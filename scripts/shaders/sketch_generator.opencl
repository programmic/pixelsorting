// Random number generator helper (LGC)
float prng(uint2 *state) {
    state->x = 1664525u * state->x + 1013904223u;
    state->y = 1664525u * state->y + 1013904223u;
    return (float)(state->x ^ state->y) / 4294967296.0f;
}

// Perceptual brightness mapper helper
float get_brightness(__global const uchar* img, int x, int y, int width) {
    int idx = (y * width + x) * 3;
    float r = (float)img[idx];
    float g = (float)img[idx+1];
    float b = (float)img[idx+2];
    return (0.299f * r + 0.587f * g + 0.114f * b) * (100.0f / 255.0f);
}

// Atomically fade the source image pixel to simulate standard canvas fading logic
void fade_pixel(__global uchar* img, int x, int y, int width) {
    int idx = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        int val = img[idx + c] + 25;
        img[idx + c] = (val > 255) ? 255 : val;
    }
}

__kernel void run_sketch_traces(
    __global uchar* img_pixels,         // Read-Write source image buffer
    __global float4* line_segments,     // Output buffer for lines: [x1, y1, x2, y2]
    __global uchar4* line_colors,       // Output buffer for line styling: [r, g, b, alpha]
    __global int* line_weights,         // Output buffer for line stroke weights
    const int width, const int height,
    const int max_count, const int sub_step,
    const int scale, const int perception,
    const float max_speed, const float noise_scale,
    const float noise_influence, const float drop_rate,
    const int drop_alpha, const int draw_alpha,
    const int draw_weight, const int total_traces,
    const int cycle
) {
    int gid = get_global_id(0);
    if (gid >= total_traces) return;

    // Seed PRNG dynamically based on global ID and current cycle
    uint2 rand_state = (uint2)(gid + 12345, cycle + 67890);

    // Track particle properties
    float pos_x = 0.0f, pos_y = 0.0f;
    float vel_x = 0.0f, vel_y = 0.0f;
    uchar3 draw_color = (uchar3)(0, 0, 0);

    // Particle trace re-initialization routine
    bool initialized = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
        float rx = prng(&rand_state) * (width - 1);
        float ry = prng(&rand_state) * (height - 1);
        if (get_brightness(img_pixels, (int)rx, (int)ry, width) < 50.0f) {
            pos_x = rx * scale;
            pos_y = ry * scale;
            int idx = ((int)ry * width + (int)rx) * 3;
            draw_color = (uchar3)(img_pixels[idx], img_pixels[idx+1], img_pixels[idx+2]);
            initialized = true;
            break;
        }
    }
    if (!initialized) return;

    int paint_count = 0;
    int scaled_perception = perception * scale;
    int half_p = scaled_perception / 2;
    float scaled_max_speed = max_speed * scale;
    int scaled_draw_weight = draw_weight * scale;

    // Execute sub-steps sequentially inside the thread channel
    for (int step = 0; step < sub_step; ++step) {
        float pre_x = pos_x;
        float pre_y = pos_y;

        float force_x = 0.0f, force_y = 0.0f;
        int count = 0;

        // Scan perception radius
        for (int i = -half_p; i <= half_p; ++i) {
            for (int j = -half_p; j <= half_p; ++j) {
                if (i == 0 && j == 0) continue;

                int src_x = (int)((pos_x + i) / scale);
                int src_y = (int)((pos_y + j) / scale);

                if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                    float b = 1.0f - (get_brightness(img_pixels, src_x, src_y, width) / 100.0f);
                    float p_mag = sqrt((float)(i * i + j * j));
                    if (p_mag > 0.0f) {
                        force_x += (i / p_mag) * b / p_mag;
                        force_y += (j / p_mag) * b / p_mag;
                        count++;
                    }
                }
            }
        }

        if (count != 0) {
            force_x /= count;
            force_y /= count;
        }

        // Evaluate trigonometry noise matrix
        float n = (sin(pos_x / (noise_scale * scale)) + cos(pos_y / (noise_scale * scale))) * (cycle * 0.01f);
        float n_mapped = (n + 2.0f) * (scaled_perception * 3.14159265f);
        float noise_x = cos(n_mapped);
        float noise_y = sin(n_mapped);

        float force_mag = sqrt(force_x * force_x + force_y * force_y);
        if (force_mag < 0.01f) {
            force_x += noise_x * (noise_influence * scaled_perception);
            force_y += noise_y * (noise_influence * scaled_perception);
        } else {
            force_x += noise_x * noise_influence;
            force_y += noise_y * noise_influence;
        }

        vel_x += force_x;
        vel_y += force_y;

        // Limit velocity magnitudes
        float vel_mag = sqrt(vel_x * vel_x + vel_y * vel_y);
        if (vel_mag > scaled_max_speed) {
            vel_x = (vel_x / vel_mag) * scaled_max_speed;
            vel_y = (vel_y / vel_mag) * scaled_max_speed;
        }

        pos_x += vel_x;
        pos_y += vel_y;
        paint_count++;

        // Canvas structural bounds checks
        if (paint_count > max_count || pos_x >= (width * scale) || pos_x < 0 || pos_y >= (height * scale) || pos_y < 0) {
            // Reinitialize particle position safely
            paint_count = 0;
            float rx = prng(&rand_state) * (width - 1);
            float ry = prng(&rand_state) * (height - 1);
            pos_x = rx * scale;
            pos_y = ry * scale;
            int idx = ((int)ry * width + (int)rx) * 3;
            draw_color = (uchar3)(img_pixels[idx], img_pixels[idx+1], img_pixels[idx+2]);
            vel_x = 0.0f; vel_y = 0.0f;
            continue; 
        }

        // Dynamic Line Weight formatting criteria
        int alpha = draw_alpha;
        int weight = scaled_draw_weight;
        float current_force_mag = sqrt(force_x * force_x + force_y * force_y);
        if (current_force_mag > 0.1f && prng(&rand_state) < drop_rate) {
            alpha = drop_alpha;
            weight = (int)(scaled_draw_weight + prng(&rand_state) * scaled_perception);
        }

        // Export data to line payload arrays for standard PIL painting cycles
        int out_idx = (gid * sub_step) + step;
        line_segments[out_idx] = (float4)(pre_x, pre_y, pos_x, pos_y);
        line_colors[out_idx] = (uchar4)(draw_color.x, draw_color.y, draw_color.z, alpha);
        line_weights[out_idx] = weight;

        // Apply canvas color fading onto source images mapping back to scale bounds
        fade_pixel(img_pixels, (int)(pre_x / scale), (int)(pre_y / scale), width);
    }
}