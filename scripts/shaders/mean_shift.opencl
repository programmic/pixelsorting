__kernel void mean_shift(
    __global const float *input_img,
    __global float *output_img,
    const int width,
    const int height,
    const int channels,
    const int spatial_radius,
    const int color_radius,
    const int max_iter
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    // Initialize mean to current pixel
    float mean[3];
    mean[0] = input_img[idx + 0];
    mean[1] = input_img[idx + 1];
    mean[2] = input_img[idx + 2];

    for (int iter = 0; iter < max_iter; ++iter) {
        float sum[3] = {0.0f, 0.0f, 0.0f};
        int count = 0;

        // Search window
        for (int dy = -spatial_radius; dy <= spatial_radius; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= height) continue;

            for (int dx = -spatial_radius; dx <= spatial_radius; ++dx) {
                int nx = x + dx;
                if (nx < 0 || nx >= width) continue;

                int nidx = (ny * width + nx) * channels;

                // Color distance
                float d0 = input_img[nidx + 0] - mean[0];
                float d1 = input_img[nidx + 1] - mean[1];
                float d2 = input_img[nidx + 2] - mean[2];
                float color_dist = sqrt(d0*d0 + d1*d1 + d2*d2);

                if (color_dist <= color_radius) {
                    sum[0] += input_img[nidx + 0];
                    sum[1] += input_img[nidx + 1];
                    sum[2] += input_img[nidx + 2];
                    count += 1;
                }
            }
        }

        if (count == 0) break;

        float new_mean[3];
        new_mean[0] = sum[0] / count;
        new_mean[1] = sum[1] / count;
        new_mean[2] = sum[2] / count;

        // Check for convergence (optional: use a threshold)
        float shift = fabs(new_mean[0] - mean[0]) +
                      fabs(new_mean[1] - mean[1]) +
                      fabs(new_mean[2] - mean[2]);
        mean[0] = new_mean[0];
        mean[1] = new_mean[1];
        mean[2] = new_mean[2];

        if (shift < 0.1f) // convergence threshold
            break;
    }

    output_img[idx + 0] = mean[0];
    output_img[idx + 1] = mean[1];
    output_img[idx + 2] = mean[2];
}