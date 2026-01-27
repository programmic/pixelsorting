__kernel void papari_aniso_kuwahara(
    __global const float* img,
    __global const float* gray,
    __global float* out,

    int W,
    int H,
    int C,

    int radius,
    int regions,

    float tensor_sigma,
    float anisotropy,
    float epsilon
){
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= W || y >= H) return;

    int idx = (y * W + x);
    int base = idx * C;

    // -----------------------------
    // 1) Compute structure tensor
    // -----------------------------

    float gx = 0.0f;
    float gy = 0.0f;

    int xm1 = max(x - 1, 0);
    int xp1 = min(x + 1, W - 1);
    int ym1 = max(y - 1, 0);
    int yp1 = min(y + 1, H - 1);

    float gxm = gray[y * W + xm1];
    float gxp = gray[y * W + xp1];
    float gym = gray[ym1 * W + x];
    float gyp = gray[yp1 * W + x];

    gx = 0.5f * (gxp - gxm);
    gy = 0.5f * (gyp - gym);

    float Jxx = gx * gx;
    float Jyy = gy * gy;
    float Jxy = gx * gy;

    // Eigen decomposition (orientation)

    float trace = Jxx + Jyy;
    float det = Jxx * Jyy - Jxy * Jxy;
    float disc = sqrt(fmax(trace * trace * 0.25f - det, 0.0f));

    float lambda1 = trace * 0.5f + disc;
    float lambda2 = trace * 0.5f - disc;

    float2 v;

    if (fabs(Jxy) > epsilon) {
        v.x = lambda1 - Jyy;
        v.y = Jxy;
    } else {
        v.x = 1.0f;
        v.y = 0.0f;
    }

    float len = sqrt(v.x * v.x + v.y * v.y) + epsilon;
    v.x /= len;
    v.y /= len;

    // anisotropic axes
    float major = radius * anisotropy;
    float minor = radius;

    // -----------------------------
    // 2) Kuwahara sectors
    // -----------------------------

    float best_var = 1e30f;
    float3 best_mean = (float3)(0,0,0);

    for(int r = 0; r < regions; r++){

        float angle0 = (2.0f * M_PI_F * r) / regions;
        float angle1 = (2.0f * M_PI_F * (r+1)) / regions;

        float3 mean = (float3)(0,0,0);
        float3 sqmean = (float3)(0,0,0);
        float wsum = 0.0f;

        for(int dy = -radius; dy <= radius; dy++){
            for(int dx = -radius; dx <= radius; dx++){

                float px = (float)dx;
                float py = (float)dy;

                float rx =  px * v.x + py * v.y;
                float ry = -px * v.y + py * v.x;

                float ell = (rx*rx)/(major*major) + (ry*ry)/(minor*minor);
                if(ell > 1.0f) continue;

                float theta = atan2(py, px);
                if(theta < 0) theta += 2.0f * M_PI_F;

                if(theta < angle0 || theta >= angle1) continue;

                int sx = clamp(x + dx, 0, W - 1);
                int sy = clamp(y + dy, 0, H - 1);
                int sidx = (sy * W + sx) * C;

                float3 col;
                col.x = img[sidx + 0];
                col.y = img[sidx + 1];
                col.z = img[sidx + 2];

                mean += col;
                sqmean += col * col;
                wsum += 1.0f;
            }
        }

        if(wsum < 1.0f) continue;

        mean /= wsum;
        sqmean /= wsum;

        float3 var3 = sqmean - mean * mean;
        float var = var3.x + var3.y + var3.z;

        if(var < best_var){
            best_var = var;
            best_mean = mean;
        }
    }

    // -----------------------------
    // 3) Write result
    // -----------------------------

    out[base + 0] = best_mean.x;
    out[base + 1] = best_mean.y;
    out[base + 2] = best_mean.z;
}
