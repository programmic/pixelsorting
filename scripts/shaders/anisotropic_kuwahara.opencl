__kernel void anisotropic_kuwahara(__global const float *input,
                                   __global float *output, const int width,
                                   const int height, const int channels,
                                   const int radius, const int regions) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height)
    return;
  int idx = (y * width + x) * channels;
  float gx = 0.0f, gy = 0.0f;
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    int idx_left = (y * width + (x - 1)) * channels;
    int idx_right = (y * width + (x + 1)) * channels;
    int idx_up = ((y - 1) * width + x) * channels;
    int idx_down = ((y + 1) * width + x) * channels;
    float gray_left = 0.299f * input[idx_left] + 0.587f * input[idx_left + 1] +
                      0.114f * input[idx_left + 2];
    float gray_right = 0.299f * input[idx_right] +
                       0.587f * input[idx_right + 1] +
                       0.114f * input[idx_right + 2];
    float gray_up = 0.299f * input[idx_up] + 0.587f * input[idx_up + 1] +
                    0.114f * input[idx_up + 2];
    float gray_down = 0.299f * input[idx_down] + 0.587f * input[idx_down + 1] +
                      0.114f * input[idx_down + 2];
    gx = gray_right - gray_left;
    gy = gray_down - gray_up;
  }
  float orientation = atan2(gy, gx);
  float mean[16][3];
  float var[16];
  int count[16];
  for (int i = 0; i < regions; i++) {
    for (int c = 0; c < 3; c++)
      mean[i][c] = 0.0f;
    var[i] = 0.0f;
    count[i] = 0;
  }
  for (int dy = -radius; dy <= radius; dy++) {
    for (int dx = -radius; dx <= radius; dx++) {
      int nx = clamp(x + dx, 0, width - 1);
      int ny = clamp(y + dy, 0, height - 1);
      int nidx = (ny * width + nx) * channels;
      float gray = 0.299f * input[nidx] + 0.587f * input[nidx + 1] +
                   0.114f * input[nidx + 2];
      float angle = atan2((float)dy, (float)dx) - orientation;
      while (angle < 0)
        angle += 6.2831853f;
      while (angle >= 6.2831853f)
        angle -= 6.2831853f;
      int region = (int)(angle / (6.2831853f / regions));
      if (region >= regions)
        region = regions - 1;
      for (int c = 0; c < channels; c++) {
        mean[region][c] += input[nidx + c];
      }
      var[region] += gray * gray;
      count[region]++;
    }
  }
  float min_var = 1e20f;
  int best_region = 0;
  for (int i = 0; i < regions; i++) {
    if (count[i] > 0) {
      float variance = var[i] / count[i];
      if (variance < min_var) {
        min_var = variance;
        best_region = i;
      }
    }
  }
  int out_idx = (y * width + x) * channels;
  for (int c = 0; c < channels; c++) {
    output[out_idx + c] = mean[best_region][c] / max(count[best_region], 1);
  }
}
