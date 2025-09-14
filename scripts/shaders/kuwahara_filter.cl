__kernel void kuwahara_filter(__global const float *input,
                              __global float *output, const int width,
                              const int height, const int channels,
                              const int radius) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= width || y >= height)
    return;

  float mean[4][3] = {0};
  float var[4] = {0};
  int count[4] = {0};

  // Calculate regions and their statistics
  // Top-left region
  for (int dy = -radius; dy <= 0; dy++) {
    for (int dx = -radius; dx <= 0; dx++) {
      int nx = clamp(x + dx, 0, width - 1);
      int ny = clamp(y + dy, 0, height - 1);
      int idx = (ny * width + nx) * channels;
      for (int c = 0; c < channels; c++) {
        mean[0][c] += input[idx + c];
      }
      float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] +
                   0.114f * input[idx + 2];
      var[0] += gray * gray;
      count[0]++;
    }
  }
  // Top-right region
  for (int dy = -radius; dy <= 0; dy++) {
    for (int dx = 0; dx <= radius; dx++) {
      int nx = clamp(x + dx, 0, width - 1);
      int ny = clamp(y + dy, 0, height - 1);
      int idx = (ny * width + nx) * channels;
      for (int c = 0; c < channels; c++) {
        mean[1][c] += input[idx + c];
      }
      float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] +
                   0.114f * input[idx + 2];
      var[1] += gray * gray;
      count[1]++;
    }
  }
  // Bottom-left region
  for (int dy = 0; dy <= radius; dy++) {
    for (int dx = -radius; dx <= 0; dx++) {
      int nx = clamp(x + dx, 0, width - 1);
      int ny = clamp(y + dy, 0, height - 1);
      int idx = (ny * width + nx) * channels;
      for (int c = 0; c < channels; c++) {
        mean[2][c] += input[idx + c];
      }
      float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] +
                   0.114f * input[idx + 2];
      var[2] += gray * gray;
      count[2]++;
    }
  }
  // Bottom-right region
  for (int dy = 0; dy <= radius; dy++) {
    for (int dx = 0; dx <= radius; dx++) {
      int nx = clamp(x + dx, 0, width - 1);
      int ny = clamp(y + dy, 0, height - 1);
      int idx = (ny * width + nx) * channels;
      for (int c = 0; c < channels; c++) {
        mean[3][c] += input[idx + c];
      }
      float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] +
                   0.114f * input[idx + 2];
      var[3] += gray * gray;
      count[3]++;
    }
  }
  // Find region with minimum variance
  float min_var = 1e20f;
  int best_region = 0;
  for (int i = 0; i < 4; i++) {
    if (count[i] > 0) {
      float variance = var[i] / count[i];
      if (variance < min_var) {
        min_var = variance;
        best_region = i;
      }
    }
  }
  // Calculate mean color for best region
  int out_idx = (y * width + x) * channels;
  for (int c = 0; c < channels; c++) {
    output[out_idx + c] = mean[best_region][c] / max(count[best_region], 1);
  }
}
