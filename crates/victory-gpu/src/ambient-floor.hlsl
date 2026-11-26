// Indirect-light floor adapted from Belobug/VIctory-Belobug-AMD 06031237.
// SPDX-License-Identifier: GPL-3.0-or-later
#define ROOT "RootConstants(num32BitConstants=1, b0), DescriptorTable(UAV(u0, flags=DESCRIPTORS_VOLATILE))"
cbuffer Floor : register(b0) { float level; };
RWTexture2D<uint> light : register(u0);

float3 unpack(uint value) {
    return float3(value & 511, (value >> 9) & 511, (value >> 18) & 511)
        * exp2(float(value >> 27) - 24.0);
}

uint pack(float3 rgb) {
    rgb = clamp(rgb, 0.0, 65408.0);
    float peak = max(rgb.r, max(rgb.g, rgb.b));
    int exponent = clamp(int(floor(log2(max(peak, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - exponent));
    if (peak * scale >= 511.5) {
        exponent = min(exponent + 1, 31);
        scale = exp2(float(24 - exponent));
    }
    uint3 mantissa = min(uint3(rgb * scale + 0.5), 511u);
    return mantissa.r | (mantissa.g << 9) | (mantissa.b << 18) | (uint(exponent) << 27);
}

[RootSignature(ROOT)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    light.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    const float3 luma = float3(0.2126, 0.7152, 0.0722);
    float3 current = unpack(light[id.xy]);
    float missing = level - dot(current, luma);
    if (missing > 0.0) light[id.xy] = pack(current + missing);
}
