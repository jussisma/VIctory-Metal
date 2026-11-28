//! VRAB v2 host binding proof: translated shaders, translated root signature,
//! exact GPU results.
//!
//! Every shader here is compiled from this repository's own HLSL (source in the
//! comments below) with Microsoft `dxc`, then translated by Apple's Metal
//! Shader Converter. The DXIL containers embed a versioned root signature that
//! names the same spaces the native host binds:
//!
//! ```text
//! USER_DATA_SPACE (0)  root constants  b0
//! ROOT_CBV_SPACE  (1)  root CBV/UAV    b0/u0
//! MEMORY_SPACE    (2)  root SRV        t0/t1
//! ```
//!
//! The host test builds the authoritative slot map from those root-signature
//! bytes with [`victory_shader::emit::translated_pipeline_root_signature`] and
//! submits wire root arguments by their root-parameter slot. It proves:
//!
//! * a translated vertex/pixel pair draws an exact RGBA8 colour from a
//!   USER_DATA_SPACE constant block plus a ROOT_CBV_SPACE buffer;
//! * a translated compute shader reads a USER_DATA_SPACE constant and a
//!   MEMORY_SPACE buffer supplied through `SetRootBufferView`, and an unbound
//!   MEMORY_SPACE parameter is served a zero-filled default buffer;
//! * an argument naming a slot the layout does not declare is refused and
//!   counted.
//!
//! The tests are `#[ignore]`d because the GPU is shared and the converter is
//! x86_64 (Rosetta). Run one at a time:
//!
//! ```sh
//! cargo test -p victory-metal --test vrab_host_binding_gpu -- --ignored --test-threads=1
//! ```
//!
//! # HLSL sources
//!
//! Vertex (compiled `-T vs_6_0 -E main`):
//!
//! ```hlsl
//! cbuffer User : register(b0, space0) { float4 tint; };
//! cbuffer Root : register(b0, space1) { float4 cbvColor; };
//! struct VSOut { float4 pos : SV_Position; float4 color : COLOR0; };
//! [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | \
//!   CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), \
//!   RootConstants(b0, space=0, num32BitConstants=4, \
//!   visibility=SHADER_VISIBILITY_VERTEX), \
//!   CBV(b0, space=1, visibility=SHADER_VISIBILITY_VERTEX)")]
//! VSOut main(uint id : SV_VertexID) {
//!     float2 p = float2((id == 1) ? 3.0 : -1.0, (id == 2) ? 3.0 : -1.0);
//!     VSOut o;
//!     o.pos = float4(p, 0.0, 1.0);
//!     o.color = tint + cbvColor;
//!     return o;
//! }
//! ```
//!
//! Pixel (compiled `-T ps_6_0 -E main`):
//!
//! ```hlsl
//! struct PSIn { float4 pos : SV_Position; float4 color : COLOR0; };
//! [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
//! float4 main(PSIn i) : SV_Target { return i.color; }
//! ```
//!
//! Compute (compiled `-T cs_6_0 -E main`):
//!
//! ```hlsl
//! cbuffer User : register(b0, space0) { float add; };
//! StructuredBuffer<float> mem0 : register(t0, space2);
//! StructuredBuffer<float> mem1 : register(t1, space2);
//! RWStructuredBuffer<float> outv : register(u0, space1);
//! [RootSignature("RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | \
//!   SAMPLER_HEAP_DIRECTLY_INDEXED), \
//!   RootConstants(b0, space=0, num32BitConstants=1, \
//!   visibility=SHADER_VISIBILITY_ALL), \
//!   UAV(u0, space=1, visibility=SHADER_VISIBILITY_ALL), \
//!   SRV(t0, space=2, visibility=SHADER_VISIBILITY_ALL), \
//!   SRV(t1, space=2, visibility=SHADER_VISIBILITY_ALL)")]
//! [numthreads(1,1,1)]
//! void main(uint3 id : SV_DispatchThreadID) {
//!     outv[id.x] = mem0[0] + add + mem1[0];
//! }
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::time::Duration;

use victory_metal::MetalTransport;
use victory_metal::compiler_pool::{CompileJob, CompilerPool, PoolConfig};
use victory_metal::macos::bindings::{ReflectedStage, RootParameterMap, RootSignatureLayout};
use victory_metal::macos::device::{DirectMetalTransport, PipelineAbi};
use victory_metal::macos::registry::{
    FunctionRole, PipelineStage as RegistryPipelineStage, PipelineState, StageLibrary, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage,
};
use victory_metal::wire::Dispatch1d;
use victory_metal::wire_v3::{
    BeginRenderPass, ColorAttachment, CommandV3, DrawState, LoadOp, PipelineRef, PrimitiveTopology,
    RootBufferView, RootConstantsV3, ScissorRect, StoreOp, SubmissionV3, Viewport,
};
use victory_shader::emit::{
    HostShader, RootSignature, SignatureFlags, UserData, translated_pipeline_root_signature,
    translated_root_signature,
};
use victory_shader::native::{HwStage, RootType};

const EDGE: u32 = 2;
const TARGET: u64 = 0x7A00;
const GRAPHICS_HANDLE: u64 = 0x7A10;
const COMPUTE_HANDLE: u64 = 0x7A20;
const PAGE_BYTES: usize = 16 * 1024;

/// Vertex DXIL: USER_DATA_SPACE b0 (4 dwords) + ROOT_CBV_SPACE b0.
const VERTEX_DXIL_HEX: &str = "44584243e0f8c36c5e272a95e0dff56e74d7da8501000000fc0e00000800000040000000500000008c000000f0000000c8010000180200004c080000680800005346493008000000000000000000000049534731340000000100000008000000000000002800000000000000060000000100000000000000010100000000000053565f5665727465784944004f5347315c00000002000000080000000000000048000000000000000100000003000000000000000f000000000000000000000054000000000000000000000003000000010000000f0000000000000053565f506f736974696f6e00434f4c4f5200000050535630d0000000380000000100000000000000000000000000000000000000ffffffff01000000010200010200000000000000000000000000000007000000000000000200000018000000020000000000000000000000000000000d00000000000000020000000100000000000000000000000d000000000000000c00000000434f4c4f52006d61696e000100000000000000100000000000000000000000010041010100000000000000000000000100440303040000010000000000000001014400030200000300000000000000000000000000000052545330480000000200000002000000180000000000000048000000010c000001000000010000003000000002000000010000003c000000000000000000000004000000000000000100000000000000535441542c060000600001008b0100004458494c0001000010000000140600004243c0de210c0000820100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d84f0ffffffff03206d3086ffffffff1f0009a800491800000300000013826042204c080600000000892000002100000032224809206485049322a484049322e384a19014124c8a8c0b84a44c10642300250014e608c0608e0029c6186390416506a018608c410ea1a386cb9fb087907c6ea38a9598fce2b61171ce3944ee192e7fc21e42f243a019160205ab106ab0416d8e2028061b648c477020601881388e70252242c1247ac4ea38130a26d91c1800000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c05100001000000000000000c79102000040000000000000018f23440000c0000000000000030e48180001800000000000000200b04000011000000321e981419114c908c092647c604431a253002500c055206e55012a55108e5513c455010444aa10446004aa20c0aa10888ce00509d01203b0340770680f05843800c0000791800008d0000001a034c90460213443420c31b4381934bb30ba32b4b018971c971817181014181934bb30ba32b6301818181818181818149d910041384a198200cc606612036080441c16e6e82301c1b86032126089546a26aae4c6e8230201b1042590862608009c2c591907a7ba3db800cce331003036c101a68030100113041a0321274696e7413842199200cca0461582608033341189a0d084255847561d90681d12608d646644ccc6ee88ded4d6e03827415615d58b64160bc0d05316ddc374110800dc08681100331d8108cc18661080332982060dd86c00c48b485a5b97199b2fab22a93a32bc34b229a2010d20481983604c40481a03608d5b06121d2400dd6800dda605803a20ddc6043f0065ca6acbea0dee6d2e8d2dedc260844354120ac0902714d1006678350d5c18685880339980336a083810e8836b00316434f4c4f52130402dbb00c7820076dc0067930d0c1d00676b041b8033dd830c0c11e001b8a3040033e90001a666c6f6174732cd2dce6e8e62608c34322cd8d6e6e82304034e6d2cebed8c868cca59d7dcdd14d10866883e207d91f80422888c2280ca4500a55d8d8ecda5cd2c8cadce8a6044115323c17bb32b9b9b437b72901d1840ccfc52e8ccdae4c6e4a50d421c37399430b232b936b7a232b639b122065c8f05ce4cae6deeae4c6cae6a6045125323c17ba3cb8b22037b737ba30bab437b7b929c2470675c8f05cecd2caee92c8a6e8c2e8caa6046650870ccfa5cc8d4e2e0fea2dcd8d6e6e4ac0075dc8f05cc6deeadce8cae4e6a604a500791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001400000036b00d97ef3cbe1050454144a5030c25610002e617b76d04d270f9cee30b11014c440834c34258c0345cbef3f88b030c62f350938fdcb60954c3e53b8f2f4d4e44a0d4f450935fdcb601100c8034000000000000484153481400000000000000f0510e382a0b7d442f1c8905bb73d8ee4458494c8c06000060000100a30100004458494c0001000010000000740600004243c0de210c00009a0100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d84f0ffffffff03206d3086ffffffff1f0009a800491800000300000013826042204c080600000000892000002100000032224809206485049322a484049322e384a19014124c8a8c0b84a44c10642300250014e608c0608e0029c6186390416506a018608c410ea1a386cb9fb087907c6ea38a9598fce2b61171ce3944ee192e7fc21e42f243a019160205ab106ab0416d8e2028061b648c477020601881388e70252242c1247ac4ea38130a26d91c1800000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c05100001000000000000000c79102000040000000000000018f23440000c0000000000000030e48180001800000000000000200b0400000d000000321e981419114c908c092647c604431a2530025014c5502065501e444aa10446004aa20c0aa108a8ce00d09d01203cd610200300791800005c0000001a034c90460213443420c31b4381934bb30ba32b4b018971c971817181014181934bb30ba32b6301818181818181818149d910041384a198200cc60661202608c3b141180c0a7673138401d93020093141a82602138421d980100b431043034c102e6a03323ccc400c0db04170a00d040044c00401ab3604d304410048b485a5b97199b2fab22a93a32bc34b229a2010cc0481683604c4048170260883b241e0860d0b716199b60d19b1751b028fcb94d517d4db5c1a5dda9bdb048178260804344120a209c2b06c103832d8b01060100662a08dc13006c456062c869e989ea4260884b46119ce200c360d0d863118b632d82098411a6c183e3500361495b5061250858dcdaecd258daccc8d6e4a105421c373b12b939b4b7b739b12104dc8f05cecc2d8eccae4a604461d323c9739b430b232b9a637b232b6294152860ccf45ae6ceead4e6eac6c6e4a10d521c373b14b2bbb4b229ba20ba32b9b124c75c8f05ccadce8e4f2a0ded2dce8e6a6046b00791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001400000036b00d97ef3cbe1050454144a5030c25610002e617b76d04d270f9cee30b11014c440834c34258c0345cbef3f88b030c62f350938fdcb60954c3e53b8f2f4d4e44a0d4f450935fdcb601100c80340000612000004b0000001304412c100000000a000000344aa1e4caae1066008a80de08009d3102100441fc174600c608401004f16f8c00044190040300002306090082600075cac36dc588410280201840de02719c316290002008068518285d5725c30dc10706d30d05120c370c0b184c371c4a306270002008064d194086188c2604c06882108c260cc26802318c181c00088241a306157306a30901309a2004a30983309a400c4624f03122818f11097c8c48e033629000200806081d80011cc0011a4423060900826080d00118c0011ca4c133629000200806081d80011cc001196023060900826080d00118c0011c98c135629000200806081d80411cc0011a1023060900826080d00118c4011ca4c130629000200806081d80411cc001190823060900826080d00118c4011c9841800000000000000000";

/// Pass-through pixel DXIL with an empty root signature.
const PIXEL_DXIL_HEX: &str = "44584243f3bd016fa434e5f70017ed2eb02f689401000000c40b0000080000004000000050000000b4000000f0000000a4010000c4010000a4060000c006000053464930080000000000000000000000495347315c00000002000000080000000000000048000000000000000100000003000000000000000f000000000000000000000054000000000000000000000003000000010000000f0f00000000000053565f506f736974696f6e00434f4c4f520000004f5347313400000001000000080000000000000028000000000000004000000003000000000000000f0000000000000053565f54617267657400000050535630ac000000380000000000000000000000000000000000000000000000ffffffff0000000002010002010000000000000000000000000000000700000000000000000000000c00000000434f4c4f52006d61696e000100000000000000100000000000000000000000010044030304000001000000000000000101440003020000000000000000000001004410030000000000000000000000000000000000000001000000020000000400000008000000525453301800000002000000000000001800000000000000180000000100000053544154d804000060000000360100004458494c0001000010000000c00400004243c0de210c00002d0100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628010450242920b42841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a0421460651180000060000001b8ce0ffffffff074002a80d84f0ffffffff03200100000049180000020000001382604220000000892000000f00000032220809206485041322a484041322e384a19014124c888c0b84844c1030230025008a19803902309823408a31445444560c20a21ac2818054200000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c06100001000000000000000c791020000400000000000000c802010d000000321e981419114c908c092647c60443a2121801288632288f922804aa922883421801288202a11d4b41080000804020100000000079180000610000001a034c90460213443420c31b4381934bb30ba32b4b018971c971817181014181934bb30ba32b6301818181818181818149d91004130462982010c406612036080441016e6e8240141b860321260802b001d83010cbb22160360c83d24c10166843f090680b4b73e33265f505f536974697f6e6364128940942b16c08880942c14c108a66c3424813555983455c008ba127a627a90942e14c10086382401c1b848ddbb00cd97455da600d57b741c03c2653565f5461726765741384e2d9b01060308541750d1671751b0231d8307c63006c2894880c0080459adb1cdddc04814068cca59d7db1914d108884c65cdad9d71cdd06c30cce000dd2400dd2a00a1b9b5d9b4b1a59991bdd9420a84286e76257263797f6e63625209a90e1b9d885b1d995c94d098a3a64782e7368616465724d6f64656c5302a412199e0b5d1e5c59909bdb1b5d185dda9bdbdc94a0a94386e76297567697443645174657362578ea90e1b994b9d1c9e541bda5b9d1cd4d09c8a00b199ecbd85b9d1b5d99dcdc94400d00000000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000000b00000016300d97ef3cfee20083d83cd4e417b76d02d570f9cee34b93131128353dd4e417b76d000403200d000000000000000048415348140000000000000097e32c74b22eb72158fe9924047145044458494cfc040000600000003f0100004458494c0001000010000000e40400004243c0de210c0000360100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628010450242920b42841032143808184b0a32428848901420434688a500193242e4480e901122c4504151818ce183e58a0421460651180000060000001b8ce0ffffffff074002a80d84f0ffffffff03200100000049180000020000001382604220000000892000000f00000032220809206485041322a484041322e384a19014124c888c0b84844c1030230025008a19803902309823408a31445444560c20a21ac2818054200000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c06100001000000000000000c791020000400000000000000c802010c000000321e981419114c908c092647c60443a2121801288a622883f2a02a893228841180222810dab114840000000804020100791800004b0000001a034c90460213443420c31b4381934bb30ba32b4b018971c971817181014181934bb30ba32b6301818181818181818149d91004130462982010c4066120260844b141180c0a70731304c2d83020093141589e0dc132411000126d61696e5ca6acbea0dee6d2e8d2dedc260845324128940d01314128960942c16c5888078aa46998080a6031f4c4f4243541289a0902714c1008648380651b96c18228e91aa681d23608d5c664caea8b2a4ceeac8c6e8250381b16a2833c891a2682d23604df868103036043d13861000055d8d8ecda5cd2c8cadce8a6044115323c17bb32b9b9b437b72901d1840ccfc52e8ccdae4c6e4a60d421c37399430b232b936b7a232b639b122475c8f05cecd2caee92c8a6e8c2e8caa6044b1d323c9732373ab93ca8b73437bab9294118000000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000000b00000016300d97ef3cfee20083d83cd4e417b76d02d570f9cee34b93131128353dd4e417b76d000403200d00000000612000001e0000001304412c100000000300000044853003500a542500000000230609008260604c440421c388410280201818542145023162900020080646654c52528c182400088281611dd4d4182306090082608058064531c488410280201820964151c530629000200806886550d4228c182400088201621914e504080000000000";

/// Compute DXIL: USER_DATA_SPACE b0, ROOT_CBV_SPACE u0, MEMORY_SPACE t0/t1.
const COMPUTE_DXIL_HEX: &str = "445842431011793e8e2c40c9b34bf9aec718e849010000006c0e000008000000400000005000000060000000700000002c010000ac010000140800003008000053464930080000000000000000000000495347310800000000000000080000004f53473108000000000000000800000050535630b4000000380000000000000000000000000000000000000000000000ffffffff05000000000000000000000001000000010000000100000001000000000000000400000018000000020000000000000000000000000000000d00000000000000050000000200000000000000000000000c00000000000000050000000200000001000000010000000c00000000000000080000000100000000000000000000000c0000000000000008000000006d61696e0000000000000052545330780000000200000004000000180000000000000078000000000c000001000000000000004800000004000000000000005400000003000000000000006000000003000000000000006c000000000000000000000001000000000000000100000000000000000000000200000000000000010000000200000000000000535441546006000060000500980100004458494c0001000010000000480600004243c0de210c00008f0100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662f8ffffffff019000491800000300000013826042204c080600000000892000003e00000032224809206485049322a484049322e384a19014124c8a8c0b84a44c10742300250014e608c0a00c630c223300470d973f610f21f9dc46152b31f9c56d23c21863e6081042f70c973f610f21f921d00c0b8182548833d4a035471014430d344623571430d418c6188308de345cfe843d84e4af84b412935fdc362a8c316694420e3506cd818099ba71608770988779708359a00779a88771a0877a908772200751a8077330877290073e300776788770a0073f4081213bd3370eec100ef3300f6e200bb7300bf4200ff5300ef4500ff2500ee4200af5600ee6500ef2c007e6c00eef100ef4e0072830848f7025224281213d47000a000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0008000000000000000000863c04100001000000000000000c79162000040000000000000018f2384000080000000000000030e4818000180000000000000060c83301012000000000000000401608000f000000321e981419114c908c092647c604431a253002500c455018855006e55012e541bc40a88d00909d01a03b0340780680f20c00e91900da33006466000079180000780000001a034c90460213443420c31b4381934bb30ba32b4b018971c971817181014181934bb30ba32b6301818181818181818149d910041384c198200cc706612036080441616c6e8230201b8603212608d744a2adac0d6682302413844199200ccb066170362484b230c4d0100f89b6b2b6980dc9a044cc3034c4b34180a40942569178aba3b39b2034d18685a0aa81181acbb29e0dc13541d82e12557365721b1022d3086270800dc1b68198300e9820581203b409c2c04c1086668230381b0ce40303220cc4608260601b04820c360465b04120cc60c3e08cc1194c10306a82a0591cc2c8c83618c81a8001110662b04170d860c3417868900668a0066d304110800dc08681800338d810c4c1866178033920d11696e63641e0a00dc3300c1b08a20e1c3bd850bc011d00dd1dd030637b0ba39b6391e63647373741181e227465785f6c6f61644ce8caf0bee6e8dee4ca362079c0e8c11ef001d30743153636bb369734b23237ba294150850ccfc5ae4c6e2eedcd6d4a403421c373b10b63b32b939b121475c8f05ce6d0c2c8cae49adec8cad8a6044819323c17b9b2b9b73ab9b1b2b9290157890ccf852e0fae2cc8cded8d2e8c2eedcd6d6e8ad0067250870ccfa5cc8d4e2e0fea2dcd8d6e6e4a70075dc8f05cc6deeadce8cae4e6a6047d0000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001800000056400d97ef3c7e40150511950e30f8c56d1b41355cbef3f801551444c44e4e44f8c56d9bc0365cbef3f84240150511950e3094840108985fdcb60d48c3e53b8f2f4404301121d00c0b6101d270f9cee34f4744008338f8c86d1b00c10048030000000000484153481400000000000000c863b44c448d923f086d3d3f1a9783444458494c34060000600005008d0100004458494c00010000100000001c0600004243c0de210c0000840100000b82200002000000130000000781239141c80449061032399201840c250508191e048b628014450242920b42a41032143808184b0a32528848901420434688a500193242e4480e909122c4504151818ce183e58a0429460651180000080000001b8ce0ffffffff074002a80d86f0ffffffff032001d50662f8ffffffff019000491800000300000013826042204c080600000000892000003e00000032224809206485049322a484049322e384a19014124c8a8c0b84a44c10742300250014e608c0a00c630c223300470d973f610f21f9dc46152b31f9c56d23c21863e6081042f70c973f610f21f921d00c0b8182548833d4a035471014430d344623571430d418c6188308de345cfe843d84e4af84b412935fdc362a8c316694420e3506cd818099ba71608770988779708359a00779a88771a0877a908772200751a8077330877290073e300776788770a0073f4081213bd3370eec100ef3300f6e200bb7300bf4200ff5300ef4500ff2500ee4200af5600ee6500ef2c007e6c00eef100ef4e0072830848f7025224281213d47000a000000131472c08774608736688779680372c0870daf500e6dd00e7a500e6d000f7a300772a0077320076d900e71a0077320076d900e78a0077320076d900e7160077a300772d006e9300772a0077320076d900e7640077a600774d006e6100776a0077320076d600e7320077a300772d006e6600774a0077640076de00e78a0077160077a300772a007764007439e0000000000000000000000863c04100001000000000000000c79162000040000000000000018f2384000080000000000000030e4818000180000000000000060c83301012000000000000000401608000b000000321e981419114c908c092647c604431a2530025014c55004855108d44600881708e51900da3300746700000079180000460000001a034c90460213443420c31b4381934bb30ba32b4b018971c971817181014181934bb30ba32b6301818181818181818149d910041384c198200cc7066120260803b241180c0a6373138421d93020093141b8220213844199200ccb0461603608c3b3212116a6210687803624c3c234c3e010d00621922608d93341689a0d0b41310331385555411b026b82b0411b100263086278800d41b681982e0d9820080089b6b034b70902e76c188661d84010def36d28b80ed8c0a00a1b9b5d9b4b1a59991bdd9420a84286e76257263797f6e63625209a90e1b9d885b1d995c94d098c3a64782e7368616465724d6f64656c5382a40c199e8b5cd9dc5b9ddc58d9dc9440ab4386e752e646279707f596e646373725000300000000791800004c0000003308801cc4e11c6614013d88433884c38c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c033dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41dde211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b680737688772680737808770908770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11ccca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc823bd4033bb0c30cc421077c70037a288776808719d1430ef8e006e4200ee7e006f6100ef2c00ee1900fef500ff4000000712000001800000056400d97ef3c7e40150511950e30f8c56d1b41355cbef3f801551444c44e4e44f8c56d9bc0365cbef3f84240150511950e3094840108985fdcb60d48c3e53b8f2f4404301121d00c0b6101d270f9cee34f4744008338f8c86d1b00c10048030061200000290000001304412c1000000006000000344aae10034a37a0ec4a3180501194c00800991900000000230609008260e060c94051ce88410280201838993258d6336290002008068eb61096058d182400088281b331c7754523060600826040780b36628000200806d3d60c59369a1000230607008260b08041436ca3090160c1009f11030400413098c0404a3c6f3421006c08e03362d0002008061019484d120681e338100200000000000000";

fn unhex(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0, "hex string has an odd length");
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("valid hex pair"))
        .collect()
}

struct HostPage {
    ptr: *mut c_void,
    len: usize,
}

impl HostPage {
    fn new(len: usize) -> Self {
        unsafe extern "C" {
            fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        }
        let mut raw: *mut c_void = null_mut();
        // SAFETY: `posix_memalign` writes one aligned, writable allocation.
        let rc = unsafe { posix_memalign(&mut raw, 16 * 1024, len) };
        assert_eq!(rc, 0, "posix_memalign failed");
        assert!(!raw.is_null());
        // SAFETY: `raw` is a fresh allocation of `len` writable bytes.
        unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, len) };
        Self { ptr: raw, len }
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) {
        assert!(
            offset + bytes.len() <= self.len,
            "host page write out of bounds"
        );
        // SAFETY: `offset + bytes.len() <= self.len` and `self.ptr` is live.
        unsafe {
            let destination = self.ptr.cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
    }

    fn raw(&self) -> *mut u8 {
        self.ptr.cast()
    }

    fn read_f32(&self, index: usize) -> f32 {
        assert!((index + 1) * 4 <= self.len, "host page read out of bounds");
        // SAFETY: the index is in bounds and the page is live.
        unsafe { *self.ptr.cast::<f32>().add(index) }
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn free(ptr: *mut c_void);
        }
        // SAFETY: `self.ptr` came from `posix_memalign` and is unregistered
        // before this guard drops.
        unsafe { free(self.ptr) };
    }
}

fn rgba8_desc() -> TextureDesc {
    TextureDesc {
        width: EDGE,
        height: EDGE,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::color_attachment(),
        label: Some("vrab-host-binding-colour".to_string()),
    }
}

fn converter_library() -> Option<PathBuf> {
    std::env::var_os("VICTORY_METAL_IRCONVERTER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::var_os("D3DMETAL_RES")
                .map(|root| PathBuf::from(root).join("libmetalirconverter.dylib"))
                .filter(|path| path.is_file())
        })
        .or_else(|| {
            let path = PathBuf::from(std::env::var_os("HOME")?).join(
                "Library/Caches/victory-macos/target/wine-cx/lib/external/\
                 D3DMetal.framework/Versions/A/Resources/libmetalirconverter.dylib",
            );
            path.is_file().then_some(path)
        })
}

fn converter_worker() -> Option<PathBuf> {
    std::env::var_os("VICTORY_IRCONVERTER_WORKER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/x86_64-apple-darwin/debug/victory-irconverter");
            path.is_file().then_some(path)
        })
}

/// Sets a converter environment variable for the life of one test.
struct ConverterEnv {
    name: &'static str,
    old: Option<std::ffi::OsString>,
}

impl ConverterEnv {
    fn set(name: &'static str, value: &Path) -> Self {
        let old = std::env::var_os(name);
        // SAFETY: this binary runs with --test-threads=1 and no other test
        // reads these variables at the same time.
        unsafe { std::env::set_var(name, value) };
        Self { name, old }
    }
}

impl Drop for ConverterEnv {
    fn drop(&mut self) {
        // SAFETY: see `ConverterEnv::set`.
        unsafe {
            match &self.old {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

/// The converter-backed pool, or `None` with a printed PARTIAL reason.
fn pool() -> Option<(CompilerPool, ConverterEnv)> {
    let converter = match converter_library() {
        Some(path) => path,
        None => {
            eprintln!(
                "PARTIAL: no libmetalirconverter.dylib (set VICTORY_METAL_IRCONVERTER or D3DMETAL_RES); translated-shader proof skipped"
            );
            return None;
        }
    };
    let worker = match converter_worker() {
        Some(path) => path,
        None => {
            eprintln!(
                "PARTIAL: no x86_64 victory-irconverter worker; build it with `cargo build -p victory-metal --bin victory-irconverter --target x86_64-apple-darwin`; translated-shader proof skipped"
            );
            return None;
        }
    };
    let env = ConverterEnv::set("VICTORY_METAL_IRCONVERTER", &converter);
    let config = PoolConfig {
        workers: 1,
        queue_depth: 4,
        job_timeout: Duration::from_secs(120),
        cache_dir: None,
        converter_version: "victory-metal-test/vrab-host-binding/1".to_string(),
    };
    Some((CompilerPool::new(config, Some(worker)), env))
}

fn translated_layout(reflection: &[u8], stage: ReflectedStage) -> RootSignatureLayout {
    let text = std::str::from_utf8(reflection).expect("reflection is UTF-8 JSON");
    let layout =
        RootSignatureLayout::from_converter_reflection_json(text).expect("reflection layout");
    assert_eq!(layout.stage(), stage, "converter stage");
    layout
}

/// A `HostShader` description that mirrors the DXIL's embedded root signature,
/// used only to serialize the authoritative root signature bytes.
fn described_shader(stage: HwStage, dwords: u32) -> HostShader {
    HostShader {
        stage,
        dxil: Vec::new(),
        user_data: (0..dwords)
            .map(|index| UserData {
                index,
                slot: 0,
                root: RootType::Constants,
                dword: index,
            })
            .collect(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_srv_bytes: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
    }
}

fn signature_map(signature: &RootSignature) -> RootParameterMap {
    let blob = signature
        .to_versioned_blob()
        .expect("serialize root signature");
    RootParameterMap::from_versioned_blob(&blob).expect("parse translated root signature")
}

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn translated_graphics_reads_user_data_and_root_cbv() {
    let Some((pool, _env)) = pool() else {
        return;
    };
    let vertex_dxil = unhex(VERTEX_DXIL_HEX);
    let pixel_dxil = unhex(PIXEL_DXIL_HEX);
    let vertex = pool
        .compile_with_reflection(&CompileJob {
            id: 1,
            stage: 1,
            entry: "main",
            dxil: &vertex_dxil,
        })
        .expect("translated vertex stage");
    let pixel = pool
        .compile_with_reflection(&CompileJob {
            id: 2,
            stage: 2,
            entry: "main",
            dxil: &pixel_dxil,
        })
        .expect("translated pixel stage");
    assert!(vertex.metallib.starts_with(b"MTLB"));
    assert!(pixel.metallib.starts_with(b"MTLB"));

    let vertex_layout = translated_layout(&vertex.reflection, ReflectedStage::Vertex);
    let pixel_layout = translated_layout(&pixel.reflection, ReflectedStage::Fragment);

    // The translated root signature is what the slot map claims: constants at
    // space 0, root CBV at space 1, nothing in space 2.
    let mut vs = described_shader(HwStage::Vs, 4);
    vs.root_cbvs = vec![0];
    let ps = described_shader(HwStage::Ps, 0);
    let signature = translated_pipeline_root_signature(
        &[&vs, &ps],
        SignatureFlags {
            allow_input_assembler_input_layout: true,
            ..SignatureFlags::default()
        },
    )
    .expect("graphics root signature");
    let map = signature_map(&signature);
    assert_eq!(map.len(), 2);
    assert_eq!(map.slot(0).expect("slot 0").space(), 0);
    assert_eq!(map.slot(1).expect("slot 1").space(), 1);

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    transport
        .register_registry_texture(TARGET, rgba8_desc())
        .expect("colour target");

    let state = PipelineState {
        color_formats: vec![TextureFormat::Rgba8Unorm],
        depth_format: None,
        fragment_entry: Some("main".to_string()),
        ..Default::default()
    }
    .encode(RegistryPipelineStage::Graphics);
    transport
        .register_content_hashed_graphics_pipeline_stages(
            GRAPHICS_HANDLE,
            &[
                StageLibrary {
                    role: FunctionRole::Vertex,
                    metallib: &vertex.metallib,
                    entry: "main",
                },
                StageLibrary {
                    role: FunctionRole::Fragment,
                    metallib: &pixel.metallib,
                    entry: "main",
                },
            ],
            &state,
            PipelineAbi::graphics(Some(vertex_layout), Some(pixel_layout))
                .with_root_parameters(map),
        )
        .expect("graphics pipeline registration");

    let mut cbv_page = HostPage::new(PAGE_BYTES);
    cbv_page.write(
        0,
        &[0.0f32, 0.5, 0.0, 0.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    // SAFETY: the page is a live 16 KiB allocation that outlives the submission.
    let cbv_id = unsafe { transport.register_zerocopy(cbv_page.raw(), PAGE_BYTES) }
        .expect("CBV page registration");

    let mut snapshot = HashMap::new();
    snapshot.insert((1u64, 1u32), victory_metal::VmBufferId(TARGET));
    snapshot.insert((4u64, 1u32), cbv_id);

    let submission = SubmissionV3 {
        sequence: 9001,
        list: None,
        commands: vec![
            CommandV3::BeginRenderPass(BeginRenderPass {
                colors: vec![ColorAttachment {
                    allocation_id: 1,
                    generation: 1,
                    mip_level: 0,
                    slice: 0,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                }],
                depth_stencil: None,
            }),
            CommandV3::SetGraphicsPipeline(PipelineRef {
                pipeline_id: GRAPHICS_HANDLE,
            }),
            CommandV3::SetViewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: f64::from(EDGE),
                height: f64::from(EDGE),
                near: 0.0,
                far: 1.0,
            }),
            CommandV3::SetScissor(ScissorRect {
                x: 0,
                y: 0,
                width: EDGE,
                height: EDGE,
            }),
            CommandV3::SetPrimitiveTopology(PrimitiveTopology::TriangleList),
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 0,
                bytes: [0.25f32, 0.0, 0.0, 1.0]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            }),
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 1,
                allocation_id: 4,
                generation: 1,
                offset: 0,
                length: 16,
            }),
            CommandV3::Draw(DrawState {
                vertex_count: 3,
                instance_count: 1,
                start_vertex: 0,
                start_instance: 0,
            }),
            CommandV3::EndRenderPass,
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &snapshot)
        .expect("graphics submission");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    assert_eq!(stats.render_encoders, 1);
    transport.wait_for_gpu().expect("GPU completion");
    let pixels = transport
        .read_registered_texture_rgba8(TARGET)
        .expect("colour readback");
    assert_eq!(pixels.len(), (EDGE * EDGE) as usize);
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            *pixel,
            [64, 128, 0, 255],
            "pixel {index}: USER_DATA 0.25 + root CBV 0.5"
        );
    }
    transport.unregister_buffer(cbv_id).expect("unregister CBV");
    eprintln!(
        "VRAB graphics proof: {} pixels == [64,128,0,255] from translated VS/PS",
        EDGE * EDGE
    );
}

#[test]
#[ignore = "GPU hardware proof; run with --ignored --test-threads=1"]
fn translated_compute_reads_user_data_and_memory_space() {
    let Some((pool, _env)) = pool() else {
        return;
    };
    let compute_dxil = unhex(COMPUTE_DXIL_HEX);
    let compute = pool
        .compile_with_reflection(&CompileJob {
            id: 3,
            stage: 8,
            entry: "main",
            dxil: &compute_dxil,
        })
        .expect("translated compute stage");
    assert!(compute.metallib.starts_with(b"MTLB"));
    let layout = translated_layout(&compute.reflection, ReflectedStage::Compute);
    assert_eq!(layout.parameters().len(), 4);

    let mut cs = described_shader(HwStage::Cs, 1);
    cs.root_uavs = vec![0];
    cs.memory = true;
    cs.view_addresses = true;
    let signature = translated_root_signature(&cs).expect("compute root signature");
    let map = signature_map(&signature);
    assert_eq!(map.len(), 4);
    assert_eq!(map.slot(0).expect("slot 0").space(), 0);
    assert_eq!(map.slot(1).expect("slot 1").space(), 1);
    assert_eq!(map.slot(2).expect("slot 2").space(), 2);
    assert_eq!(map.slot(3).expect("slot 3").space(), 2);

    let transport = DirectMetalTransport::new().expect("Metal device should open");
    let state = PipelineState::default().encode(RegistryPipelineStage::Compute);
    transport
        .register_content_hashed_compute_pipeline_stages(
            COMPUTE_HANDLE,
            &[StageLibrary {
                role: FunctionRole::Kernel,
                metallib: &compute.metallib,
                entry: "main",
            }],
            &state,
            PipelineAbi::compute(layout).with_root_parameters(map),
        )
        .expect("compute pipeline registration");

    let mut output = HostPage::new(PAGE_BYTES);
    output.write(0, &0.0f32.to_le_bytes());
    let mut memory = HostPage::new(PAGE_BYTES);
    memory.write(0, &7.5f32.to_le_bytes());
    // SAFETY: both pages are live 16 KiB allocations that outlive the submission.
    let output_id =
        unsafe { transport.register_zerocopy(output.raw(), PAGE_BYTES) }.expect("output UAV page");
    // SAFETY: same zero-copy contract.
    let memory_id =
        unsafe { transport.register_zerocopy(memory.raw(), PAGE_BYTES) }.expect("memory SRV page");

    let mut snapshot = HashMap::new();
    snapshot.insert((1u64, 1u32), output_id);
    snapshot.insert((2u64, 1u32), memory_id);

    let submission = SubmissionV3 {
        sequence: 9002,
        list: None,
        commands: vec![
            CommandV3::SetComputePipeline(PipelineRef {
                pipeline_id: COMPUTE_HANDLE,
            }),
            // Slot 0: USER_DATA_SPACE b0 (the addend).
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 0,
                bytes: 2.5f32.to_le_bytes().to_vec(),
            }),
            // Slot 1: ROOT_CBV_SPACE u0, the exact-value output.
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 1,
                allocation_id: 1,
                generation: 1,
                offset: 0,
                length: 4,
            }),
            // Slot 2: MEMORY_SPACE t0, supplied by the test.
            CommandV3::SetRootBufferView(RootBufferView {
                slot: 2,
                allocation_id: 2,
                generation: 1,
                offset: 0,
                length: 4,
            }),
            // Slot 3 (MEMORY_SPACE t1) is deliberately left unbound: the host
            // must serve a zero-filled default and count it.
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: COMPUTE_HANDLE,
                threads: 1,
                threads_per_group: 1,
                bindings: Vec::new(),
            }),
        ],
    };
    let (reply, stats) = transport
        .submit_v3_with_stats(&submission, &snapshot)
        .expect("compute submission");
    assert_eq!(reply.status, victory_metal::wire::ReplyStatus::Ok);
    assert_eq!(stats.compute_encoders, 1);
    transport.wait_for_gpu().expect("GPU completion");

    assert_eq!(
        output.read_f32(0),
        10.0,
        "7.5 (MEMORY_SPACE) + 2.5 (USER_DATA_SPACE) + 0.0 (default MEMORY_SPACE)"
    );
    assert_eq!(
        transport.memory_space_counters(),
        (1, 8),
        "one unbound MEMORY_SPACE parameter got one zero-filled 8-byte buffer"
    );

    // An argument naming a slot the layout does not declare is refused, and
    // counted, instead of silently continuing.
    let before = transport.abi_refusal_count();
    let mut other = HostPage::new(PAGE_BYTES);
    other.write(0, &0.0f32.to_le_bytes());
    // SAFETY: the page is live for the duration of the refused submission.
    let other_id = unsafe { transport.register_zerocopy(other.raw(), PAGE_BYTES) }
        .expect("second output page");
    let mut refused_snapshot = HashMap::new();
    refused_snapshot.insert((1u64, 1u32), other_id);
    let refused = SubmissionV3 {
        sequence: 9003,
        list: None,
        commands: vec![
            CommandV3::SetComputePipeline(PipelineRef {
                pipeline_id: COMPUTE_HANDLE,
            }),
            CommandV3::SetRootConstants(RootConstantsV3 {
                slot: 9,
                bytes: 1.0f32.to_le_bytes().to_vec(),
            }),
            CommandV3::Dispatch1d(Dispatch1d {
                pipeline_id: COMPUTE_HANDLE,
                threads: 1,
                threads_per_group: 1,
                bindings: Vec::new(),
            }),
        ],
    };
    let error = transport
        .submit_v3_with_stats(&refused, &refused_snapshot)
        .expect_err("an undeclared root argument must be refused");
    let text = error.to_string();
    assert!(
        text.contains("root parameter") || text.contains("root argument"),
        "bounded refusal names the root argument: {text}"
    );
    assert_eq!(transport.abi_refusal_count(), before + 1);

    transport
        .unregister_buffer(other_id)
        .expect("unregister other page");
    transport
        .unregister_buffer(memory_id)
        .expect("unregister memory page");
    transport
        .unregister_buffer(output_id)
        .expect("unregister output page");
    eprintln!(
        "VRAB compute proof: translated CS wrote 10.0 from USER_DATA + MEMORY_SPACE, default MEMORY_SPACE counted"
    );
}
