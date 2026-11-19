# Inside a .wpack file

An affine group in a 4-bit MLX checkpoint has 64 weight codes in the measured model. The codes occupy 32 bytes. Two more 16-bit values describe how to turn a code into a weight. Nearby groups often share exactly the same scale and bias.

Weightpack makes one sorted table per matrix. Each entry holds the original 16 bits of the scale and the original 16 bits of the bias. The index of an entry usually takes 10 to 13 bits on the measured Qwen. Groups can share an index only when both bit patterns match, including the sign of zero.

## Layout, version 1

The container is an uncompressed ZIP. Every archive has a JSON manifest and two binary members per packed matrix. Matrix numbers follow the order in the manifest. Binary words use little-endian byte order.

| Member | Contents |
| --- | --- |
| `manifest.json` | Format name, version, source matrix names, metadata and weight shapes, dtypes, entry counts, checksums and size summary |
| `matrices/000000.table` | Sorted 32-bit words. Bits 31 through 16 hold the scale. Bits 15 through 0 hold the bias. |
| `matrices/000000.ids` | Matrix indices in row-major order, least significant bit first. The stream ends on a 32-bit boundary followed by a zero word. |

Index width is `max(1, (entries - 1).bit_length())`. The extra word lets a GPU read an index across a word boundary at the end of a matrix. A matrix with more than 65,536 distinct pairs, or one for which the table and indices would take at least the original space, keeps its stock representation.

For example, 64 groups alternating between two pairs use 256 bytes of stock metadata. The two table entries take 8 bytes, the one-bit indices take 8 bytes, and the final zero word takes 4. The packed payload is 20 bytes.

The source checksum is SHA-256 over the scale bytes followed by the bias bytes. The payload checksum is SHA-256 over the table bytes followed by the index bytes. The reader validates sizes and checksums, reconstructs the pairs, and compares their bits against the supplied checkpoint. Verification covers the scale and bias arrays. The weight codes and tokenizer come from the supplied checkpoint.

## Metal execution

The loader replaces matching MLX affine linear and embedding modules. Embedding lookup fetches metadata for the requested vocabulary rows. Linear layers can expand one matrix's scales and biases and call MLX's quantized matrix multiply. Tied embeddings also support the output projection.

The direct kernels were checked on an M1 Max with MLX 0.32.2, FP16 input and 4-bit groups of 64. The small tiles accept 7 to 16 input rows at input width 5,120 and output widths 10,240, 12,288 or 17,408. One input row at output width 17,408 has a separate dot product. Other shapes and devices use expansion and stock MLX arithmetic. The row and device checks live alongside the kernel. [Dispatch](../src/weightpack/tiles.py) · [Metal tile](../src/weightpack/metal/tile.metal)

The CPU archive commands handle paired FP16 and BF16 metadata in 2D MLX matrices. The generation path supports standard MLX-LM affine layers whose names can be matched to the checkpoint. A model with custom quantized modules can still be scanned and packed. Loading an incompatible module reports its name.
