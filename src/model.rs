use alloc::vec::Vec;
use burn::{
    backend::Flex,
    tensor::{activation::softmax, Bool, IndexingUpdateOp, Tensor, TensorData},
};
use burn_flex::FlexDevice;

const DIM: usize = 256;
const HEAD: usize = 64;
const LAYERS: usize = 8;
const KERNEL: usize = 5;
const TOP_K: usize = 4;
pub const VOCAB: usize = 256;
pub const CONTEXT: usize = 16;

type T = Tensor<Flex, 2>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidModel,
    InvalidParameter,
    InvalidTrit,
    InvalidContext,
    Incomplete,
    InvalidOutput,
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let (head, tail) = self
            .bytes
            .split_at_checked(len)
            .ok_or(Error::InvalidModel)?;
        self.bytes = tail;
        Ok(head)
    }

    fn floats(&mut self, count: usize) -> Result<&'a [u8], Error> {
        let bytes = self.take(count * 4)?;
        if !bytes.chunks_exact(4).all(|b| read_float(b).is_finite()) {
            return Err(Error::InvalidParameter);
        }
        Ok(bytes)
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Result<Matrix<'a>, Error> {
        let scale = read_float(self.floats(1)?);
        if scale < 0.0 {
            return Err(Error::InvalidParameter);
        }
        let packed = self.take((rows * cols).div_ceil(4))?;
        for byte in packed {
            for shift in [0, 2, 4, 6] {
                if (byte >> shift) & 3 == 2 {
                    return Err(Error::InvalidTrit);
                }
            }
        }
        Ok(Matrix {
            packed,
            scale,
            rows,
            cols,
        })
    }

    fn norm(&mut self) -> Result<Norm<'a>, Error> {
        Ok(Norm {
            gamma: self.floats(DIM)?,
            beta: self.floats(DIM)?,
        })
    }
}

fn read_float(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn tensor(values: Vec<f32>, shape: [usize; 2]) -> T {
    T::from_data(TensorData::new(values, shape), &FlexDevice)
}

fn vector(bytes: &[u8]) -> T {
    tensor(bytes.chunks_exact(4).map(read_float).collect(), [1, DIM])
}

struct Matrix<'a> {
    packed: &'a [u8],
    scale: f32,
    rows: usize,
    cols: usize,
}

impl Matrix<'_> {
    fn weight(&self, row: usize, col: usize) -> f32 {
        let index = row * self.cols + col;
        match (self.packed[index / 4] >> ((index % 4) * 2)) & 3 {
            1 => self.scale,
            3 => -self.scale,
            _ => 0.0,
        }
    }

    fn project(&self, input: T) -> T {
        let [seq, _] = input.dims();
        let mut output = T::zeros([seq, self.rows], &FlexDevice);
        // Flex's GEMM scratch exceeds RP2040 SRAM. Burn broadcast/reduce avoids it.
        for row in 0..self.rows {
            let weights = tensor(
                (0..self.cols).map(|col| self.weight(row, col)).collect(),
                [1, self.cols],
            );
            let projected = (input.clone() * weights).sum_dim(1);
            output = output.slice_assign([0..seq, row..row + 1], projected);
        }
        output
    }
}

fn bounded_matmul(lhs: T, rhs: T) -> T {
    let [seq, inner] = lhs.dims();
    let [_, cols] = rhs.dims();
    let mut output = T::zeros([seq, cols], &FlexDevice);
    for col in 0..cols {
        let column = rhs.clone().slice([0..inner, col..col + 1]).transpose();
        let product = (lhs.clone() * column).sum_dim(1);
        output = output.slice_assign([0..seq, col..col + 1], product);
    }
    output
}

struct Norm<'a> {
    gamma: &'a [u8],
    beta: &'a [u8],
}

impl Norm<'_> {
    fn forward(&self, input: T) -> T {
        let centered = input.clone() - input.mean_dim(1);
        let variance = (centered.clone() * centered.clone()).mean_dim(1);
        centered / (variance + 1.0e-5).sqrt() * vector(self.gamma) + vector(self.beta)
    }
}

struct Block<'a> {
    norm: Norm<'a>,
    conv: Matrix<'a>,
    a: &'a [u8],
    b: &'a [u8],
    c: &'a [u8],
    query: Matrix<'a>,
    key: Matrix<'a>,
    value: Matrix<'a>,
    router: Matrix<'a>,
}

impl Block<'_> {
    fn local(&self, input: T) -> T {
        let [seq, _] = input.dims();
        let mut output = T::zeros([seq, DIM], &FlexDevice);
        for lag in 0..KERNEL.min(seq) {
            let weights = tensor(
                (0..DIM).map(|c| self.conv.weight(c, lag)).collect(),
                [1, DIM],
            );
            let source = input.clone().slice([0..seq - lag, 0..DIM]);
            let previous = output.clone().slice([lag..seq, 0..DIM]);
            output = output.slice_assign([lag..seq, 0..DIM], previous + source * weights);
        }
        output
    }

    fn state(&self, input: T) -> T {
        let [seq, _] = input.dims();
        let a = vector(self.a).tanh();
        let b = vector(self.b);
        let c = vector(self.c);
        let mut hidden = T::zeros([1, DIM], &FlexDevice);
        let mut output = T::zeros([seq, DIM], &FlexDevice);
        for t in 0..seq {
            hidden = a.clone() * hidden + b.clone() * input.clone().slice([t..t + 1, 0..DIM]);
            output = output.slice_assign([t..t + 1, 0..DIM], c.clone() * hidden.clone());
        }
        output
    }

    fn sparse(&self, input: T) -> T {
        let [seq, _] = input.dims();
        let q = self.query.project(input.clone());
        let k = self.key.project(input.clone());
        let v = self.value.project(input);
        let scores = bounded_matmul(q, k.transpose()) / 8.0; // sqrt(HEAD) = 8
        let mask: Vec<bool> = (0..seq)
            .flat_map(|row| (0..seq).map(move |col| col > row))
            .collect();
        let mask =
            Tensor::<Flex, 2, Bool>::from_data(TensorData::new(mask, [seq, seq]), &FlexDevice);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        let (values, indices) = scores.topk_with_indices(TOP_K.min(seq), 1);
        let probabilities = softmax(values, 1);
        let attention = T::zeros([seq, seq], &FlexDevice).scatter(
            1,
            indices,
            probabilities,
            IndexingUpdateOp::Add,
        );
        bounded_matmul(attention, v)
    }

    fn forward(&self, input: T) -> T {
        let [seq, _] = input.dims();
        let h = self.norm.forward(input.clone());
        let routes = softmax(self.router.project(h.clone()), 1);
        // Accumulate pathways one at a time to avoid retaining all three outputs.
        let mixed = self.local(h.clone()) * routes.clone().slice([0..seq, 0..1]);
        let mixed = mixed + self.state(h.clone()) * routes.clone().slice([0..seq, 1..2]);
        let mixed = mixed + self.sparse(h) * routes.slice([0..seq, 2..3]);
        input + mixed
    }
}

/// Published 944,640-parameter Atome model, backed by a borrowed ATOME01 blob.
pub struct Atome<'a> {
    embed: Matrix<'a>,
    blocks: Vec<Block<'a>>,
    final_norm: Norm<'a>,
    unembed: Matrix<'a>,
}

impl<'a> Atome<'a> {
    pub fn load(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut reader = Reader { bytes };
        if reader.take(7)? != b"ATOME01" {
            return Err(Error::InvalidModel);
        }
        let embed = reader.matrix(VOCAB, DIM)?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            blocks.push(Block {
                norm: reader.norm()?,
                conv: reader.matrix(DIM, KERNEL)?,
                a: reader.floats(DIM)?,
                b: reader.floats(DIM)?,
                c: reader.floats(DIM)?,
                query: reader.matrix(HEAD, DIM)?,
                key: reader.matrix(HEAD, DIM)?,
                value: reader.matrix(DIM, DIM)?,
                router: reader.matrix(3, DIM)?,
            });
        }
        let final_norm = reader.norm()?;
        let unembed = reader.matrix(VOCAB, DIM)?;
        if !reader.bytes.is_empty() {
            return Err(Error::InvalidModel);
        }
        Ok(Self {
            embed,
            blocks,
            final_norm,
            unembed,
        })
    }

    pub fn start<'m>(&'m self, tokens: &[u8]) -> Result<Inference<'m, 'a>, Error> {
        if tokens.is_empty() || tokens.len() > CONTEXT {
            return Err(Error::InvalidContext);
        }
        let values = tokens
            .iter()
            .flat_map(|&token| (0..DIM).map(move |col| self.embed.weight(token as usize, col)))
            .collect();
        Ok(Inference {
            model: self,
            x: tensor(values, [tokens.len(), DIM]),
            next_layer: 0,
        })
    }
}

/// A full-window inference, advanced one block at a time for cooperative scheduling.
pub struct Inference<'m, 'a> {
    model: &'m Atome<'a>,
    x: T,
    next_layer: usize,
}

impl Inference<'_, '_> {
    /// Returns true when all blocks are complete. No state is carried between windows.
    pub fn step(&mut self) -> bool {
        if self.next_layer < LAYERS {
            self.x = self.model.blocks[self.next_layer].forward(self.x.clone());
            self.next_layer += 1;
        }
        self.next_layer == LAYERS
    }

    pub fn logits(self) -> Result<[f32; VOCAB], Error> {
        if self.next_layer != LAYERS {
            return Err(Error::Incomplete);
        }
        let [seq, _] = self.x.dims();
        let last = self.x.slice([seq - 1..seq, 0..DIM]);
        let last = self.model.final_norm.forward(last);
        let data = self.model.unembed.project(last).into_data();
        let values = data.as_slice::<f32>().map_err(|_| Error::InvalidOutput)?;
        let logits: [f32; VOCAB] = values.try_into().map_err(|_| Error::InvalidOutput)?;
        if !logits.iter().all(|x| x.is_finite()) {
            return Err(Error::InvalidOutput);
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MODEL_BYTES;
    use alloc::vec;

    #[test]
    fn decodes_trits_and_unaligned_floats() {
        let bytes = [9, 0, 0, 0, 63, 0b00_11_01_00];
        let mut reader = Reader { bytes: &bytes[1..] };
        let m = reader.matrix(1, 4).unwrap();
        assert_eq!(
            (0..4).map(|c| m.weight(0, c)).collect::<Vec<_>>(),
            vec![0.0, 0.5, -0.5, 0.0]
        );
    }

    #[test]
    fn rejects_malformed_models() {
        assert!(matches!(Atome::load(b"invalid"), Err(Error::InvalidModel)));
        assert!(Atome::load(&MODEL_BYTES[..MODEL_BYTES.len() - 1]).is_err());
        let mut bytes = MODEL_BYTES.to_vec();
        bytes.push(0);
        assert!(matches!(Atome::load(&bytes), Err(Error::InvalidModel)));
        let mut bytes = MODEL_BYTES.to_vec();
        bytes[7..11].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(Atome::load(&bytes), Err(Error::InvalidParameter)));
        bytes[7..11].copy_from_slice(&1.0f32.to_le_bytes());
        bytes[11] = 2;
        assert!(matches!(Atome::load(&bytes), Err(Error::InvalidTrit)));
    }

    #[test]
    fn rejects_invalid_context_and_incomplete_inference() {
        let model = Atome::load(MODEL_BYTES).unwrap();
        assert!(matches!(model.start(&[]), Err(Error::InvalidContext)));
        assert!(matches!(
            model.start(&[0; CONTEXT + 1]),
            Err(Error::InvalidContext)
        ));
        assert!(matches!(
            model.start(b"A").unwrap().logits(),
            Err(Error::Incomplete)
        ));
    }

    #[test]
    fn tiled_projection_matches_dense_burn_matmul() {
        let packed = [0b11_01_00_01; 15];
        let m = Matrix {
            packed: &packed,
            scale: 0.25,
            rows: 12,
            cols: 5,
        };
        let input = tensor((0..15).map(|i| i as f32 / 7.0).collect(), [3, 5]);
        let weights: Vec<f32> = (0..5)
            .flat_map(|c| (0..12).map(move |r| (r, c)))
            .map(|(r, c)| m.weight(r, c))
            .collect();
        let expected = input.clone().matmul(tensor(weights, [5, 12])).into_data();
        let actual = m.project(input).into_data();
        for (a, b) in actual
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(expected.as_slice::<f32>().unwrap())
        {
            assert!((a - b).abs() < 1.0e-6);
        }
    }
}
