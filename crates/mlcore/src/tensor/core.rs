use std::{usize, vec};

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch {
        expected: (usize, usize),
        got: (usize, usize)
    },

    ShapeMismatchFlat {
        expected: usize,
        got: usize
    },
} 

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor2 {
    pub(crate) dim: (usize, usize),
    pub(crate) inner: Vec<f32>,
}

impl Tensor2 {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor2 { dim: (rows, cols), inner: vec![0.0; rows*cols] }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.dim
    }

    pub fn from_vec(shape: (usize, usize), data: Vec<f32>) -> Result<Tensor2, TensorError> {
        if data.len() != shape.0*shape.1 {
            return Err(TensorError::ShapeMismatchFlat { expected: shape.0*shape.1, got: data.len() })
        } 

        Ok(Tensor2 { dim: shape, inner: data })
    }

    pub fn matmul(&self, rhs: &Tensor2) -> Result<Tensor2, TensorError> {
        let shape_self = self.shape();
        let shape_rhs = rhs.shape();

        if shape_self.1 != shape_rhs.0 {
            return Err(TensorError::ShapeMismatch { expected: shape_self, got: shape_rhs });
        }

        let mut c_arr: Vec<f32> = vec![0.0; shape_self.0*shape_rhs.1];

        for i in 0..shape_self.0 {
            for j in 0..shape_rhs.1 {
                let c_idx = j + i*shape_rhs.1;
                for k in 0..shape_self.1 {
                    let a_idx = k + i*shape_self.1;
                    let b_idx = j + k*shape_rhs.1;

                    c_arr[c_idx] += self.inner[a_idx] * rhs.inner[b_idx];
                }
            }
        }

        Ok(Tensor2 { dim: (shape_self.0, shape_rhs.1), inner: c_arr })
    }

    pub fn transpose(&self) -> Tensor2 {
        if self.dim.0 == 1 || self.dim.1 == 1 {
            return Tensor2 { dim: (self.dim.1, self.dim.0), inner: self.inner.clone() };
        }

        let mut vec: Vec<f32> = self.inner.clone();

        for i in 1..self.dim.0 {
            for j in 1..self.dim.1 {
                let cur_idx = dbg!(self.get_idx(i, j));
                let new_idx = dbg!(self.get_idx(j, i));

                vec.swap(cur_idx, new_idx);
            }
        }

        Tensor2 { dim: (self.dim.1, self.dim.0), inner: vec }
    }
}

impl Tensor2 {
    #[inline(always)]
    fn get_idx(&self, i: usize, j: usize) -> usize {
        j + i * self.dim.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new() {
        let new = Tensor2::zeros(3, 10);

        assert_eq!(new.shape(), (3, 10));
    }

    #[test]
    fn test_from_vec_ok() {
        let vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor2::from_vec((2, 3), vec);

        assert!(tensor.is_ok());
        assert_eq!(tensor.unwrap().shape(), (2, 3))
    }

    #[test]
    fn test_from_vec_err() {
        let vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor2::from_vec((2, 5), vec);

        let err = TensorError::ShapeMismatchFlat { expected: 10, got: 6 };

        assert!(tensor.is_err());
        assert_eq!(tensor.unwrap_err(), err);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let b = Tensor2::from_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let c = a.matmul(&b);

        assert_eq!(
            c.unwrap(),
            Tensor2::from_vec((2, 2), vec![
                58.0, 64.0,
                139.0, 154.0
            ]).unwrap()
        );
    }

    #[test]
    fn test_transpose() {
        let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor1 = Tensor2::from_vec((2,3), data1).unwrap();

        let data2: Vec<f32> = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let tensor2 = Tensor2::from_vec((3,2), data2).unwrap();

        assert_eq!(tensor1.transpose(), tensor2);
    }
}
