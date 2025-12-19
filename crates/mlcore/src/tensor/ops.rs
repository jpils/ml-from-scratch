use crate::tensor::Tensor2;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

impl<'a> Add<&'a Tensor2> for &Tensor2 {
    type Output = Tensor2;

    fn add(self, rhs: &'a Tensor2) -> Self::Output {
        let shape = self.shape();
        assert!(self.shape() == rhs.shape(), "Shape of rhs and lhs do not match!");

        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);
        for (val1, val2) in self.inner.iter().zip(rhs.inner.iter()) {
            vec.push(val1+val2);
        }

        Tensor2::from_vec(shape, vec).unwrap()
    }
}

impl Add<f32> for &Tensor2 {
    type Output = Tensor2;

    fn add(self, rhs: f32) -> Self::Output {
        let shape = self.shape();
        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);
        
        for val in self.inner.iter() {
            vec.push(val + rhs);
        }

        Tensor2::from_vec(shape, vec).unwrap()
    }
}

impl<'a> AddAssign<&'a Tensor2> for Tensor2 {
    fn add_assign(&mut self, rhs: &'a Tensor2) {
        assert!(self.shape() == rhs.shape(), "Shape of rhs and lhs do not match!");

        for (val1, val2) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *val1 += val2;
        }
    }
}

impl AddAssign<f32> for Tensor2 {
    fn add_assign(&mut self, rhs: f32) {
        for val in self.inner.iter_mut() {
            *val += rhs; 
        }
    }
}

impl<'a> Sub<&'a Tensor2> for &Tensor2 {
    type Output = Tensor2;

    fn sub(self, rhs: &'a Tensor2) -> Self::Output {
        let shape = self.shape();
        assert!(shape == rhs.shape(), "Shape of rhs and lhs do not match!");

        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);
        for (val1, val2) in self.inner.iter().zip(rhs.inner.iter()) {
            vec.push(val1 - val2);
        }

        Tensor2::from_vec(shape, vec).unwrap()
    }
}

impl Sub<f32> for &Tensor2 {
    type Output = Tensor2;

    fn sub(self, rhs: f32) -> Self::Output {
        let shape = self.shape();

        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);
        for val in self.inner.iter() {
            vec.push(val - rhs);
        }

        Tensor2::from_vec(shape, vec).unwrap()
    }
}

impl<'a> SubAssign<&'a Tensor2> for Tensor2 {
    fn sub_assign(&mut self, rhs: &'a Tensor2) {
        assert!(self.shape() == rhs.shape(), "Shape of rhs and lhs do not match!");

        for (val1, val2) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *val1 -= val2;
        }
    }
}

impl SubAssign<f32> for Tensor2 {
    fn sub_assign(&mut self, rhs: f32) {
        for val in self.inner.iter_mut() {
            *val -= rhs;
        }
    }
}

impl<'a> Mul<&'a Tensor2> for &Tensor2 {
    type Output = Tensor2;

    fn mul(self, rhs: &'a Tensor2) -> Self::Output {
        let shape = self.shape();
        assert!(shape == rhs.shape(), "Shape of rhs and lhs do not match!");

        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);
        for (val1, val2) in self.inner.iter().zip(rhs.inner.iter()) {
            vec.push(val1*val2);
        }

        Tensor2::from_vec(rhs.shape(), vec).unwrap()
    }
}

impl Mul<f32> for &Tensor2 {
    type Output = Tensor2;

    fn mul(self, rhs: f32) -> Self::Output {
        let shape = self.shape();
        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);

        for val in self.inner.iter() {
            vec.push(val*rhs);
        }   

        Tensor2::from_vec(shape, vec).unwrap()
    }
}

impl<'a> MulAssign<&'a Tensor2> for Tensor2 { 
    fn mul_assign(&mut self, rhs: &'a Tensor2) {
        for (val1, val2) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *val1 *= val2;
        }
    }
}

impl MulAssign<f32> for Tensor2 {
    fn mul_assign(&mut self, rhs: f32) {
        for val in self.inner.iter_mut() {
            *val *= rhs;
        }
    }
}

impl<'a> Div<&'a Tensor2> for &Tensor2 {
    type Output = Tensor2;

    fn div(self, rhs: &'a Tensor2) -> Self::Output {
        let shape = self.shape();
        assert!(shape == rhs.shape(), "Shape of rhs and lhs do not match!");

        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);
        for (val1, val2) in self.inner.iter().zip(rhs.inner.iter()) {
            vec.push(val1/val2);
        }

        Tensor2::from_vec(rhs.shape(), vec).unwrap()
    }
}

impl Div<f32> for &Tensor2 {
    type Output = Tensor2;

    fn div(self, rhs: f32) -> Self::Output {
        let shape = self.shape();
        let mut vec: Vec<f32> = Vec::with_capacity(shape.0*shape.1);

        for val in self.inner.iter() {
            vec.push(val/rhs);
        }   

        Tensor2::from_vec(shape, vec).unwrap()
    }
}

impl<'a> DivAssign<&'a Tensor2> for Tensor2 {
    fn div_assign(&mut self, rhs: &'a Tensor2) {
        for (val1, val2) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *val1 /= val2;
        }

    }
}

impl DivAssign<f32> for Tensor2 {
    fn div_assign(&mut self, rhs: f32) {
        for val in self.inner.iter_mut() {
            *val /= rhs;
        }

    }
}

#[cfg(test)]
mod test {
    use crate::tensor::*;

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
    fn test_add() {
        let tensor1 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor2 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            (&tensor1+&tensor2), 
            Tensor2::from_vec((2, 3), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap()
        )
    }

    #[test]
    #[should_panic]
    fn test_add_panic() {
        let tensor1 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor2 = Tensor2::from_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            (&tensor1+&tensor2), 
            Tensor2::from_vec((2, 3), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap()
        )
    }

    #[test]
    fn test_add_assign() {
        let mut tensor1 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor2 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        tensor1 += &tensor2; 

        assert_eq!(
            tensor1,
            Tensor2::from_vec((2, 3), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap()
        )
    }

    #[test]
    #[should_panic]
    fn test_add_assign_panic() {
        let mut tensor1 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor2 = Tensor2::from_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

        tensor1 += &tensor2; 
    }

    #[test]
    fn test_add_f32() {
        let tensor1 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let num: f32 = 0.5;

        assert_eq!(
            (&tensor1+num), 
            Tensor2::from_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap()
        )
    }

    #[test]
    fn test_add_assign_f32() {
        let mut tensor1 = Tensor2::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let num: f32 = 0.5;

        tensor1 += num; 

        assert_eq!(
            tensor1,
            Tensor2::from_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap()
        )
    }

    #[test]
    fn test_elem_mul() {
        let tensor1 = Tensor2::from_vec((2, 3), vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]).unwrap();
        let tensor2 = Tensor2::from_vec((2, 3), vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).unwrap();

        assert_eq!(
            (&tensor1*&tensor2),
            Tensor2::from_vec((2, 3), vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]).unwrap()
        )
    }

    #[test]
    fn test_scalar_mul() {
        let tensor1 = Tensor2::from_vec((2, 3), vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]).unwrap();
        let num:f32 = 2.0;

        assert_eq!(
            (&tensor1*num),
            Tensor2::from_vec((2, 3), vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]).unwrap()
        )
    }

    #[test]
    fn test_elem_mul_assign() {
        let mut tensor1 = Tensor2::from_vec((2, 3), vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]).unwrap();
        let tensor2 = Tensor2::from_vec((2, 3), vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).unwrap();

        tensor1 *= &tensor2;

        assert_eq!(
            tensor1,
            Tensor2::from_vec((2, 3), vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]).unwrap()
        )
    }

    #[test]
    fn test_scalar_mul_assign() {
        let mut tensor1 = Tensor2::from_vec((2, 3), vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]).unwrap();
        let num:f32 = 2.0;

        tensor1 *= num;

        assert_eq!(
            tensor1,
            Tensor2::from_vec((2, 3), vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]).unwrap()
        )
    }
}
