import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1 : RowCertificate 1 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1 1) := by change Nat.Prime 7; exact ProjectEulerSolutions.P58PrimeNodes.p_7.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1 2) := by change Nat.Prime 5; exact ProjectEulerSolutions.P58PrimeNodes.p_5.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1 3) := by change Nat.Prime 3; exact ProjectEulerSolutions.P58PrimeNodes.p_3.prime
    exact decide_eq_true prime
theorem row_count_1 : row 1 = 3 := (row_1.checked).trans rfl
theorem model_count_1 : modelRow 1 = 3 := by decide
theorem modelrow_agrees_1 : modelRow 1 = row 1 := model_count_1.trans row_count_1.symm
def row_2 : RowCertificate 2 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2 1) := by change ¬ Nat.Prime 21; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 7) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2 2) := by change Nat.Prime 17; exact ProjectEulerSolutions.P58PrimeNodes.p_17.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2 3) := by change Nat.Prime 13; exact ProjectEulerSolutions.P58PrimeNodes.p_13.prime
    exact decide_eq_true prime
theorem row_count_2 : row 2 = 2 := (row_2.checked).trans rfl
theorem model_count_2 : modelRow 2 = 2 := by decide
theorem modelrow_agrees_2 : modelRow 2 = row 2 := model_count_2.trans row_count_2.symm
def row_3 : RowCertificate 3 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3 1) := by change Nat.Prime 43; exact ProjectEulerSolutions.P58PrimeNodes.p_43.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3 2) := by change Nat.Prime 37; exact ProjectEulerSolutions.P58PrimeNodes.p_37.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3 3) := by change Nat.Prime 31; exact ProjectEulerSolutions.P58PrimeNodes.p_31.prime
    exact decide_eq_true prime
theorem row_count_3 : row 3 = 3 := (row_3.checked).trans rfl
theorem model_count_3 : modelRow 3 = 3 := by decide
theorem modelrow_agrees_3 : modelRow 3 = row 3 := model_count_3.trans row_count_3.symm
def row_4 : RowCertificate 4 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4 1) := by change Nat.Prime 73; exact ProjectEulerSolutions.P58PrimeNodes.p_73.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4 2) := by change ¬ Nat.Prime 65; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4 3) := by change ¬ Nat.Prime 57; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4 : row 4 = 1 := (row_4.checked).trans rfl
theorem model_count_4 : modelRow 4 = 1 := by decide
theorem modelrow_agrees_4 : modelRow 4 = row 4 := model_count_4.trans row_count_4.symm
def row_5 : RowCertificate 5 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 5 1) := by change ¬ Nat.Prime 111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 37) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 5 2) := by change Nat.Prime 101; exact ProjectEulerSolutions.P58PrimeNodes.p_101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 5 3) := by change ¬ Nat.Prime 91; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_5 : row 5 = 1 := (row_5.checked).trans rfl
theorem model_count_5 : modelRow 5 = 1 := by decide
theorem modelrow_agrees_5 : modelRow 5 = row 5 := model_count_5.trans row_count_5.symm
def row_6 : RowCertificate 6 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6 1) := by change Nat.Prime 157; exact ProjectEulerSolutions.P58PrimeNodes.p_157.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6 2) := by change ¬ Nat.Prime 145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6 3) := by change ¬ Nat.Prime 133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 19) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6 : row 6 = 1 := (row_6.checked).trans rfl
theorem model_count_6 : modelRow 6 = 1 := by decide
theorem modelrow_agrees_6 : modelRow 6 = row 6 := model_count_6.trans row_count_6.symm
def row_7 : RowCertificate 7 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7 1) := by change Nat.Prime 211; exact ProjectEulerSolutions.P58PrimeNodes.p_211.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7 2) := by change Nat.Prime 197; exact ProjectEulerSolutions.P58PrimeNodes.p_197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7 3) := by change ¬ Nat.Prime 183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7 : row 7 = 2 := (row_7.checked).trans rfl
theorem model_count_7 : modelRow 7 = 2 := by decide
theorem modelrow_agrees_7 : modelRow 7 = row 7 := model_count_7.trans row_count_7.symm
def row_8 : RowCertificate 8 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8 1) := by change ¬ Nat.Prime 273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 91) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8 2) := by change Nat.Prime 257; exact ProjectEulerSolutions.P58PrimeNodes.p_257.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8 3) := by change Nat.Prime 241; exact ProjectEulerSolutions.P58PrimeNodes.p_241.prime
    exact decide_eq_true prime
theorem row_count_8 : row 8 = 2 := (row_8.checked).trans rfl
theorem model_count_8 : modelRow 8 = 2 := by decide
theorem modelrow_agrees_8 : modelRow 8 = row 8 := model_count_8.trans row_count_8.symm
def row_9 : RowCertificate 9 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9 1) := by change ¬ Nat.Prime 343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9 2) := by change ¬ Nat.Prime 325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9 3) := by change Nat.Prime 307; exact ProjectEulerSolutions.P58PrimeNodes.p_307.prime
    exact decide_eq_true prime
theorem row_count_9 : row 9 = 1 := (row_9.checked).trans rfl
theorem model_count_9 : modelRow 9 = 1 := by decide
theorem modelrow_agrees_9 : modelRow 9 = row 9 := model_count_9.trans row_count_9.symm
def row_10 : RowCertificate 10 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10 1) := by change Nat.Prime 421; exact ProjectEulerSolutions.P58PrimeNodes.p_421.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10 2) := by change Nat.Prime 401; exact ProjectEulerSolutions.P58PrimeNodes.p_401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10 3) := by change ¬ Nat.Prime 381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10 : row 10 = 2 := (row_10.checked).trans rfl
theorem model_count_10 : modelRow 10 = 2 := by decide
theorem modelrow_agrees_10 : modelRow 10 = row 10 := model_count_10.trans row_count_10.symm
def row_11 : RowCertificate 11 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11 1) := by change ¬ Nat.Prime 507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11 2) := by change ¬ Nat.Prime 485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 97) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 11 3) := by change Nat.Prime 463; exact ProjectEulerSolutions.P58PrimeNodes.p_463.prime
    exact decide_eq_true prime
theorem row_count_11 : row 11 = 1 := (row_11.checked).trans rfl
theorem model_count_11 : modelRow 11 = 1 := by decide
theorem modelrow_agrees_11 : modelRow 11 = row 11 := model_count_11.trans row_count_11.symm
def row_12 : RowCertificate 12 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12 1) := by change Nat.Prime 601; exact ProjectEulerSolutions.P58PrimeNodes.p_601.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12 2) := by change Nat.Prime 577; exact ProjectEulerSolutions.P58PrimeNodes.p_577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12 3) := by change ¬ Nat.Prime 553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 79) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12 : row 12 = 2 := (row_12.checked).trans rfl
theorem model_count_12 : modelRow 12 = 2 := by decide
theorem modelrow_agrees_12 : modelRow 12 = row 12 := model_count_12.trans row_count_12.symm
def row_13 : RowCertificate 13 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13 1) := by change ¬ Nat.Prime 703; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 37) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13 2) := by change Nat.Prime 677; exact ProjectEulerSolutions.P58PrimeNodes.p_677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13 3) := by change ¬ Nat.Prime 651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13 : row 13 = 1 := (row_13.checked).trans rfl
theorem model_count_13 : modelRow 13 = 1 := by decide
theorem modelrow_agrees_13 : modelRow 13 = row 13 := model_count_13.trans row_count_13.symm
def row_14 : RowCertificate 14 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 14 1) := by change ¬ Nat.Prime 813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 14 2) := by change ¬ Nat.Prime 785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 14 3) := by change Nat.Prime 757; exact ProjectEulerSolutions.P58PrimeNodes.p_757.prime
    exact decide_eq_true prime
theorem row_count_14 : row 14 = 1 := (row_14.checked).trans rfl
theorem model_count_14 : modelRow 14 = 1 := by decide
theorem modelrow_agrees_14 : modelRow 14 = row 14 := model_count_14.trans row_count_14.symm
def row_15 : RowCertificate 15 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 15 1) := by change ¬ Nat.Prime 931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 15 2) := by change ¬ Nat.Prime 901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 53) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 15 3) := by change ¬ Nat.Prime 871; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 67) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_15 : row 15 = 0 := (row_15.checked).trans rfl
theorem model_count_15 : modelRow 15 = 0 := by decide
theorem modelrow_agrees_15 : modelRow 15 = row 15 := model_count_15.trans row_count_15.symm
def row_16 : RowCertificate 16 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 16 1) := by change ¬ Nat.Prime 1057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 16 2) := by change ¬ Nat.Prime 1025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 16 3) := by change ¬ Nat.Prime 993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_16 : row 16 = 0 := (row_16.checked).trans rfl
theorem model_count_16 : modelRow 16 = 0 := by decide
theorem modelrow_agrees_16 : modelRow 16 = row 16 := model_count_16.trans row_count_16.symm
def row_17 : RowCertificate 17 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 17 1) := by change ¬ Nat.Prime 1191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 17 2) := by change ¬ Nat.Prime 1157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 89) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 17 3) := by change Nat.Prime 1123; exact ProjectEulerSolutions.P58PrimeNodes.p_1123.prime
    exact decide_eq_true prime
theorem row_count_17 : row 17 = 1 := (row_17.checked).trans rfl
theorem model_count_17 : modelRow 17 = 1 := by decide
theorem modelrow_agrees_17 : modelRow 17 = row 17 := model_count_17.trans row_count_17.symm
def row_18 : RowCertificate 18 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 18 1) := by change ¬ Nat.Prime 1333; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 43) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 18 2) := by change Nat.Prime 1297; exact ProjectEulerSolutions.P58PrimeNodes.p_1297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 18 3) := by change ¬ Nat.Prime 1261; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 97) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_18 : row 18 = 1 := (row_18.checked).trans rfl
theorem model_count_18 : modelRow 18 = 1 := by decide
theorem modelrow_agrees_18 : modelRow 18 = row 18 := model_count_18.trans row_count_18.symm
def row_19 : RowCertificate 19 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 19 1) := by change Nat.Prime 1483; exact ProjectEulerSolutions.P58PrimeNodes.p_1483.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 19 2) := by change ¬ Nat.Prime 1445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 19 3) := by change ¬ Nat.Prime 1407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_19 : row 19 = 1 := (row_19.checked).trans rfl
theorem model_count_19 : modelRow 19 = 1 := by decide
theorem modelrow_agrees_19 : modelRow 19 = row 19 := model_count_19.trans row_count_19.symm
def row_20 : RowCertificate 20 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 20 1) := by change ¬ Nat.Prime 1641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 20 2) := by change Nat.Prime 1601; exact ProjectEulerSolutions.P58PrimeNodes.p_1601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 20 3) := by change ¬ Nat.Prime 1561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_20 : row 20 = 1 := (row_20.checked).trans rfl
theorem model_count_20 : modelRow 20 = 1 := by decide
theorem modelrow_agrees_20 : modelRow 20 = row 20 := model_count_20.trans row_count_20.symm
def row_21 : RowCertificate 21 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 21 1) := by change ¬ Nat.Prime 1807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 21 2) := by change ¬ Nat.Prime 1765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 21 3) := by change Nat.Prime 1723; exact ProjectEulerSolutions.P58PrimeNodes.p_1723.prime
    exact decide_eq_true prime
theorem row_count_21 : row 21 = 1 := (row_21.checked).trans rfl
theorem model_count_21 : modelRow 21 = 1 := by decide
theorem modelrow_agrees_21 : modelRow 21 = row 21 := model_count_21.trans row_count_21.symm
def row_22 : RowCertificate 22 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 22 1) := by change ¬ Nat.Prime 1981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 22 2) := by change ¬ Nat.Prime 1937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 22 3) := by change ¬ Nat.Prime 1893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_22 : row 22 = 0 := (row_22.checked).trans rfl
theorem model_count_22 : modelRow 22 = 0 := by decide
theorem modelrow_agrees_22 : modelRow 22 = row 22 := model_count_22.trans row_count_22.symm
def row_23 : RowCertificate 23 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 23 1) := by change ¬ Nat.Prime 2163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 23 2) := by change ¬ Nat.Prime 2117; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 73) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 23 3) := by change ¬ Nat.Prime 2071; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_23 : row 23 = 0 := (row_23.checked).trans rfl
theorem model_count_23 : modelRow 23 = 0 := by decide
theorem modelrow_agrees_23 : modelRow 23 = row 23 := model_count_23.trans row_count_23.symm
def row_24 : RowCertificate 24 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 24 1) := by change ¬ Nat.Prime 2353; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 24 2) := by change ¬ Nat.Prime 2305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 24 3) := by change ¬ Nat.Prime 2257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 61) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_24 : row 24 = 0 := (row_24.checked).trans rfl
theorem model_count_24 : modelRow 24 = 0 := by decide
theorem modelrow_agrees_24 : modelRow 24 = row 24 := model_count_24.trans row_count_24.symm
def row_25 : RowCertificate 25 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 25 1) := by change Nat.Prime 2551; exact ProjectEulerSolutions.P58PrimeNodes.p_2551.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 25 2) := by change ¬ Nat.Prime 2501; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 61) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 25 3) := by change ¬ Nat.Prime 2451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_25 : row 25 = 1 := (row_25.checked).trans rfl
theorem model_count_25 : modelRow 25 = 1 := by decide
theorem modelrow_agrees_25 : modelRow 25 = row 25 := model_count_25.trans row_count_25.symm
def row_26 : RowCertificate 26 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 26 1) := by change ¬ Nat.Prime 2757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 26 2) := by change ¬ Nat.Prime 2705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 26 3) := by change ¬ Nat.Prime 2653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_26 : row 26 = 0 := (row_26.checked).trans rfl
theorem model_count_26 : modelRow 26 = 0 := by decide
theorem modelrow_agrees_26 : modelRow 26 = row 26 := model_count_26.trans row_count_26.symm
def row_27 : RowCertificate 27 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 27 1) := by change Nat.Prime 2971; exact ProjectEulerSolutions.P58PrimeNodes.p_2971.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 27 2) := by change Nat.Prime 2917; exact ProjectEulerSolutions.P58PrimeNodes.p_2917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 27 3) := by change ¬ Nat.Prime 2863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_27 : row 27 = 2 := (row_27.checked).trans rfl
theorem model_count_27 : modelRow 27 = 2 := by decide
theorem modelrow_agrees_27 : modelRow 27 = row 27 := model_count_27.trans row_count_27.symm
def row_28 : RowCertificate 28 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 28 1) := by change ¬ Nat.Prime 3193; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 28 2) := by change Nat.Prime 3137; exact ProjectEulerSolutions.P58PrimeNodes.p_3137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 28 3) := by change ¬ Nat.Prime 3081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_28 : row 28 = 1 := (row_28.checked).trans rfl
theorem model_count_28 : modelRow 28 = 1 := by decide
theorem modelrow_agrees_28 : modelRow 28 = row 28 := model_count_28.trans row_count_28.symm
def row_29 : RowCertificate 29 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 29 1) := by change ¬ Nat.Prime 3423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 29 2) := by change ¬ Nat.Prime 3365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 29 3) := by change Nat.Prime 3307; exact ProjectEulerSolutions.P58PrimeNodes.p_3307.prime
    exact decide_eq_true prime
theorem row_count_29 : row 29 = 1 := (row_29.checked).trans rfl
theorem model_count_29 : modelRow 29 = 1 := by decide
theorem modelrow_agrees_29 : modelRow 29 = row 29 := model_count_29.trans row_count_29.symm
def row_30 : RowCertificate 30 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 30 1) := by change ¬ Nat.Prime 3661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 30 2) := by change ¬ Nat.Prime 3601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 30 3) := by change Nat.Prime 3541; exact ProjectEulerSolutions.P58PrimeNodes.p_3541.prime
    exact decide_eq_true prime
theorem row_count_30 : row 30 = 1 := (row_30.checked).trans rfl
theorem model_count_30 : modelRow 30 = 1 := by decide
theorem modelrow_agrees_30 : modelRow 30 = row 30 := model_count_30.trans row_count_30.symm
def row_31 : RowCertificate 31 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 31 1) := by change Nat.Prime 3907; exact ProjectEulerSolutions.P58PrimeNodes.p_3907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 31 2) := by change ¬ Nat.Prime 3845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 31 3) := by change ¬ Nat.Prime 3783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_31 : row 31 = 1 := (row_31.checked).trans rfl
theorem model_count_31 : modelRow 31 = 1 := by decide
theorem modelrow_agrees_31 : modelRow 31 = row 31 := model_count_31.trans row_count_31.symm
def row_32 : RowCertificate 32 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 32 1) := by change ¬ Nat.Prime 4161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 32 2) := by change ¬ Nat.Prime 4097; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 32 3) := by change ¬ Nat.Prime 4033; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_32 : row 32 = 0 := (row_32.checked).trans rfl
theorem model_count_32 : modelRow 32 = 0 := by decide
theorem modelrow_agrees_32 : modelRow 32 = row 32 := model_count_32.trans row_count_32.symm
def row_33 : RowCertificate 33 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 33 1) := by change Nat.Prime 4423; exact ProjectEulerSolutions.P58PrimeNodes.p_4423.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 33 2) := by change Nat.Prime 4357; exact ProjectEulerSolutions.P58PrimeNodes.p_4357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 33 3) := by change ¬ Nat.Prime 4291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_33 : row 33 = 2 := (row_33.checked).trans rfl
theorem model_count_33 : modelRow 33 = 2 := by decide
theorem modelrow_agrees_33 : modelRow 33 = row 33 := model_count_33.trans row_count_33.symm
def row_34 : RowCertificate 34 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 34 1) := by change ¬ Nat.Prime 4693; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 34 2) := by change ¬ Nat.Prime 4625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 34 3) := by change ¬ Nat.Prime 4557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_34 : row 34 = 0 := (row_34.checked).trans rfl
theorem model_count_34 : modelRow 34 = 0 := by decide
theorem modelrow_agrees_34 : modelRow 34 = row 34 := model_count_34.trans row_count_34.symm
def row_35 : RowCertificate 35 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 35 1) := by change ¬ Nat.Prime 4971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 35 2) := by change ¬ Nat.Prime 4901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 35 3) := by change Nat.Prime 4831; exact ProjectEulerSolutions.P58PrimeNodes.p_4831.prime
    exact decide_eq_true prime
theorem row_count_35 : row 35 = 1 := (row_35.checked).trans rfl
theorem model_count_35 : modelRow 35 = 1 := by decide
theorem modelrow_agrees_35 : modelRow 35 = row 35 := model_count_35.trans row_count_35.symm
def row_36 : RowCertificate 36 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 36 1) := by change ¬ Nat.Prime 5257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 36 2) := by change ¬ Nat.Prime 5185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 36 3) := by change Nat.Prime 5113; exact ProjectEulerSolutions.P58PrimeNodes.p_5113.prime
    exact decide_eq_true prime
theorem row_count_36 : row 36 = 1 := (row_36.checked).trans rfl
theorem model_count_36 : modelRow 36 = 1 := by decide
theorem modelrow_agrees_36 : modelRow 36 = row 36 := model_count_36.trans row_count_36.symm
def row_37 : RowCertificate 37 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 37 1) := by change ¬ Nat.Prime 5551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 37 2) := by change Nat.Prime 5477; exact ProjectEulerSolutions.P58PrimeNodes.p_5477.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 37 3) := by change ¬ Nat.Prime 5403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_37 : row 37 = 1 := (row_37.checked).trans rfl
theorem model_count_37 : modelRow 37 = 1 := by decide
theorem modelrow_agrees_37 : modelRow 37 = row 37 := model_count_37.trans row_count_37.symm
def row_38 : RowCertificate 38 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 38 1) := by change ¬ Nat.Prime 5853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 38 2) := by change ¬ Nat.Prime 5777; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 38 3) := by change Nat.Prime 5701; exact ProjectEulerSolutions.P58PrimeNodes.p_5701.prime
    exact decide_eq_true prime
theorem row_count_38 : row 38 = 1 := (row_38.checked).trans rfl
theorem model_count_38 : modelRow 38 = 1 := by decide
theorem modelrow_agrees_38 : modelRow 38 = row 38 := model_count_38.trans row_count_38.symm
def row_39 : RowCertificate 39 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 39 1) := by change Nat.Prime 6163; exact ProjectEulerSolutions.P58PrimeNodes.p_6163.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 39 2) := by change ¬ Nat.Prime 6085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 39 3) := by change Nat.Prime 6007; exact ProjectEulerSolutions.P58PrimeNodes.p_6007.prime
    exact decide_eq_true prime
theorem row_count_39 : row 39 = 2 := (row_39.checked).trans rfl
theorem model_count_39 : modelRow 39 = 2 := by decide
theorem modelrow_agrees_39 : modelRow 39 = row 39 := model_count_39.trans row_count_39.symm
def row_40 : RowCertificate 40 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 40 1) := by change Nat.Prime 6481; exact ProjectEulerSolutions.P58PrimeNodes.p_6481.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 40 2) := by change ¬ Nat.Prime 6401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 40 3) := by change ¬ Nat.Prime 6321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_40 : row 40 = 1 := (row_40.checked).trans rfl
theorem model_count_40 : modelRow 40 = 1 := by decide
theorem modelrow_agrees_40 : modelRow 40 = row 40 := model_count_40.trans row_count_40.symm
def row_41 : RowCertificate 41 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 41 1) := by change ¬ Nat.Prime 6807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 41 2) := by change ¬ Nat.Prime 6725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 41 3) := by change ¬ Nat.Prime 6643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_41 : row 41 = 0 := (row_41.checked).trans rfl
theorem model_count_41 : modelRow 41 = 0 := by decide
theorem modelrow_agrees_41 : modelRow 41 = row 41 := model_count_41.trans row_count_41.symm
def row_42 : RowCertificate 42 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 42 1) := by change ¬ Nat.Prime 7141; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 42 2) := by change Nat.Prime 7057; exact ProjectEulerSolutions.P58PrimeNodes.p_7057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 42 3) := by change ¬ Nat.Prime 6973; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_42 : row 42 = 1 := (row_42.checked).trans rfl
theorem model_count_42 : modelRow 42 = 1 := by decide
theorem modelrow_agrees_42 : modelRow 42 = row 42 := model_count_42.trans row_count_42.symm
def row_43 : RowCertificate 43 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 43 1) := by change ¬ Nat.Prime 7483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 43 2) := by change ¬ Nat.Prime 7397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 43 3) := by change ¬ Nat.Prime 7311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_43 : row 43 = 0 := (row_43.checked).trans rfl
theorem model_count_43 : modelRow 43 = 0 := by decide
theorem modelrow_agrees_43 : modelRow 43 = row 43 := model_count_43.trans row_count_43.symm
def row_44 : RowCertificate 44 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 44 1) := by change ¬ Nat.Prime 7833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 44 2) := by change ¬ Nat.Prime 7745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 44 3) := by change ¬ Nat.Prime 7657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_44 : row 44 = 0 := (row_44.checked).trans rfl
theorem model_count_44 : modelRow 44 = 0 := by decide
theorem modelrow_agrees_44 : modelRow 44 = row 44 := model_count_44.trans row_count_44.symm
def row_45 : RowCertificate 45 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 45 1) := by change Nat.Prime 8191; exact ProjectEulerSolutions.P58PrimeNodes.p_8191.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 45 2) := by change Nat.Prime 8101; exact ProjectEulerSolutions.P58PrimeNodes.p_8101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 45 3) := by change Nat.Prime 8011; exact ProjectEulerSolutions.P58PrimeNodes.p_8011.prime
    exact decide_eq_true prime
theorem row_count_45 : row 45 = 3 := (row_45.checked).trans rfl
theorem model_count_45 : modelRow 45 = 3 := by decide
theorem modelrow_agrees_45 : modelRow 45 = row 45 := model_count_45.trans row_count_45.symm
def row_46 : RowCertificate 46 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 46 1) := by change ¬ Nat.Prime 8557; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 46 2) := by change ¬ Nat.Prime 8465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 46 3) := by change ¬ Nat.Prime 8373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_46 : row 46 = 0 := (row_46.checked).trans rfl
theorem model_count_46 : modelRow 46 = 0 := by decide
theorem modelrow_agrees_46 : modelRow 46 = row 46 := model_count_46.trans row_count_46.symm
def row_47 : RowCertificate 47 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 47 1) := by change ¬ Nat.Prime 8931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 47 2) := by change Nat.Prime 8837; exact ProjectEulerSolutions.P58PrimeNodes.p_8837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 47 3) := by change ¬ Nat.Prime 8743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_47 : row 47 = 1 := (row_47.checked).trans rfl
theorem model_count_47 : modelRow 47 = 1 := by decide
theorem modelrow_agrees_47 : modelRow 47 = row 47 := model_count_47.trans row_count_47.symm
def row_48 : RowCertificate 48 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 48 1) := by change ¬ Nat.Prime 9313; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 48 2) := by change ¬ Nat.Prime 9217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 48 3) := by change ¬ Nat.Prime 9121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_48 : row 48 = 0 := (row_48.checked).trans rfl
theorem model_count_48 : modelRow 48 = 0 := by decide
theorem modelrow_agrees_48 : modelRow 48 = row 48 := model_count_48.trans row_count_48.symm
def row_49 : RowCertificate 49 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 49 1) := by change ¬ Nat.Prime 9703; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 49 2) := by change ¬ Nat.Prime 9605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 49 3) := by change ¬ Nat.Prime 9507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_49 : row 49 = 0 := (row_49.checked).trans rfl
theorem model_count_49 : modelRow 49 = 0 := by decide
theorem modelrow_agrees_49 : modelRow 49 = row 49 := model_count_49.trans row_count_49.symm
def row_50 : RowCertificate 50 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 50 1) := by change ¬ Nat.Prime 10101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 50 2) := by change ¬ Nat.Prime 10001; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 50 3) := by change Nat.Prime 9901; exact ProjectEulerSolutions.P58PrimeNodes.p_9901.prime
    exact decide_eq_true prime
theorem row_count_50 : row 50 = 1 := (row_50.checked).trans rfl
theorem model_count_50 : modelRow 50 = 1 := by decide
theorem modelrow_agrees_50 : modelRow 50 = row 50 := model_count_50.trans row_count_50.symm
def row_51 : RowCertificate 51 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 51 1) := by change ¬ Nat.Prime 10507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 51 2) := by change ¬ Nat.Prime 10405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 51 3) := by change Nat.Prime 10303; exact ProjectEulerSolutions.P58PrimeNodes.p_10303.prime
    exact decide_eq_true prime
theorem row_count_51 : row 51 = 1 := (row_51.checked).trans rfl
theorem model_count_51 : modelRow 51 = 1 := by decide
theorem modelrow_agrees_51 : modelRow 51 = row 51 := model_count_51.trans row_count_51.symm
def row_52 : RowCertificate 52 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 52 1) := by change ¬ Nat.Prime 10921; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 52 2) := by change ¬ Nat.Prime 10817; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 52 3) := by change ¬ Nat.Prime 10713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_52 : row 52 = 0 := (row_52.checked).trans rfl
theorem model_count_52 : modelRow 52 = 0 := by decide
theorem modelrow_agrees_52 : modelRow 52 = row 52 := model_count_52.trans row_count_52.symm
def row_53 : RowCertificate 53 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 53 1) := by change ¬ Nat.Prime 11343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 53 2) := by change ¬ Nat.Prime 11237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 53 3) := by change Nat.Prime 11131; exact ProjectEulerSolutions.P58PrimeNodes.p_11131.prime
    exact decide_eq_true prime
theorem row_count_53 : row 53 = 1 := (row_53.checked).trans rfl
theorem model_count_53 : modelRow 53 = 1 := by decide
theorem modelrow_agrees_53 : modelRow 53 = row 53 := model_count_53.trans row_count_53.symm
def row_54 : RowCertificate 54 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 54 1) := by change ¬ Nat.Prime 11773; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 54 2) := by change ¬ Nat.Prime 11665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 54 3) := by change ¬ Nat.Prime 11557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_54 : row 54 = 0 := (row_54.checked).trans rfl
theorem model_count_54 : modelRow 54 = 0 := by decide
theorem modelrow_agrees_54 : modelRow 54 = row 54 := model_count_54.trans row_count_54.symm
def row_55 : RowCertificate 55 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 55 1) := by change Nat.Prime 12211; exact ProjectEulerSolutions.P58PrimeNodes.p_12211.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 55 2) := by change Nat.Prime 12101; exact ProjectEulerSolutions.P58PrimeNodes.p_12101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 55 3) := by change ¬ Nat.Prime 11991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_55 : row 55 = 2 := (row_55.checked).trans rfl
theorem model_count_55 : modelRow 55 = 2 := by decide
theorem modelrow_agrees_55 : modelRow 55 = row 55 := model_count_55.trans row_count_55.symm
def row_56 : RowCertificate 56 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 56 1) := by change ¬ Nat.Prime 12657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 56 2) := by change ¬ Nat.Prime 12545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 56 3) := by change Nat.Prime 12433; exact ProjectEulerSolutions.P58PrimeNodes.p_12433.prime
    exact decide_eq_true prime
theorem row_count_56 : row 56 = 1 := (row_56.checked).trans rfl
theorem model_count_56 : modelRow 56 = 1 := by decide
theorem modelrow_agrees_56 : modelRow 56 = row 56 := model_count_56.trans row_count_56.symm
def row_57 : RowCertificate 57 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 57 1) := by change ¬ Nat.Prime 13111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 57 2) := by change ¬ Nat.Prime 12997; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 57 3) := by change ¬ Nat.Prime 12883; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_57 : row 57 = 0 := (row_57.checked).trans rfl
theorem model_count_57 : modelRow 57 = 0 := by decide
theorem modelrow_agrees_57 : modelRow 57 = row 57 := model_count_57.trans row_count_57.symm
def row_58 : RowCertificate 58 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 58 1) := by change ¬ Nat.Prime 13573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 58 2) := by change Nat.Prime 13457; exact ProjectEulerSolutions.P58PrimeNodes.p_13457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 58 3) := by change ¬ Nat.Prime 13341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_58 : row 58 = 1 := (row_58.checked).trans rfl
theorem model_count_58 : modelRow 58 = 1 := by decide
theorem modelrow_agrees_58 : modelRow 58 = row 58 := model_count_58.trans row_count_58.symm
def row_59 : RowCertificate 59 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 59 1) := by change ¬ Nat.Prime 14043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 59 2) := by change ¬ Nat.Prime 13925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 59 3) := by change Nat.Prime 13807; exact ProjectEulerSolutions.P58PrimeNodes.p_13807.prime
    exact decide_eq_true prime
theorem row_count_59 : row 59 = 1 := (row_59.checked).trans rfl
theorem model_count_59 : modelRow 59 = 1 := by decide
theorem modelrow_agrees_59 : modelRow 59 = row 59 := model_count_59.trans row_count_59.symm
def row_60 : RowCertificate 60 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 60 1) := by change ¬ Nat.Prime 14521; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 60 2) := by change Nat.Prime 14401; exact ProjectEulerSolutions.P58PrimeNodes.p_14401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 60 3) := by change Nat.Prime 14281; exact ProjectEulerSolutions.P58PrimeNodes.p_14281.prime
    exact decide_eq_true prime
theorem row_count_60 : row 60 = 2 := (row_60.checked).trans rfl
theorem model_count_60 : modelRow 60 = 2 := by decide
theorem modelrow_agrees_60 : modelRow 60 = row 60 := model_count_60.trans row_count_60.symm
def row_61 : RowCertificate 61 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 61 1) := by change ¬ Nat.Prime 15007; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 61 2) := by change ¬ Nat.Prime 14885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 61 3) := by change ¬ Nat.Prime 14763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_61 : row 61 = 0 := (row_61.checked).trans rfl
theorem model_count_61 : modelRow 61 = 0 := by decide
theorem modelrow_agrees_61 : modelRow 61 = row 61 := model_count_61.trans row_count_61.symm
def row_62 : RowCertificate 62 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 62 1) := by change ¬ Nat.Prime 15501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 62 2) := by change Nat.Prime 15377; exact ProjectEulerSolutions.P58PrimeNodes.p_15377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 62 3) := by change ¬ Nat.Prime 15253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_62 : row 62 = 1 := (row_62.checked).trans rfl
theorem model_count_62 : modelRow 62 = 1 := by decide
theorem modelrow_agrees_62 : modelRow 62 = row 62 := model_count_62.trans row_count_62.symm
def row_63 : RowCertificate 63 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 63 1) := by change ¬ Nat.Prime 16003; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 63 2) := by change Nat.Prime 15877; exact ProjectEulerSolutions.P58PrimeNodes.p_15877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 63 3) := by change ¬ Nat.Prime 15751; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_63 : row 63 = 1 := (row_63.checked).trans rfl
theorem model_count_63 : modelRow 63 = 1 := by decide
theorem modelrow_agrees_63 : modelRow 63 = row 63 := model_count_63.trans row_count_63.symm
def row_64 : RowCertificate 64 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 64 1) := by change ¬ Nat.Prime 16513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 64 2) := by change ¬ Nat.Prime 16385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 64 3) := by change ¬ Nat.Prime 16257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_64 : row 64 = 0 := (row_64.checked).trans rfl
theorem model_count_64 : modelRow 64 = 0 := by decide
theorem modelrow_agrees_64 : modelRow 64 = row 64 := model_count_64.trans row_count_64.symm
def row_65 : RowCertificate 65 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 65 1) := by change ¬ Nat.Prime 17031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 65 2) := by change Nat.Prime 16901; exact ProjectEulerSolutions.P58PrimeNodes.p_16901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 65 3) := by change ¬ Nat.Prime 16771; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_65 : row 65 = 1 := (row_65.checked).trans rfl
theorem model_count_65 : modelRow 65 = 1 := by decide
theorem modelrow_agrees_65 : modelRow 65 = row 65 := model_count_65.trans row_count_65.symm
def row_66 : RowCertificate 66 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 66 1) := by change ¬ Nat.Prime 17557; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 66 2) := by change ¬ Nat.Prime 17425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 66 3) := by change Nat.Prime 17293; exact ProjectEulerSolutions.P58PrimeNodes.p_17293.prime
    exact decide_eq_true prime
theorem row_count_66 : row 66 = 1 := (row_66.checked).trans rfl
theorem model_count_66 : modelRow 66 = 1 := by decide
theorem modelrow_agrees_66 : modelRow 66 = row 66 := model_count_66.trans row_count_66.symm
def row_67 : RowCertificate 67 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 67 1) := by change ¬ Nat.Prime 18091; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 67 2) := by change Nat.Prime 17957; exact ProjectEulerSolutions.P58PrimeNodes.p_17957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 67 3) := by change ¬ Nat.Prime 17823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_67 : row 67 = 1 := (row_67.checked).trans rfl
theorem model_count_67 : modelRow 67 = 1 := by decide
theorem modelrow_agrees_67 : modelRow 67 = row 67 := model_count_67.trans row_count_67.symm
def row_68 : RowCertificate 68 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 68 1) := by change ¬ Nat.Prime 18633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 6211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 68 2) := by change ¬ Nat.Prime 18497; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 68 3) := by change ¬ Nat.Prime 18361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_68 : row 68 = 0 := (row_68.checked).trans rfl
theorem model_count_68 : modelRow 68 = 0 := by decide
theorem modelrow_agrees_68 : modelRow 68 = row 68 := model_count_68.trans row_count_68.symm
def row_69 : RowCertificate 69 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 69 1) := by change Nat.Prime 19183; exact ProjectEulerSolutions.P58PrimeNodes.p_19183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 69 2) := by change ¬ Nat.Prime 19045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 69 3) := by change ¬ Nat.Prime 18907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_69 : row 69 = 1 := (row_69.checked).trans rfl
theorem model_count_69 : modelRow 69 = 1 := by decide
theorem modelrow_agrees_69 : modelRow 69 = row 69 := model_count_69.trans row_count_69.symm
def row_70 : RowCertificate 70 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 70 1) := by change ¬ Nat.Prime 19741; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 70 2) := by change ¬ Nat.Prime 19601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 70 3) := by change ¬ Nat.Prime 19461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 6487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_70 : row 70 = 0 := (row_70.checked).trans rfl
theorem model_count_70 : modelRow 70 = 0 := by decide
theorem modelrow_agrees_70 : modelRow 70 = row 70 := model_count_70.trans row_count_70.symm
def row_71 : RowCertificate 71 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 71 1) := by change ¬ Nat.Prime 20307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 6769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 71 2) := by change ¬ Nat.Prime 20165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 4033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 71 3) := by change Nat.Prime 20023; exact ProjectEulerSolutions.P58PrimeNodes.p_20023.prime
    exact decide_eq_true prime
theorem row_count_71 : row 71 = 1 := (row_71.checked).trans rfl
theorem model_count_71 : modelRow 71 = 1 := by decide
theorem modelrow_agrees_71 : modelRow 71 = row 71 := model_count_71.trans row_count_71.symm
def row_72 : RowCertificate 72 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 72 1) := by change ¬ Nat.Prime 20881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2983) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 72 2) := by change ¬ Nat.Prime 20737; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 72 3) := by change Nat.Prime 20593; exact ProjectEulerSolutions.P58PrimeNodes.p_20593.prime
    exact decide_eq_true prime
theorem row_count_72 : row 72 = 1 := (row_72.checked).trans rfl
theorem model_count_72 : modelRow 72 = 1 := by decide
theorem modelrow_agrees_72 : modelRow 72 = row 72 := model_count_72.trans row_count_72.symm
def row_73 : RowCertificate 73 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 73 1) := by change ¬ Nat.Prime 21463; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 73 2) := by change Nat.Prime 21317; exact ProjectEulerSolutions.P58PrimeNodes.p_21317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 73 3) := by change ¬ Nat.Prime 21171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 7057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_73 : row 73 = 1 := (row_73.checked).trans rfl
theorem model_count_73 : modelRow 73 = 1 := by decide
theorem modelrow_agrees_73 : modelRow 73 = row 73 := model_count_73.trans row_count_73.symm
def row_74 : RowCertificate 74 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 74 1) := by change ¬ Nat.Prime 22053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 7351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 74 2) := by change ¬ Nat.Prime 21905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 4381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 74 3) := by change Nat.Prime 21757; exact ProjectEulerSolutions.P58PrimeNodes.p_21757.prime
    exact decide_eq_true prime
theorem row_count_74 : row 74 = 1 := (row_74.checked).trans rfl
theorem model_count_74 : modelRow 74 = 1 := by decide
theorem modelrow_agrees_74 : modelRow 74 = row 74 := model_count_74.trans row_count_74.symm
def row_75 : RowCertificate 75 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 75 1) := by change Nat.Prime 22651; exact ProjectEulerSolutions.P58PrimeNodes.p_22651.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 75 2) := by change Nat.Prime 22501; exact ProjectEulerSolutions.P58PrimeNodes.p_22501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 75 3) := by change ¬ Nat.Prime 22351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 3193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_75 : row 75 = 2 := (row_75.checked).trans rfl
theorem model_count_75 : modelRow 75 = 2 := by decide
theorem modelrow_agrees_75 : modelRow 75 = row 75 := model_count_75.trans row_count_75.symm
def row_76 : RowCertificate 76 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 76 1) := by change ¬ Nat.Prime 23257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 76 2) := by change ¬ Nat.Prime 23105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 4621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 76 3) := by change ¬ Nat.Prime 22953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 7651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_76 : row 76 = 0 := (row_76.checked).trans rfl
theorem model_count_76 : modelRow 76 = 0 := by decide
theorem modelrow_agrees_76 : modelRow 76 = row 76 := model_count_76.trans row_count_76.symm
def row_77 : RowCertificate 77 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 77 1) := by change ¬ Nat.Prime 23871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 7957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 77 2) := by change ¬ Nat.Prime 23717; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 77 3) := by change Nat.Prime 23563; exact ProjectEulerSolutions.P58PrimeNodes.p_23563.prime
    exact decide_eq_true prime
theorem row_count_77 : row 77 = 1 := (row_77.checked).trans rfl
theorem model_count_77 : modelRow 77 = 1 := by decide
theorem modelrow_agrees_77 : modelRow 77 = row 77 := model_count_77.trans row_count_77.symm
def row_78 : RowCertificate 78 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 78 1) := by change ¬ Nat.Prime 24493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 3499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 78 2) := by change Nat.Prime 24337; exact ProjectEulerSolutions.P58PrimeNodes.p_24337.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 78 3) := by change Nat.Prime 24181; exact ProjectEulerSolutions.P58PrimeNodes.p_24181.prime
    exact decide_eq_true prime
theorem row_count_78 : row 78 = 2 := (row_78.checked).trans rfl
theorem model_count_78 : modelRow 78 = 2 := by decide
theorem modelrow_agrees_78 : modelRow 78 = row 78 := model_count_78.trans row_count_78.symm
def row_79 : RowCertificate 79 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 79 1) := by change ¬ Nat.Prime 25123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 3589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 79 2) := by change ¬ Nat.Prime 24965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 4993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 79 3) := by change ¬ Nat.Prime 24807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 8269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_79 : row 79 = 0 := (row_79.checked).trans rfl
theorem model_count_79 : modelRow 79 = 0 := by decide
theorem modelrow_agrees_79 : modelRow 79 = row 79 := model_count_79.trans row_count_79.symm
def row_80 : RowCertificate 80 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 80 1) := by change ¬ Nat.Prime 25761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 8587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 80 2) := by change Nat.Prime 25601; exact ProjectEulerSolutions.P58PrimeNodes.p_25601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 80 3) := by change ¬ Nat.Prime 25441; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_80 : row 80 = 1 := (row_80.checked).trans rfl
theorem model_count_80 : modelRow 80 = 1 := by decide
theorem modelrow_agrees_80 : modelRow 80 = row 80 := model_count_80.trans row_count_80.symm
def row_81 : RowCertificate 81 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 81 1) := by change Nat.Prime 26407; exact ProjectEulerSolutions.P58PrimeNodes.p_26407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 81 2) := by change ¬ Nat.Prime 26245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 81 3) := by change Nat.Prime 26083; exact ProjectEulerSolutions.P58PrimeNodes.p_26083.prime
    exact decide_eq_true prime
theorem row_count_81 : row 81 = 2 := (row_81.checked).trans rfl
theorem model_count_81 : modelRow 81 = 2 := by decide
theorem modelrow_agrees_81 : modelRow 81 = row 81 := model_count_81.trans row_count_81.symm
def row_82 : RowCertificate 82 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 82 1) := by change Nat.Prime 27061; exact ProjectEulerSolutions.P58PrimeNodes.p_27061.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 82 2) := by change ¬ Nat.Prime 26897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 82 3) := by change ¬ Nat.Prime 26733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 8911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_82 : row 82 = 1 := (row_82.checked).trans rfl
theorem model_count_82 : modelRow 82 = 1 := by decide
theorem modelrow_agrees_82 : modelRow 82 = row 82 := model_count_82.trans row_count_82.symm
def row_83 : RowCertificate 83 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 83 1) := by change ¬ Nat.Prime 27723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 83 2) := by change ¬ Nat.Prime 27557; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 83 3) := by change ¬ Nat.Prime 27391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 3913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_83 : row 83 = 0 := (row_83.checked).trans rfl
theorem model_count_83 : modelRow 83 = 0 := by decide
theorem modelrow_agrees_83 : modelRow 83 = row 83 := model_count_83.trans row_count_83.symm
def row_84 : RowCertificate 84 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 84 1) := by change Nat.Prime 28393; exact ProjectEulerSolutions.P58PrimeNodes.p_28393.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 84 2) := by change ¬ Nat.Prime 28225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 84 3) := by change Nat.Prime 28057; exact ProjectEulerSolutions.P58PrimeNodes.p_28057.prime
    exact decide_eq_true prime
theorem row_count_84 : row 84 = 2 := (row_84.checked).trans rfl
theorem model_count_84 : modelRow 84 = 2 := by decide
theorem modelrow_agrees_84 : modelRow 84 = row 84 := model_count_84.trans row_count_84.symm
def row_85 : RowCertificate 85 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 85 1) := by change ¬ Nat.Prime 29071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 85 2) := by change Nat.Prime 28901; exact ProjectEulerSolutions.P58PrimeNodes.p_28901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 85 3) := by change ¬ Nat.Prime 28731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_85 : row 85 = 1 := (row_85.checked).trans rfl
theorem model_count_85 : modelRow 85 = 1 := by decide
theorem modelrow_agrees_85 : modelRow 85 = row 85 := model_count_85.trans row_count_85.symm
def row_86 : RowCertificate 86 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 86 1) := by change ¬ Nat.Prime 29757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 86 2) := by change ¬ Nat.Prime 29585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 86 3) := by change ¬ Nat.Prime 29413; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_86 : row 86 = 0 := (row_86.checked).trans rfl
theorem model_count_86 : modelRow 86 = 0 := by decide
theorem modelrow_agrees_86 : modelRow 86 = row 86 := model_count_86.trans row_count_86.symm
def row_87 : RowCertificate 87 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 87 1) := by change ¬ Nat.Prime 30451; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 87 2) := by change ¬ Nat.Prime 30277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 87 3) := by change Nat.Prime 30103; exact ProjectEulerSolutions.P58PrimeNodes.p_30103.prime
    exact decide_eq_true prime
theorem row_count_87 : row 87 = 1 := (row_87.checked).trans rfl
theorem model_count_87 : modelRow 87 = 1 := by decide
theorem modelrow_agrees_87 : modelRow 87 = row 87 := model_count_87.trans row_count_87.symm
def row_88 : RowCertificate 88 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 88 1) := by change Nat.Prime 31153; exact ProjectEulerSolutions.P58PrimeNodes.p_31153.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 88 2) := by change Nat.Prime 30977; exact ProjectEulerSolutions.P58PrimeNodes.p_30977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 88 3) := by change ¬ Nat.Prime 30801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_88 : row 88 = 2 := (row_88.checked).trans rfl
theorem model_count_88 : modelRow 88 = 2 := by decide
theorem modelrow_agrees_88 : modelRow 88 = row 88 := model_count_88.trans row_count_88.symm
def row_89 : RowCertificate 89 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 89 1) := by change ¬ Nat.Prime 31863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 89 2) := by change ¬ Nat.Prime 31685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 89 3) := by change ¬ Nat.Prime 31507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_89 : row 89 = 0 := (row_89.checked).trans rfl
theorem model_count_89 : modelRow 89 = 0 := by decide
theorem modelrow_agrees_89 : modelRow 89 = row 89 := model_count_89.trans row_count_89.symm
def row_90 : RowCertificate 90 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 90 1) := by change ¬ Nat.Prime 32581; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 90 2) := by change Nat.Prime 32401; exact ProjectEulerSolutions.P58PrimeNodes.p_32401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 90 3) := by change ¬ Nat.Prime 32221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_90 : row 90 = 1 := (row_90.checked).trans rfl
theorem model_count_90 : modelRow 90 = 1 := by decide
theorem modelrow_agrees_90 : modelRow 90 = row 90 := model_count_90.trans row_count_90.symm
def row_91 : RowCertificate 91 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 91 1) := by change ¬ Nat.Prime 33307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 91 2) := by change ¬ Nat.Prime 33125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 91 3) := by change ¬ Nat.Prime 32943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_91 : row 91 = 0 := (row_91.checked).trans rfl
theorem model_count_91 : modelRow 91 = 0 := by decide
theorem modelrow_agrees_91 : modelRow 91 = row 91 := model_count_91.trans row_count_91.symm
def row_92 : RowCertificate 92 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 92 1) := by change ¬ Nat.Prime 34041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 92 2) := by change Nat.Prime 33857; exact ProjectEulerSolutions.P58PrimeNodes.p_33857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 92 3) := by change ¬ Nat.Prime 33673; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_92 : row 92 = 1 := (row_92.checked).trans rfl
theorem model_count_92 : modelRow 92 = 1 := by decide
theorem modelrow_agrees_92 : modelRow 92 = row 92 := model_count_92.trans row_count_92.symm
def row_93 : RowCertificate 93 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 93 1) := by change ¬ Nat.Prime 34783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 93 2) := by change ¬ Nat.Prime 34597; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 93 3) := by change ¬ Nat.Prime 34411; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_93 : row 93 = 0 := (row_93.checked).trans rfl
theorem model_count_93 : modelRow 93 = 0 := by decide
theorem modelrow_agrees_93 : modelRow 93 = row 93 := model_count_93.trans row_count_93.symm
def row_94 : RowCertificate 94 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 94 1) := by change Nat.Prime 35533; exact ProjectEulerSolutions.P58PrimeNodes.p_35533.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 94 2) := by change ¬ Nat.Prime 35345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 94 3) := by change ¬ Nat.Prime 35157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_94 : row 94 = 1 := (row_94.checked).trans rfl
theorem model_count_94 : modelRow 94 = 1 := by decide
theorem modelrow_agrees_94 : modelRow 94 = row 94 := model_count_94.trans row_count_94.symm
def row_95 : RowCertificate 95 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 95 1) := by change ¬ Nat.Prime 36291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 95 2) := by change ¬ Nat.Prime 36101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 95 3) := by change Nat.Prime 35911; exact ProjectEulerSolutions.P58PrimeNodes.p_35911.prime
    exact decide_eq_true prime
theorem row_count_95 : row 95 = 1 := (row_95.checked).trans rfl
theorem model_count_95 : modelRow 95 = 1 := by decide
theorem modelrow_agrees_95 : modelRow 95 = row 95 := model_count_95.trans row_count_95.symm
def row_96 : RowCertificate 96 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 96 1) := by change Nat.Prime 37057; exact ProjectEulerSolutions.P58PrimeNodes.p_37057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 96 2) := by change ¬ Nat.Prime 36865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 96 3) := by change ¬ Nat.Prime 36673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_96 : row 96 = 1 := (row_96.checked).trans rfl
theorem model_count_96 : modelRow 96 = 1 := by decide
theorem modelrow_agrees_96 : modelRow 96 = row 96 := model_count_96.trans row_count_96.symm
def row_97 : RowCertificate 97 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 97 1) := by change Nat.Prime 37831; exact ProjectEulerSolutions.P58PrimeNodes.p_37831.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 97 2) := by change ¬ Nat.Prime 37637; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 97 3) := by change ¬ Nat.Prime 37443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_97 : row 97 = 1 := (row_97.checked).trans rfl
theorem model_count_97 : modelRow 97 = 1 := by decide
theorem modelrow_agrees_97 : modelRow 97 = row 97 := model_count_97.trans row_count_97.symm
def row_98 : RowCertificate 98 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 98 1) := by change ¬ Nat.Prime 38613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 98 2) := by change ¬ Nat.Prime 38417; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 98 3) := by change ¬ Nat.Prime 38221; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_98 : row 98 = 0 := (row_98.checked).trans rfl
theorem model_count_98 : modelRow 98 = 0 := by decide
theorem modelrow_agrees_98 : modelRow 98 = row 98 := model_count_98.trans row_count_98.symm
def row_99 : RowCertificate 99 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 99 1) := by change ¬ Nat.Prime 39403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 99 2) := by change ¬ Nat.Prime 39205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 99 3) := by change ¬ Nat.Prime 39007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_99 : row 99 = 0 := (row_99.checked).trans rfl
theorem model_count_99 : modelRow 99 = 0 := by decide
theorem modelrow_agrees_99 : modelRow 99 = row 99 := model_count_99.trans row_count_99.symm
def row_100 : RowCertificate 100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 100 1) := by change ¬ Nat.Prime 40201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 100 2) := by change ¬ Nat.Prime 40001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 100 3) := by change ¬ Nat.Prime 39801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_100 : row 100 = 0 := (row_100.checked).trans rfl
theorem model_count_100 : modelRow 100 = 0 := by decide
theorem modelrow_agrees_100 : modelRow 100 = row 100 := model_count_100.trans row_count_100.symm
def row_101 : RowCertificate 101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 101 1) := by change ¬ Nat.Prime 41007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 101 2) := by change ¬ Nat.Prime 40805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 101 3) := by change ¬ Nat.Prime 40603; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_101 : row 101 = 0 := (row_101.checked).trans rfl
theorem model_count_101 : modelRow 101 = 0 := by decide
theorem modelrow_agrees_101 : modelRow 101 = row 101 := model_count_101.trans row_count_101.symm
def row_102 : RowCertificate 102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 102 1) := by change ¬ Nat.Prime 41821; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 102 2) := by change Nat.Prime 41617; exact ProjectEulerSolutions.P58PrimeNodes.p_41617.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 102 3) := by change Nat.Prime 41413; exact ProjectEulerSolutions.P58PrimeNodes.p_41413.prime
    exact decide_eq_true prime
theorem row_count_102 : row 102 = 2 := (row_102.checked).trans rfl
theorem model_count_102 : modelRow 102 = 2 := by decide
theorem modelrow_agrees_102 : modelRow 102 = row 102 := model_count_102.trans row_count_102.symm
def row_103 : RowCertificate 103 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 103 1) := by change Nat.Prime 42643; exact ProjectEulerSolutions.P58PrimeNodes.p_42643.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 103 2) := by change Nat.Prime 42437; exact ProjectEulerSolutions.P58PrimeNodes.p_42437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 103 3) := by change ¬ Nat.Prime 42231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_103 : row 103 = 2 := (row_103.checked).trans rfl
theorem model_count_103 : modelRow 103 = 2 := by decide
theorem modelrow_agrees_103 : modelRow 103 = row 103 := model_count_103.trans row_count_103.symm
def row_104 : RowCertificate 104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 104 1) := by change ¬ Nat.Prime 43473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 104 2) := by change ¬ Nat.Prime 43265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 104 3) := by change ¬ Nat.Prime 43057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_104 : row 104 = 0 := (row_104.checked).trans rfl
theorem model_count_104 : modelRow 104 = 0 := by decide
theorem modelrow_agrees_104 : modelRow 104 = row 104 := model_count_104.trans row_count_104.symm
def row_105 : RowCertificate 105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 105 1) := by change ¬ Nat.Prime 44311; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 105 2) := by change Nat.Prime 44101; exact ProjectEulerSolutions.P58PrimeNodes.p_44101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 105 3) := by change Nat.Prime 43891; exact ProjectEulerSolutions.P58PrimeNodes.p_43891.prime
    exact decide_eq_true prime
theorem row_count_105 : row 105 = 2 := (row_105.checked).trans rfl
theorem model_count_105 : modelRow 105 = 2 := by decide
theorem modelrow_agrees_105 : modelRow 105 = row 105 := model_count_105.trans row_count_105.symm
def row_106 : RowCertificate 106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 106 1) := by change ¬ Nat.Prime 45157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 106 2) := by change ¬ Nat.Prime 44945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 106 3) := by change ¬ Nat.Prime 44733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_106 : row 106 = 0 := (row_106.checked).trans rfl
theorem model_count_106 : modelRow 106 = 0 := by decide
theorem modelrow_agrees_106 : modelRow 106 = row 106 := model_count_106.trans row_count_106.symm
def row_107 : RowCertificate 107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 107 1) := by change ¬ Nat.Prime 46011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 107 2) := by change ¬ Nat.Prime 45797; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 107 3) := by change ¬ Nat.Prime 45583; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_107 : row 107 = 0 := (row_107.checked).trans rfl
theorem model_count_107 : modelRow 107 = 0 := by decide
theorem modelrow_agrees_107 : modelRow 107 = row 107 := model_count_107.trans row_count_107.symm
def row_108 : RowCertificate 108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 108 1) := by change ¬ Nat.Prime 46873; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 108 2) := by change ¬ Nat.Prime 46657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 108 3) := by change Nat.Prime 46441; exact ProjectEulerSolutions.P58PrimeNodes.p_46441.prime
    exact decide_eq_true prime
theorem row_count_108 : row 108 = 1 := (row_108.checked).trans rfl
theorem model_count_108 : modelRow 108 = 1 := by decide
theorem modelrow_agrees_108 : modelRow 108 = row 108 := model_count_108.trans row_count_108.symm
def row_109 : RowCertificate 109 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 109 1) := by change Nat.Prime 47743; exact ProjectEulerSolutions.P58PrimeNodes.p_47743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 109 2) := by change ¬ Nat.Prime 47525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 109 3) := by change ¬ Nat.Prime 47307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_109 : row 109 = 1 := (row_109.checked).trans rfl
theorem model_count_109 : modelRow 109 = 1 := by decide
theorem modelrow_agrees_109 : modelRow 109 = row 109 := model_count_109.trans row_count_109.symm
def row_110 : RowCertificate 110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 110 1) := by change ¬ Nat.Prime 48621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 110 2) := by change ¬ Nat.Prime 48401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 110 3) := by change ¬ Nat.Prime 48181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_110 : row 110 = 0 := (row_110.checked).trans rfl
theorem model_count_110 : modelRow 110 = 0 := by decide
theorem modelrow_agrees_110 : modelRow 110 = row 110 := model_count_110.trans row_count_110.symm
def row_111 : RowCertificate 111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 111 1) := by change ¬ Nat.Prime 49507; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 111 2) := by change ¬ Nat.Prime 49285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 111 3) := by change ¬ Nat.Prime 49063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_111 : row 111 = 0 := (row_111.checked).trans rfl
theorem model_count_111 : modelRow 111 = 0 := by decide
theorem modelrow_agrees_111 : modelRow 111 = row 111 := model_count_111.trans row_count_111.symm
def row_112 : RowCertificate 112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 112 1) := by change ¬ Nat.Prime 50401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 112 2) := by change Nat.Prime 50177; exact ProjectEulerSolutions.P58PrimeNodes.p_50177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 112 3) := by change ¬ Nat.Prime 49953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_112 : row 112 = 1 := (row_112.checked).trans rfl
theorem model_count_112 : modelRow 112 = 1 := by decide
theorem modelrow_agrees_112 : modelRow 112 = row 112 := model_count_112.trans row_count_112.symm
def row_113 : RowCertificate 113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 113 1) := by change ¬ Nat.Prime 51303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 113 2) := by change ¬ Nat.Prime 51077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 113 3) := by change ¬ Nat.Prime 50851; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_113 : row 113 = 0 := (row_113.checked).trans rfl
theorem model_count_113 : modelRow 113 = 0 := by decide
theorem modelrow_agrees_113 : modelRow 113 = row 113 := model_count_113.trans row_count_113.symm
def row_114 : RowCertificate 114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 114 1) := by change ¬ Nat.Prime 52213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7459) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 114 2) := by change ¬ Nat.Prime 51985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 114 3) := by change ¬ Nat.Prime 51757; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_114 : row 114 = 0 := (row_114.checked).trans rfl
theorem model_count_114 : modelRow 114 = 0 := by decide
theorem modelrow_agrees_114 : modelRow 114 = row 114 := model_count_114.trans row_count_114.symm
def row_115 : RowCertificate 115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 115 1) := by change ¬ Nat.Prime 53131; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 115 2) := by change Nat.Prime 52901; exact ProjectEulerSolutions.P58PrimeNodes.p_52901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 115 3) := by change ¬ Nat.Prime 52671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_115 : row 115 = 1 := (row_115.checked).trans rfl
theorem model_count_115 : modelRow 115 = 1 := by decide
theorem modelrow_agrees_115 : modelRow 115 = row 115 := model_count_115.trans row_count_115.symm
def row_116 : RowCertificate 116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 116 1) := by change ¬ Nat.Prime 54057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 116 2) := by change ¬ Nat.Prime 53825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 116 3) := by change Nat.Prime 53593; exact ProjectEulerSolutions.P58PrimeNodes.p_53593.prime
    exact decide_eq_true prime
theorem row_count_116 : row 116 = 1 := (row_116.checked).trans rfl
theorem model_count_116 : modelRow 116 = 1 := by decide
theorem modelrow_agrees_116 : modelRow 116 = row 116 := model_count_116.trans row_count_116.symm
def row_117 : RowCertificate 117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 117 1) := by change ¬ Nat.Prime 54991; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 117 2) := by change ¬ Nat.Prime 54757; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 117 3) := by change ¬ Nat.Prime 54523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_117 : row 117 = 0 := (row_117.checked).trans rfl
theorem model_count_117 : modelRow 117 = 0 := by decide
theorem modelrow_agrees_117 : modelRow 117 = row 117 := model_count_117.trans row_count_117.symm
def row_118 : RowCertificate 118 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 118 1) := by change Nat.Prime 55933; exact ProjectEulerSolutions.P58PrimeNodes.p_55933.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 118 2) := by change Nat.Prime 55697; exact ProjectEulerSolutions.P58PrimeNodes.p_55697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 118 3) := by change ¬ Nat.Prime 55461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_118 : row 118 = 2 := (row_118.checked).trans rfl
theorem model_count_118 : modelRow 118 = 2 := by decide
theorem modelrow_agrees_118 : modelRow 118 = row 118 := model_count_118.trans row_count_118.symm
def row_119 : RowCertificate 119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 119 1) := by change ¬ Nat.Prime 56883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 119 2) := by change ¬ Nat.Prime 56645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 119 3) := by change ¬ Nat.Prime 56407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_119 : row 119 = 0 := (row_119.checked).trans rfl
theorem model_count_119 : modelRow 119 = 0 := by decide
theorem modelrow_agrees_119 : modelRow 119 = row 119 := model_count_119.trans row_count_119.symm
def row_120 : RowCertificate 120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 120 1) := by change ¬ Nat.Prime 57841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 120 2) := by change Nat.Prime 57601; exact ProjectEulerSolutions.P58PrimeNodes.p_57601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 120 3) := by change ¬ Nat.Prime 57361; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_120 : row 120 = 1 := (row_120.checked).trans rfl
theorem model_count_120 : modelRow 120 = 1 := by decide
theorem modelrow_agrees_120 : modelRow 120 = row 120 := model_count_120.trans row_count_120.symm
def row_121 : RowCertificate 121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 121 1) := by change ¬ Nat.Prime 58807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 121 2) := by change ¬ Nat.Prime 58565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 121 3) := by change ¬ Nat.Prime 58323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_121 : row 121 = 0 := (row_121.checked).trans rfl
theorem model_count_121 : modelRow 121 = 0 := by decide
theorem modelrow_agrees_121 : modelRow 121 = row 121 := model_count_121.trans row_count_121.symm
def row_122 : RowCertificate 122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 122 1) := by change ¬ Nat.Prime 59781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 122 2) := by change ¬ Nat.Prime 59537; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 122 3) := by change ¬ Nat.Prime 59293; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_122 : row 122 = 0 := (row_122.checked).trans rfl
theorem model_count_122 : modelRow 122 = 0 := by decide
theorem modelrow_agrees_122 : modelRow 122 = row 122 := model_count_122.trans row_count_122.symm
def row_123 : RowCertificate 123 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 123 1) := by change Nat.Prime 60763; exact ProjectEulerSolutions.P58PrimeNodes.p_60763.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 123 2) := by change ¬ Nat.Prime 60517; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 123 3) := by change Nat.Prime 60271; exact ProjectEulerSolutions.P58PrimeNodes.p_60271.prime
    exact decide_eq_true prime
theorem row_count_123 : row 123 = 2 := (row_123.checked).trans rfl
theorem model_count_123 : modelRow 123 = 2 := by decide
theorem modelrow_agrees_123 : modelRow 123 = row 123 := model_count_123.trans row_count_123.symm
def row_124 : RowCertificate 124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 124 1) := by change ¬ Nat.Prime 61753; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 124 2) := by change ¬ Nat.Prime 61505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 124 3) := by change ¬ Nat.Prime 61257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_124 : row 124 = 0 := (row_124.checked).trans rfl
theorem model_count_124 : modelRow 124 = 0 := by decide
theorem modelrow_agrees_124 : modelRow 124 = row 124 := model_count_124.trans row_count_124.symm
def row_125 : RowCertificate 125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 125 1) := by change ¬ Nat.Prime 62751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 125 2) := by change Nat.Prime 62501; exact ProjectEulerSolutions.P58PrimeNodes.p_62501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 125 3) := by change ¬ Nat.Prime 62251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_125 : row 125 = 1 := (row_125.checked).trans rfl
theorem model_count_125 : modelRow 125 = 1 := by decide
theorem modelrow_agrees_125 : modelRow 125 = row 125 := model_count_125.trans row_count_125.symm
def row_126 : RowCertificate 126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 126 1) := by change ¬ Nat.Prime 63757; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 126 2) := by change ¬ Nat.Prime 63505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 126 3) := by change ¬ Nat.Prime 63253; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_126 : row 126 = 0 := (row_126.checked).trans rfl
theorem model_count_126 : modelRow 126 = 0 := by decide
theorem modelrow_agrees_126 : modelRow 126 = row 126 := model_count_126.trans row_count_126.symm
def row_127 : RowCertificate 127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 127 1) := by change ¬ Nat.Prime 64771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 127 2) := by change ¬ Nat.Prime 64517; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 127 3) := by change ¬ Nat.Prime 64263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_127 : row 127 = 0 := (row_127.checked).trans rfl
theorem model_count_127 : modelRow 127 = 0 := by decide
theorem modelrow_agrees_127 : modelRow 127 = row 127 := model_count_127.trans row_count_127.symm
def row_128 : RowCertificate 128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 128 1) := by change ¬ Nat.Prime 65793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 128 2) := by change Nat.Prime 65537; exact ProjectEulerSolutions.P58PrimeNodes.p_65537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 128 3) := by change ¬ Nat.Prime 65281; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 673) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_128 : row 128 = 1 := (row_128.checked).trans rfl
theorem model_count_128 : modelRow 128 = 1 := by decide
theorem modelrow_agrees_128 : modelRow 128 = row 128 := model_count_128.trans row_count_128.symm
private def atomic_1 : CountSegment 1 10 13120 0 1 0 3 := CountSegment.single (first := 0) (initial := 0) row_count_1 (by decide)
private theorem atomic_1_agreement : RowAgreement 0 1 := RowAgreement.single (first := 0) modelrow_agrees_1
private def atomic_2 : CountSegment 1 10 13120 1 2 3 5 := CountSegment.single (first := 1) (initial := 3) row_count_2 (by decide)
private theorem atomic_2_agreement : RowAgreement 1 2 := RowAgreement.single (first := 1) modelrow_agrees_2
private def atomic_3 : CountSegment 1 10 13120 2 3 5 8 := CountSegment.single (first := 2) (initial := 5) row_count_3 (by decide)
private theorem atomic_3_agreement : RowAgreement 2 3 := RowAgreement.single (first := 2) modelrow_agrees_3
private def atomic_4 : CountSegment 1 10 13120 3 4 8 9 := CountSegment.single (first := 3) (initial := 8) row_count_4 (by decide)
private theorem atomic_4_agreement : RowAgreement 3 4 := RowAgreement.single (first := 3) modelrow_agrees_4
private def atomic_5 : CountSegment 1 10 13120 4 5 9 10 := CountSegment.single (first := 4) (initial := 9) row_count_5 (by decide)
private theorem atomic_5_agreement : RowAgreement 4 5 := RowAgreement.single (first := 4) modelrow_agrees_5
private def atomic_6 : CountSegment 1 10 13120 5 6 10 11 := CountSegment.single (first := 5) (initial := 10) row_count_6 (by decide)
private theorem atomic_6_agreement : RowAgreement 5 6 := RowAgreement.single (first := 5) modelrow_agrees_6
private def atomic_7 : CountSegment 1 10 13120 6 7 11 13 := CountSegment.single (first := 6) (initial := 11) row_count_7 (by decide)
private theorem atomic_7_agreement : RowAgreement 6 7 := RowAgreement.single (first := 6) modelrow_agrees_7
private def atomic_8 : CountSegment 1 10 13120 7 8 13 15 := CountSegment.single (first := 7) (initial := 13) row_count_8 (by decide)
private theorem atomic_8_agreement : RowAgreement 7 8 := RowAgreement.single (first := 7) modelrow_agrees_8
private def atomic_9 : CountSegment 1 10 13120 8 9 15 16 := CountSegment.single (first := 8) (initial := 15) row_count_9 (by decide)
private theorem atomic_9_agreement : RowAgreement 8 9 := RowAgreement.single (first := 8) modelrow_agrees_9
private def atomic_10 : CountSegment 1 10 13120 9 10 16 18 := CountSegment.single (first := 9) (initial := 16) row_count_10 (by decide)
private theorem atomic_10_agreement : RowAgreement 9 10 := RowAgreement.single (first := 9) modelrow_agrees_10
private def atomic_11 : CountSegment 1 10 13120 10 11 18 19 := CountSegment.single (first := 10) (initial := 18) row_count_11 (by decide)
private theorem atomic_11_agreement : RowAgreement 10 11 := RowAgreement.single (first := 10) modelrow_agrees_11
private def atomic_12 : CountSegment 1 10 13120 11 12 19 21 := CountSegment.single (first := 11) (initial := 19) row_count_12 (by decide)
private theorem atomic_12_agreement : RowAgreement 11 12 := RowAgreement.single (first := 11) modelrow_agrees_12
private def atomic_13 : CountSegment 1 10 13120 12 13 21 22 := CountSegment.single (first := 12) (initial := 21) row_count_13 (by decide)
private theorem atomic_13_agreement : RowAgreement 12 13 := RowAgreement.single (first := 12) modelrow_agrees_13
private def atomic_14 : CountSegment 1 10 13120 13 14 22 23 := CountSegment.single (first := 13) (initial := 22) row_count_14 (by decide)
private theorem atomic_14_agreement : RowAgreement 13 14 := RowAgreement.single (first := 13) modelrow_agrees_14
private def atomic_15 : CountSegment 1 10 13120 14 15 23 23 := CountSegment.single (first := 14) (initial := 23) row_count_15 (by decide)
private theorem atomic_15_agreement : RowAgreement 14 15 := RowAgreement.single (first := 14) modelrow_agrees_15
private def atomic_16 : CountSegment 1 10 13120 15 16 23 23 := CountSegment.single (first := 15) (initial := 23) row_count_16 (by decide)
private theorem atomic_16_agreement : RowAgreement 15 16 := RowAgreement.single (first := 15) modelrow_agrees_16
private def atomic_17 : CountSegment 1 10 13120 16 17 23 24 := CountSegment.single (first := 16) (initial := 23) row_count_17 (by decide)
private theorem atomic_17_agreement : RowAgreement 16 17 := RowAgreement.single (first := 16) modelrow_agrees_17
private def atomic_18 : CountSegment 1 10 13120 17 18 24 25 := CountSegment.single (first := 17) (initial := 24) row_count_18 (by decide)
private theorem atomic_18_agreement : RowAgreement 17 18 := RowAgreement.single (first := 17) modelrow_agrees_18
private def atomic_19 : CountSegment 1 10 13120 18 19 25 26 := CountSegment.single (first := 18) (initial := 25) row_count_19 (by decide)
private theorem atomic_19_agreement : RowAgreement 18 19 := RowAgreement.single (first := 18) modelrow_agrees_19
private def atomic_20 : CountSegment 1 10 13120 19 20 26 27 := CountSegment.single (first := 19) (initial := 26) row_count_20 (by decide)
private theorem atomic_20_agreement : RowAgreement 19 20 := RowAgreement.single (first := 19) modelrow_agrees_20
private def atomic_21 : CountSegment 1 10 13120 20 21 27 28 := CountSegment.single (first := 20) (initial := 27) row_count_21 (by decide)
private theorem atomic_21_agreement : RowAgreement 20 21 := RowAgreement.single (first := 20) modelrow_agrees_21
private def atomic_22 : CountSegment 1 10 13120 21 22 28 28 := CountSegment.single (first := 21) (initial := 28) row_count_22 (by decide)
private theorem atomic_22_agreement : RowAgreement 21 22 := RowAgreement.single (first := 21) modelrow_agrees_22
private def atomic_23 : CountSegment 1 10 13120 22 23 28 28 := CountSegment.single (first := 22) (initial := 28) row_count_23 (by decide)
private theorem atomic_23_agreement : RowAgreement 22 23 := RowAgreement.single (first := 22) modelrow_agrees_23
private def atomic_24 : CountSegment 1 10 13120 23 24 28 28 := CountSegment.single (first := 23) (initial := 28) row_count_24 (by decide)
private theorem atomic_24_agreement : RowAgreement 23 24 := RowAgreement.single (first := 23) modelrow_agrees_24
private def atomic_25 : CountSegment 1 10 13120 24 25 28 29 := CountSegment.single (first := 24) (initial := 28) row_count_25 (by decide)
private theorem atomic_25_agreement : RowAgreement 24 25 := RowAgreement.single (first := 24) modelrow_agrees_25
private def atomic_26 : CountSegment 1 10 13120 25 26 29 29 := CountSegment.single (first := 25) (initial := 29) row_count_26 (by decide)
private theorem atomic_26_agreement : RowAgreement 25 26 := RowAgreement.single (first := 25) modelrow_agrees_26
private def atomic_27 : CountSegment 1 10 13120 26 27 29 31 := CountSegment.single (first := 26) (initial := 29) row_count_27 (by decide)
private theorem atomic_27_agreement : RowAgreement 26 27 := RowAgreement.single (first := 26) modelrow_agrees_27
private def atomic_28 : CountSegment 1 10 13120 27 28 31 32 := CountSegment.single (first := 27) (initial := 31) row_count_28 (by decide)
private theorem atomic_28_agreement : RowAgreement 27 28 := RowAgreement.single (first := 27) modelrow_agrees_28
private def atomic_29 : CountSegment 1 10 13120 28 29 32 33 := CountSegment.single (first := 28) (initial := 32) row_count_29 (by decide)
private theorem atomic_29_agreement : RowAgreement 28 29 := RowAgreement.single (first := 28) modelrow_agrees_29
private def atomic_30 : CountSegment 1 10 13120 29 30 33 34 := CountSegment.single (first := 29) (initial := 33) row_count_30 (by decide)
private theorem atomic_30_agreement : RowAgreement 29 30 := RowAgreement.single (first := 29) modelrow_agrees_30
private def atomic_31 : CountSegment 1 10 13120 30 31 34 35 := CountSegment.single (first := 30) (initial := 34) row_count_31 (by decide)
private theorem atomic_31_agreement : RowAgreement 30 31 := RowAgreement.single (first := 30) modelrow_agrees_31
private def atomic_32 : CountSegment 1 10 13120 31 32 35 35 := CountSegment.single (first := 31) (initial := 35) row_count_32 (by decide)
private theorem atomic_32_agreement : RowAgreement 31 32 := RowAgreement.single (first := 31) modelrow_agrees_32
private def atomic_33 : CountSegment 1 10 13120 32 33 35 37 := CountSegment.single (first := 32) (initial := 35) row_count_33 (by decide)
private theorem atomic_33_agreement : RowAgreement 32 33 := RowAgreement.single (first := 32) modelrow_agrees_33
private def atomic_34 : CountSegment 1 10 13120 33 34 37 37 := CountSegment.single (first := 33) (initial := 37) row_count_34 (by decide)
private theorem atomic_34_agreement : RowAgreement 33 34 := RowAgreement.single (first := 33) modelrow_agrees_34
private def atomic_35 : CountSegment 1 10 13120 34 35 37 38 := CountSegment.single (first := 34) (initial := 37) row_count_35 (by decide)
private theorem atomic_35_agreement : RowAgreement 34 35 := RowAgreement.single (first := 34) modelrow_agrees_35
private def atomic_36 : CountSegment 1 10 13120 35 36 38 39 := CountSegment.single (first := 35) (initial := 38) row_count_36 (by decide)
private theorem atomic_36_agreement : RowAgreement 35 36 := RowAgreement.single (first := 35) modelrow_agrees_36
private def atomic_37 : CountSegment 1 10 13120 36 37 39 40 := CountSegment.single (first := 36) (initial := 39) row_count_37 (by decide)
private theorem atomic_37_agreement : RowAgreement 36 37 := RowAgreement.single (first := 36) modelrow_agrees_37
private def atomic_38 : CountSegment 1 10 13120 37 38 40 41 := CountSegment.single (first := 37) (initial := 40) row_count_38 (by decide)
private theorem atomic_38_agreement : RowAgreement 37 38 := RowAgreement.single (first := 37) modelrow_agrees_38
private def atomic_39 : CountSegment 1 10 13120 38 39 41 43 := CountSegment.single (first := 38) (initial := 41) row_count_39 (by decide)
private theorem atomic_39_agreement : RowAgreement 38 39 := RowAgreement.single (first := 38) modelrow_agrees_39
private def atomic_40 : CountSegment 1 10 13120 39 40 43 44 := CountSegment.single (first := 39) (initial := 43) row_count_40 (by decide)
private theorem atomic_40_agreement : RowAgreement 39 40 := RowAgreement.single (first := 39) modelrow_agrees_40
private def atomic_41 : CountSegment 1 10 13120 40 41 44 44 := CountSegment.single (first := 40) (initial := 44) row_count_41 (by decide)
private theorem atomic_41_agreement : RowAgreement 40 41 := RowAgreement.single (first := 40) modelrow_agrees_41
private def atomic_42 : CountSegment 1 10 13120 41 42 44 45 := CountSegment.single (first := 41) (initial := 44) row_count_42 (by decide)
private theorem atomic_42_agreement : RowAgreement 41 42 := RowAgreement.single (first := 41) modelrow_agrees_42
private def atomic_43 : CountSegment 1 10 13120 42 43 45 45 := CountSegment.single (first := 42) (initial := 45) row_count_43 (by decide)
private theorem atomic_43_agreement : RowAgreement 42 43 := RowAgreement.single (first := 42) modelrow_agrees_43
private def atomic_44 : CountSegment 1 10 13120 43 44 45 45 := CountSegment.single (first := 43) (initial := 45) row_count_44 (by decide)
private theorem atomic_44_agreement : RowAgreement 43 44 := RowAgreement.single (first := 43) modelrow_agrees_44
private def atomic_45 : CountSegment 1 10 13120 44 45 45 48 := CountSegment.single (first := 44) (initial := 45) row_count_45 (by decide)
private theorem atomic_45_agreement : RowAgreement 44 45 := RowAgreement.single (first := 44) modelrow_agrees_45
private def atomic_46 : CountSegment 1 10 13120 45 46 48 48 := CountSegment.single (first := 45) (initial := 48) row_count_46 (by decide)
private theorem atomic_46_agreement : RowAgreement 45 46 := RowAgreement.single (first := 45) modelrow_agrees_46
private def atomic_47 : CountSegment 1 10 13120 46 47 48 49 := CountSegment.single (first := 46) (initial := 48) row_count_47 (by decide)
private theorem atomic_47_agreement : RowAgreement 46 47 := RowAgreement.single (first := 46) modelrow_agrees_47
private def atomic_48 : CountSegment 1 10 13120 47 48 49 49 := CountSegment.single (first := 47) (initial := 49) row_count_48 (by decide)
private theorem atomic_48_agreement : RowAgreement 47 48 := RowAgreement.single (first := 47) modelrow_agrees_48
private def atomic_49 : CountSegment 1 10 13120 48 49 49 49 := CountSegment.single (first := 48) (initial := 49) row_count_49 (by decide)
private theorem atomic_49_agreement : RowAgreement 48 49 := RowAgreement.single (first := 48) modelrow_agrees_49
private def atomic_50 : CountSegment 1 10 13120 49 50 49 50 := CountSegment.single (first := 49) (initial := 49) row_count_50 (by decide)
private theorem atomic_50_agreement : RowAgreement 49 50 := RowAgreement.single (first := 49) modelrow_agrees_50
private def atomic_51 : CountSegment 1 10 13120 50 51 50 51 := CountSegment.single (first := 50) (initial := 50) row_count_51 (by decide)
private theorem atomic_51_agreement : RowAgreement 50 51 := RowAgreement.single (first := 50) modelrow_agrees_51
private def atomic_52 : CountSegment 1 10 13120 51 52 51 51 := CountSegment.single (first := 51) (initial := 51) row_count_52 (by decide)
private theorem atomic_52_agreement : RowAgreement 51 52 := RowAgreement.single (first := 51) modelrow_agrees_52
private def atomic_53 : CountSegment 1 10 13120 52 53 51 52 := CountSegment.single (first := 52) (initial := 51) row_count_53 (by decide)
private theorem atomic_53_agreement : RowAgreement 52 53 := RowAgreement.single (first := 52) modelrow_agrees_53
private def atomic_54 : CountSegment 1 10 13120 53 54 52 52 := CountSegment.single (first := 53) (initial := 52) row_count_54 (by decide)
private theorem atomic_54_agreement : RowAgreement 53 54 := RowAgreement.single (first := 53) modelrow_agrees_54
private def atomic_55 : CountSegment 1 10 13120 54 55 52 54 := CountSegment.single (first := 54) (initial := 52) row_count_55 (by decide)
private theorem atomic_55_agreement : RowAgreement 54 55 := RowAgreement.single (first := 54) modelrow_agrees_55
private def atomic_56 : CountSegment 1 10 13120 55 56 54 55 := CountSegment.single (first := 55) (initial := 54) row_count_56 (by decide)
private theorem atomic_56_agreement : RowAgreement 55 56 := RowAgreement.single (first := 55) modelrow_agrees_56
private def atomic_57 : CountSegment 1 10 13120 56 57 55 55 := CountSegment.single (first := 56) (initial := 55) row_count_57 (by decide)
private theorem atomic_57_agreement : RowAgreement 56 57 := RowAgreement.single (first := 56) modelrow_agrees_57
private def atomic_58 : CountSegment 1 10 13120 57 58 55 56 := CountSegment.single (first := 57) (initial := 55) row_count_58 (by decide)
private theorem atomic_58_agreement : RowAgreement 57 58 := RowAgreement.single (first := 57) modelrow_agrees_58
private def atomic_59 : CountSegment 1 10 13120 58 59 56 57 := CountSegment.single (first := 58) (initial := 56) row_count_59 (by decide)
private theorem atomic_59_agreement : RowAgreement 58 59 := RowAgreement.single (first := 58) modelrow_agrees_59
private def atomic_60 : CountSegment 1 10 13120 59 60 57 59 := CountSegment.single (first := 59) (initial := 57) row_count_60 (by decide)
private theorem atomic_60_agreement : RowAgreement 59 60 := RowAgreement.single (first := 59) modelrow_agrees_60
private def atomic_61 : CountSegment 1 10 13120 60 61 59 59 := CountSegment.single (first := 60) (initial := 59) row_count_61 (by decide)
private theorem atomic_61_agreement : RowAgreement 60 61 := RowAgreement.single (first := 60) modelrow_agrees_61
private def atomic_62 : CountSegment 1 10 13120 61 62 59 60 := CountSegment.single (first := 61) (initial := 59) row_count_62 (by decide)
private theorem atomic_62_agreement : RowAgreement 61 62 := RowAgreement.single (first := 61) modelrow_agrees_62
private def atomic_63 : CountSegment 1 10 13120 62 63 60 61 := CountSegment.single (first := 62) (initial := 60) row_count_63 (by decide)
private theorem atomic_63_agreement : RowAgreement 62 63 := RowAgreement.single (first := 62) modelrow_agrees_63
private def atomic_64 : CountSegment 1 10 13120 63 64 61 61 := CountSegment.single (first := 63) (initial := 61) row_count_64 (by decide)
private theorem atomic_64_agreement : RowAgreement 63 64 := RowAgreement.single (first := 63) modelrow_agrees_64
private def atomic_65 : CountSegment 1 10 13120 64 65 61 62 := CountSegment.single (first := 64) (initial := 61) row_count_65 (by decide)
private theorem atomic_65_agreement : RowAgreement 64 65 := RowAgreement.single (first := 64) modelrow_agrees_65
private def atomic_66 : CountSegment 1 10 13120 65 66 62 63 := CountSegment.single (first := 65) (initial := 62) row_count_66 (by decide)
private theorem atomic_66_agreement : RowAgreement 65 66 := RowAgreement.single (first := 65) modelrow_agrees_66
private def atomic_67 : CountSegment 1 10 13120 66 67 63 64 := CountSegment.single (first := 66) (initial := 63) row_count_67 (by decide)
private theorem atomic_67_agreement : RowAgreement 66 67 := RowAgreement.single (first := 66) modelrow_agrees_67
private def atomic_68 : CountSegment 1 10 13120 67 68 64 64 := CountSegment.single (first := 67) (initial := 64) row_count_68 (by decide)
private theorem atomic_68_agreement : RowAgreement 67 68 := RowAgreement.single (first := 67) modelrow_agrees_68
private def atomic_69 : CountSegment 1 10 13120 68 69 64 65 := CountSegment.single (first := 68) (initial := 64) row_count_69 (by decide)
private theorem atomic_69_agreement : RowAgreement 68 69 := RowAgreement.single (first := 68) modelrow_agrees_69
private def atomic_70 : CountSegment 1 10 13120 69 70 65 65 := CountSegment.single (first := 69) (initial := 65) row_count_70 (by decide)
private theorem atomic_70_agreement : RowAgreement 69 70 := RowAgreement.single (first := 69) modelrow_agrees_70
private def atomic_71 : CountSegment 1 10 13120 70 71 65 66 := CountSegment.single (first := 70) (initial := 65) row_count_71 (by decide)
private theorem atomic_71_agreement : RowAgreement 70 71 := RowAgreement.single (first := 70) modelrow_agrees_71
private def atomic_72 : CountSegment 1 10 13120 71 72 66 67 := CountSegment.single (first := 71) (initial := 66) row_count_72 (by decide)
private theorem atomic_72_agreement : RowAgreement 71 72 := RowAgreement.single (first := 71) modelrow_agrees_72
private def atomic_73 : CountSegment 1 10 13120 72 73 67 68 := CountSegment.single (first := 72) (initial := 67) row_count_73 (by decide)
private theorem atomic_73_agreement : RowAgreement 72 73 := RowAgreement.single (first := 72) modelrow_agrees_73
private def atomic_74 : CountSegment 1 10 13120 73 74 68 69 := CountSegment.single (first := 73) (initial := 68) row_count_74 (by decide)
private theorem atomic_74_agreement : RowAgreement 73 74 := RowAgreement.single (first := 73) modelrow_agrees_74
private def atomic_75 : CountSegment 1 10 13120 74 75 69 71 := CountSegment.single (first := 74) (initial := 69) row_count_75 (by decide)
private theorem atomic_75_agreement : RowAgreement 74 75 := RowAgreement.single (first := 74) modelrow_agrees_75
private def atomic_76 : CountSegment 1 10 13120 75 76 71 71 := CountSegment.single (first := 75) (initial := 71) row_count_76 (by decide)
private theorem atomic_76_agreement : RowAgreement 75 76 := RowAgreement.single (first := 75) modelrow_agrees_76
private def atomic_77 : CountSegment 1 10 13120 76 77 71 72 := CountSegment.single (first := 76) (initial := 71) row_count_77 (by decide)
private theorem atomic_77_agreement : RowAgreement 76 77 := RowAgreement.single (first := 76) modelrow_agrees_77
private def atomic_78 : CountSegment 1 10 13120 77 78 72 74 := CountSegment.single (first := 77) (initial := 72) row_count_78 (by decide)
private theorem atomic_78_agreement : RowAgreement 77 78 := RowAgreement.single (first := 77) modelrow_agrees_78
private def atomic_79 : CountSegment 1 10 13120 78 79 74 74 := CountSegment.single (first := 78) (initial := 74) row_count_79 (by decide)
private theorem atomic_79_agreement : RowAgreement 78 79 := RowAgreement.single (first := 78) modelrow_agrees_79
private def atomic_80 : CountSegment 1 10 13120 79 80 74 75 := CountSegment.single (first := 79) (initial := 74) row_count_80 (by decide)
private theorem atomic_80_agreement : RowAgreement 79 80 := RowAgreement.single (first := 79) modelrow_agrees_80
private def atomic_81 : CountSegment 1 10 13120 80 81 75 77 := CountSegment.single (first := 80) (initial := 75) row_count_81 (by decide)
private theorem atomic_81_agreement : RowAgreement 80 81 := RowAgreement.single (first := 80) modelrow_agrees_81
private def atomic_82 : CountSegment 1 10 13120 81 82 77 78 := CountSegment.single (first := 81) (initial := 77) row_count_82 (by decide)
private theorem atomic_82_agreement : RowAgreement 81 82 := RowAgreement.single (first := 81) modelrow_agrees_82
private def atomic_83 : CountSegment 1 10 13120 82 83 78 78 := CountSegment.single (first := 82) (initial := 78) row_count_83 (by decide)
private theorem atomic_83_agreement : RowAgreement 82 83 := RowAgreement.single (first := 82) modelrow_agrees_83
private def atomic_84 : CountSegment 1 10 13120 83 84 78 80 := CountSegment.single (first := 83) (initial := 78) row_count_84 (by decide)
private theorem atomic_84_agreement : RowAgreement 83 84 := RowAgreement.single (first := 83) modelrow_agrees_84
private def atomic_85 : CountSegment 1 10 13120 84 85 80 81 := CountSegment.single (first := 84) (initial := 80) row_count_85 (by decide)
private theorem atomic_85_agreement : RowAgreement 84 85 := RowAgreement.single (first := 84) modelrow_agrees_85
private def atomic_86 : CountSegment 1 10 13120 85 86 81 81 := CountSegment.single (first := 85) (initial := 81) row_count_86 (by decide)
private theorem atomic_86_agreement : RowAgreement 85 86 := RowAgreement.single (first := 85) modelrow_agrees_86
private def atomic_87 : CountSegment 1 10 13120 86 87 81 82 := CountSegment.single (first := 86) (initial := 81) row_count_87 (by decide)
private theorem atomic_87_agreement : RowAgreement 86 87 := RowAgreement.single (first := 86) modelrow_agrees_87
private def atomic_88 : CountSegment 1 10 13120 87 88 82 84 := CountSegment.single (first := 87) (initial := 82) row_count_88 (by decide)
private theorem atomic_88_agreement : RowAgreement 87 88 := RowAgreement.single (first := 87) modelrow_agrees_88
private def atomic_89 : CountSegment 1 10 13120 88 89 84 84 := CountSegment.single (first := 88) (initial := 84) row_count_89 (by decide)
private theorem atomic_89_agreement : RowAgreement 88 89 := RowAgreement.single (first := 88) modelrow_agrees_89
private def atomic_90 : CountSegment 1 10 13120 89 90 84 85 := CountSegment.single (first := 89) (initial := 84) row_count_90 (by decide)
private theorem atomic_90_agreement : RowAgreement 89 90 := RowAgreement.single (first := 89) modelrow_agrees_90
private def atomic_91 : CountSegment 1 10 13120 90 91 85 85 := CountSegment.single (first := 90) (initial := 85) row_count_91 (by decide)
private theorem atomic_91_agreement : RowAgreement 90 91 := RowAgreement.single (first := 90) modelrow_agrees_91
private def atomic_92 : CountSegment 1 10 13120 91 92 85 86 := CountSegment.single (first := 91) (initial := 85) row_count_92 (by decide)
private theorem atomic_92_agreement : RowAgreement 91 92 := RowAgreement.single (first := 91) modelrow_agrees_92
private def atomic_93 : CountSegment 1 10 13120 92 93 86 86 := CountSegment.single (first := 92) (initial := 86) row_count_93 (by decide)
private theorem atomic_93_agreement : RowAgreement 92 93 := RowAgreement.single (first := 92) modelrow_agrees_93
private def atomic_94 : CountSegment 1 10 13120 93 94 86 87 := CountSegment.single (first := 93) (initial := 86) row_count_94 (by decide)
private theorem atomic_94_agreement : RowAgreement 93 94 := RowAgreement.single (first := 93) modelrow_agrees_94
private def atomic_95 : CountSegment 1 10 13120 94 95 87 88 := CountSegment.single (first := 94) (initial := 87) row_count_95 (by decide)
private theorem atomic_95_agreement : RowAgreement 94 95 := RowAgreement.single (first := 94) modelrow_agrees_95
private def atomic_96 : CountSegment 1 10 13120 95 96 88 89 := CountSegment.single (first := 95) (initial := 88) row_count_96 (by decide)
private theorem atomic_96_agreement : RowAgreement 95 96 := RowAgreement.single (first := 95) modelrow_agrees_96
private def atomic_97 : CountSegment 1 10 13120 96 97 89 90 := CountSegment.single (first := 96) (initial := 89) row_count_97 (by decide)
private theorem atomic_97_agreement : RowAgreement 96 97 := RowAgreement.single (first := 96) modelrow_agrees_97
private def atomic_98 : CountSegment 1 10 13120 97 98 90 90 := CountSegment.single (first := 97) (initial := 90) row_count_98 (by decide)
private theorem atomic_98_agreement : RowAgreement 97 98 := RowAgreement.single (first := 97) modelrow_agrees_98
private def atomic_99 : CountSegment 1 10 13120 98 99 90 90 := CountSegment.single (first := 98) (initial := 90) row_count_99 (by decide)
private theorem atomic_99_agreement : RowAgreement 98 99 := RowAgreement.single (first := 98) modelrow_agrees_99
private def atomic_100 : CountSegment 1 10 13120 99 100 90 90 := CountSegment.single (first := 99) (initial := 90) row_count_100 (by decide)
private theorem atomic_100_agreement : RowAgreement 99 100 := RowAgreement.single (first := 99) modelrow_agrees_100
private def atomic_101 : CountSegment 1 10 13120 100 101 90 90 := CountSegment.single (first := 100) (initial := 90) row_count_101 (by decide)
private theorem atomic_101_agreement : RowAgreement 100 101 := RowAgreement.single (first := 100) modelrow_agrees_101
private def atomic_102 : CountSegment 1 10 13120 101 102 90 92 := CountSegment.single (first := 101) (initial := 90) row_count_102 (by decide)
private theorem atomic_102_agreement : RowAgreement 101 102 := RowAgreement.single (first := 101) modelrow_agrees_102
private def atomic_103 : CountSegment 1 10 13120 102 103 92 94 := CountSegment.single (first := 102) (initial := 92) row_count_103 (by decide)
private theorem atomic_103_agreement : RowAgreement 102 103 := RowAgreement.single (first := 102) modelrow_agrees_103
private def atomic_104 : CountSegment 1 10 13120 103 104 94 94 := CountSegment.single (first := 103) (initial := 94) row_count_104 (by decide)
private theorem atomic_104_agreement : RowAgreement 103 104 := RowAgreement.single (first := 103) modelrow_agrees_104
private def atomic_105 : CountSegment 1 10 13120 104 105 94 96 := CountSegment.single (first := 104) (initial := 94) row_count_105 (by decide)
private theorem atomic_105_agreement : RowAgreement 104 105 := RowAgreement.single (first := 104) modelrow_agrees_105
private def atomic_106 : CountSegment 1 10 13120 105 106 96 96 := CountSegment.single (first := 105) (initial := 96) row_count_106 (by decide)
private theorem atomic_106_agreement : RowAgreement 105 106 := RowAgreement.single (first := 105) modelrow_agrees_106
private def atomic_107 : CountSegment 1 10 13120 106 107 96 96 := CountSegment.single (first := 106) (initial := 96) row_count_107 (by decide)
private theorem atomic_107_agreement : RowAgreement 106 107 := RowAgreement.single (first := 106) modelrow_agrees_107
private def atomic_108 : CountSegment 1 10 13120 107 108 96 97 := CountSegment.single (first := 107) (initial := 96) row_count_108 (by decide)
private theorem atomic_108_agreement : RowAgreement 107 108 := RowAgreement.single (first := 107) modelrow_agrees_108
private def atomic_109 : CountSegment 1 10 13120 108 109 97 98 := CountSegment.single (first := 108) (initial := 97) row_count_109 (by decide)
private theorem atomic_109_agreement : RowAgreement 108 109 := RowAgreement.single (first := 108) modelrow_agrees_109
private def atomic_110 : CountSegment 1 10 13120 109 110 98 98 := CountSegment.single (first := 109) (initial := 98) row_count_110 (by decide)
private theorem atomic_110_agreement : RowAgreement 109 110 := RowAgreement.single (first := 109) modelrow_agrees_110
private def atomic_111 : CountSegment 1 10 13120 110 111 98 98 := CountSegment.single (first := 110) (initial := 98) row_count_111 (by decide)
private theorem atomic_111_agreement : RowAgreement 110 111 := RowAgreement.single (first := 110) modelrow_agrees_111
private def atomic_112 : CountSegment 1 10 13120 111 112 98 99 := CountSegment.single (first := 111) (initial := 98) row_count_112 (by decide)
private theorem atomic_112_agreement : RowAgreement 111 112 := RowAgreement.single (first := 111) modelrow_agrees_112
private def atomic_113 : CountSegment 1 10 13120 112 113 99 99 := CountSegment.single (first := 112) (initial := 99) row_count_113 (by decide)
private theorem atomic_113_agreement : RowAgreement 112 113 := RowAgreement.single (first := 112) modelrow_agrees_113
private def atomic_114 : CountSegment 1 10 13120 113 114 99 99 := CountSegment.single (first := 113) (initial := 99) row_count_114 (by decide)
private theorem atomic_114_agreement : RowAgreement 113 114 := RowAgreement.single (first := 113) modelrow_agrees_114
private def atomic_115 : CountSegment 1 10 13120 114 115 99 100 := CountSegment.single (first := 114) (initial := 99) row_count_115 (by decide)
private theorem atomic_115_agreement : RowAgreement 114 115 := RowAgreement.single (first := 114) modelrow_agrees_115
private def atomic_116 : CountSegment 1 10 13120 115 116 100 101 := CountSegment.single (first := 115) (initial := 100) row_count_116 (by decide)
private theorem atomic_116_agreement : RowAgreement 115 116 := RowAgreement.single (first := 115) modelrow_agrees_116
private def atomic_117 : CountSegment 1 10 13120 116 117 101 101 := CountSegment.single (first := 116) (initial := 101) row_count_117 (by decide)
private theorem atomic_117_agreement : RowAgreement 116 117 := RowAgreement.single (first := 116) modelrow_agrees_117
private def atomic_118 : CountSegment 1 10 13120 117 118 101 103 := CountSegment.single (first := 117) (initial := 101) row_count_118 (by decide)
private theorem atomic_118_agreement : RowAgreement 117 118 := RowAgreement.single (first := 117) modelrow_agrees_118
private def atomic_119 : CountSegment 1 10 13120 118 119 103 103 := CountSegment.single (first := 118) (initial := 103) row_count_119 (by decide)
private theorem atomic_119_agreement : RowAgreement 118 119 := RowAgreement.single (first := 118) modelrow_agrees_119
private def atomic_120 : CountSegment 1 10 13120 119 120 103 104 := CountSegment.single (first := 119) (initial := 103) row_count_120 (by decide)
private theorem atomic_120_agreement : RowAgreement 119 120 := RowAgreement.single (first := 119) modelrow_agrees_120
private def atomic_121 : CountSegment 1 10 13120 120 121 104 104 := CountSegment.single (first := 120) (initial := 104) row_count_121 (by decide)
private theorem atomic_121_agreement : RowAgreement 120 121 := RowAgreement.single (first := 120) modelrow_agrees_121
private def atomic_122 : CountSegment 1 10 13120 121 122 104 104 := CountSegment.single (first := 121) (initial := 104) row_count_122 (by decide)
private theorem atomic_122_agreement : RowAgreement 121 122 := RowAgreement.single (first := 121) modelrow_agrees_122
private def atomic_123 : CountSegment 1 10 13120 122 123 104 106 := CountSegment.single (first := 122) (initial := 104) row_count_123 (by decide)
private theorem atomic_123_agreement : RowAgreement 122 123 := RowAgreement.single (first := 122) modelrow_agrees_123
private def atomic_124 : CountSegment 1 10 13120 123 124 106 106 := CountSegment.single (first := 123) (initial := 106) row_count_124 (by decide)
private theorem atomic_124_agreement : RowAgreement 123 124 := RowAgreement.single (first := 123) modelrow_agrees_124
private def atomic_125 : CountSegment 1 10 13120 124 125 106 107 := CountSegment.single (first := 124) (initial := 106) row_count_125 (by decide)
private theorem atomic_125_agreement : RowAgreement 124 125 := RowAgreement.single (first := 124) modelrow_agrees_125
private def atomic_126 : CountSegment 1 10 13120 125 126 107 107 := CountSegment.single (first := 125) (initial := 107) row_count_126 (by decide)
private theorem atomic_126_agreement : RowAgreement 125 126 := RowAgreement.single (first := 125) modelrow_agrees_126
private def atomic_127 : CountSegment 1 10 13120 126 127 107 107 := CountSegment.single (first := 126) (initial := 107) row_count_127 (by decide)
private theorem atomic_127_agreement : RowAgreement 126 127 := RowAgreement.single (first := 126) modelrow_agrees_127
private def atomic_128 : CountSegment 1 10 13120 127 128 107 108 := CountSegment.single (first := 127) (initial := 107) row_count_128 (by decide)
private theorem atomic_128_agreement : RowAgreement 127 128 := RowAgreement.single (first := 127) modelrow_agrees_128
private def within_000_000 : CountSegment 1 10 13120 0 2 0 5 := CountSegment.join atomic_1 atomic_2
private theorem within_000_000_agreement : RowAgreement 0 2 := RowAgreement.join atomic_1_agreement atomic_2_agreement
private def within_000_001 : CountSegment 1 10 13120 2 4 5 9 := CountSegment.join atomic_3 atomic_4
private theorem within_000_001_agreement : RowAgreement 2 4 := RowAgreement.join atomic_3_agreement atomic_4_agreement
private def within_000_002 : CountSegment 1 10 13120 0 4 0 9 := CountSegment.join within_000_000 within_000_001
private theorem within_000_002_agreement : RowAgreement 0 4 := RowAgreement.join within_000_000_agreement within_000_001_agreement
private def within_000_003 : CountSegment 1 10 13120 4 6 9 11 := CountSegment.join atomic_5 atomic_6
private theorem within_000_003_agreement : RowAgreement 4 6 := RowAgreement.join atomic_5_agreement atomic_6_agreement
private def within_000_004 : CountSegment 1 10 13120 6 8 11 15 := CountSegment.join atomic_7 atomic_8
private theorem within_000_004_agreement : RowAgreement 6 8 := RowAgreement.join atomic_7_agreement atomic_8_agreement
private def within_000_005 : CountSegment 1 10 13120 4 8 9 15 := CountSegment.join within_000_003 within_000_004
private theorem within_000_005_agreement : RowAgreement 4 8 := RowAgreement.join within_000_003_agreement within_000_004_agreement
private def within_000_006 : CountSegment 1 10 13120 0 8 0 15 := CountSegment.join within_000_002 within_000_005
private theorem within_000_006_agreement : RowAgreement 0 8 := RowAgreement.join within_000_002_agreement within_000_005_agreement
private def within_000_007 : CountSegment 1 10 13120 8 10 15 18 := CountSegment.join atomic_9 atomic_10
private theorem within_000_007_agreement : RowAgreement 8 10 := RowAgreement.join atomic_9_agreement atomic_10_agreement
private def within_000_008 : CountSegment 1 10 13120 10 12 18 21 := CountSegment.join atomic_11 atomic_12
private theorem within_000_008_agreement : RowAgreement 10 12 := RowAgreement.join atomic_11_agreement atomic_12_agreement
private def within_000_009 : CountSegment 1 10 13120 8 12 15 21 := CountSegment.join within_000_007 within_000_008
private theorem within_000_009_agreement : RowAgreement 8 12 := RowAgreement.join within_000_007_agreement within_000_008_agreement
private def within_000_010 : CountSegment 1 10 13120 12 14 21 23 := CountSegment.join atomic_13 atomic_14
private theorem within_000_010_agreement : RowAgreement 12 14 := RowAgreement.join atomic_13_agreement atomic_14_agreement
private def within_000_011 : CountSegment 1 10 13120 14 16 23 23 := CountSegment.join atomic_15 atomic_16
private theorem within_000_011_agreement : RowAgreement 14 16 := RowAgreement.join atomic_15_agreement atomic_16_agreement
private def within_000_012 : CountSegment 1 10 13120 12 16 21 23 := CountSegment.join within_000_010 within_000_011
private theorem within_000_012_agreement : RowAgreement 12 16 := RowAgreement.join within_000_010_agreement within_000_011_agreement
private def within_000_013 : CountSegment 1 10 13120 8 16 15 23 := CountSegment.join within_000_009 within_000_012
private theorem within_000_013_agreement : RowAgreement 8 16 := RowAgreement.join within_000_009_agreement within_000_012_agreement
private def within_000_014 : CountSegment 1 10 13120 0 16 0 23 := CountSegment.join within_000_006 within_000_013
private theorem within_000_014_agreement : RowAgreement 0 16 := RowAgreement.join within_000_006_agreement within_000_013_agreement
private def within_000_015 : CountSegment 1 10 13120 16 18 23 25 := CountSegment.join atomic_17 atomic_18
private theorem within_000_015_agreement : RowAgreement 16 18 := RowAgreement.join atomic_17_agreement atomic_18_agreement
private def within_000_016 : CountSegment 1 10 13120 18 20 25 27 := CountSegment.join atomic_19 atomic_20
private theorem within_000_016_agreement : RowAgreement 18 20 := RowAgreement.join atomic_19_agreement atomic_20_agreement
private def within_000_017 : CountSegment 1 10 13120 16 20 23 27 := CountSegment.join within_000_015 within_000_016
private theorem within_000_017_agreement : RowAgreement 16 20 := RowAgreement.join within_000_015_agreement within_000_016_agreement
private def within_000_018 : CountSegment 1 10 13120 20 22 27 28 := CountSegment.join atomic_21 atomic_22
private theorem within_000_018_agreement : RowAgreement 20 22 := RowAgreement.join atomic_21_agreement atomic_22_agreement
private def within_000_019 : CountSegment 1 10 13120 22 24 28 28 := CountSegment.join atomic_23 atomic_24
private theorem within_000_019_agreement : RowAgreement 22 24 := RowAgreement.join atomic_23_agreement atomic_24_agreement
private def within_000_020 : CountSegment 1 10 13120 20 24 27 28 := CountSegment.join within_000_018 within_000_019
private theorem within_000_020_agreement : RowAgreement 20 24 := RowAgreement.join within_000_018_agreement within_000_019_agreement
private def within_000_021 : CountSegment 1 10 13120 16 24 23 28 := CountSegment.join within_000_017 within_000_020
private theorem within_000_021_agreement : RowAgreement 16 24 := RowAgreement.join within_000_017_agreement within_000_020_agreement
private def within_000_022 : CountSegment 1 10 13120 24 26 28 29 := CountSegment.join atomic_25 atomic_26
private theorem within_000_022_agreement : RowAgreement 24 26 := RowAgreement.join atomic_25_agreement atomic_26_agreement
private def within_000_023 : CountSegment 1 10 13120 26 28 29 32 := CountSegment.join atomic_27 atomic_28
private theorem within_000_023_agreement : RowAgreement 26 28 := RowAgreement.join atomic_27_agreement atomic_28_agreement
private def within_000_024 : CountSegment 1 10 13120 24 28 28 32 := CountSegment.join within_000_022 within_000_023
private theorem within_000_024_agreement : RowAgreement 24 28 := RowAgreement.join within_000_022_agreement within_000_023_agreement
private def within_000_025 : CountSegment 1 10 13120 28 30 32 34 := CountSegment.join atomic_29 atomic_30
private theorem within_000_025_agreement : RowAgreement 28 30 := RowAgreement.join atomic_29_agreement atomic_30_agreement
private def within_000_026 : CountSegment 1 10 13120 30 32 34 35 := CountSegment.join atomic_31 atomic_32
private theorem within_000_026_agreement : RowAgreement 30 32 := RowAgreement.join atomic_31_agreement atomic_32_agreement
private def within_000_027 : CountSegment 1 10 13120 28 32 32 35 := CountSegment.join within_000_025 within_000_026
private theorem within_000_027_agreement : RowAgreement 28 32 := RowAgreement.join within_000_025_agreement within_000_026_agreement
private def within_000_028 : CountSegment 1 10 13120 24 32 28 35 := CountSegment.join within_000_024 within_000_027
private theorem within_000_028_agreement : RowAgreement 24 32 := RowAgreement.join within_000_024_agreement within_000_027_agreement
private def within_000_029 : CountSegment 1 10 13120 16 32 23 35 := CountSegment.join within_000_021 within_000_028
private theorem within_000_029_agreement : RowAgreement 16 32 := RowAgreement.join within_000_021_agreement within_000_028_agreement
private def within_000_030 : CountSegment 1 10 13120 0 32 0 35 := CountSegment.join within_000_014 within_000_029
private theorem within_000_030_agreement : RowAgreement 0 32 := RowAgreement.join within_000_014_agreement within_000_029_agreement
private def within_000_031 : CountSegment 1 10 13120 32 34 35 37 := CountSegment.join atomic_33 atomic_34
private theorem within_000_031_agreement : RowAgreement 32 34 := RowAgreement.join atomic_33_agreement atomic_34_agreement
private def within_000_032 : CountSegment 1 10 13120 34 36 37 39 := CountSegment.join atomic_35 atomic_36
private theorem within_000_032_agreement : RowAgreement 34 36 := RowAgreement.join atomic_35_agreement atomic_36_agreement
private def within_000_033 : CountSegment 1 10 13120 32 36 35 39 := CountSegment.join within_000_031 within_000_032
private theorem within_000_033_agreement : RowAgreement 32 36 := RowAgreement.join within_000_031_agreement within_000_032_agreement
private def within_000_034 : CountSegment 1 10 13120 36 38 39 41 := CountSegment.join atomic_37 atomic_38
private theorem within_000_034_agreement : RowAgreement 36 38 := RowAgreement.join atomic_37_agreement atomic_38_agreement
private def within_000_035 : CountSegment 1 10 13120 38 40 41 44 := CountSegment.join atomic_39 atomic_40
private theorem within_000_035_agreement : RowAgreement 38 40 := RowAgreement.join atomic_39_agreement atomic_40_agreement
private def within_000_036 : CountSegment 1 10 13120 36 40 39 44 := CountSegment.join within_000_034 within_000_035
private theorem within_000_036_agreement : RowAgreement 36 40 := RowAgreement.join within_000_034_agreement within_000_035_agreement
private def within_000_037 : CountSegment 1 10 13120 32 40 35 44 := CountSegment.join within_000_033 within_000_036
private theorem within_000_037_agreement : RowAgreement 32 40 := RowAgreement.join within_000_033_agreement within_000_036_agreement
private def within_000_038 : CountSegment 1 10 13120 40 42 44 45 := CountSegment.join atomic_41 atomic_42
private theorem within_000_038_agreement : RowAgreement 40 42 := RowAgreement.join atomic_41_agreement atomic_42_agreement
private def within_000_039 : CountSegment 1 10 13120 42 44 45 45 := CountSegment.join atomic_43 atomic_44
private theorem within_000_039_agreement : RowAgreement 42 44 := RowAgreement.join atomic_43_agreement atomic_44_agreement
private def within_000_040 : CountSegment 1 10 13120 40 44 44 45 := CountSegment.join within_000_038 within_000_039
private theorem within_000_040_agreement : RowAgreement 40 44 := RowAgreement.join within_000_038_agreement within_000_039_agreement
private def within_000_041 : CountSegment 1 10 13120 44 46 45 48 := CountSegment.join atomic_45 atomic_46
private theorem within_000_041_agreement : RowAgreement 44 46 := RowAgreement.join atomic_45_agreement atomic_46_agreement
private def within_000_042 : CountSegment 1 10 13120 46 48 48 49 := CountSegment.join atomic_47 atomic_48
private theorem within_000_042_agreement : RowAgreement 46 48 := RowAgreement.join atomic_47_agreement atomic_48_agreement
private def within_000_043 : CountSegment 1 10 13120 44 48 45 49 := CountSegment.join within_000_041 within_000_042
private theorem within_000_043_agreement : RowAgreement 44 48 := RowAgreement.join within_000_041_agreement within_000_042_agreement
private def within_000_044 : CountSegment 1 10 13120 40 48 44 49 := CountSegment.join within_000_040 within_000_043
private theorem within_000_044_agreement : RowAgreement 40 48 := RowAgreement.join within_000_040_agreement within_000_043_agreement
private def within_000_045 : CountSegment 1 10 13120 32 48 35 49 := CountSegment.join within_000_037 within_000_044
private theorem within_000_045_agreement : RowAgreement 32 48 := RowAgreement.join within_000_037_agreement within_000_044_agreement
private def within_000_046 : CountSegment 1 10 13120 48 50 49 50 := CountSegment.join atomic_49 atomic_50
private theorem within_000_046_agreement : RowAgreement 48 50 := RowAgreement.join atomic_49_agreement atomic_50_agreement
private def within_000_047 : CountSegment 1 10 13120 50 52 50 51 := CountSegment.join atomic_51 atomic_52
private theorem within_000_047_agreement : RowAgreement 50 52 := RowAgreement.join atomic_51_agreement atomic_52_agreement
private def within_000_048 : CountSegment 1 10 13120 48 52 49 51 := CountSegment.join within_000_046 within_000_047
private theorem within_000_048_agreement : RowAgreement 48 52 := RowAgreement.join within_000_046_agreement within_000_047_agreement
private def within_000_049 : CountSegment 1 10 13120 52 54 51 52 := CountSegment.join atomic_53 atomic_54
private theorem within_000_049_agreement : RowAgreement 52 54 := RowAgreement.join atomic_53_agreement atomic_54_agreement
private def within_000_050 : CountSegment 1 10 13120 54 56 52 55 := CountSegment.join atomic_55 atomic_56
private theorem within_000_050_agreement : RowAgreement 54 56 := RowAgreement.join atomic_55_agreement atomic_56_agreement
private def within_000_051 : CountSegment 1 10 13120 52 56 51 55 := CountSegment.join within_000_049 within_000_050
private theorem within_000_051_agreement : RowAgreement 52 56 := RowAgreement.join within_000_049_agreement within_000_050_agreement
private def within_000_052 : CountSegment 1 10 13120 48 56 49 55 := CountSegment.join within_000_048 within_000_051
private theorem within_000_052_agreement : RowAgreement 48 56 := RowAgreement.join within_000_048_agreement within_000_051_agreement
private def within_000_053 : CountSegment 1 10 13120 56 58 55 56 := CountSegment.join atomic_57 atomic_58
private theorem within_000_053_agreement : RowAgreement 56 58 := RowAgreement.join atomic_57_agreement atomic_58_agreement
private def within_000_054 : CountSegment 1 10 13120 58 60 56 59 := CountSegment.join atomic_59 atomic_60
private theorem within_000_054_agreement : RowAgreement 58 60 := RowAgreement.join atomic_59_agreement atomic_60_agreement
private def within_000_055 : CountSegment 1 10 13120 56 60 55 59 := CountSegment.join within_000_053 within_000_054
private theorem within_000_055_agreement : RowAgreement 56 60 := RowAgreement.join within_000_053_agreement within_000_054_agreement
private def within_000_056 : CountSegment 1 10 13120 60 62 59 60 := CountSegment.join atomic_61 atomic_62
private theorem within_000_056_agreement : RowAgreement 60 62 := RowAgreement.join atomic_61_agreement atomic_62_agreement
private def within_000_057 : CountSegment 1 10 13120 62 64 60 61 := CountSegment.join atomic_63 atomic_64
private theorem within_000_057_agreement : RowAgreement 62 64 := RowAgreement.join atomic_63_agreement atomic_64_agreement
private def within_000_058 : CountSegment 1 10 13120 60 64 59 61 := CountSegment.join within_000_056 within_000_057
private theorem within_000_058_agreement : RowAgreement 60 64 := RowAgreement.join within_000_056_agreement within_000_057_agreement
private def within_000_059 : CountSegment 1 10 13120 56 64 55 61 := CountSegment.join within_000_055 within_000_058
private theorem within_000_059_agreement : RowAgreement 56 64 := RowAgreement.join within_000_055_agreement within_000_058_agreement
private def within_000_060 : CountSegment 1 10 13120 48 64 49 61 := CountSegment.join within_000_052 within_000_059
private theorem within_000_060_agreement : RowAgreement 48 64 := RowAgreement.join within_000_052_agreement within_000_059_agreement
private def within_000_061 : CountSegment 1 10 13120 32 64 35 61 := CountSegment.join within_000_045 within_000_060
private theorem within_000_061_agreement : RowAgreement 32 64 := RowAgreement.join within_000_045_agreement within_000_060_agreement
private def within_000_062 : CountSegment 1 10 13120 0 64 0 61 := CountSegment.join within_000_030 within_000_061
private theorem within_000_062_agreement : RowAgreement 0 64 := RowAgreement.join within_000_030_agreement within_000_061_agreement
private def within_000_063 : CountSegment 1 10 13120 64 66 61 63 := CountSegment.join atomic_65 atomic_66
private theorem within_000_063_agreement : RowAgreement 64 66 := RowAgreement.join atomic_65_agreement atomic_66_agreement
private def within_000_064 : CountSegment 1 10 13120 66 68 63 64 := CountSegment.join atomic_67 atomic_68
private theorem within_000_064_agreement : RowAgreement 66 68 := RowAgreement.join atomic_67_agreement atomic_68_agreement
private def within_000_065 : CountSegment 1 10 13120 64 68 61 64 := CountSegment.join within_000_063 within_000_064
private theorem within_000_065_agreement : RowAgreement 64 68 := RowAgreement.join within_000_063_agreement within_000_064_agreement
private def within_000_066 : CountSegment 1 10 13120 68 70 64 65 := CountSegment.join atomic_69 atomic_70
private theorem within_000_066_agreement : RowAgreement 68 70 := RowAgreement.join atomic_69_agreement atomic_70_agreement
private def within_000_067 : CountSegment 1 10 13120 70 72 65 67 := CountSegment.join atomic_71 atomic_72
private theorem within_000_067_agreement : RowAgreement 70 72 := RowAgreement.join atomic_71_agreement atomic_72_agreement
private def within_000_068 : CountSegment 1 10 13120 68 72 64 67 := CountSegment.join within_000_066 within_000_067
private theorem within_000_068_agreement : RowAgreement 68 72 := RowAgreement.join within_000_066_agreement within_000_067_agreement
private def within_000_069 : CountSegment 1 10 13120 64 72 61 67 := CountSegment.join within_000_065 within_000_068
private theorem within_000_069_agreement : RowAgreement 64 72 := RowAgreement.join within_000_065_agreement within_000_068_agreement
private def within_000_070 : CountSegment 1 10 13120 72 74 67 69 := CountSegment.join atomic_73 atomic_74
private theorem within_000_070_agreement : RowAgreement 72 74 := RowAgreement.join atomic_73_agreement atomic_74_agreement
private def within_000_071 : CountSegment 1 10 13120 74 76 69 71 := CountSegment.join atomic_75 atomic_76
private theorem within_000_071_agreement : RowAgreement 74 76 := RowAgreement.join atomic_75_agreement atomic_76_agreement
private def within_000_072 : CountSegment 1 10 13120 72 76 67 71 := CountSegment.join within_000_070 within_000_071
private theorem within_000_072_agreement : RowAgreement 72 76 := RowAgreement.join within_000_070_agreement within_000_071_agreement
private def within_000_073 : CountSegment 1 10 13120 76 78 71 74 := CountSegment.join atomic_77 atomic_78
private theorem within_000_073_agreement : RowAgreement 76 78 := RowAgreement.join atomic_77_agreement atomic_78_agreement
private def within_000_074 : CountSegment 1 10 13120 78 80 74 75 := CountSegment.join atomic_79 atomic_80
private theorem within_000_074_agreement : RowAgreement 78 80 := RowAgreement.join atomic_79_agreement atomic_80_agreement
private def within_000_075 : CountSegment 1 10 13120 76 80 71 75 := CountSegment.join within_000_073 within_000_074
private theorem within_000_075_agreement : RowAgreement 76 80 := RowAgreement.join within_000_073_agreement within_000_074_agreement
private def within_000_076 : CountSegment 1 10 13120 72 80 67 75 := CountSegment.join within_000_072 within_000_075
private theorem within_000_076_agreement : RowAgreement 72 80 := RowAgreement.join within_000_072_agreement within_000_075_agreement
private def within_000_077 : CountSegment 1 10 13120 64 80 61 75 := CountSegment.join within_000_069 within_000_076
private theorem within_000_077_agreement : RowAgreement 64 80 := RowAgreement.join within_000_069_agreement within_000_076_agreement
private def within_000_078 : CountSegment 1 10 13120 80 82 75 78 := CountSegment.join atomic_81 atomic_82
private theorem within_000_078_agreement : RowAgreement 80 82 := RowAgreement.join atomic_81_agreement atomic_82_agreement
private def within_000_079 : CountSegment 1 10 13120 82 84 78 80 := CountSegment.join atomic_83 atomic_84
private theorem within_000_079_agreement : RowAgreement 82 84 := RowAgreement.join atomic_83_agreement atomic_84_agreement
private def within_000_080 : CountSegment 1 10 13120 80 84 75 80 := CountSegment.join within_000_078 within_000_079
private theorem within_000_080_agreement : RowAgreement 80 84 := RowAgreement.join within_000_078_agreement within_000_079_agreement
private def within_000_081 : CountSegment 1 10 13120 84 86 80 81 := CountSegment.join atomic_85 atomic_86
private theorem within_000_081_agreement : RowAgreement 84 86 := RowAgreement.join atomic_85_agreement atomic_86_agreement
private def within_000_082 : CountSegment 1 10 13120 86 88 81 84 := CountSegment.join atomic_87 atomic_88
private theorem within_000_082_agreement : RowAgreement 86 88 := RowAgreement.join atomic_87_agreement atomic_88_agreement
private def within_000_083 : CountSegment 1 10 13120 84 88 80 84 := CountSegment.join within_000_081 within_000_082
private theorem within_000_083_agreement : RowAgreement 84 88 := RowAgreement.join within_000_081_agreement within_000_082_agreement
private def within_000_084 : CountSegment 1 10 13120 80 88 75 84 := CountSegment.join within_000_080 within_000_083
private theorem within_000_084_agreement : RowAgreement 80 88 := RowAgreement.join within_000_080_agreement within_000_083_agreement
private def within_000_085 : CountSegment 1 10 13120 88 90 84 85 := CountSegment.join atomic_89 atomic_90
private theorem within_000_085_agreement : RowAgreement 88 90 := RowAgreement.join atomic_89_agreement atomic_90_agreement
private def within_000_086 : CountSegment 1 10 13120 90 92 85 86 := CountSegment.join atomic_91 atomic_92
private theorem within_000_086_agreement : RowAgreement 90 92 := RowAgreement.join atomic_91_agreement atomic_92_agreement
private def within_000_087 : CountSegment 1 10 13120 88 92 84 86 := CountSegment.join within_000_085 within_000_086
private theorem within_000_087_agreement : RowAgreement 88 92 := RowAgreement.join within_000_085_agreement within_000_086_agreement
private def within_000_088 : CountSegment 1 10 13120 92 94 86 87 := CountSegment.join atomic_93 atomic_94
private theorem within_000_088_agreement : RowAgreement 92 94 := RowAgreement.join atomic_93_agreement atomic_94_agreement
private def within_000_089 : CountSegment 1 10 13120 94 96 87 89 := CountSegment.join atomic_95 atomic_96
private theorem within_000_089_agreement : RowAgreement 94 96 := RowAgreement.join atomic_95_agreement atomic_96_agreement
private def within_000_090 : CountSegment 1 10 13120 92 96 86 89 := CountSegment.join within_000_088 within_000_089
private theorem within_000_090_agreement : RowAgreement 92 96 := RowAgreement.join within_000_088_agreement within_000_089_agreement
private def within_000_091 : CountSegment 1 10 13120 88 96 84 89 := CountSegment.join within_000_087 within_000_090
private theorem within_000_091_agreement : RowAgreement 88 96 := RowAgreement.join within_000_087_agreement within_000_090_agreement
private def within_000_092 : CountSegment 1 10 13120 80 96 75 89 := CountSegment.join within_000_084 within_000_091
private theorem within_000_092_agreement : RowAgreement 80 96 := RowAgreement.join within_000_084_agreement within_000_091_agreement
private def within_000_093 : CountSegment 1 10 13120 64 96 61 89 := CountSegment.join within_000_077 within_000_092
private theorem within_000_093_agreement : RowAgreement 64 96 := RowAgreement.join within_000_077_agreement within_000_092_agreement
private def within_000_094 : CountSegment 1 10 13120 96 98 89 90 := CountSegment.join atomic_97 atomic_98
private theorem within_000_094_agreement : RowAgreement 96 98 := RowAgreement.join atomic_97_agreement atomic_98_agreement
private def within_000_095 : CountSegment 1 10 13120 98 100 90 90 := CountSegment.join atomic_99 atomic_100
private theorem within_000_095_agreement : RowAgreement 98 100 := RowAgreement.join atomic_99_agreement atomic_100_agreement
private def within_000_096 : CountSegment 1 10 13120 96 100 89 90 := CountSegment.join within_000_094 within_000_095
private theorem within_000_096_agreement : RowAgreement 96 100 := RowAgreement.join within_000_094_agreement within_000_095_agreement
private def within_000_097 : CountSegment 1 10 13120 100 102 90 92 := CountSegment.join atomic_101 atomic_102
private theorem within_000_097_agreement : RowAgreement 100 102 := RowAgreement.join atomic_101_agreement atomic_102_agreement
private def within_000_098 : CountSegment 1 10 13120 102 104 92 94 := CountSegment.join atomic_103 atomic_104
private theorem within_000_098_agreement : RowAgreement 102 104 := RowAgreement.join atomic_103_agreement atomic_104_agreement
private def within_000_099 : CountSegment 1 10 13120 100 104 90 94 := CountSegment.join within_000_097 within_000_098
private theorem within_000_099_agreement : RowAgreement 100 104 := RowAgreement.join within_000_097_agreement within_000_098_agreement
private def within_000_100 : CountSegment 1 10 13120 96 104 89 94 := CountSegment.join within_000_096 within_000_099
private theorem within_000_100_agreement : RowAgreement 96 104 := RowAgreement.join within_000_096_agreement within_000_099_agreement
private def within_000_101 : CountSegment 1 10 13120 104 106 94 96 := CountSegment.join atomic_105 atomic_106
private theorem within_000_101_agreement : RowAgreement 104 106 := RowAgreement.join atomic_105_agreement atomic_106_agreement
private def within_000_102 : CountSegment 1 10 13120 106 108 96 97 := CountSegment.join atomic_107 atomic_108
private theorem within_000_102_agreement : RowAgreement 106 108 := RowAgreement.join atomic_107_agreement atomic_108_agreement
private def within_000_103 : CountSegment 1 10 13120 104 108 94 97 := CountSegment.join within_000_101 within_000_102
private theorem within_000_103_agreement : RowAgreement 104 108 := RowAgreement.join within_000_101_agreement within_000_102_agreement
private def within_000_104 : CountSegment 1 10 13120 108 110 97 98 := CountSegment.join atomic_109 atomic_110
private theorem within_000_104_agreement : RowAgreement 108 110 := RowAgreement.join atomic_109_agreement atomic_110_agreement
private def within_000_105 : CountSegment 1 10 13120 110 112 98 99 := CountSegment.join atomic_111 atomic_112
private theorem within_000_105_agreement : RowAgreement 110 112 := RowAgreement.join atomic_111_agreement atomic_112_agreement
private def within_000_106 : CountSegment 1 10 13120 108 112 97 99 := CountSegment.join within_000_104 within_000_105
private theorem within_000_106_agreement : RowAgreement 108 112 := RowAgreement.join within_000_104_agreement within_000_105_agreement
private def within_000_107 : CountSegment 1 10 13120 104 112 94 99 := CountSegment.join within_000_103 within_000_106
private theorem within_000_107_agreement : RowAgreement 104 112 := RowAgreement.join within_000_103_agreement within_000_106_agreement
private def within_000_108 : CountSegment 1 10 13120 96 112 89 99 := CountSegment.join within_000_100 within_000_107
private theorem within_000_108_agreement : RowAgreement 96 112 := RowAgreement.join within_000_100_agreement within_000_107_agreement
private def within_000_109 : CountSegment 1 10 13120 112 114 99 99 := CountSegment.join atomic_113 atomic_114
private theorem within_000_109_agreement : RowAgreement 112 114 := RowAgreement.join atomic_113_agreement atomic_114_agreement
private def within_000_110 : CountSegment 1 10 13120 114 116 99 101 := CountSegment.join atomic_115 atomic_116
private theorem within_000_110_agreement : RowAgreement 114 116 := RowAgreement.join atomic_115_agreement atomic_116_agreement
private def within_000_111 : CountSegment 1 10 13120 112 116 99 101 := CountSegment.join within_000_109 within_000_110
private theorem within_000_111_agreement : RowAgreement 112 116 := RowAgreement.join within_000_109_agreement within_000_110_agreement
private def within_000_112 : CountSegment 1 10 13120 116 118 101 103 := CountSegment.join atomic_117 atomic_118
private theorem within_000_112_agreement : RowAgreement 116 118 := RowAgreement.join atomic_117_agreement atomic_118_agreement
private def within_000_113 : CountSegment 1 10 13120 118 120 103 104 := CountSegment.join atomic_119 atomic_120
private theorem within_000_113_agreement : RowAgreement 118 120 := RowAgreement.join atomic_119_agreement atomic_120_agreement
private def within_000_114 : CountSegment 1 10 13120 116 120 101 104 := CountSegment.join within_000_112 within_000_113
private theorem within_000_114_agreement : RowAgreement 116 120 := RowAgreement.join within_000_112_agreement within_000_113_agreement
private def within_000_115 : CountSegment 1 10 13120 112 120 99 104 := CountSegment.join within_000_111 within_000_114
private theorem within_000_115_agreement : RowAgreement 112 120 := RowAgreement.join within_000_111_agreement within_000_114_agreement
private def within_000_116 : CountSegment 1 10 13120 120 122 104 104 := CountSegment.join atomic_121 atomic_122
private theorem within_000_116_agreement : RowAgreement 120 122 := RowAgreement.join atomic_121_agreement atomic_122_agreement
private def within_000_117 : CountSegment 1 10 13120 122 124 104 106 := CountSegment.join atomic_123 atomic_124
private theorem within_000_117_agreement : RowAgreement 122 124 := RowAgreement.join atomic_123_agreement atomic_124_agreement
private def within_000_118 : CountSegment 1 10 13120 120 124 104 106 := CountSegment.join within_000_116 within_000_117
private theorem within_000_118_agreement : RowAgreement 120 124 := RowAgreement.join within_000_116_agreement within_000_117_agreement
private def within_000_119 : CountSegment 1 10 13120 124 126 106 107 := CountSegment.join atomic_125 atomic_126
private theorem within_000_119_agreement : RowAgreement 124 126 := RowAgreement.join atomic_125_agreement atomic_126_agreement
private def within_000_120 : CountSegment 1 10 13120 126 128 107 108 := CountSegment.join atomic_127 atomic_128
private theorem within_000_120_agreement : RowAgreement 126 128 := RowAgreement.join atomic_127_agreement atomic_128_agreement
private def within_000_121 : CountSegment 1 10 13120 124 128 106 108 := CountSegment.join within_000_119 within_000_120
private theorem within_000_121_agreement : RowAgreement 124 128 := RowAgreement.join within_000_119_agreement within_000_120_agreement
private def within_000_122 : CountSegment 1 10 13120 120 128 104 108 := CountSegment.join within_000_118 within_000_121
private theorem within_000_122_agreement : RowAgreement 120 128 := RowAgreement.join within_000_118_agreement within_000_121_agreement
private def within_000_123 : CountSegment 1 10 13120 112 128 99 108 := CountSegment.join within_000_115 within_000_122
private theorem within_000_123_agreement : RowAgreement 112 128 := RowAgreement.join within_000_115_agreement within_000_122_agreement
private def within_000_124 : CountSegment 1 10 13120 96 128 89 108 := CountSegment.join within_000_108 within_000_123
private theorem within_000_124_agreement : RowAgreement 96 128 := RowAgreement.join within_000_108_agreement within_000_123_agreement
private def within_000_125 : CountSegment 1 10 13120 64 128 61 108 := CountSegment.join within_000_093 within_000_124
private theorem within_000_125_agreement : RowAgreement 64 128 := RowAgreement.join within_000_093_agreement within_000_124_agreement
private def within_000_126 : CountSegment 1 10 13120 0 128 0 108 := CountSegment.join within_000_062 within_000_125
private theorem within_000_126_agreement : RowAgreement 0 128 := RowAgreement.join within_000_062_agreement within_000_125_agreement
def segment_000 : CountSegment 1 10 13120 0 128 0 108 := within_000_126
theorem agreement_000 : RowAgreement 0 128 := within_000_126_agreement
end ProjectEulerSolutions.P58Numerical
