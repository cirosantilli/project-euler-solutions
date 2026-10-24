import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3457 : RowCertificate 3457 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3457 1) := by change Nat.Prime 47810311; exact ProjectEulerSolutions.P58PrimeNodes.p_47810311.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3457 2) := by change ¬ Nat.Prime 47803397; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1648393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3457 3) := by change ¬ Nat.Prime 47796483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15932161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3457 : row 3457 = 1 := (row_3457.checked).trans rfl
theorem model_count_3457 : modelRow 3457 = 1 := by decide
theorem modelrow_agrees_3457 : modelRow 3457 = row 3457 := model_count_3457.trans row_count_3457.symm
def row_3458 : RowCertificate 3458 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3458 1) := by change ¬ Nat.Prime 47837973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15945991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3458 2) := by change ¬ Nat.Prime 47831057; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 42821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3458 3) := by change Nat.Prime 47824141; exact ProjectEulerSolutions.P58PrimeNodes.p_47824141.prime
    exact decide_eq_true prime
theorem row_count_3458 : row 3458 = 1 := (row_3458.checked).trans rfl
theorem model_count_3458 : modelRow 3458 = 1 := by decide
theorem modelrow_agrees_3458 : modelRow 3458 = row 3458 := model_count_3458.trans row_count_3458.symm
def row_3459 : RowCertificate 3459 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3459 1) := by change ¬ Nat.Prime 47865643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6837949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3459 2) := by change ¬ Nat.Prime 47858725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9571745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3459 3) := by change Nat.Prime 47851807; exact ProjectEulerSolutions.P58PrimeNodes.p_47851807.prime
    exact decide_eq_true prime
theorem row_count_3459 : row 3459 = 1 := (row_3459.checked).trans rfl
theorem model_count_3459 : modelRow 3459 = 1 := by decide
theorem modelrow_agrees_3459 : modelRow 3459 = row 3459 := model_count_3459.trans row_count_3459.symm
def row_3460 : RowCertificate 3460 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3460 1) := by change ¬ Nat.Prime 47893321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6841903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3460 2) := by change ¬ Nat.Prime 47886401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1167961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3460 3) := by change ¬ Nat.Prime 47879481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15959827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3460 : row 3460 = 0 := (row_3460.checked).trans rfl
theorem model_count_3460 : modelRow 3460 = 0 := by decide
theorem modelrow_agrees_3460 : modelRow 3460 = row 3460 := model_count_3460.trans row_count_3460.symm
def row_3461 : RowCertificate 3461 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3461 1) := by change ¬ Nat.Prime 47921007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15973669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3461 2) := by change ¬ Nat.Prime 47914085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9582817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3461 3) := by change Nat.Prime 47907163; exact ProjectEulerSolutions.P58PrimeNodes.p_47907163.prime
    exact decide_eq_true prime
theorem row_count_3461 : row 3461 = 1 := (row_3461.checked).trans rfl
theorem model_count_3461 : modelRow 3461 = 1 := by decide
theorem modelrow_agrees_3461 : modelRow 3461 = row 3461 := model_count_3461.trans row_count_3461.symm
def row_3462 : RowCertificate 3462 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3462 1) := by change ¬ Nat.Prime 47948701; exact PrattCertificates.composite_of_product (factor := 1669) (cofactor := 28729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3462 2) := by change ¬ Nat.Prime 47941777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3687829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3462 3) := by change ¬ Nat.Prime 47934853; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2522887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3462 : row 3462 = 0 := (row_3462.checked).trans rfl
theorem model_count_3462 : modelRow 3462 = 0 := by decide
theorem modelrow_agrees_3462 : modelRow 3462 = row 3462 := model_count_3462.trans row_count_3462.symm
def row_3463 : RowCertificate 3463 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3463 1) := by change ¬ Nat.Prime 47976403; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 657211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3463 2) := by change Nat.Prime 47969477; exact ProjectEulerSolutions.P58PrimeNodes.p_47969477.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3463 3) := by change ¬ Nat.Prime 47962551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15987517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3463 : row 3463 = 1 := (row_3463.checked).trans rfl
theorem model_count_3463 : modelRow 3463 = 1 := by decide
theorem modelrow_agrees_3463 : modelRow 3463 = row 3463 := model_count_3463.trans row_count_3463.symm
def row_3464 : RowCertificate 3464 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3464 1) := by change ¬ Nat.Prime 48004113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16001371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3464 2) := by change ¬ Nat.Prime 47997185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9599437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3464 3) := by change ¬ Nat.Prime 47990257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6855751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3464 : row 3464 = 0 := (row_3464.checked).trans rfl
theorem model_count_3464 : modelRow 3464 = 0 := by decide
theorem modelrow_agrees_3464 : modelRow 3464 = row 3464 := model_count_3464.trans row_count_3464.symm
def row_3465 : RowCertificate 3465 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3465 1) := by change ¬ Nat.Prime 48031831; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 716893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3465 2) := by change Nat.Prime 48024901; exact ProjectEulerSolutions.P58PrimeNodes.p_48024901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3465 3) := by change ¬ Nat.Prime 48017971; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1297783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3465 : row 3465 = 1 := (row_3465.checked).trans rfl
theorem model_count_3465 : modelRow 3465 = 1 := by decide
theorem modelrow_agrees_3465 : modelRow 3465 = row 3465 := model_count_3465.trans row_count_3465.symm
def row_3466 : RowCertificate 3466 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3466 1) := by change ¬ Nat.Prime 48059557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6865651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3466 2) := by change ¬ Nat.Prime 48052625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9610525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3466 3) := by change ¬ Nat.Prime 48045693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16015231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3466 : row 3466 = 0 := (row_3466.checked).trans rfl
theorem model_count_3466 : modelRow 3466 = 0 := by decide
theorem modelrow_agrees_3466 : modelRow 3466 = row 3466 := model_count_3466.trans row_count_3466.symm
def row_3467 : RowCertificate 3467 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3467 1) := by change ¬ Nat.Prime 48087291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16029097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3467 2) := by change ¬ Nat.Prime 48080357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3698489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3467 3) := by change ¬ Nat.Prime 48073423; exact PrattCertificates.composite_of_product (factor := 1723) (cofactor := 27901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3467 : row 3467 = 0 := (row_3467.checked).trans rfl
theorem model_count_3467 : modelRow 3467 = 0 := by decide
theorem modelrow_agrees_3467 : modelRow 3467 = row 3467 := model_count_3467.trans row_count_3467.symm
def row_3468 : RowCertificate 3468 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3468 1) := by change ¬ Nat.Prime 48115033; exact PrattCertificates.composite_of_product (factor := 3001) (cofactor := 16033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3468 2) := by change ¬ Nat.Prime 48108097; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 306421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3468 3) := by change ¬ Nat.Prime 48101161; exact PrattCertificates.composite_of_product (factor := 4273) (cofactor := 11257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3468 : row 3468 = 0 := (row_3468.checked).trans rfl
theorem model_count_3468 : modelRow 3468 = 0 := by decide
theorem modelrow_agrees_3468 : modelRow 3468 = row 3468 := model_count_3468.trans row_count_3468.symm
def row_3469 : RowCertificate 3469 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3469 1) := by change ¬ Nat.Prime 48142783; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3703291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3469 2) := by change ¬ Nat.Prime 48135845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9627169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3469 3) := by change ¬ Nat.Prime 48128907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16042969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3469 : row 3469 = 0 := (row_3469.checked).trans rfl
theorem model_count_3469 : modelRow 3469 = 0 := by decide
theorem modelrow_agrees_3469 : modelRow 3469 = row 3469 := model_count_3469.trans row_count_3469.symm
def row_3470 : RowCertificate 3470 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3470 1) := by change ¬ Nat.Prime 48170541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16056847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3470 2) := by change ¬ Nat.Prime 48163601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2833153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3470 3) := by change ¬ Nat.Prime 48156661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6879523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3470 : row 3470 = 0 := (row_3470.checked).trans rfl
theorem model_count_3470 : modelRow 3470 = 0 := by decide
theorem modelrow_agrees_3470 : modelRow 3470 = row 3470 := model_count_3470.trans row_count_3470.symm
def row_3471 : RowCertificate 3471 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3471 1) := by change ¬ Nat.Prime 48198307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2536753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3471 2) := by change ¬ Nat.Prime 48191365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9638273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3471 3) := by change ¬ Nat.Prime 48184423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6883489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3471 : row 3471 = 0 := (row_3471.checked).trans rfl
theorem model_count_3471 : modelRow 3471 = 0 := by decide
theorem modelrow_agrees_3471 : modelRow 3471 = row 3471 := model_count_3471.trans row_count_3471.symm
def row_3472 : RowCertificate 3472 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3472 1) := by change Nat.Prime 48226081; exact ProjectEulerSolutions.P58PrimeNodes.p_48226081.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3472 2) := by change Nat.Prime 48219137; exact ProjectEulerSolutions.P58PrimeNodes.p_48219137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3472 3) := by change ¬ Nat.Prime 48212193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16070731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3472 : row 3472 = 2 := (row_3472.checked).trans rfl
theorem model_count_3472 : modelRow 3472 = 2 := by decide
theorem modelrow_agrees_3472 : modelRow 3472 = row 3472 := model_count_3472.trans row_count_3472.symm
def row_3473 : RowCertificate 3473 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3473 1) := by change ¬ Nat.Prime 48253863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16084621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3473 2) := by change ¬ Nat.Prime 48246917; exact PrattCertificates.composite_of_product (factor := 5801) (cofactor := 8317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3473 3) := by change ¬ Nat.Prime 48239971; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3710767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3473 : row 3473 = 0 := (row_3473.checked).trans rfl
theorem model_count_3473 : modelRow 3473 = 0 := by decide
theorem modelrow_agrees_3473 : modelRow 3473 = row 3473 := model_count_3473.trans row_count_3473.symm
def row_3474 : RowCertificate 3474 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3474 1) := by change ¬ Nat.Prime 48281653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6897379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3474 2) := by change ¬ Nat.Prime 48274705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9654941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3474 3) := by change ¬ Nat.Prime 48267757; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 468619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3474 : row 3474 = 0 := (row_3474.checked).trans rfl
theorem model_count_3474 : modelRow 3474 = 0 := by decide
theorem modelrow_agrees_3474 : modelRow 3474 = row 3474 := model_count_3474.trans row_count_3474.symm
def row_3475 : RowCertificate 3475 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3475 1) := by change ¬ Nat.Prime 48309451; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 296377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3475 2) := by change ¬ Nat.Prime 48302501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3715577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3475 3) := by change ¬ Nat.Prime 48295551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16098517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3475 : row 3475 = 0 := (row_3475.checked).trans rfl
theorem model_count_3475 : modelRow 3475 = 0 := by decide
theorem modelrow_agrees_3475 : modelRow 3475 = row 3475 := model_count_3475.trans row_count_3475.symm
def row_3476 : RowCertificate 3476 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3476 1) := by change ¬ Nat.Prime 48337257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16112419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3476 2) := by change ¬ Nat.Prime 48330305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9666061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3476 3) := by change ¬ Nat.Prime 48323353; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3717181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3476 : row 3476 = 0 := (row_3476.checked).trans rfl
theorem model_count_3476 : modelRow 3476 = 0 := by decide
theorem modelrow_agrees_3476 : modelRow 3476 = row 3476 := model_count_3476.trans row_count_3476.symm
def row_3477 : RowCertificate 3477 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3477 1) := by change Nat.Prime 48365071; exact ProjectEulerSolutions.P58PrimeNodes.p_48365071.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3477 2) := by change ¬ Nat.Prime 48358117; exact PrattCertificates.composite_of_product (factor := 5281) (cofactor := 9157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3477 3) := by change ¬ Nat.Prime 48351163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6907309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3477 : row 3477 = 1 := (row_3477.checked).trans rfl
theorem model_count_3477 : modelRow 3477 = 1 := by decide
theorem modelrow_agrees_3477 : modelRow 3477 = row 3477 := model_count_3477.trans row_count_3477.symm
def row_3478 : RowCertificate 3478 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3478 1) := by change Nat.Prime 48392893; exact ProjectEulerSolutions.P58PrimeNodes.p_48392893.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3478 2) := by change Nat.Prime 48385937; exact ProjectEulerSolutions.P58PrimeNodes.p_48385937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3478 3) := by change ¬ Nat.Prime 48378981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16126327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3478 : row 3478 = 2 := (row_3478.checked).trans rfl
theorem model_count_3478 : modelRow 3478 = 2 := by decide
theorem modelrow_agrees_3478 : modelRow 3478 = row 3478 := model_count_3478.trans row_count_3478.symm
def row_3479 : RowCertificate 3479 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3479 1) := by change ¬ Nat.Prime 48420723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16140241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3479 2) := by change ¬ Nat.Prime 48413765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9682753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3479 3) := by change ¬ Nat.Prime 48406807; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 469969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3479 : row 3479 = 0 := (row_3479.checked).trans rfl
theorem model_count_3479 : modelRow 3479 = 0 := by decide
theorem modelrow_agrees_3479 : modelRow 3479 = row 3479 := model_count_3479.trans row_count_3479.symm
def row_3480 : RowCertificate 3480 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3480 1) := by change ¬ Nat.Prime 48448561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6921223) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3480 2) := by change ¬ Nat.Prime 48441601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3726277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3480 3) := by change ¬ Nat.Prime 48434641; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1126387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3480 : row 3480 = 0 := (row_3480.checked).trans rfl
theorem model_count_3480 : modelRow 3480 = 0 := by decide
theorem modelrow_agrees_3480 : modelRow 3480 = row 3480 := model_count_3480.trans row_count_3480.symm
def row_3481 : RowCertificate 3481 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3481 1) := by change ¬ Nat.Prime 48476407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6925201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3481 2) := by change ¬ Nat.Prime 48469445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9693889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3481 3) := by change ¬ Nat.Prime 48462483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16154161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3481 : row 3481 = 0 := (row_3481.checked).trans rfl
theorem model_count_3481 : modelRow 3481 = 0 := by decide
theorem modelrow_agrees_3481 : modelRow 3481 = row 3481 := model_count_3481.trans row_count_3481.symm
def row_3482 : RowCertificate 3482 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3482 1) := by change ¬ Nat.Prime 48504261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16168087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3482 2) := by change ¬ Nat.Prime 48497297; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 106121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3482 3) := by change Nat.Prime 48490333; exact ProjectEulerSolutions.P58PrimeNodes.p_48490333.prime
    exact decide_eq_true prime
theorem row_count_3482 : row 3482 = 1 := (row_3482.checked).trans rfl
theorem model_count_3482 : modelRow 3482 = 1 := by decide
theorem modelrow_agrees_3482 : modelRow 3482 = row 3482 := model_count_3482.trans row_count_3482.symm
def row_3483 : RowCertificate 3483 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3483 1) := by change ¬ Nat.Prime 48532123; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1311679) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3483 2) := by change ¬ Nat.Prime 48525157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2854421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3483 3) := by change ¬ Nat.Prime 48518191; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2553589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3483 : row 3483 = 0 := (row_3483.checked).trans rfl
theorem model_count_3483 : modelRow 3483 = 0 := by decide
theorem modelrow_agrees_3483 : modelRow 3483 = row 3483 := model_count_3483.trans row_count_3483.symm
def row_3484 : RowCertificate 3484 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3484 1) := by change Nat.Prime 48559993; exact ProjectEulerSolutions.P58PrimeNodes.p_48559993.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3484 2) := by change ¬ Nat.Prime 48553025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9710605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3484 3) := by change ¬ Nat.Prime 48546057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16182019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3484 : row 3484 = 1 := (row_3484.checked).trans rfl
theorem model_count_3484 : modelRow 3484 = 1 := by decide
theorem modelrow_agrees_3484 : modelRow 3484 = row 3484 := model_count_3484.trans row_count_3484.symm
def row_3485 : RowCertificate 3485 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3485 1) := by change ¬ Nat.Prime 48587871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16195957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3485 2) := by change Nat.Prime 48580901; exact ProjectEulerSolutions.P58PrimeNodes.p_48580901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3485 3) := by change ¬ Nat.Prime 48573931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6939133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3485 : row 3485 = 1 := (row_3485.checked).trans rfl
theorem model_count_3485 : modelRow 3485 = 1 := by decide
theorem modelrow_agrees_3485 : modelRow 3485 = row 3485 := model_count_3485.trans row_count_3485.symm
def row_3486 : RowCertificate 3486 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3486 1) := by change ¬ Nat.Prime 48615757; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1130599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3486 2) := by change ¬ Nat.Prime 48608785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9721757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3486 3) := by change ¬ Nat.Prime 48601813; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3738601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3486 : row 3486 = 0 := (row_3486.checked).trans rfl
theorem model_count_3486 : modelRow 3486 = 0 := by decide
theorem modelrow_agrees_3486 : modelRow 3486 = row 3486 := model_count_3486.trans row_count_3486.symm
def row_3487 : RowCertificate 3487 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3487 1) := by change ¬ Nat.Prime 48643651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6949093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3487 2) := by change ¬ Nat.Prime 48636677; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2860981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3487 3) := by change ¬ Nat.Prime 48629703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16209901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3487 : row 3487 = 0 := (row_3487.checked).trans rfl
theorem model_count_3487 : modelRow 3487 = 0 := by decide
theorem modelrow_agrees_3487 : modelRow 3487 = row 3487 := model_count_3487.trans row_count_3487.symm
def row_3488 : RowCertificate 3488 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3488 1) := by change ¬ Nat.Prime 48671553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16223851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3488 2) := by change ¬ Nat.Prime 48664577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3743429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3488 3) := by change ¬ Nat.Prime 48657601; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 615919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3488 : row 3488 = 0 := (row_3488.checked).trans rfl
theorem model_count_3488 : modelRow 3488 = 0 := by decide
theorem modelrow_agrees_3488 : modelRow 3488 = row 3488 := model_count_3488.trans row_count_3488.symm
def row_3489 : RowCertificate 3489 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3489 1) := by change ¬ Nat.Prime 48699463; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 322513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3489 2) := by change ¬ Nat.Prime 48692485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9738497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3489 3) := by change ¬ Nat.Prime 48685507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3745039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3489 : row 3489 = 0 := (row_3489.checked).trans rfl
theorem model_count_3489 : modelRow 3489 = 0 := by decide
theorem modelrow_agrees_3489 : modelRow 3489 = row 3489 := model_count_3489.trans row_count_3489.symm
def row_3490 : RowCertificate 3490 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3490 1) := by change ¬ Nat.Prime 48727381; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2564599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3490 2) := by change Nat.Prime 48720401; exact ProjectEulerSolutions.P58PrimeNodes.p_48720401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3490 3) := by change ¬ Nat.Prime 48713421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16237807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3490 : row 3490 = 1 := (row_3490.checked).trans rfl
theorem model_count_3490 : modelRow 3490 = 1 := by decide
theorem modelrow_agrees_3490 : modelRow 3490 = row 3490 := model_count_3490.trans row_count_3490.symm
def row_3491 : RowCertificate 3491 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3491 1) := by change ¬ Nat.Prime 48755307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16251769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3491 2) := by change ¬ Nat.Prime 48748325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9749665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3491 3) := by change ¬ Nat.Prime 48741343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6963049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3491 : row 3491 = 0 := (row_3491.checked).trans rfl
theorem model_count_3491 : modelRow 3491 = 0 := by decide
theorem modelrow_agrees_3491 : modelRow 3491 = row 3491 := model_count_3491.trans row_count_3491.symm
def row_3492 : RowCertificate 3492 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3492 1) := by change ¬ Nat.Prime 48783241; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3752557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3492 2) := by change Nat.Prime 48776257; exact ProjectEulerSolutions.P58PrimeNodes.p_48776257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3492 3) := by change ¬ Nat.Prime 48769273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6967039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3492 : row 3492 = 1 := (row_3492.checked).trans rfl
theorem model_count_3492 : modelRow 3492 = 1 := by decide
theorem modelrow_agrees_3492 : modelRow 3492 = row 3492 := model_count_3492.trans row_count_3492.symm
def row_3493 : RowCertificate 3493 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3493 1) := by change Nat.Prime 48811183; exact ProjectEulerSolutions.P58PrimeNodes.p_48811183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3493 2) := by change ¬ Nat.Prime 48804197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3754169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3493 3) := by change ¬ Nat.Prime 48797211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16265737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3493 : row 3493 = 1 := (row_3493.checked).trans rfl
theorem model_count_3493 : modelRow 3493 = 1 := by decide
theorem modelrow_agrees_3493 : modelRow 3493 = row 3493 := model_count_3493.trans row_count_3493.symm
def row_3494 : RowCertificate 3494 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3494 1) := by change ¬ Nat.Prime 48839133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16279711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3494 2) := by change ¬ Nat.Prime 48832145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9766429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3494 3) := by change Nat.Prime 48825157; exact ProjectEulerSolutions.P58PrimeNodes.p_48825157.prime
    exact decide_eq_true prime
theorem row_count_3494 : row 3494 = 1 := (row_3494.checked).trans rfl
theorem model_count_3494 : modelRow 3494 = 1 := by decide
theorem modelrow_agrees_3494 : modelRow 3494 = row 3494 := model_count_3494.trans row_count_3494.symm
def row_3495 : RowCertificate 3495 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3495 1) := by change ¬ Nat.Prime 48867091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6981013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3495 2) := by change Nat.Prime 48860101; exact ProjectEulerSolutions.P58PrimeNodes.p_48860101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3495 3) := by change ¬ Nat.Prime 48853111; exact PrattCertificates.composite_of_product (factor := 1861) (cofactor := 26251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3495 : row 3495 = 1 := (row_3495.checked).trans rfl
theorem model_count_3495 : modelRow 3495 = 1 := by decide
theorem modelrow_agrees_3495 : modelRow 3495 = row 3495 := model_count_3495.trans row_count_3495.symm
def row_3496 : RowCertificate 3496 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3496 1) := by change ¬ Nat.Prime 48895057; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 351763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3496 2) := by change ¬ Nat.Prime 48888065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9777613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3496 3) := by change ¬ Nat.Prime 48881073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16293691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3496 : row 3496 = 0 := (row_3496.checked).trans rfl
theorem model_count_3496 : modelRow 3496 = 0 := by decide
theorem modelrow_agrees_3496 : modelRow 3496 = row 3496 := model_count_3496.trans row_count_3496.symm
def row_3497 : RowCertificate 3497 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3497 1) := by change ¬ Nat.Prime 48923031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16307677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3497 2) := by change ¬ Nat.Prime 48916037; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 68993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3497 3) := by change Nat.Prime 48909043; exact ProjectEulerSolutions.P58PrimeNodes.p_48909043.prime
    exact decide_eq_true prime
theorem row_count_3497 : row 3497 = 1 := (row_3497.checked).trans rfl
theorem model_count_3497 : modelRow 3497 = 1 := by decide
theorem modelrow_agrees_3497 : modelRow 3497 = row 3497 := model_count_3497.trans row_count_3497.symm
def row_3498 : RowCertificate 3498 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3498 1) := by change ¬ Nat.Prime 48951013; exact PrattCertificates.composite_of_product (factor := 6373) (cofactor := 7681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3498 2) := by change Nat.Prime 48944017; exact ProjectEulerSolutions.P58PrimeNodes.p_48944017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3498 3) := by change ¬ Nat.Prime 48937021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6991003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3498 : row 3498 = 1 := (row_3498.checked).trans rfl
theorem model_count_3498 : modelRow 3498 = 1 := by decide
theorem modelrow_agrees_3498 : modelRow 3498 = row 3498 := model_count_3498.trans row_count_3498.symm
def row_3499 : RowCertificate 3499 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3499 1) := by change ¬ Nat.Prime 48979003; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 131311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3499 2) := by change ¬ Nat.Prime 48972005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9794401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3499 3) := by change ¬ Nat.Prime 48965007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16321669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3499 : row 3499 = 0 := (row_3499.checked).trans rfl
theorem model_count_3499 : modelRow 3499 = 0 := by decide
theorem modelrow_agrees_3499 : modelRow 3499 = row 3499 := model_count_3499.trans row_count_3499.symm
def row_3500 : RowCertificate 3500 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3500 1) := by change ¬ Nat.Prime 49007001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16335667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3500 2) := by change ¬ Nat.Prime 49000001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2882353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3500 3) := by change ¬ Nat.Prime 48993001; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2578579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3500 : row 3500 = 0 := (row_3500.checked).trans rfl
theorem model_count_3500 : modelRow 3500 = 0 := by decide
theorem modelrow_agrees_3500 : modelRow 3500 = row 3500 := model_count_3500.trans row_count_3500.symm
def row_3501 : RowCertificate 3501 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3501 1) := by change ¬ Nat.Prime 49035007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7005001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3501 2) := by change ¬ Nat.Prime 49028005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9805601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3501 3) := by change ¬ Nat.Prime 49021003; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 803623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3501 : row 3501 = 0 := (row_3501.checked).trans rfl
theorem model_count_3501 : modelRow 3501 = 0 := by decide
theorem modelrow_agrees_3501 : modelRow 3501 = row 3501 := model_count_3501.trans row_count_3501.symm
def row_3502 : RowCertificate 3502 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3502 1) := by change ¬ Nat.Prime 49063021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7009003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3502 2) := by change ¬ Nat.Prime 49056017; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 804197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3502 3) := by change ¬ Nat.Prime 49049013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16349671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3502 : row 3502 = 0 := (row_3502.checked).trans rfl
theorem model_count_3502 : modelRow 3502 = 0 := by decide
theorem modelrow_agrees_3502 : modelRow 3502 = row 3502 := model_count_3502.trans row_count_3502.symm
def row_3503 : RowCertificate 3503 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3503 1) := by change ¬ Nat.Prime 49091043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16363681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3503 2) := by change ¬ Nat.Prime 49084037; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1692553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3503 3) := by change ¬ Nat.Prime 49077031; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 732493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3503 : row 3503 = 0 := (row_3503.checked).trans rfl
theorem model_count_3503 : modelRow 3503 = 0 := by decide
theorem modelrow_agrees_3503 : modelRow 3503 = row 3503 := model_count_3503.trans row_count_3503.symm
def row_3504 : RowCertificate 3504 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3504 1) := by change ¬ Nat.Prime 49119073; exact PrattCertificates.composite_of_product (factor := 1531) (cofactor := 32083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3504 2) := by change ¬ Nat.Prime 49112065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9822413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3504 3) := by change ¬ Nat.Prime 49105057; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 621583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3504 : row 3504 = 0 := (row_3504.checked).trans rfl
theorem model_count_3504 : modelRow 3504 = 0 := by decide
theorem modelrow_agrees_3504 : modelRow 3504 = row 3504 := model_count_3504.trans row_count_3504.symm
def row_3505 : RowCertificate 3505 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3505 1) := by change ¬ Nat.Prime 49147111; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3780547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3505 2) := by change Nat.Prime 49140101; exact ProjectEulerSolutions.P58PrimeNodes.p_49140101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3505 3) := by change ¬ Nat.Prime 49133091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16377697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3505 : row 3505 = 1 := (row_3505.checked).trans rfl
theorem model_count_3505 : modelRow 3505 = 1 := by decide
theorem modelrow_agrees_3505 : modelRow 3505 = row 3505 := model_count_3505.trans row_count_3505.symm
def row_3506 : RowCertificate 3506 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3506 1) := by change ¬ Nat.Prime 49175157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16391719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3506 2) := by change ¬ Nat.Prime 49168145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9833629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3506 3) := by change ¬ Nat.Prime 49161133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7023019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3506 : row 3506 = 0 := (row_3506.checked).trans rfl
theorem model_count_3506 : modelRow 3506 = 0 := by decide
theorem modelrow_agrees_3506 : modelRow 3506 = row 3506 := model_count_3506.trans row_count_3506.symm
def row_3507 : RowCertificate 3507 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3507 1) := by change Nat.Prime 49203211; exact ProjectEulerSolutions.P58PrimeNodes.p_49203211.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3507 2) := by change Nat.Prime 49196197; exact ProjectEulerSolutions.P58PrimeNodes.p_49196197.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3507 3) := by change Nat.Prime 49189183; exact ProjectEulerSolutions.P58PrimeNodes.p_49189183.prime
    exact decide_eq_true prime
theorem row_count_3507 : row 3507 = 3 := (row_3507.checked).trans rfl
theorem model_count_3507 : modelRow 3507 = 3 := by decide
theorem modelrow_agrees_3507 : modelRow 3507 = row 3507 := model_count_3507.trans row_count_3507.symm
def row_3508 : RowCertificate 3508 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3508 1) := by change ¬ Nat.Prime 49231273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7033039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3508 2) := by change Nat.Prime 49224257; exact ProjectEulerSolutions.P58PrimeNodes.p_49224257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3508 3) := by change ¬ Nat.Prime 49217241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16405747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3508 : row 3508 = 1 := (row_3508.checked).trans rfl
theorem model_count_3508 : modelRow 3508 = 1 := by decide
theorem modelrow_agrees_3508 : modelRow 3508 = row 3508 := model_count_3508.trans row_count_3508.symm
def row_3509 : RowCertificate 3509 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3509 1) := by change ¬ Nat.Prime 49259343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16419781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3509 2) := by change ¬ Nat.Prime 49252325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9850465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3509 3) := by change ¬ Nat.Prime 49245307; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 181717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3509 : row 3509 = 0 := (row_3509.checked).trans rfl
theorem model_count_3509 : modelRow 3509 = 0 := by decide
theorem modelrow_agrees_3509 : modelRow 3509 = row 3509 := model_count_3509.trans row_count_3509.symm
def row_3510 : RowCertificate 3510 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3510 1) := by change ¬ Nat.Prime 49287421; exact PrattCertificates.composite_of_product (factor := 6529) (cofactor := 7549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3510 2) := by change ¬ Nat.Prime 49280401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1201961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3510 3) := by change ¬ Nat.Prime 49273381; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1331713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3510 : row 3510 = 0 := (row_3510.checked).trans rfl
theorem model_count_3510 : modelRow 3510 = 0 := by decide
theorem modelrow_agrees_3510 : modelRow 3510 = row 3510 := model_count_3510.trans row_count_3510.symm
def row_3511 : RowCertificate 3511 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3511 1) := by change ¬ Nat.Prime 49315507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2595553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3511 2) := by change ¬ Nat.Prime 49308485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9861697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3511 3) := by change ¬ Nat.Prime 49301463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16433821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3511 : row 3511 = 0 := (row_3511.checked).trans rfl
theorem model_count_3511 : modelRow 3511 = 0 := by decide
theorem modelrow_agrees_3511 : modelRow 3511 = row 3511 := model_count_3511.trans row_count_3511.symm
def row_3512 : RowCertificate 3512 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3512 1) := by change ¬ Nat.Prime 49343601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16447867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3512 2) := by change ¬ Nat.Prime 49336577; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1333421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3512 3) := by change ¬ Nat.Prime 49329553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7047079) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3512 : row 3512 = 0 := (row_3512.checked).trans rfl
theorem model_count_3512 : modelRow 3512 = 0 := by decide
theorem modelrow_agrees_3512 : modelRow 3512 = row 3512 := model_count_3512.trans row_count_3512.symm
def row_3513 : RowCertificate 3513 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3513 1) := by change Nat.Prime 49371703; exact ProjectEulerSolutions.P58PrimeNodes.p_49371703.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3513 2) := by change ¬ Nat.Prime 49364677; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 931409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3513 3) := by change ¬ Nat.Prime 49357651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7051093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3513 : row 3513 = 1 := (row_3513.checked).trans rfl
theorem model_count_3513 : modelRow 3513 = 1 := by decide
theorem modelrow_agrees_3513 : modelRow 3513 = row 3513 := model_count_3513.trans row_count_3513.symm
def row_3514 : RowCertificate 3514 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3514 1) := by change ¬ Nat.Prime 49399813; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 809833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3514 2) := by change ¬ Nat.Prime 49392785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9878557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3514 3) := by change ¬ Nat.Prime 49385757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16461919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3514 : row 3514 = 0 := (row_3514.checked).trans rfl
theorem model_count_3514 : modelRow 3514 = 0 := by decide
theorem modelrow_agrees_3514 : modelRow 3514 = row 3514 := model_count_3514.trans row_count_3514.symm
def row_3515 : RowCertificate 3515 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3515 1) := by change ¬ Nat.Prime 49427931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16475977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3515 2) := by change ¬ Nat.Prime 49420901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1704169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3515 3) := by change ¬ Nat.Prime 49413871; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3801067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3515 : row 3515 = 0 := (row_3515.checked).trans rfl
theorem model_count_3515 : modelRow 3515 = 0 := by decide
theorem modelrow_agrees_3515 : modelRow 3515 = row 3515 := model_count_3515.trans row_count_3515.symm
def row_3516 : RowCertificate 3516 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3516 1) := by change ¬ Nat.Prime 49456057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7065151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3516 2) := by change ¬ Nat.Prime 49449025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9889805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3516 3) := by change ¬ Nat.Prime 49441993; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1594903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3516 : row 3516 = 0 := (row_3516.checked).trans rfl
theorem model_count_3516 : modelRow 3516 = 0 := by decide
theorem modelrow_agrees_3516 : modelRow 3516 = row 3516 := model_count_3516.trans row_count_3516.symm
def row_3517 : RowCertificate 3517 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3517 1) := by change Nat.Prime 49484191; exact ProjectEulerSolutions.P58PrimeNodes.p_49484191.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3517 2) := by change ¬ Nat.Prime 49477157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2910421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3517 3) := by change ¬ Nat.Prime 49470123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16490041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3517 : row 3517 = 1 := (row_3517.checked).trans rfl
theorem model_count_3517 : modelRow 3517 = 1 := by decide
theorem modelrow_agrees_3517 : modelRow 3517 = row 3517 := model_count_3517.trans row_count_3517.symm
def row_3518 : RowCertificate 3518 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3518 1) := by change ¬ Nat.Prime 49512333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16504111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3518 2) := by change ¬ Nat.Prime 49505297; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1337981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3518 3) := by change ¬ Nat.Prime 49498261; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 47917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3518 : row 3518 = 0 := (row_3518.checked).trans rfl
theorem model_count_3518 : modelRow 3518 = 0 := by decide
theorem modelrow_agrees_3518 : modelRow 3518 = row 3518 := model_count_3518.trans row_count_3518.symm
def row_3519 : RowCertificate 3519 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3519 1) := by change ¬ Nat.Prime 49540483; exact PrattCertificates.composite_of_product (factor := 3571) (cofactor := 13873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3519 2) := by change ¬ Nat.Prime 49533445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9906689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3519 3) := by change ¬ Nat.Prime 49526407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7075201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3519 : row 3519 = 0 := (row_3519.checked).trans rfl
theorem model_count_3519 : modelRow 3519 = 0 := by decide
theorem modelrow_agrees_3519 : modelRow 3519 = row 3519 := model_count_3519.trans row_count_3519.symm
def row_3520 : RowCertificate 3520 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3520 1) := by change ¬ Nat.Prime 49568641; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1339693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3520 2) := by change ¬ Nat.Prime 49561601; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 273821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3520 3) := by change ¬ Nat.Prime 49554561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16518187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3520 : row 3520 = 0 := (row_3520.checked).trans rfl
theorem model_count_3520 : modelRow 3520 = 0 := by decide
theorem modelrow_agrees_3520 : modelRow 3520 = row 3520 := model_count_3520.trans row_count_3520.symm
def row_3521 : RowCertificate 3521 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3521 1) := by change ¬ Nat.Prime 49596807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16532269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3521 2) := by change ¬ Nat.Prime 49589765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9917953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3521 3) := by change ¬ Nat.Prime 49582723; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2609617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3521 : row 3521 = 0 := (row_3521.checked).trans rfl
theorem model_count_3521 : modelRow 3521 = 0 := by decide
theorem modelrow_agrees_3521 : modelRow 3521 = row 3521 := model_count_3521.trans row_count_3521.symm
def row_3522 : RowCertificate 3522 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3522 1) := by change ¬ Nat.Prime 49624981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7089283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3522 2) := by change Nat.Prime 49617937; exact ProjectEulerSolutions.P58PrimeNodes.p_49617937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3522 3) := by change ¬ Nat.Prime 49610893; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 161599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3522 : row 3522 = 1 := (row_3522.checked).trans rfl
theorem model_count_3522 : modelRow 3522 = 1 := by decide
theorem modelrow_agrees_3522 : modelRow 3522 = row 3522 := model_count_3522.trans row_count_3522.symm
def row_3523 : RowCertificate 3523 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3523 1) := by change ¬ Nat.Prime 49653163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7093309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3523 2) := by change ¬ Nat.Prime 49646117; exact PrattCertificates.composite_of_product (factor := 1013) (cofactor := 49009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3523 3) := by change ¬ Nat.Prime 49639071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16546357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3523 : row 3523 = 0 := (row_3523.checked).trans rfl
theorem model_count_3523 : modelRow 3523 = 0 := by decide
theorem modelrow_agrees_3523 : modelRow 3523 = row 3523 := model_count_3523.trans row_count_3523.symm
def row_3524 : RowCertificate 3524 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3524 1) := by change ¬ Nat.Prime 49681353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16560451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3524 2) := by change ¬ Nat.Prime 49674305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9934861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3524 3) := by change Nat.Prime 49667257; exact ProjectEulerSolutions.P58PrimeNodes.p_49667257.prime
    exact decide_eq_true prime
theorem row_count_3524 : row 3524 = 1 := (row_3524.checked).trans rfl
theorem model_count_3524 : modelRow 3524 = 1 := by decide
theorem modelrow_agrees_3524 : modelRow 3524 = row 3524 := model_count_3524.trans row_count_3524.symm
def row_3525 : RowCertificate 3525 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3525 1) := by change ¬ Nat.Prime 49709551; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 482617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3525 2) := by change Nat.Prime 49702501; exact ProjectEulerSolutions.P58PrimeNodes.p_49702501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3525 3) := by change ¬ Nat.Prime 49695451; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3822727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3525 : row 3525 = 1 := (row_3525.checked).trans rfl
theorem model_count_3525 : modelRow 3525 = 1 := by decide
theorem modelrow_agrees_3525 : modelRow 3525 = row 3525 := model_count_3525.trans row_count_3525.symm
def row_3526 : RowCertificate 3526 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3526 1) := by change ¬ Nat.Prime 49737757; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 316801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3526 2) := by change ¬ Nat.Prime 49730705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9946141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3526 3) := by change ¬ Nat.Prime 49723653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16574551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3526 : row 3526 = 0 := (row_3526.checked).trans rfl
theorem model_count_3526 : modelRow 3526 = 0 := by decide
theorem modelrow_agrees_3526 : modelRow 3526 = row 3526 := model_count_3526.trans row_count_3526.symm
def row_3527 : RowCertificate 3527 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3527 1) := by change ¬ Nat.Prime 49765971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16588657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3527 2) := by change ¬ Nat.Prime 49758917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3827609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3527 3) := by change ¬ Nat.Prime 49751863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7107409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3527 : row 3527 = 0 := (row_3527.checked).trans rfl
theorem model_count_3527 : modelRow 3527 = 0 := by decide
theorem modelrow_agrees_3527 : modelRow 3527 = row 3527 := model_count_3527.trans row_count_3527.symm
def row_3528 : RowCertificate 3528 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3528 1) := by change ¬ Nat.Prime 49794193; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2620747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3528 2) := by change ¬ Nat.Prime 49787137; exact PrattCertificates.composite_of_product (factor := 2089) (cofactor := 23833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3528 3) := by change ¬ Nat.Prime 49780081; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3829237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3528 : row 3528 = 0 := (row_3528.checked).trans rfl
theorem model_count_3528 : modelRow 3528 = 0 := by decide
theorem modelrow_agrees_3528 : modelRow 3528 = row 3528 := model_count_3528.trans row_count_3528.symm
def row_3529 : RowCertificate 3529 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3529 1) := by change ¬ Nat.Prime 49822423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7117489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3529 2) := by change ¬ Nat.Prime 49815365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9963073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3529 3) := by change ¬ Nat.Prime 49808307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16602769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3529 : row 3529 = 0 := (row_3529.checked).trans rfl
theorem model_count_3529 : modelRow 3529 = 0 := by decide
theorem modelrow_agrees_3529 : modelRow 3529 = row 3529 := model_count_3529.trans row_count_3529.symm
def row_3530 : RowCertificate 3530 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3530 1) := by change ¬ Nat.Prime 49850661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16616887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3530 2) := by change ¬ Nat.Prime 49843601; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 493501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3530 3) := by change Nat.Prime 49836541; exact ProjectEulerSolutions.P58PrimeNodes.p_49836541.prime
    exact decide_eq_true prime
theorem row_count_3530 : row 3530 = 1 := (row_3530.checked).trans rfl
theorem model_count_3530 : modelRow 3530 = 1 := by decide
theorem modelrow_agrees_3530 : modelRow 3530 = row 3530 := model_count_3530.trans row_count_3530.symm
def row_3531 : RowCertificate 3531 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3531 1) := by change ¬ Nat.Prime 49878907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3836839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3531 2) := by change ¬ Nat.Prime 49871845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9974369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3531 3) := by change ¬ Nat.Prime 49864783; exact PrattCertificates.composite_of_product (factor := 619) (cofactor := 80557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3531 : row 3531 = 0 := (row_3531.checked).trans rfl
theorem model_count_3531 : modelRow 3531 = 0 := by decide
theorem modelrow_agrees_3531 : modelRow 3531 = row 3531 := model_count_3531.trans row_count_3531.symm
def row_3532 : RowCertificate 3532 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3532 1) := by change ¬ Nat.Prime 49907161; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 744883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3532 2) := by change ¬ Nat.Prime 49900097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3838469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3532 3) := by change ¬ Nat.Prime 49893033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16631011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3532 : row 3532 = 0 := (row_3532.checked).trans rfl
theorem model_count_3532 : modelRow 3532 = 0 := by decide
theorem modelrow_agrees_3532 : modelRow 3532 = row 3532 := model_count_3532.trans row_count_3532.symm
def row_3533 : RowCertificate 3533 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3533 1) := by change ¬ Nat.Prime 49935423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16645141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3533 2) := by change Nat.Prime 49928357; exact ProjectEulerSolutions.P58PrimeNodes.p_49928357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3533 3) := by change ¬ Nat.Prime 49921291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7131613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3533 : row 3533 = 1 := (row_3533.checked).trans rfl
theorem model_count_3533 : modelRow 3533 = 1 := by decide
theorem modelrow_agrees_3533 : modelRow 3533 = row 3533 := model_count_3533.trans row_count_3533.symm
def row_3534 : RowCertificate 3534 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3534 1) := by change ¬ Nat.Prime 49963693; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3843361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3534 2) := by change ¬ Nat.Prime 49956625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9991325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3534 3) := by change ¬ Nat.Prime 49949557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7135651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3534 : row 3534 = 0 := (row_3534.checked).trans rfl
theorem model_count_3534 : modelRow 3534 = 0 := by decide
theorem modelrow_agrees_3534 : modelRow 3534 = row 3534 := model_count_3534.trans row_count_3534.symm
def row_3535 : RowCertificate 3535 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3535 1) := by change ¬ Nat.Prime 49991971; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 91393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3535 2) := by change ¬ Nat.Prime 49984901; exact PrattCertificates.composite_of_product (factor := 5641) (cofactor := 8861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3535 3) := by change ¬ Nat.Prime 49977831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16659277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3535 : row 3535 = 0 := (row_3535.checked).trans rfl
theorem model_count_3535 : modelRow 3535 = 0 := by decide
theorem modelrow_agrees_3535 : modelRow 3535 = row 3535 := model_count_3535.trans row_count_3535.symm
def row_3536 : RowCertificate 3536 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3536 1) := by change ¬ Nat.Prime 50020257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16673419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3536 2) := by change ¬ Nat.Prime 50013185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10002637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3536 3) := by change ¬ Nat.Prime 50006113; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 251287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3536 : row 3536 = 0 := (row_3536.checked).trans rfl
theorem model_count_3536 : modelRow 3536 = 0 := by decide
theorem modelrow_agrees_3536 : modelRow 3536 = row 3536 := model_count_3536.trans row_count_3536.symm
def row_3537 : RowCertificate 3537 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3537 1) := by change ¬ Nat.Prime 50048551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7149793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3537 2) := by change ¬ Nat.Prime 50041477; exact PrattCertificates.composite_of_product (factor := 2381) (cofactor := 21017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3537 3) := by change ¬ Nat.Prime 50034403; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1614013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3537 : row 3537 = 0 := (row_3537.checked).trans rfl
theorem model_count_3537 : modelRow 3537 = 0 := by decide
theorem modelrow_agrees_3537 : modelRow 3537 = row 3537 := model_count_3537.trans row_count_3537.symm
def row_3538 : RowCertificate 3538 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3538 1) := by change ¬ Nat.Prime 50076853; exact PrattCertificates.composite_of_product (factor := 2161) (cofactor := 23173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3538 2) := by change ¬ Nat.Prime 50069777; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2945281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3538 3) := by change ¬ Nat.Prime 50062701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16687567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3538 : row 3538 = 0 := (row_3538.checked).trans rfl
theorem model_count_3538 : modelRow 3538 = 0 := by decide
theorem modelrow_agrees_3538 : modelRow 3538 = row 3538 := model_count_3538.trans row_count_3538.symm
def row_3539 : RowCertificate 3539 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3539 1) := by change ¬ Nat.Prime 50105163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16701721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3539 2) := by change ¬ Nat.Prime 50098085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10019617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3539 3) := by change ¬ Nat.Prime 50091007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1353811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3539 : row 3539 = 0 := (row_3539.checked).trans rfl
theorem model_count_3539 : modelRow 3539 = 0 := by decide
theorem modelrow_agrees_3539 : modelRow 3539 = row 3539 := model_count_3539.trans row_count_3539.symm
def row_3540 : RowCertificate 3540 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3540 1) := by change Nat.Prime 50133481; exact ProjectEulerSolutions.P58PrimeNodes.p_50133481.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3540 2) := by change ¬ Nat.Prime 50126401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3855877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3540 3) := by change ¬ Nat.Prime 50119321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7159903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3540 : row 3540 = 1 := (row_3540.checked).trans rfl
theorem model_count_3540 : modelRow 3540 = 1 := by decide
theorem modelrow_agrees_3540 : modelRow 3540 = row 3540 := model_count_3540.trans row_count_3540.symm
def row_3541 : RowCertificate 3541 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3541 1) := by change ¬ Nat.Prime 50161807; exact PrattCertificates.composite_of_product (factor := 6133) (cofactor := 8179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3541 2) := by change ¬ Nat.Prime 50154725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10030945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3541 3) := by change ¬ Nat.Prime 50147643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16715881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3541 : row 3541 = 0 := (row_3541.checked).trans rfl
theorem model_count_3541 : modelRow 3541 = 0 := by decide
theorem modelrow_agrees_3541 : modelRow 3541 = row 3541 := model_count_3541.trans row_count_3541.symm
def row_3542 : RowCertificate 3542 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3542 1) := by change ¬ Nat.Prime 50190141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16730047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3542 2) := by change ¬ Nat.Prime 50183057; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1223977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3542 3) := by change Nat.Prime 50175973; exact ProjectEulerSolutions.P58PrimeNodes.p_50175973.prime
    exact decide_eq_true prime
theorem row_count_3542 : row 3542 = 1 := (row_3542.checked).trans rfl
theorem model_count_3542 : modelRow 3542 = 1 := by decide
theorem modelrow_agrees_3542 : modelRow 3542 = row 3542 := model_count_3542.trans row_count_3542.symm
def row_3543 : RowCertificate 3543 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3543 1) := by change ¬ Nat.Prime 50218483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7174069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3543 2) := by change ¬ Nat.Prime 50211397; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 564173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3543 3) := by change ¬ Nat.Prime 50204311; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 181243) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3543 : row 3543 = 0 := (row_3543.checked).trans rfl
theorem model_count_3543 : modelRow 3543 = 0 := by decide
theorem modelrow_agrees_3543 : modelRow 3543 = row 3543 := model_count_3543.trans row_count_3543.symm
def row_3544 : RowCertificate 3544 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3544 1) := by change ¬ Nat.Prime 50246833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7178119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3544 2) := by change ¬ Nat.Prime 50239745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10047949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3544 3) := by change ¬ Nat.Prime 50232657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16744219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3544 : row 3544 = 0 := (row_3544.checked).trans rfl
theorem model_count_3544 : modelRow 3544 = 0 := by decide
theorem modelrow_agrees_3544 : modelRow 3544 = row 3544 := model_count_3544.trans row_count_3544.symm
def row_3545 : RowCertificate 3545 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3545 1) := by change ¬ Nat.Prime 50275191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16758397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3545 2) := by change ¬ Nat.Prime 50268101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3866777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3545 3) := by change ¬ Nat.Prime 50261011; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 823951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3545 : row 3545 = 0 := (row_3545.checked).trans rfl
theorem model_count_3545 : modelRow 3545 = 0 := by decide
theorem modelrow_agrees_3545 : modelRow 3545 = row 3545 := model_count_3545.trans row_count_3545.symm
def row_3546 : RowCertificate 3546 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3546 1) := by change ¬ Nat.Prime 50303557; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 396091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3546 2) := by change ¬ Nat.Prime 50296465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10059293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3546 3) := by change Nat.Prime 50289373; exact ProjectEulerSolutions.P58PrimeNodes.p_50289373.prime
    exact decide_eq_true prime
theorem row_count_3546 : row 3546 = 1 := (row_3546.checked).trans rfl
theorem model_count_3546 : modelRow 3546 = 1 := by decide
theorem modelrow_agrees_3546 : modelRow 3546 = row 3546 := model_count_3546.trans row_count_3546.symm
def row_3547 : RowCertificate 3547 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3547 1) := by change ¬ Nat.Prime 50331931; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3871687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3547 2) := by change ¬ Nat.Prime 50324837; exact PrattCertificates.composite_of_product (factor := 821) (cofactor := 61297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3547 3) := by change ¬ Nat.Prime 50317743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16772581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3547 : row 3547 = 0 := (row_3547.checked).trans rfl
theorem model_count_3547 : modelRow 3547 = 0 := by decide
theorem modelrow_agrees_3547 : modelRow 3547 = row 3547 := model_count_3547.trans row_count_3547.symm
def row_3548 : RowCertificate 3548 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3548 1) := by change ¬ Nat.Prime 50360313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16786771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3548 2) := by change ¬ Nat.Prime 50353217; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 123113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3548 3) := by change ¬ Nat.Prime 50346121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7192303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3548 : row 3548 = 0 := (row_3548.checked).trans rfl
theorem model_count_3548 : modelRow 3548 = 0 := by decide
theorem modelrow_agrees_3548 : modelRow 3548 = row 3548 := model_count_3548.trans row_count_3548.symm
def row_3549 : RowCertificate 3549 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3549 1) := by change ¬ Nat.Prime 50388703; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2652037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3549 2) := by change ¬ Nat.Prime 50381605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10076321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3549 3) := by change ¬ Nat.Prime 50374507; exact PrattCertificates.composite_of_product (factor := 1063) (cofactor := 47389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3549 : row 3549 = 0 := (row_3549.checked).trans rfl
theorem model_count_3549 : modelRow 3549 = 0 := by decide
theorem modelrow_agrees_3549 : modelRow 3549 = row 3549 := model_count_3549.trans row_count_3549.symm
def row_3550 : RowCertificate 3550 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3550 1) := by change ¬ Nat.Prime 50417101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7202443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3550 2) := by change Nat.Prime 50410001; exact ProjectEulerSolutions.P58PrimeNodes.p_50410001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3550 3) := by change ¬ Nat.Prime 50402901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16800967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3550 : row 3550 = 1 := (row_3550.checked).trans rfl
theorem model_count_3550 : modelRow 3550 = 1 := by decide
theorem modelrow_agrees_3550 : modelRow 3550 = row 3550 := model_count_3550.trans row_count_3550.symm
def row_3551 : RowCertificate 3551 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3551 1) := by change ¬ Nat.Prime 50445507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16815169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3551 2) := by change ¬ Nat.Prime 50438405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10087681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3551 3) := by change ¬ Nat.Prime 50431303; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3879331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3551 : row 3551 = 0 := (row_3551.checked).trans rfl
theorem model_count_3551 : modelRow 3551 = 0 := by decide
theorem modelrow_agrees_3551 : modelRow 3551 = row 3551 := model_count_3551.trans row_count_3551.symm
def row_3552 : RowCertificate 3552 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3552 1) := by change ¬ Nat.Prime 50473921; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1628191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3552 2) := by change ¬ Nat.Prime 50466817; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 446609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3552 3) := by change ¬ Nat.Prime 50459713; exact PrattCertificates.composite_of_product (factor := 1471) (cofactor := 34303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3552 : row 3552 = 0 := (row_3552.checked).trans rfl
theorem model_count_3552 : modelRow 3552 = 0 := by decide
theorem modelrow_agrees_3552 : modelRow 3552 = row 3552 := model_count_3552.trans row_count_3552.symm
def row_3553 : RowCertificate 3553 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3553 1) := by change Nat.Prime 50502343; exact ProjectEulerSolutions.P58PrimeNodes.p_50502343.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3553 2) := by change ¬ Nat.Prime 50495237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3884249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3553 3) := by change ¬ Nat.Prime 50488131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16829377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3553 : row 3553 = 1 := (row_3553.checked).trans rfl
theorem model_count_3553 : modelRow 3553 = 1 := by decide
theorem modelrow_agrees_3553 : modelRow 3553 = row 3553 := model_count_3553.trans row_count_3553.symm
def row_3554 : RowCertificate 3554 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3554 1) := by change ¬ Nat.Prime 50530773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16843591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3554 2) := by change ¬ Nat.Prime 50523665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10104733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3554 3) := by change ¬ Nat.Prime 50516557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7216651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3554 : row 3554 = 0 := (row_3554.checked).trans rfl
theorem model_count_3554 : modelRow 3554 = 0 := by decide
theorem modelrow_agrees_3554 : modelRow 3554 = row 3554 := model_count_3554.trans row_count_3554.symm
def row_3555 : RowCertificate 3555 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3555 1) := by change Nat.Prime 50559211; exact ProjectEulerSolutions.P58PrimeNodes.p_50559211.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3555 2) := by change ¬ Nat.Prime 50552101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2973653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3555 3) := by change ¬ Nat.Prime 50544991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7220713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3555 : row 3555 = 1 := (row_3555.checked).trans rfl
theorem model_count_3555 : modelRow 3555 = 1 := by decide
theorem modelrow_agrees_3555 : modelRow 3555 = row 3555 := model_count_3555.trans row_count_3555.symm
def row_3556 : RowCertificate 3556 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3556 1) := by change Nat.Prime 50587657; exact ProjectEulerSolutions.P58PrimeNodes.p_50587657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3556 2) := by change ¬ Nat.Prime 50580545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10116109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3556 3) := by change ¬ Nat.Prime 50573433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16857811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3556 : row 3556 = 1 := (row_3556.checked).trans rfl
theorem model_count_3556 : modelRow 3556 = 1 := by decide
theorem modelrow_agrees_3556 : modelRow 3556 = row 3556 := model_count_3556.trans row_count_3556.symm
def row_3557 : RowCertificate 3557 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3557 1) := by change ¬ Nat.Prime 50616111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16872037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3557 2) := by change Nat.Prime 50608997; exact ProjectEulerSolutions.P58PrimeNodes.p_50608997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3557 3) := by change ¬ Nat.Prime 50601883; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2663257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3557 : row 3557 = 1 := (row_3557.checked).trans rfl
theorem model_count_3557 : modelRow 3557 = 1 := by decide
theorem modelrow_agrees_3557 : modelRow 3557 = row 3557 := model_count_3557.trans row_count_3557.symm
def row_3558 : RowCertificate 3558 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3558 1) := by change ¬ Nat.Prime 50644573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7234939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3558 2) := by change ¬ Nat.Prime 50637457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3895189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3558 3) := by change Nat.Prime 50630341; exact ProjectEulerSolutions.P58PrimeNodes.p_50630341.prime
    exact decide_eq_true prime
theorem row_count_3558 : row 3558 = 1 := (row_3558.checked).trans rfl
theorem model_count_3558 : modelRow 3558 = 1 := by decide
theorem modelrow_agrees_3558 : modelRow 3558 = row 3558 := model_count_3558.trans row_count_3558.symm
def row_3559 : RowCertificate 3559 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3559 1) := by change Nat.Prime 50673043; exact ProjectEulerSolutions.P58PrimeNodes.p_50673043.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3559 2) := by change ¬ Nat.Prime 50665925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10133185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3559 3) := by change ¬ Nat.Prime 50658807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16886269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3559 : row 3559 = 1 := (row_3559.checked).trans rfl
theorem model_count_3559 : modelRow 3559 = 1 := by decide
theorem modelrow_agrees_3559 : modelRow 3559 = row 3559 := model_count_3559.trans row_count_3559.symm
def row_3560 : RowCertificate 3560 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3560 1) := by change ¬ Nat.Prime 50701521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16900507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3560 2) := by change ¬ Nat.Prime 50694401; exact PrattCertificates.composite_of_product (factor := 5381) (cofactor := 9421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3560 3) := by change ¬ Nat.Prime 50687281; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 179107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3560 : row 3560 = 0 := (row_3560.checked).trans rfl
theorem model_count_3560 : modelRow 3560 = 0 := by decide
theorem modelrow_agrees_3560 : modelRow 3560 = row 3560 := model_count_3560.trans row_count_3560.symm
def row_3561 : RowCertificate 3561 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3561 1) := by change Nat.Prime 50730007; exact ProjectEulerSolutions.P58PrimeNodes.p_50730007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3561 2) := by change ¬ Nat.Prime 50722885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10144577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3561 3) := by change ¬ Nat.Prime 50715763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7245109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3561 : row 3561 = 1 := (row_3561.checked).trans rfl
theorem model_count_3561 : modelRow 3561 = 1 := by decide
theorem modelrow_agrees_3561 : modelRow 3561 = row 3561 := model_count_3561.trans row_count_3561.symm
def row_3562 : RowCertificate 3562 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3562 1) := by change ¬ Nat.Prime 50758501; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1637371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3562 2) := by change ¬ Nat.Prime 50751377; exact PrattCertificates.composite_of_product (factor := 2029) (cofactor := 25013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3562 3) := by change ¬ Nat.Prime 50744253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16914751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3562 : row 3562 = 0 := (row_3562.checked).trans rfl
theorem model_count_3562 : modelRow 3562 = 0 := by decide
theorem modelrow_agrees_3562 : modelRow 3562 = row 3562 := model_count_3562.trans row_count_3562.symm
def row_3563 : RowCertificate 3563 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3563 1) := by change ¬ Nat.Prime 50787003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16929001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3563 2) := by change ¬ Nat.Prime 50779877; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 832457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3563 3) := by change ¬ Nat.Prime 50772751; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 101749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3563 : row 3563 = 0 := (row_3563.checked).trans rfl
theorem model_count_3563 : modelRow 3563 = 0 := by decide
theorem modelrow_agrees_3563 : modelRow 3563 = row 3563 := model_count_3563.trans row_count_3563.symm
def row_3564 : RowCertificate 3564 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3564 1) := by change ¬ Nat.Prime 50815513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7259359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3564 2) := by change ¬ Nat.Prime 50808385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10161677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3564 3) := by change ¬ Nat.Prime 50801257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3907789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3564 : row 3564 = 0 := (row_3564.checked).trans rfl
theorem model_count_3564 : modelRow 3564 = 0 := by decide
theorem modelrow_agrees_3564 : modelRow 3564 = row 3564 := model_count_3564.trans row_count_3564.symm
def row_3565 : RowCertificate 3565 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3565 1) := by change ¬ Nat.Prime 50844031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7263433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3565 2) := by change Nat.Prime 50836901; exact ProjectEulerSolutions.P58PrimeNodes.p_50836901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3565 3) := by change ¬ Nat.Prime 50829771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16943257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3565 : row 3565 = 1 := (row_3565.checked).trans rfl
theorem model_count_3565 : modelRow 3565 = 1 := by decide
theorem modelrow_agrees_3565 : modelRow 3565 = row 3565 := model_count_3565.trans row_count_3565.symm
def row_3566 : RowCertificate 3566 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3566 1) := by change ¬ Nat.Prime 50872557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16957519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3566 2) := by change ¬ Nat.Prime 50865425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10173085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3566 3) := by change ¬ Nat.Prime 50858293; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1182751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3566 : row 3566 = 0 := (row_3566.checked).trans rfl
theorem model_count_3566 : modelRow 3566 = 0 := by decide
theorem modelrow_agrees_3566 : modelRow 3566 = row 3566 := model_count_3566.trans row_count_3566.symm
def row_3567 : RowCertificate 3567 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3567 1) := by change ¬ Nat.Prime 50901091; exact PrattCertificates.composite_of_product (factor := 3433) (cofactor := 14827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3567 2) := by change ¬ Nat.Prime 50893957; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 450389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3567 3) := by change ¬ Nat.Prime 50886823; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3914371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3567 : row 3567 = 0 := (row_3567.checked).trans rfl
theorem model_count_3567 : modelRow 3567 = 0 := by decide
theorem modelrow_agrees_3567 : modelRow 3567 = row 3567 := model_count_3567.trans row_count_3567.symm
def row_3568 : RowCertificate 3568 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3568 1) := by change ¬ Nat.Prime 50929633; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2680507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3568 2) := by change ¬ Nat.Prime 50922497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2995441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3568 3) := by change ¬ Nat.Prime 50915361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16971787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3568 : row 3568 = 0 := (row_3568.checked).trans rfl
theorem model_count_3568 : modelRow 3568 = 0 := by decide
theorem modelrow_agrees_3568 : modelRow 3568 = row 3568 := model_count_3568.trans row_count_3568.symm
def row_3569 : RowCertificate 3569 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3569 1) := by change ¬ Nat.Prime 50958183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 16986061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3569 2) := by change ¬ Nat.Prime 50951045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10190209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3569 3) := by change ¬ Nat.Prime 50943907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7277701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3569 : row 3569 = 0 := (row_3569.checked).trans rfl
theorem model_count_3569 : modelRow 3569 = 0 := by decide
theorem modelrow_agrees_3569 : modelRow 3569 = row 3569 := model_count_3569.trans row_count_3569.symm
def row_3570 : RowCertificate 3570 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3570 1) := by change ¬ Nat.Prime 50986741; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3922057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3570 2) := by change ¬ Nat.Prime 50979601; exact PrattCertificates.composite_of_product (factor := 997) (cofactor := 51133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3570 3) := by change ¬ Nat.Prime 50972461; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 760783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3570 : row 3570 = 0 := (row_3570.checked).trans rfl
theorem model_count_3570 : modelRow 3570 = 0 := by decide
theorem modelrow_agrees_3570 : modelRow 3570 = row 3570 := model_count_3570.trans row_count_3570.symm
def row_3571 : RowCertificate 3571 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3571 1) := by change ¬ Nat.Prime 51015307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7287901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3571 2) := by change ¬ Nat.Prime 51008165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10201633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3571 3) := by change ¬ Nat.Prime 51001023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17000341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3571 : row 3571 = 0 := (row_3571.checked).trans rfl
theorem model_count_3571 : modelRow 3571 = 0 := by decide
theorem modelrow_agrees_3571 : modelRow 3571 = row 3571 := model_count_3571.trans row_count_3571.symm
def row_3572 : RowCertificate 3572 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3572 1) := by change ¬ Nat.Prime 51043881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17014627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3572 2) := by change ¬ Nat.Prime 51036737; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3002161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3572 3) := by change Nat.Prime 51029593; exact ProjectEulerSolutions.P58PrimeNodes.p_51029593.prime
    exact decide_eq_true prime
theorem row_count_3572 : row 3572 = 1 := (row_3572.checked).trans rfl
theorem model_count_3572 : modelRow 3572 = 1 := by decide
theorem modelrow_agrees_3572 : modelRow 3572 = row 3572 := model_count_3572.trans row_count_3572.symm
def row_3573 : RowCertificate 3573 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3573 1) := by change ¬ Nat.Prime 51072463; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3928651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3573 2) := by change ¬ Nat.Prime 51065317; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1760873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3573 3) := by change ¬ Nat.Prime 51058171; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 699427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3573 : row 3573 = 0 := (row_3573.checked).trans rfl
theorem model_count_3573 : modelRow 3573 = 0 := by decide
theorem modelrow_agrees_3573 : modelRow 3573 = row 3573 := model_count_3573.trans row_count_3573.symm
def row_3574 : RowCertificate 3574 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3574 1) := by change ¬ Nat.Prime 51101053; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 468817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3574 2) := by change ¬ Nat.Prime 51093905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10218781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3574 3) := by change ¬ Nat.Prime 51086757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17028919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3574 : row 3574 = 0 := (row_3574.checked).trans rfl
theorem model_count_3574 : modelRow 3574 = 0 := by decide
theorem modelrow_agrees_3574 : modelRow 3574 = row 3574 := model_count_3574.trans row_count_3574.symm
def row_3575 : RowCertificate 3575 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3575 1) := by change ¬ Nat.Prime 51129651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17043217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3575 2) := by change Nat.Prime 51122501; exact ProjectEulerSolutions.P58PrimeNodes.p_51122501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3575 3) := by change ¬ Nat.Prime 51115351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7302193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3575 : row 3575 = 1 := (row_3575.checked).trans rfl
theorem model_count_3575 : modelRow 3575 = 1 := by decide
theorem modelrow_agrees_3575 : modelRow 3575 = row 3575 := model_count_3575.trans row_count_3575.symm
def row_3576 : RowCertificate 3576 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3576 1) := by change Nat.Prime 51158257; exact ProjectEulerSolutions.P58PrimeNodes.p_51158257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3576 2) := by change ¬ Nat.Prime 51151105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10230221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3576 3) := by change ¬ Nat.Prime 51143953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7306279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3576 : row 3576 = 1 := (row_3576.checked).trans rfl
theorem model_count_3576 : modelRow 3576 = 1 := by decide
theorem modelrow_agrees_3576 : modelRow 3576 = row 3576 := model_count_3576.trans row_count_3576.symm
def row_3577 : RowCertificate 3577 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3577 1) := by change ¬ Nat.Prime 51186871; exact PrattCertificates.composite_of_product (factor := 3499) (cofactor := 14629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3577 2) := by change ¬ Nat.Prime 51179717; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 575053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3577 3) := by change ¬ Nat.Prime 51172563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17057521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3577 : row 3577 = 0 := (row_3577.checked).trans rfl
theorem model_count_3577 : modelRow 3577 = 0 := by decide
theorem modelrow_agrees_3577 : modelRow 3577 = row 3577 := model_count_3577.trans row_count_3577.symm
def row_3578 : RowCertificate 3578 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3578 1) := by change ¬ Nat.Prime 51215493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17071831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3578 2) := by change ¬ Nat.Prime 51208337; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 527921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3578 3) := by change ¬ Nat.Prime 51201181; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2694799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3578 : row 3578 = 0 := (row_3578.checked).trans rfl
theorem model_count_3578 : modelRow 3578 = 0 := by decide
theorem modelrow_agrees_3578 : modelRow 3578 = row 3578 := model_count_3578.trans row_count_3578.symm
def row_3579 : RowCertificate 3579 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3579 1) := by change ¬ Nat.Prime 51244123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7320589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3579 2) := by change ¬ Nat.Prime 51236965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10247393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3579 3) := by change Nat.Prime 51229807; exact ProjectEulerSolutions.P58PrimeNodes.p_51229807.prime
    exact decide_eq_true prime
theorem row_count_3579 : row 3579 = 1 := (row_3579.checked).trans rfl
theorem model_count_3579 : modelRow 3579 = 1 := by decide
theorem modelrow_agrees_3579 : modelRow 3579 = row 3579 := model_count_3579.trans row_count_3579.symm
def row_3580 : RowCertificate 3580 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3580 1) := by change Nat.Prime 51272761; exact ProjectEulerSolutions.P58PrimeNodes.p_51272761.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3580 2) := by change Nat.Prime 51265601; exact ProjectEulerSolutions.P58PrimeNodes.p_51265601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3580 3) := by change ¬ Nat.Prime 51258441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17086147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3580 : row 3580 = 2 := (row_3580.checked).trans rfl
theorem model_count_3580 : modelRow 3580 = 2 := by decide
theorem modelrow_agrees_3580 : modelRow 3580 = row 3580 := model_count_3580.trans row_count_3580.symm
def row_3581 : RowCertificate 3581 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3581 1) := by change ¬ Nat.Prime 51301407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17100469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3581 2) := by change ¬ Nat.Prime 51294245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10258849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3581 3) := by change ¬ Nat.Prime 51287083; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 45427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3581 : row 3581 = 0 := (row_3581.checked).trans rfl
theorem model_count_3581 : modelRow 3581 = 0 := by decide
theorem modelrow_agrees_3581 : modelRow 3581 = row 3581 := model_count_3581.trans row_count_3581.symm
def row_3582 : RowCertificate 3582 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3582 1) := by change Nat.Prime 51330061; exact ProjectEulerSolutions.P58PrimeNodes.p_51330061.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3582 2) := by change Nat.Prime 51322897; exact ProjectEulerSolutions.P58PrimeNodes.p_51322897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3582 3) := by change ¬ Nat.Prime 51315733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7330819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3582 : row 3582 = 2 := (row_3582.checked).trans rfl
theorem model_count_3582 : modelRow 3582 = 2 := by decide
theorem modelrow_agrees_3582 : modelRow 3582 = row 3582 := model_count_3582.trans row_count_3582.symm
def row_3583 : RowCertificate 3583 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3583 1) := by change ¬ Nat.Prime 51358723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3950671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3583 2) := by change ¬ Nat.Prime 51351557; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1252477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3583 3) := by change ¬ Nat.Prime 51344391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17114797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3583 : row 3583 = 0 := (row_3583.checked).trans rfl
theorem model_count_3583 : modelRow 3583 = 0 := by decide
theorem modelrow_agrees_3583 : modelRow 3583 = row 3583 := model_count_3583.trans row_count_3583.symm
def row_3584 : RowCertificate 3584 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3584 1) := by change ¬ Nat.Prime 51387393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17129131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3584 2) := by change ¬ Nat.Prime 51380225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10276045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3584 3) := by change ¬ Nat.Prime 51373057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1388461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3584 : row 3584 = 0 := (row_3584.checked).trans rfl
theorem model_count_3584 : modelRow 3584 = 0 := by decide
theorem modelrow_agrees_3584 : modelRow 3584 = row 3584 := model_count_3584.trans row_count_3584.symm
private def atomic_3457 : CountSegment 1 10 13120 3456 3457 1644 1645 := CountSegment.single (first := 3456) (initial := 1644) row_count_3457 (by decide)
private theorem atomic_3457_agreement : RowAgreement 3456 3457 := RowAgreement.single (first := 3456) modelrow_agrees_3457
private def atomic_3458 : CountSegment 1 10 13120 3457 3458 1645 1646 := CountSegment.single (first := 3457) (initial := 1645) row_count_3458 (by decide)
private theorem atomic_3458_agreement : RowAgreement 3457 3458 := RowAgreement.single (first := 3457) modelrow_agrees_3458
private def atomic_3459 : CountSegment 1 10 13120 3458 3459 1646 1647 := CountSegment.single (first := 3458) (initial := 1646) row_count_3459 (by decide)
private theorem atomic_3459_agreement : RowAgreement 3458 3459 := RowAgreement.single (first := 3458) modelrow_agrees_3459
private def atomic_3460 : CountSegment 1 10 13120 3459 3460 1647 1647 := CountSegment.single (first := 3459) (initial := 1647) row_count_3460 (by decide)
private theorem atomic_3460_agreement : RowAgreement 3459 3460 := RowAgreement.single (first := 3459) modelrow_agrees_3460
private def atomic_3461 : CountSegment 1 10 13120 3460 3461 1647 1648 := CountSegment.single (first := 3460) (initial := 1647) row_count_3461 (by decide)
private theorem atomic_3461_agreement : RowAgreement 3460 3461 := RowAgreement.single (first := 3460) modelrow_agrees_3461
private def atomic_3462 : CountSegment 1 10 13120 3461 3462 1648 1648 := CountSegment.single (first := 3461) (initial := 1648) row_count_3462 (by decide)
private theorem atomic_3462_agreement : RowAgreement 3461 3462 := RowAgreement.single (first := 3461) modelrow_agrees_3462
private def atomic_3463 : CountSegment 1 10 13120 3462 3463 1648 1649 := CountSegment.single (first := 3462) (initial := 1648) row_count_3463 (by decide)
private theorem atomic_3463_agreement : RowAgreement 3462 3463 := RowAgreement.single (first := 3462) modelrow_agrees_3463
private def atomic_3464 : CountSegment 1 10 13120 3463 3464 1649 1649 := CountSegment.single (first := 3463) (initial := 1649) row_count_3464 (by decide)
private theorem atomic_3464_agreement : RowAgreement 3463 3464 := RowAgreement.single (first := 3463) modelrow_agrees_3464
private def atomic_3465 : CountSegment 1 10 13120 3464 3465 1649 1650 := CountSegment.single (first := 3464) (initial := 1649) row_count_3465 (by decide)
private theorem atomic_3465_agreement : RowAgreement 3464 3465 := RowAgreement.single (first := 3464) modelrow_agrees_3465
private def atomic_3466 : CountSegment 1 10 13120 3465 3466 1650 1650 := CountSegment.single (first := 3465) (initial := 1650) row_count_3466 (by decide)
private theorem atomic_3466_agreement : RowAgreement 3465 3466 := RowAgreement.single (first := 3465) modelrow_agrees_3466
private def atomic_3467 : CountSegment 1 10 13120 3466 3467 1650 1650 := CountSegment.single (first := 3466) (initial := 1650) row_count_3467 (by decide)
private theorem atomic_3467_agreement : RowAgreement 3466 3467 := RowAgreement.single (first := 3466) modelrow_agrees_3467
private def atomic_3468 : CountSegment 1 10 13120 3467 3468 1650 1650 := CountSegment.single (first := 3467) (initial := 1650) row_count_3468 (by decide)
private theorem atomic_3468_agreement : RowAgreement 3467 3468 := RowAgreement.single (first := 3467) modelrow_agrees_3468
private def atomic_3469 : CountSegment 1 10 13120 3468 3469 1650 1650 := CountSegment.single (first := 3468) (initial := 1650) row_count_3469 (by decide)
private theorem atomic_3469_agreement : RowAgreement 3468 3469 := RowAgreement.single (first := 3468) modelrow_agrees_3469
private def atomic_3470 : CountSegment 1 10 13120 3469 3470 1650 1650 := CountSegment.single (first := 3469) (initial := 1650) row_count_3470 (by decide)
private theorem atomic_3470_agreement : RowAgreement 3469 3470 := RowAgreement.single (first := 3469) modelrow_agrees_3470
private def atomic_3471 : CountSegment 1 10 13120 3470 3471 1650 1650 := CountSegment.single (first := 3470) (initial := 1650) row_count_3471 (by decide)
private theorem atomic_3471_agreement : RowAgreement 3470 3471 := RowAgreement.single (first := 3470) modelrow_agrees_3471
private def atomic_3472 : CountSegment 1 10 13120 3471 3472 1650 1652 := CountSegment.single (first := 3471) (initial := 1650) row_count_3472 (by decide)
private theorem atomic_3472_agreement : RowAgreement 3471 3472 := RowAgreement.single (first := 3471) modelrow_agrees_3472
private def atomic_3473 : CountSegment 1 10 13120 3472 3473 1652 1652 := CountSegment.single (first := 3472) (initial := 1652) row_count_3473 (by decide)
private theorem atomic_3473_agreement : RowAgreement 3472 3473 := RowAgreement.single (first := 3472) modelrow_agrees_3473
private def atomic_3474 : CountSegment 1 10 13120 3473 3474 1652 1652 := CountSegment.single (first := 3473) (initial := 1652) row_count_3474 (by decide)
private theorem atomic_3474_agreement : RowAgreement 3473 3474 := RowAgreement.single (first := 3473) modelrow_agrees_3474
private def atomic_3475 : CountSegment 1 10 13120 3474 3475 1652 1652 := CountSegment.single (first := 3474) (initial := 1652) row_count_3475 (by decide)
private theorem atomic_3475_agreement : RowAgreement 3474 3475 := RowAgreement.single (first := 3474) modelrow_agrees_3475
private def atomic_3476 : CountSegment 1 10 13120 3475 3476 1652 1652 := CountSegment.single (first := 3475) (initial := 1652) row_count_3476 (by decide)
private theorem atomic_3476_agreement : RowAgreement 3475 3476 := RowAgreement.single (first := 3475) modelrow_agrees_3476
private def atomic_3477 : CountSegment 1 10 13120 3476 3477 1652 1653 := CountSegment.single (first := 3476) (initial := 1652) row_count_3477 (by decide)
private theorem atomic_3477_agreement : RowAgreement 3476 3477 := RowAgreement.single (first := 3476) modelrow_agrees_3477
private def atomic_3478 : CountSegment 1 10 13120 3477 3478 1653 1655 := CountSegment.single (first := 3477) (initial := 1653) row_count_3478 (by decide)
private theorem atomic_3478_agreement : RowAgreement 3477 3478 := RowAgreement.single (first := 3477) modelrow_agrees_3478
private def atomic_3479 : CountSegment 1 10 13120 3478 3479 1655 1655 := CountSegment.single (first := 3478) (initial := 1655) row_count_3479 (by decide)
private theorem atomic_3479_agreement : RowAgreement 3478 3479 := RowAgreement.single (first := 3478) modelrow_agrees_3479
private def atomic_3480 : CountSegment 1 10 13120 3479 3480 1655 1655 := CountSegment.single (first := 3479) (initial := 1655) row_count_3480 (by decide)
private theorem atomic_3480_agreement : RowAgreement 3479 3480 := RowAgreement.single (first := 3479) modelrow_agrees_3480
private def atomic_3481 : CountSegment 1 10 13120 3480 3481 1655 1655 := CountSegment.single (first := 3480) (initial := 1655) row_count_3481 (by decide)
private theorem atomic_3481_agreement : RowAgreement 3480 3481 := RowAgreement.single (first := 3480) modelrow_agrees_3481
private def atomic_3482 : CountSegment 1 10 13120 3481 3482 1655 1656 := CountSegment.single (first := 3481) (initial := 1655) row_count_3482 (by decide)
private theorem atomic_3482_agreement : RowAgreement 3481 3482 := RowAgreement.single (first := 3481) modelrow_agrees_3482
private def atomic_3483 : CountSegment 1 10 13120 3482 3483 1656 1656 := CountSegment.single (first := 3482) (initial := 1656) row_count_3483 (by decide)
private theorem atomic_3483_agreement : RowAgreement 3482 3483 := RowAgreement.single (first := 3482) modelrow_agrees_3483
private def atomic_3484 : CountSegment 1 10 13120 3483 3484 1656 1657 := CountSegment.single (first := 3483) (initial := 1656) row_count_3484 (by decide)
private theorem atomic_3484_agreement : RowAgreement 3483 3484 := RowAgreement.single (first := 3483) modelrow_agrees_3484
private def atomic_3485 : CountSegment 1 10 13120 3484 3485 1657 1658 := CountSegment.single (first := 3484) (initial := 1657) row_count_3485 (by decide)
private theorem atomic_3485_agreement : RowAgreement 3484 3485 := RowAgreement.single (first := 3484) modelrow_agrees_3485
private def atomic_3486 : CountSegment 1 10 13120 3485 3486 1658 1658 := CountSegment.single (first := 3485) (initial := 1658) row_count_3486 (by decide)
private theorem atomic_3486_agreement : RowAgreement 3485 3486 := RowAgreement.single (first := 3485) modelrow_agrees_3486
private def atomic_3487 : CountSegment 1 10 13120 3486 3487 1658 1658 := CountSegment.single (first := 3486) (initial := 1658) row_count_3487 (by decide)
private theorem atomic_3487_agreement : RowAgreement 3486 3487 := RowAgreement.single (first := 3486) modelrow_agrees_3487
private def atomic_3488 : CountSegment 1 10 13120 3487 3488 1658 1658 := CountSegment.single (first := 3487) (initial := 1658) row_count_3488 (by decide)
private theorem atomic_3488_agreement : RowAgreement 3487 3488 := RowAgreement.single (first := 3487) modelrow_agrees_3488
private def atomic_3489 : CountSegment 1 10 13120 3488 3489 1658 1658 := CountSegment.single (first := 3488) (initial := 1658) row_count_3489 (by decide)
private theorem atomic_3489_agreement : RowAgreement 3488 3489 := RowAgreement.single (first := 3488) modelrow_agrees_3489
private def atomic_3490 : CountSegment 1 10 13120 3489 3490 1658 1659 := CountSegment.single (first := 3489) (initial := 1658) row_count_3490 (by decide)
private theorem atomic_3490_agreement : RowAgreement 3489 3490 := RowAgreement.single (first := 3489) modelrow_agrees_3490
private def atomic_3491 : CountSegment 1 10 13120 3490 3491 1659 1659 := CountSegment.single (first := 3490) (initial := 1659) row_count_3491 (by decide)
private theorem atomic_3491_agreement : RowAgreement 3490 3491 := RowAgreement.single (first := 3490) modelrow_agrees_3491
private def atomic_3492 : CountSegment 1 10 13120 3491 3492 1659 1660 := CountSegment.single (first := 3491) (initial := 1659) row_count_3492 (by decide)
private theorem atomic_3492_agreement : RowAgreement 3491 3492 := RowAgreement.single (first := 3491) modelrow_agrees_3492
private def atomic_3493 : CountSegment 1 10 13120 3492 3493 1660 1661 := CountSegment.single (first := 3492) (initial := 1660) row_count_3493 (by decide)
private theorem atomic_3493_agreement : RowAgreement 3492 3493 := RowAgreement.single (first := 3492) modelrow_agrees_3493
private def atomic_3494 : CountSegment 1 10 13120 3493 3494 1661 1662 := CountSegment.single (first := 3493) (initial := 1661) row_count_3494 (by decide)
private theorem atomic_3494_agreement : RowAgreement 3493 3494 := RowAgreement.single (first := 3493) modelrow_agrees_3494
private def atomic_3495 : CountSegment 1 10 13120 3494 3495 1662 1663 := CountSegment.single (first := 3494) (initial := 1662) row_count_3495 (by decide)
private theorem atomic_3495_agreement : RowAgreement 3494 3495 := RowAgreement.single (first := 3494) modelrow_agrees_3495
private def atomic_3496 : CountSegment 1 10 13120 3495 3496 1663 1663 := CountSegment.single (first := 3495) (initial := 1663) row_count_3496 (by decide)
private theorem atomic_3496_agreement : RowAgreement 3495 3496 := RowAgreement.single (first := 3495) modelrow_agrees_3496
private def atomic_3497 : CountSegment 1 10 13120 3496 3497 1663 1664 := CountSegment.single (first := 3496) (initial := 1663) row_count_3497 (by decide)
private theorem atomic_3497_agreement : RowAgreement 3496 3497 := RowAgreement.single (first := 3496) modelrow_agrees_3497
private def atomic_3498 : CountSegment 1 10 13120 3497 3498 1664 1665 := CountSegment.single (first := 3497) (initial := 1664) row_count_3498 (by decide)
private theorem atomic_3498_agreement : RowAgreement 3497 3498 := RowAgreement.single (first := 3497) modelrow_agrees_3498
private def atomic_3499 : CountSegment 1 10 13120 3498 3499 1665 1665 := CountSegment.single (first := 3498) (initial := 1665) row_count_3499 (by decide)
private theorem atomic_3499_agreement : RowAgreement 3498 3499 := RowAgreement.single (first := 3498) modelrow_agrees_3499
private def atomic_3500 : CountSegment 1 10 13120 3499 3500 1665 1665 := CountSegment.single (first := 3499) (initial := 1665) row_count_3500 (by decide)
private theorem atomic_3500_agreement : RowAgreement 3499 3500 := RowAgreement.single (first := 3499) modelrow_agrees_3500
private def atomic_3501 : CountSegment 1 10 13120 3500 3501 1665 1665 := CountSegment.single (first := 3500) (initial := 1665) row_count_3501 (by decide)
private theorem atomic_3501_agreement : RowAgreement 3500 3501 := RowAgreement.single (first := 3500) modelrow_agrees_3501
private def atomic_3502 : CountSegment 1 10 13120 3501 3502 1665 1665 := CountSegment.single (first := 3501) (initial := 1665) row_count_3502 (by decide)
private theorem atomic_3502_agreement : RowAgreement 3501 3502 := RowAgreement.single (first := 3501) modelrow_agrees_3502
private def atomic_3503 : CountSegment 1 10 13120 3502 3503 1665 1665 := CountSegment.single (first := 3502) (initial := 1665) row_count_3503 (by decide)
private theorem atomic_3503_agreement : RowAgreement 3502 3503 := RowAgreement.single (first := 3502) modelrow_agrees_3503
private def atomic_3504 : CountSegment 1 10 13120 3503 3504 1665 1665 := CountSegment.single (first := 3503) (initial := 1665) row_count_3504 (by decide)
private theorem atomic_3504_agreement : RowAgreement 3503 3504 := RowAgreement.single (first := 3503) modelrow_agrees_3504
private def atomic_3505 : CountSegment 1 10 13120 3504 3505 1665 1666 := CountSegment.single (first := 3504) (initial := 1665) row_count_3505 (by decide)
private theorem atomic_3505_agreement : RowAgreement 3504 3505 := RowAgreement.single (first := 3504) modelrow_agrees_3505
private def atomic_3506 : CountSegment 1 10 13120 3505 3506 1666 1666 := CountSegment.single (first := 3505) (initial := 1666) row_count_3506 (by decide)
private theorem atomic_3506_agreement : RowAgreement 3505 3506 := RowAgreement.single (first := 3505) modelrow_agrees_3506
private def atomic_3507 : CountSegment 1 10 13120 3506 3507 1666 1669 := CountSegment.single (first := 3506) (initial := 1666) row_count_3507 (by decide)
private theorem atomic_3507_agreement : RowAgreement 3506 3507 := RowAgreement.single (first := 3506) modelrow_agrees_3507
private def atomic_3508 : CountSegment 1 10 13120 3507 3508 1669 1670 := CountSegment.single (first := 3507) (initial := 1669) row_count_3508 (by decide)
private theorem atomic_3508_agreement : RowAgreement 3507 3508 := RowAgreement.single (first := 3507) modelrow_agrees_3508
private def atomic_3509 : CountSegment 1 10 13120 3508 3509 1670 1670 := CountSegment.single (first := 3508) (initial := 1670) row_count_3509 (by decide)
private theorem atomic_3509_agreement : RowAgreement 3508 3509 := RowAgreement.single (first := 3508) modelrow_agrees_3509
private def atomic_3510 : CountSegment 1 10 13120 3509 3510 1670 1670 := CountSegment.single (first := 3509) (initial := 1670) row_count_3510 (by decide)
private theorem atomic_3510_agreement : RowAgreement 3509 3510 := RowAgreement.single (first := 3509) modelrow_agrees_3510
private def atomic_3511 : CountSegment 1 10 13120 3510 3511 1670 1670 := CountSegment.single (first := 3510) (initial := 1670) row_count_3511 (by decide)
private theorem atomic_3511_agreement : RowAgreement 3510 3511 := RowAgreement.single (first := 3510) modelrow_agrees_3511
private def atomic_3512 : CountSegment 1 10 13120 3511 3512 1670 1670 := CountSegment.single (first := 3511) (initial := 1670) row_count_3512 (by decide)
private theorem atomic_3512_agreement : RowAgreement 3511 3512 := RowAgreement.single (first := 3511) modelrow_agrees_3512
private def atomic_3513 : CountSegment 1 10 13120 3512 3513 1670 1671 := CountSegment.single (first := 3512) (initial := 1670) row_count_3513 (by decide)
private theorem atomic_3513_agreement : RowAgreement 3512 3513 := RowAgreement.single (first := 3512) modelrow_agrees_3513
private def atomic_3514 : CountSegment 1 10 13120 3513 3514 1671 1671 := CountSegment.single (first := 3513) (initial := 1671) row_count_3514 (by decide)
private theorem atomic_3514_agreement : RowAgreement 3513 3514 := RowAgreement.single (first := 3513) modelrow_agrees_3514
private def atomic_3515 : CountSegment 1 10 13120 3514 3515 1671 1671 := CountSegment.single (first := 3514) (initial := 1671) row_count_3515 (by decide)
private theorem atomic_3515_agreement : RowAgreement 3514 3515 := RowAgreement.single (first := 3514) modelrow_agrees_3515
private def atomic_3516 : CountSegment 1 10 13120 3515 3516 1671 1671 := CountSegment.single (first := 3515) (initial := 1671) row_count_3516 (by decide)
private theorem atomic_3516_agreement : RowAgreement 3515 3516 := RowAgreement.single (first := 3515) modelrow_agrees_3516
private def atomic_3517 : CountSegment 1 10 13120 3516 3517 1671 1672 := CountSegment.single (first := 3516) (initial := 1671) row_count_3517 (by decide)
private theorem atomic_3517_agreement : RowAgreement 3516 3517 := RowAgreement.single (first := 3516) modelrow_agrees_3517
private def atomic_3518 : CountSegment 1 10 13120 3517 3518 1672 1672 := CountSegment.single (first := 3517) (initial := 1672) row_count_3518 (by decide)
private theorem atomic_3518_agreement : RowAgreement 3517 3518 := RowAgreement.single (first := 3517) modelrow_agrees_3518
private def atomic_3519 : CountSegment 1 10 13120 3518 3519 1672 1672 := CountSegment.single (first := 3518) (initial := 1672) row_count_3519 (by decide)
private theorem atomic_3519_agreement : RowAgreement 3518 3519 := RowAgreement.single (first := 3518) modelrow_agrees_3519
private def atomic_3520 : CountSegment 1 10 13120 3519 3520 1672 1672 := CountSegment.single (first := 3519) (initial := 1672) row_count_3520 (by decide)
private theorem atomic_3520_agreement : RowAgreement 3519 3520 := RowAgreement.single (first := 3519) modelrow_agrees_3520
private def atomic_3521 : CountSegment 1 10 13120 3520 3521 1672 1672 := CountSegment.single (first := 3520) (initial := 1672) row_count_3521 (by decide)
private theorem atomic_3521_agreement : RowAgreement 3520 3521 := RowAgreement.single (first := 3520) modelrow_agrees_3521
private def atomic_3522 : CountSegment 1 10 13120 3521 3522 1672 1673 := CountSegment.single (first := 3521) (initial := 1672) row_count_3522 (by decide)
private theorem atomic_3522_agreement : RowAgreement 3521 3522 := RowAgreement.single (first := 3521) modelrow_agrees_3522
private def atomic_3523 : CountSegment 1 10 13120 3522 3523 1673 1673 := CountSegment.single (first := 3522) (initial := 1673) row_count_3523 (by decide)
private theorem atomic_3523_agreement : RowAgreement 3522 3523 := RowAgreement.single (first := 3522) modelrow_agrees_3523
private def atomic_3524 : CountSegment 1 10 13120 3523 3524 1673 1674 := CountSegment.single (first := 3523) (initial := 1673) row_count_3524 (by decide)
private theorem atomic_3524_agreement : RowAgreement 3523 3524 := RowAgreement.single (first := 3523) modelrow_agrees_3524
private def atomic_3525 : CountSegment 1 10 13120 3524 3525 1674 1675 := CountSegment.single (first := 3524) (initial := 1674) row_count_3525 (by decide)
private theorem atomic_3525_agreement : RowAgreement 3524 3525 := RowAgreement.single (first := 3524) modelrow_agrees_3525
private def atomic_3526 : CountSegment 1 10 13120 3525 3526 1675 1675 := CountSegment.single (first := 3525) (initial := 1675) row_count_3526 (by decide)
private theorem atomic_3526_agreement : RowAgreement 3525 3526 := RowAgreement.single (first := 3525) modelrow_agrees_3526
private def atomic_3527 : CountSegment 1 10 13120 3526 3527 1675 1675 := CountSegment.single (first := 3526) (initial := 1675) row_count_3527 (by decide)
private theorem atomic_3527_agreement : RowAgreement 3526 3527 := RowAgreement.single (first := 3526) modelrow_agrees_3527
private def atomic_3528 : CountSegment 1 10 13120 3527 3528 1675 1675 := CountSegment.single (first := 3527) (initial := 1675) row_count_3528 (by decide)
private theorem atomic_3528_agreement : RowAgreement 3527 3528 := RowAgreement.single (first := 3527) modelrow_agrees_3528
private def atomic_3529 : CountSegment 1 10 13120 3528 3529 1675 1675 := CountSegment.single (first := 3528) (initial := 1675) row_count_3529 (by decide)
private theorem atomic_3529_agreement : RowAgreement 3528 3529 := RowAgreement.single (first := 3528) modelrow_agrees_3529
private def atomic_3530 : CountSegment 1 10 13120 3529 3530 1675 1676 := CountSegment.single (first := 3529) (initial := 1675) row_count_3530 (by decide)
private theorem atomic_3530_agreement : RowAgreement 3529 3530 := RowAgreement.single (first := 3529) modelrow_agrees_3530
private def atomic_3531 : CountSegment 1 10 13120 3530 3531 1676 1676 := CountSegment.single (first := 3530) (initial := 1676) row_count_3531 (by decide)
private theorem atomic_3531_agreement : RowAgreement 3530 3531 := RowAgreement.single (first := 3530) modelrow_agrees_3531
private def atomic_3532 : CountSegment 1 10 13120 3531 3532 1676 1676 := CountSegment.single (first := 3531) (initial := 1676) row_count_3532 (by decide)
private theorem atomic_3532_agreement : RowAgreement 3531 3532 := RowAgreement.single (first := 3531) modelrow_agrees_3532
private def atomic_3533 : CountSegment 1 10 13120 3532 3533 1676 1677 := CountSegment.single (first := 3532) (initial := 1676) row_count_3533 (by decide)
private theorem atomic_3533_agreement : RowAgreement 3532 3533 := RowAgreement.single (first := 3532) modelrow_agrees_3533
private def atomic_3534 : CountSegment 1 10 13120 3533 3534 1677 1677 := CountSegment.single (first := 3533) (initial := 1677) row_count_3534 (by decide)
private theorem atomic_3534_agreement : RowAgreement 3533 3534 := RowAgreement.single (first := 3533) modelrow_agrees_3534
private def atomic_3535 : CountSegment 1 10 13120 3534 3535 1677 1677 := CountSegment.single (first := 3534) (initial := 1677) row_count_3535 (by decide)
private theorem atomic_3535_agreement : RowAgreement 3534 3535 := RowAgreement.single (first := 3534) modelrow_agrees_3535
private def atomic_3536 : CountSegment 1 10 13120 3535 3536 1677 1677 := CountSegment.single (first := 3535) (initial := 1677) row_count_3536 (by decide)
private theorem atomic_3536_agreement : RowAgreement 3535 3536 := RowAgreement.single (first := 3535) modelrow_agrees_3536
private def atomic_3537 : CountSegment 1 10 13120 3536 3537 1677 1677 := CountSegment.single (first := 3536) (initial := 1677) row_count_3537 (by decide)
private theorem atomic_3537_agreement : RowAgreement 3536 3537 := RowAgreement.single (first := 3536) modelrow_agrees_3537
private def atomic_3538 : CountSegment 1 10 13120 3537 3538 1677 1677 := CountSegment.single (first := 3537) (initial := 1677) row_count_3538 (by decide)
private theorem atomic_3538_agreement : RowAgreement 3537 3538 := RowAgreement.single (first := 3537) modelrow_agrees_3538
private def atomic_3539 : CountSegment 1 10 13120 3538 3539 1677 1677 := CountSegment.single (first := 3538) (initial := 1677) row_count_3539 (by decide)
private theorem atomic_3539_agreement : RowAgreement 3538 3539 := RowAgreement.single (first := 3538) modelrow_agrees_3539
private def atomic_3540 : CountSegment 1 10 13120 3539 3540 1677 1678 := CountSegment.single (first := 3539) (initial := 1677) row_count_3540 (by decide)
private theorem atomic_3540_agreement : RowAgreement 3539 3540 := RowAgreement.single (first := 3539) modelrow_agrees_3540
private def atomic_3541 : CountSegment 1 10 13120 3540 3541 1678 1678 := CountSegment.single (first := 3540) (initial := 1678) row_count_3541 (by decide)
private theorem atomic_3541_agreement : RowAgreement 3540 3541 := RowAgreement.single (first := 3540) modelrow_agrees_3541
private def atomic_3542 : CountSegment 1 10 13120 3541 3542 1678 1679 := CountSegment.single (first := 3541) (initial := 1678) row_count_3542 (by decide)
private theorem atomic_3542_agreement : RowAgreement 3541 3542 := RowAgreement.single (first := 3541) modelrow_agrees_3542
private def atomic_3543 : CountSegment 1 10 13120 3542 3543 1679 1679 := CountSegment.single (first := 3542) (initial := 1679) row_count_3543 (by decide)
private theorem atomic_3543_agreement : RowAgreement 3542 3543 := RowAgreement.single (first := 3542) modelrow_agrees_3543
private def atomic_3544 : CountSegment 1 10 13120 3543 3544 1679 1679 := CountSegment.single (first := 3543) (initial := 1679) row_count_3544 (by decide)
private theorem atomic_3544_agreement : RowAgreement 3543 3544 := RowAgreement.single (first := 3543) modelrow_agrees_3544
private def atomic_3545 : CountSegment 1 10 13120 3544 3545 1679 1679 := CountSegment.single (first := 3544) (initial := 1679) row_count_3545 (by decide)
private theorem atomic_3545_agreement : RowAgreement 3544 3545 := RowAgreement.single (first := 3544) modelrow_agrees_3545
private def atomic_3546 : CountSegment 1 10 13120 3545 3546 1679 1680 := CountSegment.single (first := 3545) (initial := 1679) row_count_3546 (by decide)
private theorem atomic_3546_agreement : RowAgreement 3545 3546 := RowAgreement.single (first := 3545) modelrow_agrees_3546
private def atomic_3547 : CountSegment 1 10 13120 3546 3547 1680 1680 := CountSegment.single (first := 3546) (initial := 1680) row_count_3547 (by decide)
private theorem atomic_3547_agreement : RowAgreement 3546 3547 := RowAgreement.single (first := 3546) modelrow_agrees_3547
private def atomic_3548 : CountSegment 1 10 13120 3547 3548 1680 1680 := CountSegment.single (first := 3547) (initial := 1680) row_count_3548 (by decide)
private theorem atomic_3548_agreement : RowAgreement 3547 3548 := RowAgreement.single (first := 3547) modelrow_agrees_3548
private def atomic_3549 : CountSegment 1 10 13120 3548 3549 1680 1680 := CountSegment.single (first := 3548) (initial := 1680) row_count_3549 (by decide)
private theorem atomic_3549_agreement : RowAgreement 3548 3549 := RowAgreement.single (first := 3548) modelrow_agrees_3549
private def atomic_3550 : CountSegment 1 10 13120 3549 3550 1680 1681 := CountSegment.single (first := 3549) (initial := 1680) row_count_3550 (by decide)
private theorem atomic_3550_agreement : RowAgreement 3549 3550 := RowAgreement.single (first := 3549) modelrow_agrees_3550
private def atomic_3551 : CountSegment 1 10 13120 3550 3551 1681 1681 := CountSegment.single (first := 3550) (initial := 1681) row_count_3551 (by decide)
private theorem atomic_3551_agreement : RowAgreement 3550 3551 := RowAgreement.single (first := 3550) modelrow_agrees_3551
private def atomic_3552 : CountSegment 1 10 13120 3551 3552 1681 1681 := CountSegment.single (first := 3551) (initial := 1681) row_count_3552 (by decide)
private theorem atomic_3552_agreement : RowAgreement 3551 3552 := RowAgreement.single (first := 3551) modelrow_agrees_3552
private def atomic_3553 : CountSegment 1 10 13120 3552 3553 1681 1682 := CountSegment.single (first := 3552) (initial := 1681) row_count_3553 (by decide)
private theorem atomic_3553_agreement : RowAgreement 3552 3553 := RowAgreement.single (first := 3552) modelrow_agrees_3553
private def atomic_3554 : CountSegment 1 10 13120 3553 3554 1682 1682 := CountSegment.single (first := 3553) (initial := 1682) row_count_3554 (by decide)
private theorem atomic_3554_agreement : RowAgreement 3553 3554 := RowAgreement.single (first := 3553) modelrow_agrees_3554
private def atomic_3555 : CountSegment 1 10 13120 3554 3555 1682 1683 := CountSegment.single (first := 3554) (initial := 1682) row_count_3555 (by decide)
private theorem atomic_3555_agreement : RowAgreement 3554 3555 := RowAgreement.single (first := 3554) modelrow_agrees_3555
private def atomic_3556 : CountSegment 1 10 13120 3555 3556 1683 1684 := CountSegment.single (first := 3555) (initial := 1683) row_count_3556 (by decide)
private theorem atomic_3556_agreement : RowAgreement 3555 3556 := RowAgreement.single (first := 3555) modelrow_agrees_3556
private def atomic_3557 : CountSegment 1 10 13120 3556 3557 1684 1685 := CountSegment.single (first := 3556) (initial := 1684) row_count_3557 (by decide)
private theorem atomic_3557_agreement : RowAgreement 3556 3557 := RowAgreement.single (first := 3556) modelrow_agrees_3557
private def atomic_3558 : CountSegment 1 10 13120 3557 3558 1685 1686 := CountSegment.single (first := 3557) (initial := 1685) row_count_3558 (by decide)
private theorem atomic_3558_agreement : RowAgreement 3557 3558 := RowAgreement.single (first := 3557) modelrow_agrees_3558
private def atomic_3559 : CountSegment 1 10 13120 3558 3559 1686 1687 := CountSegment.single (first := 3558) (initial := 1686) row_count_3559 (by decide)
private theorem atomic_3559_agreement : RowAgreement 3558 3559 := RowAgreement.single (first := 3558) modelrow_agrees_3559
private def atomic_3560 : CountSegment 1 10 13120 3559 3560 1687 1687 := CountSegment.single (first := 3559) (initial := 1687) row_count_3560 (by decide)
private theorem atomic_3560_agreement : RowAgreement 3559 3560 := RowAgreement.single (first := 3559) modelrow_agrees_3560
private def atomic_3561 : CountSegment 1 10 13120 3560 3561 1687 1688 := CountSegment.single (first := 3560) (initial := 1687) row_count_3561 (by decide)
private theorem atomic_3561_agreement : RowAgreement 3560 3561 := RowAgreement.single (first := 3560) modelrow_agrees_3561
private def atomic_3562 : CountSegment 1 10 13120 3561 3562 1688 1688 := CountSegment.single (first := 3561) (initial := 1688) row_count_3562 (by decide)
private theorem atomic_3562_agreement : RowAgreement 3561 3562 := RowAgreement.single (first := 3561) modelrow_agrees_3562
private def atomic_3563 : CountSegment 1 10 13120 3562 3563 1688 1688 := CountSegment.single (first := 3562) (initial := 1688) row_count_3563 (by decide)
private theorem atomic_3563_agreement : RowAgreement 3562 3563 := RowAgreement.single (first := 3562) modelrow_agrees_3563
private def atomic_3564 : CountSegment 1 10 13120 3563 3564 1688 1688 := CountSegment.single (first := 3563) (initial := 1688) row_count_3564 (by decide)
private theorem atomic_3564_agreement : RowAgreement 3563 3564 := RowAgreement.single (first := 3563) modelrow_agrees_3564
private def atomic_3565 : CountSegment 1 10 13120 3564 3565 1688 1689 := CountSegment.single (first := 3564) (initial := 1688) row_count_3565 (by decide)
private theorem atomic_3565_agreement : RowAgreement 3564 3565 := RowAgreement.single (first := 3564) modelrow_agrees_3565
private def atomic_3566 : CountSegment 1 10 13120 3565 3566 1689 1689 := CountSegment.single (first := 3565) (initial := 1689) row_count_3566 (by decide)
private theorem atomic_3566_agreement : RowAgreement 3565 3566 := RowAgreement.single (first := 3565) modelrow_agrees_3566
private def atomic_3567 : CountSegment 1 10 13120 3566 3567 1689 1689 := CountSegment.single (first := 3566) (initial := 1689) row_count_3567 (by decide)
private theorem atomic_3567_agreement : RowAgreement 3566 3567 := RowAgreement.single (first := 3566) modelrow_agrees_3567
private def atomic_3568 : CountSegment 1 10 13120 3567 3568 1689 1689 := CountSegment.single (first := 3567) (initial := 1689) row_count_3568 (by decide)
private theorem atomic_3568_agreement : RowAgreement 3567 3568 := RowAgreement.single (first := 3567) modelrow_agrees_3568
private def atomic_3569 : CountSegment 1 10 13120 3568 3569 1689 1689 := CountSegment.single (first := 3568) (initial := 1689) row_count_3569 (by decide)
private theorem atomic_3569_agreement : RowAgreement 3568 3569 := RowAgreement.single (first := 3568) modelrow_agrees_3569
private def atomic_3570 : CountSegment 1 10 13120 3569 3570 1689 1689 := CountSegment.single (first := 3569) (initial := 1689) row_count_3570 (by decide)
private theorem atomic_3570_agreement : RowAgreement 3569 3570 := RowAgreement.single (first := 3569) modelrow_agrees_3570
private def atomic_3571 : CountSegment 1 10 13120 3570 3571 1689 1689 := CountSegment.single (first := 3570) (initial := 1689) row_count_3571 (by decide)
private theorem atomic_3571_agreement : RowAgreement 3570 3571 := RowAgreement.single (first := 3570) modelrow_agrees_3571
private def atomic_3572 : CountSegment 1 10 13120 3571 3572 1689 1690 := CountSegment.single (first := 3571) (initial := 1689) row_count_3572 (by decide)
private theorem atomic_3572_agreement : RowAgreement 3571 3572 := RowAgreement.single (first := 3571) modelrow_agrees_3572
private def atomic_3573 : CountSegment 1 10 13120 3572 3573 1690 1690 := CountSegment.single (first := 3572) (initial := 1690) row_count_3573 (by decide)
private theorem atomic_3573_agreement : RowAgreement 3572 3573 := RowAgreement.single (first := 3572) modelrow_agrees_3573
private def atomic_3574 : CountSegment 1 10 13120 3573 3574 1690 1690 := CountSegment.single (first := 3573) (initial := 1690) row_count_3574 (by decide)
private theorem atomic_3574_agreement : RowAgreement 3573 3574 := RowAgreement.single (first := 3573) modelrow_agrees_3574
private def atomic_3575 : CountSegment 1 10 13120 3574 3575 1690 1691 := CountSegment.single (first := 3574) (initial := 1690) row_count_3575 (by decide)
private theorem atomic_3575_agreement : RowAgreement 3574 3575 := RowAgreement.single (first := 3574) modelrow_agrees_3575
private def atomic_3576 : CountSegment 1 10 13120 3575 3576 1691 1692 := CountSegment.single (first := 3575) (initial := 1691) row_count_3576 (by decide)
private theorem atomic_3576_agreement : RowAgreement 3575 3576 := RowAgreement.single (first := 3575) modelrow_agrees_3576
private def atomic_3577 : CountSegment 1 10 13120 3576 3577 1692 1692 := CountSegment.single (first := 3576) (initial := 1692) row_count_3577 (by decide)
private theorem atomic_3577_agreement : RowAgreement 3576 3577 := RowAgreement.single (first := 3576) modelrow_agrees_3577
private def atomic_3578 : CountSegment 1 10 13120 3577 3578 1692 1692 := CountSegment.single (first := 3577) (initial := 1692) row_count_3578 (by decide)
private theorem atomic_3578_agreement : RowAgreement 3577 3578 := RowAgreement.single (first := 3577) modelrow_agrees_3578
private def atomic_3579 : CountSegment 1 10 13120 3578 3579 1692 1693 := CountSegment.single (first := 3578) (initial := 1692) row_count_3579 (by decide)
private theorem atomic_3579_agreement : RowAgreement 3578 3579 := RowAgreement.single (first := 3578) modelrow_agrees_3579
private def atomic_3580 : CountSegment 1 10 13120 3579 3580 1693 1695 := CountSegment.single (first := 3579) (initial := 1693) row_count_3580 (by decide)
private theorem atomic_3580_agreement : RowAgreement 3579 3580 := RowAgreement.single (first := 3579) modelrow_agrees_3580
private def atomic_3581 : CountSegment 1 10 13120 3580 3581 1695 1695 := CountSegment.single (first := 3580) (initial := 1695) row_count_3581 (by decide)
private theorem atomic_3581_agreement : RowAgreement 3580 3581 := RowAgreement.single (first := 3580) modelrow_agrees_3581
private def atomic_3582 : CountSegment 1 10 13120 3581 3582 1695 1697 := CountSegment.single (first := 3581) (initial := 1695) row_count_3582 (by decide)
private theorem atomic_3582_agreement : RowAgreement 3581 3582 := RowAgreement.single (first := 3581) modelrow_agrees_3582
private def atomic_3583 : CountSegment 1 10 13120 3582 3583 1697 1697 := CountSegment.single (first := 3582) (initial := 1697) row_count_3583 (by decide)
private theorem atomic_3583_agreement : RowAgreement 3582 3583 := RowAgreement.single (first := 3582) modelrow_agrees_3583
private def atomic_3584 : CountSegment 1 10 13120 3583 3584 1697 1697 := CountSegment.single (first := 3583) (initial := 1697) row_count_3584 (by decide)
private theorem atomic_3584_agreement : RowAgreement 3583 3584 := RowAgreement.single (first := 3583) modelrow_agrees_3584
private def within_027_000 : CountSegment 1 10 13120 3456 3458 1644 1646 := CountSegment.join atomic_3457 atomic_3458
private theorem within_027_000_agreement : RowAgreement 3456 3458 := RowAgreement.join atomic_3457_agreement atomic_3458_agreement
private def within_027_001 : CountSegment 1 10 13120 3458 3460 1646 1647 := CountSegment.join atomic_3459 atomic_3460
private theorem within_027_001_agreement : RowAgreement 3458 3460 := RowAgreement.join atomic_3459_agreement atomic_3460_agreement
private def within_027_002 : CountSegment 1 10 13120 3456 3460 1644 1647 := CountSegment.join within_027_000 within_027_001
private theorem within_027_002_agreement : RowAgreement 3456 3460 := RowAgreement.join within_027_000_agreement within_027_001_agreement
private def within_027_003 : CountSegment 1 10 13120 3460 3462 1647 1648 := CountSegment.join atomic_3461 atomic_3462
private theorem within_027_003_agreement : RowAgreement 3460 3462 := RowAgreement.join atomic_3461_agreement atomic_3462_agreement
private def within_027_004 : CountSegment 1 10 13120 3462 3464 1648 1649 := CountSegment.join atomic_3463 atomic_3464
private theorem within_027_004_agreement : RowAgreement 3462 3464 := RowAgreement.join atomic_3463_agreement atomic_3464_agreement
private def within_027_005 : CountSegment 1 10 13120 3460 3464 1647 1649 := CountSegment.join within_027_003 within_027_004
private theorem within_027_005_agreement : RowAgreement 3460 3464 := RowAgreement.join within_027_003_agreement within_027_004_agreement
private def within_027_006 : CountSegment 1 10 13120 3456 3464 1644 1649 := CountSegment.join within_027_002 within_027_005
private theorem within_027_006_agreement : RowAgreement 3456 3464 := RowAgreement.join within_027_002_agreement within_027_005_agreement
private def within_027_007 : CountSegment 1 10 13120 3464 3466 1649 1650 := CountSegment.join atomic_3465 atomic_3466
private theorem within_027_007_agreement : RowAgreement 3464 3466 := RowAgreement.join atomic_3465_agreement atomic_3466_agreement
private def within_027_008 : CountSegment 1 10 13120 3466 3468 1650 1650 := CountSegment.join atomic_3467 atomic_3468
private theorem within_027_008_agreement : RowAgreement 3466 3468 := RowAgreement.join atomic_3467_agreement atomic_3468_agreement
private def within_027_009 : CountSegment 1 10 13120 3464 3468 1649 1650 := CountSegment.join within_027_007 within_027_008
private theorem within_027_009_agreement : RowAgreement 3464 3468 := RowAgreement.join within_027_007_agreement within_027_008_agreement
private def within_027_010 : CountSegment 1 10 13120 3468 3470 1650 1650 := CountSegment.join atomic_3469 atomic_3470
private theorem within_027_010_agreement : RowAgreement 3468 3470 := RowAgreement.join atomic_3469_agreement atomic_3470_agreement
private def within_027_011 : CountSegment 1 10 13120 3470 3472 1650 1652 := CountSegment.join atomic_3471 atomic_3472
private theorem within_027_011_agreement : RowAgreement 3470 3472 := RowAgreement.join atomic_3471_agreement atomic_3472_agreement
private def within_027_012 : CountSegment 1 10 13120 3468 3472 1650 1652 := CountSegment.join within_027_010 within_027_011
private theorem within_027_012_agreement : RowAgreement 3468 3472 := RowAgreement.join within_027_010_agreement within_027_011_agreement
private def within_027_013 : CountSegment 1 10 13120 3464 3472 1649 1652 := CountSegment.join within_027_009 within_027_012
private theorem within_027_013_agreement : RowAgreement 3464 3472 := RowAgreement.join within_027_009_agreement within_027_012_agreement
private def within_027_014 : CountSegment 1 10 13120 3456 3472 1644 1652 := CountSegment.join within_027_006 within_027_013
private theorem within_027_014_agreement : RowAgreement 3456 3472 := RowAgreement.join within_027_006_agreement within_027_013_agreement
private def within_027_015 : CountSegment 1 10 13120 3472 3474 1652 1652 := CountSegment.join atomic_3473 atomic_3474
private theorem within_027_015_agreement : RowAgreement 3472 3474 := RowAgreement.join atomic_3473_agreement atomic_3474_agreement
private def within_027_016 : CountSegment 1 10 13120 3474 3476 1652 1652 := CountSegment.join atomic_3475 atomic_3476
private theorem within_027_016_agreement : RowAgreement 3474 3476 := RowAgreement.join atomic_3475_agreement atomic_3476_agreement
private def within_027_017 : CountSegment 1 10 13120 3472 3476 1652 1652 := CountSegment.join within_027_015 within_027_016
private theorem within_027_017_agreement : RowAgreement 3472 3476 := RowAgreement.join within_027_015_agreement within_027_016_agreement
private def within_027_018 : CountSegment 1 10 13120 3476 3478 1652 1655 := CountSegment.join atomic_3477 atomic_3478
private theorem within_027_018_agreement : RowAgreement 3476 3478 := RowAgreement.join atomic_3477_agreement atomic_3478_agreement
private def within_027_019 : CountSegment 1 10 13120 3478 3480 1655 1655 := CountSegment.join atomic_3479 atomic_3480
private theorem within_027_019_agreement : RowAgreement 3478 3480 := RowAgreement.join atomic_3479_agreement atomic_3480_agreement
private def within_027_020 : CountSegment 1 10 13120 3476 3480 1652 1655 := CountSegment.join within_027_018 within_027_019
private theorem within_027_020_agreement : RowAgreement 3476 3480 := RowAgreement.join within_027_018_agreement within_027_019_agreement
private def within_027_021 : CountSegment 1 10 13120 3472 3480 1652 1655 := CountSegment.join within_027_017 within_027_020
private theorem within_027_021_agreement : RowAgreement 3472 3480 := RowAgreement.join within_027_017_agreement within_027_020_agreement
private def within_027_022 : CountSegment 1 10 13120 3480 3482 1655 1656 := CountSegment.join atomic_3481 atomic_3482
private theorem within_027_022_agreement : RowAgreement 3480 3482 := RowAgreement.join atomic_3481_agreement atomic_3482_agreement
private def within_027_023 : CountSegment 1 10 13120 3482 3484 1656 1657 := CountSegment.join atomic_3483 atomic_3484
private theorem within_027_023_agreement : RowAgreement 3482 3484 := RowAgreement.join atomic_3483_agreement atomic_3484_agreement
private def within_027_024 : CountSegment 1 10 13120 3480 3484 1655 1657 := CountSegment.join within_027_022 within_027_023
private theorem within_027_024_agreement : RowAgreement 3480 3484 := RowAgreement.join within_027_022_agreement within_027_023_agreement
private def within_027_025 : CountSegment 1 10 13120 3484 3486 1657 1658 := CountSegment.join atomic_3485 atomic_3486
private theorem within_027_025_agreement : RowAgreement 3484 3486 := RowAgreement.join atomic_3485_agreement atomic_3486_agreement
private def within_027_026 : CountSegment 1 10 13120 3486 3488 1658 1658 := CountSegment.join atomic_3487 atomic_3488
private theorem within_027_026_agreement : RowAgreement 3486 3488 := RowAgreement.join atomic_3487_agreement atomic_3488_agreement
private def within_027_027 : CountSegment 1 10 13120 3484 3488 1657 1658 := CountSegment.join within_027_025 within_027_026
private theorem within_027_027_agreement : RowAgreement 3484 3488 := RowAgreement.join within_027_025_agreement within_027_026_agreement
private def within_027_028 : CountSegment 1 10 13120 3480 3488 1655 1658 := CountSegment.join within_027_024 within_027_027
private theorem within_027_028_agreement : RowAgreement 3480 3488 := RowAgreement.join within_027_024_agreement within_027_027_agreement
private def within_027_029 : CountSegment 1 10 13120 3472 3488 1652 1658 := CountSegment.join within_027_021 within_027_028
private theorem within_027_029_agreement : RowAgreement 3472 3488 := RowAgreement.join within_027_021_agreement within_027_028_agreement
private def within_027_030 : CountSegment 1 10 13120 3456 3488 1644 1658 := CountSegment.join within_027_014 within_027_029
private theorem within_027_030_agreement : RowAgreement 3456 3488 := RowAgreement.join within_027_014_agreement within_027_029_agreement
private def within_027_031 : CountSegment 1 10 13120 3488 3490 1658 1659 := CountSegment.join atomic_3489 atomic_3490
private theorem within_027_031_agreement : RowAgreement 3488 3490 := RowAgreement.join atomic_3489_agreement atomic_3490_agreement
private def within_027_032 : CountSegment 1 10 13120 3490 3492 1659 1660 := CountSegment.join atomic_3491 atomic_3492
private theorem within_027_032_agreement : RowAgreement 3490 3492 := RowAgreement.join atomic_3491_agreement atomic_3492_agreement
private def within_027_033 : CountSegment 1 10 13120 3488 3492 1658 1660 := CountSegment.join within_027_031 within_027_032
private theorem within_027_033_agreement : RowAgreement 3488 3492 := RowAgreement.join within_027_031_agreement within_027_032_agreement
private def within_027_034 : CountSegment 1 10 13120 3492 3494 1660 1662 := CountSegment.join atomic_3493 atomic_3494
private theorem within_027_034_agreement : RowAgreement 3492 3494 := RowAgreement.join atomic_3493_agreement atomic_3494_agreement
private def within_027_035 : CountSegment 1 10 13120 3494 3496 1662 1663 := CountSegment.join atomic_3495 atomic_3496
private theorem within_027_035_agreement : RowAgreement 3494 3496 := RowAgreement.join atomic_3495_agreement atomic_3496_agreement
private def within_027_036 : CountSegment 1 10 13120 3492 3496 1660 1663 := CountSegment.join within_027_034 within_027_035
private theorem within_027_036_agreement : RowAgreement 3492 3496 := RowAgreement.join within_027_034_agreement within_027_035_agreement
private def within_027_037 : CountSegment 1 10 13120 3488 3496 1658 1663 := CountSegment.join within_027_033 within_027_036
private theorem within_027_037_agreement : RowAgreement 3488 3496 := RowAgreement.join within_027_033_agreement within_027_036_agreement
private def within_027_038 : CountSegment 1 10 13120 3496 3498 1663 1665 := CountSegment.join atomic_3497 atomic_3498
private theorem within_027_038_agreement : RowAgreement 3496 3498 := RowAgreement.join atomic_3497_agreement atomic_3498_agreement
private def within_027_039 : CountSegment 1 10 13120 3498 3500 1665 1665 := CountSegment.join atomic_3499 atomic_3500
private theorem within_027_039_agreement : RowAgreement 3498 3500 := RowAgreement.join atomic_3499_agreement atomic_3500_agreement
private def within_027_040 : CountSegment 1 10 13120 3496 3500 1663 1665 := CountSegment.join within_027_038 within_027_039
private theorem within_027_040_agreement : RowAgreement 3496 3500 := RowAgreement.join within_027_038_agreement within_027_039_agreement
private def within_027_041 : CountSegment 1 10 13120 3500 3502 1665 1665 := CountSegment.join atomic_3501 atomic_3502
private theorem within_027_041_agreement : RowAgreement 3500 3502 := RowAgreement.join atomic_3501_agreement atomic_3502_agreement
private def within_027_042 : CountSegment 1 10 13120 3502 3504 1665 1665 := CountSegment.join atomic_3503 atomic_3504
private theorem within_027_042_agreement : RowAgreement 3502 3504 := RowAgreement.join atomic_3503_agreement atomic_3504_agreement
private def within_027_043 : CountSegment 1 10 13120 3500 3504 1665 1665 := CountSegment.join within_027_041 within_027_042
private theorem within_027_043_agreement : RowAgreement 3500 3504 := RowAgreement.join within_027_041_agreement within_027_042_agreement
private def within_027_044 : CountSegment 1 10 13120 3496 3504 1663 1665 := CountSegment.join within_027_040 within_027_043
private theorem within_027_044_agreement : RowAgreement 3496 3504 := RowAgreement.join within_027_040_agreement within_027_043_agreement
private def within_027_045 : CountSegment 1 10 13120 3488 3504 1658 1665 := CountSegment.join within_027_037 within_027_044
private theorem within_027_045_agreement : RowAgreement 3488 3504 := RowAgreement.join within_027_037_agreement within_027_044_agreement
private def within_027_046 : CountSegment 1 10 13120 3504 3506 1665 1666 := CountSegment.join atomic_3505 atomic_3506
private theorem within_027_046_agreement : RowAgreement 3504 3506 := RowAgreement.join atomic_3505_agreement atomic_3506_agreement
private def within_027_047 : CountSegment 1 10 13120 3506 3508 1666 1670 := CountSegment.join atomic_3507 atomic_3508
private theorem within_027_047_agreement : RowAgreement 3506 3508 := RowAgreement.join atomic_3507_agreement atomic_3508_agreement
private def within_027_048 : CountSegment 1 10 13120 3504 3508 1665 1670 := CountSegment.join within_027_046 within_027_047
private theorem within_027_048_agreement : RowAgreement 3504 3508 := RowAgreement.join within_027_046_agreement within_027_047_agreement
private def within_027_049 : CountSegment 1 10 13120 3508 3510 1670 1670 := CountSegment.join atomic_3509 atomic_3510
private theorem within_027_049_agreement : RowAgreement 3508 3510 := RowAgreement.join atomic_3509_agreement atomic_3510_agreement
private def within_027_050 : CountSegment 1 10 13120 3510 3512 1670 1670 := CountSegment.join atomic_3511 atomic_3512
private theorem within_027_050_agreement : RowAgreement 3510 3512 := RowAgreement.join atomic_3511_agreement atomic_3512_agreement
private def within_027_051 : CountSegment 1 10 13120 3508 3512 1670 1670 := CountSegment.join within_027_049 within_027_050
private theorem within_027_051_agreement : RowAgreement 3508 3512 := RowAgreement.join within_027_049_agreement within_027_050_agreement
private def within_027_052 : CountSegment 1 10 13120 3504 3512 1665 1670 := CountSegment.join within_027_048 within_027_051
private theorem within_027_052_agreement : RowAgreement 3504 3512 := RowAgreement.join within_027_048_agreement within_027_051_agreement
private def within_027_053 : CountSegment 1 10 13120 3512 3514 1670 1671 := CountSegment.join atomic_3513 atomic_3514
private theorem within_027_053_agreement : RowAgreement 3512 3514 := RowAgreement.join atomic_3513_agreement atomic_3514_agreement
private def within_027_054 : CountSegment 1 10 13120 3514 3516 1671 1671 := CountSegment.join atomic_3515 atomic_3516
private theorem within_027_054_agreement : RowAgreement 3514 3516 := RowAgreement.join atomic_3515_agreement atomic_3516_agreement
private def within_027_055 : CountSegment 1 10 13120 3512 3516 1670 1671 := CountSegment.join within_027_053 within_027_054
private theorem within_027_055_agreement : RowAgreement 3512 3516 := RowAgreement.join within_027_053_agreement within_027_054_agreement
private def within_027_056 : CountSegment 1 10 13120 3516 3518 1671 1672 := CountSegment.join atomic_3517 atomic_3518
private theorem within_027_056_agreement : RowAgreement 3516 3518 := RowAgreement.join atomic_3517_agreement atomic_3518_agreement
private def within_027_057 : CountSegment 1 10 13120 3518 3520 1672 1672 := CountSegment.join atomic_3519 atomic_3520
private theorem within_027_057_agreement : RowAgreement 3518 3520 := RowAgreement.join atomic_3519_agreement atomic_3520_agreement
private def within_027_058 : CountSegment 1 10 13120 3516 3520 1671 1672 := CountSegment.join within_027_056 within_027_057
private theorem within_027_058_agreement : RowAgreement 3516 3520 := RowAgreement.join within_027_056_agreement within_027_057_agreement
private def within_027_059 : CountSegment 1 10 13120 3512 3520 1670 1672 := CountSegment.join within_027_055 within_027_058
private theorem within_027_059_agreement : RowAgreement 3512 3520 := RowAgreement.join within_027_055_agreement within_027_058_agreement
private def within_027_060 : CountSegment 1 10 13120 3504 3520 1665 1672 := CountSegment.join within_027_052 within_027_059
private theorem within_027_060_agreement : RowAgreement 3504 3520 := RowAgreement.join within_027_052_agreement within_027_059_agreement
private def within_027_061 : CountSegment 1 10 13120 3488 3520 1658 1672 := CountSegment.join within_027_045 within_027_060
private theorem within_027_061_agreement : RowAgreement 3488 3520 := RowAgreement.join within_027_045_agreement within_027_060_agreement
private def within_027_062 : CountSegment 1 10 13120 3456 3520 1644 1672 := CountSegment.join within_027_030 within_027_061
private theorem within_027_062_agreement : RowAgreement 3456 3520 := RowAgreement.join within_027_030_agreement within_027_061_agreement
private def within_027_063 : CountSegment 1 10 13120 3520 3522 1672 1673 := CountSegment.join atomic_3521 atomic_3522
private theorem within_027_063_agreement : RowAgreement 3520 3522 := RowAgreement.join atomic_3521_agreement atomic_3522_agreement
private def within_027_064 : CountSegment 1 10 13120 3522 3524 1673 1674 := CountSegment.join atomic_3523 atomic_3524
private theorem within_027_064_agreement : RowAgreement 3522 3524 := RowAgreement.join atomic_3523_agreement atomic_3524_agreement
private def within_027_065 : CountSegment 1 10 13120 3520 3524 1672 1674 := CountSegment.join within_027_063 within_027_064
private theorem within_027_065_agreement : RowAgreement 3520 3524 := RowAgreement.join within_027_063_agreement within_027_064_agreement
private def within_027_066 : CountSegment 1 10 13120 3524 3526 1674 1675 := CountSegment.join atomic_3525 atomic_3526
private theorem within_027_066_agreement : RowAgreement 3524 3526 := RowAgreement.join atomic_3525_agreement atomic_3526_agreement
private def within_027_067 : CountSegment 1 10 13120 3526 3528 1675 1675 := CountSegment.join atomic_3527 atomic_3528
private theorem within_027_067_agreement : RowAgreement 3526 3528 := RowAgreement.join atomic_3527_agreement atomic_3528_agreement
private def within_027_068 : CountSegment 1 10 13120 3524 3528 1674 1675 := CountSegment.join within_027_066 within_027_067
private theorem within_027_068_agreement : RowAgreement 3524 3528 := RowAgreement.join within_027_066_agreement within_027_067_agreement
private def within_027_069 : CountSegment 1 10 13120 3520 3528 1672 1675 := CountSegment.join within_027_065 within_027_068
private theorem within_027_069_agreement : RowAgreement 3520 3528 := RowAgreement.join within_027_065_agreement within_027_068_agreement
private def within_027_070 : CountSegment 1 10 13120 3528 3530 1675 1676 := CountSegment.join atomic_3529 atomic_3530
private theorem within_027_070_agreement : RowAgreement 3528 3530 := RowAgreement.join atomic_3529_agreement atomic_3530_agreement
private def within_027_071 : CountSegment 1 10 13120 3530 3532 1676 1676 := CountSegment.join atomic_3531 atomic_3532
private theorem within_027_071_agreement : RowAgreement 3530 3532 := RowAgreement.join atomic_3531_agreement atomic_3532_agreement
private def within_027_072 : CountSegment 1 10 13120 3528 3532 1675 1676 := CountSegment.join within_027_070 within_027_071
private theorem within_027_072_agreement : RowAgreement 3528 3532 := RowAgreement.join within_027_070_agreement within_027_071_agreement
private def within_027_073 : CountSegment 1 10 13120 3532 3534 1676 1677 := CountSegment.join atomic_3533 atomic_3534
private theorem within_027_073_agreement : RowAgreement 3532 3534 := RowAgreement.join atomic_3533_agreement atomic_3534_agreement
private def within_027_074 : CountSegment 1 10 13120 3534 3536 1677 1677 := CountSegment.join atomic_3535 atomic_3536
private theorem within_027_074_agreement : RowAgreement 3534 3536 := RowAgreement.join atomic_3535_agreement atomic_3536_agreement
private def within_027_075 : CountSegment 1 10 13120 3532 3536 1676 1677 := CountSegment.join within_027_073 within_027_074
private theorem within_027_075_agreement : RowAgreement 3532 3536 := RowAgreement.join within_027_073_agreement within_027_074_agreement
private def within_027_076 : CountSegment 1 10 13120 3528 3536 1675 1677 := CountSegment.join within_027_072 within_027_075
private theorem within_027_076_agreement : RowAgreement 3528 3536 := RowAgreement.join within_027_072_agreement within_027_075_agreement
private def within_027_077 : CountSegment 1 10 13120 3520 3536 1672 1677 := CountSegment.join within_027_069 within_027_076
private theorem within_027_077_agreement : RowAgreement 3520 3536 := RowAgreement.join within_027_069_agreement within_027_076_agreement
private def within_027_078 : CountSegment 1 10 13120 3536 3538 1677 1677 := CountSegment.join atomic_3537 atomic_3538
private theorem within_027_078_agreement : RowAgreement 3536 3538 := RowAgreement.join atomic_3537_agreement atomic_3538_agreement
private def within_027_079 : CountSegment 1 10 13120 3538 3540 1677 1678 := CountSegment.join atomic_3539 atomic_3540
private theorem within_027_079_agreement : RowAgreement 3538 3540 := RowAgreement.join atomic_3539_agreement atomic_3540_agreement
private def within_027_080 : CountSegment 1 10 13120 3536 3540 1677 1678 := CountSegment.join within_027_078 within_027_079
private theorem within_027_080_agreement : RowAgreement 3536 3540 := RowAgreement.join within_027_078_agreement within_027_079_agreement
private def within_027_081 : CountSegment 1 10 13120 3540 3542 1678 1679 := CountSegment.join atomic_3541 atomic_3542
private theorem within_027_081_agreement : RowAgreement 3540 3542 := RowAgreement.join atomic_3541_agreement atomic_3542_agreement
private def within_027_082 : CountSegment 1 10 13120 3542 3544 1679 1679 := CountSegment.join atomic_3543 atomic_3544
private theorem within_027_082_agreement : RowAgreement 3542 3544 := RowAgreement.join atomic_3543_agreement atomic_3544_agreement
private def within_027_083 : CountSegment 1 10 13120 3540 3544 1678 1679 := CountSegment.join within_027_081 within_027_082
private theorem within_027_083_agreement : RowAgreement 3540 3544 := RowAgreement.join within_027_081_agreement within_027_082_agreement
private def within_027_084 : CountSegment 1 10 13120 3536 3544 1677 1679 := CountSegment.join within_027_080 within_027_083
private theorem within_027_084_agreement : RowAgreement 3536 3544 := RowAgreement.join within_027_080_agreement within_027_083_agreement
private def within_027_085 : CountSegment 1 10 13120 3544 3546 1679 1680 := CountSegment.join atomic_3545 atomic_3546
private theorem within_027_085_agreement : RowAgreement 3544 3546 := RowAgreement.join atomic_3545_agreement atomic_3546_agreement
private def within_027_086 : CountSegment 1 10 13120 3546 3548 1680 1680 := CountSegment.join atomic_3547 atomic_3548
private theorem within_027_086_agreement : RowAgreement 3546 3548 := RowAgreement.join atomic_3547_agreement atomic_3548_agreement
private def within_027_087 : CountSegment 1 10 13120 3544 3548 1679 1680 := CountSegment.join within_027_085 within_027_086
private theorem within_027_087_agreement : RowAgreement 3544 3548 := RowAgreement.join within_027_085_agreement within_027_086_agreement
private def within_027_088 : CountSegment 1 10 13120 3548 3550 1680 1681 := CountSegment.join atomic_3549 atomic_3550
private theorem within_027_088_agreement : RowAgreement 3548 3550 := RowAgreement.join atomic_3549_agreement atomic_3550_agreement
private def within_027_089 : CountSegment 1 10 13120 3550 3552 1681 1681 := CountSegment.join atomic_3551 atomic_3552
private theorem within_027_089_agreement : RowAgreement 3550 3552 := RowAgreement.join atomic_3551_agreement atomic_3552_agreement
private def within_027_090 : CountSegment 1 10 13120 3548 3552 1680 1681 := CountSegment.join within_027_088 within_027_089
private theorem within_027_090_agreement : RowAgreement 3548 3552 := RowAgreement.join within_027_088_agreement within_027_089_agreement
private def within_027_091 : CountSegment 1 10 13120 3544 3552 1679 1681 := CountSegment.join within_027_087 within_027_090
private theorem within_027_091_agreement : RowAgreement 3544 3552 := RowAgreement.join within_027_087_agreement within_027_090_agreement
private def within_027_092 : CountSegment 1 10 13120 3536 3552 1677 1681 := CountSegment.join within_027_084 within_027_091
private theorem within_027_092_agreement : RowAgreement 3536 3552 := RowAgreement.join within_027_084_agreement within_027_091_agreement
private def within_027_093 : CountSegment 1 10 13120 3520 3552 1672 1681 := CountSegment.join within_027_077 within_027_092
private theorem within_027_093_agreement : RowAgreement 3520 3552 := RowAgreement.join within_027_077_agreement within_027_092_agreement
private def within_027_094 : CountSegment 1 10 13120 3552 3554 1681 1682 := CountSegment.join atomic_3553 atomic_3554
private theorem within_027_094_agreement : RowAgreement 3552 3554 := RowAgreement.join atomic_3553_agreement atomic_3554_agreement
private def within_027_095 : CountSegment 1 10 13120 3554 3556 1682 1684 := CountSegment.join atomic_3555 atomic_3556
private theorem within_027_095_agreement : RowAgreement 3554 3556 := RowAgreement.join atomic_3555_agreement atomic_3556_agreement
private def within_027_096 : CountSegment 1 10 13120 3552 3556 1681 1684 := CountSegment.join within_027_094 within_027_095
private theorem within_027_096_agreement : RowAgreement 3552 3556 := RowAgreement.join within_027_094_agreement within_027_095_agreement
private def within_027_097 : CountSegment 1 10 13120 3556 3558 1684 1686 := CountSegment.join atomic_3557 atomic_3558
private theorem within_027_097_agreement : RowAgreement 3556 3558 := RowAgreement.join atomic_3557_agreement atomic_3558_agreement
private def within_027_098 : CountSegment 1 10 13120 3558 3560 1686 1687 := CountSegment.join atomic_3559 atomic_3560
private theorem within_027_098_agreement : RowAgreement 3558 3560 := RowAgreement.join atomic_3559_agreement atomic_3560_agreement
private def within_027_099 : CountSegment 1 10 13120 3556 3560 1684 1687 := CountSegment.join within_027_097 within_027_098
private theorem within_027_099_agreement : RowAgreement 3556 3560 := RowAgreement.join within_027_097_agreement within_027_098_agreement
private def within_027_100 : CountSegment 1 10 13120 3552 3560 1681 1687 := CountSegment.join within_027_096 within_027_099
private theorem within_027_100_agreement : RowAgreement 3552 3560 := RowAgreement.join within_027_096_agreement within_027_099_agreement
private def within_027_101 : CountSegment 1 10 13120 3560 3562 1687 1688 := CountSegment.join atomic_3561 atomic_3562
private theorem within_027_101_agreement : RowAgreement 3560 3562 := RowAgreement.join atomic_3561_agreement atomic_3562_agreement
private def within_027_102 : CountSegment 1 10 13120 3562 3564 1688 1688 := CountSegment.join atomic_3563 atomic_3564
private theorem within_027_102_agreement : RowAgreement 3562 3564 := RowAgreement.join atomic_3563_agreement atomic_3564_agreement
private def within_027_103 : CountSegment 1 10 13120 3560 3564 1687 1688 := CountSegment.join within_027_101 within_027_102
private theorem within_027_103_agreement : RowAgreement 3560 3564 := RowAgreement.join within_027_101_agreement within_027_102_agreement
private def within_027_104 : CountSegment 1 10 13120 3564 3566 1688 1689 := CountSegment.join atomic_3565 atomic_3566
private theorem within_027_104_agreement : RowAgreement 3564 3566 := RowAgreement.join atomic_3565_agreement atomic_3566_agreement
private def within_027_105 : CountSegment 1 10 13120 3566 3568 1689 1689 := CountSegment.join atomic_3567 atomic_3568
private theorem within_027_105_agreement : RowAgreement 3566 3568 := RowAgreement.join atomic_3567_agreement atomic_3568_agreement
private def within_027_106 : CountSegment 1 10 13120 3564 3568 1688 1689 := CountSegment.join within_027_104 within_027_105
private theorem within_027_106_agreement : RowAgreement 3564 3568 := RowAgreement.join within_027_104_agreement within_027_105_agreement
private def within_027_107 : CountSegment 1 10 13120 3560 3568 1687 1689 := CountSegment.join within_027_103 within_027_106
private theorem within_027_107_agreement : RowAgreement 3560 3568 := RowAgreement.join within_027_103_agreement within_027_106_agreement
private def within_027_108 : CountSegment 1 10 13120 3552 3568 1681 1689 := CountSegment.join within_027_100 within_027_107
private theorem within_027_108_agreement : RowAgreement 3552 3568 := RowAgreement.join within_027_100_agreement within_027_107_agreement
private def within_027_109 : CountSegment 1 10 13120 3568 3570 1689 1689 := CountSegment.join atomic_3569 atomic_3570
private theorem within_027_109_agreement : RowAgreement 3568 3570 := RowAgreement.join atomic_3569_agreement atomic_3570_agreement
private def within_027_110 : CountSegment 1 10 13120 3570 3572 1689 1690 := CountSegment.join atomic_3571 atomic_3572
private theorem within_027_110_agreement : RowAgreement 3570 3572 := RowAgreement.join atomic_3571_agreement atomic_3572_agreement
private def within_027_111 : CountSegment 1 10 13120 3568 3572 1689 1690 := CountSegment.join within_027_109 within_027_110
private theorem within_027_111_agreement : RowAgreement 3568 3572 := RowAgreement.join within_027_109_agreement within_027_110_agreement
private def within_027_112 : CountSegment 1 10 13120 3572 3574 1690 1690 := CountSegment.join atomic_3573 atomic_3574
private theorem within_027_112_agreement : RowAgreement 3572 3574 := RowAgreement.join atomic_3573_agreement atomic_3574_agreement
private def within_027_113 : CountSegment 1 10 13120 3574 3576 1690 1692 := CountSegment.join atomic_3575 atomic_3576
private theorem within_027_113_agreement : RowAgreement 3574 3576 := RowAgreement.join atomic_3575_agreement atomic_3576_agreement
private def within_027_114 : CountSegment 1 10 13120 3572 3576 1690 1692 := CountSegment.join within_027_112 within_027_113
private theorem within_027_114_agreement : RowAgreement 3572 3576 := RowAgreement.join within_027_112_agreement within_027_113_agreement
private def within_027_115 : CountSegment 1 10 13120 3568 3576 1689 1692 := CountSegment.join within_027_111 within_027_114
private theorem within_027_115_agreement : RowAgreement 3568 3576 := RowAgreement.join within_027_111_agreement within_027_114_agreement
private def within_027_116 : CountSegment 1 10 13120 3576 3578 1692 1692 := CountSegment.join atomic_3577 atomic_3578
private theorem within_027_116_agreement : RowAgreement 3576 3578 := RowAgreement.join atomic_3577_agreement atomic_3578_agreement
private def within_027_117 : CountSegment 1 10 13120 3578 3580 1692 1695 := CountSegment.join atomic_3579 atomic_3580
private theorem within_027_117_agreement : RowAgreement 3578 3580 := RowAgreement.join atomic_3579_agreement atomic_3580_agreement
private def within_027_118 : CountSegment 1 10 13120 3576 3580 1692 1695 := CountSegment.join within_027_116 within_027_117
private theorem within_027_118_agreement : RowAgreement 3576 3580 := RowAgreement.join within_027_116_agreement within_027_117_agreement
private def within_027_119 : CountSegment 1 10 13120 3580 3582 1695 1697 := CountSegment.join atomic_3581 atomic_3582
private theorem within_027_119_agreement : RowAgreement 3580 3582 := RowAgreement.join atomic_3581_agreement atomic_3582_agreement
private def within_027_120 : CountSegment 1 10 13120 3582 3584 1697 1697 := CountSegment.join atomic_3583 atomic_3584
private theorem within_027_120_agreement : RowAgreement 3582 3584 := RowAgreement.join atomic_3583_agreement atomic_3584_agreement
private def within_027_121 : CountSegment 1 10 13120 3580 3584 1695 1697 := CountSegment.join within_027_119 within_027_120
private theorem within_027_121_agreement : RowAgreement 3580 3584 := RowAgreement.join within_027_119_agreement within_027_120_agreement
private def within_027_122 : CountSegment 1 10 13120 3576 3584 1692 1697 := CountSegment.join within_027_118 within_027_121
private theorem within_027_122_agreement : RowAgreement 3576 3584 := RowAgreement.join within_027_118_agreement within_027_121_agreement
private def within_027_123 : CountSegment 1 10 13120 3568 3584 1689 1697 := CountSegment.join within_027_115 within_027_122
private theorem within_027_123_agreement : RowAgreement 3568 3584 := RowAgreement.join within_027_115_agreement within_027_122_agreement
private def within_027_124 : CountSegment 1 10 13120 3552 3584 1681 1697 := CountSegment.join within_027_108 within_027_123
private theorem within_027_124_agreement : RowAgreement 3552 3584 := RowAgreement.join within_027_108_agreement within_027_123_agreement
private def within_027_125 : CountSegment 1 10 13120 3520 3584 1672 1697 := CountSegment.join within_027_093 within_027_124
private theorem within_027_125_agreement : RowAgreement 3520 3584 := RowAgreement.join within_027_093_agreement within_027_124_agreement
private def within_027_126 : CountSegment 1 10 13120 3456 3584 1644 1697 := CountSegment.join within_027_062 within_027_125
private theorem within_027_126_agreement : RowAgreement 3456 3584 := RowAgreement.join within_027_062_agreement within_027_125_agreement
def segment_027 : CountSegment 1 10 13120 3456 3584 1644 1697 := within_027_126
theorem agreement_027 : RowAgreement 3456 3584 := within_027_126_agreement
end ProjectEulerSolutions.P58Numerical
