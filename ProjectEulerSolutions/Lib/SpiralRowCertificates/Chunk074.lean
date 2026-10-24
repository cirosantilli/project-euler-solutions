import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9473 : RowCertificate 9473 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9473 1) := by change ¬ Nat.Prime 358969863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119656621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9473 2) := by change ¬ Nat.Prime 358950917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27611609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9473 3) := by change ¬ Nat.Prime 358931971; exact PrattCertificates.composite_of_product (factor := 5791) (cofactor := 61981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9473 : row 9473 = 0 := (row_9473.checked).trans rfl
theorem model_count_9473 : modelRow 9473 = 0 := by decide
theorem modelrow_agrees_9473 : modelRow 9473 = row 9473 := model_count_9473.trans row_count_9473.symm
def row_9474 : RowCertificate 9474 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9474 1) := by change Nat.Prime 359045653; exact ProjectEulerSolutions.P58PrimeNodes.p_359045653.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9474 2) := by change ¬ Nat.Prime 359026705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71805341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9474 3) := by change Nat.Prime 359007757; exact ProjectEulerSolutions.P58PrimeNodes.p_359007757.prime
    exact decide_eq_true prime
theorem row_count_9474 : row 9474 = 2 := (row_9474.checked).trans rfl
theorem model_count_9474 : modelRow 9474 = 2 := by decide
theorem modelrow_agrees_9474 : modelRow 9474 = row 9474 := model_count_9474.trans row_count_9474.symm
def row_9475 : RowCertificate 9475 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9475 1) := by change ¬ Nat.Prime 359121451; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27624727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9475 2) := by change ¬ Nat.Prime 359102501; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9705473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9475 3) := by change ¬ Nat.Prime 359083551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119694517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9475 : row 9475 = 0 := (row_9475.checked).trans rfl
theorem model_count_9475 : modelRow 9475 = 0 := by decide
theorem modelrow_agrees_9475 : modelRow 9475 = row 9475 := model_count_9475.trans row_count_9475.symm
def row_9476 : RowCertificate 9476 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9476 1) := by change ¬ Nat.Prime 359197257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119732419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9476 2) := by change ¬ Nat.Prime 359178305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71835661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9476 3) := by change ¬ Nat.Prime 359159353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51308479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9476 : row 9476 = 0 := (row_9476.checked).trans rfl
theorem model_count_9476 : modelRow 9476 = 0 := by decide
theorem modelrow_agrees_9476 : modelRow 9476 = row 9476 := model_count_9476.trans row_count_9476.symm
def row_9477 : RowCertificate 9477 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9477 1) := by change ¬ Nat.Prime 359273071; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18909109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9477 2) := by change ¬ Nat.Prime 359254117; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12388073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9477 3) := by change ¬ Nat.Prime 359235163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51319309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9477 : row 9477 = 0 := (row_9477.checked).trans rfl
theorem model_count_9477 : modelRow 9477 = 0 := by decide
theorem modelrow_agrees_9477 : modelRow 9477 = row 9477 := model_count_9477.trans row_count_9477.symm
def row_9478 : RowCertificate 9478 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9478 1) := by change ¬ Nat.Prime 359348893; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8356951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9478 2) := by change ¬ Nat.Prime 359329937; exact PrattCertificates.composite_of_product (factor := 7789) (cofactor := 46133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9478 3) := by change ¬ Nat.Prime 359310981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119770327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9478 : row 9478 = 0 := (row_9478.checked).trans rfl
theorem model_count_9478 : modelRow 9478 = 0 := by decide
theorem modelrow_agrees_9478 : modelRow 9478 = row 9478 := model_count_9478.trans row_count_9478.symm
def row_9479 : RowCertificate 9479 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9479 1) := by change ¬ Nat.Prime 359424723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119808241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9479 2) := by change ¬ Nat.Prime 359405765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71881153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9479 3) := by change ¬ Nat.Prime 359386807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27645139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9479 : row 9479 = 0 := (row_9479.checked).trans rfl
theorem model_count_9479 : modelRow 9479 = 0 := by decide
theorem modelrow_agrees_9479 : modelRow 9479 = row 9479 := model_count_9479.trans row_count_9479.symm
def row_9480 : RowCertificate 9480 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9480 1) := by change ¬ Nat.Prime 359500561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51357223) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9480 2) := by change ¬ Nat.Prime 359481601; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5893141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9480 3) := by change ¬ Nat.Prime 359462641; exact PrattCertificates.composite_of_product (factor := 15601) (cofactor := 23041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9480 : row 9480 = 0 := (row_9480.checked).trans rfl
theorem model_count_9480 : modelRow 9480 = 0 := by decide
theorem modelrow_agrees_9480 : modelRow 9480 = row 9480 := model_count_9480.trans row_count_9480.symm
def row_9481 : RowCertificate 9481 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9481 1) := by change ¬ Nat.Prime 359576407; exact PrattCertificates.composite_of_product (factor := 7477) (cofactor := 48091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9481 2) := by change ¬ Nat.Prime 359557445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71911489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9481 3) := by change ¬ Nat.Prime 359538483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119846161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9481 : row 9481 = 0 := (row_9481.checked).trans rfl
theorem model_count_9481 : modelRow 9481 = 0 := by decide
theorem modelrow_agrees_9481 : modelRow 9481 = row 9481 := model_count_9481.trans row_count_9481.symm
def row_9482 : RowCertificate 9482 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9482 1) := by change ¬ Nat.Prime 359652261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119884087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9482 2) := by change ¬ Nat.Prime 359633297; exact PrattCertificates.composite_of_product (factor := 2957) (cofactor := 121621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9482 3) := by change ¬ Nat.Prime 359614333; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27662641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9482 : row 9482 = 0 := (row_9482.checked).trans rfl
theorem model_count_9482 : modelRow 9482 = 0 := by decide
theorem modelrow_agrees_9482 : modelRow 9482 = row 9482 := model_count_9482.trans row_count_9482.symm
def row_9483 : RowCertificate 9483 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9483 1) := by change ¬ Nat.Prime 359728123; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11604133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9483 2) := by change Nat.Prime 359709157; exact ProjectEulerSolutions.P58PrimeNodes.p_359709157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9483 3) := by change ¬ Nat.Prime 359690191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51384313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9483 : row 9483 = 1 := (row_9483.checked).trans rfl
theorem model_count_9483 : modelRow 9483 = 1 := by decide
theorem modelrow_agrees_9483 : modelRow 9483 = row 9483 := model_count_9483.trans row_count_9483.symm
def row_9484 : RowCertificate 9484 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9484 1) := by change ¬ Nat.Prime 359803993; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1030957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9484 2) := by change ¬ Nat.Prime 359785025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71957005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9484 3) := by change ¬ Nat.Prime 359766057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119922019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9484 : row 9484 = 0 := (row_9484.checked).trans rfl
theorem model_count_9484 : modelRow 9484 = 0 := by decide
theorem modelrow_agrees_9484 : modelRow 9484 = row 9484 := model_count_9484.trans row_count_9484.symm
def row_9485 : RowCertificate 9485 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9485 1) := by change ¬ Nat.Prime 359879871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119959957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9485 2) := by change Nat.Prime 359860901; exact ProjectEulerSolutions.P58PrimeNodes.p_359860901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9485 3) := by change ¬ Nat.Prime 359841931; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18939049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9485 : row 9485 = 1 := (row_9485.checked).trans rfl
theorem model_count_9485 : modelRow 9485 = 1 := by decide
theorem modelrow_agrees_9485 : modelRow 9485 = row 9485 := model_count_9485.trans row_count_9485.symm
def row_9486 : RowCertificate 9486 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9486 1) := by change ¬ Nat.Prime 359955757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51422251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9486 2) := by change ¬ Nat.Prime 359936785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71987357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9486 3) := by change ¬ Nat.Prime 359917813; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4930381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9486 : row 9486 = 0 := (row_9486.checked).trans rfl
theorem model_count_9486 : modelRow 9486 = 0 := by decide
theorem modelrow_agrees_9486 : modelRow 9486 = row 9486 := model_count_9486.trans row_count_9486.symm
def row_9487 : RowCertificate 9487 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9487 1) := by change ¬ Nat.Prime 360031651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51433093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9487 2) := by change ¬ Nat.Prime 360012677; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8780797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9487 3) := by change ¬ Nat.Prime 359993703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119997901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9487 : row 9487 = 0 := (row_9487.checked).trans rfl
theorem model_count_9487 : modelRow 9487 = 0 := by decide
theorem modelrow_agrees_9487 : modelRow 9487 = row 9487 := model_count_9487.trans row_count_9487.symm
def row_9488 : RowCertificate 9488 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9488 1) := by change ¬ Nat.Prime 360107553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120035851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9488 2) := by change ¬ Nat.Prime 360088577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21181681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9488 3) := by change ¬ Nat.Prime 360069601; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1706491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9488 : row 9488 = 0 := (row_9488.checked).trans rfl
theorem model_count_9488 : modelRow 9488 = 0 := by decide
theorem modelrow_agrees_9488 : modelRow 9488 = row 9488 := model_count_9488.trans row_count_9488.symm
def row_9489 : RowCertificate 9489 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9489 1) := by change Nat.Prime 360183463; exact ProjectEulerSolutions.P58PrimeNodes.p_360183463.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9489 2) := by change ¬ Nat.Prime 360164485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72032897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9489 3) := by change ¬ Nat.Prime 360145507; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11617597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9489 : row 9489 = 1 := (row_9489.checked).trans rfl
theorem model_count_9489 : modelRow 9489 = 1 := by decide
theorem modelrow_agrees_9489 : modelRow 9489 = row 9489 := model_count_9489.trans row_count_9489.symm
def row_9490 : RowCertificate 9490 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9490 1) := by change Nat.Prime 360259381; exact ProjectEulerSolutions.P58PrimeNodes.p_360259381.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9490 2) := by change Nat.Prime 360240401; exact ProjectEulerSolutions.P58PrimeNodes.p_360240401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9490 3) := by change ¬ Nat.Prime 360221421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120073807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9490 : row 9490 = 2 := (row_9490.checked).trans rfl
theorem model_count_9490 : modelRow 9490 = 2 := by decide
theorem modelrow_agrees_9490 : modelRow 9490 = row 9490 := model_count_9490.trans row_count_9490.symm
def row_9491 : RowCertificate 9491 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9491 1) := by change ¬ Nat.Prime 360335307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120111769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9491 2) := by change ¬ Nat.Prime 360316325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72063265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9491 3) := by change ¬ Nat.Prime 360297343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51471049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9491 : row 9491 = 0 := (row_9491.checked).trans rfl
theorem model_count_9491 : modelRow 9491 = 0 := by decide
theorem modelrow_agrees_9491 : modelRow 9491 = row 9491 := model_count_9491.trans row_count_9491.symm
def row_9492 : RowCertificate 9492 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9492 1) := by change ¬ Nat.Prime 360411241; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5908381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9492 2) := by change Nat.Prime 360392257; exact ProjectEulerSolutions.P58PrimeNodes.p_360392257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9492 3) := by change ¬ Nat.Prime 360373273; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27721021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9492 : row 9492 = 1 := (row_9492.checked).trans rfl
theorem model_count_9492 : modelRow 9492 = 1 := by decide
theorem modelrow_agrees_9492 : modelRow 9492 = row 9492 := model_count_9492.trans row_count_9492.symm
def row_9493 : RowCertificate 9493 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9493 1) := by change ¬ Nat.Prime 360487183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51498169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9493 2) := by change ¬ Nat.Prime 360468197; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1991537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9493 3) := by change ¬ Nat.Prime 360449211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120149737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9493 : row 9493 = 0 := (row_9493.checked).trans rfl
theorem model_count_9493 : modelRow 9493 = 0 := by decide
theorem modelrow_agrees_9493 : modelRow 9493 = row 9493 := model_count_9493.trans row_count_9493.symm
def row_9494 : RowCertificate 9494 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9494 1) := by change ¬ Nat.Prime 360563133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120187711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9494 2) := by change ¬ Nat.Prime 360544145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72108829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9494 3) := by change Nat.Prime 360525157; exact ProjectEulerSolutions.P58PrimeNodes.p_360525157.prime
    exact decide_eq_true prime
theorem row_count_9494 : row 9494 = 1 := (row_9494.checked).trans rfl
theorem model_count_9494 : modelRow 9494 = 1 := by decide
theorem modelrow_agrees_9494 : modelRow 9494 = row 9494 := model_count_9494.trans row_count_9494.symm
def row_9495 : RowCertificate 9495 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9495 1) := by change ¬ Nat.Prime 360639091; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5382673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9495 2) := by change ¬ Nat.Prime 360620101; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3717733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9495 3) := by change ¬ Nat.Prime 360601111; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27738547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9495 : row 9495 = 0 := (row_9495.checked).trans rfl
theorem model_count_9495 : modelRow 9495 = 0 := by decide
theorem modelrow_agrees_9495 : modelRow 9495 = row 9495 := model_count_9495.trans row_count_9495.symm
def row_9496 : RowCertificate 9496 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9496 1) := by change ¬ Nat.Prime 360715057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18985003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9496 2) := by change ¬ Nat.Prime 360696065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72139213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9496 3) := by change ¬ Nat.Prime 360677073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120225691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9496 : row 9496 = 0 := (row_9496.checked).trans rfl
theorem model_count_9496 : modelRow 9496 = 0 := by decide
theorem modelrow_agrees_9496 : modelRow 9496 = row 9496 := model_count_9496.trans row_count_9496.symm
def row_9497 : RowCertificate 9497 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9497 1) := by change ¬ Nat.Prime 360791031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120263677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9497 2) := by change Nat.Prime 360772037; exact ProjectEulerSolutions.P58PrimeNodes.p_360772037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9497 3) := by change ¬ Nat.Prime 360753043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51536149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9497 : row 9497 = 1 := (row_9497.checked).trans rfl
theorem model_count_9497 : modelRow 9497 = 1 := by decide
theorem modelrow_agrees_9497 : modelRow 9497 = row 9497 := model_count_9497.trans row_count_9497.symm
def row_9498 : RowCertificate 9498 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9498 1) := by change ¬ Nat.Prime 360867013; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27759001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9498 2) := by change Nat.Prime 360848017; exact ProjectEulerSolutions.P58PrimeNodes.p_360848017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9498 3) := by change ¬ Nat.Prime 360829021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51547003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9498 : row 9498 = 1 := (row_9498.checked).trans rfl
theorem model_count_9498 : modelRow 9498 = 1 := by decide
theorem modelrow_agrees_9498 : modelRow 9498 = row 9498 := model_count_9498.trans row_count_9498.symm
def row_9499 : RowCertificate 9499 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9499 1) := by change ¬ Nat.Prime 360943003; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5387209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9499 2) := by change ¬ Nat.Prime 360924005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72184801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9499 3) := by change ¬ Nat.Prime 360905007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120301669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9499 : row 9499 = 0 := (row_9499.checked).trans rfl
theorem model_count_9499 : modelRow 9499 = 0 := by decide
theorem modelrow_agrees_9499 : modelRow 9499 = row 9499 := model_count_9499.trans row_count_9499.symm
def row_9500 : RowCertificate 9500 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9500 1) := by change ¬ Nat.Prime 361019001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120339667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9500 2) := by change Nat.Prime 361000001; exact ProjectEulerSolutions.P58PrimeNodes.p_361000001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9500 3) := by change ¬ Nat.Prime 360981001; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8394907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9500 : row 9500 = 1 := (row_9500.checked).trans rfl
theorem model_count_9500 : modelRow 9500 = 1 := by decide
theorem modelrow_agrees_9500 : modelRow 9500 = row 9500 := model_count_9500.trans row_count_9500.symm
def row_9501 : RowCertificate 9501 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9501 1) := by change ¬ Nat.Prime 361095007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51585001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9501 2) := by change ¬ Nat.Prime 361076005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72215201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9501 3) := by change ¬ Nat.Prime 361057003; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 952657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9501 : row 9501 = 0 := (row_9501.checked).trans rfl
theorem model_count_9501 : modelRow 9501 = 0 := by decide
theorem modelrow_agrees_9501 : modelRow 9501 = row 9501 := model_count_9501.trans row_count_9501.symm
def row_9502 : RowCertificate 9502 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9502 1) := by change ¬ Nat.Prime 361171021; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1711711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9502 2) := by change ¬ Nat.Prime 361152017; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6814189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9502 3) := by change ¬ Nat.Prime 361133013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120377671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9502 : row 9502 = 0 := (row_9502.checked).trans rfl
theorem model_count_9502 : modelRow 9502 = 0 := by decide
theorem modelrow_agrees_9502 : modelRow 9502 = row 9502 := model_count_9502.trans row_count_9502.symm
def row_9503 : RowCertificate 9503 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9503 1) := by change ¬ Nat.Prime 361247043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120415681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9503 2) := by change ¬ Nat.Prime 361228037; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 2636701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9503 3) := by change ¬ Nat.Prime 361209031; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 1276357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9503 : row 9503 = 0 := (row_9503.checked).trans rfl
theorem model_count_9503 : modelRow 9503 = 0 := by decide
theorem modelrow_agrees_9503 : modelRow 9503 = row 9503 := model_count_9503.trans row_count_9503.symm
def row_9504 : RowCertificate 9504 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9504 1) := by change ¬ Nat.Prime 361323073; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11655583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9504 2) := by change ¬ Nat.Prime 361304065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72260813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9504 3) := by change ¬ Nat.Prime 361285057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51612151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9504 : row 9504 = 0 := (row_9504.checked).trans rfl
theorem model_count_9504 : modelRow 9504 = 0 := by decide
theorem modelrow_agrees_9504 : modelRow 9504 = row 9504 := model_count_9504.trans row_count_9504.symm
def row_9505 : RowCertificate 9505 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9505 1) := by change Nat.Prime 361399111; exact ProjectEulerSolutions.P58PrimeNodes.p_361399111.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9505 2) := by change ¬ Nat.Prime 361380101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21257653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9505 3) := by change ¬ Nat.Prime 361361091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120453697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9505 : row 9505 = 1 := (row_9505.checked).trans rfl
theorem model_count_9505 : modelRow 9505 = 1 := by decide
theorem modelrow_agrees_9505 : modelRow 9505 = row 9505 := model_count_9505.trans row_count_9505.symm
def row_9506 : RowCertificate 9506 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9506 1) := by change ¬ Nat.Prime 361475157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120491719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9506 2) := by change ¬ Nat.Prime 361456145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72291229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9506 3) := by change ¬ Nat.Prime 361437133; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19023007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9506 : row 9506 = 0 := (row_9506.checked).trans rfl
theorem model_count_9506 : modelRow 9506 = 0 := by decide
theorem modelrow_agrees_9506 : modelRow 9506 = row 9506 := model_count_9506.trans row_count_9506.symm
def row_9507 : RowCertificate 9507 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9507 1) := by change ¬ Nat.Prime 361551211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51650173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9507 2) := by change ¬ Nat.Prime 361532197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27810169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9507 3) := by change Nat.Prime 361513183; exact ProjectEulerSolutions.P58PrimeNodes.p_361513183.prime
    exact decide_eq_true prime
theorem row_count_9507 : row 9507 = 1 := (row_9507.checked).trans rfl
theorem model_count_9507 : modelRow 9507 = 1 := by decide
theorem modelrow_agrees_9507 : modelRow 9507 = row 9507 := model_count_9507.trans row_count_9507.symm
def row_9508 : RowCertificate 9508 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9508 1) := by change ¬ Nat.Prime 361627273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51661039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9508 2) := by change Nat.Prime 361608257; exact ProjectEulerSolutions.P58PrimeNodes.p_361608257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9508 3) := by change ¬ Nat.Prime 361589241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120529747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9508 : row 9508 = 1 := (row_9508.checked).trans rfl
theorem model_count_9508 : modelRow 9508 = 1 := by decide
theorem modelrow_agrees_9508 : modelRow 9508 = row 9508 := model_count_9508.trans row_count_9508.symm
def row_9509 : RowCertificate 9509 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9509 1) := by change ¬ Nat.Prime 361703343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120567781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9509 2) := by change ¬ Nat.Prime 361684325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72336865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9509 3) := by change ¬ Nat.Prime 361665307; exact PrattCertificates.composite_of_product (factor := 6091) (cofactor := 59377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9509 : row 9509 = 0 := (row_9509.checked).trans rfl
theorem model_count_9509 : modelRow 9509 = 0 := by decide
theorem modelrow_agrees_9509 : modelRow 9509 = row 9509 := model_count_9509.trans row_count_9509.symm
def row_9510 : RowCertificate 9510 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9510 1) := by change ¬ Nat.Prime 361779421; exact PrattCertificates.composite_of_product (factor := 3079) (cofactor := 117499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9510 2) := by change ¬ Nat.Prime 361760401; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 1073473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9510 3) := by change ¬ Nat.Prime 361741381; exact PrattCertificates.composite_of_product (factor := 9811) (cofactor := 36871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9510 : row 9510 = 0 := (row_9510.checked).trans rfl
theorem model_count_9510 : modelRow 9510 = 0 := by decide
theorem modelrow_agrees_9510 : modelRow 9510 = row 9510 := model_count_9510.trans row_count_9510.symm
def row_9511 : RowCertificate 9511 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9511 1) := by change ¬ Nat.Prime 361855507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27835039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9511 2) := by change ¬ Nat.Prime 361836485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72367297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9511 3) := by change ¬ Nat.Prime 361817463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120605821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9511 : row 9511 = 0 := (row_9511.checked).trans rfl
theorem model_count_9511 : modelRow 9511 = 0 := by decide
theorem modelrow_agrees_9511 : modelRow 9511 = row 9511 := model_count_9511.trans row_count_9511.symm
def row_9512 : RowCertificate 9512 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9512 1) := by change ¬ Nat.Prime 361931601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120643867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9512 2) := by change ¬ Nat.Prime 361912577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27839429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9512 3) := by change ¬ Nat.Prime 361893553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51699079) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9512 : row 9512 = 0 := (row_9512.checked).trans rfl
theorem model_count_9512 : modelRow 9512 = 0 := by decide
theorem modelrow_agrees_9512 : modelRow 9512 = row 9512 := model_count_9512.trans row_count_9512.symm
def row_9513 : RowCertificate 9513 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9513 1) := by change ¬ Nat.Prime 362007703; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19053037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9513 2) := by change ¬ Nat.Prime 361988677; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4958749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9513 3) := by change Nat.Prime 361969651; exact ProjectEulerSolutions.P58PrimeNodes.p_361969651.prime
    exact decide_eq_true prime
theorem row_count_9513 : row 9513 = 1 := (row_9513.checked).trans rfl
theorem model_count_9513 : modelRow 9513 = 1 := by decide
theorem modelrow_agrees_9513 : modelRow 9513 = row 9513 := model_count_9513.trans row_count_9513.symm
def row_9514 : RowCertificate 9514 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9514 1) := by change ¬ Nat.Prime 362083813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51726259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9514 2) := by change ¬ Nat.Prime 362064785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72412957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9514 3) := by change ¬ Nat.Prime 362045757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120681919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9514 : row 9514 = 0 := (row_9514.checked).trans rfl
theorem model_count_9514 : modelRow 9514 = 0 := by decide
theorem modelrow_agrees_9514 : modelRow 9514 = row 9514 := model_count_9514.trans row_count_9514.symm
def row_9515 : RowCertificate 9515 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9515 1) := by change ¬ Nat.Prime 362159931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120719977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9515 2) := by change Nat.Prime 362140901; exact ProjectEulerSolutions.P58PrimeNodes.p_362140901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9515 3) := by change ¬ Nat.Prime 362121871; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3322219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9515 : row 9515 = 1 := (row_9515.checked).trans rfl
theorem model_count_9515 : modelRow 9515 = 1 := by decide
theorem modelrow_agrees_9515 : modelRow 9515 = row 9515 := model_count_9515.trans row_count_9515.symm
def row_9516 : RowCertificate 9516 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9516 1) := by change Nat.Prime 362236057; exact ProjectEulerSolutions.P58PrimeNodes.p_362236057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9516 2) := by change ¬ Nat.Prime 362217025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72443405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9516 3) := by change ¬ Nat.Prime 362197993; exact PrattCertificates.composite_of_product (factor := 2137) (cofactor := 169489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9516 : row 9516 = 1 := (row_9516.checked).trans rfl
theorem model_count_9516 : modelRow 9516 = 1 := by decide
theorem modelrow_agrees_9516 : modelRow 9516 = row 9516 := model_count_9516.trans row_count_9516.symm
def row_9517 : RowCertificate 9517 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9517 1) := by change Nat.Prime 362312191; exact ProjectEulerSolutions.P58PrimeNodes.p_362312191.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9517 2) := by change ¬ Nat.Prime 362293157; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3734981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9517 3) := by change ¬ Nat.Prime 362274123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120758041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9517 : row 9517 = 1 := (row_9517.checked).trans rfl
theorem model_count_9517 : modelRow 9517 = 1 := by decide
theorem modelrow_agrees_9517 : modelRow 9517 = row 9517 := model_count_9517.trans row_count_9517.symm
def row_9518 : RowCertificate 9518 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9518 1) := by change ¬ Nat.Prime 362388333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120796111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9518 2) := by change ¬ Nat.Prime 362369297; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21315841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9518 3) := by change ¬ Nat.Prime 362350261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51764323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9518 : row 9518 = 0 := (row_9518.checked).trans rfl
theorem model_count_9518 : modelRow 9518 = 0 := by decide
theorem modelrow_agrees_9518 : modelRow 9518 = row 9518 := model_count_9518.trans row_count_9518.symm
def row_9519 : RowCertificate 9519 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9519 1) := by change Nat.Prime 362464483; exact ProjectEulerSolutions.P58PrimeNodes.p_362464483.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9519 2) := by change ¬ Nat.Prime 362445445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72489089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9519 3) := by change ¬ Nat.Prime 362426407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51775201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9519 : row 9519 = 1 := (row_9519.checked).trans rfl
theorem model_count_9519 : modelRow 9519 = 1 := by decide
theorem modelrow_agrees_9519 : modelRow 9519 = row 9519 := model_count_9519.trans row_count_9519.symm
def row_9520 : RowCertificate 9520 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9520 1) := by change ¬ Nat.Prime 362540641; exact PrattCertificates.composite_of_product (factor := 2767) (cofactor := 131023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9520 2) := by change ¬ Nat.Prime 362521601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27886277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9520 3) := by change ¬ Nat.Prime 362502561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120834187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9520 : row 9520 = 0 := (row_9520.checked).trans rfl
theorem model_count_9520 : modelRow 9520 = 0 := by decide
theorem modelrow_agrees_9520 : modelRow 9520 = row 9520 := model_count_9520.trans row_count_9520.symm
def row_9521 : RowCertificate 9521 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9521 1) := by change ¬ Nat.Prime 362616807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120872269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9521 2) := by change ¬ Nat.Prime 362597765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72519553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9521 3) := by change ¬ Nat.Prime 362578723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27890671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9521 : row 9521 = 0 := (row_9521.checked).trans rfl
theorem model_count_9521 : modelRow 9521 = 0 := by decide
theorem modelrow_agrees_9521 : modelRow 9521 = row 9521 := model_count_9521.trans row_count_9521.symm
def row_9522 : RowCertificate 9522 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9522 1) := by change ¬ Nat.Prime 362692981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51813283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9522 2) := by change ¬ Nat.Prime 362673937; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21333761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9522 3) := by change ¬ Nat.Prime 362654893; exact PrattCertificates.composite_of_product (factor := 3709) (cofactor := 97777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9522 : row 9522 = 0 := (row_9522.checked).trans rfl
theorem model_count_9522 : modelRow 9522 = 0 := by decide
theorem modelrow_agrees_9522 : modelRow 9522 = row 9522 := model_count_9522.trans row_count_9522.symm
def row_9523 : RowCertificate 9523 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9523 1) := by change Nat.Prime 362769163; exact ProjectEulerSolutions.P58PrimeNodes.p_362769163.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9523 2) := by change Nat.Prime 362750117; exact ProjectEulerSolutions.P58PrimeNodes.p_362750117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9523 3) := by change ¬ Nat.Prime 362731071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120910357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9523 : row 9523 = 2 := (row_9523.checked).trans rfl
theorem model_count_9523 : modelRow 9523 = 2 := by decide
theorem modelrow_agrees_9523 : modelRow 9523 = row 9523 := model_count_9523.trans row_count_9523.symm
def row_9524 : RowCertificate 9524 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9524 1) := by change ¬ Nat.Prime 362845353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120948451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9524 2) := by change ¬ Nat.Prime 362826305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72565261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9524 3) := by change ¬ Nat.Prime 362807257; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3740281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9524 : row 9524 = 0 := (row_9524.checked).trans rfl
theorem model_count_9524 : modelRow 9524 = 0 := by decide
theorem modelrow_agrees_9524 : modelRow 9524 = row 9524 := model_count_9524.trans row_count_9524.symm
def row_9525 : RowCertificate 9525 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9525 1) := by change Nat.Prime 362921551; exact ProjectEulerSolutions.P58PrimeNodes.p_362921551.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9525 2) := by change ¬ Nat.Prime 362902501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27915577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9525 3) := by change ¬ Nat.Prime 362883451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51840493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9525 : row 9525 = 1 := (row_9525.checked).trans rfl
theorem model_count_9525 : modelRow 9525 = 1 := by decide
theorem modelrow_agrees_9525 : modelRow 9525 = row 9525 := model_count_9525.trans row_count_9525.symm
def row_9526 : RowCertificate 9526 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9526 1) := by change Nat.Prime 362997757; exact ProjectEulerSolutions.P58PrimeNodes.p_362997757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9526 2) := by change ¬ Nat.Prime 362978705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72595741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9526 3) := by change ¬ Nat.Prime 362959653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120986551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9526 : row 9526 = 1 := (row_9526.checked).trans rfl
theorem model_count_9526 : modelRow 9526 = 1 := by decide
theorem modelrow_agrees_9526 : modelRow 9526 = row 9526 := model_count_9526.trans row_count_9526.symm
def row_9527 : RowCertificate 9527 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9527 1) := by change ¬ Nat.Prime 363073971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121024657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9527 2) := by change Nat.Prime 363054917; exact ProjectEulerSolutions.P58PrimeNodes.p_363054917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9527 3) := by change ¬ Nat.Prime 363035863; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2005723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9527 : row 9527 = 1 := (row_9527.checked).trans rfl
theorem model_count_9527 : modelRow 9527 = 1 := by decide
theorem modelrow_agrees_9527 : modelRow 9527 = row 9527 := model_count_9527.trans row_count_9527.symm
def row_9528 : RowCertificate 9528 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9528 1) := by change ¬ Nat.Prime 363150193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51878599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9528 2) := by change ¬ Nat.Prime 363131137; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8856857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9528 3) := by change ¬ Nat.Prime 363112081; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8444467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9528 : row 9528 = 0 := (row_9528.checked).trans rfl
theorem model_count_9528 : modelRow 9528 = 0 := by decide
theorem modelrow_agrees_9528 : modelRow 9528 = row 9528 := model_count_9528.trans row_count_9528.symm
def row_9529 : RowCertificate 9529 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9529 1) := by change ¬ Nat.Prime 363226423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51889489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9529 2) := by change ¬ Nat.Prime 363207365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72641473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9529 3) := by change ¬ Nat.Prime 363188307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121062769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9529 : row 9529 = 0 := (row_9529.checked).trans rfl
theorem model_count_9529 : modelRow 9529 = 0 := by decide
theorem modelrow_agrees_9529 : modelRow 9529 = row 9529 := model_count_9529.trans row_count_9529.symm
def row_9530 : RowCertificate 9530 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9530 1) := by change ¬ Nat.Prime 363302661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121100887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9530 2) := by change Nat.Prime 363283601; exact ProjectEulerSolutions.P58PrimeNodes.p_363283601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9530 3) := by change ¬ Nat.Prime 363264541; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11718211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9530 : row 9530 = 1 := (row_9530.checked).trans rfl
theorem model_count_9530 : modelRow 9530 = 1 := by decide
theorem modelrow_agrees_9530 : modelRow 9530 = row 9530 := model_count_9530.trans row_count_9530.symm
def row_9531 : RowCertificate 9531 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9531 1) := by change ¬ Nat.Prime 363378907; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4599733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9531 2) := by change ¬ Nat.Prime 363359845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72671969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9531 3) := by change ¬ Nat.Prime 363340783; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27949291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9531 : row 9531 = 0 := (row_9531.checked).trans rfl
theorem model_count_9531 : modelRow 9531 = 0 := by decide
theorem modelrow_agrees_9531 : modelRow 9531 = row 9531 := model_count_9531.trans row_count_9531.symm
def row_9532 : RowCertificate 9532 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9532 1) := by change ¬ Nat.Prime 363455161; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19129219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9532 2) := by change ¬ Nat.Prime 363436097; exact PrattCertificates.composite_of_product (factor := 761) (cofactor := 477577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9532 3) := by change ¬ Nat.Prime 363417033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121139011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9532 : row 9532 = 0 := (row_9532.checked).trans rfl
theorem model_count_9532 : modelRow 9532 = 0 := by decide
theorem modelrow_agrees_9532 : modelRow 9532 = row 9532 := model_count_9532.trans row_count_9532.symm
def row_9533 : RowCertificate 9533 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9533 1) := by change ¬ Nat.Prime 363531423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121177141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9533 2) := by change ¬ Nat.Prime 363512357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27962489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9533 3) := by change ¬ Nat.Prime 363493291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51927613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9533 : row 9533 = 0 := (row_9533.checked).trans rfl
theorem model_count_9533 : modelRow 9533 = 0 := by decide
theorem modelrow_agrees_9533 : modelRow 9533 = row 9533 := model_count_9533.trans row_count_9533.symm
def row_9534 : RowCertificate 9534 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9534 1) := by change ¬ Nat.Prime 363607693; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19137247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9534 2) := by change ¬ Nat.Prime 363588625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72717725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9534 3) := by change ¬ Nat.Prime 363569557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27966889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9534 : row 9534 = 0 := (row_9534.checked).trans rfl
theorem model_count_9534 : modelRow 9534 = 0 := by decide
theorem modelrow_agrees_9534 : modelRow 9534 = row 9534 := model_count_9534.trans row_count_9534.symm
def row_9535 : RowCertificate 9535 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9535 1) := by change ¬ Nat.Prime 363683971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51954853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9535 2) := by change ¬ Nat.Prime 363664901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21392053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9535 3) := by change ¬ Nat.Prime 363645831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121215277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9535 : row 9535 = 0 := (row_9535.checked).trans rfl
theorem model_count_9535 : modelRow 9535 = 0 := by decide
theorem modelrow_agrees_9535 : modelRow 9535 = row 9535 := model_count_9535.trans row_count_9535.symm
def row_9536 : RowCertificate 9536 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9536 1) := by change ¬ Nat.Prime 363760257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121253419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9536 2) := by change ¬ Nat.Prime 363741185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72748237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9536 3) := by change Nat.Prime 363722113; exact ProjectEulerSolutions.P58PrimeNodes.p_363722113.prime
    exact decide_eq_true prime
theorem row_count_9536 : row 9536 = 1 := (row_9536.checked).trans rfl
theorem model_count_9536 : modelRow 9536 = 1 := by decide
theorem modelrow_agrees_9536 : modelRow 9536 = row 9536 := model_count_9536.trans row_count_9536.symm
def row_9537 : RowCertificate 9537 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9537 1) := by change ¬ Nat.Prime 363836551; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27987427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9537 2) := by change ¬ Nat.Prime 363817477; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8873597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9537 3) := by change ¬ Nat.Prime 363798403; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3750499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9537 : row 9537 = 0 := (row_9537.checked).trans rfl
theorem model_count_9537 : modelRow 9537 = 0 := by decide
theorem modelrow_agrees_9537 : modelRow 9537 = row 9537 := model_count_9537.trans row_count_9537.symm
def row_9538 : RowCertificate 9538 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9538 1) := by change Nat.Prime 363912853; exact ProjectEulerSolutions.P58PrimeNodes.p_363912853.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9538 2) := by change ¬ Nat.Prime 363893777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27991829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9538 3) := by change ¬ Nat.Prime 363874701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121291567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9538 : row 9538 = 1 := (row_9538.checked).trans rfl
theorem model_count_9538 : modelRow 9538 = 1 := by decide
theorem modelrow_agrees_9538 : modelRow 9538 = row 9538 := model_count_9538.trans row_count_9538.symm
def row_9539 : RowCertificate 9539 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9539 1) := by change ¬ Nat.Prime 363989163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121329721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9539 2) := by change ¬ Nat.Prime 363970085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72794017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9539 3) := by change ¬ Nat.Prime 363951007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51993001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9539 : row 9539 = 0 := (row_9539.checked).trans rfl
theorem model_count_9539 : modelRow 9539 = 0 := by decide
theorem modelrow_agrees_9539 : modelRow 9539 = row 9539 := model_count_9539.trans row_count_9539.symm
def row_9540 : RowCertificate 9540 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9540 1) := by change ¬ Nat.Prime 364065481; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28005037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9540 2) := by change ¬ Nat.Prime 364046401; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4986937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9540 3) := by change ¬ Nat.Prime 364027321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52003903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9540 : row 9540 = 0 := (row_9540.checked).trans rfl
theorem model_count_9540 : modelRow 9540 = 0 := by decide
theorem modelrow_agrees_9540 : modelRow 9540 = row 9540 := model_count_9540.trans row_count_9540.symm
def row_9541 : RowCertificate 9541 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9541 1) := by change Nat.Prime 364141807; exact ProjectEulerSolutions.P58PrimeNodes.p_364141807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9541 2) := by change ¬ Nat.Prime 364122725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72824545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9541 3) := by change ¬ Nat.Prime 364103643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121367881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9541 : row 9541 = 1 := (row_9541.checked).trans rfl
theorem model_count_9541 : modelRow 9541 = 1 := by decide
theorem modelrow_agrees_9541 : modelRow 9541 = row 9541 := model_count_9541.trans row_count_9541.symm
def row_9542 : RowCertificate 9542 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9542 1) := by change ¬ Nat.Prime 364218141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121406047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9542 2) := by change Nat.Prime 364199057; exact ProjectEulerSolutions.P58PrimeNodes.p_364199057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9542 3) := by change ¬ Nat.Prime 364179973; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19167367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9542 : row 9542 = 1 := (row_9542.checked).trans rfl
theorem model_count_9542 : modelRow 9542 = 1 := by decide
theorem modelrow_agrees_9542 : modelRow 9542 = row 9542 := model_count_9542.trans row_count_9542.symm
def row_9543 : RowCertificate 9543 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9543 1) := by change ¬ Nat.Prime 364294483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52042069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9543 2) := by change ¬ Nat.Prime 364275397; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9845281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9543 3) := by change ¬ Nat.Prime 364256311; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8471077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9543 : row 9543 = 0 := (row_9543.checked).trans rfl
theorem model_count_9543 : modelRow 9543 = 0 := by decide
theorem modelrow_agrees_9543 : modelRow 9543 = row 9543 := model_count_9543.trans row_count_9543.symm
def row_9544 : RowCertificate 9544 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9544 1) := by change ¬ Nat.Prime 364370833; exact PrattCertificates.composite_of_product (factor := 883) (cofactor := 412651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9544 2) := by change ¬ Nat.Prime 364351745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72870349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9544 3) := by change ¬ Nat.Prime 364332657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121444219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9544 : row 9544 = 0 := (row_9544.checked).trans rfl
theorem model_count_9544 : modelRow 9544 = 0 := by decide
theorem modelrow_agrees_9544 : modelRow 9544 = row 9544 := model_count_9544.trans row_count_9544.symm
def row_9545 : RowCertificate 9545 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9545 1) := by change ¬ Nat.Prime 364447191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121482397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9545 2) := by change ¬ Nat.Prime 364428101; exact PrattCertificates.composite_of_product (factor := 7333) (cofactor := 49697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9545 3) := by change ¬ Nat.Prime 364409011; exact PrattCertificates.composite_of_product (factor := 1297) (cofactor := 280963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9545 : row 9545 = 0 := (row_9545.checked).trans rfl
theorem model_count_9545 : modelRow 9545 = 0 := by decide
theorem modelrow_agrees_9545 : modelRow 9545 = row 9545 := model_count_9545.trans row_count_9545.symm
def row_9546 : RowCertificate 9546 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9546 1) := by change Nat.Prime 364523557; exact ProjectEulerSolutions.P58PrimeNodes.p_364523557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9546 2) := by change ¬ Nat.Prime 364504465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72900893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9546 3) := by change ¬ Nat.Prime 364485373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52069339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9546 : row 9546 = 1 := (row_9546.checked).trans rfl
theorem model_count_9546 : modelRow 9546 = 1 := by decide
theorem modelrow_agrees_9546 : modelRow 9546 = row 9546 := model_count_9546.trans row_count_9546.symm
def row_9547 : RowCertificate 9547 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9547 1) := by change ¬ Nat.Prime 364599931; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4615189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9547 2) := by change ¬ Nat.Prime 364580837; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12571753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9547 3) := by change ¬ Nat.Prime 364561743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121520581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9547 : row 9547 = 0 := (row_9547.checked).trans rfl
theorem model_count_9547 : modelRow 9547 = 0 := by decide
theorem modelrow_agrees_9547 : modelRow 9547 = row 9547 := model_count_9547.trans row_count_9547.symm
def row_9548 : RowCertificate 9548 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9548 1) := by change ¬ Nat.Prime 364676313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121558771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9548 2) := by change ¬ Nat.Prime 364657217; exact PrattCertificates.composite_of_product (factor := 1993) (cofactor := 182969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9548 3) := by change ¬ Nat.Prime 364638121; exact PrattCertificates.composite_of_product (factor := 6529) (cofactor := 55849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9548 : row 9548 = 0 := (row_9548.checked).trans rfl
theorem model_count_9548 : modelRow 9548 = 0 := by decide
theorem modelrow_agrees_9548 : modelRow 9548 = row 9548 := model_count_9548.trans row_count_9548.symm
def row_9549 : RowCertificate 9549 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9549 1) := by change ¬ Nat.Prime 364752703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52107529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9549 2) := by change ¬ Nat.Prime 364733605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72946721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9549 3) := by change ¬ Nat.Prime 364714507; exact PrattCertificates.composite_of_product (factor := 3217) (cofactor := 113371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9549 : row 9549 = 0 := (row_9549.checked).trans rfl
theorem model_count_9549 : modelRow 9549 = 0 := by decide
theorem modelrow_agrees_9549 : modelRow 9549 = row 9549 := model_count_9549.trans row_count_9549.symm
def row_9550 : RowCertificate 9550 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9550 1) := by change ¬ Nat.Prime 364829101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52118443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9550 2) := by change Nat.Prime 364810001; exact ProjectEulerSolutions.P58PrimeNodes.p_364810001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9550 3) := by change ¬ Nat.Prime 364790901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121596967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9550 : row 9550 = 1 := (row_9550.checked).trans rfl
theorem model_count_9550 : modelRow 9550 = 1 := by decide
theorem modelrow_agrees_9550 : modelRow 9550 = row 9550 := model_count_9550.trans row_count_9550.symm
def row_9551 : RowCertificate 9551 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9551 1) := by change ¬ Nat.Prime 364905507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121635169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9551 2) := by change ¬ Nat.Prime 364886405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72977281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9551 3) := by change ¬ Nat.Prime 364867303; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11769913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9551 : row 9551 = 0 := (row_9551.checked).trans rfl
theorem model_count_9551 : modelRow 9551 = 0 := by decide
theorem modelrow_agrees_9551 : modelRow 9551 = row 9551 := model_count_9551.trans row_count_9551.symm
def row_9552 : RowCertificate 9552 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9552 1) := by change ¬ Nat.Prime 364981921; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 667243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9552 2) := by change ¬ Nat.Prime 364962817; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21468401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9552 3) := by change ¬ Nat.Prime 364943713; exact PrattCertificates.composite_of_product (factor := 823) (cofactor := 443431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9552 : row 9552 = 0 := (row_9552.checked).trans rfl
theorem model_count_9552 : modelRow 9552 = 0 := by decide
theorem modelrow_agrees_9552 : modelRow 9552 = row 9552 := model_count_9552.trans row_count_9552.symm
def row_9553 : RowCertificate 9553 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9553 1) := by change ¬ Nat.Prime 365058343; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28081411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9553 2) := by change ¬ Nat.Prime 365039237; exact PrattCertificates.composite_of_product (factor := 4001) (cofactor := 91237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9553 3) := by change ¬ Nat.Prime 365020131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121673377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9553 : row 9553 = 0 := (row_9553.checked).trans rfl
theorem model_count_9553 : modelRow 9553 = 0 := by decide
theorem modelrow_agrees_9553 : modelRow 9553 = row 9553 := model_count_9553.trans row_count_9553.symm
def row_9554 : RowCertificate 9554 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9554 1) := by change ¬ Nat.Prime 365134773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121711591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9554 2) := by change ¬ Nat.Prime 365115665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73023133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9554 3) := by change ¬ Nat.Prime 365096557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52156651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9554 : row 9554 = 0 := (row_9554.checked).trans rfl
theorem model_count_9554 : modelRow 9554 = 0 := by decide
theorem modelrow_agrees_9554 : modelRow 9554 = row 9554 := model_count_9554.trans row_count_9554.symm
def row_9555 : RowCertificate 9555 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9555 1) := by change Nat.Prime 365211211; exact ProjectEulerSolutions.P58PrimeNodes.p_365211211.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9555 2) := by change ¬ Nat.Prime 365192101; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6890417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9555 3) := by change ¬ Nat.Prime 365172991; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 326923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9555 : row 9555 = 1 := (row_9555.checked).trans rfl
theorem model_count_9555 : modelRow 9555 = 1 := by decide
theorem modelrow_agrees_9555 : modelRow 9555 = row 9555 := model_count_9555.trans row_count_9555.symm
def row_9556 : RowCertificate 9556 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9556 1) := by change ¬ Nat.Prime 365287657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52183951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9556 2) := by change ¬ Nat.Prime 365268545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73053709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9556 3) := by change ¬ Nat.Prime 365249433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121749811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9556 : row 9556 = 0 := (row_9556.checked).trans rfl
theorem model_count_9556 : modelRow 9556 = 0 := by decide
theorem modelrow_agrees_9556 : modelRow 9556 = row 9556 := model_count_9556.trans row_count_9556.symm
def row_9557 : RowCertificate 9557 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9557 1) := by change ¬ Nat.Prime 365364111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121788037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9557 2) := by change ¬ Nat.Prime 365344997; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1595393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9557 3) := by change ¬ Nat.Prime 365325883; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28101991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9557 : row 9557 = 0 := (row_9557.checked).trans rfl
theorem model_count_9557 : modelRow 9557 = 0 := by decide
theorem modelrow_agrees_9557 : modelRow 9557 = row 9557 := model_count_9557.trans row_count_9557.symm
def row_9558 : RowCertificate 9558 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9558 1) := by change Nat.Prime 365440573; exact ProjectEulerSolutions.P58PrimeNodes.p_365440573.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9558 2) := by change ¬ Nat.Prime 365421457; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2452493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9558 3) := by change Nat.Prime 365402341; exact ProjectEulerSolutions.P58PrimeNodes.p_365402341.prime
    exact decide_eq_true prime
theorem row_count_9558 : row 9558 = 2 := (row_9558.checked).trans rfl
theorem model_count_9558 : modelRow 9558 = 2 := by decide
theorem modelrow_agrees_9558 : modelRow 9558 = row 9558 := model_count_9558.trans row_count_9558.symm
def row_9559 : RowCertificate 9559 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9559 1) := by change ¬ Nat.Prime 365517043; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9878839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9559 2) := by change ¬ Nat.Prime 365497925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73099585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9559 3) := by change ¬ Nat.Prime 365478807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121826269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9559 : row 9559 = 0 := (row_9559.checked).trans rfl
theorem model_count_9559 : modelRow 9559 = 0 := by decide
theorem modelrow_agrees_9559 : modelRow 9559 = row 9559 := model_count_9559.trans row_count_9559.symm
def row_9560 : RowCertificate 9560 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9560 1) := by change ¬ Nat.Prime 365593521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121864507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9560 2) := by change Nat.Prime 365574401; exact ProjectEulerSolutions.P58PrimeNodes.p_365574401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9560 3) := by change ¬ Nat.Prime 365555281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52222183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9560 : row 9560 = 1 := (row_9560.checked).trans rfl
theorem model_count_9560 : modelRow 9560 = 1 := by decide
theorem modelrow_agrees_9560 : modelRow 9560 = row 9560 := model_count_9560.trans row_count_9560.symm
def row_9561 : RowCertificate 9561 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9561 1) := by change ¬ Nat.Prime 365670007; exact PrattCertificates.composite_of_product (factor := 11551) (cofactor := 31657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9561 2) := by change ¬ Nat.Prime 365650885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73130177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9561 3) := by change ¬ Nat.Prime 365631763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52233109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9561 : row 9561 = 0 := (row_9561.checked).trans rfl
theorem model_count_9561 : modelRow 9561 = 0 := by decide
theorem modelrow_agrees_9561 : modelRow 9561 = row 9561 := model_count_9561.trans row_count_9561.symm
def row_9562 : RowCertificate 9562 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9562 1) := by change ¬ Nat.Prime 365746501; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5458903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9562 2) := by change Nat.Prime 365727377; exact ProjectEulerSolutions.P58PrimeNodes.p_365727377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9562 3) := by change ¬ Nat.Prime 365708253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121902751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9562 : row 9562 = 1 := (row_9562.checked).trans rfl
theorem model_count_9562 : modelRow 9562 = 1 := by decide
theorem modelrow_agrees_9562 : modelRow 9562 = row 9562 := model_count_9562.trans row_count_9562.symm
def row_9563 : RowCertificate 9563 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9563 1) := by change ¬ Nat.Prime 365823003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121941001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9563 2) := by change ¬ Nat.Prime 365803877; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2329961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9563 3) := by change ¬ Nat.Prime 365784751; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19251829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9563 : row 9563 = 0 := (row_9563.checked).trans rfl
theorem model_count_9563 : modelRow 9563 = 0 := by decide
theorem modelrow_agrees_9563 : modelRow 9563 = row 9563 := model_count_9563.trans row_count_9563.symm
def row_9564 : RowCertificate 9564 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9564 1) := by change ¬ Nat.Prime 365899513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52271359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9564 2) := by change ¬ Nat.Prime 365880385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73176077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9564 3) := by change Nat.Prime 365861257; exact ProjectEulerSolutions.P58PrimeNodes.p_365861257.prime
    exact decide_eq_true prime
theorem row_count_9564 : row 9564 = 1 := (row_9564.checked).trans rfl
theorem model_count_9564 : modelRow 9564 = 1 := by decide
theorem modelrow_agrees_9564 : modelRow 9564 = row 9564 := model_count_9564.trans row_count_9564.symm
def row_9565 : RowCertificate 9565 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9565 1) := by change ¬ Nat.Prime 365976031; exact PrattCertificates.composite_of_product (factor := 3469) (cofactor := 105499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9565 2) := by change Nat.Prime 365956901; exact ProjectEulerSolutions.P58PrimeNodes.p_365956901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9565 3) := by change ¬ Nat.Prime 365937771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121979257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9565 : row 9565 = 1 := (row_9565.checked).trans rfl
theorem model_count_9565 : modelRow 9565 = 1 := by decide
theorem modelrow_agrees_9565 : modelRow 9565 = row 9565 := model_count_9565.trans row_count_9565.symm
def row_9566 : RowCertificate 9566 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9566 1) := by change ¬ Nat.Prime 366052557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122017519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9566 2) := by change ¬ Nat.Prime 366033425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73206685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9566 3) := by change ¬ Nat.Prime 366014293; exact PrattCertificates.composite_of_product (factor := 14929) (cofactor := 24517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9566 : row 9566 = 0 := (row_9566.checked).trans rfl
theorem model_count_9566 : modelRow 9566 = 0 := by decide
theorem modelrow_agrees_9566 : modelRow 9566 = row 9566 := model_count_9566.trans row_count_9566.symm
def row_9567 : RowCertificate 9567 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9567 1) := by change ¬ Nat.Prime 366129091; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5015467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9567 2) := by change ¬ Nat.Prime 366109957; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 1154921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9567 3) := by change ¬ Nat.Prime 366090823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52298689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9567 : row 9567 = 0 := (row_9567.checked).trans rfl
theorem model_count_9567 : modelRow 9567 = 0 := by decide
theorem modelrow_agrees_9567 : modelRow 9567 = row 9567 := model_count_9567.trans row_count_9567.symm
def row_9568 : RowCertificate 9568 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9568 1) := by change Nat.Prime 366205633; exact ProjectEulerSolutions.P58PrimeNodes.p_366205633.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9568 2) := by change Nat.Prime 366186497; exact ProjectEulerSolutions.P58PrimeNodes.p_366186497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9568 3) := by change ¬ Nat.Prime 366167361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122055787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9568 : row 9568 = 2 := (row_9568.checked).trans rfl
theorem model_count_9568 : modelRow 9568 = 2 := by decide
theorem modelrow_agrees_9568 : modelRow 9568 = row 9568 := model_count_9568.trans row_count_9568.symm
def row_9569 : RowCertificate 9569 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9569 1) := by change ¬ Nat.Prime 366282183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122094061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9569 2) := by change ¬ Nat.Prime 366263045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73252609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9569 3) := by change ¬ Nat.Prime 366243907; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2023447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9569 : row 9569 = 0 := (row_9569.checked).trans rfl
theorem model_count_9569 : modelRow 9569 = 0 := by decide
theorem modelrow_agrees_9569 : modelRow 9569 = row 9569 := model_count_9569.trans row_count_9569.symm
def row_9570 : RowCertificate 9570 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9570 1) := by change ¬ Nat.Prime 366358741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52336963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9570 2) := by change Nat.Prime 366339601; exact ProjectEulerSolutions.P58PrimeNodes.p_366339601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9570 3) := by change ¬ Nat.Prime 366320461; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28178497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9570 : row 9570 = 1 := (row_9570.checked).trans rfl
theorem model_count_9570 : modelRow 9570 = 1 := by decide
theorem modelrow_agrees_9570 : modelRow 9570 = row 9570 := model_count_9570.trans row_count_9570.symm
def row_9571 : RowCertificate 9571 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9571 1) := by change ¬ Nat.Prime 366435307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52347901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9571 2) := by change ¬ Nat.Prime 366416165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73283233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9571 3) := by change ¬ Nat.Prime 366397023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122132341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9571 : row 9571 = 0 := (row_9571.checked).trans rfl
theorem model_count_9571 : modelRow 9571 = 0 := by decide
theorem modelrow_agrees_9571 : modelRow 9571 = row 9571 := model_count_9571.trans row_count_9571.symm
def row_9572 : RowCertificate 9572 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9572 1) := by change ¬ Nat.Prime 366511881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122170627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9572 2) := by change ¬ Nat.Prime 366492737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28191749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9572 3) := by change Nat.Prime 366473593; exact ProjectEulerSolutions.P58PrimeNodes.p_366473593.prime
    exact decide_eq_true prime
theorem row_count_9572 : row 9572 = 1 := (row_9572.checked).trans rfl
theorem model_count_9572 : modelRow 9572 = 1 := by decide
theorem modelrow_agrees_9572 : modelRow 9572 = row 9572 := model_count_9572.trans row_count_9572.symm
def row_9573 : RowCertificate 9573 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9573 1) := by change ¬ Nat.Prime 366588463; exact PrattCertificates.composite_of_product (factor := 6871) (cofactor := 53353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9573 2) := by change ¬ Nat.Prime 366569317; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21562901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9573 3) := by change ¬ Nat.Prime 366550171; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28196167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9573 : row 9573 = 0 := (row_9573.checked).trans rfl
theorem model_count_9573 : modelRow 9573 = 0 := by decide
theorem modelrow_agrees_9573 : modelRow 9573 = row 9573 := model_count_9573.trans row_count_9573.symm
def row_9574 : RowCertificate 9574 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9574 1) := by change Nat.Prime 366665053; exact ProjectEulerSolutions.P58PrimeNodes.p_366665053.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9574 2) := by change ¬ Nat.Prime 366645905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73329181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9574 3) := by change ¬ Nat.Prime 366626757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122208919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9574 : row 9574 = 1 := (row_9574.checked).trans rfl
theorem model_count_9574 : modelRow 9574 = 1 := by decide
theorem modelrow_agrees_9574 : modelRow 9574 = row 9574 := model_count_9574.trans row_count_9574.symm
def row_9575 : RowCertificate 9575 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9575 1) := by change ¬ Nat.Prime 366741651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122247217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9575 2) := by change Nat.Prime 366722501; exact ProjectEulerSolutions.P58PrimeNodes.p_366722501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9575 3) := by change ¬ Nat.Prime 366703351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52386193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9575 : row 9575 = 1 := (row_9575.checked).trans rfl
theorem model_count_9575 : modelRow 9575 = 1 := by decide
theorem modelrow_agrees_9575 : modelRow 9575 = row 9575 := model_count_9575.trans row_count_9575.symm
def row_9576 : RowCertificate 9576 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9576 1) := by change ¬ Nat.Prime 366818257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28216789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9576 2) := by change ¬ Nat.Prime 366799105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73359821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9576 3) := by change Nat.Prime 366779953; exact ProjectEulerSolutions.P58PrimeNodes.p_366779953.prime
    exact decide_eq_true prime
theorem row_count_9576 : row 9576 = 1 := (row_9576.checked).trans rfl
theorem model_count_9576 : modelRow 9576 = 1 := by decide
theorem modelrow_agrees_9576 : modelRow 9576 = row 9576 := model_count_9576.trans row_count_9576.symm
def row_9577 : RowCertificate 9577 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9577 1) := by change ¬ Nat.Prime 366894871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52413553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9577 2) := by change ¬ Nat.Prime 366875717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28221209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9577 3) := by change ¬ Nat.Prime 366856563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122285521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9577 : row 9577 = 0 := (row_9577.checked).trans rfl
theorem model_count_9577 : modelRow 9577 = 0 := by decide
theorem modelrow_agrees_9577 : modelRow 9577 = row 9577 := model_count_9577.trans row_count_9577.symm
def row_9578 : RowCertificate 9578 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9578 1) := by change ¬ Nat.Prime 366971493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122323831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9578 2) := by change ¬ Nat.Prime 366952337; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8950057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9578 3) := by change ¬ Nat.Prime 366933181; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9917113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9578 : row 9578 = 0 := (row_9578.checked).trans rfl
theorem model_count_9578 : modelRow 9578 = 0 := by decide
theorem modelrow_agrees_9578 : modelRow 9578 = row 9578 := model_count_9578.trans row_count_9578.symm
def row_9579 : RowCertificate 9579 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9579 1) := by change ¬ Nat.Prime 367048123; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28234471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9579 2) := by change ¬ Nat.Prime 367028965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73405793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9579 3) := by change ¬ Nat.Prime 367009807; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2889841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9579 : row 9579 = 0 := (row_9579.checked).trans rfl
theorem model_count_9579 : modelRow 9579 = 0 := by decide
theorem modelrow_agrees_9579 : modelRow 9579 = row 9579 := model_count_9579.trans row_count_9579.symm
def row_9580 : RowCertificate 9580 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9580 1) := by change ¬ Nat.Prime 367124761; exact PrattCertificates.composite_of_product (factor := 3877) (cofactor := 94693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9580 2) := by change ¬ Nat.Prime 367105601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9921773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9580 3) := by change ¬ Nat.Prime 367086441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122362147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9580 : row 9580 = 0 := (row_9580.checked).trans rfl
theorem model_count_9580 : modelRow 9580 = 0 := by decide
theorem modelrow_agrees_9580 : modelRow 9580 = row 9580 := model_count_9580.trans row_count_9580.symm
def row_9581 : RowCertificate 9581 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9581 1) := by change ¬ Nat.Prime 367201407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122400469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9581 2) := by change ¬ Nat.Prime 367182245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73436449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9581 3) := by change ¬ Nat.Prime 367163083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52451869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9581 : row 9581 = 0 := (row_9581.checked).trans rfl
theorem model_count_9581 : modelRow 9581 = 0 := by decide
theorem modelrow_agrees_9581 : modelRow 9581 = row 9581 := model_count_9581.trans row_count_9581.symm
def row_9582 : RowCertificate 9582 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9582 1) := by change Nat.Prime 367278061; exact ProjectEulerSolutions.P58PrimeNodes.p_367278061.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9582 2) := by change ¬ Nat.Prime 367258897; exact PrattCertificates.composite_of_product (factor := 12589) (cofactor := 29173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9582 3) := by change ¬ Nat.Prime 367239733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52462819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9582 : row 9582 = 1 := (row_9582.checked).trans rfl
theorem model_count_9582 : modelRow 9582 = 1 := by decide
theorem modelrow_agrees_9582 : modelRow 9582 = row 9582 := model_count_9582.trans row_count_9582.symm
def row_9583 : RowCertificate 9583 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9583 1) := by change ¬ Nat.Prime 367354723; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2339839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9583 2) := by change Nat.Prime 367335557; exact ProjectEulerSolutions.P58PrimeNodes.p_367335557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9583 3) := by change ¬ Nat.Prime 367316391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122438797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9583 : row 9583 = 1 := (row_9583.checked).trans rfl
theorem model_count_9583 : modelRow 9583 = 1 := by decide
theorem modelrow_agrees_9583 : modelRow 9583 = row 9583 := model_count_9583.trans row_count_9583.symm
def row_9584 : RowCertificate 9584 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9584 1) := by change ¬ Nat.Prime 367431393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122477131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9584 2) := by change ¬ Nat.Prime 367412225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73482445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9584 3) := by change ¬ Nat.Prime 367393057; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6022837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9584 : row 9584 = 0 := (row_9584.checked).trans rfl
theorem model_count_9584 : modelRow 9584 = 0 := by decide
theorem modelrow_agrees_9584 : modelRow 9584 = row 9584 := model_count_9584.trans row_count_9584.symm
def row_9585 : RowCertificate 9585 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9585 1) := by change ¬ Nat.Prime 367508071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52501153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9585 2) := by change ¬ Nat.Prime 367488901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28268377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9585 3) := by change ¬ Nat.Prime 367469731; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 898459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9585 : row 9585 = 0 := (row_9585.checked).trans rfl
theorem model_count_9585 : modelRow 9585 = 0 := by decide
theorem modelrow_agrees_9585 : modelRow 9585 = row 9585 := model_count_9585.trans row_count_9585.symm
def row_9586 : RowCertificate 9586 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9586 1) := by change Nat.Prime 367584757; exact ProjectEulerSolutions.P58PrimeNodes.p_367584757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9586 2) := by change ¬ Nat.Prime 367565585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73513117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9586 3) := by change ¬ Nat.Prime 367546413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122515471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9586 : row 9586 = 1 := (row_9586.checked).trans rfl
theorem model_count_9586 : modelRow 9586 = 1 := by decide
theorem modelrow_agrees_9586 : modelRow 9586 = row 9586 := model_count_9586.trans row_count_9586.symm
def row_9587 : RowCertificate 9587 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9587 1) := by change ¬ Nat.Prime 367661451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122553817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9587 2) := by change ¬ Nat.Prime 367642277; exact PrattCertificates.composite_of_product (factor := 4217) (cofactor := 87181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9587 3) := by change ¬ Nat.Prime 367623103; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4653457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9587 : row 9587 = 0 := (row_9587.checked).trans rfl
theorem model_count_9587 : modelRow 9587 = 0 := by decide
theorem modelrow_agrees_9587 : modelRow 9587 = row 9587 := model_count_9587.trans row_count_9587.symm
def row_9588 : RowCertificate 9588 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9588 1) := by change ¬ Nat.Prime 367738153; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9938869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9588 2) := by change ¬ Nat.Prime 367718977; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 945293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9588 3) := by change ¬ Nat.Prime 367699801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52528543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9588 : row 9588 = 0 := (row_9588.checked).trans rfl
theorem model_count_9588 : modelRow 9588 = 0 := by decide
theorem modelrow_agrees_9588 : modelRow 9588 = row 9588 := model_count_9588.trans row_count_9588.symm
def row_9589 : RowCertificate 9589 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9589 1) := by change ¬ Nat.Prime 367814863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28293451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9589 2) := by change ¬ Nat.Prime 367795685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73559137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9589 3) := by change ¬ Nat.Prime 367776507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122592169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9589 : row 9589 = 0 := (row_9589.checked).trans rfl
theorem model_count_9589 : modelRow 9589 = 0 := by decide
theorem modelrow_agrees_9589 : modelRow 9589 = row 9589 := model_count_9589.trans row_count_9589.symm
def row_9590 : RowCertificate 9590 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9590 1) := by change ¬ Nat.Prime 367891581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122630527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9590 2) := by change ¬ Nat.Prime 367872401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28297877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9590 3) := by change ¬ Nat.Prime 367853221; exact PrattCertificates.composite_of_product (factor := 17491) (cofactor := 21031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9590 : row 9590 = 0 := (row_9590.checked).trans rfl
theorem model_count_9590 : modelRow 9590 = 0 := by decide
theorem modelrow_agrees_9590 : modelRow 9590 = row 9590 := model_count_9590.trans row_count_9590.symm
def row_9591 : RowCertificate 9591 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9591 1) := by change ¬ Nat.Prime 367968307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52566901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9591 2) := by change ¬ Nat.Prime 367949125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73589825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9591 3) := by change ¬ Nat.Prime 367929943; exact PrattCertificates.composite_of_product (factor := 1999) (cofactor := 184057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9591 : row 9591 = 0 := (row_9591.checked).trans rfl
theorem model_count_9591 : modelRow 9591 = 0 := by decide
theorem modelrow_agrees_9591 : modelRow 9591 = row 9591 := model_count_9591.trans row_count_9591.symm
def row_9592 : RowCertificate 9592 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9592 1) := by change ¬ Nat.Prime 368045041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52577863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9592 2) := by change ¬ Nat.Prime 368025857; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3794081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9592 3) := by change ¬ Nat.Prime 368006673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122668891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9592 : row 9592 = 0 := (row_9592.checked).trans rfl
theorem model_count_9592 : modelRow 9592 = 0 := by decide
theorem modelrow_agrees_9592 : modelRow 9592 = row 9592 := model_count_9592.trans row_count_9592.symm
def row_9593 : RowCertificate 9593 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9593 1) := by change ¬ Nat.Prime 368121783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122707261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9593 2) := by change ¬ Nat.Prime 368102597; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12693193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9593 3) := by change ¬ Nat.Prime 368083411; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 672913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9593 : row 9593 = 0 := (row_9593.checked).trans rfl
theorem model_count_9593 : modelRow 9593 = 0 := by decide
theorem modelrow_agrees_9593 : modelRow 9593 = row 9593 := model_count_9593.trans row_count_9593.symm
def row_9594 : RowCertificate 9594 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9594 1) := by change Nat.Prime 368198533; exact ProjectEulerSolutions.P58PrimeNodes.p_368198533.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9594 2) := by change ¬ Nat.Prime 368179345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73635869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9594 3) := by change ¬ Nat.Prime 368160157; exact PrattCertificates.composite_of_product (factor := 8863) (cofactor := 41539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9594 : row 9594 = 1 := (row_9594.checked).trans rfl
theorem model_count_9594 : modelRow 9594 = 1 := by decide
theorem modelrow_agrees_9594 : modelRow 9594 = row 9594 := model_count_9594.trans row_count_9594.symm
def row_9595 : RowCertificate 9595 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9595 1) := by change ¬ Nat.Prime 368275291; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5044867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9595 2) := by change ¬ Nat.Prime 368256101; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 4137709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9595 3) := by change ¬ Nat.Prime 368236911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122745637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9595 : row 9595 = 0 := (row_9595.checked).trans rfl
theorem model_count_9595 : modelRow 9595 = 0 := by decide
theorem modelrow_agrees_9595 : modelRow 9595 = row 9595 := model_count_9595.trans row_count_9595.symm
def row_9596 : RowCertificate 9596 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9596 1) := by change ¬ Nat.Prime 368352057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122784019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9596 2) := by change ¬ Nat.Prime 368332865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73666573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9596 3) := by change ¬ Nat.Prime 368313673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52616239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9596 : row 9596 = 0 := (row_9596.checked).trans rfl
theorem model_count_9596 : modelRow 9596 = 0 := by decide
theorem modelrow_agrees_9596 : modelRow 9596 = row 9596 := model_count_9596.trans row_count_9596.symm
def row_9597 : RowCertificate 9597 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9597 1) := by change ¬ Nat.Prime 368428831; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11884801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9597 2) := by change Nat.Prime 368409637; exact ProjectEulerSolutions.P58PrimeNodes.p_368409637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9597 3) := by change ¬ Nat.Prime 368390443; exact PrattCertificates.composite_of_product (factor := 15559) (cofactor := 23677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9597 : row 9597 = 1 := (row_9597.checked).trans rfl
theorem model_count_9597 : modelRow 9597 = 1 := by decide
theorem modelrow_agrees_9597 : modelRow 9597 = row 9597 := model_count_9597.trans row_count_9597.symm
def row_9598 : RowCertificate 9598 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9598 1) := by change ¬ Nat.Prime 368505613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52643659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9598 2) := by change ¬ Nat.Prime 368486417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28345109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9598 3) := by change ¬ Nat.Prime 368467221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122822407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9598 : row 9598 = 0 := (row_9598.checked).trans rfl
theorem model_count_9598 : modelRow 9598 = 0 := by decide
theorem modelrow_agrees_9598 : modelRow 9598 = row 9598 := model_count_9598.trans row_count_9598.symm
def row_9599 : RowCertificate 9599 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9599 1) := by change ¬ Nat.Prime 368582403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122860801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9599 2) := by change ¬ Nat.Prime 368563205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73712641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9599 3) := by change ¬ Nat.Prime 368544007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28349539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9599 : row 9599 = 0 := (row_9599.checked).trans rfl
theorem model_count_9599 : modelRow 9599 = 0 := by decide
theorem modelrow_agrees_9599 : modelRow 9599 = row 9599 := model_count_9599.trans row_count_9599.symm
def row_9600 : RowCertificate 9600 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9600 1) := by change ¬ Nat.Prime 368659201; exact PrattCertificates.composite_of_product (factor := 12637) (cofactor := 29173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9600 2) := by change ¬ Nat.Prime 368640001; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3649901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9600 3) := by change ¬ Nat.Prime 368620801; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5501803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9600 : row 9600 = 0 := (row_9600.checked).trans rfl
theorem model_count_9600 : modelRow 9600 = 0 := by decide
theorem modelrow_agrees_9600 : modelRow 9600 = row 9600 := model_count_9600.trans row_count_9600.symm
private def atomic_9473 : CountSegment 1 10 13120 9472 9473 3934 3934 := CountSegment.single (first := 9472) (initial := 3934) row_count_9473 (by decide)
private theorem atomic_9473_agreement : RowAgreement 9472 9473 := RowAgreement.single (first := 9472) modelrow_agrees_9473
private def atomic_9474 : CountSegment 1 10 13120 9473 9474 3934 3936 := CountSegment.single (first := 9473) (initial := 3934) row_count_9474 (by decide)
private theorem atomic_9474_agreement : RowAgreement 9473 9474 := RowAgreement.single (first := 9473) modelrow_agrees_9474
private def atomic_9475 : CountSegment 1 10 13120 9474 9475 3936 3936 := CountSegment.single (first := 9474) (initial := 3936) row_count_9475 (by decide)
private theorem atomic_9475_agreement : RowAgreement 9474 9475 := RowAgreement.single (first := 9474) modelrow_agrees_9475
private def atomic_9476 : CountSegment 1 10 13120 9475 9476 3936 3936 := CountSegment.single (first := 9475) (initial := 3936) row_count_9476 (by decide)
private theorem atomic_9476_agreement : RowAgreement 9475 9476 := RowAgreement.single (first := 9475) modelrow_agrees_9476
private def atomic_9477 : CountSegment 1 10 13120 9476 9477 3936 3936 := CountSegment.single (first := 9476) (initial := 3936) row_count_9477 (by decide)
private theorem atomic_9477_agreement : RowAgreement 9476 9477 := RowAgreement.single (first := 9476) modelrow_agrees_9477
private def atomic_9478 : CountSegment 1 10 13120 9477 9478 3936 3936 := CountSegment.single (first := 9477) (initial := 3936) row_count_9478 (by decide)
private theorem atomic_9478_agreement : RowAgreement 9477 9478 := RowAgreement.single (first := 9477) modelrow_agrees_9478
private def atomic_9479 : CountSegment 1 10 13120 9478 9479 3936 3936 := CountSegment.single (first := 9478) (initial := 3936) row_count_9479 (by decide)
private theorem atomic_9479_agreement : RowAgreement 9478 9479 := RowAgreement.single (first := 9478) modelrow_agrees_9479
private def atomic_9480 : CountSegment 1 10 13120 9479 9480 3936 3936 := CountSegment.single (first := 9479) (initial := 3936) row_count_9480 (by decide)
private theorem atomic_9480_agreement : RowAgreement 9479 9480 := RowAgreement.single (first := 9479) modelrow_agrees_9480
private def atomic_9481 : CountSegment 1 10 13120 9480 9481 3936 3936 := CountSegment.single (first := 9480) (initial := 3936) row_count_9481 (by decide)
private theorem atomic_9481_agreement : RowAgreement 9480 9481 := RowAgreement.single (first := 9480) modelrow_agrees_9481
private def atomic_9482 : CountSegment 1 10 13120 9481 9482 3936 3936 := CountSegment.single (first := 9481) (initial := 3936) row_count_9482 (by decide)
private theorem atomic_9482_agreement : RowAgreement 9481 9482 := RowAgreement.single (first := 9481) modelrow_agrees_9482
private def atomic_9483 : CountSegment 1 10 13120 9482 9483 3936 3937 := CountSegment.single (first := 9482) (initial := 3936) row_count_9483 (by decide)
private theorem atomic_9483_agreement : RowAgreement 9482 9483 := RowAgreement.single (first := 9482) modelrow_agrees_9483
private def atomic_9484 : CountSegment 1 10 13120 9483 9484 3937 3937 := CountSegment.single (first := 9483) (initial := 3937) row_count_9484 (by decide)
private theorem atomic_9484_agreement : RowAgreement 9483 9484 := RowAgreement.single (first := 9483) modelrow_agrees_9484
private def atomic_9485 : CountSegment 1 10 13120 9484 9485 3937 3938 := CountSegment.single (first := 9484) (initial := 3937) row_count_9485 (by decide)
private theorem atomic_9485_agreement : RowAgreement 9484 9485 := RowAgreement.single (first := 9484) modelrow_agrees_9485
private def atomic_9486 : CountSegment 1 10 13120 9485 9486 3938 3938 := CountSegment.single (first := 9485) (initial := 3938) row_count_9486 (by decide)
private theorem atomic_9486_agreement : RowAgreement 9485 9486 := RowAgreement.single (first := 9485) modelrow_agrees_9486
private def atomic_9487 : CountSegment 1 10 13120 9486 9487 3938 3938 := CountSegment.single (first := 9486) (initial := 3938) row_count_9487 (by decide)
private theorem atomic_9487_agreement : RowAgreement 9486 9487 := RowAgreement.single (first := 9486) modelrow_agrees_9487
private def atomic_9488 : CountSegment 1 10 13120 9487 9488 3938 3938 := CountSegment.single (first := 9487) (initial := 3938) row_count_9488 (by decide)
private theorem atomic_9488_agreement : RowAgreement 9487 9488 := RowAgreement.single (first := 9487) modelrow_agrees_9488
private def atomic_9489 : CountSegment 1 10 13120 9488 9489 3938 3939 := CountSegment.single (first := 9488) (initial := 3938) row_count_9489 (by decide)
private theorem atomic_9489_agreement : RowAgreement 9488 9489 := RowAgreement.single (first := 9488) modelrow_agrees_9489
private def atomic_9490 : CountSegment 1 10 13120 9489 9490 3939 3941 := CountSegment.single (first := 9489) (initial := 3939) row_count_9490 (by decide)
private theorem atomic_9490_agreement : RowAgreement 9489 9490 := RowAgreement.single (first := 9489) modelrow_agrees_9490
private def atomic_9491 : CountSegment 1 10 13120 9490 9491 3941 3941 := CountSegment.single (first := 9490) (initial := 3941) row_count_9491 (by decide)
private theorem atomic_9491_agreement : RowAgreement 9490 9491 := RowAgreement.single (first := 9490) modelrow_agrees_9491
private def atomic_9492 : CountSegment 1 10 13120 9491 9492 3941 3942 := CountSegment.single (first := 9491) (initial := 3941) row_count_9492 (by decide)
private theorem atomic_9492_agreement : RowAgreement 9491 9492 := RowAgreement.single (first := 9491) modelrow_agrees_9492
private def atomic_9493 : CountSegment 1 10 13120 9492 9493 3942 3942 := CountSegment.single (first := 9492) (initial := 3942) row_count_9493 (by decide)
private theorem atomic_9493_agreement : RowAgreement 9492 9493 := RowAgreement.single (first := 9492) modelrow_agrees_9493
private def atomic_9494 : CountSegment 1 10 13120 9493 9494 3942 3943 := CountSegment.single (first := 9493) (initial := 3942) row_count_9494 (by decide)
private theorem atomic_9494_agreement : RowAgreement 9493 9494 := RowAgreement.single (first := 9493) modelrow_agrees_9494
private def atomic_9495 : CountSegment 1 10 13120 9494 9495 3943 3943 := CountSegment.single (first := 9494) (initial := 3943) row_count_9495 (by decide)
private theorem atomic_9495_agreement : RowAgreement 9494 9495 := RowAgreement.single (first := 9494) modelrow_agrees_9495
private def atomic_9496 : CountSegment 1 10 13120 9495 9496 3943 3943 := CountSegment.single (first := 9495) (initial := 3943) row_count_9496 (by decide)
private theorem atomic_9496_agreement : RowAgreement 9495 9496 := RowAgreement.single (first := 9495) modelrow_agrees_9496
private def atomic_9497 : CountSegment 1 10 13120 9496 9497 3943 3944 := CountSegment.single (first := 9496) (initial := 3943) row_count_9497 (by decide)
private theorem atomic_9497_agreement : RowAgreement 9496 9497 := RowAgreement.single (first := 9496) modelrow_agrees_9497
private def atomic_9498 : CountSegment 1 10 13120 9497 9498 3944 3945 := CountSegment.single (first := 9497) (initial := 3944) row_count_9498 (by decide)
private theorem atomic_9498_agreement : RowAgreement 9497 9498 := RowAgreement.single (first := 9497) modelrow_agrees_9498
private def atomic_9499 : CountSegment 1 10 13120 9498 9499 3945 3945 := CountSegment.single (first := 9498) (initial := 3945) row_count_9499 (by decide)
private theorem atomic_9499_agreement : RowAgreement 9498 9499 := RowAgreement.single (first := 9498) modelrow_agrees_9499
private def atomic_9500 : CountSegment 1 10 13120 9499 9500 3945 3946 := CountSegment.single (first := 9499) (initial := 3945) row_count_9500 (by decide)
private theorem atomic_9500_agreement : RowAgreement 9499 9500 := RowAgreement.single (first := 9499) modelrow_agrees_9500
private def atomic_9501 : CountSegment 1 10 13120 9500 9501 3946 3946 := CountSegment.single (first := 9500) (initial := 3946) row_count_9501 (by decide)
private theorem atomic_9501_agreement : RowAgreement 9500 9501 := RowAgreement.single (first := 9500) modelrow_agrees_9501
private def atomic_9502 : CountSegment 1 10 13120 9501 9502 3946 3946 := CountSegment.single (first := 9501) (initial := 3946) row_count_9502 (by decide)
private theorem atomic_9502_agreement : RowAgreement 9501 9502 := RowAgreement.single (first := 9501) modelrow_agrees_9502
private def atomic_9503 : CountSegment 1 10 13120 9502 9503 3946 3946 := CountSegment.single (first := 9502) (initial := 3946) row_count_9503 (by decide)
private theorem atomic_9503_agreement : RowAgreement 9502 9503 := RowAgreement.single (first := 9502) modelrow_agrees_9503
private def atomic_9504 : CountSegment 1 10 13120 9503 9504 3946 3946 := CountSegment.single (first := 9503) (initial := 3946) row_count_9504 (by decide)
private theorem atomic_9504_agreement : RowAgreement 9503 9504 := RowAgreement.single (first := 9503) modelrow_agrees_9504
private def atomic_9505 : CountSegment 1 10 13120 9504 9505 3946 3947 := CountSegment.single (first := 9504) (initial := 3946) row_count_9505 (by decide)
private theorem atomic_9505_agreement : RowAgreement 9504 9505 := RowAgreement.single (first := 9504) modelrow_agrees_9505
private def atomic_9506 : CountSegment 1 10 13120 9505 9506 3947 3947 := CountSegment.single (first := 9505) (initial := 3947) row_count_9506 (by decide)
private theorem atomic_9506_agreement : RowAgreement 9505 9506 := RowAgreement.single (first := 9505) modelrow_agrees_9506
private def atomic_9507 : CountSegment 1 10 13120 9506 9507 3947 3948 := CountSegment.single (first := 9506) (initial := 3947) row_count_9507 (by decide)
private theorem atomic_9507_agreement : RowAgreement 9506 9507 := RowAgreement.single (first := 9506) modelrow_agrees_9507
private def atomic_9508 : CountSegment 1 10 13120 9507 9508 3948 3949 := CountSegment.single (first := 9507) (initial := 3948) row_count_9508 (by decide)
private theorem atomic_9508_agreement : RowAgreement 9507 9508 := RowAgreement.single (first := 9507) modelrow_agrees_9508
private def atomic_9509 : CountSegment 1 10 13120 9508 9509 3949 3949 := CountSegment.single (first := 9508) (initial := 3949) row_count_9509 (by decide)
private theorem atomic_9509_agreement : RowAgreement 9508 9509 := RowAgreement.single (first := 9508) modelrow_agrees_9509
private def atomic_9510 : CountSegment 1 10 13120 9509 9510 3949 3949 := CountSegment.single (first := 9509) (initial := 3949) row_count_9510 (by decide)
private theorem atomic_9510_agreement : RowAgreement 9509 9510 := RowAgreement.single (first := 9509) modelrow_agrees_9510
private def atomic_9511 : CountSegment 1 10 13120 9510 9511 3949 3949 := CountSegment.single (first := 9510) (initial := 3949) row_count_9511 (by decide)
private theorem atomic_9511_agreement : RowAgreement 9510 9511 := RowAgreement.single (first := 9510) modelrow_agrees_9511
private def atomic_9512 : CountSegment 1 10 13120 9511 9512 3949 3949 := CountSegment.single (first := 9511) (initial := 3949) row_count_9512 (by decide)
private theorem atomic_9512_agreement : RowAgreement 9511 9512 := RowAgreement.single (first := 9511) modelrow_agrees_9512
private def atomic_9513 : CountSegment 1 10 13120 9512 9513 3949 3950 := CountSegment.single (first := 9512) (initial := 3949) row_count_9513 (by decide)
private theorem atomic_9513_agreement : RowAgreement 9512 9513 := RowAgreement.single (first := 9512) modelrow_agrees_9513
private def atomic_9514 : CountSegment 1 10 13120 9513 9514 3950 3950 := CountSegment.single (first := 9513) (initial := 3950) row_count_9514 (by decide)
private theorem atomic_9514_agreement : RowAgreement 9513 9514 := RowAgreement.single (first := 9513) modelrow_agrees_9514
private def atomic_9515 : CountSegment 1 10 13120 9514 9515 3950 3951 := CountSegment.single (first := 9514) (initial := 3950) row_count_9515 (by decide)
private theorem atomic_9515_agreement : RowAgreement 9514 9515 := RowAgreement.single (first := 9514) modelrow_agrees_9515
private def atomic_9516 : CountSegment 1 10 13120 9515 9516 3951 3952 := CountSegment.single (first := 9515) (initial := 3951) row_count_9516 (by decide)
private theorem atomic_9516_agreement : RowAgreement 9515 9516 := RowAgreement.single (first := 9515) modelrow_agrees_9516
private def atomic_9517 : CountSegment 1 10 13120 9516 9517 3952 3953 := CountSegment.single (first := 9516) (initial := 3952) row_count_9517 (by decide)
private theorem atomic_9517_agreement : RowAgreement 9516 9517 := RowAgreement.single (first := 9516) modelrow_agrees_9517
private def atomic_9518 : CountSegment 1 10 13120 9517 9518 3953 3953 := CountSegment.single (first := 9517) (initial := 3953) row_count_9518 (by decide)
private theorem atomic_9518_agreement : RowAgreement 9517 9518 := RowAgreement.single (first := 9517) modelrow_agrees_9518
private def atomic_9519 : CountSegment 1 10 13120 9518 9519 3953 3954 := CountSegment.single (first := 9518) (initial := 3953) row_count_9519 (by decide)
private theorem atomic_9519_agreement : RowAgreement 9518 9519 := RowAgreement.single (first := 9518) modelrow_agrees_9519
private def atomic_9520 : CountSegment 1 10 13120 9519 9520 3954 3954 := CountSegment.single (first := 9519) (initial := 3954) row_count_9520 (by decide)
private theorem atomic_9520_agreement : RowAgreement 9519 9520 := RowAgreement.single (first := 9519) modelrow_agrees_9520
private def atomic_9521 : CountSegment 1 10 13120 9520 9521 3954 3954 := CountSegment.single (first := 9520) (initial := 3954) row_count_9521 (by decide)
private theorem atomic_9521_agreement : RowAgreement 9520 9521 := RowAgreement.single (first := 9520) modelrow_agrees_9521
private def atomic_9522 : CountSegment 1 10 13120 9521 9522 3954 3954 := CountSegment.single (first := 9521) (initial := 3954) row_count_9522 (by decide)
private theorem atomic_9522_agreement : RowAgreement 9521 9522 := RowAgreement.single (first := 9521) modelrow_agrees_9522
private def atomic_9523 : CountSegment 1 10 13120 9522 9523 3954 3956 := CountSegment.single (first := 9522) (initial := 3954) row_count_9523 (by decide)
private theorem atomic_9523_agreement : RowAgreement 9522 9523 := RowAgreement.single (first := 9522) modelrow_agrees_9523
private def atomic_9524 : CountSegment 1 10 13120 9523 9524 3956 3956 := CountSegment.single (first := 9523) (initial := 3956) row_count_9524 (by decide)
private theorem atomic_9524_agreement : RowAgreement 9523 9524 := RowAgreement.single (first := 9523) modelrow_agrees_9524
private def atomic_9525 : CountSegment 1 10 13120 9524 9525 3956 3957 := CountSegment.single (first := 9524) (initial := 3956) row_count_9525 (by decide)
private theorem atomic_9525_agreement : RowAgreement 9524 9525 := RowAgreement.single (first := 9524) modelrow_agrees_9525
private def atomic_9526 : CountSegment 1 10 13120 9525 9526 3957 3958 := CountSegment.single (first := 9525) (initial := 3957) row_count_9526 (by decide)
private theorem atomic_9526_agreement : RowAgreement 9525 9526 := RowAgreement.single (first := 9525) modelrow_agrees_9526
private def atomic_9527 : CountSegment 1 10 13120 9526 9527 3958 3959 := CountSegment.single (first := 9526) (initial := 3958) row_count_9527 (by decide)
private theorem atomic_9527_agreement : RowAgreement 9526 9527 := RowAgreement.single (first := 9526) modelrow_agrees_9527
private def atomic_9528 : CountSegment 1 10 13120 9527 9528 3959 3959 := CountSegment.single (first := 9527) (initial := 3959) row_count_9528 (by decide)
private theorem atomic_9528_agreement : RowAgreement 9527 9528 := RowAgreement.single (first := 9527) modelrow_agrees_9528
private def atomic_9529 : CountSegment 1 10 13120 9528 9529 3959 3959 := CountSegment.single (first := 9528) (initial := 3959) row_count_9529 (by decide)
private theorem atomic_9529_agreement : RowAgreement 9528 9529 := RowAgreement.single (first := 9528) modelrow_agrees_9529
private def atomic_9530 : CountSegment 1 10 13120 9529 9530 3959 3960 := CountSegment.single (first := 9529) (initial := 3959) row_count_9530 (by decide)
private theorem atomic_9530_agreement : RowAgreement 9529 9530 := RowAgreement.single (first := 9529) modelrow_agrees_9530
private def atomic_9531 : CountSegment 1 10 13120 9530 9531 3960 3960 := CountSegment.single (first := 9530) (initial := 3960) row_count_9531 (by decide)
private theorem atomic_9531_agreement : RowAgreement 9530 9531 := RowAgreement.single (first := 9530) modelrow_agrees_9531
private def atomic_9532 : CountSegment 1 10 13120 9531 9532 3960 3960 := CountSegment.single (first := 9531) (initial := 3960) row_count_9532 (by decide)
private theorem atomic_9532_agreement : RowAgreement 9531 9532 := RowAgreement.single (first := 9531) modelrow_agrees_9532
private def atomic_9533 : CountSegment 1 10 13120 9532 9533 3960 3960 := CountSegment.single (first := 9532) (initial := 3960) row_count_9533 (by decide)
private theorem atomic_9533_agreement : RowAgreement 9532 9533 := RowAgreement.single (first := 9532) modelrow_agrees_9533
private def atomic_9534 : CountSegment 1 10 13120 9533 9534 3960 3960 := CountSegment.single (first := 9533) (initial := 3960) row_count_9534 (by decide)
private theorem atomic_9534_agreement : RowAgreement 9533 9534 := RowAgreement.single (first := 9533) modelrow_agrees_9534
private def atomic_9535 : CountSegment 1 10 13120 9534 9535 3960 3960 := CountSegment.single (first := 9534) (initial := 3960) row_count_9535 (by decide)
private theorem atomic_9535_agreement : RowAgreement 9534 9535 := RowAgreement.single (first := 9534) modelrow_agrees_9535
private def atomic_9536 : CountSegment 1 10 13120 9535 9536 3960 3961 := CountSegment.single (first := 9535) (initial := 3960) row_count_9536 (by decide)
private theorem atomic_9536_agreement : RowAgreement 9535 9536 := RowAgreement.single (first := 9535) modelrow_agrees_9536
private def atomic_9537 : CountSegment 1 10 13120 9536 9537 3961 3961 := CountSegment.single (first := 9536) (initial := 3961) row_count_9537 (by decide)
private theorem atomic_9537_agreement : RowAgreement 9536 9537 := RowAgreement.single (first := 9536) modelrow_agrees_9537
private def atomic_9538 : CountSegment 1 10 13120 9537 9538 3961 3962 := CountSegment.single (first := 9537) (initial := 3961) row_count_9538 (by decide)
private theorem atomic_9538_agreement : RowAgreement 9537 9538 := RowAgreement.single (first := 9537) modelrow_agrees_9538
private def atomic_9539 : CountSegment 1 10 13120 9538 9539 3962 3962 := CountSegment.single (first := 9538) (initial := 3962) row_count_9539 (by decide)
private theorem atomic_9539_agreement : RowAgreement 9538 9539 := RowAgreement.single (first := 9538) modelrow_agrees_9539
private def atomic_9540 : CountSegment 1 10 13120 9539 9540 3962 3962 := CountSegment.single (first := 9539) (initial := 3962) row_count_9540 (by decide)
private theorem atomic_9540_agreement : RowAgreement 9539 9540 := RowAgreement.single (first := 9539) modelrow_agrees_9540
private def atomic_9541 : CountSegment 1 10 13120 9540 9541 3962 3963 := CountSegment.single (first := 9540) (initial := 3962) row_count_9541 (by decide)
private theorem atomic_9541_agreement : RowAgreement 9540 9541 := RowAgreement.single (first := 9540) modelrow_agrees_9541
private def atomic_9542 : CountSegment 1 10 13120 9541 9542 3963 3964 := CountSegment.single (first := 9541) (initial := 3963) row_count_9542 (by decide)
private theorem atomic_9542_agreement : RowAgreement 9541 9542 := RowAgreement.single (first := 9541) modelrow_agrees_9542
private def atomic_9543 : CountSegment 1 10 13120 9542 9543 3964 3964 := CountSegment.single (first := 9542) (initial := 3964) row_count_9543 (by decide)
private theorem atomic_9543_agreement : RowAgreement 9542 9543 := RowAgreement.single (first := 9542) modelrow_agrees_9543
private def atomic_9544 : CountSegment 1 10 13120 9543 9544 3964 3964 := CountSegment.single (first := 9543) (initial := 3964) row_count_9544 (by decide)
private theorem atomic_9544_agreement : RowAgreement 9543 9544 := RowAgreement.single (first := 9543) modelrow_agrees_9544
private def atomic_9545 : CountSegment 1 10 13120 9544 9545 3964 3964 := CountSegment.single (first := 9544) (initial := 3964) row_count_9545 (by decide)
private theorem atomic_9545_agreement : RowAgreement 9544 9545 := RowAgreement.single (first := 9544) modelrow_agrees_9545
private def atomic_9546 : CountSegment 1 10 13120 9545 9546 3964 3965 := CountSegment.single (first := 9545) (initial := 3964) row_count_9546 (by decide)
private theorem atomic_9546_agreement : RowAgreement 9545 9546 := RowAgreement.single (first := 9545) modelrow_agrees_9546
private def atomic_9547 : CountSegment 1 10 13120 9546 9547 3965 3965 := CountSegment.single (first := 9546) (initial := 3965) row_count_9547 (by decide)
private theorem atomic_9547_agreement : RowAgreement 9546 9547 := RowAgreement.single (first := 9546) modelrow_agrees_9547
private def atomic_9548 : CountSegment 1 10 13120 9547 9548 3965 3965 := CountSegment.single (first := 9547) (initial := 3965) row_count_9548 (by decide)
private theorem atomic_9548_agreement : RowAgreement 9547 9548 := RowAgreement.single (first := 9547) modelrow_agrees_9548
private def atomic_9549 : CountSegment 1 10 13120 9548 9549 3965 3965 := CountSegment.single (first := 9548) (initial := 3965) row_count_9549 (by decide)
private theorem atomic_9549_agreement : RowAgreement 9548 9549 := RowAgreement.single (first := 9548) modelrow_agrees_9549
private def atomic_9550 : CountSegment 1 10 13120 9549 9550 3965 3966 := CountSegment.single (first := 9549) (initial := 3965) row_count_9550 (by decide)
private theorem atomic_9550_agreement : RowAgreement 9549 9550 := RowAgreement.single (first := 9549) modelrow_agrees_9550
private def atomic_9551 : CountSegment 1 10 13120 9550 9551 3966 3966 := CountSegment.single (first := 9550) (initial := 3966) row_count_9551 (by decide)
private theorem atomic_9551_agreement : RowAgreement 9550 9551 := RowAgreement.single (first := 9550) modelrow_agrees_9551
private def atomic_9552 : CountSegment 1 10 13120 9551 9552 3966 3966 := CountSegment.single (first := 9551) (initial := 3966) row_count_9552 (by decide)
private theorem atomic_9552_agreement : RowAgreement 9551 9552 := RowAgreement.single (first := 9551) modelrow_agrees_9552
private def atomic_9553 : CountSegment 1 10 13120 9552 9553 3966 3966 := CountSegment.single (first := 9552) (initial := 3966) row_count_9553 (by decide)
private theorem atomic_9553_agreement : RowAgreement 9552 9553 := RowAgreement.single (first := 9552) modelrow_agrees_9553
private def atomic_9554 : CountSegment 1 10 13120 9553 9554 3966 3966 := CountSegment.single (first := 9553) (initial := 3966) row_count_9554 (by decide)
private theorem atomic_9554_agreement : RowAgreement 9553 9554 := RowAgreement.single (first := 9553) modelrow_agrees_9554
private def atomic_9555 : CountSegment 1 10 13120 9554 9555 3966 3967 := CountSegment.single (first := 9554) (initial := 3966) row_count_9555 (by decide)
private theorem atomic_9555_agreement : RowAgreement 9554 9555 := RowAgreement.single (first := 9554) modelrow_agrees_9555
private def atomic_9556 : CountSegment 1 10 13120 9555 9556 3967 3967 := CountSegment.single (first := 9555) (initial := 3967) row_count_9556 (by decide)
private theorem atomic_9556_agreement : RowAgreement 9555 9556 := RowAgreement.single (first := 9555) modelrow_agrees_9556
private def atomic_9557 : CountSegment 1 10 13120 9556 9557 3967 3967 := CountSegment.single (first := 9556) (initial := 3967) row_count_9557 (by decide)
private theorem atomic_9557_agreement : RowAgreement 9556 9557 := RowAgreement.single (first := 9556) modelrow_agrees_9557
private def atomic_9558 : CountSegment 1 10 13120 9557 9558 3967 3969 := CountSegment.single (first := 9557) (initial := 3967) row_count_9558 (by decide)
private theorem atomic_9558_agreement : RowAgreement 9557 9558 := RowAgreement.single (first := 9557) modelrow_agrees_9558
private def atomic_9559 : CountSegment 1 10 13120 9558 9559 3969 3969 := CountSegment.single (first := 9558) (initial := 3969) row_count_9559 (by decide)
private theorem atomic_9559_agreement : RowAgreement 9558 9559 := RowAgreement.single (first := 9558) modelrow_agrees_9559
private def atomic_9560 : CountSegment 1 10 13120 9559 9560 3969 3970 := CountSegment.single (first := 9559) (initial := 3969) row_count_9560 (by decide)
private theorem atomic_9560_agreement : RowAgreement 9559 9560 := RowAgreement.single (first := 9559) modelrow_agrees_9560
private def atomic_9561 : CountSegment 1 10 13120 9560 9561 3970 3970 := CountSegment.single (first := 9560) (initial := 3970) row_count_9561 (by decide)
private theorem atomic_9561_agreement : RowAgreement 9560 9561 := RowAgreement.single (first := 9560) modelrow_agrees_9561
private def atomic_9562 : CountSegment 1 10 13120 9561 9562 3970 3971 := CountSegment.single (first := 9561) (initial := 3970) row_count_9562 (by decide)
private theorem atomic_9562_agreement : RowAgreement 9561 9562 := RowAgreement.single (first := 9561) modelrow_agrees_9562
private def atomic_9563 : CountSegment 1 10 13120 9562 9563 3971 3971 := CountSegment.single (first := 9562) (initial := 3971) row_count_9563 (by decide)
private theorem atomic_9563_agreement : RowAgreement 9562 9563 := RowAgreement.single (first := 9562) modelrow_agrees_9563
private def atomic_9564 : CountSegment 1 10 13120 9563 9564 3971 3972 := CountSegment.single (first := 9563) (initial := 3971) row_count_9564 (by decide)
private theorem atomic_9564_agreement : RowAgreement 9563 9564 := RowAgreement.single (first := 9563) modelrow_agrees_9564
private def atomic_9565 : CountSegment 1 10 13120 9564 9565 3972 3973 := CountSegment.single (first := 9564) (initial := 3972) row_count_9565 (by decide)
private theorem atomic_9565_agreement : RowAgreement 9564 9565 := RowAgreement.single (first := 9564) modelrow_agrees_9565
private def atomic_9566 : CountSegment 1 10 13120 9565 9566 3973 3973 := CountSegment.single (first := 9565) (initial := 3973) row_count_9566 (by decide)
private theorem atomic_9566_agreement : RowAgreement 9565 9566 := RowAgreement.single (first := 9565) modelrow_agrees_9566
private def atomic_9567 : CountSegment 1 10 13120 9566 9567 3973 3973 := CountSegment.single (first := 9566) (initial := 3973) row_count_9567 (by decide)
private theorem atomic_9567_agreement : RowAgreement 9566 9567 := RowAgreement.single (first := 9566) modelrow_agrees_9567
private def atomic_9568 : CountSegment 1 10 13120 9567 9568 3973 3975 := CountSegment.single (first := 9567) (initial := 3973) row_count_9568 (by decide)
private theorem atomic_9568_agreement : RowAgreement 9567 9568 := RowAgreement.single (first := 9567) modelrow_agrees_9568
private def atomic_9569 : CountSegment 1 10 13120 9568 9569 3975 3975 := CountSegment.single (first := 9568) (initial := 3975) row_count_9569 (by decide)
private theorem atomic_9569_agreement : RowAgreement 9568 9569 := RowAgreement.single (first := 9568) modelrow_agrees_9569
private def atomic_9570 : CountSegment 1 10 13120 9569 9570 3975 3976 := CountSegment.single (first := 9569) (initial := 3975) row_count_9570 (by decide)
private theorem atomic_9570_agreement : RowAgreement 9569 9570 := RowAgreement.single (first := 9569) modelrow_agrees_9570
private def atomic_9571 : CountSegment 1 10 13120 9570 9571 3976 3976 := CountSegment.single (first := 9570) (initial := 3976) row_count_9571 (by decide)
private theorem atomic_9571_agreement : RowAgreement 9570 9571 := RowAgreement.single (first := 9570) modelrow_agrees_9571
private def atomic_9572 : CountSegment 1 10 13120 9571 9572 3976 3977 := CountSegment.single (first := 9571) (initial := 3976) row_count_9572 (by decide)
private theorem atomic_9572_agreement : RowAgreement 9571 9572 := RowAgreement.single (first := 9571) modelrow_agrees_9572
private def atomic_9573 : CountSegment 1 10 13120 9572 9573 3977 3977 := CountSegment.single (first := 9572) (initial := 3977) row_count_9573 (by decide)
private theorem atomic_9573_agreement : RowAgreement 9572 9573 := RowAgreement.single (first := 9572) modelrow_agrees_9573
private def atomic_9574 : CountSegment 1 10 13120 9573 9574 3977 3978 := CountSegment.single (first := 9573) (initial := 3977) row_count_9574 (by decide)
private theorem atomic_9574_agreement : RowAgreement 9573 9574 := RowAgreement.single (first := 9573) modelrow_agrees_9574
private def atomic_9575 : CountSegment 1 10 13120 9574 9575 3978 3979 := CountSegment.single (first := 9574) (initial := 3978) row_count_9575 (by decide)
private theorem atomic_9575_agreement : RowAgreement 9574 9575 := RowAgreement.single (first := 9574) modelrow_agrees_9575
private def atomic_9576 : CountSegment 1 10 13120 9575 9576 3979 3980 := CountSegment.single (first := 9575) (initial := 3979) row_count_9576 (by decide)
private theorem atomic_9576_agreement : RowAgreement 9575 9576 := RowAgreement.single (first := 9575) modelrow_agrees_9576
private def atomic_9577 : CountSegment 1 10 13120 9576 9577 3980 3980 := CountSegment.single (first := 9576) (initial := 3980) row_count_9577 (by decide)
private theorem atomic_9577_agreement : RowAgreement 9576 9577 := RowAgreement.single (first := 9576) modelrow_agrees_9577
private def atomic_9578 : CountSegment 1 10 13120 9577 9578 3980 3980 := CountSegment.single (first := 9577) (initial := 3980) row_count_9578 (by decide)
private theorem atomic_9578_agreement : RowAgreement 9577 9578 := RowAgreement.single (first := 9577) modelrow_agrees_9578
private def atomic_9579 : CountSegment 1 10 13120 9578 9579 3980 3980 := CountSegment.single (first := 9578) (initial := 3980) row_count_9579 (by decide)
private theorem atomic_9579_agreement : RowAgreement 9578 9579 := RowAgreement.single (first := 9578) modelrow_agrees_9579
private def atomic_9580 : CountSegment 1 10 13120 9579 9580 3980 3980 := CountSegment.single (first := 9579) (initial := 3980) row_count_9580 (by decide)
private theorem atomic_9580_agreement : RowAgreement 9579 9580 := RowAgreement.single (first := 9579) modelrow_agrees_9580
private def atomic_9581 : CountSegment 1 10 13120 9580 9581 3980 3980 := CountSegment.single (first := 9580) (initial := 3980) row_count_9581 (by decide)
private theorem atomic_9581_agreement : RowAgreement 9580 9581 := RowAgreement.single (first := 9580) modelrow_agrees_9581
private def atomic_9582 : CountSegment 1 10 13120 9581 9582 3980 3981 := CountSegment.single (first := 9581) (initial := 3980) row_count_9582 (by decide)
private theorem atomic_9582_agreement : RowAgreement 9581 9582 := RowAgreement.single (first := 9581) modelrow_agrees_9582
private def atomic_9583 : CountSegment 1 10 13120 9582 9583 3981 3982 := CountSegment.single (first := 9582) (initial := 3981) row_count_9583 (by decide)
private theorem atomic_9583_agreement : RowAgreement 9582 9583 := RowAgreement.single (first := 9582) modelrow_agrees_9583
private def atomic_9584 : CountSegment 1 10 13120 9583 9584 3982 3982 := CountSegment.single (first := 9583) (initial := 3982) row_count_9584 (by decide)
private theorem atomic_9584_agreement : RowAgreement 9583 9584 := RowAgreement.single (first := 9583) modelrow_agrees_9584
private def atomic_9585 : CountSegment 1 10 13120 9584 9585 3982 3982 := CountSegment.single (first := 9584) (initial := 3982) row_count_9585 (by decide)
private theorem atomic_9585_agreement : RowAgreement 9584 9585 := RowAgreement.single (first := 9584) modelrow_agrees_9585
private def atomic_9586 : CountSegment 1 10 13120 9585 9586 3982 3983 := CountSegment.single (first := 9585) (initial := 3982) row_count_9586 (by decide)
private theorem atomic_9586_agreement : RowAgreement 9585 9586 := RowAgreement.single (first := 9585) modelrow_agrees_9586
private def atomic_9587 : CountSegment 1 10 13120 9586 9587 3983 3983 := CountSegment.single (first := 9586) (initial := 3983) row_count_9587 (by decide)
private theorem atomic_9587_agreement : RowAgreement 9586 9587 := RowAgreement.single (first := 9586) modelrow_agrees_9587
private def atomic_9588 : CountSegment 1 10 13120 9587 9588 3983 3983 := CountSegment.single (first := 9587) (initial := 3983) row_count_9588 (by decide)
private theorem atomic_9588_agreement : RowAgreement 9587 9588 := RowAgreement.single (first := 9587) modelrow_agrees_9588
private def atomic_9589 : CountSegment 1 10 13120 9588 9589 3983 3983 := CountSegment.single (first := 9588) (initial := 3983) row_count_9589 (by decide)
private theorem atomic_9589_agreement : RowAgreement 9588 9589 := RowAgreement.single (first := 9588) modelrow_agrees_9589
private def atomic_9590 : CountSegment 1 10 13120 9589 9590 3983 3983 := CountSegment.single (first := 9589) (initial := 3983) row_count_9590 (by decide)
private theorem atomic_9590_agreement : RowAgreement 9589 9590 := RowAgreement.single (first := 9589) modelrow_agrees_9590
private def atomic_9591 : CountSegment 1 10 13120 9590 9591 3983 3983 := CountSegment.single (first := 9590) (initial := 3983) row_count_9591 (by decide)
private theorem atomic_9591_agreement : RowAgreement 9590 9591 := RowAgreement.single (first := 9590) modelrow_agrees_9591
private def atomic_9592 : CountSegment 1 10 13120 9591 9592 3983 3983 := CountSegment.single (first := 9591) (initial := 3983) row_count_9592 (by decide)
private theorem atomic_9592_agreement : RowAgreement 9591 9592 := RowAgreement.single (first := 9591) modelrow_agrees_9592
private def atomic_9593 : CountSegment 1 10 13120 9592 9593 3983 3983 := CountSegment.single (first := 9592) (initial := 3983) row_count_9593 (by decide)
private theorem atomic_9593_agreement : RowAgreement 9592 9593 := RowAgreement.single (first := 9592) modelrow_agrees_9593
private def atomic_9594 : CountSegment 1 10 13120 9593 9594 3983 3984 := CountSegment.single (first := 9593) (initial := 3983) row_count_9594 (by decide)
private theorem atomic_9594_agreement : RowAgreement 9593 9594 := RowAgreement.single (first := 9593) modelrow_agrees_9594
private def atomic_9595 : CountSegment 1 10 13120 9594 9595 3984 3984 := CountSegment.single (first := 9594) (initial := 3984) row_count_9595 (by decide)
private theorem atomic_9595_agreement : RowAgreement 9594 9595 := RowAgreement.single (first := 9594) modelrow_agrees_9595
private def atomic_9596 : CountSegment 1 10 13120 9595 9596 3984 3984 := CountSegment.single (first := 9595) (initial := 3984) row_count_9596 (by decide)
private theorem atomic_9596_agreement : RowAgreement 9595 9596 := RowAgreement.single (first := 9595) modelrow_agrees_9596
private def atomic_9597 : CountSegment 1 10 13120 9596 9597 3984 3985 := CountSegment.single (first := 9596) (initial := 3984) row_count_9597 (by decide)
private theorem atomic_9597_agreement : RowAgreement 9596 9597 := RowAgreement.single (first := 9596) modelrow_agrees_9597
private def atomic_9598 : CountSegment 1 10 13120 9597 9598 3985 3985 := CountSegment.single (first := 9597) (initial := 3985) row_count_9598 (by decide)
private theorem atomic_9598_agreement : RowAgreement 9597 9598 := RowAgreement.single (first := 9597) modelrow_agrees_9598
private def atomic_9599 : CountSegment 1 10 13120 9598 9599 3985 3985 := CountSegment.single (first := 9598) (initial := 3985) row_count_9599 (by decide)
private theorem atomic_9599_agreement : RowAgreement 9598 9599 := RowAgreement.single (first := 9598) modelrow_agrees_9599
private def atomic_9600 : CountSegment 1 10 13120 9599 9600 3985 3985 := CountSegment.single (first := 9599) (initial := 3985) row_count_9600 (by decide)
private theorem atomic_9600_agreement : RowAgreement 9599 9600 := RowAgreement.single (first := 9599) modelrow_agrees_9600
private def within_074_000 : CountSegment 1 10 13120 9472 9474 3934 3936 := CountSegment.join atomic_9473 atomic_9474
private theorem within_074_000_agreement : RowAgreement 9472 9474 := RowAgreement.join atomic_9473_agreement atomic_9474_agreement
private def within_074_001 : CountSegment 1 10 13120 9474 9476 3936 3936 := CountSegment.join atomic_9475 atomic_9476
private theorem within_074_001_agreement : RowAgreement 9474 9476 := RowAgreement.join atomic_9475_agreement atomic_9476_agreement
private def within_074_002 : CountSegment 1 10 13120 9472 9476 3934 3936 := CountSegment.join within_074_000 within_074_001
private theorem within_074_002_agreement : RowAgreement 9472 9476 := RowAgreement.join within_074_000_agreement within_074_001_agreement
private def within_074_003 : CountSegment 1 10 13120 9476 9478 3936 3936 := CountSegment.join atomic_9477 atomic_9478
private theorem within_074_003_agreement : RowAgreement 9476 9478 := RowAgreement.join atomic_9477_agreement atomic_9478_agreement
private def within_074_004 : CountSegment 1 10 13120 9478 9480 3936 3936 := CountSegment.join atomic_9479 atomic_9480
private theorem within_074_004_agreement : RowAgreement 9478 9480 := RowAgreement.join atomic_9479_agreement atomic_9480_agreement
private def within_074_005 : CountSegment 1 10 13120 9476 9480 3936 3936 := CountSegment.join within_074_003 within_074_004
private theorem within_074_005_agreement : RowAgreement 9476 9480 := RowAgreement.join within_074_003_agreement within_074_004_agreement
private def within_074_006 : CountSegment 1 10 13120 9472 9480 3934 3936 := CountSegment.join within_074_002 within_074_005
private theorem within_074_006_agreement : RowAgreement 9472 9480 := RowAgreement.join within_074_002_agreement within_074_005_agreement
private def within_074_007 : CountSegment 1 10 13120 9480 9482 3936 3936 := CountSegment.join atomic_9481 atomic_9482
private theorem within_074_007_agreement : RowAgreement 9480 9482 := RowAgreement.join atomic_9481_agreement atomic_9482_agreement
private def within_074_008 : CountSegment 1 10 13120 9482 9484 3936 3937 := CountSegment.join atomic_9483 atomic_9484
private theorem within_074_008_agreement : RowAgreement 9482 9484 := RowAgreement.join atomic_9483_agreement atomic_9484_agreement
private def within_074_009 : CountSegment 1 10 13120 9480 9484 3936 3937 := CountSegment.join within_074_007 within_074_008
private theorem within_074_009_agreement : RowAgreement 9480 9484 := RowAgreement.join within_074_007_agreement within_074_008_agreement
private def within_074_010 : CountSegment 1 10 13120 9484 9486 3937 3938 := CountSegment.join atomic_9485 atomic_9486
private theorem within_074_010_agreement : RowAgreement 9484 9486 := RowAgreement.join atomic_9485_agreement atomic_9486_agreement
private def within_074_011 : CountSegment 1 10 13120 9486 9488 3938 3938 := CountSegment.join atomic_9487 atomic_9488
private theorem within_074_011_agreement : RowAgreement 9486 9488 := RowAgreement.join atomic_9487_agreement atomic_9488_agreement
private def within_074_012 : CountSegment 1 10 13120 9484 9488 3937 3938 := CountSegment.join within_074_010 within_074_011
private theorem within_074_012_agreement : RowAgreement 9484 9488 := RowAgreement.join within_074_010_agreement within_074_011_agreement
private def within_074_013 : CountSegment 1 10 13120 9480 9488 3936 3938 := CountSegment.join within_074_009 within_074_012
private theorem within_074_013_agreement : RowAgreement 9480 9488 := RowAgreement.join within_074_009_agreement within_074_012_agreement
private def within_074_014 : CountSegment 1 10 13120 9472 9488 3934 3938 := CountSegment.join within_074_006 within_074_013
private theorem within_074_014_agreement : RowAgreement 9472 9488 := RowAgreement.join within_074_006_agreement within_074_013_agreement
private def within_074_015 : CountSegment 1 10 13120 9488 9490 3938 3941 := CountSegment.join atomic_9489 atomic_9490
private theorem within_074_015_agreement : RowAgreement 9488 9490 := RowAgreement.join atomic_9489_agreement atomic_9490_agreement
private def within_074_016 : CountSegment 1 10 13120 9490 9492 3941 3942 := CountSegment.join atomic_9491 atomic_9492
private theorem within_074_016_agreement : RowAgreement 9490 9492 := RowAgreement.join atomic_9491_agreement atomic_9492_agreement
private def within_074_017 : CountSegment 1 10 13120 9488 9492 3938 3942 := CountSegment.join within_074_015 within_074_016
private theorem within_074_017_agreement : RowAgreement 9488 9492 := RowAgreement.join within_074_015_agreement within_074_016_agreement
private def within_074_018 : CountSegment 1 10 13120 9492 9494 3942 3943 := CountSegment.join atomic_9493 atomic_9494
private theorem within_074_018_agreement : RowAgreement 9492 9494 := RowAgreement.join atomic_9493_agreement atomic_9494_agreement
private def within_074_019 : CountSegment 1 10 13120 9494 9496 3943 3943 := CountSegment.join atomic_9495 atomic_9496
private theorem within_074_019_agreement : RowAgreement 9494 9496 := RowAgreement.join atomic_9495_agreement atomic_9496_agreement
private def within_074_020 : CountSegment 1 10 13120 9492 9496 3942 3943 := CountSegment.join within_074_018 within_074_019
private theorem within_074_020_agreement : RowAgreement 9492 9496 := RowAgreement.join within_074_018_agreement within_074_019_agreement
private def within_074_021 : CountSegment 1 10 13120 9488 9496 3938 3943 := CountSegment.join within_074_017 within_074_020
private theorem within_074_021_agreement : RowAgreement 9488 9496 := RowAgreement.join within_074_017_agreement within_074_020_agreement
private def within_074_022 : CountSegment 1 10 13120 9496 9498 3943 3945 := CountSegment.join atomic_9497 atomic_9498
private theorem within_074_022_agreement : RowAgreement 9496 9498 := RowAgreement.join atomic_9497_agreement atomic_9498_agreement
private def within_074_023 : CountSegment 1 10 13120 9498 9500 3945 3946 := CountSegment.join atomic_9499 atomic_9500
private theorem within_074_023_agreement : RowAgreement 9498 9500 := RowAgreement.join atomic_9499_agreement atomic_9500_agreement
private def within_074_024 : CountSegment 1 10 13120 9496 9500 3943 3946 := CountSegment.join within_074_022 within_074_023
private theorem within_074_024_agreement : RowAgreement 9496 9500 := RowAgreement.join within_074_022_agreement within_074_023_agreement
private def within_074_025 : CountSegment 1 10 13120 9500 9502 3946 3946 := CountSegment.join atomic_9501 atomic_9502
private theorem within_074_025_agreement : RowAgreement 9500 9502 := RowAgreement.join atomic_9501_agreement atomic_9502_agreement
private def within_074_026 : CountSegment 1 10 13120 9502 9504 3946 3946 := CountSegment.join atomic_9503 atomic_9504
private theorem within_074_026_agreement : RowAgreement 9502 9504 := RowAgreement.join atomic_9503_agreement atomic_9504_agreement
private def within_074_027 : CountSegment 1 10 13120 9500 9504 3946 3946 := CountSegment.join within_074_025 within_074_026
private theorem within_074_027_agreement : RowAgreement 9500 9504 := RowAgreement.join within_074_025_agreement within_074_026_agreement
private def within_074_028 : CountSegment 1 10 13120 9496 9504 3943 3946 := CountSegment.join within_074_024 within_074_027
private theorem within_074_028_agreement : RowAgreement 9496 9504 := RowAgreement.join within_074_024_agreement within_074_027_agreement
private def within_074_029 : CountSegment 1 10 13120 9488 9504 3938 3946 := CountSegment.join within_074_021 within_074_028
private theorem within_074_029_agreement : RowAgreement 9488 9504 := RowAgreement.join within_074_021_agreement within_074_028_agreement
private def within_074_030 : CountSegment 1 10 13120 9472 9504 3934 3946 := CountSegment.join within_074_014 within_074_029
private theorem within_074_030_agreement : RowAgreement 9472 9504 := RowAgreement.join within_074_014_agreement within_074_029_agreement
private def within_074_031 : CountSegment 1 10 13120 9504 9506 3946 3947 := CountSegment.join atomic_9505 atomic_9506
private theorem within_074_031_agreement : RowAgreement 9504 9506 := RowAgreement.join atomic_9505_agreement atomic_9506_agreement
private def within_074_032 : CountSegment 1 10 13120 9506 9508 3947 3949 := CountSegment.join atomic_9507 atomic_9508
private theorem within_074_032_agreement : RowAgreement 9506 9508 := RowAgreement.join atomic_9507_agreement atomic_9508_agreement
private def within_074_033 : CountSegment 1 10 13120 9504 9508 3946 3949 := CountSegment.join within_074_031 within_074_032
private theorem within_074_033_agreement : RowAgreement 9504 9508 := RowAgreement.join within_074_031_agreement within_074_032_agreement
private def within_074_034 : CountSegment 1 10 13120 9508 9510 3949 3949 := CountSegment.join atomic_9509 atomic_9510
private theorem within_074_034_agreement : RowAgreement 9508 9510 := RowAgreement.join atomic_9509_agreement atomic_9510_agreement
private def within_074_035 : CountSegment 1 10 13120 9510 9512 3949 3949 := CountSegment.join atomic_9511 atomic_9512
private theorem within_074_035_agreement : RowAgreement 9510 9512 := RowAgreement.join atomic_9511_agreement atomic_9512_agreement
private def within_074_036 : CountSegment 1 10 13120 9508 9512 3949 3949 := CountSegment.join within_074_034 within_074_035
private theorem within_074_036_agreement : RowAgreement 9508 9512 := RowAgreement.join within_074_034_agreement within_074_035_agreement
private def within_074_037 : CountSegment 1 10 13120 9504 9512 3946 3949 := CountSegment.join within_074_033 within_074_036
private theorem within_074_037_agreement : RowAgreement 9504 9512 := RowAgreement.join within_074_033_agreement within_074_036_agreement
private def within_074_038 : CountSegment 1 10 13120 9512 9514 3949 3950 := CountSegment.join atomic_9513 atomic_9514
private theorem within_074_038_agreement : RowAgreement 9512 9514 := RowAgreement.join atomic_9513_agreement atomic_9514_agreement
private def within_074_039 : CountSegment 1 10 13120 9514 9516 3950 3952 := CountSegment.join atomic_9515 atomic_9516
private theorem within_074_039_agreement : RowAgreement 9514 9516 := RowAgreement.join atomic_9515_agreement atomic_9516_agreement
private def within_074_040 : CountSegment 1 10 13120 9512 9516 3949 3952 := CountSegment.join within_074_038 within_074_039
private theorem within_074_040_agreement : RowAgreement 9512 9516 := RowAgreement.join within_074_038_agreement within_074_039_agreement
private def within_074_041 : CountSegment 1 10 13120 9516 9518 3952 3953 := CountSegment.join atomic_9517 atomic_9518
private theorem within_074_041_agreement : RowAgreement 9516 9518 := RowAgreement.join atomic_9517_agreement atomic_9518_agreement
private def within_074_042 : CountSegment 1 10 13120 9518 9520 3953 3954 := CountSegment.join atomic_9519 atomic_9520
private theorem within_074_042_agreement : RowAgreement 9518 9520 := RowAgreement.join atomic_9519_agreement atomic_9520_agreement
private def within_074_043 : CountSegment 1 10 13120 9516 9520 3952 3954 := CountSegment.join within_074_041 within_074_042
private theorem within_074_043_agreement : RowAgreement 9516 9520 := RowAgreement.join within_074_041_agreement within_074_042_agreement
private def within_074_044 : CountSegment 1 10 13120 9512 9520 3949 3954 := CountSegment.join within_074_040 within_074_043
private theorem within_074_044_agreement : RowAgreement 9512 9520 := RowAgreement.join within_074_040_agreement within_074_043_agreement
private def within_074_045 : CountSegment 1 10 13120 9504 9520 3946 3954 := CountSegment.join within_074_037 within_074_044
private theorem within_074_045_agreement : RowAgreement 9504 9520 := RowAgreement.join within_074_037_agreement within_074_044_agreement
private def within_074_046 : CountSegment 1 10 13120 9520 9522 3954 3954 := CountSegment.join atomic_9521 atomic_9522
private theorem within_074_046_agreement : RowAgreement 9520 9522 := RowAgreement.join atomic_9521_agreement atomic_9522_agreement
private def within_074_047 : CountSegment 1 10 13120 9522 9524 3954 3956 := CountSegment.join atomic_9523 atomic_9524
private theorem within_074_047_agreement : RowAgreement 9522 9524 := RowAgreement.join atomic_9523_agreement atomic_9524_agreement
private def within_074_048 : CountSegment 1 10 13120 9520 9524 3954 3956 := CountSegment.join within_074_046 within_074_047
private theorem within_074_048_agreement : RowAgreement 9520 9524 := RowAgreement.join within_074_046_agreement within_074_047_agreement
private def within_074_049 : CountSegment 1 10 13120 9524 9526 3956 3958 := CountSegment.join atomic_9525 atomic_9526
private theorem within_074_049_agreement : RowAgreement 9524 9526 := RowAgreement.join atomic_9525_agreement atomic_9526_agreement
private def within_074_050 : CountSegment 1 10 13120 9526 9528 3958 3959 := CountSegment.join atomic_9527 atomic_9528
private theorem within_074_050_agreement : RowAgreement 9526 9528 := RowAgreement.join atomic_9527_agreement atomic_9528_agreement
private def within_074_051 : CountSegment 1 10 13120 9524 9528 3956 3959 := CountSegment.join within_074_049 within_074_050
private theorem within_074_051_agreement : RowAgreement 9524 9528 := RowAgreement.join within_074_049_agreement within_074_050_agreement
private def within_074_052 : CountSegment 1 10 13120 9520 9528 3954 3959 := CountSegment.join within_074_048 within_074_051
private theorem within_074_052_agreement : RowAgreement 9520 9528 := RowAgreement.join within_074_048_agreement within_074_051_agreement
private def within_074_053 : CountSegment 1 10 13120 9528 9530 3959 3960 := CountSegment.join atomic_9529 atomic_9530
private theorem within_074_053_agreement : RowAgreement 9528 9530 := RowAgreement.join atomic_9529_agreement atomic_9530_agreement
private def within_074_054 : CountSegment 1 10 13120 9530 9532 3960 3960 := CountSegment.join atomic_9531 atomic_9532
private theorem within_074_054_agreement : RowAgreement 9530 9532 := RowAgreement.join atomic_9531_agreement atomic_9532_agreement
private def within_074_055 : CountSegment 1 10 13120 9528 9532 3959 3960 := CountSegment.join within_074_053 within_074_054
private theorem within_074_055_agreement : RowAgreement 9528 9532 := RowAgreement.join within_074_053_agreement within_074_054_agreement
private def within_074_056 : CountSegment 1 10 13120 9532 9534 3960 3960 := CountSegment.join atomic_9533 atomic_9534
private theorem within_074_056_agreement : RowAgreement 9532 9534 := RowAgreement.join atomic_9533_agreement atomic_9534_agreement
private def within_074_057 : CountSegment 1 10 13120 9534 9536 3960 3961 := CountSegment.join atomic_9535 atomic_9536
private theorem within_074_057_agreement : RowAgreement 9534 9536 := RowAgreement.join atomic_9535_agreement atomic_9536_agreement
private def within_074_058 : CountSegment 1 10 13120 9532 9536 3960 3961 := CountSegment.join within_074_056 within_074_057
private theorem within_074_058_agreement : RowAgreement 9532 9536 := RowAgreement.join within_074_056_agreement within_074_057_agreement
private def within_074_059 : CountSegment 1 10 13120 9528 9536 3959 3961 := CountSegment.join within_074_055 within_074_058
private theorem within_074_059_agreement : RowAgreement 9528 9536 := RowAgreement.join within_074_055_agreement within_074_058_agreement
private def within_074_060 : CountSegment 1 10 13120 9520 9536 3954 3961 := CountSegment.join within_074_052 within_074_059
private theorem within_074_060_agreement : RowAgreement 9520 9536 := RowAgreement.join within_074_052_agreement within_074_059_agreement
private def within_074_061 : CountSegment 1 10 13120 9504 9536 3946 3961 := CountSegment.join within_074_045 within_074_060
private theorem within_074_061_agreement : RowAgreement 9504 9536 := RowAgreement.join within_074_045_agreement within_074_060_agreement
private def within_074_062 : CountSegment 1 10 13120 9472 9536 3934 3961 := CountSegment.join within_074_030 within_074_061
private theorem within_074_062_agreement : RowAgreement 9472 9536 := RowAgreement.join within_074_030_agreement within_074_061_agreement
private def within_074_063 : CountSegment 1 10 13120 9536 9538 3961 3962 := CountSegment.join atomic_9537 atomic_9538
private theorem within_074_063_agreement : RowAgreement 9536 9538 := RowAgreement.join atomic_9537_agreement atomic_9538_agreement
private def within_074_064 : CountSegment 1 10 13120 9538 9540 3962 3962 := CountSegment.join atomic_9539 atomic_9540
private theorem within_074_064_agreement : RowAgreement 9538 9540 := RowAgreement.join atomic_9539_agreement atomic_9540_agreement
private def within_074_065 : CountSegment 1 10 13120 9536 9540 3961 3962 := CountSegment.join within_074_063 within_074_064
private theorem within_074_065_agreement : RowAgreement 9536 9540 := RowAgreement.join within_074_063_agreement within_074_064_agreement
private def within_074_066 : CountSegment 1 10 13120 9540 9542 3962 3964 := CountSegment.join atomic_9541 atomic_9542
private theorem within_074_066_agreement : RowAgreement 9540 9542 := RowAgreement.join atomic_9541_agreement atomic_9542_agreement
private def within_074_067 : CountSegment 1 10 13120 9542 9544 3964 3964 := CountSegment.join atomic_9543 atomic_9544
private theorem within_074_067_agreement : RowAgreement 9542 9544 := RowAgreement.join atomic_9543_agreement atomic_9544_agreement
private def within_074_068 : CountSegment 1 10 13120 9540 9544 3962 3964 := CountSegment.join within_074_066 within_074_067
private theorem within_074_068_agreement : RowAgreement 9540 9544 := RowAgreement.join within_074_066_agreement within_074_067_agreement
private def within_074_069 : CountSegment 1 10 13120 9536 9544 3961 3964 := CountSegment.join within_074_065 within_074_068
private theorem within_074_069_agreement : RowAgreement 9536 9544 := RowAgreement.join within_074_065_agreement within_074_068_agreement
private def within_074_070 : CountSegment 1 10 13120 9544 9546 3964 3965 := CountSegment.join atomic_9545 atomic_9546
private theorem within_074_070_agreement : RowAgreement 9544 9546 := RowAgreement.join atomic_9545_agreement atomic_9546_agreement
private def within_074_071 : CountSegment 1 10 13120 9546 9548 3965 3965 := CountSegment.join atomic_9547 atomic_9548
private theorem within_074_071_agreement : RowAgreement 9546 9548 := RowAgreement.join atomic_9547_agreement atomic_9548_agreement
private def within_074_072 : CountSegment 1 10 13120 9544 9548 3964 3965 := CountSegment.join within_074_070 within_074_071
private theorem within_074_072_agreement : RowAgreement 9544 9548 := RowAgreement.join within_074_070_agreement within_074_071_agreement
private def within_074_073 : CountSegment 1 10 13120 9548 9550 3965 3966 := CountSegment.join atomic_9549 atomic_9550
private theorem within_074_073_agreement : RowAgreement 9548 9550 := RowAgreement.join atomic_9549_agreement atomic_9550_agreement
private def within_074_074 : CountSegment 1 10 13120 9550 9552 3966 3966 := CountSegment.join atomic_9551 atomic_9552
private theorem within_074_074_agreement : RowAgreement 9550 9552 := RowAgreement.join atomic_9551_agreement atomic_9552_agreement
private def within_074_075 : CountSegment 1 10 13120 9548 9552 3965 3966 := CountSegment.join within_074_073 within_074_074
private theorem within_074_075_agreement : RowAgreement 9548 9552 := RowAgreement.join within_074_073_agreement within_074_074_agreement
private def within_074_076 : CountSegment 1 10 13120 9544 9552 3964 3966 := CountSegment.join within_074_072 within_074_075
private theorem within_074_076_agreement : RowAgreement 9544 9552 := RowAgreement.join within_074_072_agreement within_074_075_agreement
private def within_074_077 : CountSegment 1 10 13120 9536 9552 3961 3966 := CountSegment.join within_074_069 within_074_076
private theorem within_074_077_agreement : RowAgreement 9536 9552 := RowAgreement.join within_074_069_agreement within_074_076_agreement
private def within_074_078 : CountSegment 1 10 13120 9552 9554 3966 3966 := CountSegment.join atomic_9553 atomic_9554
private theorem within_074_078_agreement : RowAgreement 9552 9554 := RowAgreement.join atomic_9553_agreement atomic_9554_agreement
private def within_074_079 : CountSegment 1 10 13120 9554 9556 3966 3967 := CountSegment.join atomic_9555 atomic_9556
private theorem within_074_079_agreement : RowAgreement 9554 9556 := RowAgreement.join atomic_9555_agreement atomic_9556_agreement
private def within_074_080 : CountSegment 1 10 13120 9552 9556 3966 3967 := CountSegment.join within_074_078 within_074_079
private theorem within_074_080_agreement : RowAgreement 9552 9556 := RowAgreement.join within_074_078_agreement within_074_079_agreement
private def within_074_081 : CountSegment 1 10 13120 9556 9558 3967 3969 := CountSegment.join atomic_9557 atomic_9558
private theorem within_074_081_agreement : RowAgreement 9556 9558 := RowAgreement.join atomic_9557_agreement atomic_9558_agreement
private def within_074_082 : CountSegment 1 10 13120 9558 9560 3969 3970 := CountSegment.join atomic_9559 atomic_9560
private theorem within_074_082_agreement : RowAgreement 9558 9560 := RowAgreement.join atomic_9559_agreement atomic_9560_agreement
private def within_074_083 : CountSegment 1 10 13120 9556 9560 3967 3970 := CountSegment.join within_074_081 within_074_082
private theorem within_074_083_agreement : RowAgreement 9556 9560 := RowAgreement.join within_074_081_agreement within_074_082_agreement
private def within_074_084 : CountSegment 1 10 13120 9552 9560 3966 3970 := CountSegment.join within_074_080 within_074_083
private theorem within_074_084_agreement : RowAgreement 9552 9560 := RowAgreement.join within_074_080_agreement within_074_083_agreement
private def within_074_085 : CountSegment 1 10 13120 9560 9562 3970 3971 := CountSegment.join atomic_9561 atomic_9562
private theorem within_074_085_agreement : RowAgreement 9560 9562 := RowAgreement.join atomic_9561_agreement atomic_9562_agreement
private def within_074_086 : CountSegment 1 10 13120 9562 9564 3971 3972 := CountSegment.join atomic_9563 atomic_9564
private theorem within_074_086_agreement : RowAgreement 9562 9564 := RowAgreement.join atomic_9563_agreement atomic_9564_agreement
private def within_074_087 : CountSegment 1 10 13120 9560 9564 3970 3972 := CountSegment.join within_074_085 within_074_086
private theorem within_074_087_agreement : RowAgreement 9560 9564 := RowAgreement.join within_074_085_agreement within_074_086_agreement
private def within_074_088 : CountSegment 1 10 13120 9564 9566 3972 3973 := CountSegment.join atomic_9565 atomic_9566
private theorem within_074_088_agreement : RowAgreement 9564 9566 := RowAgreement.join atomic_9565_agreement atomic_9566_agreement
private def within_074_089 : CountSegment 1 10 13120 9566 9568 3973 3975 := CountSegment.join atomic_9567 atomic_9568
private theorem within_074_089_agreement : RowAgreement 9566 9568 := RowAgreement.join atomic_9567_agreement atomic_9568_agreement
private def within_074_090 : CountSegment 1 10 13120 9564 9568 3972 3975 := CountSegment.join within_074_088 within_074_089
private theorem within_074_090_agreement : RowAgreement 9564 9568 := RowAgreement.join within_074_088_agreement within_074_089_agreement
private def within_074_091 : CountSegment 1 10 13120 9560 9568 3970 3975 := CountSegment.join within_074_087 within_074_090
private theorem within_074_091_agreement : RowAgreement 9560 9568 := RowAgreement.join within_074_087_agreement within_074_090_agreement
private def within_074_092 : CountSegment 1 10 13120 9552 9568 3966 3975 := CountSegment.join within_074_084 within_074_091
private theorem within_074_092_agreement : RowAgreement 9552 9568 := RowAgreement.join within_074_084_agreement within_074_091_agreement
private def within_074_093 : CountSegment 1 10 13120 9536 9568 3961 3975 := CountSegment.join within_074_077 within_074_092
private theorem within_074_093_agreement : RowAgreement 9536 9568 := RowAgreement.join within_074_077_agreement within_074_092_agreement
private def within_074_094 : CountSegment 1 10 13120 9568 9570 3975 3976 := CountSegment.join atomic_9569 atomic_9570
private theorem within_074_094_agreement : RowAgreement 9568 9570 := RowAgreement.join atomic_9569_agreement atomic_9570_agreement
private def within_074_095 : CountSegment 1 10 13120 9570 9572 3976 3977 := CountSegment.join atomic_9571 atomic_9572
private theorem within_074_095_agreement : RowAgreement 9570 9572 := RowAgreement.join atomic_9571_agreement atomic_9572_agreement
private def within_074_096 : CountSegment 1 10 13120 9568 9572 3975 3977 := CountSegment.join within_074_094 within_074_095
private theorem within_074_096_agreement : RowAgreement 9568 9572 := RowAgreement.join within_074_094_agreement within_074_095_agreement
private def within_074_097 : CountSegment 1 10 13120 9572 9574 3977 3978 := CountSegment.join atomic_9573 atomic_9574
private theorem within_074_097_agreement : RowAgreement 9572 9574 := RowAgreement.join atomic_9573_agreement atomic_9574_agreement
private def within_074_098 : CountSegment 1 10 13120 9574 9576 3978 3980 := CountSegment.join atomic_9575 atomic_9576
private theorem within_074_098_agreement : RowAgreement 9574 9576 := RowAgreement.join atomic_9575_agreement atomic_9576_agreement
private def within_074_099 : CountSegment 1 10 13120 9572 9576 3977 3980 := CountSegment.join within_074_097 within_074_098
private theorem within_074_099_agreement : RowAgreement 9572 9576 := RowAgreement.join within_074_097_agreement within_074_098_agreement
private def within_074_100 : CountSegment 1 10 13120 9568 9576 3975 3980 := CountSegment.join within_074_096 within_074_099
private theorem within_074_100_agreement : RowAgreement 9568 9576 := RowAgreement.join within_074_096_agreement within_074_099_agreement
private def within_074_101 : CountSegment 1 10 13120 9576 9578 3980 3980 := CountSegment.join atomic_9577 atomic_9578
private theorem within_074_101_agreement : RowAgreement 9576 9578 := RowAgreement.join atomic_9577_agreement atomic_9578_agreement
private def within_074_102 : CountSegment 1 10 13120 9578 9580 3980 3980 := CountSegment.join atomic_9579 atomic_9580
private theorem within_074_102_agreement : RowAgreement 9578 9580 := RowAgreement.join atomic_9579_agreement atomic_9580_agreement
private def within_074_103 : CountSegment 1 10 13120 9576 9580 3980 3980 := CountSegment.join within_074_101 within_074_102
private theorem within_074_103_agreement : RowAgreement 9576 9580 := RowAgreement.join within_074_101_agreement within_074_102_agreement
private def within_074_104 : CountSegment 1 10 13120 9580 9582 3980 3981 := CountSegment.join atomic_9581 atomic_9582
private theorem within_074_104_agreement : RowAgreement 9580 9582 := RowAgreement.join atomic_9581_agreement atomic_9582_agreement
private def within_074_105 : CountSegment 1 10 13120 9582 9584 3981 3982 := CountSegment.join atomic_9583 atomic_9584
private theorem within_074_105_agreement : RowAgreement 9582 9584 := RowAgreement.join atomic_9583_agreement atomic_9584_agreement
private def within_074_106 : CountSegment 1 10 13120 9580 9584 3980 3982 := CountSegment.join within_074_104 within_074_105
private theorem within_074_106_agreement : RowAgreement 9580 9584 := RowAgreement.join within_074_104_agreement within_074_105_agreement
private def within_074_107 : CountSegment 1 10 13120 9576 9584 3980 3982 := CountSegment.join within_074_103 within_074_106
private theorem within_074_107_agreement : RowAgreement 9576 9584 := RowAgreement.join within_074_103_agreement within_074_106_agreement
private def within_074_108 : CountSegment 1 10 13120 9568 9584 3975 3982 := CountSegment.join within_074_100 within_074_107
private theorem within_074_108_agreement : RowAgreement 9568 9584 := RowAgreement.join within_074_100_agreement within_074_107_agreement
private def within_074_109 : CountSegment 1 10 13120 9584 9586 3982 3983 := CountSegment.join atomic_9585 atomic_9586
private theorem within_074_109_agreement : RowAgreement 9584 9586 := RowAgreement.join atomic_9585_agreement atomic_9586_agreement
private def within_074_110 : CountSegment 1 10 13120 9586 9588 3983 3983 := CountSegment.join atomic_9587 atomic_9588
private theorem within_074_110_agreement : RowAgreement 9586 9588 := RowAgreement.join atomic_9587_agreement atomic_9588_agreement
private def within_074_111 : CountSegment 1 10 13120 9584 9588 3982 3983 := CountSegment.join within_074_109 within_074_110
private theorem within_074_111_agreement : RowAgreement 9584 9588 := RowAgreement.join within_074_109_agreement within_074_110_agreement
private def within_074_112 : CountSegment 1 10 13120 9588 9590 3983 3983 := CountSegment.join atomic_9589 atomic_9590
private theorem within_074_112_agreement : RowAgreement 9588 9590 := RowAgreement.join atomic_9589_agreement atomic_9590_agreement
private def within_074_113 : CountSegment 1 10 13120 9590 9592 3983 3983 := CountSegment.join atomic_9591 atomic_9592
private theorem within_074_113_agreement : RowAgreement 9590 9592 := RowAgreement.join atomic_9591_agreement atomic_9592_agreement
private def within_074_114 : CountSegment 1 10 13120 9588 9592 3983 3983 := CountSegment.join within_074_112 within_074_113
private theorem within_074_114_agreement : RowAgreement 9588 9592 := RowAgreement.join within_074_112_agreement within_074_113_agreement
private def within_074_115 : CountSegment 1 10 13120 9584 9592 3982 3983 := CountSegment.join within_074_111 within_074_114
private theorem within_074_115_agreement : RowAgreement 9584 9592 := RowAgreement.join within_074_111_agreement within_074_114_agreement
private def within_074_116 : CountSegment 1 10 13120 9592 9594 3983 3984 := CountSegment.join atomic_9593 atomic_9594
private theorem within_074_116_agreement : RowAgreement 9592 9594 := RowAgreement.join atomic_9593_agreement atomic_9594_agreement
private def within_074_117 : CountSegment 1 10 13120 9594 9596 3984 3984 := CountSegment.join atomic_9595 atomic_9596
private theorem within_074_117_agreement : RowAgreement 9594 9596 := RowAgreement.join atomic_9595_agreement atomic_9596_agreement
private def within_074_118 : CountSegment 1 10 13120 9592 9596 3983 3984 := CountSegment.join within_074_116 within_074_117
private theorem within_074_118_agreement : RowAgreement 9592 9596 := RowAgreement.join within_074_116_agreement within_074_117_agreement
private def within_074_119 : CountSegment 1 10 13120 9596 9598 3984 3985 := CountSegment.join atomic_9597 atomic_9598
private theorem within_074_119_agreement : RowAgreement 9596 9598 := RowAgreement.join atomic_9597_agreement atomic_9598_agreement
private def within_074_120 : CountSegment 1 10 13120 9598 9600 3985 3985 := CountSegment.join atomic_9599 atomic_9600
private theorem within_074_120_agreement : RowAgreement 9598 9600 := RowAgreement.join atomic_9599_agreement atomic_9600_agreement
private def within_074_121 : CountSegment 1 10 13120 9596 9600 3984 3985 := CountSegment.join within_074_119 within_074_120
private theorem within_074_121_agreement : RowAgreement 9596 9600 := RowAgreement.join within_074_119_agreement within_074_120_agreement
private def within_074_122 : CountSegment 1 10 13120 9592 9600 3983 3985 := CountSegment.join within_074_118 within_074_121
private theorem within_074_122_agreement : RowAgreement 9592 9600 := RowAgreement.join within_074_118_agreement within_074_121_agreement
private def within_074_123 : CountSegment 1 10 13120 9584 9600 3982 3985 := CountSegment.join within_074_115 within_074_122
private theorem within_074_123_agreement : RowAgreement 9584 9600 := RowAgreement.join within_074_115_agreement within_074_122_agreement
private def within_074_124 : CountSegment 1 10 13120 9568 9600 3975 3985 := CountSegment.join within_074_108 within_074_123
private theorem within_074_124_agreement : RowAgreement 9568 9600 := RowAgreement.join within_074_108_agreement within_074_123_agreement
private def within_074_125 : CountSegment 1 10 13120 9536 9600 3961 3985 := CountSegment.join within_074_093 within_074_124
private theorem within_074_125_agreement : RowAgreement 9536 9600 := RowAgreement.join within_074_093_agreement within_074_124_agreement
private def within_074_126 : CountSegment 1 10 13120 9472 9600 3934 3985 := CountSegment.join within_074_062 within_074_125
private theorem within_074_126_agreement : RowAgreement 9472 9600 := RowAgreement.join within_074_062_agreement within_074_125_agreement
def segment_074 : CountSegment 1 10 13120 9472 9600 3934 3985 := within_074_126
theorem agreement_074 : RowAgreement 9472 9600 := within_074_126_agreement
end ProjectEulerSolutions.P58Numerical
