import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3329 : RowCertificate 3329 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3329 1) := by change ¬ Nat.Prime 44335623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14778541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3329 2) := by change ¬ Nat.Prime 44328965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8865793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3329 3) := by change ¬ Nat.Prime 44322307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2332753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3329 : row 3329 = 0 := (row_3329.checked).trans rfl
theorem model_count_3329 : modelRow 3329 = 0 := by decide
theorem modelrow_agrees_3329 : modelRow 3329 = row 3329 := model_count_3329.trans row_count_3329.symm
def row_3330 : RowCertificate 3330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3330 1) := by change ¬ Nat.Prime 44362261; exact PrattCertificates.composite_of_product (factor := 2179) (cofactor := 20359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3330 2) := by change ¬ Nat.Prime 44355601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2609153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3330 3) := by change ¬ Nat.Prime 44348941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6335563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3330 : row 3330 = 0 := (row_3330.checked).trans rfl
theorem model_count_3330 : modelRow 3330 = 0 := by decide
theorem modelrow_agrees_3330 : modelRow 3330 = row 3330 := model_count_3330.trans row_count_3330.symm
def row_3331 : RowCertificate 3331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3331 1) := by change ¬ Nat.Prime 44388907; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 727687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3331 2) := by change ¬ Nat.Prime 44382245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8876449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3331 3) := by change ¬ Nat.Prime 44375583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14791861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3331 : row 3331 = 0 := (row_3331.checked).trans rfl
theorem model_count_3331 : modelRow 3331 = 0 := by decide
theorem modelrow_agrees_3331 : modelRow 3331 = row 3331 := model_count_3331.trans row_count_3331.symm
def row_3332 : RowCertificate 3332 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3332 1) := by change ¬ Nat.Prime 44415561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14805187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3332 2) := by change ¬ Nat.Prime 44408897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3416069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3332 3) := by change ¬ Nat.Prime 44402233; exact PrattCertificates.composite_of_product (factor := 6451) (cofactor := 6883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3332 : row 3332 = 0 := (row_3332.checked).trans rfl
theorem model_count_3332 : modelRow 3332 = 0 := by decide
theorem modelrow_agrees_3332 : modelRow 3332 = row 3332 := model_count_3332.trans row_count_3332.symm
def row_3333 : RowCertificate 3333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3333 1) := by change ¬ Nat.Prime 44442223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6348889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3333 2) := by change ¬ Nat.Prime 44435557; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1200961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3333 3) := by change ¬ Nat.Prime 44428891; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3417607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3333 : row 3333 = 0 := (row_3333.checked).trans rfl
theorem model_count_3333 : modelRow 3333 = 0 := by decide
theorem modelrow_agrees_3333 : modelRow 3333 = row 3333 := model_count_3333.trans row_count_3333.symm
def row_3334 : RowCertificate 3334 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3334 1) := by change ¬ Nat.Prime 44468893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6352699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3334 2) := by change ¬ Nat.Prime 44462225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8892445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3334 3) := by change ¬ Nat.Prime 44455557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14818519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3334 : row 3334 = 0 := (row_3334.checked).trans rfl
theorem model_count_3334 : modelRow 3334 = 0 := by decide
theorem modelrow_agrees_3334 : modelRow 3334 = row 3334 := model_count_3334.trans row_count_3334.symm
def row_3335 : RowCertificate 3335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3335 1) := by change ¬ Nat.Prime 44495571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14831857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3335 2) := by change ¬ Nat.Prime 44488901; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 609437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3335 3) := by change Nat.Prime 44482231; exact ProjectEulerSolutions.P58PrimeNodes.p_44482231.prime
    exact decide_eq_true prime
theorem row_count_3335 : row 3335 = 1 := (row_3335.checked).trans rfl
theorem model_count_3335 : modelRow 3335 = 1 := by decide
theorem modelrow_agrees_3335 : modelRow 3335 = row 3335 := model_count_3335.trans row_count_3335.symm
def row_3336 : RowCertificate 3336 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3336 1) := by change ¬ Nat.Prime 44522257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3424789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3336 2) := by change ¬ Nat.Prime 44515585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8903117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3336 3) := by change ¬ Nat.Prime 44508913; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1035091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3336 : row 3336 = 0 := (row_3336.checked).trans rfl
theorem model_count_3336 : modelRow 3336 = 0 := by decide
theorem modelrow_agrees_3336 : modelRow 3336 = row 3336 := model_count_3336.trans row_count_3336.symm
def row_3337 : RowCertificate 3337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3337 1) := by change ¬ Nat.Prime 44548951; exact PrattCertificates.composite_of_product (factor := 997) (cofactor := 44683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3337 2) := by change ¬ Nat.Prime 44542277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3426329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3337 3) := by change ¬ Nat.Prime 44535603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14845201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3337 : row 3337 = 0 := (row_3337.checked).trans rfl
theorem model_count_3337 : modelRow 3337 = 0 := by decide
theorem modelrow_agrees_3337 : modelRow 3337 = row 3337 := model_count_3337.trans row_count_3337.symm
def row_3338 : RowCertificate 3338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3338 1) := by change ¬ Nat.Prime 44575653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14858551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3338 2) := by change ¬ Nat.Prime 44568977; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 441277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3338 3) := by change ¬ Nat.Prime 44562301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6366043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3338 : row 3338 = 0 := (row_3338.checked).trans rfl
theorem model_count_3338 : modelRow 3338 = 0 := by decide
theorem modelrow_agrees_3338 : modelRow 3338 = row 3338 := model_count_3338.trans row_count_3338.symm
def row_3339 : RowCertificate 3339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3339 1) := by change ¬ Nat.Prime 44602363; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3430951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3339 2) := by change ¬ Nat.Prime 44595685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8919137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3339 3) := by change ¬ Nat.Prime 44589007; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 72739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3339 : row 3339 = 0 := (row_3339.checked).trans rfl
theorem model_count_3339 : modelRow 3339 = 0 := by decide
theorem modelrow_agrees_3339 : modelRow 3339 = row 3339 := model_count_3339.trans row_count_3339.symm
def row_3340 : RowCertificate 3340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3340 1) := by change ¬ Nat.Prime 44629081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6375583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3340 2) := by change ¬ Nat.Prime 44622401; exact PrattCertificates.composite_of_product (factor := 1693) (cofactor := 26357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3340 3) := by change ¬ Nat.Prime 44615721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14871907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3340 : row 3340 = 0 := (row_3340.checked).trans rfl
theorem model_count_3340 : modelRow 3340 = 0 := by decide
theorem modelrow_agrees_3340 : modelRow 3340 = row 3340 := model_count_3340.trans row_count_3340.symm
def row_3341 : RowCertificate 3341 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3341 1) := by change ¬ Nat.Prime 44655807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14885269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3341 2) := by change ¬ Nat.Prime 44649125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8929825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3341 3) := by change Nat.Prime 44642443; exact ProjectEulerSolutions.P58PrimeNodes.p_44642443.prime
    exact decide_eq_true prime
theorem row_count_3341 : row 3341 = 1 := (row_3341.checked).trans rfl
theorem model_count_3341 : modelRow 3341 = 1 := by decide
theorem modelrow_agrees_3341 : modelRow 3341 = row 3341 := model_count_3341.trans row_count_3341.symm
def row_3342 : RowCertificate 3342 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3342 1) := by change ¬ Nat.Prime 44682541; exact PrattCertificates.composite_of_product (factor := 5281) (cofactor := 8461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3342 2) := by change ¬ Nat.Prime 44675857; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 226781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3342 3) := by change ¬ Nat.Prime 44669173; exact PrattCertificates.composite_of_product (factor := 3061) (cofactor := 14593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3342 : row 3342 = 0 := (row_3342.checked).trans rfl
theorem model_count_3342 : modelRow 3342 = 0 := by decide
theorem modelrow_agrees_3342 : modelRow 3342 = row 3342 := model_count_3342.trans row_count_3342.symm
def row_3343 : RowCertificate 3343 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3343 1) := by change ¬ Nat.Prime 44709283; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1208359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3343 2) := by change ¬ Nat.Prime 44702597; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 112601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3343 3) := by change ¬ Nat.Prime 44695911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14898637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3343 : row 3343 = 0 := (row_3343.checked).trans rfl
theorem model_count_3343 : modelRow 3343 = 0 := by decide
theorem modelrow_agrees_3343 : modelRow 3343 = row 3343 := model_count_3343.trans row_count_3343.symm
def row_3344 : RowCertificate 3344 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3344 1) := by change ¬ Nat.Prime 44736033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14912011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3344 2) := by change ¬ Nat.Prime 44729345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8945869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3344 3) := by change ¬ Nat.Prime 44722657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6388951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3344 : row 3344 = 0 := (row_3344.checked).trans rfl
theorem model_count_3344 : modelRow 3344 = 0 := by decide
theorem modelrow_agrees_3344 : modelRow 3344 = row 3344 := model_count_3344.trans row_count_3344.symm
def row_3345 : RowCertificate 3345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3345 1) := by change ¬ Nat.Prime 44762791; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1443961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3345 2) := by change ¬ Nat.Prime 44756101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3442777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3345 3) := by change ¬ Nat.Prime 44749411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6392773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3345 : row 3345 = 0 := (row_3345.checked).trans rfl
theorem model_count_3345 : modelRow 3345 = 0 := by decide
theorem modelrow_agrees_3345 : modelRow 3345 = row 3345 := model_count_3345.trans row_count_3345.symm
def row_3346 : RowCertificate 3346 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3346 1) := by change Nat.Prime 44789557; exact ProjectEulerSolutions.P58PrimeNodes.p_44789557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3346 2) := by change ¬ Nat.Prime 44782865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8956573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3346 3) := by change ¬ Nat.Prime 44776173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14925391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3346 : row 3346 = 1 := (row_3346.checked).trans rfl
theorem model_count_3346 : modelRow 3346 = 1 := by decide
theorem modelrow_agrees_3346 : modelRow 3346 = row 3346 := model_count_3346.trans row_count_3346.symm
def row_3347 : RowCertificate 3347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3347 1) := by change ¬ Nat.Prime 44816331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14938777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3347 2) := by change ¬ Nat.Prime 44809637; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2635861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3347 3) := by change ¬ Nat.Prime 44802943; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 103471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3347 : row 3347 = 0 := (row_3347.checked).trans rfl
theorem model_count_3347 : modelRow 3347 = 0 := by decide
theorem modelrow_agrees_3347 : modelRow 3347 = row 3347 := model_count_3347.trans row_count_3347.symm
def row_3348 : RowCertificate 3348 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3348 1) := by change ¬ Nat.Prime 44843113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6406159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3348 2) := by change ¬ Nat.Prime 44836417; exact PrattCertificates.composite_of_product (factor := 2909) (cofactor := 15413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3348 3) := by change ¬ Nat.Prime 44829721; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2359459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3348 : row 3348 = 0 := (row_3348.checked).trans rfl
theorem model_count_3348 : modelRow 3348 = 0 := by decide
theorem modelrow_agrees_3348 : modelRow 3348 = row 3348 := model_count_3348.trans row_count_3348.symm
def row_3349 : RowCertificate 3349 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3349 1) := by change ¬ Nat.Prime 44869903; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3451531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3349 2) := by change ¬ Nat.Prime 44863205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8972641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3349 3) := by change ¬ Nat.Prime 44856507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14952169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3349 : row 3349 = 0 := (row_3349.checked).trans rfl
theorem model_count_3349 : modelRow 3349 = 0 := by decide
theorem modelrow_agrees_3349 : modelRow 3349 = row 3349 := model_count_3349.trans row_count_3349.symm
def row_3350 : RowCertificate 3350 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3350 1) := by change ¬ Nat.Prime 44896701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14965567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3350 2) := by change ¬ Nat.Prime 44890001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3453077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3350 3) := by change ¬ Nat.Prime 44883301; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2362279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3350 : row 3350 = 0 := (row_3350.checked).trans rfl
theorem model_count_3350 : modelRow 3350 = 0 := by decide
theorem modelrow_agrees_3350 : modelRow 3350 = row 3350 := model_count_3350.trans row_count_3350.symm
def row_3351 : RowCertificate 3351 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3351 1) := by change Nat.Prime 44923507; exact ProjectEulerSolutions.P58PrimeNodes.p_44923507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3351 2) := by change ¬ Nat.Prime 44916805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8983361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3351 3) := by change ¬ Nat.Prime 44910103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6415729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3351 : row 3351 = 1 := (row_3351.checked).trans rfl
theorem model_count_3351 : modelRow 3351 = 1 := by decide
theorem modelrow_agrees_3351 : modelRow 3351 = row 3351 := model_count_3351.trans row_count_3351.symm
def row_3352 : RowCertificate 3352 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3352 1) := by change ¬ Nat.Prime 44950321; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3457717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3352 2) := by change Nat.Prime 44943617; exact ProjectEulerSolutions.P58PrimeNodes.p_44943617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3352 3) := by change ¬ Nat.Prime 44936913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14978971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3352 : row 3352 = 1 := (row_3352.checked).trans rfl
theorem model_count_3352 : modelRow 3352 = 1 := by decide
theorem modelrow_agrees_3352 : modelRow 3352 = row 3352 := model_count_3352.trans row_count_3352.symm
def row_3353 : RowCertificate 3353 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3353 1) := by change ¬ Nat.Prime 44977143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14992381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3353 2) := by change ¬ Nat.Prime 44970437; exact PrattCertificates.composite_of_product (factor := 4493) (cofactor := 10009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3353 3) := by change ¬ Nat.Prime 44963731; exact PrattCertificates.composite_of_product (factor := 4057) (cofactor := 11083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3353 : row 3353 = 0 := (row_3353.checked).trans rfl
theorem model_count_3353 : modelRow 3353 = 0 := by decide
theorem modelrow_agrees_3353 : modelRow 3353 = row 3353 := model_count_3353.trans row_count_3353.symm
def row_3354 : RowCertificate 3354 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3354 1) := by change ¬ Nat.Prime 45003973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6429139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3354 2) := by change ¬ Nat.Prime 44997265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8999453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3354 3) := by change ¬ Nat.Prime 44990557; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1215961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3354 : row 3354 = 0 := (row_3354.checked).trans rfl
theorem model_count_3354 : modelRow 3354 = 0 := by decide
theorem modelrow_agrees_3354 : modelRow 3354 = row 3354 := model_count_3354.trans row_count_3354.symm
def row_3355 : RowCertificate 3355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3355 1) := by change ¬ Nat.Prime 45030811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6432973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3355 2) := by change Nat.Prime 45024101; exact ProjectEulerSolutions.P58PrimeNodes.p_45024101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3355 3) := by change ¬ Nat.Prime 45017391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15005797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3355 : row 3355 = 1 := (row_3355.checked).trans rfl
theorem model_count_3355 : modelRow 3355 = 1 := by decide
theorem modelrow_agrees_3355 : modelRow 3355 = row 3355 := model_count_3355.trans row_count_3355.symm
def row_3356 : RowCertificate 3356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3356 1) := by change ¬ Nat.Prime 45057657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15019219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3356 2) := by change ¬ Nat.Prime 45050945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9010189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3356 3) := by change ¬ Nat.Prime 45044233; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3464941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3356 : row 3356 = 0 := (row_3356.checked).trans rfl
theorem model_count_3356 : modelRow 3356 = 0 := by decide
theorem modelrow_agrees_3356 : modelRow 3356 = row 3356 := model_count_3356.trans row_count_3356.symm
def row_3357 : RowCertificate 3357 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3357 1) := by change ¬ Nat.Prime 45084511; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2372869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3357 2) := by change Nat.Prime 45077797; exact ProjectEulerSolutions.P58PrimeNodes.p_45077797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3357 3) := by change ¬ Nat.Prime 45071083; exact PrattCertificates.composite_of_product (factor := 4159) (cofactor := 10837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3357 : row 3357 = 1 := (row_3357.checked).trans rfl
theorem model_count_3357 : modelRow 3357 = 1 := by decide
theorem modelrow_agrees_3357 : modelRow 3357 = row 3357 := model_count_3357.trans row_count_3357.symm
def row_3358 : RowCertificate 3358 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3358 1) := by change ¬ Nat.Prime 45111373; exact PrattCertificates.composite_of_product (factor := 2851) (cofactor := 15823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3358 2) := by change ¬ Nat.Prime 45104657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3469589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3358 3) := by change ¬ Nat.Prime 45097941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15032647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3358 : row 3358 = 0 := (row_3358.checked).trans rfl
theorem model_count_3358 : modelRow 3358 = 0 := by decide
theorem modelrow_agrees_3358 : modelRow 3358 = row 3358 := model_count_3358.trans row_count_3358.symm
def row_3359 : RowCertificate 3359 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3359 1) := by change ¬ Nat.Prime 45138243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15046081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3359 2) := by change ¬ Nat.Prime 45131525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9026305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3359 3) := by change ¬ Nat.Prime 45124807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6446401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3359 : row 3359 = 0 := (row_3359.checked).trans rfl
theorem model_count_3359 : modelRow 3359 = 0 := by decide
theorem modelrow_agrees_3359 : modelRow 3359 = row 3359 := model_count_3359.trans row_count_3359.symm
def row_3360 : RowCertificate 3360 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3360 1) := by change Nat.Prime 45165121; exact ProjectEulerSolutions.P58PrimeNodes.p_45165121.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3360 2) := by change ¬ Nat.Prime 45158401; exact PrattCertificates.composite_of_product (factor := 1049) (cofactor := 43049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3360 3) := by change Nat.Prime 45151681; exact ProjectEulerSolutions.P58PrimeNodes.p_45151681.prime
    exact decide_eq_true prime
theorem row_count_3360 : row 3360 = 2 := (row_3360.checked).trans rfl
theorem model_count_3360 : modelRow 3360 = 2 := by decide
theorem modelrow_agrees_3360 : modelRow 3360 = row 3360 := model_count_3360.trans row_count_3360.symm
def row_3361 : RowCertificate 3361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3361 1) := by change ¬ Nat.Prime 45192007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6456001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3361 2) := by change ¬ Nat.Prime 45185285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9037057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3361 3) := by change ¬ Nat.Prime 45178563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15059521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3361 : row 3361 = 0 := (row_3361.checked).trans rfl
theorem model_count_3361 : modelRow 3361 = 0 := by decide
theorem modelrow_agrees_3361 : modelRow 3361 = row 3361 := model_count_3361.trans row_count_3361.symm
def row_3362 : RowCertificate 3362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3362 1) := by change ¬ Nat.Prime 45218901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15072967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3362 2) := by change Nat.Prime 45212177; exact ProjectEulerSolutions.P58PrimeNodes.p_45212177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3362 3) := by change ¬ Nat.Prime 45205453; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1221769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3362 : row 3362 = 1 := (row_3362.checked).trans rfl
theorem model_count_3362 : modelRow 3362 = 1 := by decide
theorem modelrow_agrees_3362 : modelRow 3362 = row 3362 := model_count_3362.trans row_count_3362.symm
def row_3363 : RowCertificate 3363 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3363 1) := by change ¬ Nat.Prime 45245803; exact PrattCertificates.composite_of_product (factor := 883) (cofactor := 51241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3363 2) := by change ¬ Nat.Prime 45239077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3479929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3363 3) := by change Nat.Prime 45232351; exact ProjectEulerSolutions.P58PrimeNodes.p_45232351.prime
    exact decide_eq_true prime
theorem row_count_3363 : row 3363 = 1 := (row_3363.checked).trans rfl
theorem model_count_3363 : modelRow 3363 = 1 := by decide
theorem modelrow_agrees_3363 : modelRow 3363 = row 3363 := model_count_3363.trans row_count_3363.symm
def row_3364 : RowCertificate 3364 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3364 1) := by change ¬ Nat.Prime 45272713; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 466729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3364 2) := by change ¬ Nat.Prime 45265985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9053197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3364 3) := by change ¬ Nat.Prime 45259257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15086419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3364 : row 3364 = 0 := (row_3364.checked).trans rfl
theorem model_count_3364 : modelRow 3364 = 0 := by decide
theorem modelrow_agrees_3364 : modelRow 3364 = row 3364 := model_count_3364.trans row_count_3364.symm
def row_3365 : RowCertificate 3365 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3365 1) := by change ¬ Nat.Prime 45299631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15099877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3365 2) := by change ¬ Nat.Prime 45292901; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 508909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3365 3) := by change ¬ Nat.Prime 45286171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6469453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3365 : row 3365 = 0 := (row_3365.checked).trans rfl
theorem model_count_3365 : modelRow 3365 = 0 := by decide
theorem modelrow_agrees_3365 : modelRow 3365 = row 3365 := model_count_3365.trans row_count_3365.symm
def row_3366 : RowCertificate 3366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3366 1) := by change ¬ Nat.Prime 45326557; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1462147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3366 2) := by change ¬ Nat.Prime 45319825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9063965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3366 3) := by change ¬ Nat.Prime 45313093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6473299) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3366 : row 3366 = 0 := (row_3366.checked).trans rfl
theorem model_count_3366 : modelRow 3366 = 0 := by decide
theorem modelrow_agrees_3366 : modelRow 3366 = row 3366 := model_count_3366.trans row_count_3366.symm
def row_3367 : RowCertificate 3367 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3367 1) := by change Nat.Prime 45353491; exact ProjectEulerSolutions.P58PrimeNodes.p_45353491.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3367 2) := by change Nat.Prime 45346757; exact ProjectEulerSolutions.P58PrimeNodes.p_45346757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3367 3) := by change ¬ Nat.Prime 45340023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15113341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3367 : row 3367 = 2 := (row_3367.checked).trans rfl
theorem model_count_3367 : modelRow 3367 = 2 := by decide
theorem modelrow_agrees_3367 : modelRow 3367 = row 3367 := model_count_3367.trans row_count_3367.symm
def row_3368 : RowCertificate 3368 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3368 1) := by change ¬ Nat.Prime 45380433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15126811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3368 2) := by change ¬ Nat.Prime 45373697; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2669041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3368 3) := by change ¬ Nat.Prime 45366961; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 198109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3368 : row 3368 = 0 := (row_3368.checked).trans rfl
theorem model_count_3368 : modelRow 3368 = 0 := by decide
theorem modelrow_agrees_3368 : modelRow 3368 = row 3368 := model_count_3368.trans row_count_3368.symm
def row_3369 : RowCertificate 3369 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3369 1) := by change ¬ Nat.Prime 45407383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6486769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3369 2) := by change ¬ Nat.Prime 45400645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9080129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3369 3) := by change ¬ Nat.Prime 45393907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3491839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3369 : row 3369 = 0 := (row_3369.checked).trans rfl
theorem model_count_3369 : modelRow 3369 = 0 := by decide
theorem modelrow_agrees_3369 : modelRow 3369 = row 3369 := model_count_3369.trans row_count_3369.symm
def row_3370 : RowCertificate 3370 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3370 1) := by change Nat.Prime 45434341; exact ProjectEulerSolutions.P58PrimeNodes.p_45434341.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3370 2) := by change ¬ Nat.Prime 45427601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1566469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3370 3) := by change ¬ Nat.Prime 45420861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15140287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3370 : row 3370 = 1 := (row_3370.checked).trans rfl
theorem model_count_3370 : modelRow 3370 = 1 := by decide
theorem modelrow_agrees_3370 : modelRow 3370 = row 3370 := model_count_3370.trans row_count_3370.symm
def row_3371 : RowCertificate 3371 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3371 1) := by change ¬ Nat.Prime 45461307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15153769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3371 2) := by change ¬ Nat.Prime 45454565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9090913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3371 3) := by change ¬ Nat.Prime 45447823; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 441241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3371 : row 3371 = 0 := (row_3371.checked).trans rfl
theorem model_count_3371 : modelRow 3371 = 0 := by decide
theorem modelrow_agrees_3371 : modelRow 3371 = row 3371 := model_count_3371.trans row_count_3371.symm
def row_3372 : RowCertificate 3372 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3372 1) := by change ¬ Nat.Prime 45488281; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1229413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3372 2) := by change ¬ Nat.Prime 45481537; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 67181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3372 3) := by change ¬ Nat.Prime 45474793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6496399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3372 : row 3372 = 0 := (row_3372.checked).trans rfl
theorem model_count_3372 : modelRow 3372 = 0 := by decide
theorem modelrow_agrees_3372 : modelRow 3372 = row 3372 := model_count_3372.trans row_count_3372.symm
def row_3373 : RowCertificate 3373 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3373 1) := by change ¬ Nat.Prime 45515263; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 167953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3373 2) := by change ¬ Nat.Prime 45508517; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 99581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3373 3) := by change ¬ Nat.Prime 45501771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15167257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3373 : row 3373 = 0 := (row_3373.checked).trans rfl
theorem model_count_3373 : modelRow 3373 = 0 := by decide
theorem modelrow_agrees_3373 : modelRow 3373 = row 3373 := model_count_3373.trans row_count_3373.symm
def row_3374 : RowCertificate 3374 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3374 1) := by change ¬ Nat.Prime 45542253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15180751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3374 2) := by change ¬ Nat.Prime 45535505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9107101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3374 3) := by change ¬ Nat.Prime 45528757; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 160879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3374 : row 3374 = 0 := (row_3374.checked).trans rfl
theorem model_count_3374 : modelRow 3374 = 0 := by decide
theorem modelrow_agrees_3374 : modelRow 3374 = row 3374 := model_count_3374.trans row_count_3374.symm
def row_3375 : RowCertificate 3375 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3375 1) := by change ¬ Nat.Prime 45569251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6509893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3375 2) := by change ¬ Nat.Prime 45562501; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 332573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3375 3) := by change Nat.Prime 45555751; exact ProjectEulerSolutions.P58PrimeNodes.p_45555751.prime
    exact decide_eq_true prime
theorem row_count_3375 : row 3375 = 1 := (row_3375.checked).trans rfl
theorem model_count_3375 : modelRow 3375 = 1 := by decide
theorem modelrow_agrees_3375 : modelRow 3375 = row 3375 := model_count_3375.trans row_count_3375.symm
def row_3376 : RowCertificate 3376 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3376 1) := by change ¬ Nat.Prime 45596257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6513751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3376 2) := by change ¬ Nat.Prime 45589505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9117901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3376 3) := by change ¬ Nat.Prime 45582753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15194251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3376 : row 3376 = 0 := (row_3376.checked).trans rfl
theorem model_count_3376 : modelRow 3376 = 0 := by decide
theorem modelrow_agrees_3376 : modelRow 3376 = row 3376 := model_count_3376.trans row_count_3376.symm
def row_3377 : RowCertificate 3377 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3377 1) := by change ¬ Nat.Prime 45623271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15207757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3377 2) := by change ¬ Nat.Prime 45616517; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 860689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3377 3) := by change ¬ Nat.Prime 45609763; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 130687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3377 : row 3377 = 0 := (row_3377.checked).trans rfl
theorem model_count_3377 : modelRow 3377 = 0 := by decide
theorem modelrow_agrees_3377 : modelRow 3377 = row 3377 := model_count_3377.trans row_count_3377.symm
def row_3378 : RowCertificate 3378 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3378 1) := by change ¬ Nat.Prime 45650293; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3511561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3378 2) := by change ¬ Nat.Prime 45643537; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1113257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3378 3) := by change ¬ Nat.Prime 45636781; exact PrattCertificates.composite_of_product (factor := 4663) (cofactor := 9787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3378 : row 3378 = 0 := (row_3378.checked).trans rfl
theorem model_count_3378 : modelRow 3378 = 0 := by decide
theorem modelrow_agrees_3378 : modelRow 3378 = row 3378 := model_count_3378.trans row_count_3378.symm
def row_3379 : RowCertificate 3379 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3379 1) := by change ¬ Nat.Prime 45677323; exact PrattCertificates.composite_of_product (factor := 823) (cofactor := 55501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3379 2) := by change ¬ Nat.Prime 45670565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9134113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3379 3) := by change ¬ Nat.Prime 45663807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15221269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3379 : row 3379 = 0 := (row_3379.checked).trans rfl
theorem model_count_3379 : modelRow 3379 = 0 := by decide
theorem modelrow_agrees_3379 : modelRow 3379 = row 3379 := model_count_3379.trans row_count_3379.symm
def row_3380 : RowCertificate 3380 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3380 1) := by change ¬ Nat.Prime 45704361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15234787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3380 2) := by change ¬ Nat.Prime 45697601; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 749141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3380 3) := by change ¬ Nat.Prime 45690841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6527263) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3380 : row 3380 = 0 := (row_3380.checked).trans rfl
theorem model_count_3380 : modelRow 3380 = 0 := by decide
theorem modelrow_agrees_3380 : modelRow 3380 = row 3380 := model_count_3380.trans row_count_3380.symm
def row_3381 : RowCertificate 3381 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3381 1) := by change ¬ Nat.Prime 45731407; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 302857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3381 2) := by change ¬ Nat.Prime 45724645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9144929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3381 3) := by change Nat.Prime 45717883; exact ProjectEulerSolutions.P58PrimeNodes.p_45717883.prime
    exact decide_eq_true prime
theorem row_count_3381 : row 3381 = 1 := (row_3381.checked).trans rfl
theorem model_count_3381 : modelRow 3381 = 1 := by decide
theorem modelrow_agrees_3381 : modelRow 3381 = row 3381 := model_count_3381.trans row_count_3381.symm
def row_3382 : RowCertificate 3382 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3382 1) := by change ¬ Nat.Prime 45758461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6536923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3382 2) := by change Nat.Prime 45751697; exact ProjectEulerSolutions.P58PrimeNodes.p_45751697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3382 3) := by change ¬ Nat.Prime 45744933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15248311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3382 : row 3382 = 1 := (row_3382.checked).trans rfl
theorem model_count_3382 : modelRow 3382 = 1 := by decide
theorem modelrow_agrees_3382 : modelRow 3382 = row 3382 := model_count_3382.trans row_count_3382.symm
def row_3383 : RowCertificate 3383 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3383 1) := by change ¬ Nat.Prime 45785523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15261841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3383 2) := by change ¬ Nat.Prime 45778757; exact PrattCertificates.composite_of_product (factor := 4561) (cofactor := 10037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3383 3) := by change ¬ Nat.Prime 45771991; exact PrattCertificates.composite_of_product (factor := 4729) (cofactor := 9679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3383 : row 3383 = 0 := (row_3383.checked).trans rfl
theorem model_count_3383 : modelRow 3383 = 0 := by decide
theorem modelrow_agrees_3383 : modelRow 3383 = row 3383 := model_count_3383.trans row_count_3383.symm
def row_3384 : RowCertificate 3384 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3384 1) := by change ¬ Nat.Prime 45812593; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 329587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3384 2) := by change ¬ Nat.Prime 45805825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9161165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3384 3) := by change Nat.Prime 45799057; exact ProjectEulerSolutions.P58PrimeNodes.p_45799057.prime
    exact decide_eq_true prime
theorem row_count_3384 : row 3384 = 1 := (row_3384.checked).trans rfl
theorem model_count_3384 : modelRow 3384 = 1 := by decide
theorem modelrow_agrees_3384 : modelRow 3384 = row 3384 := model_count_3384.trans row_count_3384.symm
def row_3385 : RowCertificate 3385 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3385 1) := by change ¬ Nat.Prime 45839671; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 580249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3385 2) := by change ¬ Nat.Prime 45832901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2696053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3385 3) := by change ¬ Nat.Prime 45826131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15275377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3385 : row 3385 = 0 := (row_3385.checked).trans rfl
theorem model_count_3385 : modelRow 3385 = 0 := by decide
theorem modelrow_agrees_3385 : modelRow 3385 = row 3385 := model_count_3385.trans row_count_3385.symm
def row_3386 : RowCertificate 3386 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3386 1) := by change ¬ Nat.Prime 45866757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15288919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3386 2) := by change ¬ Nat.Prime 45859985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9171997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3386 3) := by change ¬ Nat.Prime 45853213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6550459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3386 : row 3386 = 0 := (row_3386.checked).trans rfl
theorem model_count_3386 : modelRow 3386 = 0 := by decide
theorem modelrow_agrees_3386 : modelRow 3386 = row 3386 := model_count_3386.trans row_count_3386.symm
def row_3387 : RowCertificate 3387 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3387 1) := by change Nat.Prime 45893851; exact ProjectEulerSolutions.P58PrimeNodes.p_45893851.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3387 2) := by change ¬ Nat.Prime 45887077; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1582313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3387 3) := by change ¬ Nat.Prime 45880303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6554329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3387 : row 3387 = 1 := (row_3387.checked).trans rfl
theorem model_count_3387 : modelRow 3387 = 1 := by decide
theorem modelrow_agrees_3387 : modelRow 3387 = row 3387 := model_count_3387.trans row_count_3387.symm
def row_3388 : RowCertificate 3388 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3388 1) := by change ¬ Nat.Prime 45920953; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3532381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3388 2) := by change Nat.Prime 45914177; exact ProjectEulerSolutions.P58PrimeNodes.p_45914177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3388 3) := by change ¬ Nat.Prime 45907401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15302467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3388 : row 3388 = 1 := (row_3388.checked).trans rfl
theorem model_count_3388 : modelRow 3388 = 1 := by decide
theorem modelrow_agrees_3388 : modelRow 3388 = row 3388 := model_count_3388.trans row_count_3388.symm
def row_3389 : RowCertificate 3389 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3389 1) := by change ¬ Nat.Prime 45948063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15316021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3389 2) := by change ¬ Nat.Prime 45941285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9188257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3389 3) := by change ¬ Nat.Prime 45934507; exact PrattCertificates.composite_of_product (factor := 2113) (cofactor := 21739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3389 : row 3389 = 0 := (row_3389.checked).trans rfl
theorem model_count_3389 : modelRow 3389 = 0 := by decide
theorem modelrow_agrees_3389 : modelRow 3389 = row 3389 := model_count_3389.trans row_count_3389.symm
def row_3390 : RowCertificate 3390 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3390 1) := by change ¬ Nat.Prime 45975181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6567883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3390 2) := by change Nat.Prime 45968401; exact ProjectEulerSolutions.P58PrimeNodes.p_45968401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3390 3) := by change ¬ Nat.Prime 45961621; exact PrattCertificates.composite_of_product (factor := 1087) (cofactor := 42283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3390 : row 3390 = 1 := (row_3390.checked).trans rfl
theorem model_count_3390 : modelRow 3390 = 1 := by decide
theorem modelrow_agrees_3390 : modelRow 3390 = row 3390 := model_count_3390.trans row_count_3390.symm
def row_3391 : RowCertificate 3391 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3391 1) := by change ¬ Nat.Prime 46002307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3538639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3391 2) := by change ¬ Nat.Prime 45995525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9199105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3391 3) := by change ¬ Nat.Prime 45988743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15329581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3391 : row 3391 = 0 := (row_3391.checked).trans rfl
theorem model_count_3391 : modelRow 3391 = 0 := by decide
theorem modelrow_agrees_3391 : modelRow 3391 = row 3391 := model_count_3391.trans row_count_3391.symm
def row_3392 : RowCertificate 3392 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3392 1) := by change ¬ Nat.Prime 46029441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15343147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3392 2) := by change Nat.Prime 46022657; exact ProjectEulerSolutions.P58PrimeNodes.p_46022657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3392 3) := by change ¬ Nat.Prime 46015873; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1484383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3392 : row 3392 = 1 := (row_3392.checked).trans rfl
theorem model_count_3392 : modelRow 3392 = 1 := by decide
theorem modelrow_agrees_3392 : modelRow 3392 = row 3392 := model_count_3392.trans row_count_3392.symm
def row_3393 : RowCertificate 3393 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3393 1) := by change ¬ Nat.Prime 46056583; exact PrattCertificates.composite_of_product (factor := 2617) (cofactor := 17599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3393 2) := by change Nat.Prime 46049797; exact ProjectEulerSolutions.P58PrimeNodes.p_46049797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3393 3) := by change ¬ Nat.Prime 46043011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6577573) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3393 : row 3393 = 1 := (row_3393.checked).trans rfl
theorem model_count_3393 : modelRow 3393 = 1 := by decide
theorem modelrow_agrees_3393 : modelRow 3393 = row 3393 := model_count_3393.trans row_count_3393.symm
def row_3394 : RowCertificate 3394 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3394 1) := by change Nat.Prime 46083733; exact ProjectEulerSolutions.P58PrimeNodes.p_46083733.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3394 2) := by change ¬ Nat.Prime 46076945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9215389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3394 3) := by change ¬ Nat.Prime 46070157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15356719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3394 : row 3394 = 1 := (row_3394.checked).trans rfl
theorem model_count_3394 : modelRow 3394 = 1 := by decide
theorem modelrow_agrees_3394 : modelRow 3394 = row 3394 := model_count_3394.trans row_count_3394.symm
def row_3395 : RowCertificate 3395 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3395 1) := by change ¬ Nat.Prime 46110891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15370297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3395 2) := by change ¬ Nat.Prime 46104101; exact PrattCertificates.composite_of_product (factor := 809) (cofactor := 56989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3395 3) := by change ¬ Nat.Prime 46097311; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3545947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3395 : row 3395 = 0 := (row_3395.checked).trans rfl
theorem model_count_3395 : modelRow 3395 = 0 := by decide
theorem modelrow_agrees_3395 : modelRow 3395 = row 3395 := model_count_3395.trans row_count_3395.symm
def row_3396 : RowCertificate 3396 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3396 1) := by change ¬ Nat.Prime 46138057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6591151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3396 2) := by change ¬ Nat.Prime 46131265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9226253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3396 3) := by change Nat.Prime 46124473; exact ProjectEulerSolutions.P58PrimeNodes.p_46124473.prime
    exact decide_eq_true prime
theorem row_count_3396 : row 3396 = 1 := (row_3396.checked).trans rfl
theorem model_count_3396 : modelRow 3396 = 1 := by decide
theorem modelrow_agrees_3396 : modelRow 3396 = row 3396 := model_count_3396.trans row_count_3396.symm
def row_3397 : RowCertificate 3397 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3397 1) := by change ¬ Nat.Prime 46165231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6595033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3397 2) := by change ¬ Nat.Prime 46158437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3550649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3397 3) := by change ¬ Nat.Prime 46151643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15383881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3397 : row 3397 = 0 := (row_3397.checked).trans rfl
theorem model_count_3397 : modelRow 3397 = 0 := by decide
theorem modelrow_agrees_3397 : modelRow 3397 = row 3397 := model_count_3397.trans row_count_3397.symm
def row_3398 : RowCertificate 3398 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3398 1) := by change ¬ Nat.Prime 46192413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15397471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3398 2) := by change ¬ Nat.Prime 46185617; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2716801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3398 3) := by change ¬ Nat.Prime 46178821; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3552217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3398 : row 3398 = 0 := (row_3398.checked).trans rfl
theorem model_count_3398 : modelRow 3398 = 0 := by decide
theorem modelrow_agrees_3398 : modelRow 3398 = row 3398 := model_count_3398.trans row_count_3398.symm
def row_3399 : RowCertificate 3399 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3399 1) := by change Nat.Prime 46219603; exact ProjectEulerSolutions.P58PrimeNodes.p_46219603.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3399 2) := by change ¬ Nat.Prime 46212805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9242561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3399 3) := by change ¬ Nat.Prime 46206007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1248811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3399 : row 3399 = 1 := (row_3399.checked).trans rfl
theorem model_count_3399 : modelRow 3399 = 1 := by decide
theorem modelrow_agrees_3399 : modelRow 3399 = row 3399 := model_count_3399.trans row_count_3399.symm
def row_3400 : RowCertificate 3400 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3400 1) := by change ¬ Nat.Prime 46246801; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1075507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3400 2) := by change Nat.Prime 46240001; exact ProjectEulerSolutions.P58PrimeNodes.p_46240001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3400 3) := by change ¬ Nat.Prime 46233201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15411067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3400 : row 3400 = 1 := (row_3400.checked).trans rfl
theorem model_count_3400 : modelRow 3400 = 1 := by decide
theorem modelrow_agrees_3400 : modelRow 3400 = row 3400 := model_count_3400.trans row_count_3400.symm
def row_3401 : RowCertificate 3401 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3401 1) := by change ¬ Nat.Prime 46274007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15424669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3401 2) := by change ¬ Nat.Prime 46267205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9253441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3401 3) := by change ¬ Nat.Prime 46260403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6608629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3401 : row 3401 = 0 := (row_3401.checked).trans rfl
theorem model_count_3401 : modelRow 3401 = 0 := by decide
theorem modelrow_agrees_3401 : modelRow 3401 = row 3401 := model_count_3401.trans row_count_3401.symm
def row_3402 : RowCertificate 3402 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3402 1) := by change ¬ Nat.Prime 46301221; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 691063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3402 2) := by change ¬ Nat.Prime 46294417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3561109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3402 3) := by change ¬ Nat.Prime 46287613; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 424657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3402 : row 3402 = 0 := (row_3402.checked).trans rfl
theorem model_count_3402 : modelRow 3402 = 0 := by decide
theorem modelrow_agrees_3402 : modelRow 3402 = row 3402 := model_count_3402.trans row_count_3402.symm
def row_3403 : RowCertificate 3403 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3403 1) := by change ¬ Nat.Prime 46328443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6618349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3403 2) := by change Nat.Prime 46321637; exact ProjectEulerSolutions.P58PrimeNodes.p_46321637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3403 3) := by change ¬ Nat.Prime 46314831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15438277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3403 : row 3403 = 1 := (row_3403.checked).trans rfl
theorem model_count_3403 : modelRow 3403 = 1 := by decide
theorem modelrow_agrees_3403 : modelRow 3403 = row 3403 := model_count_3403.trans row_count_3403.symm
def row_3404 : RowCertificate 3404 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3404 1) := by change ¬ Nat.Prime 46355673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15451891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3404 2) := by change ¬ Nat.Prime 46348865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9269773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3404 3) := by change Nat.Prime 46342057; exact ProjectEulerSolutions.P58PrimeNodes.p_46342057.prime
    exact decide_eq_true prime
theorem row_count_3404 : row 3404 = 1 := (row_3404.checked).trans rfl
theorem model_count_3404 : modelRow 3404 = 1 := by decide
theorem modelrow_agrees_3404 : modelRow 3404 = row 3404 := model_count_3404.trans row_count_3404.symm
def row_3405 : RowCertificate 3405 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3405 1) := by change Nat.Prime 46382911; exact ProjectEulerSolutions.P58PrimeNodes.p_46382911.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3405 2) := by change ¬ Nat.Prime 46376101; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 311249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3405 3) := by change ¬ Nat.Prime 46369291; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2440489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3405 : row 3405 = 1 := (row_3405.checked).trans rfl
theorem model_count_3405 : modelRow 3405 = 1 := by decide
theorem modelrow_agrees_3405 : modelRow 3405 = row 3405 := model_count_3405.trans row_count_3405.symm
def row_3406 : RowCertificate 3406 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3406 1) := by change Nat.Prime 46410157; exact ProjectEulerSolutions.P58PrimeNodes.p_46410157.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3406 2) := by change ¬ Nat.Prime 46403345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9280669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3406 3) := by change ¬ Nat.Prime 46396533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15465511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3406 : row 3406 = 1 := (row_3406.checked).trans rfl
theorem model_count_3406 : modelRow 3406 = 1 := by decide
theorem modelrow_agrees_3406 : modelRow 3406 = row 3406 := model_count_3406.trans row_count_3406.symm
def row_3407 : RowCertificate 3407 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3407 1) := by change ¬ Nat.Prime 46437411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15479137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3407 2) := by change ¬ Nat.Prime 46430597; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1254881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3407 3) := by change ¬ Nat.Prime 46423783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6631969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3407 : row 3407 = 0 := (row_3407.checked).trans rfl
theorem model_count_3407 : modelRow 3407 = 0 := by decide
theorem modelrow_agrees_3407 : modelRow 3407 = row 3407 := model_count_3407.trans row_count_3407.symm
def row_3408 : RowCertificate 3408 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3408 1) := by change Nat.Prime 46464673; exact ProjectEulerSolutions.P58PrimeNodes.p_46464673.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3408 2) := by change ¬ Nat.Prime 46457857; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 636409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3408 3) := by change ¬ Nat.Prime 46451041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6635863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3408 : row 3408 = 1 := (row_3408.checked).trans rfl
theorem model_count_3408 : modelRow 3408 = 1 := by decide
theorem modelrow_agrees_3408 : modelRow 3408 = row 3408 := model_count_3408.trans row_count_3408.symm
def row_3409 : RowCertificate 3409 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3409 1) := by change ¬ Nat.Prime 46491943; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1256539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3409 2) := by change ¬ Nat.Prime 46485125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9297025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3409 3) := by change ¬ Nat.Prime 46478307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15492769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3409 : row 3409 = 0 := (row_3409.checked).trans rfl
theorem model_count_3409 : modelRow 3409 = 0 := by decide
theorem modelrow_agrees_3409 : modelRow 3409 = row 3409 := model_count_3409.trans row_count_3409.symm
def row_3410 : RowCertificate 3410 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3410 1) := by change ¬ Nat.Prime 46519221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15506407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3410 2) := by change ¬ Nat.Prime 46512401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3577877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3410 3) := by change ¬ Nat.Prime 46505581; exact PrattCertificates.composite_of_product (factor := 3037) (cofactor := 15313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3410 : row 3410 = 0 := (row_3410.checked).trans rfl
theorem model_count_3410 : modelRow 3410 = 0 := by decide
theorem modelrow_agrees_3410 : modelRow 3410 = row 3410 := model_count_3410.trans row_count_3410.symm
def row_3411 : RowCertificate 3411 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3411 1) := by change ¬ Nat.Prime 46546507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6649501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3411 2) := by change ¬ Nat.Prime 46539685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9307937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3411 3) := by change ¬ Nat.Prime 46532863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3579451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3411 : row 3411 = 0 := (row_3411.checked).trans rfl
theorem model_count_3411 : modelRow 3411 = 0 := by decide
theorem modelrow_agrees_3411 : modelRow 3411 = row 3411 := model_count_3411.trans row_count_3411.symm
def row_3412 : RowCertificate 3412 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3412 1) := by change ¬ Nat.Prime 46573801; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 64063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3412 2) := by change Nat.Prime 46566977; exact ProjectEulerSolutions.P58PrimeNodes.p_46566977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3412 3) := by change ¬ Nat.Prime 46560153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15520051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3412 : row 3412 = 1 := (row_3412.checked).trans rfl
theorem model_count_3412 : modelRow 3412 = 1 := by decide
theorem modelrow_agrees_3412 : modelRow 3412 = row 3412 := model_count_3412.trans row_count_3412.symm
def row_3413 : RowCertificate 3413 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3413 1) := by change ¬ Nat.Prime 46601103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15533701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3413 2) := by change Nat.Prime 46594277; exact ProjectEulerSolutions.P58PrimeNodes.p_46594277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3413 3) := by change ¬ Nat.Prime 46587451; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1502821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3413 : row 3413 = 1 := (row_3413.checked).trans rfl
theorem model_count_3413 : modelRow 3413 = 1 := by decide
theorem modelrow_agrees_3413 : modelRow 3413 = row 3413 := model_count_3413.trans row_count_3413.symm
def row_3414 : RowCertificate 3414 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3414 1) := by change ¬ Nat.Prime 46628413; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3586801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3414 2) := by change ¬ Nat.Prime 46621585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9324317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3414 3) := by change ¬ Nat.Prime 46614757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6659251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3414 : row 3414 = 0 := (row_3414.checked).trans rfl
theorem model_count_3414 : modelRow 3414 = 0 := by decide
theorem modelrow_agrees_3414 : modelRow 3414 = row 3414 := model_count_3414.trans row_count_3414.symm
def row_3415 : RowCertificate 3415 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3415 1) := by change ¬ Nat.Prime 46655731; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1085017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3415 2) := by change ¬ Nat.Prime 46648901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3588377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3415 3) := by change ¬ Nat.Prime 46642071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15547357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3415 : row 3415 = 0 := (row_3415.checked).trans rfl
theorem model_count_3415 : modelRow 3415 = 0 := by decide
theorem modelrow_agrees_3415 : modelRow 3415 = row 3415 := model_count_3415.trans row_count_3415.symm
def row_3416 : RowCertificate 3416 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3416 1) := by change ¬ Nat.Prime 46683057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15561019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3416 2) := by change ¬ Nat.Prime 46676225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9335245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3416 3) := by change ¬ Nat.Prime 46669393; exact PrattCertificates.composite_of_product (factor := 751) (cofactor := 62143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3416 : row 3416 = 0 := (row_3416.checked).trans rfl
theorem model_count_3416 : modelRow 3416 = 0 := by decide
theorem modelrow_agrees_3416 : modelRow 3416 = row 3416 := model_count_3416.trans row_count_3416.symm
def row_3417 : RowCertificate 3417 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3417 1) := by change ¬ Nat.Prime 46710391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6672913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3417 2) := by change ¬ Nat.Prime 46703557; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 428473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3417 3) := by change Nat.Prime 46696723; exact ProjectEulerSolutions.P58PrimeNodes.p_46696723.prime
    exact decide_eq_true prime
theorem row_count_3417 : row 3417 = 1 := (row_3417.checked).trans rfl
theorem model_count_3417 : modelRow 3417 = 1 := by decide
theorem modelrow_agrees_3417 : modelRow 3417 = row 3417 := model_count_3417.trans row_count_3417.symm
def row_3418 : RowCertificate 3418 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3418 1) := by change ¬ Nat.Prime 46737733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6676819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3418 2) := by change ¬ Nat.Prime 46730897; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 242129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3418 3) := by change ¬ Nat.Prime 46724061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15574687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3418 : row 3418 = 0 := (row_3418.checked).trans rfl
theorem model_count_3418 : modelRow 3418 = 0 := by decide
theorem modelrow_agrees_3418 : modelRow 3418 = row 3418 := model_count_3418.trans row_count_3418.symm
def row_3419 : RowCertificate 3419 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3419 1) := by change ¬ Nat.Prime 46765083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15588361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3419 2) := by change ¬ Nat.Prime 46758245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9351649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3419 3) := by change Nat.Prime 46751407; exact ProjectEulerSolutions.P58PrimeNodes.p_46751407.prime
    exact decide_eq_true prime
theorem row_count_3419 : row 3419 = 1 := (row_3419.checked).trans rfl
theorem model_count_3419 : modelRow 3419 = 1 := by decide
theorem modelrow_agrees_3419 : modelRow 3419 = row 3419 := model_count_3419.trans row_count_3419.symm
def row_3420 : RowCertificate 3420 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3420 1) := by change ¬ Nat.Prime 46792441; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 61813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3420 2) := by change ¬ Nat.Prime 46785601; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 270437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3420 3) := by change ¬ Nat.Prime 46778761; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 242377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3420 : row 3420 = 0 := (row_3420.checked).trans rfl
theorem model_count_3420 : modelRow 3420 = 0 := by decide
theorem modelrow_agrees_3420 : modelRow 3420 = row 3420 := model_count_3420.trans row_count_3420.symm
def row_3421 : RowCertificate 3421 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3421 1) := by change Nat.Prime 46819807; exact ProjectEulerSolutions.P58PrimeNodes.p_46819807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3421 2) := by change ¬ Nat.Prime 46812965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9362593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3421 3) := by change ¬ Nat.Prime 46806123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15602041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3421 : row 3421 = 1 := (row_3421.checked).trans rfl
theorem model_count_3421 : modelRow 3421 = 1 := by decide
theorem modelrow_agrees_3421 : modelRow 3421 = row 3421 := model_count_3421.trans row_count_3421.symm
def row_3422 : RowCertificate 3422 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3422 1) := by change ¬ Nat.Prime 46847181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15615727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3422 2) := by change ¬ Nat.Prime 46840337; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 134213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3422 3) := by change ¬ Nat.Prime 46833493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6690499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3422 : row 3422 = 0 := (row_3422.checked).trans rfl
theorem model_count_3422 : modelRow 3422 = 0 := by decide
theorem modelrow_agrees_3422 : modelRow 3422 = row 3422 := model_count_3422.trans row_count_3422.symm
def row_3423 : RowCertificate 3423 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3423 1) := by change ¬ Nat.Prime 46874563; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 93937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3423 2) := by change ¬ Nat.Prime 46867717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3605209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3423 3) := by change ¬ Nat.Prime 46860871; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1511641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3423 : row 3423 = 0 := (row_3423.checked).trans rfl
theorem model_count_3423 : modelRow 3423 = 0 := by decide
theorem modelrow_agrees_3423 : modelRow 3423 = row 3423 := model_count_3423.trans row_count_3423.symm
def row_3424 : RowCertificate 3424 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3424 1) := by change ¬ Nat.Prime 46901953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6700279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3424 2) := by change ¬ Nat.Prime 46895105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9379021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3424 3) := by change ¬ Nat.Prime 46888257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15629419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3424 : row 3424 = 0 := (row_3424.checked).trans rfl
theorem model_count_3424 : modelRow 3424 = 0 := by decide
theorem modelrow_agrees_3424 : modelRow 3424 = row 3424 := model_count_3424.trans row_count_3424.symm
def row_3425 : RowCertificate 3425 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3425 1) := by change ¬ Nat.Prime 46929351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15643117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3425 2) := by change Nat.Prime 46922501; exact ProjectEulerSolutions.P58PrimeNodes.p_46922501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3425 3) := by change ¬ Nat.Prime 46915651; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 593869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3425 : row 3425 = 1 := (row_3425.checked).trans rfl
theorem model_count_3425 : modelRow 3425 = 1 := by decide
theorem modelrow_agrees_3425 : modelRow 3425 = row 3425 := model_count_3425.trans row_count_3425.symm
def row_3426 : RowCertificate 3426 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3426 1) := by change ¬ Nat.Prime 46956757; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 106963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3426 2) := by change ¬ Nat.Prime 46949905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9389981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3426 3) := by change ¬ Nat.Prime 46943053; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2470687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3426 : row 3426 = 0 := (row_3426.checked).trans rfl
theorem model_count_3426 : modelRow 3426 = 0 := by decide
theorem modelrow_agrees_3426 : modelRow 3426 = row 3426 := model_count_3426.trans row_count_3426.symm
def row_3427 : RowCertificate 3427 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3427 1) := by change ¬ Nat.Prime 46984171; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3614167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3427 2) := by change Nat.Prime 46977317; exact ProjectEulerSolutions.P58PrimeNodes.p_46977317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3427 3) := by change ¬ Nat.Prime 46970463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15656821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3427 : row 3427 = 1 := (row_3427.checked).trans rfl
theorem model_count_3427 : modelRow 3427 = 1 := by decide
theorem modelrow_agrees_3427 : modelRow 3427 = row 3427 := model_count_3427.trans row_count_3427.symm
def row_3428 : RowCertificate 3428 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3428 1) := by change ¬ Nat.Prime 47011593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15670531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3428 2) := by change ¬ Nat.Prime 47004737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3615749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3428 3) := by change ¬ Nat.Prime 46997881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6713983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3428 : row 3428 = 0 := (row_3428.checked).trans rfl
theorem model_count_3428 : modelRow 3428 = 0 := by decide
theorem modelrow_agrees_3428 : modelRow 3428 = row 3428 := model_count_3428.trans row_count_3428.symm
def row_3429 : RowCertificate 3429 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3429 1) := by change ¬ Nat.Prime 47039023; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 236377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3429 2) := by change ¬ Nat.Prime 47032165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9406433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3429 3) := by change ¬ Nat.Prime 47025307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6717901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3429 : row 3429 = 0 := (row_3429.checked).trans rfl
theorem model_count_3429 : modelRow 3429 = 0 := by decide
theorem modelrow_agrees_3429 : modelRow 3429 = row 3429 := model_count_3429.trans row_count_3429.symm
def row_3430 : RowCertificate 3430 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3430 1) := by change ¬ Nat.Prime 47066461; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3620497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3430 2) := by change ¬ Nat.Prime 47059601; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 887917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3430 3) := by change ¬ Nat.Prime 47052741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15684247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3430 : row 3430 = 0 := (row_3430.checked).trans rfl
theorem model_count_3430 : modelRow 3430 = 0 := by decide
theorem modelrow_agrees_3430 : modelRow 3430 = row 3430 := model_count_3430.trans row_count_3430.symm
def row_3431 : RowCertificate 3431 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3431 1) := by change ¬ Nat.Prime 47093907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15697969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3431 2) := by change ¬ Nat.Prime 47087045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9417409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3431 3) := by change ¬ Nat.Prime 47080183; exact PrattCertificates.composite_of_product (factor := 2131) (cofactor := 22093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3431 : row 3431 = 0 := (row_3431.checked).trans rfl
theorem model_count_3431 : modelRow 3431 = 0 := by decide
theorem modelrow_agrees_3431 : modelRow 3431 = row 3431 := model_count_3431.trans row_count_3431.symm
def row_3432 : RowCertificate 3432 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3432 1) := by change ¬ Nat.Prime 47121361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6731623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3432 2) := by change ¬ Nat.Prime 47114497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2771441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3432 3) := by change ¬ Nat.Prime 47107633; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 703099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3432 : row 3432 = 0 := (row_3432.checked).trans rfl
theorem model_count_3432 : modelRow 3432 = 0 := by decide
theorem modelrow_agrees_3432 : modelRow 3432 = row 3432 := model_count_3432.trans row_count_3432.symm
def row_3433 : RowCertificate 3433 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3433 1) := by change ¬ Nat.Prime 47148823; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2481517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3433 2) := by change Nat.Prime 47141957; exact ProjectEulerSolutions.P58PrimeNodes.p_47141957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3433 3) := by change ¬ Nat.Prime 47135091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15711697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3433 : row 3433 = 1 := (row_3433.checked).trans rfl
theorem model_count_3433 : modelRow 3433 = 1 := by decide
theorem modelrow_agrees_3433 : modelRow 3433 = row 3433 := model_count_3433.trans row_count_3433.symm
def row_3434 : RowCertificate 3434 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3434 1) := by change ¬ Nat.Prime 47176293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15725431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3434 2) := by change ¬ Nat.Prime 47169425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9433885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3434 3) := by change ¬ Nat.Prime 47162557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3627889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3434 : row 3434 = 0 := (row_3434.checked).trans rfl
theorem model_count_3434 : modelRow 3434 = 0 := by decide
theorem modelrow_agrees_3434 : modelRow 3434 = row 3434 := model_count_3434.trans row_count_3434.symm
def row_3435 : RowCertificate 3435 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3435 1) := by change ¬ Nat.Prime 47203771; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2484409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3435 2) := by change ¬ Nat.Prime 47196901; exact PrattCertificates.composite_of_product (factor := 2341) (cofactor := 20161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3435 3) := by change ¬ Nat.Prime 47190031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6741433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3435 : row 3435 = 0 := (row_3435.checked).trans rfl
theorem model_count_3435 : modelRow 3435 = 0 := by decide
theorem modelrow_agrees_3435 : modelRow 3435 = row 3435 := model_count_3435.trans row_count_3435.symm
def row_3436 : RowCertificate 3436 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3436 1) := by change ¬ Nat.Prime 47231257; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 237343) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3436 2) := by change ¬ Nat.Prime 47224385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9444877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3436 3) := by change ¬ Nat.Prime 47217513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15739171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3436 : row 3436 = 0 := (row_3436.checked).trans rfl
theorem model_count_3436 : modelRow 3436 = 0 := by decide
theorem modelrow_agrees_3436 : modelRow 3436 = row 3436 := model_count_3436.trans row_count_3436.symm
def row_3437 : RowCertificate 3437 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3437 1) := by change ¬ Nat.Prime 47258751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15752917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3437 2) := by change Nat.Prime 47251877; exact ProjectEulerSolutions.P58PrimeNodes.p_47251877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3437 3) := by change ¬ Nat.Prime 47245003; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3634231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3437 : row 3437 = 1 := (row_3437.checked).trans rfl
theorem model_count_3437 : modelRow 3437 = 1 := by decide
theorem modelrow_agrees_3437 : modelRow 3437 = row 3437 := model_count_3437.trans row_count_3437.symm
def row_3438 : RowCertificate 3438 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3438 1) := by change ¬ Nat.Prime 47286253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6755179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3438 2) := by change ¬ Nat.Prime 47279377; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1277821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3438 3) := by change Nat.Prime 47272501; exact ProjectEulerSolutions.P58PrimeNodes.p_47272501.prime
    exact decide_eq_true prime
theorem row_count_3438 : row 3438 = 1 := (row_3438.checked).trans rfl
theorem model_count_3438 : modelRow 3438 = 1 := by decide
theorem modelrow_agrees_3438 : modelRow 3438 = row 3438 := model_count_3438.trans row_count_3438.symm
def row_3439 : RowCertificate 3439 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3439 1) := by change ¬ Nat.Prime 47313763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6759109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3439 2) := by change ¬ Nat.Prime 47306885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9461377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3439 3) := by change ¬ Nat.Prime 47300007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15766669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3439 : row 3439 = 0 := (row_3439.checked).trans rfl
theorem model_count_3439 : modelRow 3439 = 0 := by decide
theorem modelrow_agrees_3439 : modelRow 3439 = row 3439 := model_count_3439.trans row_count_3439.symm
def row_3440 : RowCertificate 3440 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3440 1) := by change ¬ Nat.Prime 47341281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15780427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3440 2) := by change ¬ Nat.Prime 47334401; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 301493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3440 3) := by change ¬ Nat.Prime 47327521; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 775861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3440 : row 3440 = 0 := (row_3440.checked).trans rfl
theorem model_count_3440 : modelRow 3440 = 0 := by decide
theorem modelrow_agrees_3440 : modelRow 3440 = row 3440 := model_count_3440.trans row_count_3440.symm
def row_3441 : RowCertificate 3441 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3441 1) := by change Nat.Prime 47368807; exact ProjectEulerSolutions.P58PrimeNodes.p_47368807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3441 2) := by change ¬ Nat.Prime 47361925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9472385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3441 3) := by change ¬ Nat.Prime 47355043; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 50539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3441 : row 3441 = 1 := (row_3441.checked).trans rfl
theorem model_count_3441 : modelRow 3441 = 1 := by decide
theorem modelrow_agrees_3441 : modelRow 3441 = row 3441 := model_count_3441.trans row_count_3441.symm
def row_3442 : RowCertificate 3442 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3442 1) := by change Nat.Prime 47396341; exact ProjectEulerSolutions.P58PrimeNodes.p_47396341.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3442 2) := by change Nat.Prime 47389457; exact ProjectEulerSolutions.P58PrimeNodes.p_47389457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3442 3) := by change ¬ Nat.Prime 47382573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15794191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3442 : row 3442 = 2 := (row_3442.checked).trans rfl
theorem model_count_3442 : modelRow 3442 = 2 := by decide
theorem modelrow_agrees_3442 : modelRow 3442 = row 3442 := model_count_3442.trans row_count_3442.symm
def row_3443 : RowCertificate 3443 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3443 1) := by change ¬ Nat.Prime 47423883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15807961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3443 2) := by change ¬ Nat.Prime 47416997; exact PrattCertificates.composite_of_product (factor := 5693) (cofactor := 8329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3443 3) := by change ¬ Nat.Prime 47410111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6772873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3443 : row 3443 = 0 := (row_3443.checked).trans rfl
theorem model_count_3443 : modelRow 3443 = 0 := by decide
theorem modelrow_agrees_3443 : modelRow 3443 = row 3443 := model_count_3443.trans row_count_3443.symm
def row_3444 : RowCertificate 3444 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3444 1) := by change Nat.Prime 47451433; exact ProjectEulerSolutions.P58PrimeNodes.p_47451433.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3444 2) := by change ¬ Nat.Prime 47444545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9488909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3444 3) := by change ¬ Nat.Prime 47437657; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1530247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3444 : row 3444 = 1 := (row_3444.checked).trans rfl
theorem model_count_3444 : modelRow 3444 = 1 := by decide
theorem modelrow_agrees_3444 : modelRow 3444 = row 3444 := model_count_3444.trans row_count_3444.symm
def row_3445 : RowCertificate 3445 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3445 1) := by change ¬ Nat.Prime 47478991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6782713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3445 2) := by change ¬ Nat.Prime 47472101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1636969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3445 3) := by change ¬ Nat.Prime 47465211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15821737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3445 : row 3445 = 0 := (row_3445.checked).trans rfl
theorem model_count_3445 : modelRow 3445 = 0 := by decide
theorem modelrow_agrees_3445 : modelRow 3445 = row 3445 := model_count_3445.trans row_count_3445.symm
def row_3446 : RowCertificate 3446 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3446 1) := by change ¬ Nat.Prime 47506557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15835519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3446 2) := by change ¬ Nat.Prime 47499665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9499933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3446 3) := by change ¬ Nat.Prime 47492773; exact PrattCertificates.composite_of_product (factor := 2719) (cofactor := 17467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3446 : row 3446 = 0 := (row_3446.checked).trans rfl
theorem model_count_3446 : modelRow 3446 = 0 := by decide
theorem modelrow_agrees_3446 : modelRow 3446 = row 3446 := model_count_3446.trans row_count_3446.symm
def row_3447 : RowCertificate 3447 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3447 1) := by change ¬ Nat.Prime 47534131; exact PrattCertificates.composite_of_product (factor := 2203) (cofactor := 21577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3447 2) := by change ¬ Nat.Prime 47527237; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 162209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3447 3) := by change ¬ Nat.Prime 47520343; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3655411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3447 : row 3447 = 0 := (row_3447.checked).trans rfl
theorem model_count_3447 : modelRow 3447 = 0 := by decide
theorem modelrow_agrees_3447 : modelRow 3447 = row 3447 := model_count_3447.trans row_count_3447.symm
def row_3448 : RowCertificate 3448 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3448 1) := by change ¬ Nat.Prime 47561713; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 602047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3448 2) := by change ¬ Nat.Prime 47554817; exact PrattCertificates.composite_of_product (factor := 1669) (cofactor := 28493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3448 3) := by change ¬ Nat.Prime 47547921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15849307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3448 : row 3448 = 0 := (row_3448.checked).trans rfl
theorem model_count_3448 : modelRow 3448 = 0 := by decide
theorem modelrow_agrees_3448 : modelRow 3448 = row 3448 := model_count_3448.trans row_count_3448.symm
def row_3449 : RowCertificate 3449 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3449 1) := by change ¬ Nat.Prime 47589303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15863101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3449 2) := by change ¬ Nat.Prime 47582405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9516481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3449 3) := by change ¬ Nat.Prime 47575507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6796501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3449 : row 3449 = 0 := (row_3449.checked).trans rfl
theorem model_count_3449 : modelRow 3449 = 0 := by decide
theorem modelrow_agrees_3449 : modelRow 3449 = row 3449 := model_count_3449.trans row_count_3449.symm
def row_3450 : RowCertificate 3450 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3450 1) := by change ¬ Nat.Prime 47616901; exact PrattCertificates.composite_of_product (factor := 6343) (cofactor := 7507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3450 2) := by change ¬ Nat.Prime 47610001; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 436789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3450 3) := by change ¬ Nat.Prime 47603101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6800443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3450 : row 3450 = 0 := (row_3450.checked).trans rfl
theorem model_count_3450 : modelRow 3450 = 0 := by decide
theorem modelrow_agrees_3450 : modelRow 3450 = row 3450 := model_count_3450.trans row_count_3450.symm
def row_3451 : RowCertificate 3451 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3451 1) := by change ¬ Nat.Prime 47644507; exact PrattCertificates.composite_of_product (factor := 1321) (cofactor := 36067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3451 2) := by change ¬ Nat.Prime 47637605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9527521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3451 3) := by change ¬ Nat.Prime 47630703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15876901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3451 : row 3451 = 0 := (row_3451.checked).trans rfl
theorem model_count_3451 : modelRow 3451 = 0 := by decide
theorem modelrow_agrees_3451 : modelRow 3451 = row 3451 := model_count_3451.trans row_count_3451.symm
def row_3452 : RowCertificate 3452 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3452 1) := by change ¬ Nat.Prime 47672121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15890707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3452 2) := by change ¬ Nat.Prime 47665217; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 781397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3452 3) := by change ¬ Nat.Prime 47658313; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 88093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3452 : row 3452 = 0 := (row_3452.checked).trans rfl
theorem model_count_3452 : modelRow 3452 = 0 := by decide
theorem modelrow_agrees_3452 : modelRow 3452 = row 3452 := model_count_3452.trans row_count_3452.symm
def row_3453 : RowCertificate 3453 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3453 1) := by change ¬ Nat.Prime 47699743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6814249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3453 2) := by change ¬ Nat.Prime 47692837; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2805461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3453 3) := by change Nat.Prime 47685931; exact ProjectEulerSolutions.P58PrimeNodes.p_47685931.prime
    exact decide_eq_true prime
theorem row_count_3453 : row 3453 = 1 := (row_3453.checked).trans rfl
theorem model_count_3453 : modelRow 3453 = 1 := by decide
theorem modelrow_agrees_3453 : modelRow 3453 = row 3453 := model_count_3453.trans row_count_3453.symm
def row_3454 : RowCertificate 3454 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3454 1) := by change ¬ Nat.Prime 47727373; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2511967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3454 2) := by change ¬ Nat.Prime 47720465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9544093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3454 3) := by change ¬ Nat.Prime 47713557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15904519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3454 : row 3454 = 0 := (row_3454.checked).trans rfl
theorem model_count_3454 : modelRow 3454 = 0 := by decide
theorem modelrow_agrees_3454 : modelRow 3454 = row 3454 := model_count_3454.trans row_count_3454.symm
def row_3455 : RowCertificate 3455 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3455 1) := by change ¬ Nat.Prime 47755011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 15918337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3455 2) := by change Nat.Prime 47748101; exact ProjectEulerSolutions.P58PrimeNodes.p_47748101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3455 3) := by change Nat.Prime 47741191; exact ProjectEulerSolutions.P58PrimeNodes.p_47741191.prime
    exact decide_eq_true prime
theorem row_count_3455 : row 3455 = 2 := (row_3455.checked).trans rfl
theorem model_count_3455 : modelRow 3455 = 2 := by decide
theorem modelrow_agrees_3455 : modelRow 3455 = row 3455 := model_count_3455.trans row_count_3455.symm
def row_3456 : RowCertificate 3456 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3456 1) := by change ¬ Nat.Prime 47782657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3675589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3456 2) := by change ¬ Nat.Prime 47775745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 9555149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3456 3) := by change ¬ Nat.Prime 47768833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6824119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3456 : row 3456 = 0 := (row_3456.checked).trans rfl
theorem model_count_3456 : modelRow 3456 = 0 := by decide
theorem modelrow_agrees_3456 : modelRow 3456 = row 3456 := model_count_3456.trans row_count_3456.symm
private def atomic_3329 : CountSegment 1 10 13120 3328 3329 1595 1595 := CountSegment.single (first := 3328) (initial := 1595) row_count_3329 (by decide)
private theorem atomic_3329_agreement : RowAgreement 3328 3329 := RowAgreement.single (first := 3328) modelrow_agrees_3329
private def atomic_3330 : CountSegment 1 10 13120 3329 3330 1595 1595 := CountSegment.single (first := 3329) (initial := 1595) row_count_3330 (by decide)
private theorem atomic_3330_agreement : RowAgreement 3329 3330 := RowAgreement.single (first := 3329) modelrow_agrees_3330
private def atomic_3331 : CountSegment 1 10 13120 3330 3331 1595 1595 := CountSegment.single (first := 3330) (initial := 1595) row_count_3331 (by decide)
private theorem atomic_3331_agreement : RowAgreement 3330 3331 := RowAgreement.single (first := 3330) modelrow_agrees_3331
private def atomic_3332 : CountSegment 1 10 13120 3331 3332 1595 1595 := CountSegment.single (first := 3331) (initial := 1595) row_count_3332 (by decide)
private theorem atomic_3332_agreement : RowAgreement 3331 3332 := RowAgreement.single (first := 3331) modelrow_agrees_3332
private def atomic_3333 : CountSegment 1 10 13120 3332 3333 1595 1595 := CountSegment.single (first := 3332) (initial := 1595) row_count_3333 (by decide)
private theorem atomic_3333_agreement : RowAgreement 3332 3333 := RowAgreement.single (first := 3332) modelrow_agrees_3333
private def atomic_3334 : CountSegment 1 10 13120 3333 3334 1595 1595 := CountSegment.single (first := 3333) (initial := 1595) row_count_3334 (by decide)
private theorem atomic_3334_agreement : RowAgreement 3333 3334 := RowAgreement.single (first := 3333) modelrow_agrees_3334
private def atomic_3335 : CountSegment 1 10 13120 3334 3335 1595 1596 := CountSegment.single (first := 3334) (initial := 1595) row_count_3335 (by decide)
private theorem atomic_3335_agreement : RowAgreement 3334 3335 := RowAgreement.single (first := 3334) modelrow_agrees_3335
private def atomic_3336 : CountSegment 1 10 13120 3335 3336 1596 1596 := CountSegment.single (first := 3335) (initial := 1596) row_count_3336 (by decide)
private theorem atomic_3336_agreement : RowAgreement 3335 3336 := RowAgreement.single (first := 3335) modelrow_agrees_3336
private def atomic_3337 : CountSegment 1 10 13120 3336 3337 1596 1596 := CountSegment.single (first := 3336) (initial := 1596) row_count_3337 (by decide)
private theorem atomic_3337_agreement : RowAgreement 3336 3337 := RowAgreement.single (first := 3336) modelrow_agrees_3337
private def atomic_3338 : CountSegment 1 10 13120 3337 3338 1596 1596 := CountSegment.single (first := 3337) (initial := 1596) row_count_3338 (by decide)
private theorem atomic_3338_agreement : RowAgreement 3337 3338 := RowAgreement.single (first := 3337) modelrow_agrees_3338
private def atomic_3339 : CountSegment 1 10 13120 3338 3339 1596 1596 := CountSegment.single (first := 3338) (initial := 1596) row_count_3339 (by decide)
private theorem atomic_3339_agreement : RowAgreement 3338 3339 := RowAgreement.single (first := 3338) modelrow_agrees_3339
private def atomic_3340 : CountSegment 1 10 13120 3339 3340 1596 1596 := CountSegment.single (first := 3339) (initial := 1596) row_count_3340 (by decide)
private theorem atomic_3340_agreement : RowAgreement 3339 3340 := RowAgreement.single (first := 3339) modelrow_agrees_3340
private def atomic_3341 : CountSegment 1 10 13120 3340 3341 1596 1597 := CountSegment.single (first := 3340) (initial := 1596) row_count_3341 (by decide)
private theorem atomic_3341_agreement : RowAgreement 3340 3341 := RowAgreement.single (first := 3340) modelrow_agrees_3341
private def atomic_3342 : CountSegment 1 10 13120 3341 3342 1597 1597 := CountSegment.single (first := 3341) (initial := 1597) row_count_3342 (by decide)
private theorem atomic_3342_agreement : RowAgreement 3341 3342 := RowAgreement.single (first := 3341) modelrow_agrees_3342
private def atomic_3343 : CountSegment 1 10 13120 3342 3343 1597 1597 := CountSegment.single (first := 3342) (initial := 1597) row_count_3343 (by decide)
private theorem atomic_3343_agreement : RowAgreement 3342 3343 := RowAgreement.single (first := 3342) modelrow_agrees_3343
private def atomic_3344 : CountSegment 1 10 13120 3343 3344 1597 1597 := CountSegment.single (first := 3343) (initial := 1597) row_count_3344 (by decide)
private theorem atomic_3344_agreement : RowAgreement 3343 3344 := RowAgreement.single (first := 3343) modelrow_agrees_3344
private def atomic_3345 : CountSegment 1 10 13120 3344 3345 1597 1597 := CountSegment.single (first := 3344) (initial := 1597) row_count_3345 (by decide)
private theorem atomic_3345_agreement : RowAgreement 3344 3345 := RowAgreement.single (first := 3344) modelrow_agrees_3345
private def atomic_3346 : CountSegment 1 10 13120 3345 3346 1597 1598 := CountSegment.single (first := 3345) (initial := 1597) row_count_3346 (by decide)
private theorem atomic_3346_agreement : RowAgreement 3345 3346 := RowAgreement.single (first := 3345) modelrow_agrees_3346
private def atomic_3347 : CountSegment 1 10 13120 3346 3347 1598 1598 := CountSegment.single (first := 3346) (initial := 1598) row_count_3347 (by decide)
private theorem atomic_3347_agreement : RowAgreement 3346 3347 := RowAgreement.single (first := 3346) modelrow_agrees_3347
private def atomic_3348 : CountSegment 1 10 13120 3347 3348 1598 1598 := CountSegment.single (first := 3347) (initial := 1598) row_count_3348 (by decide)
private theorem atomic_3348_agreement : RowAgreement 3347 3348 := RowAgreement.single (first := 3347) modelrow_agrees_3348
private def atomic_3349 : CountSegment 1 10 13120 3348 3349 1598 1598 := CountSegment.single (first := 3348) (initial := 1598) row_count_3349 (by decide)
private theorem atomic_3349_agreement : RowAgreement 3348 3349 := RowAgreement.single (first := 3348) modelrow_agrees_3349
private def atomic_3350 : CountSegment 1 10 13120 3349 3350 1598 1598 := CountSegment.single (first := 3349) (initial := 1598) row_count_3350 (by decide)
private theorem atomic_3350_agreement : RowAgreement 3349 3350 := RowAgreement.single (first := 3349) modelrow_agrees_3350
private def atomic_3351 : CountSegment 1 10 13120 3350 3351 1598 1599 := CountSegment.single (first := 3350) (initial := 1598) row_count_3351 (by decide)
private theorem atomic_3351_agreement : RowAgreement 3350 3351 := RowAgreement.single (first := 3350) modelrow_agrees_3351
private def atomic_3352 : CountSegment 1 10 13120 3351 3352 1599 1600 := CountSegment.single (first := 3351) (initial := 1599) row_count_3352 (by decide)
private theorem atomic_3352_agreement : RowAgreement 3351 3352 := RowAgreement.single (first := 3351) modelrow_agrees_3352
private def atomic_3353 : CountSegment 1 10 13120 3352 3353 1600 1600 := CountSegment.single (first := 3352) (initial := 1600) row_count_3353 (by decide)
private theorem atomic_3353_agreement : RowAgreement 3352 3353 := RowAgreement.single (first := 3352) modelrow_agrees_3353
private def atomic_3354 : CountSegment 1 10 13120 3353 3354 1600 1600 := CountSegment.single (first := 3353) (initial := 1600) row_count_3354 (by decide)
private theorem atomic_3354_agreement : RowAgreement 3353 3354 := RowAgreement.single (first := 3353) modelrow_agrees_3354
private def atomic_3355 : CountSegment 1 10 13120 3354 3355 1600 1601 := CountSegment.single (first := 3354) (initial := 1600) row_count_3355 (by decide)
private theorem atomic_3355_agreement : RowAgreement 3354 3355 := RowAgreement.single (first := 3354) modelrow_agrees_3355
private def atomic_3356 : CountSegment 1 10 13120 3355 3356 1601 1601 := CountSegment.single (first := 3355) (initial := 1601) row_count_3356 (by decide)
private theorem atomic_3356_agreement : RowAgreement 3355 3356 := RowAgreement.single (first := 3355) modelrow_agrees_3356
private def atomic_3357 : CountSegment 1 10 13120 3356 3357 1601 1602 := CountSegment.single (first := 3356) (initial := 1601) row_count_3357 (by decide)
private theorem atomic_3357_agreement : RowAgreement 3356 3357 := RowAgreement.single (first := 3356) modelrow_agrees_3357
private def atomic_3358 : CountSegment 1 10 13120 3357 3358 1602 1602 := CountSegment.single (first := 3357) (initial := 1602) row_count_3358 (by decide)
private theorem atomic_3358_agreement : RowAgreement 3357 3358 := RowAgreement.single (first := 3357) modelrow_agrees_3358
private def atomic_3359 : CountSegment 1 10 13120 3358 3359 1602 1602 := CountSegment.single (first := 3358) (initial := 1602) row_count_3359 (by decide)
private theorem atomic_3359_agreement : RowAgreement 3358 3359 := RowAgreement.single (first := 3358) modelrow_agrees_3359
private def atomic_3360 : CountSegment 1 10 13120 3359 3360 1602 1604 := CountSegment.single (first := 3359) (initial := 1602) row_count_3360 (by decide)
private theorem atomic_3360_agreement : RowAgreement 3359 3360 := RowAgreement.single (first := 3359) modelrow_agrees_3360
private def atomic_3361 : CountSegment 1 10 13120 3360 3361 1604 1604 := CountSegment.single (first := 3360) (initial := 1604) row_count_3361 (by decide)
private theorem atomic_3361_agreement : RowAgreement 3360 3361 := RowAgreement.single (first := 3360) modelrow_agrees_3361
private def atomic_3362 : CountSegment 1 10 13120 3361 3362 1604 1605 := CountSegment.single (first := 3361) (initial := 1604) row_count_3362 (by decide)
private theorem atomic_3362_agreement : RowAgreement 3361 3362 := RowAgreement.single (first := 3361) modelrow_agrees_3362
private def atomic_3363 : CountSegment 1 10 13120 3362 3363 1605 1606 := CountSegment.single (first := 3362) (initial := 1605) row_count_3363 (by decide)
private theorem atomic_3363_agreement : RowAgreement 3362 3363 := RowAgreement.single (first := 3362) modelrow_agrees_3363
private def atomic_3364 : CountSegment 1 10 13120 3363 3364 1606 1606 := CountSegment.single (first := 3363) (initial := 1606) row_count_3364 (by decide)
private theorem atomic_3364_agreement : RowAgreement 3363 3364 := RowAgreement.single (first := 3363) modelrow_agrees_3364
private def atomic_3365 : CountSegment 1 10 13120 3364 3365 1606 1606 := CountSegment.single (first := 3364) (initial := 1606) row_count_3365 (by decide)
private theorem atomic_3365_agreement : RowAgreement 3364 3365 := RowAgreement.single (first := 3364) modelrow_agrees_3365
private def atomic_3366 : CountSegment 1 10 13120 3365 3366 1606 1606 := CountSegment.single (first := 3365) (initial := 1606) row_count_3366 (by decide)
private theorem atomic_3366_agreement : RowAgreement 3365 3366 := RowAgreement.single (first := 3365) modelrow_agrees_3366
private def atomic_3367 : CountSegment 1 10 13120 3366 3367 1606 1608 := CountSegment.single (first := 3366) (initial := 1606) row_count_3367 (by decide)
private theorem atomic_3367_agreement : RowAgreement 3366 3367 := RowAgreement.single (first := 3366) modelrow_agrees_3367
private def atomic_3368 : CountSegment 1 10 13120 3367 3368 1608 1608 := CountSegment.single (first := 3367) (initial := 1608) row_count_3368 (by decide)
private theorem atomic_3368_agreement : RowAgreement 3367 3368 := RowAgreement.single (first := 3367) modelrow_agrees_3368
private def atomic_3369 : CountSegment 1 10 13120 3368 3369 1608 1608 := CountSegment.single (first := 3368) (initial := 1608) row_count_3369 (by decide)
private theorem atomic_3369_agreement : RowAgreement 3368 3369 := RowAgreement.single (first := 3368) modelrow_agrees_3369
private def atomic_3370 : CountSegment 1 10 13120 3369 3370 1608 1609 := CountSegment.single (first := 3369) (initial := 1608) row_count_3370 (by decide)
private theorem atomic_3370_agreement : RowAgreement 3369 3370 := RowAgreement.single (first := 3369) modelrow_agrees_3370
private def atomic_3371 : CountSegment 1 10 13120 3370 3371 1609 1609 := CountSegment.single (first := 3370) (initial := 1609) row_count_3371 (by decide)
private theorem atomic_3371_agreement : RowAgreement 3370 3371 := RowAgreement.single (first := 3370) modelrow_agrees_3371
private def atomic_3372 : CountSegment 1 10 13120 3371 3372 1609 1609 := CountSegment.single (first := 3371) (initial := 1609) row_count_3372 (by decide)
private theorem atomic_3372_agreement : RowAgreement 3371 3372 := RowAgreement.single (first := 3371) modelrow_agrees_3372
private def atomic_3373 : CountSegment 1 10 13120 3372 3373 1609 1609 := CountSegment.single (first := 3372) (initial := 1609) row_count_3373 (by decide)
private theorem atomic_3373_agreement : RowAgreement 3372 3373 := RowAgreement.single (first := 3372) modelrow_agrees_3373
private def atomic_3374 : CountSegment 1 10 13120 3373 3374 1609 1609 := CountSegment.single (first := 3373) (initial := 1609) row_count_3374 (by decide)
private theorem atomic_3374_agreement : RowAgreement 3373 3374 := RowAgreement.single (first := 3373) modelrow_agrees_3374
private def atomic_3375 : CountSegment 1 10 13120 3374 3375 1609 1610 := CountSegment.single (first := 3374) (initial := 1609) row_count_3375 (by decide)
private theorem atomic_3375_agreement : RowAgreement 3374 3375 := RowAgreement.single (first := 3374) modelrow_agrees_3375
private def atomic_3376 : CountSegment 1 10 13120 3375 3376 1610 1610 := CountSegment.single (first := 3375) (initial := 1610) row_count_3376 (by decide)
private theorem atomic_3376_agreement : RowAgreement 3375 3376 := RowAgreement.single (first := 3375) modelrow_agrees_3376
private def atomic_3377 : CountSegment 1 10 13120 3376 3377 1610 1610 := CountSegment.single (first := 3376) (initial := 1610) row_count_3377 (by decide)
private theorem atomic_3377_agreement : RowAgreement 3376 3377 := RowAgreement.single (first := 3376) modelrow_agrees_3377
private def atomic_3378 : CountSegment 1 10 13120 3377 3378 1610 1610 := CountSegment.single (first := 3377) (initial := 1610) row_count_3378 (by decide)
private theorem atomic_3378_agreement : RowAgreement 3377 3378 := RowAgreement.single (first := 3377) modelrow_agrees_3378
private def atomic_3379 : CountSegment 1 10 13120 3378 3379 1610 1610 := CountSegment.single (first := 3378) (initial := 1610) row_count_3379 (by decide)
private theorem atomic_3379_agreement : RowAgreement 3378 3379 := RowAgreement.single (first := 3378) modelrow_agrees_3379
private def atomic_3380 : CountSegment 1 10 13120 3379 3380 1610 1610 := CountSegment.single (first := 3379) (initial := 1610) row_count_3380 (by decide)
private theorem atomic_3380_agreement : RowAgreement 3379 3380 := RowAgreement.single (first := 3379) modelrow_agrees_3380
private def atomic_3381 : CountSegment 1 10 13120 3380 3381 1610 1611 := CountSegment.single (first := 3380) (initial := 1610) row_count_3381 (by decide)
private theorem atomic_3381_agreement : RowAgreement 3380 3381 := RowAgreement.single (first := 3380) modelrow_agrees_3381
private def atomic_3382 : CountSegment 1 10 13120 3381 3382 1611 1612 := CountSegment.single (first := 3381) (initial := 1611) row_count_3382 (by decide)
private theorem atomic_3382_agreement : RowAgreement 3381 3382 := RowAgreement.single (first := 3381) modelrow_agrees_3382
private def atomic_3383 : CountSegment 1 10 13120 3382 3383 1612 1612 := CountSegment.single (first := 3382) (initial := 1612) row_count_3383 (by decide)
private theorem atomic_3383_agreement : RowAgreement 3382 3383 := RowAgreement.single (first := 3382) modelrow_agrees_3383
private def atomic_3384 : CountSegment 1 10 13120 3383 3384 1612 1613 := CountSegment.single (first := 3383) (initial := 1612) row_count_3384 (by decide)
private theorem atomic_3384_agreement : RowAgreement 3383 3384 := RowAgreement.single (first := 3383) modelrow_agrees_3384
private def atomic_3385 : CountSegment 1 10 13120 3384 3385 1613 1613 := CountSegment.single (first := 3384) (initial := 1613) row_count_3385 (by decide)
private theorem atomic_3385_agreement : RowAgreement 3384 3385 := RowAgreement.single (first := 3384) modelrow_agrees_3385
private def atomic_3386 : CountSegment 1 10 13120 3385 3386 1613 1613 := CountSegment.single (first := 3385) (initial := 1613) row_count_3386 (by decide)
private theorem atomic_3386_agreement : RowAgreement 3385 3386 := RowAgreement.single (first := 3385) modelrow_agrees_3386
private def atomic_3387 : CountSegment 1 10 13120 3386 3387 1613 1614 := CountSegment.single (first := 3386) (initial := 1613) row_count_3387 (by decide)
private theorem atomic_3387_agreement : RowAgreement 3386 3387 := RowAgreement.single (first := 3386) modelrow_agrees_3387
private def atomic_3388 : CountSegment 1 10 13120 3387 3388 1614 1615 := CountSegment.single (first := 3387) (initial := 1614) row_count_3388 (by decide)
private theorem atomic_3388_agreement : RowAgreement 3387 3388 := RowAgreement.single (first := 3387) modelrow_agrees_3388
private def atomic_3389 : CountSegment 1 10 13120 3388 3389 1615 1615 := CountSegment.single (first := 3388) (initial := 1615) row_count_3389 (by decide)
private theorem atomic_3389_agreement : RowAgreement 3388 3389 := RowAgreement.single (first := 3388) modelrow_agrees_3389
private def atomic_3390 : CountSegment 1 10 13120 3389 3390 1615 1616 := CountSegment.single (first := 3389) (initial := 1615) row_count_3390 (by decide)
private theorem atomic_3390_agreement : RowAgreement 3389 3390 := RowAgreement.single (first := 3389) modelrow_agrees_3390
private def atomic_3391 : CountSegment 1 10 13120 3390 3391 1616 1616 := CountSegment.single (first := 3390) (initial := 1616) row_count_3391 (by decide)
private theorem atomic_3391_agreement : RowAgreement 3390 3391 := RowAgreement.single (first := 3390) modelrow_agrees_3391
private def atomic_3392 : CountSegment 1 10 13120 3391 3392 1616 1617 := CountSegment.single (first := 3391) (initial := 1616) row_count_3392 (by decide)
private theorem atomic_3392_agreement : RowAgreement 3391 3392 := RowAgreement.single (first := 3391) modelrow_agrees_3392
private def atomic_3393 : CountSegment 1 10 13120 3392 3393 1617 1618 := CountSegment.single (first := 3392) (initial := 1617) row_count_3393 (by decide)
private theorem atomic_3393_agreement : RowAgreement 3392 3393 := RowAgreement.single (first := 3392) modelrow_agrees_3393
private def atomic_3394 : CountSegment 1 10 13120 3393 3394 1618 1619 := CountSegment.single (first := 3393) (initial := 1618) row_count_3394 (by decide)
private theorem atomic_3394_agreement : RowAgreement 3393 3394 := RowAgreement.single (first := 3393) modelrow_agrees_3394
private def atomic_3395 : CountSegment 1 10 13120 3394 3395 1619 1619 := CountSegment.single (first := 3394) (initial := 1619) row_count_3395 (by decide)
private theorem atomic_3395_agreement : RowAgreement 3394 3395 := RowAgreement.single (first := 3394) modelrow_agrees_3395
private def atomic_3396 : CountSegment 1 10 13120 3395 3396 1619 1620 := CountSegment.single (first := 3395) (initial := 1619) row_count_3396 (by decide)
private theorem atomic_3396_agreement : RowAgreement 3395 3396 := RowAgreement.single (first := 3395) modelrow_agrees_3396
private def atomic_3397 : CountSegment 1 10 13120 3396 3397 1620 1620 := CountSegment.single (first := 3396) (initial := 1620) row_count_3397 (by decide)
private theorem atomic_3397_agreement : RowAgreement 3396 3397 := RowAgreement.single (first := 3396) modelrow_agrees_3397
private def atomic_3398 : CountSegment 1 10 13120 3397 3398 1620 1620 := CountSegment.single (first := 3397) (initial := 1620) row_count_3398 (by decide)
private theorem atomic_3398_agreement : RowAgreement 3397 3398 := RowAgreement.single (first := 3397) modelrow_agrees_3398
private def atomic_3399 : CountSegment 1 10 13120 3398 3399 1620 1621 := CountSegment.single (first := 3398) (initial := 1620) row_count_3399 (by decide)
private theorem atomic_3399_agreement : RowAgreement 3398 3399 := RowAgreement.single (first := 3398) modelrow_agrees_3399
private def atomic_3400 : CountSegment 1 10 13120 3399 3400 1621 1622 := CountSegment.single (first := 3399) (initial := 1621) row_count_3400 (by decide)
private theorem atomic_3400_agreement : RowAgreement 3399 3400 := RowAgreement.single (first := 3399) modelrow_agrees_3400
private def atomic_3401 : CountSegment 1 10 13120 3400 3401 1622 1622 := CountSegment.single (first := 3400) (initial := 1622) row_count_3401 (by decide)
private theorem atomic_3401_agreement : RowAgreement 3400 3401 := RowAgreement.single (first := 3400) modelrow_agrees_3401
private def atomic_3402 : CountSegment 1 10 13120 3401 3402 1622 1622 := CountSegment.single (first := 3401) (initial := 1622) row_count_3402 (by decide)
private theorem atomic_3402_agreement : RowAgreement 3401 3402 := RowAgreement.single (first := 3401) modelrow_agrees_3402
private def atomic_3403 : CountSegment 1 10 13120 3402 3403 1622 1623 := CountSegment.single (first := 3402) (initial := 1622) row_count_3403 (by decide)
private theorem atomic_3403_agreement : RowAgreement 3402 3403 := RowAgreement.single (first := 3402) modelrow_agrees_3403
private def atomic_3404 : CountSegment 1 10 13120 3403 3404 1623 1624 := CountSegment.single (first := 3403) (initial := 1623) row_count_3404 (by decide)
private theorem atomic_3404_agreement : RowAgreement 3403 3404 := RowAgreement.single (first := 3403) modelrow_agrees_3404
private def atomic_3405 : CountSegment 1 10 13120 3404 3405 1624 1625 := CountSegment.single (first := 3404) (initial := 1624) row_count_3405 (by decide)
private theorem atomic_3405_agreement : RowAgreement 3404 3405 := RowAgreement.single (first := 3404) modelrow_agrees_3405
private def atomic_3406 : CountSegment 1 10 13120 3405 3406 1625 1626 := CountSegment.single (first := 3405) (initial := 1625) row_count_3406 (by decide)
private theorem atomic_3406_agreement : RowAgreement 3405 3406 := RowAgreement.single (first := 3405) modelrow_agrees_3406
private def atomic_3407 : CountSegment 1 10 13120 3406 3407 1626 1626 := CountSegment.single (first := 3406) (initial := 1626) row_count_3407 (by decide)
private theorem atomic_3407_agreement : RowAgreement 3406 3407 := RowAgreement.single (first := 3406) modelrow_agrees_3407
private def atomic_3408 : CountSegment 1 10 13120 3407 3408 1626 1627 := CountSegment.single (first := 3407) (initial := 1626) row_count_3408 (by decide)
private theorem atomic_3408_agreement : RowAgreement 3407 3408 := RowAgreement.single (first := 3407) modelrow_agrees_3408
private def atomic_3409 : CountSegment 1 10 13120 3408 3409 1627 1627 := CountSegment.single (first := 3408) (initial := 1627) row_count_3409 (by decide)
private theorem atomic_3409_agreement : RowAgreement 3408 3409 := RowAgreement.single (first := 3408) modelrow_agrees_3409
private def atomic_3410 : CountSegment 1 10 13120 3409 3410 1627 1627 := CountSegment.single (first := 3409) (initial := 1627) row_count_3410 (by decide)
private theorem atomic_3410_agreement : RowAgreement 3409 3410 := RowAgreement.single (first := 3409) modelrow_agrees_3410
private def atomic_3411 : CountSegment 1 10 13120 3410 3411 1627 1627 := CountSegment.single (first := 3410) (initial := 1627) row_count_3411 (by decide)
private theorem atomic_3411_agreement : RowAgreement 3410 3411 := RowAgreement.single (first := 3410) modelrow_agrees_3411
private def atomic_3412 : CountSegment 1 10 13120 3411 3412 1627 1628 := CountSegment.single (first := 3411) (initial := 1627) row_count_3412 (by decide)
private theorem atomic_3412_agreement : RowAgreement 3411 3412 := RowAgreement.single (first := 3411) modelrow_agrees_3412
private def atomic_3413 : CountSegment 1 10 13120 3412 3413 1628 1629 := CountSegment.single (first := 3412) (initial := 1628) row_count_3413 (by decide)
private theorem atomic_3413_agreement : RowAgreement 3412 3413 := RowAgreement.single (first := 3412) modelrow_agrees_3413
private def atomic_3414 : CountSegment 1 10 13120 3413 3414 1629 1629 := CountSegment.single (first := 3413) (initial := 1629) row_count_3414 (by decide)
private theorem atomic_3414_agreement : RowAgreement 3413 3414 := RowAgreement.single (first := 3413) modelrow_agrees_3414
private def atomic_3415 : CountSegment 1 10 13120 3414 3415 1629 1629 := CountSegment.single (first := 3414) (initial := 1629) row_count_3415 (by decide)
private theorem atomic_3415_agreement : RowAgreement 3414 3415 := RowAgreement.single (first := 3414) modelrow_agrees_3415
private def atomic_3416 : CountSegment 1 10 13120 3415 3416 1629 1629 := CountSegment.single (first := 3415) (initial := 1629) row_count_3416 (by decide)
private theorem atomic_3416_agreement : RowAgreement 3415 3416 := RowAgreement.single (first := 3415) modelrow_agrees_3416
private def atomic_3417 : CountSegment 1 10 13120 3416 3417 1629 1630 := CountSegment.single (first := 3416) (initial := 1629) row_count_3417 (by decide)
private theorem atomic_3417_agreement : RowAgreement 3416 3417 := RowAgreement.single (first := 3416) modelrow_agrees_3417
private def atomic_3418 : CountSegment 1 10 13120 3417 3418 1630 1630 := CountSegment.single (first := 3417) (initial := 1630) row_count_3418 (by decide)
private theorem atomic_3418_agreement : RowAgreement 3417 3418 := RowAgreement.single (first := 3417) modelrow_agrees_3418
private def atomic_3419 : CountSegment 1 10 13120 3418 3419 1630 1631 := CountSegment.single (first := 3418) (initial := 1630) row_count_3419 (by decide)
private theorem atomic_3419_agreement : RowAgreement 3418 3419 := RowAgreement.single (first := 3418) modelrow_agrees_3419
private def atomic_3420 : CountSegment 1 10 13120 3419 3420 1631 1631 := CountSegment.single (first := 3419) (initial := 1631) row_count_3420 (by decide)
private theorem atomic_3420_agreement : RowAgreement 3419 3420 := RowAgreement.single (first := 3419) modelrow_agrees_3420
private def atomic_3421 : CountSegment 1 10 13120 3420 3421 1631 1632 := CountSegment.single (first := 3420) (initial := 1631) row_count_3421 (by decide)
private theorem atomic_3421_agreement : RowAgreement 3420 3421 := RowAgreement.single (first := 3420) modelrow_agrees_3421
private def atomic_3422 : CountSegment 1 10 13120 3421 3422 1632 1632 := CountSegment.single (first := 3421) (initial := 1632) row_count_3422 (by decide)
private theorem atomic_3422_agreement : RowAgreement 3421 3422 := RowAgreement.single (first := 3421) modelrow_agrees_3422
private def atomic_3423 : CountSegment 1 10 13120 3422 3423 1632 1632 := CountSegment.single (first := 3422) (initial := 1632) row_count_3423 (by decide)
private theorem atomic_3423_agreement : RowAgreement 3422 3423 := RowAgreement.single (first := 3422) modelrow_agrees_3423
private def atomic_3424 : CountSegment 1 10 13120 3423 3424 1632 1632 := CountSegment.single (first := 3423) (initial := 1632) row_count_3424 (by decide)
private theorem atomic_3424_agreement : RowAgreement 3423 3424 := RowAgreement.single (first := 3423) modelrow_agrees_3424
private def atomic_3425 : CountSegment 1 10 13120 3424 3425 1632 1633 := CountSegment.single (first := 3424) (initial := 1632) row_count_3425 (by decide)
private theorem atomic_3425_agreement : RowAgreement 3424 3425 := RowAgreement.single (first := 3424) modelrow_agrees_3425
private def atomic_3426 : CountSegment 1 10 13120 3425 3426 1633 1633 := CountSegment.single (first := 3425) (initial := 1633) row_count_3426 (by decide)
private theorem atomic_3426_agreement : RowAgreement 3425 3426 := RowAgreement.single (first := 3425) modelrow_agrees_3426
private def atomic_3427 : CountSegment 1 10 13120 3426 3427 1633 1634 := CountSegment.single (first := 3426) (initial := 1633) row_count_3427 (by decide)
private theorem atomic_3427_agreement : RowAgreement 3426 3427 := RowAgreement.single (first := 3426) modelrow_agrees_3427
private def atomic_3428 : CountSegment 1 10 13120 3427 3428 1634 1634 := CountSegment.single (first := 3427) (initial := 1634) row_count_3428 (by decide)
private theorem atomic_3428_agreement : RowAgreement 3427 3428 := RowAgreement.single (first := 3427) modelrow_agrees_3428
private def atomic_3429 : CountSegment 1 10 13120 3428 3429 1634 1634 := CountSegment.single (first := 3428) (initial := 1634) row_count_3429 (by decide)
private theorem atomic_3429_agreement : RowAgreement 3428 3429 := RowAgreement.single (first := 3428) modelrow_agrees_3429
private def atomic_3430 : CountSegment 1 10 13120 3429 3430 1634 1634 := CountSegment.single (first := 3429) (initial := 1634) row_count_3430 (by decide)
private theorem atomic_3430_agreement : RowAgreement 3429 3430 := RowAgreement.single (first := 3429) modelrow_agrees_3430
private def atomic_3431 : CountSegment 1 10 13120 3430 3431 1634 1634 := CountSegment.single (first := 3430) (initial := 1634) row_count_3431 (by decide)
private theorem atomic_3431_agreement : RowAgreement 3430 3431 := RowAgreement.single (first := 3430) modelrow_agrees_3431
private def atomic_3432 : CountSegment 1 10 13120 3431 3432 1634 1634 := CountSegment.single (first := 3431) (initial := 1634) row_count_3432 (by decide)
private theorem atomic_3432_agreement : RowAgreement 3431 3432 := RowAgreement.single (first := 3431) modelrow_agrees_3432
private def atomic_3433 : CountSegment 1 10 13120 3432 3433 1634 1635 := CountSegment.single (first := 3432) (initial := 1634) row_count_3433 (by decide)
private theorem atomic_3433_agreement : RowAgreement 3432 3433 := RowAgreement.single (first := 3432) modelrow_agrees_3433
private def atomic_3434 : CountSegment 1 10 13120 3433 3434 1635 1635 := CountSegment.single (first := 3433) (initial := 1635) row_count_3434 (by decide)
private theorem atomic_3434_agreement : RowAgreement 3433 3434 := RowAgreement.single (first := 3433) modelrow_agrees_3434
private def atomic_3435 : CountSegment 1 10 13120 3434 3435 1635 1635 := CountSegment.single (first := 3434) (initial := 1635) row_count_3435 (by decide)
private theorem atomic_3435_agreement : RowAgreement 3434 3435 := RowAgreement.single (first := 3434) modelrow_agrees_3435
private def atomic_3436 : CountSegment 1 10 13120 3435 3436 1635 1635 := CountSegment.single (first := 3435) (initial := 1635) row_count_3436 (by decide)
private theorem atomic_3436_agreement : RowAgreement 3435 3436 := RowAgreement.single (first := 3435) modelrow_agrees_3436
private def atomic_3437 : CountSegment 1 10 13120 3436 3437 1635 1636 := CountSegment.single (first := 3436) (initial := 1635) row_count_3437 (by decide)
private theorem atomic_3437_agreement : RowAgreement 3436 3437 := RowAgreement.single (first := 3436) modelrow_agrees_3437
private def atomic_3438 : CountSegment 1 10 13120 3437 3438 1636 1637 := CountSegment.single (first := 3437) (initial := 1636) row_count_3438 (by decide)
private theorem atomic_3438_agreement : RowAgreement 3437 3438 := RowAgreement.single (first := 3437) modelrow_agrees_3438
private def atomic_3439 : CountSegment 1 10 13120 3438 3439 1637 1637 := CountSegment.single (first := 3438) (initial := 1637) row_count_3439 (by decide)
private theorem atomic_3439_agreement : RowAgreement 3438 3439 := RowAgreement.single (first := 3438) modelrow_agrees_3439
private def atomic_3440 : CountSegment 1 10 13120 3439 3440 1637 1637 := CountSegment.single (first := 3439) (initial := 1637) row_count_3440 (by decide)
private theorem atomic_3440_agreement : RowAgreement 3439 3440 := RowAgreement.single (first := 3439) modelrow_agrees_3440
private def atomic_3441 : CountSegment 1 10 13120 3440 3441 1637 1638 := CountSegment.single (first := 3440) (initial := 1637) row_count_3441 (by decide)
private theorem atomic_3441_agreement : RowAgreement 3440 3441 := RowAgreement.single (first := 3440) modelrow_agrees_3441
private def atomic_3442 : CountSegment 1 10 13120 3441 3442 1638 1640 := CountSegment.single (first := 3441) (initial := 1638) row_count_3442 (by decide)
private theorem atomic_3442_agreement : RowAgreement 3441 3442 := RowAgreement.single (first := 3441) modelrow_agrees_3442
private def atomic_3443 : CountSegment 1 10 13120 3442 3443 1640 1640 := CountSegment.single (first := 3442) (initial := 1640) row_count_3443 (by decide)
private theorem atomic_3443_agreement : RowAgreement 3442 3443 := RowAgreement.single (first := 3442) modelrow_agrees_3443
private def atomic_3444 : CountSegment 1 10 13120 3443 3444 1640 1641 := CountSegment.single (first := 3443) (initial := 1640) row_count_3444 (by decide)
private theorem atomic_3444_agreement : RowAgreement 3443 3444 := RowAgreement.single (first := 3443) modelrow_agrees_3444
private def atomic_3445 : CountSegment 1 10 13120 3444 3445 1641 1641 := CountSegment.single (first := 3444) (initial := 1641) row_count_3445 (by decide)
private theorem atomic_3445_agreement : RowAgreement 3444 3445 := RowAgreement.single (first := 3444) modelrow_agrees_3445
private def atomic_3446 : CountSegment 1 10 13120 3445 3446 1641 1641 := CountSegment.single (first := 3445) (initial := 1641) row_count_3446 (by decide)
private theorem atomic_3446_agreement : RowAgreement 3445 3446 := RowAgreement.single (first := 3445) modelrow_agrees_3446
private def atomic_3447 : CountSegment 1 10 13120 3446 3447 1641 1641 := CountSegment.single (first := 3446) (initial := 1641) row_count_3447 (by decide)
private theorem atomic_3447_agreement : RowAgreement 3446 3447 := RowAgreement.single (first := 3446) modelrow_agrees_3447
private def atomic_3448 : CountSegment 1 10 13120 3447 3448 1641 1641 := CountSegment.single (first := 3447) (initial := 1641) row_count_3448 (by decide)
private theorem atomic_3448_agreement : RowAgreement 3447 3448 := RowAgreement.single (first := 3447) modelrow_agrees_3448
private def atomic_3449 : CountSegment 1 10 13120 3448 3449 1641 1641 := CountSegment.single (first := 3448) (initial := 1641) row_count_3449 (by decide)
private theorem atomic_3449_agreement : RowAgreement 3448 3449 := RowAgreement.single (first := 3448) modelrow_agrees_3449
private def atomic_3450 : CountSegment 1 10 13120 3449 3450 1641 1641 := CountSegment.single (first := 3449) (initial := 1641) row_count_3450 (by decide)
private theorem atomic_3450_agreement : RowAgreement 3449 3450 := RowAgreement.single (first := 3449) modelrow_agrees_3450
private def atomic_3451 : CountSegment 1 10 13120 3450 3451 1641 1641 := CountSegment.single (first := 3450) (initial := 1641) row_count_3451 (by decide)
private theorem atomic_3451_agreement : RowAgreement 3450 3451 := RowAgreement.single (first := 3450) modelrow_agrees_3451
private def atomic_3452 : CountSegment 1 10 13120 3451 3452 1641 1641 := CountSegment.single (first := 3451) (initial := 1641) row_count_3452 (by decide)
private theorem atomic_3452_agreement : RowAgreement 3451 3452 := RowAgreement.single (first := 3451) modelrow_agrees_3452
private def atomic_3453 : CountSegment 1 10 13120 3452 3453 1641 1642 := CountSegment.single (first := 3452) (initial := 1641) row_count_3453 (by decide)
private theorem atomic_3453_agreement : RowAgreement 3452 3453 := RowAgreement.single (first := 3452) modelrow_agrees_3453
private def atomic_3454 : CountSegment 1 10 13120 3453 3454 1642 1642 := CountSegment.single (first := 3453) (initial := 1642) row_count_3454 (by decide)
private theorem atomic_3454_agreement : RowAgreement 3453 3454 := RowAgreement.single (first := 3453) modelrow_agrees_3454
private def atomic_3455 : CountSegment 1 10 13120 3454 3455 1642 1644 := CountSegment.single (first := 3454) (initial := 1642) row_count_3455 (by decide)
private theorem atomic_3455_agreement : RowAgreement 3454 3455 := RowAgreement.single (first := 3454) modelrow_agrees_3455
private def atomic_3456 : CountSegment 1 10 13120 3455 3456 1644 1644 := CountSegment.single (first := 3455) (initial := 1644) row_count_3456 (by decide)
private theorem atomic_3456_agreement : RowAgreement 3455 3456 := RowAgreement.single (first := 3455) modelrow_agrees_3456
private def within_026_000 : CountSegment 1 10 13120 3328 3330 1595 1595 := CountSegment.join atomic_3329 atomic_3330
private theorem within_026_000_agreement : RowAgreement 3328 3330 := RowAgreement.join atomic_3329_agreement atomic_3330_agreement
private def within_026_001 : CountSegment 1 10 13120 3330 3332 1595 1595 := CountSegment.join atomic_3331 atomic_3332
private theorem within_026_001_agreement : RowAgreement 3330 3332 := RowAgreement.join atomic_3331_agreement atomic_3332_agreement
private def within_026_002 : CountSegment 1 10 13120 3328 3332 1595 1595 := CountSegment.join within_026_000 within_026_001
private theorem within_026_002_agreement : RowAgreement 3328 3332 := RowAgreement.join within_026_000_agreement within_026_001_agreement
private def within_026_003 : CountSegment 1 10 13120 3332 3334 1595 1595 := CountSegment.join atomic_3333 atomic_3334
private theorem within_026_003_agreement : RowAgreement 3332 3334 := RowAgreement.join atomic_3333_agreement atomic_3334_agreement
private def within_026_004 : CountSegment 1 10 13120 3334 3336 1595 1596 := CountSegment.join atomic_3335 atomic_3336
private theorem within_026_004_agreement : RowAgreement 3334 3336 := RowAgreement.join atomic_3335_agreement atomic_3336_agreement
private def within_026_005 : CountSegment 1 10 13120 3332 3336 1595 1596 := CountSegment.join within_026_003 within_026_004
private theorem within_026_005_agreement : RowAgreement 3332 3336 := RowAgreement.join within_026_003_agreement within_026_004_agreement
private def within_026_006 : CountSegment 1 10 13120 3328 3336 1595 1596 := CountSegment.join within_026_002 within_026_005
private theorem within_026_006_agreement : RowAgreement 3328 3336 := RowAgreement.join within_026_002_agreement within_026_005_agreement
private def within_026_007 : CountSegment 1 10 13120 3336 3338 1596 1596 := CountSegment.join atomic_3337 atomic_3338
private theorem within_026_007_agreement : RowAgreement 3336 3338 := RowAgreement.join atomic_3337_agreement atomic_3338_agreement
private def within_026_008 : CountSegment 1 10 13120 3338 3340 1596 1596 := CountSegment.join atomic_3339 atomic_3340
private theorem within_026_008_agreement : RowAgreement 3338 3340 := RowAgreement.join atomic_3339_agreement atomic_3340_agreement
private def within_026_009 : CountSegment 1 10 13120 3336 3340 1596 1596 := CountSegment.join within_026_007 within_026_008
private theorem within_026_009_agreement : RowAgreement 3336 3340 := RowAgreement.join within_026_007_agreement within_026_008_agreement
private def within_026_010 : CountSegment 1 10 13120 3340 3342 1596 1597 := CountSegment.join atomic_3341 atomic_3342
private theorem within_026_010_agreement : RowAgreement 3340 3342 := RowAgreement.join atomic_3341_agreement atomic_3342_agreement
private def within_026_011 : CountSegment 1 10 13120 3342 3344 1597 1597 := CountSegment.join atomic_3343 atomic_3344
private theorem within_026_011_agreement : RowAgreement 3342 3344 := RowAgreement.join atomic_3343_agreement atomic_3344_agreement
private def within_026_012 : CountSegment 1 10 13120 3340 3344 1596 1597 := CountSegment.join within_026_010 within_026_011
private theorem within_026_012_agreement : RowAgreement 3340 3344 := RowAgreement.join within_026_010_agreement within_026_011_agreement
private def within_026_013 : CountSegment 1 10 13120 3336 3344 1596 1597 := CountSegment.join within_026_009 within_026_012
private theorem within_026_013_agreement : RowAgreement 3336 3344 := RowAgreement.join within_026_009_agreement within_026_012_agreement
private def within_026_014 : CountSegment 1 10 13120 3328 3344 1595 1597 := CountSegment.join within_026_006 within_026_013
private theorem within_026_014_agreement : RowAgreement 3328 3344 := RowAgreement.join within_026_006_agreement within_026_013_agreement
private def within_026_015 : CountSegment 1 10 13120 3344 3346 1597 1598 := CountSegment.join atomic_3345 atomic_3346
private theorem within_026_015_agreement : RowAgreement 3344 3346 := RowAgreement.join atomic_3345_agreement atomic_3346_agreement
private def within_026_016 : CountSegment 1 10 13120 3346 3348 1598 1598 := CountSegment.join atomic_3347 atomic_3348
private theorem within_026_016_agreement : RowAgreement 3346 3348 := RowAgreement.join atomic_3347_agreement atomic_3348_agreement
private def within_026_017 : CountSegment 1 10 13120 3344 3348 1597 1598 := CountSegment.join within_026_015 within_026_016
private theorem within_026_017_agreement : RowAgreement 3344 3348 := RowAgreement.join within_026_015_agreement within_026_016_agreement
private def within_026_018 : CountSegment 1 10 13120 3348 3350 1598 1598 := CountSegment.join atomic_3349 atomic_3350
private theorem within_026_018_agreement : RowAgreement 3348 3350 := RowAgreement.join atomic_3349_agreement atomic_3350_agreement
private def within_026_019 : CountSegment 1 10 13120 3350 3352 1598 1600 := CountSegment.join atomic_3351 atomic_3352
private theorem within_026_019_agreement : RowAgreement 3350 3352 := RowAgreement.join atomic_3351_agreement atomic_3352_agreement
private def within_026_020 : CountSegment 1 10 13120 3348 3352 1598 1600 := CountSegment.join within_026_018 within_026_019
private theorem within_026_020_agreement : RowAgreement 3348 3352 := RowAgreement.join within_026_018_agreement within_026_019_agreement
private def within_026_021 : CountSegment 1 10 13120 3344 3352 1597 1600 := CountSegment.join within_026_017 within_026_020
private theorem within_026_021_agreement : RowAgreement 3344 3352 := RowAgreement.join within_026_017_agreement within_026_020_agreement
private def within_026_022 : CountSegment 1 10 13120 3352 3354 1600 1600 := CountSegment.join atomic_3353 atomic_3354
private theorem within_026_022_agreement : RowAgreement 3352 3354 := RowAgreement.join atomic_3353_agreement atomic_3354_agreement
private def within_026_023 : CountSegment 1 10 13120 3354 3356 1600 1601 := CountSegment.join atomic_3355 atomic_3356
private theorem within_026_023_agreement : RowAgreement 3354 3356 := RowAgreement.join atomic_3355_agreement atomic_3356_agreement
private def within_026_024 : CountSegment 1 10 13120 3352 3356 1600 1601 := CountSegment.join within_026_022 within_026_023
private theorem within_026_024_agreement : RowAgreement 3352 3356 := RowAgreement.join within_026_022_agreement within_026_023_agreement
private def within_026_025 : CountSegment 1 10 13120 3356 3358 1601 1602 := CountSegment.join atomic_3357 atomic_3358
private theorem within_026_025_agreement : RowAgreement 3356 3358 := RowAgreement.join atomic_3357_agreement atomic_3358_agreement
private def within_026_026 : CountSegment 1 10 13120 3358 3360 1602 1604 := CountSegment.join atomic_3359 atomic_3360
private theorem within_026_026_agreement : RowAgreement 3358 3360 := RowAgreement.join atomic_3359_agreement atomic_3360_agreement
private def within_026_027 : CountSegment 1 10 13120 3356 3360 1601 1604 := CountSegment.join within_026_025 within_026_026
private theorem within_026_027_agreement : RowAgreement 3356 3360 := RowAgreement.join within_026_025_agreement within_026_026_agreement
private def within_026_028 : CountSegment 1 10 13120 3352 3360 1600 1604 := CountSegment.join within_026_024 within_026_027
private theorem within_026_028_agreement : RowAgreement 3352 3360 := RowAgreement.join within_026_024_agreement within_026_027_agreement
private def within_026_029 : CountSegment 1 10 13120 3344 3360 1597 1604 := CountSegment.join within_026_021 within_026_028
private theorem within_026_029_agreement : RowAgreement 3344 3360 := RowAgreement.join within_026_021_agreement within_026_028_agreement
private def within_026_030 : CountSegment 1 10 13120 3328 3360 1595 1604 := CountSegment.join within_026_014 within_026_029
private theorem within_026_030_agreement : RowAgreement 3328 3360 := RowAgreement.join within_026_014_agreement within_026_029_agreement
private def within_026_031 : CountSegment 1 10 13120 3360 3362 1604 1605 := CountSegment.join atomic_3361 atomic_3362
private theorem within_026_031_agreement : RowAgreement 3360 3362 := RowAgreement.join atomic_3361_agreement atomic_3362_agreement
private def within_026_032 : CountSegment 1 10 13120 3362 3364 1605 1606 := CountSegment.join atomic_3363 atomic_3364
private theorem within_026_032_agreement : RowAgreement 3362 3364 := RowAgreement.join atomic_3363_agreement atomic_3364_agreement
private def within_026_033 : CountSegment 1 10 13120 3360 3364 1604 1606 := CountSegment.join within_026_031 within_026_032
private theorem within_026_033_agreement : RowAgreement 3360 3364 := RowAgreement.join within_026_031_agreement within_026_032_agreement
private def within_026_034 : CountSegment 1 10 13120 3364 3366 1606 1606 := CountSegment.join atomic_3365 atomic_3366
private theorem within_026_034_agreement : RowAgreement 3364 3366 := RowAgreement.join atomic_3365_agreement atomic_3366_agreement
private def within_026_035 : CountSegment 1 10 13120 3366 3368 1606 1608 := CountSegment.join atomic_3367 atomic_3368
private theorem within_026_035_agreement : RowAgreement 3366 3368 := RowAgreement.join atomic_3367_agreement atomic_3368_agreement
private def within_026_036 : CountSegment 1 10 13120 3364 3368 1606 1608 := CountSegment.join within_026_034 within_026_035
private theorem within_026_036_agreement : RowAgreement 3364 3368 := RowAgreement.join within_026_034_agreement within_026_035_agreement
private def within_026_037 : CountSegment 1 10 13120 3360 3368 1604 1608 := CountSegment.join within_026_033 within_026_036
private theorem within_026_037_agreement : RowAgreement 3360 3368 := RowAgreement.join within_026_033_agreement within_026_036_agreement
private def within_026_038 : CountSegment 1 10 13120 3368 3370 1608 1609 := CountSegment.join atomic_3369 atomic_3370
private theorem within_026_038_agreement : RowAgreement 3368 3370 := RowAgreement.join atomic_3369_agreement atomic_3370_agreement
private def within_026_039 : CountSegment 1 10 13120 3370 3372 1609 1609 := CountSegment.join atomic_3371 atomic_3372
private theorem within_026_039_agreement : RowAgreement 3370 3372 := RowAgreement.join atomic_3371_agreement atomic_3372_agreement
private def within_026_040 : CountSegment 1 10 13120 3368 3372 1608 1609 := CountSegment.join within_026_038 within_026_039
private theorem within_026_040_agreement : RowAgreement 3368 3372 := RowAgreement.join within_026_038_agreement within_026_039_agreement
private def within_026_041 : CountSegment 1 10 13120 3372 3374 1609 1609 := CountSegment.join atomic_3373 atomic_3374
private theorem within_026_041_agreement : RowAgreement 3372 3374 := RowAgreement.join atomic_3373_agreement atomic_3374_agreement
private def within_026_042 : CountSegment 1 10 13120 3374 3376 1609 1610 := CountSegment.join atomic_3375 atomic_3376
private theorem within_026_042_agreement : RowAgreement 3374 3376 := RowAgreement.join atomic_3375_agreement atomic_3376_agreement
private def within_026_043 : CountSegment 1 10 13120 3372 3376 1609 1610 := CountSegment.join within_026_041 within_026_042
private theorem within_026_043_agreement : RowAgreement 3372 3376 := RowAgreement.join within_026_041_agreement within_026_042_agreement
private def within_026_044 : CountSegment 1 10 13120 3368 3376 1608 1610 := CountSegment.join within_026_040 within_026_043
private theorem within_026_044_agreement : RowAgreement 3368 3376 := RowAgreement.join within_026_040_agreement within_026_043_agreement
private def within_026_045 : CountSegment 1 10 13120 3360 3376 1604 1610 := CountSegment.join within_026_037 within_026_044
private theorem within_026_045_agreement : RowAgreement 3360 3376 := RowAgreement.join within_026_037_agreement within_026_044_agreement
private def within_026_046 : CountSegment 1 10 13120 3376 3378 1610 1610 := CountSegment.join atomic_3377 atomic_3378
private theorem within_026_046_agreement : RowAgreement 3376 3378 := RowAgreement.join atomic_3377_agreement atomic_3378_agreement
private def within_026_047 : CountSegment 1 10 13120 3378 3380 1610 1610 := CountSegment.join atomic_3379 atomic_3380
private theorem within_026_047_agreement : RowAgreement 3378 3380 := RowAgreement.join atomic_3379_agreement atomic_3380_agreement
private def within_026_048 : CountSegment 1 10 13120 3376 3380 1610 1610 := CountSegment.join within_026_046 within_026_047
private theorem within_026_048_agreement : RowAgreement 3376 3380 := RowAgreement.join within_026_046_agreement within_026_047_agreement
private def within_026_049 : CountSegment 1 10 13120 3380 3382 1610 1612 := CountSegment.join atomic_3381 atomic_3382
private theorem within_026_049_agreement : RowAgreement 3380 3382 := RowAgreement.join atomic_3381_agreement atomic_3382_agreement
private def within_026_050 : CountSegment 1 10 13120 3382 3384 1612 1613 := CountSegment.join atomic_3383 atomic_3384
private theorem within_026_050_agreement : RowAgreement 3382 3384 := RowAgreement.join atomic_3383_agreement atomic_3384_agreement
private def within_026_051 : CountSegment 1 10 13120 3380 3384 1610 1613 := CountSegment.join within_026_049 within_026_050
private theorem within_026_051_agreement : RowAgreement 3380 3384 := RowAgreement.join within_026_049_agreement within_026_050_agreement
private def within_026_052 : CountSegment 1 10 13120 3376 3384 1610 1613 := CountSegment.join within_026_048 within_026_051
private theorem within_026_052_agreement : RowAgreement 3376 3384 := RowAgreement.join within_026_048_agreement within_026_051_agreement
private def within_026_053 : CountSegment 1 10 13120 3384 3386 1613 1613 := CountSegment.join atomic_3385 atomic_3386
private theorem within_026_053_agreement : RowAgreement 3384 3386 := RowAgreement.join atomic_3385_agreement atomic_3386_agreement
private def within_026_054 : CountSegment 1 10 13120 3386 3388 1613 1615 := CountSegment.join atomic_3387 atomic_3388
private theorem within_026_054_agreement : RowAgreement 3386 3388 := RowAgreement.join atomic_3387_agreement atomic_3388_agreement
private def within_026_055 : CountSegment 1 10 13120 3384 3388 1613 1615 := CountSegment.join within_026_053 within_026_054
private theorem within_026_055_agreement : RowAgreement 3384 3388 := RowAgreement.join within_026_053_agreement within_026_054_agreement
private def within_026_056 : CountSegment 1 10 13120 3388 3390 1615 1616 := CountSegment.join atomic_3389 atomic_3390
private theorem within_026_056_agreement : RowAgreement 3388 3390 := RowAgreement.join atomic_3389_agreement atomic_3390_agreement
private def within_026_057 : CountSegment 1 10 13120 3390 3392 1616 1617 := CountSegment.join atomic_3391 atomic_3392
private theorem within_026_057_agreement : RowAgreement 3390 3392 := RowAgreement.join atomic_3391_agreement atomic_3392_agreement
private def within_026_058 : CountSegment 1 10 13120 3388 3392 1615 1617 := CountSegment.join within_026_056 within_026_057
private theorem within_026_058_agreement : RowAgreement 3388 3392 := RowAgreement.join within_026_056_agreement within_026_057_agreement
private def within_026_059 : CountSegment 1 10 13120 3384 3392 1613 1617 := CountSegment.join within_026_055 within_026_058
private theorem within_026_059_agreement : RowAgreement 3384 3392 := RowAgreement.join within_026_055_agreement within_026_058_agreement
private def within_026_060 : CountSegment 1 10 13120 3376 3392 1610 1617 := CountSegment.join within_026_052 within_026_059
private theorem within_026_060_agreement : RowAgreement 3376 3392 := RowAgreement.join within_026_052_agreement within_026_059_agreement
private def within_026_061 : CountSegment 1 10 13120 3360 3392 1604 1617 := CountSegment.join within_026_045 within_026_060
private theorem within_026_061_agreement : RowAgreement 3360 3392 := RowAgreement.join within_026_045_agreement within_026_060_agreement
private def within_026_062 : CountSegment 1 10 13120 3328 3392 1595 1617 := CountSegment.join within_026_030 within_026_061
private theorem within_026_062_agreement : RowAgreement 3328 3392 := RowAgreement.join within_026_030_agreement within_026_061_agreement
private def within_026_063 : CountSegment 1 10 13120 3392 3394 1617 1619 := CountSegment.join atomic_3393 atomic_3394
private theorem within_026_063_agreement : RowAgreement 3392 3394 := RowAgreement.join atomic_3393_agreement atomic_3394_agreement
private def within_026_064 : CountSegment 1 10 13120 3394 3396 1619 1620 := CountSegment.join atomic_3395 atomic_3396
private theorem within_026_064_agreement : RowAgreement 3394 3396 := RowAgreement.join atomic_3395_agreement atomic_3396_agreement
private def within_026_065 : CountSegment 1 10 13120 3392 3396 1617 1620 := CountSegment.join within_026_063 within_026_064
private theorem within_026_065_agreement : RowAgreement 3392 3396 := RowAgreement.join within_026_063_agreement within_026_064_agreement
private def within_026_066 : CountSegment 1 10 13120 3396 3398 1620 1620 := CountSegment.join atomic_3397 atomic_3398
private theorem within_026_066_agreement : RowAgreement 3396 3398 := RowAgreement.join atomic_3397_agreement atomic_3398_agreement
private def within_026_067 : CountSegment 1 10 13120 3398 3400 1620 1622 := CountSegment.join atomic_3399 atomic_3400
private theorem within_026_067_agreement : RowAgreement 3398 3400 := RowAgreement.join atomic_3399_agreement atomic_3400_agreement
private def within_026_068 : CountSegment 1 10 13120 3396 3400 1620 1622 := CountSegment.join within_026_066 within_026_067
private theorem within_026_068_agreement : RowAgreement 3396 3400 := RowAgreement.join within_026_066_agreement within_026_067_agreement
private def within_026_069 : CountSegment 1 10 13120 3392 3400 1617 1622 := CountSegment.join within_026_065 within_026_068
private theorem within_026_069_agreement : RowAgreement 3392 3400 := RowAgreement.join within_026_065_agreement within_026_068_agreement
private def within_026_070 : CountSegment 1 10 13120 3400 3402 1622 1622 := CountSegment.join atomic_3401 atomic_3402
private theorem within_026_070_agreement : RowAgreement 3400 3402 := RowAgreement.join atomic_3401_agreement atomic_3402_agreement
private def within_026_071 : CountSegment 1 10 13120 3402 3404 1622 1624 := CountSegment.join atomic_3403 atomic_3404
private theorem within_026_071_agreement : RowAgreement 3402 3404 := RowAgreement.join atomic_3403_agreement atomic_3404_agreement
private def within_026_072 : CountSegment 1 10 13120 3400 3404 1622 1624 := CountSegment.join within_026_070 within_026_071
private theorem within_026_072_agreement : RowAgreement 3400 3404 := RowAgreement.join within_026_070_agreement within_026_071_agreement
private def within_026_073 : CountSegment 1 10 13120 3404 3406 1624 1626 := CountSegment.join atomic_3405 atomic_3406
private theorem within_026_073_agreement : RowAgreement 3404 3406 := RowAgreement.join atomic_3405_agreement atomic_3406_agreement
private def within_026_074 : CountSegment 1 10 13120 3406 3408 1626 1627 := CountSegment.join atomic_3407 atomic_3408
private theorem within_026_074_agreement : RowAgreement 3406 3408 := RowAgreement.join atomic_3407_agreement atomic_3408_agreement
private def within_026_075 : CountSegment 1 10 13120 3404 3408 1624 1627 := CountSegment.join within_026_073 within_026_074
private theorem within_026_075_agreement : RowAgreement 3404 3408 := RowAgreement.join within_026_073_agreement within_026_074_agreement
private def within_026_076 : CountSegment 1 10 13120 3400 3408 1622 1627 := CountSegment.join within_026_072 within_026_075
private theorem within_026_076_agreement : RowAgreement 3400 3408 := RowAgreement.join within_026_072_agreement within_026_075_agreement
private def within_026_077 : CountSegment 1 10 13120 3392 3408 1617 1627 := CountSegment.join within_026_069 within_026_076
private theorem within_026_077_agreement : RowAgreement 3392 3408 := RowAgreement.join within_026_069_agreement within_026_076_agreement
private def within_026_078 : CountSegment 1 10 13120 3408 3410 1627 1627 := CountSegment.join atomic_3409 atomic_3410
private theorem within_026_078_agreement : RowAgreement 3408 3410 := RowAgreement.join atomic_3409_agreement atomic_3410_agreement
private def within_026_079 : CountSegment 1 10 13120 3410 3412 1627 1628 := CountSegment.join atomic_3411 atomic_3412
private theorem within_026_079_agreement : RowAgreement 3410 3412 := RowAgreement.join atomic_3411_agreement atomic_3412_agreement
private def within_026_080 : CountSegment 1 10 13120 3408 3412 1627 1628 := CountSegment.join within_026_078 within_026_079
private theorem within_026_080_agreement : RowAgreement 3408 3412 := RowAgreement.join within_026_078_agreement within_026_079_agreement
private def within_026_081 : CountSegment 1 10 13120 3412 3414 1628 1629 := CountSegment.join atomic_3413 atomic_3414
private theorem within_026_081_agreement : RowAgreement 3412 3414 := RowAgreement.join atomic_3413_agreement atomic_3414_agreement
private def within_026_082 : CountSegment 1 10 13120 3414 3416 1629 1629 := CountSegment.join atomic_3415 atomic_3416
private theorem within_026_082_agreement : RowAgreement 3414 3416 := RowAgreement.join atomic_3415_agreement atomic_3416_agreement
private def within_026_083 : CountSegment 1 10 13120 3412 3416 1628 1629 := CountSegment.join within_026_081 within_026_082
private theorem within_026_083_agreement : RowAgreement 3412 3416 := RowAgreement.join within_026_081_agreement within_026_082_agreement
private def within_026_084 : CountSegment 1 10 13120 3408 3416 1627 1629 := CountSegment.join within_026_080 within_026_083
private theorem within_026_084_agreement : RowAgreement 3408 3416 := RowAgreement.join within_026_080_agreement within_026_083_agreement
private def within_026_085 : CountSegment 1 10 13120 3416 3418 1629 1630 := CountSegment.join atomic_3417 atomic_3418
private theorem within_026_085_agreement : RowAgreement 3416 3418 := RowAgreement.join atomic_3417_agreement atomic_3418_agreement
private def within_026_086 : CountSegment 1 10 13120 3418 3420 1630 1631 := CountSegment.join atomic_3419 atomic_3420
private theorem within_026_086_agreement : RowAgreement 3418 3420 := RowAgreement.join atomic_3419_agreement atomic_3420_agreement
private def within_026_087 : CountSegment 1 10 13120 3416 3420 1629 1631 := CountSegment.join within_026_085 within_026_086
private theorem within_026_087_agreement : RowAgreement 3416 3420 := RowAgreement.join within_026_085_agreement within_026_086_agreement
private def within_026_088 : CountSegment 1 10 13120 3420 3422 1631 1632 := CountSegment.join atomic_3421 atomic_3422
private theorem within_026_088_agreement : RowAgreement 3420 3422 := RowAgreement.join atomic_3421_agreement atomic_3422_agreement
private def within_026_089 : CountSegment 1 10 13120 3422 3424 1632 1632 := CountSegment.join atomic_3423 atomic_3424
private theorem within_026_089_agreement : RowAgreement 3422 3424 := RowAgreement.join atomic_3423_agreement atomic_3424_agreement
private def within_026_090 : CountSegment 1 10 13120 3420 3424 1631 1632 := CountSegment.join within_026_088 within_026_089
private theorem within_026_090_agreement : RowAgreement 3420 3424 := RowAgreement.join within_026_088_agreement within_026_089_agreement
private def within_026_091 : CountSegment 1 10 13120 3416 3424 1629 1632 := CountSegment.join within_026_087 within_026_090
private theorem within_026_091_agreement : RowAgreement 3416 3424 := RowAgreement.join within_026_087_agreement within_026_090_agreement
private def within_026_092 : CountSegment 1 10 13120 3408 3424 1627 1632 := CountSegment.join within_026_084 within_026_091
private theorem within_026_092_agreement : RowAgreement 3408 3424 := RowAgreement.join within_026_084_agreement within_026_091_agreement
private def within_026_093 : CountSegment 1 10 13120 3392 3424 1617 1632 := CountSegment.join within_026_077 within_026_092
private theorem within_026_093_agreement : RowAgreement 3392 3424 := RowAgreement.join within_026_077_agreement within_026_092_agreement
private def within_026_094 : CountSegment 1 10 13120 3424 3426 1632 1633 := CountSegment.join atomic_3425 atomic_3426
private theorem within_026_094_agreement : RowAgreement 3424 3426 := RowAgreement.join atomic_3425_agreement atomic_3426_agreement
private def within_026_095 : CountSegment 1 10 13120 3426 3428 1633 1634 := CountSegment.join atomic_3427 atomic_3428
private theorem within_026_095_agreement : RowAgreement 3426 3428 := RowAgreement.join atomic_3427_agreement atomic_3428_agreement
private def within_026_096 : CountSegment 1 10 13120 3424 3428 1632 1634 := CountSegment.join within_026_094 within_026_095
private theorem within_026_096_agreement : RowAgreement 3424 3428 := RowAgreement.join within_026_094_agreement within_026_095_agreement
private def within_026_097 : CountSegment 1 10 13120 3428 3430 1634 1634 := CountSegment.join atomic_3429 atomic_3430
private theorem within_026_097_agreement : RowAgreement 3428 3430 := RowAgreement.join atomic_3429_agreement atomic_3430_agreement
private def within_026_098 : CountSegment 1 10 13120 3430 3432 1634 1634 := CountSegment.join atomic_3431 atomic_3432
private theorem within_026_098_agreement : RowAgreement 3430 3432 := RowAgreement.join atomic_3431_agreement atomic_3432_agreement
private def within_026_099 : CountSegment 1 10 13120 3428 3432 1634 1634 := CountSegment.join within_026_097 within_026_098
private theorem within_026_099_agreement : RowAgreement 3428 3432 := RowAgreement.join within_026_097_agreement within_026_098_agreement
private def within_026_100 : CountSegment 1 10 13120 3424 3432 1632 1634 := CountSegment.join within_026_096 within_026_099
private theorem within_026_100_agreement : RowAgreement 3424 3432 := RowAgreement.join within_026_096_agreement within_026_099_agreement
private def within_026_101 : CountSegment 1 10 13120 3432 3434 1634 1635 := CountSegment.join atomic_3433 atomic_3434
private theorem within_026_101_agreement : RowAgreement 3432 3434 := RowAgreement.join atomic_3433_agreement atomic_3434_agreement
private def within_026_102 : CountSegment 1 10 13120 3434 3436 1635 1635 := CountSegment.join atomic_3435 atomic_3436
private theorem within_026_102_agreement : RowAgreement 3434 3436 := RowAgreement.join atomic_3435_agreement atomic_3436_agreement
private def within_026_103 : CountSegment 1 10 13120 3432 3436 1634 1635 := CountSegment.join within_026_101 within_026_102
private theorem within_026_103_agreement : RowAgreement 3432 3436 := RowAgreement.join within_026_101_agreement within_026_102_agreement
private def within_026_104 : CountSegment 1 10 13120 3436 3438 1635 1637 := CountSegment.join atomic_3437 atomic_3438
private theorem within_026_104_agreement : RowAgreement 3436 3438 := RowAgreement.join atomic_3437_agreement atomic_3438_agreement
private def within_026_105 : CountSegment 1 10 13120 3438 3440 1637 1637 := CountSegment.join atomic_3439 atomic_3440
private theorem within_026_105_agreement : RowAgreement 3438 3440 := RowAgreement.join atomic_3439_agreement atomic_3440_agreement
private def within_026_106 : CountSegment 1 10 13120 3436 3440 1635 1637 := CountSegment.join within_026_104 within_026_105
private theorem within_026_106_agreement : RowAgreement 3436 3440 := RowAgreement.join within_026_104_agreement within_026_105_agreement
private def within_026_107 : CountSegment 1 10 13120 3432 3440 1634 1637 := CountSegment.join within_026_103 within_026_106
private theorem within_026_107_agreement : RowAgreement 3432 3440 := RowAgreement.join within_026_103_agreement within_026_106_agreement
private def within_026_108 : CountSegment 1 10 13120 3424 3440 1632 1637 := CountSegment.join within_026_100 within_026_107
private theorem within_026_108_agreement : RowAgreement 3424 3440 := RowAgreement.join within_026_100_agreement within_026_107_agreement
private def within_026_109 : CountSegment 1 10 13120 3440 3442 1637 1640 := CountSegment.join atomic_3441 atomic_3442
private theorem within_026_109_agreement : RowAgreement 3440 3442 := RowAgreement.join atomic_3441_agreement atomic_3442_agreement
private def within_026_110 : CountSegment 1 10 13120 3442 3444 1640 1641 := CountSegment.join atomic_3443 atomic_3444
private theorem within_026_110_agreement : RowAgreement 3442 3444 := RowAgreement.join atomic_3443_agreement atomic_3444_agreement
private def within_026_111 : CountSegment 1 10 13120 3440 3444 1637 1641 := CountSegment.join within_026_109 within_026_110
private theorem within_026_111_agreement : RowAgreement 3440 3444 := RowAgreement.join within_026_109_agreement within_026_110_agreement
private def within_026_112 : CountSegment 1 10 13120 3444 3446 1641 1641 := CountSegment.join atomic_3445 atomic_3446
private theorem within_026_112_agreement : RowAgreement 3444 3446 := RowAgreement.join atomic_3445_agreement atomic_3446_agreement
private def within_026_113 : CountSegment 1 10 13120 3446 3448 1641 1641 := CountSegment.join atomic_3447 atomic_3448
private theorem within_026_113_agreement : RowAgreement 3446 3448 := RowAgreement.join atomic_3447_agreement atomic_3448_agreement
private def within_026_114 : CountSegment 1 10 13120 3444 3448 1641 1641 := CountSegment.join within_026_112 within_026_113
private theorem within_026_114_agreement : RowAgreement 3444 3448 := RowAgreement.join within_026_112_agreement within_026_113_agreement
private def within_026_115 : CountSegment 1 10 13120 3440 3448 1637 1641 := CountSegment.join within_026_111 within_026_114
private theorem within_026_115_agreement : RowAgreement 3440 3448 := RowAgreement.join within_026_111_agreement within_026_114_agreement
private def within_026_116 : CountSegment 1 10 13120 3448 3450 1641 1641 := CountSegment.join atomic_3449 atomic_3450
private theorem within_026_116_agreement : RowAgreement 3448 3450 := RowAgreement.join atomic_3449_agreement atomic_3450_agreement
private def within_026_117 : CountSegment 1 10 13120 3450 3452 1641 1641 := CountSegment.join atomic_3451 atomic_3452
private theorem within_026_117_agreement : RowAgreement 3450 3452 := RowAgreement.join atomic_3451_agreement atomic_3452_agreement
private def within_026_118 : CountSegment 1 10 13120 3448 3452 1641 1641 := CountSegment.join within_026_116 within_026_117
private theorem within_026_118_agreement : RowAgreement 3448 3452 := RowAgreement.join within_026_116_agreement within_026_117_agreement
private def within_026_119 : CountSegment 1 10 13120 3452 3454 1641 1642 := CountSegment.join atomic_3453 atomic_3454
private theorem within_026_119_agreement : RowAgreement 3452 3454 := RowAgreement.join atomic_3453_agreement atomic_3454_agreement
private def within_026_120 : CountSegment 1 10 13120 3454 3456 1642 1644 := CountSegment.join atomic_3455 atomic_3456
private theorem within_026_120_agreement : RowAgreement 3454 3456 := RowAgreement.join atomic_3455_agreement atomic_3456_agreement
private def within_026_121 : CountSegment 1 10 13120 3452 3456 1641 1644 := CountSegment.join within_026_119 within_026_120
private theorem within_026_121_agreement : RowAgreement 3452 3456 := RowAgreement.join within_026_119_agreement within_026_120_agreement
private def within_026_122 : CountSegment 1 10 13120 3448 3456 1641 1644 := CountSegment.join within_026_118 within_026_121
private theorem within_026_122_agreement : RowAgreement 3448 3456 := RowAgreement.join within_026_118_agreement within_026_121_agreement
private def within_026_123 : CountSegment 1 10 13120 3440 3456 1637 1644 := CountSegment.join within_026_115 within_026_122
private theorem within_026_123_agreement : RowAgreement 3440 3456 := RowAgreement.join within_026_115_agreement within_026_122_agreement
private def within_026_124 : CountSegment 1 10 13120 3424 3456 1632 1644 := CountSegment.join within_026_108 within_026_123
private theorem within_026_124_agreement : RowAgreement 3424 3456 := RowAgreement.join within_026_108_agreement within_026_123_agreement
private def within_026_125 : CountSegment 1 10 13120 3392 3456 1617 1644 := CountSegment.join within_026_093 within_026_124
private theorem within_026_125_agreement : RowAgreement 3392 3456 := RowAgreement.join within_026_093_agreement within_026_124_agreement
private def within_026_126 : CountSegment 1 10 13120 3328 3456 1595 1644 := CountSegment.join within_026_062 within_026_125
private theorem within_026_126_agreement : RowAgreement 3328 3456 := RowAgreement.join within_026_062_agreement within_026_125_agreement
def segment_026 : CountSegment 1 10 13120 3328 3456 1595 1644 := within_026_126
theorem agreement_026 : RowAgreement 3328 3456 := within_026_126_agreement
end ProjectEulerSolutions.P58Numerical
