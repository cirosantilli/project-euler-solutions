import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_4353 : RowCertificate 4353 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4353 1) := by change ¬ Nat.Prime 75803143; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5831011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4353 2) := by change Nat.Prime 75794437; exact ProjectEulerSolutions.P58PrimeNodes.p_75794437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4353 3) := by change ¬ Nat.Prime 75785731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10826533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4353 : row 4353 = 1 := (row_4353.checked).trans rfl
theorem model_count_4353 : modelRow 4353 = 1 := by decide
theorem modelrow_agrees_4353 : modelRow 4353 = row 4353 := model_count_4353.trans row_count_4353.symm
def row_4354 : RowCertificate 4354 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4354 1) := by change ¬ Nat.Prime 75837973; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 736291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4354 2) := by change ¬ Nat.Prime 75829265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15165853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4354 3) := by change ¬ Nat.Prime 75820557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25273519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4354 : row 4354 = 0 := (row_4354.checked).trans rfl
theorem model_count_4354 : modelRow 4354 = 0 := by decide
theorem modelrow_agrees_4354 : modelRow 4354 = row 4354 := model_count_4354.trans row_count_4354.symm
def row_4355 : RowCertificate 4355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4355 1) := by change ¬ Nat.Prime 75872811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25290937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4355 2) := by change ¬ Nat.Prime 75864101; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 325597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4355 3) := by change ¬ Nat.Prime 75855391; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3992389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4355 : row 4355 = 0 := (row_4355.checked).trans rfl
theorem model_count_4355 : modelRow 4355 = 0 := by decide
theorem modelrow_agrees_4355 : modelRow 4355 = row 4355 := model_count_4355.trans row_count_4355.symm
def row_4356 : RowCertificate 4356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4356 1) := by change ¬ Nat.Prime 75907657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10843951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4356 2) := by change ¬ Nat.Prime 75898945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15179789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4356 3) := by change ¬ Nat.Prime 75890233; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 138739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4356 : row 4356 = 0 := (row_4356.checked).trans rfl
theorem model_count_4356 : modelRow 4356 = 0 := by decide
theorem modelrow_agrees_4356 : modelRow 4356 = row 4356 := model_count_4356.trans row_count_4356.symm
def row_4357 : RowCertificate 4357 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4357 1) := by change ¬ Nat.Prime 75942511; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 546349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4357 2) := by change ¬ Nat.Prime 75933797; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1040189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4357 3) := by change ¬ Nat.Prime 75925083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25308361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4357 : row 4357 = 0 := (row_4357.checked).trans rfl
theorem model_count_4357 : modelRow 4357 = 0 := by decide
theorem modelrow_agrees_4357 : modelRow 4357 = row 4357 := model_count_4357.trans row_count_4357.symm
def row_4358 : RowCertificate 4358 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4358 1) := by change ¬ Nat.Prime 75977373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25325791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4358 2) := by change ¬ Nat.Prime 75968657; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 672289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4358 3) := by change Nat.Prime 75959941; exact ProjectEulerSolutions.P58PrimeNodes.p_75959941.prime
    exact decide_eq_true prime
theorem row_count_4358 : row 4358 = 1 := (row_4358.checked).trans rfl
theorem model_count_4358 : modelRow 4358 = 1 := by decide
theorem modelrow_agrees_4358 : modelRow 4358 = row 4358 := model_count_4358.trans row_count_4358.symm
def row_4359 : RowCertificate 4359 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4359 1) := by change Nat.Prime 76012243; exact ProjectEulerSolutions.P58PrimeNodes.p_76012243.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4359 2) := by change ¬ Nat.Prime 76003525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15200705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4359 3) := by change ¬ Nat.Prime 75994807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10856401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4359 : row 4359 = 1 := (row_4359.checked).trans rfl
theorem model_count_4359 : modelRow 4359 = 1 := by decide
theorem modelrow_agrees_4359 : modelRow 4359 = row 4359 := model_count_4359.trans row_count_4359.symm
def row_4360 : RowCertificate 4360 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4360 1) := by change Nat.Prime 76047121; exact ProjectEulerSolutions.P58PrimeNodes.p_76047121.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4360 2) := by change Nat.Prime 76038401; exact ProjectEulerSolutions.P58PrimeNodes.p_76038401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4360 3) := by change ¬ Nat.Prime 76029681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25343227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4360 : row 4360 = 2 := (row_4360.checked).trans rfl
theorem model_count_4360 : modelRow 4360 = 2 := by decide
theorem modelrow_agrees_4360 : modelRow 4360 = row 4360 := model_count_4360.trans row_count_4360.symm
def row_4361 : RowCertificate 4361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4361 1) := by change ¬ Nat.Prime 76082007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25360669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4361 2) := by change ¬ Nat.Prime 76073285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15214657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4361 3) := by change ¬ Nat.Prime 76064563; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2055799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4361 : row 4361 = 0 := (row_4361.checked).trans rfl
theorem model_count_4361 : modelRow 4361 = 0 := by decide
theorem modelrow_agrees_4361 : modelRow 4361 = row 4361 := model_count_4361.trans row_count_4361.symm
def row_4362 : RowCertificate 4362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4362 1) := by change ¬ Nat.Prime 76116901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10873843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4362 2) := by change ¬ Nat.Prime 76108177; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1856297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4362 3) := by change ¬ Nat.Prime 76099453; exact PrattCertificates.composite_of_product (factor := 3469) (cofactor := 21937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4362 : row 4362 = 0 := (row_4362.checked).trans rfl
theorem model_count_4362 : modelRow 4362 = 0 := by decide
theorem modelrow_agrees_4362 : modelRow 4362 = row 4362 := model_count_4362.trans row_count_4362.symm
def row_4363 : RowCertificate 4363 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4363 1) := by change ¬ Nat.Prime 76151803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10878829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4363 2) := by change ¬ Nat.Prime 76143077; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2057921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4363 3) := by change ¬ Nat.Prime 76134351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25378117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4363 : row 4363 = 0 := (row_4363.checked).trans rfl
theorem model_count_4363 : modelRow 4363 = 0 := by decide
theorem modelrow_agrees_4363 : modelRow 4363 = row 4363 := model_count_4363.trans row_count_4363.symm
def row_4364 : RowCertificate 4364 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4364 1) := by change ¬ Nat.Prime 76186713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25395571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4364 2) := by change ¬ Nat.Prime 76177985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15235597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4364 3) := by change Nat.Prime 76169257; exact ProjectEulerSolutions.P58PrimeNodes.p_76169257.prime
    exact decide_eq_true prime
theorem row_count_4364 : row 4364 = 1 := (row_4364.checked).trans rfl
theorem model_count_4364 : modelRow 4364 = 1 := by decide
theorem modelrow_agrees_4364 : modelRow 4364 = row 4364 := model_count_4364.trans row_count_4364.symm
def row_4365 : RowCertificate 4365 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4365 1) := by change Nat.Prime 76221631; exact ProjectEulerSolutions.P58PrimeNodes.p_76221631.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4365 2) := by change ¬ Nat.Prime 76212901; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 440537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4365 3) := by change ¬ Nat.Prime 76204171; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 110281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4365 : row 4365 = 1 := (row_4365.checked).trans rfl
theorem model_count_4365 : modelRow 4365 = 1 := by decide
theorem modelrow_agrees_4365 : modelRow 4365 = row 4365 := model_count_4365.trans row_count_4365.symm
def row_4366 : RowCertificate 4366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4366 1) := by change ¬ Nat.Prime 76256557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5865889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4366 2) := by change ¬ Nat.Prime 76247825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15249565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4366 3) := by change ¬ Nat.Prime 76239093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25413031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4366 : row 4366 = 0 := (row_4366.checked).trans rfl
theorem model_count_4366 : modelRow 4366 = 0 := by decide
theorem modelrow_agrees_4366 : modelRow 4366 = row 4366 := model_count_4366.trans row_count_4366.symm
def row_4367 : RowCertificate 4367 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4367 1) := by change ¬ Nat.Prime 76291491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25430497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4367 2) := by change ¬ Nat.Prime 76282757; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4487221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4367 3) := by change ¬ Nat.Prime 76274023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10896289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4367 : row 4367 = 0 := (row_4367.checked).trans rfl
theorem model_count_4367 : modelRow 4367 = 0 := by decide
theorem modelrow_agrees_4367 : modelRow 4367 = row 4367 := model_count_4367.trans row_count_4367.symm
def row_4368 : RowCertificate 4368 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4368 1) := by change ¬ Nat.Prime 76326433; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2462143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4368 2) := by change ¬ Nat.Prime 76317697; exact PrattCertificates.composite_of_product (factor := 821) (cofactor := 92957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4368 3) := by change ¬ Nat.Prime 76308961; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1774627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4368 : row 4368 = 0 := (row_4368.checked).trans rfl
theorem model_count_4368 : modelRow 4368 = 0 := by decide
theorem modelrow_agrees_4368 : modelRow 4368 = row 4368 := model_count_4368.trans row_count_4368.symm
def row_4369 : RowCertificate 4369 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4369 1) := by change ¬ Nat.Prime 76361383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10908769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4369 2) := by change ¬ Nat.Prime 76352645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15270529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4369 3) := by change ¬ Nat.Prime 76343907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25447969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4369 : row 4369 = 0 := (row_4369.checked).trans rfl
theorem model_count_4369 : modelRow 4369 = 0 := by decide
theorem modelrow_agrees_4369 : modelRow 4369 = row 4369 := model_count_4369.trans row_count_4369.symm
def row_4370 : RowCertificate 4370 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4370 1) := by change ¬ Nat.Prime 76396341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25465447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4370 2) := by change ¬ Nat.Prime 76387601; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 316961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4370 3) := by change ¬ Nat.Prime 76378861; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5875297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4370 : row 4370 = 0 := (row_4370.checked).trans rfl
theorem model_count_4370 : modelRow 4370 = 0 := by decide
theorem modelrow_agrees_4370 : modelRow 4370 = row 4370 := model_count_4370.trans row_count_4370.symm
def row_4371 : RowCertificate 4371 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4371 1) := by change ¬ Nat.Prime 76431307; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2065711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4371 2) := by change ¬ Nat.Prime 76422565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15284513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4371 3) := by change Nat.Prime 76413823; exact ProjectEulerSolutions.P58PrimeNodes.p_76413823.prime
    exact decide_eq_true prime
theorem row_count_4371 : row 4371 = 1 := (row_4371.checked).trans rfl
theorem model_count_4371 : modelRow 4371 = 1 := by decide
theorem modelrow_agrees_4371 : modelRow 4371 = row 4371 := model_count_4371.trans row_count_4371.symm
def row_4372 : RowCertificate 4372 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4372 1) := by change Nat.Prime 76466281; exact ProjectEulerSolutions.P58PrimeNodes.p_76466281.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4372 2) := by change ¬ Nat.Prime 76457537; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5881349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4372 3) := by change ¬ Nat.Prime 76448793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25482931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4372 : row 4372 = 1 := (row_4372.checked).trans rfl
theorem model_count_4372 : modelRow 4372 = 1 := by decide
theorem modelrow_agrees_4372 : modelRow 4372 = row 4372 := model_count_4372.trans row_count_4372.symm
def row_4373 : RowCertificate 4373 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4373 1) := by change ¬ Nat.Prime 76501263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25500421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4373 2) := by change ¬ Nat.Prime 76492517; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2637673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4373 3) := by change ¬ Nat.Prime 76483771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10926253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4373 : row 4373 = 0 := (row_4373.checked).trans rfl
theorem model_count_4373 : modelRow 4373 = 0 := by decide
theorem modelrow_agrees_4373 : modelRow 4373 = row 4373 := model_count_4373.trans row_count_4373.symm
def row_4374 : RowCertificate 4374 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4374 1) := by change Nat.Prime 76536253; exact ProjectEulerSolutions.P58PrimeNodes.p_76536253.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4374 2) := by change ¬ Nat.Prime 76527505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15305501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4374 3) := by change ¬ Nat.Prime 76518757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10931251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4374 : row 4374 = 1 := (row_4374.checked).trans rfl
theorem model_count_4374 : modelRow 4374 = 1 := by decide
theorem modelrow_agrees_4374 : modelRow 4374 = row 4374 := model_count_4374.trans row_count_4374.symm
def row_4375 : RowCertificate 4375 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4375 1) := by change ¬ Nat.Prime 76571251; exact PrattCertificates.composite_of_product (factor := 853) (cofactor := 89767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4375 2) := by change Nat.Prime 76562501; exact ProjectEulerSolutions.P58PrimeNodes.p_76562501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4375 3) := by change ¬ Nat.Prime 76553751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25517917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4375 : row 4375 = 1 := (row_4375.checked).trans rfl
theorem model_count_4375 : modelRow 4375 = 1 := by decide
theorem modelrow_agrees_4375 : modelRow 4375 = row 4375 := model_count_4375.trans row_count_4375.symm
def row_4376 : RowCertificate 4376 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4376 1) := by change ¬ Nat.Prime 76606257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25535419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4376 2) := by change ¬ Nat.Prime 76597505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15319501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4376 3) := by change ¬ Nat.Prime 76588753; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4030987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4376 : row 4376 = 0 := (row_4376.checked).trans rfl
theorem model_count_4376 : modelRow 4376 = 0 := by decide
theorem modelrow_agrees_4376 : modelRow 4376 = row 4376 := model_count_4376.trans row_count_4376.symm
def row_4377 : RowCertificate 4377 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4377 1) := by change ¬ Nat.Prime 76641271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10948753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4377 2) := by change ¬ Nat.Prime 76632517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5894809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4377 3) := by change Nat.Prime 76623763; exact ProjectEulerSolutions.P58PrimeNodes.p_76623763.prime
    exact decide_eq_true prime
theorem row_count_4377 : row 4377 = 1 := (row_4377.checked).trans rfl
theorem model_count_4377 : modelRow 4377 = 1 := by decide
theorem modelrow_agrees_4377 : modelRow 4377 = row 4377 := model_count_4377.trans row_count_4377.symm
def row_4378 : RowCertificate 4378 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4378 1) := by change Nat.Prime 76676293; exact ProjectEulerSolutions.P58PrimeNodes.p_76676293.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4378 2) := by change ¬ Nat.Prime 76667537; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 861433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4378 3) := by change ¬ Nat.Prime 76658781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25552927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4378 : row 4378 = 1 := (row_4378.checked).trans rfl
theorem model_count_4378 : modelRow 4378 = 1 := by decide
theorem modelrow_agrees_4378 : modelRow 4378 = row 4378 := model_count_4378.trans row_count_4378.symm
def row_4379 : RowCertificate 4379 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4379 1) := by change ¬ Nat.Prime 76711323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25570441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4379 2) := by change ¬ Nat.Prime 76702565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15340513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4379 3) := by change Nat.Prime 76693807; exact ProjectEulerSolutions.P58PrimeNodes.p_76693807.prime
    exact decide_eq_true prime
theorem row_count_4379 : row 4379 = 1 := (row_4379.checked).trans rfl
theorem model_count_4379 : modelRow 4379 = 1 := by decide
theorem modelrow_agrees_4379 : modelRow 4379 = row 4379 := model_count_4379.trans row_count_4379.symm
def row_4380 : RowCertificate 4380 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4380 1) := by change Nat.Prime 76746361; exact ProjectEulerSolutions.P58PrimeNodes.p_76746361.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4380 2) := by change Nat.Prime 76737601; exact ProjectEulerSolutions.P58PrimeNodes.p_76737601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4380 3) := by change ¬ Nat.Prime 76728841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10961263) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4380 : row 4380 = 2 := (row_4380.checked).trans rfl
theorem model_count_4380 : modelRow 4380 = 2 := by decide
theorem modelrow_agrees_4380 : modelRow 4380 = row 4380 := model_count_4380.trans row_count_4380.symm
def row_4381 : RowCertificate 4381 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4381 1) := by change Nat.Prime 76781407; exact ProjectEulerSolutions.P58PrimeNodes.p_76781407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4381 2) := by change ¬ Nat.Prime 76772645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15354529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4381 3) := by change ¬ Nat.Prime 76763883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25587961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4381 : row 4381 = 1 := (row_4381.checked).trans rfl
theorem model_count_4381 : modelRow 4381 = 1 := by decide
theorem modelrow_agrees_4381 : modelRow 4381 = row 4381 := model_count_4381.trans row_count_4381.symm
def row_4382 : RowCertificate 4382 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4382 1) := by change ¬ Nat.Prime 76816461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25605487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4382 2) := by change ¬ Nat.Prime 76807697; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 489221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4382 3) := by change ¬ Nat.Prime 76798933; exact PrattCertificates.composite_of_product (factor := 6907) (cofactor := 11119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4382 : row 4382 = 0 := (row_4382.checked).trans rfl
theorem model_count_4382 : modelRow 4382 = 0 := by decide
theorem modelrow_agrees_4382 : modelRow 4382 = row 4382 := model_count_4382.trans row_count_4382.symm
def row_4383 : RowCertificate 4383 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4383 1) := by change ¬ Nat.Prime 76851523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10978789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4383 2) := by change ¬ Nat.Prime 76842757; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 398149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4383 3) := by change ¬ Nat.Prime 76833991; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5910307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4383 : row 4383 = 0 := (row_4383.checked).trans rfl
theorem model_count_4383 : modelRow 4383 = 0 := by decide
theorem modelrow_agrees_4383 : modelRow 4383 = row 4383 := model_count_4383.trans row_count_4383.symm
def row_4384 : RowCertificate 4384 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4384 1) := by change ¬ Nat.Prime 76886593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10983799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4384 2) := by change ¬ Nat.Prime 76877825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15375565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4384 3) := by change ¬ Nat.Prime 76869057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25623019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4384 : row 4384 = 0 := (row_4384.checked).trans rfl
theorem model_count_4384 : modelRow 4384 = 0 := by decide
theorem modelrow_agrees_4384 : modelRow 4384 = row 4384 := model_count_4384.trans row_count_4384.symm
def row_4385 : RowCertificate 4385 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4385 1) := by change ¬ Nat.Prime 76921671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25640557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4385 2) := by change ¬ Nat.Prime 76912901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5916377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4385 3) := by change ¬ Nat.Prime 76904131; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 398467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4385 : row 4385 = 0 := (row_4385.checked).trans rfl
theorem model_count_4385 : modelRow 4385 = 0 := by decide
theorem modelrow_agrees_4385 : modelRow 4385 = row 4385 := model_count_4385.trans row_count_4385.symm
def row_4386 : RowCertificate 4386 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4386 1) := by change ¬ Nat.Prime 76956757; exact PrattCertificates.composite_of_product (factor := 3229) (cofactor := 23833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4386 2) := by change ¬ Nat.Prime 76947985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15389597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4386 3) := by change ¬ Nat.Prime 76939213; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5918401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4386 : row 4386 = 0 := (row_4386.checked).trans rfl
theorem model_count_4386 : modelRow 4386 = 0 := by decide
theorem modelrow_agrees_4386 : modelRow 4386 = row 4386 := model_count_4386.trans row_count_4386.symm
def row_4387 : RowCertificate 4387 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4387 1) := by change Nat.Prime 76991851; exact ProjectEulerSolutions.P58PrimeNodes.p_76991851.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4387 2) := by change Nat.Prime 76983077; exact ProjectEulerSolutions.P58PrimeNodes.p_76983077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4387 3) := by change ¬ Nat.Prime 76974303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25658101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4387 : row 4387 = 2 := (row_4387.checked).trans rfl
theorem model_count_4387 : modelRow 4387 = 2 := by decide
theorem modelrow_agrees_4387 : modelRow 4387 = row 4387 := model_count_4387.trans row_count_4387.symm
def row_4388 : RowCertificate 4388 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4388 1) := by change ¬ Nat.Prime 77026953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25675651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4388 2) := by change ¬ Nat.Prime 77018177; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4530481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4388 3) := by change ¬ Nat.Prime 77009401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11001343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4388 : row 4388 = 0 := (row_4388.checked).trans rfl
theorem model_count_4388 : modelRow 4388 = 0 := by decide
theorem modelrow_agrees_4388 : modelRow 4388 = row 4388 := model_count_4388.trans row_count_4388.symm
def row_4389 : RowCertificate 4389 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4389 1) := by change ¬ Nat.Prime 77062063; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5927851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4389 2) := by change ¬ Nat.Prime 77053285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15410657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4389 3) := by change Nat.Prime 77044507; exact ProjectEulerSolutions.P58PrimeNodes.p_77044507.prime
    exact decide_eq_true prime
theorem row_count_4389 : row 4389 = 1 := (row_4389.checked).trans rfl
theorem model_count_4389 : modelRow 4389 = 1 := by decide
theorem modelrow_agrees_4389 : modelRow 4389 = row 4389 := model_count_4389.trans row_count_4389.symm
def row_4390 : RowCertificate 4390 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4390 1) := by change ¬ Nat.Prime 77097181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11013883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4390 2) := by change ¬ Nat.Prime 77088401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5929877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4390 3) := by change ¬ Nat.Prime 77079621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25693207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4390 : row 4390 = 0 := (row_4390.checked).trans rfl
theorem model_count_4390 : modelRow 4390 = 0 := by decide
theorem modelrow_agrees_4390 : modelRow 4390 = row 4390 := model_count_4390.trans row_count_4390.symm
def row_4391 : RowCertificate 4391 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4391 1) := by change ¬ Nat.Prime 77132307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25710769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4391 2) := by change ¬ Nat.Prime 77123525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15424705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4391 3) := by change ¬ Nat.Prime 77114743; exact PrattCertificates.composite_of_product (factor := 2083) (cofactor := 37021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4391 : row 4391 = 0 := (row_4391.checked).trans rfl
theorem model_count_4391 : modelRow 4391 = 0 := by decide
theorem modelrow_agrees_4391 : modelRow 4391 = row 4391 := model_count_4391.trans row_count_4391.symm
def row_4392 : RowCertificate 4392 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4392 1) := by change ¬ Nat.Prime 77167441; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5935957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4392 2) := by change Nat.Prime 77158657; exact ProjectEulerSolutions.P58PrimeNodes.p_77158657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4392 3) := by change ¬ Nat.Prime 77149873; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 707797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4392 : row 4392 = 1 := (row_4392.checked).trans rfl
theorem model_count_4392 : modelRow 4392 = 1 := by decide
theorem modelrow_agrees_4392 : modelRow 4392 = row 4392 := model_count_4392.trans row_count_4392.symm
def row_4393 : RowCertificate 4393 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4393 1) := by change ¬ Nat.Prime 77202583; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 32479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4393 2) := by change Nat.Prime 77193797; exact ProjectEulerSolutions.P58PrimeNodes.p_77193797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4393 3) := by change ¬ Nat.Prime 77185011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25728337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4393 : row 4393 = 1 := (row_4393.checked).trans rfl
theorem model_count_4393 : modelRow 4393 = 1 := by decide
theorem modelrow_agrees_4393 : modelRow 4393 = row 4393 := model_count_4393.trans row_count_4393.symm
def row_4394 : RowCertificate 4394 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4394 1) := by change ¬ Nat.Prime 77237733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25745911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4394 2) := by change ¬ Nat.Prime 77228945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15445789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4394 3) := by change ¬ Nat.Prime 77220157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11031451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4394 : row 4394 = 0 := (row_4394.checked).trans rfl
theorem model_count_4394 : modelRow 4394 = 0 := by decide
theorem modelrow_agrees_4394 : modelRow 4394 = row 4394 := model_count_4394.trans row_count_4394.symm
def row_4395 : RowCertificate 4395 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4395 1) := by change ¬ Nat.Prime 77272891; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 511741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4395 2) := by change Nat.Prime 77264101; exact ProjectEulerSolutions.P58PrimeNodes.p_77264101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4395 3) := by change ¬ Nat.Prime 77255311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11036473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4395 : row 4395 = 1 := (row_4395.checked).trans rfl
theorem model_count_4395 : modelRow 4395 = 1 := by decide
theorem modelrow_agrees_4395 : modelRow 4395 = row 4395 := model_count_4395.trans row_count_4395.symm
def row_4396 : RowCertificate 4396 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4396 1) := by change ¬ Nat.Prime 77308057; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 978583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4396 2) := by change ¬ Nat.Prime 77299265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15459853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4396 3) := by change ¬ Nat.Prime 77290473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25763491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4396 : row 4396 = 0 := (row_4396.checked).trans rfl
theorem model_count_4396 : modelRow 4396 = 0 := by decide
theorem modelrow_agrees_4396 : modelRow 4396 = row 4396 := model_count_4396.trans row_count_4396.symm
def row_4397 : RowCertificate 4397 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4397 1) := by change ¬ Nat.Prime 77343231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25781077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4397 2) := by change ¬ Nat.Prime 77334437; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 138841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4397 3) := by change ¬ Nat.Prime 77325643; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 400651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4397 : row 4397 = 0 := (row_4397.checked).trans rfl
theorem model_count_4397 : modelRow 4397 = 0 := by decide
theorem modelrow_agrees_4397 : modelRow 4397 = row 4397 := model_count_4397.trans row_count_4397.symm
def row_4398 : RowCertificate 4398 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4398 1) := by change ¬ Nat.Prime 77378413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11054059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4398 2) := by change ¬ Nat.Prime 77369617; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5951509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4398 3) := by change ¬ Nat.Prime 77360821; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2090833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4398 : row 4398 = 0 := (row_4398.checked).trans rfl
theorem model_count_4398 : modelRow 4398 = 0 := by decide
theorem modelrow_agrees_4398 : modelRow 4398 = row 4398 := model_count_4398.trans row_count_4398.symm
def row_4399 : RowCertificate 4399 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4399 1) := by change ¬ Nat.Prime 77413603; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2497213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4399 2) := by change ¬ Nat.Prime 77404805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15480961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4399 3) := by change ¬ Nat.Prime 77396007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25798669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4399 : row 4399 = 0 := (row_4399.checked).trans rfl
theorem model_count_4399 : modelRow 4399 = 0 := by decide
theorem modelrow_agrees_4399 : modelRow 4399 = row 4399 := model_count_4399.trans row_count_4399.symm
def row_4400 : RowCertificate 4400 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4400 1) := by change ¬ Nat.Prime 77448801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25816267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4400 2) := by change ¬ Nat.Prime 77440001; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2092973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4400 3) := by change ¬ Nat.Prime 77431201; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 557059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4400 : row 4400 = 0 := (row_4400.checked).trans rfl
theorem model_count_4400 : modelRow 4400 = 0 := by decide
theorem modelrow_agrees_4400 : modelRow 4400 = row 4400 := model_count_4400.trans row_count_4400.symm
def row_4401 : RowCertificate 4401 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4401 1) := by change Nat.Prime 77484007; exact ProjectEulerSolutions.P58PrimeNodes.p_77484007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4401 2) := by change ¬ Nat.Prime 77475205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15495041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4401 3) := by change ¬ Nat.Prime 77466403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11066629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4401 : row 4401 = 1 := (row_4401.checked).trans rfl
theorem model_count_4401 : modelRow 4401 = 1 := by decide
theorem modelrow_agrees_4401 : modelRow 4401 = row 4401 := model_count_4401.trans row_count_4401.symm
def row_4402 : RowCertificate 4402 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4402 1) := by change ¬ Nat.Prime 77519221; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5963017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4402 2) := by change ¬ Nat.Prime 77510417; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2672773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4402 3) := by change ¬ Nat.Prime 77501613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25833871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4402 : row 4402 = 0 := (row_4402.checked).trans rfl
theorem model_count_4402 : modelRow 4402 = 0 := by decide
theorem modelrow_agrees_4402 : modelRow 4402 = row 4402 := model_count_4402.trans row_count_4402.symm
def row_4403 : RowCertificate 4403 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4403 1) := by change ¬ Nat.Prime 77554443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25851481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4403 2) := by change ¬ Nat.Prime 77545637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5965049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4403 3) := by change ¬ Nat.Prime 77536831; exact PrattCertificates.composite_of_product (factor := 991) (cofactor := 78241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4403 : row 4403 = 0 := (row_4403.checked).trans rfl
theorem model_count_4403 : modelRow 4403 = 0 := by decide
theorem modelrow_agrees_4403 : modelRow 4403 = row 4403 := model_count_4403.trans row_count_4403.symm
def row_4404 : RowCertificate 4404 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4404 1) := by change ¬ Nat.Prime 77589673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11084239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4404 2) := by change ¬ Nat.Prime 77580865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15516173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4404 3) := by change Nat.Prime 77572057; exact ProjectEulerSolutions.P58PrimeNodes.p_77572057.prime
    exact decide_eq_true prime
theorem row_count_4404 : row 4404 = 1 := (row_4404.checked).trans rfl
theorem model_count_4404 : modelRow 4404 = 1 := by decide
theorem modelrow_agrees_4404 : modelRow 4404 = row 4404 := model_count_4404.trans row_count_4404.symm
def row_4405 : RowCertificate 4405 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4405 1) := by change ¬ Nat.Prime 77624911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11089273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4405 2) := by change ¬ Nat.Prime 77616101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4565653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4405 3) := by change ¬ Nat.Prime 77607291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25869097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4405 : row 4405 = 0 := (row_4405.checked).trans rfl
theorem model_count_4405 : modelRow 4405 = 0 := by decide
theorem modelrow_agrees_4405 : modelRow 4405 = row 4405 := model_count_4405.trans row_count_4405.symm
def row_4406 : RowCertificate 4406 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4406 1) := by change ¬ Nat.Prime 77660157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25886719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4406 2) := by change ¬ Nat.Prime 77651345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15530269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4406 3) := by change ¬ Nat.Prime 77642533; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 753811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4406 : row 4406 = 0 := (row_4406.checked).trans rfl
theorem model_count_4406 : modelRow 4406 = 0 := by decide
theorem modelrow_agrees_4406 : modelRow 4406 = row 4406 := model_count_4406.trans row_count_4406.symm
def row_4407 : RowCertificate 4407 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4407 1) := by change ¬ Nat.Prime 77695411; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1159633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4407 2) := by change ¬ Nat.Prime 77686597; exact PrattCertificates.composite_of_product (factor := 5309) (cofactor := 14633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4407 3) := by change Nat.Prime 77677783; exact ProjectEulerSolutions.P58PrimeNodes.p_77677783.prime
    exact decide_eq_true prime
theorem row_count_4407 : row 4407 = 1 := (row_4407.checked).trans rfl
theorem model_count_4407 : modelRow 4407 = 1 := by decide
theorem modelrow_agrees_4407 : modelRow 4407 = row 4407 := model_count_4407.trans row_count_4407.symm
def row_4408 : RowCertificate 4408 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4408 1) := by change ¬ Nat.Prime 77730673; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2100829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4408 2) := by change Nat.Prime 77721857; exact ProjectEulerSolutions.P58PrimeNodes.p_77721857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4408 3) := by change ¬ Nat.Prime 77713041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25904347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4408 : row 4408 = 1 := (row_4408.checked).trans rfl
theorem model_count_4408 : modelRow 4408 = 1 := by decide
theorem modelrow_agrees_4408 : modelRow 4408 = row 4408 := model_count_4408.trans row_count_4408.symm
def row_4409 : RowCertificate 4409 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4409 1) := by change ¬ Nat.Prime 77765943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25921981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4409 2) := by change ¬ Nat.Prime 77757125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15551425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4409 3) := by change ¬ Nat.Prime 77748307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11106901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4409 : row 4409 = 0 := (row_4409.checked).trans rfl
theorem model_count_4409 : modelRow 4409 = 0 := by decide
theorem modelrow_agrees_4409 : modelRow 4409 = row 4409 := model_count_4409.trans row_count_4409.symm
def row_4410 : RowCertificate 4410 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4410 1) := by change ¬ Nat.Prime 77801221; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 429841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4410 2) := by change ¬ Nat.Prime 77792401; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 495493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4410 3) := by change ¬ Nat.Prime 77783581; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 101149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4410 : row 4410 = 0 := (row_4410.checked).trans rfl
theorem model_count_4410 : modelRow 4410 = 0 := by decide
theorem modelrow_agrees_4410 : modelRow 4410 = row 4410 := model_count_4410.trans row_count_4410.symm
def row_4411 : RowCertificate 4411 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4411 1) := by change ¬ Nat.Prime 77836507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11119501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4411 2) := by change ¬ Nat.Prime 77827685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15565537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4411 3) := by change ¬ Nat.Prime 77818863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25939621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4411 : row 4411 = 0 := (row_4411.checked).trans rfl
theorem model_count_4411 : modelRow 4411 = 0 := by decide
theorem modelrow_agrees_4411 : modelRow 4411 = row 4411 := model_count_4411.trans row_count_4411.symm
def row_4412 : RowCertificate 4412 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4412 1) := by change ¬ Nat.Prime 77871801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25957267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4412 2) := by change ¬ Nat.Prime 77862977; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1899097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4412 3) := by change ¬ Nat.Prime 77854153; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5988781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4412 : row 4412 = 0 := (row_4412.checked).trans rfl
theorem model_count_4412 : modelRow 4412 = 0 := by decide
theorem modelrow_agrees_4412 : modelRow 4412 = row 4412 := model_count_4412.trans row_count_4412.symm
def row_4413 : RowCertificate 4413 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4413 1) := by change Nat.Prime 77907103; exact ProjectEulerSolutions.P58PrimeNodes.p_77907103.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4413 2) := by change ¬ Nat.Prime 77898277; exact PrattCertificates.composite_of_product (factor := 1277) (cofactor := 61001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4413 3) := by change Nat.Prime 77889451; exact ProjectEulerSolutions.P58PrimeNodes.p_77889451.prime
    exact decide_eq_true prime
theorem row_count_4413 : row 4413 = 2 := (row_4413.checked).trans rfl
theorem model_count_4413 : modelRow 4413 = 2 := by decide
theorem modelrow_agrees_4413 : modelRow 4413 = row 4413 := model_count_4413.trans row_count_4413.symm
def row_4414 : RowCertificate 4414 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4414 1) := by change ¬ Nat.Prime 77942413; exact PrattCertificates.composite_of_product (factor := 4003) (cofactor := 19471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4414 2) := by change ¬ Nat.Prime 77933585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15586717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4414 3) := by change ¬ Nat.Prime 77924757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25974919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4414 : row 4414 = 0 := (row_4414.checked).trans rfl
theorem model_count_4414 : modelRow 4414 = 0 := by decide
theorem modelrow_agrees_4414 : modelRow 4414 = row 4414 := model_count_4414.trans row_count_4414.symm
def row_4415 : RowCertificate 4415 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4415 1) := by change ¬ Nat.Prime 77977731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25992577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4415 2) := by change ¬ Nat.Prime 77968901; exact PrattCertificates.composite_of_product (factor := 3037) (cofactor := 25673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4415 3) := by change ¬ Nat.Prime 77960071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11137153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4415 : row 4415 = 0 := (row_4415.checked).trans rfl
theorem model_count_4415 : modelRow 4415 = 0 := by decide
theorem modelrow_agrees_4415 : modelRow 4415 = row 4415 := model_count_4415.trans row_count_4415.symm
def row_4416 : RowCertificate 4416 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4416 1) := by change ¬ Nat.Prime 78013057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2108461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4416 2) := by change ¬ Nat.Prime 78004225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15600845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4416 3) := by change ¬ Nat.Prime 77995393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11142199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4416 : row 4416 = 0 := (row_4416.checked).trans rfl
theorem model_count_4416 : modelRow 4416 = 0 := by decide
theorem modelrow_agrees_4416 : modelRow 4416 = row 4416 := model_count_4416.trans row_count_4416.symm
def row_4417 : RowCertificate 4417 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4417 1) := by change ¬ Nat.Prime 78048391; exact PrattCertificates.composite_of_product (factor := 1579) (cofactor := 49429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4417 2) := by change ¬ Nat.Prime 78039557; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1279337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4417 3) := by change ¬ Nat.Prime 78030723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26010241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4417 : row 4417 = 0 := (row_4417.checked).trans rfl
theorem model_count_4417 : modelRow 4417 = 0 := by decide
theorem modelrow_agrees_4417 : modelRow 4417 = row 4417 := model_count_4417.trans row_count_4417.symm
def row_4418 : RowCertificate 4418 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4418 1) := by change ¬ Nat.Prime 78083733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26027911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4418 2) := by change ¬ Nat.Prime 78074897; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4592641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4418 3) := by change Nat.Prime 78066061; exact ProjectEulerSolutions.P58PrimeNodes.p_78066061.prime
    exact decide_eq_true prime
theorem row_count_4418 : row 4418 = 1 := (row_4418.checked).trans rfl
theorem model_count_4418 : modelRow 4418 = 1 := by decide
theorem modelrow_agrees_4418 : modelRow 4418 = row 4418 := model_count_4418.trans row_count_4418.symm
def row_4419 : RowCertificate 4419 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4419 1) := by change ¬ Nat.Prime 78119083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11159869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4419 2) := by change ¬ Nat.Prime 78110245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15622049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4419 3) := by change ¬ Nat.Prime 78101407; exact PrattCertificates.composite_of_product (factor := 4987) (cofactor := 15661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4419 : row 4419 = 0 := (row_4419.checked).trans rfl
theorem model_count_4419 : modelRow 4419 = 0 := by decide
theorem modelrow_agrees_4419 : modelRow 4419 = row 4419 := model_count_4419.trans row_count_4419.symm
def row_4420 : RowCertificate 4420 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4420 1) := by change ¬ Nat.Prime 78154441; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2521111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4420 2) := by change ¬ Nat.Prime 78145601; exact PrattCertificates.composite_of_product (factor := 6529) (cofactor := 11969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4420 3) := by change ¬ Nat.Prime 78136761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26045587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4420 : row 4420 = 0 := (row_4420.checked).trans rfl
theorem model_count_4420 : modelRow 4420 = 0 := by decide
theorem modelrow_agrees_4420 : modelRow 4420 = row 4420 := model_count_4420.trans row_count_4420.symm
def row_4421 : RowCertificate 4421 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4421 1) := by change ¬ Nat.Prime 78189807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26063269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4421 2) := by change ¬ Nat.Prime 78180965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15636193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4421 3) := by change ¬ Nat.Prime 78172123; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1070851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4421 : row 4421 = 0 := (row_4421.checked).trans rfl
theorem model_count_4421 : modelRow 4421 = 0 := by decide
theorem modelrow_agrees_4421 : modelRow 4421 = row 4421 := model_count_4421.trans row_count_4421.symm
def row_4422 : RowCertificate 4422 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4422 1) := by change ¬ Nat.Prime 78225181; exact PrattCertificates.composite_of_product (factor := 6199) (cofactor := 12619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4422 2) := by change ¬ Nat.Prime 78216337; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4600961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4422 3) := by change ¬ Nat.Prime 78207493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11172499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4422 : row 4422 = 0 := (row_4422.checked).trans rfl
theorem model_count_4422 : modelRow 4422 = 0 := by decide
theorem modelrow_agrees_4422 : modelRow 4422 = row 4422 := model_count_4422.trans row_count_4422.symm
def row_4423 : RowCertificate 4423 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4423 1) := by change ¬ Nat.Prime 78260563; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4118977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4423 2) := by change Nat.Prime 78251717; exact ProjectEulerSolutions.P58PrimeNodes.p_78251717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4423 3) := by change ¬ Nat.Prime 78242871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26080957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4423 : row 4423 = 1 := (row_4423.checked).trans rfl
theorem model_count_4423 : modelRow 4423 = 1 := by decide
theorem modelrow_agrees_4423 : modelRow 4423 = row 4423 := model_count_4423.trans row_count_4423.symm
def row_4424 : RowCertificate 4424 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4424 1) := by change ¬ Nat.Prime 78295953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26098651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4424 2) := by change ¬ Nat.Prime 78287105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15657421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4424 3) := by change ¬ Nat.Prime 78278257; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 370987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4424 : row 4424 = 0 := (row_4424.checked).trans rfl
theorem model_count_4424 : modelRow 4424 = 0 := by decide
theorem modelrow_agrees_4424 : modelRow 4424 = row 4424 := model_count_4424.trans row_count_4424.symm
def row_4425 : RowCertificate 4425 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4425 1) := by change ¬ Nat.Prime 78331351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11190193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4425 2) := by change ¬ Nat.Prime 78322501; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 432721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4425 3) := by change ¬ Nat.Prime 78313651; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6024127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4425 : row 4425 = 0 := (row_4425.checked).trans rfl
theorem model_count_4425 : modelRow 4425 = 0 := by decide
theorem modelrow_agrees_4425 : modelRow 4425 = row 4425 := model_count_4425.trans row_count_4425.symm
def row_4426 : RowCertificate 4426 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4426 1) := by change ¬ Nat.Prime 78366757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11195251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4426 2) := by change ¬ Nat.Prime 78357905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15671581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4426 3) := by change ¬ Nat.Prime 78349053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26116351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4426 : row 4426 = 0 := (row_4426.checked).trans rfl
theorem model_count_4426 : modelRow 4426 = 0 := by decide
theorem modelrow_agrees_4426 : modelRow 4426 = row 4426 := model_count_4426.trans row_count_4426.symm
def row_4427 : RowCertificate 4427 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4427 1) := by change ¬ Nat.Prime 78402171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26134057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4427 2) := by change Nat.Prime 78393317; exact ProjectEulerSolutions.P58PrimeNodes.p_78393317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4427 3) := by change ¬ Nat.Prime 78384463; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2118499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4427 : row 4427 = 1 := (row_4427.checked).trans rfl
theorem model_count_4427 : modelRow 4427 = 1 := by decide
theorem modelrow_agrees_4427 : modelRow 4427 = row 4427 := model_count_4427.trans row_count_4427.symm
def row_4428 : RowCertificate 4428 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4428 1) := by change ¬ Nat.Prime 78437593; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6033661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4428 2) := by change ¬ Nat.Prime 78428737; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1285717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4428 3) := by change ¬ Nat.Prime 78419881; exact PrattCertificates.composite_of_product (factor := 8287) (cofactor := 9463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4428 : row 4428 = 0 := (row_4428.checked).trans rfl
theorem model_count_4428 : modelRow 4428 = 0 := by decide
theorem modelrow_agrees_4428 : modelRow 4428 = row 4428 := model_count_4428.trans row_count_4428.symm
def row_4429 : RowCertificate 4429 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4429 1) := by change ¬ Nat.Prime 78473023; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1286443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4429 2) := by change ¬ Nat.Prime 78464165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15692833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4429 3) := by change ¬ Nat.Prime 78455307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26151769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4429 : row 4429 = 0 := (row_4429.checked).trans rfl
theorem model_count_4429 : modelRow 4429 = 0 := by decide
theorem modelrow_agrees_4429 : modelRow 4429 = row 4429 := model_count_4429.trans row_count_4429.symm
def row_4430 : RowCertificate 4430 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4430 1) := by change ¬ Nat.Prime 78508461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26169487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4430 2) := by change ¬ Nat.Prime 78499601; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1075337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4430 3) := by change ¬ Nat.Prime 78490741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11212963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4430 : row 4430 = 0 := (row_4430.checked).trans rfl
theorem model_count_4430 : modelRow 4430 = 0 := by decide
theorem modelrow_agrees_4430 : modelRow 4430 = row 4430 := model_count_4430.trans row_count_4430.symm
def row_4431 : RowCertificate 4431 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4431 1) := by change ¬ Nat.Prime 78543907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6041839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4431 2) := by change ¬ Nat.Prime 78535045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15707009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4431 3) := by change ¬ Nat.Prime 78526183; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4132957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4431 : row 4431 = 0 := (row_4431.checked).trans rfl
theorem model_count_4431 : modelRow 4431 = 0 := by decide
theorem modelrow_agrees_4431 : modelRow 4431 = row 4431 := model_count_4431.trans row_count_4431.symm
def row_4432 : RowCertificate 4432 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4432 1) := by change ¬ Nat.Prime 78579361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11225623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4432 2) := by change ¬ Nat.Prime 78570497; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 69593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4432 3) := by change ¬ Nat.Prime 78561633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26187211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4432 : row 4432 = 0 := (row_4432.checked).trans rfl
theorem model_count_4432 : modelRow 4432 = 0 := by decide
theorem modelrow_agrees_4432 : modelRow 4432 = row 4432 := model_count_4432.trans row_count_4432.symm
def row_4433 : RowCertificate 4433 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4433 1) := by change ¬ Nat.Prime 78614823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26204941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4433 2) := by change ¬ Nat.Prime 78605957; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 883213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4433 3) := by change ¬ Nat.Prime 78597091; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4136689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4433 : row 4433 = 0 := (row_4433.checked).trans rfl
theorem model_count_4433 : modelRow 4433 = 0 := by decide
theorem modelrow_agrees_4433 : modelRow 4433 = row 4433 := model_count_4433.trans row_count_4433.symm
def row_4434 : RowCertificate 4434 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4434 1) := by change ¬ Nat.Prime 78650293; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 136309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4434 2) := by change ¬ Nat.Prime 78641425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15728285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4434 3) := by change ¬ Nat.Prime 78632557; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 102253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4434 : row 4434 = 0 := (row_4434.checked).trans rfl
theorem model_count_4434 : modelRow 4434 = 0 := by decide
theorem modelrow_agrees_4434 : modelRow 4434 = row 4434 := model_count_4434.trans row_count_4434.symm
def row_4435 : RowCertificate 4435 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4435 1) := by change ¬ Nat.Prime 78685771; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 619573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4435 2) := by change ¬ Nat.Prime 78676901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4628053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4435 3) := by change ¬ Nat.Prime 78668031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26222677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4435 : row 4435 = 0 := (row_4435.checked).trans rfl
theorem model_count_4435 : modelRow 4435 = 0 := by decide
theorem modelrow_agrees_4435 : modelRow 4435 = row 4435 := model_count_4435.trans row_count_4435.symm
def row_4436 : RowCertificate 4436 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4436 1) := by change ¬ Nat.Prime 78721257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26240419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4436 2) := by change ¬ Nat.Prime 78712385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15742477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4436 3) := by change ¬ Nat.Prime 78703513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11243359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4436 : row 4436 = 0 := (row_4436.checked).trans rfl
theorem model_count_4436 : modelRow 4436 = 0 := by decide
theorem modelrow_agrees_4436 : modelRow 4436 = row 4436 := model_count_4436.trans row_count_4436.symm
def row_4437 : RowCertificate 4437 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4437 1) := by change ¬ Nat.Prime 78756751; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 722539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4437 2) := by change ¬ Nat.Prime 78747877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6057529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4437 3) := by change ¬ Nat.Prime 78739003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11248429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4437 : row 4437 = 0 := (row_4437.checked).trans rfl
theorem model_count_4437 : modelRow 4437 = 0 := by decide
theorem modelrow_agrees_4437 : modelRow 4437 = row 4437 := model_count_4437.trans row_count_4437.symm
def row_4438 : RowCertificate 4438 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4438 1) := by change ¬ Nat.Prime 78792253; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 521803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4438 2) := by change Nat.Prime 78783377; exact ProjectEulerSolutions.P58PrimeNodes.p_78783377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4438 3) := by change ¬ Nat.Prime 78774501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26258167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4438 : row 4438 = 1 := (row_4438.checked).trans rfl
theorem model_count_4438 : modelRow 4438 = 1 := by decide
theorem modelrow_agrees_4438 : modelRow 4438 = row 4438 := model_count_4438.trans row_count_4438.symm
def row_4439 : RowCertificate 4439 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4439 1) := by change ¬ Nat.Prime 78827763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26275921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4439 2) := by change ¬ Nat.Prime 78818885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15763777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4439 3) := by change ¬ Nat.Prime 78810007; exact PrattCertificates.composite_of_product (factor := 631) (cofactor := 124897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4439 : row 4439 = 0 := (row_4439.checked).trans rfl
theorem model_count_4439 : modelRow 4439 = 0 := by decide
theorem modelrow_agrees_4439 : modelRow 4439 = row 4439 := model_count_4439.trans row_count_4439.symm
def row_4440 : RowCertificate 4440 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4440 1) := by change ¬ Nat.Prime 78863281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11266183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4440 2) := by change Nat.Prime 78854401; exact ProjectEulerSolutions.P58PrimeNodes.p_78854401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4440 3) := by change ¬ Nat.Prime 78845521; exact PrattCertificates.composite_of_product (factor := 6547) (cofactor := 12043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4440 : row 4440 = 1 := (row_4440.checked).trans rfl
theorem model_count_4440 : modelRow 4440 = 1 := by decide
theorem modelrow_agrees_4440 : modelRow 4440 = row 4440 := model_count_4440.trans row_count_4440.symm
def row_4441 : RowCertificate 4441 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4441 1) := by change ¬ Nat.Prime 78898807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6069139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4441 2) := by change ¬ Nat.Prime 78889925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15777985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4441 3) := by change ¬ Nat.Prime 78881043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26293681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4441 : row 4441 = 0 := (row_4441.checked).trans rfl
theorem model_count_4441 : modelRow 4441 = 0 := by decide
theorem modelrow_agrees_4441 : modelRow 4441 = row 4441 := model_count_4441.trans row_count_4441.symm
def row_4442 : RowCertificate 4442 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4442 1) := by change ¬ Nat.Prime 78934341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26311447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4442 2) := by change ¬ Nat.Prime 78925457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6071189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4442 3) := by change Nat.Prime 78916573; exact ProjectEulerSolutions.P58PrimeNodes.p_78916573.prime
    exact decide_eq_true prime
theorem row_count_4442 : row 4442 = 1 := (row_4442.checked).trans rfl
theorem model_count_4442 : modelRow 4442 = 1 := by decide
theorem modelrow_agrees_4442 : modelRow 4442 = row 4442 := model_count_4442.trans row_count_4442.symm
def row_4443 : RowCertificate 4443 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4443 1) := by change Nat.Prime 78969883; exact ProjectEulerSolutions.P58PrimeNodes.p_78969883.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4443 2) := by change ¬ Nat.Prime 78960997; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2722793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4443 3) := by change ¬ Nat.Prime 78952111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11278873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4443 : row 4443 = 1 := (row_4443.checked).trans rfl
theorem model_count_4443 : modelRow 4443 = 1 := by decide
theorem modelrow_agrees_4443 : modelRow 4443 = row 4443 := model_count_4443.trans row_count_4443.symm
def row_4444 : RowCertificate 4444 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4444 1) := by change ¬ Nat.Prime 79005433; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6077341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4444 2) := by change ¬ Nat.Prime 78996545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15799309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4444 3) := by change ¬ Nat.Prime 78987657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26329219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4444 : row 4444 = 0 := (row_4444.checked).trans rfl
theorem model_count_4444 : modelRow 4444 = 0 := by decide
theorem modelrow_agrees_4444 : modelRow 4444 = row 4444 := model_count_4444.trans row_count_4444.symm
def row_4445 : RowCertificate 4445 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4445 1) := by change ¬ Nat.Prime 79040991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26346997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4445 2) := by change ¬ Nat.Prime 79032101; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 131501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4445 3) := by change ¬ Nat.Prime 79023211; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 119551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4445 : row 4445 = 0 := (row_4445.checked).trans rfl
theorem model_count_4445 : modelRow 4445 = 0 := by decide
theorem modelrow_agrees_4445 : modelRow 4445 = row 4445 := model_count_4445.trans row_count_4445.symm
def row_4446 : RowCertificate 4446 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4446 1) := by change ¬ Nat.Prime 79076557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11296651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4446 2) := by change ¬ Nat.Prime 79067665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15813533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4446 3) := by change ¬ Nat.Prime 79058773; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2550283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4446 : row 4446 = 0 := (row_4446.checked).trans rfl
theorem model_count_4446 : modelRow 4446 = 0 := by decide
theorem modelrow_agrees_4446 : modelRow 4446 = row 4446 := model_count_4446.trans row_count_4446.symm
def row_4447 : RowCertificate 4447 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4447 1) := by change ¬ Nat.Prime 79112131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11301733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4447 2) := by change Nat.Prime 79103237; exact ProjectEulerSolutions.P58PrimeNodes.p_79103237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4447 3) := by change ¬ Nat.Prime 79094343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26364781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4447 : row 4447 = 1 := (row_4447.checked).trans rfl
theorem model_count_4447 : modelRow 4447 = 1 := by decide
theorem modelrow_agrees_4447 : modelRow 4447 = row 4447 := model_count_4447.trans row_count_4447.symm
def row_4448 : RowCertificate 4448 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4448 1) := by change ¬ Nat.Prime 79147713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26382571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4448 2) := by change ¬ Nat.Prime 79138817; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 531133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4448 3) := by change ¬ Nat.Prime 79129921; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6086917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4448 : row 4448 = 0 := (row_4448.checked).trans rfl
theorem model_count_4448 : modelRow 4448 = 0 := by decide
theorem modelrow_agrees_4448 : modelRow 4448 = row 4448 := model_count_4448.trans row_count_4448.symm
def row_4449 : RowCertificate 4449 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4449 1) := by change ¬ Nat.Prime 79183303; exact PrattCertificates.composite_of_product (factor := 1693) (cofactor := 46771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4449 2) := by change ¬ Nat.Prime 79174405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15834881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4449 3) := by change ¬ Nat.Prime 79165507; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1084459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4449 : row 4449 = 0 := (row_4449.checked).trans rfl
theorem model_count_4449 : modelRow 4449 = 0 := by decide
theorem modelrow_agrees_4449 : modelRow 4449 = row 4449 := model_count_4449.trans row_count_4449.symm
def row_4450 : RowCertificate 4450 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4450 1) := by change Nat.Prime 79218901; exact ProjectEulerSolutions.P58PrimeNodes.p_79218901.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4450 2) := by change ¬ Nat.Prime 79210001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6093077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4450 3) := by change ¬ Nat.Prime 79201101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26400367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4450 : row 4450 = 1 := (row_4450.checked).trans rfl
theorem model_count_4450 : modelRow 4450 = 1 := by decide
theorem modelrow_agrees_4450 : modelRow 4450 = row 4450 := model_count_4450.trans row_count_4450.symm
def row_4451 : RowCertificate 4451 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4451 1) := by change ¬ Nat.Prime 79254507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26418169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4451 2) := by change ¬ Nat.Prime 79245605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15849121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4451 3) := by change ¬ Nat.Prime 79236703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11319529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4451 : row 4451 = 0 := (row_4451.checked).trans rfl
theorem model_count_4451 : modelRow 4451 = 0 := by decide
theorem modelrow_agrees_4451 : modelRow 4451 = row 4451 := model_count_4451.trans row_count_4451.symm
def row_4452 : RowCertificate 4452 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4452 1) := by change ¬ Nat.Prime 79290121; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 769807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4452 2) := by change ¬ Nat.Prime 79281217; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4663601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4452 3) := by change ¬ Nat.Prime 79272313; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4172227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4452 : row 4452 = 0 := (row_4452.checked).trans rfl
theorem model_count_4452 : modelRow 4452 = 0 := by decide
theorem modelrow_agrees_4452 : modelRow 4452 = row 4452 := model_count_4452.trans row_count_4452.symm
def row_4453 : RowCertificate 4453 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4453 1) := by change ¬ Nat.Prime 79325743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11332249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4453 2) := by change ¬ Nat.Prime 79316837; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1934557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4453 3) := by change ¬ Nat.Prime 79307931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26435977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4453 : row 4453 = 0 := (row_4453.checked).trans rfl
theorem model_count_4453 : modelRow 4453 = 0 := by decide
theorem modelrow_agrees_4453 : modelRow 4453 = row 4453 := model_count_4453.trans row_count_4453.symm
def row_4454 : RowCertificate 4454 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4454 1) := by change ¬ Nat.Prime 79361373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26453791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4454 2) := by change ¬ Nat.Prime 79352465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15870493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4454 3) := by change ¬ Nat.Prime 79343557; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1845199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4454 : row 4454 = 0 := (row_4454.checked).trans rfl
theorem model_count_4454 : modelRow 4454 = 0 := by decide
theorem modelrow_agrees_4454 : modelRow 4454 = row 4454 := model_count_4454.trans row_count_4454.symm
def row_4455 : RowCertificate 4455 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4455 1) := by change Nat.Prime 79397011; exact ProjectEulerSolutions.P58PrimeNodes.p_79397011.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4455 2) := by change ¬ Nat.Prime 79388101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6106777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4455 3) := by change ¬ Nat.Prime 79379191; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 625033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4455 : row 4455 = 1 := (row_4455.checked).trans rfl
theorem model_count_4455 : modelRow 4455 = 1 := by decide
theorem modelrow_agrees_4455 : modelRow 4455 = row 4455 := model_count_4455.trans row_count_4455.symm
def row_4456 : RowCertificate 4456 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4456 1) := by change ¬ Nat.Prime 79432657; exact PrattCertificates.composite_of_product (factor := 4423) (cofactor := 17959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4456 2) := by change ¬ Nat.Prime 79423745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15884749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4456 3) := by change ¬ Nat.Prime 79414833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26471611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4456 : row 4456 = 0 := (row_4456.checked).trans rfl
theorem model_count_4456 : modelRow 4456 = 0 := by decide
theorem modelrow_agrees_4456 : modelRow 4456 = row 4456 := model_count_4456.trans row_count_4456.symm
def row_4457 : RowCertificate 4457 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4457 1) := by change ¬ Nat.Prime 79468311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26489437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4457 2) := by change ¬ Nat.Prime 79459397; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 183509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4457 3) := by change ¬ Nat.Prime 79450483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11350069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4457 : row 4457 = 0 := (row_4457.checked).trans rfl
theorem model_count_4457 : modelRow 4457 = 0 := by decide
theorem modelrow_agrees_4457 : modelRow 4457 = row 4457 := model_count_4457.trans row_count_4457.symm
def row_4458 : RowCertificate 4458 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4458 1) := by change ¬ Nat.Prime 79503973; exact PrattCertificates.composite_of_product (factor := 2593) (cofactor := 30661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4458 2) := by change ¬ Nat.Prime 79495057; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 459509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4458 3) := by change ¬ Nat.Prime 79486141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11355163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4458 : row 4458 = 0 := (row_4458.checked).trans rfl
theorem model_count_4458 : modelRow 4458 = 0 := by decide
theorem modelrow_agrees_4458 : modelRow 4458 = row 4458 := model_count_4458.trans row_count_4458.symm
def row_4459 : RowCertificate 4459 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4459 1) := by change ¬ Nat.Prime 79539643; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4186297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4459 2) := by change ¬ Nat.Prime 79530725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15906145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4459 3) := by change ¬ Nat.Prime 79521807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26507269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4459 : row 4459 = 0 := (row_4459.checked).trans rfl
theorem model_count_4459 : modelRow 4459 = 0 := by decide
theorem modelrow_agrees_4459 : modelRow 4459 = row 4459 := model_count_4459.trans row_count_4459.symm
def row_4460 : RowCertificate 4460 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4460 1) := by change ¬ Nat.Prime 79575321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26525107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4460 2) := by change ¬ Nat.Prime 79566401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2743669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4460 3) := by change ¬ Nat.Prime 79557481; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1304221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4460 : row 4460 = 0 := (row_4460.checked).trans rfl
theorem model_count_4460 : modelRow 4460 = 0 := by decide
theorem modelrow_agrees_4460 : modelRow 4460 = row 4460 := model_count_4460.trans row_count_4460.symm
def row_4461 : RowCertificate 4461 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4461 1) := by change ¬ Nat.Prime 79611007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11373001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4461 2) := by change ¬ Nat.Prime 79602085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15920417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4461 3) := by change ¬ Nat.Prime 79593163; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6122551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4461 : row 4461 = 0 := (row_4461.checked).trans rfl
theorem model_count_4461 : modelRow 4461 = 0 := by decide
theorem modelrow_agrees_4461 : modelRow 4461 = row 4461 := model_count_4461.trans row_count_4461.symm
def row_4462 : RowCertificate 4462 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4462 1) := by change ¬ Nat.Prime 79646701; exact PrattCertificates.composite_of_product (factor := 3259) (cofactor := 24439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4462 2) := by change ¬ Nat.Prime 79637777; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 287501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4462 3) := by change ¬ Nat.Prime 79628853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26542951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4462 : row 4462 = 0 := (row_4462.checked).trans rfl
theorem model_count_4462 : modelRow 4462 = 0 := by decide
theorem modelrow_agrees_4462 : modelRow 4462 = row 4462 := model_count_4462.trans row_count_4462.symm
def row_4463 : RowCertificate 4463 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4463 1) := by change ¬ Nat.Prime 79682403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26560801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4463 2) := by change ¬ Nat.Prime 79673477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6128729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4463 3) := by change Nat.Prime 79664551; exact ProjectEulerSolutions.P58PrimeNodes.p_79664551.prime
    exact decide_eq_true prime
theorem row_count_4463 : row 4463 = 1 := (row_4463.checked).trans rfl
theorem model_count_4463 : modelRow 4463 = 1 := by decide
theorem modelrow_agrees_4463 : modelRow 4463 = row 4463 := model_count_4463.trans row_count_4463.symm
def row_4464 : RowCertificate 4464 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4464 1) := by change Nat.Prime 79718113; exact ProjectEulerSolutions.P58PrimeNodes.p_79718113.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4464 2) := by change ¬ Nat.Prime 79709185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15941837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4464 3) := by change ¬ Nat.Prime 79700257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11385751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4464 : row 4464 = 1 := (row_4464.checked).trans rfl
theorem model_count_4464 : modelRow 4464 = 1 := by decide
theorem modelrow_agrees_4464 : modelRow 4464 = row 4464 := model_count_4464.trans row_count_4464.symm
def row_4465 : RowCertificate 4465 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4465 1) := by change Nat.Prime 79753831; exact ProjectEulerSolutions.P58PrimeNodes.p_79753831.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4465 2) := by change ¬ Nat.Prime 79744901; exact PrattCertificates.composite_of_product (factor := 3461) (cofactor := 23041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4465 3) := by change ¬ Nat.Prime 79735971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26578657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4465 : row 4465 = 1 := (row_4465.checked).trans rfl
theorem model_count_4465 : modelRow 4465 = 1 := by decide
theorem modelrow_agrees_4465 : modelRow 4465 = row 4465 := model_count_4465.trans row_count_4465.symm
def row_4466 : RowCertificate 4466 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4466 1) := by change ¬ Nat.Prime 79789557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26596519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4466 2) := by change ¬ Nat.Prime 79780625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15956125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4466 3) := by change ¬ Nat.Prime 79771693; exact PrattCertificates.composite_of_product (factor := 1801) (cofactor := 44293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4466 : row 4466 = 0 := (row_4466.checked).trans rfl
theorem model_count_4466 : modelRow 4466 = 0 := by decide
theorem modelrow_agrees_4466 : modelRow 4466 = row 4466 := model_count_4466.trans row_count_4466.symm
def row_4467 : RowCertificate 4467 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4467 1) := by change ¬ Nat.Prime 79825291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11403613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4467 2) := by change ¬ Nat.Prime 79816357; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1505969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4467 3) := by change ¬ Nat.Prime 79807423; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2574433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4467 : row 4467 = 0 := (row_4467.checked).trans rfl
theorem model_count_4467 : modelRow 4467 = 0 := by decide
theorem modelrow_agrees_4467 : modelRow 4467 = row 4467 := model_count_4467.trans row_count_4467.symm
def row_4468 : RowCertificate 4468 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4468 1) := by change ¬ Nat.Prime 79861033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11408719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4468 2) := by change ¬ Nat.Prime 79852097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6142469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4468 3) := by change ¬ Nat.Prime 79843161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26614387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4468 : row 4468 = 0 := (row_4468.checked).trans rfl
theorem model_count_4468 : modelRow 4468 = 0 := by decide
theorem modelrow_agrees_4468 : modelRow 4468 = row 4468 := model_count_4468.trans row_count_4468.symm
def row_4469 : RowCertificate 4469 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4469 1) := by change ¬ Nat.Prime 79896783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26632261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4469 2) := by change ¬ Nat.Prime 79887845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15977569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4469 3) := by change ¬ Nat.Prime 79878907; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4204153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4469 : row 4469 = 0 := (row_4469.checked).trans rfl
theorem model_count_4469 : modelRow 4469 = 0 := by decide
theorem modelrow_agrees_4469 : modelRow 4469 = row 4469 := model_count_4469.trans row_count_4469.symm
def row_4470 : RowCertificate 4470 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4470 1) := by change ¬ Nat.Prime 79932541; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6148657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4470 2) := by change Nat.Prime 79923601; exact ProjectEulerSolutions.P58PrimeNodes.p_79923601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4470 3) := by change ¬ Nat.Prime 79914661; exact PrattCertificates.composite_of_product (factor := 751) (cofactor := 106411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4470 : row 4470 = 1 := (row_4470.checked).trans rfl
theorem model_count_4470 : modelRow 4470 = 1 := by decide
theorem modelrow_agrees_4470 : modelRow 4470 = row 4470 := model_count_4470.trans row_count_4470.symm
def row_4471 : RowCertificate 4471 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4471 1) := by change Nat.Prime 79968307; exact ProjectEulerSolutions.P58PrimeNodes.p_79968307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4471 2) := by change ¬ Nat.Prime 79959365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15991873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4471 3) := by change ¬ Nat.Prime 79950423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26650141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4471 : row 4471 = 1 := (row_4471.checked).trans rfl
theorem model_count_4471 : modelRow 4471 = 1 := by decide
theorem modelrow_agrees_4471 : modelRow 4471 = row 4471 := model_count_4471.trans row_count_4471.symm
def row_4472 : RowCertificate 4472 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4472 1) := by change ¬ Nat.Prime 80004081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26668027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4472 2) := by change ¬ Nat.Prime 79995137; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2758453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4472 3) := by change ¬ Nat.Prime 79986193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11426599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4472 : row 4472 = 0 := (row_4472.checked).trans rfl
theorem model_count_4472 : modelRow 4472 = 0 := by decide
theorem modelrow_agrees_4472 : modelRow 4472 = row 4472 := model_count_4472.trans row_count_4472.symm
def row_4473 : RowCertificate 4473 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4473 1) := by change Nat.Prime 80039863; exact ProjectEulerSolutions.P58PrimeNodes.p_80039863.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4473 2) := by change ¬ Nat.Prime 80030917; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4707701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4473 3) := by change Nat.Prime 80021971; exact ProjectEulerSolutions.P58PrimeNodes.p_80021971.prime
    exact decide_eq_true prime
theorem row_count_4473 : row 4473 = 2 := (row_4473.checked).trans rfl
theorem model_count_4473 : modelRow 4473 = 2 := by decide
theorem modelrow_agrees_4473 : modelRow 4473 = row 4473 := model_count_4473.trans row_count_4473.symm
def row_4474 : RowCertificate 4474 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4474 1) := by change ¬ Nat.Prime 80075653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11439379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4474 2) := by change ¬ Nat.Prime 80066705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 16013341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4474 3) := by change ¬ Nat.Prime 80057757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26685919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4474 : row 4474 = 0 := (row_4474.checked).trans rfl
theorem model_count_4474 : modelRow 4474 = 0 := by decide
theorem modelrow_agrees_4474 : modelRow 4474 = row 4474 := model_count_4474.trans row_count_4474.symm
def row_4475 : RowCertificate 4475 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4475 1) := by change ¬ Nat.Prime 80111451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26703817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4475 2) := by change ¬ Nat.Prime 80102501; exact PrattCertificates.composite_of_product (factor := 6733) (cofactor := 11897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4475 3) := by change ¬ Nat.Prime 80093551; exact PrattCertificates.composite_of_product (factor := 1321) (cofactor := 60631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4475 : row 4475 = 0 := (row_4475.checked).trans rfl
theorem model_count_4475 : modelRow 4475 = 0 := by decide
theorem modelrow_agrees_4475 : modelRow 4475 = row 4475 := model_count_4475.trans row_count_4475.symm
def row_4476 : RowCertificate 4476 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4476 1) := by change ¬ Nat.Prime 80147257; exact PrattCertificates.composite_of_product (factor := 3163) (cofactor := 25339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4476 2) := by change ¬ Nat.Prime 80138305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 16027661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4476 3) := by change ¬ Nat.Prime 80129353; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 229597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4476 : row 4476 = 0 := (row_4476.checked).trans rfl
theorem model_count_4476 : modelRow 4476 = 0 := by decide
theorem modelrow_agrees_4476 : modelRow 4476 = row 4476 := model_count_4476.trans row_count_4476.symm
def row_4477 : RowCertificate 4477 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4477 1) := by change Nat.Prime 80183071; exact ProjectEulerSolutions.P58PrimeNodes.p_80183071.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4477 2) := by change ¬ Nat.Prime 80174117; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 157513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4477 3) := by change ¬ Nat.Prime 80165163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26721721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4477 : row 4477 = 1 := (row_4477.checked).trans rfl
theorem model_count_4477 : modelRow 4477 = 1 := by decide
theorem modelrow_agrees_4477 : modelRow 4477 = row 4477 := model_count_4477.trans row_count_4477.symm
def row_4478 : RowCertificate 4478 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4478 1) := by change ¬ Nat.Prime 80218893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26739631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4478 2) := by change ¬ Nat.Prime 80209937; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1314917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4478 3) := by change ¬ Nat.Prime 80200981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11457283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4478 : row 4478 = 0 := (row_4478.checked).trans rfl
theorem model_count_4478 : modelRow 4478 = 0 := by decide
theorem modelrow_agrees_4478 : modelRow 4478 = row 4478 := model_count_4478.trans row_count_4478.symm
def row_4479 : RowCertificate 4479 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4479 1) := by change ¬ Nat.Prime 80254723; exact PrattCertificates.composite_of_product (factor := 5689) (cofactor := 14107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4479 2) := by change ¬ Nat.Prime 80245765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 16049153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4479 3) := by change ¬ Nat.Prime 80236807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 11462401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4479 : row 4479 = 0 := (row_4479.checked).trans rfl
theorem model_count_4479 : modelRow 4479 = 0 := by decide
theorem modelrow_agrees_4479 : modelRow 4479 = row 4479 := model_count_4479.trans row_count_4479.symm
def row_4480 : RowCertificate 4480 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4480 1) := by change ¬ Nat.Prime 80290561; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6176197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4480 2) := by change ¬ Nat.Prime 80281601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2169773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4480 3) := by change ¬ Nat.Prime 80272641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 26757547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4480 : row 4480 = 0 := (row_4480.checked).trans rfl
theorem model_count_4480 : modelRow 4480 = 0 := by decide
theorem modelrow_agrees_4480 : modelRow 4480 = row 4480 := model_count_4480.trans row_count_4480.symm
private def atomic_4353 : CountSegment 1 10 13120 4352 4353 1983 1984 := CountSegment.single (first := 4352) (initial := 1983) row_count_4353 (by decide)
private theorem atomic_4353_agreement : RowAgreement 4352 4353 := RowAgreement.single (first := 4352) modelrow_agrees_4353
private def atomic_4354 : CountSegment 1 10 13120 4353 4354 1984 1984 := CountSegment.single (first := 4353) (initial := 1984) row_count_4354 (by decide)
private theorem atomic_4354_agreement : RowAgreement 4353 4354 := RowAgreement.single (first := 4353) modelrow_agrees_4354
private def atomic_4355 : CountSegment 1 10 13120 4354 4355 1984 1984 := CountSegment.single (first := 4354) (initial := 1984) row_count_4355 (by decide)
private theorem atomic_4355_agreement : RowAgreement 4354 4355 := RowAgreement.single (first := 4354) modelrow_agrees_4355
private def atomic_4356 : CountSegment 1 10 13120 4355 4356 1984 1984 := CountSegment.single (first := 4355) (initial := 1984) row_count_4356 (by decide)
private theorem atomic_4356_agreement : RowAgreement 4355 4356 := RowAgreement.single (first := 4355) modelrow_agrees_4356
private def atomic_4357 : CountSegment 1 10 13120 4356 4357 1984 1984 := CountSegment.single (first := 4356) (initial := 1984) row_count_4357 (by decide)
private theorem atomic_4357_agreement : RowAgreement 4356 4357 := RowAgreement.single (first := 4356) modelrow_agrees_4357
private def atomic_4358 : CountSegment 1 10 13120 4357 4358 1984 1985 := CountSegment.single (first := 4357) (initial := 1984) row_count_4358 (by decide)
private theorem atomic_4358_agreement : RowAgreement 4357 4358 := RowAgreement.single (first := 4357) modelrow_agrees_4358
private def atomic_4359 : CountSegment 1 10 13120 4358 4359 1985 1986 := CountSegment.single (first := 4358) (initial := 1985) row_count_4359 (by decide)
private theorem atomic_4359_agreement : RowAgreement 4358 4359 := RowAgreement.single (first := 4358) modelrow_agrees_4359
private def atomic_4360 : CountSegment 1 10 13120 4359 4360 1986 1988 := CountSegment.single (first := 4359) (initial := 1986) row_count_4360 (by decide)
private theorem atomic_4360_agreement : RowAgreement 4359 4360 := RowAgreement.single (first := 4359) modelrow_agrees_4360
private def atomic_4361 : CountSegment 1 10 13120 4360 4361 1988 1988 := CountSegment.single (first := 4360) (initial := 1988) row_count_4361 (by decide)
private theorem atomic_4361_agreement : RowAgreement 4360 4361 := RowAgreement.single (first := 4360) modelrow_agrees_4361
private def atomic_4362 : CountSegment 1 10 13120 4361 4362 1988 1988 := CountSegment.single (first := 4361) (initial := 1988) row_count_4362 (by decide)
private theorem atomic_4362_agreement : RowAgreement 4361 4362 := RowAgreement.single (first := 4361) modelrow_agrees_4362
private def atomic_4363 : CountSegment 1 10 13120 4362 4363 1988 1988 := CountSegment.single (first := 4362) (initial := 1988) row_count_4363 (by decide)
private theorem atomic_4363_agreement : RowAgreement 4362 4363 := RowAgreement.single (first := 4362) modelrow_agrees_4363
private def atomic_4364 : CountSegment 1 10 13120 4363 4364 1988 1989 := CountSegment.single (first := 4363) (initial := 1988) row_count_4364 (by decide)
private theorem atomic_4364_agreement : RowAgreement 4363 4364 := RowAgreement.single (first := 4363) modelrow_agrees_4364
private def atomic_4365 : CountSegment 1 10 13120 4364 4365 1989 1990 := CountSegment.single (first := 4364) (initial := 1989) row_count_4365 (by decide)
private theorem atomic_4365_agreement : RowAgreement 4364 4365 := RowAgreement.single (first := 4364) modelrow_agrees_4365
private def atomic_4366 : CountSegment 1 10 13120 4365 4366 1990 1990 := CountSegment.single (first := 4365) (initial := 1990) row_count_4366 (by decide)
private theorem atomic_4366_agreement : RowAgreement 4365 4366 := RowAgreement.single (first := 4365) modelrow_agrees_4366
private def atomic_4367 : CountSegment 1 10 13120 4366 4367 1990 1990 := CountSegment.single (first := 4366) (initial := 1990) row_count_4367 (by decide)
private theorem atomic_4367_agreement : RowAgreement 4366 4367 := RowAgreement.single (first := 4366) modelrow_agrees_4367
private def atomic_4368 : CountSegment 1 10 13120 4367 4368 1990 1990 := CountSegment.single (first := 4367) (initial := 1990) row_count_4368 (by decide)
private theorem atomic_4368_agreement : RowAgreement 4367 4368 := RowAgreement.single (first := 4367) modelrow_agrees_4368
private def atomic_4369 : CountSegment 1 10 13120 4368 4369 1990 1990 := CountSegment.single (first := 4368) (initial := 1990) row_count_4369 (by decide)
private theorem atomic_4369_agreement : RowAgreement 4368 4369 := RowAgreement.single (first := 4368) modelrow_agrees_4369
private def atomic_4370 : CountSegment 1 10 13120 4369 4370 1990 1990 := CountSegment.single (first := 4369) (initial := 1990) row_count_4370 (by decide)
private theorem atomic_4370_agreement : RowAgreement 4369 4370 := RowAgreement.single (first := 4369) modelrow_agrees_4370
private def atomic_4371 : CountSegment 1 10 13120 4370 4371 1990 1991 := CountSegment.single (first := 4370) (initial := 1990) row_count_4371 (by decide)
private theorem atomic_4371_agreement : RowAgreement 4370 4371 := RowAgreement.single (first := 4370) modelrow_agrees_4371
private def atomic_4372 : CountSegment 1 10 13120 4371 4372 1991 1992 := CountSegment.single (first := 4371) (initial := 1991) row_count_4372 (by decide)
private theorem atomic_4372_agreement : RowAgreement 4371 4372 := RowAgreement.single (first := 4371) modelrow_agrees_4372
private def atomic_4373 : CountSegment 1 10 13120 4372 4373 1992 1992 := CountSegment.single (first := 4372) (initial := 1992) row_count_4373 (by decide)
private theorem atomic_4373_agreement : RowAgreement 4372 4373 := RowAgreement.single (first := 4372) modelrow_agrees_4373
private def atomic_4374 : CountSegment 1 10 13120 4373 4374 1992 1993 := CountSegment.single (first := 4373) (initial := 1992) row_count_4374 (by decide)
private theorem atomic_4374_agreement : RowAgreement 4373 4374 := RowAgreement.single (first := 4373) modelrow_agrees_4374
private def atomic_4375 : CountSegment 1 10 13120 4374 4375 1993 1994 := CountSegment.single (first := 4374) (initial := 1993) row_count_4375 (by decide)
private theorem atomic_4375_agreement : RowAgreement 4374 4375 := RowAgreement.single (first := 4374) modelrow_agrees_4375
private def atomic_4376 : CountSegment 1 10 13120 4375 4376 1994 1994 := CountSegment.single (first := 4375) (initial := 1994) row_count_4376 (by decide)
private theorem atomic_4376_agreement : RowAgreement 4375 4376 := RowAgreement.single (first := 4375) modelrow_agrees_4376
private def atomic_4377 : CountSegment 1 10 13120 4376 4377 1994 1995 := CountSegment.single (first := 4376) (initial := 1994) row_count_4377 (by decide)
private theorem atomic_4377_agreement : RowAgreement 4376 4377 := RowAgreement.single (first := 4376) modelrow_agrees_4377
private def atomic_4378 : CountSegment 1 10 13120 4377 4378 1995 1996 := CountSegment.single (first := 4377) (initial := 1995) row_count_4378 (by decide)
private theorem atomic_4378_agreement : RowAgreement 4377 4378 := RowAgreement.single (first := 4377) modelrow_agrees_4378
private def atomic_4379 : CountSegment 1 10 13120 4378 4379 1996 1997 := CountSegment.single (first := 4378) (initial := 1996) row_count_4379 (by decide)
private theorem atomic_4379_agreement : RowAgreement 4378 4379 := RowAgreement.single (first := 4378) modelrow_agrees_4379
private def atomic_4380 : CountSegment 1 10 13120 4379 4380 1997 1999 := CountSegment.single (first := 4379) (initial := 1997) row_count_4380 (by decide)
private theorem atomic_4380_agreement : RowAgreement 4379 4380 := RowAgreement.single (first := 4379) modelrow_agrees_4380
private def atomic_4381 : CountSegment 1 10 13120 4380 4381 1999 2000 := CountSegment.single (first := 4380) (initial := 1999) row_count_4381 (by decide)
private theorem atomic_4381_agreement : RowAgreement 4380 4381 := RowAgreement.single (first := 4380) modelrow_agrees_4381
private def atomic_4382 : CountSegment 1 10 13120 4381 4382 2000 2000 := CountSegment.single (first := 4381) (initial := 2000) row_count_4382 (by decide)
private theorem atomic_4382_agreement : RowAgreement 4381 4382 := RowAgreement.single (first := 4381) modelrow_agrees_4382
private def atomic_4383 : CountSegment 1 10 13120 4382 4383 2000 2000 := CountSegment.single (first := 4382) (initial := 2000) row_count_4383 (by decide)
private theorem atomic_4383_agreement : RowAgreement 4382 4383 := RowAgreement.single (first := 4382) modelrow_agrees_4383
private def atomic_4384 : CountSegment 1 10 13120 4383 4384 2000 2000 := CountSegment.single (first := 4383) (initial := 2000) row_count_4384 (by decide)
private theorem atomic_4384_agreement : RowAgreement 4383 4384 := RowAgreement.single (first := 4383) modelrow_agrees_4384
private def atomic_4385 : CountSegment 1 10 13120 4384 4385 2000 2000 := CountSegment.single (first := 4384) (initial := 2000) row_count_4385 (by decide)
private theorem atomic_4385_agreement : RowAgreement 4384 4385 := RowAgreement.single (first := 4384) modelrow_agrees_4385
private def atomic_4386 : CountSegment 1 10 13120 4385 4386 2000 2000 := CountSegment.single (first := 4385) (initial := 2000) row_count_4386 (by decide)
private theorem atomic_4386_agreement : RowAgreement 4385 4386 := RowAgreement.single (first := 4385) modelrow_agrees_4386
private def atomic_4387 : CountSegment 1 10 13120 4386 4387 2000 2002 := CountSegment.single (first := 4386) (initial := 2000) row_count_4387 (by decide)
private theorem atomic_4387_agreement : RowAgreement 4386 4387 := RowAgreement.single (first := 4386) modelrow_agrees_4387
private def atomic_4388 : CountSegment 1 10 13120 4387 4388 2002 2002 := CountSegment.single (first := 4387) (initial := 2002) row_count_4388 (by decide)
private theorem atomic_4388_agreement : RowAgreement 4387 4388 := RowAgreement.single (first := 4387) modelrow_agrees_4388
private def atomic_4389 : CountSegment 1 10 13120 4388 4389 2002 2003 := CountSegment.single (first := 4388) (initial := 2002) row_count_4389 (by decide)
private theorem atomic_4389_agreement : RowAgreement 4388 4389 := RowAgreement.single (first := 4388) modelrow_agrees_4389
private def atomic_4390 : CountSegment 1 10 13120 4389 4390 2003 2003 := CountSegment.single (first := 4389) (initial := 2003) row_count_4390 (by decide)
private theorem atomic_4390_agreement : RowAgreement 4389 4390 := RowAgreement.single (first := 4389) modelrow_agrees_4390
private def atomic_4391 : CountSegment 1 10 13120 4390 4391 2003 2003 := CountSegment.single (first := 4390) (initial := 2003) row_count_4391 (by decide)
private theorem atomic_4391_agreement : RowAgreement 4390 4391 := RowAgreement.single (first := 4390) modelrow_agrees_4391
private def atomic_4392 : CountSegment 1 10 13120 4391 4392 2003 2004 := CountSegment.single (first := 4391) (initial := 2003) row_count_4392 (by decide)
private theorem atomic_4392_agreement : RowAgreement 4391 4392 := RowAgreement.single (first := 4391) modelrow_agrees_4392
private def atomic_4393 : CountSegment 1 10 13120 4392 4393 2004 2005 := CountSegment.single (first := 4392) (initial := 2004) row_count_4393 (by decide)
private theorem atomic_4393_agreement : RowAgreement 4392 4393 := RowAgreement.single (first := 4392) modelrow_agrees_4393
private def atomic_4394 : CountSegment 1 10 13120 4393 4394 2005 2005 := CountSegment.single (first := 4393) (initial := 2005) row_count_4394 (by decide)
private theorem atomic_4394_agreement : RowAgreement 4393 4394 := RowAgreement.single (first := 4393) modelrow_agrees_4394
private def atomic_4395 : CountSegment 1 10 13120 4394 4395 2005 2006 := CountSegment.single (first := 4394) (initial := 2005) row_count_4395 (by decide)
private theorem atomic_4395_agreement : RowAgreement 4394 4395 := RowAgreement.single (first := 4394) modelrow_agrees_4395
private def atomic_4396 : CountSegment 1 10 13120 4395 4396 2006 2006 := CountSegment.single (first := 4395) (initial := 2006) row_count_4396 (by decide)
private theorem atomic_4396_agreement : RowAgreement 4395 4396 := RowAgreement.single (first := 4395) modelrow_agrees_4396
private def atomic_4397 : CountSegment 1 10 13120 4396 4397 2006 2006 := CountSegment.single (first := 4396) (initial := 2006) row_count_4397 (by decide)
private theorem atomic_4397_agreement : RowAgreement 4396 4397 := RowAgreement.single (first := 4396) modelrow_agrees_4397
private def atomic_4398 : CountSegment 1 10 13120 4397 4398 2006 2006 := CountSegment.single (first := 4397) (initial := 2006) row_count_4398 (by decide)
private theorem atomic_4398_agreement : RowAgreement 4397 4398 := RowAgreement.single (first := 4397) modelrow_agrees_4398
private def atomic_4399 : CountSegment 1 10 13120 4398 4399 2006 2006 := CountSegment.single (first := 4398) (initial := 2006) row_count_4399 (by decide)
private theorem atomic_4399_agreement : RowAgreement 4398 4399 := RowAgreement.single (first := 4398) modelrow_agrees_4399
private def atomic_4400 : CountSegment 1 10 13120 4399 4400 2006 2006 := CountSegment.single (first := 4399) (initial := 2006) row_count_4400 (by decide)
private theorem atomic_4400_agreement : RowAgreement 4399 4400 := RowAgreement.single (first := 4399) modelrow_agrees_4400
private def atomic_4401 : CountSegment 1 10 13120 4400 4401 2006 2007 := CountSegment.single (first := 4400) (initial := 2006) row_count_4401 (by decide)
private theorem atomic_4401_agreement : RowAgreement 4400 4401 := RowAgreement.single (first := 4400) modelrow_agrees_4401
private def atomic_4402 : CountSegment 1 10 13120 4401 4402 2007 2007 := CountSegment.single (first := 4401) (initial := 2007) row_count_4402 (by decide)
private theorem atomic_4402_agreement : RowAgreement 4401 4402 := RowAgreement.single (first := 4401) modelrow_agrees_4402
private def atomic_4403 : CountSegment 1 10 13120 4402 4403 2007 2007 := CountSegment.single (first := 4402) (initial := 2007) row_count_4403 (by decide)
private theorem atomic_4403_agreement : RowAgreement 4402 4403 := RowAgreement.single (first := 4402) modelrow_agrees_4403
private def atomic_4404 : CountSegment 1 10 13120 4403 4404 2007 2008 := CountSegment.single (first := 4403) (initial := 2007) row_count_4404 (by decide)
private theorem atomic_4404_agreement : RowAgreement 4403 4404 := RowAgreement.single (first := 4403) modelrow_agrees_4404
private def atomic_4405 : CountSegment 1 10 13120 4404 4405 2008 2008 := CountSegment.single (first := 4404) (initial := 2008) row_count_4405 (by decide)
private theorem atomic_4405_agreement : RowAgreement 4404 4405 := RowAgreement.single (first := 4404) modelrow_agrees_4405
private def atomic_4406 : CountSegment 1 10 13120 4405 4406 2008 2008 := CountSegment.single (first := 4405) (initial := 2008) row_count_4406 (by decide)
private theorem atomic_4406_agreement : RowAgreement 4405 4406 := RowAgreement.single (first := 4405) modelrow_agrees_4406
private def atomic_4407 : CountSegment 1 10 13120 4406 4407 2008 2009 := CountSegment.single (first := 4406) (initial := 2008) row_count_4407 (by decide)
private theorem atomic_4407_agreement : RowAgreement 4406 4407 := RowAgreement.single (first := 4406) modelrow_agrees_4407
private def atomic_4408 : CountSegment 1 10 13120 4407 4408 2009 2010 := CountSegment.single (first := 4407) (initial := 2009) row_count_4408 (by decide)
private theorem atomic_4408_agreement : RowAgreement 4407 4408 := RowAgreement.single (first := 4407) modelrow_agrees_4408
private def atomic_4409 : CountSegment 1 10 13120 4408 4409 2010 2010 := CountSegment.single (first := 4408) (initial := 2010) row_count_4409 (by decide)
private theorem atomic_4409_agreement : RowAgreement 4408 4409 := RowAgreement.single (first := 4408) modelrow_agrees_4409
private def atomic_4410 : CountSegment 1 10 13120 4409 4410 2010 2010 := CountSegment.single (first := 4409) (initial := 2010) row_count_4410 (by decide)
private theorem atomic_4410_agreement : RowAgreement 4409 4410 := RowAgreement.single (first := 4409) modelrow_agrees_4410
private def atomic_4411 : CountSegment 1 10 13120 4410 4411 2010 2010 := CountSegment.single (first := 4410) (initial := 2010) row_count_4411 (by decide)
private theorem atomic_4411_agreement : RowAgreement 4410 4411 := RowAgreement.single (first := 4410) modelrow_agrees_4411
private def atomic_4412 : CountSegment 1 10 13120 4411 4412 2010 2010 := CountSegment.single (first := 4411) (initial := 2010) row_count_4412 (by decide)
private theorem atomic_4412_agreement : RowAgreement 4411 4412 := RowAgreement.single (first := 4411) modelrow_agrees_4412
private def atomic_4413 : CountSegment 1 10 13120 4412 4413 2010 2012 := CountSegment.single (first := 4412) (initial := 2010) row_count_4413 (by decide)
private theorem atomic_4413_agreement : RowAgreement 4412 4413 := RowAgreement.single (first := 4412) modelrow_agrees_4413
private def atomic_4414 : CountSegment 1 10 13120 4413 4414 2012 2012 := CountSegment.single (first := 4413) (initial := 2012) row_count_4414 (by decide)
private theorem atomic_4414_agreement : RowAgreement 4413 4414 := RowAgreement.single (first := 4413) modelrow_agrees_4414
private def atomic_4415 : CountSegment 1 10 13120 4414 4415 2012 2012 := CountSegment.single (first := 4414) (initial := 2012) row_count_4415 (by decide)
private theorem atomic_4415_agreement : RowAgreement 4414 4415 := RowAgreement.single (first := 4414) modelrow_agrees_4415
private def atomic_4416 : CountSegment 1 10 13120 4415 4416 2012 2012 := CountSegment.single (first := 4415) (initial := 2012) row_count_4416 (by decide)
private theorem atomic_4416_agreement : RowAgreement 4415 4416 := RowAgreement.single (first := 4415) modelrow_agrees_4416
private def atomic_4417 : CountSegment 1 10 13120 4416 4417 2012 2012 := CountSegment.single (first := 4416) (initial := 2012) row_count_4417 (by decide)
private theorem atomic_4417_agreement : RowAgreement 4416 4417 := RowAgreement.single (first := 4416) modelrow_agrees_4417
private def atomic_4418 : CountSegment 1 10 13120 4417 4418 2012 2013 := CountSegment.single (first := 4417) (initial := 2012) row_count_4418 (by decide)
private theorem atomic_4418_agreement : RowAgreement 4417 4418 := RowAgreement.single (first := 4417) modelrow_agrees_4418
private def atomic_4419 : CountSegment 1 10 13120 4418 4419 2013 2013 := CountSegment.single (first := 4418) (initial := 2013) row_count_4419 (by decide)
private theorem atomic_4419_agreement : RowAgreement 4418 4419 := RowAgreement.single (first := 4418) modelrow_agrees_4419
private def atomic_4420 : CountSegment 1 10 13120 4419 4420 2013 2013 := CountSegment.single (first := 4419) (initial := 2013) row_count_4420 (by decide)
private theorem atomic_4420_agreement : RowAgreement 4419 4420 := RowAgreement.single (first := 4419) modelrow_agrees_4420
private def atomic_4421 : CountSegment 1 10 13120 4420 4421 2013 2013 := CountSegment.single (first := 4420) (initial := 2013) row_count_4421 (by decide)
private theorem atomic_4421_agreement : RowAgreement 4420 4421 := RowAgreement.single (first := 4420) modelrow_agrees_4421
private def atomic_4422 : CountSegment 1 10 13120 4421 4422 2013 2013 := CountSegment.single (first := 4421) (initial := 2013) row_count_4422 (by decide)
private theorem atomic_4422_agreement : RowAgreement 4421 4422 := RowAgreement.single (first := 4421) modelrow_agrees_4422
private def atomic_4423 : CountSegment 1 10 13120 4422 4423 2013 2014 := CountSegment.single (first := 4422) (initial := 2013) row_count_4423 (by decide)
private theorem atomic_4423_agreement : RowAgreement 4422 4423 := RowAgreement.single (first := 4422) modelrow_agrees_4423
private def atomic_4424 : CountSegment 1 10 13120 4423 4424 2014 2014 := CountSegment.single (first := 4423) (initial := 2014) row_count_4424 (by decide)
private theorem atomic_4424_agreement : RowAgreement 4423 4424 := RowAgreement.single (first := 4423) modelrow_agrees_4424
private def atomic_4425 : CountSegment 1 10 13120 4424 4425 2014 2014 := CountSegment.single (first := 4424) (initial := 2014) row_count_4425 (by decide)
private theorem atomic_4425_agreement : RowAgreement 4424 4425 := RowAgreement.single (first := 4424) modelrow_agrees_4425
private def atomic_4426 : CountSegment 1 10 13120 4425 4426 2014 2014 := CountSegment.single (first := 4425) (initial := 2014) row_count_4426 (by decide)
private theorem atomic_4426_agreement : RowAgreement 4425 4426 := RowAgreement.single (first := 4425) modelrow_agrees_4426
private def atomic_4427 : CountSegment 1 10 13120 4426 4427 2014 2015 := CountSegment.single (first := 4426) (initial := 2014) row_count_4427 (by decide)
private theorem atomic_4427_agreement : RowAgreement 4426 4427 := RowAgreement.single (first := 4426) modelrow_agrees_4427
private def atomic_4428 : CountSegment 1 10 13120 4427 4428 2015 2015 := CountSegment.single (first := 4427) (initial := 2015) row_count_4428 (by decide)
private theorem atomic_4428_agreement : RowAgreement 4427 4428 := RowAgreement.single (first := 4427) modelrow_agrees_4428
private def atomic_4429 : CountSegment 1 10 13120 4428 4429 2015 2015 := CountSegment.single (first := 4428) (initial := 2015) row_count_4429 (by decide)
private theorem atomic_4429_agreement : RowAgreement 4428 4429 := RowAgreement.single (first := 4428) modelrow_agrees_4429
private def atomic_4430 : CountSegment 1 10 13120 4429 4430 2015 2015 := CountSegment.single (first := 4429) (initial := 2015) row_count_4430 (by decide)
private theorem atomic_4430_agreement : RowAgreement 4429 4430 := RowAgreement.single (first := 4429) modelrow_agrees_4430
private def atomic_4431 : CountSegment 1 10 13120 4430 4431 2015 2015 := CountSegment.single (first := 4430) (initial := 2015) row_count_4431 (by decide)
private theorem atomic_4431_agreement : RowAgreement 4430 4431 := RowAgreement.single (first := 4430) modelrow_agrees_4431
private def atomic_4432 : CountSegment 1 10 13120 4431 4432 2015 2015 := CountSegment.single (first := 4431) (initial := 2015) row_count_4432 (by decide)
private theorem atomic_4432_agreement : RowAgreement 4431 4432 := RowAgreement.single (first := 4431) modelrow_agrees_4432
private def atomic_4433 : CountSegment 1 10 13120 4432 4433 2015 2015 := CountSegment.single (first := 4432) (initial := 2015) row_count_4433 (by decide)
private theorem atomic_4433_agreement : RowAgreement 4432 4433 := RowAgreement.single (first := 4432) modelrow_agrees_4433
private def atomic_4434 : CountSegment 1 10 13120 4433 4434 2015 2015 := CountSegment.single (first := 4433) (initial := 2015) row_count_4434 (by decide)
private theorem atomic_4434_agreement : RowAgreement 4433 4434 := RowAgreement.single (first := 4433) modelrow_agrees_4434
private def atomic_4435 : CountSegment 1 10 13120 4434 4435 2015 2015 := CountSegment.single (first := 4434) (initial := 2015) row_count_4435 (by decide)
private theorem atomic_4435_agreement : RowAgreement 4434 4435 := RowAgreement.single (first := 4434) modelrow_agrees_4435
private def atomic_4436 : CountSegment 1 10 13120 4435 4436 2015 2015 := CountSegment.single (first := 4435) (initial := 2015) row_count_4436 (by decide)
private theorem atomic_4436_agreement : RowAgreement 4435 4436 := RowAgreement.single (first := 4435) modelrow_agrees_4436
private def atomic_4437 : CountSegment 1 10 13120 4436 4437 2015 2015 := CountSegment.single (first := 4436) (initial := 2015) row_count_4437 (by decide)
private theorem atomic_4437_agreement : RowAgreement 4436 4437 := RowAgreement.single (first := 4436) modelrow_agrees_4437
private def atomic_4438 : CountSegment 1 10 13120 4437 4438 2015 2016 := CountSegment.single (first := 4437) (initial := 2015) row_count_4438 (by decide)
private theorem atomic_4438_agreement : RowAgreement 4437 4438 := RowAgreement.single (first := 4437) modelrow_agrees_4438
private def atomic_4439 : CountSegment 1 10 13120 4438 4439 2016 2016 := CountSegment.single (first := 4438) (initial := 2016) row_count_4439 (by decide)
private theorem atomic_4439_agreement : RowAgreement 4438 4439 := RowAgreement.single (first := 4438) modelrow_agrees_4439
private def atomic_4440 : CountSegment 1 10 13120 4439 4440 2016 2017 := CountSegment.single (first := 4439) (initial := 2016) row_count_4440 (by decide)
private theorem atomic_4440_agreement : RowAgreement 4439 4440 := RowAgreement.single (first := 4439) modelrow_agrees_4440
private def atomic_4441 : CountSegment 1 10 13120 4440 4441 2017 2017 := CountSegment.single (first := 4440) (initial := 2017) row_count_4441 (by decide)
private theorem atomic_4441_agreement : RowAgreement 4440 4441 := RowAgreement.single (first := 4440) modelrow_agrees_4441
private def atomic_4442 : CountSegment 1 10 13120 4441 4442 2017 2018 := CountSegment.single (first := 4441) (initial := 2017) row_count_4442 (by decide)
private theorem atomic_4442_agreement : RowAgreement 4441 4442 := RowAgreement.single (first := 4441) modelrow_agrees_4442
private def atomic_4443 : CountSegment 1 10 13120 4442 4443 2018 2019 := CountSegment.single (first := 4442) (initial := 2018) row_count_4443 (by decide)
private theorem atomic_4443_agreement : RowAgreement 4442 4443 := RowAgreement.single (first := 4442) modelrow_agrees_4443
private def atomic_4444 : CountSegment 1 10 13120 4443 4444 2019 2019 := CountSegment.single (first := 4443) (initial := 2019) row_count_4444 (by decide)
private theorem atomic_4444_agreement : RowAgreement 4443 4444 := RowAgreement.single (first := 4443) modelrow_agrees_4444
private def atomic_4445 : CountSegment 1 10 13120 4444 4445 2019 2019 := CountSegment.single (first := 4444) (initial := 2019) row_count_4445 (by decide)
private theorem atomic_4445_agreement : RowAgreement 4444 4445 := RowAgreement.single (first := 4444) modelrow_agrees_4445
private def atomic_4446 : CountSegment 1 10 13120 4445 4446 2019 2019 := CountSegment.single (first := 4445) (initial := 2019) row_count_4446 (by decide)
private theorem atomic_4446_agreement : RowAgreement 4445 4446 := RowAgreement.single (first := 4445) modelrow_agrees_4446
private def atomic_4447 : CountSegment 1 10 13120 4446 4447 2019 2020 := CountSegment.single (first := 4446) (initial := 2019) row_count_4447 (by decide)
private theorem atomic_4447_agreement : RowAgreement 4446 4447 := RowAgreement.single (first := 4446) modelrow_agrees_4447
private def atomic_4448 : CountSegment 1 10 13120 4447 4448 2020 2020 := CountSegment.single (first := 4447) (initial := 2020) row_count_4448 (by decide)
private theorem atomic_4448_agreement : RowAgreement 4447 4448 := RowAgreement.single (first := 4447) modelrow_agrees_4448
private def atomic_4449 : CountSegment 1 10 13120 4448 4449 2020 2020 := CountSegment.single (first := 4448) (initial := 2020) row_count_4449 (by decide)
private theorem atomic_4449_agreement : RowAgreement 4448 4449 := RowAgreement.single (first := 4448) modelrow_agrees_4449
private def atomic_4450 : CountSegment 1 10 13120 4449 4450 2020 2021 := CountSegment.single (first := 4449) (initial := 2020) row_count_4450 (by decide)
private theorem atomic_4450_agreement : RowAgreement 4449 4450 := RowAgreement.single (first := 4449) modelrow_agrees_4450
private def atomic_4451 : CountSegment 1 10 13120 4450 4451 2021 2021 := CountSegment.single (first := 4450) (initial := 2021) row_count_4451 (by decide)
private theorem atomic_4451_agreement : RowAgreement 4450 4451 := RowAgreement.single (first := 4450) modelrow_agrees_4451
private def atomic_4452 : CountSegment 1 10 13120 4451 4452 2021 2021 := CountSegment.single (first := 4451) (initial := 2021) row_count_4452 (by decide)
private theorem atomic_4452_agreement : RowAgreement 4451 4452 := RowAgreement.single (first := 4451) modelrow_agrees_4452
private def atomic_4453 : CountSegment 1 10 13120 4452 4453 2021 2021 := CountSegment.single (first := 4452) (initial := 2021) row_count_4453 (by decide)
private theorem atomic_4453_agreement : RowAgreement 4452 4453 := RowAgreement.single (first := 4452) modelrow_agrees_4453
private def atomic_4454 : CountSegment 1 10 13120 4453 4454 2021 2021 := CountSegment.single (first := 4453) (initial := 2021) row_count_4454 (by decide)
private theorem atomic_4454_agreement : RowAgreement 4453 4454 := RowAgreement.single (first := 4453) modelrow_agrees_4454
private def atomic_4455 : CountSegment 1 10 13120 4454 4455 2021 2022 := CountSegment.single (first := 4454) (initial := 2021) row_count_4455 (by decide)
private theorem atomic_4455_agreement : RowAgreement 4454 4455 := RowAgreement.single (first := 4454) modelrow_agrees_4455
private def atomic_4456 : CountSegment 1 10 13120 4455 4456 2022 2022 := CountSegment.single (first := 4455) (initial := 2022) row_count_4456 (by decide)
private theorem atomic_4456_agreement : RowAgreement 4455 4456 := RowAgreement.single (first := 4455) modelrow_agrees_4456
private def atomic_4457 : CountSegment 1 10 13120 4456 4457 2022 2022 := CountSegment.single (first := 4456) (initial := 2022) row_count_4457 (by decide)
private theorem atomic_4457_agreement : RowAgreement 4456 4457 := RowAgreement.single (first := 4456) modelrow_agrees_4457
private def atomic_4458 : CountSegment 1 10 13120 4457 4458 2022 2022 := CountSegment.single (first := 4457) (initial := 2022) row_count_4458 (by decide)
private theorem atomic_4458_agreement : RowAgreement 4457 4458 := RowAgreement.single (first := 4457) modelrow_agrees_4458
private def atomic_4459 : CountSegment 1 10 13120 4458 4459 2022 2022 := CountSegment.single (first := 4458) (initial := 2022) row_count_4459 (by decide)
private theorem atomic_4459_agreement : RowAgreement 4458 4459 := RowAgreement.single (first := 4458) modelrow_agrees_4459
private def atomic_4460 : CountSegment 1 10 13120 4459 4460 2022 2022 := CountSegment.single (first := 4459) (initial := 2022) row_count_4460 (by decide)
private theorem atomic_4460_agreement : RowAgreement 4459 4460 := RowAgreement.single (first := 4459) modelrow_agrees_4460
private def atomic_4461 : CountSegment 1 10 13120 4460 4461 2022 2022 := CountSegment.single (first := 4460) (initial := 2022) row_count_4461 (by decide)
private theorem atomic_4461_agreement : RowAgreement 4460 4461 := RowAgreement.single (first := 4460) modelrow_agrees_4461
private def atomic_4462 : CountSegment 1 10 13120 4461 4462 2022 2022 := CountSegment.single (first := 4461) (initial := 2022) row_count_4462 (by decide)
private theorem atomic_4462_agreement : RowAgreement 4461 4462 := RowAgreement.single (first := 4461) modelrow_agrees_4462
private def atomic_4463 : CountSegment 1 10 13120 4462 4463 2022 2023 := CountSegment.single (first := 4462) (initial := 2022) row_count_4463 (by decide)
private theorem atomic_4463_agreement : RowAgreement 4462 4463 := RowAgreement.single (first := 4462) modelrow_agrees_4463
private def atomic_4464 : CountSegment 1 10 13120 4463 4464 2023 2024 := CountSegment.single (first := 4463) (initial := 2023) row_count_4464 (by decide)
private theorem atomic_4464_agreement : RowAgreement 4463 4464 := RowAgreement.single (first := 4463) modelrow_agrees_4464
private def atomic_4465 : CountSegment 1 10 13120 4464 4465 2024 2025 := CountSegment.single (first := 4464) (initial := 2024) row_count_4465 (by decide)
private theorem atomic_4465_agreement : RowAgreement 4464 4465 := RowAgreement.single (first := 4464) modelrow_agrees_4465
private def atomic_4466 : CountSegment 1 10 13120 4465 4466 2025 2025 := CountSegment.single (first := 4465) (initial := 2025) row_count_4466 (by decide)
private theorem atomic_4466_agreement : RowAgreement 4465 4466 := RowAgreement.single (first := 4465) modelrow_agrees_4466
private def atomic_4467 : CountSegment 1 10 13120 4466 4467 2025 2025 := CountSegment.single (first := 4466) (initial := 2025) row_count_4467 (by decide)
private theorem atomic_4467_agreement : RowAgreement 4466 4467 := RowAgreement.single (first := 4466) modelrow_agrees_4467
private def atomic_4468 : CountSegment 1 10 13120 4467 4468 2025 2025 := CountSegment.single (first := 4467) (initial := 2025) row_count_4468 (by decide)
private theorem atomic_4468_agreement : RowAgreement 4467 4468 := RowAgreement.single (first := 4467) modelrow_agrees_4468
private def atomic_4469 : CountSegment 1 10 13120 4468 4469 2025 2025 := CountSegment.single (first := 4468) (initial := 2025) row_count_4469 (by decide)
private theorem atomic_4469_agreement : RowAgreement 4468 4469 := RowAgreement.single (first := 4468) modelrow_agrees_4469
private def atomic_4470 : CountSegment 1 10 13120 4469 4470 2025 2026 := CountSegment.single (first := 4469) (initial := 2025) row_count_4470 (by decide)
private theorem atomic_4470_agreement : RowAgreement 4469 4470 := RowAgreement.single (first := 4469) modelrow_agrees_4470
private def atomic_4471 : CountSegment 1 10 13120 4470 4471 2026 2027 := CountSegment.single (first := 4470) (initial := 2026) row_count_4471 (by decide)
private theorem atomic_4471_agreement : RowAgreement 4470 4471 := RowAgreement.single (first := 4470) modelrow_agrees_4471
private def atomic_4472 : CountSegment 1 10 13120 4471 4472 2027 2027 := CountSegment.single (first := 4471) (initial := 2027) row_count_4472 (by decide)
private theorem atomic_4472_agreement : RowAgreement 4471 4472 := RowAgreement.single (first := 4471) modelrow_agrees_4472
private def atomic_4473 : CountSegment 1 10 13120 4472 4473 2027 2029 := CountSegment.single (first := 4472) (initial := 2027) row_count_4473 (by decide)
private theorem atomic_4473_agreement : RowAgreement 4472 4473 := RowAgreement.single (first := 4472) modelrow_agrees_4473
private def atomic_4474 : CountSegment 1 10 13120 4473 4474 2029 2029 := CountSegment.single (first := 4473) (initial := 2029) row_count_4474 (by decide)
private theorem atomic_4474_agreement : RowAgreement 4473 4474 := RowAgreement.single (first := 4473) modelrow_agrees_4474
private def atomic_4475 : CountSegment 1 10 13120 4474 4475 2029 2029 := CountSegment.single (first := 4474) (initial := 2029) row_count_4475 (by decide)
private theorem atomic_4475_agreement : RowAgreement 4474 4475 := RowAgreement.single (first := 4474) modelrow_agrees_4475
private def atomic_4476 : CountSegment 1 10 13120 4475 4476 2029 2029 := CountSegment.single (first := 4475) (initial := 2029) row_count_4476 (by decide)
private theorem atomic_4476_agreement : RowAgreement 4475 4476 := RowAgreement.single (first := 4475) modelrow_agrees_4476
private def atomic_4477 : CountSegment 1 10 13120 4476 4477 2029 2030 := CountSegment.single (first := 4476) (initial := 2029) row_count_4477 (by decide)
private theorem atomic_4477_agreement : RowAgreement 4476 4477 := RowAgreement.single (first := 4476) modelrow_agrees_4477
private def atomic_4478 : CountSegment 1 10 13120 4477 4478 2030 2030 := CountSegment.single (first := 4477) (initial := 2030) row_count_4478 (by decide)
private theorem atomic_4478_agreement : RowAgreement 4477 4478 := RowAgreement.single (first := 4477) modelrow_agrees_4478
private def atomic_4479 : CountSegment 1 10 13120 4478 4479 2030 2030 := CountSegment.single (first := 4478) (initial := 2030) row_count_4479 (by decide)
private theorem atomic_4479_agreement : RowAgreement 4478 4479 := RowAgreement.single (first := 4478) modelrow_agrees_4479
private def atomic_4480 : CountSegment 1 10 13120 4479 4480 2030 2030 := CountSegment.single (first := 4479) (initial := 2030) row_count_4480 (by decide)
private theorem atomic_4480_agreement : RowAgreement 4479 4480 := RowAgreement.single (first := 4479) modelrow_agrees_4480
private def within_034_000 : CountSegment 1 10 13120 4352 4354 1983 1984 := CountSegment.join atomic_4353 atomic_4354
private theorem within_034_000_agreement : RowAgreement 4352 4354 := RowAgreement.join atomic_4353_agreement atomic_4354_agreement
private def within_034_001 : CountSegment 1 10 13120 4354 4356 1984 1984 := CountSegment.join atomic_4355 atomic_4356
private theorem within_034_001_agreement : RowAgreement 4354 4356 := RowAgreement.join atomic_4355_agreement atomic_4356_agreement
private def within_034_002 : CountSegment 1 10 13120 4352 4356 1983 1984 := CountSegment.join within_034_000 within_034_001
private theorem within_034_002_agreement : RowAgreement 4352 4356 := RowAgreement.join within_034_000_agreement within_034_001_agreement
private def within_034_003 : CountSegment 1 10 13120 4356 4358 1984 1985 := CountSegment.join atomic_4357 atomic_4358
private theorem within_034_003_agreement : RowAgreement 4356 4358 := RowAgreement.join atomic_4357_agreement atomic_4358_agreement
private def within_034_004 : CountSegment 1 10 13120 4358 4360 1985 1988 := CountSegment.join atomic_4359 atomic_4360
private theorem within_034_004_agreement : RowAgreement 4358 4360 := RowAgreement.join atomic_4359_agreement atomic_4360_agreement
private def within_034_005 : CountSegment 1 10 13120 4356 4360 1984 1988 := CountSegment.join within_034_003 within_034_004
private theorem within_034_005_agreement : RowAgreement 4356 4360 := RowAgreement.join within_034_003_agreement within_034_004_agreement
private def within_034_006 : CountSegment 1 10 13120 4352 4360 1983 1988 := CountSegment.join within_034_002 within_034_005
private theorem within_034_006_agreement : RowAgreement 4352 4360 := RowAgreement.join within_034_002_agreement within_034_005_agreement
private def within_034_007 : CountSegment 1 10 13120 4360 4362 1988 1988 := CountSegment.join atomic_4361 atomic_4362
private theorem within_034_007_agreement : RowAgreement 4360 4362 := RowAgreement.join atomic_4361_agreement atomic_4362_agreement
private def within_034_008 : CountSegment 1 10 13120 4362 4364 1988 1989 := CountSegment.join atomic_4363 atomic_4364
private theorem within_034_008_agreement : RowAgreement 4362 4364 := RowAgreement.join atomic_4363_agreement atomic_4364_agreement
private def within_034_009 : CountSegment 1 10 13120 4360 4364 1988 1989 := CountSegment.join within_034_007 within_034_008
private theorem within_034_009_agreement : RowAgreement 4360 4364 := RowAgreement.join within_034_007_agreement within_034_008_agreement
private def within_034_010 : CountSegment 1 10 13120 4364 4366 1989 1990 := CountSegment.join atomic_4365 atomic_4366
private theorem within_034_010_agreement : RowAgreement 4364 4366 := RowAgreement.join atomic_4365_agreement atomic_4366_agreement
private def within_034_011 : CountSegment 1 10 13120 4366 4368 1990 1990 := CountSegment.join atomic_4367 atomic_4368
private theorem within_034_011_agreement : RowAgreement 4366 4368 := RowAgreement.join atomic_4367_agreement atomic_4368_agreement
private def within_034_012 : CountSegment 1 10 13120 4364 4368 1989 1990 := CountSegment.join within_034_010 within_034_011
private theorem within_034_012_agreement : RowAgreement 4364 4368 := RowAgreement.join within_034_010_agreement within_034_011_agreement
private def within_034_013 : CountSegment 1 10 13120 4360 4368 1988 1990 := CountSegment.join within_034_009 within_034_012
private theorem within_034_013_agreement : RowAgreement 4360 4368 := RowAgreement.join within_034_009_agreement within_034_012_agreement
private def within_034_014 : CountSegment 1 10 13120 4352 4368 1983 1990 := CountSegment.join within_034_006 within_034_013
private theorem within_034_014_agreement : RowAgreement 4352 4368 := RowAgreement.join within_034_006_agreement within_034_013_agreement
private def within_034_015 : CountSegment 1 10 13120 4368 4370 1990 1990 := CountSegment.join atomic_4369 atomic_4370
private theorem within_034_015_agreement : RowAgreement 4368 4370 := RowAgreement.join atomic_4369_agreement atomic_4370_agreement
private def within_034_016 : CountSegment 1 10 13120 4370 4372 1990 1992 := CountSegment.join atomic_4371 atomic_4372
private theorem within_034_016_agreement : RowAgreement 4370 4372 := RowAgreement.join atomic_4371_agreement atomic_4372_agreement
private def within_034_017 : CountSegment 1 10 13120 4368 4372 1990 1992 := CountSegment.join within_034_015 within_034_016
private theorem within_034_017_agreement : RowAgreement 4368 4372 := RowAgreement.join within_034_015_agreement within_034_016_agreement
private def within_034_018 : CountSegment 1 10 13120 4372 4374 1992 1993 := CountSegment.join atomic_4373 atomic_4374
private theorem within_034_018_agreement : RowAgreement 4372 4374 := RowAgreement.join atomic_4373_agreement atomic_4374_agreement
private def within_034_019 : CountSegment 1 10 13120 4374 4376 1993 1994 := CountSegment.join atomic_4375 atomic_4376
private theorem within_034_019_agreement : RowAgreement 4374 4376 := RowAgreement.join atomic_4375_agreement atomic_4376_agreement
private def within_034_020 : CountSegment 1 10 13120 4372 4376 1992 1994 := CountSegment.join within_034_018 within_034_019
private theorem within_034_020_agreement : RowAgreement 4372 4376 := RowAgreement.join within_034_018_agreement within_034_019_agreement
private def within_034_021 : CountSegment 1 10 13120 4368 4376 1990 1994 := CountSegment.join within_034_017 within_034_020
private theorem within_034_021_agreement : RowAgreement 4368 4376 := RowAgreement.join within_034_017_agreement within_034_020_agreement
private def within_034_022 : CountSegment 1 10 13120 4376 4378 1994 1996 := CountSegment.join atomic_4377 atomic_4378
private theorem within_034_022_agreement : RowAgreement 4376 4378 := RowAgreement.join atomic_4377_agreement atomic_4378_agreement
private def within_034_023 : CountSegment 1 10 13120 4378 4380 1996 1999 := CountSegment.join atomic_4379 atomic_4380
private theorem within_034_023_agreement : RowAgreement 4378 4380 := RowAgreement.join atomic_4379_agreement atomic_4380_agreement
private def within_034_024 : CountSegment 1 10 13120 4376 4380 1994 1999 := CountSegment.join within_034_022 within_034_023
private theorem within_034_024_agreement : RowAgreement 4376 4380 := RowAgreement.join within_034_022_agreement within_034_023_agreement
private def within_034_025 : CountSegment 1 10 13120 4380 4382 1999 2000 := CountSegment.join atomic_4381 atomic_4382
private theorem within_034_025_agreement : RowAgreement 4380 4382 := RowAgreement.join atomic_4381_agreement atomic_4382_agreement
private def within_034_026 : CountSegment 1 10 13120 4382 4384 2000 2000 := CountSegment.join atomic_4383 atomic_4384
private theorem within_034_026_agreement : RowAgreement 4382 4384 := RowAgreement.join atomic_4383_agreement atomic_4384_agreement
private def within_034_027 : CountSegment 1 10 13120 4380 4384 1999 2000 := CountSegment.join within_034_025 within_034_026
private theorem within_034_027_agreement : RowAgreement 4380 4384 := RowAgreement.join within_034_025_agreement within_034_026_agreement
private def within_034_028 : CountSegment 1 10 13120 4376 4384 1994 2000 := CountSegment.join within_034_024 within_034_027
private theorem within_034_028_agreement : RowAgreement 4376 4384 := RowAgreement.join within_034_024_agreement within_034_027_agreement
private def within_034_029 : CountSegment 1 10 13120 4368 4384 1990 2000 := CountSegment.join within_034_021 within_034_028
private theorem within_034_029_agreement : RowAgreement 4368 4384 := RowAgreement.join within_034_021_agreement within_034_028_agreement
private def within_034_030 : CountSegment 1 10 13120 4352 4384 1983 2000 := CountSegment.join within_034_014 within_034_029
private theorem within_034_030_agreement : RowAgreement 4352 4384 := RowAgreement.join within_034_014_agreement within_034_029_agreement
private def within_034_031 : CountSegment 1 10 13120 4384 4386 2000 2000 := CountSegment.join atomic_4385 atomic_4386
private theorem within_034_031_agreement : RowAgreement 4384 4386 := RowAgreement.join atomic_4385_agreement atomic_4386_agreement
private def within_034_032 : CountSegment 1 10 13120 4386 4388 2000 2002 := CountSegment.join atomic_4387 atomic_4388
private theorem within_034_032_agreement : RowAgreement 4386 4388 := RowAgreement.join atomic_4387_agreement atomic_4388_agreement
private def within_034_033 : CountSegment 1 10 13120 4384 4388 2000 2002 := CountSegment.join within_034_031 within_034_032
private theorem within_034_033_agreement : RowAgreement 4384 4388 := RowAgreement.join within_034_031_agreement within_034_032_agreement
private def within_034_034 : CountSegment 1 10 13120 4388 4390 2002 2003 := CountSegment.join atomic_4389 atomic_4390
private theorem within_034_034_agreement : RowAgreement 4388 4390 := RowAgreement.join atomic_4389_agreement atomic_4390_agreement
private def within_034_035 : CountSegment 1 10 13120 4390 4392 2003 2004 := CountSegment.join atomic_4391 atomic_4392
private theorem within_034_035_agreement : RowAgreement 4390 4392 := RowAgreement.join atomic_4391_agreement atomic_4392_agreement
private def within_034_036 : CountSegment 1 10 13120 4388 4392 2002 2004 := CountSegment.join within_034_034 within_034_035
private theorem within_034_036_agreement : RowAgreement 4388 4392 := RowAgreement.join within_034_034_agreement within_034_035_agreement
private def within_034_037 : CountSegment 1 10 13120 4384 4392 2000 2004 := CountSegment.join within_034_033 within_034_036
private theorem within_034_037_agreement : RowAgreement 4384 4392 := RowAgreement.join within_034_033_agreement within_034_036_agreement
private def within_034_038 : CountSegment 1 10 13120 4392 4394 2004 2005 := CountSegment.join atomic_4393 atomic_4394
private theorem within_034_038_agreement : RowAgreement 4392 4394 := RowAgreement.join atomic_4393_agreement atomic_4394_agreement
private def within_034_039 : CountSegment 1 10 13120 4394 4396 2005 2006 := CountSegment.join atomic_4395 atomic_4396
private theorem within_034_039_agreement : RowAgreement 4394 4396 := RowAgreement.join atomic_4395_agreement atomic_4396_agreement
private def within_034_040 : CountSegment 1 10 13120 4392 4396 2004 2006 := CountSegment.join within_034_038 within_034_039
private theorem within_034_040_agreement : RowAgreement 4392 4396 := RowAgreement.join within_034_038_agreement within_034_039_agreement
private def within_034_041 : CountSegment 1 10 13120 4396 4398 2006 2006 := CountSegment.join atomic_4397 atomic_4398
private theorem within_034_041_agreement : RowAgreement 4396 4398 := RowAgreement.join atomic_4397_agreement atomic_4398_agreement
private def within_034_042 : CountSegment 1 10 13120 4398 4400 2006 2006 := CountSegment.join atomic_4399 atomic_4400
private theorem within_034_042_agreement : RowAgreement 4398 4400 := RowAgreement.join atomic_4399_agreement atomic_4400_agreement
private def within_034_043 : CountSegment 1 10 13120 4396 4400 2006 2006 := CountSegment.join within_034_041 within_034_042
private theorem within_034_043_agreement : RowAgreement 4396 4400 := RowAgreement.join within_034_041_agreement within_034_042_agreement
private def within_034_044 : CountSegment 1 10 13120 4392 4400 2004 2006 := CountSegment.join within_034_040 within_034_043
private theorem within_034_044_agreement : RowAgreement 4392 4400 := RowAgreement.join within_034_040_agreement within_034_043_agreement
private def within_034_045 : CountSegment 1 10 13120 4384 4400 2000 2006 := CountSegment.join within_034_037 within_034_044
private theorem within_034_045_agreement : RowAgreement 4384 4400 := RowAgreement.join within_034_037_agreement within_034_044_agreement
private def within_034_046 : CountSegment 1 10 13120 4400 4402 2006 2007 := CountSegment.join atomic_4401 atomic_4402
private theorem within_034_046_agreement : RowAgreement 4400 4402 := RowAgreement.join atomic_4401_agreement atomic_4402_agreement
private def within_034_047 : CountSegment 1 10 13120 4402 4404 2007 2008 := CountSegment.join atomic_4403 atomic_4404
private theorem within_034_047_agreement : RowAgreement 4402 4404 := RowAgreement.join atomic_4403_agreement atomic_4404_agreement
private def within_034_048 : CountSegment 1 10 13120 4400 4404 2006 2008 := CountSegment.join within_034_046 within_034_047
private theorem within_034_048_agreement : RowAgreement 4400 4404 := RowAgreement.join within_034_046_agreement within_034_047_agreement
private def within_034_049 : CountSegment 1 10 13120 4404 4406 2008 2008 := CountSegment.join atomic_4405 atomic_4406
private theorem within_034_049_agreement : RowAgreement 4404 4406 := RowAgreement.join atomic_4405_agreement atomic_4406_agreement
private def within_034_050 : CountSegment 1 10 13120 4406 4408 2008 2010 := CountSegment.join atomic_4407 atomic_4408
private theorem within_034_050_agreement : RowAgreement 4406 4408 := RowAgreement.join atomic_4407_agreement atomic_4408_agreement
private def within_034_051 : CountSegment 1 10 13120 4404 4408 2008 2010 := CountSegment.join within_034_049 within_034_050
private theorem within_034_051_agreement : RowAgreement 4404 4408 := RowAgreement.join within_034_049_agreement within_034_050_agreement
private def within_034_052 : CountSegment 1 10 13120 4400 4408 2006 2010 := CountSegment.join within_034_048 within_034_051
private theorem within_034_052_agreement : RowAgreement 4400 4408 := RowAgreement.join within_034_048_agreement within_034_051_agreement
private def within_034_053 : CountSegment 1 10 13120 4408 4410 2010 2010 := CountSegment.join atomic_4409 atomic_4410
private theorem within_034_053_agreement : RowAgreement 4408 4410 := RowAgreement.join atomic_4409_agreement atomic_4410_agreement
private def within_034_054 : CountSegment 1 10 13120 4410 4412 2010 2010 := CountSegment.join atomic_4411 atomic_4412
private theorem within_034_054_agreement : RowAgreement 4410 4412 := RowAgreement.join atomic_4411_agreement atomic_4412_agreement
private def within_034_055 : CountSegment 1 10 13120 4408 4412 2010 2010 := CountSegment.join within_034_053 within_034_054
private theorem within_034_055_agreement : RowAgreement 4408 4412 := RowAgreement.join within_034_053_agreement within_034_054_agreement
private def within_034_056 : CountSegment 1 10 13120 4412 4414 2010 2012 := CountSegment.join atomic_4413 atomic_4414
private theorem within_034_056_agreement : RowAgreement 4412 4414 := RowAgreement.join atomic_4413_agreement atomic_4414_agreement
private def within_034_057 : CountSegment 1 10 13120 4414 4416 2012 2012 := CountSegment.join atomic_4415 atomic_4416
private theorem within_034_057_agreement : RowAgreement 4414 4416 := RowAgreement.join atomic_4415_agreement atomic_4416_agreement
private def within_034_058 : CountSegment 1 10 13120 4412 4416 2010 2012 := CountSegment.join within_034_056 within_034_057
private theorem within_034_058_agreement : RowAgreement 4412 4416 := RowAgreement.join within_034_056_agreement within_034_057_agreement
private def within_034_059 : CountSegment 1 10 13120 4408 4416 2010 2012 := CountSegment.join within_034_055 within_034_058
private theorem within_034_059_agreement : RowAgreement 4408 4416 := RowAgreement.join within_034_055_agreement within_034_058_agreement
private def within_034_060 : CountSegment 1 10 13120 4400 4416 2006 2012 := CountSegment.join within_034_052 within_034_059
private theorem within_034_060_agreement : RowAgreement 4400 4416 := RowAgreement.join within_034_052_agreement within_034_059_agreement
private def within_034_061 : CountSegment 1 10 13120 4384 4416 2000 2012 := CountSegment.join within_034_045 within_034_060
private theorem within_034_061_agreement : RowAgreement 4384 4416 := RowAgreement.join within_034_045_agreement within_034_060_agreement
private def within_034_062 : CountSegment 1 10 13120 4352 4416 1983 2012 := CountSegment.join within_034_030 within_034_061
private theorem within_034_062_agreement : RowAgreement 4352 4416 := RowAgreement.join within_034_030_agreement within_034_061_agreement
private def within_034_063 : CountSegment 1 10 13120 4416 4418 2012 2013 := CountSegment.join atomic_4417 atomic_4418
private theorem within_034_063_agreement : RowAgreement 4416 4418 := RowAgreement.join atomic_4417_agreement atomic_4418_agreement
private def within_034_064 : CountSegment 1 10 13120 4418 4420 2013 2013 := CountSegment.join atomic_4419 atomic_4420
private theorem within_034_064_agreement : RowAgreement 4418 4420 := RowAgreement.join atomic_4419_agreement atomic_4420_agreement
private def within_034_065 : CountSegment 1 10 13120 4416 4420 2012 2013 := CountSegment.join within_034_063 within_034_064
private theorem within_034_065_agreement : RowAgreement 4416 4420 := RowAgreement.join within_034_063_agreement within_034_064_agreement
private def within_034_066 : CountSegment 1 10 13120 4420 4422 2013 2013 := CountSegment.join atomic_4421 atomic_4422
private theorem within_034_066_agreement : RowAgreement 4420 4422 := RowAgreement.join atomic_4421_agreement atomic_4422_agreement
private def within_034_067 : CountSegment 1 10 13120 4422 4424 2013 2014 := CountSegment.join atomic_4423 atomic_4424
private theorem within_034_067_agreement : RowAgreement 4422 4424 := RowAgreement.join atomic_4423_agreement atomic_4424_agreement
private def within_034_068 : CountSegment 1 10 13120 4420 4424 2013 2014 := CountSegment.join within_034_066 within_034_067
private theorem within_034_068_agreement : RowAgreement 4420 4424 := RowAgreement.join within_034_066_agreement within_034_067_agreement
private def within_034_069 : CountSegment 1 10 13120 4416 4424 2012 2014 := CountSegment.join within_034_065 within_034_068
private theorem within_034_069_agreement : RowAgreement 4416 4424 := RowAgreement.join within_034_065_agreement within_034_068_agreement
private def within_034_070 : CountSegment 1 10 13120 4424 4426 2014 2014 := CountSegment.join atomic_4425 atomic_4426
private theorem within_034_070_agreement : RowAgreement 4424 4426 := RowAgreement.join atomic_4425_agreement atomic_4426_agreement
private def within_034_071 : CountSegment 1 10 13120 4426 4428 2014 2015 := CountSegment.join atomic_4427 atomic_4428
private theorem within_034_071_agreement : RowAgreement 4426 4428 := RowAgreement.join atomic_4427_agreement atomic_4428_agreement
private def within_034_072 : CountSegment 1 10 13120 4424 4428 2014 2015 := CountSegment.join within_034_070 within_034_071
private theorem within_034_072_agreement : RowAgreement 4424 4428 := RowAgreement.join within_034_070_agreement within_034_071_agreement
private def within_034_073 : CountSegment 1 10 13120 4428 4430 2015 2015 := CountSegment.join atomic_4429 atomic_4430
private theorem within_034_073_agreement : RowAgreement 4428 4430 := RowAgreement.join atomic_4429_agreement atomic_4430_agreement
private def within_034_074 : CountSegment 1 10 13120 4430 4432 2015 2015 := CountSegment.join atomic_4431 atomic_4432
private theorem within_034_074_agreement : RowAgreement 4430 4432 := RowAgreement.join atomic_4431_agreement atomic_4432_agreement
private def within_034_075 : CountSegment 1 10 13120 4428 4432 2015 2015 := CountSegment.join within_034_073 within_034_074
private theorem within_034_075_agreement : RowAgreement 4428 4432 := RowAgreement.join within_034_073_agreement within_034_074_agreement
private def within_034_076 : CountSegment 1 10 13120 4424 4432 2014 2015 := CountSegment.join within_034_072 within_034_075
private theorem within_034_076_agreement : RowAgreement 4424 4432 := RowAgreement.join within_034_072_agreement within_034_075_agreement
private def within_034_077 : CountSegment 1 10 13120 4416 4432 2012 2015 := CountSegment.join within_034_069 within_034_076
private theorem within_034_077_agreement : RowAgreement 4416 4432 := RowAgreement.join within_034_069_agreement within_034_076_agreement
private def within_034_078 : CountSegment 1 10 13120 4432 4434 2015 2015 := CountSegment.join atomic_4433 atomic_4434
private theorem within_034_078_agreement : RowAgreement 4432 4434 := RowAgreement.join atomic_4433_agreement atomic_4434_agreement
private def within_034_079 : CountSegment 1 10 13120 4434 4436 2015 2015 := CountSegment.join atomic_4435 atomic_4436
private theorem within_034_079_agreement : RowAgreement 4434 4436 := RowAgreement.join atomic_4435_agreement atomic_4436_agreement
private def within_034_080 : CountSegment 1 10 13120 4432 4436 2015 2015 := CountSegment.join within_034_078 within_034_079
private theorem within_034_080_agreement : RowAgreement 4432 4436 := RowAgreement.join within_034_078_agreement within_034_079_agreement
private def within_034_081 : CountSegment 1 10 13120 4436 4438 2015 2016 := CountSegment.join atomic_4437 atomic_4438
private theorem within_034_081_agreement : RowAgreement 4436 4438 := RowAgreement.join atomic_4437_agreement atomic_4438_agreement
private def within_034_082 : CountSegment 1 10 13120 4438 4440 2016 2017 := CountSegment.join atomic_4439 atomic_4440
private theorem within_034_082_agreement : RowAgreement 4438 4440 := RowAgreement.join atomic_4439_agreement atomic_4440_agreement
private def within_034_083 : CountSegment 1 10 13120 4436 4440 2015 2017 := CountSegment.join within_034_081 within_034_082
private theorem within_034_083_agreement : RowAgreement 4436 4440 := RowAgreement.join within_034_081_agreement within_034_082_agreement
private def within_034_084 : CountSegment 1 10 13120 4432 4440 2015 2017 := CountSegment.join within_034_080 within_034_083
private theorem within_034_084_agreement : RowAgreement 4432 4440 := RowAgreement.join within_034_080_agreement within_034_083_agreement
private def within_034_085 : CountSegment 1 10 13120 4440 4442 2017 2018 := CountSegment.join atomic_4441 atomic_4442
private theorem within_034_085_agreement : RowAgreement 4440 4442 := RowAgreement.join atomic_4441_agreement atomic_4442_agreement
private def within_034_086 : CountSegment 1 10 13120 4442 4444 2018 2019 := CountSegment.join atomic_4443 atomic_4444
private theorem within_034_086_agreement : RowAgreement 4442 4444 := RowAgreement.join atomic_4443_agreement atomic_4444_agreement
private def within_034_087 : CountSegment 1 10 13120 4440 4444 2017 2019 := CountSegment.join within_034_085 within_034_086
private theorem within_034_087_agreement : RowAgreement 4440 4444 := RowAgreement.join within_034_085_agreement within_034_086_agreement
private def within_034_088 : CountSegment 1 10 13120 4444 4446 2019 2019 := CountSegment.join atomic_4445 atomic_4446
private theorem within_034_088_agreement : RowAgreement 4444 4446 := RowAgreement.join atomic_4445_agreement atomic_4446_agreement
private def within_034_089 : CountSegment 1 10 13120 4446 4448 2019 2020 := CountSegment.join atomic_4447 atomic_4448
private theorem within_034_089_agreement : RowAgreement 4446 4448 := RowAgreement.join atomic_4447_agreement atomic_4448_agreement
private def within_034_090 : CountSegment 1 10 13120 4444 4448 2019 2020 := CountSegment.join within_034_088 within_034_089
private theorem within_034_090_agreement : RowAgreement 4444 4448 := RowAgreement.join within_034_088_agreement within_034_089_agreement
private def within_034_091 : CountSegment 1 10 13120 4440 4448 2017 2020 := CountSegment.join within_034_087 within_034_090
private theorem within_034_091_agreement : RowAgreement 4440 4448 := RowAgreement.join within_034_087_agreement within_034_090_agreement
private def within_034_092 : CountSegment 1 10 13120 4432 4448 2015 2020 := CountSegment.join within_034_084 within_034_091
private theorem within_034_092_agreement : RowAgreement 4432 4448 := RowAgreement.join within_034_084_agreement within_034_091_agreement
private def within_034_093 : CountSegment 1 10 13120 4416 4448 2012 2020 := CountSegment.join within_034_077 within_034_092
private theorem within_034_093_agreement : RowAgreement 4416 4448 := RowAgreement.join within_034_077_agreement within_034_092_agreement
private def within_034_094 : CountSegment 1 10 13120 4448 4450 2020 2021 := CountSegment.join atomic_4449 atomic_4450
private theorem within_034_094_agreement : RowAgreement 4448 4450 := RowAgreement.join atomic_4449_agreement atomic_4450_agreement
private def within_034_095 : CountSegment 1 10 13120 4450 4452 2021 2021 := CountSegment.join atomic_4451 atomic_4452
private theorem within_034_095_agreement : RowAgreement 4450 4452 := RowAgreement.join atomic_4451_agreement atomic_4452_agreement
private def within_034_096 : CountSegment 1 10 13120 4448 4452 2020 2021 := CountSegment.join within_034_094 within_034_095
private theorem within_034_096_agreement : RowAgreement 4448 4452 := RowAgreement.join within_034_094_agreement within_034_095_agreement
private def within_034_097 : CountSegment 1 10 13120 4452 4454 2021 2021 := CountSegment.join atomic_4453 atomic_4454
private theorem within_034_097_agreement : RowAgreement 4452 4454 := RowAgreement.join atomic_4453_agreement atomic_4454_agreement
private def within_034_098 : CountSegment 1 10 13120 4454 4456 2021 2022 := CountSegment.join atomic_4455 atomic_4456
private theorem within_034_098_agreement : RowAgreement 4454 4456 := RowAgreement.join atomic_4455_agreement atomic_4456_agreement
private def within_034_099 : CountSegment 1 10 13120 4452 4456 2021 2022 := CountSegment.join within_034_097 within_034_098
private theorem within_034_099_agreement : RowAgreement 4452 4456 := RowAgreement.join within_034_097_agreement within_034_098_agreement
private def within_034_100 : CountSegment 1 10 13120 4448 4456 2020 2022 := CountSegment.join within_034_096 within_034_099
private theorem within_034_100_agreement : RowAgreement 4448 4456 := RowAgreement.join within_034_096_agreement within_034_099_agreement
private def within_034_101 : CountSegment 1 10 13120 4456 4458 2022 2022 := CountSegment.join atomic_4457 atomic_4458
private theorem within_034_101_agreement : RowAgreement 4456 4458 := RowAgreement.join atomic_4457_agreement atomic_4458_agreement
private def within_034_102 : CountSegment 1 10 13120 4458 4460 2022 2022 := CountSegment.join atomic_4459 atomic_4460
private theorem within_034_102_agreement : RowAgreement 4458 4460 := RowAgreement.join atomic_4459_agreement atomic_4460_agreement
private def within_034_103 : CountSegment 1 10 13120 4456 4460 2022 2022 := CountSegment.join within_034_101 within_034_102
private theorem within_034_103_agreement : RowAgreement 4456 4460 := RowAgreement.join within_034_101_agreement within_034_102_agreement
private def within_034_104 : CountSegment 1 10 13120 4460 4462 2022 2022 := CountSegment.join atomic_4461 atomic_4462
private theorem within_034_104_agreement : RowAgreement 4460 4462 := RowAgreement.join atomic_4461_agreement atomic_4462_agreement
private def within_034_105 : CountSegment 1 10 13120 4462 4464 2022 2024 := CountSegment.join atomic_4463 atomic_4464
private theorem within_034_105_agreement : RowAgreement 4462 4464 := RowAgreement.join atomic_4463_agreement atomic_4464_agreement
private def within_034_106 : CountSegment 1 10 13120 4460 4464 2022 2024 := CountSegment.join within_034_104 within_034_105
private theorem within_034_106_agreement : RowAgreement 4460 4464 := RowAgreement.join within_034_104_agreement within_034_105_agreement
private def within_034_107 : CountSegment 1 10 13120 4456 4464 2022 2024 := CountSegment.join within_034_103 within_034_106
private theorem within_034_107_agreement : RowAgreement 4456 4464 := RowAgreement.join within_034_103_agreement within_034_106_agreement
private def within_034_108 : CountSegment 1 10 13120 4448 4464 2020 2024 := CountSegment.join within_034_100 within_034_107
private theorem within_034_108_agreement : RowAgreement 4448 4464 := RowAgreement.join within_034_100_agreement within_034_107_agreement
private def within_034_109 : CountSegment 1 10 13120 4464 4466 2024 2025 := CountSegment.join atomic_4465 atomic_4466
private theorem within_034_109_agreement : RowAgreement 4464 4466 := RowAgreement.join atomic_4465_agreement atomic_4466_agreement
private def within_034_110 : CountSegment 1 10 13120 4466 4468 2025 2025 := CountSegment.join atomic_4467 atomic_4468
private theorem within_034_110_agreement : RowAgreement 4466 4468 := RowAgreement.join atomic_4467_agreement atomic_4468_agreement
private def within_034_111 : CountSegment 1 10 13120 4464 4468 2024 2025 := CountSegment.join within_034_109 within_034_110
private theorem within_034_111_agreement : RowAgreement 4464 4468 := RowAgreement.join within_034_109_agreement within_034_110_agreement
private def within_034_112 : CountSegment 1 10 13120 4468 4470 2025 2026 := CountSegment.join atomic_4469 atomic_4470
private theorem within_034_112_agreement : RowAgreement 4468 4470 := RowAgreement.join atomic_4469_agreement atomic_4470_agreement
private def within_034_113 : CountSegment 1 10 13120 4470 4472 2026 2027 := CountSegment.join atomic_4471 atomic_4472
private theorem within_034_113_agreement : RowAgreement 4470 4472 := RowAgreement.join atomic_4471_agreement atomic_4472_agreement
private def within_034_114 : CountSegment 1 10 13120 4468 4472 2025 2027 := CountSegment.join within_034_112 within_034_113
private theorem within_034_114_agreement : RowAgreement 4468 4472 := RowAgreement.join within_034_112_agreement within_034_113_agreement
private def within_034_115 : CountSegment 1 10 13120 4464 4472 2024 2027 := CountSegment.join within_034_111 within_034_114
private theorem within_034_115_agreement : RowAgreement 4464 4472 := RowAgreement.join within_034_111_agreement within_034_114_agreement
private def within_034_116 : CountSegment 1 10 13120 4472 4474 2027 2029 := CountSegment.join atomic_4473 atomic_4474
private theorem within_034_116_agreement : RowAgreement 4472 4474 := RowAgreement.join atomic_4473_agreement atomic_4474_agreement
private def within_034_117 : CountSegment 1 10 13120 4474 4476 2029 2029 := CountSegment.join atomic_4475 atomic_4476
private theorem within_034_117_agreement : RowAgreement 4474 4476 := RowAgreement.join atomic_4475_agreement atomic_4476_agreement
private def within_034_118 : CountSegment 1 10 13120 4472 4476 2027 2029 := CountSegment.join within_034_116 within_034_117
private theorem within_034_118_agreement : RowAgreement 4472 4476 := RowAgreement.join within_034_116_agreement within_034_117_agreement
private def within_034_119 : CountSegment 1 10 13120 4476 4478 2029 2030 := CountSegment.join atomic_4477 atomic_4478
private theorem within_034_119_agreement : RowAgreement 4476 4478 := RowAgreement.join atomic_4477_agreement atomic_4478_agreement
private def within_034_120 : CountSegment 1 10 13120 4478 4480 2030 2030 := CountSegment.join atomic_4479 atomic_4480
private theorem within_034_120_agreement : RowAgreement 4478 4480 := RowAgreement.join atomic_4479_agreement atomic_4480_agreement
private def within_034_121 : CountSegment 1 10 13120 4476 4480 2029 2030 := CountSegment.join within_034_119 within_034_120
private theorem within_034_121_agreement : RowAgreement 4476 4480 := RowAgreement.join within_034_119_agreement within_034_120_agreement
private def within_034_122 : CountSegment 1 10 13120 4472 4480 2027 2030 := CountSegment.join within_034_118 within_034_121
private theorem within_034_122_agreement : RowAgreement 4472 4480 := RowAgreement.join within_034_118_agreement within_034_121_agreement
private def within_034_123 : CountSegment 1 10 13120 4464 4480 2024 2030 := CountSegment.join within_034_115 within_034_122
private theorem within_034_123_agreement : RowAgreement 4464 4480 := RowAgreement.join within_034_115_agreement within_034_122_agreement
private def within_034_124 : CountSegment 1 10 13120 4448 4480 2020 2030 := CountSegment.join within_034_108 within_034_123
private theorem within_034_124_agreement : RowAgreement 4448 4480 := RowAgreement.join within_034_108_agreement within_034_123_agreement
private def within_034_125 : CountSegment 1 10 13120 4416 4480 2012 2030 := CountSegment.join within_034_093 within_034_124
private theorem within_034_125_agreement : RowAgreement 4416 4480 := RowAgreement.join within_034_093_agreement within_034_124_agreement
private def within_034_126 : CountSegment 1 10 13120 4352 4480 1983 2030 := CountSegment.join within_034_062 within_034_125
private theorem within_034_126_agreement : RowAgreement 4352 4480 := RowAgreement.join within_034_062_agreement within_034_125_agreement
def segment_034 : CountSegment 1 10 13120 4352 4480 1983 2030 := within_034_126
theorem agreement_034 : RowAgreement 4352 4480 := within_034_126_agreement
end ProjectEulerSolutions.P58Numerical
