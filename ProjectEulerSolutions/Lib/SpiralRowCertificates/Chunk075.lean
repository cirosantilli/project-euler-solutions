import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9601 : RowCertificate 9601 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9601 1) := by change Nat.Prime 368736007; exact ProjectEulerSolutions.P58PrimeNodes.p_368736007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9601 2) := by change ¬ Nat.Prime 368716805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73743361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9601 3) := by change ¬ Nat.Prime 368697603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122899201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9601 : row 9601 = 1 := (row_9601.checked).trans rfl
theorem model_count_9601 : modelRow 9601 = 1 := by decide
theorem modelrow_agrees_9601 : modelRow 9601 = row 9601 := model_count_9601.trans row_count_9601.symm
def row_9602 : RowCertificate 9602 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9602 1) := by change ¬ Nat.Prime 368812821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122937607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9602 2) := by change ¬ Nat.Prime 368793617; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6045797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9602 3) := by change ¬ Nat.Prime 368774413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52682059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9602 : row 9602 = 0 := (row_9602.checked).trans rfl
theorem model_count_9602 : modelRow 9602 = 0 := by decide
theorem modelrow_agrees_9602 : modelRow 9602 = row 9602 := model_count_9602.trans row_count_9602.symm
def row_9603 : RowCertificate 9603 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9603 1) := by change Nat.Prime 368889643; exact ProjectEulerSolutions.P58PrimeNodes.p_368889643.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9603 2) := by change ¬ Nat.Prime 368870437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28374649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9603 3) := by change ¬ Nat.Prime 368851231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52693033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9603 : row 9603 = 1 := (row_9603.checked).trans rfl
theorem model_count_9603 : modelRow 9603 = 1 := by decide
theorem modelrow_agrees_9603 : modelRow 9603 = row 9603 := model_count_9603.trans row_count_9603.symm
def row_9604 : RowCertificate 9604 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9604 1) := by change Nat.Prime 368966473; exact ProjectEulerSolutions.P58PrimeNodes.p_368966473.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9604 2) := by change ¬ Nat.Prime 368947265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73789453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9604 3) := by change ¬ Nat.Prime 368928057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 122976019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9604 : row 9604 = 1 := (row_9604.checked).trans rfl
theorem model_count_9604 : modelRow 9604 = 1 := by decide
theorem modelrow_agrees_9604 : modelRow 9604 = row 9604 := model_count_9604.trans row_count_9604.symm
def row_9605 : RowCertificate 9605 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9605 1) := by change ¬ Nat.Prime 369043311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123014437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9605 2) := by change ¬ Nat.Prime 369024101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12724969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9605 3) := by change ¬ Nat.Prime 369004891; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2443741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9605 : row 9605 = 0 := (row_9605.checked).trans rfl
theorem model_count_9605 : modelRow 9605 = 0 := by decide
theorem modelrow_agrees_9605 : modelRow 9605 = row 9605 := model_count_9605.trans row_count_9605.symm
def row_9606 : RowCertificate 9606 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9606 1) := by change ¬ Nat.Prime 369120157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52731451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9606 2) := by change ¬ Nat.Prime 369100945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73820189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9606 3) := by change Nat.Prime 369081733; exact ProjectEulerSolutions.P58PrimeNodes.p_369081733.prime
    exact decide_eq_true prime
theorem row_count_9606 : row 9606 = 1 := (row_9606.checked).trans rfl
theorem model_count_9606 : modelRow 9606 = 1 := by decide
theorem modelrow_agrees_9606 : modelRow 9606 = row 9606 := model_count_9606.trans row_count_9606.symm
def row_9607 : RowCertificate 9607 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9607 1) := by change ¬ Nat.Prime 369197011; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11909581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9607 2) := by change ¬ Nat.Prime 369177797; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21716341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9607 3) := by change ¬ Nat.Prime 369158583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123052861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9607 : row 9607 = 0 := (row_9607.checked).trans rfl
theorem model_count_9607 : modelRow 9607 = 0 := by decide
theorem modelrow_agrees_9607 : modelRow 9607 = row 9607 := model_count_9607.trans row_count_9607.symm
def row_9608 : RowCertificate 9608 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9608 1) := by change ¬ Nat.Prime 369273873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123091291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9608 2) := by change ¬ Nat.Prime 369254657; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6967069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9608 3) := by change ¬ Nat.Prime 369235441; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1913137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9608 : row 9608 = 0 := (row_9608.checked).trans rfl
theorem model_count_9608 : modelRow 9608 = 0 := by decide
theorem modelrow_agrees_9608 : modelRow 9608 = row 9608 := model_count_9608.trans row_count_9608.symm
def row_9609 : RowCertificate 9609 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9609 1) := by change ¬ Nat.Prime 369350743; exact PrattCertificates.composite_of_product (factor := 5179) (cofactor := 71317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9609 2) := by change ¬ Nat.Prime 369331525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73866305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9609 3) := by change ¬ Nat.Prime 369312307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52758901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9609 : row 9609 = 0 := (row_9609.checked).trans rfl
theorem model_count_9609 : modelRow 9609 = 0 := by decide
theorem modelrow_agrees_9609 : modelRow 9609 = row 9609 := model_count_9609.trans row_count_9609.symm
def row_9610 : RowCertificate 9610 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9610 1) := by change ¬ Nat.Prime 369427621; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19443559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9610 2) := by change ¬ Nat.Prime 369408401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9009961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9610 3) := by change ¬ Nat.Prime 369389181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123129727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9610 : row 9610 = 0 := (row_9610.checked).trans rfl
theorem model_count_9610 : modelRow 9610 = 0 := by decide
theorem modelrow_agrees_9610 : modelRow 9610 = row 9610 := model_count_9610.trans row_count_9610.symm
def row_9611 : RowCertificate 9611 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9611 1) := by change ¬ Nat.Prime 369504507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123168169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9611 2) := by change ¬ Nat.Prime 369485285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73897057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9611 3) := by change Nat.Prime 369466063; exact ProjectEulerSolutions.P58PrimeNodes.p_369466063.prime
    exact decide_eq_true prime
theorem row_count_9611 : row 9611 = 1 := (row_9611.checked).trans rfl
theorem model_count_9611 : modelRow 9611 = 1 := by decide
theorem modelrow_agrees_9611 : modelRow 9611 = row 9611 := model_count_9611.trans row_count_9611.symm
def row_9612 : RowCertificate 9612 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9612 1) := by change ¬ Nat.Prime 369581401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52797343) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9612 2) := by change Nat.Prime 369562177; exact ProjectEulerSolutions.P58PrimeNodes.p_369562177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9612 3) := by change ¬ Nat.Prime 369542953; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28426381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9612 : row 9612 = 1 := (row_9612.checked).trans rfl
theorem model_count_9612 : modelRow 9612 = 1 := by decide
theorem modelrow_agrees_9612 : modelRow 9612 = row 9612 := model_count_9612.trans row_count_9612.symm
def row_9613 : RowCertificate 9613 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9613 1) := by change ¬ Nat.Prime 369658303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52808329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9613 2) := by change ¬ Nat.Prime 369639077; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6059657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9613 3) := by change ¬ Nat.Prime 369619851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123206617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9613 : row 9613 = 0 := (row_9613.checked).trans rfl
theorem model_count_9613 : modelRow 9613 = 0 := by decide
theorem modelrow_agrees_9613 : modelRow 9613 = row 9613 := model_count_9613.trans row_count_9613.symm
def row_9614 : RowCertificate 9614 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9614 1) := by change ¬ Nat.Prime 369735213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123245071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9614 2) := by change ¬ Nat.Prime 369715985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73943197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9614 3) := by change ¬ Nat.Prime 369696757; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8597599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9614 : row 9614 = 0 := (row_9614.checked).trans rfl
theorem model_count_9614 : modelRow 9614 = 0 := by decide
theorem modelrow_agrees_9614 : modelRow 9614 = row 9614 := model_count_9614.trans row_count_9614.symm
def row_9615 : RowCertificate 9615 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9615 1) := by change ¬ Nat.Prime 369812131; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28447087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9615 2) := by change Nat.Prime 369792901; exact ProjectEulerSolutions.P58PrimeNodes.p_369792901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9615 3) := by change ¬ Nat.Prime 369773671; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9993883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9615 : row 9615 = 1 := (row_9615.checked).trans rfl
theorem model_count_9615 : modelRow 9615 = 1 := by decide
theorem modelrow_agrees_9615 : modelRow 9615 = row 9615 := model_count_9615.trans row_count_9615.symm
def row_9616 : RowCertificate 9616 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9616 1) := by change Nat.Prime 369889057; exact ProjectEulerSolutions.P58PrimeNodes.p_369889057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9616 2) := by change ¬ Nat.Prime 369869825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73973965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9616 3) := by change ¬ Nat.Prime 369850593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123283531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9616 : row 9616 = 1 := (row_9616.checked).trans rfl
theorem model_count_9616 : modelRow 9616 = 1 := by decide
theorem modelrow_agrees_9616 : modelRow 9616 = row 9616 := model_count_9616.trans row_count_9616.symm
def row_9617 : RowCertificate 9617 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9617 1) := by change ¬ Nat.Prime 369965991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123321997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9617 2) := by change ¬ Nat.Prime 369946757; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9998561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9617 3) := by change ¬ Nat.Prime 369927523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52846789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9617 : row 9617 = 0 := (row_9617.checked).trans rfl
theorem model_count_9617 : modelRow 9617 = 0 := by decide
theorem modelrow_agrees_9617 : modelRow 9617 = row 9617 := model_count_9617.trans row_count_9617.symm
def row_9618 : RowCertificate 9618 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9618 1) := by change ¬ Nat.Prime 370042933; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28464841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9618 2) := by change Nat.Prime 370023697; exact ProjectEulerSolutions.P58PrimeNodes.p_370023697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9618 3) := by change ¬ Nat.Prime 370004461; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19473919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9618 : row 9618 = 1 := (row_9618.checked).trans rfl
theorem model_count_9618 : modelRow 9618 = 1 := by decide
theorem modelrow_agrees_9618 : modelRow 9618 = row 9618 := model_count_9618.trans row_count_9618.symm
def row_9619 : RowCertificate 9619 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9619 1) := by change ¬ Nat.Prime 370119883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52874269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9619 2) := by change ¬ Nat.Prime 370100645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74020129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9619 3) := by change ¬ Nat.Prime 370081407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123360469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9619 : row 9619 = 0 := (row_9619.checked).trans rfl
theorem model_count_9619 : modelRow 9619 = 0 := by decide
theorem modelrow_agrees_9619 : modelRow 9619 = row 9619 := model_count_9619.trans row_count_9619.symm
def row_9620 : RowCertificate 9620 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9620 1) := by change ¬ Nat.Prime 370196841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123398947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9620 2) := by change ¬ Nat.Prime 370177601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21775153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9620 3) := by change ¬ Nat.Prime 370158361; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19482019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9620 : row 9620 = 0 := (row_9620.checked).trans rfl
theorem model_count_9620 : modelRow 9620 = 0 := by decide
theorem modelrow_agrees_9620 : modelRow 9620 = row 9620 := model_count_9620.trans row_count_9620.symm
def row_9621 : RowCertificate 9621 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9621 1) := by change Nat.Prime 370273807; exact ProjectEulerSolutions.P58PrimeNodes.p_370273807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9621 2) := by change ¬ Nat.Prime 370254565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74050913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9621 3) := by change ¬ Nat.Prime 370235323; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3816859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9621 : row 9621 = 1 := (row_9621.checked).trans rfl
theorem model_count_9621 : modelRow 9621 = 1 := by decide
theorem modelrow_agrees_9621 : modelRow 9621 = row 9621 := model_count_9621.trans row_count_9621.symm
def row_9622 : RowCertificate 9622 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9622 1) := by change Nat.Prime 370350781; exact ProjectEulerSolutions.P58PrimeNodes.p_370350781.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9622 2) := by change ¬ Nat.Prime 370331537; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12770053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9622 3) := by change ¬ Nat.Prime 370312293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123437431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9622 : row 9622 = 1 := (row_9622.checked).trans rfl
theorem model_count_9622 : modelRow 9622 = 1 := by decide
theorem modelrow_agrees_9622 : modelRow 9622 = row 9622 := model_count_9622.trans row_count_9622.symm
def row_9623 : RowCertificate 9623 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9623 1) := by change ¬ Nat.Prime 370427763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123475921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9623 2) := by change ¬ Nat.Prime 370408517; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10011041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9623 3) := by change ¬ Nat.Prime 370389271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52912753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9623 : row 9623 = 0 := (row_9623.checked).trans rfl
theorem model_count_9623 : modelRow 9623 = 0 := by decide
theorem modelrow_agrees_9623 : modelRow 9623 = row 9623 := model_count_9623.trans row_count_9623.symm
def row_9624 : RowCertificate 9624 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9624 1) := by change Nat.Prime 370504753; exact ProjectEulerSolutions.P58PrimeNodes.p_370504753.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9624 2) := by change ¬ Nat.Prime 370485505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74097101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9624 3) := by change ¬ Nat.Prime 370466257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52923751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9624 : row 9624 = 1 := (row_9624.checked).trans rfl
theorem model_count_9624 : modelRow 9624 = 1 := by decide
theorem modelrow_agrees_9624 : modelRow 9624 = row 9624 := model_count_9624.trans row_count_9624.symm
def row_9625 : RowCertificate 9625 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9625 1) := by change ¬ Nat.Prime 370581751; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10015723) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9625 2) := by change ¬ Nat.Prime 370562501; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 1590397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9625 3) := by change ¬ Nat.Prime 370543251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123514417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9625 : row 9625 = 0 := (row_9625.checked).trans rfl
theorem model_count_9625 : modelRow 9625 = 0 := by decide
theorem modelrow_agrees_9625 : modelRow 9625 = row 9625 := model_count_9625.trans row_count_9625.symm
def row_9626 : RowCertificate 9626 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9626 1) := by change ¬ Nat.Prime 370658757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123552919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9626 2) := by change ¬ Nat.Prime 370639505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74127901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9626 3) := by change Nat.Prime 370620253; exact ProjectEulerSolutions.P58PrimeNodes.p_370620253.prime
    exact decide_eq_true prime
theorem row_count_9626 : row 9626 = 1 := (row_9626.checked).trans rfl
theorem model_count_9626 : modelRow 9626 = 1 := by decide
theorem modelrow_agrees_9626 : modelRow 9626 = row 9626 := model_count_9626.trans row_count_9626.symm
def row_9627 : RowCertificate 9627 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9627 1) := by change ¬ Nat.Prime 370735771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52962253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9627 2) := by change Nat.Prime 370716517; exact ProjectEulerSolutions.P58PrimeNodes.p_370716517.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9627 3) := by change Nat.Prime 370697263; exact ProjectEulerSolutions.P58PrimeNodes.p_370697263.prime
    exact decide_eq_true prime
theorem row_count_9627 : row 9627 = 2 := (row_9627.checked).trans rfl
theorem model_count_9627 : modelRow 9627 = 2 := by decide
theorem modelrow_agrees_9627 : modelRow 9627 = row 9627 := model_count_9627.trans row_count_9627.symm
def row_9628 : RowCertificate 9628 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9628 1) := by change ¬ Nat.Prime 370812793; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28524061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9628 2) := by change ¬ Nat.Prime 370793537; exact PrattCertificates.composite_of_product (factor := 14593) (cofactor := 25409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9628 3) := by change ¬ Nat.Prime 370774281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123591427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9628 : row 9628 = 0 := (row_9628.checked).trans rfl
theorem model_count_9628 : modelRow 9628 = 0 := by decide
theorem modelrow_agrees_9628 : modelRow 9628 = row 9628 := model_count_9628.trans row_count_9628.symm
def row_9629 : RowCertificate 9629 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9629 1) := by change ¬ Nat.Prime 370889823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123629941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9629 2) := by change ¬ Nat.Prime 370870565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74174113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9629 3) := by change ¬ Nat.Prime 370851307; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8624449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9629 : row 9629 = 0 := (row_9629.checked).trans rfl
theorem model_count_9629 : modelRow 9629 = 0 := by decide
theorem modelrow_agrees_9629 : modelRow 9629 = row 9629 := model_count_9629.trans row_count_9629.symm
def row_9630 : RowCertificate 9630 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9630 1) := by change Nat.Prime 370966861; exact ProjectEulerSolutions.P58PrimeNodes.p_370966861.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9630 2) := by change ¬ Nat.Prime 370947601; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3403189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9630 3) := by change ¬ Nat.Prime 370928341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52989763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9630 : row 9630 = 1 := (row_9630.checked).trans rfl
theorem model_count_9630 : modelRow 9630 = 1 := by decide
theorem modelrow_agrees_9630 : modelRow 9630 = row 9630 := model_count_9630.trans row_count_9630.symm
def row_9631 : RowCertificate 9631 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9631 1) := by change ¬ Nat.Prime 371043907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28541839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9631 2) := by change ¬ Nat.Prime 371024645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74204929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9631 3) := by change ¬ Nat.Prime 371005383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123668461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9631 : row 9631 = 0 := (row_9631.checked).trans rfl
theorem model_count_9631 : modelRow 9631 = 0 := by decide
theorem modelrow_agrees_9631 : modelRow 9631 = row 9631 := model_count_9631.trans row_count_9631.symm
def row_9632 : RowCertificate 9632 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9632 1) := by change ¬ Nat.Prime 371120961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123706987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9632 2) := by change ¬ Nat.Prime 371101697; exact PrattCertificates.composite_of_product (factor := 6133) (cofactor := 60509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9632 3) := by change ¬ Nat.Prime 371082433; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5083321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9632 : row 9632 = 0 := (row_9632.checked).trans rfl
theorem model_count_9632 : modelRow 9632 = 0 := by decide
theorem modelrow_agrees_9632 : modelRow 9632 = row 9632 := model_count_9632.trans row_count_9632.symm
def row_9633 : RowCertificate 9633 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9633 1) := by change ¬ Nat.Prime 371198023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53028289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9633 2) := by change ¬ Nat.Prime 371178757; exact PrattCertificates.composite_of_product (factor := 3361) (cofactor := 110437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9633 3) := by change Nat.Prime 371159491; exact ProjectEulerSolutions.P58PrimeNodes.p_371159491.prime
    exact decide_eq_true prime
theorem row_count_9633 : row 9633 = 1 := (row_9633.checked).trans rfl
theorem model_count_9633 : modelRow 9633 = 1 := by decide
theorem modelrow_agrees_9633 : modelRow 9633 = row 9633 := model_count_9633.trans row_count_9633.symm
def row_9634 : RowCertificate 9634 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9634 1) := by change ¬ Nat.Prime 371275093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53039299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9634 2) := by change ¬ Nat.Prime 371255825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74251165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9634 3) := by change ¬ Nat.Prime 371236557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123745519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9634 : row 9634 = 0 := (row_9634.checked).trans rfl
theorem model_count_9634 : modelRow 9634 = 0 := by decide
theorem modelrow_agrees_9634 : modelRow 9634 = row 9634 := model_count_9634.trans row_count_9634.symm
def row_9635 : RowCertificate 9635 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9635 1) := by change ¬ Nat.Prime 371352171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123784057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9635 2) := by change ¬ Nat.Prime 371332901; exact PrattCertificates.composite_of_product (factor := 10253) (cofactor := 36217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9635 3) := by change ¬ Nat.Prime 371313631; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28562587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9635 : row 9635 = 0 := (row_9635.checked).trans rfl
theorem model_count_9635 : modelRow 9635 = 0 := by decide
theorem modelrow_agrees_9635 : modelRow 9635 = row 9635 := model_count_9635.trans row_count_9635.symm
def row_9636 : RowCertificate 9636 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9636 1) := by change Nat.Prime 371429257; exact ProjectEulerSolutions.P58PrimeNodes.p_371429257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9636 2) := by change ¬ Nat.Prime 371409985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74281997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9636 3) := by change Nat.Prime 371390713; exact ProjectEulerSolutions.P58PrimeNodes.p_371390713.prime
    exact decide_eq_true prime
theorem row_count_9636 : row 9636 = 2 := (row_9636.checked).trans rfl
theorem model_count_9636 : modelRow 9636 = 2 := by decide
theorem modelrow_agrees_9636 : modelRow 9636 = row 9636 := model_count_9636.trans row_count_9636.symm
def row_9637 : RowCertificate 9637 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9637 1) := by change Nat.Prime 371506351; exact ProjectEulerSolutions.P58PrimeNodes.p_371506351.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9637 2) := by change ¬ Nat.Prime 371487077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28575929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9637 3) := by change ¬ Nat.Prime 371467803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123822601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9637 : row 9637 = 1 := (row_9637.checked).trans rfl
theorem model_count_9637 : modelRow 9637 = 1 := by decide
theorem modelrow_agrees_9637 : modelRow 9637 = row 9637 := model_count_9637.trans row_count_9637.symm
def row_9638 : RowCertificate 9638 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9638 1) := by change ¬ Nat.Prime 371583453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123861151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9638 2) := by change Nat.Prime 371564177; exact ProjectEulerSolutions.P58PrimeNodes.p_371564177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9638 3) := by change ¬ Nat.Prime 371544901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53077843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9638 : row 9638 = 1 := (row_9638.checked).trans rfl
theorem model_count_9638 : modelRow 9638 = 1 := by decide
theorem modelrow_agrees_9638 : modelRow 9638 = row 9638 := model_count_9638.trans row_count_9638.symm
def row_9639 : RowCertificate 9639 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9639 1) := by change ¬ Nat.Prime 371660563; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2673817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9639 2) := by change ¬ Nat.Prime 371641285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74328257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9639 3) := by change ¬ Nat.Prime 371622007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19559053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9639 : row 9639 = 0 := (row_9639.checked).trans rfl
theorem model_count_9639 : modelRow 9639 = 0 := by decide
theorem modelrow_agrees_9639 : modelRow 9639 = row 9639 := model_count_9639.trans row_count_9639.symm
def row_9640 : RowCertificate 9640 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9640 1) := by change ¬ Nat.Prime 371737681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53105383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9640 2) := by change ¬ Nat.Prime 371718401; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 2713273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9640 3) := by change ¬ Nat.Prime 371699121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123899707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9640 : row 9640 = 0 := (row_9640.checked).trans rfl
theorem model_count_9640 : modelRow 9640 = 0 := by decide
theorem modelrow_agrees_9640 : modelRow 9640 = row 9640 := model_count_9640.trans row_count_9640.symm
def row_9641 : RowCertificate 9641 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9641 1) := by change ¬ Nat.Prime 371814807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123938269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9641 2) := by change ¬ Nat.Prime 371795525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74359105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9641 3) := by change ¬ Nat.Prime 371776243; exact PrattCertificates.composite_of_product (factor := 8689) (cofactor := 42787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9641 : row 9641 = 0 := (row_9641.checked).trans rfl
theorem model_count_9641 : modelRow 9641 = 0 := by decide
theorem modelrow_agrees_9641 : modelRow 9641 = row 9641 := model_count_9641.trans row_count_9641.symm
def row_9642 : RowCertificate 9642 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9642 1) := by change ¬ Nat.Prime 371891941; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2928283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9642 2) := by change ¬ Nat.Prime 371872657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28605589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9642 3) := by change Nat.Prime 371853373; exact ProjectEulerSolutions.P58PrimeNodes.p_371853373.prime
    exact decide_eq_true prime
theorem row_count_9642 : row 9642 = 1 := (row_9642.checked).trans rfl
theorem model_count_9642 : modelRow 9642 = 1 := by decide
theorem modelrow_agrees_9642 : modelRow 9642 = row 9642 := model_count_9642.trans row_count_9642.symm
def row_9643 : RowCertificate 9643 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9643 1) := by change ¬ Nat.Prime 371969083; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 859051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9643 2) := by change ¬ Nat.Prime 371949797; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 1382713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9643 3) := by change ¬ Nat.Prime 371930511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123976837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9643 : row 9643 = 0 := (row_9643.checked).trans rfl
theorem model_count_9643 : modelRow 9643 = 0 := by decide
theorem modelrow_agrees_9643 : modelRow 9643 = row 9643 := model_count_9643.trans row_count_9643.symm
def row_9644 : RowCertificate 9644 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9644 1) := by change ¬ Nat.Prime 372046233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124015411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9644 2) := by change ¬ Nat.Prime 372026945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74405389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9644 3) := by change ¬ Nat.Prime 372007657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53143951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9644 : row 9644 = 0 := (row_9644.checked).trans rfl
theorem model_count_9644 : modelRow 9644 = 0 := by decide
theorem modelrow_agrees_9644 : modelRow 9644 = row 9644 := model_count_9644.trans row_count_9644.symm
def row_9645 : RowCertificate 9645 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9645 1) := by change ¬ Nat.Prime 372123391; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 711517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9645 2) := by change Nat.Prime 372104101; exact ProjectEulerSolutions.P58PrimeNodes.p_372104101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9645 3) := by change ¬ Nat.Prime 372084811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53154973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9645 : row 9645 = 1 := (row_9645.checked).trans rfl
theorem model_count_9645 : modelRow 9645 = 1 := by decide
theorem modelrow_agrees_9645 : modelRow 9645 = row 9645 := model_count_9645.trans row_count_9645.symm
def row_9646 : RowCertificate 9646 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9646 1) := by change ¬ Nat.Prime 372200557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19589503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9646 2) := by change ¬ Nat.Prime 372181265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74436253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9646 3) := by change ¬ Nat.Prime 372161973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124053991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9646 : row 9646 = 0 := (row_9646.checked).trans rfl
theorem model_count_9646 : modelRow 9646 = 0 := by decide
theorem modelrow_agrees_9646 : modelRow 9646 = row 9646 := model_count_9646.trans row_count_9646.symm
def row_9647 : RowCertificate 9647 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9647 1) := by change ¬ Nat.Prime 372277731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124092577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9647 2) := by change Nat.Prime 372258437; exact ProjectEulerSolutions.P58PrimeNodes.p_372258437.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9647 3) := by change Nat.Prime 372239143; exact ProjectEulerSolutions.P58PrimeNodes.p_372239143.prime
    exact decide_eq_true prime
theorem row_count_9647 : row 9647 = 2 := (row_9647.checked).trans rfl
theorem model_count_9647 : modelRow 9647 = 2 := by decide
theorem modelrow_agrees_9647 : modelRow 9647 = row 9647 := model_count_9647.trans row_count_9647.symm
def row_9648 : RowCertificate 9648 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9648 1) := by change ¬ Nat.Prime 372354913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53193559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9648 2) := by change ¬ Nat.Prime 372335617; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 2152229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9648 3) := by change ¬ Nat.Prime 372316321; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28639717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9648 : row 9648 = 0 := (row_9648.checked).trans rfl
theorem model_count_9648 : modelRow 9648 = 0 := by decide
theorem modelrow_agrees_9648 : modelRow 9648 = row 9648 := model_count_9648.trans row_count_9648.symm
def row_9649 : RowCertificate 9649 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9649 1) := by change ¬ Nat.Prime 372432103; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2372179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9649 2) := by change ¬ Nat.Prime 372412805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74482561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9649 3) := by change ¬ Nat.Prime 372393507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124131169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9649 : row 9649 = 0 := (row_9649.checked).trans rfl
theorem model_count_9649 : modelRow 9649 = 0 := by decide
theorem modelrow_agrees_9649 : modelRow 9649 = row 9649 := model_count_9649.trans row_count_9649.symm
def row_9650 : RowCertificate 9650 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9650 1) := by change ¬ Nat.Prime 372509301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124169767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9650 2) := by change ¬ Nat.Prime 372490001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28653077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9650 3) := by change ¬ Nat.Prime 372470701; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4714819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9650 : row 9650 = 0 := (row_9650.checked).trans rfl
theorem model_count_9650 : modelRow 9650 = 0 := by decide
theorem modelrow_agrees_9650 : modelRow 9650 = row 9650 := model_count_9650.trans row_count_9650.symm
def row_9651 : RowCertificate 9651 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9651 1) := by change ¬ Nat.Prime 372586507; exact PrattCertificates.composite_of_product (factor := 17239) (cofactor := 21613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9651 2) := by change ¬ Nat.Prime 372567205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74513441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9651 3) := by change ¬ Nat.Prime 372547903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53221129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9651 : row 9651 = 0 := (row_9651.checked).trans rfl
theorem model_count_9651 : modelRow 9651 = 0 := by decide
theorem modelrow_agrees_9651 : modelRow 9651 = row 9651 := model_count_9651.trans row_count_9651.symm
def row_9652 : RowCertificate 9652 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9652 1) := by change ¬ Nat.Prime 372663721; exact PrattCertificates.composite_of_product (factor := 571) (cofactor := 652651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9652 2) := by change Nat.Prime 372644417; exact ProjectEulerSolutions.P58PrimeNodes.p_372644417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9652 3) := by change ¬ Nat.Prime 372625113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124208371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9652 : row 9652 = 1 := (row_9652.checked).trans rfl
theorem model_count_9652 : modelRow 9652 = 1 := by decide
theorem modelrow_agrees_9652 : modelRow 9652 = row 9652 := model_count_9652.trans row_count_9652.symm
def row_9653 : RowCertificate 9653 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9653 1) := by change ¬ Nat.Prime 372740943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124246981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9653 2) := by change ¬ Nat.Prime 372721637; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 860789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9653 3) := by change Nat.Prime 372702331; exact ProjectEulerSolutions.P58PrimeNodes.p_372702331.prime
    exact decide_eq_true prime
theorem row_count_9653 : row 9653 = 1 := (row_9653.checked).trans rfl
theorem model_count_9653 : modelRow 9653 = 1 := by decide
theorem modelrow_agrees_9653 : modelRow 9653 = row 9653 := model_count_9653.trans row_count_9653.symm
def row_9654 : RowCertificate 9654 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9654 1) := by change ¬ Nat.Prime 372818173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53259739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9654 2) := by change ¬ Nat.Prime 372798865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74559773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9654 3) := by change ¬ Nat.Prime 372779557; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12025147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9654 : row 9654 = 0 := (row_9654.checked).trans rfl
theorem model_count_9654 : modelRow 9654 = 0 := by decide
theorem modelrow_agrees_9654 : modelRow 9654 = row 9654 := model_count_9654.trans row_count_9654.symm
def row_9655 : RowCertificate 9655 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9655 1) := by change ¬ Nat.Prime 372895411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53270773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9655 2) := by change ¬ Nat.Prime 372876101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28682777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9655 3) := by change ¬ Nat.Prime 372856791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124285597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9655 : row 9655 = 0 := (row_9655.checked).trans rfl
theorem model_count_9655 : modelRow 9655 = 0 := by decide
theorem modelrow_agrees_9655 : modelRow 9655 = row 9655 := model_count_9655.trans row_count_9655.symm
def row_9656 : RowCertificate 9656 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9656 1) := by change ¬ Nat.Prime 372972657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124324219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9656 2) := by change ¬ Nat.Prime 372953345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74590669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9656 3) := by change ¬ Nat.Prime 372934033; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19628107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9656 : row 9656 = 0 := (row_9656.checked).trans rfl
theorem model_count_9656 : modelRow 9656 = 0 := by decide
theorem modelrow_agrees_9656 : modelRow 9656 = row 9656 := model_count_9656.trans row_count_9656.symm
def row_9657 : RowCertificate 9657 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9657 1) := by change ¬ Nat.Prime 373049911; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28696147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9657 2) := by change ¬ Nat.Prime 373030597; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 809177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9657 3) := by change ¬ Nat.Prime 373011283; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8674681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9657 : row 9657 = 0 := (row_9657.checked).trans rfl
theorem model_count_9657 : modelRow 9657 = 0 := by decide
theorem modelrow_agrees_9657 : modelRow 9657 = row 9657 := model_count_9657.trans row_count_9657.symm
def row_9658 : RowCertificate 9658 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9658 1) := by change Nat.Prime 373127173; exact ProjectEulerSolutions.P58PrimeNodes.p_373127173.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9658 2) := by change ¬ Nat.Prime 373107857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21947521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9658 3) := by change ¬ Nat.Prime 373088541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124362847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9658 : row 9658 = 1 := (row_9658.checked).trans rfl
theorem model_count_9658 : modelRow 9658 = 1 := by decide
theorem modelrow_agrees_9658 : modelRow 9658 = row 9658 := model_count_9658.trans row_count_9658.symm
def row_9659 : RowCertificate 9659 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9659 1) := by change ¬ Nat.Prime 373204443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124401481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9659 2) := by change ¬ Nat.Prime 373185125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74637025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9659 3) := by change ¬ Nat.Prime 373165807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53309401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9659 : row 9659 = 0 := (row_9659.checked).trans rfl
theorem model_count_9659 : modelRow 9659 = 0 := by decide
theorem modelrow_agrees_9659 : modelRow 9659 = row 9659 := model_count_9659.trans row_count_9659.symm
def row_9660 : RowCertificate 9660 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9660 1) := by change ¬ Nat.Prime 373281721; exact PrattCertificates.composite_of_product (factor := 14197) (cofactor := 26293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9660 2) := by change ¬ Nat.Prime 373262401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10088173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9660 3) := by change Nat.Prime 373243081; exact ProjectEulerSolutions.P58PrimeNodes.p_373243081.prime
    exact decide_eq_true prime
theorem row_count_9660 : row 9660 = 1 := (row_9660.checked).trans rfl
theorem model_count_9660 : modelRow 9660 = 1 := by decide
theorem modelrow_agrees_9660 : modelRow 9660 = row 9660 := model_count_9660.trans row_count_9660.symm
def row_9661 : RowCertificate 9661 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9661 1) := by change ¬ Nat.Prime 373359007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53337001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9661 2) := by change ¬ Nat.Prime 373339685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74667937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9661 3) := by change ¬ Nat.Prime 373320363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124440121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9661 : row 9661 = 0 := (row_9661.checked).trans rfl
theorem model_count_9661 : modelRow 9661 = 0 := by decide
theorem modelrow_agrees_9661 : modelRow 9661 = row 9661 := model_count_9661.trans row_count_9661.symm
def row_9662 : RowCertificate 9662 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9662 1) := by change ¬ Nat.Prime 373436301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124478767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9662 2) := by change Nat.Prime 373416977; exact ProjectEulerSolutions.P58PrimeNodes.p_373416977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9662 3) := by change ¬ Nat.Prime 373397653; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6121273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9662 : row 9662 = 1 := (row_9662.checked).trans rfl
theorem model_count_9662 : modelRow 9662 = 1 := by decide
theorem modelrow_agrees_9662 : modelRow 9662 = row 9662 := model_count_9662.trans row_count_9662.symm
def row_9663 : RowCertificate 9663 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9663 1) := by change ¬ Nat.Prime 373513603; exact PrattCertificates.composite_of_product (factor := 17359) (cofactor := 21517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9663 2) := by change ¬ Nat.Prime 373494277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28730329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9663 3) := by change ¬ Nat.Prime 373474951; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5574253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9663 : row 9663 = 0 := (row_9663.checked).trans rfl
theorem model_count_9663 : modelRow 9663 = 0 := by decide
theorem modelrow_agrees_9663 : modelRow 9663 = row 9663 := model_count_9663.trans row_count_9663.symm
def row_9664 : RowCertificate 9664 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9664 1) := by change ¬ Nat.Prime 373590913; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 349477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9664 2) := by change ¬ Nat.Prime 373571585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74714317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9664 3) := by change ¬ Nat.Prime 373552257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124517419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9664 : row 9664 = 0 := (row_9664.checked).trans rfl
theorem model_count_9664 : modelRow 9664 = 0 := by decide
theorem modelrow_agrees_9664 : modelRow 9664 = row 9664 := model_count_9664.trans row_count_9664.symm
def row_9665 : RowCertificate 9665 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9665 1) := by change ¬ Nat.Prime 373668231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124556077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9665 2) := by change ¬ Nat.Prime 373648901; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1348913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9665 3) := by change ¬ Nat.Prime 373629571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53375653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9665 : row 9665 = 0 := (row_9665.checked).trans rfl
theorem model_count_9665 : modelRow 9665 = 0 := by decide
theorem modelrow_agrees_9665 : modelRow 9665 = row 9665 := model_count_9665.trans row_count_9665.symm
def row_9666 : RowCertificate 9666 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9666 1) := by change Nat.Prime 373745557; exact ProjectEulerSolutions.P58PrimeNodes.p_373745557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9666 2) := by change ¬ Nat.Prime 373726225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74745245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9666 3) := by change ¬ Nat.Prime 373706893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53386699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9666 : row 9666 = 1 := (row_9666.checked).trans rfl
theorem model_count_9666 : modelRow 9666 = 1 := by decide
theorem modelrow_agrees_9666 : modelRow 9666 = row 9666 := model_count_9666.trans row_count_9666.symm
def row_9667 : RowCertificate 9667 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9667 1) := by change ¬ Nat.Prime 373822891; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28755607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9667 2) := by change Nat.Prime 373803557; exact ProjectEulerSolutions.P58PrimeNodes.p_373803557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9667 3) := by change ¬ Nat.Prime 373784223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124594741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9667 : row 9667 = 1 := (row_9667.checked).trans rfl
theorem model_count_9667 : modelRow 9667 = 1 := by decide
theorem modelrow_agrees_9667 : modelRow 9667 = row 9667 := model_count_9667.trans row_count_9667.symm
def row_9668 : RowCertificate 9668 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9668 1) := by change ¬ Nat.Prime 373900233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124633411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9668 2) := by change ¬ Nat.Prime 373880897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28760069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9668 3) := by change Nat.Prime 373861561; exact ProjectEulerSolutions.P58PrimeNodes.p_373861561.prime
    exact decide_eq_true prime
theorem row_count_9668 : row 9668 = 1 := (row_9668.checked).trans rfl
theorem model_count_9668 : modelRow 9668 = 1 := by decide
theorem modelrow_agrees_9668 : modelRow 9668 = row 9668 := model_count_9668.trans row_count_9668.symm
def row_9669 : RowCertificate 9669 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9669 1) := by change ¬ Nat.Prime 373977583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53425369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9669 2) := by change ¬ Nat.Prime 373958245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74791649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9669 3) := by change ¬ Nat.Prime 373938907; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 1109611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9669 : row 9669 = 0 := (row_9669.checked).trans rfl
theorem model_count_9669 : modelRow 9669 = 0 := by decide
theorem modelrow_agrees_9669 : modelRow 9669 = row 9669 := model_count_9669.trans row_count_9669.symm
def row_9670 : RowCertificate 9670 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9670 1) := by change ¬ Nat.Prime 374054941; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28773457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9670 2) := by change Nat.Prime 374035601; exact ProjectEulerSolutions.P58PrimeNodes.p_374035601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9670 3) := by change ¬ Nat.Prime 374016261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124672087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9670 : row 9670 = 1 := (row_9670.checked).trans rfl
theorem model_count_9670 : modelRow 9670 = 1 := by decide
theorem modelrow_agrees_9670 : modelRow 9670 = row 9670 := model_count_9670.trans row_count_9670.symm
def row_9671 : RowCertificate 9671 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9671 1) := by change ¬ Nat.Prime 374132307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124710769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9671 2) := by change ¬ Nat.Prime 374112965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74822593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9671 3) := by change ¬ Nat.Prime 374093623; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 261787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9671 : row 9671 = 0 := (row_9671.checked).trans rfl
theorem model_count_9671 : modelRow 9671 = 0 := by decide
theorem modelrow_agrees_9671 : modelRow 9671 = row 9671 := model_count_9671.trans row_count_9671.symm
def row_9672 : RowCertificate 9672 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9672 1) := by change ¬ Nat.Prime 374209681; exact PrattCertificates.composite_of_product (factor := 11701) (cofactor := 31981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9672 2) := by change ¬ Nat.Prime 374190337; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1552657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9672 3) := by change ¬ Nat.Prime 374170993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53452999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9672 : row 9672 = 0 := (row_9672.checked).trans rfl
theorem model_count_9672 : modelRow 9672 = 0 := by decide
theorem modelrow_agrees_9672 : modelRow 9672 = row 9672 := model_count_9672.trans row_count_9672.symm
def row_9673 : RowCertificate 9673 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9673 1) := by change ¬ Nat.Prime 374287063; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 691843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9673 2) := by change ¬ Nat.Prime 374267717; exact PrattCertificates.composite_of_product (factor := 13721) (cofactor := 27277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9673 3) := by change ¬ Nat.Prime 374248371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124749457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9673 : row 9673 = 0 := (row_9673.checked).trans rfl
theorem model_count_9673 : modelRow 9673 = 0 := by decide
theorem modelrow_agrees_9673 : modelRow 9673 = row 9673 := model_count_9673.trans row_count_9673.symm
def row_9674 : RowCertificate 9674 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9674 1) := by change ¬ Nat.Prime 374364453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124788151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9674 2) := by change ¬ Nat.Prime 374345105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74869021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9674 3) := by change ¬ Nat.Prime 374325757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28794289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9674 : row 9674 = 0 := (row_9674.checked).trans rfl
theorem model_count_9674 : modelRow 9674 = 0 := by decide
theorem modelrow_agrees_9674 : modelRow 9674 = row 9674 := model_count_9674.trans row_count_9674.symm
def row_9675 : RowCertificate 9675 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9675 1) := by change ¬ Nat.Prime 374441851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53491693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9675 2) := by change ¬ Nat.Prime 374422501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22024853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9675 3) := by change ¬ Nat.Prime 374403151; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19705429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9675 : row 9675 = 0 := (row_9675.checked).trans rfl
theorem model_count_9675 : modelRow 9675 = 0 := by decide
theorem modelrow_agrees_9675 : modelRow 9675 = row 9675 := model_count_9675.trans row_count_9675.symm
def row_9676 : RowCertificate 9676 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9676 1) := by change ¬ Nat.Prime 374519257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53502751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9676 2) := by change ¬ Nat.Prime 374499905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74899981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9676 3) := by change ¬ Nat.Prime 374480553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124826851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9676 : row 9676 = 0 := (row_9676.checked).trans rfl
theorem model_count_9676 : modelRow 9676 = 0 := by decide
theorem modelrow_agrees_9676 : modelRow 9676 = row 9676 := model_count_9676.trans row_count_9676.symm
def row_9677 : RowCertificate 9677 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9677 1) := by change ¬ Nat.Prime 374596671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124865557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9677 2) := by change ¬ Nat.Prime 374577317; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 2734141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9677 3) := by change ¬ Nat.Prime 374557963; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28812151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9677 : row 9677 = 0 := (row_9677.checked).trans rfl
theorem model_count_9677 : modelRow 9677 = 0 := by decide
theorem modelrow_agrees_9677 : modelRow 9677 = row 9677 := model_count_9677.trans row_count_9677.symm
def row_9678 : RowCertificate 9678 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9678 1) := by change ¬ Nat.Prime 374674093; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8713351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9678 2) := by change ¬ Nat.Prime 374654737; exact PrattCertificates.composite_of_product (factor := 2789) (cofactor := 134333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9678 3) := by change Nat.Prime 374635381; exact ProjectEulerSolutions.P58PrimeNodes.p_374635381.prime
    exact decide_eq_true prime
theorem row_count_9678 : row 9678 = 1 := (row_9678.checked).trans rfl
theorem model_count_9678 : modelRow 9678 = 1 := by decide
theorem modelrow_agrees_9678 : modelRow 9678 = row 9678 := model_count_9678.trans row_count_9678.symm
def row_9679 : RowCertificate 9679 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9679 1) := by change ¬ Nat.Prime 374751523; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 1220689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9679 2) := by change ¬ Nat.Prime 374732165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74946433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9679 3) := by change ¬ Nat.Prime 374712807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124904269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9679 : row 9679 = 0 := (row_9679.checked).trans rfl
theorem model_count_9679 : modelRow 9679 = 0 := by decide
theorem modelrow_agrees_9679 : modelRow 9679 = row 9679 := model_count_9679.trans row_count_9679.symm
def row_9680 : RowCertificate 9680 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9680 1) := by change ¬ Nat.Prime 374828961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124942987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9680 2) := by change ¬ Nat.Prime 374809601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12924469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9680 3) := by change ¬ Nat.Prime 374790241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53541463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9680 : row 9680 = 0 := (row_9680.checked).trans rfl
theorem model_count_9680 : modelRow 9680 = 0 := by decide
theorem modelrow_agrees_9680 : modelRow 9680 = row 9680 := model_count_9680.trans row_count_9680.symm
def row_9681 : RowCertificate 9681 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9681 1) := by change ¬ Nat.Prime 374906407; exact PrattCertificates.composite_of_product (factor := 6481) (cofactor := 57847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9681 2) := by change ¬ Nat.Prime 374887045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74977409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9681 3) := by change ¬ Nat.Prime 374867683; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10131559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9681 : row 9681 = 0 := (row_9681.checked).trans rfl
theorem model_count_9681 : modelRow 9681 = 0 := by decide
theorem modelrow_agrees_9681 : modelRow 9681 = row 9681 := model_count_9681.trans row_count_9681.symm
def row_9682 : RowCertificate 9682 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9682 1) := by change ¬ Nat.Prime 374983861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53569123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9682 2) := by change Nat.Prime 374964497; exact ProjectEulerSolutions.P58PrimeNodes.p_374964497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9682 3) := by change ¬ Nat.Prime 374945133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124981711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9682 : row 9682 = 1 := (row_9682.checked).trans rfl
theorem model_count_9682 : modelRow 9682 = 1 := by decide
theorem modelrow_agrees_9682 : modelRow 9682 = row 9682 := model_count_9682.trans row_count_9682.symm
def row_9683 : RowCertificate 9683 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9683 1) := by change ¬ Nat.Prime 375061323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125020441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9683 2) := by change ¬ Nat.Prime 375041957; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 916973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9683 3) := by change ¬ Nat.Prime 375022591; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 511627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9683 : row 9683 = 0 := (row_9683.checked).trans rfl
theorem model_count_9683 : modelRow 9683 = 0 := by decide
theorem modelrow_agrees_9683 : modelRow 9683 = row 9683 := model_count_9683.trans row_count_9683.symm
def row_9684 : RowCertificate 9684 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9684 1) := by change ¬ Nat.Prime 375138793; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19744147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9684 2) := by change ¬ Nat.Prime 375119425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75023885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9684 3) := by change ¬ Nat.Prime 375100057; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 1022071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9684 : row 9684 = 0 := (row_9684.checked).trans rfl
theorem model_count_9684 : modelRow 9684 = 0 := by decide
theorem modelrow_agrees_9684 : modelRow 9684 = row 9684 := model_count_9684.trans row_count_9684.symm
def row_9685 : RowCertificate 9685 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9685 1) := by change ¬ Nat.Prime 375216271; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1638499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9685 2) := by change Nat.Prime 375196901; exact ProjectEulerSolutions.P58PrimeNodes.p_375196901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9685 3) := by change ¬ Nat.Prime 375177531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125059177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9685 : row 9685 = 1 := (row_9685.checked).trans rfl
theorem model_count_9685 : modelRow 9685 = 1 := by decide
theorem modelrow_agrees_9685 : modelRow 9685 = row 9685 := model_count_9685.trans row_count_9685.symm
def row_9686 : RowCertificate 9686 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9686 1) := by change ¬ Nat.Prime 375293757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125097919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9686 2) := by change ¬ Nat.Prime 375274385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75054877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9686 3) := by change ¬ Nat.Prime 375255013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53607859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9686 : row 9686 = 0 := (row_9686.checked).trans rfl
theorem model_count_9686 : modelRow 9686 = 0 := by decide
theorem modelrow_agrees_9686 : modelRow 9686 = row 9686 := model_count_9686.trans row_count_9686.symm
def row_9687 : RowCertificate 9687 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9687 1) := by change ¬ Nat.Prime 375371251; exact PrattCertificates.composite_of_product (factor := 1471) (cofactor := 255181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9687 2) := by change ¬ Nat.Prime 375351877; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 835973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9687 3) := by change ¬ Nat.Prime 375332503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53618929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9687 : row 9687 = 0 := (row_9687.checked).trans rfl
theorem model_count_9687 : modelRow 9687 = 0 := by decide
theorem modelrow_agrees_9687 : modelRow 9687 = row 9687 := model_count_9687.trans row_count_9687.symm
def row_9688 : RowCertificate 9688 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9688 1) := by change ¬ Nat.Prime 375448753; exact PrattCertificates.composite_of_product (factor := 17209) (cofactor := 21817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9688 2) := by change ¬ Nat.Prime 375429377; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22084081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9688 3) := by change ¬ Nat.Prime 375410001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125136667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9688 : row 9688 = 0 := (row_9688.checked).trans rfl
theorem model_count_9688 : modelRow 9688 = 0 := by decide
theorem modelrow_agrees_9688 : modelRow 9688 = row 9688 := model_count_9688.trans row_count_9688.symm
def row_9689 : RowCertificate 9689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9689 1) := by change ¬ Nat.Prime 375526263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125175421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9689 2) := by change ¬ Nat.Prime 375506885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75101377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9689 3) := by change ¬ Nat.Prime 375487507; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10148311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9689 : row 9689 = 0 := (row_9689.checked).trans rfl
theorem model_count_9689 : modelRow 9689 = 0 := by decide
theorem modelrow_agrees_9689 : modelRow 9689 = row 9689 := model_count_9689.trans row_count_9689.symm
def row_9690 : RowCertificate 9690 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9690 1) := by change ¬ Nat.Prime 375603781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53657683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9690 2) := by change Nat.Prime 375584401; exact ProjectEulerSolutions.P58PrimeNodes.p_375584401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9690 3) := by change ¬ Nat.Prime 375565021; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28889617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9690 : row 9690 = 1 := (row_9690.checked).trans rfl
theorem model_count_9690 : modelRow 9690 = 1 := by decide
theorem modelrow_agrees_9690 : modelRow 9690 = row 9690 := model_count_9690.trans row_count_9690.symm
def row_9691 : RowCertificate 9691 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9691 1) := by change Nat.Prime 375681307; exact ProjectEulerSolutions.P58PrimeNodes.p_375681307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9691 2) := by change ¬ Nat.Prime 375661925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75132385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9691 3) := by change ¬ Nat.Prime 375642543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125214181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9691 : row 9691 = 1 := (row_9691.checked).trans rfl
theorem model_count_9691 : modelRow 9691 = 1 := by decide
theorem modelrow_agrees_9691 : modelRow 9691 = row 9691 := model_count_9691.trans row_count_9691.symm
def row_9692 : RowCertificate 9692 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9692 1) := by change ¬ Nat.Prime 375758841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125252947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9692 2) := by change ¬ Nat.Prime 375739457; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22102321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9692 3) := by change ¬ Nat.Prime 375720073; exact PrattCertificates.composite_of_product (factor := 2617) (cofactor := 143569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9692 : row 9692 = 0 := (row_9692.checked).trans rfl
theorem model_count_9692 : modelRow 9692 = 0 := by decide
theorem modelrow_agrees_9692 : modelRow 9692 = row 9692 := model_count_9692.trans row_count_9692.symm
def row_9693 : RowCertificate 9693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9693 1) := by change ¬ Nat.Prime 375836383; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28910491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9693 2) := by change ¬ Nat.Prime 375816997; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2076337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9693 3) := by change ¬ Nat.Prime 375797611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53685373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9693 : row 9693 = 0 := (row_9693.checked).trans rfl
theorem model_count_9693 : modelRow 9693 = 0 := by decide
theorem modelrow_agrees_9693 : modelRow 9693 = row 9693 := model_count_9693.trans row_count_9693.symm
def row_9694 : RowCertificate 9694 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9694 1) := by change Nat.Prime 375913933; exact ProjectEulerSolutions.P58PrimeNodes.p_375913933.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9694 2) := by change ¬ Nat.Prime 375894545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75178909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9694 3) := by change ¬ Nat.Prime 375875157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125291719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9694 : row 9694 = 1 := (row_9694.checked).trans rfl
theorem model_count_9694 : modelRow 9694 = 1 := by decide
theorem modelrow_agrees_9694 : modelRow 9694 = row 9694 := model_count_9694.trans row_count_9694.symm
def row_9695 : RowCertificate 9695 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9695 1) := by change ¬ Nat.Prime 375991491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125330497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9695 2) := by change ¬ Nat.Prime 375972101; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 947033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9695 3) := by change ¬ Nat.Prime 375952711; exact PrattCertificates.composite_of_product (factor := 3739) (cofactor := 100549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9695 : row 9695 = 0 := (row_9695.checked).trans rfl
theorem model_count_9695 : modelRow 9695 = 0 := by decide
theorem modelrow_agrees_9695 : modelRow 9695 = row 9695 := model_count_9695.trans row_count_9695.symm
def row_9696 : RowCertificate 9696 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9696 1) := by change ¬ Nat.Prime 376069057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53724151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9696 2) := by change ¬ Nat.Prime 376049665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75209933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9696 3) := by change ¬ Nat.Prime 376030273; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19791067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9696 : row 9696 = 0 := (row_9696.checked).trans rfl
theorem model_count_9696 : modelRow 9696 = 0 := by decide
theorem modelrow_agrees_9696 : modelRow 9696 = row 9696 := model_count_9696.trans row_count_9696.symm
def row_9697 : RowCertificate 9697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9697 1) := by change ¬ Nat.Prime 376146631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53735233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9697 2) := by change ¬ Nat.Prime 376127237; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10165601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9697 3) := by change ¬ Nat.Prime 376107843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125369281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9697 : row 9697 = 0 := (row_9697.checked).trans rfl
theorem model_count_9697 : modelRow 9697 = 0 := by decide
theorem modelrow_agrees_9697 : modelRow 9697 = row 9697 := model_count_9697.trans row_count_9697.symm
def row_9698 : RowCertificate 9698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9698 1) := by change ¬ Nat.Prime 376224213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125408071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9698 2) := by change ¬ Nat.Prime 376204817; exact PrattCertificates.composite_of_product (factor := 4373) (cofactor := 86029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9698 3) := by change ¬ Nat.Prime 376185421; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1890379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9698 : row 9698 = 0 := (row_9698.checked).trans rfl
theorem model_count_9698 : modelRow 9698 = 0 := by decide
theorem modelrow_agrees_9698 : modelRow 9698 = row 9698 := model_count_9698.trans row_count_9698.symm
def row_9699 : RowCertificate 9699 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9699 1) := by change ¬ Nat.Prime 376301803; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10170319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9699 2) := by change ¬ Nat.Prime 376282405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75256481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9699 3) := by change ¬ Nat.Prime 376263007; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1783237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9699 : row 9699 = 0 := (row_9699.checked).trans rfl
theorem model_count_9699 : modelRow 9699 = 0 := by decide
theorem modelrow_agrees_9699 : modelRow 9699 = row 9699 := model_count_9699.trans row_count_9699.symm
def row_9700 : RowCertificate 9700 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9700 1) := by change ¬ Nat.Prime 376379401; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12141271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9700 2) := by change Nat.Prime 376360001; exact ProjectEulerSolutions.P58PrimeNodes.p_376360001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9700 3) := by change ¬ Nat.Prime 376340601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125446867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9700 : row 9700 = 1 := (row_9700.checked).trans rfl
theorem model_count_9700 : modelRow 9700 = 1 := by decide
theorem modelrow_agrees_9700 : modelRow 9700 = row 9700 := model_count_9700.trans row_count_9700.symm
def row_9701 : RowCertificate 9701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9701 1) := by change ¬ Nat.Prime 376457007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125485669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9701 2) := by change ¬ Nat.Prime 376437605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75287521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9701 3) := by change ¬ Nat.Prime 376418203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53774029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9701 : row 9701 = 0 := (row_9701.checked).trans rfl
theorem model_count_9701 : modelRow 9701 = 0 := by decide
theorem modelrow_agrees_9701 : modelRow 9701 = row 9701 := model_count_9701.trans row_count_9701.symm
def row_9702 : RowCertificate 9702 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9702 1) := by change ¬ Nat.Prime 376534621; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 652573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9702 2) := by change ¬ Nat.Prime 376515217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28962709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9702 3) := by change Nat.Prime 376495813; exact ProjectEulerSolutions.P58PrimeNodes.p_376495813.prime
    exact decide_eq_true prime
theorem row_count_9702 : row 9702 = 1 := (row_9702.checked).trans rfl
theorem model_count_9702 : modelRow 9702 = 1 := by decide
theorem modelrow_agrees_9702 : modelRow 9702 = row 9702 := model_count_9702.trans row_count_9702.symm
def row_9703 : RowCertificate 9703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9703 1) := by change ¬ Nat.Prime 376612243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53801749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9703 2) := by change ¬ Nat.Prime 376592837; exact PrattCertificates.composite_of_product (factor := 4093) (cofactor := 92009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9703 3) := by change ¬ Nat.Prime 376573431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125524477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9703 : row 9703 = 0 := (row_9703.checked).trans rfl
theorem model_count_9703 : modelRow 9703 = 0 := by decide
theorem modelrow_agrees_9703 : modelRow 9703 = row 9703 := model_count_9703.trans row_count_9703.symm
def row_9704 : RowCertificate 9704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9704 1) := by change ¬ Nat.Prime 376689873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125563291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9704 2) := by change ¬ Nat.Prime 376670465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75334093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9704 3) := by change Nat.Prime 376651057; exact ProjectEulerSolutions.P58PrimeNodes.p_376651057.prime
    exact decide_eq_true prime
theorem row_count_9704 : row 9704 = 1 := (row_9704.checked).trans rfl
theorem model_count_9704 : modelRow 9704 = 1 := by decide
theorem modelrow_agrees_9704 : modelRow 9704 = row 9704 := model_count_9704.trans row_count_9704.symm
def row_9705 : RowCertificate 9705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9705 1) := by change ¬ Nat.Prime 376767511; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19829869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9705 2) := by change ¬ Nat.Prime 376748101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22161653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9705 3) := by change ¬ Nat.Prime 376728691; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5160667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9705 : row 9705 = 0 := (row_9705.checked).trans rfl
theorem model_count_9705 : modelRow 9705 = 0 := by decide
theorem modelrow_agrees_9705 : modelRow 9705 = row 9705 := model_count_9705.trans row_count_9705.symm
def row_9706 : RowCertificate 9706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9706 1) := by change ¬ Nat.Prime 376845157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28988089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9706 2) := by change ¬ Nat.Prime 376825745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75365149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9706 3) := by change ¬ Nat.Prime 376806333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125602111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9706 : row 9706 = 0 := (row_9706.checked).trans rfl
theorem model_count_9706 : modelRow 9706 = 0 := by decide
theorem modelrow_agrees_9706 : modelRow 9706 = row 9706 := model_count_9706.trans row_count_9706.symm
def row_9707 : RowCertificate 9707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9707 1) := by change ¬ Nat.Prime 376922811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125640937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9707 2) := by change ¬ Nat.Prime 376903397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28992569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9707 3) := by change ¬ Nat.Prime 376883983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53840569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9707 : row 9707 = 0 := (row_9707.checked).trans rfl
theorem model_count_9707 : modelRow 9707 = 0 := by decide
theorem modelrow_agrees_9707 : modelRow 9707 = row 9707 := model_count_9707.trans row_count_9707.symm
def row_9708 : RowCertificate 9708 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9708 1) := by change Nat.Prime 377000473; exact ProjectEulerSolutions.P58PrimeNodes.p_377000473.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9708 2) := by change Nat.Prime 376981057; exact ProjectEulerSolutions.P58PrimeNodes.p_376981057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9708 3) := by change ¬ Nat.Prime 376961641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53851663) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9708 : row 9708 = 2 := (row_9708.checked).trans rfl
theorem model_count_9708 : modelRow 9708 = 2 := by decide
theorem modelrow_agrees_9708 : modelRow 9708 = row 9708 := model_count_9708.trans row_count_9708.symm
def row_9709 : RowCertificate 9709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9709 1) := by change ¬ Nat.Prime 377078143; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29006011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9709 2) := by change ¬ Nat.Prime 377058725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75411745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9709 3) := by change ¬ Nat.Prime 377039307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125679769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9709 : row 9709 = 0 := (row_9709.checked).trans rfl
theorem model_count_9709 : modelRow 9709 = 0 := by decide
theorem modelrow_agrees_9709 : modelRow 9709 = row 9709 := model_count_9709.trans row_count_9709.symm
def row_9710 : RowCertificate 9710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9710 1) := by change ¬ Nat.Prime 377155821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125718607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9710 2) := by change Nat.Prime 377136401; exact ProjectEulerSolutions.P58PrimeNodes.p_377136401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9710 3) := by change ¬ Nat.Prime 377116981; exact PrattCertificates.composite_of_product (factor := 1783) (cofactor := 211507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9710 : row 9710 = 1 := (row_9710.checked).trans rfl
theorem model_count_9710 : modelRow 9710 = 1 := by decide
theorem modelrow_agrees_9710 : modelRow 9710 = row 9710 := model_count_9710.trans row_count_9710.symm
def row_9711 : RowCertificate 9711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9711 1) := by change ¬ Nat.Prime 377233507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53890501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9711 2) := by change ¬ Nat.Prime 377214085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75442817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9711 3) := by change ¬ Nat.Prime 377194663; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1080787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9711 : row 9711 = 0 := (row_9711.checked).trans rfl
theorem model_count_9711 : modelRow 9711 = 0 := by decide
theorem modelrow_agrees_9711 : modelRow 9711 = row 9711 := model_count_9711.trans row_count_9711.symm
def row_9712 : RowCertificate 9712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9712 1) := by change ¬ Nat.Prime 377311201; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 814927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9712 2) := by change ¬ Nat.Prime 377291777; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 697397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9712 3) := by change ¬ Nat.Prime 377272353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125757451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9712 : row 9712 = 0 := (row_9712.checked).trans rfl
theorem model_count_9712 : modelRow 9712 = 0 := by decide
theorem modelrow_agrees_9712 : modelRow 9712 = row 9712 := model_count_9712.trans row_count_9712.symm
def row_9713 : RowCertificate 9713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9713 1) := by change ¬ Nat.Prime 377388903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125796301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9713 2) := by change ¬ Nat.Prime 377369477; exact PrattCertificates.composite_of_product (factor := 13477) (cofactor := 28001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9713 3) := by change ¬ Nat.Prime 377350051; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29026927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9713 : row 9713 = 0 := (row_9713.checked).trans rfl
theorem model_count_9713 : modelRow 9713 = 0 := by decide
theorem modelrow_agrees_9713 : modelRow 9713 = row 9713 := model_count_9713.trans row_count_9713.symm
def row_9714 : RowCertificate 9714 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9714 1) := by change Nat.Prime 377466613; exact ProjectEulerSolutions.P58PrimeNodes.p_377466613.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9714 2) := by change ¬ Nat.Prime 377447185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75489437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9714 3) := by change ¬ Nat.Prime 377427757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53918251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9714 : row 9714 = 1 := (row_9714.checked).trans rfl
theorem model_count_9714 : modelRow 9714 = 1 := by decide
theorem modelrow_agrees_9714 : modelRow 9714 = row 9714 := model_count_9714.trans row_count_9714.symm
def row_9715 : RowCertificate 9715 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9715 1) := by change Nat.Prime 377544331; exact ProjectEulerSolutions.P58PrimeNodes.p_377544331.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9715 2) := by change ¬ Nat.Prime 377524901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29040377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9715 3) := by change ¬ Nat.Prime 377505471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125835157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9715 : row 9715 = 1 := (row_9715.checked).trans rfl
theorem model_count_9715 : modelRow 9715 = 1 := by decide
theorem modelrow_agrees_9715 : modelRow 9715 = row 9715 := model_count_9715.trans row_count_9715.symm
def row_9716 : RowCertificate 9716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9716 1) := by change ¬ Nat.Prime 377622057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125874019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9716 2) := by change ¬ Nat.Prime 377602625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75520525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9716 3) := by change ¬ Nat.Prime 377583193; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29044861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9716 : row 9716 = 0 := (row_9716.checked).trans rfl
theorem model_count_9716 : modelRow 9716 = 0 := by decide
theorem modelrow_agrees_9716 : modelRow 9716 = row 9716 := model_count_9716.trans row_count_9716.symm
def row_9717 : RowCertificate 9717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9717 1) := by change ¬ Nat.Prime 377699791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53957113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9717 2) := by change Nat.Prime 377680357; exact ProjectEulerSolutions.P58PrimeNodes.p_377680357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9717 3) := by change ¬ Nat.Prime 377660923; exact PrattCertificates.composite_of_product (factor := 1951) (cofactor := 193573) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9717 : row 9717 = 1 := (row_9717.checked).trans rfl
theorem model_count_9717 : modelRow 9717 = 1 := by decide
theorem modelrow_agrees_9717 : modelRow 9717 = row 9717 := model_count_9717.trans row_count_9717.symm
def row_9718 : RowCertificate 9718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9718 1) := by change ¬ Nat.Prime 377777533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53968219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9718 2) := by change ¬ Nat.Prime 377758097; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 4244473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9718 3) := by change ¬ Nat.Prime 377738661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125912887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9718 : row 9718 = 0 := (row_9718.checked).trans rfl
theorem model_count_9718 : modelRow 9718 = 0 := by decide
theorem modelrow_agrees_9718 : modelRow 9718 = row 9718 := model_count_9718.trans row_count_9718.symm
def row_9719 : RowCertificate 9719 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9719 1) := by change ¬ Nat.Prime 377855283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125951761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9719 2) := by change ¬ Nat.Prime 377835845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75567169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9719 3) := by change Nat.Prime 377816407; exact ProjectEulerSolutions.P58PrimeNodes.p_377816407.prime
    exact decide_eq_true prime
theorem row_count_9719 : row 9719 = 1 := (row_9719.checked).trans rfl
theorem model_count_9719 : modelRow 9719 = 1 := by decide
theorem modelrow_agrees_9719 : modelRow 9719 = row 9719 := model_count_9719.trans row_count_9719.symm
def row_9720 : RowCertificate 9720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9720 1) := by change ¬ Nat.Prime 377933041; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1694767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9720 2) := by change ¬ Nat.Prime 377913601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29070277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9720 3) := by change Nat.Prime 377894161; exact ProjectEulerSolutions.P58PrimeNodes.p_377894161.prime
    exact decide_eq_true prime
theorem row_count_9720 : row 9720 = 1 := (row_9720.checked).trans rfl
theorem model_count_9720 : modelRow 9720 = 1 := by decide
theorem modelrow_agrees_9720 : modelRow 9720 = row 9720 := model_count_9720.trans row_count_9720.symm
def row_9721 : RowCertificate 9721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9721 1) := by change ¬ Nat.Prime 378010807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12193897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9721 2) := by change ¬ Nat.Prime 377991365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75598273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9721 3) := by change ¬ Nat.Prime 377971923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125990641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9721 : row 9721 = 0 := (row_9721.checked).trans rfl
theorem model_count_9721 : modelRow 9721 = 0 := by decide
theorem modelrow_agrees_9721 : modelRow 9721 = row 9721 := model_count_9721.trans row_count_9721.symm
def row_9722 : RowCertificate 9722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9722 1) := by change ¬ Nat.Prime 378088581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126029527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9722 2) := by change ¬ Nat.Prime 378069137; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22239361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9722 3) := by change ¬ Nat.Prime 378049693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54007099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9722 : row 9722 = 0 := (row_9722.checked).trans rfl
theorem model_count_9722 : modelRow 9722 = 0 := by decide
theorem modelrow_agrees_9722 : modelRow 9722 = row 9722 := model_count_9722.trans row_count_9722.symm
def row_9723 : RowCertificate 9723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9723 1) := by change ¬ Nat.Prime 378166363; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2504413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9723 2) := by change Nat.Prime 378146917; exact ProjectEulerSolutions.P58PrimeNodes.p_378146917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9723 3) := by change ¬ Nat.Prime 378127471; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6198811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9723 : row 9723 = 1 := (row_9723.checked).trans rfl
theorem model_count_9723 : modelRow 9723 = 1 := by decide
theorem modelrow_agrees_9723 : modelRow 9723 = row 9723 := model_count_9723.trans row_count_9723.symm
def row_9724 : RowCertificate 9724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9724 1) := by change ¬ Nat.Prime 378244153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54034879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9724 2) := by change ¬ Nat.Prime 378224705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75644941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9724 3) := by change ¬ Nat.Prime 378205257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126068419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9724 : row 9724 = 0 := (row_9724.checked).trans rfl
theorem model_count_9724 : modelRow 9724 = 0 := by decide
theorem modelrow_agrees_9724 : modelRow 9724 = row 9724 := model_count_9724.trans row_count_9724.symm
def row_9725 : RowCertificate 9725 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9725 1) := by change ¬ Nat.Prime 378321951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126107317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9725 2) := by change ¬ Nat.Prime 378302501; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 1406329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9725 3) := by change ¬ Nat.Prime 378283051; exact PrattCertificates.composite_of_product (factor := 1303) (cofactor := 290317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9725 : row 9725 = 0 := (row_9725.checked).trans rfl
theorem model_count_9725 : modelRow 9725 = 0 := by decide
theorem modelrow_agrees_9725 : modelRow 9725 = row 9725 := model_count_9725.trans row_count_9725.symm
def row_9726 : RowCertificate 9726 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9726 1) := by change Nat.Prime 378399757; exact ProjectEulerSolutions.P58PrimeNodes.p_378399757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9726 2) := by change ¬ Nat.Prime 378380305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75676061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9726 3) := by change ¬ Nat.Prime 378360853; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29104681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9726 : row 9726 = 1 := (row_9726.checked).trans rfl
theorem model_count_9726 : modelRow 9726 = 1 := by decide
theorem modelrow_agrees_9726 : modelRow 9726 = row 9726 := model_count_9726.trans row_count_9726.symm
def row_9727 : RowCertificate 9727 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9727 1) := by change Nat.Prime 378477571; exact ProjectEulerSolutions.P58PrimeNodes.p_378477571.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9727 2) := by change Nat.Prime 378458117; exact ProjectEulerSolutions.P58PrimeNodes.p_378458117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9727 3) := by change ¬ Nat.Prime 378438663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126146221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9727 : row 9727 = 2 := (row_9727.checked).trans rfl
theorem model_count_9727 : modelRow 9727 = 2 := by decide
theorem modelrow_agrees_9727 : modelRow 9727 = row 9727 := model_count_9727.trans row_count_9727.symm
def row_9728 : RowCertificate 9728 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9728 1) := by change ¬ Nat.Prime 378555393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126185131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9728 2) := by change ¬ Nat.Prime 378535937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29118149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9728 3) := by change ¬ Nat.Prime 378516481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54073783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9728 : row 9728 = 0 := (row_9728.checked).trans rfl
theorem model_count_9728 : modelRow 9728 = 0 := by decide
theorem modelrow_agrees_9728 : modelRow 9728 = row 9728 := model_count_9728.trans row_count_9728.symm
private def atomic_9601 : CountSegment 1 10 13120 9600 9601 3985 3986 := CountSegment.single (first := 9600) (initial := 3985) row_count_9601 (by decide)
private theorem atomic_9601_agreement : RowAgreement 9600 9601 := RowAgreement.single (first := 9600) modelrow_agrees_9601
private def atomic_9602 : CountSegment 1 10 13120 9601 9602 3986 3986 := CountSegment.single (first := 9601) (initial := 3986) row_count_9602 (by decide)
private theorem atomic_9602_agreement : RowAgreement 9601 9602 := RowAgreement.single (first := 9601) modelrow_agrees_9602
private def atomic_9603 : CountSegment 1 10 13120 9602 9603 3986 3987 := CountSegment.single (first := 9602) (initial := 3986) row_count_9603 (by decide)
private theorem atomic_9603_agreement : RowAgreement 9602 9603 := RowAgreement.single (first := 9602) modelrow_agrees_9603
private def atomic_9604 : CountSegment 1 10 13120 9603 9604 3987 3988 := CountSegment.single (first := 9603) (initial := 3987) row_count_9604 (by decide)
private theorem atomic_9604_agreement : RowAgreement 9603 9604 := RowAgreement.single (first := 9603) modelrow_agrees_9604
private def atomic_9605 : CountSegment 1 10 13120 9604 9605 3988 3988 := CountSegment.single (first := 9604) (initial := 3988) row_count_9605 (by decide)
private theorem atomic_9605_agreement : RowAgreement 9604 9605 := RowAgreement.single (first := 9604) modelrow_agrees_9605
private def atomic_9606 : CountSegment 1 10 13120 9605 9606 3988 3989 := CountSegment.single (first := 9605) (initial := 3988) row_count_9606 (by decide)
private theorem atomic_9606_agreement : RowAgreement 9605 9606 := RowAgreement.single (first := 9605) modelrow_agrees_9606
private def atomic_9607 : CountSegment 1 10 13120 9606 9607 3989 3989 := CountSegment.single (first := 9606) (initial := 3989) row_count_9607 (by decide)
private theorem atomic_9607_agreement : RowAgreement 9606 9607 := RowAgreement.single (first := 9606) modelrow_agrees_9607
private def atomic_9608 : CountSegment 1 10 13120 9607 9608 3989 3989 := CountSegment.single (first := 9607) (initial := 3989) row_count_9608 (by decide)
private theorem atomic_9608_agreement : RowAgreement 9607 9608 := RowAgreement.single (first := 9607) modelrow_agrees_9608
private def atomic_9609 : CountSegment 1 10 13120 9608 9609 3989 3989 := CountSegment.single (first := 9608) (initial := 3989) row_count_9609 (by decide)
private theorem atomic_9609_agreement : RowAgreement 9608 9609 := RowAgreement.single (first := 9608) modelrow_agrees_9609
private def atomic_9610 : CountSegment 1 10 13120 9609 9610 3989 3989 := CountSegment.single (first := 9609) (initial := 3989) row_count_9610 (by decide)
private theorem atomic_9610_agreement : RowAgreement 9609 9610 := RowAgreement.single (first := 9609) modelrow_agrees_9610
private def atomic_9611 : CountSegment 1 10 13120 9610 9611 3989 3990 := CountSegment.single (first := 9610) (initial := 3989) row_count_9611 (by decide)
private theorem atomic_9611_agreement : RowAgreement 9610 9611 := RowAgreement.single (first := 9610) modelrow_agrees_9611
private def atomic_9612 : CountSegment 1 10 13120 9611 9612 3990 3991 := CountSegment.single (first := 9611) (initial := 3990) row_count_9612 (by decide)
private theorem atomic_9612_agreement : RowAgreement 9611 9612 := RowAgreement.single (first := 9611) modelrow_agrees_9612
private def atomic_9613 : CountSegment 1 10 13120 9612 9613 3991 3991 := CountSegment.single (first := 9612) (initial := 3991) row_count_9613 (by decide)
private theorem atomic_9613_agreement : RowAgreement 9612 9613 := RowAgreement.single (first := 9612) modelrow_agrees_9613
private def atomic_9614 : CountSegment 1 10 13120 9613 9614 3991 3991 := CountSegment.single (first := 9613) (initial := 3991) row_count_9614 (by decide)
private theorem atomic_9614_agreement : RowAgreement 9613 9614 := RowAgreement.single (first := 9613) modelrow_agrees_9614
private def atomic_9615 : CountSegment 1 10 13120 9614 9615 3991 3992 := CountSegment.single (first := 9614) (initial := 3991) row_count_9615 (by decide)
private theorem atomic_9615_agreement : RowAgreement 9614 9615 := RowAgreement.single (first := 9614) modelrow_agrees_9615
private def atomic_9616 : CountSegment 1 10 13120 9615 9616 3992 3993 := CountSegment.single (first := 9615) (initial := 3992) row_count_9616 (by decide)
private theorem atomic_9616_agreement : RowAgreement 9615 9616 := RowAgreement.single (first := 9615) modelrow_agrees_9616
private def atomic_9617 : CountSegment 1 10 13120 9616 9617 3993 3993 := CountSegment.single (first := 9616) (initial := 3993) row_count_9617 (by decide)
private theorem atomic_9617_agreement : RowAgreement 9616 9617 := RowAgreement.single (first := 9616) modelrow_agrees_9617
private def atomic_9618 : CountSegment 1 10 13120 9617 9618 3993 3994 := CountSegment.single (first := 9617) (initial := 3993) row_count_9618 (by decide)
private theorem atomic_9618_agreement : RowAgreement 9617 9618 := RowAgreement.single (first := 9617) modelrow_agrees_9618
private def atomic_9619 : CountSegment 1 10 13120 9618 9619 3994 3994 := CountSegment.single (first := 9618) (initial := 3994) row_count_9619 (by decide)
private theorem atomic_9619_agreement : RowAgreement 9618 9619 := RowAgreement.single (first := 9618) modelrow_agrees_9619
private def atomic_9620 : CountSegment 1 10 13120 9619 9620 3994 3994 := CountSegment.single (first := 9619) (initial := 3994) row_count_9620 (by decide)
private theorem atomic_9620_agreement : RowAgreement 9619 9620 := RowAgreement.single (first := 9619) modelrow_agrees_9620
private def atomic_9621 : CountSegment 1 10 13120 9620 9621 3994 3995 := CountSegment.single (first := 9620) (initial := 3994) row_count_9621 (by decide)
private theorem atomic_9621_agreement : RowAgreement 9620 9621 := RowAgreement.single (first := 9620) modelrow_agrees_9621
private def atomic_9622 : CountSegment 1 10 13120 9621 9622 3995 3996 := CountSegment.single (first := 9621) (initial := 3995) row_count_9622 (by decide)
private theorem atomic_9622_agreement : RowAgreement 9621 9622 := RowAgreement.single (first := 9621) modelrow_agrees_9622
private def atomic_9623 : CountSegment 1 10 13120 9622 9623 3996 3996 := CountSegment.single (first := 9622) (initial := 3996) row_count_9623 (by decide)
private theorem atomic_9623_agreement : RowAgreement 9622 9623 := RowAgreement.single (first := 9622) modelrow_agrees_9623
private def atomic_9624 : CountSegment 1 10 13120 9623 9624 3996 3997 := CountSegment.single (first := 9623) (initial := 3996) row_count_9624 (by decide)
private theorem atomic_9624_agreement : RowAgreement 9623 9624 := RowAgreement.single (first := 9623) modelrow_agrees_9624
private def atomic_9625 : CountSegment 1 10 13120 9624 9625 3997 3997 := CountSegment.single (first := 9624) (initial := 3997) row_count_9625 (by decide)
private theorem atomic_9625_agreement : RowAgreement 9624 9625 := RowAgreement.single (first := 9624) modelrow_agrees_9625
private def atomic_9626 : CountSegment 1 10 13120 9625 9626 3997 3998 := CountSegment.single (first := 9625) (initial := 3997) row_count_9626 (by decide)
private theorem atomic_9626_agreement : RowAgreement 9625 9626 := RowAgreement.single (first := 9625) modelrow_agrees_9626
private def atomic_9627 : CountSegment 1 10 13120 9626 9627 3998 4000 := CountSegment.single (first := 9626) (initial := 3998) row_count_9627 (by decide)
private theorem atomic_9627_agreement : RowAgreement 9626 9627 := RowAgreement.single (first := 9626) modelrow_agrees_9627
private def atomic_9628 : CountSegment 1 10 13120 9627 9628 4000 4000 := CountSegment.single (first := 9627) (initial := 4000) row_count_9628 (by decide)
private theorem atomic_9628_agreement : RowAgreement 9627 9628 := RowAgreement.single (first := 9627) modelrow_agrees_9628
private def atomic_9629 : CountSegment 1 10 13120 9628 9629 4000 4000 := CountSegment.single (first := 9628) (initial := 4000) row_count_9629 (by decide)
private theorem atomic_9629_agreement : RowAgreement 9628 9629 := RowAgreement.single (first := 9628) modelrow_agrees_9629
private def atomic_9630 : CountSegment 1 10 13120 9629 9630 4000 4001 := CountSegment.single (first := 9629) (initial := 4000) row_count_9630 (by decide)
private theorem atomic_9630_agreement : RowAgreement 9629 9630 := RowAgreement.single (first := 9629) modelrow_agrees_9630
private def atomic_9631 : CountSegment 1 10 13120 9630 9631 4001 4001 := CountSegment.single (first := 9630) (initial := 4001) row_count_9631 (by decide)
private theorem atomic_9631_agreement : RowAgreement 9630 9631 := RowAgreement.single (first := 9630) modelrow_agrees_9631
private def atomic_9632 : CountSegment 1 10 13120 9631 9632 4001 4001 := CountSegment.single (first := 9631) (initial := 4001) row_count_9632 (by decide)
private theorem atomic_9632_agreement : RowAgreement 9631 9632 := RowAgreement.single (first := 9631) modelrow_agrees_9632
private def atomic_9633 : CountSegment 1 10 13120 9632 9633 4001 4002 := CountSegment.single (first := 9632) (initial := 4001) row_count_9633 (by decide)
private theorem atomic_9633_agreement : RowAgreement 9632 9633 := RowAgreement.single (first := 9632) modelrow_agrees_9633
private def atomic_9634 : CountSegment 1 10 13120 9633 9634 4002 4002 := CountSegment.single (first := 9633) (initial := 4002) row_count_9634 (by decide)
private theorem atomic_9634_agreement : RowAgreement 9633 9634 := RowAgreement.single (first := 9633) modelrow_agrees_9634
private def atomic_9635 : CountSegment 1 10 13120 9634 9635 4002 4002 := CountSegment.single (first := 9634) (initial := 4002) row_count_9635 (by decide)
private theorem atomic_9635_agreement : RowAgreement 9634 9635 := RowAgreement.single (first := 9634) modelrow_agrees_9635
private def atomic_9636 : CountSegment 1 10 13120 9635 9636 4002 4004 := CountSegment.single (first := 9635) (initial := 4002) row_count_9636 (by decide)
private theorem atomic_9636_agreement : RowAgreement 9635 9636 := RowAgreement.single (first := 9635) modelrow_agrees_9636
private def atomic_9637 : CountSegment 1 10 13120 9636 9637 4004 4005 := CountSegment.single (first := 9636) (initial := 4004) row_count_9637 (by decide)
private theorem atomic_9637_agreement : RowAgreement 9636 9637 := RowAgreement.single (first := 9636) modelrow_agrees_9637
private def atomic_9638 : CountSegment 1 10 13120 9637 9638 4005 4006 := CountSegment.single (first := 9637) (initial := 4005) row_count_9638 (by decide)
private theorem atomic_9638_agreement : RowAgreement 9637 9638 := RowAgreement.single (first := 9637) modelrow_agrees_9638
private def atomic_9639 : CountSegment 1 10 13120 9638 9639 4006 4006 := CountSegment.single (first := 9638) (initial := 4006) row_count_9639 (by decide)
private theorem atomic_9639_agreement : RowAgreement 9638 9639 := RowAgreement.single (first := 9638) modelrow_agrees_9639
private def atomic_9640 : CountSegment 1 10 13120 9639 9640 4006 4006 := CountSegment.single (first := 9639) (initial := 4006) row_count_9640 (by decide)
private theorem atomic_9640_agreement : RowAgreement 9639 9640 := RowAgreement.single (first := 9639) modelrow_agrees_9640
private def atomic_9641 : CountSegment 1 10 13120 9640 9641 4006 4006 := CountSegment.single (first := 9640) (initial := 4006) row_count_9641 (by decide)
private theorem atomic_9641_agreement : RowAgreement 9640 9641 := RowAgreement.single (first := 9640) modelrow_agrees_9641
private def atomic_9642 : CountSegment 1 10 13120 9641 9642 4006 4007 := CountSegment.single (first := 9641) (initial := 4006) row_count_9642 (by decide)
private theorem atomic_9642_agreement : RowAgreement 9641 9642 := RowAgreement.single (first := 9641) modelrow_agrees_9642
private def atomic_9643 : CountSegment 1 10 13120 9642 9643 4007 4007 := CountSegment.single (first := 9642) (initial := 4007) row_count_9643 (by decide)
private theorem atomic_9643_agreement : RowAgreement 9642 9643 := RowAgreement.single (first := 9642) modelrow_agrees_9643
private def atomic_9644 : CountSegment 1 10 13120 9643 9644 4007 4007 := CountSegment.single (first := 9643) (initial := 4007) row_count_9644 (by decide)
private theorem atomic_9644_agreement : RowAgreement 9643 9644 := RowAgreement.single (first := 9643) modelrow_agrees_9644
private def atomic_9645 : CountSegment 1 10 13120 9644 9645 4007 4008 := CountSegment.single (first := 9644) (initial := 4007) row_count_9645 (by decide)
private theorem atomic_9645_agreement : RowAgreement 9644 9645 := RowAgreement.single (first := 9644) modelrow_agrees_9645
private def atomic_9646 : CountSegment 1 10 13120 9645 9646 4008 4008 := CountSegment.single (first := 9645) (initial := 4008) row_count_9646 (by decide)
private theorem atomic_9646_agreement : RowAgreement 9645 9646 := RowAgreement.single (first := 9645) modelrow_agrees_9646
private def atomic_9647 : CountSegment 1 10 13120 9646 9647 4008 4010 := CountSegment.single (first := 9646) (initial := 4008) row_count_9647 (by decide)
private theorem atomic_9647_agreement : RowAgreement 9646 9647 := RowAgreement.single (first := 9646) modelrow_agrees_9647
private def atomic_9648 : CountSegment 1 10 13120 9647 9648 4010 4010 := CountSegment.single (first := 9647) (initial := 4010) row_count_9648 (by decide)
private theorem atomic_9648_agreement : RowAgreement 9647 9648 := RowAgreement.single (first := 9647) modelrow_agrees_9648
private def atomic_9649 : CountSegment 1 10 13120 9648 9649 4010 4010 := CountSegment.single (first := 9648) (initial := 4010) row_count_9649 (by decide)
private theorem atomic_9649_agreement : RowAgreement 9648 9649 := RowAgreement.single (first := 9648) modelrow_agrees_9649
private def atomic_9650 : CountSegment 1 10 13120 9649 9650 4010 4010 := CountSegment.single (first := 9649) (initial := 4010) row_count_9650 (by decide)
private theorem atomic_9650_agreement : RowAgreement 9649 9650 := RowAgreement.single (first := 9649) modelrow_agrees_9650
private def atomic_9651 : CountSegment 1 10 13120 9650 9651 4010 4010 := CountSegment.single (first := 9650) (initial := 4010) row_count_9651 (by decide)
private theorem atomic_9651_agreement : RowAgreement 9650 9651 := RowAgreement.single (first := 9650) modelrow_agrees_9651
private def atomic_9652 : CountSegment 1 10 13120 9651 9652 4010 4011 := CountSegment.single (first := 9651) (initial := 4010) row_count_9652 (by decide)
private theorem atomic_9652_agreement : RowAgreement 9651 9652 := RowAgreement.single (first := 9651) modelrow_agrees_9652
private def atomic_9653 : CountSegment 1 10 13120 9652 9653 4011 4012 := CountSegment.single (first := 9652) (initial := 4011) row_count_9653 (by decide)
private theorem atomic_9653_agreement : RowAgreement 9652 9653 := RowAgreement.single (first := 9652) modelrow_agrees_9653
private def atomic_9654 : CountSegment 1 10 13120 9653 9654 4012 4012 := CountSegment.single (first := 9653) (initial := 4012) row_count_9654 (by decide)
private theorem atomic_9654_agreement : RowAgreement 9653 9654 := RowAgreement.single (first := 9653) modelrow_agrees_9654
private def atomic_9655 : CountSegment 1 10 13120 9654 9655 4012 4012 := CountSegment.single (first := 9654) (initial := 4012) row_count_9655 (by decide)
private theorem atomic_9655_agreement : RowAgreement 9654 9655 := RowAgreement.single (first := 9654) modelrow_agrees_9655
private def atomic_9656 : CountSegment 1 10 13120 9655 9656 4012 4012 := CountSegment.single (first := 9655) (initial := 4012) row_count_9656 (by decide)
private theorem atomic_9656_agreement : RowAgreement 9655 9656 := RowAgreement.single (first := 9655) modelrow_agrees_9656
private def atomic_9657 : CountSegment 1 10 13120 9656 9657 4012 4012 := CountSegment.single (first := 9656) (initial := 4012) row_count_9657 (by decide)
private theorem atomic_9657_agreement : RowAgreement 9656 9657 := RowAgreement.single (first := 9656) modelrow_agrees_9657
private def atomic_9658 : CountSegment 1 10 13120 9657 9658 4012 4013 := CountSegment.single (first := 9657) (initial := 4012) row_count_9658 (by decide)
private theorem atomic_9658_agreement : RowAgreement 9657 9658 := RowAgreement.single (first := 9657) modelrow_agrees_9658
private def atomic_9659 : CountSegment 1 10 13120 9658 9659 4013 4013 := CountSegment.single (first := 9658) (initial := 4013) row_count_9659 (by decide)
private theorem atomic_9659_agreement : RowAgreement 9658 9659 := RowAgreement.single (first := 9658) modelrow_agrees_9659
private def atomic_9660 : CountSegment 1 10 13120 9659 9660 4013 4014 := CountSegment.single (first := 9659) (initial := 4013) row_count_9660 (by decide)
private theorem atomic_9660_agreement : RowAgreement 9659 9660 := RowAgreement.single (first := 9659) modelrow_agrees_9660
private def atomic_9661 : CountSegment 1 10 13120 9660 9661 4014 4014 := CountSegment.single (first := 9660) (initial := 4014) row_count_9661 (by decide)
private theorem atomic_9661_agreement : RowAgreement 9660 9661 := RowAgreement.single (first := 9660) modelrow_agrees_9661
private def atomic_9662 : CountSegment 1 10 13120 9661 9662 4014 4015 := CountSegment.single (first := 9661) (initial := 4014) row_count_9662 (by decide)
private theorem atomic_9662_agreement : RowAgreement 9661 9662 := RowAgreement.single (first := 9661) modelrow_agrees_9662
private def atomic_9663 : CountSegment 1 10 13120 9662 9663 4015 4015 := CountSegment.single (first := 9662) (initial := 4015) row_count_9663 (by decide)
private theorem atomic_9663_agreement : RowAgreement 9662 9663 := RowAgreement.single (first := 9662) modelrow_agrees_9663
private def atomic_9664 : CountSegment 1 10 13120 9663 9664 4015 4015 := CountSegment.single (first := 9663) (initial := 4015) row_count_9664 (by decide)
private theorem atomic_9664_agreement : RowAgreement 9663 9664 := RowAgreement.single (first := 9663) modelrow_agrees_9664
private def atomic_9665 : CountSegment 1 10 13120 9664 9665 4015 4015 := CountSegment.single (first := 9664) (initial := 4015) row_count_9665 (by decide)
private theorem atomic_9665_agreement : RowAgreement 9664 9665 := RowAgreement.single (first := 9664) modelrow_agrees_9665
private def atomic_9666 : CountSegment 1 10 13120 9665 9666 4015 4016 := CountSegment.single (first := 9665) (initial := 4015) row_count_9666 (by decide)
private theorem atomic_9666_agreement : RowAgreement 9665 9666 := RowAgreement.single (first := 9665) modelrow_agrees_9666
private def atomic_9667 : CountSegment 1 10 13120 9666 9667 4016 4017 := CountSegment.single (first := 9666) (initial := 4016) row_count_9667 (by decide)
private theorem atomic_9667_agreement : RowAgreement 9666 9667 := RowAgreement.single (first := 9666) modelrow_agrees_9667
private def atomic_9668 : CountSegment 1 10 13120 9667 9668 4017 4018 := CountSegment.single (first := 9667) (initial := 4017) row_count_9668 (by decide)
private theorem atomic_9668_agreement : RowAgreement 9667 9668 := RowAgreement.single (first := 9667) modelrow_agrees_9668
private def atomic_9669 : CountSegment 1 10 13120 9668 9669 4018 4018 := CountSegment.single (first := 9668) (initial := 4018) row_count_9669 (by decide)
private theorem atomic_9669_agreement : RowAgreement 9668 9669 := RowAgreement.single (first := 9668) modelrow_agrees_9669
private def atomic_9670 : CountSegment 1 10 13120 9669 9670 4018 4019 := CountSegment.single (first := 9669) (initial := 4018) row_count_9670 (by decide)
private theorem atomic_9670_agreement : RowAgreement 9669 9670 := RowAgreement.single (first := 9669) modelrow_agrees_9670
private def atomic_9671 : CountSegment 1 10 13120 9670 9671 4019 4019 := CountSegment.single (first := 9670) (initial := 4019) row_count_9671 (by decide)
private theorem atomic_9671_agreement : RowAgreement 9670 9671 := RowAgreement.single (first := 9670) modelrow_agrees_9671
private def atomic_9672 : CountSegment 1 10 13120 9671 9672 4019 4019 := CountSegment.single (first := 9671) (initial := 4019) row_count_9672 (by decide)
private theorem atomic_9672_agreement : RowAgreement 9671 9672 := RowAgreement.single (first := 9671) modelrow_agrees_9672
private def atomic_9673 : CountSegment 1 10 13120 9672 9673 4019 4019 := CountSegment.single (first := 9672) (initial := 4019) row_count_9673 (by decide)
private theorem atomic_9673_agreement : RowAgreement 9672 9673 := RowAgreement.single (first := 9672) modelrow_agrees_9673
private def atomic_9674 : CountSegment 1 10 13120 9673 9674 4019 4019 := CountSegment.single (first := 9673) (initial := 4019) row_count_9674 (by decide)
private theorem atomic_9674_agreement : RowAgreement 9673 9674 := RowAgreement.single (first := 9673) modelrow_agrees_9674
private def atomic_9675 : CountSegment 1 10 13120 9674 9675 4019 4019 := CountSegment.single (first := 9674) (initial := 4019) row_count_9675 (by decide)
private theorem atomic_9675_agreement : RowAgreement 9674 9675 := RowAgreement.single (first := 9674) modelrow_agrees_9675
private def atomic_9676 : CountSegment 1 10 13120 9675 9676 4019 4019 := CountSegment.single (first := 9675) (initial := 4019) row_count_9676 (by decide)
private theorem atomic_9676_agreement : RowAgreement 9675 9676 := RowAgreement.single (first := 9675) modelrow_agrees_9676
private def atomic_9677 : CountSegment 1 10 13120 9676 9677 4019 4019 := CountSegment.single (first := 9676) (initial := 4019) row_count_9677 (by decide)
private theorem atomic_9677_agreement : RowAgreement 9676 9677 := RowAgreement.single (first := 9676) modelrow_agrees_9677
private def atomic_9678 : CountSegment 1 10 13120 9677 9678 4019 4020 := CountSegment.single (first := 9677) (initial := 4019) row_count_9678 (by decide)
private theorem atomic_9678_agreement : RowAgreement 9677 9678 := RowAgreement.single (first := 9677) modelrow_agrees_9678
private def atomic_9679 : CountSegment 1 10 13120 9678 9679 4020 4020 := CountSegment.single (first := 9678) (initial := 4020) row_count_9679 (by decide)
private theorem atomic_9679_agreement : RowAgreement 9678 9679 := RowAgreement.single (first := 9678) modelrow_agrees_9679
private def atomic_9680 : CountSegment 1 10 13120 9679 9680 4020 4020 := CountSegment.single (first := 9679) (initial := 4020) row_count_9680 (by decide)
private theorem atomic_9680_agreement : RowAgreement 9679 9680 := RowAgreement.single (first := 9679) modelrow_agrees_9680
private def atomic_9681 : CountSegment 1 10 13120 9680 9681 4020 4020 := CountSegment.single (first := 9680) (initial := 4020) row_count_9681 (by decide)
private theorem atomic_9681_agreement : RowAgreement 9680 9681 := RowAgreement.single (first := 9680) modelrow_agrees_9681
private def atomic_9682 : CountSegment 1 10 13120 9681 9682 4020 4021 := CountSegment.single (first := 9681) (initial := 4020) row_count_9682 (by decide)
private theorem atomic_9682_agreement : RowAgreement 9681 9682 := RowAgreement.single (first := 9681) modelrow_agrees_9682
private def atomic_9683 : CountSegment 1 10 13120 9682 9683 4021 4021 := CountSegment.single (first := 9682) (initial := 4021) row_count_9683 (by decide)
private theorem atomic_9683_agreement : RowAgreement 9682 9683 := RowAgreement.single (first := 9682) modelrow_agrees_9683
private def atomic_9684 : CountSegment 1 10 13120 9683 9684 4021 4021 := CountSegment.single (first := 9683) (initial := 4021) row_count_9684 (by decide)
private theorem atomic_9684_agreement : RowAgreement 9683 9684 := RowAgreement.single (first := 9683) modelrow_agrees_9684
private def atomic_9685 : CountSegment 1 10 13120 9684 9685 4021 4022 := CountSegment.single (first := 9684) (initial := 4021) row_count_9685 (by decide)
private theorem atomic_9685_agreement : RowAgreement 9684 9685 := RowAgreement.single (first := 9684) modelrow_agrees_9685
private def atomic_9686 : CountSegment 1 10 13120 9685 9686 4022 4022 := CountSegment.single (first := 9685) (initial := 4022) row_count_9686 (by decide)
private theorem atomic_9686_agreement : RowAgreement 9685 9686 := RowAgreement.single (first := 9685) modelrow_agrees_9686
private def atomic_9687 : CountSegment 1 10 13120 9686 9687 4022 4022 := CountSegment.single (first := 9686) (initial := 4022) row_count_9687 (by decide)
private theorem atomic_9687_agreement : RowAgreement 9686 9687 := RowAgreement.single (first := 9686) modelrow_agrees_9687
private def atomic_9688 : CountSegment 1 10 13120 9687 9688 4022 4022 := CountSegment.single (first := 9687) (initial := 4022) row_count_9688 (by decide)
private theorem atomic_9688_agreement : RowAgreement 9687 9688 := RowAgreement.single (first := 9687) modelrow_agrees_9688
private def atomic_9689 : CountSegment 1 10 13120 9688 9689 4022 4022 := CountSegment.single (first := 9688) (initial := 4022) row_count_9689 (by decide)
private theorem atomic_9689_agreement : RowAgreement 9688 9689 := RowAgreement.single (first := 9688) modelrow_agrees_9689
private def atomic_9690 : CountSegment 1 10 13120 9689 9690 4022 4023 := CountSegment.single (first := 9689) (initial := 4022) row_count_9690 (by decide)
private theorem atomic_9690_agreement : RowAgreement 9689 9690 := RowAgreement.single (first := 9689) modelrow_agrees_9690
private def atomic_9691 : CountSegment 1 10 13120 9690 9691 4023 4024 := CountSegment.single (first := 9690) (initial := 4023) row_count_9691 (by decide)
private theorem atomic_9691_agreement : RowAgreement 9690 9691 := RowAgreement.single (first := 9690) modelrow_agrees_9691
private def atomic_9692 : CountSegment 1 10 13120 9691 9692 4024 4024 := CountSegment.single (first := 9691) (initial := 4024) row_count_9692 (by decide)
private theorem atomic_9692_agreement : RowAgreement 9691 9692 := RowAgreement.single (first := 9691) modelrow_agrees_9692
private def atomic_9693 : CountSegment 1 10 13120 9692 9693 4024 4024 := CountSegment.single (first := 9692) (initial := 4024) row_count_9693 (by decide)
private theorem atomic_9693_agreement : RowAgreement 9692 9693 := RowAgreement.single (first := 9692) modelrow_agrees_9693
private def atomic_9694 : CountSegment 1 10 13120 9693 9694 4024 4025 := CountSegment.single (first := 9693) (initial := 4024) row_count_9694 (by decide)
private theorem atomic_9694_agreement : RowAgreement 9693 9694 := RowAgreement.single (first := 9693) modelrow_agrees_9694
private def atomic_9695 : CountSegment 1 10 13120 9694 9695 4025 4025 := CountSegment.single (first := 9694) (initial := 4025) row_count_9695 (by decide)
private theorem atomic_9695_agreement : RowAgreement 9694 9695 := RowAgreement.single (first := 9694) modelrow_agrees_9695
private def atomic_9696 : CountSegment 1 10 13120 9695 9696 4025 4025 := CountSegment.single (first := 9695) (initial := 4025) row_count_9696 (by decide)
private theorem atomic_9696_agreement : RowAgreement 9695 9696 := RowAgreement.single (first := 9695) modelrow_agrees_9696
private def atomic_9697 : CountSegment 1 10 13120 9696 9697 4025 4025 := CountSegment.single (first := 9696) (initial := 4025) row_count_9697 (by decide)
private theorem atomic_9697_agreement : RowAgreement 9696 9697 := RowAgreement.single (first := 9696) modelrow_agrees_9697
private def atomic_9698 : CountSegment 1 10 13120 9697 9698 4025 4025 := CountSegment.single (first := 9697) (initial := 4025) row_count_9698 (by decide)
private theorem atomic_9698_agreement : RowAgreement 9697 9698 := RowAgreement.single (first := 9697) modelrow_agrees_9698
private def atomic_9699 : CountSegment 1 10 13120 9698 9699 4025 4025 := CountSegment.single (first := 9698) (initial := 4025) row_count_9699 (by decide)
private theorem atomic_9699_agreement : RowAgreement 9698 9699 := RowAgreement.single (first := 9698) modelrow_agrees_9699
private def atomic_9700 : CountSegment 1 10 13120 9699 9700 4025 4026 := CountSegment.single (first := 9699) (initial := 4025) row_count_9700 (by decide)
private theorem atomic_9700_agreement : RowAgreement 9699 9700 := RowAgreement.single (first := 9699) modelrow_agrees_9700
private def atomic_9701 : CountSegment 1 10 13120 9700 9701 4026 4026 := CountSegment.single (first := 9700) (initial := 4026) row_count_9701 (by decide)
private theorem atomic_9701_agreement : RowAgreement 9700 9701 := RowAgreement.single (first := 9700) modelrow_agrees_9701
private def atomic_9702 : CountSegment 1 10 13120 9701 9702 4026 4027 := CountSegment.single (first := 9701) (initial := 4026) row_count_9702 (by decide)
private theorem atomic_9702_agreement : RowAgreement 9701 9702 := RowAgreement.single (first := 9701) modelrow_agrees_9702
private def atomic_9703 : CountSegment 1 10 13120 9702 9703 4027 4027 := CountSegment.single (first := 9702) (initial := 4027) row_count_9703 (by decide)
private theorem atomic_9703_agreement : RowAgreement 9702 9703 := RowAgreement.single (first := 9702) modelrow_agrees_9703
private def atomic_9704 : CountSegment 1 10 13120 9703 9704 4027 4028 := CountSegment.single (first := 9703) (initial := 4027) row_count_9704 (by decide)
private theorem atomic_9704_agreement : RowAgreement 9703 9704 := RowAgreement.single (first := 9703) modelrow_agrees_9704
private def atomic_9705 : CountSegment 1 10 13120 9704 9705 4028 4028 := CountSegment.single (first := 9704) (initial := 4028) row_count_9705 (by decide)
private theorem atomic_9705_agreement : RowAgreement 9704 9705 := RowAgreement.single (first := 9704) modelrow_agrees_9705
private def atomic_9706 : CountSegment 1 10 13120 9705 9706 4028 4028 := CountSegment.single (first := 9705) (initial := 4028) row_count_9706 (by decide)
private theorem atomic_9706_agreement : RowAgreement 9705 9706 := RowAgreement.single (first := 9705) modelrow_agrees_9706
private def atomic_9707 : CountSegment 1 10 13120 9706 9707 4028 4028 := CountSegment.single (first := 9706) (initial := 4028) row_count_9707 (by decide)
private theorem atomic_9707_agreement : RowAgreement 9706 9707 := RowAgreement.single (first := 9706) modelrow_agrees_9707
private def atomic_9708 : CountSegment 1 10 13120 9707 9708 4028 4030 := CountSegment.single (first := 9707) (initial := 4028) row_count_9708 (by decide)
private theorem atomic_9708_agreement : RowAgreement 9707 9708 := RowAgreement.single (first := 9707) modelrow_agrees_9708
private def atomic_9709 : CountSegment 1 10 13120 9708 9709 4030 4030 := CountSegment.single (first := 9708) (initial := 4030) row_count_9709 (by decide)
private theorem atomic_9709_agreement : RowAgreement 9708 9709 := RowAgreement.single (first := 9708) modelrow_agrees_9709
private def atomic_9710 : CountSegment 1 10 13120 9709 9710 4030 4031 := CountSegment.single (first := 9709) (initial := 4030) row_count_9710 (by decide)
private theorem atomic_9710_agreement : RowAgreement 9709 9710 := RowAgreement.single (first := 9709) modelrow_agrees_9710
private def atomic_9711 : CountSegment 1 10 13120 9710 9711 4031 4031 := CountSegment.single (first := 9710) (initial := 4031) row_count_9711 (by decide)
private theorem atomic_9711_agreement : RowAgreement 9710 9711 := RowAgreement.single (first := 9710) modelrow_agrees_9711
private def atomic_9712 : CountSegment 1 10 13120 9711 9712 4031 4031 := CountSegment.single (first := 9711) (initial := 4031) row_count_9712 (by decide)
private theorem atomic_9712_agreement : RowAgreement 9711 9712 := RowAgreement.single (first := 9711) modelrow_agrees_9712
private def atomic_9713 : CountSegment 1 10 13120 9712 9713 4031 4031 := CountSegment.single (first := 9712) (initial := 4031) row_count_9713 (by decide)
private theorem atomic_9713_agreement : RowAgreement 9712 9713 := RowAgreement.single (first := 9712) modelrow_agrees_9713
private def atomic_9714 : CountSegment 1 10 13120 9713 9714 4031 4032 := CountSegment.single (first := 9713) (initial := 4031) row_count_9714 (by decide)
private theorem atomic_9714_agreement : RowAgreement 9713 9714 := RowAgreement.single (first := 9713) modelrow_agrees_9714
private def atomic_9715 : CountSegment 1 10 13120 9714 9715 4032 4033 := CountSegment.single (first := 9714) (initial := 4032) row_count_9715 (by decide)
private theorem atomic_9715_agreement : RowAgreement 9714 9715 := RowAgreement.single (first := 9714) modelrow_agrees_9715
private def atomic_9716 : CountSegment 1 10 13120 9715 9716 4033 4033 := CountSegment.single (first := 9715) (initial := 4033) row_count_9716 (by decide)
private theorem atomic_9716_agreement : RowAgreement 9715 9716 := RowAgreement.single (first := 9715) modelrow_agrees_9716
private def atomic_9717 : CountSegment 1 10 13120 9716 9717 4033 4034 := CountSegment.single (first := 9716) (initial := 4033) row_count_9717 (by decide)
private theorem atomic_9717_agreement : RowAgreement 9716 9717 := RowAgreement.single (first := 9716) modelrow_agrees_9717
private def atomic_9718 : CountSegment 1 10 13120 9717 9718 4034 4034 := CountSegment.single (first := 9717) (initial := 4034) row_count_9718 (by decide)
private theorem atomic_9718_agreement : RowAgreement 9717 9718 := RowAgreement.single (first := 9717) modelrow_agrees_9718
private def atomic_9719 : CountSegment 1 10 13120 9718 9719 4034 4035 := CountSegment.single (first := 9718) (initial := 4034) row_count_9719 (by decide)
private theorem atomic_9719_agreement : RowAgreement 9718 9719 := RowAgreement.single (first := 9718) modelrow_agrees_9719
private def atomic_9720 : CountSegment 1 10 13120 9719 9720 4035 4036 := CountSegment.single (first := 9719) (initial := 4035) row_count_9720 (by decide)
private theorem atomic_9720_agreement : RowAgreement 9719 9720 := RowAgreement.single (first := 9719) modelrow_agrees_9720
private def atomic_9721 : CountSegment 1 10 13120 9720 9721 4036 4036 := CountSegment.single (first := 9720) (initial := 4036) row_count_9721 (by decide)
private theorem atomic_9721_agreement : RowAgreement 9720 9721 := RowAgreement.single (first := 9720) modelrow_agrees_9721
private def atomic_9722 : CountSegment 1 10 13120 9721 9722 4036 4036 := CountSegment.single (first := 9721) (initial := 4036) row_count_9722 (by decide)
private theorem atomic_9722_agreement : RowAgreement 9721 9722 := RowAgreement.single (first := 9721) modelrow_agrees_9722
private def atomic_9723 : CountSegment 1 10 13120 9722 9723 4036 4037 := CountSegment.single (first := 9722) (initial := 4036) row_count_9723 (by decide)
private theorem atomic_9723_agreement : RowAgreement 9722 9723 := RowAgreement.single (first := 9722) modelrow_agrees_9723
private def atomic_9724 : CountSegment 1 10 13120 9723 9724 4037 4037 := CountSegment.single (first := 9723) (initial := 4037) row_count_9724 (by decide)
private theorem atomic_9724_agreement : RowAgreement 9723 9724 := RowAgreement.single (first := 9723) modelrow_agrees_9724
private def atomic_9725 : CountSegment 1 10 13120 9724 9725 4037 4037 := CountSegment.single (first := 9724) (initial := 4037) row_count_9725 (by decide)
private theorem atomic_9725_agreement : RowAgreement 9724 9725 := RowAgreement.single (first := 9724) modelrow_agrees_9725
private def atomic_9726 : CountSegment 1 10 13120 9725 9726 4037 4038 := CountSegment.single (first := 9725) (initial := 4037) row_count_9726 (by decide)
private theorem atomic_9726_agreement : RowAgreement 9725 9726 := RowAgreement.single (first := 9725) modelrow_agrees_9726
private def atomic_9727 : CountSegment 1 10 13120 9726 9727 4038 4040 := CountSegment.single (first := 9726) (initial := 4038) row_count_9727 (by decide)
private theorem atomic_9727_agreement : RowAgreement 9726 9727 := RowAgreement.single (first := 9726) modelrow_agrees_9727
private def atomic_9728 : CountSegment 1 10 13120 9727 9728 4040 4040 := CountSegment.single (first := 9727) (initial := 4040) row_count_9728 (by decide)
private theorem atomic_9728_agreement : RowAgreement 9727 9728 := RowAgreement.single (first := 9727) modelrow_agrees_9728
private def within_075_000 : CountSegment 1 10 13120 9600 9602 3985 3986 := CountSegment.join atomic_9601 atomic_9602
private theorem within_075_000_agreement : RowAgreement 9600 9602 := RowAgreement.join atomic_9601_agreement atomic_9602_agreement
private def within_075_001 : CountSegment 1 10 13120 9602 9604 3986 3988 := CountSegment.join atomic_9603 atomic_9604
private theorem within_075_001_agreement : RowAgreement 9602 9604 := RowAgreement.join atomic_9603_agreement atomic_9604_agreement
private def within_075_002 : CountSegment 1 10 13120 9600 9604 3985 3988 := CountSegment.join within_075_000 within_075_001
private theorem within_075_002_agreement : RowAgreement 9600 9604 := RowAgreement.join within_075_000_agreement within_075_001_agreement
private def within_075_003 : CountSegment 1 10 13120 9604 9606 3988 3989 := CountSegment.join atomic_9605 atomic_9606
private theorem within_075_003_agreement : RowAgreement 9604 9606 := RowAgreement.join atomic_9605_agreement atomic_9606_agreement
private def within_075_004 : CountSegment 1 10 13120 9606 9608 3989 3989 := CountSegment.join atomic_9607 atomic_9608
private theorem within_075_004_agreement : RowAgreement 9606 9608 := RowAgreement.join atomic_9607_agreement atomic_9608_agreement
private def within_075_005 : CountSegment 1 10 13120 9604 9608 3988 3989 := CountSegment.join within_075_003 within_075_004
private theorem within_075_005_agreement : RowAgreement 9604 9608 := RowAgreement.join within_075_003_agreement within_075_004_agreement
private def within_075_006 : CountSegment 1 10 13120 9600 9608 3985 3989 := CountSegment.join within_075_002 within_075_005
private theorem within_075_006_agreement : RowAgreement 9600 9608 := RowAgreement.join within_075_002_agreement within_075_005_agreement
private def within_075_007 : CountSegment 1 10 13120 9608 9610 3989 3989 := CountSegment.join atomic_9609 atomic_9610
private theorem within_075_007_agreement : RowAgreement 9608 9610 := RowAgreement.join atomic_9609_agreement atomic_9610_agreement
private def within_075_008 : CountSegment 1 10 13120 9610 9612 3989 3991 := CountSegment.join atomic_9611 atomic_9612
private theorem within_075_008_agreement : RowAgreement 9610 9612 := RowAgreement.join atomic_9611_agreement atomic_9612_agreement
private def within_075_009 : CountSegment 1 10 13120 9608 9612 3989 3991 := CountSegment.join within_075_007 within_075_008
private theorem within_075_009_agreement : RowAgreement 9608 9612 := RowAgreement.join within_075_007_agreement within_075_008_agreement
private def within_075_010 : CountSegment 1 10 13120 9612 9614 3991 3991 := CountSegment.join atomic_9613 atomic_9614
private theorem within_075_010_agreement : RowAgreement 9612 9614 := RowAgreement.join atomic_9613_agreement atomic_9614_agreement
private def within_075_011 : CountSegment 1 10 13120 9614 9616 3991 3993 := CountSegment.join atomic_9615 atomic_9616
private theorem within_075_011_agreement : RowAgreement 9614 9616 := RowAgreement.join atomic_9615_agreement atomic_9616_agreement
private def within_075_012 : CountSegment 1 10 13120 9612 9616 3991 3993 := CountSegment.join within_075_010 within_075_011
private theorem within_075_012_agreement : RowAgreement 9612 9616 := RowAgreement.join within_075_010_agreement within_075_011_agreement
private def within_075_013 : CountSegment 1 10 13120 9608 9616 3989 3993 := CountSegment.join within_075_009 within_075_012
private theorem within_075_013_agreement : RowAgreement 9608 9616 := RowAgreement.join within_075_009_agreement within_075_012_agreement
private def within_075_014 : CountSegment 1 10 13120 9600 9616 3985 3993 := CountSegment.join within_075_006 within_075_013
private theorem within_075_014_agreement : RowAgreement 9600 9616 := RowAgreement.join within_075_006_agreement within_075_013_agreement
private def within_075_015 : CountSegment 1 10 13120 9616 9618 3993 3994 := CountSegment.join atomic_9617 atomic_9618
private theorem within_075_015_agreement : RowAgreement 9616 9618 := RowAgreement.join atomic_9617_agreement atomic_9618_agreement
private def within_075_016 : CountSegment 1 10 13120 9618 9620 3994 3994 := CountSegment.join atomic_9619 atomic_9620
private theorem within_075_016_agreement : RowAgreement 9618 9620 := RowAgreement.join atomic_9619_agreement atomic_9620_agreement
private def within_075_017 : CountSegment 1 10 13120 9616 9620 3993 3994 := CountSegment.join within_075_015 within_075_016
private theorem within_075_017_agreement : RowAgreement 9616 9620 := RowAgreement.join within_075_015_agreement within_075_016_agreement
private def within_075_018 : CountSegment 1 10 13120 9620 9622 3994 3996 := CountSegment.join atomic_9621 atomic_9622
private theorem within_075_018_agreement : RowAgreement 9620 9622 := RowAgreement.join atomic_9621_agreement atomic_9622_agreement
private def within_075_019 : CountSegment 1 10 13120 9622 9624 3996 3997 := CountSegment.join atomic_9623 atomic_9624
private theorem within_075_019_agreement : RowAgreement 9622 9624 := RowAgreement.join atomic_9623_agreement atomic_9624_agreement
private def within_075_020 : CountSegment 1 10 13120 9620 9624 3994 3997 := CountSegment.join within_075_018 within_075_019
private theorem within_075_020_agreement : RowAgreement 9620 9624 := RowAgreement.join within_075_018_agreement within_075_019_agreement
private def within_075_021 : CountSegment 1 10 13120 9616 9624 3993 3997 := CountSegment.join within_075_017 within_075_020
private theorem within_075_021_agreement : RowAgreement 9616 9624 := RowAgreement.join within_075_017_agreement within_075_020_agreement
private def within_075_022 : CountSegment 1 10 13120 9624 9626 3997 3998 := CountSegment.join atomic_9625 atomic_9626
private theorem within_075_022_agreement : RowAgreement 9624 9626 := RowAgreement.join atomic_9625_agreement atomic_9626_agreement
private def within_075_023 : CountSegment 1 10 13120 9626 9628 3998 4000 := CountSegment.join atomic_9627 atomic_9628
private theorem within_075_023_agreement : RowAgreement 9626 9628 := RowAgreement.join atomic_9627_agreement atomic_9628_agreement
private def within_075_024 : CountSegment 1 10 13120 9624 9628 3997 4000 := CountSegment.join within_075_022 within_075_023
private theorem within_075_024_agreement : RowAgreement 9624 9628 := RowAgreement.join within_075_022_agreement within_075_023_agreement
private def within_075_025 : CountSegment 1 10 13120 9628 9630 4000 4001 := CountSegment.join atomic_9629 atomic_9630
private theorem within_075_025_agreement : RowAgreement 9628 9630 := RowAgreement.join atomic_9629_agreement atomic_9630_agreement
private def within_075_026 : CountSegment 1 10 13120 9630 9632 4001 4001 := CountSegment.join atomic_9631 atomic_9632
private theorem within_075_026_agreement : RowAgreement 9630 9632 := RowAgreement.join atomic_9631_agreement atomic_9632_agreement
private def within_075_027 : CountSegment 1 10 13120 9628 9632 4000 4001 := CountSegment.join within_075_025 within_075_026
private theorem within_075_027_agreement : RowAgreement 9628 9632 := RowAgreement.join within_075_025_agreement within_075_026_agreement
private def within_075_028 : CountSegment 1 10 13120 9624 9632 3997 4001 := CountSegment.join within_075_024 within_075_027
private theorem within_075_028_agreement : RowAgreement 9624 9632 := RowAgreement.join within_075_024_agreement within_075_027_agreement
private def within_075_029 : CountSegment 1 10 13120 9616 9632 3993 4001 := CountSegment.join within_075_021 within_075_028
private theorem within_075_029_agreement : RowAgreement 9616 9632 := RowAgreement.join within_075_021_agreement within_075_028_agreement
private def within_075_030 : CountSegment 1 10 13120 9600 9632 3985 4001 := CountSegment.join within_075_014 within_075_029
private theorem within_075_030_agreement : RowAgreement 9600 9632 := RowAgreement.join within_075_014_agreement within_075_029_agreement
private def within_075_031 : CountSegment 1 10 13120 9632 9634 4001 4002 := CountSegment.join atomic_9633 atomic_9634
private theorem within_075_031_agreement : RowAgreement 9632 9634 := RowAgreement.join atomic_9633_agreement atomic_9634_agreement
private def within_075_032 : CountSegment 1 10 13120 9634 9636 4002 4004 := CountSegment.join atomic_9635 atomic_9636
private theorem within_075_032_agreement : RowAgreement 9634 9636 := RowAgreement.join atomic_9635_agreement atomic_9636_agreement
private def within_075_033 : CountSegment 1 10 13120 9632 9636 4001 4004 := CountSegment.join within_075_031 within_075_032
private theorem within_075_033_agreement : RowAgreement 9632 9636 := RowAgreement.join within_075_031_agreement within_075_032_agreement
private def within_075_034 : CountSegment 1 10 13120 9636 9638 4004 4006 := CountSegment.join atomic_9637 atomic_9638
private theorem within_075_034_agreement : RowAgreement 9636 9638 := RowAgreement.join atomic_9637_agreement atomic_9638_agreement
private def within_075_035 : CountSegment 1 10 13120 9638 9640 4006 4006 := CountSegment.join atomic_9639 atomic_9640
private theorem within_075_035_agreement : RowAgreement 9638 9640 := RowAgreement.join atomic_9639_agreement atomic_9640_agreement
private def within_075_036 : CountSegment 1 10 13120 9636 9640 4004 4006 := CountSegment.join within_075_034 within_075_035
private theorem within_075_036_agreement : RowAgreement 9636 9640 := RowAgreement.join within_075_034_agreement within_075_035_agreement
private def within_075_037 : CountSegment 1 10 13120 9632 9640 4001 4006 := CountSegment.join within_075_033 within_075_036
private theorem within_075_037_agreement : RowAgreement 9632 9640 := RowAgreement.join within_075_033_agreement within_075_036_agreement
private def within_075_038 : CountSegment 1 10 13120 9640 9642 4006 4007 := CountSegment.join atomic_9641 atomic_9642
private theorem within_075_038_agreement : RowAgreement 9640 9642 := RowAgreement.join atomic_9641_agreement atomic_9642_agreement
private def within_075_039 : CountSegment 1 10 13120 9642 9644 4007 4007 := CountSegment.join atomic_9643 atomic_9644
private theorem within_075_039_agreement : RowAgreement 9642 9644 := RowAgreement.join atomic_9643_agreement atomic_9644_agreement
private def within_075_040 : CountSegment 1 10 13120 9640 9644 4006 4007 := CountSegment.join within_075_038 within_075_039
private theorem within_075_040_agreement : RowAgreement 9640 9644 := RowAgreement.join within_075_038_agreement within_075_039_agreement
private def within_075_041 : CountSegment 1 10 13120 9644 9646 4007 4008 := CountSegment.join atomic_9645 atomic_9646
private theorem within_075_041_agreement : RowAgreement 9644 9646 := RowAgreement.join atomic_9645_agreement atomic_9646_agreement
private def within_075_042 : CountSegment 1 10 13120 9646 9648 4008 4010 := CountSegment.join atomic_9647 atomic_9648
private theorem within_075_042_agreement : RowAgreement 9646 9648 := RowAgreement.join atomic_9647_agreement atomic_9648_agreement
private def within_075_043 : CountSegment 1 10 13120 9644 9648 4007 4010 := CountSegment.join within_075_041 within_075_042
private theorem within_075_043_agreement : RowAgreement 9644 9648 := RowAgreement.join within_075_041_agreement within_075_042_agreement
private def within_075_044 : CountSegment 1 10 13120 9640 9648 4006 4010 := CountSegment.join within_075_040 within_075_043
private theorem within_075_044_agreement : RowAgreement 9640 9648 := RowAgreement.join within_075_040_agreement within_075_043_agreement
private def within_075_045 : CountSegment 1 10 13120 9632 9648 4001 4010 := CountSegment.join within_075_037 within_075_044
private theorem within_075_045_agreement : RowAgreement 9632 9648 := RowAgreement.join within_075_037_agreement within_075_044_agreement
private def within_075_046 : CountSegment 1 10 13120 9648 9650 4010 4010 := CountSegment.join atomic_9649 atomic_9650
private theorem within_075_046_agreement : RowAgreement 9648 9650 := RowAgreement.join atomic_9649_agreement atomic_9650_agreement
private def within_075_047 : CountSegment 1 10 13120 9650 9652 4010 4011 := CountSegment.join atomic_9651 atomic_9652
private theorem within_075_047_agreement : RowAgreement 9650 9652 := RowAgreement.join atomic_9651_agreement atomic_9652_agreement
private def within_075_048 : CountSegment 1 10 13120 9648 9652 4010 4011 := CountSegment.join within_075_046 within_075_047
private theorem within_075_048_agreement : RowAgreement 9648 9652 := RowAgreement.join within_075_046_agreement within_075_047_agreement
private def within_075_049 : CountSegment 1 10 13120 9652 9654 4011 4012 := CountSegment.join atomic_9653 atomic_9654
private theorem within_075_049_agreement : RowAgreement 9652 9654 := RowAgreement.join atomic_9653_agreement atomic_9654_agreement
private def within_075_050 : CountSegment 1 10 13120 9654 9656 4012 4012 := CountSegment.join atomic_9655 atomic_9656
private theorem within_075_050_agreement : RowAgreement 9654 9656 := RowAgreement.join atomic_9655_agreement atomic_9656_agreement
private def within_075_051 : CountSegment 1 10 13120 9652 9656 4011 4012 := CountSegment.join within_075_049 within_075_050
private theorem within_075_051_agreement : RowAgreement 9652 9656 := RowAgreement.join within_075_049_agreement within_075_050_agreement
private def within_075_052 : CountSegment 1 10 13120 9648 9656 4010 4012 := CountSegment.join within_075_048 within_075_051
private theorem within_075_052_agreement : RowAgreement 9648 9656 := RowAgreement.join within_075_048_agreement within_075_051_agreement
private def within_075_053 : CountSegment 1 10 13120 9656 9658 4012 4013 := CountSegment.join atomic_9657 atomic_9658
private theorem within_075_053_agreement : RowAgreement 9656 9658 := RowAgreement.join atomic_9657_agreement atomic_9658_agreement
private def within_075_054 : CountSegment 1 10 13120 9658 9660 4013 4014 := CountSegment.join atomic_9659 atomic_9660
private theorem within_075_054_agreement : RowAgreement 9658 9660 := RowAgreement.join atomic_9659_agreement atomic_9660_agreement
private def within_075_055 : CountSegment 1 10 13120 9656 9660 4012 4014 := CountSegment.join within_075_053 within_075_054
private theorem within_075_055_agreement : RowAgreement 9656 9660 := RowAgreement.join within_075_053_agreement within_075_054_agreement
private def within_075_056 : CountSegment 1 10 13120 9660 9662 4014 4015 := CountSegment.join atomic_9661 atomic_9662
private theorem within_075_056_agreement : RowAgreement 9660 9662 := RowAgreement.join atomic_9661_agreement atomic_9662_agreement
private def within_075_057 : CountSegment 1 10 13120 9662 9664 4015 4015 := CountSegment.join atomic_9663 atomic_9664
private theorem within_075_057_agreement : RowAgreement 9662 9664 := RowAgreement.join atomic_9663_agreement atomic_9664_agreement
private def within_075_058 : CountSegment 1 10 13120 9660 9664 4014 4015 := CountSegment.join within_075_056 within_075_057
private theorem within_075_058_agreement : RowAgreement 9660 9664 := RowAgreement.join within_075_056_agreement within_075_057_agreement
private def within_075_059 : CountSegment 1 10 13120 9656 9664 4012 4015 := CountSegment.join within_075_055 within_075_058
private theorem within_075_059_agreement : RowAgreement 9656 9664 := RowAgreement.join within_075_055_agreement within_075_058_agreement
private def within_075_060 : CountSegment 1 10 13120 9648 9664 4010 4015 := CountSegment.join within_075_052 within_075_059
private theorem within_075_060_agreement : RowAgreement 9648 9664 := RowAgreement.join within_075_052_agreement within_075_059_agreement
private def within_075_061 : CountSegment 1 10 13120 9632 9664 4001 4015 := CountSegment.join within_075_045 within_075_060
private theorem within_075_061_agreement : RowAgreement 9632 9664 := RowAgreement.join within_075_045_agreement within_075_060_agreement
private def within_075_062 : CountSegment 1 10 13120 9600 9664 3985 4015 := CountSegment.join within_075_030 within_075_061
private theorem within_075_062_agreement : RowAgreement 9600 9664 := RowAgreement.join within_075_030_agreement within_075_061_agreement
private def within_075_063 : CountSegment 1 10 13120 9664 9666 4015 4016 := CountSegment.join atomic_9665 atomic_9666
private theorem within_075_063_agreement : RowAgreement 9664 9666 := RowAgreement.join atomic_9665_agreement atomic_9666_agreement
private def within_075_064 : CountSegment 1 10 13120 9666 9668 4016 4018 := CountSegment.join atomic_9667 atomic_9668
private theorem within_075_064_agreement : RowAgreement 9666 9668 := RowAgreement.join atomic_9667_agreement atomic_9668_agreement
private def within_075_065 : CountSegment 1 10 13120 9664 9668 4015 4018 := CountSegment.join within_075_063 within_075_064
private theorem within_075_065_agreement : RowAgreement 9664 9668 := RowAgreement.join within_075_063_agreement within_075_064_agreement
private def within_075_066 : CountSegment 1 10 13120 9668 9670 4018 4019 := CountSegment.join atomic_9669 atomic_9670
private theorem within_075_066_agreement : RowAgreement 9668 9670 := RowAgreement.join atomic_9669_agreement atomic_9670_agreement
private def within_075_067 : CountSegment 1 10 13120 9670 9672 4019 4019 := CountSegment.join atomic_9671 atomic_9672
private theorem within_075_067_agreement : RowAgreement 9670 9672 := RowAgreement.join atomic_9671_agreement atomic_9672_agreement
private def within_075_068 : CountSegment 1 10 13120 9668 9672 4018 4019 := CountSegment.join within_075_066 within_075_067
private theorem within_075_068_agreement : RowAgreement 9668 9672 := RowAgreement.join within_075_066_agreement within_075_067_agreement
private def within_075_069 : CountSegment 1 10 13120 9664 9672 4015 4019 := CountSegment.join within_075_065 within_075_068
private theorem within_075_069_agreement : RowAgreement 9664 9672 := RowAgreement.join within_075_065_agreement within_075_068_agreement
private def within_075_070 : CountSegment 1 10 13120 9672 9674 4019 4019 := CountSegment.join atomic_9673 atomic_9674
private theorem within_075_070_agreement : RowAgreement 9672 9674 := RowAgreement.join atomic_9673_agreement atomic_9674_agreement
private def within_075_071 : CountSegment 1 10 13120 9674 9676 4019 4019 := CountSegment.join atomic_9675 atomic_9676
private theorem within_075_071_agreement : RowAgreement 9674 9676 := RowAgreement.join atomic_9675_agreement atomic_9676_agreement
private def within_075_072 : CountSegment 1 10 13120 9672 9676 4019 4019 := CountSegment.join within_075_070 within_075_071
private theorem within_075_072_agreement : RowAgreement 9672 9676 := RowAgreement.join within_075_070_agreement within_075_071_agreement
private def within_075_073 : CountSegment 1 10 13120 9676 9678 4019 4020 := CountSegment.join atomic_9677 atomic_9678
private theorem within_075_073_agreement : RowAgreement 9676 9678 := RowAgreement.join atomic_9677_agreement atomic_9678_agreement
private def within_075_074 : CountSegment 1 10 13120 9678 9680 4020 4020 := CountSegment.join atomic_9679 atomic_9680
private theorem within_075_074_agreement : RowAgreement 9678 9680 := RowAgreement.join atomic_9679_agreement atomic_9680_agreement
private def within_075_075 : CountSegment 1 10 13120 9676 9680 4019 4020 := CountSegment.join within_075_073 within_075_074
private theorem within_075_075_agreement : RowAgreement 9676 9680 := RowAgreement.join within_075_073_agreement within_075_074_agreement
private def within_075_076 : CountSegment 1 10 13120 9672 9680 4019 4020 := CountSegment.join within_075_072 within_075_075
private theorem within_075_076_agreement : RowAgreement 9672 9680 := RowAgreement.join within_075_072_agreement within_075_075_agreement
private def within_075_077 : CountSegment 1 10 13120 9664 9680 4015 4020 := CountSegment.join within_075_069 within_075_076
private theorem within_075_077_agreement : RowAgreement 9664 9680 := RowAgreement.join within_075_069_agreement within_075_076_agreement
private def within_075_078 : CountSegment 1 10 13120 9680 9682 4020 4021 := CountSegment.join atomic_9681 atomic_9682
private theorem within_075_078_agreement : RowAgreement 9680 9682 := RowAgreement.join atomic_9681_agreement atomic_9682_agreement
private def within_075_079 : CountSegment 1 10 13120 9682 9684 4021 4021 := CountSegment.join atomic_9683 atomic_9684
private theorem within_075_079_agreement : RowAgreement 9682 9684 := RowAgreement.join atomic_9683_agreement atomic_9684_agreement
private def within_075_080 : CountSegment 1 10 13120 9680 9684 4020 4021 := CountSegment.join within_075_078 within_075_079
private theorem within_075_080_agreement : RowAgreement 9680 9684 := RowAgreement.join within_075_078_agreement within_075_079_agreement
private def within_075_081 : CountSegment 1 10 13120 9684 9686 4021 4022 := CountSegment.join atomic_9685 atomic_9686
private theorem within_075_081_agreement : RowAgreement 9684 9686 := RowAgreement.join atomic_9685_agreement atomic_9686_agreement
private def within_075_082 : CountSegment 1 10 13120 9686 9688 4022 4022 := CountSegment.join atomic_9687 atomic_9688
private theorem within_075_082_agreement : RowAgreement 9686 9688 := RowAgreement.join atomic_9687_agreement atomic_9688_agreement
private def within_075_083 : CountSegment 1 10 13120 9684 9688 4021 4022 := CountSegment.join within_075_081 within_075_082
private theorem within_075_083_agreement : RowAgreement 9684 9688 := RowAgreement.join within_075_081_agreement within_075_082_agreement
private def within_075_084 : CountSegment 1 10 13120 9680 9688 4020 4022 := CountSegment.join within_075_080 within_075_083
private theorem within_075_084_agreement : RowAgreement 9680 9688 := RowAgreement.join within_075_080_agreement within_075_083_agreement
private def within_075_085 : CountSegment 1 10 13120 9688 9690 4022 4023 := CountSegment.join atomic_9689 atomic_9690
private theorem within_075_085_agreement : RowAgreement 9688 9690 := RowAgreement.join atomic_9689_agreement atomic_9690_agreement
private def within_075_086 : CountSegment 1 10 13120 9690 9692 4023 4024 := CountSegment.join atomic_9691 atomic_9692
private theorem within_075_086_agreement : RowAgreement 9690 9692 := RowAgreement.join atomic_9691_agreement atomic_9692_agreement
private def within_075_087 : CountSegment 1 10 13120 9688 9692 4022 4024 := CountSegment.join within_075_085 within_075_086
private theorem within_075_087_agreement : RowAgreement 9688 9692 := RowAgreement.join within_075_085_agreement within_075_086_agreement
private def within_075_088 : CountSegment 1 10 13120 9692 9694 4024 4025 := CountSegment.join atomic_9693 atomic_9694
private theorem within_075_088_agreement : RowAgreement 9692 9694 := RowAgreement.join atomic_9693_agreement atomic_9694_agreement
private def within_075_089 : CountSegment 1 10 13120 9694 9696 4025 4025 := CountSegment.join atomic_9695 atomic_9696
private theorem within_075_089_agreement : RowAgreement 9694 9696 := RowAgreement.join atomic_9695_agreement atomic_9696_agreement
private def within_075_090 : CountSegment 1 10 13120 9692 9696 4024 4025 := CountSegment.join within_075_088 within_075_089
private theorem within_075_090_agreement : RowAgreement 9692 9696 := RowAgreement.join within_075_088_agreement within_075_089_agreement
private def within_075_091 : CountSegment 1 10 13120 9688 9696 4022 4025 := CountSegment.join within_075_087 within_075_090
private theorem within_075_091_agreement : RowAgreement 9688 9696 := RowAgreement.join within_075_087_agreement within_075_090_agreement
private def within_075_092 : CountSegment 1 10 13120 9680 9696 4020 4025 := CountSegment.join within_075_084 within_075_091
private theorem within_075_092_agreement : RowAgreement 9680 9696 := RowAgreement.join within_075_084_agreement within_075_091_agreement
private def within_075_093 : CountSegment 1 10 13120 9664 9696 4015 4025 := CountSegment.join within_075_077 within_075_092
private theorem within_075_093_agreement : RowAgreement 9664 9696 := RowAgreement.join within_075_077_agreement within_075_092_agreement
private def within_075_094 : CountSegment 1 10 13120 9696 9698 4025 4025 := CountSegment.join atomic_9697 atomic_9698
private theorem within_075_094_agreement : RowAgreement 9696 9698 := RowAgreement.join atomic_9697_agreement atomic_9698_agreement
private def within_075_095 : CountSegment 1 10 13120 9698 9700 4025 4026 := CountSegment.join atomic_9699 atomic_9700
private theorem within_075_095_agreement : RowAgreement 9698 9700 := RowAgreement.join atomic_9699_agreement atomic_9700_agreement
private def within_075_096 : CountSegment 1 10 13120 9696 9700 4025 4026 := CountSegment.join within_075_094 within_075_095
private theorem within_075_096_agreement : RowAgreement 9696 9700 := RowAgreement.join within_075_094_agreement within_075_095_agreement
private def within_075_097 : CountSegment 1 10 13120 9700 9702 4026 4027 := CountSegment.join atomic_9701 atomic_9702
private theorem within_075_097_agreement : RowAgreement 9700 9702 := RowAgreement.join atomic_9701_agreement atomic_9702_agreement
private def within_075_098 : CountSegment 1 10 13120 9702 9704 4027 4028 := CountSegment.join atomic_9703 atomic_9704
private theorem within_075_098_agreement : RowAgreement 9702 9704 := RowAgreement.join atomic_9703_agreement atomic_9704_agreement
private def within_075_099 : CountSegment 1 10 13120 9700 9704 4026 4028 := CountSegment.join within_075_097 within_075_098
private theorem within_075_099_agreement : RowAgreement 9700 9704 := RowAgreement.join within_075_097_agreement within_075_098_agreement
private def within_075_100 : CountSegment 1 10 13120 9696 9704 4025 4028 := CountSegment.join within_075_096 within_075_099
private theorem within_075_100_agreement : RowAgreement 9696 9704 := RowAgreement.join within_075_096_agreement within_075_099_agreement
private def within_075_101 : CountSegment 1 10 13120 9704 9706 4028 4028 := CountSegment.join atomic_9705 atomic_9706
private theorem within_075_101_agreement : RowAgreement 9704 9706 := RowAgreement.join atomic_9705_agreement atomic_9706_agreement
private def within_075_102 : CountSegment 1 10 13120 9706 9708 4028 4030 := CountSegment.join atomic_9707 atomic_9708
private theorem within_075_102_agreement : RowAgreement 9706 9708 := RowAgreement.join atomic_9707_agreement atomic_9708_agreement
private def within_075_103 : CountSegment 1 10 13120 9704 9708 4028 4030 := CountSegment.join within_075_101 within_075_102
private theorem within_075_103_agreement : RowAgreement 9704 9708 := RowAgreement.join within_075_101_agreement within_075_102_agreement
private def within_075_104 : CountSegment 1 10 13120 9708 9710 4030 4031 := CountSegment.join atomic_9709 atomic_9710
private theorem within_075_104_agreement : RowAgreement 9708 9710 := RowAgreement.join atomic_9709_agreement atomic_9710_agreement
private def within_075_105 : CountSegment 1 10 13120 9710 9712 4031 4031 := CountSegment.join atomic_9711 atomic_9712
private theorem within_075_105_agreement : RowAgreement 9710 9712 := RowAgreement.join atomic_9711_agreement atomic_9712_agreement
private def within_075_106 : CountSegment 1 10 13120 9708 9712 4030 4031 := CountSegment.join within_075_104 within_075_105
private theorem within_075_106_agreement : RowAgreement 9708 9712 := RowAgreement.join within_075_104_agreement within_075_105_agreement
private def within_075_107 : CountSegment 1 10 13120 9704 9712 4028 4031 := CountSegment.join within_075_103 within_075_106
private theorem within_075_107_agreement : RowAgreement 9704 9712 := RowAgreement.join within_075_103_agreement within_075_106_agreement
private def within_075_108 : CountSegment 1 10 13120 9696 9712 4025 4031 := CountSegment.join within_075_100 within_075_107
private theorem within_075_108_agreement : RowAgreement 9696 9712 := RowAgreement.join within_075_100_agreement within_075_107_agreement
private def within_075_109 : CountSegment 1 10 13120 9712 9714 4031 4032 := CountSegment.join atomic_9713 atomic_9714
private theorem within_075_109_agreement : RowAgreement 9712 9714 := RowAgreement.join atomic_9713_agreement atomic_9714_agreement
private def within_075_110 : CountSegment 1 10 13120 9714 9716 4032 4033 := CountSegment.join atomic_9715 atomic_9716
private theorem within_075_110_agreement : RowAgreement 9714 9716 := RowAgreement.join atomic_9715_agreement atomic_9716_agreement
private def within_075_111 : CountSegment 1 10 13120 9712 9716 4031 4033 := CountSegment.join within_075_109 within_075_110
private theorem within_075_111_agreement : RowAgreement 9712 9716 := RowAgreement.join within_075_109_agreement within_075_110_agreement
private def within_075_112 : CountSegment 1 10 13120 9716 9718 4033 4034 := CountSegment.join atomic_9717 atomic_9718
private theorem within_075_112_agreement : RowAgreement 9716 9718 := RowAgreement.join atomic_9717_agreement atomic_9718_agreement
private def within_075_113 : CountSegment 1 10 13120 9718 9720 4034 4036 := CountSegment.join atomic_9719 atomic_9720
private theorem within_075_113_agreement : RowAgreement 9718 9720 := RowAgreement.join atomic_9719_agreement atomic_9720_agreement
private def within_075_114 : CountSegment 1 10 13120 9716 9720 4033 4036 := CountSegment.join within_075_112 within_075_113
private theorem within_075_114_agreement : RowAgreement 9716 9720 := RowAgreement.join within_075_112_agreement within_075_113_agreement
private def within_075_115 : CountSegment 1 10 13120 9712 9720 4031 4036 := CountSegment.join within_075_111 within_075_114
private theorem within_075_115_agreement : RowAgreement 9712 9720 := RowAgreement.join within_075_111_agreement within_075_114_agreement
private def within_075_116 : CountSegment 1 10 13120 9720 9722 4036 4036 := CountSegment.join atomic_9721 atomic_9722
private theorem within_075_116_agreement : RowAgreement 9720 9722 := RowAgreement.join atomic_9721_agreement atomic_9722_agreement
private def within_075_117 : CountSegment 1 10 13120 9722 9724 4036 4037 := CountSegment.join atomic_9723 atomic_9724
private theorem within_075_117_agreement : RowAgreement 9722 9724 := RowAgreement.join atomic_9723_agreement atomic_9724_agreement
private def within_075_118 : CountSegment 1 10 13120 9720 9724 4036 4037 := CountSegment.join within_075_116 within_075_117
private theorem within_075_118_agreement : RowAgreement 9720 9724 := RowAgreement.join within_075_116_agreement within_075_117_agreement
private def within_075_119 : CountSegment 1 10 13120 9724 9726 4037 4038 := CountSegment.join atomic_9725 atomic_9726
private theorem within_075_119_agreement : RowAgreement 9724 9726 := RowAgreement.join atomic_9725_agreement atomic_9726_agreement
private def within_075_120 : CountSegment 1 10 13120 9726 9728 4038 4040 := CountSegment.join atomic_9727 atomic_9728
private theorem within_075_120_agreement : RowAgreement 9726 9728 := RowAgreement.join atomic_9727_agreement atomic_9728_agreement
private def within_075_121 : CountSegment 1 10 13120 9724 9728 4037 4040 := CountSegment.join within_075_119 within_075_120
private theorem within_075_121_agreement : RowAgreement 9724 9728 := RowAgreement.join within_075_119_agreement within_075_120_agreement
private def within_075_122 : CountSegment 1 10 13120 9720 9728 4036 4040 := CountSegment.join within_075_118 within_075_121
private theorem within_075_122_agreement : RowAgreement 9720 9728 := RowAgreement.join within_075_118_agreement within_075_121_agreement
private def within_075_123 : CountSegment 1 10 13120 9712 9728 4031 4040 := CountSegment.join within_075_115 within_075_122
private theorem within_075_123_agreement : RowAgreement 9712 9728 := RowAgreement.join within_075_115_agreement within_075_122_agreement
private def within_075_124 : CountSegment 1 10 13120 9696 9728 4025 4040 := CountSegment.join within_075_108 within_075_123
private theorem within_075_124_agreement : RowAgreement 9696 9728 := RowAgreement.join within_075_108_agreement within_075_123_agreement
private def within_075_125 : CountSegment 1 10 13120 9664 9728 4015 4040 := CountSegment.join within_075_093 within_075_124
private theorem within_075_125_agreement : RowAgreement 9664 9728 := RowAgreement.join within_075_093_agreement within_075_124_agreement
private def within_075_126 : CountSegment 1 10 13120 9600 9728 3985 4040 := CountSegment.join within_075_062 within_075_125
private theorem within_075_126_agreement : RowAgreement 9600 9728 := RowAgreement.join within_075_062_agreement within_075_125_agreement
def segment_075 : CountSegment 1 10 13120 9600 9728 3985 4040 := within_075_126
theorem agreement_075 : RowAgreement 9600 9728 := within_075_126_agreement
end ProjectEulerSolutions.P58Numerical
