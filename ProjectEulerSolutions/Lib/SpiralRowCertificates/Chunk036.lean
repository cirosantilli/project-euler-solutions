import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_4609 : RowCertificate 4609 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4609 1) := by change Nat.Prime 84980743; exact ProjectEulerSolutions.P58PrimeNodes.p_84980743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4609 2) := by change ¬ Nat.Prime 84971525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 16994305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4609 3) := by change ¬ Nat.Prime 84962307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28320769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4609 : row 4609 = 1 := (row_4609.checked).trans rfl
theorem model_count_4609 : modelRow 4609 = 1 := by decide
theorem modelrow_agrees_4609 : modelRow 4609 = row 4609 := model_count_4609.trans row_count_4609.symm
def row_4610 : RowCertificate 4610 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4610 1) := by change ¬ Nat.Prime 85017621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28339207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4610 2) := by change ¬ Nat.Prime 85008401; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 302521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4610 3) := by change ¬ Nat.Prime 84999181; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 1075939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4610 : row 4610 = 0 := (row_4610.checked).trans rfl
theorem model_count_4610 : modelRow 4610 = 0 := by decide
theorem modelrow_agrees_4610 : modelRow 4610 = row 4610 := model_count_4610.trans row_count_4610.symm
def row_4611 : RowCertificate 4611 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4611 1) := by change ¬ Nat.Prime 85054507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4476553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4611 2) := by change ¬ Nat.Prime 85045285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17009057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4611 3) := by change ¬ Nat.Prime 85036063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12148009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4611 : row 4611 = 0 := (row_4611.checked).trans rfl
theorem model_count_4611 : modelRow 4611 = 0 := by decide
theorem modelrow_agrees_4611 : modelRow 4611 = row 4611 := model_count_4611.trans row_count_4611.symm
def row_4612 : RowCertificate 4612 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4612 1) := by change ¬ Nat.Prime 85091401; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1394941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4612 2) := by change ¬ Nat.Prime 85082177; exact PrattCertificates.composite_of_product (factor := 2333) (cofactor := 36469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4612 3) := by change ¬ Nat.Prime 85072953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28357651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4612 : row 4612 = 0 := (row_4612.checked).trans rfl
theorem model_count_4612 : modelRow 4612 = 0 := by decide
theorem modelrow_agrees_4612 : modelRow 4612 = row 4612 := model_count_4612.trans row_count_4612.symm
def row_4613 : RowCertificate 4613 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4613 1) := by change ¬ Nat.Prime 85128303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28376101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4613 2) := by change ¬ Nat.Prime 85119077; exact PrattCertificates.composite_of_product (factor := 1109) (cofactor := 76753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4613 3) := by change ¬ Nat.Prime 85109851; exact PrattCertificates.composite_of_product (factor := 5209) (cofactor := 16339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4613 : row 4613 = 0 := (row_4613.checked).trans rfl
theorem model_count_4613 : modelRow 4613 = 0 := by decide
theorem modelrow_agrees_4613 : modelRow 4613 = row 4613 := model_count_4613.trans row_count_4613.symm
def row_4614 : RowCertificate 4614 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4614 1) := by change ¬ Nat.Prime 85165213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12166459) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4614 2) := by change ¬ Nat.Prime 85155985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17031197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4614 3) := by change ¬ Nat.Prime 85146757; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 277351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4614 : row 4614 = 0 := (row_4614.checked).trans rfl
theorem model_count_4614 : modelRow 4614 = 0 := by decide
theorem modelrow_agrees_4614 : modelRow 4614 = row 4614 := model_count_4614.trans row_count_4614.symm
def row_4615 : RowCertificate 4615 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4615 1) := by change ¬ Nat.Prime 85202131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12171733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4615 2) := by change ¬ Nat.Prime 85192901; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 138977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4615 3) := by change ¬ Nat.Prime 85183671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28394557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4615 : row 4615 = 0 := (row_4615.checked).trans rfl
theorem model_count_4615 : modelRow 4615 = 0 := by decide
theorem modelrow_agrees_4615 : modelRow 4615 = row 4615 := model_count_4615.trans row_count_4615.symm
def row_4616 : RowCertificate 4616 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4616 1) := by change ¬ Nat.Prime 85239057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28413019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4616 2) := by change ¬ Nat.Prime 85229825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17045965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4616 3) := by change Nat.Prime 85220593; exact ProjectEulerSolutions.P58PrimeNodes.p_85220593.prime
    exact decide_eq_true prime
theorem row_count_4616 : row 4616 = 1 := (row_4616.checked).trans rfl
theorem model_count_4616 : modelRow 4616 = 1 := by decide
theorem modelrow_agrees_4616 : modelRow 4616 = row 4616 := model_count_4616.trans row_count_4616.symm
def row_4617 : RowCertificate 4617 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4617 1) := by change ¬ Nat.Prime 85275991; exact PrattCertificates.composite_of_product (factor := 2593) (cofactor := 32887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4617 2) := by change ¬ Nat.Prime 85266757; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2940233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4617 3) := by change ¬ Nat.Prime 85257523; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6558271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4617 : row 4617 = 0 := (row_4617.checked).trans rfl
theorem model_count_4617 : modelRow 4617 = 0 := by decide
theorem modelrow_agrees_4617 : modelRow 4617 = row 4617 := model_count_4617.trans row_count_4617.symm
def row_4618 : RowCertificate 4618 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4618 1) := by change ¬ Nat.Prime 85312933; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 257743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4618 2) := by change ¬ Nat.Prime 85303697; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 331921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4618 3) := by change ¬ Nat.Prime 85294461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28431487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4618 : row 4618 = 0 := (row_4618.checked).trans rfl
theorem model_count_4618 : modelRow 4618 = 0 := by decide
theorem modelrow_agrees_4618 : modelRow 4618 = row 4618 := model_count_4618.trans row_count_4618.symm
def row_4619 : RowCertificate 4619 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4619 1) := by change ¬ Nat.Prime 85349883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28449961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4619 2) := by change ¬ Nat.Prime 85340645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17068129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4619 3) := by change ¬ Nat.Prime 85331407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12190201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4619 : row 4619 = 0 := (row_4619.checked).trans rfl
theorem model_count_4619 : modelRow 4619 = 0 := by decide
theorem modelrow_agrees_4619 : modelRow 4619 = row 4619 := model_count_4619.trans row_count_4619.symm
def row_4620 : RowCertificate 4620 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4620 1) := by change Nat.Prime 85386841; exact ProjectEulerSolutions.P58PrimeNodes.p_85386841.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4620 2) := by change Nat.Prime 85377601; exact ProjectEulerSolutions.P58PrimeNodes.p_85377601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4620 3) := by change ¬ Nat.Prime 85368361; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6566797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4620 : row 4620 = 2 := (row_4620.checked).trans rfl
theorem model_count_4620 : modelRow 4620 = 2 := by decide
theorem modelrow_agrees_4620 : modelRow 4620 = row 4620 := model_count_4620.trans row_count_4620.symm
def row_4621 : RowCertificate 4621 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4621 1) := by change ¬ Nat.Prime 85423807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12203401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4621 2) := by change ¬ Nat.Prime 85414565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17082913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4621 3) := by change ¬ Nat.Prime 85405323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28468441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4621 : row 4621 = 0 := (row_4621.checked).trans rfl
theorem model_count_4621 : modelRow 4621 = 0 := by decide
theorem modelrow_agrees_4621 : modelRow 4621 = row 4621 := model_count_4621.trans row_count_4621.symm
def row_4622 : RowCertificate 4622 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4622 1) := by change ¬ Nat.Prime 85460781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28486927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4622 2) := by change ¬ Nat.Prime 85451537; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5026561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4622 3) := by change ¬ Nat.Prime 85442293; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2756203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4622 : row 4622 = 0 := (row_4622.checked).trans rfl
theorem model_count_4622 : modelRow 4622 = 0 := by decide
theorem modelrow_agrees_4622 : modelRow 4622 = row 4622 := model_count_4622.trans row_count_4622.symm
def row_4623 : RowCertificate 4623 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4623 1) := by change ¬ Nat.Prime 85497763; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6576751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4623 2) := by change Nat.Prime 85488517; exact ProjectEulerSolutions.P58PrimeNodes.p_85488517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4623 3) := by change ¬ Nat.Prime 85479271; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4498909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4623 : row 4623 = 1 := (row_4623.checked).trans rfl
theorem model_count_4623 : modelRow 4623 = 1 := by decide
theorem modelrow_agrees_4623 : modelRow 4623 = row 4623 := model_count_4623.trans row_count_4623.symm
def row_4624 : RowCertificate 4624 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4624 1) := by change Nat.Prime 85534753; exact ProjectEulerSolutions.P58PrimeNodes.p_85534753.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4624 2) := by change ¬ Nat.Prime 85525505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17105101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4624 3) := by change ¬ Nat.Prime 85516257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28505419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4624 : row 4624 = 1 := (row_4624.checked).trans rfl
theorem model_count_4624 : modelRow 4624 = 1 := by decide
theorem modelrow_agrees_4624 : modelRow 4624 = row 4624 := model_count_4624.trans row_count_4624.symm
def row_4625 : RowCertificate 4625 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4625 1) := by change ¬ Nat.Prime 85571751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28523917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4625 2) := by change ¬ Nat.Prime 85562501; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 472721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4625 3) := by change ¬ Nat.Prime 85553251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12221893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4625 : row 4625 = 0 := (row_4625.checked).trans rfl
theorem model_count_4625 : modelRow 4625 = 0 := by decide
theorem modelrow_agrees_4625 : modelRow 4625 = row 4625 := model_count_4625.trans row_count_4625.symm
def row_4626 : RowCertificate 4626 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4626 1) := by change ¬ Nat.Prime 85608757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6585289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4626 2) := by change ¬ Nat.Prime 85599505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17119901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4626 3) := by change ¬ Nat.Prime 85590253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12227179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4626 : row 4626 = 0 := (row_4626.checked).trans rfl
theorem model_count_4626 : modelRow 4626 = 0 := by decide
theorem modelrow_agrees_4626 : modelRow 4626 = row 4626 := model_count_4626.trans row_count_4626.symm
def row_4627 : RowCertificate 4627 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4627 1) := by change ¬ Nat.Prime 85645771; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 373999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4627 2) := by change ¬ Nat.Prime 85636517; exact PrattCertificates.composite_of_product (factor := 5689) (cofactor := 15053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4627 3) := by change ¬ Nat.Prime 85627263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28542421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4627 : row 4627 = 0 := (row_4627.checked).trans rfl
theorem model_count_4627 : modelRow 4627 = 0 := by decide
theorem modelrow_agrees_4627 : modelRow 4627 = row 4627 := model_count_4627.trans row_count_4627.symm
def row_4628 : RowCertificate 4628 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4628 1) := by change ¬ Nat.Prime 85682793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28560931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4628 2) := by change ¬ Nat.Prime 85673537; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2315501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4628 3) := by change Nat.Prime 85664281; exact ProjectEulerSolutions.P58PrimeNodes.p_85664281.prime
    exact decide_eq_true prime
theorem row_count_4628 : row 4628 = 1 := (row_4628.checked).trans rfl
theorem model_count_4628 : modelRow 4628 = 1 := by decide
theorem modelrow_agrees_4628 : modelRow 4628 = row 4628 := model_count_4628.trans row_count_4628.symm
def row_4629 : RowCertificate 4629 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4629 1) := by change ¬ Nat.Prime 85719823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12245689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4629 2) := by change ¬ Nat.Prime 85710565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17142113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4629 3) := by change ¬ Nat.Prime 85701307; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 309391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4629 : row 4629 = 0 := (row_4629.checked).trans rfl
theorem model_count_4629 : modelRow 4629 = 0 := by decide
theorem modelrow_agrees_4629 : modelRow 4629 = row 4629 := model_count_4629.trans row_count_4629.symm
def row_4630 : RowCertificate 4630 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4630 1) := by change ¬ Nat.Prime 85756861; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4513519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4630 2) := by change Nat.Prime 85747601; exact ProjectEulerSolutions.P58PrimeNodes.p_85747601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4630 3) := by change ¬ Nat.Prime 85738341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28579447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4630 : row 4630 = 1 := (row_4630.checked).trans rfl
theorem model_count_4630 : modelRow 4630 = 1 := by decide
theorem modelrow_agrees_4630 : modelRow 4630 = row 4630 := model_count_4630.trans row_count_4630.symm
def row_4631 : RowCertificate 4631 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4631 1) := by change ¬ Nat.Prime 85793907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28597969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4631 2) := by change ¬ Nat.Prime 85784645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17156929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4631 3) := by change Nat.Prime 85775383; exact ProjectEulerSolutions.P58PrimeNodes.p_85775383.prime
    exact decide_eq_true prime
theorem row_count_4631 : row 4631 = 1 := (row_4631.checked).trans rfl
theorem model_count_4631 : modelRow 4631 = 1 := by decide
theorem modelrow_agrees_4631 : modelRow 4631 = row 4631 := model_count_4631.trans row_count_4631.symm
def row_4632 : RowCertificate 4632 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4632 1) := by change ¬ Nat.Prime 85830961; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4517419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4632 2) := by change ¬ Nat.Prime 85821697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6601669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4632 3) := by change ¬ Nat.Prime 85812433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12258919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4632 : row 4632 = 0 := (row_4632.checked).trans rfl
theorem model_count_4632 : modelRow 4632 = 0 := by decide
theorem modelrow_agrees_4632 : modelRow 4632 = row 4632 := model_count_4632.trans row_count_4632.symm
def row_4633 : RowCertificate 4633 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4633 1) := by change ¬ Nat.Prime 85868023; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 1086937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4633 2) := by change Nat.Prime 85858757; exact ProjectEulerSolutions.P58PrimeNodes.p_85858757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4633 3) := by change ¬ Nat.Prime 85849491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28616497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4633 : row 4633 = 1 := (row_4633.checked).trans rfl
theorem model_count_4633 : modelRow 4633 = 1 := by decide
theorem modelrow_agrees_4633 : modelRow 4633 = row 4633 := model_count_4633.trans row_count_4633.symm
def row_4634 : RowCertificate 4634 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4634 1) := by change ¬ Nat.Prime 85905093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28635031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4634 2) := by change ¬ Nat.Prime 85895825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17179165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4634 3) := by change ¬ Nat.Prime 85886557; exact PrattCertificates.composite_of_product (factor := 7873) (cofactor := 10909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4634 : row 4634 = 0 := (row_4634.checked).trans rfl
theorem model_count_4634 : modelRow 4634 = 0 := by decide
theorem modelrow_agrees_4634 : modelRow 4634 = row 4634 := model_count_4634.trans row_count_4634.symm
def row_4635 : RowCertificate 4635 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4635 1) := by change ¬ Nat.Prime 85942171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12277453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4635 2) := by change Nat.Prime 85932901; exact ProjectEulerSolutions.P58PrimeNodes.p_85932901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4635 3) := by change ¬ Nat.Prime 85923631; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 407221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4635 : row 4635 = 1 := (row_4635.checked).trans rfl
theorem model_count_4635 : modelRow 4635 = 1 := by decide
theorem modelrow_agrees_4635 : modelRow 4635 = row 4635 := model_count_4635.trans row_count_4635.symm
def row_4636 : RowCertificate 4636 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4636 1) := by change ¬ Nat.Prime 85979257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12282751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4636 2) := by change ¬ Nat.Prime 85969985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17193997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4636 3) := by change ¬ Nat.Prime 85960713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28653571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4636 : row 4636 = 0 := (row_4636.checked).trans rfl
theorem model_count_4636 : modelRow 4636 = 0 := by decide
theorem modelrow_agrees_4636 : modelRow 4636 = row 4636 := model_count_4636.trans row_count_4636.symm
def row_4637 : RowCertificate 4637 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4637 1) := by change ¬ Nat.Prime 86016351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28672117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4637 2) := by change ¬ Nat.Prime 86007077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6615929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4637 3) := by change ¬ Nat.Prime 85997803; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 188179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4637 : row 4637 = 0 := (row_4637.checked).trans rfl
theorem model_count_4637 : modelRow 4637 = 0 := by decide
theorem modelrow_agrees_4637 : modelRow 4637 = row 4637 := model_count_4637.trans row_count_4637.symm
def row_4638 : RowCertificate 4638 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4638 1) := by change ¬ Nat.Prime 86053453; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2325769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4638 2) := by change Nat.Prime 86044177; exact ProjectEulerSolutions.P58PrimeNodes.p_86044177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4638 3) := by change ¬ Nat.Prime 86034901; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1284103) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4638 : row 4638 = 1 := (row_4638.checked).trans rfl
theorem model_count_4638 : modelRow 4638 = 1 := by decide
theorem modelrow_agrees_4638 : modelRow 4638 = row 4638 := model_count_4638.trans row_count_4638.symm
def row_4639 : RowCertificate 4639 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4639 1) := by change ¬ Nat.Prime 86090563; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6622351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4639 2) := by change ¬ Nat.Prime 86081285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17216257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4639 3) := by change ¬ Nat.Prime 86072007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28690669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4639 : row 4639 = 0 := (row_4639.checked).trans rfl
theorem model_count_4639 : modelRow 4639 = 0 := by decide
theorem modelrow_agrees_4639 : modelRow 4639 = row 4639 := model_count_4639.trans row_count_4639.symm
def row_4640 : RowCertificate 4640 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4640 1) := by change ¬ Nat.Prime 86127681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28709227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4640 2) := by change Nat.Prime 86118401; exact ProjectEulerSolutions.P58PrimeNodes.p_86118401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4640 3) := by change ¬ Nat.Prime 86109121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12301303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4640 : row 4640 = 1 := (row_4640.checked).trans rfl
theorem model_count_4640 : modelRow 4640 = 1 := by decide
theorem modelrow_agrees_4640 : modelRow 4640 = row 4640 := model_count_4640.trans row_count_4640.symm
def row_4641 : RowCertificate 4641 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4641 1) := by change ¬ Nat.Prime 86164807; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 204667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4641 2) := by change ¬ Nat.Prime 86155525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17231105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4641 3) := by change ¬ Nat.Prime 86146243; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2003401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4641 : row 4641 = 0 := (row_4641.checked).trans rfl
theorem model_count_4641 : modelRow 4641 = 0 := by decide
theorem modelrow_agrees_4641 : modelRow 4641 = row 4641 := model_count_4641.trans row_count_4641.symm
def row_4642 : RowCertificate 4642 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4642 1) := by change ¬ Nat.Prime 86201941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12314563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4642 2) := by change ¬ Nat.Prime 86192657; exact PrattCertificates.composite_of_product (factor := 3121) (cofactor := 27617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4642 3) := by change ¬ Nat.Prime 86183373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28727791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4642 : row 4642 = 0 := (row_4642.checked).trans rfl
theorem model_count_4642 : modelRow 4642 = 0 := by decide
theorem modelrow_agrees_4642 : modelRow 4642 = row 4642 := model_count_4642.trans row_count_4642.symm
def row_4643 : RowCertificate 4643 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4643 1) := by change ¬ Nat.Prime 86239083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28746361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4643 2) := by change ¬ Nat.Prime 86229797; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5072341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4643 3) := by change ¬ Nat.Prime 86220511; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6632347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4643 : row 4643 = 0 := (row_4643.checked).trans rfl
theorem model_count_4643 : modelRow 4643 = 0 := by decide
theorem modelrow_agrees_4643 : modelRow 4643 = row 4643 := model_count_4643.trans row_count_4643.symm
def row_4644 : RowCertificate 4644 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4644 1) := by change Nat.Prime 86276233; exact ProjectEulerSolutions.P58PrimeNodes.p_86276233.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4644 2) := by change ¬ Nat.Prime 86266945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17253389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4644 3) := by change ¬ Nat.Prime 86257657; exact PrattCertificates.composite_of_product (factor := 1987) (cofactor := 43411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4644 : row 4644 = 1 := (row_4644.checked).trans rfl
theorem model_count_4644 : modelRow 4644 = 1 := by decide
theorem modelrow_agrees_4644 : modelRow 4644 = row 4644 := model_count_4644.trans row_count_4644.symm
def row_4645 : RowCertificate 4645 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4645 1) := by change ¬ Nat.Prime 86313391; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 679633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4645 2) := by change ¬ Nat.Prime 86304101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6638777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4645 3) := by change ¬ Nat.Prime 86294811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28764937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4645 : row 4645 = 0 := (row_4645.checked).trans rfl
theorem model_count_4645 : modelRow 4645 = 0 := by decide
theorem modelrow_agrees_4645 : modelRow 4645 = row 4645 := model_count_4645.trans row_count_4645.symm
def row_4646 : RowCertificate 4646 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4646 1) := by change ¬ Nat.Prime 86350557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28783519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4646 2) := by change ¬ Nat.Prime 86341265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17268253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4646 3) := by change ¬ Nat.Prime 86331973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12333139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4646 : row 4646 = 0 := (row_4646.checked).trans rfl
theorem model_count_4646 : modelRow 4646 = 0 := by decide
theorem modelrow_agrees_4646 : modelRow 4646 = row 4646 := model_count_4646.trans row_count_4646.symm
def row_4647 : RowCertificate 4647 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4647 1) := by change ¬ Nat.Prime 86387731; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2786701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4647 2) := by change Nat.Prime 86378437; exact ProjectEulerSolutions.P58PrimeNodes.p_86378437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4647 3) := by change ¬ Nat.Prime 86369143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12338449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4647 : row 4647 = 1 := (row_4647.checked).trans rfl
theorem model_count_4647 : modelRow 4647 = 1 := by decide
theorem modelrow_agrees_4647 : modelRow 4647 = row 4647 := model_count_4647.trans row_count_4647.symm
def row_4648 : RowCertificate 4648 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4648 1) := by change ¬ Nat.Prime 86424913; exact PrattCertificates.composite_of_product (factor := 3019) (cofactor := 28627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4648 2) := by change ¬ Nat.Prime 86415617; exact PrattCertificates.composite_of_product (factor := 5077) (cofactor := 17021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4648 3) := by change ¬ Nat.Prime 86406321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28802107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4648 : row 4648 = 0 := (row_4648.checked).trans rfl
theorem model_count_4648 : modelRow 4648 = 0 := by decide
theorem modelrow_agrees_4648 : modelRow 4648 = row 4648 := model_count_4648.trans row_count_4648.symm
def row_4649 : RowCertificate 4649 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4649 1) := by change ¬ Nat.Prime 86462103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28820701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4649 2) := by change ¬ Nat.Prime 86452805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17290561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4649 3) := by change ¬ Nat.Prime 86443507; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2336311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4649 : row 4649 = 0 := (row_4649.checked).trans rfl
theorem model_count_4649 : modelRow 4649 = 0 := by decide
theorem modelrow_agrees_4649 : modelRow 4649 = row 4649 := model_count_4649.trans row_count_4649.symm
def row_4650 : RowCertificate 4650 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4650 1) := by change ¬ Nat.Prime 86499301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12357043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4650 2) := by change ¬ Nat.Prime 86490001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6653077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4650 3) := by change ¬ Nat.Prime 86480701; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 261271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4650 : row 4650 = 0 := (row_4650.checked).trans rfl
theorem model_count_4650 : modelRow 4650 = 0 := by decide
theorem modelrow_agrees_4650 : modelRow 4650 = row 4650 := model_count_4650.trans row_count_4650.symm
def row_4651 : RowCertificate 4651 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4651 1) := by change ¬ Nat.Prime 86536507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4554553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4651 2) := by change ¬ Nat.Prime 86527205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17305441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4651 3) := by change ¬ Nat.Prime 86517903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28839301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4651 : row 4651 = 0 := (row_4651.checked).trans rfl
theorem model_count_4651 : modelRow 4651 = 0 := by decide
theorem modelrow_agrees_4651 : modelRow 4651 = row 4651 := model_count_4651.trans row_count_4651.symm
def row_4652 : RowCertificate 4652 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4652 1) := by change ¬ Nat.Prime 86573721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28857907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4652 2) := by change ¬ Nat.Prime 86564417; exact PrattCertificates.composite_of_product (factor := 881) (cofactor := 98257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4652 3) := by change ¬ Nat.Prime 86555113; exact PrattCertificates.composite_of_product (factor := 1759) (cofactor := 49207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4652 : row 4652 = 0 := (row_4652.checked).trans rfl
theorem model_count_4652 : modelRow 4652 = 0 := by decide
theorem modelrow_agrees_4652 : modelRow 4652 = row 4652 := model_count_4652.trans row_count_4652.symm
def row_4653 : RowCertificate 4653 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4653 1) := by change Nat.Prime 86610943; exact ProjectEulerSolutions.P58PrimeNodes.p_86610943.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4653 2) := by change ¬ Nat.Prime 86601637; exact PrattCertificates.composite_of_product (factor := 4349) (cofactor := 19913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4653 3) := by change ¬ Nat.Prime 86592331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12370333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4653 : row 4653 = 1 := (row_4653.checked).trans rfl
theorem model_count_4653 : modelRow 4653 = 1 := by decide
theorem modelrow_agrees_4653 : modelRow 4653 = row 4653 := model_count_4653.trans row_count_4653.symm
def row_4654 : RowCertificate 4654 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4654 1) := by change Nat.Prime 86648173; exact ProjectEulerSolutions.P58PrimeNodes.p_86648173.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4654 2) := by change ¬ Nat.Prime 86638865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17327773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4654 3) := by change ¬ Nat.Prime 86629557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28876519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4654 : row 4654 = 1 := (row_4654.checked).trans rfl
theorem model_count_4654 : modelRow 4654 = 1 := by decide
theorem modelrow_agrees_4654 : modelRow 4654 = row 4654 := model_count_4654.trans row_count_4654.symm
def row_4655 : RowCertificate 4655 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4655 1) := by change ¬ Nat.Prime 86685411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28895137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4655 2) := by change Nat.Prime 86676101; exact ProjectEulerSolutions.P58PrimeNodes.p_86676101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4655 3) := by change Nat.Prime 86666791; exact ProjectEulerSolutions.P58PrimeNodes.p_86666791.prime
    exact decide_eq_true prime
theorem row_count_4655 : row 4655 = 2 := (row_4655.checked).trans rfl
theorem model_count_4655 : modelRow 4655 = 2 := by decide
theorem modelrow_agrees_4655 : modelRow 4655 = row 4655 := model_count_4655.trans row_count_4655.symm
def row_4656 : RowCertificate 4656 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4656 1) := by change ¬ Nat.Prime 86722657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12388951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4656 2) := by change ¬ Nat.Prime 86713345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17342669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4656 3) := by change ¬ Nat.Prime 86704033; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6669541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4656 : row 4656 = 0 := (row_4656.checked).trans rfl
theorem model_count_4656 : modelRow 4656 = 0 := by decide
theorem modelrow_agrees_4656 : modelRow 4656 = row 4656 := model_count_4656.trans row_count_4656.symm
def row_4657 : RowCertificate 4657 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4657 1) := by change ¬ Nat.Prime 86759911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12394273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4657 2) := by change Nat.Prime 86750597; exact ProjectEulerSolutions.P58PrimeNodes.p_86750597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4657 3) := by change ¬ Nat.Prime 86741283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28913761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4657 : row 4657 = 1 := (row_4657.checked).trans rfl
theorem model_count_4657 : modelRow 4657 = 1 := by decide
theorem modelrow_agrees_4657 : modelRow 4657 = row 4657 := model_count_4657.trans row_count_4657.symm
def row_4658 : RowCertificate 4658 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4658 1) := by change ¬ Nat.Prime 86797173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28932391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4658 2) := by change ¬ Nat.Prime 86787857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6675989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4658 3) := by change ¬ Nat.Prime 86778541; exact PrattCertificates.composite_of_product (factor := 607) (cofactor := 142963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4658 : row 4658 = 0 := (row_4658.checked).trans rfl
theorem model_count_4658 : modelRow 4658 = 0 := by decide
theorem modelrow_agrees_4658 : modelRow 4658 = row 4658 := model_count_4658.trans row_count_4658.symm
def row_4659 : RowCertificate 4659 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4659 1) := by change Nat.Prime 86834443; exact ProjectEulerSolutions.P58PrimeNodes.p_86834443.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4659 2) := by change ¬ Nat.Prime 86825125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17365025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4659 3) := by change ¬ Nat.Prime 86815807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6678139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4659 : row 4659 = 1 := (row_4659.checked).trans rfl
theorem model_count_4659 : modelRow 4659 = 1 := by decide
theorem modelrow_agrees_4659 : modelRow 4659 = row 4659 := model_count_4659.trans row_count_4659.symm
def row_4660 : RowCertificate 4660 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4660 1) := by change Nat.Prime 86871721; exact ProjectEulerSolutions.P58PrimeNodes.p_86871721.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4660 2) := by change ¬ Nat.Prime 86862401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5109553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4660 3) := by change ¬ Nat.Prime 86853081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28951027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4660 : row 4660 = 1 := (row_4660.checked).trans rfl
theorem model_count_4660 : modelRow 4660 = 1 := by decide
theorem modelrow_agrees_4660 : modelRow 4660 = row 4660 := model_count_4660.trans row_count_4660.symm
def row_4661 : RowCertificate 4661 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4661 1) := by change ¬ Nat.Prime 86909007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28969669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4661 2) := by change ¬ Nat.Prime 86899685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17379937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4661 3) := by change ¬ Nat.Prime 86890363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12412909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4661 : row 4661 = 0 := (row_4661.checked).trans rfl
theorem model_count_4661 : modelRow 4661 = 0 := by decide
theorem modelrow_agrees_4661 : modelRow 4661 = row 4661 := model_count_4661.trans row_count_4661.symm
def row_4662 : RowCertificate 4662 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4662 1) := by change ¬ Nat.Prime 86946301; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6688177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4662 2) := by change Nat.Prime 86936977; exact ProjectEulerSolutions.P58PrimeNodes.p_86936977.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4662 3) := by change Nat.Prime 86927653; exact ProjectEulerSolutions.P58PrimeNodes.p_86927653.prime
    exact decide_eq_true prime
theorem row_count_4662 : row 4662 = 2 := (row_4662.checked).trans rfl
theorem model_count_4662 : modelRow 4662 = 2 := by decide
theorem modelrow_agrees_4662 : modelRow 4662 = row 4662 := model_count_4662.trans row_count_4662.symm
def row_4663 : RowCertificate 4663 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4663 1) := by change ¬ Nat.Prime 86983603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12426229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4663 2) := by change ¬ Nat.Prime 86974277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6690329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4663 3) := by change ¬ Nat.Prime 86964951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 28988317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4663 : row 4663 = 0 := (row_4663.checked).trans rfl
theorem model_count_4663 : modelRow 4663 = 0 := by decide
theorem modelrow_agrees_4663 : modelRow 4663 = row 4663 := model_count_4663.trans row_count_4663.symm
def row_4664 : RowCertificate 4664 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4664 1) := by change ¬ Nat.Prime 87020913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29006971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4664 2) := by change ¬ Nat.Prime 87011585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17402317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4664 3) := by change ¬ Nat.Prime 87002257; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 200929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4664 : row 4664 = 0 := (row_4664.checked).trans rfl
theorem model_count_4664 : modelRow 4664 = 0 := by decide
theorem modelrow_agrees_4664 : modelRow 4664 = row 4664 := model_count_4664.trans row_count_4664.symm
def row_4665 : RowCertificate 4665 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4665 1) := by change ¬ Nat.Prime 87058231; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6696787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4665 2) := by change ¬ Nat.Prime 87048901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2352673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4665 3) := by change ¬ Nat.Prime 87039571; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 576421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4665 : row 4665 = 0 := (row_4665.checked).trans rfl
theorem model_count_4665 : modelRow 4665 = 0 := by decide
theorem modelrow_agrees_4665 : modelRow 4665 = row 4665 := model_count_4665.trans row_count_4665.symm
def row_4666 : RowCertificate 4666 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4666 1) := by change ¬ Nat.Prime 87095557; exact PrattCertificates.composite_of_product (factor := 7867) (cofactor := 11071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4666 2) := by change ¬ Nat.Prime 87086225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17417245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4666 3) := by change ¬ Nat.Prime 87076893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29025631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4666 : row 4666 = 0 := (row_4666.checked).trans rfl
theorem model_count_4666 : modelRow 4666 = 0 := by decide
theorem modelrow_agrees_4666 : modelRow 4666 = row 4666 := model_count_4666.trans row_count_4666.symm
def row_4667 : RowCertificate 4667 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4667 1) := by change ¬ Nat.Prime 87132891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29044297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4667 2) := by change ¬ Nat.Prime 87123557; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 898181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4667 3) := by change ¬ Nat.Prime 87114223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12444889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4667 : row 4667 = 0 := (row_4667.checked).trans rfl
theorem model_count_4667 : modelRow 4667 = 0 := by decide
theorem modelrow_agrees_4667 : modelRow 4667 = row 4667 := model_count_4667.trans row_count_4667.symm
def row_4668 : RowCertificate 4668 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4668 1) := by change ¬ Nat.Prime 87170233; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4587907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4668 2) := by change Nat.Prime 87160897; exact ProjectEulerSolutions.P58PrimeNodes.p_87160897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4668 3) := by change ¬ Nat.Prime 87151561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12450223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4668 : row 4668 = 1 := (row_4668.checked).trans rfl
theorem model_count_4668 : modelRow 4668 = 1 := by decide
theorem modelrow_agrees_4668 : modelRow 4668 = row 4668 := model_count_4668.trans row_count_4668.symm
def row_4669 : RowCertificate 4669 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4669 1) := by change ¬ Nat.Prime 87207583; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 61027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4669 2) := by change ¬ Nat.Prime 87198245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17439649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4669 3) := by change ¬ Nat.Prime 87188907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29062969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4669 : row 4669 = 0 := (row_4669.checked).trans rfl
theorem model_count_4669 : modelRow 4669 = 0 := by decide
theorem modelrow_agrees_4669 : modelRow 4669 = row 4669 := model_count_4669.trans row_count_4669.symm
def row_4670 : RowCertificate 4670 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4670 1) := by change ¬ Nat.Prime 87244941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29081647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4670 2) := by change Nat.Prime 87235601; exact ProjectEulerSolutions.P58PrimeNodes.p_87235601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4670 3) := by change ¬ Nat.Prime 87226261; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 159463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4670 : row 4670 = 1 := (row_4670.checked).trans rfl
theorem model_count_4670 : modelRow 4670 = 1 := by decide
theorem modelrow_agrees_4670 : modelRow 4670 = row 4670 := model_count_4670.trans row_count_4670.symm
def row_4671 : RowCertificate 4671 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4671 1) := by change ¬ Nat.Prime 87282307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12468901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4671 2) := by change ¬ Nat.Prime 87272965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17454593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4671 3) := by change ¬ Nat.Prime 87263623; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 174877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4671 : row 4671 = 0 := (row_4671.checked).trans rfl
theorem model_count_4671 : modelRow 4671 = 0 := by decide
theorem modelrow_agrees_4671 : modelRow 4671 = row 4671 := model_count_4671.trans row_count_4671.symm
def row_4672 : RowCertificate 4672 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4672 1) := by change Nat.Prime 87319681; exact ProjectEulerSolutions.P58PrimeNodes.p_87319681.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4672 2) := by change ¬ Nat.Prime 87310337; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1431317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4672 3) := by change ¬ Nat.Prime 87300993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29100331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4672 : row 4672 = 1 := (row_4672.checked).trans rfl
theorem model_count_4672 : modelRow 4672 = 1 := by decide
theorem modelrow_agrees_4672 : modelRow 4672 = row 4672 := model_count_4672.trans row_count_4672.symm
def row_4673 : RowCertificate 4673 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4673 1) := by change ¬ Nat.Prime 87357063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29119021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4673 2) := by change ¬ Nat.Prime 87347717; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5138101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4673 3) := by change ¬ Nat.Prime 87338371; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 1105549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4673 : row 4673 = 0 := (row_4673.checked).trans rfl
theorem model_count_4673 : modelRow 4673 = 0 := by decide
theorem modelrow_agrees_4673 : modelRow 4673 = row 4673 := model_count_4673.trans row_count_4673.symm
def row_4674 : RowCertificate 4674 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4674 1) := by change ¬ Nat.Prime 87394453; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 452821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4674 2) := by change ¬ Nat.Prime 87385105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17477021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4674 3) := by change ¬ Nat.Prime 87375757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12482251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4674 : row 4674 = 0 := (row_4674.checked).trans rfl
theorem model_count_4674 : modelRow 4674 = 0 := by decide
theorem modelrow_agrees_4674 : modelRow 4674 = row 4674 := model_count_4674.trans row_count_4674.symm
def row_4675 : RowCertificate 4675 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4675 1) := by change ¬ Nat.Prime 87431851; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6725527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4675 2) := by change ¬ Nat.Prime 87422501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3014569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4675 3) := by change ¬ Nat.Prime 87413151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29137717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4675 : row 4675 = 0 := (row_4675.checked).trans rfl
theorem model_count_4675 : modelRow 4675 = 0 := by decide
theorem modelrow_agrees_4675 : modelRow 4675 = row 4675 := model_count_4675.trans row_count_4675.symm
def row_4676 : RowCertificate 4676 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4676 1) := by change ¬ Nat.Prime 87469257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29156419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4676 2) := by change ¬ Nat.Prime 87459905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17491981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4676 3) := by change Nat.Prime 87450553; exact ProjectEulerSolutions.P58PrimeNodes.p_87450553.prime
    exact decide_eq_true prime
theorem row_count_4676 : row 4676 = 1 := (row_4676.checked).trans rfl
theorem model_count_4676 : modelRow 4676 = 1 := by decide
theorem modelrow_agrees_4676 : modelRow 4676 = row 4676 := model_count_4676.trans row_count_4676.symm
def row_4677 : RowCertificate 4677 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4677 1) := by change ¬ Nat.Prime 87506671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12500953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4677 2) := by change ¬ Nat.Prime 87497317; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5146901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4677 3) := by change ¬ Nat.Prime 87487963; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 213907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4677 : row 4677 = 0 := (row_4677.checked).trans rfl
theorem model_count_4677 : modelRow 4677 = 0 := by decide
theorem modelrow_agrees_4677 : modelRow 4677 = row 4677 := model_count_4677.trans row_count_4677.symm
def row_4678 : RowCertificate 4678 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4678 1) := by change ¬ Nat.Prime 87544093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12506299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4678 2) := by change Nat.Prime 87534737; exact ProjectEulerSolutions.P58PrimeNodes.p_87534737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4678 3) := by change ¬ Nat.Prime 87525381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29175127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4678 : row 4678 = 1 := (row_4678.checked).trans rfl
theorem model_count_4678 : modelRow 4678 = 1 := by decide
theorem modelrow_agrees_4678 : modelRow 4678 = row 4678 := model_count_4678.trans row_count_4678.symm
def row_4679 : RowCertificate 4679 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4679 1) := by change ¬ Nat.Prime 87581523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29193841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4679 2) := by change ¬ Nat.Prime 87572165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17514433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4679 3) := by change Nat.Prime 87562807; exact ProjectEulerSolutions.P58PrimeNodes.p_87562807.prime
    exact decide_eq_true prime
theorem row_count_4679 : row 4679 = 1 := (row_4679.checked).trans rfl
theorem model_count_4679 : modelRow 4679 = 1 := by decide
theorem modelrow_agrees_4679 : modelRow 4679 = row 4679 := model_count_4679.trans row_count_4679.symm
def row_4680 : RowCertificate 4680 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4680 1) := by change Nat.Prime 87618961; exact ProjectEulerSolutions.P58PrimeNodes.p_87618961.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4680 2) := by change ¬ Nat.Prime 87609601; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 132541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4680 3) := by change ¬ Nat.Prime 87600241; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4610539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4680 : row 4680 = 1 := (row_4680.checked).trans rfl
theorem model_count_4680 : modelRow 4680 = 1 := by decide
theorem modelrow_agrees_4680 : modelRow 4680 = row 4680 := model_count_4680.trans row_count_4680.symm
def row_4681 : RowCertificate 4681 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4681 1) := by change Nat.Prime 87656407; exact ProjectEulerSolutions.P58PrimeNodes.p_87656407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4681 2) := by change ¬ Nat.Prime 87647045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17529409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4681 3) := by change ¬ Nat.Prime 87637683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29212561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4681 : row 4681 = 1 := (row_4681.checked).trans rfl
theorem model_count_4681 : modelRow 4681 = 1 := by decide
theorem modelrow_agrees_4681 : modelRow 4681 = row 4681 := model_count_4681.trans row_count_4681.symm
def row_4682 : RowCertificate 4682 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4682 1) := by change ¬ Nat.Prime 87693861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29231287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4682 2) := by change ¬ Nat.Prime 87684497; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 775969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4682 3) := by change ¬ Nat.Prime 87675133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12525019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4682 : row 4682 = 0 := (row_4682.checked).trans rfl
theorem model_count_4682 : modelRow 4682 = 0 := by decide
theorem modelrow_agrees_4682 : modelRow 4682 = row 4682 := model_count_4682.trans row_count_4682.symm
def row_4683 : RowCertificate 4683 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4683 1) := by change Nat.Prime 87731323; exact ProjectEulerSolutions.P58PrimeNodes.p_87731323.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4683 2) := by change ¬ Nat.Prime 87721957; exact PrattCertificates.composite_of_product (factor := 1861) (cofactor := 47137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4683 3) := by change ¬ Nat.Prime 87712591; exact PrattCertificates.composite_of_product (factor := 4561) (cofactor := 19231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4683 : row 4683 = 1 := (row_4683.checked).trans rfl
theorem model_count_4683 : modelRow 4683 = 1 := by decide
theorem modelrow_agrees_4683 : modelRow 4683 = row 4683 := model_count_4683.trans row_count_4683.symm
def row_4684 : RowCertificate 4684 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4684 1) := by change ¬ Nat.Prime 87768793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12538399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4684 2) := by change ¬ Nat.Prime 87759425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17551885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4684 3) := by change ¬ Nat.Prime 87750057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29250019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4684 : row 4684 = 0 := (row_4684.checked).trans rfl
theorem model_count_4684 : modelRow 4684 = 0 := by decide
theorem modelrow_agrees_4684 : modelRow 4684 = row 4684 := model_count_4684.trans row_count_4684.symm
def row_4685 : RowCertificate 4685 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4685 1) := by change ¬ Nat.Prime 87806271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29268757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4685 2) := by change ¬ Nat.Prime 87796901; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 172489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4685 3) := by change ¬ Nat.Prime 87787531; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6752887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4685 : row 4685 = 0 := (row_4685.checked).trans rfl
theorem model_count_4685 : modelRow 4685 = 0 := by decide
theorem modelrow_agrees_4685 : modelRow 4685 = row 4685 := model_count_4685.trans row_count_4685.symm
def row_4686 : RowCertificate 4686 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4686 1) := by change ¬ Nat.Prime 87843757; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 455149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4686 2) := by change ¬ Nat.Prime 87834385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17566877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4686 3) := by change ¬ Nat.Prime 87825013; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2373649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4686 : row 4686 = 0 := (row_4686.checked).trans rfl
theorem model_count_4686 : modelRow 4686 = 0 := by decide
theorem modelrow_agrees_4686 : modelRow 4686 = row 4686 := model_count_4686.trans row_count_4686.symm
def row_4687 : RowCertificate 4687 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4687 1) := by change ¬ Nat.Prime 87881251; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4625329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4687 2) := by change Nat.Prime 87871877; exact ProjectEulerSolutions.P58PrimeNodes.p_87871877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4687 3) := by change ¬ Nat.Prime 87862503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29287501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4687 : row 4687 = 1 := (row_4687.checked).trans rfl
theorem model_count_4687 : modelRow 4687 = 1 := by decide
theorem modelrow_agrees_4687 : modelRow 4687 = row 4687 := model_count_4687.trans row_count_4687.symm
def row_4688 : RowCertificate 4688 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4688 1) := by change ¬ Nat.Prime 87918753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29306251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4688 2) := by change ¬ Nat.Prime 87909377; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 455489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4688 3) := by change ¬ Nat.Prime 87900001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12557143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4688 : row 4688 = 0 := (row_4688.checked).trans rfl
theorem model_count_4688 : modelRow 4688 = 0 := by decide
theorem modelrow_agrees_4688 : modelRow 4688 = row 4688 := model_count_4688.trans row_count_4688.symm
def row_4689 : RowCertificate 4689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4689 1) := by change ¬ Nat.Prime 87956263; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4629277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4689 2) := by change ¬ Nat.Prime 87946885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17589377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4689 3) := by change ¬ Nat.Prime 87937507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12562501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4689 : row 4689 = 0 := (row_4689.checked).trans rfl
theorem model_count_4689 : modelRow 4689 = 0 := by decide
theorem modelrow_agrees_4689 : modelRow 4689 = row 4689 := model_count_4689.trans row_count_4689.symm
def row_4690 : RowCertificate 4690 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4690 1) := by change ¬ Nat.Prime 87993781; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2046367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4690 2) := by change ¬ Nat.Prime 87984401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5175553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4690 3) := by change ¬ Nat.Prime 87975021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29325007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4690 : row 4690 = 0 := (row_4690.checked).trans rfl
theorem model_count_4690 : modelRow 4690 = 0 := by decide
theorem modelrow_agrees_4690 : modelRow 4690 = row 4690 := model_count_4690.trans row_count_4690.symm
def row_4691 : RowCertificate 4691 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4691 1) := by change ¬ Nat.Prime 88031307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29343769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4691 2) := by change ¬ Nat.Prime 88021925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17604385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4691 3) := by change ¬ Nat.Prime 88012543; exact PrattCertificates.composite_of_product (factor := 8329) (cofactor := 10567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4691 : row 4691 = 0 := (row_4691.checked).trans rfl
theorem model_count_4691 : modelRow 4691 = 0 := by decide
theorem modelrow_agrees_4691 : modelRow 4691 = row 4691 := model_count_4691.trans row_count_4691.symm
def row_4692 : RowCertificate 4692 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4692 1) := by change ¬ Nat.Prime 88068841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12581263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4692 2) := by change ¬ Nat.Prime 88059457; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3036533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4692 3) := by change ¬ Nat.Prime 88050073; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 365353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4692 : row 4692 = 0 := (row_4692.checked).trans rfl
theorem model_count_4692 : modelRow 4692 = 0 := by decide
theorem modelrow_agrees_4692 : modelRow 4692 = row 4692 := model_count_4692.trans row_count_4692.symm
def row_4693 : RowCertificate 4693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4693 1) := by change ¬ Nat.Prime 88106383; exact PrattCertificates.composite_of_product (factor := 1447) (cofactor := 60889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4693 2) := by change Nat.Prime 88096997; exact ProjectEulerSolutions.P58PrimeNodes.p_88096997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4693 3) := by change ¬ Nat.Prime 88087611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29362537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4693 : row 4693 = 1 := (row_4693.checked).trans rfl
theorem model_count_4693 : modelRow 4693 = 1 := by decide
theorem modelrow_agrees_4693 : modelRow 4693 = row 4693 := model_count_4693.trans row_count_4693.symm
def row_4694 : RowCertificate 4694 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4694 1) := by change ¬ Nat.Prime 88143933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29381311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4694 2) := by change ¬ Nat.Prime 88134545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17626909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4694 3) := by change ¬ Nat.Prime 88125157; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2842747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4694 : row 4694 = 0 := (row_4694.checked).trans rfl
theorem model_count_4694 : modelRow 4694 = 0 := by decide
theorem modelrow_agrees_4694 : modelRow 4694 = row 4694 := model_count_4694.trans row_count_4694.symm
def row_4695 : RowCertificate 4695 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4695 1) := by change ¬ Nat.Prime 88181491; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 200869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4695 2) := by change ¬ Nat.Prime 88172101; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1207837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4695 3) := by change ¬ Nat.Prime 88162711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12594673) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4695 : row 4695 = 0 := (row_4695.checked).trans rfl
theorem model_count_4695 : modelRow 4695 = 0 := by decide
theorem modelrow_agrees_4695 : modelRow 4695 = row 4695 := model_count_4695.trans row_count_4695.symm
def row_4696 : RowCertificate 4696 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4696 1) := by change ¬ Nat.Prime 88219057; exact PrattCertificates.composite_of_product (factor := 2557) (cofactor := 34501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4696 2) := by change ¬ Nat.Prime 88209665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17641933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4696 3) := by change ¬ Nat.Prime 88200273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29400091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4696 : row 4696 = 0 := (row_4696.checked).trans rfl
theorem model_count_4696 : modelRow 4696 = 0 := by decide
theorem modelrow_agrees_4696 : modelRow 4696 = row 4696 := model_count_4696.trans row_count_4696.symm
def row_4697 : RowCertificate 4697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4697 1) := by change ¬ Nat.Prime 88256631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29418877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4697 2) := by change ¬ Nat.Prime 88247237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6788249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4697 3) := by change ¬ Nat.Prime 88237843; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4644097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4697 : row 4697 = 0 := (row_4697.checked).trans rfl
theorem model_count_4697 : modelRow 4697 = 0 := by decide
theorem modelrow_agrees_4697 : modelRow 4697 = row 4697 := model_count_4697.trans row_count_4697.symm
def row_4698 : RowCertificate 4698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4698 1) := by change ¬ Nat.Prime 88294213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12613459) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4698 2) := by change ¬ Nat.Prime 88284817; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 278501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4698 3) := by change ¬ Nat.Prime 88275421; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6790417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4698 : row 4698 = 0 := (row_4698.checked).trans rfl
theorem model_count_4698 : modelRow 4698 = 0 := by decide
theorem modelrow_agrees_4698 : modelRow 4698 = row 4698 := model_count_4698.trans row_count_4698.symm
def row_4699 : RowCertificate 4699 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4699 1) := by change ¬ Nat.Prime 88331803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12618829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4699 2) := by change ¬ Nat.Prime 88322405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17664481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4699 3) := by change ¬ Nat.Prime 88313007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29437669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4699 : row 4699 = 0 := (row_4699.checked).trans rfl
theorem model_count_4699 : modelRow 4699 = 0 := by decide
theorem modelrow_agrees_4699 : modelRow 4699 = row 4699 := model_count_4699.trans row_count_4699.symm
def row_4700 : RowCertificate 4700 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4700 1) := by change ¬ Nat.Prime 88369401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29456467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4700 2) := by change ¬ Nat.Prime 88360001; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 992809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4700 3) := by change Nat.Prime 88350601; exact ProjectEulerSolutions.P58PrimeNodes.p_88350601.prime
    exact decide_eq_true prime
theorem row_count_4700 : row 4700 = 1 := (row_4700.checked).trans rfl
theorem model_count_4700 : modelRow 4700 = 1 := by decide
theorem modelrow_agrees_4700 : modelRow 4700 = row 4700 := model_count_4700.trans row_count_4700.symm
def row_4701 : RowCertificate 4701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4701 1) := by change ¬ Nat.Prime 88407007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6800539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4701 2) := by change ¬ Nat.Prime 88397605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17679521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4701 3) := by change Nat.Prime 88388203; exact ProjectEulerSolutions.P58PrimeNodes.p_88388203.prime
    exact decide_eq_true prime
theorem row_count_4701 : row 4701 = 1 := (row_4701.checked).trans rfl
theorem model_count_4701 : modelRow 4701 = 1 := by decide
theorem modelrow_agrees_4701 : modelRow 4701 = row 4701 := model_count_4701.trans row_count_4701.symm
def row_4702 : RowCertificate 4702 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4702 1) := by change Nat.Prime 88444621; exact ProjectEulerSolutions.P58PrimeNodes.p_88444621.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4702 2) := by change ¬ Nat.Prime 88435217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6802709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4702 3) := by change ¬ Nat.Prime 88425813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29475271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4702 : row 4702 = 1 := (row_4702.checked).trans rfl
theorem model_count_4702 : modelRow 4702 = 1 := by decide
theorem modelrow_agrees_4702 : modelRow 4702 = row 4702 := model_count_4702.trans row_count_4702.symm
def row_4703 : RowCertificate 4703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4703 1) := by change ¬ Nat.Prime 88482243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29494081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4703 2) := by change Nat.Prime 88472837; exact ProjectEulerSolutions.P58PrimeNodes.p_88472837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4703 3) := by change ¬ Nat.Prime 88463431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12637633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4703 : row 4703 = 1 := (row_4703.checked).trans rfl
theorem model_count_4703 : modelRow 4703 = 1 := by decide
theorem modelrow_agrees_4703 : modelRow 4703 = row 4703 := model_count_4703.trans row_count_4703.symm
def row_4704 : RowCertificate 4704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4704 1) := by change ¬ Nat.Prime 88519873; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6809221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4704 2) := by change ¬ Nat.Prime 88510465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17702093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4704 3) := by change ¬ Nat.Prime 88501057; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1450837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4704 : row 4704 = 0 := (row_4704.checked).trans rfl
theorem model_count_4704 : modelRow 4704 = 0 := by decide
theorem modelrow_agrees_4704 : modelRow 4704 = row 4704 := model_count_4704.trans row_count_4704.symm
def row_4705 : RowCertificate 4705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4705 1) := by change ¬ Nat.Prime 88557511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12651073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4705 2) := by change ¬ Nat.Prime 88548101; exact PrattCertificates.composite_of_product (factor := 7349) (cofactor := 12049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4705 3) := by change ¬ Nat.Prime 88538691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29512897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4705 : row 4705 = 0 := (row_4705.checked).trans rfl
theorem model_count_4705 : modelRow 4705 = 0 := by decide
theorem modelrow_agrees_4705 : modelRow 4705 = row 4705 := model_count_4705.trans row_count_4705.symm
def row_4706 : RowCertificate 4706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4706 1) := by change ¬ Nat.Prime 88595157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29531719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4706 2) := by change ¬ Nat.Prime 88585745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17717149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4706 3) := by change ¬ Nat.Prime 88576333; exact PrattCertificates.composite_of_product (factor := 1453) (cofactor := 60961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4706 : row 4706 = 0 := (row_4706.checked).trans rfl
theorem model_count_4706 : modelRow 4706 = 0 := by decide
theorem modelrow_agrees_4706 : modelRow 4706 = row 4706 := model_count_4706.trans row_count_4706.symm
def row_4707 : RowCertificate 4707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4707 1) := by change ¬ Nat.Prime 88632811; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 367771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4707 2) := by change ¬ Nat.Prime 88623397; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5213141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4707 3) := by change ¬ Nat.Prime 88613983; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 237571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4707 : row 4707 = 0 := (row_4707.checked).trans rfl
theorem model_count_4707 : modelRow 4707 = 0 := by decide
theorem modelrow_agrees_4707 : modelRow 4707 = row 4707 := model_count_4707.trans row_count_4707.symm
def row_4708 : RowCertificate 4708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4708 1) := by change ¬ Nat.Prime 88670473; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4666867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4708 2) := by change ¬ Nat.Prime 88661057; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 647161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4708 3) := by change ¬ Nat.Prime 88651641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29550547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4708 : row 4708 = 0 := (row_4708.checked).trans rfl
theorem model_count_4708 : modelRow 4708 = 0 := by decide
theorem modelrow_agrees_4708 : modelRow 4708 = row 4708 := model_count_4708.trans row_count_4708.symm
def row_4709 : RowCertificate 4709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4709 1) := by change ¬ Nat.Prime 88708143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29569381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4709 2) := by change ¬ Nat.Prime 88698725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17739745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4709 3) := by change ¬ Nat.Prime 88689307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12669901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4709 : row 4709 = 0 := (row_4709.checked).trans rfl
theorem model_count_4709 : modelRow 4709 = 0 := by decide
theorem modelrow_agrees_4709 : modelRow 4709 = row 4709 := model_count_4709.trans row_count_4709.symm
def row_4710 : RowCertificate 4710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4710 1) := by change ¬ Nat.Prime 88745821; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 128431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4710 2) := by change ¬ Nat.Prime 88736401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6825877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4710 3) := by change ¬ Nat.Prime 88726981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12675283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4710 : row 4710 = 0 := (row_4710.checked).trans rfl
theorem model_count_4710 : modelRow 4710 = 0 := by decide
theorem modelrow_agrees_4710 : modelRow 4710 = row 4710 := model_count_4710.trans row_count_4710.symm
def row_4711 : RowCertificate 4711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4711 1) := by change ¬ Nat.Prime 88783507; exact PrattCertificates.composite_of_product (factor := 2719) (cofactor := 32653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4711 2) := by change ¬ Nat.Prime 88774085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17754817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4711 3) := by change ¬ Nat.Prime 88764663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29588221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4711 : row 4711 = 0 := (row_4711.checked).trans rfl
theorem model_count_4711 : modelRow 4711 = 0 := by decide
theorem modelrow_agrees_4711 : modelRow 4711 = row 4711 := model_count_4711.trans row_count_4711.symm
def row_4712 : RowCertificate 4712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4712 1) := by change ¬ Nat.Prime 88821201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29607067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4712 2) := by change Nat.Prime 88811777; exact ProjectEulerSolutions.P58PrimeNodes.p_88811777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4712 3) := by change ¬ Nat.Prime 88802353; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2065171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4712 : row 4712 = 1 := (row_4712.checked).trans rfl
theorem model_count_4712 : modelRow 4712 = 1 := by decide
theorem modelrow_agrees_4712 : modelRow 4712 = row 4712 := model_count_4712.trans row_count_4712.symm
def row_4713 : RowCertificate 4713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4713 1) := by change ¬ Nat.Prime 88858903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12694129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4713 2) := by change Nat.Prime 88849477; exact ProjectEulerSolutions.P58PrimeNodes.p_88849477.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4713 3) := by change ¬ Nat.Prime 88840051; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1216987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4713 : row 4713 = 1 := (row_4713.checked).trans rfl
theorem model_count_4713 : modelRow 4713 = 1 := by decide
theorem modelrow_agrees_4713 : modelRow 4713 = row 4713 := model_count_4713.trans row_count_4713.symm
def row_4714 : RowCertificate 4714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4714 1) := by change ¬ Nat.Prime 88896613; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6838201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4714 2) := by change ¬ Nat.Prime 88887185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17777437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4714 3) := by change ¬ Nat.Prime 88877757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29625919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4714 : row 4714 = 0 := (row_4714.checked).trans rfl
theorem model_count_4714 : modelRow 4714 = 0 := by decide
theorem modelrow_agrees_4714 : modelRow 4714 = row 4714 := model_count_4714.trans row_count_4714.symm
def row_4715 : RowCertificate 4715 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4715 1) := by change ¬ Nat.Prime 88934331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29644777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4715 2) := by change ¬ Nat.Prime 88924901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6840377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4715 3) := by change ¬ Nat.Prime 88915471; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2868241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4715 : row 4715 = 0 := (row_4715.checked).trans rfl
theorem model_count_4715 : modelRow 4715 = 0 := by decide
theorem modelrow_agrees_4715 : modelRow 4715 = row 4715 := model_count_4715.trans row_count_4715.symm
def row_4716 : RowCertificate 4716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4716 1) := by change ¬ Nat.Prime 88972057; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 566701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4716 2) := by change ¬ Nat.Prime 88962625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17792525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4716 3) := by change ¬ Nat.Prime 88953193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12707599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4716 : row 4716 = 0 := (row_4716.checked).trans rfl
theorem model_count_4716 : modelRow 4716 = 0 := by decide
theorem modelrow_agrees_4716 : modelRow 4716 = row 4716 := model_count_4716.trans row_count_4716.symm
def row_4717 : RowCertificate 4717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4717 1) := by change ¬ Nat.Prime 89009791; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6846907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4717 2) := by change Nat.Prime 89000357; exact ProjectEulerSolutions.P58PrimeNodes.p_89000357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4717 3) := by change ¬ Nat.Prime 88990923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29663641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4717 : row 4717 = 1 := (row_4717.checked).trans rfl
theorem model_count_4717 : modelRow 4717 = 1 := by decide
theorem modelrow_agrees_4717 : modelRow 4717 = row 4717 := model_count_4717.trans row_count_4717.symm
def row_4718 : RowCertificate 4718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4718 1) := by change ¬ Nat.Prime 89047533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29682511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4718 2) := by change Nat.Prime 89038097; exact ProjectEulerSolutions.P58PrimeNodes.p_89038097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4718 3) := by change ¬ Nat.Prime 89028661; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4685719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4718 : row 4718 = 1 := (row_4718.checked).trans rfl
theorem model_count_4718 : modelRow 4718 = 1 := by decide
theorem modelrow_agrees_4718 : modelRow 4718 = row 4718 := model_count_4718.trans row_count_4718.symm
def row_4719 : RowCertificate 4719 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4719 1) := by change ¬ Nat.Prime 89085283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12726469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4719 2) := by change ¬ Nat.Prime 89075845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17815169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4719 3) := by change ¬ Nat.Prime 89066407; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 817123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4719 : row 4719 = 0 := (row_4719.checked).trans rfl
theorem model_count_4719 : modelRow 4719 = 0 := by decide
theorem modelrow_agrees_4719 : modelRow 4719 = row 4719 := model_count_4719.trans row_count_4719.symm
def row_4720 : RowCertificate 4720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4720 1) := by change ¬ Nat.Prime 89123041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12731863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4720 2) := by change ¬ Nat.Prime 89113601; exact PrattCertificates.composite_of_product (factor := 2333) (cofactor := 38197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4720 3) := by change ¬ Nat.Prime 89104161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29701387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4720 : row 4720 = 0 := (row_4720.checked).trans rfl
theorem model_count_4720 : modelRow 4720 = 0 := by decide
theorem modelrow_agrees_4720 : modelRow 4720 = row 4720 := model_count_4720.trans row_count_4720.symm
def row_4721 : RowCertificate 4721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4721 1) := by change ¬ Nat.Prime 89160807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29720269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4721 2) := by change ¬ Nat.Prime 89151365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17830273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4721 3) := by change ¬ Nat.Prime 89141923; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6857071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4721 : row 4721 = 0 := (row_4721.checked).trans rfl
theorem model_count_4721 : modelRow 4721 = 0 := by decide
theorem modelrow_agrees_4721 : modelRow 4721 = row 4721 := model_count_4721.trans row_count_4721.symm
def row_4722 : RowCertificate 4722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4722 1) := by change ¬ Nat.Prime 89198581; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 919573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4722 2) := by change ¬ Nat.Prime 89189137; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1462117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4722 3) := by change Nat.Prime 89179693; exact ProjectEulerSolutions.P58PrimeNodes.p_89179693.prime
    exact decide_eq_true prime
theorem row_count_4722 : row 4722 = 1 := (row_4722.checked).trans rfl
theorem model_count_4722 : modelRow 4722 = 1 := by decide
theorem modelrow_agrees_4722 : modelRow 4722 = row 4722 := model_count_4722.trans row_count_4722.symm
def row_4723 : RowCertificate 4723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4723 1) := by change ¬ Nat.Prime 89236363; exact PrattCertificates.composite_of_product (factor := 3217) (cofactor := 27739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4723 2) := by change ¬ Nat.Prime 89226917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6863609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4723 3) := by change ¬ Nat.Prime 89217471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29739157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4723 : row 4723 = 0 := (row_4723.checked).trans rfl
theorem model_count_4723 : modelRow 4723 = 0 := by decide
theorem modelrow_agrees_4723 : modelRow 4723 = row 4723 := model_count_4723.trans row_count_4723.symm
def row_4724 : RowCertificate 4724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4724 1) := by change ¬ Nat.Prime 89274153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29758051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4724 2) := by change ¬ Nat.Prime 89264705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17852941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4724 3) := by change ¬ Nat.Prime 89255257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12750751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4724 : row 4724 = 0 := (row_4724.checked).trans rfl
theorem model_count_4724 : modelRow 4724 = 0 := by decide
theorem modelrow_agrees_4724 : modelRow 4724 = row 4724 := model_count_4724.trans row_count_4724.symm
def row_4725 : RowCertificate 4725 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4725 1) := by change ¬ Nat.Prime 89311951; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4700629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4725 2) := by change ¬ Nat.Prime 89302501; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 819289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4725 3) := by change ¬ Nat.Prime 89293051; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2880421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4725 : row 4725 = 0 := (row_4725.checked).trans rfl
theorem model_count_4725 : modelRow 4725 = 0 := by decide
theorem modelrow_agrees_4725 : modelRow 4725 = row 4725 := model_count_4725.trans row_count_4725.symm
def row_4726 : RowCertificate 4726 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4726 1) := by change ¬ Nat.Prime 89349757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12764251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4726 2) := by change ¬ Nat.Prime 89340305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17868061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4726 3) := by change ¬ Nat.Prime 89330853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29776951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4726 : row 4726 = 0 := (row_4726.checked).trans rfl
theorem model_count_4726 : modelRow 4726 = 0 := by decide
theorem modelrow_agrees_4726 : modelRow 4726 = row 4726 := model_count_4726.trans row_count_4726.symm
def row_4727 : RowCertificate 4727 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4727 1) := by change ¬ Nat.Prime 89387571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29795857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4727 2) := by change ¬ Nat.Prime 89378117; exact PrattCertificates.composite_of_product (factor := 2593) (cofactor := 34469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4727 3) := by change ¬ Nat.Prime 89368663; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2078341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4727 : row 4727 = 0 := (row_4727.checked).trans rfl
theorem model_count_4727 : modelRow 4727 = 0 := by decide
theorem modelrow_agrees_4727 : modelRow 4727 = row 4727 := model_count_4727.trans row_count_4727.symm
def row_4728 : RowCertificate 4728 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4728 1) := by change ¬ Nat.Prime 89425393; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 1131967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4728 2) := by change ¬ Nat.Prime 89415937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6878149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4728 3) := by change ¬ Nat.Prime 89406481; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 88609) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4728 : row 4728 = 0 := (row_4728.checked).trans rfl
theorem model_count_4728 : modelRow 4728 = 0 := by decide
theorem modelrow_agrees_4728 : modelRow 4728 = row 4728 := model_count_4728.trans row_count_4728.symm
def row_4729 : RowCertificate 4729 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4729 1) := by change ¬ Nat.Prime 89463223; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 243769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4729 2) := by change ¬ Nat.Prime 89453765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17890753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4729 3) := by change ¬ Nat.Prime 89444307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29814769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4729 : row 4729 = 0 := (row_4729.checked).trans rfl
theorem model_count_4729 : modelRow 4729 = 0 := by decide
theorem modelrow_agrees_4729 : modelRow 4729 = row 4729 := model_count_4729.trans row_count_4729.symm
def row_4730 : RowCertificate 4730 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4730 1) := by change ¬ Nat.Prime 89501061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29833687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4730 2) := by change Nat.Prime 89491601; exact ProjectEulerSolutions.P58PrimeNodes.p_89491601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4730 3) := by change ¬ Nat.Prime 89482141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12783163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4730 : row 4730 = 1 := (row_4730.checked).trans rfl
theorem model_count_4730 : modelRow 4730 = 1 := by decide
theorem modelrow_agrees_4730 : modelRow 4730 = row 4730 := model_count_4730.trans row_count_4730.symm
def row_4731 : RowCertificate 4731 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4731 1) := by change Nat.Prime 89538907; exact ProjectEulerSolutions.P58PrimeNodes.p_89538907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4731 2) := by change ¬ Nat.Prime 89529445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17905889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4731 3) := by change ¬ Nat.Prime 89519983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12788569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4731 : row 4731 = 1 := (row_4731.checked).trans rfl
theorem model_count_4731 : modelRow 4731 = 1 := by decide
theorem modelrow_agrees_4731 : modelRow 4731 = row 4731 := model_count_4731.trans row_count_4731.symm
def row_4732 : RowCertificate 4732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4732 1) := by change ¬ Nat.Prime 89576761; exact PrattCertificates.composite_of_product (factor := 3373) (cofactor := 26557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4732 2) := by change ¬ Nat.Prime 89567297; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1689949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4732 3) := by change ¬ Nat.Prime 89557833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29852611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4732 : row 4732 = 0 := (row_4732.checked).trans rfl
theorem model_count_4732 : modelRow 4732 = 0 := by decide
theorem modelrow_agrees_4732 : modelRow 4732 = row 4732 := model_count_4732.trans row_count_4732.symm
def row_4733 : RowCertificate 4733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4733 1) := by change ¬ Nat.Prime 89614623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29871541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4733 2) := by change ¬ Nat.Prime 89605157; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3089833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4733 3) := by change Nat.Prime 89595691; exact ProjectEulerSolutions.P58PrimeNodes.p_89595691.prime
    exact decide_eq_true prime
theorem row_count_4733 : row 4733 = 1 := (row_4733.checked).trans rfl
theorem model_count_4733 : modelRow 4733 = 1 := by decide
theorem modelrow_agrees_4733 : modelRow 4733 = row 4733 := model_count_4733.trans row_count_4733.symm
def row_4734 : RowCertificate 4734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4734 1) := by change ¬ Nat.Prime 89652493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 12807499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4734 2) := by change ¬ Nat.Prime 89643025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17928605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4734 3) := by change ¬ Nat.Prime 89633557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 6894889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4734 : row 4734 = 0 := (row_4734.checked).trans rfl
theorem model_count_4734 : modelRow 4734 = 0 := by decide
theorem modelrow_agrees_4734 : modelRow 4734 = row 4734 := model_count_4734.trans row_count_4734.symm
def row_4735 : RowCertificate 4735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4735 1) := by change ¬ Nat.Prime 89690371; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 924643) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4735 2) := by change ¬ Nat.Prime 89680901; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 455233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4735 3) := by change ¬ Nat.Prime 89671431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29890477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4735 : row 4735 = 0 := (row_4735.checked).trans rfl
theorem model_count_4735 : modelRow 4735 = 0 := by decide
theorem modelrow_agrees_4735 : modelRow 4735 = row 4735 := model_count_4735.trans row_count_4735.symm
def row_4736 : RowCertificate 4736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4736 1) := by change ¬ Nat.Prime 89728257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 29909419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4736 2) := by change ¬ Nat.Prime 89718785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 17943757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4736 3) := by change ¬ Nat.Prime 89709313; exact PrattCertificates.composite_of_product (factor := 4483) (cofactor := 20011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4736 : row 4736 = 0 := (row_4736.checked).trans rfl
theorem model_count_4736 : modelRow 4736 = 0 := by decide
theorem modelrow_agrees_4736 : modelRow 4736 = row 4736 := model_count_4736.trans row_count_4736.symm
private def atomic_4609 : CountSegment 1 10 13120 4608 4609 2090 2091 := CountSegment.single (first := 4608) (initial := 2090) row_count_4609 (by decide)
private theorem atomic_4609_agreement : RowAgreement 4608 4609 := RowAgreement.single (first := 4608) modelrow_agrees_4609
private def atomic_4610 : CountSegment 1 10 13120 4609 4610 2091 2091 := CountSegment.single (first := 4609) (initial := 2091) row_count_4610 (by decide)
private theorem atomic_4610_agreement : RowAgreement 4609 4610 := RowAgreement.single (first := 4609) modelrow_agrees_4610
private def atomic_4611 : CountSegment 1 10 13120 4610 4611 2091 2091 := CountSegment.single (first := 4610) (initial := 2091) row_count_4611 (by decide)
private theorem atomic_4611_agreement : RowAgreement 4610 4611 := RowAgreement.single (first := 4610) modelrow_agrees_4611
private def atomic_4612 : CountSegment 1 10 13120 4611 4612 2091 2091 := CountSegment.single (first := 4611) (initial := 2091) row_count_4612 (by decide)
private theorem atomic_4612_agreement : RowAgreement 4611 4612 := RowAgreement.single (first := 4611) modelrow_agrees_4612
private def atomic_4613 : CountSegment 1 10 13120 4612 4613 2091 2091 := CountSegment.single (first := 4612) (initial := 2091) row_count_4613 (by decide)
private theorem atomic_4613_agreement : RowAgreement 4612 4613 := RowAgreement.single (first := 4612) modelrow_agrees_4613
private def atomic_4614 : CountSegment 1 10 13120 4613 4614 2091 2091 := CountSegment.single (first := 4613) (initial := 2091) row_count_4614 (by decide)
private theorem atomic_4614_agreement : RowAgreement 4613 4614 := RowAgreement.single (first := 4613) modelrow_agrees_4614
private def atomic_4615 : CountSegment 1 10 13120 4614 4615 2091 2091 := CountSegment.single (first := 4614) (initial := 2091) row_count_4615 (by decide)
private theorem atomic_4615_agreement : RowAgreement 4614 4615 := RowAgreement.single (first := 4614) modelrow_agrees_4615
private def atomic_4616 : CountSegment 1 10 13120 4615 4616 2091 2092 := CountSegment.single (first := 4615) (initial := 2091) row_count_4616 (by decide)
private theorem atomic_4616_agreement : RowAgreement 4615 4616 := RowAgreement.single (first := 4615) modelrow_agrees_4616
private def atomic_4617 : CountSegment 1 10 13120 4616 4617 2092 2092 := CountSegment.single (first := 4616) (initial := 2092) row_count_4617 (by decide)
private theorem atomic_4617_agreement : RowAgreement 4616 4617 := RowAgreement.single (first := 4616) modelrow_agrees_4617
private def atomic_4618 : CountSegment 1 10 13120 4617 4618 2092 2092 := CountSegment.single (first := 4617) (initial := 2092) row_count_4618 (by decide)
private theorem atomic_4618_agreement : RowAgreement 4617 4618 := RowAgreement.single (first := 4617) modelrow_agrees_4618
private def atomic_4619 : CountSegment 1 10 13120 4618 4619 2092 2092 := CountSegment.single (first := 4618) (initial := 2092) row_count_4619 (by decide)
private theorem atomic_4619_agreement : RowAgreement 4618 4619 := RowAgreement.single (first := 4618) modelrow_agrees_4619
private def atomic_4620 : CountSegment 1 10 13120 4619 4620 2092 2094 := CountSegment.single (first := 4619) (initial := 2092) row_count_4620 (by decide)
private theorem atomic_4620_agreement : RowAgreement 4619 4620 := RowAgreement.single (first := 4619) modelrow_agrees_4620
private def atomic_4621 : CountSegment 1 10 13120 4620 4621 2094 2094 := CountSegment.single (first := 4620) (initial := 2094) row_count_4621 (by decide)
private theorem atomic_4621_agreement : RowAgreement 4620 4621 := RowAgreement.single (first := 4620) modelrow_agrees_4621
private def atomic_4622 : CountSegment 1 10 13120 4621 4622 2094 2094 := CountSegment.single (first := 4621) (initial := 2094) row_count_4622 (by decide)
private theorem atomic_4622_agreement : RowAgreement 4621 4622 := RowAgreement.single (first := 4621) modelrow_agrees_4622
private def atomic_4623 : CountSegment 1 10 13120 4622 4623 2094 2095 := CountSegment.single (first := 4622) (initial := 2094) row_count_4623 (by decide)
private theorem atomic_4623_agreement : RowAgreement 4622 4623 := RowAgreement.single (first := 4622) modelrow_agrees_4623
private def atomic_4624 : CountSegment 1 10 13120 4623 4624 2095 2096 := CountSegment.single (first := 4623) (initial := 2095) row_count_4624 (by decide)
private theorem atomic_4624_agreement : RowAgreement 4623 4624 := RowAgreement.single (first := 4623) modelrow_agrees_4624
private def atomic_4625 : CountSegment 1 10 13120 4624 4625 2096 2096 := CountSegment.single (first := 4624) (initial := 2096) row_count_4625 (by decide)
private theorem atomic_4625_agreement : RowAgreement 4624 4625 := RowAgreement.single (first := 4624) modelrow_agrees_4625
private def atomic_4626 : CountSegment 1 10 13120 4625 4626 2096 2096 := CountSegment.single (first := 4625) (initial := 2096) row_count_4626 (by decide)
private theorem atomic_4626_agreement : RowAgreement 4625 4626 := RowAgreement.single (first := 4625) modelrow_agrees_4626
private def atomic_4627 : CountSegment 1 10 13120 4626 4627 2096 2096 := CountSegment.single (first := 4626) (initial := 2096) row_count_4627 (by decide)
private theorem atomic_4627_agreement : RowAgreement 4626 4627 := RowAgreement.single (first := 4626) modelrow_agrees_4627
private def atomic_4628 : CountSegment 1 10 13120 4627 4628 2096 2097 := CountSegment.single (first := 4627) (initial := 2096) row_count_4628 (by decide)
private theorem atomic_4628_agreement : RowAgreement 4627 4628 := RowAgreement.single (first := 4627) modelrow_agrees_4628
private def atomic_4629 : CountSegment 1 10 13120 4628 4629 2097 2097 := CountSegment.single (first := 4628) (initial := 2097) row_count_4629 (by decide)
private theorem atomic_4629_agreement : RowAgreement 4628 4629 := RowAgreement.single (first := 4628) modelrow_agrees_4629
private def atomic_4630 : CountSegment 1 10 13120 4629 4630 2097 2098 := CountSegment.single (first := 4629) (initial := 2097) row_count_4630 (by decide)
private theorem atomic_4630_agreement : RowAgreement 4629 4630 := RowAgreement.single (first := 4629) modelrow_agrees_4630
private def atomic_4631 : CountSegment 1 10 13120 4630 4631 2098 2099 := CountSegment.single (first := 4630) (initial := 2098) row_count_4631 (by decide)
private theorem atomic_4631_agreement : RowAgreement 4630 4631 := RowAgreement.single (first := 4630) modelrow_agrees_4631
private def atomic_4632 : CountSegment 1 10 13120 4631 4632 2099 2099 := CountSegment.single (first := 4631) (initial := 2099) row_count_4632 (by decide)
private theorem atomic_4632_agreement : RowAgreement 4631 4632 := RowAgreement.single (first := 4631) modelrow_agrees_4632
private def atomic_4633 : CountSegment 1 10 13120 4632 4633 2099 2100 := CountSegment.single (first := 4632) (initial := 2099) row_count_4633 (by decide)
private theorem atomic_4633_agreement : RowAgreement 4632 4633 := RowAgreement.single (first := 4632) modelrow_agrees_4633
private def atomic_4634 : CountSegment 1 10 13120 4633 4634 2100 2100 := CountSegment.single (first := 4633) (initial := 2100) row_count_4634 (by decide)
private theorem atomic_4634_agreement : RowAgreement 4633 4634 := RowAgreement.single (first := 4633) modelrow_agrees_4634
private def atomic_4635 : CountSegment 1 10 13120 4634 4635 2100 2101 := CountSegment.single (first := 4634) (initial := 2100) row_count_4635 (by decide)
private theorem atomic_4635_agreement : RowAgreement 4634 4635 := RowAgreement.single (first := 4634) modelrow_agrees_4635
private def atomic_4636 : CountSegment 1 10 13120 4635 4636 2101 2101 := CountSegment.single (first := 4635) (initial := 2101) row_count_4636 (by decide)
private theorem atomic_4636_agreement : RowAgreement 4635 4636 := RowAgreement.single (first := 4635) modelrow_agrees_4636
private def atomic_4637 : CountSegment 1 10 13120 4636 4637 2101 2101 := CountSegment.single (first := 4636) (initial := 2101) row_count_4637 (by decide)
private theorem atomic_4637_agreement : RowAgreement 4636 4637 := RowAgreement.single (first := 4636) modelrow_agrees_4637
private def atomic_4638 : CountSegment 1 10 13120 4637 4638 2101 2102 := CountSegment.single (first := 4637) (initial := 2101) row_count_4638 (by decide)
private theorem atomic_4638_agreement : RowAgreement 4637 4638 := RowAgreement.single (first := 4637) modelrow_agrees_4638
private def atomic_4639 : CountSegment 1 10 13120 4638 4639 2102 2102 := CountSegment.single (first := 4638) (initial := 2102) row_count_4639 (by decide)
private theorem atomic_4639_agreement : RowAgreement 4638 4639 := RowAgreement.single (first := 4638) modelrow_agrees_4639
private def atomic_4640 : CountSegment 1 10 13120 4639 4640 2102 2103 := CountSegment.single (first := 4639) (initial := 2102) row_count_4640 (by decide)
private theorem atomic_4640_agreement : RowAgreement 4639 4640 := RowAgreement.single (first := 4639) modelrow_agrees_4640
private def atomic_4641 : CountSegment 1 10 13120 4640 4641 2103 2103 := CountSegment.single (first := 4640) (initial := 2103) row_count_4641 (by decide)
private theorem atomic_4641_agreement : RowAgreement 4640 4641 := RowAgreement.single (first := 4640) modelrow_agrees_4641
private def atomic_4642 : CountSegment 1 10 13120 4641 4642 2103 2103 := CountSegment.single (first := 4641) (initial := 2103) row_count_4642 (by decide)
private theorem atomic_4642_agreement : RowAgreement 4641 4642 := RowAgreement.single (first := 4641) modelrow_agrees_4642
private def atomic_4643 : CountSegment 1 10 13120 4642 4643 2103 2103 := CountSegment.single (first := 4642) (initial := 2103) row_count_4643 (by decide)
private theorem atomic_4643_agreement : RowAgreement 4642 4643 := RowAgreement.single (first := 4642) modelrow_agrees_4643
private def atomic_4644 : CountSegment 1 10 13120 4643 4644 2103 2104 := CountSegment.single (first := 4643) (initial := 2103) row_count_4644 (by decide)
private theorem atomic_4644_agreement : RowAgreement 4643 4644 := RowAgreement.single (first := 4643) modelrow_agrees_4644
private def atomic_4645 : CountSegment 1 10 13120 4644 4645 2104 2104 := CountSegment.single (first := 4644) (initial := 2104) row_count_4645 (by decide)
private theorem atomic_4645_agreement : RowAgreement 4644 4645 := RowAgreement.single (first := 4644) modelrow_agrees_4645
private def atomic_4646 : CountSegment 1 10 13120 4645 4646 2104 2104 := CountSegment.single (first := 4645) (initial := 2104) row_count_4646 (by decide)
private theorem atomic_4646_agreement : RowAgreement 4645 4646 := RowAgreement.single (first := 4645) modelrow_agrees_4646
private def atomic_4647 : CountSegment 1 10 13120 4646 4647 2104 2105 := CountSegment.single (first := 4646) (initial := 2104) row_count_4647 (by decide)
private theorem atomic_4647_agreement : RowAgreement 4646 4647 := RowAgreement.single (first := 4646) modelrow_agrees_4647
private def atomic_4648 : CountSegment 1 10 13120 4647 4648 2105 2105 := CountSegment.single (first := 4647) (initial := 2105) row_count_4648 (by decide)
private theorem atomic_4648_agreement : RowAgreement 4647 4648 := RowAgreement.single (first := 4647) modelrow_agrees_4648
private def atomic_4649 : CountSegment 1 10 13120 4648 4649 2105 2105 := CountSegment.single (first := 4648) (initial := 2105) row_count_4649 (by decide)
private theorem atomic_4649_agreement : RowAgreement 4648 4649 := RowAgreement.single (first := 4648) modelrow_agrees_4649
private def atomic_4650 : CountSegment 1 10 13120 4649 4650 2105 2105 := CountSegment.single (first := 4649) (initial := 2105) row_count_4650 (by decide)
private theorem atomic_4650_agreement : RowAgreement 4649 4650 := RowAgreement.single (first := 4649) modelrow_agrees_4650
private def atomic_4651 : CountSegment 1 10 13120 4650 4651 2105 2105 := CountSegment.single (first := 4650) (initial := 2105) row_count_4651 (by decide)
private theorem atomic_4651_agreement : RowAgreement 4650 4651 := RowAgreement.single (first := 4650) modelrow_agrees_4651
private def atomic_4652 : CountSegment 1 10 13120 4651 4652 2105 2105 := CountSegment.single (first := 4651) (initial := 2105) row_count_4652 (by decide)
private theorem atomic_4652_agreement : RowAgreement 4651 4652 := RowAgreement.single (first := 4651) modelrow_agrees_4652
private def atomic_4653 : CountSegment 1 10 13120 4652 4653 2105 2106 := CountSegment.single (first := 4652) (initial := 2105) row_count_4653 (by decide)
private theorem atomic_4653_agreement : RowAgreement 4652 4653 := RowAgreement.single (first := 4652) modelrow_agrees_4653
private def atomic_4654 : CountSegment 1 10 13120 4653 4654 2106 2107 := CountSegment.single (first := 4653) (initial := 2106) row_count_4654 (by decide)
private theorem atomic_4654_agreement : RowAgreement 4653 4654 := RowAgreement.single (first := 4653) modelrow_agrees_4654
private def atomic_4655 : CountSegment 1 10 13120 4654 4655 2107 2109 := CountSegment.single (first := 4654) (initial := 2107) row_count_4655 (by decide)
private theorem atomic_4655_agreement : RowAgreement 4654 4655 := RowAgreement.single (first := 4654) modelrow_agrees_4655
private def atomic_4656 : CountSegment 1 10 13120 4655 4656 2109 2109 := CountSegment.single (first := 4655) (initial := 2109) row_count_4656 (by decide)
private theorem atomic_4656_agreement : RowAgreement 4655 4656 := RowAgreement.single (first := 4655) modelrow_agrees_4656
private def atomic_4657 : CountSegment 1 10 13120 4656 4657 2109 2110 := CountSegment.single (first := 4656) (initial := 2109) row_count_4657 (by decide)
private theorem atomic_4657_agreement : RowAgreement 4656 4657 := RowAgreement.single (first := 4656) modelrow_agrees_4657
private def atomic_4658 : CountSegment 1 10 13120 4657 4658 2110 2110 := CountSegment.single (first := 4657) (initial := 2110) row_count_4658 (by decide)
private theorem atomic_4658_agreement : RowAgreement 4657 4658 := RowAgreement.single (first := 4657) modelrow_agrees_4658
private def atomic_4659 : CountSegment 1 10 13120 4658 4659 2110 2111 := CountSegment.single (first := 4658) (initial := 2110) row_count_4659 (by decide)
private theorem atomic_4659_agreement : RowAgreement 4658 4659 := RowAgreement.single (first := 4658) modelrow_agrees_4659
private def atomic_4660 : CountSegment 1 10 13120 4659 4660 2111 2112 := CountSegment.single (first := 4659) (initial := 2111) row_count_4660 (by decide)
private theorem atomic_4660_agreement : RowAgreement 4659 4660 := RowAgreement.single (first := 4659) modelrow_agrees_4660
private def atomic_4661 : CountSegment 1 10 13120 4660 4661 2112 2112 := CountSegment.single (first := 4660) (initial := 2112) row_count_4661 (by decide)
private theorem atomic_4661_agreement : RowAgreement 4660 4661 := RowAgreement.single (first := 4660) modelrow_agrees_4661
private def atomic_4662 : CountSegment 1 10 13120 4661 4662 2112 2114 := CountSegment.single (first := 4661) (initial := 2112) row_count_4662 (by decide)
private theorem atomic_4662_agreement : RowAgreement 4661 4662 := RowAgreement.single (first := 4661) modelrow_agrees_4662
private def atomic_4663 : CountSegment 1 10 13120 4662 4663 2114 2114 := CountSegment.single (first := 4662) (initial := 2114) row_count_4663 (by decide)
private theorem atomic_4663_agreement : RowAgreement 4662 4663 := RowAgreement.single (first := 4662) modelrow_agrees_4663
private def atomic_4664 : CountSegment 1 10 13120 4663 4664 2114 2114 := CountSegment.single (first := 4663) (initial := 2114) row_count_4664 (by decide)
private theorem atomic_4664_agreement : RowAgreement 4663 4664 := RowAgreement.single (first := 4663) modelrow_agrees_4664
private def atomic_4665 : CountSegment 1 10 13120 4664 4665 2114 2114 := CountSegment.single (first := 4664) (initial := 2114) row_count_4665 (by decide)
private theorem atomic_4665_agreement : RowAgreement 4664 4665 := RowAgreement.single (first := 4664) modelrow_agrees_4665
private def atomic_4666 : CountSegment 1 10 13120 4665 4666 2114 2114 := CountSegment.single (first := 4665) (initial := 2114) row_count_4666 (by decide)
private theorem atomic_4666_agreement : RowAgreement 4665 4666 := RowAgreement.single (first := 4665) modelrow_agrees_4666
private def atomic_4667 : CountSegment 1 10 13120 4666 4667 2114 2114 := CountSegment.single (first := 4666) (initial := 2114) row_count_4667 (by decide)
private theorem atomic_4667_agreement : RowAgreement 4666 4667 := RowAgreement.single (first := 4666) modelrow_agrees_4667
private def atomic_4668 : CountSegment 1 10 13120 4667 4668 2114 2115 := CountSegment.single (first := 4667) (initial := 2114) row_count_4668 (by decide)
private theorem atomic_4668_agreement : RowAgreement 4667 4668 := RowAgreement.single (first := 4667) modelrow_agrees_4668
private def atomic_4669 : CountSegment 1 10 13120 4668 4669 2115 2115 := CountSegment.single (first := 4668) (initial := 2115) row_count_4669 (by decide)
private theorem atomic_4669_agreement : RowAgreement 4668 4669 := RowAgreement.single (first := 4668) modelrow_agrees_4669
private def atomic_4670 : CountSegment 1 10 13120 4669 4670 2115 2116 := CountSegment.single (first := 4669) (initial := 2115) row_count_4670 (by decide)
private theorem atomic_4670_agreement : RowAgreement 4669 4670 := RowAgreement.single (first := 4669) modelrow_agrees_4670
private def atomic_4671 : CountSegment 1 10 13120 4670 4671 2116 2116 := CountSegment.single (first := 4670) (initial := 2116) row_count_4671 (by decide)
private theorem atomic_4671_agreement : RowAgreement 4670 4671 := RowAgreement.single (first := 4670) modelrow_agrees_4671
private def atomic_4672 : CountSegment 1 10 13120 4671 4672 2116 2117 := CountSegment.single (first := 4671) (initial := 2116) row_count_4672 (by decide)
private theorem atomic_4672_agreement : RowAgreement 4671 4672 := RowAgreement.single (first := 4671) modelrow_agrees_4672
private def atomic_4673 : CountSegment 1 10 13120 4672 4673 2117 2117 := CountSegment.single (first := 4672) (initial := 2117) row_count_4673 (by decide)
private theorem atomic_4673_agreement : RowAgreement 4672 4673 := RowAgreement.single (first := 4672) modelrow_agrees_4673
private def atomic_4674 : CountSegment 1 10 13120 4673 4674 2117 2117 := CountSegment.single (first := 4673) (initial := 2117) row_count_4674 (by decide)
private theorem atomic_4674_agreement : RowAgreement 4673 4674 := RowAgreement.single (first := 4673) modelrow_agrees_4674
private def atomic_4675 : CountSegment 1 10 13120 4674 4675 2117 2117 := CountSegment.single (first := 4674) (initial := 2117) row_count_4675 (by decide)
private theorem atomic_4675_agreement : RowAgreement 4674 4675 := RowAgreement.single (first := 4674) modelrow_agrees_4675
private def atomic_4676 : CountSegment 1 10 13120 4675 4676 2117 2118 := CountSegment.single (first := 4675) (initial := 2117) row_count_4676 (by decide)
private theorem atomic_4676_agreement : RowAgreement 4675 4676 := RowAgreement.single (first := 4675) modelrow_agrees_4676
private def atomic_4677 : CountSegment 1 10 13120 4676 4677 2118 2118 := CountSegment.single (first := 4676) (initial := 2118) row_count_4677 (by decide)
private theorem atomic_4677_agreement : RowAgreement 4676 4677 := RowAgreement.single (first := 4676) modelrow_agrees_4677
private def atomic_4678 : CountSegment 1 10 13120 4677 4678 2118 2119 := CountSegment.single (first := 4677) (initial := 2118) row_count_4678 (by decide)
private theorem atomic_4678_agreement : RowAgreement 4677 4678 := RowAgreement.single (first := 4677) modelrow_agrees_4678
private def atomic_4679 : CountSegment 1 10 13120 4678 4679 2119 2120 := CountSegment.single (first := 4678) (initial := 2119) row_count_4679 (by decide)
private theorem atomic_4679_agreement : RowAgreement 4678 4679 := RowAgreement.single (first := 4678) modelrow_agrees_4679
private def atomic_4680 : CountSegment 1 10 13120 4679 4680 2120 2121 := CountSegment.single (first := 4679) (initial := 2120) row_count_4680 (by decide)
private theorem atomic_4680_agreement : RowAgreement 4679 4680 := RowAgreement.single (first := 4679) modelrow_agrees_4680
private def atomic_4681 : CountSegment 1 10 13120 4680 4681 2121 2122 := CountSegment.single (first := 4680) (initial := 2121) row_count_4681 (by decide)
private theorem atomic_4681_agreement : RowAgreement 4680 4681 := RowAgreement.single (first := 4680) modelrow_agrees_4681
private def atomic_4682 : CountSegment 1 10 13120 4681 4682 2122 2122 := CountSegment.single (first := 4681) (initial := 2122) row_count_4682 (by decide)
private theorem atomic_4682_agreement : RowAgreement 4681 4682 := RowAgreement.single (first := 4681) modelrow_agrees_4682
private def atomic_4683 : CountSegment 1 10 13120 4682 4683 2122 2123 := CountSegment.single (first := 4682) (initial := 2122) row_count_4683 (by decide)
private theorem atomic_4683_agreement : RowAgreement 4682 4683 := RowAgreement.single (first := 4682) modelrow_agrees_4683
private def atomic_4684 : CountSegment 1 10 13120 4683 4684 2123 2123 := CountSegment.single (first := 4683) (initial := 2123) row_count_4684 (by decide)
private theorem atomic_4684_agreement : RowAgreement 4683 4684 := RowAgreement.single (first := 4683) modelrow_agrees_4684
private def atomic_4685 : CountSegment 1 10 13120 4684 4685 2123 2123 := CountSegment.single (first := 4684) (initial := 2123) row_count_4685 (by decide)
private theorem atomic_4685_agreement : RowAgreement 4684 4685 := RowAgreement.single (first := 4684) modelrow_agrees_4685
private def atomic_4686 : CountSegment 1 10 13120 4685 4686 2123 2123 := CountSegment.single (first := 4685) (initial := 2123) row_count_4686 (by decide)
private theorem atomic_4686_agreement : RowAgreement 4685 4686 := RowAgreement.single (first := 4685) modelrow_agrees_4686
private def atomic_4687 : CountSegment 1 10 13120 4686 4687 2123 2124 := CountSegment.single (first := 4686) (initial := 2123) row_count_4687 (by decide)
private theorem atomic_4687_agreement : RowAgreement 4686 4687 := RowAgreement.single (first := 4686) modelrow_agrees_4687
private def atomic_4688 : CountSegment 1 10 13120 4687 4688 2124 2124 := CountSegment.single (first := 4687) (initial := 2124) row_count_4688 (by decide)
private theorem atomic_4688_agreement : RowAgreement 4687 4688 := RowAgreement.single (first := 4687) modelrow_agrees_4688
private def atomic_4689 : CountSegment 1 10 13120 4688 4689 2124 2124 := CountSegment.single (first := 4688) (initial := 2124) row_count_4689 (by decide)
private theorem atomic_4689_agreement : RowAgreement 4688 4689 := RowAgreement.single (first := 4688) modelrow_agrees_4689
private def atomic_4690 : CountSegment 1 10 13120 4689 4690 2124 2124 := CountSegment.single (first := 4689) (initial := 2124) row_count_4690 (by decide)
private theorem atomic_4690_agreement : RowAgreement 4689 4690 := RowAgreement.single (first := 4689) modelrow_agrees_4690
private def atomic_4691 : CountSegment 1 10 13120 4690 4691 2124 2124 := CountSegment.single (first := 4690) (initial := 2124) row_count_4691 (by decide)
private theorem atomic_4691_agreement : RowAgreement 4690 4691 := RowAgreement.single (first := 4690) modelrow_agrees_4691
private def atomic_4692 : CountSegment 1 10 13120 4691 4692 2124 2124 := CountSegment.single (first := 4691) (initial := 2124) row_count_4692 (by decide)
private theorem atomic_4692_agreement : RowAgreement 4691 4692 := RowAgreement.single (first := 4691) modelrow_agrees_4692
private def atomic_4693 : CountSegment 1 10 13120 4692 4693 2124 2125 := CountSegment.single (first := 4692) (initial := 2124) row_count_4693 (by decide)
private theorem atomic_4693_agreement : RowAgreement 4692 4693 := RowAgreement.single (first := 4692) modelrow_agrees_4693
private def atomic_4694 : CountSegment 1 10 13120 4693 4694 2125 2125 := CountSegment.single (first := 4693) (initial := 2125) row_count_4694 (by decide)
private theorem atomic_4694_agreement : RowAgreement 4693 4694 := RowAgreement.single (first := 4693) modelrow_agrees_4694
private def atomic_4695 : CountSegment 1 10 13120 4694 4695 2125 2125 := CountSegment.single (first := 4694) (initial := 2125) row_count_4695 (by decide)
private theorem atomic_4695_agreement : RowAgreement 4694 4695 := RowAgreement.single (first := 4694) modelrow_agrees_4695
private def atomic_4696 : CountSegment 1 10 13120 4695 4696 2125 2125 := CountSegment.single (first := 4695) (initial := 2125) row_count_4696 (by decide)
private theorem atomic_4696_agreement : RowAgreement 4695 4696 := RowAgreement.single (first := 4695) modelrow_agrees_4696
private def atomic_4697 : CountSegment 1 10 13120 4696 4697 2125 2125 := CountSegment.single (first := 4696) (initial := 2125) row_count_4697 (by decide)
private theorem atomic_4697_agreement : RowAgreement 4696 4697 := RowAgreement.single (first := 4696) modelrow_agrees_4697
private def atomic_4698 : CountSegment 1 10 13120 4697 4698 2125 2125 := CountSegment.single (first := 4697) (initial := 2125) row_count_4698 (by decide)
private theorem atomic_4698_agreement : RowAgreement 4697 4698 := RowAgreement.single (first := 4697) modelrow_agrees_4698
private def atomic_4699 : CountSegment 1 10 13120 4698 4699 2125 2125 := CountSegment.single (first := 4698) (initial := 2125) row_count_4699 (by decide)
private theorem atomic_4699_agreement : RowAgreement 4698 4699 := RowAgreement.single (first := 4698) modelrow_agrees_4699
private def atomic_4700 : CountSegment 1 10 13120 4699 4700 2125 2126 := CountSegment.single (first := 4699) (initial := 2125) row_count_4700 (by decide)
private theorem atomic_4700_agreement : RowAgreement 4699 4700 := RowAgreement.single (first := 4699) modelrow_agrees_4700
private def atomic_4701 : CountSegment 1 10 13120 4700 4701 2126 2127 := CountSegment.single (first := 4700) (initial := 2126) row_count_4701 (by decide)
private theorem atomic_4701_agreement : RowAgreement 4700 4701 := RowAgreement.single (first := 4700) modelrow_agrees_4701
private def atomic_4702 : CountSegment 1 10 13120 4701 4702 2127 2128 := CountSegment.single (first := 4701) (initial := 2127) row_count_4702 (by decide)
private theorem atomic_4702_agreement : RowAgreement 4701 4702 := RowAgreement.single (first := 4701) modelrow_agrees_4702
private def atomic_4703 : CountSegment 1 10 13120 4702 4703 2128 2129 := CountSegment.single (first := 4702) (initial := 2128) row_count_4703 (by decide)
private theorem atomic_4703_agreement : RowAgreement 4702 4703 := RowAgreement.single (first := 4702) modelrow_agrees_4703
private def atomic_4704 : CountSegment 1 10 13120 4703 4704 2129 2129 := CountSegment.single (first := 4703) (initial := 2129) row_count_4704 (by decide)
private theorem atomic_4704_agreement : RowAgreement 4703 4704 := RowAgreement.single (first := 4703) modelrow_agrees_4704
private def atomic_4705 : CountSegment 1 10 13120 4704 4705 2129 2129 := CountSegment.single (first := 4704) (initial := 2129) row_count_4705 (by decide)
private theorem atomic_4705_agreement : RowAgreement 4704 4705 := RowAgreement.single (first := 4704) modelrow_agrees_4705
private def atomic_4706 : CountSegment 1 10 13120 4705 4706 2129 2129 := CountSegment.single (first := 4705) (initial := 2129) row_count_4706 (by decide)
private theorem atomic_4706_agreement : RowAgreement 4705 4706 := RowAgreement.single (first := 4705) modelrow_agrees_4706
private def atomic_4707 : CountSegment 1 10 13120 4706 4707 2129 2129 := CountSegment.single (first := 4706) (initial := 2129) row_count_4707 (by decide)
private theorem atomic_4707_agreement : RowAgreement 4706 4707 := RowAgreement.single (first := 4706) modelrow_agrees_4707
private def atomic_4708 : CountSegment 1 10 13120 4707 4708 2129 2129 := CountSegment.single (first := 4707) (initial := 2129) row_count_4708 (by decide)
private theorem atomic_4708_agreement : RowAgreement 4707 4708 := RowAgreement.single (first := 4707) modelrow_agrees_4708
private def atomic_4709 : CountSegment 1 10 13120 4708 4709 2129 2129 := CountSegment.single (first := 4708) (initial := 2129) row_count_4709 (by decide)
private theorem atomic_4709_agreement : RowAgreement 4708 4709 := RowAgreement.single (first := 4708) modelrow_agrees_4709
private def atomic_4710 : CountSegment 1 10 13120 4709 4710 2129 2129 := CountSegment.single (first := 4709) (initial := 2129) row_count_4710 (by decide)
private theorem atomic_4710_agreement : RowAgreement 4709 4710 := RowAgreement.single (first := 4709) modelrow_agrees_4710
private def atomic_4711 : CountSegment 1 10 13120 4710 4711 2129 2129 := CountSegment.single (first := 4710) (initial := 2129) row_count_4711 (by decide)
private theorem atomic_4711_agreement : RowAgreement 4710 4711 := RowAgreement.single (first := 4710) modelrow_agrees_4711
private def atomic_4712 : CountSegment 1 10 13120 4711 4712 2129 2130 := CountSegment.single (first := 4711) (initial := 2129) row_count_4712 (by decide)
private theorem atomic_4712_agreement : RowAgreement 4711 4712 := RowAgreement.single (first := 4711) modelrow_agrees_4712
private def atomic_4713 : CountSegment 1 10 13120 4712 4713 2130 2131 := CountSegment.single (first := 4712) (initial := 2130) row_count_4713 (by decide)
private theorem atomic_4713_agreement : RowAgreement 4712 4713 := RowAgreement.single (first := 4712) modelrow_agrees_4713
private def atomic_4714 : CountSegment 1 10 13120 4713 4714 2131 2131 := CountSegment.single (first := 4713) (initial := 2131) row_count_4714 (by decide)
private theorem atomic_4714_agreement : RowAgreement 4713 4714 := RowAgreement.single (first := 4713) modelrow_agrees_4714
private def atomic_4715 : CountSegment 1 10 13120 4714 4715 2131 2131 := CountSegment.single (first := 4714) (initial := 2131) row_count_4715 (by decide)
private theorem atomic_4715_agreement : RowAgreement 4714 4715 := RowAgreement.single (first := 4714) modelrow_agrees_4715
private def atomic_4716 : CountSegment 1 10 13120 4715 4716 2131 2131 := CountSegment.single (first := 4715) (initial := 2131) row_count_4716 (by decide)
private theorem atomic_4716_agreement : RowAgreement 4715 4716 := RowAgreement.single (first := 4715) modelrow_agrees_4716
private def atomic_4717 : CountSegment 1 10 13120 4716 4717 2131 2132 := CountSegment.single (first := 4716) (initial := 2131) row_count_4717 (by decide)
private theorem atomic_4717_agreement : RowAgreement 4716 4717 := RowAgreement.single (first := 4716) modelrow_agrees_4717
private def atomic_4718 : CountSegment 1 10 13120 4717 4718 2132 2133 := CountSegment.single (first := 4717) (initial := 2132) row_count_4718 (by decide)
private theorem atomic_4718_agreement : RowAgreement 4717 4718 := RowAgreement.single (first := 4717) modelrow_agrees_4718
private def atomic_4719 : CountSegment 1 10 13120 4718 4719 2133 2133 := CountSegment.single (first := 4718) (initial := 2133) row_count_4719 (by decide)
private theorem atomic_4719_agreement : RowAgreement 4718 4719 := RowAgreement.single (first := 4718) modelrow_agrees_4719
private def atomic_4720 : CountSegment 1 10 13120 4719 4720 2133 2133 := CountSegment.single (first := 4719) (initial := 2133) row_count_4720 (by decide)
private theorem atomic_4720_agreement : RowAgreement 4719 4720 := RowAgreement.single (first := 4719) modelrow_agrees_4720
private def atomic_4721 : CountSegment 1 10 13120 4720 4721 2133 2133 := CountSegment.single (first := 4720) (initial := 2133) row_count_4721 (by decide)
private theorem atomic_4721_agreement : RowAgreement 4720 4721 := RowAgreement.single (first := 4720) modelrow_agrees_4721
private def atomic_4722 : CountSegment 1 10 13120 4721 4722 2133 2134 := CountSegment.single (first := 4721) (initial := 2133) row_count_4722 (by decide)
private theorem atomic_4722_agreement : RowAgreement 4721 4722 := RowAgreement.single (first := 4721) modelrow_agrees_4722
private def atomic_4723 : CountSegment 1 10 13120 4722 4723 2134 2134 := CountSegment.single (first := 4722) (initial := 2134) row_count_4723 (by decide)
private theorem atomic_4723_agreement : RowAgreement 4722 4723 := RowAgreement.single (first := 4722) modelrow_agrees_4723
private def atomic_4724 : CountSegment 1 10 13120 4723 4724 2134 2134 := CountSegment.single (first := 4723) (initial := 2134) row_count_4724 (by decide)
private theorem atomic_4724_agreement : RowAgreement 4723 4724 := RowAgreement.single (first := 4723) modelrow_agrees_4724
private def atomic_4725 : CountSegment 1 10 13120 4724 4725 2134 2134 := CountSegment.single (first := 4724) (initial := 2134) row_count_4725 (by decide)
private theorem atomic_4725_agreement : RowAgreement 4724 4725 := RowAgreement.single (first := 4724) modelrow_agrees_4725
private def atomic_4726 : CountSegment 1 10 13120 4725 4726 2134 2134 := CountSegment.single (first := 4725) (initial := 2134) row_count_4726 (by decide)
private theorem atomic_4726_agreement : RowAgreement 4725 4726 := RowAgreement.single (first := 4725) modelrow_agrees_4726
private def atomic_4727 : CountSegment 1 10 13120 4726 4727 2134 2134 := CountSegment.single (first := 4726) (initial := 2134) row_count_4727 (by decide)
private theorem atomic_4727_agreement : RowAgreement 4726 4727 := RowAgreement.single (first := 4726) modelrow_agrees_4727
private def atomic_4728 : CountSegment 1 10 13120 4727 4728 2134 2134 := CountSegment.single (first := 4727) (initial := 2134) row_count_4728 (by decide)
private theorem atomic_4728_agreement : RowAgreement 4727 4728 := RowAgreement.single (first := 4727) modelrow_agrees_4728
private def atomic_4729 : CountSegment 1 10 13120 4728 4729 2134 2134 := CountSegment.single (first := 4728) (initial := 2134) row_count_4729 (by decide)
private theorem atomic_4729_agreement : RowAgreement 4728 4729 := RowAgreement.single (first := 4728) modelrow_agrees_4729
private def atomic_4730 : CountSegment 1 10 13120 4729 4730 2134 2135 := CountSegment.single (first := 4729) (initial := 2134) row_count_4730 (by decide)
private theorem atomic_4730_agreement : RowAgreement 4729 4730 := RowAgreement.single (first := 4729) modelrow_agrees_4730
private def atomic_4731 : CountSegment 1 10 13120 4730 4731 2135 2136 := CountSegment.single (first := 4730) (initial := 2135) row_count_4731 (by decide)
private theorem atomic_4731_agreement : RowAgreement 4730 4731 := RowAgreement.single (first := 4730) modelrow_agrees_4731
private def atomic_4732 : CountSegment 1 10 13120 4731 4732 2136 2136 := CountSegment.single (first := 4731) (initial := 2136) row_count_4732 (by decide)
private theorem atomic_4732_agreement : RowAgreement 4731 4732 := RowAgreement.single (first := 4731) modelrow_agrees_4732
private def atomic_4733 : CountSegment 1 10 13120 4732 4733 2136 2137 := CountSegment.single (first := 4732) (initial := 2136) row_count_4733 (by decide)
private theorem atomic_4733_agreement : RowAgreement 4732 4733 := RowAgreement.single (first := 4732) modelrow_agrees_4733
private def atomic_4734 : CountSegment 1 10 13120 4733 4734 2137 2137 := CountSegment.single (first := 4733) (initial := 2137) row_count_4734 (by decide)
private theorem atomic_4734_agreement : RowAgreement 4733 4734 := RowAgreement.single (first := 4733) modelrow_agrees_4734
private def atomic_4735 : CountSegment 1 10 13120 4734 4735 2137 2137 := CountSegment.single (first := 4734) (initial := 2137) row_count_4735 (by decide)
private theorem atomic_4735_agreement : RowAgreement 4734 4735 := RowAgreement.single (first := 4734) modelrow_agrees_4735
private def atomic_4736 : CountSegment 1 10 13120 4735 4736 2137 2137 := CountSegment.single (first := 4735) (initial := 2137) row_count_4736 (by decide)
private theorem atomic_4736_agreement : RowAgreement 4735 4736 := RowAgreement.single (first := 4735) modelrow_agrees_4736
private def within_036_000 : CountSegment 1 10 13120 4608 4610 2090 2091 := CountSegment.join atomic_4609 atomic_4610
private theorem within_036_000_agreement : RowAgreement 4608 4610 := RowAgreement.join atomic_4609_agreement atomic_4610_agreement
private def within_036_001 : CountSegment 1 10 13120 4610 4612 2091 2091 := CountSegment.join atomic_4611 atomic_4612
private theorem within_036_001_agreement : RowAgreement 4610 4612 := RowAgreement.join atomic_4611_agreement atomic_4612_agreement
private def within_036_002 : CountSegment 1 10 13120 4608 4612 2090 2091 := CountSegment.join within_036_000 within_036_001
private theorem within_036_002_agreement : RowAgreement 4608 4612 := RowAgreement.join within_036_000_agreement within_036_001_agreement
private def within_036_003 : CountSegment 1 10 13120 4612 4614 2091 2091 := CountSegment.join atomic_4613 atomic_4614
private theorem within_036_003_agreement : RowAgreement 4612 4614 := RowAgreement.join atomic_4613_agreement atomic_4614_agreement
private def within_036_004 : CountSegment 1 10 13120 4614 4616 2091 2092 := CountSegment.join atomic_4615 atomic_4616
private theorem within_036_004_agreement : RowAgreement 4614 4616 := RowAgreement.join atomic_4615_agreement atomic_4616_agreement
private def within_036_005 : CountSegment 1 10 13120 4612 4616 2091 2092 := CountSegment.join within_036_003 within_036_004
private theorem within_036_005_agreement : RowAgreement 4612 4616 := RowAgreement.join within_036_003_agreement within_036_004_agreement
private def within_036_006 : CountSegment 1 10 13120 4608 4616 2090 2092 := CountSegment.join within_036_002 within_036_005
private theorem within_036_006_agreement : RowAgreement 4608 4616 := RowAgreement.join within_036_002_agreement within_036_005_agreement
private def within_036_007 : CountSegment 1 10 13120 4616 4618 2092 2092 := CountSegment.join atomic_4617 atomic_4618
private theorem within_036_007_agreement : RowAgreement 4616 4618 := RowAgreement.join atomic_4617_agreement atomic_4618_agreement
private def within_036_008 : CountSegment 1 10 13120 4618 4620 2092 2094 := CountSegment.join atomic_4619 atomic_4620
private theorem within_036_008_agreement : RowAgreement 4618 4620 := RowAgreement.join atomic_4619_agreement atomic_4620_agreement
private def within_036_009 : CountSegment 1 10 13120 4616 4620 2092 2094 := CountSegment.join within_036_007 within_036_008
private theorem within_036_009_agreement : RowAgreement 4616 4620 := RowAgreement.join within_036_007_agreement within_036_008_agreement
private def within_036_010 : CountSegment 1 10 13120 4620 4622 2094 2094 := CountSegment.join atomic_4621 atomic_4622
private theorem within_036_010_agreement : RowAgreement 4620 4622 := RowAgreement.join atomic_4621_agreement atomic_4622_agreement
private def within_036_011 : CountSegment 1 10 13120 4622 4624 2094 2096 := CountSegment.join atomic_4623 atomic_4624
private theorem within_036_011_agreement : RowAgreement 4622 4624 := RowAgreement.join atomic_4623_agreement atomic_4624_agreement
private def within_036_012 : CountSegment 1 10 13120 4620 4624 2094 2096 := CountSegment.join within_036_010 within_036_011
private theorem within_036_012_agreement : RowAgreement 4620 4624 := RowAgreement.join within_036_010_agreement within_036_011_agreement
private def within_036_013 : CountSegment 1 10 13120 4616 4624 2092 2096 := CountSegment.join within_036_009 within_036_012
private theorem within_036_013_agreement : RowAgreement 4616 4624 := RowAgreement.join within_036_009_agreement within_036_012_agreement
private def within_036_014 : CountSegment 1 10 13120 4608 4624 2090 2096 := CountSegment.join within_036_006 within_036_013
private theorem within_036_014_agreement : RowAgreement 4608 4624 := RowAgreement.join within_036_006_agreement within_036_013_agreement
private def within_036_015 : CountSegment 1 10 13120 4624 4626 2096 2096 := CountSegment.join atomic_4625 atomic_4626
private theorem within_036_015_agreement : RowAgreement 4624 4626 := RowAgreement.join atomic_4625_agreement atomic_4626_agreement
private def within_036_016 : CountSegment 1 10 13120 4626 4628 2096 2097 := CountSegment.join atomic_4627 atomic_4628
private theorem within_036_016_agreement : RowAgreement 4626 4628 := RowAgreement.join atomic_4627_agreement atomic_4628_agreement
private def within_036_017 : CountSegment 1 10 13120 4624 4628 2096 2097 := CountSegment.join within_036_015 within_036_016
private theorem within_036_017_agreement : RowAgreement 4624 4628 := RowAgreement.join within_036_015_agreement within_036_016_agreement
private def within_036_018 : CountSegment 1 10 13120 4628 4630 2097 2098 := CountSegment.join atomic_4629 atomic_4630
private theorem within_036_018_agreement : RowAgreement 4628 4630 := RowAgreement.join atomic_4629_agreement atomic_4630_agreement
private def within_036_019 : CountSegment 1 10 13120 4630 4632 2098 2099 := CountSegment.join atomic_4631 atomic_4632
private theorem within_036_019_agreement : RowAgreement 4630 4632 := RowAgreement.join atomic_4631_agreement atomic_4632_agreement
private def within_036_020 : CountSegment 1 10 13120 4628 4632 2097 2099 := CountSegment.join within_036_018 within_036_019
private theorem within_036_020_agreement : RowAgreement 4628 4632 := RowAgreement.join within_036_018_agreement within_036_019_agreement
private def within_036_021 : CountSegment 1 10 13120 4624 4632 2096 2099 := CountSegment.join within_036_017 within_036_020
private theorem within_036_021_agreement : RowAgreement 4624 4632 := RowAgreement.join within_036_017_agreement within_036_020_agreement
private def within_036_022 : CountSegment 1 10 13120 4632 4634 2099 2100 := CountSegment.join atomic_4633 atomic_4634
private theorem within_036_022_agreement : RowAgreement 4632 4634 := RowAgreement.join atomic_4633_agreement atomic_4634_agreement
private def within_036_023 : CountSegment 1 10 13120 4634 4636 2100 2101 := CountSegment.join atomic_4635 atomic_4636
private theorem within_036_023_agreement : RowAgreement 4634 4636 := RowAgreement.join atomic_4635_agreement atomic_4636_agreement
private def within_036_024 : CountSegment 1 10 13120 4632 4636 2099 2101 := CountSegment.join within_036_022 within_036_023
private theorem within_036_024_agreement : RowAgreement 4632 4636 := RowAgreement.join within_036_022_agreement within_036_023_agreement
private def within_036_025 : CountSegment 1 10 13120 4636 4638 2101 2102 := CountSegment.join atomic_4637 atomic_4638
private theorem within_036_025_agreement : RowAgreement 4636 4638 := RowAgreement.join atomic_4637_agreement atomic_4638_agreement
private def within_036_026 : CountSegment 1 10 13120 4638 4640 2102 2103 := CountSegment.join atomic_4639 atomic_4640
private theorem within_036_026_agreement : RowAgreement 4638 4640 := RowAgreement.join atomic_4639_agreement atomic_4640_agreement
private def within_036_027 : CountSegment 1 10 13120 4636 4640 2101 2103 := CountSegment.join within_036_025 within_036_026
private theorem within_036_027_agreement : RowAgreement 4636 4640 := RowAgreement.join within_036_025_agreement within_036_026_agreement
private def within_036_028 : CountSegment 1 10 13120 4632 4640 2099 2103 := CountSegment.join within_036_024 within_036_027
private theorem within_036_028_agreement : RowAgreement 4632 4640 := RowAgreement.join within_036_024_agreement within_036_027_agreement
private def within_036_029 : CountSegment 1 10 13120 4624 4640 2096 2103 := CountSegment.join within_036_021 within_036_028
private theorem within_036_029_agreement : RowAgreement 4624 4640 := RowAgreement.join within_036_021_agreement within_036_028_agreement
private def within_036_030 : CountSegment 1 10 13120 4608 4640 2090 2103 := CountSegment.join within_036_014 within_036_029
private theorem within_036_030_agreement : RowAgreement 4608 4640 := RowAgreement.join within_036_014_agreement within_036_029_agreement
private def within_036_031 : CountSegment 1 10 13120 4640 4642 2103 2103 := CountSegment.join atomic_4641 atomic_4642
private theorem within_036_031_agreement : RowAgreement 4640 4642 := RowAgreement.join atomic_4641_agreement atomic_4642_agreement
private def within_036_032 : CountSegment 1 10 13120 4642 4644 2103 2104 := CountSegment.join atomic_4643 atomic_4644
private theorem within_036_032_agreement : RowAgreement 4642 4644 := RowAgreement.join atomic_4643_agreement atomic_4644_agreement
private def within_036_033 : CountSegment 1 10 13120 4640 4644 2103 2104 := CountSegment.join within_036_031 within_036_032
private theorem within_036_033_agreement : RowAgreement 4640 4644 := RowAgreement.join within_036_031_agreement within_036_032_agreement
private def within_036_034 : CountSegment 1 10 13120 4644 4646 2104 2104 := CountSegment.join atomic_4645 atomic_4646
private theorem within_036_034_agreement : RowAgreement 4644 4646 := RowAgreement.join atomic_4645_agreement atomic_4646_agreement
private def within_036_035 : CountSegment 1 10 13120 4646 4648 2104 2105 := CountSegment.join atomic_4647 atomic_4648
private theorem within_036_035_agreement : RowAgreement 4646 4648 := RowAgreement.join atomic_4647_agreement atomic_4648_agreement
private def within_036_036 : CountSegment 1 10 13120 4644 4648 2104 2105 := CountSegment.join within_036_034 within_036_035
private theorem within_036_036_agreement : RowAgreement 4644 4648 := RowAgreement.join within_036_034_agreement within_036_035_agreement
private def within_036_037 : CountSegment 1 10 13120 4640 4648 2103 2105 := CountSegment.join within_036_033 within_036_036
private theorem within_036_037_agreement : RowAgreement 4640 4648 := RowAgreement.join within_036_033_agreement within_036_036_agreement
private def within_036_038 : CountSegment 1 10 13120 4648 4650 2105 2105 := CountSegment.join atomic_4649 atomic_4650
private theorem within_036_038_agreement : RowAgreement 4648 4650 := RowAgreement.join atomic_4649_agreement atomic_4650_agreement
private def within_036_039 : CountSegment 1 10 13120 4650 4652 2105 2105 := CountSegment.join atomic_4651 atomic_4652
private theorem within_036_039_agreement : RowAgreement 4650 4652 := RowAgreement.join atomic_4651_agreement atomic_4652_agreement
private def within_036_040 : CountSegment 1 10 13120 4648 4652 2105 2105 := CountSegment.join within_036_038 within_036_039
private theorem within_036_040_agreement : RowAgreement 4648 4652 := RowAgreement.join within_036_038_agreement within_036_039_agreement
private def within_036_041 : CountSegment 1 10 13120 4652 4654 2105 2107 := CountSegment.join atomic_4653 atomic_4654
private theorem within_036_041_agreement : RowAgreement 4652 4654 := RowAgreement.join atomic_4653_agreement atomic_4654_agreement
private def within_036_042 : CountSegment 1 10 13120 4654 4656 2107 2109 := CountSegment.join atomic_4655 atomic_4656
private theorem within_036_042_agreement : RowAgreement 4654 4656 := RowAgreement.join atomic_4655_agreement atomic_4656_agreement
private def within_036_043 : CountSegment 1 10 13120 4652 4656 2105 2109 := CountSegment.join within_036_041 within_036_042
private theorem within_036_043_agreement : RowAgreement 4652 4656 := RowAgreement.join within_036_041_agreement within_036_042_agreement
private def within_036_044 : CountSegment 1 10 13120 4648 4656 2105 2109 := CountSegment.join within_036_040 within_036_043
private theorem within_036_044_agreement : RowAgreement 4648 4656 := RowAgreement.join within_036_040_agreement within_036_043_agreement
private def within_036_045 : CountSegment 1 10 13120 4640 4656 2103 2109 := CountSegment.join within_036_037 within_036_044
private theorem within_036_045_agreement : RowAgreement 4640 4656 := RowAgreement.join within_036_037_agreement within_036_044_agreement
private def within_036_046 : CountSegment 1 10 13120 4656 4658 2109 2110 := CountSegment.join atomic_4657 atomic_4658
private theorem within_036_046_agreement : RowAgreement 4656 4658 := RowAgreement.join atomic_4657_agreement atomic_4658_agreement
private def within_036_047 : CountSegment 1 10 13120 4658 4660 2110 2112 := CountSegment.join atomic_4659 atomic_4660
private theorem within_036_047_agreement : RowAgreement 4658 4660 := RowAgreement.join atomic_4659_agreement atomic_4660_agreement
private def within_036_048 : CountSegment 1 10 13120 4656 4660 2109 2112 := CountSegment.join within_036_046 within_036_047
private theorem within_036_048_agreement : RowAgreement 4656 4660 := RowAgreement.join within_036_046_agreement within_036_047_agreement
private def within_036_049 : CountSegment 1 10 13120 4660 4662 2112 2114 := CountSegment.join atomic_4661 atomic_4662
private theorem within_036_049_agreement : RowAgreement 4660 4662 := RowAgreement.join atomic_4661_agreement atomic_4662_agreement
private def within_036_050 : CountSegment 1 10 13120 4662 4664 2114 2114 := CountSegment.join atomic_4663 atomic_4664
private theorem within_036_050_agreement : RowAgreement 4662 4664 := RowAgreement.join atomic_4663_agreement atomic_4664_agreement
private def within_036_051 : CountSegment 1 10 13120 4660 4664 2112 2114 := CountSegment.join within_036_049 within_036_050
private theorem within_036_051_agreement : RowAgreement 4660 4664 := RowAgreement.join within_036_049_agreement within_036_050_agreement
private def within_036_052 : CountSegment 1 10 13120 4656 4664 2109 2114 := CountSegment.join within_036_048 within_036_051
private theorem within_036_052_agreement : RowAgreement 4656 4664 := RowAgreement.join within_036_048_agreement within_036_051_agreement
private def within_036_053 : CountSegment 1 10 13120 4664 4666 2114 2114 := CountSegment.join atomic_4665 atomic_4666
private theorem within_036_053_agreement : RowAgreement 4664 4666 := RowAgreement.join atomic_4665_agreement atomic_4666_agreement
private def within_036_054 : CountSegment 1 10 13120 4666 4668 2114 2115 := CountSegment.join atomic_4667 atomic_4668
private theorem within_036_054_agreement : RowAgreement 4666 4668 := RowAgreement.join atomic_4667_agreement atomic_4668_agreement
private def within_036_055 : CountSegment 1 10 13120 4664 4668 2114 2115 := CountSegment.join within_036_053 within_036_054
private theorem within_036_055_agreement : RowAgreement 4664 4668 := RowAgreement.join within_036_053_agreement within_036_054_agreement
private def within_036_056 : CountSegment 1 10 13120 4668 4670 2115 2116 := CountSegment.join atomic_4669 atomic_4670
private theorem within_036_056_agreement : RowAgreement 4668 4670 := RowAgreement.join atomic_4669_agreement atomic_4670_agreement
private def within_036_057 : CountSegment 1 10 13120 4670 4672 2116 2117 := CountSegment.join atomic_4671 atomic_4672
private theorem within_036_057_agreement : RowAgreement 4670 4672 := RowAgreement.join atomic_4671_agreement atomic_4672_agreement
private def within_036_058 : CountSegment 1 10 13120 4668 4672 2115 2117 := CountSegment.join within_036_056 within_036_057
private theorem within_036_058_agreement : RowAgreement 4668 4672 := RowAgreement.join within_036_056_agreement within_036_057_agreement
private def within_036_059 : CountSegment 1 10 13120 4664 4672 2114 2117 := CountSegment.join within_036_055 within_036_058
private theorem within_036_059_agreement : RowAgreement 4664 4672 := RowAgreement.join within_036_055_agreement within_036_058_agreement
private def within_036_060 : CountSegment 1 10 13120 4656 4672 2109 2117 := CountSegment.join within_036_052 within_036_059
private theorem within_036_060_agreement : RowAgreement 4656 4672 := RowAgreement.join within_036_052_agreement within_036_059_agreement
private def within_036_061 : CountSegment 1 10 13120 4640 4672 2103 2117 := CountSegment.join within_036_045 within_036_060
private theorem within_036_061_agreement : RowAgreement 4640 4672 := RowAgreement.join within_036_045_agreement within_036_060_agreement
private def within_036_062 : CountSegment 1 10 13120 4608 4672 2090 2117 := CountSegment.join within_036_030 within_036_061
private theorem within_036_062_agreement : RowAgreement 4608 4672 := RowAgreement.join within_036_030_agreement within_036_061_agreement
private def within_036_063 : CountSegment 1 10 13120 4672 4674 2117 2117 := CountSegment.join atomic_4673 atomic_4674
private theorem within_036_063_agreement : RowAgreement 4672 4674 := RowAgreement.join atomic_4673_agreement atomic_4674_agreement
private def within_036_064 : CountSegment 1 10 13120 4674 4676 2117 2118 := CountSegment.join atomic_4675 atomic_4676
private theorem within_036_064_agreement : RowAgreement 4674 4676 := RowAgreement.join atomic_4675_agreement atomic_4676_agreement
private def within_036_065 : CountSegment 1 10 13120 4672 4676 2117 2118 := CountSegment.join within_036_063 within_036_064
private theorem within_036_065_agreement : RowAgreement 4672 4676 := RowAgreement.join within_036_063_agreement within_036_064_agreement
private def within_036_066 : CountSegment 1 10 13120 4676 4678 2118 2119 := CountSegment.join atomic_4677 atomic_4678
private theorem within_036_066_agreement : RowAgreement 4676 4678 := RowAgreement.join atomic_4677_agreement atomic_4678_agreement
private def within_036_067 : CountSegment 1 10 13120 4678 4680 2119 2121 := CountSegment.join atomic_4679 atomic_4680
private theorem within_036_067_agreement : RowAgreement 4678 4680 := RowAgreement.join atomic_4679_agreement atomic_4680_agreement
private def within_036_068 : CountSegment 1 10 13120 4676 4680 2118 2121 := CountSegment.join within_036_066 within_036_067
private theorem within_036_068_agreement : RowAgreement 4676 4680 := RowAgreement.join within_036_066_agreement within_036_067_agreement
private def within_036_069 : CountSegment 1 10 13120 4672 4680 2117 2121 := CountSegment.join within_036_065 within_036_068
private theorem within_036_069_agreement : RowAgreement 4672 4680 := RowAgreement.join within_036_065_agreement within_036_068_agreement
private def within_036_070 : CountSegment 1 10 13120 4680 4682 2121 2122 := CountSegment.join atomic_4681 atomic_4682
private theorem within_036_070_agreement : RowAgreement 4680 4682 := RowAgreement.join atomic_4681_agreement atomic_4682_agreement
private def within_036_071 : CountSegment 1 10 13120 4682 4684 2122 2123 := CountSegment.join atomic_4683 atomic_4684
private theorem within_036_071_agreement : RowAgreement 4682 4684 := RowAgreement.join atomic_4683_agreement atomic_4684_agreement
private def within_036_072 : CountSegment 1 10 13120 4680 4684 2121 2123 := CountSegment.join within_036_070 within_036_071
private theorem within_036_072_agreement : RowAgreement 4680 4684 := RowAgreement.join within_036_070_agreement within_036_071_agreement
private def within_036_073 : CountSegment 1 10 13120 4684 4686 2123 2123 := CountSegment.join atomic_4685 atomic_4686
private theorem within_036_073_agreement : RowAgreement 4684 4686 := RowAgreement.join atomic_4685_agreement atomic_4686_agreement
private def within_036_074 : CountSegment 1 10 13120 4686 4688 2123 2124 := CountSegment.join atomic_4687 atomic_4688
private theorem within_036_074_agreement : RowAgreement 4686 4688 := RowAgreement.join atomic_4687_agreement atomic_4688_agreement
private def within_036_075 : CountSegment 1 10 13120 4684 4688 2123 2124 := CountSegment.join within_036_073 within_036_074
private theorem within_036_075_agreement : RowAgreement 4684 4688 := RowAgreement.join within_036_073_agreement within_036_074_agreement
private def within_036_076 : CountSegment 1 10 13120 4680 4688 2121 2124 := CountSegment.join within_036_072 within_036_075
private theorem within_036_076_agreement : RowAgreement 4680 4688 := RowAgreement.join within_036_072_agreement within_036_075_agreement
private def within_036_077 : CountSegment 1 10 13120 4672 4688 2117 2124 := CountSegment.join within_036_069 within_036_076
private theorem within_036_077_agreement : RowAgreement 4672 4688 := RowAgreement.join within_036_069_agreement within_036_076_agreement
private def within_036_078 : CountSegment 1 10 13120 4688 4690 2124 2124 := CountSegment.join atomic_4689 atomic_4690
private theorem within_036_078_agreement : RowAgreement 4688 4690 := RowAgreement.join atomic_4689_agreement atomic_4690_agreement
private def within_036_079 : CountSegment 1 10 13120 4690 4692 2124 2124 := CountSegment.join atomic_4691 atomic_4692
private theorem within_036_079_agreement : RowAgreement 4690 4692 := RowAgreement.join atomic_4691_agreement atomic_4692_agreement
private def within_036_080 : CountSegment 1 10 13120 4688 4692 2124 2124 := CountSegment.join within_036_078 within_036_079
private theorem within_036_080_agreement : RowAgreement 4688 4692 := RowAgreement.join within_036_078_agreement within_036_079_agreement
private def within_036_081 : CountSegment 1 10 13120 4692 4694 2124 2125 := CountSegment.join atomic_4693 atomic_4694
private theorem within_036_081_agreement : RowAgreement 4692 4694 := RowAgreement.join atomic_4693_agreement atomic_4694_agreement
private def within_036_082 : CountSegment 1 10 13120 4694 4696 2125 2125 := CountSegment.join atomic_4695 atomic_4696
private theorem within_036_082_agreement : RowAgreement 4694 4696 := RowAgreement.join atomic_4695_agreement atomic_4696_agreement
private def within_036_083 : CountSegment 1 10 13120 4692 4696 2124 2125 := CountSegment.join within_036_081 within_036_082
private theorem within_036_083_agreement : RowAgreement 4692 4696 := RowAgreement.join within_036_081_agreement within_036_082_agreement
private def within_036_084 : CountSegment 1 10 13120 4688 4696 2124 2125 := CountSegment.join within_036_080 within_036_083
private theorem within_036_084_agreement : RowAgreement 4688 4696 := RowAgreement.join within_036_080_agreement within_036_083_agreement
private def within_036_085 : CountSegment 1 10 13120 4696 4698 2125 2125 := CountSegment.join atomic_4697 atomic_4698
private theorem within_036_085_agreement : RowAgreement 4696 4698 := RowAgreement.join atomic_4697_agreement atomic_4698_agreement
private def within_036_086 : CountSegment 1 10 13120 4698 4700 2125 2126 := CountSegment.join atomic_4699 atomic_4700
private theorem within_036_086_agreement : RowAgreement 4698 4700 := RowAgreement.join atomic_4699_agreement atomic_4700_agreement
private def within_036_087 : CountSegment 1 10 13120 4696 4700 2125 2126 := CountSegment.join within_036_085 within_036_086
private theorem within_036_087_agreement : RowAgreement 4696 4700 := RowAgreement.join within_036_085_agreement within_036_086_agreement
private def within_036_088 : CountSegment 1 10 13120 4700 4702 2126 2128 := CountSegment.join atomic_4701 atomic_4702
private theorem within_036_088_agreement : RowAgreement 4700 4702 := RowAgreement.join atomic_4701_agreement atomic_4702_agreement
private def within_036_089 : CountSegment 1 10 13120 4702 4704 2128 2129 := CountSegment.join atomic_4703 atomic_4704
private theorem within_036_089_agreement : RowAgreement 4702 4704 := RowAgreement.join atomic_4703_agreement atomic_4704_agreement
private def within_036_090 : CountSegment 1 10 13120 4700 4704 2126 2129 := CountSegment.join within_036_088 within_036_089
private theorem within_036_090_agreement : RowAgreement 4700 4704 := RowAgreement.join within_036_088_agreement within_036_089_agreement
private def within_036_091 : CountSegment 1 10 13120 4696 4704 2125 2129 := CountSegment.join within_036_087 within_036_090
private theorem within_036_091_agreement : RowAgreement 4696 4704 := RowAgreement.join within_036_087_agreement within_036_090_agreement
private def within_036_092 : CountSegment 1 10 13120 4688 4704 2124 2129 := CountSegment.join within_036_084 within_036_091
private theorem within_036_092_agreement : RowAgreement 4688 4704 := RowAgreement.join within_036_084_agreement within_036_091_agreement
private def within_036_093 : CountSegment 1 10 13120 4672 4704 2117 2129 := CountSegment.join within_036_077 within_036_092
private theorem within_036_093_agreement : RowAgreement 4672 4704 := RowAgreement.join within_036_077_agreement within_036_092_agreement
private def within_036_094 : CountSegment 1 10 13120 4704 4706 2129 2129 := CountSegment.join atomic_4705 atomic_4706
private theorem within_036_094_agreement : RowAgreement 4704 4706 := RowAgreement.join atomic_4705_agreement atomic_4706_agreement
private def within_036_095 : CountSegment 1 10 13120 4706 4708 2129 2129 := CountSegment.join atomic_4707 atomic_4708
private theorem within_036_095_agreement : RowAgreement 4706 4708 := RowAgreement.join atomic_4707_agreement atomic_4708_agreement
private def within_036_096 : CountSegment 1 10 13120 4704 4708 2129 2129 := CountSegment.join within_036_094 within_036_095
private theorem within_036_096_agreement : RowAgreement 4704 4708 := RowAgreement.join within_036_094_agreement within_036_095_agreement
private def within_036_097 : CountSegment 1 10 13120 4708 4710 2129 2129 := CountSegment.join atomic_4709 atomic_4710
private theorem within_036_097_agreement : RowAgreement 4708 4710 := RowAgreement.join atomic_4709_agreement atomic_4710_agreement
private def within_036_098 : CountSegment 1 10 13120 4710 4712 2129 2130 := CountSegment.join atomic_4711 atomic_4712
private theorem within_036_098_agreement : RowAgreement 4710 4712 := RowAgreement.join atomic_4711_agreement atomic_4712_agreement
private def within_036_099 : CountSegment 1 10 13120 4708 4712 2129 2130 := CountSegment.join within_036_097 within_036_098
private theorem within_036_099_agreement : RowAgreement 4708 4712 := RowAgreement.join within_036_097_agreement within_036_098_agreement
private def within_036_100 : CountSegment 1 10 13120 4704 4712 2129 2130 := CountSegment.join within_036_096 within_036_099
private theorem within_036_100_agreement : RowAgreement 4704 4712 := RowAgreement.join within_036_096_agreement within_036_099_agreement
private def within_036_101 : CountSegment 1 10 13120 4712 4714 2130 2131 := CountSegment.join atomic_4713 atomic_4714
private theorem within_036_101_agreement : RowAgreement 4712 4714 := RowAgreement.join atomic_4713_agreement atomic_4714_agreement
private def within_036_102 : CountSegment 1 10 13120 4714 4716 2131 2131 := CountSegment.join atomic_4715 atomic_4716
private theorem within_036_102_agreement : RowAgreement 4714 4716 := RowAgreement.join atomic_4715_agreement atomic_4716_agreement
private def within_036_103 : CountSegment 1 10 13120 4712 4716 2130 2131 := CountSegment.join within_036_101 within_036_102
private theorem within_036_103_agreement : RowAgreement 4712 4716 := RowAgreement.join within_036_101_agreement within_036_102_agreement
private def within_036_104 : CountSegment 1 10 13120 4716 4718 2131 2133 := CountSegment.join atomic_4717 atomic_4718
private theorem within_036_104_agreement : RowAgreement 4716 4718 := RowAgreement.join atomic_4717_agreement atomic_4718_agreement
private def within_036_105 : CountSegment 1 10 13120 4718 4720 2133 2133 := CountSegment.join atomic_4719 atomic_4720
private theorem within_036_105_agreement : RowAgreement 4718 4720 := RowAgreement.join atomic_4719_agreement atomic_4720_agreement
private def within_036_106 : CountSegment 1 10 13120 4716 4720 2131 2133 := CountSegment.join within_036_104 within_036_105
private theorem within_036_106_agreement : RowAgreement 4716 4720 := RowAgreement.join within_036_104_agreement within_036_105_agreement
private def within_036_107 : CountSegment 1 10 13120 4712 4720 2130 2133 := CountSegment.join within_036_103 within_036_106
private theorem within_036_107_agreement : RowAgreement 4712 4720 := RowAgreement.join within_036_103_agreement within_036_106_agreement
private def within_036_108 : CountSegment 1 10 13120 4704 4720 2129 2133 := CountSegment.join within_036_100 within_036_107
private theorem within_036_108_agreement : RowAgreement 4704 4720 := RowAgreement.join within_036_100_agreement within_036_107_agreement
private def within_036_109 : CountSegment 1 10 13120 4720 4722 2133 2134 := CountSegment.join atomic_4721 atomic_4722
private theorem within_036_109_agreement : RowAgreement 4720 4722 := RowAgreement.join atomic_4721_agreement atomic_4722_agreement
private def within_036_110 : CountSegment 1 10 13120 4722 4724 2134 2134 := CountSegment.join atomic_4723 atomic_4724
private theorem within_036_110_agreement : RowAgreement 4722 4724 := RowAgreement.join atomic_4723_agreement atomic_4724_agreement
private def within_036_111 : CountSegment 1 10 13120 4720 4724 2133 2134 := CountSegment.join within_036_109 within_036_110
private theorem within_036_111_agreement : RowAgreement 4720 4724 := RowAgreement.join within_036_109_agreement within_036_110_agreement
private def within_036_112 : CountSegment 1 10 13120 4724 4726 2134 2134 := CountSegment.join atomic_4725 atomic_4726
private theorem within_036_112_agreement : RowAgreement 4724 4726 := RowAgreement.join atomic_4725_agreement atomic_4726_agreement
private def within_036_113 : CountSegment 1 10 13120 4726 4728 2134 2134 := CountSegment.join atomic_4727 atomic_4728
private theorem within_036_113_agreement : RowAgreement 4726 4728 := RowAgreement.join atomic_4727_agreement atomic_4728_agreement
private def within_036_114 : CountSegment 1 10 13120 4724 4728 2134 2134 := CountSegment.join within_036_112 within_036_113
private theorem within_036_114_agreement : RowAgreement 4724 4728 := RowAgreement.join within_036_112_agreement within_036_113_agreement
private def within_036_115 : CountSegment 1 10 13120 4720 4728 2133 2134 := CountSegment.join within_036_111 within_036_114
private theorem within_036_115_agreement : RowAgreement 4720 4728 := RowAgreement.join within_036_111_agreement within_036_114_agreement
private def within_036_116 : CountSegment 1 10 13120 4728 4730 2134 2135 := CountSegment.join atomic_4729 atomic_4730
private theorem within_036_116_agreement : RowAgreement 4728 4730 := RowAgreement.join atomic_4729_agreement atomic_4730_agreement
private def within_036_117 : CountSegment 1 10 13120 4730 4732 2135 2136 := CountSegment.join atomic_4731 atomic_4732
private theorem within_036_117_agreement : RowAgreement 4730 4732 := RowAgreement.join atomic_4731_agreement atomic_4732_agreement
private def within_036_118 : CountSegment 1 10 13120 4728 4732 2134 2136 := CountSegment.join within_036_116 within_036_117
private theorem within_036_118_agreement : RowAgreement 4728 4732 := RowAgreement.join within_036_116_agreement within_036_117_agreement
private def within_036_119 : CountSegment 1 10 13120 4732 4734 2136 2137 := CountSegment.join atomic_4733 atomic_4734
private theorem within_036_119_agreement : RowAgreement 4732 4734 := RowAgreement.join atomic_4733_agreement atomic_4734_agreement
private def within_036_120 : CountSegment 1 10 13120 4734 4736 2137 2137 := CountSegment.join atomic_4735 atomic_4736
private theorem within_036_120_agreement : RowAgreement 4734 4736 := RowAgreement.join atomic_4735_agreement atomic_4736_agreement
private def within_036_121 : CountSegment 1 10 13120 4732 4736 2136 2137 := CountSegment.join within_036_119 within_036_120
private theorem within_036_121_agreement : RowAgreement 4732 4736 := RowAgreement.join within_036_119_agreement within_036_120_agreement
private def within_036_122 : CountSegment 1 10 13120 4728 4736 2134 2137 := CountSegment.join within_036_118 within_036_121
private theorem within_036_122_agreement : RowAgreement 4728 4736 := RowAgreement.join within_036_118_agreement within_036_121_agreement
private def within_036_123 : CountSegment 1 10 13120 4720 4736 2133 2137 := CountSegment.join within_036_115 within_036_122
private theorem within_036_123_agreement : RowAgreement 4720 4736 := RowAgreement.join within_036_115_agreement within_036_122_agreement
private def within_036_124 : CountSegment 1 10 13120 4704 4736 2129 2137 := CountSegment.join within_036_108 within_036_123
private theorem within_036_124_agreement : RowAgreement 4704 4736 := RowAgreement.join within_036_108_agreement within_036_123_agreement
private def within_036_125 : CountSegment 1 10 13120 4672 4736 2117 2137 := CountSegment.join within_036_093 within_036_124
private theorem within_036_125_agreement : RowAgreement 4672 4736 := RowAgreement.join within_036_093_agreement within_036_124_agreement
private def within_036_126 : CountSegment 1 10 13120 4608 4736 2090 2137 := CountSegment.join within_036_062 within_036_125
private theorem within_036_126_agreement : RowAgreement 4608 4736 := RowAgreement.join within_036_062_agreement within_036_125_agreement
def segment_036 : CountSegment 1 10 13120 4608 4736 2090 2137 := within_036_126
theorem agreement_036 : RowAgreement 4608 4736 := within_036_126_agreement
end ProjectEulerSolutions.P58Numerical
