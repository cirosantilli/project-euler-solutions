import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_7681 : RowCertificate 7681 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7681 1) := by change ¬ Nat.Prime 236006407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33715201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7681 2) := by change ¬ Nat.Prime 235991045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47198209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7681 3) := by change ¬ Nat.Prime 235975683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78658561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7681 : row 7681 = 0 := (row_7681.checked).trans rfl
theorem model_count_7681 : modelRow 7681 = 0 := by decide
theorem modelrow_agrees_7681 : modelRow 7681 = row 7681 := model_count_7681.trans row_count_7681.symm
def row_7682 : RowCertificate 7682 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7682 1) := by change ¬ Nat.Prime 236067861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78689287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7682 2) := by change ¬ Nat.Prime 236052497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 13885441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7682 3) := by change ¬ Nat.Prime 236037133; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12423007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7682 : row 7682 = 0 := (row_7682.checked).trans rfl
theorem model_count_7682 : modelRow 7682 = 0 := by decide
theorem modelrow_agrees_7682 : modelRow 7682 = row 7682 := model_count_7682.trans row_count_7682.symm
def row_7683 : RowCertificate 7683 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7683 1) := by change ¬ Nat.Prime 236129323; exact PrattCertificates.composite_of_product (factor := 2347) (cofactor := 100609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7683 2) := by change ¬ Nat.Prime 236113957; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 5758877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7683 3) := by change ¬ Nat.Prime 236098591; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6381043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7683 : row 7683 = 0 := (row_7683.checked).trans rfl
theorem model_count_7683 : modelRow 7683 = 0 := by decide
theorem modelrow_agrees_7683 : modelRow 7683 = row 7683 := model_count_7683.trans row_count_7683.symm
def row_7684 : RowCertificate 7684 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7684 1) := by change Nat.Prime 236190793; exact ProjectEulerSolutions.P58PrimeNodes.p_236190793.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7684 2) := by change ¬ Nat.Prime 236175425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47235085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7684 3) := by change ¬ Nat.Prime 236160057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78720019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7684 : row 7684 = 1 := (row_7684.checked).trans rfl
theorem model_count_7684 : modelRow 7684 = 1 := by decide
theorem modelrow_agrees_7684 : modelRow 7684 = row 7684 := model_count_7684.trans row_count_7684.symm
def row_7685 : RowCertificate 7685 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7685 1) := by change ¬ Nat.Prime 236252271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78750757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7685 2) := by change Nat.Prime 236236901; exact ProjectEulerSolutions.P58PrimeNodes.p_236236901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7685 3) := by change ¬ Nat.Prime 236221531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33745933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7685 : row 7685 = 1 := (row_7685.checked).trans rfl
theorem model_count_7685 : modelRow 7685 = 1 := by decide
theorem modelrow_agrees_7685 : modelRow 7685 = row 7685 := model_count_7685.trans row_count_7685.symm
def row_7686 : RowCertificate 7686 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7686 1) := by change ¬ Nat.Prime 236313757; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3527071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7686 2) := by change ¬ Nat.Prime 236298385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47259677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7686 3) := by change Nat.Prime 236283013; exact ProjectEulerSolutions.P58PrimeNodes.p_236283013.prime
    exact decide_eq_true prime
theorem row_count_7686 : row 7686 = 1 := (row_7686.checked).trans rfl
theorem model_count_7686 : modelRow 7686 = 1 := by decide
theorem modelrow_agrees_7686 : modelRow 7686 = row 7686 := model_count_7686.trans row_count_7686.symm
def row_7687 : RowCertificate 7687 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7687 1) := by change ¬ Nat.Prime 236375251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33767893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7687 2) := by change ¬ Nat.Prime 236359877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18181529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7687 3) := by change ¬ Nat.Prime 236344503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78781501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7687 : row 7687 = 0 := (row_7687.checked).trans rfl
theorem model_count_7687 : modelRow 7687 = 0 := by decide
theorem modelrow_agrees_7687 : modelRow 7687 = row 7687 := model_count_7687.trans row_count_7687.symm
def row_7688 : RowCertificate 7688 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7688 1) := by change ¬ Nat.Prime 236436753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78812251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7688 2) := by change ¬ Nat.Prime 236421377; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3238649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7688 3) := by change ¬ Nat.Prime 236406001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18185077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7688 : row 7688 = 0 := (row_7688.checked).trans rfl
theorem model_count_7688 : modelRow 7688 = 0 := by decide
theorem modelrow_agrees_7688 : modelRow 7688 = row 7688 := model_count_7688.trans row_count_7688.symm
def row_7689 : RowCertificate 7689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7689 1) := by change ¬ Nat.Prime 236498263; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12447277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7689 2) := by change ¬ Nat.Prime 236482885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47296577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7689 3) := by change ¬ Nat.Prime 236467507; exact PrattCertificates.composite_of_product (factor := 2239) (cofactor := 105613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7689 : row 7689 = 0 := (row_7689.checked).trans rfl
theorem model_count_7689 : modelRow 7689 = 0 := by decide
theorem modelrow_agrees_7689 : modelRow 7689 = row 7689 := model_count_7689.trans row_count_7689.symm
def row_7690 : RowCertificate 7690 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7690 1) := by change ¬ Nat.Prime 236559781; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3530743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7690 2) := by change ¬ Nat.Prime 236544401; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 1200733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7690 3) := by change ¬ Nat.Prime 236529021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78843007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7690 : row 7690 = 0 := (row_7690.checked).trans rfl
theorem model_count_7690 : modelRow 7690 = 0 := by decide
theorem modelrow_agrees_7690 : modelRow 7690 = row 7690 := model_count_7690.trans row_count_7690.symm
def row_7691 : RowCertificate 7691 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7691 1) := by change ¬ Nat.Prime 236621307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78873769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7691 2) := by change ¬ Nat.Prime 236605925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47321185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7691 3) := by change ¬ Nat.Prime 236590543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33798649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7691 : row 7691 = 0 := (row_7691.checked).trans rfl
theorem model_count_7691 : modelRow 7691 = 0 := by decide
theorem modelrow_agrees_7691 : modelRow 7691 = row 7691 := model_count_7691.trans row_count_7691.symm
def row_7692 : RowCertificate 7692 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7692 1) := by change Nat.Prime 236682841; exact ProjectEulerSolutions.P58PrimeNodes.p_236682841.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7692 2) := by change ¬ Nat.Prime 236667457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18205189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7692 3) := by change ¬ Nat.Prime 236652073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33807439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7692 : row 7692 = 1 := (row_7692.checked).trans rfl
theorem model_count_7692 : modelRow 7692 = 1 := by decide
theorem modelrow_agrees_7692 : modelRow 7692 = row 7692 := model_count_7692.trans row_count_7692.symm
def row_7693 : RowCertificate 7693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7693 1) := by change ¬ Nat.Prime 236744383; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1864129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7693 2) := by change ¬ Nat.Prime 236728997; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6398081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7693 3) := by change ¬ Nat.Prime 236713611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78904537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7693 : row 7693 = 0 := (row_7693.checked).trans rfl
theorem model_count_7693 : modelRow 7693 = 0 := by decide
theorem modelrow_agrees_7693 : modelRow 7693 = row 7693 := model_count_7693.trans row_count_7693.symm
def row_7694 : RowCertificate 7694 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7694 1) := by change ¬ Nat.Prime 236805933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78935311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7694 2) := by change ¬ Nat.Prime 236790545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47358109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7694 3) := by change ¬ Nat.Prime 236775157; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5506399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7694 : row 7694 = 0 := (row_7694.checked).trans rfl
theorem model_count_7694 : modelRow 7694 = 0 := by decide
theorem modelrow_agrees_7694 : modelRow 7694 = row 7694 := model_count_7694.trans row_count_7694.symm
def row_7695 : RowCertificate 7695 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7695 1) := by change ¬ Nat.Prime 236867491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33838213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7695 2) := by change ¬ Nat.Prime 236852101; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 191473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7695 3) := by change ¬ Nat.Prime 236836711; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 546967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7695 : row 7695 = 0 := (row_7695.checked).trans rfl
theorem model_count_7695 : modelRow 7695 = 0 := by decide
theorem modelrow_agrees_7695 : modelRow 7695 = row 7695 := model_count_7695.trans row_count_7695.symm
def row_7696 : RowCertificate 7696 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7696 1) := by change Nat.Prime 236929057; exact ProjectEulerSolutions.P58PrimeNodes.p_236929057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7696 2) := by change ¬ Nat.Prime 236913665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47382733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7696 3) := by change ¬ Nat.Prime 236898273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78966091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7696 : row 7696 = 1 := (row_7696.checked).trans rfl
theorem model_count_7696 : modelRow 7696 = 1 := by decide
theorem modelrow_agrees_7696 : modelRow 7696 = row 7696 := model_count_7696.trans row_count_7696.symm
def row_7697 : RowCertificate 7697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7697 1) := by change ¬ Nat.Prime 236990631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 78996877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7697 2) := by change Nat.Prime 236975237; exact ProjectEulerSolutions.P58PrimeNodes.p_236975237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7697 3) := by change ¬ Nat.Prime 236959843; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2300581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7697 : row 7697 = 1 := (row_7697.checked).trans rfl
theorem model_count_7697 : modelRow 7697 = 1 := by decide
theorem modelrow_agrees_7697 : modelRow 7697 = row 7697 := model_count_7697.trans row_count_7697.symm
def row_7698 : RowCertificate 7698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7698 1) := by change ¬ Nat.Prime 237052213; exact PrattCertificates.composite_of_product (factor := 4261) (cofactor := 55633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7698 2) := by change ¬ Nat.Prime 237036817; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 99721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7698 3) := by change ¬ Nat.Prime 237021421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33860203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7698 : row 7698 = 0 := (row_7698.checked).trans rfl
theorem model_count_7698 : modelRow 7698 = 0 := by decide
theorem modelrow_agrees_7698 : modelRow 7698 = row 7698 := model_count_7698.trans row_count_7698.symm
def row_7699 : RowCertificate 7699 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7699 1) := by change ¬ Nat.Prime 237113803; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1510279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7699 2) := by change ¬ Nat.Prime 237098405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47419681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7699 3) := by change ¬ Nat.Prime 237083007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79027669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7699 : row 7699 = 0 := (row_7699.checked).trans rfl
theorem model_count_7699 : modelRow 7699 = 0 := by decide
theorem modelrow_agrees_7699 : modelRow 7699 = row 7699 := model_count_7699.trans row_count_7699.symm
def row_7700 : RowCertificate 7700 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7700 1) := by change ¬ Nat.Prime 237175401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79058467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7700 2) := by change ¬ Nat.Prime 237160001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18243077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7700 3) := by change Nat.Prime 237144601; exact ProjectEulerSolutions.P58PrimeNodes.p_237144601.prime
    exact decide_eq_true prime
theorem row_count_7700 : row 7700 = 1 := (row_7700.checked).trans rfl
theorem model_count_7700 : modelRow 7700 = 1 := by decide
theorem modelrow_agrees_7700 : modelRow 7700 = row 7700 := model_count_7700.trans row_count_7700.symm
def row_7701 : RowCertificate 7701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7701 1) := by change ¬ Nat.Prime 237237007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33891001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7701 2) := by change ¬ Nat.Prime 237221605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47444321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7701 3) := by change ¬ Nat.Prime 237206203; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18246631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7701 : row 7701 = 0 := (row_7701.checked).trans rfl
theorem model_count_7701 : modelRow 7701 = 0 := by decide
theorem modelrow_agrees_7701 : modelRow 7701 = row 7701 := model_count_7701.trans row_count_7701.symm
def row_7702 : RowCertificate 7702 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7702 1) := by change ¬ Nat.Prime 237298621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33899803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7702 2) := by change ¬ Nat.Prime 237283217; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1310957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7702 3) := by change ¬ Nat.Prime 237267813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79089271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7702 : row 7702 = 0 := (row_7702.checked).trans rfl
theorem model_count_7702 : modelRow 7702 = 0 := by decide
theorem modelrow_agrees_7702 : modelRow 7702 = row 7702 := model_count_7702.trans row_count_7702.symm
def row_7703 : RowCertificate 7703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7703 1) := by change ¬ Nat.Prime 237360243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79120081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7703 2) := by change ¬ Nat.Prime 237344837; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 13961461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7703 3) := by change Nat.Prime 237329431; exact ProjectEulerSolutions.P58PrimeNodes.p_237329431.prime
    exact decide_eq_true prime
theorem row_count_7703 : row 7703 = 1 := (row_7703.checked).trans rfl
theorem model_count_7703 : modelRow 7703 = 1 := by decide
theorem modelrow_agrees_7703 : modelRow 7703 = row 7703 := model_count_7703.trans row_count_7703.symm
def row_7704 : RowCertificate 7704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7704 1) := by change ¬ Nat.Prime 237421873; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18263221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7704 2) := by change ¬ Nat.Prime 237406465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47481293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7704 3) := by change Nat.Prime 237391057; exact ProjectEulerSolutions.P58PrimeNodes.p_237391057.prime
    exact decide_eq_true prime
theorem row_count_7704 : row 7704 = 1 := (row_7704.checked).trans rfl
theorem model_count_7704 : modelRow 7704 = 1 := by decide
theorem modelrow_agrees_7704 : modelRow 7704 = row 7704 := model_count_7704.trans row_count_7704.symm
def row_7705 : RowCertificate 7705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7705 1) := by change ¬ Nat.Prime 237483511; exact PrattCertificates.composite_of_product (factor := 5113) (cofactor := 46447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7705 2) := by change ¬ Nat.Prime 237468101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18266777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7705 3) := by change ¬ Nat.Prime 237452691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79150897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7705 : row 7705 = 0 := (row_7705.checked).trans rfl
theorem model_count_7705 : modelRow 7705 = 0 := by decide
theorem modelrow_agrees_7705 : modelRow 7705 = row 7705 := model_count_7705.trans row_count_7705.symm
def row_7706 : RowCertificate 7706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7706 1) := by change ¬ Nat.Prime 237545157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79181719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7706 2) := by change ¬ Nat.Prime 237529745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47505949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7706 3) := by change ¬ Nat.Prime 237514333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33930619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7706 : row 7706 = 0 := (row_7706.checked).trans rfl
theorem model_count_7706 : modelRow 7706 = 0 := by decide
theorem modelrow_agrees_7706 : modelRow 7706 = row 7706 := model_count_7706.trans row_count_7706.symm
def row_7707 : RowCertificate 7707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7707 1) := by change ¬ Nat.Prime 237606811; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18277447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7707 2) := by change ¬ Nat.Prime 237591397; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1513321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7707 3) := by change Nat.Prime 237575983; exact ProjectEulerSolutions.P58PrimeNodes.p_237575983.prime
    exact decide_eq_true prime
theorem row_count_7707 : row 7707 = 1 := (row_7707.checked).trans rfl
theorem model_count_7707 : modelRow 7707 = 1 := by decide
theorem modelrow_agrees_7707 : modelRow 7707 = row 7707 := model_count_7707.trans row_count_7707.symm
def row_7708 : RowCertificate 7708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7708 1) := by change ¬ Nat.Prime 237668473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33952639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7708 2) := by change ¬ Nat.Prime 237653057; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8194933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7708 3) := by change ¬ Nat.Prime 237637641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79212547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7708 : row 7708 = 0 := (row_7708.checked).trans rfl
theorem model_count_7708 : modelRow 7708 = 0 := by decide
theorem modelrow_agrees_7708 : modelRow 7708 = row 7708 := model_count_7708.trans row_count_7708.symm
def row_7709 : RowCertificate 7709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7709 1) := by change ¬ Nat.Prime 237730143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79243381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7709 2) := by change ¬ Nat.Prime 237714725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47542945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7709 3) := by change ¬ Nat.Prime 237699307; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1126537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7709 : row 7709 = 0 := (row_7709.checked).trans rfl
theorem model_count_7709 : modelRow 7709 = 0 := by decide
theorem modelrow_agrees_7709 : modelRow 7709 = row 7709 := model_count_7709.trans row_count_7709.symm
def row_7710 : RowCertificate 7710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7710 1) := by change ¬ Nat.Prime 237791821; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12515359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7710 2) := by change ¬ Nat.Prime 237776401; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 598933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7710 3) := by change ¬ Nat.Prime 237760981; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3897721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7710 : row 7710 = 0 := (row_7710.checked).trans rfl
theorem model_count_7710 : modelRow 7710 = 0 := by decide
theorem modelrow_agrees_7710 : modelRow 7710 = row 7710 := model_count_7710.trans row_count_7710.symm
def row_7711 : RowCertificate 7711 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7711 1) := by change Nat.Prime 237853507; exact ProjectEulerSolutions.P58PrimeNodes.p_237853507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7711 2) := by change ¬ Nat.Prime 237838085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47567617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7711 3) := by change ¬ Nat.Prime 237822663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79274221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7711 : row 7711 = 1 := (row_7711.checked).trans rfl
theorem model_count_7711 : modelRow 7711 = 1 := by decide
theorem modelrow_agrees_7711 : modelRow 7711 = row 7711 := model_count_7711.trans row_count_7711.symm
def row_7712 : RowCertificate 7712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7712 1) := by change ¬ Nat.Prime 237915201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79305067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7712 2) := by change ¬ Nat.Prime 237899777; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 212981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7712 3) := by change ¬ Nat.Prime 237884353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33983479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7712 : row 7712 = 0 := (row_7712.checked).trans rfl
theorem model_count_7712 : modelRow 7712 = 0 := by decide
theorem modelrow_agrees_7712 : modelRow 7712 = row 7712 := model_count_7712.trans row_count_7712.symm
def row_7713 : RowCertificate 7713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7713 1) := by change ¬ Nat.Prime 237976903; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1459981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7713 2) := by change ¬ Nat.Prime 237961477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18304729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7713 3) := by change ¬ Nat.Prime 237946051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 33992293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7713 : row 7713 = 0 := (row_7713.checked).trans rfl
theorem model_count_7713 : modelRow 7713 = 0 := by decide
theorem modelrow_agrees_7713 : modelRow 7713 = row 7713 := model_count_7713.trans row_count_7713.symm
def row_7714 : RowCertificate 7714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7714 1) := by change ¬ Nat.Prime 238038613; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3013147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7714 2) := by change ¬ Nat.Prime 238023185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47604637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7714 3) := by change ¬ Nat.Prime 238007757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79335919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7714 : row 7714 = 0 := (row_7714.checked).trans rfl
theorem model_count_7714 : modelRow 7714 = 0 := by decide
theorem modelrow_agrees_7714 : modelRow 7714 = row 7714 := model_count_7714.trans row_count_7714.symm
def row_7715 : RowCertificate 7715 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7715 1) := by change ¬ Nat.Prime 238100331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79366777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7715 2) := by change ¬ Nat.Prime 238084901; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3261437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7715 3) := by change ¬ Nat.Prime 238069471; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1196329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7715 : row 7715 = 0 := (row_7715.checked).trans rfl
theorem model_count_7715 : modelRow 7715 = 0 := by decide
theorem modelrow_agrees_7715 : modelRow 7715 = row 7715 := model_count_7715.trans row_count_7715.symm
def row_7716 : RowCertificate 7716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7716 1) := by change ¬ Nat.Prime 238162057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34023151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7716 2) := by change ¬ Nat.Prime 238146625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47629325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7716 3) := by change Nat.Prime 238131193; exact ProjectEulerSolutions.P58PrimeNodes.p_238131193.prime
    exact decide_eq_true prime
theorem row_count_7716 : row 7716 = 1 := (row_7716.checked).trans rfl
theorem model_count_7716 : modelRow 7716 = 1 := by decide
theorem modelrow_agrees_7716 : modelRow 7716 = row 7716 := model_count_7716.trans row_count_7716.symm
def row_7717 : RowCertificate 7717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7717 1) := by change ¬ Nat.Prime 238223791; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18324907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7717 2) := by change ¬ Nat.Prime 238208357; exact PrattCertificates.composite_of_product (factor := 7877) (cofactor := 30241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7717 3) := by change ¬ Nat.Prime 238192923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79397641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7717 : row 7717 = 0 := (row_7717.checked).trans rfl
theorem model_count_7717 : modelRow 7717 = 0 := by decide
theorem modelrow_agrees_7717 : modelRow 7717 = row 7717 := model_count_7717.trans row_count_7717.symm
def row_7718 : RowCertificate 7718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7718 1) := by change ¬ Nat.Prime 238285533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79428511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7718 2) := by change ¬ Nat.Prime 238270097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18328469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7718 3) := by change ¬ Nat.Prime 238254661; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12539719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7718 : row 7718 = 0 := (row_7718.checked).trans rfl
theorem model_count_7718 : modelRow 7718 = 0 := by decide
theorem modelrow_agrees_7718 : modelRow 7718 = row 7718 := model_count_7718.trans row_count_7718.symm
def row_7719 : RowCertificate 7719 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7719 1) := by change Nat.Prime 238347283; exact ProjectEulerSolutions.P58PrimeNodes.p_238347283.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7719 2) := by change ¬ Nat.Prime 238331845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47666369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7719 3) := by change ¬ Nat.Prime 238316407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34045201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7719 : row 7719 = 1 := (row_7719.checked).trans rfl
theorem model_count_7719 : modelRow 7719 = 1 := by decide
theorem modelrow_agrees_7719 : modelRow 7719 = row 7719 := model_count_7719.trans row_count_7719.symm
def row_7720 : RowCertificate 7720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7720 1) := by change ¬ Nat.Prime 238409041; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18339157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7720 2) := by change ¬ Nat.Prime 238393601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14023153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7720 3) := by change ¬ Nat.Prime 238378161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79459387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7720 : row 7720 = 0 := (row_7720.checked).trans rfl
theorem model_count_7720 : modelRow 7720 = 0 := by decide
theorem modelrow_agrees_7720 : modelRow 7720 = row 7720 := model_count_7720.trans row_count_7720.symm
def row_7721 : RowCertificate 7721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7721 1) := by change ¬ Nat.Prime 238470807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79490269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7721 2) := by change ¬ Nat.Prime 238455365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47691073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7721 3) := by change ¬ Nat.Prime 238439923; exact PrattCertificates.composite_of_product (factor := 3313) (cofactor := 71971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7721 : row 7721 = 0 := (row_7721.checked).trans rfl
theorem model_count_7721 : modelRow 7721 = 0 := by decide
theorem modelrow_agrees_7721 : modelRow 7721 = row 7721 := model_count_7721.trans row_count_7721.symm
def row_7722 : RowCertificate 7722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7722 1) := by change ¬ Nat.Prime 238532581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34076083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7722 2) := by change ¬ Nat.Prime 238517137; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3910117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7722 3) := by change ¬ Nat.Prime 238501693; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7693603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7722 : row 7722 = 0 := (row_7722.checked).trans rfl
theorem model_count_7722 : modelRow 7722 = 0 := by decide
theorem modelrow_agrees_7722 : modelRow 7722 = row 7722 := model_count_7722.trans row_count_7722.symm
def row_7723 : RowCertificate 7723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7723 1) := by change ¬ Nat.Prime 238594363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34084909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7723 2) := by change ¬ Nat.Prime 238578917; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 4501489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7723 3) := by change ¬ Nat.Prime 238563471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79521157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7723 : row 7723 = 0 := (row_7723.checked).trans rfl
theorem model_count_7723 : modelRow 7723 = 0 := by decide
theorem modelrow_agrees_7723 : modelRow 7723 = row 7723 := model_count_7723.trans row_count_7723.symm
def row_7724 : RowCertificate 7724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7724 1) := by change ¬ Nat.Prime 238656153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79552051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7724 2) := by change ¬ Nat.Prime 238640705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47728141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7724 3) := by change ¬ Nat.Prime 238625257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18355789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7724 : row 7724 = 0 := (row_7724.checked).trans rfl
theorem model_count_7724 : modelRow 7724 = 0 := by decide
theorem modelrow_agrees_7724 : modelRow 7724 = row 7724 := model_count_7724.trans row_count_7724.symm
def row_7725 : RowCertificate 7725 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7725 1) := by change Nat.Prime 238717951; exact ProjectEulerSolutions.P58PrimeNodes.p_238717951.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7725 2) := by change ¬ Nat.Prime 238702501; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1042369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7725 3) := by change ¬ Nat.Prime 238687051; exact PrattCertificates.composite_of_product (factor := 2089) (cofactor := 114259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7725 : row 7725 = 1 := (row_7725.checked).trans rfl
theorem model_count_7725 : modelRow 7725 = 1 := by decide
theorem modelrow_agrees_7725 : modelRow 7725 = row 7725 := model_count_7725.trans row_count_7725.symm
def row_7726 : RowCertificate 7726 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7726 1) := by change Nat.Prime 238779757; exact ProjectEulerSolutions.P58PrimeNodes.p_238779757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7726 2) := by change ¬ Nat.Prime 238764305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47752861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7726 3) := by change ¬ Nat.Prime 238748853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79582951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7726 : row 7726 = 1 := (row_7726.checked).trans rfl
theorem model_count_7726 : modelRow 7726 = 1 := by decide
theorem modelrow_agrees_7726 : modelRow 7726 = row 7726 := model_count_7726.trans row_count_7726.symm
def row_7727 : RowCertificate 7727 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7727 1) := by change ¬ Nat.Prime 238841571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79613857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7727 2) := by change ¬ Nat.Prime 238826117; exact PrattCertificates.composite_of_product (factor := 12037) (cofactor := 19841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7727 3) := by change ¬ Nat.Prime 238810663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34115809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7727 : row 7727 = 0 := (row_7727.checked).trans rfl
theorem model_count_7727 : modelRow 7727 = 0 := by decide
theorem modelrow_agrees_7727 : modelRow 7727 = row 7727 := model_count_7727.trans row_count_7727.symm
def row_7728 : RowCertificate 7728 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7728 1) := by change Nat.Prime 238903393; exact ProjectEulerSolutions.P58PrimeNodes.p_238903393.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7728 2) := by change ¬ Nat.Prime 238887937; exact PrattCertificates.composite_of_product (factor := 5953) (cofactor := 40129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7728 3) := by change ¬ Nat.Prime 238872481; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6456013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7728 : row 7728 = 1 := (row_7728.checked).trans rfl
theorem model_count_7728 : modelRow 7728 = 1 := by decide
theorem modelrow_agrees_7728 : modelRow 7728 = row 7728 := model_count_7728.trans row_count_7728.symm
def row_7729 : RowCertificate 7729 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7729 1) := by change ¬ Nat.Prime 238965223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34137889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7729 2) := by change ¬ Nat.Prime 238949765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47789953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7729 3) := by change ¬ Nat.Prime 238934307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79644769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7729 : row 7729 = 0 := (row_7729.checked).trans rfl
theorem model_count_7729 : modelRow 7729 = 0 := by decide
theorem modelrow_agrees_7729 : modelRow 7729 = row 7729 := model_count_7729.trans row_count_7729.symm
def row_7730 : RowCertificate 7730 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7730 1) := by change ¬ Nat.Prime 239027061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79675687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7730 2) := by change ¬ Nat.Prime 239011601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6459773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7730 3) := by change Nat.Prime 238996141; exact ProjectEulerSolutions.P58PrimeNodes.p_238996141.prime
    exact decide_eq_true prime
theorem row_count_7730 : row 7730 = 1 := (row_7730.checked).trans rfl
theorem model_count_7730 : modelRow 7730 = 1 := by decide
theorem modelrow_agrees_7730 : modelRow 7730 = row 7730 := model_count_7730.trans row_count_7730.symm
def row_7731 : RowCertificate 7731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7731 1) := by change ¬ Nat.Prime 239088907; exact PrattCertificates.composite_of_product (factor := 823) (cofactor := 290509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7731 2) := by change ¬ Nat.Prime 239073445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47814689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7731 3) := by change ¬ Nat.Prime 239057983; exact PrattCertificates.composite_of_product (factor := 2683) (cofactor := 89101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7731 : row 7731 = 0 := (row_7731.checked).trans rfl
theorem model_count_7731 : modelRow 7731 = 0 := by decide
theorem modelrow_agrees_7731 : modelRow 7731 = row 7731 := model_count_7731.trans row_count_7731.symm
def row_7732 : RowCertificate 7732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7732 1) := by change ¬ Nat.Prime 239150761; exact PrattCertificates.composite_of_product (factor := 3181) (cofactor := 75181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7732 2) := by change Nat.Prime 239135297; exact ProjectEulerSolutions.P58PrimeNodes.p_239135297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7732 3) := by change ¬ Nat.Prime 239119833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79706611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7732 : row 7732 = 1 := (row_7732.checked).trans rfl
theorem model_count_7732 : modelRow 7732 = 1 := by decide
theorem modelrow_agrees_7732 : modelRow 7732 = row 7732 := model_count_7732.trans row_count_7732.symm
def row_7733 : RowCertificate 7733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7733 1) := by change ¬ Nat.Prime 239212623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79737541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7733 2) := by change ¬ Nat.Prime 239197157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14070421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7733 3) := by change ¬ Nat.Prime 239181691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34168813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7733 : row 7733 = 0 := (row_7733.checked).trans rfl
theorem model_count_7733 : modelRow 7733 = 0 := by decide
theorem modelrow_agrees_7733 : modelRow 7733 = row 7733 := model_count_7733.trans row_count_7733.symm
def row_7734 : RowCertificate 7734 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7734 1) := by change Nat.Prime 239274493; exact ProjectEulerSolutions.P58PrimeNodes.p_239274493.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7734 2) := by change ¬ Nat.Prime 239259025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47851805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7734 3) := by change ¬ Nat.Prime 239243557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34177651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7734 : row 7734 = 1 := (row_7734.checked).trans rfl
theorem model_count_7734 : modelRow 7734 = 1 := by decide
theorem modelrow_agrees_7734 : modelRow 7734 = row 7734 := model_count_7734.trans row_count_7734.symm
def row_7735 : RowCertificate 7735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7735 1) := by change ¬ Nat.Prime 239336371; exact PrattCertificates.composite_of_product (factor := 14713) (cofactor := 16267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7735 2) := by change Nat.Prime 239320901; exact ProjectEulerSolutions.P58PrimeNodes.p_239320901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7735 3) := by change ¬ Nat.Prime 239305431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79768477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7735 : row 7735 = 1 := (row_7735.checked).trans rfl
theorem model_count_7735 : modelRow 7735 = 1 := by decide
theorem modelrow_agrees_7735 : modelRow 7735 = row 7735 := model_count_7735.trans row_count_7735.symm
def row_7736 : RowCertificate 7736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7736 1) := by change ¬ Nat.Prime 239398257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79799419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7736 2) := by change ¬ Nat.Prime 239382785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47876557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7736 3) := by change ¬ Nat.Prime 239367313; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1722067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7736 : row 7736 = 0 := (row_7736.checked).trans rfl
theorem model_count_7736 : modelRow 7736 = 0 := by decide
theorem modelrow_agrees_7736 : modelRow 7736 = row 7736 := model_count_7736.trans row_count_7736.symm
def row_7737 : RowCertificate 7737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7737 1) := by change ¬ Nat.Prime 239460151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34208593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7737 2) := by change ¬ Nat.Prime 239444677; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14084981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7737 3) := by change ¬ Nat.Prime 239429203; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18417631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7737 : row 7737 = 0 := (row_7737.checked).trans rfl
theorem model_count_7737 : modelRow 7737 = 0 := by decide
theorem modelrow_agrees_7737 : modelRow 7737 = row 7737 := model_count_7737.trans row_count_7737.symm
def row_7738 : RowCertificate 7738 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7738 1) := by change ¬ Nat.Prime 239522053; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6473569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7738 2) := by change Nat.Prime 239506577; exact ProjectEulerSolutions.P58PrimeNodes.p_239506577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7738 3) := by change ¬ Nat.Prime 239491101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79830367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7738 : row 7738 = 1 := (row_7738.checked).trans rfl
theorem model_count_7738 : modelRow 7738 = 1 := by decide
theorem modelrow_agrees_7738 : modelRow 7738 = row 7738 := model_count_7738.trans row_count_7738.symm
def row_7739 : RowCertificate 7739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7739 1) := by change ¬ Nat.Prime 239583963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79861321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7739 2) := by change ¬ Nat.Prime 239568485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47913697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7739 3) := by change ¬ Nat.Prime 239553007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12608053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7739 : row 7739 = 0 := (row_7739.checked).trans rfl
theorem model_count_7739 : modelRow 7739 = 0 := by decide
theorem modelrow_agrees_7739 : modelRow 7739 = row 7739 := model_count_7739.trans row_count_7739.symm
def row_7740 : RowCertificate 7740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7740 1) := by change ¬ Nat.Prime 239645881; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3928621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7740 2) := by change Nat.Prime 239630401; exact ProjectEulerSolutions.P58PrimeNodes.p_239630401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7740 3) := by change ¬ Nat.Prime 239614921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34230703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7740 : row 7740 = 1 := (row_7740.checked).trans rfl
theorem model_count_7740 : modelRow 7740 = 1 := by decide
theorem modelrow_agrees_7740 : modelRow 7740 = row 7740 := model_count_7740.trans row_count_7740.symm
def row_7741 : RowCertificate 7741 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7741 1) := by change Nat.Prime 239707807; exact ProjectEulerSolutions.P58PrimeNodes.p_239707807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7741 2) := by change ¬ Nat.Prime 239692325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47938465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7741 3) := by change ¬ Nat.Prime 239676843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79892281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7741 : row 7741 = 1 := (row_7741.checked).trans rfl
theorem model_count_7741 : modelRow 7741 = 1 := by decide
theorem modelrow_agrees_7741 : modelRow 7741 = row 7741 := model_count_7741.trans row_count_7741.symm
def row_7742 : RowCertificate 7742 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7742 1) := by change ¬ Nat.Prime 239769741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79923247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7742 2) := by change ¬ Nat.Prime 239754257; exact PrattCertificates.composite_of_product (factor := 7841) (cofactor := 30577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7742 3) := by change Nat.Prime 239738773; exact ProjectEulerSolutions.P58PrimeNodes.p_239738773.prime
    exact decide_eq_true prime
theorem row_count_7742 : row 7742 = 1 := (row_7742.checked).trans rfl
theorem model_count_7742 : modelRow 7742 = 1 := by decide
theorem modelrow_agrees_7742 : modelRow 7742 = row 7742 := model_count_7742.trans row_count_7742.symm
def row_7743 : RowCertificate 7743 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7743 1) := by change ¬ Nat.Prime 239831683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34261669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7743 2) := by change ¬ Nat.Prime 239816197; exact PrattCertificates.composite_of_product (factor := 1453) (cofactor := 165049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7743 3) := by change Nat.Prime 239800711; exact ProjectEulerSolutions.P58PrimeNodes.p_239800711.prime
    exact decide_eq_true prime
theorem row_count_7743 : row 7743 = 1 := (row_7743.checked).trans rfl
theorem model_count_7743 : modelRow 7743 = 1 := by decide
theorem modelrow_agrees_7743 : modelRow 7743 = row 7743 := model_count_7743.trans row_count_7743.symm
def row_7744 : RowCertificate 7744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7744 1) := by change ¬ Nat.Prime 239893633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34270519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7744 2) := by change ¬ Nat.Prime 239878145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 47975629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7744 3) := by change ¬ Nat.Prime 239862657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79954219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7744 : row 7744 = 0 := (row_7744.checked).trans rfl
theorem model_count_7744 : modelRow 7744 = 0 := by decide
theorem modelrow_agrees_7744 : modelRow 7744 = row 7744 := model_count_7744.trans row_count_7744.symm
def row_7745 : RowCertificate 7745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7745 1) := by change ¬ Nat.Prime 239955591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 79985197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7745 2) := by change ¬ Nat.Prime 239940101; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 1386937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7745 3) := by change Nat.Prime 239924611; exact ProjectEulerSolutions.P58PrimeNodes.p_239924611.prime
    exact decide_eq_true prime
theorem row_count_7745 : row 7745 = 1 := (row_7745.checked).trans rfl
theorem model_count_7745 : modelRow 7745 = 1 := by decide
theorem modelrow_agrees_7745 : modelRow 7745 = row 7745 := model_count_7745.trans row_count_7745.symm
def row_7746 : RowCertificate 7746 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7746 1) := by change ¬ Nat.Prime 240017557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18462889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7746 2) := by change ¬ Nat.Prime 240002065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48000413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7746 3) := by change ¬ Nat.Prime 239986573; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 208141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7746 : row 7746 = 0 := (row_7746.checked).trans rfl
theorem model_count_7746 : modelRow 7746 = 0 := by decide
theorem modelrow_agrees_7746 : modelRow 7746 = row 7746 := model_count_7746.trans row_count_7746.symm
def row_7747 : RowCertificate 7747 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7747 1) := by change ¬ Nat.Prime 240079531; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7744501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7747 2) := by change ¬ Nat.Prime 240064037; exact PrattCertificates.composite_of_product (factor := 5641) (cofactor := 42557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7747 3) := by change ¬ Nat.Prime 240048543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80016181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7747 : row 7747 = 0 := (row_7747.checked).trans rfl
theorem model_count_7747 : modelRow 7747 = 0 := by decide
theorem modelrow_agrees_7747 : modelRow 7747 = row 7747 := model_count_7747.trans row_count_7747.symm
def row_7748 : RowCertificate 7748 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7748 1) := by change ¬ Nat.Prime 240141513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80047171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7748 2) := by change ¬ Nat.Prime 240126017; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2125009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7748 3) := by change ¬ Nat.Prime 240110521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34301503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7748 : row 7748 = 0 := (row_7748.checked).trans rfl
theorem model_count_7748 : modelRow 7748 = 0 := by decide
theorem modelrow_agrees_7748 : modelRow 7748 = row 7748 := model_count_7748.trans row_count_7748.symm
def row_7749 : RowCertificate 7749 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7749 1) := by change Nat.Prime 240203503; exact ProjectEulerSolutions.P58PrimeNodes.p_240203503.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7749 2) := by change ¬ Nat.Prime 240188005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48037601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7749 3) := by change ¬ Nat.Prime 240172507; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 654421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7749 : row 7749 = 1 := (row_7749.checked).trans rfl
theorem model_count_7749 : modelRow 7749 = 1 := by decide
theorem modelrow_agrees_7749 : modelRow 7749 = row 7749 := model_count_7749.trans row_count_7749.symm
def row_7750 : RowCertificate 7750 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7750 1) := by change ¬ Nat.Prime 240265501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34323643) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7750 2) := by change ¬ Nat.Prime 240250001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14132353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7750 3) := by change ¬ Nat.Prime 240234501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80078167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7750 : row 7750 = 0 := (row_7750.checked).trans rfl
theorem model_count_7750 : modelRow 7750 = 0 := by decide
theorem modelrow_agrees_7750 : modelRow 7750 = row 7750 := model_count_7750.trans row_count_7750.symm
def row_7751 : RowCertificate 7751 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7751 1) := by change ¬ Nat.Prime 240327507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80109169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7751 2) := by change ¬ Nat.Prime 240312005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48062401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7751 3) := by change Nat.Prime 240296503; exact ProjectEulerSolutions.P58PrimeNodes.p_240296503.prime
    exact decide_eq_true prime
theorem row_count_7751 : row 7751 = 1 := (row_7751.checked).trans rfl
theorem model_count_7751 : modelRow 7751 = 1 := by decide
theorem modelrow_agrees_7751 : modelRow 7751 = row 7751 := model_count_7751.trans row_count_7751.symm
def row_7752 : RowCertificate 7752 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7752 1) := by change Nat.Prime 240389521; exact ProjectEulerSolutions.P58PrimeNodes.p_240389521.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7752 2) := by change ¬ Nat.Prime 240374017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18490309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7752 3) := by change ¬ Nat.Prime 240358513; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 274069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7752 : row 7752 = 1 := (row_7752.checked).trans rfl
theorem model_count_7752 : modelRow 7752 = 1 := by decide
theorem modelrow_agrees_7752 : modelRow 7752 = row 7752 := model_count_7752.trans row_count_7752.symm
def row_7753 : RowCertificate 7753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7753 1) := by change ¬ Nat.Prime 240451543; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3588829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7753 2) := by change ¬ Nat.Prime 240436037; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 4536529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7753 3) := by change ¬ Nat.Prime 240420531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80140177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7753 : row 7753 = 0 := (row_7753.checked).trans rfl
theorem model_count_7753 : modelRow 7753 = 0 := by decide
theorem modelrow_agrees_7753 : modelRow 7753 = row 7753 := model_count_7753.trans row_count_7753.symm
def row_7754 : RowCertificate 7754 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7754 1) := by change ¬ Nat.Prime 240513573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80171191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7754 2) := by change ¬ Nat.Prime 240498065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48099613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7754 3) := by change ¬ Nat.Prime 240482557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34354651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7754 : row 7754 = 0 := (row_7754.checked).trans rfl
theorem model_count_7754 : modelRow 7754 = 0 := by decide
theorem modelrow_agrees_7754 : modelRow 7754 = row 7754 := model_count_7754.trans row_count_7754.symm
def row_7755 : RowCertificate 7755 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7755 1) := by change Nat.Prime 240575611; exact ProjectEulerSolutions.P58PrimeNodes.p_240575611.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7755 2) := by change Nat.Prime 240560101; exact ProjectEulerSolutions.P58PrimeNodes.p_240560101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7755 3) := by change ¬ Nat.Prime 240544591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34363513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7755 : row 7755 = 2 := (row_7755.checked).trans rfl
theorem model_count_7755 : modelRow 7755 = 2 := by decide
theorem modelrow_agrees_7755 : modelRow 7755 = row 7755 := model_count_7755.trans row_count_7755.symm
def row_7756 : RowCertificate 7756 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7756 1) := by change ¬ Nat.Prime 240637657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18510589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7756 2) := by change ¬ Nat.Prime 240622145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48124429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7756 3) := by change ¬ Nat.Prime 240606633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80202211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7756 : row 7756 = 0 := (row_7756.checked).trans rfl
theorem model_count_7756 : modelRow 7756 = 0 := by decide
theorem modelrow_agrees_7756 : modelRow 7756 = row 7756 := model_count_7756.trans row_count_7756.symm
def row_7757 : RowCertificate 7757 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7757 1) := by change ¬ Nat.Prime 240699711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80233237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7757 2) := by change ¬ Nat.Prime 240684197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18514169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7757 3) := by change ¬ Nat.Prime 240668683; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6504559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7757 : row 7757 = 0 := (row_7757.checked).trans rfl
theorem model_count_7757 : modelRow 7757 = 0 := by decide
theorem modelrow_agrees_7757 : modelRow 7757 = row 7757 := model_count_7757.trans row_count_7757.symm
def row_7758 : RowCertificate 7758 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7758 1) := by change ¬ Nat.Prime 240761773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34394539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7758 2) := by change Nat.Prime 240746257; exact ProjectEulerSolutions.P58PrimeNodes.p_240746257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7758 3) := by change ¬ Nat.Prime 240730741; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12670039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7758 : row 7758 = 1 := (row_7758.checked).trans rfl
theorem model_count_7758 : modelRow 7758 = 1 := by decide
theorem modelrow_agrees_7758 : modelRow 7758 = row 7758 := model_count_7758.trans row_count_7758.symm
def row_7759 : RowCertificate 7759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7759 1) := by change ¬ Nat.Prime 240823843; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18524911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7759 2) := by change ¬ Nat.Prime 240808325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48161665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7759 3) := by change ¬ Nat.Prime 240792807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80264269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7759 : row 7759 = 0 := (row_7759.checked).trans rfl
theorem model_count_7759 : modelRow 7759 = 0 := by decide
theorem modelrow_agrees_7759 : modelRow 7759 = row 7759 := model_count_7759.trans row_count_7759.symm
def row_7760 : RowCertificate 7760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7760 1) := by change ¬ Nat.Prime 240885921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80295307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7760 2) := by change ¬ Nat.Prime 240870401; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 2706409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7760 3) := by change Nat.Prime 240854881; exact ProjectEulerSolutions.P58PrimeNodes.p_240854881.prime
    exact decide_eq_true prime
theorem row_count_7760 : row 7760 = 1 := (row_7760.checked).trans rfl
theorem model_count_7760 : modelRow 7760 = 1 := by decide
theorem modelrow_agrees_7760 : modelRow 7760 = row 7760 := model_count_7760.trans row_count_7760.symm
def row_7761 : RowCertificate 7761 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7761 1) := by change Nat.Prime 240948007; exact ProjectEulerSolutions.P58PrimeNodes.p_240948007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7761 2) := by change ¬ Nat.Prime 240932485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48186497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7761 3) := by change ¬ Nat.Prime 240916963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34416709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7761 : row 7761 = 1 := (row_7761.checked).trans rfl
theorem model_count_7761 : modelRow 7761 = 1 := by decide
theorem modelrow_agrees_7761 : modelRow 7761 = row 7761 := model_count_7761.trans row_count_7761.symm
def row_7762 : RowCertificate 7762 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7762 1) := by change ¬ Nat.Prime 241010101; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1248757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7762 2) := by change Nat.Prime 240994577; exact ProjectEulerSolutions.P58PrimeNodes.p_240994577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7762 3) := by change ¬ Nat.Prime 240979053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80326351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7762 : row 7762 = 1 := (row_7762.checked).trans rfl
theorem model_count_7762 : modelRow 7762 = 1 := by decide
theorem modelrow_agrees_7762 : modelRow 7762 = row 7762 := model_count_7762.trans row_count_7762.symm
def row_7763 : RowCertificate 7763 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7763 1) := by change ¬ Nat.Prime 241072203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80357401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7763 2) := by change ¬ Nat.Prime 241056677; exact PrattCertificates.composite_of_product (factor := 3121) (cofactor := 77237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7763 3) := by change ¬ Nat.Prime 241041151; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18541627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7763 : row 7763 = 0 := (row_7763.checked).trans rfl
theorem model_count_7763 : modelRow 7763 = 0 := by decide
theorem modelrow_agrees_7763 : modelRow 7763 = row 7763 := model_count_7763.trans row_count_7763.symm
def row_7764 : RowCertificate 7764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7764 1) := by change ¬ Nat.Prime 241134313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34447759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7764 2) := by change ¬ Nat.Prime 241118785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48223757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7764 3) := by change Nat.Prime 241103257; exact ProjectEulerSolutions.P58PrimeNodes.p_241103257.prime
    exact decide_eq_true prime
theorem row_count_7764 : row 7764 = 1 := (row_7764.checked).trans rfl
theorem model_count_7764 : modelRow 7764 = 1 := by decide
theorem modelrow_agrees_7764 : modelRow 7764 = row 7764 := model_count_7764.trans row_count_7764.symm
def row_7765 : RowCertificate 7765 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7765 1) := by change ¬ Nat.Prime 241196431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34456633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7765 2) := by change ¬ Nat.Prime 241180901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18552377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7765 3) := by change ¬ Nat.Prime 241165371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80388457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7765 : row 7765 = 0 := (row_7765.checked).trans rfl
theorem model_count_7765 : modelRow 7765 = 0 := by decide
theorem modelrow_agrees_7765 : modelRow 7765 = row 7765 := model_count_7765.trans row_count_7765.symm
def row_7766 : RowCertificate 7766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7766 1) := by change ¬ Nat.Prime 241258557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80419519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7766 2) := by change ¬ Nat.Prime 241243025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48248605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7766 3) := by change ¬ Nat.Prime 241227493; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18555961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7766 : row 7766 = 0 := (row_7766.checked).trans rfl
theorem model_count_7766 : modelRow 7766 = 0 := by decide
theorem modelrow_agrees_7766 : modelRow 7766 = row 7766 := model_count_7766.trans row_count_7766.symm
def row_7767 : RowCertificate 7767 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7767 1) := by change ¬ Nat.Prime 241320691; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12701089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7767 2) := by change ¬ Nat.Prime 241305157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14194421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7767 3) := by change Nat.Prime 241289623; exact ProjectEulerSolutions.P58PrimeNodes.p_241289623.prime
    exact decide_eq_true prime
theorem row_count_7767 : row 7767 = 1 := (row_7767.checked).trans rfl
theorem model_count_7767 : modelRow 7767 = 1 := by decide
theorem modelrow_agrees_7767 : modelRow 7767 = row 7767 := model_count_7767.trans row_count_7767.symm
def row_7768 : RowCertificate 7768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7768 1) := by change ¬ Nat.Prime 241382833; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7786543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7768 2) := by change Nat.Prime 241367297; exact ProjectEulerSolutions.P58PrimeNodes.p_241367297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7768 3) := by change ¬ Nat.Prime 241351761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80450587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7768 : row 7768 = 1 := (row_7768.checked).trans rfl
theorem model_count_7768 : modelRow 7768 = 1 := by decide
theorem modelrow_agrees_7768 : modelRow 7768 = row 7768 := model_count_7768.trans row_count_7768.symm
def row_7769 : RowCertificate 7769 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7769 1) := by change ¬ Nat.Prime 241444983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80481661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7769 2) := by change ¬ Nat.Prime 241429445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48285889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7769 3) := by change ¬ Nat.Prime 241413907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34487701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7769 : row 7769 = 0 := (row_7769.checked).trans rfl
theorem model_count_7769 : modelRow 7769 = 0 := by decide
theorem modelrow_agrees_7769 : modelRow 7769 = row 7769 := model_count_7769.trans row_count_7769.symm
def row_7770 : RowCertificate 7770 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7770 1) := by change ¬ Nat.Prime 241507141; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3308317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7770 2) := by change ¬ Nat.Prime 241491601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18576277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7770 3) := by change ¬ Nat.Prime 241476061; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3056659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7770 : row 7770 = 0 := (row_7770.checked).trans rfl
theorem model_count_7770 : modelRow 7770 = 0 := by decide
theorem modelrow_agrees_7770 : modelRow 7770 = row 7770 := model_count_7770.trans row_count_7770.symm
def row_7771 : RowCertificate 7771 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7771 1) := by change ¬ Nat.Prime 241569307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34509901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7771 2) := by change ¬ Nat.Prime 241553765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48310753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7771 3) := by change ¬ Nat.Prime 241538223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80512741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7771 : row 7771 = 0 := (row_7771.checked).trans rfl
theorem model_count_7771 : modelRow 7771 = 0 := by decide
theorem modelrow_agrees_7771 : modelRow 7771 = row 7771 := model_count_7771.trans row_count_7771.symm
def row_7772 : RowCertificate 7772 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7772 1) := by change ¬ Nat.Prime 241631481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80543827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7772 2) := by change ¬ Nat.Prime 241615937; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3960917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7772 3) := by change ¬ Nat.Prime 241600393; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1482211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7772 : row 7772 = 0 := (row_7772.checked).trans rfl
theorem model_count_7772 : modelRow 7772 = 0 := by decide
theorem modelrow_agrees_7772 : modelRow 7772 = row 7772 := model_count_7772.trans row_count_7772.symm
def row_7773 : RowCertificate 7773 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7773 1) := by change Nat.Prime 241693663; exact ProjectEulerSolutions.P58PrimeNodes.p_241693663.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7773 2) := by change ¬ Nat.Prime 241678117; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6531841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7773 3) := by change ¬ Nat.Prime 241662571; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 174991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7773 : row 7773 = 1 := (row_7773.checked).trans rfl
theorem model_count_7773 : modelRow 7773 = 1 := by decide
theorem modelrow_agrees_7773 : modelRow 7773 = row 7773 := model_count_7773.trans row_count_7773.symm
def row_7774 : RowCertificate 7774 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7774 1) := by change ¬ Nat.Prime 241755853; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1252621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7774 2) := by change ¬ Nat.Prime 241740305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48348061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7774 3) := by change ¬ Nat.Prime 241724757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80574919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7774 : row 7774 = 0 := (row_7774.checked).trans rfl
theorem model_count_7774 : modelRow 7774 = 0 := by decide
theorem modelrow_agrees_7774 : modelRow 7774 = row 7774 := model_count_7774.trans row_count_7774.symm
def row_7775 : RowCertificate 7775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7775 1) := by change ¬ Nat.Prime 241818051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80606017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7775 2) := by change Nat.Prime 241802501; exact ProjectEulerSolutions.P58PrimeNodes.p_241802501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7775 3) := by change ¬ Nat.Prime 241786951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34540993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7775 : row 7775 = 1 := (row_7775.checked).trans rfl
theorem model_count_7775 : modelRow 7775 = 1 := by decide
theorem modelrow_agrees_7775 : modelRow 7775 = row 7775 := model_count_7775.trans row_count_7775.symm
def row_7776 : RowCertificate 7776 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7776 1) := by change Nat.Prime 241880257; exact ProjectEulerSolutions.P58PrimeNodes.p_241880257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7776 2) := by change ¬ Nat.Prime 241864705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48372941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7776 3) := by change ¬ Nat.Prime 241849153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34549879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7776 : row 7776 = 1 := (row_7776.checked).trans rfl
theorem model_count_7776 : modelRow 7776 = 1 := by decide
theorem modelrow_agrees_7776 : modelRow 7776 = row 7776 := model_count_7776.trans row_count_7776.symm
def row_7777 : RowCertificate 7777 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7777 1) := by change ¬ Nat.Prime 241942471; exact PrattCertificates.composite_of_product (factor := 1321) (cofactor := 183151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7777 2) := by change ¬ Nat.Prime 241926917; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2219513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7777 3) := by change ¬ Nat.Prime 241911363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80637121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7777 : row 7777 = 0 := (row_7777.checked).trans rfl
theorem model_count_7777 : modelRow 7777 = 0 := by decide
theorem modelrow_agrees_7777 : modelRow 7777 = row 7777 := model_count_7777.trans row_count_7777.symm
def row_7778 : RowCertificate 7778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7778 1) := by change ¬ Nat.Prime 242004693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80668231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7778 2) := by change ¬ Nat.Prime 241989137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18614549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7778 3) := by change ¬ Nat.Prime 241973581; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2494573) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7778 : row 7778 = 0 := (row_7778.checked).trans rfl
theorem model_count_7778 : modelRow 7778 = 0 := by decide
theorem modelrow_agrees_7778 : modelRow 7778 = row 7778 := model_count_7778.trans row_count_7778.symm
def row_7779 : RowCertificate 7779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7779 1) := by change ¬ Nat.Prime 242066923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34580989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7779 2) := by change ¬ Nat.Prime 242051365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48410273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7779 3) := by change ¬ Nat.Prime 242035807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18618139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7779 : row 7779 = 0 := (row_7779.checked).trans rfl
theorem model_count_7779 : modelRow 7779 = 0 := by decide
theorem modelrow_agrees_7779 : modelRow 7779 = row 7779 := model_count_7779.trans row_count_7779.symm
def row_7780 : RowCertificate 7780 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7780 1) := by change Nat.Prime 242129161; exact ProjectEulerSolutions.P58PrimeNodes.p_242129161.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7780 2) := by change Nat.Prime 242113601; exact ProjectEulerSolutions.P58PrimeNodes.p_242113601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7780 3) := by change ¬ Nat.Prime 242098041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80699347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7780 : row 7780 = 2 := (row_7780.checked).trans rfl
theorem model_count_7780 : modelRow 7780 = 2 := by decide
theorem modelrow_agrees_7780 : modelRow 7780 = row 7780 := model_count_7780.trans row_count_7780.symm
def row_7781 : RowCertificate 7781 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7781 1) := by change ¬ Nat.Prime 242191407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80730469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7781 2) := by change ¬ Nat.Prime 242175845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48435169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7781 3) := by change ¬ Nat.Prime 242160283; exact PrattCertificates.composite_of_product (factor := 5347) (cofactor := 45289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7781 : row 7781 = 0 := (row_7781.checked).trans rfl
theorem model_count_7781 : modelRow 7781 = 0 := by decide
theorem modelrow_agrees_7781 : modelRow 7781 = row 7781 := model_count_7781.trans row_count_7781.symm
def row_7782 : RowCertificate 7782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7782 1) := by change ¬ Nat.Prime 242253661; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18634897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7782 2) := by change ¬ Nat.Prime 242238097; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 2398397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7782 3) := by change ¬ Nat.Prime 242222533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34603219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7782 : row 7782 = 0 := (row_7782.checked).trans rfl
theorem model_count_7782 : modelRow 7782 = 0 := by decide
theorem modelrow_agrees_7782 : modelRow 7782 = row 7782 := model_count_7782.trans row_count_7782.symm
def row_7783 : RowCertificate 7783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7783 1) := by change ¬ Nat.Prime 242315923; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6549079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7783 2) := by change ¬ Nat.Prime 242300357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18638489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7783 3) := by change ¬ Nat.Prime 242284791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80761597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7783 : row 7783 = 0 := (row_7783.checked).trans rfl
theorem model_count_7783 : modelRow 7783 = 0 := by decide
theorem modelrow_agrees_7783 : modelRow 7783 = row 7783 := model_count_7783.trans row_count_7783.symm
def row_7784 : RowCertificate 7784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7784 1) := by change ¬ Nat.Prime 242378193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80792731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7784 2) := by change ¬ Nat.Prime 242362625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48472525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7784 3) := by change ¬ Nat.Prime 242347057; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7817647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7784 : row 7784 = 0 := (row_7784.checked).trans rfl
theorem model_count_7784 : modelRow 7784 = 0 := by decide
theorem modelrow_agrees_7784 : modelRow 7784 = row 7784 := model_count_7784.trans row_count_7784.symm
def row_7785 : RowCertificate 7785 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7785 1) := by change ¬ Nat.Prime 242440471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34634353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7785 2) := by change Nat.Prime 242424901; exact ProjectEulerSolutions.P58PrimeNodes.p_242424901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7785 3) := by change Nat.Prime 242409331; exact ProjectEulerSolutions.P58PrimeNodes.p_242409331.prime
    exact decide_eq_true prime
theorem row_count_7785 : row 7785 = 2 := (row_7785.checked).trans rfl
theorem model_count_7785 : modelRow 7785 = 2 := by decide
theorem modelrow_agrees_7785 : modelRow 7785 = row 7785 := model_count_7785.trans row_count_7785.symm
def row_7786 : RowCertificate 7786 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7786 1) := by change ¬ Nat.Prime 242502757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34643251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7786 2) := by change ¬ Nat.Prime 242487185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48497437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7786 3) := by change ¬ Nat.Prime 242471613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80823871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7786 : row 7786 = 0 := (row_7786.checked).trans rfl
theorem model_count_7786 : modelRow 7786 = 0 := by decide
theorem modelrow_agrees_7786 : modelRow 7786 = row 7786 := model_count_7786.trans row_count_7786.symm
def row_7787 : RowCertificate 7787 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7787 1) := by change ¬ Nat.Prime 242565051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80855017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7787 2) := by change ¬ Nat.Prime 242549477; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 687109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7787 3) := by change ¬ Nat.Prime 242533903; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3619909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7787 : row 7787 = 0 := (row_7787.checked).trans rfl
theorem model_count_7787 : modelRow 7787 = 0 := by decide
theorem modelrow_agrees_7787 : modelRow 7787 = row 7787 := model_count_7787.trans row_count_7787.symm
def row_7788 : RowCertificate 7788 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7788 1) := by change ¬ Nat.Prime 242627353; exact PrattCertificates.composite_of_product (factor := 1327) (cofactor := 182839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7788 2) := by change ¬ Nat.Prime 242611777; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14271281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7788 3) := by change Nat.Prime 242596201; exact ProjectEulerSolutions.P58PrimeNodes.p_242596201.prime
    exact decide_eq_true prime
theorem row_count_7788 : row 7788 = 1 := (row_7788.checked).trans rfl
theorem model_count_7788 : modelRow 7788 = 1 := by decide
theorem modelrow_agrees_7788 : modelRow 7788 = row 7788 := model_count_7788.trans row_count_7788.symm
def row_7789 : RowCertificate 7789 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7789 1) := by change ¬ Nat.Prime 242689663; exact PrattCertificates.composite_of_product (factor := 11257) (cofactor := 21559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7789 2) := by change ¬ Nat.Prime 242674085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48534817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7789 3) := by change ¬ Nat.Prime 242658507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80886169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7789 : row 7789 = 0 := (row_7789.checked).trans rfl
theorem model_count_7789 : modelRow 7789 = 0 := by decide
theorem modelrow_agrees_7789 : modelRow 7789 = row 7789 := model_count_7789.trans row_count_7789.symm
def row_7790 : RowCertificate 7790 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7790 1) := by change ¬ Nat.Prime 242751981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80917327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7790 2) := by change Nat.Prime 242736401; exact ProjectEulerSolutions.P58PrimeNodes.p_242736401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7790 3) := by change ¬ Nat.Prime 242720821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34674403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7790 : row 7790 = 1 := (row_7790.checked).trans rfl
theorem model_count_7790 : modelRow 7790 = 1 := by decide
theorem modelrow_agrees_7790 : modelRow 7790 = row 7790 := model_count_7790.trans row_count_7790.symm
def row_7791 : RowCertificate 7791 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7791 1) := by change ¬ Nat.Prime 242814307; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 695743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7791 2) := by change ¬ Nat.Prime 242798725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48559745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7791 3) := by change ¬ Nat.Prime 242783143; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3623629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7791 : row 7791 = 0 := (row_7791.checked).trans rfl
theorem model_count_7791 : modelRow 7791 = 0 := by decide
theorem modelrow_agrees_7791 : modelRow 7791 = row 7791 := model_count_7791.trans row_count_7791.symm
def row_7792 : RowCertificate 7792 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7792 1) := by change ¬ Nat.Prime 242876641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34696663) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7792 2) := by change Nat.Prime 242861057; exact ProjectEulerSolutions.P58PrimeNodes.p_242861057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7792 3) := by change ¬ Nat.Prime 242845473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80948491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7792 : row 7792 = 1 := (row_7792.checked).trans rfl
theorem model_count_7792 : modelRow 7792 = 1 := by decide
theorem modelrow_agrees_7792 : modelRow 7792 = row 7792 := model_count_7792.trans row_count_7792.symm
def row_7793 : RowCertificate 7793 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7793 1) := by change ¬ Nat.Prime 242938983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 80979661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7793 2) := by change Nat.Prime 242923397; exact ProjectEulerSolutions.P58PrimeNodes.p_242923397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7793 3) := by change ¬ Nat.Prime 242907811; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1608661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7793 : row 7793 = 1 := (row_7793.checked).trans rfl
theorem model_count_7793 : modelRow 7793 = 1 := by decide
theorem modelrow_agrees_7793 : modelRow 7793 = row 7793 := model_count_7793.trans row_count_7793.symm
def row_7794 : RowCertificate 7794 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7794 1) := by change ¬ Nat.Prime 243001333; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 734143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7794 2) := by change ¬ Nat.Prime 242985745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48597149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7794 3) := by change ¬ Nat.Prime 242970157; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12787903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7794 : row 7794 = 0 := (row_7794.checked).trans rfl
theorem model_count_7794 : modelRow 7794 = 0 := by decide
theorem modelrow_agrees_7794 : modelRow 7794 = row 7794 := model_count_7794.trans row_count_7794.symm
def row_7795 : RowCertificate 7795 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7795 1) := by change ¬ Nat.Prime 243063691; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18697207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7795 2) := by change ¬ Nat.Prime 243048101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8380969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7795 3) := by change ¬ Nat.Prime 243032511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81010837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7795 : row 7795 = 0 := (row_7795.checked).trans rfl
theorem model_count_7795 : modelRow 7795 = 0 := by decide
theorem modelrow_agrees_7795 : modelRow 7795 = row 7795 := model_count_7795.trans row_count_7795.symm
def row_7796 : RowCertificate 7796 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7796 1) := by change ¬ Nat.Prime 243126057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81042019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7796 2) := by change ¬ Nat.Prime 243110465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48622093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7796 3) := by change ¬ Nat.Prime 243094873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34727839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7796 : row 7796 = 0 := (row_7796.checked).trans rfl
theorem model_count_7796 : modelRow 7796 = 0 := by decide
theorem modelrow_agrees_7796 : modelRow 7796 = row 7796 := model_count_7796.trans row_count_7796.symm
def row_7797 : RowCertificate 7797 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7797 1) := by change ¬ Nat.Prime 243188431; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 138727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7797 2) := by change ¬ Nat.Prime 243172837; exact PrattCertificates.composite_of_product (factor := 3413) (cofactor := 71249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7797 3) := by change ¬ Nat.Prime 243157243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34736749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7797 : row 7797 = 0 := (row_7797.checked).trans rfl
theorem model_count_7797 : modelRow 7797 = 0 := by decide
theorem modelrow_agrees_7797 : modelRow 7797 = row 7797 := model_count_7797.trans row_count_7797.symm
def row_7798 : RowCertificate 7798 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7798 1) := by change ¬ Nat.Prime 243250813; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18711601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7798 2) := by change ¬ Nat.Prime 243235217; exact PrattCertificates.composite_of_product (factor := 1181) (cofactor := 205957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7798 3) := by change ¬ Nat.Prime 243219621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81073207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7798 : row 7798 = 0 := (row_7798.checked).trans rfl
theorem model_count_7798 : modelRow 7798 = 0 := by decide
theorem modelrow_agrees_7798 : modelRow 7798 = row 7798 := model_count_7798.trans row_count_7798.symm
def row_7799 : RowCertificate 7799 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7799 1) := by change ¬ Nat.Prime 243313203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81104401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7799 2) := by change ¬ Nat.Prime 243297605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48659521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7799 3) := by change ¬ Nat.Prime 243282007; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 577867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7799 : row 7799 = 0 := (row_7799.checked).trans rfl
theorem model_count_7799 : modelRow 7799 = 0 := by decide
theorem modelrow_agrees_7799 : modelRow 7799 = row 7799 := model_count_7799.trans row_count_7799.symm
def row_7800 : RowCertificate 7800 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7800 1) := by change ¬ Nat.Prime 243375601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34767943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7800 2) := by change ¬ Nat.Prime 243360001; exact PrattCertificates.composite_of_product (factor := 1297) (cofactor := 187633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7800 3) := by change ¬ Nat.Prime 243344401; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2362567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7800 : row 7800 = 0 := (row_7800.checked).trans rfl
theorem model_count_7800 : modelRow 7800 = 0 := by decide
theorem modelrow_agrees_7800 : modelRow 7800 = row 7800 := model_count_7800.trans row_count_7800.symm
def row_7801 : RowCertificate 7801 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7801 1) := by change ¬ Nat.Prime 243438007; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5661349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7801 2) := by change ¬ Nat.Prime 243422405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48684481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7801 3) := by change ¬ Nat.Prime 243406803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81135601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7801 : row 7801 = 0 := (row_7801.checked).trans rfl
theorem model_count_7801 : modelRow 7801 = 0 := by decide
theorem modelrow_agrees_7801 : modelRow 7801 = row 7801 := model_count_7801.trans row_count_7801.symm
def row_7802 : RowCertificate 7802 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7802 1) := by change ¬ Nat.Prime 243500421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81166807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7802 2) := by change ¬ Nat.Prime 243484817; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 241313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7802 3) := by change ¬ Nat.Prime 243469213; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18728401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7802 : row 7802 = 0 := (row_7802.checked).trans rfl
theorem model_count_7802 : modelRow 7802 = 0 := by decide
theorem modelrow_agrees_7802 : modelRow 7802 = row 7802 := model_count_7802.trans row_count_7802.symm
def row_7803 : RowCertificate 7803 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7803 1) := by change ¬ Nat.Prime 243562843; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12819097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7803 2) := by change ¬ Nat.Prime 243547237; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 578497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7803 3) := by change ¬ Nat.Prime 243531631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34790233) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7803 : row 7803 = 0 := (row_7803.checked).trans rfl
theorem model_count_7803 : modelRow 7803 = 0 := by decide
theorem modelrow_agrees_7803 : modelRow 7803 = row 7803 := model_count_7803.trans row_count_7803.symm
def row_7804 : RowCertificate 7804 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7804 1) := by change ¬ Nat.Prime 243625273; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 488227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7804 2) := by change ¬ Nat.Prime 243609665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48721933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7804 3) := by change ¬ Nat.Prime 243594057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81198019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7804 : row 7804 = 0 := (row_7804.checked).trans rfl
theorem model_count_7804 : modelRow 7804 = 0 := by decide
theorem modelrow_agrees_7804 : modelRow 7804 = row 7804 := model_count_7804.trans row_count_7804.symm
def row_7805 : RowCertificate 7805 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7805 1) := by change ¬ Nat.Prime 243687711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81229237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7805 2) := by change ¬ Nat.Prime 243672101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14333653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7805 3) := by change ¬ Nat.Prime 243656491; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18742807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7805 : row 7805 = 0 := (row_7805.checked).trans rfl
theorem model_count_7805 : modelRow 7805 = 0 := by decide
theorem modelrow_agrees_7805 : modelRow 7805 = row 7805 := model_count_7805.trans row_count_7805.symm
def row_7806 : RowCertificate 7806 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7806 1) := by change ¬ Nat.Prime 243750157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34821451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7806 2) := by change ¬ Nat.Prime 243734545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48746909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7806 3) := by change ¬ Nat.Prime 243718933; exact PrattCertificates.composite_of_product (factor := 10531) (cofactor := 23143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7806 : row 7806 = 0 := (row_7806.checked).trans rfl
theorem model_count_7806 : modelRow 7806 = 0 := by decide
theorem modelrow_agrees_7806 : modelRow 7806 = row 7806 := model_count_7806.trans row_count_7806.symm
def row_7807 : RowCertificate 7807 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7807 1) := by change ¬ Nat.Prime 243812611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34830373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7807 2) := by change ¬ Nat.Prime 243796997; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8406793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7807 3) := by change ¬ Nat.Prime 243781383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81260461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7807 : row 7807 = 0 := (row_7807.checked).trans rfl
theorem model_count_7807 : modelRow 7807 = 0 := by decide
theorem modelrow_agrees_7807 : modelRow 7807 = row 7807 := model_count_7807.trans row_count_7807.symm
def row_7808 : RowCertificate 7808 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7808 1) := by change ¬ Nat.Prime 243875073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81291691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7808 2) := by change ¬ Nat.Prime 243859457; exact PrattCertificates.composite_of_product (factor := 12373) (cofactor := 19709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7808 3) := by change ¬ Nat.Prime 243843841; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5670787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7808 : row 7808 = 0 := (row_7808.checked).trans rfl
theorem model_count_7808 : modelRow 7808 = 0 := by decide
theorem modelrow_agrees_7808 : modelRow 7808 = row 7808 := model_count_7808.trans row_count_7808.symm
private def atomic_7681 : CountSegment 1 10 13120 7680 7681 3261 3261 := CountSegment.single (first := 7680) (initial := 3261) row_count_7681 (by decide)
private theorem atomic_7681_agreement : RowAgreement 7680 7681 := RowAgreement.single (first := 7680) modelrow_agrees_7681
private def atomic_7682 : CountSegment 1 10 13120 7681 7682 3261 3261 := CountSegment.single (first := 7681) (initial := 3261) row_count_7682 (by decide)
private theorem atomic_7682_agreement : RowAgreement 7681 7682 := RowAgreement.single (first := 7681) modelrow_agrees_7682
private def atomic_7683 : CountSegment 1 10 13120 7682 7683 3261 3261 := CountSegment.single (first := 7682) (initial := 3261) row_count_7683 (by decide)
private theorem atomic_7683_agreement : RowAgreement 7682 7683 := RowAgreement.single (first := 7682) modelrow_agrees_7683
private def atomic_7684 : CountSegment 1 10 13120 7683 7684 3261 3262 := CountSegment.single (first := 7683) (initial := 3261) row_count_7684 (by decide)
private theorem atomic_7684_agreement : RowAgreement 7683 7684 := RowAgreement.single (first := 7683) modelrow_agrees_7684
private def atomic_7685 : CountSegment 1 10 13120 7684 7685 3262 3263 := CountSegment.single (first := 7684) (initial := 3262) row_count_7685 (by decide)
private theorem atomic_7685_agreement : RowAgreement 7684 7685 := RowAgreement.single (first := 7684) modelrow_agrees_7685
private def atomic_7686 : CountSegment 1 10 13120 7685 7686 3263 3264 := CountSegment.single (first := 7685) (initial := 3263) row_count_7686 (by decide)
private theorem atomic_7686_agreement : RowAgreement 7685 7686 := RowAgreement.single (first := 7685) modelrow_agrees_7686
private def atomic_7687 : CountSegment 1 10 13120 7686 7687 3264 3264 := CountSegment.single (first := 7686) (initial := 3264) row_count_7687 (by decide)
private theorem atomic_7687_agreement : RowAgreement 7686 7687 := RowAgreement.single (first := 7686) modelrow_agrees_7687
private def atomic_7688 : CountSegment 1 10 13120 7687 7688 3264 3264 := CountSegment.single (first := 7687) (initial := 3264) row_count_7688 (by decide)
private theorem atomic_7688_agreement : RowAgreement 7687 7688 := RowAgreement.single (first := 7687) modelrow_agrees_7688
private def atomic_7689 : CountSegment 1 10 13120 7688 7689 3264 3264 := CountSegment.single (first := 7688) (initial := 3264) row_count_7689 (by decide)
private theorem atomic_7689_agreement : RowAgreement 7688 7689 := RowAgreement.single (first := 7688) modelrow_agrees_7689
private def atomic_7690 : CountSegment 1 10 13120 7689 7690 3264 3264 := CountSegment.single (first := 7689) (initial := 3264) row_count_7690 (by decide)
private theorem atomic_7690_agreement : RowAgreement 7689 7690 := RowAgreement.single (first := 7689) modelrow_agrees_7690
private def atomic_7691 : CountSegment 1 10 13120 7690 7691 3264 3264 := CountSegment.single (first := 7690) (initial := 3264) row_count_7691 (by decide)
private theorem atomic_7691_agreement : RowAgreement 7690 7691 := RowAgreement.single (first := 7690) modelrow_agrees_7691
private def atomic_7692 : CountSegment 1 10 13120 7691 7692 3264 3265 := CountSegment.single (first := 7691) (initial := 3264) row_count_7692 (by decide)
private theorem atomic_7692_agreement : RowAgreement 7691 7692 := RowAgreement.single (first := 7691) modelrow_agrees_7692
private def atomic_7693 : CountSegment 1 10 13120 7692 7693 3265 3265 := CountSegment.single (first := 7692) (initial := 3265) row_count_7693 (by decide)
private theorem atomic_7693_agreement : RowAgreement 7692 7693 := RowAgreement.single (first := 7692) modelrow_agrees_7693
private def atomic_7694 : CountSegment 1 10 13120 7693 7694 3265 3265 := CountSegment.single (first := 7693) (initial := 3265) row_count_7694 (by decide)
private theorem atomic_7694_agreement : RowAgreement 7693 7694 := RowAgreement.single (first := 7693) modelrow_agrees_7694
private def atomic_7695 : CountSegment 1 10 13120 7694 7695 3265 3265 := CountSegment.single (first := 7694) (initial := 3265) row_count_7695 (by decide)
private theorem atomic_7695_agreement : RowAgreement 7694 7695 := RowAgreement.single (first := 7694) modelrow_agrees_7695
private def atomic_7696 : CountSegment 1 10 13120 7695 7696 3265 3266 := CountSegment.single (first := 7695) (initial := 3265) row_count_7696 (by decide)
private theorem atomic_7696_agreement : RowAgreement 7695 7696 := RowAgreement.single (first := 7695) modelrow_agrees_7696
private def atomic_7697 : CountSegment 1 10 13120 7696 7697 3266 3267 := CountSegment.single (first := 7696) (initial := 3266) row_count_7697 (by decide)
private theorem atomic_7697_agreement : RowAgreement 7696 7697 := RowAgreement.single (first := 7696) modelrow_agrees_7697
private def atomic_7698 : CountSegment 1 10 13120 7697 7698 3267 3267 := CountSegment.single (first := 7697) (initial := 3267) row_count_7698 (by decide)
private theorem atomic_7698_agreement : RowAgreement 7697 7698 := RowAgreement.single (first := 7697) modelrow_agrees_7698
private def atomic_7699 : CountSegment 1 10 13120 7698 7699 3267 3267 := CountSegment.single (first := 7698) (initial := 3267) row_count_7699 (by decide)
private theorem atomic_7699_agreement : RowAgreement 7698 7699 := RowAgreement.single (first := 7698) modelrow_agrees_7699
private def atomic_7700 : CountSegment 1 10 13120 7699 7700 3267 3268 := CountSegment.single (first := 7699) (initial := 3267) row_count_7700 (by decide)
private theorem atomic_7700_agreement : RowAgreement 7699 7700 := RowAgreement.single (first := 7699) modelrow_agrees_7700
private def atomic_7701 : CountSegment 1 10 13120 7700 7701 3268 3268 := CountSegment.single (first := 7700) (initial := 3268) row_count_7701 (by decide)
private theorem atomic_7701_agreement : RowAgreement 7700 7701 := RowAgreement.single (first := 7700) modelrow_agrees_7701
private def atomic_7702 : CountSegment 1 10 13120 7701 7702 3268 3268 := CountSegment.single (first := 7701) (initial := 3268) row_count_7702 (by decide)
private theorem atomic_7702_agreement : RowAgreement 7701 7702 := RowAgreement.single (first := 7701) modelrow_agrees_7702
private def atomic_7703 : CountSegment 1 10 13120 7702 7703 3268 3269 := CountSegment.single (first := 7702) (initial := 3268) row_count_7703 (by decide)
private theorem atomic_7703_agreement : RowAgreement 7702 7703 := RowAgreement.single (first := 7702) modelrow_agrees_7703
private def atomic_7704 : CountSegment 1 10 13120 7703 7704 3269 3270 := CountSegment.single (first := 7703) (initial := 3269) row_count_7704 (by decide)
private theorem atomic_7704_agreement : RowAgreement 7703 7704 := RowAgreement.single (first := 7703) modelrow_agrees_7704
private def atomic_7705 : CountSegment 1 10 13120 7704 7705 3270 3270 := CountSegment.single (first := 7704) (initial := 3270) row_count_7705 (by decide)
private theorem atomic_7705_agreement : RowAgreement 7704 7705 := RowAgreement.single (first := 7704) modelrow_agrees_7705
private def atomic_7706 : CountSegment 1 10 13120 7705 7706 3270 3270 := CountSegment.single (first := 7705) (initial := 3270) row_count_7706 (by decide)
private theorem atomic_7706_agreement : RowAgreement 7705 7706 := RowAgreement.single (first := 7705) modelrow_agrees_7706
private def atomic_7707 : CountSegment 1 10 13120 7706 7707 3270 3271 := CountSegment.single (first := 7706) (initial := 3270) row_count_7707 (by decide)
private theorem atomic_7707_agreement : RowAgreement 7706 7707 := RowAgreement.single (first := 7706) modelrow_agrees_7707
private def atomic_7708 : CountSegment 1 10 13120 7707 7708 3271 3271 := CountSegment.single (first := 7707) (initial := 3271) row_count_7708 (by decide)
private theorem atomic_7708_agreement : RowAgreement 7707 7708 := RowAgreement.single (first := 7707) modelrow_agrees_7708
private def atomic_7709 : CountSegment 1 10 13120 7708 7709 3271 3271 := CountSegment.single (first := 7708) (initial := 3271) row_count_7709 (by decide)
private theorem atomic_7709_agreement : RowAgreement 7708 7709 := RowAgreement.single (first := 7708) modelrow_agrees_7709
private def atomic_7710 : CountSegment 1 10 13120 7709 7710 3271 3271 := CountSegment.single (first := 7709) (initial := 3271) row_count_7710 (by decide)
private theorem atomic_7710_agreement : RowAgreement 7709 7710 := RowAgreement.single (first := 7709) modelrow_agrees_7710
private def atomic_7711 : CountSegment 1 10 13120 7710 7711 3271 3272 := CountSegment.single (first := 7710) (initial := 3271) row_count_7711 (by decide)
private theorem atomic_7711_agreement : RowAgreement 7710 7711 := RowAgreement.single (first := 7710) modelrow_agrees_7711
private def atomic_7712 : CountSegment 1 10 13120 7711 7712 3272 3272 := CountSegment.single (first := 7711) (initial := 3272) row_count_7712 (by decide)
private theorem atomic_7712_agreement : RowAgreement 7711 7712 := RowAgreement.single (first := 7711) modelrow_agrees_7712
private def atomic_7713 : CountSegment 1 10 13120 7712 7713 3272 3272 := CountSegment.single (first := 7712) (initial := 3272) row_count_7713 (by decide)
private theorem atomic_7713_agreement : RowAgreement 7712 7713 := RowAgreement.single (first := 7712) modelrow_agrees_7713
private def atomic_7714 : CountSegment 1 10 13120 7713 7714 3272 3272 := CountSegment.single (first := 7713) (initial := 3272) row_count_7714 (by decide)
private theorem atomic_7714_agreement : RowAgreement 7713 7714 := RowAgreement.single (first := 7713) modelrow_agrees_7714
private def atomic_7715 : CountSegment 1 10 13120 7714 7715 3272 3272 := CountSegment.single (first := 7714) (initial := 3272) row_count_7715 (by decide)
private theorem atomic_7715_agreement : RowAgreement 7714 7715 := RowAgreement.single (first := 7714) modelrow_agrees_7715
private def atomic_7716 : CountSegment 1 10 13120 7715 7716 3272 3273 := CountSegment.single (first := 7715) (initial := 3272) row_count_7716 (by decide)
private theorem atomic_7716_agreement : RowAgreement 7715 7716 := RowAgreement.single (first := 7715) modelrow_agrees_7716
private def atomic_7717 : CountSegment 1 10 13120 7716 7717 3273 3273 := CountSegment.single (first := 7716) (initial := 3273) row_count_7717 (by decide)
private theorem atomic_7717_agreement : RowAgreement 7716 7717 := RowAgreement.single (first := 7716) modelrow_agrees_7717
private def atomic_7718 : CountSegment 1 10 13120 7717 7718 3273 3273 := CountSegment.single (first := 7717) (initial := 3273) row_count_7718 (by decide)
private theorem atomic_7718_agreement : RowAgreement 7717 7718 := RowAgreement.single (first := 7717) modelrow_agrees_7718
private def atomic_7719 : CountSegment 1 10 13120 7718 7719 3273 3274 := CountSegment.single (first := 7718) (initial := 3273) row_count_7719 (by decide)
private theorem atomic_7719_agreement : RowAgreement 7718 7719 := RowAgreement.single (first := 7718) modelrow_agrees_7719
private def atomic_7720 : CountSegment 1 10 13120 7719 7720 3274 3274 := CountSegment.single (first := 7719) (initial := 3274) row_count_7720 (by decide)
private theorem atomic_7720_agreement : RowAgreement 7719 7720 := RowAgreement.single (first := 7719) modelrow_agrees_7720
private def atomic_7721 : CountSegment 1 10 13120 7720 7721 3274 3274 := CountSegment.single (first := 7720) (initial := 3274) row_count_7721 (by decide)
private theorem atomic_7721_agreement : RowAgreement 7720 7721 := RowAgreement.single (first := 7720) modelrow_agrees_7721
private def atomic_7722 : CountSegment 1 10 13120 7721 7722 3274 3274 := CountSegment.single (first := 7721) (initial := 3274) row_count_7722 (by decide)
private theorem atomic_7722_agreement : RowAgreement 7721 7722 := RowAgreement.single (first := 7721) modelrow_agrees_7722
private def atomic_7723 : CountSegment 1 10 13120 7722 7723 3274 3274 := CountSegment.single (first := 7722) (initial := 3274) row_count_7723 (by decide)
private theorem atomic_7723_agreement : RowAgreement 7722 7723 := RowAgreement.single (first := 7722) modelrow_agrees_7723
private def atomic_7724 : CountSegment 1 10 13120 7723 7724 3274 3274 := CountSegment.single (first := 7723) (initial := 3274) row_count_7724 (by decide)
private theorem atomic_7724_agreement : RowAgreement 7723 7724 := RowAgreement.single (first := 7723) modelrow_agrees_7724
private def atomic_7725 : CountSegment 1 10 13120 7724 7725 3274 3275 := CountSegment.single (first := 7724) (initial := 3274) row_count_7725 (by decide)
private theorem atomic_7725_agreement : RowAgreement 7724 7725 := RowAgreement.single (first := 7724) modelrow_agrees_7725
private def atomic_7726 : CountSegment 1 10 13120 7725 7726 3275 3276 := CountSegment.single (first := 7725) (initial := 3275) row_count_7726 (by decide)
private theorem atomic_7726_agreement : RowAgreement 7725 7726 := RowAgreement.single (first := 7725) modelrow_agrees_7726
private def atomic_7727 : CountSegment 1 10 13120 7726 7727 3276 3276 := CountSegment.single (first := 7726) (initial := 3276) row_count_7727 (by decide)
private theorem atomic_7727_agreement : RowAgreement 7726 7727 := RowAgreement.single (first := 7726) modelrow_agrees_7727
private def atomic_7728 : CountSegment 1 10 13120 7727 7728 3276 3277 := CountSegment.single (first := 7727) (initial := 3276) row_count_7728 (by decide)
private theorem atomic_7728_agreement : RowAgreement 7727 7728 := RowAgreement.single (first := 7727) modelrow_agrees_7728
private def atomic_7729 : CountSegment 1 10 13120 7728 7729 3277 3277 := CountSegment.single (first := 7728) (initial := 3277) row_count_7729 (by decide)
private theorem atomic_7729_agreement : RowAgreement 7728 7729 := RowAgreement.single (first := 7728) modelrow_agrees_7729
private def atomic_7730 : CountSegment 1 10 13120 7729 7730 3277 3278 := CountSegment.single (first := 7729) (initial := 3277) row_count_7730 (by decide)
private theorem atomic_7730_agreement : RowAgreement 7729 7730 := RowAgreement.single (first := 7729) modelrow_agrees_7730
private def atomic_7731 : CountSegment 1 10 13120 7730 7731 3278 3278 := CountSegment.single (first := 7730) (initial := 3278) row_count_7731 (by decide)
private theorem atomic_7731_agreement : RowAgreement 7730 7731 := RowAgreement.single (first := 7730) modelrow_agrees_7731
private def atomic_7732 : CountSegment 1 10 13120 7731 7732 3278 3279 := CountSegment.single (first := 7731) (initial := 3278) row_count_7732 (by decide)
private theorem atomic_7732_agreement : RowAgreement 7731 7732 := RowAgreement.single (first := 7731) modelrow_agrees_7732
private def atomic_7733 : CountSegment 1 10 13120 7732 7733 3279 3279 := CountSegment.single (first := 7732) (initial := 3279) row_count_7733 (by decide)
private theorem atomic_7733_agreement : RowAgreement 7732 7733 := RowAgreement.single (first := 7732) modelrow_agrees_7733
private def atomic_7734 : CountSegment 1 10 13120 7733 7734 3279 3280 := CountSegment.single (first := 7733) (initial := 3279) row_count_7734 (by decide)
private theorem atomic_7734_agreement : RowAgreement 7733 7734 := RowAgreement.single (first := 7733) modelrow_agrees_7734
private def atomic_7735 : CountSegment 1 10 13120 7734 7735 3280 3281 := CountSegment.single (first := 7734) (initial := 3280) row_count_7735 (by decide)
private theorem atomic_7735_agreement : RowAgreement 7734 7735 := RowAgreement.single (first := 7734) modelrow_agrees_7735
private def atomic_7736 : CountSegment 1 10 13120 7735 7736 3281 3281 := CountSegment.single (first := 7735) (initial := 3281) row_count_7736 (by decide)
private theorem atomic_7736_agreement : RowAgreement 7735 7736 := RowAgreement.single (first := 7735) modelrow_agrees_7736
private def atomic_7737 : CountSegment 1 10 13120 7736 7737 3281 3281 := CountSegment.single (first := 7736) (initial := 3281) row_count_7737 (by decide)
private theorem atomic_7737_agreement : RowAgreement 7736 7737 := RowAgreement.single (first := 7736) modelrow_agrees_7737
private def atomic_7738 : CountSegment 1 10 13120 7737 7738 3281 3282 := CountSegment.single (first := 7737) (initial := 3281) row_count_7738 (by decide)
private theorem atomic_7738_agreement : RowAgreement 7737 7738 := RowAgreement.single (first := 7737) modelrow_agrees_7738
private def atomic_7739 : CountSegment 1 10 13120 7738 7739 3282 3282 := CountSegment.single (first := 7738) (initial := 3282) row_count_7739 (by decide)
private theorem atomic_7739_agreement : RowAgreement 7738 7739 := RowAgreement.single (first := 7738) modelrow_agrees_7739
private def atomic_7740 : CountSegment 1 10 13120 7739 7740 3282 3283 := CountSegment.single (first := 7739) (initial := 3282) row_count_7740 (by decide)
private theorem atomic_7740_agreement : RowAgreement 7739 7740 := RowAgreement.single (first := 7739) modelrow_agrees_7740
private def atomic_7741 : CountSegment 1 10 13120 7740 7741 3283 3284 := CountSegment.single (first := 7740) (initial := 3283) row_count_7741 (by decide)
private theorem atomic_7741_agreement : RowAgreement 7740 7741 := RowAgreement.single (first := 7740) modelrow_agrees_7741
private def atomic_7742 : CountSegment 1 10 13120 7741 7742 3284 3285 := CountSegment.single (first := 7741) (initial := 3284) row_count_7742 (by decide)
private theorem atomic_7742_agreement : RowAgreement 7741 7742 := RowAgreement.single (first := 7741) modelrow_agrees_7742
private def atomic_7743 : CountSegment 1 10 13120 7742 7743 3285 3286 := CountSegment.single (first := 7742) (initial := 3285) row_count_7743 (by decide)
private theorem atomic_7743_agreement : RowAgreement 7742 7743 := RowAgreement.single (first := 7742) modelrow_agrees_7743
private def atomic_7744 : CountSegment 1 10 13120 7743 7744 3286 3286 := CountSegment.single (first := 7743) (initial := 3286) row_count_7744 (by decide)
private theorem atomic_7744_agreement : RowAgreement 7743 7744 := RowAgreement.single (first := 7743) modelrow_agrees_7744
private def atomic_7745 : CountSegment 1 10 13120 7744 7745 3286 3287 := CountSegment.single (first := 7744) (initial := 3286) row_count_7745 (by decide)
private theorem atomic_7745_agreement : RowAgreement 7744 7745 := RowAgreement.single (first := 7744) modelrow_agrees_7745
private def atomic_7746 : CountSegment 1 10 13120 7745 7746 3287 3287 := CountSegment.single (first := 7745) (initial := 3287) row_count_7746 (by decide)
private theorem atomic_7746_agreement : RowAgreement 7745 7746 := RowAgreement.single (first := 7745) modelrow_agrees_7746
private def atomic_7747 : CountSegment 1 10 13120 7746 7747 3287 3287 := CountSegment.single (first := 7746) (initial := 3287) row_count_7747 (by decide)
private theorem atomic_7747_agreement : RowAgreement 7746 7747 := RowAgreement.single (first := 7746) modelrow_agrees_7747
private def atomic_7748 : CountSegment 1 10 13120 7747 7748 3287 3287 := CountSegment.single (first := 7747) (initial := 3287) row_count_7748 (by decide)
private theorem atomic_7748_agreement : RowAgreement 7747 7748 := RowAgreement.single (first := 7747) modelrow_agrees_7748
private def atomic_7749 : CountSegment 1 10 13120 7748 7749 3287 3288 := CountSegment.single (first := 7748) (initial := 3287) row_count_7749 (by decide)
private theorem atomic_7749_agreement : RowAgreement 7748 7749 := RowAgreement.single (first := 7748) modelrow_agrees_7749
private def atomic_7750 : CountSegment 1 10 13120 7749 7750 3288 3288 := CountSegment.single (first := 7749) (initial := 3288) row_count_7750 (by decide)
private theorem atomic_7750_agreement : RowAgreement 7749 7750 := RowAgreement.single (first := 7749) modelrow_agrees_7750
private def atomic_7751 : CountSegment 1 10 13120 7750 7751 3288 3289 := CountSegment.single (first := 7750) (initial := 3288) row_count_7751 (by decide)
private theorem atomic_7751_agreement : RowAgreement 7750 7751 := RowAgreement.single (first := 7750) modelrow_agrees_7751
private def atomic_7752 : CountSegment 1 10 13120 7751 7752 3289 3290 := CountSegment.single (first := 7751) (initial := 3289) row_count_7752 (by decide)
private theorem atomic_7752_agreement : RowAgreement 7751 7752 := RowAgreement.single (first := 7751) modelrow_agrees_7752
private def atomic_7753 : CountSegment 1 10 13120 7752 7753 3290 3290 := CountSegment.single (first := 7752) (initial := 3290) row_count_7753 (by decide)
private theorem atomic_7753_agreement : RowAgreement 7752 7753 := RowAgreement.single (first := 7752) modelrow_agrees_7753
private def atomic_7754 : CountSegment 1 10 13120 7753 7754 3290 3290 := CountSegment.single (first := 7753) (initial := 3290) row_count_7754 (by decide)
private theorem atomic_7754_agreement : RowAgreement 7753 7754 := RowAgreement.single (first := 7753) modelrow_agrees_7754
private def atomic_7755 : CountSegment 1 10 13120 7754 7755 3290 3292 := CountSegment.single (first := 7754) (initial := 3290) row_count_7755 (by decide)
private theorem atomic_7755_agreement : RowAgreement 7754 7755 := RowAgreement.single (first := 7754) modelrow_agrees_7755
private def atomic_7756 : CountSegment 1 10 13120 7755 7756 3292 3292 := CountSegment.single (first := 7755) (initial := 3292) row_count_7756 (by decide)
private theorem atomic_7756_agreement : RowAgreement 7755 7756 := RowAgreement.single (first := 7755) modelrow_agrees_7756
private def atomic_7757 : CountSegment 1 10 13120 7756 7757 3292 3292 := CountSegment.single (first := 7756) (initial := 3292) row_count_7757 (by decide)
private theorem atomic_7757_agreement : RowAgreement 7756 7757 := RowAgreement.single (first := 7756) modelrow_agrees_7757
private def atomic_7758 : CountSegment 1 10 13120 7757 7758 3292 3293 := CountSegment.single (first := 7757) (initial := 3292) row_count_7758 (by decide)
private theorem atomic_7758_agreement : RowAgreement 7757 7758 := RowAgreement.single (first := 7757) modelrow_agrees_7758
private def atomic_7759 : CountSegment 1 10 13120 7758 7759 3293 3293 := CountSegment.single (first := 7758) (initial := 3293) row_count_7759 (by decide)
private theorem atomic_7759_agreement : RowAgreement 7758 7759 := RowAgreement.single (first := 7758) modelrow_agrees_7759
private def atomic_7760 : CountSegment 1 10 13120 7759 7760 3293 3294 := CountSegment.single (first := 7759) (initial := 3293) row_count_7760 (by decide)
private theorem atomic_7760_agreement : RowAgreement 7759 7760 := RowAgreement.single (first := 7759) modelrow_agrees_7760
private def atomic_7761 : CountSegment 1 10 13120 7760 7761 3294 3295 := CountSegment.single (first := 7760) (initial := 3294) row_count_7761 (by decide)
private theorem atomic_7761_agreement : RowAgreement 7760 7761 := RowAgreement.single (first := 7760) modelrow_agrees_7761
private def atomic_7762 : CountSegment 1 10 13120 7761 7762 3295 3296 := CountSegment.single (first := 7761) (initial := 3295) row_count_7762 (by decide)
private theorem atomic_7762_agreement : RowAgreement 7761 7762 := RowAgreement.single (first := 7761) modelrow_agrees_7762
private def atomic_7763 : CountSegment 1 10 13120 7762 7763 3296 3296 := CountSegment.single (first := 7762) (initial := 3296) row_count_7763 (by decide)
private theorem atomic_7763_agreement : RowAgreement 7762 7763 := RowAgreement.single (first := 7762) modelrow_agrees_7763
private def atomic_7764 : CountSegment 1 10 13120 7763 7764 3296 3297 := CountSegment.single (first := 7763) (initial := 3296) row_count_7764 (by decide)
private theorem atomic_7764_agreement : RowAgreement 7763 7764 := RowAgreement.single (first := 7763) modelrow_agrees_7764
private def atomic_7765 : CountSegment 1 10 13120 7764 7765 3297 3297 := CountSegment.single (first := 7764) (initial := 3297) row_count_7765 (by decide)
private theorem atomic_7765_agreement : RowAgreement 7764 7765 := RowAgreement.single (first := 7764) modelrow_agrees_7765
private def atomic_7766 : CountSegment 1 10 13120 7765 7766 3297 3297 := CountSegment.single (first := 7765) (initial := 3297) row_count_7766 (by decide)
private theorem atomic_7766_agreement : RowAgreement 7765 7766 := RowAgreement.single (first := 7765) modelrow_agrees_7766
private def atomic_7767 : CountSegment 1 10 13120 7766 7767 3297 3298 := CountSegment.single (first := 7766) (initial := 3297) row_count_7767 (by decide)
private theorem atomic_7767_agreement : RowAgreement 7766 7767 := RowAgreement.single (first := 7766) modelrow_agrees_7767
private def atomic_7768 : CountSegment 1 10 13120 7767 7768 3298 3299 := CountSegment.single (first := 7767) (initial := 3298) row_count_7768 (by decide)
private theorem atomic_7768_agreement : RowAgreement 7767 7768 := RowAgreement.single (first := 7767) modelrow_agrees_7768
private def atomic_7769 : CountSegment 1 10 13120 7768 7769 3299 3299 := CountSegment.single (first := 7768) (initial := 3299) row_count_7769 (by decide)
private theorem atomic_7769_agreement : RowAgreement 7768 7769 := RowAgreement.single (first := 7768) modelrow_agrees_7769
private def atomic_7770 : CountSegment 1 10 13120 7769 7770 3299 3299 := CountSegment.single (first := 7769) (initial := 3299) row_count_7770 (by decide)
private theorem atomic_7770_agreement : RowAgreement 7769 7770 := RowAgreement.single (first := 7769) modelrow_agrees_7770
private def atomic_7771 : CountSegment 1 10 13120 7770 7771 3299 3299 := CountSegment.single (first := 7770) (initial := 3299) row_count_7771 (by decide)
private theorem atomic_7771_agreement : RowAgreement 7770 7771 := RowAgreement.single (first := 7770) modelrow_agrees_7771
private def atomic_7772 : CountSegment 1 10 13120 7771 7772 3299 3299 := CountSegment.single (first := 7771) (initial := 3299) row_count_7772 (by decide)
private theorem atomic_7772_agreement : RowAgreement 7771 7772 := RowAgreement.single (first := 7771) modelrow_agrees_7772
private def atomic_7773 : CountSegment 1 10 13120 7772 7773 3299 3300 := CountSegment.single (first := 7772) (initial := 3299) row_count_7773 (by decide)
private theorem atomic_7773_agreement : RowAgreement 7772 7773 := RowAgreement.single (first := 7772) modelrow_agrees_7773
private def atomic_7774 : CountSegment 1 10 13120 7773 7774 3300 3300 := CountSegment.single (first := 7773) (initial := 3300) row_count_7774 (by decide)
private theorem atomic_7774_agreement : RowAgreement 7773 7774 := RowAgreement.single (first := 7773) modelrow_agrees_7774
private def atomic_7775 : CountSegment 1 10 13120 7774 7775 3300 3301 := CountSegment.single (first := 7774) (initial := 3300) row_count_7775 (by decide)
private theorem atomic_7775_agreement : RowAgreement 7774 7775 := RowAgreement.single (first := 7774) modelrow_agrees_7775
private def atomic_7776 : CountSegment 1 10 13120 7775 7776 3301 3302 := CountSegment.single (first := 7775) (initial := 3301) row_count_7776 (by decide)
private theorem atomic_7776_agreement : RowAgreement 7775 7776 := RowAgreement.single (first := 7775) modelrow_agrees_7776
private def atomic_7777 : CountSegment 1 10 13120 7776 7777 3302 3302 := CountSegment.single (first := 7776) (initial := 3302) row_count_7777 (by decide)
private theorem atomic_7777_agreement : RowAgreement 7776 7777 := RowAgreement.single (first := 7776) modelrow_agrees_7777
private def atomic_7778 : CountSegment 1 10 13120 7777 7778 3302 3302 := CountSegment.single (first := 7777) (initial := 3302) row_count_7778 (by decide)
private theorem atomic_7778_agreement : RowAgreement 7777 7778 := RowAgreement.single (first := 7777) modelrow_agrees_7778
private def atomic_7779 : CountSegment 1 10 13120 7778 7779 3302 3302 := CountSegment.single (first := 7778) (initial := 3302) row_count_7779 (by decide)
private theorem atomic_7779_agreement : RowAgreement 7778 7779 := RowAgreement.single (first := 7778) modelrow_agrees_7779
private def atomic_7780 : CountSegment 1 10 13120 7779 7780 3302 3304 := CountSegment.single (first := 7779) (initial := 3302) row_count_7780 (by decide)
private theorem atomic_7780_agreement : RowAgreement 7779 7780 := RowAgreement.single (first := 7779) modelrow_agrees_7780
private def atomic_7781 : CountSegment 1 10 13120 7780 7781 3304 3304 := CountSegment.single (first := 7780) (initial := 3304) row_count_7781 (by decide)
private theorem atomic_7781_agreement : RowAgreement 7780 7781 := RowAgreement.single (first := 7780) modelrow_agrees_7781
private def atomic_7782 : CountSegment 1 10 13120 7781 7782 3304 3304 := CountSegment.single (first := 7781) (initial := 3304) row_count_7782 (by decide)
private theorem atomic_7782_agreement : RowAgreement 7781 7782 := RowAgreement.single (first := 7781) modelrow_agrees_7782
private def atomic_7783 : CountSegment 1 10 13120 7782 7783 3304 3304 := CountSegment.single (first := 7782) (initial := 3304) row_count_7783 (by decide)
private theorem atomic_7783_agreement : RowAgreement 7782 7783 := RowAgreement.single (first := 7782) modelrow_agrees_7783
private def atomic_7784 : CountSegment 1 10 13120 7783 7784 3304 3304 := CountSegment.single (first := 7783) (initial := 3304) row_count_7784 (by decide)
private theorem atomic_7784_agreement : RowAgreement 7783 7784 := RowAgreement.single (first := 7783) modelrow_agrees_7784
private def atomic_7785 : CountSegment 1 10 13120 7784 7785 3304 3306 := CountSegment.single (first := 7784) (initial := 3304) row_count_7785 (by decide)
private theorem atomic_7785_agreement : RowAgreement 7784 7785 := RowAgreement.single (first := 7784) modelrow_agrees_7785
private def atomic_7786 : CountSegment 1 10 13120 7785 7786 3306 3306 := CountSegment.single (first := 7785) (initial := 3306) row_count_7786 (by decide)
private theorem atomic_7786_agreement : RowAgreement 7785 7786 := RowAgreement.single (first := 7785) modelrow_agrees_7786
private def atomic_7787 : CountSegment 1 10 13120 7786 7787 3306 3306 := CountSegment.single (first := 7786) (initial := 3306) row_count_7787 (by decide)
private theorem atomic_7787_agreement : RowAgreement 7786 7787 := RowAgreement.single (first := 7786) modelrow_agrees_7787
private def atomic_7788 : CountSegment 1 10 13120 7787 7788 3306 3307 := CountSegment.single (first := 7787) (initial := 3306) row_count_7788 (by decide)
private theorem atomic_7788_agreement : RowAgreement 7787 7788 := RowAgreement.single (first := 7787) modelrow_agrees_7788
private def atomic_7789 : CountSegment 1 10 13120 7788 7789 3307 3307 := CountSegment.single (first := 7788) (initial := 3307) row_count_7789 (by decide)
private theorem atomic_7789_agreement : RowAgreement 7788 7789 := RowAgreement.single (first := 7788) modelrow_agrees_7789
private def atomic_7790 : CountSegment 1 10 13120 7789 7790 3307 3308 := CountSegment.single (first := 7789) (initial := 3307) row_count_7790 (by decide)
private theorem atomic_7790_agreement : RowAgreement 7789 7790 := RowAgreement.single (first := 7789) modelrow_agrees_7790
private def atomic_7791 : CountSegment 1 10 13120 7790 7791 3308 3308 := CountSegment.single (first := 7790) (initial := 3308) row_count_7791 (by decide)
private theorem atomic_7791_agreement : RowAgreement 7790 7791 := RowAgreement.single (first := 7790) modelrow_agrees_7791
private def atomic_7792 : CountSegment 1 10 13120 7791 7792 3308 3309 := CountSegment.single (first := 7791) (initial := 3308) row_count_7792 (by decide)
private theorem atomic_7792_agreement : RowAgreement 7791 7792 := RowAgreement.single (first := 7791) modelrow_agrees_7792
private def atomic_7793 : CountSegment 1 10 13120 7792 7793 3309 3310 := CountSegment.single (first := 7792) (initial := 3309) row_count_7793 (by decide)
private theorem atomic_7793_agreement : RowAgreement 7792 7793 := RowAgreement.single (first := 7792) modelrow_agrees_7793
private def atomic_7794 : CountSegment 1 10 13120 7793 7794 3310 3310 := CountSegment.single (first := 7793) (initial := 3310) row_count_7794 (by decide)
private theorem atomic_7794_agreement : RowAgreement 7793 7794 := RowAgreement.single (first := 7793) modelrow_agrees_7794
private def atomic_7795 : CountSegment 1 10 13120 7794 7795 3310 3310 := CountSegment.single (first := 7794) (initial := 3310) row_count_7795 (by decide)
private theorem atomic_7795_agreement : RowAgreement 7794 7795 := RowAgreement.single (first := 7794) modelrow_agrees_7795
private def atomic_7796 : CountSegment 1 10 13120 7795 7796 3310 3310 := CountSegment.single (first := 7795) (initial := 3310) row_count_7796 (by decide)
private theorem atomic_7796_agreement : RowAgreement 7795 7796 := RowAgreement.single (first := 7795) modelrow_agrees_7796
private def atomic_7797 : CountSegment 1 10 13120 7796 7797 3310 3310 := CountSegment.single (first := 7796) (initial := 3310) row_count_7797 (by decide)
private theorem atomic_7797_agreement : RowAgreement 7796 7797 := RowAgreement.single (first := 7796) modelrow_agrees_7797
private def atomic_7798 : CountSegment 1 10 13120 7797 7798 3310 3310 := CountSegment.single (first := 7797) (initial := 3310) row_count_7798 (by decide)
private theorem atomic_7798_agreement : RowAgreement 7797 7798 := RowAgreement.single (first := 7797) modelrow_agrees_7798
private def atomic_7799 : CountSegment 1 10 13120 7798 7799 3310 3310 := CountSegment.single (first := 7798) (initial := 3310) row_count_7799 (by decide)
private theorem atomic_7799_agreement : RowAgreement 7798 7799 := RowAgreement.single (first := 7798) modelrow_agrees_7799
private def atomic_7800 : CountSegment 1 10 13120 7799 7800 3310 3310 := CountSegment.single (first := 7799) (initial := 3310) row_count_7800 (by decide)
private theorem atomic_7800_agreement : RowAgreement 7799 7800 := RowAgreement.single (first := 7799) modelrow_agrees_7800
private def atomic_7801 : CountSegment 1 10 13120 7800 7801 3310 3310 := CountSegment.single (first := 7800) (initial := 3310) row_count_7801 (by decide)
private theorem atomic_7801_agreement : RowAgreement 7800 7801 := RowAgreement.single (first := 7800) modelrow_agrees_7801
private def atomic_7802 : CountSegment 1 10 13120 7801 7802 3310 3310 := CountSegment.single (first := 7801) (initial := 3310) row_count_7802 (by decide)
private theorem atomic_7802_agreement : RowAgreement 7801 7802 := RowAgreement.single (first := 7801) modelrow_agrees_7802
private def atomic_7803 : CountSegment 1 10 13120 7802 7803 3310 3310 := CountSegment.single (first := 7802) (initial := 3310) row_count_7803 (by decide)
private theorem atomic_7803_agreement : RowAgreement 7802 7803 := RowAgreement.single (first := 7802) modelrow_agrees_7803
private def atomic_7804 : CountSegment 1 10 13120 7803 7804 3310 3310 := CountSegment.single (first := 7803) (initial := 3310) row_count_7804 (by decide)
private theorem atomic_7804_agreement : RowAgreement 7803 7804 := RowAgreement.single (first := 7803) modelrow_agrees_7804
private def atomic_7805 : CountSegment 1 10 13120 7804 7805 3310 3310 := CountSegment.single (first := 7804) (initial := 3310) row_count_7805 (by decide)
private theorem atomic_7805_agreement : RowAgreement 7804 7805 := RowAgreement.single (first := 7804) modelrow_agrees_7805
private def atomic_7806 : CountSegment 1 10 13120 7805 7806 3310 3310 := CountSegment.single (first := 7805) (initial := 3310) row_count_7806 (by decide)
private theorem atomic_7806_agreement : RowAgreement 7805 7806 := RowAgreement.single (first := 7805) modelrow_agrees_7806
private def atomic_7807 : CountSegment 1 10 13120 7806 7807 3310 3310 := CountSegment.single (first := 7806) (initial := 3310) row_count_7807 (by decide)
private theorem atomic_7807_agreement : RowAgreement 7806 7807 := RowAgreement.single (first := 7806) modelrow_agrees_7807
private def atomic_7808 : CountSegment 1 10 13120 7807 7808 3310 3310 := CountSegment.single (first := 7807) (initial := 3310) row_count_7808 (by decide)
private theorem atomic_7808_agreement : RowAgreement 7807 7808 := RowAgreement.single (first := 7807) modelrow_agrees_7808
private def within_060_000 : CountSegment 1 10 13120 7680 7682 3261 3261 := CountSegment.join atomic_7681 atomic_7682
private theorem within_060_000_agreement : RowAgreement 7680 7682 := RowAgreement.join atomic_7681_agreement atomic_7682_agreement
private def within_060_001 : CountSegment 1 10 13120 7682 7684 3261 3262 := CountSegment.join atomic_7683 atomic_7684
private theorem within_060_001_agreement : RowAgreement 7682 7684 := RowAgreement.join atomic_7683_agreement atomic_7684_agreement
private def within_060_002 : CountSegment 1 10 13120 7680 7684 3261 3262 := CountSegment.join within_060_000 within_060_001
private theorem within_060_002_agreement : RowAgreement 7680 7684 := RowAgreement.join within_060_000_agreement within_060_001_agreement
private def within_060_003 : CountSegment 1 10 13120 7684 7686 3262 3264 := CountSegment.join atomic_7685 atomic_7686
private theorem within_060_003_agreement : RowAgreement 7684 7686 := RowAgreement.join atomic_7685_agreement atomic_7686_agreement
private def within_060_004 : CountSegment 1 10 13120 7686 7688 3264 3264 := CountSegment.join atomic_7687 atomic_7688
private theorem within_060_004_agreement : RowAgreement 7686 7688 := RowAgreement.join atomic_7687_agreement atomic_7688_agreement
private def within_060_005 : CountSegment 1 10 13120 7684 7688 3262 3264 := CountSegment.join within_060_003 within_060_004
private theorem within_060_005_agreement : RowAgreement 7684 7688 := RowAgreement.join within_060_003_agreement within_060_004_agreement
private def within_060_006 : CountSegment 1 10 13120 7680 7688 3261 3264 := CountSegment.join within_060_002 within_060_005
private theorem within_060_006_agreement : RowAgreement 7680 7688 := RowAgreement.join within_060_002_agreement within_060_005_agreement
private def within_060_007 : CountSegment 1 10 13120 7688 7690 3264 3264 := CountSegment.join atomic_7689 atomic_7690
private theorem within_060_007_agreement : RowAgreement 7688 7690 := RowAgreement.join atomic_7689_agreement atomic_7690_agreement
private def within_060_008 : CountSegment 1 10 13120 7690 7692 3264 3265 := CountSegment.join atomic_7691 atomic_7692
private theorem within_060_008_agreement : RowAgreement 7690 7692 := RowAgreement.join atomic_7691_agreement atomic_7692_agreement
private def within_060_009 : CountSegment 1 10 13120 7688 7692 3264 3265 := CountSegment.join within_060_007 within_060_008
private theorem within_060_009_agreement : RowAgreement 7688 7692 := RowAgreement.join within_060_007_agreement within_060_008_agreement
private def within_060_010 : CountSegment 1 10 13120 7692 7694 3265 3265 := CountSegment.join atomic_7693 atomic_7694
private theorem within_060_010_agreement : RowAgreement 7692 7694 := RowAgreement.join atomic_7693_agreement atomic_7694_agreement
private def within_060_011 : CountSegment 1 10 13120 7694 7696 3265 3266 := CountSegment.join atomic_7695 atomic_7696
private theorem within_060_011_agreement : RowAgreement 7694 7696 := RowAgreement.join atomic_7695_agreement atomic_7696_agreement
private def within_060_012 : CountSegment 1 10 13120 7692 7696 3265 3266 := CountSegment.join within_060_010 within_060_011
private theorem within_060_012_agreement : RowAgreement 7692 7696 := RowAgreement.join within_060_010_agreement within_060_011_agreement
private def within_060_013 : CountSegment 1 10 13120 7688 7696 3264 3266 := CountSegment.join within_060_009 within_060_012
private theorem within_060_013_agreement : RowAgreement 7688 7696 := RowAgreement.join within_060_009_agreement within_060_012_agreement
private def within_060_014 : CountSegment 1 10 13120 7680 7696 3261 3266 := CountSegment.join within_060_006 within_060_013
private theorem within_060_014_agreement : RowAgreement 7680 7696 := RowAgreement.join within_060_006_agreement within_060_013_agreement
private def within_060_015 : CountSegment 1 10 13120 7696 7698 3266 3267 := CountSegment.join atomic_7697 atomic_7698
private theorem within_060_015_agreement : RowAgreement 7696 7698 := RowAgreement.join atomic_7697_agreement atomic_7698_agreement
private def within_060_016 : CountSegment 1 10 13120 7698 7700 3267 3268 := CountSegment.join atomic_7699 atomic_7700
private theorem within_060_016_agreement : RowAgreement 7698 7700 := RowAgreement.join atomic_7699_agreement atomic_7700_agreement
private def within_060_017 : CountSegment 1 10 13120 7696 7700 3266 3268 := CountSegment.join within_060_015 within_060_016
private theorem within_060_017_agreement : RowAgreement 7696 7700 := RowAgreement.join within_060_015_agreement within_060_016_agreement
private def within_060_018 : CountSegment 1 10 13120 7700 7702 3268 3268 := CountSegment.join atomic_7701 atomic_7702
private theorem within_060_018_agreement : RowAgreement 7700 7702 := RowAgreement.join atomic_7701_agreement atomic_7702_agreement
private def within_060_019 : CountSegment 1 10 13120 7702 7704 3268 3270 := CountSegment.join atomic_7703 atomic_7704
private theorem within_060_019_agreement : RowAgreement 7702 7704 := RowAgreement.join atomic_7703_agreement atomic_7704_agreement
private def within_060_020 : CountSegment 1 10 13120 7700 7704 3268 3270 := CountSegment.join within_060_018 within_060_019
private theorem within_060_020_agreement : RowAgreement 7700 7704 := RowAgreement.join within_060_018_agreement within_060_019_agreement
private def within_060_021 : CountSegment 1 10 13120 7696 7704 3266 3270 := CountSegment.join within_060_017 within_060_020
private theorem within_060_021_agreement : RowAgreement 7696 7704 := RowAgreement.join within_060_017_agreement within_060_020_agreement
private def within_060_022 : CountSegment 1 10 13120 7704 7706 3270 3270 := CountSegment.join atomic_7705 atomic_7706
private theorem within_060_022_agreement : RowAgreement 7704 7706 := RowAgreement.join atomic_7705_agreement atomic_7706_agreement
private def within_060_023 : CountSegment 1 10 13120 7706 7708 3270 3271 := CountSegment.join atomic_7707 atomic_7708
private theorem within_060_023_agreement : RowAgreement 7706 7708 := RowAgreement.join atomic_7707_agreement atomic_7708_agreement
private def within_060_024 : CountSegment 1 10 13120 7704 7708 3270 3271 := CountSegment.join within_060_022 within_060_023
private theorem within_060_024_agreement : RowAgreement 7704 7708 := RowAgreement.join within_060_022_agreement within_060_023_agreement
private def within_060_025 : CountSegment 1 10 13120 7708 7710 3271 3271 := CountSegment.join atomic_7709 atomic_7710
private theorem within_060_025_agreement : RowAgreement 7708 7710 := RowAgreement.join atomic_7709_agreement atomic_7710_agreement
private def within_060_026 : CountSegment 1 10 13120 7710 7712 3271 3272 := CountSegment.join atomic_7711 atomic_7712
private theorem within_060_026_agreement : RowAgreement 7710 7712 := RowAgreement.join atomic_7711_agreement atomic_7712_agreement
private def within_060_027 : CountSegment 1 10 13120 7708 7712 3271 3272 := CountSegment.join within_060_025 within_060_026
private theorem within_060_027_agreement : RowAgreement 7708 7712 := RowAgreement.join within_060_025_agreement within_060_026_agreement
private def within_060_028 : CountSegment 1 10 13120 7704 7712 3270 3272 := CountSegment.join within_060_024 within_060_027
private theorem within_060_028_agreement : RowAgreement 7704 7712 := RowAgreement.join within_060_024_agreement within_060_027_agreement
private def within_060_029 : CountSegment 1 10 13120 7696 7712 3266 3272 := CountSegment.join within_060_021 within_060_028
private theorem within_060_029_agreement : RowAgreement 7696 7712 := RowAgreement.join within_060_021_agreement within_060_028_agreement
private def within_060_030 : CountSegment 1 10 13120 7680 7712 3261 3272 := CountSegment.join within_060_014 within_060_029
private theorem within_060_030_agreement : RowAgreement 7680 7712 := RowAgreement.join within_060_014_agreement within_060_029_agreement
private def within_060_031 : CountSegment 1 10 13120 7712 7714 3272 3272 := CountSegment.join atomic_7713 atomic_7714
private theorem within_060_031_agreement : RowAgreement 7712 7714 := RowAgreement.join atomic_7713_agreement atomic_7714_agreement
private def within_060_032 : CountSegment 1 10 13120 7714 7716 3272 3273 := CountSegment.join atomic_7715 atomic_7716
private theorem within_060_032_agreement : RowAgreement 7714 7716 := RowAgreement.join atomic_7715_agreement atomic_7716_agreement
private def within_060_033 : CountSegment 1 10 13120 7712 7716 3272 3273 := CountSegment.join within_060_031 within_060_032
private theorem within_060_033_agreement : RowAgreement 7712 7716 := RowAgreement.join within_060_031_agreement within_060_032_agreement
private def within_060_034 : CountSegment 1 10 13120 7716 7718 3273 3273 := CountSegment.join atomic_7717 atomic_7718
private theorem within_060_034_agreement : RowAgreement 7716 7718 := RowAgreement.join atomic_7717_agreement atomic_7718_agreement
private def within_060_035 : CountSegment 1 10 13120 7718 7720 3273 3274 := CountSegment.join atomic_7719 atomic_7720
private theorem within_060_035_agreement : RowAgreement 7718 7720 := RowAgreement.join atomic_7719_agreement atomic_7720_agreement
private def within_060_036 : CountSegment 1 10 13120 7716 7720 3273 3274 := CountSegment.join within_060_034 within_060_035
private theorem within_060_036_agreement : RowAgreement 7716 7720 := RowAgreement.join within_060_034_agreement within_060_035_agreement
private def within_060_037 : CountSegment 1 10 13120 7712 7720 3272 3274 := CountSegment.join within_060_033 within_060_036
private theorem within_060_037_agreement : RowAgreement 7712 7720 := RowAgreement.join within_060_033_agreement within_060_036_agreement
private def within_060_038 : CountSegment 1 10 13120 7720 7722 3274 3274 := CountSegment.join atomic_7721 atomic_7722
private theorem within_060_038_agreement : RowAgreement 7720 7722 := RowAgreement.join atomic_7721_agreement atomic_7722_agreement
private def within_060_039 : CountSegment 1 10 13120 7722 7724 3274 3274 := CountSegment.join atomic_7723 atomic_7724
private theorem within_060_039_agreement : RowAgreement 7722 7724 := RowAgreement.join atomic_7723_agreement atomic_7724_agreement
private def within_060_040 : CountSegment 1 10 13120 7720 7724 3274 3274 := CountSegment.join within_060_038 within_060_039
private theorem within_060_040_agreement : RowAgreement 7720 7724 := RowAgreement.join within_060_038_agreement within_060_039_agreement
private def within_060_041 : CountSegment 1 10 13120 7724 7726 3274 3276 := CountSegment.join atomic_7725 atomic_7726
private theorem within_060_041_agreement : RowAgreement 7724 7726 := RowAgreement.join atomic_7725_agreement atomic_7726_agreement
private def within_060_042 : CountSegment 1 10 13120 7726 7728 3276 3277 := CountSegment.join atomic_7727 atomic_7728
private theorem within_060_042_agreement : RowAgreement 7726 7728 := RowAgreement.join atomic_7727_agreement atomic_7728_agreement
private def within_060_043 : CountSegment 1 10 13120 7724 7728 3274 3277 := CountSegment.join within_060_041 within_060_042
private theorem within_060_043_agreement : RowAgreement 7724 7728 := RowAgreement.join within_060_041_agreement within_060_042_agreement
private def within_060_044 : CountSegment 1 10 13120 7720 7728 3274 3277 := CountSegment.join within_060_040 within_060_043
private theorem within_060_044_agreement : RowAgreement 7720 7728 := RowAgreement.join within_060_040_agreement within_060_043_agreement
private def within_060_045 : CountSegment 1 10 13120 7712 7728 3272 3277 := CountSegment.join within_060_037 within_060_044
private theorem within_060_045_agreement : RowAgreement 7712 7728 := RowAgreement.join within_060_037_agreement within_060_044_agreement
private def within_060_046 : CountSegment 1 10 13120 7728 7730 3277 3278 := CountSegment.join atomic_7729 atomic_7730
private theorem within_060_046_agreement : RowAgreement 7728 7730 := RowAgreement.join atomic_7729_agreement atomic_7730_agreement
private def within_060_047 : CountSegment 1 10 13120 7730 7732 3278 3279 := CountSegment.join atomic_7731 atomic_7732
private theorem within_060_047_agreement : RowAgreement 7730 7732 := RowAgreement.join atomic_7731_agreement atomic_7732_agreement
private def within_060_048 : CountSegment 1 10 13120 7728 7732 3277 3279 := CountSegment.join within_060_046 within_060_047
private theorem within_060_048_agreement : RowAgreement 7728 7732 := RowAgreement.join within_060_046_agreement within_060_047_agreement
private def within_060_049 : CountSegment 1 10 13120 7732 7734 3279 3280 := CountSegment.join atomic_7733 atomic_7734
private theorem within_060_049_agreement : RowAgreement 7732 7734 := RowAgreement.join atomic_7733_agreement atomic_7734_agreement
private def within_060_050 : CountSegment 1 10 13120 7734 7736 3280 3281 := CountSegment.join atomic_7735 atomic_7736
private theorem within_060_050_agreement : RowAgreement 7734 7736 := RowAgreement.join atomic_7735_agreement atomic_7736_agreement
private def within_060_051 : CountSegment 1 10 13120 7732 7736 3279 3281 := CountSegment.join within_060_049 within_060_050
private theorem within_060_051_agreement : RowAgreement 7732 7736 := RowAgreement.join within_060_049_agreement within_060_050_agreement
private def within_060_052 : CountSegment 1 10 13120 7728 7736 3277 3281 := CountSegment.join within_060_048 within_060_051
private theorem within_060_052_agreement : RowAgreement 7728 7736 := RowAgreement.join within_060_048_agreement within_060_051_agreement
private def within_060_053 : CountSegment 1 10 13120 7736 7738 3281 3282 := CountSegment.join atomic_7737 atomic_7738
private theorem within_060_053_agreement : RowAgreement 7736 7738 := RowAgreement.join atomic_7737_agreement atomic_7738_agreement
private def within_060_054 : CountSegment 1 10 13120 7738 7740 3282 3283 := CountSegment.join atomic_7739 atomic_7740
private theorem within_060_054_agreement : RowAgreement 7738 7740 := RowAgreement.join atomic_7739_agreement atomic_7740_agreement
private def within_060_055 : CountSegment 1 10 13120 7736 7740 3281 3283 := CountSegment.join within_060_053 within_060_054
private theorem within_060_055_agreement : RowAgreement 7736 7740 := RowAgreement.join within_060_053_agreement within_060_054_agreement
private def within_060_056 : CountSegment 1 10 13120 7740 7742 3283 3285 := CountSegment.join atomic_7741 atomic_7742
private theorem within_060_056_agreement : RowAgreement 7740 7742 := RowAgreement.join atomic_7741_agreement atomic_7742_agreement
private def within_060_057 : CountSegment 1 10 13120 7742 7744 3285 3286 := CountSegment.join atomic_7743 atomic_7744
private theorem within_060_057_agreement : RowAgreement 7742 7744 := RowAgreement.join atomic_7743_agreement atomic_7744_agreement
private def within_060_058 : CountSegment 1 10 13120 7740 7744 3283 3286 := CountSegment.join within_060_056 within_060_057
private theorem within_060_058_agreement : RowAgreement 7740 7744 := RowAgreement.join within_060_056_agreement within_060_057_agreement
private def within_060_059 : CountSegment 1 10 13120 7736 7744 3281 3286 := CountSegment.join within_060_055 within_060_058
private theorem within_060_059_agreement : RowAgreement 7736 7744 := RowAgreement.join within_060_055_agreement within_060_058_agreement
private def within_060_060 : CountSegment 1 10 13120 7728 7744 3277 3286 := CountSegment.join within_060_052 within_060_059
private theorem within_060_060_agreement : RowAgreement 7728 7744 := RowAgreement.join within_060_052_agreement within_060_059_agreement
private def within_060_061 : CountSegment 1 10 13120 7712 7744 3272 3286 := CountSegment.join within_060_045 within_060_060
private theorem within_060_061_agreement : RowAgreement 7712 7744 := RowAgreement.join within_060_045_agreement within_060_060_agreement
private def within_060_062 : CountSegment 1 10 13120 7680 7744 3261 3286 := CountSegment.join within_060_030 within_060_061
private theorem within_060_062_agreement : RowAgreement 7680 7744 := RowAgreement.join within_060_030_agreement within_060_061_agreement
private def within_060_063 : CountSegment 1 10 13120 7744 7746 3286 3287 := CountSegment.join atomic_7745 atomic_7746
private theorem within_060_063_agreement : RowAgreement 7744 7746 := RowAgreement.join atomic_7745_agreement atomic_7746_agreement
private def within_060_064 : CountSegment 1 10 13120 7746 7748 3287 3287 := CountSegment.join atomic_7747 atomic_7748
private theorem within_060_064_agreement : RowAgreement 7746 7748 := RowAgreement.join atomic_7747_agreement atomic_7748_agreement
private def within_060_065 : CountSegment 1 10 13120 7744 7748 3286 3287 := CountSegment.join within_060_063 within_060_064
private theorem within_060_065_agreement : RowAgreement 7744 7748 := RowAgreement.join within_060_063_agreement within_060_064_agreement
private def within_060_066 : CountSegment 1 10 13120 7748 7750 3287 3288 := CountSegment.join atomic_7749 atomic_7750
private theorem within_060_066_agreement : RowAgreement 7748 7750 := RowAgreement.join atomic_7749_agreement atomic_7750_agreement
private def within_060_067 : CountSegment 1 10 13120 7750 7752 3288 3290 := CountSegment.join atomic_7751 atomic_7752
private theorem within_060_067_agreement : RowAgreement 7750 7752 := RowAgreement.join atomic_7751_agreement atomic_7752_agreement
private def within_060_068 : CountSegment 1 10 13120 7748 7752 3287 3290 := CountSegment.join within_060_066 within_060_067
private theorem within_060_068_agreement : RowAgreement 7748 7752 := RowAgreement.join within_060_066_agreement within_060_067_agreement
private def within_060_069 : CountSegment 1 10 13120 7744 7752 3286 3290 := CountSegment.join within_060_065 within_060_068
private theorem within_060_069_agreement : RowAgreement 7744 7752 := RowAgreement.join within_060_065_agreement within_060_068_agreement
private def within_060_070 : CountSegment 1 10 13120 7752 7754 3290 3290 := CountSegment.join atomic_7753 atomic_7754
private theorem within_060_070_agreement : RowAgreement 7752 7754 := RowAgreement.join atomic_7753_agreement atomic_7754_agreement
private def within_060_071 : CountSegment 1 10 13120 7754 7756 3290 3292 := CountSegment.join atomic_7755 atomic_7756
private theorem within_060_071_agreement : RowAgreement 7754 7756 := RowAgreement.join atomic_7755_agreement atomic_7756_agreement
private def within_060_072 : CountSegment 1 10 13120 7752 7756 3290 3292 := CountSegment.join within_060_070 within_060_071
private theorem within_060_072_agreement : RowAgreement 7752 7756 := RowAgreement.join within_060_070_agreement within_060_071_agreement
private def within_060_073 : CountSegment 1 10 13120 7756 7758 3292 3293 := CountSegment.join atomic_7757 atomic_7758
private theorem within_060_073_agreement : RowAgreement 7756 7758 := RowAgreement.join atomic_7757_agreement atomic_7758_agreement
private def within_060_074 : CountSegment 1 10 13120 7758 7760 3293 3294 := CountSegment.join atomic_7759 atomic_7760
private theorem within_060_074_agreement : RowAgreement 7758 7760 := RowAgreement.join atomic_7759_agreement atomic_7760_agreement
private def within_060_075 : CountSegment 1 10 13120 7756 7760 3292 3294 := CountSegment.join within_060_073 within_060_074
private theorem within_060_075_agreement : RowAgreement 7756 7760 := RowAgreement.join within_060_073_agreement within_060_074_agreement
private def within_060_076 : CountSegment 1 10 13120 7752 7760 3290 3294 := CountSegment.join within_060_072 within_060_075
private theorem within_060_076_agreement : RowAgreement 7752 7760 := RowAgreement.join within_060_072_agreement within_060_075_agreement
private def within_060_077 : CountSegment 1 10 13120 7744 7760 3286 3294 := CountSegment.join within_060_069 within_060_076
private theorem within_060_077_agreement : RowAgreement 7744 7760 := RowAgreement.join within_060_069_agreement within_060_076_agreement
private def within_060_078 : CountSegment 1 10 13120 7760 7762 3294 3296 := CountSegment.join atomic_7761 atomic_7762
private theorem within_060_078_agreement : RowAgreement 7760 7762 := RowAgreement.join atomic_7761_agreement atomic_7762_agreement
private def within_060_079 : CountSegment 1 10 13120 7762 7764 3296 3297 := CountSegment.join atomic_7763 atomic_7764
private theorem within_060_079_agreement : RowAgreement 7762 7764 := RowAgreement.join atomic_7763_agreement atomic_7764_agreement
private def within_060_080 : CountSegment 1 10 13120 7760 7764 3294 3297 := CountSegment.join within_060_078 within_060_079
private theorem within_060_080_agreement : RowAgreement 7760 7764 := RowAgreement.join within_060_078_agreement within_060_079_agreement
private def within_060_081 : CountSegment 1 10 13120 7764 7766 3297 3297 := CountSegment.join atomic_7765 atomic_7766
private theorem within_060_081_agreement : RowAgreement 7764 7766 := RowAgreement.join atomic_7765_agreement atomic_7766_agreement
private def within_060_082 : CountSegment 1 10 13120 7766 7768 3297 3299 := CountSegment.join atomic_7767 atomic_7768
private theorem within_060_082_agreement : RowAgreement 7766 7768 := RowAgreement.join atomic_7767_agreement atomic_7768_agreement
private def within_060_083 : CountSegment 1 10 13120 7764 7768 3297 3299 := CountSegment.join within_060_081 within_060_082
private theorem within_060_083_agreement : RowAgreement 7764 7768 := RowAgreement.join within_060_081_agreement within_060_082_agreement
private def within_060_084 : CountSegment 1 10 13120 7760 7768 3294 3299 := CountSegment.join within_060_080 within_060_083
private theorem within_060_084_agreement : RowAgreement 7760 7768 := RowAgreement.join within_060_080_agreement within_060_083_agreement
private def within_060_085 : CountSegment 1 10 13120 7768 7770 3299 3299 := CountSegment.join atomic_7769 atomic_7770
private theorem within_060_085_agreement : RowAgreement 7768 7770 := RowAgreement.join atomic_7769_agreement atomic_7770_agreement
private def within_060_086 : CountSegment 1 10 13120 7770 7772 3299 3299 := CountSegment.join atomic_7771 atomic_7772
private theorem within_060_086_agreement : RowAgreement 7770 7772 := RowAgreement.join atomic_7771_agreement atomic_7772_agreement
private def within_060_087 : CountSegment 1 10 13120 7768 7772 3299 3299 := CountSegment.join within_060_085 within_060_086
private theorem within_060_087_agreement : RowAgreement 7768 7772 := RowAgreement.join within_060_085_agreement within_060_086_agreement
private def within_060_088 : CountSegment 1 10 13120 7772 7774 3299 3300 := CountSegment.join atomic_7773 atomic_7774
private theorem within_060_088_agreement : RowAgreement 7772 7774 := RowAgreement.join atomic_7773_agreement atomic_7774_agreement
private def within_060_089 : CountSegment 1 10 13120 7774 7776 3300 3302 := CountSegment.join atomic_7775 atomic_7776
private theorem within_060_089_agreement : RowAgreement 7774 7776 := RowAgreement.join atomic_7775_agreement atomic_7776_agreement
private def within_060_090 : CountSegment 1 10 13120 7772 7776 3299 3302 := CountSegment.join within_060_088 within_060_089
private theorem within_060_090_agreement : RowAgreement 7772 7776 := RowAgreement.join within_060_088_agreement within_060_089_agreement
private def within_060_091 : CountSegment 1 10 13120 7768 7776 3299 3302 := CountSegment.join within_060_087 within_060_090
private theorem within_060_091_agreement : RowAgreement 7768 7776 := RowAgreement.join within_060_087_agreement within_060_090_agreement
private def within_060_092 : CountSegment 1 10 13120 7760 7776 3294 3302 := CountSegment.join within_060_084 within_060_091
private theorem within_060_092_agreement : RowAgreement 7760 7776 := RowAgreement.join within_060_084_agreement within_060_091_agreement
private def within_060_093 : CountSegment 1 10 13120 7744 7776 3286 3302 := CountSegment.join within_060_077 within_060_092
private theorem within_060_093_agreement : RowAgreement 7744 7776 := RowAgreement.join within_060_077_agreement within_060_092_agreement
private def within_060_094 : CountSegment 1 10 13120 7776 7778 3302 3302 := CountSegment.join atomic_7777 atomic_7778
private theorem within_060_094_agreement : RowAgreement 7776 7778 := RowAgreement.join atomic_7777_agreement atomic_7778_agreement
private def within_060_095 : CountSegment 1 10 13120 7778 7780 3302 3304 := CountSegment.join atomic_7779 atomic_7780
private theorem within_060_095_agreement : RowAgreement 7778 7780 := RowAgreement.join atomic_7779_agreement atomic_7780_agreement
private def within_060_096 : CountSegment 1 10 13120 7776 7780 3302 3304 := CountSegment.join within_060_094 within_060_095
private theorem within_060_096_agreement : RowAgreement 7776 7780 := RowAgreement.join within_060_094_agreement within_060_095_agreement
private def within_060_097 : CountSegment 1 10 13120 7780 7782 3304 3304 := CountSegment.join atomic_7781 atomic_7782
private theorem within_060_097_agreement : RowAgreement 7780 7782 := RowAgreement.join atomic_7781_agreement atomic_7782_agreement
private def within_060_098 : CountSegment 1 10 13120 7782 7784 3304 3304 := CountSegment.join atomic_7783 atomic_7784
private theorem within_060_098_agreement : RowAgreement 7782 7784 := RowAgreement.join atomic_7783_agreement atomic_7784_agreement
private def within_060_099 : CountSegment 1 10 13120 7780 7784 3304 3304 := CountSegment.join within_060_097 within_060_098
private theorem within_060_099_agreement : RowAgreement 7780 7784 := RowAgreement.join within_060_097_agreement within_060_098_agreement
private def within_060_100 : CountSegment 1 10 13120 7776 7784 3302 3304 := CountSegment.join within_060_096 within_060_099
private theorem within_060_100_agreement : RowAgreement 7776 7784 := RowAgreement.join within_060_096_agreement within_060_099_agreement
private def within_060_101 : CountSegment 1 10 13120 7784 7786 3304 3306 := CountSegment.join atomic_7785 atomic_7786
private theorem within_060_101_agreement : RowAgreement 7784 7786 := RowAgreement.join atomic_7785_agreement atomic_7786_agreement
private def within_060_102 : CountSegment 1 10 13120 7786 7788 3306 3307 := CountSegment.join atomic_7787 atomic_7788
private theorem within_060_102_agreement : RowAgreement 7786 7788 := RowAgreement.join atomic_7787_agreement atomic_7788_agreement
private def within_060_103 : CountSegment 1 10 13120 7784 7788 3304 3307 := CountSegment.join within_060_101 within_060_102
private theorem within_060_103_agreement : RowAgreement 7784 7788 := RowAgreement.join within_060_101_agreement within_060_102_agreement
private def within_060_104 : CountSegment 1 10 13120 7788 7790 3307 3308 := CountSegment.join atomic_7789 atomic_7790
private theorem within_060_104_agreement : RowAgreement 7788 7790 := RowAgreement.join atomic_7789_agreement atomic_7790_agreement
private def within_060_105 : CountSegment 1 10 13120 7790 7792 3308 3309 := CountSegment.join atomic_7791 atomic_7792
private theorem within_060_105_agreement : RowAgreement 7790 7792 := RowAgreement.join atomic_7791_agreement atomic_7792_agreement
private def within_060_106 : CountSegment 1 10 13120 7788 7792 3307 3309 := CountSegment.join within_060_104 within_060_105
private theorem within_060_106_agreement : RowAgreement 7788 7792 := RowAgreement.join within_060_104_agreement within_060_105_agreement
private def within_060_107 : CountSegment 1 10 13120 7784 7792 3304 3309 := CountSegment.join within_060_103 within_060_106
private theorem within_060_107_agreement : RowAgreement 7784 7792 := RowAgreement.join within_060_103_agreement within_060_106_agreement
private def within_060_108 : CountSegment 1 10 13120 7776 7792 3302 3309 := CountSegment.join within_060_100 within_060_107
private theorem within_060_108_agreement : RowAgreement 7776 7792 := RowAgreement.join within_060_100_agreement within_060_107_agreement
private def within_060_109 : CountSegment 1 10 13120 7792 7794 3309 3310 := CountSegment.join atomic_7793 atomic_7794
private theorem within_060_109_agreement : RowAgreement 7792 7794 := RowAgreement.join atomic_7793_agreement atomic_7794_agreement
private def within_060_110 : CountSegment 1 10 13120 7794 7796 3310 3310 := CountSegment.join atomic_7795 atomic_7796
private theorem within_060_110_agreement : RowAgreement 7794 7796 := RowAgreement.join atomic_7795_agreement atomic_7796_agreement
private def within_060_111 : CountSegment 1 10 13120 7792 7796 3309 3310 := CountSegment.join within_060_109 within_060_110
private theorem within_060_111_agreement : RowAgreement 7792 7796 := RowAgreement.join within_060_109_agreement within_060_110_agreement
private def within_060_112 : CountSegment 1 10 13120 7796 7798 3310 3310 := CountSegment.join atomic_7797 atomic_7798
private theorem within_060_112_agreement : RowAgreement 7796 7798 := RowAgreement.join atomic_7797_agreement atomic_7798_agreement
private def within_060_113 : CountSegment 1 10 13120 7798 7800 3310 3310 := CountSegment.join atomic_7799 atomic_7800
private theorem within_060_113_agreement : RowAgreement 7798 7800 := RowAgreement.join atomic_7799_agreement atomic_7800_agreement
private def within_060_114 : CountSegment 1 10 13120 7796 7800 3310 3310 := CountSegment.join within_060_112 within_060_113
private theorem within_060_114_agreement : RowAgreement 7796 7800 := RowAgreement.join within_060_112_agreement within_060_113_agreement
private def within_060_115 : CountSegment 1 10 13120 7792 7800 3309 3310 := CountSegment.join within_060_111 within_060_114
private theorem within_060_115_agreement : RowAgreement 7792 7800 := RowAgreement.join within_060_111_agreement within_060_114_agreement
private def within_060_116 : CountSegment 1 10 13120 7800 7802 3310 3310 := CountSegment.join atomic_7801 atomic_7802
private theorem within_060_116_agreement : RowAgreement 7800 7802 := RowAgreement.join atomic_7801_agreement atomic_7802_agreement
private def within_060_117 : CountSegment 1 10 13120 7802 7804 3310 3310 := CountSegment.join atomic_7803 atomic_7804
private theorem within_060_117_agreement : RowAgreement 7802 7804 := RowAgreement.join atomic_7803_agreement atomic_7804_agreement
private def within_060_118 : CountSegment 1 10 13120 7800 7804 3310 3310 := CountSegment.join within_060_116 within_060_117
private theorem within_060_118_agreement : RowAgreement 7800 7804 := RowAgreement.join within_060_116_agreement within_060_117_agreement
private def within_060_119 : CountSegment 1 10 13120 7804 7806 3310 3310 := CountSegment.join atomic_7805 atomic_7806
private theorem within_060_119_agreement : RowAgreement 7804 7806 := RowAgreement.join atomic_7805_agreement atomic_7806_agreement
private def within_060_120 : CountSegment 1 10 13120 7806 7808 3310 3310 := CountSegment.join atomic_7807 atomic_7808
private theorem within_060_120_agreement : RowAgreement 7806 7808 := RowAgreement.join atomic_7807_agreement atomic_7808_agreement
private def within_060_121 : CountSegment 1 10 13120 7804 7808 3310 3310 := CountSegment.join within_060_119 within_060_120
private theorem within_060_121_agreement : RowAgreement 7804 7808 := RowAgreement.join within_060_119_agreement within_060_120_agreement
private def within_060_122 : CountSegment 1 10 13120 7800 7808 3310 3310 := CountSegment.join within_060_118 within_060_121
private theorem within_060_122_agreement : RowAgreement 7800 7808 := RowAgreement.join within_060_118_agreement within_060_121_agreement
private def within_060_123 : CountSegment 1 10 13120 7792 7808 3309 3310 := CountSegment.join within_060_115 within_060_122
private theorem within_060_123_agreement : RowAgreement 7792 7808 := RowAgreement.join within_060_115_agreement within_060_122_agreement
private def within_060_124 : CountSegment 1 10 13120 7776 7808 3302 3310 := CountSegment.join within_060_108 within_060_123
private theorem within_060_124_agreement : RowAgreement 7776 7808 := RowAgreement.join within_060_108_agreement within_060_123_agreement
private def within_060_125 : CountSegment 1 10 13120 7744 7808 3286 3310 := CountSegment.join within_060_093 within_060_124
private theorem within_060_125_agreement : RowAgreement 7744 7808 := RowAgreement.join within_060_093_agreement within_060_124_agreement
private def within_060_126 : CountSegment 1 10 13120 7680 7808 3261 3310 := CountSegment.join within_060_062 within_060_125
private theorem within_060_126_agreement : RowAgreement 7680 7808 := RowAgreement.join within_060_062_agreement within_060_125_agreement
def segment_060 : CountSegment 1 10 13120 7680 7808 3261 3310 := within_060_126
theorem agreement_060 : RowAgreement 7680 7808 := within_060_126_agreement
end ProjectEulerSolutions.P58Numerical
