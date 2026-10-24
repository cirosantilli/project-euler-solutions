import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_8705 : RowCertificate 8705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8705 1) := by change ¬ Nat.Prime 303125511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101041837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8705 2) := by change ¬ Nat.Prime 303108101; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3405709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8705 3) := by change Nat.Prime 303090691; exact ProjectEulerSolutions.P58PrimeNodes.p_303090691.prime
    exact decide_eq_true prime
theorem row_count_8705 : row 8705 = 1 := (row_8705.checked).trans rfl
theorem model_count_8705 : modelRow 8705 = 1 := by decide
theorem modelrow_agrees_8705 : modelRow 8705 = row 8705 := model_count_8705.trans row_count_8705.symm
def row_8706 : RowCertificate 8706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8706 1) := by change ¬ Nat.Prime 303195157; exact PrattCertificates.composite_of_product (factor := 2389) (cofactor := 126913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8706 2) := by change ¬ Nat.Prime 303177745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60635549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8706 3) := by change ¬ Nat.Prime 303160333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43308619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8706 : row 8706 = 0 := (row_8706.checked).trans rfl
theorem model_count_8706 : modelRow 8706 = 0 := by decide
theorem modelrow_agrees_8706 : modelRow 8706 = row 8706 := model_count_8706.trans row_count_8706.symm
def row_8707 : RowCertificate 8707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8707 1) := by change ¬ Nat.Prime 303264811; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1931623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8707 2) := by change ¬ Nat.Prime 303247397; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5721649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8707 3) := by change ¬ Nat.Prime 303229983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101076661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8707 : row 8707 = 0 := (row_8707.checked).trans rfl
theorem model_count_8707 : modelRow 8707 = 0 := by decide
theorem modelrow_agrees_8707 : modelRow 8707 = row 8707 := model_count_8707.trans row_count_8707.symm
def row_8708 : RowCertificate 8708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8708 1) := by change ¬ Nat.Prime 303334473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101111491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8708 2) := by change ¬ Nat.Prime 303317057; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 7397977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8708 3) := by change ¬ Nat.Prime 303299641; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 15963139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8708 : row 8708 = 0 := (row_8708.checked).trans rfl
theorem model_count_8708 : modelRow 8708 = 0 := by decide
theorem modelrow_agrees_8708 : modelRow 8708 = row 8708 := model_count_8708.trans row_count_8708.symm
def row_8709 : RowCertificate 8709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8709 1) := by change ¬ Nat.Prime 303404143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43343449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8709 2) := by change ¬ Nat.Prime 303386725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60677345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8709 3) := by change ¬ Nat.Prime 303369307; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2182513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8709 : row 8709 = 0 := (row_8709.checked).trans rfl
theorem model_count_8709 : modelRow 8709 = 0 := by decide
theorem modelrow_agrees_8709 : modelRow 8709 = row 8709 := model_count_8709.trans row_count_8709.symm
def row_8710 : RowCertificate 8710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8710 1) := by change ¬ Nat.Prime 303473821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43353403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8710 2) := by change ¬ Nat.Prime 303456401; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4156937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8710 3) := by change ¬ Nat.Prime 303438981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101146327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8710 : row 8710 = 0 := (row_8710.checked).trans rfl
theorem model_count_8710 : modelRow 8710 = 0 := by decide
theorem modelrow_agrees_8710 : modelRow 8710 = row 8710 := model_count_8710.trans row_count_8710.symm
def row_8711 : RowCertificate 8711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8711 1) := by change ¬ Nat.Prime 303543507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101181169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8711 2) := by change ¬ Nat.Prime 303526085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60705217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8711 3) := by change ¬ Nat.Prime 303508663; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7058341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8711 : row 8711 = 0 := (row_8711.checked).trans rfl
theorem model_count_8711 : modelRow 8711 = 0 := by decide
theorem modelrow_agrees_8711 : modelRow 8711 = row 8711 := model_count_8711.trans row_count_8711.symm
def row_8712 : RowCertificate 8712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8712 1) := by change ¬ Nat.Prime 303613201; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3130033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8712 2) := by change Nat.Prime 303595777; exact ProjectEulerSolutions.P58PrimeNodes.p_303595777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8712 3) := by change ¬ Nat.Prime 303578353; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23352181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8712 : row 8712 = 1 := (row_8712.checked).trans rfl
theorem model_count_8712 : modelRow 8712 = 1 := by decide
theorem modelrow_agrees_8712 : modelRow 8712 = row 8712 := model_count_8712.trans row_count_8712.symm
def row_8713 : RowCertificate 8713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8713 1) := by change ¬ Nat.Prime 303682903; exact PrattCertificates.composite_of_product (factor := 4483) (cofactor := 67741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8713 2) := by change Nat.Prime 303665477; exact ProjectEulerSolutions.P58PrimeNodes.p_303665477.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8713 3) := by change ¬ Nat.Prime 303648051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101216017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8713 : row 8713 = 1 := (row_8713.checked).trans rfl
theorem model_count_8713 : modelRow 8713 = 1 := by decide
theorem modelrow_agrees_8713 : modelRow 8713 = row 8713 := model_count_8713.trans row_count_8713.symm
def row_8714 : RowCertificate 8714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8714 1) := by change ¬ Nat.Prime 303752613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101250871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8714 2) := by change ¬ Nat.Prime 303735185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60747037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8714 3) := by change ¬ Nat.Prime 303717757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43388251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8714 : row 8714 = 0 := (row_8714.checked).trans rfl
theorem model_count_8714 : modelRow 8714 = 0 := by decide
theorem modelrow_agrees_8714 : modelRow 8714 = row 8714 := model_count_8714.trans row_count_8714.symm
def row_8715 : RowCertificate 8715 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8715 1) := by change ¬ Nat.Prime 303822331; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 15990649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8715 2) := by change ¬ Nat.Prime 303804901; exact PrattCertificates.composite_of_product (factor := 1321) (cofactor := 229981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8715 3) := by change ¬ Nat.Prime 303787471; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23368267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8715 : row 8715 = 0 := (row_8715.checked).trans rfl
theorem model_count_8715 : modelRow 8715 = 0 := by decide
theorem modelrow_agrees_8715 : modelRow 8715 = row 8715 := model_count_8715.trans row_count_8715.symm
def row_8716 : RowCertificate 8716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8716 1) := by change ¬ Nat.Prime 303892057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43413151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8716 2) := by change ¬ Nat.Prime 303874625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60774925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8716 3) := by change ¬ Nat.Prime 303857193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101285731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8716 : row 8716 = 0 := (row_8716.checked).trans rfl
theorem model_count_8716 : modelRow 8716 = 0 := by decide
theorem modelrow_agrees_8716 : modelRow 8716 = row 8716 := model_count_8716.trans row_count_8716.symm
def row_8717 : RowCertificate 8717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8717 1) := by change ¬ Nat.Prime 303961791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101320597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8717 2) := by change ¬ Nat.Prime 303944357; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 7413277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8717 3) := by change ¬ Nat.Prime 303926923; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1362901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8717 : row 8717 = 0 := (row_8717.checked).trans rfl
theorem model_count_8717 : modelRow 8717 = 0 := by decide
theorem modelrow_agrees_8717 : modelRow 8717 = row 8717 := model_count_8717.trans row_count_8717.symm
def row_8718 : RowCertificate 8718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8718 1) := by change ¬ Nat.Prime 304031533; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23387041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8718 2) := by change ¬ Nat.Prime 304014097; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 2219081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8718 3) := by change ¬ Nat.Prime 303996661; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3848059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8718 : row 8718 = 0 := (row_8718.checked).trans rfl
theorem model_count_8718 : modelRow 8718 = 0 := by decide
theorem modelrow_agrees_8718 : modelRow 8718 = row 8718 := model_count_8718.trans row_count_8718.symm
def row_8719 : RowCertificate 8719 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8719 1) := by change ¬ Nat.Prime 304101283; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4165771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8719 2) := by change ¬ Nat.Prime 304083845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60816769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8719 3) := by change ¬ Nat.Prime 304066407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101355469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8719 : row 8719 = 0 := (row_8719.checked).trans rfl
theorem model_count_8719 : modelRow 8719 = 0 := by decide
theorem modelrow_agrees_8719 : modelRow 8719 = row 8719 := model_count_8719.trans row_count_8719.symm
def row_8720 : RowCertificate 8720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8720 1) := by change ¬ Nat.Prime 304171041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101390347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8720 2) := by change ¬ Nat.Prime 304153601; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 451937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8720 3) := by change ¬ Nat.Prime 304136161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43448023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8720 : row 8720 = 0 := (row_8720.checked).trans rfl
theorem model_count_8720 : modelRow 8720 = 0 := by decide
theorem modelrow_agrees_8720 : modelRow 8720 = row 8720 := model_count_8720.trans row_count_8720.symm
def row_8721 : RowCertificate 8721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8721 1) := by change ¬ Nat.Prime 304240807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23403139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8721 2) := by change ¬ Nat.Prime 304223365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60844673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8721 3) := by change ¬ Nat.Prime 304205923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43457989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8721 : row 8721 = 0 := (row_8721.checked).trans rfl
theorem model_count_8721 : modelRow 8721 = 0 := by decide
theorem modelrow_agrees_8721 : modelRow 8721 = row 8721 := model_count_8721.trans row_count_8721.symm
def row_8722 : RowCertificate 8722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8722 1) := by change ¬ Nat.Prime 304310581; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 253381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8722 2) := by change ¬ Nat.Prime 304293137; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 743993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8722 3) := by change ¬ Nat.Prime 304275693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101425231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8722 : row 8722 = 0 := (row_8722.checked).trans rfl
theorem model_count_8722 : modelRow 8722 = 0 := by decide
theorem modelrow_agrees_8722 : modelRow 8722 = row 8722 := model_count_8722.trans row_count_8722.symm
def row_8723 : RowCertificate 8723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8723 1) := by change ¬ Nat.Prime 304380363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101460121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8723 2) := by change ¬ Nat.Prime 304362917; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 17903701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8723 3) := by change ¬ Nat.Prime 304345471; exact PrattCertificates.composite_of_product (factor := 7489) (cofactor := 40639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8723 : row 8723 = 0 := (row_8723.checked).trans rfl
theorem model_count_8723 : modelRow 8723 = 0 := by decide
theorem modelrow_agrees_8723 : modelRow 8723 = row 8723 := model_count_8723.trans row_count_8723.symm
def row_8724 : RowCertificate 8724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8724 1) := by change ¬ Nat.Prime 304450153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43492879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8724 2) := by change ¬ Nat.Prime 304432705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60886541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8724 3) := by change ¬ Nat.Prime 304415257; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9819847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8724 : row 8724 = 0 := (row_8724.checked).trans rfl
theorem model_count_8724 : modelRow 8724 = 0 := by decide
theorem modelrow_agrees_8724 : modelRow 8724 = row 8724 := model_count_8724.trans row_count_8724.symm
def row_8725 : RowCertificate 8725 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8725 1) := by change Nat.Prime 304519951; exact ProjectEulerSolutions.P58PrimeNodes.p_304519951.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8725 2) := by change Nat.Prime 304502501; exact ProjectEulerSolutions.P58PrimeNodes.p_304502501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8725 3) := by change ¬ Nat.Prime 304485051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101495017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8725 : row 8725 = 2 := (row_8725.checked).trans rfl
theorem model_count_8725 : modelRow 8725 = 2 := by decide
theorem modelrow_agrees_8725 : modelRow 8725 = row 8725 := model_count_8725.trans row_count_8725.symm
def row_8726 : RowCertificate 8726 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8726 1) := by change ¬ Nat.Prime 304589757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101529919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8726 2) := by change ¬ Nat.Prime 304572305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60914461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8726 3) := by change ¬ Nat.Prime 304554853; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7082671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8726 : row 8726 = 0 := (row_8726.checked).trans rfl
theorem model_count_8726 : modelRow 8726 = 0 := by decide
theorem modelrow_agrees_8726 : modelRow 8726 = row 8726 := model_count_8726.trans row_count_8726.symm
def row_8727 : RowCertificate 8727 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8727 1) := by change ¬ Nat.Prime 304659571; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1578547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8727 2) := by change ¬ Nat.Prime 304642117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23434009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8727 3) := by change ¬ Nat.Prime 304624663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43517809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8727 : row 8727 = 0 := (row_8727.checked).trans rfl
theorem model_count_8727 : modelRow 8727 = 0 := by decide
theorem modelrow_agrees_8727 : modelRow 8727 = row 8727 := model_count_8727.trans row_count_8727.symm
def row_8728 : RowCertificate 8728 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8728 1) := by change Nat.Prime 304729393; exact ProjectEulerSolutions.P58PrimeNodes.p_304729393.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8728 2) := by change ¬ Nat.Prime 304711937; exact PrattCertificates.composite_of_product (factor := 2153) (cofactor := 141529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8728 3) := by change ¬ Nat.Prime 304694481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101564827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8728 : row 8728 = 1 := (row_8728.checked).trans rfl
theorem model_count_8728 : modelRow 8728 = 1 := by decide
theorem modelrow_agrees_8728 : modelRow 8728 = row 8728 := model_count_8728.trans row_count_8728.symm
def row_8729 : RowCertificate 8729 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8729 1) := by change ¬ Nat.Prime 304799223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101599741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8729 2) := by change ¬ Nat.Prime 304781765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60956353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8729 3) := by change ¬ Nat.Prime 304764307; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4548721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8729 : row 8729 = 0 := (row_8729.checked).trans rfl
theorem model_count_8729 : modelRow 8729 = 0 := by decide
theorem modelrow_agrees_8729 : modelRow 8729 = row 8729 := model_count_8729.trans row_count_8729.symm
def row_8730 : RowCertificate 8730 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8730 1) := by change ¬ Nat.Prime 304869061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43552723) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8730 2) := by change ¬ Nat.Prime 304851601; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5751917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8730 3) := by change ¬ Nat.Prime 304834141; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4997281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8730 : row 8730 = 0 := (row_8730.checked).trans rfl
theorem model_count_8730 : modelRow 8730 = 0 := by decide
theorem modelrow_agrees_8730 : modelRow 8730 = row 8730 := model_count_8730.trans row_count_8730.symm
def row_8731 : RowCertificate 8731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8731 1) := by change ¬ Nat.Prime 304938907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43562701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8731 2) := by change ¬ Nat.Prime 304921445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60984289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8731 3) := by change ¬ Nat.Prime 304903983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101634661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8731 : row 8731 = 0 := (row_8731.checked).trans rfl
theorem model_count_8731 : modelRow 8731 = 0 := by decide
theorem modelrow_agrees_8731 : modelRow 8731 = row 8731 := model_count_8731.trans row_count_8731.symm
def row_8732 : RowCertificate 8732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8732 1) := by change ¬ Nat.Prime 305008761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101669587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8732 2) := by change ¬ Nat.Prime 304991297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23460869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8732 3) := by change ¬ Nat.Prime 304973833; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2960911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8732 : row 8732 = 0 := (row_8732.checked).trans rfl
theorem model_count_8732 : modelRow 8732 = 0 := by decide
theorem modelrow_agrees_8732 : modelRow 8732 = row 8732 := model_count_8732.trans row_count_8732.symm
def row_8733 : RowCertificate 8733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8733 1) := by change ¬ Nat.Prime 305078623; exact PrattCertificates.composite_of_product (factor := 2539) (cofactor := 120157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8733 2) := by change Nat.Prime 305061157; exact ProjectEulerSolutions.P58PrimeNodes.p_305061157.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8733 3) := by change Nat.Prime 305043691; exact ProjectEulerSolutions.P58PrimeNodes.p_305043691.prime
    exact decide_eq_true prime
theorem row_count_8733 : row 8733 = 2 := (row_8733.checked).trans rfl
theorem model_count_8733 : modelRow 8733 = 2 := by decide
theorem modelrow_agrees_8733 : modelRow 8733 = row 8733 := model_count_8733.trans row_count_8733.symm
def row_8734 : RowCertificate 8734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8734 1) := by change ¬ Nat.Prime 305148493; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23472961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8734 2) := by change ¬ Nat.Prime 305131025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61026205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8734 3) := by change ¬ Nat.Prime 305113557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101704519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8734 : row 8734 = 0 := (row_8734.checked).trans rfl
theorem model_count_8734 : modelRow 8734 = 0 := by decide
theorem modelrow_agrees_8734 : modelRow 8734 = row 8734 := model_count_8734.trans row_count_8734.symm
def row_8735 : RowCertificate 8735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8735 1) := by change ¬ Nat.Prime 305218371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101739457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8735 2) := by change ¬ Nat.Prime 305200901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10524169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8735 3) := by change ¬ Nat.Prime 305183431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43597633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8735 : row 8735 = 0 := (row_8735.checked).trans rfl
theorem model_count_8735 : modelRow 8735 = 0 := by decide
theorem modelrow_agrees_8735 : modelRow 8735 = row 8735 := model_count_8735.trans row_count_8735.symm
def row_8736 : RowCertificate 8736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8736 1) := by change ¬ Nat.Prime 305288257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16067803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8736 2) := by change ¬ Nat.Prime 305270785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61054157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8736 3) := by change ¬ Nat.Prime 305253313; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2196067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8736 : row 8736 = 0 := (row_8736.checked).trans rfl
theorem model_count_8736 : modelRow 8736 = 0 := by decide
theorem modelrow_agrees_8736 : modelRow 8736 = row 8736 := model_count_8736.trans row_count_8736.symm
def row_8737 : RowCertificate 8737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8737 1) := by change ¬ Nat.Prime 305358151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43622593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8737 2) := by change ¬ Nat.Prime 305340677; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4182749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8737 3) := by change ¬ Nat.Prime 305323203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101774401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8737 : row 8737 = 0 := (row_8737.checked).trans rfl
theorem model_count_8737 : modelRow 8737 = 0 := by decide
theorem modelrow_agrees_8737 : modelRow 8737 = row 8737 := model_count_8737.trans row_count_8737.symm
def row_8738 : RowCertificate 8738 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8738 1) := by change ¬ Nat.Prime 305428053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101809351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8738 2) := by change ¬ Nat.Prime 305410577; exact PrattCertificates.composite_of_product (factor := 2729) (cofactor := 111913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8738 3) := by change ¬ Nat.Prime 305393101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23491777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8738 : row 8738 = 0 := (row_8738.checked).trans rfl
theorem model_count_8738 : modelRow 8738 = 0 := by decide
theorem modelrow_agrees_8738 : modelRow 8738 = row 8738 := model_count_8738.trans row_count_8738.symm
def row_8739 : RowCertificate 8739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8739 1) := by change ¬ Nat.Prime 305497963; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9854773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8739 2) := by change ¬ Nat.Prime 305480485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61096097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8739 3) := by change Nat.Prime 305463007; exact ProjectEulerSolutions.P58PrimeNodes.p_305463007.prime
    exact decide_eq_true prime
theorem row_count_8739 : row 8739 = 1 := (row_8739.checked).trans rfl
theorem model_count_8739 : modelRow 8739 = 1 := by decide
theorem modelrow_agrees_8739 : modelRow 8739 = row 8739 := model_count_8739.trans row_count_8739.symm
def row_8740 : RowCertificate 8740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8740 1) := by change ¬ Nat.Prime 305567881; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 326113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8740 2) := by change ¬ Nat.Prime 305550401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23503877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8740 3) := by change ¬ Nat.Prime 305532921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101844307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8740 : row 8740 = 0 := (row_8740.checked).trans rfl
theorem model_count_8740 : modelRow 8740 = 0 := by decide
theorem modelrow_agrees_8740 : modelRow 8740 = row 8740 := model_count_8740.trans row_count_8740.symm
def row_8741 : RowCertificate 8741 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8741 1) := by change ¬ Nat.Prime 305637807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101879269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8741 2) := by change ¬ Nat.Prime 305620325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61124065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8741 3) := by change ¬ Nat.Prime 305602843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43657549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8741 : row 8741 = 0 := (row_8741.checked).trans rfl
theorem model_count_8741 : modelRow 8741 = 0 := by decide
theorem modelrow_agrees_8741 : modelRow 8741 = row 8741 := model_count_8741.trans row_count_8741.symm
def row_8742 : RowCertificate 8742 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8742 1) := by change Nat.Prime 305707741; exact ProjectEulerSolutions.P58PrimeNodes.p_305707741.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8742 2) := by change ¬ Nat.Prime 305690257; exact PrattCertificates.composite_of_product (factor := 1493) (cofactor := 204749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8742 3) := by change ¬ Nat.Prime 305672773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43667539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8742 : row 8742 = 1 := (row_8742.checked).trans rfl
theorem model_count_8742 : modelRow 8742 = 1 := by decide
theorem modelrow_agrees_8742 : modelRow 8742 = row 8742 := model_count_8742.trans row_count_8742.symm
def row_8743 : RowCertificate 8743 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8743 1) := by change Nat.Prime 305777683; exact ProjectEulerSolutions.P58PrimeNodes.p_305777683.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8743 2) := by change ¬ Nat.Prime 305760197; exact PrattCertificates.composite_of_product (factor := 2549) (cofactor := 119953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8743 3) := by change ¬ Nat.Prime 305742711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101914237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8743 : row 8743 = 1 := (row_8743.checked).trans rfl
theorem model_count_8743 : modelRow 8743 = 1 := by decide
theorem modelrow_agrees_8743 : modelRow 8743 = row 8743 := model_count_8743.trans row_count_8743.symm
def row_8744 : RowCertificate 8744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8744 1) := by change ¬ Nat.Prime 305847633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101949211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8744 2) := by change ¬ Nat.Prime 305830145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61166029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8744 3) := by change ¬ Nat.Prime 305812657; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16095403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8744 : row 8744 = 0 := (row_8744.checked).trans rfl
theorem model_count_8744 : modelRow 8744 = 0 := by decide
theorem modelrow_agrees_8744 : modelRow 8744 = row 8744 := model_count_8744.trans row_count_8744.symm
def row_8745 : RowCertificate 8745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8745 1) := by change ¬ Nat.Prime 305917591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43702513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8745 2) := by change ¬ Nat.Prime 305900101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23530777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8745 3) := by change ¬ Nat.Prime 305882611; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9867181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8745 : row 8745 = 0 := (row_8745.checked).trans rfl
theorem model_count_8745 : modelRow 8745 = 0 := by decide
theorem modelrow_agrees_8745 : modelRow 8745 = row 8745 := model_count_8745.trans row_count_8745.symm
def row_8746 : RowCertificate 8746 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8746 1) := by change ¬ Nat.Prime 305987557; exact PrattCertificates.composite_of_product (factor := 2857) (cofactor := 107101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8746 2) := by change ¬ Nat.Prime 305970065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61194013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8746 3) := by change ¬ Nat.Prime 305952573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101984191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8746 : row 8746 = 0 := (row_8746.checked).trans rfl
theorem model_count_8746 : modelRow 8746 = 0 := by decide
theorem modelrow_agrees_8746 : modelRow 8746 = row 8746 := model_count_8746.trans row_count_8746.symm
def row_8747 : RowCertificate 8747 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8747 1) := by change ¬ Nat.Prime 306057531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102019177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8747 2) := by change ¬ Nat.Prime 306040037; exact PrattCertificates.composite_of_product (factor := 4357) (cofactor := 70241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8747 3) := by change ¬ Nat.Prime 306022543; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5016763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8747 : row 8747 = 0 := (row_8747.checked).trans rfl
theorem model_count_8747 : modelRow 8747 = 0 := by decide
theorem modelrow_agrees_8747 : modelRow 8747 = row 8747 := model_count_8747.trans row_count_8747.symm
def row_8748 : RowCertificate 8748 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8748 1) := by change Nat.Prime 306127513; exact ProjectEulerSolutions.P58PrimeNodes.p_306127513.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8748 2) := by change ¬ Nat.Prime 306110017; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5018197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8748 3) := by change ¬ Nat.Prime 306092521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43727503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8748 : row 8748 = 1 := (row_8748.checked).trans rfl
theorem model_count_8748 : modelRow 8748 = 1 := by decide
theorem modelrow_agrees_8748 : modelRow 8748 = row 8748 := model_count_8748.trans row_count_8748.symm
def row_8749 : RowCertificate 8749 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8749 1) := by change ¬ Nat.Prime 306197503; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1451173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8749 2) := by change ¬ Nat.Prime 306180005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61236001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8749 3) := by change ¬ Nat.Prime 306162507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102054169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8749 : row 8749 = 0 := (row_8749.checked).trans rfl
theorem model_count_8749 : modelRow 8749 = 0 := by decide
theorem modelrow_agrees_8749 : modelRow 8749 = row 8749 := model_count_8749.trans row_count_8749.symm
def row_8750 : RowCertificate 8750 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8750 1) := by change ¬ Nat.Prime 306267501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102089167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8750 2) := by change ¬ Nat.Prime 306250001; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2710177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8750 3) := by change ¬ Nat.Prime 306232501; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1878727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8750 : row 8750 = 0 := (row_8750.checked).trans rfl
theorem model_count_8750 : modelRow 8750 = 0 := by decide
theorem modelrow_agrees_8750 : modelRow 8750 = row 8750 := model_count_8750.trans row_count_8750.symm
def row_8751 : RowCertificate 8751 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8751 1) := by change ¬ Nat.Prime 306337507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43762501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8751 2) := by change ¬ Nat.Prime 306320005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61264001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8751 3) := by change ¬ Nat.Prime 306302503; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23561731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8751 : row 8751 = 0 := (row_8751.checked).trans rfl
theorem model_count_8751 : modelRow 8751 = 0 := by decide
theorem modelrow_agrees_8751 : modelRow 8751 = row 8751 := model_count_8751.trans row_count_8751.symm
def row_8752 : RowCertificate 8752 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8752 1) := by change ¬ Nat.Prime 306407521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43772503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8752 2) := by change ¬ Nat.Prime 306390017; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10565173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8752 3) := by change ¬ Nat.Prime 306372513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102124171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8752 : row 8752 = 0 := (row_8752.checked).trans rfl
theorem model_count_8752 : modelRow 8752 = 0 := by decide
theorem modelrow_agrees_8752 : modelRow 8752 = row 8752 := model_count_8752.trans row_count_8752.symm
def row_8753 : RowCertificate 8753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8753 1) := by change ¬ Nat.Prime 306477543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102159181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8753 2) := by change ¬ Nat.Prime 306460037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23573849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8753 3) := by change Nat.Prime 306442531; exact ProjectEulerSolutions.P58PrimeNodes.p_306442531.prime
    exact decide_eq_true prime
theorem row_count_8753 : row 8753 = 1 := (row_8753.checked).trans rfl
theorem model_count_8753 : modelRow 8753 = 1 := by decide
theorem modelrow_agrees_8753 : modelRow 8753 = row 8753 := model_count_8753.trans row_count_8753.symm
def row_8754 : RowCertificate 8754 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8754 1) := by change ¬ Nat.Prime 306547573; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1693633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8754 2) := by change ¬ Nat.Prime 306530065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61306013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8754 3) := by change ¬ Nat.Prime 306512557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23577889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8754 : row 8754 = 0 := (row_8754.checked).trans rfl
theorem model_count_8754 : modelRow 8754 = 0 := by decide
theorem modelrow_agrees_8754 : modelRow 8754 = row 8754 := model_count_8754.trans row_count_8754.symm
def row_8755 : RowCertificate 8755 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8755 1) := by change ¬ Nat.Prime 306617611; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16137769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8755 2) := by change Nat.Prime 306600101; exact ProjectEulerSolutions.P58PrimeNodes.p_306600101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8755 3) := by change ¬ Nat.Prime 306582591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102194197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8755 : row 8755 = 1 := (row_8755.checked).trans rfl
theorem model_count_8755 : modelRow 8755 = 1 := by decide
theorem modelrow_agrees_8755 : modelRow 8755 = row 8755 := model_count_8755.trans row_count_8755.symm
def row_8756 : RowCertificate 8756 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8756 1) := by change ¬ Nat.Prime 306687657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102229219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8756 2) := by change ¬ Nat.Prime 306670145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61334029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8756 3) := by change ¬ Nat.Prime 306652633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43807519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8756 : row 8756 = 0 := (row_8756.checked).trans rfl
theorem model_count_8756 : modelRow 8756 = 0 := by decide
theorem modelrow_agrees_8756 : modelRow 8756 = row 8756 := model_count_8756.trans row_count_8756.symm
def row_8757 : RowCertificate 8757 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8757 1) := by change ¬ Nat.Prime 306757711; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23596747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8757 2) := by change ¬ Nat.Prime 306740197; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18043541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8757 3) := by change ¬ Nat.Prime 306722683; exact PrattCertificates.composite_of_product (factor := 919) (cofactor := 333757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8757 : row 8757 = 0 := (row_8757.checked).trans rfl
theorem model_count_8757 : modelRow 8757 = 0 := by decide
theorem modelrow_agrees_8757 : modelRow 8757 = row 8757 := model_count_8757.trans row_count_8757.symm
def row_8758 : RowCertificate 8758 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8758 1) := by change ¬ Nat.Prime 306827773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43832539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8758 2) := by change ¬ Nat.Prime 306810257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23600789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8758 3) := by change ¬ Nat.Prime 306792741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102264247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8758 : row 8758 = 0 := (row_8758.checked).trans rfl
theorem model_count_8758 : modelRow 8758 = 0 := by decide
theorem modelrow_agrees_8758 : modelRow 8758 = row 8758 := model_count_8758.trans row_count_8758.symm
def row_8759 : RowCertificate 8759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8759 1) := by change ¬ Nat.Prime 306897843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102299281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8759 2) := by change ¬ Nat.Prime 306880325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61376065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8759 3) := by change ¬ Nat.Prime 306862807; exact PrattCertificates.composite_of_product (factor := 9721) (cofactor := 31567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8759 : row 8759 = 0 := (row_8759.checked).trans rfl
theorem model_count_8759 : modelRow 8759 = 0 := by decide
theorem modelrow_agrees_8759 : modelRow 8759 = row 8759 := model_count_8759.trans row_count_8759.symm
def row_8760 : RowCertificate 8760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8760 1) := by change ¬ Nat.Prime 306967921; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23612917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8760 2) := by change ¬ Nat.Prime 306950401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5791517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8760 3) := by change ¬ Nat.Prime 306932881; exact PrattCertificates.composite_of_product (factor := 3373) (cofactor := 90997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8760 : row 8760 = 0 := (row_8760.checked).trans rfl
theorem model_count_8760 : modelRow 8760 = 0 := by decide
theorem modelrow_agrees_8760 : modelRow 8760 = row 8760 := model_count_8760.trans row_count_8760.symm
def row_8761 : RowCertificate 8761 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8761 1) := by change ¬ Nat.Prime 307038007; exact PrattCertificates.composite_of_product (factor := 1039) (cofactor := 295513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8761 2) := by change ¬ Nat.Prime 307020485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61404097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8761 3) := by change ¬ Nat.Prime 307002963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102334321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8761 : row 8761 = 0 := (row_8761.checked).trans rfl
theorem model_count_8761 : modelRow 8761 = 0 := by decide
theorem modelrow_agrees_8761 : modelRow 8761 = row 8761 := model_count_8761.trans row_count_8761.symm
def row_8762 : RowCertificate 8762 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8762 1) := by change ¬ Nat.Prime 307108101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102369367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8762 2) := by change Nat.Prime 307090577; exact ProjectEulerSolutions.P58PrimeNodes.p_307090577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8762 3) := by change ¬ Nat.Prime 307073053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43867579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8762 : row 8762 = 1 := (row_8762.checked).trans rfl
theorem model_count_8762 : modelRow 8762 = 1 := by decide
theorem modelrow_agrees_8762 : modelRow 8762 = row 8762 := model_count_8762.trans row_count_8762.symm
def row_8763 : RowCertificate 8763 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8763 1) := by change Nat.Prime 307178203; exact ProjectEulerSolutions.P58PrimeNodes.p_307178203.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8763 2) := by change ¬ Nat.Prime 307160677; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1341313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8763 3) := by change ¬ Nat.Prime 307143151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43877593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8763 : row 8763 = 1 := (row_8763.checked).trans rfl
theorem model_count_8763 : modelRow 8763 = 1 := by decide
theorem modelrow_agrees_8763 : modelRow 8763 = row 8763 := model_count_8763.trans row_count_8763.symm
def row_8764 : RowCertificate 8764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8764 1) := by change ¬ Nat.Prime 307248313; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4208881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8764 2) := by change ¬ Nat.Prime 307230785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61446157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8764 3) := by change ¬ Nat.Prime 307213257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102404419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8764 : row 8764 = 0 := (row_8764.checked).trans rfl
theorem model_count_8764 : modelRow 8764 = 0 := by decide
theorem modelrow_agrees_8764 : modelRow 8764 = row 8764 := model_count_8764.trans row_count_8764.symm
def row_8765 : RowCertificate 8765 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8765 1) := by change ¬ Nat.Prime 307318431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102439477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8765 2) := by change ¬ Nat.Prime 307300901; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2719477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8765 3) := by change ¬ Nat.Prime 307283371; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16172809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8765 : row 8765 = 0 := (row_8765.checked).trans rfl
theorem model_count_8765 : modelRow 8765 = 0 := by decide
theorem modelrow_agrees_8765 : modelRow 8765 = row 8765 := model_count_8765.trans row_count_8765.symm
def row_8766 : RowCertificate 8766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8766 1) := by change ¬ Nat.Prime 307388557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43912651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8766 2) := by change ¬ Nat.Prime 307371025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61474205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8766 3) := by change Nat.Prime 307353493; exact ProjectEulerSolutions.P58PrimeNodes.p_307353493.prime
    exact decide_eq_true prime
theorem row_count_8766 : row 8766 = 1 := (row_8766.checked).trans rfl
theorem model_count_8766 : modelRow 8766 = 1 := by decide
theorem modelrow_agrees_8766 : modelRow 8766 = row 8766 := model_count_8766.trans row_count_8766.symm
def row_8767 : RowCertificate 8767 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8767 1) := by change ¬ Nat.Prime 307458691; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 370879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8767 2) := by change ¬ Nat.Prime 307441157; exact PrattCertificates.composite_of_product (factor := 6133) (cofactor := 50129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8767 3) := by change ¬ Nat.Prime 307423623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102474541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8767 : row 8767 = 0 := (row_8767.checked).trans rfl
theorem model_count_8767 : modelRow 8767 = 0 := by decide
theorem modelrow_agrees_8767 : modelRow 8767 = row 8767 := model_count_8767.trans row_count_8767.symm
def row_8768 : RowCertificate 8768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8768 1) := by change ¬ Nat.Prime 307528833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102509611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8768 2) := by change ¬ Nat.Prime 307511297; exact PrattCertificates.composite_of_product (factor := 821) (cofactor := 374557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8768 3) := by change ¬ Nat.Prime 307493761; exact PrattCertificates.composite_of_product (factor := 6067) (cofactor := 50683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8768 : row 8768 = 0 := (row_8768.checked).trans rfl
theorem model_count_8768 : modelRow 8768 = 0 := by decide
theorem modelrow_agrees_8768 : modelRow 8768 = row 8768 := model_count_8768.trans row_count_8768.symm
def row_8769 : RowCertificate 8769 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8769 1) := by change ¬ Nat.Prime 307598983; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1343227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8769 2) := by change ¬ Nat.Prime 307581445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61516289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8769 3) := by change ¬ Nat.Prime 307563907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43937701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8769 : row 8769 = 0 := (row_8769.checked).trans rfl
theorem model_count_8769 : modelRow 8769 = 0 := by decide
theorem modelrow_agrees_8769 : modelRow 8769 = row 8769 := model_count_8769.trans row_count_8769.symm
def row_8770 : RowCertificate 8770 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8770 1) := by change ¬ Nat.Prime 307669141; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23666857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8770 2) := by change ¬ Nat.Prime 307651601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18097153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8770 3) := by change ¬ Nat.Prime 307634061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102544687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8770 : row 8770 = 0 := (row_8770.checked).trans rfl
theorem model_count_8770 : modelRow 8770 = 0 := by decide
theorem modelrow_agrees_8770 : modelRow 8770 = row 8770 := model_count_8770.trans row_count_8770.symm
def row_8771 : RowCertificate 8771 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8771 1) := by change ¬ Nat.Prime 307739307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102579769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8771 2) := by change ¬ Nat.Prime 307721765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61544353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8771 3) := by change Nat.Prime 307704223; exact ProjectEulerSolutions.P58PrimeNodes.p_307704223.prime
    exact decide_eq_true prime
theorem row_count_8771 : row 8771 = 1 := (row_8771.checked).trans rfl
theorem model_count_8771 : modelRow 8771 = 1 := by decide
theorem modelrow_agrees_8771 : modelRow 8771 = row 8771 := model_count_8771.trans row_count_8771.symm
def row_8772 : RowCertificate 8772 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8772 1) := by change ¬ Nat.Prime 307809481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43972783) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8772 2) := by change ¬ Nat.Prime 307791937; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8318701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8772 3) := by change Nat.Prime 307774393; exact ProjectEulerSolutions.P58PrimeNodes.p_307774393.prime
    exact decide_eq_true prime
theorem row_count_8772 : row 8772 = 1 := (row_8772.checked).trans rfl
theorem model_count_8772 : modelRow 8772 = 1 := by decide
theorem modelrow_agrees_8772 : modelRow 8772 = row 8772 := model_count_8772.trans row_count_8772.symm
def row_8773 : RowCertificate 8773 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8773 1) := by change ¬ Nat.Prime 307879663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 43982809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8773 2) := by change ¬ Nat.Prime 307862117; exact PrattCertificates.composite_of_product (factor := 11617) (cofactor := 26501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8773 3) := by change ¬ Nat.Prime 307844571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102614857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8773 : row 8773 = 0 := (row_8773.checked).trans rfl
theorem model_count_8773 : modelRow 8773 = 0 := by decide
theorem modelrow_agrees_8773 : modelRow 8773 = row 8773 := model_count_8773.trans row_count_8773.symm
def row_8774 : RowCertificate 8774 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8774 1) := by change ¬ Nat.Prime 307949853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102649951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8774 2) := by change ¬ Nat.Prime 307932305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61586461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8774 3) := by change Nat.Prime 307914757; exact ProjectEulerSolutions.P58PrimeNodes.p_307914757.prime
    exact decide_eq_true prime
theorem row_count_8774 : row 8774 = 1 := (row_8774.checked).trans rfl
theorem model_count_8774 : modelRow 8774 = 1 := by decide
theorem modelrow_agrees_8774 : modelRow 8774 = row 8774 := model_count_8774.trans row_count_8774.symm
def row_8775 : RowCertificate 8775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8775 1) := by change ¬ Nat.Prime 308020051; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7163257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8775 2) := by change Nat.Prime 308002501; exact ProjectEulerSolutions.P58PrimeNodes.p_308002501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8775 3) := by change ¬ Nat.Prime 307984951; exact PrattCertificates.composite_of_product (factor := 7963) (cofactor := 38677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8775 : row 8775 = 1 := (row_8775.checked).trans rfl
theorem model_count_8775 : modelRow 8775 = 1 := by decide
theorem modelrow_agrees_8775 : modelRow 8775 = row 8775 := model_count_8775.trans row_count_8775.symm
def row_8776 : RowCertificate 8776 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8776 1) := by change Nat.Prime 308090257; exact ProjectEulerSolutions.P58PrimeNodes.p_308090257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8776 2) := by change ¬ Nat.Prime 308072705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61614541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8776 3) := by change ¬ Nat.Prime 308055153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102685051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8776 : row 8776 = 1 := (row_8776.checked).trans rfl
theorem model_count_8776 : modelRow 8776 = 1 := by decide
theorem modelrow_agrees_8776 : modelRow 8776 = row 8776 := model_count_8776.trans row_count_8776.symm
def row_8777 : RowCertificate 8777 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8777 1) := by change ¬ Nat.Prime 308160471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102720157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8777 2) := by change ¬ Nat.Prime 308142917; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 512717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8777 3) := by change ¬ Nat.Prime 308125363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44017909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8777 : row 8777 = 0 := (row_8777.checked).trans rfl
theorem model_count_8777 : modelRow 8777 = 0 := by decide
theorem modelrow_agrees_8777 : modelRow 8777 = row 8777 := model_count_8777.trans row_count_8777.symm
def row_8778 : RowCertificate 8778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8778 1) := by change ¬ Nat.Prime 308230693; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2992531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8778 2) := by change ¬ Nat.Prime 308213137; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1963141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8778 3) := by change ¬ Nat.Prime 308195581; exact PrattCertificates.composite_of_product (factor := 2179) (cofactor := 141439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8778 : row 8778 = 0 := (row_8778.checked).trans rfl
theorem model_count_8778 : modelRow 8778 = 0 := by decide
theorem modelrow_agrees_8778 : modelRow 8778 = row 8778 := model_count_8778.trans row_count_8778.symm
def row_8779 : RowCertificate 8779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8779 1) := by change ¬ Nat.Prime 308300923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44042989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8779 2) := by change ¬ Nat.Prime 308283365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61656673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8779 3) := by change ¬ Nat.Prime 308265807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102755269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8779 : row 8779 = 0 := (row_8779.checked).trans rfl
theorem model_count_8779 : modelRow 8779 = 0 := by decide
theorem modelrow_agrees_8779 : modelRow 8779 = row 8779 := model_count_8779.trans row_count_8779.symm
def row_8780 : RowCertificate 8780 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8780 1) := by change ¬ Nat.Prime 308371161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102790387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8780 2) := by change ¬ Nat.Prime 308353601; exact PrattCertificates.composite_of_product (factor := 5197) (cofactor := 59333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8780 3) := by change ¬ Nat.Prime 308336041; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23718157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8780 : row 8780 = 0 := (row_8780.checked).trans rfl
theorem model_count_8780 : modelRow 8780 = 0 := by decide
theorem modelrow_agrees_8780 : modelRow 8780 = row 8780 := model_count_8780.trans row_count_8780.symm
def row_8781 : RowCertificate 8781 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8781 1) := by change Nat.Prime 308441407; exact ProjectEulerSolutions.P58PrimeNodes.p_308441407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8781 2) := by change ¬ Nat.Prime 308423845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61684769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8781 3) := by change ¬ Nat.Prime 308406283; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3903877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8781 : row 8781 = 1 := (row_8781.checked).trans rfl
theorem model_count_8781 : modelRow 8781 = 1 := by decide
theorem modelrow_agrees_8781 : modelRow 8781 = row 8781 := model_count_8781.trans row_count_8781.symm
def row_8782 : RowCertificate 8782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8782 1) := by change ¬ Nat.Prime 308511661; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8338153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8782 2) := by change ¬ Nat.Prime 308494097; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3054397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8782 3) := by change ¬ Nat.Prime 308476533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102825511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8782 : row 8782 = 0 := (row_8782.checked).trans rfl
theorem model_count_8782 : modelRow 8782 = 0 := by decide
theorem modelrow_agrees_8782 : modelRow 8782 = row 8782 := model_count_8782.trans row_count_8782.symm
def row_8783 : RowCertificate 8783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8783 1) := by change ¬ Nat.Prime 308581923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102860641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8783 2) := by change ¬ Nat.Prime 308564357; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5821969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8783 3) := by change ¬ Nat.Prime 308546791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44078113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8783 : row 8783 = 0 := (row_8783.checked).trans rfl
theorem model_count_8783 : modelRow 8783 = 0 := by decide
theorem modelrow_agrees_8783 : modelRow 8783 = row 8783 := model_count_8783.trans row_count_8783.symm
def row_8784 : RowCertificate 8784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8784 1) := by change ¬ Nat.Prime 308652193; exact PrattCertificates.composite_of_product (factor := 3109) (cofactor := 99277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8784 2) := by change ¬ Nat.Prime 308634625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61726925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8784 3) := by change ¬ Nat.Prime 308617057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44088151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8784 : row 8784 = 0 := (row_8784.checked).trans rfl
theorem model_count_8784 : modelRow 8784 = 0 := by decide
theorem modelrow_agrees_8784 : modelRow 8784 = row 8784 := model_count_8784.trans row_count_8784.symm
def row_8785 : RowCertificate 8785 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8785 1) := by change Nat.Prime 308722471; exact ProjectEulerSolutions.P58PrimeNodes.p_308722471.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8785 2) := by change Nat.Prime 308704901; exact ProjectEulerSolutions.P58PrimeNodes.p_308704901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8785 3) := by change ¬ Nat.Prime 308687331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102895777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8785 : row 8785 = 2 := (row_8785.checked).trans rfl
theorem model_count_8785 : modelRow 8785 = 2 := by decide
theorem modelrow_agrees_8785 : modelRow 8785 = row 8785 := model_count_8785.trans row_count_8785.symm
def row_8786 : RowCertificate 8786 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8786 1) := by change ¬ Nat.Prime 308792757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102930919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8786 2) := by change ¬ Nat.Prime 308775185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61755037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8786 3) := by change ¬ Nat.Prime 308757613; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9959923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8786 : row 8786 = 0 := (row_8786.checked).trans rfl
theorem model_count_8786 : modelRow 8786 = 0 := by decide
theorem modelrow_agrees_8786 : modelRow 8786 = row 8786 := model_count_8786.trans row_count_8786.symm
def row_8787 : RowCertificate 8787 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8787 1) := by change ¬ Nat.Prime 308863051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44123293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8787 2) := by change ¬ Nat.Prime 308845477; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18167381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8787 3) := by change Nat.Prime 308827903; exact ProjectEulerSolutions.P58PrimeNodes.p_308827903.prime
    exact decide_eq_true prime
theorem row_count_8787 : row 8787 = 1 := (row_8787.checked).trans rfl
theorem model_count_8787 : modelRow 8787 = 1 := by decide
theorem modelrow_agrees_8787 : modelRow 8787 = row 8787 := model_count_8787.trans row_count_8787.symm
def row_8788 : RowCertificate 8788 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8788 1) := by change Nat.Prime 308933353; exact ProjectEulerSolutions.P58PrimeNodes.p_308933353.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8788 2) := by change ¬ Nat.Prime 308915777; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1706717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8788 3) := by change ¬ Nat.Prime 308898201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102966067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8788 : row 8788 = 1 := (row_8788.checked).trans rfl
theorem model_count_8788 : modelRow 8788 = 1 := by decide
theorem modelrow_agrees_8788 : modelRow 8788 = row 8788 := model_count_8788.trans row_count_8788.symm
def row_8789 : RowCertificate 8789 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8789 1) := by change ¬ Nat.Prime 309003663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103001221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8789 2) := by change ¬ Nat.Prime 308986085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61797217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8789 3) := by change ¬ Nat.Prime 308968507; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1385509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8789 : row 8789 = 0 := (row_8789.checked).trans rfl
theorem model_count_8789 : modelRow 8789 = 0 := by decide
theorem modelrow_agrees_8789 : modelRow 8789 = row 8789 := model_count_8789.trans row_count_8789.symm
def row_8790 : RowCertificate 8790 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8790 1) := by change ¬ Nat.Prime 309073981; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7187767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8790 2) := by change ¬ Nat.Prime 309056401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 7537961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8790 3) := by change ¬ Nat.Prime 309038821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44148403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8790 : row 8790 = 0 := (row_8790.checked).trans rfl
theorem model_count_8790 : modelRow 8790 = 0 := by decide
theorem modelrow_agrees_8790 : modelRow 8790 = row 8790 := model_count_8790.trans row_count_8790.symm
def row_8791 : RowCertificate 8791 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8791 1) := by change ¬ Nat.Prime 309144307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16270753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8791 2) := by change ¬ Nat.Prime 309126725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61825345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8791 3) := by change ¬ Nat.Prime 309109143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103036381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8791 : row 8791 = 0 := (row_8791.checked).trans rfl
theorem model_count_8791 : modelRow 8791 = 0 := by decide
theorem modelrow_agrees_8791 : modelRow 8791 = row 8791 := model_count_8791.trans row_count_8791.symm
def row_8792 : RowCertificate 8792 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8792 1) := by change ¬ Nat.Prime 309214641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103071547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8792 2) := by change ¬ Nat.Prime 309197057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23784389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8792 3) := by change ¬ Nat.Prime 309179473; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4614619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8792 : row 8792 = 0 := (row_8792.checked).trans rfl
theorem model_count_8792 : modelRow 8792 = 0 := by decide
theorem modelrow_agrees_8792 : modelRow 8792 = row 8792 := model_count_8792.trans row_count_8792.symm
def row_8793 : RowCertificate 8793 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8793 1) := by change ¬ Nat.Prime 309284983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44183569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8793 2) := by change ¬ Nat.Prime 309267397; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10664393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8793 3) := by change ¬ Nat.Prime 309249811; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23788447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8793 : row 8793 = 0 := (row_8793.checked).trans rfl
theorem model_count_8793 : modelRow 8793 = 0 := by decide
theorem modelrow_agrees_8793 : modelRow 8793 = row 8793 := model_count_8793.trans row_count_8793.symm
def row_8794 : RowCertificate 8794 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8794 1) := by change ¬ Nat.Prime 309355333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44193619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8794 2) := by change ¬ Nat.Prime 309337745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61867549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8794 3) := by change ¬ Nat.Prime 309320157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103106719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8794 : row 8794 = 0 := (row_8794.checked).trans rfl
theorem model_count_8794 : modelRow 8794 = 0 := by decide
theorem modelrow_agrees_8794 : modelRow 8794 = row 8794 := model_count_8794.trans row_count_8794.symm
def row_8795 : RowCertificate 8795 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8795 1) := by change ¬ Nat.Prime 309425691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103141897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8795 2) := by change Nat.Prime 309408101; exact ProjectEulerSolutions.P58PrimeNodes.p_309408101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8795 3) := by change Nat.Prime 309390511; exact ProjectEulerSolutions.P58PrimeNodes.p_309390511.prime
    exact decide_eq_true prime
theorem row_count_8795 : row 8795 = 2 := (row_8795.checked).trans rfl
theorem model_count_8795 : modelRow 8795 = 2 := by decide
theorem modelrow_agrees_8795 : modelRow 8795 = row 8795 := model_count_8795.trans row_count_8795.symm
def row_8796 : RowCertificate 8796 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8796 1) := by change ¬ Nat.Prime 309496057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23807389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8796 2) := by change ¬ Nat.Prime 309478465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61895693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8796 3) := by change ¬ Nat.Prime 309460873; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4618819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8796 : row 8796 = 0 := (row_8796.checked).trans rfl
theorem model_count_8796 : modelRow 8796 = 0 := by decide
theorem modelrow_agrees_8796 : modelRow 8796 = row 8796 := model_count_8796.trans row_count_8796.symm
def row_8797 : RowCertificate 8797 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8797 1) := by change ¬ Nat.Prime 309566431; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2840059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8797 2) := by change ¬ Nat.Prime 309548837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23811449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8797 3) := by change ¬ Nat.Prime 309531243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103177081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8797 : row 8797 = 0 := (row_8797.checked).trans rfl
theorem model_count_8797 : modelRow 8797 = 0 := by decide
theorem modelrow_agrees_8797 : modelRow 8797 = row 8797 := model_count_8797.trans row_count_8797.symm
def row_8798 : RowCertificate 8798 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8798 1) := by change ¬ Nat.Prime 309636813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103212271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8798 2) := by change ¬ Nat.Prime 309619217; exact PrattCertificates.composite_of_product (factor := 953) (cofactor := 324889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8798 3) := by change ¬ Nat.Prime 309601621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44228803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8798 : row 8798 = 0 := (row_8798.checked).trans rfl
theorem model_count_8798 : modelRow 8798 = 0 := by decide
theorem modelrow_agrees_8798 : modelRow 8798 = row 8798 := model_count_8798.trans row_count_8798.symm
def row_8799 : RowCertificate 8799 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8799 1) := by change ¬ Nat.Prime 309707203; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23823631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8799 2) := by change ¬ Nat.Prime 309689605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61937921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8799 3) := by change Nat.Prime 309672007; exact ProjectEulerSolutions.P58PrimeNodes.p_309672007.prime
    exact decide_eq_true prime
theorem row_count_8799 : row 8799 = 1 := (row_8799.checked).trans rfl
theorem model_count_8799 : modelRow 8799 = 1 := by decide
theorem modelrow_agrees_8799 : modelRow 8799 = row 8799 := model_count_8799.trans row_count_8799.symm
def row_8800 : RowCertificate 8800 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8800 1) := by change ¬ Nat.Prime 309777601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44253943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8800 2) := by change Nat.Prime 309760001; exact ProjectEulerSolutions.P58PrimeNodes.p_309760001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8800 3) := by change ¬ Nat.Prime 309742401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103247467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8800 : row 8800 = 1 := (row_8800.checked).trans rfl
theorem model_count_8800 : modelRow 8800 = 1 := by decide
theorem modelrow_agrees_8800 : modelRow 8800 = row 8800 := model_count_8800.trans row_count_8800.symm
def row_8801 : RowCertificate 8801 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8801 1) := by change ¬ Nat.Prime 309848007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103282669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8801 2) := by change ¬ Nat.Prime 309830405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 61966081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8801 3) := by change ¬ Nat.Prime 309812803; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16305937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8801 : row 8801 = 0 := (row_8801.checked).trans rfl
theorem model_count_8801 : modelRow 8801 = 0 := by decide
theorem modelrow_agrees_8801 : modelRow 8801 = row 8801 := model_count_8801.trans row_count_8801.symm
def row_8802 : RowCertificate 8802 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8802 1) := by change ¬ Nat.Prime 309918421; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1557379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8802 2) := by change Nat.Prime 309900817; exact ProjectEulerSolutions.P58PrimeNodes.p_309900817.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8802 3) := by change Nat.Prime 309883213; exact ProjectEulerSolutions.P58PrimeNodes.p_309883213.prime
    exact decide_eq_true prime
theorem row_count_8802 : row 8802 = 2 := (row_8802.checked).trans rfl
theorem model_count_8802 : modelRow 8802 = 2 := by decide
theorem modelrow_agrees_8802 : modelRow 8802 = row 8802 := model_count_8802.trans row_count_8802.symm
def row_8803 : RowCertificate 8803 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8803 1) := by change Nat.Prime 309988843; exact ProjectEulerSolutions.P58PrimeNodes.p_309988843.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8803 2) := by change ¬ Nat.Prime 309971237; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8377601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8803 3) := by change ¬ Nat.Prime 309953631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103317877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8803 : row 8803 = 1 := (row_8803.checked).trans rfl
theorem model_count_8803 : modelRow 8803 = 1 := by decide
theorem modelrow_agrees_8803 : modelRow 8803 = row 8803 := model_count_8803.trans row_count_8803.symm
def row_8804 : RowCertificate 8804 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8804 1) := by change ¬ Nat.Prime 310059273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103353091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8804 2) := by change ¬ Nat.Prime 310041665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62008333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8804 3) := by change ¬ Nat.Prime 310024057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44289151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8804 : row 8804 = 0 := (row_8804.checked).trans rfl
theorem model_count_8804 : modelRow 8804 = 0 := by decide
theorem modelrow_agrees_8804 : modelRow 8804 = row 8804 := model_count_8804.trans row_count_8804.symm
def row_8805 : RowCertificate 8805 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8805 1) := by change ¬ Nat.Prime 310129711; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2231149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8805 2) := by change ¬ Nat.Prime 310112101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23854777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8805 3) := by change ¬ Nat.Prime 310094491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44299213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8805 : row 8805 = 0 := (row_8805.checked).trans rfl
theorem model_count_8805 : modelRow 8805 = 0 := by decide
theorem modelrow_agrees_8805 : modelRow 8805 = row 8805 := model_count_8805.trans row_count_8805.symm
def row_8806 : RowCertificate 8806 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8806 1) := by change ¬ Nat.Prime 310200157; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2845873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8806 2) := by change ¬ Nat.Prime 310182545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62036509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8806 3) := by change ¬ Nat.Prime 310164933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103388311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8806 : row 8806 = 0 := (row_8806.checked).trans rfl
theorem model_count_8806 : modelRow 8806 = 0 := by decide
theorem modelrow_agrees_8806 : modelRow 8806 = row 8806 := model_count_8806.trans row_count_8806.symm
def row_8807 : RowCertificate 8807 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8807 1) := by change ¬ Nat.Prime 310270611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103423537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8807 2) := by change Nat.Prime 310252997; exact ProjectEulerSolutions.P58PrimeNodes.p_310252997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8807 3) := by change ¬ Nat.Prime 310235383; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10007593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8807 : row 8807 = 1 := (row_8807.checked).trans rfl
theorem model_count_8807 : modelRow 8807 = 1 := by decide
theorem modelrow_agrees_8807 : modelRow 8807 = row 8807 := model_count_8807.trans row_count_8807.symm
def row_8808 : RowCertificate 8808 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8808 1) := by change ¬ Nat.Prime 310341073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44334439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8808 2) := by change ¬ Nat.Prime 310323457; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18254321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8808 3) := by change ¬ Nat.Prime 310305841; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5086981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8808 : row 8808 = 0 := (row_8808.checked).trans rfl
theorem model_count_8808 : modelRow 8808 = 0 := by decide
theorem modelrow_agrees_8808 : modelRow 8808 = row 8808 := model_count_8808.trans row_count_8808.symm
def row_8809 : RowCertificate 8809 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8809 1) := by change ¬ Nat.Prime 310411543; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23877811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8809 2) := by change ¬ Nat.Prime 310393925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62078785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8809 3) := by change ¬ Nat.Prime 310376307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103458769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8809 : row 8809 = 0 := (row_8809.checked).trans rfl
theorem model_count_8809 : modelRow 8809 = 0 := by decide
theorem modelrow_agrees_8809 : modelRow 8809 = row 8809 := model_count_8809.trans row_count_8809.symm
def row_8810 : RowCertificate 8810 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8810 1) := by change ¬ Nat.Prime 310482021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103494007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8810 2) := by change ¬ Nat.Prime 310464401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23881877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8810 3) := by change ¬ Nat.Prime 310446781; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 832297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8810 : row 8810 = 0 := (row_8810.checked).trans rfl
theorem model_count_8810 : modelRow 8810 = 0 := by decide
theorem modelrow_agrees_8810 : modelRow 8810 = row 8810 := model_count_8810.trans row_count_8810.symm
def row_8811 : RowCertificate 8811 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8811 1) := by change ¬ Nat.Prime 310552507; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8393311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8811 2) := by change ¬ Nat.Prime 310534885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62106977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8811 3) := by change ¬ Nat.Prime 310517263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44359609) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8811 : row 8811 = 0 := (row_8811.checked).trans rfl
theorem model_count_8811 : modelRow 8811 = 0 := by decide
theorem modelrow_agrees_8811 : modelRow 8811 = row 8811 := model_count_8811.trans row_count_8811.symm
def row_8812 : RowCertificate 8812 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8812 1) := by change ¬ Nat.Prime 310623001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23894077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8812 2) := by change ¬ Nat.Prime 310605377; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 774577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8812 3) := by change ¬ Nat.Prime 310587753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103529251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8812 : row 8812 = 0 := (row_8812.checked).trans rfl
theorem model_count_8812 : modelRow 8812 = 0 := by decide
theorem modelrow_agrees_8812 : modelRow 8812 = row 8812 := model_count_8812.trans row_count_8812.symm
def row_8813 : RowCertificate 8813 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8813 1) := by change ¬ Nat.Prime 310693503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103564501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8813 2) := by change ¬ Nat.Prime 310675877; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5861809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8813 3) := by change ¬ Nat.Prime 310658251; exact PrattCertificates.composite_of_product (factor := 4423) (cofactor := 70237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8813 : row 8813 = 0 := (row_8813.checked).trans rfl
theorem model_count_8813 : modelRow 8813 = 0 := by decide
theorem modelrow_agrees_8813 : modelRow 8813 = row 8813 := model_count_8813.trans row_count_8813.symm
def row_8814 : RowCertificate 8814 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8814 1) := by change ¬ Nat.Prime 310764013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44394859) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8814 2) := by change ¬ Nat.Prime 310746385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62149277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8814 3) := by change Nat.Prime 310728757; exact ProjectEulerSolutions.P58PrimeNodes.p_310728757.prime
    exact decide_eq_true prime
theorem row_count_8814 : row 8814 = 1 := (row_8814.checked).trans rfl
theorem model_count_8814 : modelRow 8814 = 1 := by decide
theorem modelrow_agrees_8814 : modelRow 8814 = row 8814 := model_count_8814.trans row_count_8814.symm
def row_8815 : RowCertificate 8815 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8815 1) := by change ¬ Nat.Prime 310834531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44404933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8815 2) := by change ¬ Nat.Prime 310816901; exact PrattCertificates.composite_of_product (factor := 7669) (cofactor := 40529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8815 3) := by change ¬ Nat.Prime 310799271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103599757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8815 : row 8815 = 0 := (row_8815.checked).trans rfl
theorem model_count_8815 : modelRow 8815 = 0 := by decide
theorem modelrow_agrees_8815 : modelRow 8815 = row 8815 := model_count_8815.trans row_count_8815.symm
def row_8816 : RowCertificate 8816 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8816 1) := by change ¬ Nat.Prime 310905057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103635019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8816 2) := by change ¬ Nat.Prime 310887425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62177485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8816 3) := by change ¬ Nat.Prime 310869793; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23913061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8816 : row 8816 = 0 := (row_8816.checked).trans rfl
theorem model_count_8816 : modelRow 8816 = 0 := by decide
theorem modelrow_agrees_8816 : modelRow 8816 = row 8816 := model_count_8816.trans row_count_8816.symm
def row_8817 : RowCertificate 8817 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8817 1) := by change ¬ Nat.Prime 310975591; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2059441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8817 2) := by change Nat.Prime 310957957; exact ProjectEulerSolutions.P58PrimeNodes.p_310957957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8817 3) := by change ¬ Nat.Prime 310940323; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10030333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8817 : row 8817 = 1 := (row_8817.checked).trans rfl
theorem model_count_8817 : modelRow 8817 = 1 := by decide
theorem modelrow_agrees_8817 : modelRow 8817 = row 8817 := model_count_8817.trans row_count_8817.symm
def row_8818 : RowCertificate 8818 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8818 1) := by change ¬ Nat.Prime 311046133; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7233631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8818 2) := by change ¬ Nat.Prime 311028497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23925269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8818 3) := by change ¬ Nat.Prime 311010861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103670287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8818 : row 8818 = 0 := (row_8818.checked).trans rfl
theorem model_count_8818 : modelRow 8818 = 0 := by decide
theorem modelrow_agrees_8818 : modelRow 8818 = row 8818 := model_count_8818.trans row_count_8818.symm
def row_8819 : RowCertificate 8819 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8819 1) := by change ¬ Nat.Prime 311116683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103705561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8819 2) := by change ¬ Nat.Prime 311099045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62219809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8819 3) := by change ¬ Nat.Prime 311081407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44440201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8819 : row 8819 = 0 := (row_8819.checked).trans rfl
theorem model_count_8819 : modelRow 8819 = 0 := by decide
theorem modelrow_agrees_8819 : modelRow 8819 = row 8819 := model_count_8819.trans row_count_8819.symm
def row_8820 : RowCertificate 8820 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8820 1) := by change ¬ Nat.Prime 311187241; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3939079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8820 2) := by change ¬ Nat.Prime 311169601; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5101141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8820 3) := by change ¬ Nat.Prime 311151961; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16376419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8820 : row 8820 = 0 := (row_8820.checked).trans rfl
theorem model_count_8820 : modelRow 8820 = 0 := by decide
theorem modelrow_agrees_8820 : modelRow 8820 = row 8820 := model_count_8820.trans row_count_8820.symm
def row_8821 : RowCertificate 8821 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8821 1) := by change ¬ Nat.Prime 311257807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44465401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8821 2) := by change ¬ Nat.Prime 311240165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62248033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8821 3) := by change ¬ Nat.Prime 311222523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103740841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8821 : row 8821 = 0 := (row_8821.checked).trans rfl
theorem model_count_8821 : modelRow 8821 = 0 := by decide
theorem modelrow_agrees_8821 : modelRow 8821 = row 8821 := model_count_8821.trans row_count_8821.symm
def row_8822 : RowCertificate 8822 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8822 1) := by change ¬ Nat.Prime 311328381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103776127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8822 2) := by change ¬ Nat.Prime 311310737; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10734853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8822 3) := by change ¬ Nat.Prime 311293093; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16383847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8822 : row 8822 = 0 := (row_8822.checked).trans rfl
theorem model_count_8822 : modelRow 8822 = 0 := by decide
theorem modelrow_agrees_8822 : modelRow 8822 = row 8822 := model_count_8822.trans row_count_8822.symm
def row_8823 : RowCertificate 8823 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8823 1) := by change ¬ Nat.Prime 311398963; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 192103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8823 2) := by change ¬ Nat.Prime 311381317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23952409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8823 3) := by change ¬ Nat.Prime 311363671; exact PrattCertificates.composite_of_product (factor := 3541) (cofactor := 87931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8823 : row 8823 = 0 := (row_8823.checked).trans rfl
theorem model_count_8823 : modelRow 8823 = 0 := by decide
theorem modelrow_agrees_8823 : modelRow 8823 = row 8823 := model_count_8823.trans row_count_8823.symm
def row_8824 : RowCertificate 8824 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8824 1) := by change ¬ Nat.Prime 311469553; exact PrattCertificates.composite_of_product (factor := 7687) (cofactor := 40519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8824 2) := by change ¬ Nat.Prime 311451905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62290381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8824 3) := by change ¬ Nat.Prime 311434257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103811419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8824 : row 8824 = 0 := (row_8824.checked).trans rfl
theorem model_count_8824 : modelRow 8824 = 0 := by decide
theorem modelrow_agrees_8824 : modelRow 8824 = row 8824 := model_count_8824.trans row_count_8824.symm
def row_8825 : RowCertificate 8825 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8825 1) := by change ¬ Nat.Prime 311540151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103846717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8825 2) := by change ¬ Nat.Prime 311522501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18324853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8825 3) := by change ¬ Nat.Prime 311504851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44500693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8825 : row 8825 = 0 := (row_8825.checked).trans rfl
theorem model_count_8825 : modelRow 8825 = 0 := by decide
theorem modelrow_agrees_8825 : modelRow 8825 = row 8825 := model_count_8825.trans row_count_8825.symm
def row_8826 : RowCertificate 8826 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8826 1) := by change ¬ Nat.Prime 311610757; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 924661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8826 2) := by change ¬ Nat.Prime 311593105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62318621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8826 3) := by change ¬ Nat.Prime 311575453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44510779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8826 : row 8826 = 0 := (row_8826.checked).trans rfl
theorem model_count_8826 : modelRow 8826 = 0 := by decide
theorem modelrow_agrees_8826 : modelRow 8826 = row 8826 := model_count_8826.trans row_count_8826.symm
def row_8827 : RowCertificate 8827 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8827 1) := by change ¬ Nat.Prime 311681371; exact PrattCertificates.composite_of_product (factor := 1543) (cofactor := 201997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8827 2) := by change Nat.Prime 311663717; exact ProjectEulerSolutions.P58PrimeNodes.p_311663717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8827 3) := by change ¬ Nat.Prime 311646063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103882021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8827 : row 8827 = 1 := (row_8827.checked).trans rfl
theorem model_count_8827 : modelRow 8827 = 1 := by decide
theorem modelrow_agrees_8827 : modelRow 8827 = row 8827 := model_count_8827.trans row_count_8827.symm
def row_8828 : RowCertificate 8828 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8828 1) := by change ¬ Nat.Prime 311751993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103917331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8828 2) := by change ¬ Nat.Prime 311734337; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3502633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8828 3) := by change Nat.Prime 311716681; exact ProjectEulerSolutions.P58PrimeNodes.p_311716681.prime
    exact decide_eq_true prime
theorem row_count_8828 : row 8828 = 1 := (row_8828.checked).trans rfl
theorem model_count_8828 : modelRow 8828 = 1 := by decide
theorem modelrow_agrees_8828 : modelRow 8828 = row 8828 := model_count_8828.trans row_count_8828.symm
def row_8829 : RowCertificate 8829 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8829 1) := by change ¬ Nat.Prime 311822623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44546089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8829 2) := by change ¬ Nat.Prime 311804965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62360993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8829 3) := by change ¬ Nat.Prime 311787307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23983639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8829 : row 8829 = 0 := (row_8829.checked).trans rfl
theorem model_count_8829 : modelRow 8829 = 0 := by decide
theorem modelrow_agrees_8829 : modelRow 8829 = row 8829 := model_count_8829.trans row_count_8829.symm
def row_8830 : RowCertificate 8830 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8830 1) := by change ¬ Nat.Prime 311893261; exact PrattCertificates.composite_of_product (factor := 4657) (cofactor := 66973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8830 2) := by change ¬ Nat.Prime 311875601; exact PrattCertificates.composite_of_product (factor := 701) (cofactor := 444901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8830 3) := by change ¬ Nat.Prime 311857941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103952647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8830 : row 8830 = 0 := (row_8830.checked).trans rfl
theorem model_count_8830 : modelRow 8830 = 0 := by decide
theorem modelrow_agrees_8830 : modelRow 8830 = row 8830 := model_count_8830.trans row_count_8830.symm
def row_8831 : RowCertificate 8831 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8831 1) := by change ¬ Nat.Prime 311963907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103987969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8831 2) := by change ¬ Nat.Prime 311946245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62389249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8831 3) := by change ¬ Nat.Prime 311928583; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 740923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8831 : row 8831 = 0 := (row_8831.checked).trans rfl
theorem model_count_8831 : modelRow 8831 = 0 := by decide
theorem modelrow_agrees_8831 : modelRow 8831 = row 8831 := model_count_8831.trans row_count_8831.symm
def row_8832 : RowCertificate 8832 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8832 1) := by change ¬ Nat.Prime 312034561; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10065631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8832 2) := by change ¬ Nat.Prime 312016897; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 778097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8832 3) := by change ¬ Nat.Prime 311999233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44571319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8832 : row 8832 = 0 := (row_8832.checked).trans rfl
theorem model_count_8832 : modelRow 8832 = 0 := by decide
theorem modelrow_agrees_8832 : modelRow 8832 = row 8832 := model_count_8832.trans row_count_8832.symm
private def atomic_8705 : CountSegment 1 10 13120 8704 8705 3655 3656 := CountSegment.single (first := 8704) (initial := 3655) row_count_8705 (by decide)
private theorem atomic_8705_agreement : RowAgreement 8704 8705 := RowAgreement.single (first := 8704) modelrow_agrees_8705
private def atomic_8706 : CountSegment 1 10 13120 8705 8706 3656 3656 := CountSegment.single (first := 8705) (initial := 3656) row_count_8706 (by decide)
private theorem atomic_8706_agreement : RowAgreement 8705 8706 := RowAgreement.single (first := 8705) modelrow_agrees_8706
private def atomic_8707 : CountSegment 1 10 13120 8706 8707 3656 3656 := CountSegment.single (first := 8706) (initial := 3656) row_count_8707 (by decide)
private theorem atomic_8707_agreement : RowAgreement 8706 8707 := RowAgreement.single (first := 8706) modelrow_agrees_8707
private def atomic_8708 : CountSegment 1 10 13120 8707 8708 3656 3656 := CountSegment.single (first := 8707) (initial := 3656) row_count_8708 (by decide)
private theorem atomic_8708_agreement : RowAgreement 8707 8708 := RowAgreement.single (first := 8707) modelrow_agrees_8708
private def atomic_8709 : CountSegment 1 10 13120 8708 8709 3656 3656 := CountSegment.single (first := 8708) (initial := 3656) row_count_8709 (by decide)
private theorem atomic_8709_agreement : RowAgreement 8708 8709 := RowAgreement.single (first := 8708) modelrow_agrees_8709
private def atomic_8710 : CountSegment 1 10 13120 8709 8710 3656 3656 := CountSegment.single (first := 8709) (initial := 3656) row_count_8710 (by decide)
private theorem atomic_8710_agreement : RowAgreement 8709 8710 := RowAgreement.single (first := 8709) modelrow_agrees_8710
private def atomic_8711 : CountSegment 1 10 13120 8710 8711 3656 3656 := CountSegment.single (first := 8710) (initial := 3656) row_count_8711 (by decide)
private theorem atomic_8711_agreement : RowAgreement 8710 8711 := RowAgreement.single (first := 8710) modelrow_agrees_8711
private def atomic_8712 : CountSegment 1 10 13120 8711 8712 3656 3657 := CountSegment.single (first := 8711) (initial := 3656) row_count_8712 (by decide)
private theorem atomic_8712_agreement : RowAgreement 8711 8712 := RowAgreement.single (first := 8711) modelrow_agrees_8712
private def atomic_8713 : CountSegment 1 10 13120 8712 8713 3657 3658 := CountSegment.single (first := 8712) (initial := 3657) row_count_8713 (by decide)
private theorem atomic_8713_agreement : RowAgreement 8712 8713 := RowAgreement.single (first := 8712) modelrow_agrees_8713
private def atomic_8714 : CountSegment 1 10 13120 8713 8714 3658 3658 := CountSegment.single (first := 8713) (initial := 3658) row_count_8714 (by decide)
private theorem atomic_8714_agreement : RowAgreement 8713 8714 := RowAgreement.single (first := 8713) modelrow_agrees_8714
private def atomic_8715 : CountSegment 1 10 13120 8714 8715 3658 3658 := CountSegment.single (first := 8714) (initial := 3658) row_count_8715 (by decide)
private theorem atomic_8715_agreement : RowAgreement 8714 8715 := RowAgreement.single (first := 8714) modelrow_agrees_8715
private def atomic_8716 : CountSegment 1 10 13120 8715 8716 3658 3658 := CountSegment.single (first := 8715) (initial := 3658) row_count_8716 (by decide)
private theorem atomic_8716_agreement : RowAgreement 8715 8716 := RowAgreement.single (first := 8715) modelrow_agrees_8716
private def atomic_8717 : CountSegment 1 10 13120 8716 8717 3658 3658 := CountSegment.single (first := 8716) (initial := 3658) row_count_8717 (by decide)
private theorem atomic_8717_agreement : RowAgreement 8716 8717 := RowAgreement.single (first := 8716) modelrow_agrees_8717
private def atomic_8718 : CountSegment 1 10 13120 8717 8718 3658 3658 := CountSegment.single (first := 8717) (initial := 3658) row_count_8718 (by decide)
private theorem atomic_8718_agreement : RowAgreement 8717 8718 := RowAgreement.single (first := 8717) modelrow_agrees_8718
private def atomic_8719 : CountSegment 1 10 13120 8718 8719 3658 3658 := CountSegment.single (first := 8718) (initial := 3658) row_count_8719 (by decide)
private theorem atomic_8719_agreement : RowAgreement 8718 8719 := RowAgreement.single (first := 8718) modelrow_agrees_8719
private def atomic_8720 : CountSegment 1 10 13120 8719 8720 3658 3658 := CountSegment.single (first := 8719) (initial := 3658) row_count_8720 (by decide)
private theorem atomic_8720_agreement : RowAgreement 8719 8720 := RowAgreement.single (first := 8719) modelrow_agrees_8720
private def atomic_8721 : CountSegment 1 10 13120 8720 8721 3658 3658 := CountSegment.single (first := 8720) (initial := 3658) row_count_8721 (by decide)
private theorem atomic_8721_agreement : RowAgreement 8720 8721 := RowAgreement.single (first := 8720) modelrow_agrees_8721
private def atomic_8722 : CountSegment 1 10 13120 8721 8722 3658 3658 := CountSegment.single (first := 8721) (initial := 3658) row_count_8722 (by decide)
private theorem atomic_8722_agreement : RowAgreement 8721 8722 := RowAgreement.single (first := 8721) modelrow_agrees_8722
private def atomic_8723 : CountSegment 1 10 13120 8722 8723 3658 3658 := CountSegment.single (first := 8722) (initial := 3658) row_count_8723 (by decide)
private theorem atomic_8723_agreement : RowAgreement 8722 8723 := RowAgreement.single (first := 8722) modelrow_agrees_8723
private def atomic_8724 : CountSegment 1 10 13120 8723 8724 3658 3658 := CountSegment.single (first := 8723) (initial := 3658) row_count_8724 (by decide)
private theorem atomic_8724_agreement : RowAgreement 8723 8724 := RowAgreement.single (first := 8723) modelrow_agrees_8724
private def atomic_8725 : CountSegment 1 10 13120 8724 8725 3658 3660 := CountSegment.single (first := 8724) (initial := 3658) row_count_8725 (by decide)
private theorem atomic_8725_agreement : RowAgreement 8724 8725 := RowAgreement.single (first := 8724) modelrow_agrees_8725
private def atomic_8726 : CountSegment 1 10 13120 8725 8726 3660 3660 := CountSegment.single (first := 8725) (initial := 3660) row_count_8726 (by decide)
private theorem atomic_8726_agreement : RowAgreement 8725 8726 := RowAgreement.single (first := 8725) modelrow_agrees_8726
private def atomic_8727 : CountSegment 1 10 13120 8726 8727 3660 3660 := CountSegment.single (first := 8726) (initial := 3660) row_count_8727 (by decide)
private theorem atomic_8727_agreement : RowAgreement 8726 8727 := RowAgreement.single (first := 8726) modelrow_agrees_8727
private def atomic_8728 : CountSegment 1 10 13120 8727 8728 3660 3661 := CountSegment.single (first := 8727) (initial := 3660) row_count_8728 (by decide)
private theorem atomic_8728_agreement : RowAgreement 8727 8728 := RowAgreement.single (first := 8727) modelrow_agrees_8728
private def atomic_8729 : CountSegment 1 10 13120 8728 8729 3661 3661 := CountSegment.single (first := 8728) (initial := 3661) row_count_8729 (by decide)
private theorem atomic_8729_agreement : RowAgreement 8728 8729 := RowAgreement.single (first := 8728) modelrow_agrees_8729
private def atomic_8730 : CountSegment 1 10 13120 8729 8730 3661 3661 := CountSegment.single (first := 8729) (initial := 3661) row_count_8730 (by decide)
private theorem atomic_8730_agreement : RowAgreement 8729 8730 := RowAgreement.single (first := 8729) modelrow_agrees_8730
private def atomic_8731 : CountSegment 1 10 13120 8730 8731 3661 3661 := CountSegment.single (first := 8730) (initial := 3661) row_count_8731 (by decide)
private theorem atomic_8731_agreement : RowAgreement 8730 8731 := RowAgreement.single (first := 8730) modelrow_agrees_8731
private def atomic_8732 : CountSegment 1 10 13120 8731 8732 3661 3661 := CountSegment.single (first := 8731) (initial := 3661) row_count_8732 (by decide)
private theorem atomic_8732_agreement : RowAgreement 8731 8732 := RowAgreement.single (first := 8731) modelrow_agrees_8732
private def atomic_8733 : CountSegment 1 10 13120 8732 8733 3661 3663 := CountSegment.single (first := 8732) (initial := 3661) row_count_8733 (by decide)
private theorem atomic_8733_agreement : RowAgreement 8732 8733 := RowAgreement.single (first := 8732) modelrow_agrees_8733
private def atomic_8734 : CountSegment 1 10 13120 8733 8734 3663 3663 := CountSegment.single (first := 8733) (initial := 3663) row_count_8734 (by decide)
private theorem atomic_8734_agreement : RowAgreement 8733 8734 := RowAgreement.single (first := 8733) modelrow_agrees_8734
private def atomic_8735 : CountSegment 1 10 13120 8734 8735 3663 3663 := CountSegment.single (first := 8734) (initial := 3663) row_count_8735 (by decide)
private theorem atomic_8735_agreement : RowAgreement 8734 8735 := RowAgreement.single (first := 8734) modelrow_agrees_8735
private def atomic_8736 : CountSegment 1 10 13120 8735 8736 3663 3663 := CountSegment.single (first := 8735) (initial := 3663) row_count_8736 (by decide)
private theorem atomic_8736_agreement : RowAgreement 8735 8736 := RowAgreement.single (first := 8735) modelrow_agrees_8736
private def atomic_8737 : CountSegment 1 10 13120 8736 8737 3663 3663 := CountSegment.single (first := 8736) (initial := 3663) row_count_8737 (by decide)
private theorem atomic_8737_agreement : RowAgreement 8736 8737 := RowAgreement.single (first := 8736) modelrow_agrees_8737
private def atomic_8738 : CountSegment 1 10 13120 8737 8738 3663 3663 := CountSegment.single (first := 8737) (initial := 3663) row_count_8738 (by decide)
private theorem atomic_8738_agreement : RowAgreement 8737 8738 := RowAgreement.single (first := 8737) modelrow_agrees_8738
private def atomic_8739 : CountSegment 1 10 13120 8738 8739 3663 3664 := CountSegment.single (first := 8738) (initial := 3663) row_count_8739 (by decide)
private theorem atomic_8739_agreement : RowAgreement 8738 8739 := RowAgreement.single (first := 8738) modelrow_agrees_8739
private def atomic_8740 : CountSegment 1 10 13120 8739 8740 3664 3664 := CountSegment.single (first := 8739) (initial := 3664) row_count_8740 (by decide)
private theorem atomic_8740_agreement : RowAgreement 8739 8740 := RowAgreement.single (first := 8739) modelrow_agrees_8740
private def atomic_8741 : CountSegment 1 10 13120 8740 8741 3664 3664 := CountSegment.single (first := 8740) (initial := 3664) row_count_8741 (by decide)
private theorem atomic_8741_agreement : RowAgreement 8740 8741 := RowAgreement.single (first := 8740) modelrow_agrees_8741
private def atomic_8742 : CountSegment 1 10 13120 8741 8742 3664 3665 := CountSegment.single (first := 8741) (initial := 3664) row_count_8742 (by decide)
private theorem atomic_8742_agreement : RowAgreement 8741 8742 := RowAgreement.single (first := 8741) modelrow_agrees_8742
private def atomic_8743 : CountSegment 1 10 13120 8742 8743 3665 3666 := CountSegment.single (first := 8742) (initial := 3665) row_count_8743 (by decide)
private theorem atomic_8743_agreement : RowAgreement 8742 8743 := RowAgreement.single (first := 8742) modelrow_agrees_8743
private def atomic_8744 : CountSegment 1 10 13120 8743 8744 3666 3666 := CountSegment.single (first := 8743) (initial := 3666) row_count_8744 (by decide)
private theorem atomic_8744_agreement : RowAgreement 8743 8744 := RowAgreement.single (first := 8743) modelrow_agrees_8744
private def atomic_8745 : CountSegment 1 10 13120 8744 8745 3666 3666 := CountSegment.single (first := 8744) (initial := 3666) row_count_8745 (by decide)
private theorem atomic_8745_agreement : RowAgreement 8744 8745 := RowAgreement.single (first := 8744) modelrow_agrees_8745
private def atomic_8746 : CountSegment 1 10 13120 8745 8746 3666 3666 := CountSegment.single (first := 8745) (initial := 3666) row_count_8746 (by decide)
private theorem atomic_8746_agreement : RowAgreement 8745 8746 := RowAgreement.single (first := 8745) modelrow_agrees_8746
private def atomic_8747 : CountSegment 1 10 13120 8746 8747 3666 3666 := CountSegment.single (first := 8746) (initial := 3666) row_count_8747 (by decide)
private theorem atomic_8747_agreement : RowAgreement 8746 8747 := RowAgreement.single (first := 8746) modelrow_agrees_8747
private def atomic_8748 : CountSegment 1 10 13120 8747 8748 3666 3667 := CountSegment.single (first := 8747) (initial := 3666) row_count_8748 (by decide)
private theorem atomic_8748_agreement : RowAgreement 8747 8748 := RowAgreement.single (first := 8747) modelrow_agrees_8748
private def atomic_8749 : CountSegment 1 10 13120 8748 8749 3667 3667 := CountSegment.single (first := 8748) (initial := 3667) row_count_8749 (by decide)
private theorem atomic_8749_agreement : RowAgreement 8748 8749 := RowAgreement.single (first := 8748) modelrow_agrees_8749
private def atomic_8750 : CountSegment 1 10 13120 8749 8750 3667 3667 := CountSegment.single (first := 8749) (initial := 3667) row_count_8750 (by decide)
private theorem atomic_8750_agreement : RowAgreement 8749 8750 := RowAgreement.single (first := 8749) modelrow_agrees_8750
private def atomic_8751 : CountSegment 1 10 13120 8750 8751 3667 3667 := CountSegment.single (first := 8750) (initial := 3667) row_count_8751 (by decide)
private theorem atomic_8751_agreement : RowAgreement 8750 8751 := RowAgreement.single (first := 8750) modelrow_agrees_8751
private def atomic_8752 : CountSegment 1 10 13120 8751 8752 3667 3667 := CountSegment.single (first := 8751) (initial := 3667) row_count_8752 (by decide)
private theorem atomic_8752_agreement : RowAgreement 8751 8752 := RowAgreement.single (first := 8751) modelrow_agrees_8752
private def atomic_8753 : CountSegment 1 10 13120 8752 8753 3667 3668 := CountSegment.single (first := 8752) (initial := 3667) row_count_8753 (by decide)
private theorem atomic_8753_agreement : RowAgreement 8752 8753 := RowAgreement.single (first := 8752) modelrow_agrees_8753
private def atomic_8754 : CountSegment 1 10 13120 8753 8754 3668 3668 := CountSegment.single (first := 8753) (initial := 3668) row_count_8754 (by decide)
private theorem atomic_8754_agreement : RowAgreement 8753 8754 := RowAgreement.single (first := 8753) modelrow_agrees_8754
private def atomic_8755 : CountSegment 1 10 13120 8754 8755 3668 3669 := CountSegment.single (first := 8754) (initial := 3668) row_count_8755 (by decide)
private theorem atomic_8755_agreement : RowAgreement 8754 8755 := RowAgreement.single (first := 8754) modelrow_agrees_8755
private def atomic_8756 : CountSegment 1 10 13120 8755 8756 3669 3669 := CountSegment.single (first := 8755) (initial := 3669) row_count_8756 (by decide)
private theorem atomic_8756_agreement : RowAgreement 8755 8756 := RowAgreement.single (first := 8755) modelrow_agrees_8756
private def atomic_8757 : CountSegment 1 10 13120 8756 8757 3669 3669 := CountSegment.single (first := 8756) (initial := 3669) row_count_8757 (by decide)
private theorem atomic_8757_agreement : RowAgreement 8756 8757 := RowAgreement.single (first := 8756) modelrow_agrees_8757
private def atomic_8758 : CountSegment 1 10 13120 8757 8758 3669 3669 := CountSegment.single (first := 8757) (initial := 3669) row_count_8758 (by decide)
private theorem atomic_8758_agreement : RowAgreement 8757 8758 := RowAgreement.single (first := 8757) modelrow_agrees_8758
private def atomic_8759 : CountSegment 1 10 13120 8758 8759 3669 3669 := CountSegment.single (first := 8758) (initial := 3669) row_count_8759 (by decide)
private theorem atomic_8759_agreement : RowAgreement 8758 8759 := RowAgreement.single (first := 8758) modelrow_agrees_8759
private def atomic_8760 : CountSegment 1 10 13120 8759 8760 3669 3669 := CountSegment.single (first := 8759) (initial := 3669) row_count_8760 (by decide)
private theorem atomic_8760_agreement : RowAgreement 8759 8760 := RowAgreement.single (first := 8759) modelrow_agrees_8760
private def atomic_8761 : CountSegment 1 10 13120 8760 8761 3669 3669 := CountSegment.single (first := 8760) (initial := 3669) row_count_8761 (by decide)
private theorem atomic_8761_agreement : RowAgreement 8760 8761 := RowAgreement.single (first := 8760) modelrow_agrees_8761
private def atomic_8762 : CountSegment 1 10 13120 8761 8762 3669 3670 := CountSegment.single (first := 8761) (initial := 3669) row_count_8762 (by decide)
private theorem atomic_8762_agreement : RowAgreement 8761 8762 := RowAgreement.single (first := 8761) modelrow_agrees_8762
private def atomic_8763 : CountSegment 1 10 13120 8762 8763 3670 3671 := CountSegment.single (first := 8762) (initial := 3670) row_count_8763 (by decide)
private theorem atomic_8763_agreement : RowAgreement 8762 8763 := RowAgreement.single (first := 8762) modelrow_agrees_8763
private def atomic_8764 : CountSegment 1 10 13120 8763 8764 3671 3671 := CountSegment.single (first := 8763) (initial := 3671) row_count_8764 (by decide)
private theorem atomic_8764_agreement : RowAgreement 8763 8764 := RowAgreement.single (first := 8763) modelrow_agrees_8764
private def atomic_8765 : CountSegment 1 10 13120 8764 8765 3671 3671 := CountSegment.single (first := 8764) (initial := 3671) row_count_8765 (by decide)
private theorem atomic_8765_agreement : RowAgreement 8764 8765 := RowAgreement.single (first := 8764) modelrow_agrees_8765
private def atomic_8766 : CountSegment 1 10 13120 8765 8766 3671 3672 := CountSegment.single (first := 8765) (initial := 3671) row_count_8766 (by decide)
private theorem atomic_8766_agreement : RowAgreement 8765 8766 := RowAgreement.single (first := 8765) modelrow_agrees_8766
private def atomic_8767 : CountSegment 1 10 13120 8766 8767 3672 3672 := CountSegment.single (first := 8766) (initial := 3672) row_count_8767 (by decide)
private theorem atomic_8767_agreement : RowAgreement 8766 8767 := RowAgreement.single (first := 8766) modelrow_agrees_8767
private def atomic_8768 : CountSegment 1 10 13120 8767 8768 3672 3672 := CountSegment.single (first := 8767) (initial := 3672) row_count_8768 (by decide)
private theorem atomic_8768_agreement : RowAgreement 8767 8768 := RowAgreement.single (first := 8767) modelrow_agrees_8768
private def atomic_8769 : CountSegment 1 10 13120 8768 8769 3672 3672 := CountSegment.single (first := 8768) (initial := 3672) row_count_8769 (by decide)
private theorem atomic_8769_agreement : RowAgreement 8768 8769 := RowAgreement.single (first := 8768) modelrow_agrees_8769
private def atomic_8770 : CountSegment 1 10 13120 8769 8770 3672 3672 := CountSegment.single (first := 8769) (initial := 3672) row_count_8770 (by decide)
private theorem atomic_8770_agreement : RowAgreement 8769 8770 := RowAgreement.single (first := 8769) modelrow_agrees_8770
private def atomic_8771 : CountSegment 1 10 13120 8770 8771 3672 3673 := CountSegment.single (first := 8770) (initial := 3672) row_count_8771 (by decide)
private theorem atomic_8771_agreement : RowAgreement 8770 8771 := RowAgreement.single (first := 8770) modelrow_agrees_8771
private def atomic_8772 : CountSegment 1 10 13120 8771 8772 3673 3674 := CountSegment.single (first := 8771) (initial := 3673) row_count_8772 (by decide)
private theorem atomic_8772_agreement : RowAgreement 8771 8772 := RowAgreement.single (first := 8771) modelrow_agrees_8772
private def atomic_8773 : CountSegment 1 10 13120 8772 8773 3674 3674 := CountSegment.single (first := 8772) (initial := 3674) row_count_8773 (by decide)
private theorem atomic_8773_agreement : RowAgreement 8772 8773 := RowAgreement.single (first := 8772) modelrow_agrees_8773
private def atomic_8774 : CountSegment 1 10 13120 8773 8774 3674 3675 := CountSegment.single (first := 8773) (initial := 3674) row_count_8774 (by decide)
private theorem atomic_8774_agreement : RowAgreement 8773 8774 := RowAgreement.single (first := 8773) modelrow_agrees_8774
private def atomic_8775 : CountSegment 1 10 13120 8774 8775 3675 3676 := CountSegment.single (first := 8774) (initial := 3675) row_count_8775 (by decide)
private theorem atomic_8775_agreement : RowAgreement 8774 8775 := RowAgreement.single (first := 8774) modelrow_agrees_8775
private def atomic_8776 : CountSegment 1 10 13120 8775 8776 3676 3677 := CountSegment.single (first := 8775) (initial := 3676) row_count_8776 (by decide)
private theorem atomic_8776_agreement : RowAgreement 8775 8776 := RowAgreement.single (first := 8775) modelrow_agrees_8776
private def atomic_8777 : CountSegment 1 10 13120 8776 8777 3677 3677 := CountSegment.single (first := 8776) (initial := 3677) row_count_8777 (by decide)
private theorem atomic_8777_agreement : RowAgreement 8776 8777 := RowAgreement.single (first := 8776) modelrow_agrees_8777
private def atomic_8778 : CountSegment 1 10 13120 8777 8778 3677 3677 := CountSegment.single (first := 8777) (initial := 3677) row_count_8778 (by decide)
private theorem atomic_8778_agreement : RowAgreement 8777 8778 := RowAgreement.single (first := 8777) modelrow_agrees_8778
private def atomic_8779 : CountSegment 1 10 13120 8778 8779 3677 3677 := CountSegment.single (first := 8778) (initial := 3677) row_count_8779 (by decide)
private theorem atomic_8779_agreement : RowAgreement 8778 8779 := RowAgreement.single (first := 8778) modelrow_agrees_8779
private def atomic_8780 : CountSegment 1 10 13120 8779 8780 3677 3677 := CountSegment.single (first := 8779) (initial := 3677) row_count_8780 (by decide)
private theorem atomic_8780_agreement : RowAgreement 8779 8780 := RowAgreement.single (first := 8779) modelrow_agrees_8780
private def atomic_8781 : CountSegment 1 10 13120 8780 8781 3677 3678 := CountSegment.single (first := 8780) (initial := 3677) row_count_8781 (by decide)
private theorem atomic_8781_agreement : RowAgreement 8780 8781 := RowAgreement.single (first := 8780) modelrow_agrees_8781
private def atomic_8782 : CountSegment 1 10 13120 8781 8782 3678 3678 := CountSegment.single (first := 8781) (initial := 3678) row_count_8782 (by decide)
private theorem atomic_8782_agreement : RowAgreement 8781 8782 := RowAgreement.single (first := 8781) modelrow_agrees_8782
private def atomic_8783 : CountSegment 1 10 13120 8782 8783 3678 3678 := CountSegment.single (first := 8782) (initial := 3678) row_count_8783 (by decide)
private theorem atomic_8783_agreement : RowAgreement 8782 8783 := RowAgreement.single (first := 8782) modelrow_agrees_8783
private def atomic_8784 : CountSegment 1 10 13120 8783 8784 3678 3678 := CountSegment.single (first := 8783) (initial := 3678) row_count_8784 (by decide)
private theorem atomic_8784_agreement : RowAgreement 8783 8784 := RowAgreement.single (first := 8783) modelrow_agrees_8784
private def atomic_8785 : CountSegment 1 10 13120 8784 8785 3678 3680 := CountSegment.single (first := 8784) (initial := 3678) row_count_8785 (by decide)
private theorem atomic_8785_agreement : RowAgreement 8784 8785 := RowAgreement.single (first := 8784) modelrow_agrees_8785
private def atomic_8786 : CountSegment 1 10 13120 8785 8786 3680 3680 := CountSegment.single (first := 8785) (initial := 3680) row_count_8786 (by decide)
private theorem atomic_8786_agreement : RowAgreement 8785 8786 := RowAgreement.single (first := 8785) modelrow_agrees_8786
private def atomic_8787 : CountSegment 1 10 13120 8786 8787 3680 3681 := CountSegment.single (first := 8786) (initial := 3680) row_count_8787 (by decide)
private theorem atomic_8787_agreement : RowAgreement 8786 8787 := RowAgreement.single (first := 8786) modelrow_agrees_8787
private def atomic_8788 : CountSegment 1 10 13120 8787 8788 3681 3682 := CountSegment.single (first := 8787) (initial := 3681) row_count_8788 (by decide)
private theorem atomic_8788_agreement : RowAgreement 8787 8788 := RowAgreement.single (first := 8787) modelrow_agrees_8788
private def atomic_8789 : CountSegment 1 10 13120 8788 8789 3682 3682 := CountSegment.single (first := 8788) (initial := 3682) row_count_8789 (by decide)
private theorem atomic_8789_agreement : RowAgreement 8788 8789 := RowAgreement.single (first := 8788) modelrow_agrees_8789
private def atomic_8790 : CountSegment 1 10 13120 8789 8790 3682 3682 := CountSegment.single (first := 8789) (initial := 3682) row_count_8790 (by decide)
private theorem atomic_8790_agreement : RowAgreement 8789 8790 := RowAgreement.single (first := 8789) modelrow_agrees_8790
private def atomic_8791 : CountSegment 1 10 13120 8790 8791 3682 3682 := CountSegment.single (first := 8790) (initial := 3682) row_count_8791 (by decide)
private theorem atomic_8791_agreement : RowAgreement 8790 8791 := RowAgreement.single (first := 8790) modelrow_agrees_8791
private def atomic_8792 : CountSegment 1 10 13120 8791 8792 3682 3682 := CountSegment.single (first := 8791) (initial := 3682) row_count_8792 (by decide)
private theorem atomic_8792_agreement : RowAgreement 8791 8792 := RowAgreement.single (first := 8791) modelrow_agrees_8792
private def atomic_8793 : CountSegment 1 10 13120 8792 8793 3682 3682 := CountSegment.single (first := 8792) (initial := 3682) row_count_8793 (by decide)
private theorem atomic_8793_agreement : RowAgreement 8792 8793 := RowAgreement.single (first := 8792) modelrow_agrees_8793
private def atomic_8794 : CountSegment 1 10 13120 8793 8794 3682 3682 := CountSegment.single (first := 8793) (initial := 3682) row_count_8794 (by decide)
private theorem atomic_8794_agreement : RowAgreement 8793 8794 := RowAgreement.single (first := 8793) modelrow_agrees_8794
private def atomic_8795 : CountSegment 1 10 13120 8794 8795 3682 3684 := CountSegment.single (first := 8794) (initial := 3682) row_count_8795 (by decide)
private theorem atomic_8795_agreement : RowAgreement 8794 8795 := RowAgreement.single (first := 8794) modelrow_agrees_8795
private def atomic_8796 : CountSegment 1 10 13120 8795 8796 3684 3684 := CountSegment.single (first := 8795) (initial := 3684) row_count_8796 (by decide)
private theorem atomic_8796_agreement : RowAgreement 8795 8796 := RowAgreement.single (first := 8795) modelrow_agrees_8796
private def atomic_8797 : CountSegment 1 10 13120 8796 8797 3684 3684 := CountSegment.single (first := 8796) (initial := 3684) row_count_8797 (by decide)
private theorem atomic_8797_agreement : RowAgreement 8796 8797 := RowAgreement.single (first := 8796) modelrow_agrees_8797
private def atomic_8798 : CountSegment 1 10 13120 8797 8798 3684 3684 := CountSegment.single (first := 8797) (initial := 3684) row_count_8798 (by decide)
private theorem atomic_8798_agreement : RowAgreement 8797 8798 := RowAgreement.single (first := 8797) modelrow_agrees_8798
private def atomic_8799 : CountSegment 1 10 13120 8798 8799 3684 3685 := CountSegment.single (first := 8798) (initial := 3684) row_count_8799 (by decide)
private theorem atomic_8799_agreement : RowAgreement 8798 8799 := RowAgreement.single (first := 8798) modelrow_agrees_8799
private def atomic_8800 : CountSegment 1 10 13120 8799 8800 3685 3686 := CountSegment.single (first := 8799) (initial := 3685) row_count_8800 (by decide)
private theorem atomic_8800_agreement : RowAgreement 8799 8800 := RowAgreement.single (first := 8799) modelrow_agrees_8800
private def atomic_8801 : CountSegment 1 10 13120 8800 8801 3686 3686 := CountSegment.single (first := 8800) (initial := 3686) row_count_8801 (by decide)
private theorem atomic_8801_agreement : RowAgreement 8800 8801 := RowAgreement.single (first := 8800) modelrow_agrees_8801
private def atomic_8802 : CountSegment 1 10 13120 8801 8802 3686 3688 := CountSegment.single (first := 8801) (initial := 3686) row_count_8802 (by decide)
private theorem atomic_8802_agreement : RowAgreement 8801 8802 := RowAgreement.single (first := 8801) modelrow_agrees_8802
private def atomic_8803 : CountSegment 1 10 13120 8802 8803 3688 3689 := CountSegment.single (first := 8802) (initial := 3688) row_count_8803 (by decide)
private theorem atomic_8803_agreement : RowAgreement 8802 8803 := RowAgreement.single (first := 8802) modelrow_agrees_8803
private def atomic_8804 : CountSegment 1 10 13120 8803 8804 3689 3689 := CountSegment.single (first := 8803) (initial := 3689) row_count_8804 (by decide)
private theorem atomic_8804_agreement : RowAgreement 8803 8804 := RowAgreement.single (first := 8803) modelrow_agrees_8804
private def atomic_8805 : CountSegment 1 10 13120 8804 8805 3689 3689 := CountSegment.single (first := 8804) (initial := 3689) row_count_8805 (by decide)
private theorem atomic_8805_agreement : RowAgreement 8804 8805 := RowAgreement.single (first := 8804) modelrow_agrees_8805
private def atomic_8806 : CountSegment 1 10 13120 8805 8806 3689 3689 := CountSegment.single (first := 8805) (initial := 3689) row_count_8806 (by decide)
private theorem atomic_8806_agreement : RowAgreement 8805 8806 := RowAgreement.single (first := 8805) modelrow_agrees_8806
private def atomic_8807 : CountSegment 1 10 13120 8806 8807 3689 3690 := CountSegment.single (first := 8806) (initial := 3689) row_count_8807 (by decide)
private theorem atomic_8807_agreement : RowAgreement 8806 8807 := RowAgreement.single (first := 8806) modelrow_agrees_8807
private def atomic_8808 : CountSegment 1 10 13120 8807 8808 3690 3690 := CountSegment.single (first := 8807) (initial := 3690) row_count_8808 (by decide)
private theorem atomic_8808_agreement : RowAgreement 8807 8808 := RowAgreement.single (first := 8807) modelrow_agrees_8808
private def atomic_8809 : CountSegment 1 10 13120 8808 8809 3690 3690 := CountSegment.single (first := 8808) (initial := 3690) row_count_8809 (by decide)
private theorem atomic_8809_agreement : RowAgreement 8808 8809 := RowAgreement.single (first := 8808) modelrow_agrees_8809
private def atomic_8810 : CountSegment 1 10 13120 8809 8810 3690 3690 := CountSegment.single (first := 8809) (initial := 3690) row_count_8810 (by decide)
private theorem atomic_8810_agreement : RowAgreement 8809 8810 := RowAgreement.single (first := 8809) modelrow_agrees_8810
private def atomic_8811 : CountSegment 1 10 13120 8810 8811 3690 3690 := CountSegment.single (first := 8810) (initial := 3690) row_count_8811 (by decide)
private theorem atomic_8811_agreement : RowAgreement 8810 8811 := RowAgreement.single (first := 8810) modelrow_agrees_8811
private def atomic_8812 : CountSegment 1 10 13120 8811 8812 3690 3690 := CountSegment.single (first := 8811) (initial := 3690) row_count_8812 (by decide)
private theorem atomic_8812_agreement : RowAgreement 8811 8812 := RowAgreement.single (first := 8811) modelrow_agrees_8812
private def atomic_8813 : CountSegment 1 10 13120 8812 8813 3690 3690 := CountSegment.single (first := 8812) (initial := 3690) row_count_8813 (by decide)
private theorem atomic_8813_agreement : RowAgreement 8812 8813 := RowAgreement.single (first := 8812) modelrow_agrees_8813
private def atomic_8814 : CountSegment 1 10 13120 8813 8814 3690 3691 := CountSegment.single (first := 8813) (initial := 3690) row_count_8814 (by decide)
private theorem atomic_8814_agreement : RowAgreement 8813 8814 := RowAgreement.single (first := 8813) modelrow_agrees_8814
private def atomic_8815 : CountSegment 1 10 13120 8814 8815 3691 3691 := CountSegment.single (first := 8814) (initial := 3691) row_count_8815 (by decide)
private theorem atomic_8815_agreement : RowAgreement 8814 8815 := RowAgreement.single (first := 8814) modelrow_agrees_8815
private def atomic_8816 : CountSegment 1 10 13120 8815 8816 3691 3691 := CountSegment.single (first := 8815) (initial := 3691) row_count_8816 (by decide)
private theorem atomic_8816_agreement : RowAgreement 8815 8816 := RowAgreement.single (first := 8815) modelrow_agrees_8816
private def atomic_8817 : CountSegment 1 10 13120 8816 8817 3691 3692 := CountSegment.single (first := 8816) (initial := 3691) row_count_8817 (by decide)
private theorem atomic_8817_agreement : RowAgreement 8816 8817 := RowAgreement.single (first := 8816) modelrow_agrees_8817
private def atomic_8818 : CountSegment 1 10 13120 8817 8818 3692 3692 := CountSegment.single (first := 8817) (initial := 3692) row_count_8818 (by decide)
private theorem atomic_8818_agreement : RowAgreement 8817 8818 := RowAgreement.single (first := 8817) modelrow_agrees_8818
private def atomic_8819 : CountSegment 1 10 13120 8818 8819 3692 3692 := CountSegment.single (first := 8818) (initial := 3692) row_count_8819 (by decide)
private theorem atomic_8819_agreement : RowAgreement 8818 8819 := RowAgreement.single (first := 8818) modelrow_agrees_8819
private def atomic_8820 : CountSegment 1 10 13120 8819 8820 3692 3692 := CountSegment.single (first := 8819) (initial := 3692) row_count_8820 (by decide)
private theorem atomic_8820_agreement : RowAgreement 8819 8820 := RowAgreement.single (first := 8819) modelrow_agrees_8820
private def atomic_8821 : CountSegment 1 10 13120 8820 8821 3692 3692 := CountSegment.single (first := 8820) (initial := 3692) row_count_8821 (by decide)
private theorem atomic_8821_agreement : RowAgreement 8820 8821 := RowAgreement.single (first := 8820) modelrow_agrees_8821
private def atomic_8822 : CountSegment 1 10 13120 8821 8822 3692 3692 := CountSegment.single (first := 8821) (initial := 3692) row_count_8822 (by decide)
private theorem atomic_8822_agreement : RowAgreement 8821 8822 := RowAgreement.single (first := 8821) modelrow_agrees_8822
private def atomic_8823 : CountSegment 1 10 13120 8822 8823 3692 3692 := CountSegment.single (first := 8822) (initial := 3692) row_count_8823 (by decide)
private theorem atomic_8823_agreement : RowAgreement 8822 8823 := RowAgreement.single (first := 8822) modelrow_agrees_8823
private def atomic_8824 : CountSegment 1 10 13120 8823 8824 3692 3692 := CountSegment.single (first := 8823) (initial := 3692) row_count_8824 (by decide)
private theorem atomic_8824_agreement : RowAgreement 8823 8824 := RowAgreement.single (first := 8823) modelrow_agrees_8824
private def atomic_8825 : CountSegment 1 10 13120 8824 8825 3692 3692 := CountSegment.single (first := 8824) (initial := 3692) row_count_8825 (by decide)
private theorem atomic_8825_agreement : RowAgreement 8824 8825 := RowAgreement.single (first := 8824) modelrow_agrees_8825
private def atomic_8826 : CountSegment 1 10 13120 8825 8826 3692 3692 := CountSegment.single (first := 8825) (initial := 3692) row_count_8826 (by decide)
private theorem atomic_8826_agreement : RowAgreement 8825 8826 := RowAgreement.single (first := 8825) modelrow_agrees_8826
private def atomic_8827 : CountSegment 1 10 13120 8826 8827 3692 3693 := CountSegment.single (first := 8826) (initial := 3692) row_count_8827 (by decide)
private theorem atomic_8827_agreement : RowAgreement 8826 8827 := RowAgreement.single (first := 8826) modelrow_agrees_8827
private def atomic_8828 : CountSegment 1 10 13120 8827 8828 3693 3694 := CountSegment.single (first := 8827) (initial := 3693) row_count_8828 (by decide)
private theorem atomic_8828_agreement : RowAgreement 8827 8828 := RowAgreement.single (first := 8827) modelrow_agrees_8828
private def atomic_8829 : CountSegment 1 10 13120 8828 8829 3694 3694 := CountSegment.single (first := 8828) (initial := 3694) row_count_8829 (by decide)
private theorem atomic_8829_agreement : RowAgreement 8828 8829 := RowAgreement.single (first := 8828) modelrow_agrees_8829
private def atomic_8830 : CountSegment 1 10 13120 8829 8830 3694 3694 := CountSegment.single (first := 8829) (initial := 3694) row_count_8830 (by decide)
private theorem atomic_8830_agreement : RowAgreement 8829 8830 := RowAgreement.single (first := 8829) modelrow_agrees_8830
private def atomic_8831 : CountSegment 1 10 13120 8830 8831 3694 3694 := CountSegment.single (first := 8830) (initial := 3694) row_count_8831 (by decide)
private theorem atomic_8831_agreement : RowAgreement 8830 8831 := RowAgreement.single (first := 8830) modelrow_agrees_8831
private def atomic_8832 : CountSegment 1 10 13120 8831 8832 3694 3694 := CountSegment.single (first := 8831) (initial := 3694) row_count_8832 (by decide)
private theorem atomic_8832_agreement : RowAgreement 8831 8832 := RowAgreement.single (first := 8831) modelrow_agrees_8832
private def within_068_000 : CountSegment 1 10 13120 8704 8706 3655 3656 := CountSegment.join atomic_8705 atomic_8706
private theorem within_068_000_agreement : RowAgreement 8704 8706 := RowAgreement.join atomic_8705_agreement atomic_8706_agreement
private def within_068_001 : CountSegment 1 10 13120 8706 8708 3656 3656 := CountSegment.join atomic_8707 atomic_8708
private theorem within_068_001_agreement : RowAgreement 8706 8708 := RowAgreement.join atomic_8707_agreement atomic_8708_agreement
private def within_068_002 : CountSegment 1 10 13120 8704 8708 3655 3656 := CountSegment.join within_068_000 within_068_001
private theorem within_068_002_agreement : RowAgreement 8704 8708 := RowAgreement.join within_068_000_agreement within_068_001_agreement
private def within_068_003 : CountSegment 1 10 13120 8708 8710 3656 3656 := CountSegment.join atomic_8709 atomic_8710
private theorem within_068_003_agreement : RowAgreement 8708 8710 := RowAgreement.join atomic_8709_agreement atomic_8710_agreement
private def within_068_004 : CountSegment 1 10 13120 8710 8712 3656 3657 := CountSegment.join atomic_8711 atomic_8712
private theorem within_068_004_agreement : RowAgreement 8710 8712 := RowAgreement.join atomic_8711_agreement atomic_8712_agreement
private def within_068_005 : CountSegment 1 10 13120 8708 8712 3656 3657 := CountSegment.join within_068_003 within_068_004
private theorem within_068_005_agreement : RowAgreement 8708 8712 := RowAgreement.join within_068_003_agreement within_068_004_agreement
private def within_068_006 : CountSegment 1 10 13120 8704 8712 3655 3657 := CountSegment.join within_068_002 within_068_005
private theorem within_068_006_agreement : RowAgreement 8704 8712 := RowAgreement.join within_068_002_agreement within_068_005_agreement
private def within_068_007 : CountSegment 1 10 13120 8712 8714 3657 3658 := CountSegment.join atomic_8713 atomic_8714
private theorem within_068_007_agreement : RowAgreement 8712 8714 := RowAgreement.join atomic_8713_agreement atomic_8714_agreement
private def within_068_008 : CountSegment 1 10 13120 8714 8716 3658 3658 := CountSegment.join atomic_8715 atomic_8716
private theorem within_068_008_agreement : RowAgreement 8714 8716 := RowAgreement.join atomic_8715_agreement atomic_8716_agreement
private def within_068_009 : CountSegment 1 10 13120 8712 8716 3657 3658 := CountSegment.join within_068_007 within_068_008
private theorem within_068_009_agreement : RowAgreement 8712 8716 := RowAgreement.join within_068_007_agreement within_068_008_agreement
private def within_068_010 : CountSegment 1 10 13120 8716 8718 3658 3658 := CountSegment.join atomic_8717 atomic_8718
private theorem within_068_010_agreement : RowAgreement 8716 8718 := RowAgreement.join atomic_8717_agreement atomic_8718_agreement
private def within_068_011 : CountSegment 1 10 13120 8718 8720 3658 3658 := CountSegment.join atomic_8719 atomic_8720
private theorem within_068_011_agreement : RowAgreement 8718 8720 := RowAgreement.join atomic_8719_agreement atomic_8720_agreement
private def within_068_012 : CountSegment 1 10 13120 8716 8720 3658 3658 := CountSegment.join within_068_010 within_068_011
private theorem within_068_012_agreement : RowAgreement 8716 8720 := RowAgreement.join within_068_010_agreement within_068_011_agreement
private def within_068_013 : CountSegment 1 10 13120 8712 8720 3657 3658 := CountSegment.join within_068_009 within_068_012
private theorem within_068_013_agreement : RowAgreement 8712 8720 := RowAgreement.join within_068_009_agreement within_068_012_agreement
private def within_068_014 : CountSegment 1 10 13120 8704 8720 3655 3658 := CountSegment.join within_068_006 within_068_013
private theorem within_068_014_agreement : RowAgreement 8704 8720 := RowAgreement.join within_068_006_agreement within_068_013_agreement
private def within_068_015 : CountSegment 1 10 13120 8720 8722 3658 3658 := CountSegment.join atomic_8721 atomic_8722
private theorem within_068_015_agreement : RowAgreement 8720 8722 := RowAgreement.join atomic_8721_agreement atomic_8722_agreement
private def within_068_016 : CountSegment 1 10 13120 8722 8724 3658 3658 := CountSegment.join atomic_8723 atomic_8724
private theorem within_068_016_agreement : RowAgreement 8722 8724 := RowAgreement.join atomic_8723_agreement atomic_8724_agreement
private def within_068_017 : CountSegment 1 10 13120 8720 8724 3658 3658 := CountSegment.join within_068_015 within_068_016
private theorem within_068_017_agreement : RowAgreement 8720 8724 := RowAgreement.join within_068_015_agreement within_068_016_agreement
private def within_068_018 : CountSegment 1 10 13120 8724 8726 3658 3660 := CountSegment.join atomic_8725 atomic_8726
private theorem within_068_018_agreement : RowAgreement 8724 8726 := RowAgreement.join atomic_8725_agreement atomic_8726_agreement
private def within_068_019 : CountSegment 1 10 13120 8726 8728 3660 3661 := CountSegment.join atomic_8727 atomic_8728
private theorem within_068_019_agreement : RowAgreement 8726 8728 := RowAgreement.join atomic_8727_agreement atomic_8728_agreement
private def within_068_020 : CountSegment 1 10 13120 8724 8728 3658 3661 := CountSegment.join within_068_018 within_068_019
private theorem within_068_020_agreement : RowAgreement 8724 8728 := RowAgreement.join within_068_018_agreement within_068_019_agreement
private def within_068_021 : CountSegment 1 10 13120 8720 8728 3658 3661 := CountSegment.join within_068_017 within_068_020
private theorem within_068_021_agreement : RowAgreement 8720 8728 := RowAgreement.join within_068_017_agreement within_068_020_agreement
private def within_068_022 : CountSegment 1 10 13120 8728 8730 3661 3661 := CountSegment.join atomic_8729 atomic_8730
private theorem within_068_022_agreement : RowAgreement 8728 8730 := RowAgreement.join atomic_8729_agreement atomic_8730_agreement
private def within_068_023 : CountSegment 1 10 13120 8730 8732 3661 3661 := CountSegment.join atomic_8731 atomic_8732
private theorem within_068_023_agreement : RowAgreement 8730 8732 := RowAgreement.join atomic_8731_agreement atomic_8732_agreement
private def within_068_024 : CountSegment 1 10 13120 8728 8732 3661 3661 := CountSegment.join within_068_022 within_068_023
private theorem within_068_024_agreement : RowAgreement 8728 8732 := RowAgreement.join within_068_022_agreement within_068_023_agreement
private def within_068_025 : CountSegment 1 10 13120 8732 8734 3661 3663 := CountSegment.join atomic_8733 atomic_8734
private theorem within_068_025_agreement : RowAgreement 8732 8734 := RowAgreement.join atomic_8733_agreement atomic_8734_agreement
private def within_068_026 : CountSegment 1 10 13120 8734 8736 3663 3663 := CountSegment.join atomic_8735 atomic_8736
private theorem within_068_026_agreement : RowAgreement 8734 8736 := RowAgreement.join atomic_8735_agreement atomic_8736_agreement
private def within_068_027 : CountSegment 1 10 13120 8732 8736 3661 3663 := CountSegment.join within_068_025 within_068_026
private theorem within_068_027_agreement : RowAgreement 8732 8736 := RowAgreement.join within_068_025_agreement within_068_026_agreement
private def within_068_028 : CountSegment 1 10 13120 8728 8736 3661 3663 := CountSegment.join within_068_024 within_068_027
private theorem within_068_028_agreement : RowAgreement 8728 8736 := RowAgreement.join within_068_024_agreement within_068_027_agreement
private def within_068_029 : CountSegment 1 10 13120 8720 8736 3658 3663 := CountSegment.join within_068_021 within_068_028
private theorem within_068_029_agreement : RowAgreement 8720 8736 := RowAgreement.join within_068_021_agreement within_068_028_agreement
private def within_068_030 : CountSegment 1 10 13120 8704 8736 3655 3663 := CountSegment.join within_068_014 within_068_029
private theorem within_068_030_agreement : RowAgreement 8704 8736 := RowAgreement.join within_068_014_agreement within_068_029_agreement
private def within_068_031 : CountSegment 1 10 13120 8736 8738 3663 3663 := CountSegment.join atomic_8737 atomic_8738
private theorem within_068_031_agreement : RowAgreement 8736 8738 := RowAgreement.join atomic_8737_agreement atomic_8738_agreement
private def within_068_032 : CountSegment 1 10 13120 8738 8740 3663 3664 := CountSegment.join atomic_8739 atomic_8740
private theorem within_068_032_agreement : RowAgreement 8738 8740 := RowAgreement.join atomic_8739_agreement atomic_8740_agreement
private def within_068_033 : CountSegment 1 10 13120 8736 8740 3663 3664 := CountSegment.join within_068_031 within_068_032
private theorem within_068_033_agreement : RowAgreement 8736 8740 := RowAgreement.join within_068_031_agreement within_068_032_agreement
private def within_068_034 : CountSegment 1 10 13120 8740 8742 3664 3665 := CountSegment.join atomic_8741 atomic_8742
private theorem within_068_034_agreement : RowAgreement 8740 8742 := RowAgreement.join atomic_8741_agreement atomic_8742_agreement
private def within_068_035 : CountSegment 1 10 13120 8742 8744 3665 3666 := CountSegment.join atomic_8743 atomic_8744
private theorem within_068_035_agreement : RowAgreement 8742 8744 := RowAgreement.join atomic_8743_agreement atomic_8744_agreement
private def within_068_036 : CountSegment 1 10 13120 8740 8744 3664 3666 := CountSegment.join within_068_034 within_068_035
private theorem within_068_036_agreement : RowAgreement 8740 8744 := RowAgreement.join within_068_034_agreement within_068_035_agreement
private def within_068_037 : CountSegment 1 10 13120 8736 8744 3663 3666 := CountSegment.join within_068_033 within_068_036
private theorem within_068_037_agreement : RowAgreement 8736 8744 := RowAgreement.join within_068_033_agreement within_068_036_agreement
private def within_068_038 : CountSegment 1 10 13120 8744 8746 3666 3666 := CountSegment.join atomic_8745 atomic_8746
private theorem within_068_038_agreement : RowAgreement 8744 8746 := RowAgreement.join atomic_8745_agreement atomic_8746_agreement
private def within_068_039 : CountSegment 1 10 13120 8746 8748 3666 3667 := CountSegment.join atomic_8747 atomic_8748
private theorem within_068_039_agreement : RowAgreement 8746 8748 := RowAgreement.join atomic_8747_agreement atomic_8748_agreement
private def within_068_040 : CountSegment 1 10 13120 8744 8748 3666 3667 := CountSegment.join within_068_038 within_068_039
private theorem within_068_040_agreement : RowAgreement 8744 8748 := RowAgreement.join within_068_038_agreement within_068_039_agreement
private def within_068_041 : CountSegment 1 10 13120 8748 8750 3667 3667 := CountSegment.join atomic_8749 atomic_8750
private theorem within_068_041_agreement : RowAgreement 8748 8750 := RowAgreement.join atomic_8749_agreement atomic_8750_agreement
private def within_068_042 : CountSegment 1 10 13120 8750 8752 3667 3667 := CountSegment.join atomic_8751 atomic_8752
private theorem within_068_042_agreement : RowAgreement 8750 8752 := RowAgreement.join atomic_8751_agreement atomic_8752_agreement
private def within_068_043 : CountSegment 1 10 13120 8748 8752 3667 3667 := CountSegment.join within_068_041 within_068_042
private theorem within_068_043_agreement : RowAgreement 8748 8752 := RowAgreement.join within_068_041_agreement within_068_042_agreement
private def within_068_044 : CountSegment 1 10 13120 8744 8752 3666 3667 := CountSegment.join within_068_040 within_068_043
private theorem within_068_044_agreement : RowAgreement 8744 8752 := RowAgreement.join within_068_040_agreement within_068_043_agreement
private def within_068_045 : CountSegment 1 10 13120 8736 8752 3663 3667 := CountSegment.join within_068_037 within_068_044
private theorem within_068_045_agreement : RowAgreement 8736 8752 := RowAgreement.join within_068_037_agreement within_068_044_agreement
private def within_068_046 : CountSegment 1 10 13120 8752 8754 3667 3668 := CountSegment.join atomic_8753 atomic_8754
private theorem within_068_046_agreement : RowAgreement 8752 8754 := RowAgreement.join atomic_8753_agreement atomic_8754_agreement
private def within_068_047 : CountSegment 1 10 13120 8754 8756 3668 3669 := CountSegment.join atomic_8755 atomic_8756
private theorem within_068_047_agreement : RowAgreement 8754 8756 := RowAgreement.join atomic_8755_agreement atomic_8756_agreement
private def within_068_048 : CountSegment 1 10 13120 8752 8756 3667 3669 := CountSegment.join within_068_046 within_068_047
private theorem within_068_048_agreement : RowAgreement 8752 8756 := RowAgreement.join within_068_046_agreement within_068_047_agreement
private def within_068_049 : CountSegment 1 10 13120 8756 8758 3669 3669 := CountSegment.join atomic_8757 atomic_8758
private theorem within_068_049_agreement : RowAgreement 8756 8758 := RowAgreement.join atomic_8757_agreement atomic_8758_agreement
private def within_068_050 : CountSegment 1 10 13120 8758 8760 3669 3669 := CountSegment.join atomic_8759 atomic_8760
private theorem within_068_050_agreement : RowAgreement 8758 8760 := RowAgreement.join atomic_8759_agreement atomic_8760_agreement
private def within_068_051 : CountSegment 1 10 13120 8756 8760 3669 3669 := CountSegment.join within_068_049 within_068_050
private theorem within_068_051_agreement : RowAgreement 8756 8760 := RowAgreement.join within_068_049_agreement within_068_050_agreement
private def within_068_052 : CountSegment 1 10 13120 8752 8760 3667 3669 := CountSegment.join within_068_048 within_068_051
private theorem within_068_052_agreement : RowAgreement 8752 8760 := RowAgreement.join within_068_048_agreement within_068_051_agreement
private def within_068_053 : CountSegment 1 10 13120 8760 8762 3669 3670 := CountSegment.join atomic_8761 atomic_8762
private theorem within_068_053_agreement : RowAgreement 8760 8762 := RowAgreement.join atomic_8761_agreement atomic_8762_agreement
private def within_068_054 : CountSegment 1 10 13120 8762 8764 3670 3671 := CountSegment.join atomic_8763 atomic_8764
private theorem within_068_054_agreement : RowAgreement 8762 8764 := RowAgreement.join atomic_8763_agreement atomic_8764_agreement
private def within_068_055 : CountSegment 1 10 13120 8760 8764 3669 3671 := CountSegment.join within_068_053 within_068_054
private theorem within_068_055_agreement : RowAgreement 8760 8764 := RowAgreement.join within_068_053_agreement within_068_054_agreement
private def within_068_056 : CountSegment 1 10 13120 8764 8766 3671 3672 := CountSegment.join atomic_8765 atomic_8766
private theorem within_068_056_agreement : RowAgreement 8764 8766 := RowAgreement.join atomic_8765_agreement atomic_8766_agreement
private def within_068_057 : CountSegment 1 10 13120 8766 8768 3672 3672 := CountSegment.join atomic_8767 atomic_8768
private theorem within_068_057_agreement : RowAgreement 8766 8768 := RowAgreement.join atomic_8767_agreement atomic_8768_agreement
private def within_068_058 : CountSegment 1 10 13120 8764 8768 3671 3672 := CountSegment.join within_068_056 within_068_057
private theorem within_068_058_agreement : RowAgreement 8764 8768 := RowAgreement.join within_068_056_agreement within_068_057_agreement
private def within_068_059 : CountSegment 1 10 13120 8760 8768 3669 3672 := CountSegment.join within_068_055 within_068_058
private theorem within_068_059_agreement : RowAgreement 8760 8768 := RowAgreement.join within_068_055_agreement within_068_058_agreement
private def within_068_060 : CountSegment 1 10 13120 8752 8768 3667 3672 := CountSegment.join within_068_052 within_068_059
private theorem within_068_060_agreement : RowAgreement 8752 8768 := RowAgreement.join within_068_052_agreement within_068_059_agreement
private def within_068_061 : CountSegment 1 10 13120 8736 8768 3663 3672 := CountSegment.join within_068_045 within_068_060
private theorem within_068_061_agreement : RowAgreement 8736 8768 := RowAgreement.join within_068_045_agreement within_068_060_agreement
private def within_068_062 : CountSegment 1 10 13120 8704 8768 3655 3672 := CountSegment.join within_068_030 within_068_061
private theorem within_068_062_agreement : RowAgreement 8704 8768 := RowAgreement.join within_068_030_agreement within_068_061_agreement
private def within_068_063 : CountSegment 1 10 13120 8768 8770 3672 3672 := CountSegment.join atomic_8769 atomic_8770
private theorem within_068_063_agreement : RowAgreement 8768 8770 := RowAgreement.join atomic_8769_agreement atomic_8770_agreement
private def within_068_064 : CountSegment 1 10 13120 8770 8772 3672 3674 := CountSegment.join atomic_8771 atomic_8772
private theorem within_068_064_agreement : RowAgreement 8770 8772 := RowAgreement.join atomic_8771_agreement atomic_8772_agreement
private def within_068_065 : CountSegment 1 10 13120 8768 8772 3672 3674 := CountSegment.join within_068_063 within_068_064
private theorem within_068_065_agreement : RowAgreement 8768 8772 := RowAgreement.join within_068_063_agreement within_068_064_agreement
private def within_068_066 : CountSegment 1 10 13120 8772 8774 3674 3675 := CountSegment.join atomic_8773 atomic_8774
private theorem within_068_066_agreement : RowAgreement 8772 8774 := RowAgreement.join atomic_8773_agreement atomic_8774_agreement
private def within_068_067 : CountSegment 1 10 13120 8774 8776 3675 3677 := CountSegment.join atomic_8775 atomic_8776
private theorem within_068_067_agreement : RowAgreement 8774 8776 := RowAgreement.join atomic_8775_agreement atomic_8776_agreement
private def within_068_068 : CountSegment 1 10 13120 8772 8776 3674 3677 := CountSegment.join within_068_066 within_068_067
private theorem within_068_068_agreement : RowAgreement 8772 8776 := RowAgreement.join within_068_066_agreement within_068_067_agreement
private def within_068_069 : CountSegment 1 10 13120 8768 8776 3672 3677 := CountSegment.join within_068_065 within_068_068
private theorem within_068_069_agreement : RowAgreement 8768 8776 := RowAgreement.join within_068_065_agreement within_068_068_agreement
private def within_068_070 : CountSegment 1 10 13120 8776 8778 3677 3677 := CountSegment.join atomic_8777 atomic_8778
private theorem within_068_070_agreement : RowAgreement 8776 8778 := RowAgreement.join atomic_8777_agreement atomic_8778_agreement
private def within_068_071 : CountSegment 1 10 13120 8778 8780 3677 3677 := CountSegment.join atomic_8779 atomic_8780
private theorem within_068_071_agreement : RowAgreement 8778 8780 := RowAgreement.join atomic_8779_agreement atomic_8780_agreement
private def within_068_072 : CountSegment 1 10 13120 8776 8780 3677 3677 := CountSegment.join within_068_070 within_068_071
private theorem within_068_072_agreement : RowAgreement 8776 8780 := RowAgreement.join within_068_070_agreement within_068_071_agreement
private def within_068_073 : CountSegment 1 10 13120 8780 8782 3677 3678 := CountSegment.join atomic_8781 atomic_8782
private theorem within_068_073_agreement : RowAgreement 8780 8782 := RowAgreement.join atomic_8781_agreement atomic_8782_agreement
private def within_068_074 : CountSegment 1 10 13120 8782 8784 3678 3678 := CountSegment.join atomic_8783 atomic_8784
private theorem within_068_074_agreement : RowAgreement 8782 8784 := RowAgreement.join atomic_8783_agreement atomic_8784_agreement
private def within_068_075 : CountSegment 1 10 13120 8780 8784 3677 3678 := CountSegment.join within_068_073 within_068_074
private theorem within_068_075_agreement : RowAgreement 8780 8784 := RowAgreement.join within_068_073_agreement within_068_074_agreement
private def within_068_076 : CountSegment 1 10 13120 8776 8784 3677 3678 := CountSegment.join within_068_072 within_068_075
private theorem within_068_076_agreement : RowAgreement 8776 8784 := RowAgreement.join within_068_072_agreement within_068_075_agreement
private def within_068_077 : CountSegment 1 10 13120 8768 8784 3672 3678 := CountSegment.join within_068_069 within_068_076
private theorem within_068_077_agreement : RowAgreement 8768 8784 := RowAgreement.join within_068_069_agreement within_068_076_agreement
private def within_068_078 : CountSegment 1 10 13120 8784 8786 3678 3680 := CountSegment.join atomic_8785 atomic_8786
private theorem within_068_078_agreement : RowAgreement 8784 8786 := RowAgreement.join atomic_8785_agreement atomic_8786_agreement
private def within_068_079 : CountSegment 1 10 13120 8786 8788 3680 3682 := CountSegment.join atomic_8787 atomic_8788
private theorem within_068_079_agreement : RowAgreement 8786 8788 := RowAgreement.join atomic_8787_agreement atomic_8788_agreement
private def within_068_080 : CountSegment 1 10 13120 8784 8788 3678 3682 := CountSegment.join within_068_078 within_068_079
private theorem within_068_080_agreement : RowAgreement 8784 8788 := RowAgreement.join within_068_078_agreement within_068_079_agreement
private def within_068_081 : CountSegment 1 10 13120 8788 8790 3682 3682 := CountSegment.join atomic_8789 atomic_8790
private theorem within_068_081_agreement : RowAgreement 8788 8790 := RowAgreement.join atomic_8789_agreement atomic_8790_agreement
private def within_068_082 : CountSegment 1 10 13120 8790 8792 3682 3682 := CountSegment.join atomic_8791 atomic_8792
private theorem within_068_082_agreement : RowAgreement 8790 8792 := RowAgreement.join atomic_8791_agreement atomic_8792_agreement
private def within_068_083 : CountSegment 1 10 13120 8788 8792 3682 3682 := CountSegment.join within_068_081 within_068_082
private theorem within_068_083_agreement : RowAgreement 8788 8792 := RowAgreement.join within_068_081_agreement within_068_082_agreement
private def within_068_084 : CountSegment 1 10 13120 8784 8792 3678 3682 := CountSegment.join within_068_080 within_068_083
private theorem within_068_084_agreement : RowAgreement 8784 8792 := RowAgreement.join within_068_080_agreement within_068_083_agreement
private def within_068_085 : CountSegment 1 10 13120 8792 8794 3682 3682 := CountSegment.join atomic_8793 atomic_8794
private theorem within_068_085_agreement : RowAgreement 8792 8794 := RowAgreement.join atomic_8793_agreement atomic_8794_agreement
private def within_068_086 : CountSegment 1 10 13120 8794 8796 3682 3684 := CountSegment.join atomic_8795 atomic_8796
private theorem within_068_086_agreement : RowAgreement 8794 8796 := RowAgreement.join atomic_8795_agreement atomic_8796_agreement
private def within_068_087 : CountSegment 1 10 13120 8792 8796 3682 3684 := CountSegment.join within_068_085 within_068_086
private theorem within_068_087_agreement : RowAgreement 8792 8796 := RowAgreement.join within_068_085_agreement within_068_086_agreement
private def within_068_088 : CountSegment 1 10 13120 8796 8798 3684 3684 := CountSegment.join atomic_8797 atomic_8798
private theorem within_068_088_agreement : RowAgreement 8796 8798 := RowAgreement.join atomic_8797_agreement atomic_8798_agreement
private def within_068_089 : CountSegment 1 10 13120 8798 8800 3684 3686 := CountSegment.join atomic_8799 atomic_8800
private theorem within_068_089_agreement : RowAgreement 8798 8800 := RowAgreement.join atomic_8799_agreement atomic_8800_agreement
private def within_068_090 : CountSegment 1 10 13120 8796 8800 3684 3686 := CountSegment.join within_068_088 within_068_089
private theorem within_068_090_agreement : RowAgreement 8796 8800 := RowAgreement.join within_068_088_agreement within_068_089_agreement
private def within_068_091 : CountSegment 1 10 13120 8792 8800 3682 3686 := CountSegment.join within_068_087 within_068_090
private theorem within_068_091_agreement : RowAgreement 8792 8800 := RowAgreement.join within_068_087_agreement within_068_090_agreement
private def within_068_092 : CountSegment 1 10 13120 8784 8800 3678 3686 := CountSegment.join within_068_084 within_068_091
private theorem within_068_092_agreement : RowAgreement 8784 8800 := RowAgreement.join within_068_084_agreement within_068_091_agreement
private def within_068_093 : CountSegment 1 10 13120 8768 8800 3672 3686 := CountSegment.join within_068_077 within_068_092
private theorem within_068_093_agreement : RowAgreement 8768 8800 := RowAgreement.join within_068_077_agreement within_068_092_agreement
private def within_068_094 : CountSegment 1 10 13120 8800 8802 3686 3688 := CountSegment.join atomic_8801 atomic_8802
private theorem within_068_094_agreement : RowAgreement 8800 8802 := RowAgreement.join atomic_8801_agreement atomic_8802_agreement
private def within_068_095 : CountSegment 1 10 13120 8802 8804 3688 3689 := CountSegment.join atomic_8803 atomic_8804
private theorem within_068_095_agreement : RowAgreement 8802 8804 := RowAgreement.join atomic_8803_agreement atomic_8804_agreement
private def within_068_096 : CountSegment 1 10 13120 8800 8804 3686 3689 := CountSegment.join within_068_094 within_068_095
private theorem within_068_096_agreement : RowAgreement 8800 8804 := RowAgreement.join within_068_094_agreement within_068_095_agreement
private def within_068_097 : CountSegment 1 10 13120 8804 8806 3689 3689 := CountSegment.join atomic_8805 atomic_8806
private theorem within_068_097_agreement : RowAgreement 8804 8806 := RowAgreement.join atomic_8805_agreement atomic_8806_agreement
private def within_068_098 : CountSegment 1 10 13120 8806 8808 3689 3690 := CountSegment.join atomic_8807 atomic_8808
private theorem within_068_098_agreement : RowAgreement 8806 8808 := RowAgreement.join atomic_8807_agreement atomic_8808_agreement
private def within_068_099 : CountSegment 1 10 13120 8804 8808 3689 3690 := CountSegment.join within_068_097 within_068_098
private theorem within_068_099_agreement : RowAgreement 8804 8808 := RowAgreement.join within_068_097_agreement within_068_098_agreement
private def within_068_100 : CountSegment 1 10 13120 8800 8808 3686 3690 := CountSegment.join within_068_096 within_068_099
private theorem within_068_100_agreement : RowAgreement 8800 8808 := RowAgreement.join within_068_096_agreement within_068_099_agreement
private def within_068_101 : CountSegment 1 10 13120 8808 8810 3690 3690 := CountSegment.join atomic_8809 atomic_8810
private theorem within_068_101_agreement : RowAgreement 8808 8810 := RowAgreement.join atomic_8809_agreement atomic_8810_agreement
private def within_068_102 : CountSegment 1 10 13120 8810 8812 3690 3690 := CountSegment.join atomic_8811 atomic_8812
private theorem within_068_102_agreement : RowAgreement 8810 8812 := RowAgreement.join atomic_8811_agreement atomic_8812_agreement
private def within_068_103 : CountSegment 1 10 13120 8808 8812 3690 3690 := CountSegment.join within_068_101 within_068_102
private theorem within_068_103_agreement : RowAgreement 8808 8812 := RowAgreement.join within_068_101_agreement within_068_102_agreement
private def within_068_104 : CountSegment 1 10 13120 8812 8814 3690 3691 := CountSegment.join atomic_8813 atomic_8814
private theorem within_068_104_agreement : RowAgreement 8812 8814 := RowAgreement.join atomic_8813_agreement atomic_8814_agreement
private def within_068_105 : CountSegment 1 10 13120 8814 8816 3691 3691 := CountSegment.join atomic_8815 atomic_8816
private theorem within_068_105_agreement : RowAgreement 8814 8816 := RowAgreement.join atomic_8815_agreement atomic_8816_agreement
private def within_068_106 : CountSegment 1 10 13120 8812 8816 3690 3691 := CountSegment.join within_068_104 within_068_105
private theorem within_068_106_agreement : RowAgreement 8812 8816 := RowAgreement.join within_068_104_agreement within_068_105_agreement
private def within_068_107 : CountSegment 1 10 13120 8808 8816 3690 3691 := CountSegment.join within_068_103 within_068_106
private theorem within_068_107_agreement : RowAgreement 8808 8816 := RowAgreement.join within_068_103_agreement within_068_106_agreement
private def within_068_108 : CountSegment 1 10 13120 8800 8816 3686 3691 := CountSegment.join within_068_100 within_068_107
private theorem within_068_108_agreement : RowAgreement 8800 8816 := RowAgreement.join within_068_100_agreement within_068_107_agreement
private def within_068_109 : CountSegment 1 10 13120 8816 8818 3691 3692 := CountSegment.join atomic_8817 atomic_8818
private theorem within_068_109_agreement : RowAgreement 8816 8818 := RowAgreement.join atomic_8817_agreement atomic_8818_agreement
private def within_068_110 : CountSegment 1 10 13120 8818 8820 3692 3692 := CountSegment.join atomic_8819 atomic_8820
private theorem within_068_110_agreement : RowAgreement 8818 8820 := RowAgreement.join atomic_8819_agreement atomic_8820_agreement
private def within_068_111 : CountSegment 1 10 13120 8816 8820 3691 3692 := CountSegment.join within_068_109 within_068_110
private theorem within_068_111_agreement : RowAgreement 8816 8820 := RowAgreement.join within_068_109_agreement within_068_110_agreement
private def within_068_112 : CountSegment 1 10 13120 8820 8822 3692 3692 := CountSegment.join atomic_8821 atomic_8822
private theorem within_068_112_agreement : RowAgreement 8820 8822 := RowAgreement.join atomic_8821_agreement atomic_8822_agreement
private def within_068_113 : CountSegment 1 10 13120 8822 8824 3692 3692 := CountSegment.join atomic_8823 atomic_8824
private theorem within_068_113_agreement : RowAgreement 8822 8824 := RowAgreement.join atomic_8823_agreement atomic_8824_agreement
private def within_068_114 : CountSegment 1 10 13120 8820 8824 3692 3692 := CountSegment.join within_068_112 within_068_113
private theorem within_068_114_agreement : RowAgreement 8820 8824 := RowAgreement.join within_068_112_agreement within_068_113_agreement
private def within_068_115 : CountSegment 1 10 13120 8816 8824 3691 3692 := CountSegment.join within_068_111 within_068_114
private theorem within_068_115_agreement : RowAgreement 8816 8824 := RowAgreement.join within_068_111_agreement within_068_114_agreement
private def within_068_116 : CountSegment 1 10 13120 8824 8826 3692 3692 := CountSegment.join atomic_8825 atomic_8826
private theorem within_068_116_agreement : RowAgreement 8824 8826 := RowAgreement.join atomic_8825_agreement atomic_8826_agreement
private def within_068_117 : CountSegment 1 10 13120 8826 8828 3692 3694 := CountSegment.join atomic_8827 atomic_8828
private theorem within_068_117_agreement : RowAgreement 8826 8828 := RowAgreement.join atomic_8827_agreement atomic_8828_agreement
private def within_068_118 : CountSegment 1 10 13120 8824 8828 3692 3694 := CountSegment.join within_068_116 within_068_117
private theorem within_068_118_agreement : RowAgreement 8824 8828 := RowAgreement.join within_068_116_agreement within_068_117_agreement
private def within_068_119 : CountSegment 1 10 13120 8828 8830 3694 3694 := CountSegment.join atomic_8829 atomic_8830
private theorem within_068_119_agreement : RowAgreement 8828 8830 := RowAgreement.join atomic_8829_agreement atomic_8830_agreement
private def within_068_120 : CountSegment 1 10 13120 8830 8832 3694 3694 := CountSegment.join atomic_8831 atomic_8832
private theorem within_068_120_agreement : RowAgreement 8830 8832 := RowAgreement.join atomic_8831_agreement atomic_8832_agreement
private def within_068_121 : CountSegment 1 10 13120 8828 8832 3694 3694 := CountSegment.join within_068_119 within_068_120
private theorem within_068_121_agreement : RowAgreement 8828 8832 := RowAgreement.join within_068_119_agreement within_068_120_agreement
private def within_068_122 : CountSegment 1 10 13120 8824 8832 3692 3694 := CountSegment.join within_068_118 within_068_121
private theorem within_068_122_agreement : RowAgreement 8824 8832 := RowAgreement.join within_068_118_agreement within_068_121_agreement
private def within_068_123 : CountSegment 1 10 13120 8816 8832 3691 3694 := CountSegment.join within_068_115 within_068_122
private theorem within_068_123_agreement : RowAgreement 8816 8832 := RowAgreement.join within_068_115_agreement within_068_122_agreement
private def within_068_124 : CountSegment 1 10 13120 8800 8832 3686 3694 := CountSegment.join within_068_108 within_068_123
private theorem within_068_124_agreement : RowAgreement 8800 8832 := RowAgreement.join within_068_108_agreement within_068_123_agreement
private def within_068_125 : CountSegment 1 10 13120 8768 8832 3672 3694 := CountSegment.join within_068_093 within_068_124
private theorem within_068_125_agreement : RowAgreement 8768 8832 := RowAgreement.join within_068_093_agreement within_068_124_agreement
private def within_068_126 : CountSegment 1 10 13120 8704 8832 3655 3694 := CountSegment.join within_068_062 within_068_125
private theorem within_068_126_agreement : RowAgreement 8704 8832 := RowAgreement.join within_068_062_agreement within_068_125_agreement
def segment_068 : CountSegment 1 10 13120 8704 8832 3655 3694 := within_068_126
theorem agreement_068 : RowAgreement 8704 8832 := within_068_126_agreement
end ProjectEulerSolutions.P58Numerical
