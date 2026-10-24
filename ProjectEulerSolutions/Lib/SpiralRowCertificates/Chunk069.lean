import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_8833 : RowCertificate 8833 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8833 1) := by change ¬ Nat.Prime 312105223; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7258261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8833 2) := by change ¬ Nat.Prime 312087557; exact PrattCertificates.composite_of_product (factor := 9277) (cofactor := 33641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8833 3) := by change ¬ Nat.Prime 312069891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104023297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8833 : row 8833 = 0 := (row_8833.checked).trans rfl
theorem model_count_8833 : modelRow 8833 = 0 := by decide
theorem modelrow_agrees_8833 : modelRow 8833 = row 8833 := model_count_8833.trans row_count_8833.symm
def row_8834 : RowCertificate 8834 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8834 1) := by change ¬ Nat.Prime 312175893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104058631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8834 2) := by change ¬ Nat.Prime 312158225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62431645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8834 3) := by change ¬ Nat.Prime 312140557; exact PrattCertificates.composite_of_product (factor := 7699) (cofactor := 40543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8834 : row 8834 = 0 := (row_8834.checked).trans rfl
theorem model_count_8834 : modelRow 8834 = 0 := by decide
theorem modelrow_agrees_8834 : modelRow 8834 = row 8834 := model_count_8834.trans row_count_8834.symm
def row_8835 : RowCertificate 8835 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8835 1) := by change ¬ Nat.Prime 312246571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44606653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8835 2) := by change Nat.Prime 312228901; exact ProjectEulerSolutions.P58PrimeNodes.p_312228901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8835 3) := by change ¬ Nat.Prime 312211231; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3031177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8835 : row 8835 = 1 := (row_8835.checked).trans rfl
theorem model_count_8835 : modelRow 8835 = 1 := by decide
theorem modelrow_agrees_8835 : modelRow 8835 = row 8835 := model_count_8835.trans row_count_8835.symm
def row_8836 : RowCertificate 8836 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8836 1) := by change ¬ Nat.Prime 312317257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44616751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8836 2) := by change ¬ Nat.Prime 312299585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62459917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8836 3) := by change ¬ Nat.Prime 312281913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104093971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8836 : row 8836 = 0 := (row_8836.checked).trans rfl
theorem model_count_8836 : modelRow 8836 = 0 := by decide
theorem modelrow_agrees_8836 : modelRow 8836 = row 8836 := model_count_8836.trans row_count_8836.symm
def row_8837 : RowCertificate 8837 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8837 1) := by change ¬ Nat.Prime 312387951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104129317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8837 2) := by change Nat.Prime 312370277; exact ProjectEulerSolutions.P58PrimeNodes.p_312370277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8837 3) := by change ¬ Nat.Prime 312352603; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 541339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8837 : row 8837 = 1 := (row_8837.checked).trans rfl
theorem model_count_8837 : modelRow 8837 = 1 := by decide
theorem modelrow_agrees_8837 : modelRow 8837 = row 8837 := model_count_8837.trans row_count_8837.symm
def row_8838 : RowCertificate 8838 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8838 1) := by change ¬ Nat.Prime 312458653; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24035281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8838 2) := by change ¬ Nat.Prime 312440977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18378881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8838 3) := by change ¬ Nat.Prime 312423301; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10078171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8838 : row 8838 = 0 := (row_8838.checked).trans rfl
theorem model_count_8838 : modelRow 8838 = 0 := by decide
theorem modelrow_agrees_8838 : modelRow 8838 = row 8838 := model_count_8838.trans row_count_8838.symm
def row_8839 : RowCertificate 8839 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8839 1) := by change Nat.Prime 312529363; exact ProjectEulerSolutions.P58PrimeNodes.p_312529363.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8839 2) := by change ¬ Nat.Prime 312511685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62502337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8839 3) := by change ¬ Nat.Prime 312494007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104164669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8839 : row 8839 = 1 := (row_8839.checked).trans rfl
theorem model_count_8839 : modelRow 8839 = 1 := by decide
theorem modelrow_agrees_8839 : modelRow 8839 = row 8839 := model_count_8839.trans row_count_8839.symm
def row_8840 : RowCertificate 8840 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8840 1) := by change ¬ Nat.Prime 312600081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104200027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8840 2) := by change ¬ Nat.Prime 312582401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8448173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8840 3) := by change ¬ Nat.Prime 312564721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44652103) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8840 : row 8840 = 0 := (row_8840.checked).trans rfl
theorem model_count_8840 : modelRow 8840 = 0 := by decide
theorem modelrow_agrees_8840 : modelRow 8840 = row 8840 := model_count_8840.trans row_count_8840.symm
def row_8841 : RowCertificate 8841 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8841 1) := by change ¬ Nat.Prime 312670807; exact PrattCertificates.composite_of_product (factor := 3463) (cofactor := 90289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8841 2) := by change ¬ Nat.Prime 312653125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62530625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8841 3) := by change ¬ Nat.Prime 312635443; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16454497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8841 : row 8841 = 0 := (row_8841.checked).trans rfl
theorem model_count_8841 : modelRow 8841 = 0 := by decide
theorem modelrow_agrees_8841 : modelRow 8841 = row 8841 := model_count_8841.trans row_count_8841.symm
def row_8842 : RowCertificate 8842 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8842 1) := by change ¬ Nat.Prime 312741541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44677363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8842 2) := by change ¬ Nat.Prime 312723857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18395521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8842 3) := by change ¬ Nat.Prime 312706173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104235391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8842 : row 8842 = 0 := (row_8842.checked).trans rfl
theorem model_count_8842 : modelRow 8842 = 0 := by decide
theorem modelrow_agrees_8842 : modelRow 8842 = row 8842 := model_count_8842.trans row_count_8842.symm
def row_8843 : RowCertificate 8843 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8843 1) := by change ¬ Nat.Prime 312812283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104270761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8843 2) := by change Nat.Prime 312794597; exact ProjectEulerSolutions.P58PrimeNodes.p_312794597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8843 3) := by change ¬ Nat.Prime 312776911; exact PrattCertificates.composite_of_product (factor := 14851) (cofactor := 21061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8843 : row 8843 = 1 := (row_8843.checked).trans rfl
theorem model_count_8843 : modelRow 8843 = 1 := by decide
theorem modelrow_agrees_8843 : modelRow 8843 = row 8843 := model_count_8843.trans row_count_8843.symm
def row_8844 : RowCertificate 8844 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8844 1) := by change ¬ Nat.Prime 312883033; exact PrattCertificates.composite_of_product (factor := 3313) (cofactor := 94441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8844 2) := by change ¬ Nat.Prime 312865345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62573069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8844 3) := by change Nat.Prime 312847657; exact ProjectEulerSolutions.P58PrimeNodes.p_312847657.prime
    exact decide_eq_true prime
theorem row_count_8844 : row 8844 = 1 := (row_8844.checked).trans rfl
theorem model_count_8844 : modelRow 8844 = 1 := by decide
theorem modelrow_agrees_8844 : modelRow 8844 = row 8844 := model_count_8844.trans row_count_8844.symm
def row_8845 : RowCertificate 8845 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8845 1) := by change Nat.Prime 312953791; exact ProjectEulerSolutions.P58PrimeNodes.p_312953791.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8845 2) := by change Nat.Prime 312936101; exact ProjectEulerSolutions.P58PrimeNodes.p_312936101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8845 3) := by change ¬ Nat.Prime 312918411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104306137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8845 : row 8845 = 2 := (row_8845.checked).trans rfl
theorem model_count_8845 : modelRow 8845 = 2 := by decide
theorem modelrow_agrees_8845 : modelRow 8845 = row 8845 := model_count_8845.trans row_count_8845.symm
def row_8846 : RowCertificate 8846 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8846 1) := by change ¬ Nat.Prime 313024557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104341519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8846 2) := by change ¬ Nat.Prime 313006865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62601373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8846 3) := by change ¬ Nat.Prime 312989173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44712739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8846 : row 8846 = 0 := (row_8846.checked).trans rfl
theorem model_count_8846 : modelRow 8846 = 0 := by decide
theorem modelrow_agrees_8846 : modelRow 8846 = row 8846 := model_count_8846.trans row_count_8846.symm
def row_8847 : RowCertificate 8847 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8847 1) := by change Nat.Prime 313095331; exact ProjectEulerSolutions.P58PrimeNodes.p_313095331.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8847 2) := by change Nat.Prime 313077637; exact ProjectEulerSolutions.P58PrimeNodes.p_313077637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8847 3) := by change ¬ Nat.Prime 313059943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44722849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8847 : row 8847 = 2 := (row_8847.checked).trans rfl
theorem model_count_8847 : modelRow 8847 = 2 := by decide
theorem modelrow_agrees_8847 : modelRow 8847 = row 8847 := model_count_8847.trans row_count_8847.symm
def row_8848 : RowCertificate 8848 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8848 1) := by change ¬ Nat.Prime 313166113; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24089701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8848 2) := by change Nat.Prime 313148417; exact ProjectEulerSolutions.P58PrimeNodes.p_313148417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8848 3) := by change ¬ Nat.Prime 313130721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104376907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8848 : row 8848 = 1 := (row_8848.checked).trans rfl
theorem model_count_8848 : modelRow 8848 = 1 := by decide
theorem modelrow_agrees_8848 : modelRow 8848 = row 8848 := model_count_8848.trans row_count_8848.symm
def row_8849 : RowCertificate 8849 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8849 1) := by change ¬ Nat.Prime 313236903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104412301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8849 2) := by change ¬ Nat.Prime 313219205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62643841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8849 3) := by change ¬ Nat.Prime 313201507; exact PrattCertificates.composite_of_product (factor := 3361) (cofactor := 93187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8849 : row 8849 = 0 := (row_8849.checked).trans rfl
theorem model_count_8849 : modelRow 8849 = 0 := by decide
theorem modelrow_agrees_8849 : modelRow 8849 = row 8849 := model_count_8849.trans row_count_8849.symm
def row_8850 : RowCertificate 8850 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8850 1) := by change ¬ Nat.Prime 313307701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44758243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8850 2) := by change ¬ Nat.Prime 313290001; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 258277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8850 3) := by change ¬ Nat.Prime 313272301; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2074651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8850 : row 8850 = 0 := (row_8850.checked).trans rfl
theorem model_count_8850 : modelRow 8850 = 0 := by decide
theorem modelrow_agrees_8850 : modelRow 8850 = row 8850 := model_count_8850.trans row_count_8850.symm
def row_8851 : RowCertificate 8851 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8851 1) := by change ¬ Nat.Prime 313378507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24106039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8851 2) := by change ¬ Nat.Prime 313360805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62672161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8851 3) := by change ¬ Nat.Prime 313343103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104447701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8851 : row 8851 = 0 := (row_8851.checked).trans rfl
theorem model_count_8851 : modelRow 8851 = 0 := by decide
theorem modelrow_agrees_8851 : modelRow 8851 = row 8851 := model_count_8851.trans row_count_8851.symm
def row_8852 : RowCertificate 8852 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8852 1) := by change ¬ Nat.Prime 313449321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104483107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8852 2) := by change Nat.Prime 313431617; exact ProjectEulerSolutions.P58PrimeNodes.p_313431617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8852 3) := by change ¬ Nat.Prime 313413913; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5137933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8852 : row 8852 = 1 := (row_8852.checked).trans rfl
theorem model_count_8852 : modelRow 8852 = 1 := by decide
theorem modelrow_agrees_8852 : modelRow 8852 = row 8852 := model_count_8852.trans row_count_8852.symm
def row_8853 : RowCertificate 8853 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8853 1) := by change ¬ Nat.Prime 313520143; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10113553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8853 2) := by change ¬ Nat.Prime 313502437; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 562841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8853 3) := by change ¬ Nat.Prime 313484731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44783533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8853 : row 8853 = 0 := (row_8853.checked).trans rfl
theorem model_count_8853 : modelRow 8853 = 0 := by decide
theorem modelrow_agrees_8853 : modelRow 8853 = row 8853 := model_count_8853.trans row_count_8853.symm
def row_8854 : RowCertificate 8854 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8854 1) := by change ¬ Nat.Prime 313590973; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1923871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8854 2) := by change ¬ Nat.Prime 313573265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62714653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8854 3) := by change ¬ Nat.Prime 313555557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104518519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8854 : row 8854 = 0 := (row_8854.checked).trans rfl
theorem model_count_8854 : modelRow 8854 = 0 := by decide
theorem modelrow_agrees_8854 : modelRow 8854 = row 8854 := model_count_8854.trans row_count_8854.symm
def row_8855 : RowCertificate 8855 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8855 1) := by change ¬ Nat.Prime 313661811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104553937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8855 2) := by change ¬ Nat.Prime 313644101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18449653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8855 3) := by change ¬ Nat.Prime 313626391; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24125107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8855 : row 8855 = 0 := (row_8855.checked).trans rfl
theorem model_count_8855 : modelRow 8855 = 0 := by decide
theorem modelrow_agrees_8855 : modelRow 8855 = row 8855 := model_count_8855.trans row_count_8855.symm
def row_8856 : RowCertificate 8856 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8856 1) := by change ¬ Nat.Prime 313732657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44818951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8856 2) := by change ¬ Nat.Prime 313714945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62742989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8856 3) := by change ¬ Nat.Prime 313697233; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 511741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8856 : row 8856 = 0 := (row_8856.checked).trans rfl
theorem model_count_8856 : modelRow 8856 = 0 := by decide
theorem modelrow_agrees_8856 : modelRow 8856 = row 8856 := model_count_8856.trans row_count_8856.symm
def row_8857 : RowCertificate 8857 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8857 1) := by change ¬ Nat.Prime 313803511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44829073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8857 2) := by change ¬ Nat.Prime 313785797; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24137369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8857 3) := by change ¬ Nat.Prime 313768083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104589361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8857 : row 8857 = 0 := (row_8857.checked).trans rfl
theorem model_count_8857 : modelRow 8857 = 0 := by decide
theorem modelrow_agrees_8857 : modelRow 8857 = row 8857 := model_count_8857.trans row_count_8857.symm
def row_8858 : RowCertificate 8858 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8858 1) := by change ¬ Nat.Prime 313874373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104624791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8858 2) := by change ¬ Nat.Prime 313856657; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 1593181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8858 3) := by change ¬ Nat.Prime 313838941; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24141457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8858 : row 8858 = 0 := (row_8858.checked).trans rfl
theorem model_count_8858 : modelRow 8858 = 0 := by decide
theorem modelrow_agrees_8858 : modelRow 8858 = row 8858 := model_count_8858.trans row_count_8858.symm
def row_8859 : RowCertificate 8859 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8859 1) := by change Nat.Prime 313945243; exact ProjectEulerSolutions.P58PrimeNodes.p_313945243.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8859 2) := by change ¬ Nat.Prime 313927525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62785505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8859 3) := by change ¬ Nat.Prime 313909807; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4685221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8859 : row 8859 = 1 := (row_8859.checked).trans rfl
theorem model_count_8859 : modelRow 8859 = 1 := by decide
theorem modelrow_agrees_8859 : modelRow 8859 = row 8859 := model_count_8859.trans row_count_8859.symm
def row_8860 : RowCertificate 8860 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8860 1) := by change ¬ Nat.Prime 314016121; exact PrattCertificates.composite_of_product (factor := 10687) (cofactor := 29383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8860 2) := by change Nat.Prime 313998401; exact ProjectEulerSolutions.P58PrimeNodes.p_313998401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8860 3) := by change ¬ Nat.Prime 313980681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104660227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8860 : row 8860 = 1 := (row_8860.checked).trans rfl
theorem model_count_8860 : modelRow 8860 = 1 := by decide
theorem modelrow_agrees_8860 : modelRow 8860 = row 8860 := model_count_8860.trans row_count_8860.symm
def row_8861 : RowCertificate 8861 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8861 1) := by change ¬ Nat.Prime 314087007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104695669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8861 2) := by change ¬ Nat.Prime 314069285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62813857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8861 3) := by change ¬ Nat.Prime 314051563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44864509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8861 : row 8861 = 0 := (row_8861.checked).trans rfl
theorem model_count_8861 : modelRow 8861 = 0 := by decide
theorem modelrow_agrees_8861 : modelRow 8861 = row 8861 := model_count_8861.trans row_count_8861.symm
def row_8862 : RowCertificate 8862 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8862 1) := by change ¬ Nat.Prime 314157901; exact PrattCertificates.composite_of_product (factor := 991) (cofactor := 317011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8862 2) := by change ¬ Nat.Prime 314140177; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24164629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8862 3) := by change ¬ Nat.Prime 314122453; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 580633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8862 : row 8862 = 0 := (row_8862.checked).trans rfl
theorem model_count_8862 : modelRow 8862 = 0 := by decide
theorem modelrow_agrees_8862 : modelRow 8862 = row 8862 := model_count_8862.trans row_count_8862.symm
def row_8863 : RowCertificate 8863 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8863 1) := by change ¬ Nat.Prime 314228803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44889829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8863 2) := by change ¬ Nat.Prime 314211077; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2780629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8863 3) := by change ¬ Nat.Prime 314193351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104731117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8863 : row 8863 = 0 := (row_8863.checked).trans rfl
theorem model_count_8863 : modelRow 8863 = 0 := by decide
theorem modelrow_agrees_8863 : modelRow 8863 = row 8863 := model_count_8863.trans row_count_8863.symm
def row_8864 : RowCertificate 8864 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8864 1) := by change ¬ Nat.Prime 314299713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104766571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8864 2) := by change ¬ Nat.Prime 314281985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62856397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8864 3) := by change ¬ Nat.Prime 314264257; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1372333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8864 : row 8864 = 0 := (row_8864.checked).trans rfl
theorem model_count_8864 : modelRow 8864 = 0 := by decide
theorem modelrow_agrees_8864 : modelRow 8864 = row 8864 := model_count_8864.trans row_count_8864.symm
def row_8865 : RowCertificate 8865 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8865 1) := by change ¬ Nat.Prime 314370631; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4306447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8865 2) := by change ¬ Nat.Prime 314352901; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 768589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8865 3) := by change ¬ Nat.Prime 314335171; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 432373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8865 : row 8865 = 0 := (row_8865.checked).trans rfl
theorem model_count_8865 : modelRow 8865 = 0 := by decide
theorem modelrow_agrees_8865 : modelRow 8865 = row 8865 := model_count_8865.trans row_count_8865.symm
def row_8866 : RowCertificate 8866 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8866 1) := by change ¬ Nat.Prime 314441557; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 933061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8866 2) := by change ¬ Nat.Prime 314423825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62884765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8866 3) := by change ¬ Nat.Prime 314406093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104802031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8866 : row 8866 = 0 := (row_8866.checked).trans rfl
theorem model_count_8866 : modelRow 8866 = 0 := by decide
theorem modelrow_agrees_8866 : modelRow 8866 = row 8866 := model_count_8866.trans row_count_8866.symm
def row_8867 : RowCertificate 8867 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8867 1) := by change ¬ Nat.Prime 314512491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104837497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8867 2) := by change ¬ Nat.Prime 314494757; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2885273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8867 3) := by change ¬ Nat.Prime 314477023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44925289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8867 : row 8867 = 0 := (row_8867.checked).trans rfl
theorem model_count_8867 : modelRow 8867 = 0 := by decide
theorem modelrow_agrees_8867 : modelRow 8867 = row 8867 := model_count_8867.trans row_count_8867.symm
def row_8868 : RowCertificate 8868 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8868 1) := by change ¬ Nat.Prime 314583433; exact PrattCertificates.composite_of_product (factor := 14869) (cofactor := 21157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8868 2) := by change ¬ Nat.Prime 314565697; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10847093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8868 3) := by change ¬ Nat.Prime 314547961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44935423) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8868 : row 8868 = 0 := (row_8868.checked).trans rfl
theorem model_count_8868 : modelRow 8868 = 0 := by decide
theorem modelrow_agrees_8868 : modelRow 8868 = row 8868 := model_count_8868.trans row_count_8868.symm
def row_8869 : RowCertificate 8869 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8869 1) := by change ¬ Nat.Prime 314654383; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16560757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8869 2) := by change ¬ Nat.Prime 314636645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62927329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8869 3) := by change ¬ Nat.Prime 314618907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104872969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8869 : row 8869 = 0 := (row_8869.checked).trans rfl
theorem model_count_8869 : modelRow 8869 = 0 := by decide
theorem modelrow_agrees_8869 : modelRow 8869 = row 8869 := model_count_8869.trans row_count_8869.symm
def row_8870 : RowCertificate 8870 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8870 1) := by change ¬ Nat.Prime 314725341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104908447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8870 2) := by change ¬ Nat.Prime 314707601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24208277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8870 3) := by change Nat.Prime 314689861; exact ProjectEulerSolutions.P58PrimeNodes.p_314689861.prime
    exact decide_eq_true prime
theorem row_count_8870 : row 8870 = 1 := (row_8870.checked).trans rfl
theorem model_count_8870 : modelRow 8870 = 1 := by decide
theorem modelrow_agrees_8870 : modelRow 8870 = row 8870 := model_count_8870.trans row_count_8870.symm
def row_8871 : RowCertificate 8871 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8871 1) := by change ¬ Nat.Prime 314796307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44970901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8871 2) := by change ¬ Nat.Prime 314778565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62955713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8871 3) := by change ¬ Nat.Prime 314760823; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24212371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8871 : row 8871 = 0 := (row_8871.checked).trans rfl
theorem model_count_8871 : modelRow 8871 = 0 := by decide
theorem modelrow_agrees_8871 : modelRow 8871 = row 8871 := model_count_8871.trans row_count_8871.symm
def row_8872 : RowCertificate 8872 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8872 1) := by change Nat.Prime 314867281; exact ProjectEulerSolutions.P58PrimeNodes.p_314867281.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8872 2) := by change ¬ Nat.Prime 314849537; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18520561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8872 3) := by change ¬ Nat.Prime 314831793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104943931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8872 : row 8872 = 1 := (row_8872.checked).trans rfl
theorem model_count_8872 : modelRow 8872 = 1 := by decide
theorem modelrow_agrees_8872 : modelRow 8872 = row 8872 := model_count_8872.trans row_count_8872.symm
def row_8873 : RowCertificate 8873 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8873 1) := by change ¬ Nat.Prime 314938263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104979421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8873 2) := by change ¬ Nat.Prime 314920517; exact PrattCertificates.composite_of_product (factor := 12653) (cofactor := 24889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8873 3) := by change ¬ Nat.Prime 314902771; exact PrattCertificates.composite_of_product (factor := 12157) (cofactor := 25903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8873 : row 8873 = 0 := (row_8873.checked).trans rfl
theorem model_count_8873 : modelRow 8873 = 0 := by decide
theorem modelrow_agrees_8873 : modelRow 8873 = row 8873 := model_count_8873.trans row_count_8873.symm
def row_8874 : RowCertificate 8874 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8874 1) := by change ¬ Nat.Prime 315009253; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24231481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8874 2) := by change ¬ Nat.Prime 314991505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62998301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8874 3) := by change ¬ Nat.Prime 314973757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44996251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8874 : row 8874 = 0 := (row_8874.checked).trans rfl
theorem model_count_8874 : modelRow 8874 = 0 := by decide
theorem modelrow_agrees_8874 : modelRow 8874 = row 8874 := model_count_8874.trans row_count_8874.symm
def row_8875 : RowCertificate 8875 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8875 1) := by change Nat.Prime 315080251; exact ProjectEulerSolutions.P58PrimeNodes.p_315080251.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8875 2) := by change ¬ Nat.Prime 315062501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24235577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8875 3) := by change ¬ Nat.Prime 315044751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105014917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8875 : row 8875 = 1 := (row_8875.checked).trans rfl
theorem model_count_8875 : modelRow 8875 = 1 := by decide
theorem modelrow_agrees_8875 : modelRow 8875 = row 8875 := model_count_8875.trans row_count_8875.symm
def row_8876 : RowCertificate 8876 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8876 1) := by change ¬ Nat.Prime 315151257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105050419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8876 2) := by change ¬ Nat.Prime 315133505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63026701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8876 3) := by change ¬ Nat.Prime 315115753; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3988807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8876 : row 8876 = 0 := (row_8876.checked).trans rfl
theorem model_count_8876 : modelRow 8876 = 0 := by decide
theorem modelrow_agrees_8876 : modelRow 8876 = row 8876 := model_count_8876.trans row_count_8876.symm
def row_8877 : RowCertificate 8877 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8877 1) := by change ¬ Nat.Prime 315222271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45031753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8877 2) := by change ¬ Nat.Prime 315204517; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8519041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8877 3) := by change ¬ Nat.Prime 315186763; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16588777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8877 : row 8877 = 0 := (row_8877.checked).trans rfl
theorem model_count_8877 : modelRow 8877 = 0 := by decide
theorem modelrow_agrees_8877 : modelRow 8877 = row 8877 := model_count_8877.trans row_count_8877.symm
def row_8878 : RowCertificate 8878 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8878 1) := by change ¬ Nat.Prime 315293293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45041899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8878 2) := by change ¬ Nat.Prime 315275537; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2790049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8878 3) := by change ¬ Nat.Prime 315257781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105085927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8878 : row 8878 = 0 := (row_8878.checked).trans rfl
theorem model_count_8878 : modelRow 8878 = 0 := by decide
theorem modelrow_agrees_8878 : modelRow 8878 = row 8878 := model_count_8878.trans row_count_8878.symm
def row_8879 : RowCertificate 8879 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8879 1) := by change ¬ Nat.Prime 315364323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105121441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8879 2) := by change ¬ Nat.Prime 315346565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63069313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8879 3) := by change ¬ Nat.Prime 315328807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16596253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8879 : row 8879 = 0 := (row_8879.checked).trans rfl
theorem model_count_8879 : modelRow 8879 = 0 := by decide
theorem modelrow_agrees_8879 : modelRow 8879 = row 8879 := model_count_8879.trans row_count_8879.symm
def row_8880 : RowCertificate 8880 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8880 1) := by change ¬ Nat.Prime 315435361; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2483743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8880 2) := by change ¬ Nat.Prime 315417601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10876469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8880 3) := by change Nat.Prime 315399841; exact ProjectEulerSolutions.P58PrimeNodes.p_315399841.prime
    exact decide_eq_true prime
theorem row_count_8880 : row 8880 = 1 := (row_8880.checked).trans rfl
theorem model_count_8880 : modelRow 8880 = 1 := by decide
theorem modelrow_agrees_8880 : modelRow 8880 = row 8880 := model_count_8880.trans row_count_8880.symm
def row_8881 : RowCertificate 8881 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8881 1) := by change ¬ Nat.Prime 315506407; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3063169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8881 2) := by change ¬ Nat.Prime 315488645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63097729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8881 3) := by change ¬ Nat.Prime 315470883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105156961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8881 : row 8881 = 0 := (row_8881.checked).trans rfl
theorem model_count_8881 : modelRow 8881 = 0 := by decide
theorem modelrow_agrees_8881 : modelRow 8881 = row 8881 := model_count_8881.trans row_count_8881.symm
def row_8882 : RowCertificate 8882 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8882 1) := by change ¬ Nat.Prime 315577461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105192487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8882 2) := by change Nat.Prime 315559697; exact ProjectEulerSolutions.P58PrimeNodes.p_315559697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8882 3) := by change ¬ Nat.Prime 315541933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45077419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8882 : row 8882 = 1 := (row_8882.checked).trans rfl
theorem model_count_8882 : modelRow 8882 = 1 := by decide
theorem modelrow_agrees_8882 : modelRow 8882 = row 8882 := model_count_8882.trans row_count_8882.symm
def row_8883 : RowCertificate 8883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8883 1) := by change ¬ Nat.Prime 315648523; exact PrattCertificates.composite_of_product (factor := 1741) (cofactor := 181303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8883 2) := by change ¬ Nat.Prime 315630757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24279289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8883 3) := by change ¬ Nat.Prime 315612991; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7339837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8883 : row 8883 = 0 := (row_8883.checked).trans rfl
theorem model_count_8883 : modelRow 8883 = 0 := by decide
theorem modelrow_agrees_8883 : modelRow 8883 = row 8883 := model_count_8883.trans row_count_8883.symm
def row_8884 : RowCertificate 8884 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8884 1) := by change ¬ Nat.Prime 315719593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45102799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8884 2) := by change ¬ Nat.Prime 315701825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63140365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8884 3) := by change ¬ Nat.Prime 315684057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105228019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8884 : row 8884 = 0 := (row_8884.checked).trans rfl
theorem model_count_8884 : modelRow 8884 = 0 := by decide
theorem modelrow_agrees_8884 : modelRow 8884 = row 8884 := model_count_8884.trans row_count_8884.symm
def row_8885 : RowCertificate 8885 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8885 1) := by change ¬ Nat.Prime 315790671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105263557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8885 2) := by change ¬ Nat.Prime 315772901; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1310261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8885 3) := by change Nat.Prime 315755131; exact ProjectEulerSolutions.P58PrimeNodes.p_315755131.prime
    exact decide_eq_true prime
theorem row_count_8885 : row 8885 = 1 := (row_8885.checked).trans rfl
theorem model_count_8885 : modelRow 8885 = 1 := by decide
theorem modelrow_agrees_8885 : modelRow 8885 = row 8885 := model_count_8885.trans row_count_8885.symm
def row_8886 : RowCertificate 8886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8886 1) := by change ¬ Nat.Prime 315861757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16624303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8886 2) := by change ¬ Nat.Prime 315843985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63168797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8886 3) := by change ¬ Nat.Prime 315826213; exact PrattCertificates.composite_of_product (factor := 2971) (cofactor := 106303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8886 : row 8886 = 0 := (row_8886.checked).trans rfl
theorem model_count_8886 : modelRow 8886 = 0 := by decide
theorem modelrow_agrees_8886 : modelRow 8886 = row 8886 := model_count_8886.trans row_count_8886.symm
def row_8887 : RowCertificate 8887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8887 1) := by change ¬ Nat.Prime 315932851; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24302527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8887 2) := by change ¬ Nat.Prime 315915077; exact PrattCertificates.composite_of_product (factor := 9473) (cofactor := 33349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8887 3) := by change ¬ Nat.Prime 315897303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105299101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8887 : row 8887 = 0 := (row_8887.checked).trans rfl
theorem model_count_8887 : modelRow 8887 = 0 := by decide
theorem modelrow_agrees_8887 : modelRow 8887 = row 8887 := model_count_8887.trans row_count_8887.symm
def row_8888 : RowCertificate 8888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8888 1) := by change ¬ Nat.Prime 316003953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105334651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8888 2) := by change ¬ Nat.Prime 315986177; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24306629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8888 3) := by change ¬ Nat.Prime 315968401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45138343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8888 : row 8888 = 0 := (row_8888.checked).trans rfl
theorem model_count_8888 : modelRow 8888 = 0 := by decide
theorem modelrow_agrees_8888 : modelRow 8888 = row 8888 := model_count_8888.trans row_count_8888.symm
def row_8889 : RowCertificate 8889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8889 1) := by change ¬ Nat.Prime 316075063; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 367957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8889 2) := by change ¬ Nat.Prime 316057285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63211457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8889 3) := by change ¬ Nat.Prime 316039507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45148501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8889 : row 8889 = 0 := (row_8889.checked).trans rfl
theorem model_count_8889 : modelRow 8889 = 0 := by decide
theorem modelrow_agrees_8889 : modelRow 8889 = row 8889 := model_count_8889.trans row_count_8889.symm
def row_8890 : RowCertificate 8890 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8890 1) := by change ¬ Nat.Prime 316146181; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24318937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8890 2) := by change ¬ Nat.Prime 316128401; exact PrattCertificates.composite_of_product (factor := 653) (cofactor := 484117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8890 3) := by change ¬ Nat.Prime 316110621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105370207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8890 : row 8890 = 0 := (row_8890.checked).trans rfl
theorem model_count_8890 : modelRow 8890 = 0 := by decide
theorem modelrow_agrees_8890 : modelRow 8890 = row 8890 := model_count_8890.trans row_count_8890.symm
def row_8891 : RowCertificate 8891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8891 1) := by change ¬ Nat.Prime 316217307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105405769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8891 2) := by change ¬ Nat.Prime 316199525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63239905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8891 3) := by change Nat.Prime 316181743; exact ProjectEulerSolutions.P58PrimeNodes.p_316181743.prime
    exact decide_eq_true prime
theorem row_count_8891 : row 8891 = 1 := (row_8891.checked).trans rfl
theorem model_count_8891 : modelRow 8891 = 1 := by decide
theorem modelrow_agrees_8891 : modelRow 8891 = row 8891 := model_count_8891.trans row_count_8891.symm
def row_8892 : RowCertificate 8892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8892 1) := by change ¬ Nat.Prime 316288441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45184063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8892 2) := by change ¬ Nat.Prime 316270657; exact PrattCertificates.composite_of_product (factor := 2309) (cofactor := 136973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8892 3) := by change ¬ Nat.Prime 316252873; exact PrattCertificates.composite_of_product (factor := 6823) (cofactor := 46351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8892 : row 8892 = 0 := (row_8892.checked).trans rfl
theorem model_count_8892 : modelRow 8892 = 0 := by decide
theorem modelrow_agrees_8892 : modelRow 8892 = row 8892 := model_count_8892.trans row_count_8892.symm
def row_8893 : RowCertificate 8893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8893 1) := by change ¬ Nat.Prime 316359583; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8550259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8893 2) := by change ¬ Nat.Prime 316341797; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18608341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8893 3) := by change ¬ Nat.Prime 316324011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105441337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8893 : row 8893 = 0 := (row_8893.checked).trans rfl
theorem model_count_8893 : modelRow 8893 = 0 := by decide
theorem modelrow_agrees_8893 : modelRow 8893 = row 8893 := model_count_8893.trans row_count_8893.symm
def row_8894 : RowCertificate 8894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8894 1) := by change ¬ Nat.Prime 316430733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105476911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8894 2) := by change ¬ Nat.Prime 316412945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63282589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8894 3) := by change ¬ Nat.Prime 316395157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24338089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8894 : row 8894 = 0 := (row_8894.checked).trans rfl
theorem model_count_8894 : modelRow 8894 = 0 := by decide
theorem modelrow_agrees_8894 : modelRow 8894 = row 8894 := model_count_8894.trans row_count_8894.symm
def row_8895 : RowCertificate 8895 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8895 1) := by change Nat.Prime 316501891; exact ProjectEulerSolutions.P58PrimeNodes.p_316501891.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8895 2) := by change ¬ Nat.Prime 316484101; exact PrattCertificates.composite_of_product (factor := 857) (cofactor := 369293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8895 3) := by change ¬ Nat.Prime 316466311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45209473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8895 : row 8895 = 1 := (row_8895.checked).trans rfl
theorem model_count_8895 : modelRow 8895 = 1 := by decide
theorem modelrow_agrees_8895 : modelRow 8895 = row 8895 := model_count_8895.trans row_count_8895.symm
def row_8896 : RowCertificate 8896 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8896 1) := by change ¬ Nat.Prime 316573057; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4724971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8896 2) := by change ¬ Nat.Prime 316555265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63311053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8896 3) := by change ¬ Nat.Prime 316537473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105512491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8896 : row 8896 = 0 := (row_8896.checked).trans rfl
theorem model_count_8896 : modelRow 8896 = 0 := by decide
theorem modelrow_agrees_8896 : modelRow 8896 = row 8896 := model_count_8896.trans row_count_8896.symm
def row_8897 : RowCertificate 8897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8897 1) := by change ¬ Nat.Prime 316644231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105548077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8897 2) := by change ¬ Nat.Prime 316626437; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 10918153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8897 3) := by change ¬ Nat.Prime 316608643; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24354511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8897 : row 8897 = 0 := (row_8897.checked).trans rfl
theorem model_count_8897 : modelRow 8897 = 0 := by decide
theorem modelrow_agrees_8897 : modelRow 8897 = row 8897 := model_count_8897.trans row_count_8897.symm
def row_8898 : RowCertificate 8898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8898 1) := by change ¬ Nat.Prime 316715413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45245059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8898 2) := by change Nat.Prime 316697617; exact ProjectEulerSolutions.P58PrimeNodes.p_316697617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8898 3) := by change ¬ Nat.Prime 316679821; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16667359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8898 : row 8898 = 1 := (row_8898.checked).trans rfl
theorem model_count_8898 : modelRow 8898 = 1 := by decide
theorem modelrow_agrees_8898 : modelRow 8898 = row 8898 := model_count_8898.trans row_count_8898.symm
def row_8899 : RowCertificate 8899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8899 1) := by change ¬ Nat.Prime 316786603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45255229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8899 2) := by change ¬ Nat.Prime 316768805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63353761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8899 3) := by change ¬ Nat.Prime 316751007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105583669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8899 : row 8899 = 0 := (row_8899.checked).trans rfl
theorem model_count_8899 : modelRow 8899 = 0 := by decide
theorem modelrow_agrees_8899 : modelRow 8899 = row 8899 := model_count_8899.trans row_count_8899.symm
def row_8900 : RowCertificate 8900 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8900 1) := by change ¬ Nat.Prime 316857801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105619267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8900 2) := by change ¬ Nat.Prime 316840001; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2906789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8900 3) := by change ¬ Nat.Prime 316822201; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10220071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8900 : row 8900 = 0 := (row_8900.checked).trans rfl
theorem model_count_8900 : modelRow 8900 = 0 := by decide
theorem modelrow_agrees_8900 : modelRow 8900 = row 8900 := model_count_8900.trans row_count_8900.symm
def row_8901 : RowCertificate 8901 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8901 1) := by change Nat.Prime 316929007; exact ProjectEulerSolutions.P58PrimeNodes.p_316929007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8901 2) := by change ¬ Nat.Prime 316911205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63382241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8901 3) := by change Nat.Prime 316893403; exact ProjectEulerSolutions.P58PrimeNodes.p_316893403.prime
    exact decide_eq_true prime
theorem row_count_8901 : row 8901 = 2 := (row_8901.checked).trans rfl
theorem model_count_8901 : modelRow 8901 = 2 := by decide
theorem modelrow_agrees_8901 : modelRow 8901 = row 8901 := model_count_8901.trans row_count_8901.symm
def row_8902 : RowCertificate 8902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8902 1) := by change ¬ Nat.Prime 317000221; exact PrattCertificates.composite_of_product (factor := 3061) (cofactor := 103561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8902 2) := by change Nat.Prime 316982417; exact ProjectEulerSolutions.P58PrimeNodes.p_316982417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8902 3) := by change ¬ Nat.Prime 316964613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105654871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8902 : row 8902 = 1 := (row_8902.checked).trans rfl
theorem model_count_8902 : modelRow 8902 = 1 := by decide
theorem modelrow_agrees_8902 : modelRow 8902 = row 8902 := model_count_8902.trans row_count_8902.symm
def row_8903 : RowCertificate 8903 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8903 1) := by change ¬ Nat.Prime 317071443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105690481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8903 2) := by change ¬ Nat.Prime 317053637; exact PrattCertificates.composite_of_product (factor := 17389) (cofactor := 18233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8903 3) := by change ¬ Nat.Prime 317035831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45290833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8903 : row 8903 = 0 := (row_8903.checked).trans rfl
theorem model_count_8903 : modelRow 8903 = 0 := by decide
theorem modelrow_agrees_8903 : modelRow 8903 = row 8903 := model_count_8903.trans row_count_8903.symm
def row_8904 : RowCertificate 8904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8904 1) := by change ¬ Nat.Prime 317142673; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7375411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8904 2) := by change ¬ Nat.Prime 317124865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63424973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8904 3) := by change ¬ Nat.Prime 317107057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8570461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8904 : row 8904 = 0 := (row_8904.checked).trans rfl
theorem model_count_8904 : modelRow 8904 = 0 := by decide
theorem modelrow_agrees_8904 : modelRow 8904 = row 8904 := model_count_8904.trans row_count_8904.symm
def row_8905 : RowCertificate 8905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8905 1) := by change ¬ Nat.Prime 317213911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45316273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8905 2) := by change Nat.Prime 317196101; exact ProjectEulerSolutions.P58PrimeNodes.p_317196101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8905 3) := by change ¬ Nat.Prime 317178291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105726097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8905 : row 8905 = 1 := (row_8905.checked).trans rfl
theorem model_count_8905 : modelRow 8905 = 1 := by decide
theorem modelrow_agrees_8905 : modelRow 8905 = row 8905 := model_count_8905.trans row_count_8905.symm
def row_8906 : RowCertificate 8906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8906 1) := by change ¬ Nat.Prime 317285157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105761719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8906 2) := by change ¬ Nat.Prime 317267345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63453469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8906 3) := by change ¬ Nat.Prime 317249533; exact PrattCertificates.composite_of_product (factor := 11059) (cofactor := 28687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8906 : row 8906 = 0 := (row_8906.checked).trans rfl
theorem model_count_8906 : modelRow 8906 = 0 := by decide
theorem modelrow_agrees_8906 : modelRow 8906 = row 8906 := model_count_8906.trans row_count_8906.symm
def row_8907 : RowCertificate 8907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8907 1) := by change ¬ Nat.Prime 317356411; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16702969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8907 2) := by change ¬ Nat.Prime 317338597; exact PrattCertificates.composite_of_product (factor := 3733) (cofactor := 85009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8907 3) := by change ¬ Nat.Prime 317320783; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24409291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8907 : row 8907 = 0 := (row_8907.checked).trans rfl
theorem model_count_8907 : modelRow 8907 = 0 := by decide
theorem modelrow_agrees_8907 : modelRow 8907 = row 8907 := model_count_8907.trans row_count_8907.symm
def row_8908 : RowCertificate 8908 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8908 1) := by change Nat.Prime 317427673; exact ProjectEulerSolutions.P58PrimeNodes.p_317427673.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8908 2) := by change ¬ Nat.Prime 317409857; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1014089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8908 3) := by change ¬ Nat.Prime 317392041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105797347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8908 : row 8908 = 1 := (row_8908.checked).trans rfl
theorem model_count_8908 : modelRow 8908 = 1 := by decide
theorem modelrow_agrees_8908 : modelRow 8908 = row 8908 := model_count_8908.trans row_count_8908.symm
def row_8909 : RowCertificate 8909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8909 1) := by change ¬ Nat.Prime 317498943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105832981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8909 2) := by change ¬ Nat.Prime 317481125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63496225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8909 3) := by change ¬ Nat.Prime 317463307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45351901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8909 : row 8909 = 0 := (row_8909.checked).trans rfl
theorem model_count_8909 : modelRow 8909 = 0 := by decide
theorem modelrow_agrees_8909 : modelRow 8909 = row 8909 := model_count_8909.trans row_count_8909.symm
def row_8910 : RowCertificate 8910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8910 1) := by change ¬ Nat.Prime 317570221; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4350277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8910 2) := by change ¬ Nat.Prime 317552401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18679553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8910 3) := by change ¬ Nat.Prime 317534581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45362083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8910 : row 8910 = 0 := (row_8910.checked).trans rfl
theorem model_count_8910 : modelRow 8910 = 0 := by decide
theorem modelrow_agrees_8910 : modelRow 8910 = row 8910 := model_count_8910.trans row_count_8910.symm
def row_8911 : RowCertificate 8911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8911 1) := by change ¬ Nat.Prime 317641507; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 362191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8911 2) := by change ¬ Nat.Prime 317623685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63524737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8911 3) := by change ¬ Nat.Prime 317605863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105868621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8911 : row 8911 = 0 := (row_8911.checked).trans rfl
theorem model_count_8911 : modelRow 8911 = 0 := by decide
theorem modelrow_agrees_8911 : modelRow 8911 = row 8911 := model_count_8911.trans row_count_8911.symm
def row_8912 : RowCertificate 8912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8912 1) := by change ¬ Nat.Prime 317712801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105904267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8912 2) := by change Nat.Prime 317694977; exact ProjectEulerSolutions.P58PrimeNodes.p_317694977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8912 3) := by change ¬ Nat.Prime 317677153; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8585869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8912 : row 8912 = 1 := (row_8912.checked).trans rfl
theorem model_count_8912 : modelRow 8912 = 1 := by decide
theorem modelrow_agrees_8912 : modelRow 8912 = row 8912 := model_count_8912.trans row_count_8912.symm
def row_8913 : RowCertificate 8913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8913 1) := by change ¬ Nat.Prime 317784103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45397729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8913 2) := by change ¬ Nat.Prime 317766277; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 7750397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8913 3) := by change ¬ Nat.Prime 317748451; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5208991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8913 : row 8913 = 0 := (row_8913.checked).trans rfl
theorem model_count_8913 : modelRow 8913 = 0 := by decide
theorem modelrow_agrees_8913 : modelRow 8913 = row 8913 := model_count_8913.trans row_count_8913.symm
def row_8914 : RowCertificate 8914 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8914 1) := by change Nat.Prime 317855413; exact ProjectEulerSolutions.P58PrimeNodes.p_317855413.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8914 2) := by change ¬ Nat.Prime 317837585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63567517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8914 3) := by change ¬ Nat.Prime 317819757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105939919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8914 : row 8914 = 1 := (row_8914.checked).trans rfl
theorem model_count_8914 : modelRow 8914 = 1 := by decide
theorem modelrow_agrees_8914 : modelRow 8914 = row 8914 := model_count_8914.trans row_count_8914.symm
def row_8915 : RowCertificate 8915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8915 1) := by change ¬ Nat.Prime 317926731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105975577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8915 2) := by change Nat.Prime 317908901; exact ProjectEulerSolutions.P58PrimeNodes.p_317908901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8915 3) := by change ¬ Nat.Prime 317891071; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16731109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8915 : row 8915 = 1 := (row_8915.checked).trans rfl
theorem model_count_8915 : modelRow 8915 = 1 := by decide
theorem modelrow_agrees_8915 : modelRow 8915 = row 8915 := model_count_8915.trans row_count_8915.symm
def row_8916 : RowCertificate 8916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8916 1) := by change ¬ Nat.Prime 317998057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24461389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8916 2) := by change ¬ Nat.Prime 317980225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63596045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8916 3) := by change ¬ Nat.Prime 317962393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45423199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8916 : row 8916 = 0 := (row_8916.checked).trans rfl
theorem model_count_8916 : modelRow 8916 = 0 := by decide
theorem modelrow_agrees_8916 : modelRow 8916 = row 8916 := model_count_8916.trans row_count_8916.symm
def row_8917 : RowCertificate 8917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8917 1) := by change ¬ Nat.Prime 318069391; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2288269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8917 2) := by change ¬ Nat.Prime 318051557; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3573613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8917 3) := by change ¬ Nat.Prime 318033723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106011241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8917 : row 8917 = 0 := (row_8917.checked).trans rfl
theorem model_count_8917 : modelRow 8917 = 0 := by decide
theorem modelrow_agrees_8917 : modelRow 8917 = row 8917 := model_count_8917.trans row_count_8917.symm
def row_8918 : RowCertificate 8918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8918 1) := by change ¬ Nat.Prime 318140733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106046911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8918 2) := by change ¬ Nat.Prime 318122897; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2135053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8918 3) := by change ¬ Nat.Prime 318105061; exact PrattCertificates.composite_of_product (factor := 1867) (cofactor := 170383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8918 : row 8918 = 0 := (row_8918.checked).trans rfl
theorem model_count_8918 : modelRow 8918 = 0 := by decide
theorem modelrow_agrees_8918 : modelRow 8918 = row 8918 := model_count_8918.trans row_count_8918.symm
def row_8919 : RowCertificate 8919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8919 1) := by change ¬ Nat.Prime 318212083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45458869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8919 2) := by change ¬ Nat.Prime 318194245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63638849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8919 3) := by change Nat.Prime 318176407; exact ProjectEulerSolutions.P58PrimeNodes.p_318176407.prime
    exact decide_eq_true prime
theorem row_count_8919 : row 8919 = 1 := (row_8919.checked).trans rfl
theorem model_count_8919 : modelRow 8919 = 1 := by decide
theorem modelrow_agrees_8919 : modelRow 8919 = row 8919 := model_count_8919.trans row_count_8919.symm
def row_8920 : RowCertificate 8920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8920 1) := by change ¬ Nat.Prime 318283441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45469063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8920 2) := by change ¬ Nat.Prime 318265601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8601773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8920 3) := by change ¬ Nat.Prime 318247761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106082587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8920 : row 8920 = 0 := (row_8920.checked).trans rfl
theorem model_count_8920 : modelRow 8920 = 0 := by decide
theorem modelrow_agrees_8920 : modelRow 8920 = row 8920 := model_count_8920.trans row_count_8920.symm
def row_8921 : RowCertificate 8921 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8921 1) := by change ¬ Nat.Prime 318354807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106118269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8921 2) := by change ¬ Nat.Prime 318336965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63667393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8921 3) := by change ¬ Nat.Prime 318319123; exact PrattCertificates.composite_of_product (factor := 1747) (cofactor := 182209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8921 : row 8921 = 0 := (row_8921.checked).trans rfl
theorem model_count_8921 : modelRow 8921 = 0 := by decide
theorem modelrow_agrees_8921 : modelRow 8921 = row 8921 := model_count_8921.trans row_count_8921.symm
def row_8922 : RowCertificate 8922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8922 1) := by change ¬ Nat.Prime 318426181; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8606113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8922 2) := by change ¬ Nat.Prime 318408337; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24492949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8922 3) := by change ¬ Nat.Prime 318390493; exact PrattCertificates.composite_of_product (factor := 3517) (cofactor := 90529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8922 : row 8922 = 0 := (row_8922.checked).trans rfl
theorem model_count_8922 : modelRow 8922 = 0 := by decide
theorem modelrow_agrees_8922 : modelRow 8922 = row 8922 := model_count_8922.trans row_count_8922.symm
def row_8923 : RowCertificate 8923 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8923 1) := by change Nat.Prime 318497563; exact ProjectEulerSolutions.P58PrimeNodes.p_318497563.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8923 2) := by change ¬ Nat.Prime 318479717; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18734101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8923 3) := by change ¬ Nat.Prime 318461871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106153957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8923 : row 8923 = 1 := (row_8923.checked).trans rfl
theorem model_count_8923 : modelRow 8923 = 1 := by decide
theorem modelrow_agrees_8923 : modelRow 8923 = row 8923 := model_count_8923.trans row_count_8923.symm
def row_8924 : RowCertificate 8924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8924 1) := by change ¬ Nat.Prime 318568953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106189651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8924 2) := by change ¬ Nat.Prime 318551105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63710221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8924 3) := by change ¬ Nat.Prime 318533257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45504751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8924 : row 8924 = 0 := (row_8924.checked).trans rfl
theorem model_count_8924 : modelRow 8924 = 0 := by decide
theorem modelrow_agrees_8924 : modelRow 8924 = row 8924 := model_count_8924.trans row_count_8924.symm
def row_8925 : RowCertificate 8925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8925 1) := by change ¬ Nat.Prime 318640351; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10278721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8925 2) := by change Nat.Prime 318622501; exact ProjectEulerSolutions.P58PrimeNodes.p_318622501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8925 3) := by change ¬ Nat.Prime 318604651; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1322011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8925 : row 8925 = 1 := (row_8925.checked).trans rfl
theorem model_count_8925 : modelRow 8925 = 1 := by decide
theorem modelrow_agrees_8925 : modelRow 8925 = row 8925 := model_count_8925.trans row_count_8925.symm
def row_8926 : RowCertificate 8926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8926 1) := by change ¬ Nat.Prime 318711757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45530251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8926 2) := by change ¬ Nat.Prime 318693905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63738781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8926 3) := by change ¬ Nat.Prime 318676053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106225351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8926 : row 8926 = 0 := (row_8926.checked).trans rfl
theorem model_count_8926 : modelRow 8926 = 0 := by decide
theorem modelrow_agrees_8926 : modelRow 8926 = row 8926 := model_count_8926.trans row_count_8926.symm
def row_8927 : RowCertificate 8927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8927 1) := by change ¬ Nat.Prime 318783171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106261057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8927 2) := by change ¬ Nat.Prime 318765317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24520409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8927 3) := by change Nat.Prime 318747463; exact ProjectEulerSolutions.P58PrimeNodes.p_318747463.prime
    exact decide_eq_true prime
theorem row_count_8927 : row 8927 = 1 := (row_8927.checked).trans rfl
theorem model_count_8927 : modelRow 8927 = 1 := by decide
theorem modelrow_agrees_8927 : modelRow 8927 = row 8927 := model_count_8927.trans row_count_8927.symm
def row_8928 : RowCertificate 8928 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8928 1) := by change Nat.Prime 318854593; exact ProjectEulerSolutions.P58PrimeNodes.p_318854593.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8928 2) := by change Nat.Prime 318836737; exact ProjectEulerSolutions.P58PrimeNodes.p_318836737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8928 3) := by change ¬ Nat.Prime 318818881; exact PrattCertificates.composite_of_product (factor := 1423) (cofactor := 224047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8928 : row 8928 = 2 := (row_8928.checked).trans rfl
theorem model_count_8928 : modelRow 8928 = 2 := by decide
theorem modelrow_agrees_8928 : modelRow 8928 = row 8928 := model_count_8928.trans row_count_8928.symm
def row_8929 : RowCertificate 8929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8929 1) := by change ¬ Nat.Prime 318926023; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24532771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8929 2) := by change ¬ Nat.Prime 318908165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63781633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8929 3) := by change ¬ Nat.Prime 318890307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106296769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8929 : row 8929 = 0 := (row_8929.checked).trans rfl
theorem model_count_8929 : modelRow 8929 = 0 := by decide
theorem modelrow_agrees_8929 : modelRow 8929 = row 8929 := model_count_8929.trans row_count_8929.symm
def row_8930 : RowCertificate 8930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8930 1) := by change ¬ Nat.Prime 318997461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106332487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8930 2) := by change Nat.Prime 318979601; exact ProjectEulerSolutions.P58PrimeNodes.p_318979601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8930 3) := by change ¬ Nat.Prime 318961741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45565963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8930 : row 8930 = 1 := (row_8930.checked).trans rfl
theorem model_count_8930 : modelRow 8930 = 1 := by decide
theorem modelrow_agrees_8930 : modelRow 8930 = row 8930 := model_count_8930.trans row_count_8930.symm
def row_8931 : RowCertificate 8931 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8931 1) := by change Nat.Prime 319068907; exact ProjectEulerSolutions.P58PrimeNodes.p_319068907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8931 2) := by change ¬ Nat.Prime 319051045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63810209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8931 3) := by change ¬ Nat.Prime 319033183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45576169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8931 : row 8931 = 1 := (row_8931.checked).trans rfl
theorem model_count_8931 : modelRow 8931 = 1 := by decide
theorem modelrow_agrees_8931 : modelRow 8931 = row 8931 := model_count_8931.trans row_count_8931.symm
def row_8932 : RowCertificate 8932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8932 1) := by change ¬ Nat.Prime 319140361; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1653577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8932 2) := by change ¬ Nat.Prime 319122497; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 257981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8932 3) := by change ¬ Nat.Prime 319104633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106368211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8932 : row 8932 = 0 := (row_8932.checked).trans rfl
theorem model_count_8932 : modelRow 8932 = 0 := by decide
theorem modelrow_agrees_8932 : modelRow 8932 = row 8932 := model_count_8932.trans row_count_8932.symm
def row_8933 : RowCertificate 8933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8933 1) := by change ¬ Nat.Prime 319211823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106403941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8933 2) := by change ¬ Nat.Prime 319193957; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1019789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8933 3) := by change ¬ Nat.Prime 319176091; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24552007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8933 : row 8933 = 0 := (row_8933.checked).trans rfl
theorem model_count_8933 : modelRow 8933 = 0 := by decide
theorem modelrow_agrees_8933 : modelRow 8933 = row 8933 := model_count_8933.trans row_count_8933.symm
def row_8934 : RowCertificate 8934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8934 1) := by change ¬ Nat.Prime 319283293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45611899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8934 2) := by change ¬ Nat.Prime 319265425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63853085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8934 3) := by change ¬ Nat.Prime 319247557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16802503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8934 : row 8934 = 0 := (row_8934.checked).trans rfl
theorem model_count_8934 : modelRow 8934 = 0 := by decide
theorem modelrow_agrees_8934 : modelRow 8934 = row 8934 := model_count_8934.trans row_count_8934.symm
def row_8935 : RowCertificate 8935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8935 1) := by change ¬ Nat.Prime 319354771; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1764391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8935 2) := by change ¬ Nat.Prime 319336901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24564377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8935 3) := by change ¬ Nat.Prime 319319031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106439677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8935 : row 8935 = 0 := (row_8935.checked).trans rfl
theorem model_count_8935 : modelRow 8935 = 0 := by decide
theorem modelrow_agrees_8935 : modelRow 8935 = row 8935 := model_count_8935.trans row_count_8935.symm
def row_8936 : RowCertificate 8936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8936 1) := by change ¬ Nat.Prime 319426257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106475419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8936 2) := by change ¬ Nat.Prime 319408385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63881677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8936 3) := by change ¬ Nat.Prime 319390513; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24568501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8936 : row 8936 = 0 := (row_8936.checked).trans rfl
theorem model_count_8936 : modelRow 8936 = 0 := by decide
theorem modelrow_agrees_8936 : modelRow 8936 = row 8936 := model_count_8936.trans row_count_8936.symm
def row_8937 : RowCertificate 8937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8937 1) := by change ¬ Nat.Prime 319497751; exact PrattCertificates.composite_of_product (factor := 3019) (cofactor := 105829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8937 2) := by change ¬ Nat.Prime 319479877; exact PrattCertificates.composite_of_product (factor := 2609) (cofactor := 122453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8937 3) := by change ¬ Nat.Prime 319462003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45637429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8937 : row 8937 = 0 := (row_8937.checked).trans rfl
theorem model_count_8937 : modelRow 8937 = 0 := by decide
theorem modelrow_agrees_8937 : modelRow 8937 = row 8937 := model_count_8937.trans row_count_8937.symm
def row_8938 : RowCertificate 8938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8938 1) := by change ¬ Nat.Prime 319569253; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4377661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8938 2) := by change ¬ Nat.Prime 319551377; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11019013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8938 3) := by change ¬ Nat.Prime 319533501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106511167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8938 : row 8938 = 0 := (row_8938.checked).trans rfl
theorem model_count_8938 : modelRow 8938 = 0 := by decide
theorem modelrow_agrees_8938 : modelRow 8938 = row 8938 := model_count_8938.trans row_count_8938.symm
def row_8939 : RowCertificate 8939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8939 1) := by change ¬ Nat.Prime 319640763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106546921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8939 2) := by change ¬ Nat.Prime 319622885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63924577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8939 3) := by change ¬ Nat.Prime 319605007; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4045633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8939 : row 8939 = 0 := (row_8939.checked).trans rfl
theorem model_count_8939 : modelRow 8939 = 0 := by decide
theorem modelrow_agrees_8939 : modelRow 8939 = row 8939 := model_count_8939.trans row_count_8939.symm
def row_8940 : RowCertificate 8940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8940 1) := by change ¬ Nat.Prime 319712281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45673183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8940 2) := by change ¬ Nat.Prime 319694401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24591877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8940 3) := by change ¬ Nat.Prime 319676521; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1433527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8940 : row 8940 = 0 := (row_8940.checked).trans rfl
theorem model_count_8940 : modelRow 8940 = 0 := by decide
theorem modelrow_agrees_8940 : modelRow 8940 = row 8940 := model_count_8940.trans row_count_8940.symm
def row_8941 : RowCertificate 8941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8941 1) := by change ¬ Nat.Prime 319783807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45683401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8941 2) := by change ¬ Nat.Prime 319765925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63953185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8941 3) := by change ¬ Nat.Prime 319748043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106582681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8941 : row 8941 = 0 := (row_8941.checked).trans rfl
theorem model_count_8941 : modelRow 8941 = 0 := by decide
theorem modelrow_agrees_8941 : modelRow 8941 = row 8941 := model_count_8941.trans row_count_8941.symm
def row_8942 : RowCertificate 8942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8942 1) := by change ¬ Nat.Prime 319855341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106618447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8942 2) := by change ¬ Nat.Prime 319837457; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6034669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8942 3) := by change ¬ Nat.Prime 319819573; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3297109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8942 : row 8942 = 0 := (row_8942.checked).trans rfl
theorem model_count_8942 : modelRow 8942 = 0 := by decide
theorem modelrow_agrees_8942 : modelRow 8942 = row 8942 := model_count_8942.trans row_count_8942.symm
def row_8943 : RowCertificate 8943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8943 1) := by change ¬ Nat.Prime 319926883; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16838257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8943 2) := by change Nat.Prime 319908997; exact ProjectEulerSolutions.P58PrimeNodes.p_319908997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8943 3) := by change ¬ Nat.Prime 319891111; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2037523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8943 : row 8943 = 1 := (row_8943.checked).trans rfl
theorem model_count_8943 : modelRow 8943 = 1 := by decide
theorem modelrow_agrees_8943 : modelRow 8943 = row 8943 := model_count_8943.trans row_count_8943.symm
def row_8944 : RowCertificate 8944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8944 1) := by change ¬ Nat.Prime 319998433; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2302147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8944 2) := by change ¬ Nat.Prime 319980545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63996109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8944 3) := by change ¬ Nat.Prime 319962657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106654219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8944 : row 8944 = 0 := (row_8944.checked).trans rfl
theorem model_count_8944 : modelRow 8944 = 0 := by decide
theorem modelrow_agrees_8944 : modelRow 8944 = row 8944 := model_count_8944.trans row_count_8944.symm
def row_8945 : RowCertificate 8945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8945 1) := by change ¬ Nat.Prime 320069991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106689997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8945 2) := by change Nat.Prime 320052101; exact ProjectEulerSolutions.P58PrimeNodes.p_320052101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8945 3) := by change ¬ Nat.Prime 320034211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45719173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8945 : row 8945 = 1 := (row_8945.checked).trans rfl
theorem model_count_8945 : modelRow 8945 = 1 := by decide
theorem modelrow_agrees_8945 : modelRow 8945 = row 8945 := model_count_8945.trans row_count_8945.symm
def row_8946 : RowCertificate 8946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8946 1) := by change ¬ Nat.Prime 320141557; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10327147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8946 2) := by change ¬ Nat.Prime 320123665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64024733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8946 3) := by change ¬ Nat.Prime 320105773; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24623521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8946 : row 8946 = 0 := (row_8946.checked).trans rfl
theorem model_count_8946 : modelRow 8946 = 0 := by decide
theorem modelrow_agrees_8946 : modelRow 8946 = row 8946 := model_count_8946.trans row_count_8946.symm
def row_8947 : RowCertificate 8947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8947 1) := by change ¬ Nat.Prime 320213131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45744733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8947 2) := by change ¬ Nat.Prime 320195237; exact PrattCertificates.composite_of_product (factor := 2273) (cofactor := 140869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8947 3) := by change ¬ Nat.Prime 320177343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106725781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8947 : row 8947 = 0 := (row_8947.checked).trans rfl
theorem model_count_8947 : modelRow 8947 = 0 := by decide
theorem modelrow_agrees_8947 : modelRow 8947 = row 8947 := model_count_8947.trans row_count_8947.symm
def row_8948 : RowCertificate 8948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8948 1) := by change ¬ Nat.Prime 320284713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106761571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8948 2) := by change ¬ Nat.Prime 320266817; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24635909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8948 3) := by change Nat.Prime 320248921; exact ProjectEulerSolutions.P58PrimeNodes.p_320248921.prime
    exact decide_eq_true prime
theorem row_count_8948 : row 8948 = 1 := (row_8948.checked).trans rfl
theorem model_count_8948 : modelRow 8948 = 1 := by decide
theorem modelrow_agrees_8948 : modelRow 8948 = row 8948 := model_count_8948.trans row_count_8948.symm
def row_8949 : RowCertificate 8949 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8949 1) := by change Nat.Prime 320356303; exact ProjectEulerSolutions.P58PrimeNodes.p_320356303.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8949 2) := by change ¬ Nat.Prime 320338405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64067681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8949 3) := by change ¬ Nat.Prime 320320507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24640039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8949 : row 8949 = 1 := (row_8949.checked).trans rfl
theorem model_count_8949 : modelRow 8949 = 1 := by decide
theorem modelrow_agrees_8949 : modelRow 8949 = row 8949 := model_count_8949.trans row_count_8949.symm
def row_8950 : RowCertificate 8950 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8950 1) := by change Nat.Prime 320427901; exact ProjectEulerSolutions.P58PrimeNodes.p_320427901.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8950 2) := by change ¬ Nat.Prime 320410001; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1770221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8950 3) := by change ¬ Nat.Prime 320392101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106797367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8950 : row 8950 = 1 := (row_8950.checked).trans rfl
theorem model_count_8950 : modelRow 8950 = 1 := by decide
theorem modelrow_agrees_8950 : modelRow 8950 = row 8950 := model_count_8950.trans row_count_8950.symm
def row_8951 : RowCertificate 8951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8951 1) := by change ¬ Nat.Prime 320499507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106833169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8951 2) := by change ¬ Nat.Prime 320481605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64096321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8951 3) := by change ¬ Nat.Prime 320463703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45780529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8951 : row 8951 = 0 := (row_8951.checked).trans rfl
theorem model_count_8951 : modelRow 8951 = 0 := by decide
theorem modelrow_agrees_8951 : modelRow 8951 = row 8951 := model_count_8951.trans row_count_8951.symm
def row_8952 : RowCertificate 8952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8952 1) := by change ¬ Nat.Prime 320571121; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24659317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8952 2) := by change Nat.Prime 320553217; exact ProjectEulerSolutions.P58PrimeNodes.p_320553217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8952 3) := by change ¬ Nat.Prime 320535313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45790759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8952 : row 8952 = 1 := (row_8952.checked).trans rfl
theorem model_count_8952 : modelRow 8952 = 1 := by decide
theorem modelrow_agrees_8952 : modelRow 8952 = row 8952 := model_count_8952.trans row_count_8952.symm
def row_8953 : RowCertificate 8953 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8953 1) := by change Nat.Prime 320642743; exact ProjectEulerSolutions.P58PrimeNodes.p_320642743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8953 2) := by change ¬ Nat.Prime 320624837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24663449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8953 3) := by change ¬ Nat.Prime 320606931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106868977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8953 : row 8953 = 1 := (row_8953.checked).trans rfl
theorem model_count_8953 : modelRow 8953 = 1 := by decide
theorem modelrow_agrees_8953 : modelRow 8953 = row 8953 := model_count_8953.trans row_count_8953.symm
def row_8954 : RowCertificate 8954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8954 1) := by change ¬ Nat.Prime 320714373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106904791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8954 2) := by change ¬ Nat.Prime 320696465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64139293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8954 3) := by change ¬ Nat.Prime 320678557; exact PrattCertificates.composite_of_product (factor := 2551) (cofactor := 125707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8954 : row 8954 = 0 := (row_8954.checked).trans rfl
theorem model_count_8954 : modelRow 8954 = 0 := by decide
theorem modelrow_agrees_8954 : modelRow 8954 = row 8954 := model_count_8954.trans row_count_8954.symm
def row_8955 : RowCertificate 8955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8955 1) := by change ¬ Nat.Prime 320786011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45826573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8955 2) := by change ¬ Nat.Prime 320768101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11060969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8955 3) := by change ¬ Nat.Prime 320750191; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16881589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8955 : row 8955 = 0 := (row_8955.checked).trans rfl
theorem model_count_8955 : modelRow 8955 = 0 := by decide
theorem modelrow_agrees_8955 : modelRow 8955 = row 8955 := model_count_8955.trans row_count_8955.symm
def row_8956 : RowCertificate 8956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8956 1) := by change ¬ Nat.Prime 320857657; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10350247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8956 2) := by change ¬ Nat.Prime 320839745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64167949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8956 3) := by change ¬ Nat.Prime 320821833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106940611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8956 : row 8956 = 0 := (row_8956.checked).trans rfl
theorem model_count_8956 : modelRow 8956 = 0 := by decide
theorem modelrow_agrees_8956 : modelRow 8956 = row 8956 := model_count_8956.trans row_count_8956.symm
def row_8957 : RowCertificate 8957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8957 1) := by change ¬ Nat.Prime 320929311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106976437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8957 2) := by change ¬ Nat.Prime 320911397; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18877141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8957 3) := by change Nat.Prime 320893483; exact ProjectEulerSolutions.P58PrimeNodes.p_320893483.prime
    exact decide_eq_true prime
theorem row_count_8957 : row 8957 = 1 := (row_8957.checked).trans rfl
theorem model_count_8957 : modelRow 8957 = 1 := by decide
theorem modelrow_agrees_8957 : modelRow 8957 = row 8957 := model_count_8957.trans row_count_8957.symm
def row_8958 : RowCertificate 8958 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8958 1) := by change Nat.Prime 321000973; exact ProjectEulerSolutions.P58PrimeNodes.p_321000973.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8958 2) := by change ¬ Nat.Prime 320983057; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 310729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8958 3) := by change ¬ Nat.Prime 320965141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45852163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8958 : row 8958 = 1 := (row_8958.checked).trans rfl
theorem model_count_8958 : modelRow 8958 = 1 := by decide
theorem modelrow_agrees_8958 : modelRow 8958 = row 8958 := model_count_8958.trans row_count_8958.symm
def row_8959 : RowCertificate 8959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8959 1) := by change ¬ Nat.Prime 321072643; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8677639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8959 2) := by change ¬ Nat.Prime 321054725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64210945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8959 3) := by change ¬ Nat.Prime 321036807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107012269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8959 : row 8959 = 0 := (row_8959.checked).trans rfl
theorem model_count_8959 : modelRow 8959 = 0 := by decide
theorem modelrow_agrees_8959 : modelRow 8959 = row 8959 := model_count_8959.trans row_count_8959.symm
def row_8960 : RowCertificate 8960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8960 1) := by change ¬ Nat.Prime 321144321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107048107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8960 2) := by change ¬ Nat.Prime 321126401; exact PrattCertificates.composite_of_product (factor := 3253) (cofactor := 98717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8960 3) := by change ¬ Nat.Prime 321108481; exact PrattCertificates.composite_of_product (factor := 5881) (cofactor := 54601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8960 : row 8960 = 0 := (row_8960.checked).trans rfl
theorem model_count_8960 : modelRow 8960 = 0 := by decide
theorem modelrow_agrees_8960 : modelRow 8960 = row 8960 := model_count_8960.trans row_count_8960.symm
private def atomic_8833 : CountSegment 1 10 13120 8832 8833 3694 3694 := CountSegment.single (first := 8832) (initial := 3694) row_count_8833 (by decide)
private theorem atomic_8833_agreement : RowAgreement 8832 8833 := RowAgreement.single (first := 8832) modelrow_agrees_8833
private def atomic_8834 : CountSegment 1 10 13120 8833 8834 3694 3694 := CountSegment.single (first := 8833) (initial := 3694) row_count_8834 (by decide)
private theorem atomic_8834_agreement : RowAgreement 8833 8834 := RowAgreement.single (first := 8833) modelrow_agrees_8834
private def atomic_8835 : CountSegment 1 10 13120 8834 8835 3694 3695 := CountSegment.single (first := 8834) (initial := 3694) row_count_8835 (by decide)
private theorem atomic_8835_agreement : RowAgreement 8834 8835 := RowAgreement.single (first := 8834) modelrow_agrees_8835
private def atomic_8836 : CountSegment 1 10 13120 8835 8836 3695 3695 := CountSegment.single (first := 8835) (initial := 3695) row_count_8836 (by decide)
private theorem atomic_8836_agreement : RowAgreement 8835 8836 := RowAgreement.single (first := 8835) modelrow_agrees_8836
private def atomic_8837 : CountSegment 1 10 13120 8836 8837 3695 3696 := CountSegment.single (first := 8836) (initial := 3695) row_count_8837 (by decide)
private theorem atomic_8837_agreement : RowAgreement 8836 8837 := RowAgreement.single (first := 8836) modelrow_agrees_8837
private def atomic_8838 : CountSegment 1 10 13120 8837 8838 3696 3696 := CountSegment.single (first := 8837) (initial := 3696) row_count_8838 (by decide)
private theorem atomic_8838_agreement : RowAgreement 8837 8838 := RowAgreement.single (first := 8837) modelrow_agrees_8838
private def atomic_8839 : CountSegment 1 10 13120 8838 8839 3696 3697 := CountSegment.single (first := 8838) (initial := 3696) row_count_8839 (by decide)
private theorem atomic_8839_agreement : RowAgreement 8838 8839 := RowAgreement.single (first := 8838) modelrow_agrees_8839
private def atomic_8840 : CountSegment 1 10 13120 8839 8840 3697 3697 := CountSegment.single (first := 8839) (initial := 3697) row_count_8840 (by decide)
private theorem atomic_8840_agreement : RowAgreement 8839 8840 := RowAgreement.single (first := 8839) modelrow_agrees_8840
private def atomic_8841 : CountSegment 1 10 13120 8840 8841 3697 3697 := CountSegment.single (first := 8840) (initial := 3697) row_count_8841 (by decide)
private theorem atomic_8841_agreement : RowAgreement 8840 8841 := RowAgreement.single (first := 8840) modelrow_agrees_8841
private def atomic_8842 : CountSegment 1 10 13120 8841 8842 3697 3697 := CountSegment.single (first := 8841) (initial := 3697) row_count_8842 (by decide)
private theorem atomic_8842_agreement : RowAgreement 8841 8842 := RowAgreement.single (first := 8841) modelrow_agrees_8842
private def atomic_8843 : CountSegment 1 10 13120 8842 8843 3697 3698 := CountSegment.single (first := 8842) (initial := 3697) row_count_8843 (by decide)
private theorem atomic_8843_agreement : RowAgreement 8842 8843 := RowAgreement.single (first := 8842) modelrow_agrees_8843
private def atomic_8844 : CountSegment 1 10 13120 8843 8844 3698 3699 := CountSegment.single (first := 8843) (initial := 3698) row_count_8844 (by decide)
private theorem atomic_8844_agreement : RowAgreement 8843 8844 := RowAgreement.single (first := 8843) modelrow_agrees_8844
private def atomic_8845 : CountSegment 1 10 13120 8844 8845 3699 3701 := CountSegment.single (first := 8844) (initial := 3699) row_count_8845 (by decide)
private theorem atomic_8845_agreement : RowAgreement 8844 8845 := RowAgreement.single (first := 8844) modelrow_agrees_8845
private def atomic_8846 : CountSegment 1 10 13120 8845 8846 3701 3701 := CountSegment.single (first := 8845) (initial := 3701) row_count_8846 (by decide)
private theorem atomic_8846_agreement : RowAgreement 8845 8846 := RowAgreement.single (first := 8845) modelrow_agrees_8846
private def atomic_8847 : CountSegment 1 10 13120 8846 8847 3701 3703 := CountSegment.single (first := 8846) (initial := 3701) row_count_8847 (by decide)
private theorem atomic_8847_agreement : RowAgreement 8846 8847 := RowAgreement.single (first := 8846) modelrow_agrees_8847
private def atomic_8848 : CountSegment 1 10 13120 8847 8848 3703 3704 := CountSegment.single (first := 8847) (initial := 3703) row_count_8848 (by decide)
private theorem atomic_8848_agreement : RowAgreement 8847 8848 := RowAgreement.single (first := 8847) modelrow_agrees_8848
private def atomic_8849 : CountSegment 1 10 13120 8848 8849 3704 3704 := CountSegment.single (first := 8848) (initial := 3704) row_count_8849 (by decide)
private theorem atomic_8849_agreement : RowAgreement 8848 8849 := RowAgreement.single (first := 8848) modelrow_agrees_8849
private def atomic_8850 : CountSegment 1 10 13120 8849 8850 3704 3704 := CountSegment.single (first := 8849) (initial := 3704) row_count_8850 (by decide)
private theorem atomic_8850_agreement : RowAgreement 8849 8850 := RowAgreement.single (first := 8849) modelrow_agrees_8850
private def atomic_8851 : CountSegment 1 10 13120 8850 8851 3704 3704 := CountSegment.single (first := 8850) (initial := 3704) row_count_8851 (by decide)
private theorem atomic_8851_agreement : RowAgreement 8850 8851 := RowAgreement.single (first := 8850) modelrow_agrees_8851
private def atomic_8852 : CountSegment 1 10 13120 8851 8852 3704 3705 := CountSegment.single (first := 8851) (initial := 3704) row_count_8852 (by decide)
private theorem atomic_8852_agreement : RowAgreement 8851 8852 := RowAgreement.single (first := 8851) modelrow_agrees_8852
private def atomic_8853 : CountSegment 1 10 13120 8852 8853 3705 3705 := CountSegment.single (first := 8852) (initial := 3705) row_count_8853 (by decide)
private theorem atomic_8853_agreement : RowAgreement 8852 8853 := RowAgreement.single (first := 8852) modelrow_agrees_8853
private def atomic_8854 : CountSegment 1 10 13120 8853 8854 3705 3705 := CountSegment.single (first := 8853) (initial := 3705) row_count_8854 (by decide)
private theorem atomic_8854_agreement : RowAgreement 8853 8854 := RowAgreement.single (first := 8853) modelrow_agrees_8854
private def atomic_8855 : CountSegment 1 10 13120 8854 8855 3705 3705 := CountSegment.single (first := 8854) (initial := 3705) row_count_8855 (by decide)
private theorem atomic_8855_agreement : RowAgreement 8854 8855 := RowAgreement.single (first := 8854) modelrow_agrees_8855
private def atomic_8856 : CountSegment 1 10 13120 8855 8856 3705 3705 := CountSegment.single (first := 8855) (initial := 3705) row_count_8856 (by decide)
private theorem atomic_8856_agreement : RowAgreement 8855 8856 := RowAgreement.single (first := 8855) modelrow_agrees_8856
private def atomic_8857 : CountSegment 1 10 13120 8856 8857 3705 3705 := CountSegment.single (first := 8856) (initial := 3705) row_count_8857 (by decide)
private theorem atomic_8857_agreement : RowAgreement 8856 8857 := RowAgreement.single (first := 8856) modelrow_agrees_8857
private def atomic_8858 : CountSegment 1 10 13120 8857 8858 3705 3705 := CountSegment.single (first := 8857) (initial := 3705) row_count_8858 (by decide)
private theorem atomic_8858_agreement : RowAgreement 8857 8858 := RowAgreement.single (first := 8857) modelrow_agrees_8858
private def atomic_8859 : CountSegment 1 10 13120 8858 8859 3705 3706 := CountSegment.single (first := 8858) (initial := 3705) row_count_8859 (by decide)
private theorem atomic_8859_agreement : RowAgreement 8858 8859 := RowAgreement.single (first := 8858) modelrow_agrees_8859
private def atomic_8860 : CountSegment 1 10 13120 8859 8860 3706 3707 := CountSegment.single (first := 8859) (initial := 3706) row_count_8860 (by decide)
private theorem atomic_8860_agreement : RowAgreement 8859 8860 := RowAgreement.single (first := 8859) modelrow_agrees_8860
private def atomic_8861 : CountSegment 1 10 13120 8860 8861 3707 3707 := CountSegment.single (first := 8860) (initial := 3707) row_count_8861 (by decide)
private theorem atomic_8861_agreement : RowAgreement 8860 8861 := RowAgreement.single (first := 8860) modelrow_agrees_8861
private def atomic_8862 : CountSegment 1 10 13120 8861 8862 3707 3707 := CountSegment.single (first := 8861) (initial := 3707) row_count_8862 (by decide)
private theorem atomic_8862_agreement : RowAgreement 8861 8862 := RowAgreement.single (first := 8861) modelrow_agrees_8862
private def atomic_8863 : CountSegment 1 10 13120 8862 8863 3707 3707 := CountSegment.single (first := 8862) (initial := 3707) row_count_8863 (by decide)
private theorem atomic_8863_agreement : RowAgreement 8862 8863 := RowAgreement.single (first := 8862) modelrow_agrees_8863
private def atomic_8864 : CountSegment 1 10 13120 8863 8864 3707 3707 := CountSegment.single (first := 8863) (initial := 3707) row_count_8864 (by decide)
private theorem atomic_8864_agreement : RowAgreement 8863 8864 := RowAgreement.single (first := 8863) modelrow_agrees_8864
private def atomic_8865 : CountSegment 1 10 13120 8864 8865 3707 3707 := CountSegment.single (first := 8864) (initial := 3707) row_count_8865 (by decide)
private theorem atomic_8865_agreement : RowAgreement 8864 8865 := RowAgreement.single (first := 8864) modelrow_agrees_8865
private def atomic_8866 : CountSegment 1 10 13120 8865 8866 3707 3707 := CountSegment.single (first := 8865) (initial := 3707) row_count_8866 (by decide)
private theorem atomic_8866_agreement : RowAgreement 8865 8866 := RowAgreement.single (first := 8865) modelrow_agrees_8866
private def atomic_8867 : CountSegment 1 10 13120 8866 8867 3707 3707 := CountSegment.single (first := 8866) (initial := 3707) row_count_8867 (by decide)
private theorem atomic_8867_agreement : RowAgreement 8866 8867 := RowAgreement.single (first := 8866) modelrow_agrees_8867
private def atomic_8868 : CountSegment 1 10 13120 8867 8868 3707 3707 := CountSegment.single (first := 8867) (initial := 3707) row_count_8868 (by decide)
private theorem atomic_8868_agreement : RowAgreement 8867 8868 := RowAgreement.single (first := 8867) modelrow_agrees_8868
private def atomic_8869 : CountSegment 1 10 13120 8868 8869 3707 3707 := CountSegment.single (first := 8868) (initial := 3707) row_count_8869 (by decide)
private theorem atomic_8869_agreement : RowAgreement 8868 8869 := RowAgreement.single (first := 8868) modelrow_agrees_8869
private def atomic_8870 : CountSegment 1 10 13120 8869 8870 3707 3708 := CountSegment.single (first := 8869) (initial := 3707) row_count_8870 (by decide)
private theorem atomic_8870_agreement : RowAgreement 8869 8870 := RowAgreement.single (first := 8869) modelrow_agrees_8870
private def atomic_8871 : CountSegment 1 10 13120 8870 8871 3708 3708 := CountSegment.single (first := 8870) (initial := 3708) row_count_8871 (by decide)
private theorem atomic_8871_agreement : RowAgreement 8870 8871 := RowAgreement.single (first := 8870) modelrow_agrees_8871
private def atomic_8872 : CountSegment 1 10 13120 8871 8872 3708 3709 := CountSegment.single (first := 8871) (initial := 3708) row_count_8872 (by decide)
private theorem atomic_8872_agreement : RowAgreement 8871 8872 := RowAgreement.single (first := 8871) modelrow_agrees_8872
private def atomic_8873 : CountSegment 1 10 13120 8872 8873 3709 3709 := CountSegment.single (first := 8872) (initial := 3709) row_count_8873 (by decide)
private theorem atomic_8873_agreement : RowAgreement 8872 8873 := RowAgreement.single (first := 8872) modelrow_agrees_8873
private def atomic_8874 : CountSegment 1 10 13120 8873 8874 3709 3709 := CountSegment.single (first := 8873) (initial := 3709) row_count_8874 (by decide)
private theorem atomic_8874_agreement : RowAgreement 8873 8874 := RowAgreement.single (first := 8873) modelrow_agrees_8874
private def atomic_8875 : CountSegment 1 10 13120 8874 8875 3709 3710 := CountSegment.single (first := 8874) (initial := 3709) row_count_8875 (by decide)
private theorem atomic_8875_agreement : RowAgreement 8874 8875 := RowAgreement.single (first := 8874) modelrow_agrees_8875
private def atomic_8876 : CountSegment 1 10 13120 8875 8876 3710 3710 := CountSegment.single (first := 8875) (initial := 3710) row_count_8876 (by decide)
private theorem atomic_8876_agreement : RowAgreement 8875 8876 := RowAgreement.single (first := 8875) modelrow_agrees_8876
private def atomic_8877 : CountSegment 1 10 13120 8876 8877 3710 3710 := CountSegment.single (first := 8876) (initial := 3710) row_count_8877 (by decide)
private theorem atomic_8877_agreement : RowAgreement 8876 8877 := RowAgreement.single (first := 8876) modelrow_agrees_8877
private def atomic_8878 : CountSegment 1 10 13120 8877 8878 3710 3710 := CountSegment.single (first := 8877) (initial := 3710) row_count_8878 (by decide)
private theorem atomic_8878_agreement : RowAgreement 8877 8878 := RowAgreement.single (first := 8877) modelrow_agrees_8878
private def atomic_8879 : CountSegment 1 10 13120 8878 8879 3710 3710 := CountSegment.single (first := 8878) (initial := 3710) row_count_8879 (by decide)
private theorem atomic_8879_agreement : RowAgreement 8878 8879 := RowAgreement.single (first := 8878) modelrow_agrees_8879
private def atomic_8880 : CountSegment 1 10 13120 8879 8880 3710 3711 := CountSegment.single (first := 8879) (initial := 3710) row_count_8880 (by decide)
private theorem atomic_8880_agreement : RowAgreement 8879 8880 := RowAgreement.single (first := 8879) modelrow_agrees_8880
private def atomic_8881 : CountSegment 1 10 13120 8880 8881 3711 3711 := CountSegment.single (first := 8880) (initial := 3711) row_count_8881 (by decide)
private theorem atomic_8881_agreement : RowAgreement 8880 8881 := RowAgreement.single (first := 8880) modelrow_agrees_8881
private def atomic_8882 : CountSegment 1 10 13120 8881 8882 3711 3712 := CountSegment.single (first := 8881) (initial := 3711) row_count_8882 (by decide)
private theorem atomic_8882_agreement : RowAgreement 8881 8882 := RowAgreement.single (first := 8881) modelrow_agrees_8882
private def atomic_8883 : CountSegment 1 10 13120 8882 8883 3712 3712 := CountSegment.single (first := 8882) (initial := 3712) row_count_8883 (by decide)
private theorem atomic_8883_agreement : RowAgreement 8882 8883 := RowAgreement.single (first := 8882) modelrow_agrees_8883
private def atomic_8884 : CountSegment 1 10 13120 8883 8884 3712 3712 := CountSegment.single (first := 8883) (initial := 3712) row_count_8884 (by decide)
private theorem atomic_8884_agreement : RowAgreement 8883 8884 := RowAgreement.single (first := 8883) modelrow_agrees_8884
private def atomic_8885 : CountSegment 1 10 13120 8884 8885 3712 3713 := CountSegment.single (first := 8884) (initial := 3712) row_count_8885 (by decide)
private theorem atomic_8885_agreement : RowAgreement 8884 8885 := RowAgreement.single (first := 8884) modelrow_agrees_8885
private def atomic_8886 : CountSegment 1 10 13120 8885 8886 3713 3713 := CountSegment.single (first := 8885) (initial := 3713) row_count_8886 (by decide)
private theorem atomic_8886_agreement : RowAgreement 8885 8886 := RowAgreement.single (first := 8885) modelrow_agrees_8886
private def atomic_8887 : CountSegment 1 10 13120 8886 8887 3713 3713 := CountSegment.single (first := 8886) (initial := 3713) row_count_8887 (by decide)
private theorem atomic_8887_agreement : RowAgreement 8886 8887 := RowAgreement.single (first := 8886) modelrow_agrees_8887
private def atomic_8888 : CountSegment 1 10 13120 8887 8888 3713 3713 := CountSegment.single (first := 8887) (initial := 3713) row_count_8888 (by decide)
private theorem atomic_8888_agreement : RowAgreement 8887 8888 := RowAgreement.single (first := 8887) modelrow_agrees_8888
private def atomic_8889 : CountSegment 1 10 13120 8888 8889 3713 3713 := CountSegment.single (first := 8888) (initial := 3713) row_count_8889 (by decide)
private theorem atomic_8889_agreement : RowAgreement 8888 8889 := RowAgreement.single (first := 8888) modelrow_agrees_8889
private def atomic_8890 : CountSegment 1 10 13120 8889 8890 3713 3713 := CountSegment.single (first := 8889) (initial := 3713) row_count_8890 (by decide)
private theorem atomic_8890_agreement : RowAgreement 8889 8890 := RowAgreement.single (first := 8889) modelrow_agrees_8890
private def atomic_8891 : CountSegment 1 10 13120 8890 8891 3713 3714 := CountSegment.single (first := 8890) (initial := 3713) row_count_8891 (by decide)
private theorem atomic_8891_agreement : RowAgreement 8890 8891 := RowAgreement.single (first := 8890) modelrow_agrees_8891
private def atomic_8892 : CountSegment 1 10 13120 8891 8892 3714 3714 := CountSegment.single (first := 8891) (initial := 3714) row_count_8892 (by decide)
private theorem atomic_8892_agreement : RowAgreement 8891 8892 := RowAgreement.single (first := 8891) modelrow_agrees_8892
private def atomic_8893 : CountSegment 1 10 13120 8892 8893 3714 3714 := CountSegment.single (first := 8892) (initial := 3714) row_count_8893 (by decide)
private theorem atomic_8893_agreement : RowAgreement 8892 8893 := RowAgreement.single (first := 8892) modelrow_agrees_8893
private def atomic_8894 : CountSegment 1 10 13120 8893 8894 3714 3714 := CountSegment.single (first := 8893) (initial := 3714) row_count_8894 (by decide)
private theorem atomic_8894_agreement : RowAgreement 8893 8894 := RowAgreement.single (first := 8893) modelrow_agrees_8894
private def atomic_8895 : CountSegment 1 10 13120 8894 8895 3714 3715 := CountSegment.single (first := 8894) (initial := 3714) row_count_8895 (by decide)
private theorem atomic_8895_agreement : RowAgreement 8894 8895 := RowAgreement.single (first := 8894) modelrow_agrees_8895
private def atomic_8896 : CountSegment 1 10 13120 8895 8896 3715 3715 := CountSegment.single (first := 8895) (initial := 3715) row_count_8896 (by decide)
private theorem atomic_8896_agreement : RowAgreement 8895 8896 := RowAgreement.single (first := 8895) modelrow_agrees_8896
private def atomic_8897 : CountSegment 1 10 13120 8896 8897 3715 3715 := CountSegment.single (first := 8896) (initial := 3715) row_count_8897 (by decide)
private theorem atomic_8897_agreement : RowAgreement 8896 8897 := RowAgreement.single (first := 8896) modelrow_agrees_8897
private def atomic_8898 : CountSegment 1 10 13120 8897 8898 3715 3716 := CountSegment.single (first := 8897) (initial := 3715) row_count_8898 (by decide)
private theorem atomic_8898_agreement : RowAgreement 8897 8898 := RowAgreement.single (first := 8897) modelrow_agrees_8898
private def atomic_8899 : CountSegment 1 10 13120 8898 8899 3716 3716 := CountSegment.single (first := 8898) (initial := 3716) row_count_8899 (by decide)
private theorem atomic_8899_agreement : RowAgreement 8898 8899 := RowAgreement.single (first := 8898) modelrow_agrees_8899
private def atomic_8900 : CountSegment 1 10 13120 8899 8900 3716 3716 := CountSegment.single (first := 8899) (initial := 3716) row_count_8900 (by decide)
private theorem atomic_8900_agreement : RowAgreement 8899 8900 := RowAgreement.single (first := 8899) modelrow_agrees_8900
private def atomic_8901 : CountSegment 1 10 13120 8900 8901 3716 3718 := CountSegment.single (first := 8900) (initial := 3716) row_count_8901 (by decide)
private theorem atomic_8901_agreement : RowAgreement 8900 8901 := RowAgreement.single (first := 8900) modelrow_agrees_8901
private def atomic_8902 : CountSegment 1 10 13120 8901 8902 3718 3719 := CountSegment.single (first := 8901) (initial := 3718) row_count_8902 (by decide)
private theorem atomic_8902_agreement : RowAgreement 8901 8902 := RowAgreement.single (first := 8901) modelrow_agrees_8902
private def atomic_8903 : CountSegment 1 10 13120 8902 8903 3719 3719 := CountSegment.single (first := 8902) (initial := 3719) row_count_8903 (by decide)
private theorem atomic_8903_agreement : RowAgreement 8902 8903 := RowAgreement.single (first := 8902) modelrow_agrees_8903
private def atomic_8904 : CountSegment 1 10 13120 8903 8904 3719 3719 := CountSegment.single (first := 8903) (initial := 3719) row_count_8904 (by decide)
private theorem atomic_8904_agreement : RowAgreement 8903 8904 := RowAgreement.single (first := 8903) modelrow_agrees_8904
private def atomic_8905 : CountSegment 1 10 13120 8904 8905 3719 3720 := CountSegment.single (first := 8904) (initial := 3719) row_count_8905 (by decide)
private theorem atomic_8905_agreement : RowAgreement 8904 8905 := RowAgreement.single (first := 8904) modelrow_agrees_8905
private def atomic_8906 : CountSegment 1 10 13120 8905 8906 3720 3720 := CountSegment.single (first := 8905) (initial := 3720) row_count_8906 (by decide)
private theorem atomic_8906_agreement : RowAgreement 8905 8906 := RowAgreement.single (first := 8905) modelrow_agrees_8906
private def atomic_8907 : CountSegment 1 10 13120 8906 8907 3720 3720 := CountSegment.single (first := 8906) (initial := 3720) row_count_8907 (by decide)
private theorem atomic_8907_agreement : RowAgreement 8906 8907 := RowAgreement.single (first := 8906) modelrow_agrees_8907
private def atomic_8908 : CountSegment 1 10 13120 8907 8908 3720 3721 := CountSegment.single (first := 8907) (initial := 3720) row_count_8908 (by decide)
private theorem atomic_8908_agreement : RowAgreement 8907 8908 := RowAgreement.single (first := 8907) modelrow_agrees_8908
private def atomic_8909 : CountSegment 1 10 13120 8908 8909 3721 3721 := CountSegment.single (first := 8908) (initial := 3721) row_count_8909 (by decide)
private theorem atomic_8909_agreement : RowAgreement 8908 8909 := RowAgreement.single (first := 8908) modelrow_agrees_8909
private def atomic_8910 : CountSegment 1 10 13120 8909 8910 3721 3721 := CountSegment.single (first := 8909) (initial := 3721) row_count_8910 (by decide)
private theorem atomic_8910_agreement : RowAgreement 8909 8910 := RowAgreement.single (first := 8909) modelrow_agrees_8910
private def atomic_8911 : CountSegment 1 10 13120 8910 8911 3721 3721 := CountSegment.single (first := 8910) (initial := 3721) row_count_8911 (by decide)
private theorem atomic_8911_agreement : RowAgreement 8910 8911 := RowAgreement.single (first := 8910) modelrow_agrees_8911
private def atomic_8912 : CountSegment 1 10 13120 8911 8912 3721 3722 := CountSegment.single (first := 8911) (initial := 3721) row_count_8912 (by decide)
private theorem atomic_8912_agreement : RowAgreement 8911 8912 := RowAgreement.single (first := 8911) modelrow_agrees_8912
private def atomic_8913 : CountSegment 1 10 13120 8912 8913 3722 3722 := CountSegment.single (first := 8912) (initial := 3722) row_count_8913 (by decide)
private theorem atomic_8913_agreement : RowAgreement 8912 8913 := RowAgreement.single (first := 8912) modelrow_agrees_8913
private def atomic_8914 : CountSegment 1 10 13120 8913 8914 3722 3723 := CountSegment.single (first := 8913) (initial := 3722) row_count_8914 (by decide)
private theorem atomic_8914_agreement : RowAgreement 8913 8914 := RowAgreement.single (first := 8913) modelrow_agrees_8914
private def atomic_8915 : CountSegment 1 10 13120 8914 8915 3723 3724 := CountSegment.single (first := 8914) (initial := 3723) row_count_8915 (by decide)
private theorem atomic_8915_agreement : RowAgreement 8914 8915 := RowAgreement.single (first := 8914) modelrow_agrees_8915
private def atomic_8916 : CountSegment 1 10 13120 8915 8916 3724 3724 := CountSegment.single (first := 8915) (initial := 3724) row_count_8916 (by decide)
private theorem atomic_8916_agreement : RowAgreement 8915 8916 := RowAgreement.single (first := 8915) modelrow_agrees_8916
private def atomic_8917 : CountSegment 1 10 13120 8916 8917 3724 3724 := CountSegment.single (first := 8916) (initial := 3724) row_count_8917 (by decide)
private theorem atomic_8917_agreement : RowAgreement 8916 8917 := RowAgreement.single (first := 8916) modelrow_agrees_8917
private def atomic_8918 : CountSegment 1 10 13120 8917 8918 3724 3724 := CountSegment.single (first := 8917) (initial := 3724) row_count_8918 (by decide)
private theorem atomic_8918_agreement : RowAgreement 8917 8918 := RowAgreement.single (first := 8917) modelrow_agrees_8918
private def atomic_8919 : CountSegment 1 10 13120 8918 8919 3724 3725 := CountSegment.single (first := 8918) (initial := 3724) row_count_8919 (by decide)
private theorem atomic_8919_agreement : RowAgreement 8918 8919 := RowAgreement.single (first := 8918) modelrow_agrees_8919
private def atomic_8920 : CountSegment 1 10 13120 8919 8920 3725 3725 := CountSegment.single (first := 8919) (initial := 3725) row_count_8920 (by decide)
private theorem atomic_8920_agreement : RowAgreement 8919 8920 := RowAgreement.single (first := 8919) modelrow_agrees_8920
private def atomic_8921 : CountSegment 1 10 13120 8920 8921 3725 3725 := CountSegment.single (first := 8920) (initial := 3725) row_count_8921 (by decide)
private theorem atomic_8921_agreement : RowAgreement 8920 8921 := RowAgreement.single (first := 8920) modelrow_agrees_8921
private def atomic_8922 : CountSegment 1 10 13120 8921 8922 3725 3725 := CountSegment.single (first := 8921) (initial := 3725) row_count_8922 (by decide)
private theorem atomic_8922_agreement : RowAgreement 8921 8922 := RowAgreement.single (first := 8921) modelrow_agrees_8922
private def atomic_8923 : CountSegment 1 10 13120 8922 8923 3725 3726 := CountSegment.single (first := 8922) (initial := 3725) row_count_8923 (by decide)
private theorem atomic_8923_agreement : RowAgreement 8922 8923 := RowAgreement.single (first := 8922) modelrow_agrees_8923
private def atomic_8924 : CountSegment 1 10 13120 8923 8924 3726 3726 := CountSegment.single (first := 8923) (initial := 3726) row_count_8924 (by decide)
private theorem atomic_8924_agreement : RowAgreement 8923 8924 := RowAgreement.single (first := 8923) modelrow_agrees_8924
private def atomic_8925 : CountSegment 1 10 13120 8924 8925 3726 3727 := CountSegment.single (first := 8924) (initial := 3726) row_count_8925 (by decide)
private theorem atomic_8925_agreement : RowAgreement 8924 8925 := RowAgreement.single (first := 8924) modelrow_agrees_8925
private def atomic_8926 : CountSegment 1 10 13120 8925 8926 3727 3727 := CountSegment.single (first := 8925) (initial := 3727) row_count_8926 (by decide)
private theorem atomic_8926_agreement : RowAgreement 8925 8926 := RowAgreement.single (first := 8925) modelrow_agrees_8926
private def atomic_8927 : CountSegment 1 10 13120 8926 8927 3727 3728 := CountSegment.single (first := 8926) (initial := 3727) row_count_8927 (by decide)
private theorem atomic_8927_agreement : RowAgreement 8926 8927 := RowAgreement.single (first := 8926) modelrow_agrees_8927
private def atomic_8928 : CountSegment 1 10 13120 8927 8928 3728 3730 := CountSegment.single (first := 8927) (initial := 3728) row_count_8928 (by decide)
private theorem atomic_8928_agreement : RowAgreement 8927 8928 := RowAgreement.single (first := 8927) modelrow_agrees_8928
private def atomic_8929 : CountSegment 1 10 13120 8928 8929 3730 3730 := CountSegment.single (first := 8928) (initial := 3730) row_count_8929 (by decide)
private theorem atomic_8929_agreement : RowAgreement 8928 8929 := RowAgreement.single (first := 8928) modelrow_agrees_8929
private def atomic_8930 : CountSegment 1 10 13120 8929 8930 3730 3731 := CountSegment.single (first := 8929) (initial := 3730) row_count_8930 (by decide)
private theorem atomic_8930_agreement : RowAgreement 8929 8930 := RowAgreement.single (first := 8929) modelrow_agrees_8930
private def atomic_8931 : CountSegment 1 10 13120 8930 8931 3731 3732 := CountSegment.single (first := 8930) (initial := 3731) row_count_8931 (by decide)
private theorem atomic_8931_agreement : RowAgreement 8930 8931 := RowAgreement.single (first := 8930) modelrow_agrees_8931
private def atomic_8932 : CountSegment 1 10 13120 8931 8932 3732 3732 := CountSegment.single (first := 8931) (initial := 3732) row_count_8932 (by decide)
private theorem atomic_8932_agreement : RowAgreement 8931 8932 := RowAgreement.single (first := 8931) modelrow_agrees_8932
private def atomic_8933 : CountSegment 1 10 13120 8932 8933 3732 3732 := CountSegment.single (first := 8932) (initial := 3732) row_count_8933 (by decide)
private theorem atomic_8933_agreement : RowAgreement 8932 8933 := RowAgreement.single (first := 8932) modelrow_agrees_8933
private def atomic_8934 : CountSegment 1 10 13120 8933 8934 3732 3732 := CountSegment.single (first := 8933) (initial := 3732) row_count_8934 (by decide)
private theorem atomic_8934_agreement : RowAgreement 8933 8934 := RowAgreement.single (first := 8933) modelrow_agrees_8934
private def atomic_8935 : CountSegment 1 10 13120 8934 8935 3732 3732 := CountSegment.single (first := 8934) (initial := 3732) row_count_8935 (by decide)
private theorem atomic_8935_agreement : RowAgreement 8934 8935 := RowAgreement.single (first := 8934) modelrow_agrees_8935
private def atomic_8936 : CountSegment 1 10 13120 8935 8936 3732 3732 := CountSegment.single (first := 8935) (initial := 3732) row_count_8936 (by decide)
private theorem atomic_8936_agreement : RowAgreement 8935 8936 := RowAgreement.single (first := 8935) modelrow_agrees_8936
private def atomic_8937 : CountSegment 1 10 13120 8936 8937 3732 3732 := CountSegment.single (first := 8936) (initial := 3732) row_count_8937 (by decide)
private theorem atomic_8937_agreement : RowAgreement 8936 8937 := RowAgreement.single (first := 8936) modelrow_agrees_8937
private def atomic_8938 : CountSegment 1 10 13120 8937 8938 3732 3732 := CountSegment.single (first := 8937) (initial := 3732) row_count_8938 (by decide)
private theorem atomic_8938_agreement : RowAgreement 8937 8938 := RowAgreement.single (first := 8937) modelrow_agrees_8938
private def atomic_8939 : CountSegment 1 10 13120 8938 8939 3732 3732 := CountSegment.single (first := 8938) (initial := 3732) row_count_8939 (by decide)
private theorem atomic_8939_agreement : RowAgreement 8938 8939 := RowAgreement.single (first := 8938) modelrow_agrees_8939
private def atomic_8940 : CountSegment 1 10 13120 8939 8940 3732 3732 := CountSegment.single (first := 8939) (initial := 3732) row_count_8940 (by decide)
private theorem atomic_8940_agreement : RowAgreement 8939 8940 := RowAgreement.single (first := 8939) modelrow_agrees_8940
private def atomic_8941 : CountSegment 1 10 13120 8940 8941 3732 3732 := CountSegment.single (first := 8940) (initial := 3732) row_count_8941 (by decide)
private theorem atomic_8941_agreement : RowAgreement 8940 8941 := RowAgreement.single (first := 8940) modelrow_agrees_8941
private def atomic_8942 : CountSegment 1 10 13120 8941 8942 3732 3732 := CountSegment.single (first := 8941) (initial := 3732) row_count_8942 (by decide)
private theorem atomic_8942_agreement : RowAgreement 8941 8942 := RowAgreement.single (first := 8941) modelrow_agrees_8942
private def atomic_8943 : CountSegment 1 10 13120 8942 8943 3732 3733 := CountSegment.single (first := 8942) (initial := 3732) row_count_8943 (by decide)
private theorem atomic_8943_agreement : RowAgreement 8942 8943 := RowAgreement.single (first := 8942) modelrow_agrees_8943
private def atomic_8944 : CountSegment 1 10 13120 8943 8944 3733 3733 := CountSegment.single (first := 8943) (initial := 3733) row_count_8944 (by decide)
private theorem atomic_8944_agreement : RowAgreement 8943 8944 := RowAgreement.single (first := 8943) modelrow_agrees_8944
private def atomic_8945 : CountSegment 1 10 13120 8944 8945 3733 3734 := CountSegment.single (first := 8944) (initial := 3733) row_count_8945 (by decide)
private theorem atomic_8945_agreement : RowAgreement 8944 8945 := RowAgreement.single (first := 8944) modelrow_agrees_8945
private def atomic_8946 : CountSegment 1 10 13120 8945 8946 3734 3734 := CountSegment.single (first := 8945) (initial := 3734) row_count_8946 (by decide)
private theorem atomic_8946_agreement : RowAgreement 8945 8946 := RowAgreement.single (first := 8945) modelrow_agrees_8946
private def atomic_8947 : CountSegment 1 10 13120 8946 8947 3734 3734 := CountSegment.single (first := 8946) (initial := 3734) row_count_8947 (by decide)
private theorem atomic_8947_agreement : RowAgreement 8946 8947 := RowAgreement.single (first := 8946) modelrow_agrees_8947
private def atomic_8948 : CountSegment 1 10 13120 8947 8948 3734 3735 := CountSegment.single (first := 8947) (initial := 3734) row_count_8948 (by decide)
private theorem atomic_8948_agreement : RowAgreement 8947 8948 := RowAgreement.single (first := 8947) modelrow_agrees_8948
private def atomic_8949 : CountSegment 1 10 13120 8948 8949 3735 3736 := CountSegment.single (first := 8948) (initial := 3735) row_count_8949 (by decide)
private theorem atomic_8949_agreement : RowAgreement 8948 8949 := RowAgreement.single (first := 8948) modelrow_agrees_8949
private def atomic_8950 : CountSegment 1 10 13120 8949 8950 3736 3737 := CountSegment.single (first := 8949) (initial := 3736) row_count_8950 (by decide)
private theorem atomic_8950_agreement : RowAgreement 8949 8950 := RowAgreement.single (first := 8949) modelrow_agrees_8950
private def atomic_8951 : CountSegment 1 10 13120 8950 8951 3737 3737 := CountSegment.single (first := 8950) (initial := 3737) row_count_8951 (by decide)
private theorem atomic_8951_agreement : RowAgreement 8950 8951 := RowAgreement.single (first := 8950) modelrow_agrees_8951
private def atomic_8952 : CountSegment 1 10 13120 8951 8952 3737 3738 := CountSegment.single (first := 8951) (initial := 3737) row_count_8952 (by decide)
private theorem atomic_8952_agreement : RowAgreement 8951 8952 := RowAgreement.single (first := 8951) modelrow_agrees_8952
private def atomic_8953 : CountSegment 1 10 13120 8952 8953 3738 3739 := CountSegment.single (first := 8952) (initial := 3738) row_count_8953 (by decide)
private theorem atomic_8953_agreement : RowAgreement 8952 8953 := RowAgreement.single (first := 8952) modelrow_agrees_8953
private def atomic_8954 : CountSegment 1 10 13120 8953 8954 3739 3739 := CountSegment.single (first := 8953) (initial := 3739) row_count_8954 (by decide)
private theorem atomic_8954_agreement : RowAgreement 8953 8954 := RowAgreement.single (first := 8953) modelrow_agrees_8954
private def atomic_8955 : CountSegment 1 10 13120 8954 8955 3739 3739 := CountSegment.single (first := 8954) (initial := 3739) row_count_8955 (by decide)
private theorem atomic_8955_agreement : RowAgreement 8954 8955 := RowAgreement.single (first := 8954) modelrow_agrees_8955
private def atomic_8956 : CountSegment 1 10 13120 8955 8956 3739 3739 := CountSegment.single (first := 8955) (initial := 3739) row_count_8956 (by decide)
private theorem atomic_8956_agreement : RowAgreement 8955 8956 := RowAgreement.single (first := 8955) modelrow_agrees_8956
private def atomic_8957 : CountSegment 1 10 13120 8956 8957 3739 3740 := CountSegment.single (first := 8956) (initial := 3739) row_count_8957 (by decide)
private theorem atomic_8957_agreement : RowAgreement 8956 8957 := RowAgreement.single (first := 8956) modelrow_agrees_8957
private def atomic_8958 : CountSegment 1 10 13120 8957 8958 3740 3741 := CountSegment.single (first := 8957) (initial := 3740) row_count_8958 (by decide)
private theorem atomic_8958_agreement : RowAgreement 8957 8958 := RowAgreement.single (first := 8957) modelrow_agrees_8958
private def atomic_8959 : CountSegment 1 10 13120 8958 8959 3741 3741 := CountSegment.single (first := 8958) (initial := 3741) row_count_8959 (by decide)
private theorem atomic_8959_agreement : RowAgreement 8958 8959 := RowAgreement.single (first := 8958) modelrow_agrees_8959
private def atomic_8960 : CountSegment 1 10 13120 8959 8960 3741 3741 := CountSegment.single (first := 8959) (initial := 3741) row_count_8960 (by decide)
private theorem atomic_8960_agreement : RowAgreement 8959 8960 := RowAgreement.single (first := 8959) modelrow_agrees_8960
private def within_069_000 : CountSegment 1 10 13120 8832 8834 3694 3694 := CountSegment.join atomic_8833 atomic_8834
private theorem within_069_000_agreement : RowAgreement 8832 8834 := RowAgreement.join atomic_8833_agreement atomic_8834_agreement
private def within_069_001 : CountSegment 1 10 13120 8834 8836 3694 3695 := CountSegment.join atomic_8835 atomic_8836
private theorem within_069_001_agreement : RowAgreement 8834 8836 := RowAgreement.join atomic_8835_agreement atomic_8836_agreement
private def within_069_002 : CountSegment 1 10 13120 8832 8836 3694 3695 := CountSegment.join within_069_000 within_069_001
private theorem within_069_002_agreement : RowAgreement 8832 8836 := RowAgreement.join within_069_000_agreement within_069_001_agreement
private def within_069_003 : CountSegment 1 10 13120 8836 8838 3695 3696 := CountSegment.join atomic_8837 atomic_8838
private theorem within_069_003_agreement : RowAgreement 8836 8838 := RowAgreement.join atomic_8837_agreement atomic_8838_agreement
private def within_069_004 : CountSegment 1 10 13120 8838 8840 3696 3697 := CountSegment.join atomic_8839 atomic_8840
private theorem within_069_004_agreement : RowAgreement 8838 8840 := RowAgreement.join atomic_8839_agreement atomic_8840_agreement
private def within_069_005 : CountSegment 1 10 13120 8836 8840 3695 3697 := CountSegment.join within_069_003 within_069_004
private theorem within_069_005_agreement : RowAgreement 8836 8840 := RowAgreement.join within_069_003_agreement within_069_004_agreement
private def within_069_006 : CountSegment 1 10 13120 8832 8840 3694 3697 := CountSegment.join within_069_002 within_069_005
private theorem within_069_006_agreement : RowAgreement 8832 8840 := RowAgreement.join within_069_002_agreement within_069_005_agreement
private def within_069_007 : CountSegment 1 10 13120 8840 8842 3697 3697 := CountSegment.join atomic_8841 atomic_8842
private theorem within_069_007_agreement : RowAgreement 8840 8842 := RowAgreement.join atomic_8841_agreement atomic_8842_agreement
private def within_069_008 : CountSegment 1 10 13120 8842 8844 3697 3699 := CountSegment.join atomic_8843 atomic_8844
private theorem within_069_008_agreement : RowAgreement 8842 8844 := RowAgreement.join atomic_8843_agreement atomic_8844_agreement
private def within_069_009 : CountSegment 1 10 13120 8840 8844 3697 3699 := CountSegment.join within_069_007 within_069_008
private theorem within_069_009_agreement : RowAgreement 8840 8844 := RowAgreement.join within_069_007_agreement within_069_008_agreement
private def within_069_010 : CountSegment 1 10 13120 8844 8846 3699 3701 := CountSegment.join atomic_8845 atomic_8846
private theorem within_069_010_agreement : RowAgreement 8844 8846 := RowAgreement.join atomic_8845_agreement atomic_8846_agreement
private def within_069_011 : CountSegment 1 10 13120 8846 8848 3701 3704 := CountSegment.join atomic_8847 atomic_8848
private theorem within_069_011_agreement : RowAgreement 8846 8848 := RowAgreement.join atomic_8847_agreement atomic_8848_agreement
private def within_069_012 : CountSegment 1 10 13120 8844 8848 3699 3704 := CountSegment.join within_069_010 within_069_011
private theorem within_069_012_agreement : RowAgreement 8844 8848 := RowAgreement.join within_069_010_agreement within_069_011_agreement
private def within_069_013 : CountSegment 1 10 13120 8840 8848 3697 3704 := CountSegment.join within_069_009 within_069_012
private theorem within_069_013_agreement : RowAgreement 8840 8848 := RowAgreement.join within_069_009_agreement within_069_012_agreement
private def within_069_014 : CountSegment 1 10 13120 8832 8848 3694 3704 := CountSegment.join within_069_006 within_069_013
private theorem within_069_014_agreement : RowAgreement 8832 8848 := RowAgreement.join within_069_006_agreement within_069_013_agreement
private def within_069_015 : CountSegment 1 10 13120 8848 8850 3704 3704 := CountSegment.join atomic_8849 atomic_8850
private theorem within_069_015_agreement : RowAgreement 8848 8850 := RowAgreement.join atomic_8849_agreement atomic_8850_agreement
private def within_069_016 : CountSegment 1 10 13120 8850 8852 3704 3705 := CountSegment.join atomic_8851 atomic_8852
private theorem within_069_016_agreement : RowAgreement 8850 8852 := RowAgreement.join atomic_8851_agreement atomic_8852_agreement
private def within_069_017 : CountSegment 1 10 13120 8848 8852 3704 3705 := CountSegment.join within_069_015 within_069_016
private theorem within_069_017_agreement : RowAgreement 8848 8852 := RowAgreement.join within_069_015_agreement within_069_016_agreement
private def within_069_018 : CountSegment 1 10 13120 8852 8854 3705 3705 := CountSegment.join atomic_8853 atomic_8854
private theorem within_069_018_agreement : RowAgreement 8852 8854 := RowAgreement.join atomic_8853_agreement atomic_8854_agreement
private def within_069_019 : CountSegment 1 10 13120 8854 8856 3705 3705 := CountSegment.join atomic_8855 atomic_8856
private theorem within_069_019_agreement : RowAgreement 8854 8856 := RowAgreement.join atomic_8855_agreement atomic_8856_agreement
private def within_069_020 : CountSegment 1 10 13120 8852 8856 3705 3705 := CountSegment.join within_069_018 within_069_019
private theorem within_069_020_agreement : RowAgreement 8852 8856 := RowAgreement.join within_069_018_agreement within_069_019_agreement
private def within_069_021 : CountSegment 1 10 13120 8848 8856 3704 3705 := CountSegment.join within_069_017 within_069_020
private theorem within_069_021_agreement : RowAgreement 8848 8856 := RowAgreement.join within_069_017_agreement within_069_020_agreement
private def within_069_022 : CountSegment 1 10 13120 8856 8858 3705 3705 := CountSegment.join atomic_8857 atomic_8858
private theorem within_069_022_agreement : RowAgreement 8856 8858 := RowAgreement.join atomic_8857_agreement atomic_8858_agreement
private def within_069_023 : CountSegment 1 10 13120 8858 8860 3705 3707 := CountSegment.join atomic_8859 atomic_8860
private theorem within_069_023_agreement : RowAgreement 8858 8860 := RowAgreement.join atomic_8859_agreement atomic_8860_agreement
private def within_069_024 : CountSegment 1 10 13120 8856 8860 3705 3707 := CountSegment.join within_069_022 within_069_023
private theorem within_069_024_agreement : RowAgreement 8856 8860 := RowAgreement.join within_069_022_agreement within_069_023_agreement
private def within_069_025 : CountSegment 1 10 13120 8860 8862 3707 3707 := CountSegment.join atomic_8861 atomic_8862
private theorem within_069_025_agreement : RowAgreement 8860 8862 := RowAgreement.join atomic_8861_agreement atomic_8862_agreement
private def within_069_026 : CountSegment 1 10 13120 8862 8864 3707 3707 := CountSegment.join atomic_8863 atomic_8864
private theorem within_069_026_agreement : RowAgreement 8862 8864 := RowAgreement.join atomic_8863_agreement atomic_8864_agreement
private def within_069_027 : CountSegment 1 10 13120 8860 8864 3707 3707 := CountSegment.join within_069_025 within_069_026
private theorem within_069_027_agreement : RowAgreement 8860 8864 := RowAgreement.join within_069_025_agreement within_069_026_agreement
private def within_069_028 : CountSegment 1 10 13120 8856 8864 3705 3707 := CountSegment.join within_069_024 within_069_027
private theorem within_069_028_agreement : RowAgreement 8856 8864 := RowAgreement.join within_069_024_agreement within_069_027_agreement
private def within_069_029 : CountSegment 1 10 13120 8848 8864 3704 3707 := CountSegment.join within_069_021 within_069_028
private theorem within_069_029_agreement : RowAgreement 8848 8864 := RowAgreement.join within_069_021_agreement within_069_028_agreement
private def within_069_030 : CountSegment 1 10 13120 8832 8864 3694 3707 := CountSegment.join within_069_014 within_069_029
private theorem within_069_030_agreement : RowAgreement 8832 8864 := RowAgreement.join within_069_014_agreement within_069_029_agreement
private def within_069_031 : CountSegment 1 10 13120 8864 8866 3707 3707 := CountSegment.join atomic_8865 atomic_8866
private theorem within_069_031_agreement : RowAgreement 8864 8866 := RowAgreement.join atomic_8865_agreement atomic_8866_agreement
private def within_069_032 : CountSegment 1 10 13120 8866 8868 3707 3707 := CountSegment.join atomic_8867 atomic_8868
private theorem within_069_032_agreement : RowAgreement 8866 8868 := RowAgreement.join atomic_8867_agreement atomic_8868_agreement
private def within_069_033 : CountSegment 1 10 13120 8864 8868 3707 3707 := CountSegment.join within_069_031 within_069_032
private theorem within_069_033_agreement : RowAgreement 8864 8868 := RowAgreement.join within_069_031_agreement within_069_032_agreement
private def within_069_034 : CountSegment 1 10 13120 8868 8870 3707 3708 := CountSegment.join atomic_8869 atomic_8870
private theorem within_069_034_agreement : RowAgreement 8868 8870 := RowAgreement.join atomic_8869_agreement atomic_8870_agreement
private def within_069_035 : CountSegment 1 10 13120 8870 8872 3708 3709 := CountSegment.join atomic_8871 atomic_8872
private theorem within_069_035_agreement : RowAgreement 8870 8872 := RowAgreement.join atomic_8871_agreement atomic_8872_agreement
private def within_069_036 : CountSegment 1 10 13120 8868 8872 3707 3709 := CountSegment.join within_069_034 within_069_035
private theorem within_069_036_agreement : RowAgreement 8868 8872 := RowAgreement.join within_069_034_agreement within_069_035_agreement
private def within_069_037 : CountSegment 1 10 13120 8864 8872 3707 3709 := CountSegment.join within_069_033 within_069_036
private theorem within_069_037_agreement : RowAgreement 8864 8872 := RowAgreement.join within_069_033_agreement within_069_036_agreement
private def within_069_038 : CountSegment 1 10 13120 8872 8874 3709 3709 := CountSegment.join atomic_8873 atomic_8874
private theorem within_069_038_agreement : RowAgreement 8872 8874 := RowAgreement.join atomic_8873_agreement atomic_8874_agreement
private def within_069_039 : CountSegment 1 10 13120 8874 8876 3709 3710 := CountSegment.join atomic_8875 atomic_8876
private theorem within_069_039_agreement : RowAgreement 8874 8876 := RowAgreement.join atomic_8875_agreement atomic_8876_agreement
private def within_069_040 : CountSegment 1 10 13120 8872 8876 3709 3710 := CountSegment.join within_069_038 within_069_039
private theorem within_069_040_agreement : RowAgreement 8872 8876 := RowAgreement.join within_069_038_agreement within_069_039_agreement
private def within_069_041 : CountSegment 1 10 13120 8876 8878 3710 3710 := CountSegment.join atomic_8877 atomic_8878
private theorem within_069_041_agreement : RowAgreement 8876 8878 := RowAgreement.join atomic_8877_agreement atomic_8878_agreement
private def within_069_042 : CountSegment 1 10 13120 8878 8880 3710 3711 := CountSegment.join atomic_8879 atomic_8880
private theorem within_069_042_agreement : RowAgreement 8878 8880 := RowAgreement.join atomic_8879_agreement atomic_8880_agreement
private def within_069_043 : CountSegment 1 10 13120 8876 8880 3710 3711 := CountSegment.join within_069_041 within_069_042
private theorem within_069_043_agreement : RowAgreement 8876 8880 := RowAgreement.join within_069_041_agreement within_069_042_agreement
private def within_069_044 : CountSegment 1 10 13120 8872 8880 3709 3711 := CountSegment.join within_069_040 within_069_043
private theorem within_069_044_agreement : RowAgreement 8872 8880 := RowAgreement.join within_069_040_agreement within_069_043_agreement
private def within_069_045 : CountSegment 1 10 13120 8864 8880 3707 3711 := CountSegment.join within_069_037 within_069_044
private theorem within_069_045_agreement : RowAgreement 8864 8880 := RowAgreement.join within_069_037_agreement within_069_044_agreement
private def within_069_046 : CountSegment 1 10 13120 8880 8882 3711 3712 := CountSegment.join atomic_8881 atomic_8882
private theorem within_069_046_agreement : RowAgreement 8880 8882 := RowAgreement.join atomic_8881_agreement atomic_8882_agreement
private def within_069_047 : CountSegment 1 10 13120 8882 8884 3712 3712 := CountSegment.join atomic_8883 atomic_8884
private theorem within_069_047_agreement : RowAgreement 8882 8884 := RowAgreement.join atomic_8883_agreement atomic_8884_agreement
private def within_069_048 : CountSegment 1 10 13120 8880 8884 3711 3712 := CountSegment.join within_069_046 within_069_047
private theorem within_069_048_agreement : RowAgreement 8880 8884 := RowAgreement.join within_069_046_agreement within_069_047_agreement
private def within_069_049 : CountSegment 1 10 13120 8884 8886 3712 3713 := CountSegment.join atomic_8885 atomic_8886
private theorem within_069_049_agreement : RowAgreement 8884 8886 := RowAgreement.join atomic_8885_agreement atomic_8886_agreement
private def within_069_050 : CountSegment 1 10 13120 8886 8888 3713 3713 := CountSegment.join atomic_8887 atomic_8888
private theorem within_069_050_agreement : RowAgreement 8886 8888 := RowAgreement.join atomic_8887_agreement atomic_8888_agreement
private def within_069_051 : CountSegment 1 10 13120 8884 8888 3712 3713 := CountSegment.join within_069_049 within_069_050
private theorem within_069_051_agreement : RowAgreement 8884 8888 := RowAgreement.join within_069_049_agreement within_069_050_agreement
private def within_069_052 : CountSegment 1 10 13120 8880 8888 3711 3713 := CountSegment.join within_069_048 within_069_051
private theorem within_069_052_agreement : RowAgreement 8880 8888 := RowAgreement.join within_069_048_agreement within_069_051_agreement
private def within_069_053 : CountSegment 1 10 13120 8888 8890 3713 3713 := CountSegment.join atomic_8889 atomic_8890
private theorem within_069_053_agreement : RowAgreement 8888 8890 := RowAgreement.join atomic_8889_agreement atomic_8890_agreement
private def within_069_054 : CountSegment 1 10 13120 8890 8892 3713 3714 := CountSegment.join atomic_8891 atomic_8892
private theorem within_069_054_agreement : RowAgreement 8890 8892 := RowAgreement.join atomic_8891_agreement atomic_8892_agreement
private def within_069_055 : CountSegment 1 10 13120 8888 8892 3713 3714 := CountSegment.join within_069_053 within_069_054
private theorem within_069_055_agreement : RowAgreement 8888 8892 := RowAgreement.join within_069_053_agreement within_069_054_agreement
private def within_069_056 : CountSegment 1 10 13120 8892 8894 3714 3714 := CountSegment.join atomic_8893 atomic_8894
private theorem within_069_056_agreement : RowAgreement 8892 8894 := RowAgreement.join atomic_8893_agreement atomic_8894_agreement
private def within_069_057 : CountSegment 1 10 13120 8894 8896 3714 3715 := CountSegment.join atomic_8895 atomic_8896
private theorem within_069_057_agreement : RowAgreement 8894 8896 := RowAgreement.join atomic_8895_agreement atomic_8896_agreement
private def within_069_058 : CountSegment 1 10 13120 8892 8896 3714 3715 := CountSegment.join within_069_056 within_069_057
private theorem within_069_058_agreement : RowAgreement 8892 8896 := RowAgreement.join within_069_056_agreement within_069_057_agreement
private def within_069_059 : CountSegment 1 10 13120 8888 8896 3713 3715 := CountSegment.join within_069_055 within_069_058
private theorem within_069_059_agreement : RowAgreement 8888 8896 := RowAgreement.join within_069_055_agreement within_069_058_agreement
private def within_069_060 : CountSegment 1 10 13120 8880 8896 3711 3715 := CountSegment.join within_069_052 within_069_059
private theorem within_069_060_agreement : RowAgreement 8880 8896 := RowAgreement.join within_069_052_agreement within_069_059_agreement
private def within_069_061 : CountSegment 1 10 13120 8864 8896 3707 3715 := CountSegment.join within_069_045 within_069_060
private theorem within_069_061_agreement : RowAgreement 8864 8896 := RowAgreement.join within_069_045_agreement within_069_060_agreement
private def within_069_062 : CountSegment 1 10 13120 8832 8896 3694 3715 := CountSegment.join within_069_030 within_069_061
private theorem within_069_062_agreement : RowAgreement 8832 8896 := RowAgreement.join within_069_030_agreement within_069_061_agreement
private def within_069_063 : CountSegment 1 10 13120 8896 8898 3715 3716 := CountSegment.join atomic_8897 atomic_8898
private theorem within_069_063_agreement : RowAgreement 8896 8898 := RowAgreement.join atomic_8897_agreement atomic_8898_agreement
private def within_069_064 : CountSegment 1 10 13120 8898 8900 3716 3716 := CountSegment.join atomic_8899 atomic_8900
private theorem within_069_064_agreement : RowAgreement 8898 8900 := RowAgreement.join atomic_8899_agreement atomic_8900_agreement
private def within_069_065 : CountSegment 1 10 13120 8896 8900 3715 3716 := CountSegment.join within_069_063 within_069_064
private theorem within_069_065_agreement : RowAgreement 8896 8900 := RowAgreement.join within_069_063_agreement within_069_064_agreement
private def within_069_066 : CountSegment 1 10 13120 8900 8902 3716 3719 := CountSegment.join atomic_8901 atomic_8902
private theorem within_069_066_agreement : RowAgreement 8900 8902 := RowAgreement.join atomic_8901_agreement atomic_8902_agreement
private def within_069_067 : CountSegment 1 10 13120 8902 8904 3719 3719 := CountSegment.join atomic_8903 atomic_8904
private theorem within_069_067_agreement : RowAgreement 8902 8904 := RowAgreement.join atomic_8903_agreement atomic_8904_agreement
private def within_069_068 : CountSegment 1 10 13120 8900 8904 3716 3719 := CountSegment.join within_069_066 within_069_067
private theorem within_069_068_agreement : RowAgreement 8900 8904 := RowAgreement.join within_069_066_agreement within_069_067_agreement
private def within_069_069 : CountSegment 1 10 13120 8896 8904 3715 3719 := CountSegment.join within_069_065 within_069_068
private theorem within_069_069_agreement : RowAgreement 8896 8904 := RowAgreement.join within_069_065_agreement within_069_068_agreement
private def within_069_070 : CountSegment 1 10 13120 8904 8906 3719 3720 := CountSegment.join atomic_8905 atomic_8906
private theorem within_069_070_agreement : RowAgreement 8904 8906 := RowAgreement.join atomic_8905_agreement atomic_8906_agreement
private def within_069_071 : CountSegment 1 10 13120 8906 8908 3720 3721 := CountSegment.join atomic_8907 atomic_8908
private theorem within_069_071_agreement : RowAgreement 8906 8908 := RowAgreement.join atomic_8907_agreement atomic_8908_agreement
private def within_069_072 : CountSegment 1 10 13120 8904 8908 3719 3721 := CountSegment.join within_069_070 within_069_071
private theorem within_069_072_agreement : RowAgreement 8904 8908 := RowAgreement.join within_069_070_agreement within_069_071_agreement
private def within_069_073 : CountSegment 1 10 13120 8908 8910 3721 3721 := CountSegment.join atomic_8909 atomic_8910
private theorem within_069_073_agreement : RowAgreement 8908 8910 := RowAgreement.join atomic_8909_agreement atomic_8910_agreement
private def within_069_074 : CountSegment 1 10 13120 8910 8912 3721 3722 := CountSegment.join atomic_8911 atomic_8912
private theorem within_069_074_agreement : RowAgreement 8910 8912 := RowAgreement.join atomic_8911_agreement atomic_8912_agreement
private def within_069_075 : CountSegment 1 10 13120 8908 8912 3721 3722 := CountSegment.join within_069_073 within_069_074
private theorem within_069_075_agreement : RowAgreement 8908 8912 := RowAgreement.join within_069_073_agreement within_069_074_agreement
private def within_069_076 : CountSegment 1 10 13120 8904 8912 3719 3722 := CountSegment.join within_069_072 within_069_075
private theorem within_069_076_agreement : RowAgreement 8904 8912 := RowAgreement.join within_069_072_agreement within_069_075_agreement
private def within_069_077 : CountSegment 1 10 13120 8896 8912 3715 3722 := CountSegment.join within_069_069 within_069_076
private theorem within_069_077_agreement : RowAgreement 8896 8912 := RowAgreement.join within_069_069_agreement within_069_076_agreement
private def within_069_078 : CountSegment 1 10 13120 8912 8914 3722 3723 := CountSegment.join atomic_8913 atomic_8914
private theorem within_069_078_agreement : RowAgreement 8912 8914 := RowAgreement.join atomic_8913_agreement atomic_8914_agreement
private def within_069_079 : CountSegment 1 10 13120 8914 8916 3723 3724 := CountSegment.join atomic_8915 atomic_8916
private theorem within_069_079_agreement : RowAgreement 8914 8916 := RowAgreement.join atomic_8915_agreement atomic_8916_agreement
private def within_069_080 : CountSegment 1 10 13120 8912 8916 3722 3724 := CountSegment.join within_069_078 within_069_079
private theorem within_069_080_agreement : RowAgreement 8912 8916 := RowAgreement.join within_069_078_agreement within_069_079_agreement
private def within_069_081 : CountSegment 1 10 13120 8916 8918 3724 3724 := CountSegment.join atomic_8917 atomic_8918
private theorem within_069_081_agreement : RowAgreement 8916 8918 := RowAgreement.join atomic_8917_agreement atomic_8918_agreement
private def within_069_082 : CountSegment 1 10 13120 8918 8920 3724 3725 := CountSegment.join atomic_8919 atomic_8920
private theorem within_069_082_agreement : RowAgreement 8918 8920 := RowAgreement.join atomic_8919_agreement atomic_8920_agreement
private def within_069_083 : CountSegment 1 10 13120 8916 8920 3724 3725 := CountSegment.join within_069_081 within_069_082
private theorem within_069_083_agreement : RowAgreement 8916 8920 := RowAgreement.join within_069_081_agreement within_069_082_agreement
private def within_069_084 : CountSegment 1 10 13120 8912 8920 3722 3725 := CountSegment.join within_069_080 within_069_083
private theorem within_069_084_agreement : RowAgreement 8912 8920 := RowAgreement.join within_069_080_agreement within_069_083_agreement
private def within_069_085 : CountSegment 1 10 13120 8920 8922 3725 3725 := CountSegment.join atomic_8921 atomic_8922
private theorem within_069_085_agreement : RowAgreement 8920 8922 := RowAgreement.join atomic_8921_agreement atomic_8922_agreement
private def within_069_086 : CountSegment 1 10 13120 8922 8924 3725 3726 := CountSegment.join atomic_8923 atomic_8924
private theorem within_069_086_agreement : RowAgreement 8922 8924 := RowAgreement.join atomic_8923_agreement atomic_8924_agreement
private def within_069_087 : CountSegment 1 10 13120 8920 8924 3725 3726 := CountSegment.join within_069_085 within_069_086
private theorem within_069_087_agreement : RowAgreement 8920 8924 := RowAgreement.join within_069_085_agreement within_069_086_agreement
private def within_069_088 : CountSegment 1 10 13120 8924 8926 3726 3727 := CountSegment.join atomic_8925 atomic_8926
private theorem within_069_088_agreement : RowAgreement 8924 8926 := RowAgreement.join atomic_8925_agreement atomic_8926_agreement
private def within_069_089 : CountSegment 1 10 13120 8926 8928 3727 3730 := CountSegment.join atomic_8927 atomic_8928
private theorem within_069_089_agreement : RowAgreement 8926 8928 := RowAgreement.join atomic_8927_agreement atomic_8928_agreement
private def within_069_090 : CountSegment 1 10 13120 8924 8928 3726 3730 := CountSegment.join within_069_088 within_069_089
private theorem within_069_090_agreement : RowAgreement 8924 8928 := RowAgreement.join within_069_088_agreement within_069_089_agreement
private def within_069_091 : CountSegment 1 10 13120 8920 8928 3725 3730 := CountSegment.join within_069_087 within_069_090
private theorem within_069_091_agreement : RowAgreement 8920 8928 := RowAgreement.join within_069_087_agreement within_069_090_agreement
private def within_069_092 : CountSegment 1 10 13120 8912 8928 3722 3730 := CountSegment.join within_069_084 within_069_091
private theorem within_069_092_agreement : RowAgreement 8912 8928 := RowAgreement.join within_069_084_agreement within_069_091_agreement
private def within_069_093 : CountSegment 1 10 13120 8896 8928 3715 3730 := CountSegment.join within_069_077 within_069_092
private theorem within_069_093_agreement : RowAgreement 8896 8928 := RowAgreement.join within_069_077_agreement within_069_092_agreement
private def within_069_094 : CountSegment 1 10 13120 8928 8930 3730 3731 := CountSegment.join atomic_8929 atomic_8930
private theorem within_069_094_agreement : RowAgreement 8928 8930 := RowAgreement.join atomic_8929_agreement atomic_8930_agreement
private def within_069_095 : CountSegment 1 10 13120 8930 8932 3731 3732 := CountSegment.join atomic_8931 atomic_8932
private theorem within_069_095_agreement : RowAgreement 8930 8932 := RowAgreement.join atomic_8931_agreement atomic_8932_agreement
private def within_069_096 : CountSegment 1 10 13120 8928 8932 3730 3732 := CountSegment.join within_069_094 within_069_095
private theorem within_069_096_agreement : RowAgreement 8928 8932 := RowAgreement.join within_069_094_agreement within_069_095_agreement
private def within_069_097 : CountSegment 1 10 13120 8932 8934 3732 3732 := CountSegment.join atomic_8933 atomic_8934
private theorem within_069_097_agreement : RowAgreement 8932 8934 := RowAgreement.join atomic_8933_agreement atomic_8934_agreement
private def within_069_098 : CountSegment 1 10 13120 8934 8936 3732 3732 := CountSegment.join atomic_8935 atomic_8936
private theorem within_069_098_agreement : RowAgreement 8934 8936 := RowAgreement.join atomic_8935_agreement atomic_8936_agreement
private def within_069_099 : CountSegment 1 10 13120 8932 8936 3732 3732 := CountSegment.join within_069_097 within_069_098
private theorem within_069_099_agreement : RowAgreement 8932 8936 := RowAgreement.join within_069_097_agreement within_069_098_agreement
private def within_069_100 : CountSegment 1 10 13120 8928 8936 3730 3732 := CountSegment.join within_069_096 within_069_099
private theorem within_069_100_agreement : RowAgreement 8928 8936 := RowAgreement.join within_069_096_agreement within_069_099_agreement
private def within_069_101 : CountSegment 1 10 13120 8936 8938 3732 3732 := CountSegment.join atomic_8937 atomic_8938
private theorem within_069_101_agreement : RowAgreement 8936 8938 := RowAgreement.join atomic_8937_agreement atomic_8938_agreement
private def within_069_102 : CountSegment 1 10 13120 8938 8940 3732 3732 := CountSegment.join atomic_8939 atomic_8940
private theorem within_069_102_agreement : RowAgreement 8938 8940 := RowAgreement.join atomic_8939_agreement atomic_8940_agreement
private def within_069_103 : CountSegment 1 10 13120 8936 8940 3732 3732 := CountSegment.join within_069_101 within_069_102
private theorem within_069_103_agreement : RowAgreement 8936 8940 := RowAgreement.join within_069_101_agreement within_069_102_agreement
private def within_069_104 : CountSegment 1 10 13120 8940 8942 3732 3732 := CountSegment.join atomic_8941 atomic_8942
private theorem within_069_104_agreement : RowAgreement 8940 8942 := RowAgreement.join atomic_8941_agreement atomic_8942_agreement
private def within_069_105 : CountSegment 1 10 13120 8942 8944 3732 3733 := CountSegment.join atomic_8943 atomic_8944
private theorem within_069_105_agreement : RowAgreement 8942 8944 := RowAgreement.join atomic_8943_agreement atomic_8944_agreement
private def within_069_106 : CountSegment 1 10 13120 8940 8944 3732 3733 := CountSegment.join within_069_104 within_069_105
private theorem within_069_106_agreement : RowAgreement 8940 8944 := RowAgreement.join within_069_104_agreement within_069_105_agreement
private def within_069_107 : CountSegment 1 10 13120 8936 8944 3732 3733 := CountSegment.join within_069_103 within_069_106
private theorem within_069_107_agreement : RowAgreement 8936 8944 := RowAgreement.join within_069_103_agreement within_069_106_agreement
private def within_069_108 : CountSegment 1 10 13120 8928 8944 3730 3733 := CountSegment.join within_069_100 within_069_107
private theorem within_069_108_agreement : RowAgreement 8928 8944 := RowAgreement.join within_069_100_agreement within_069_107_agreement
private def within_069_109 : CountSegment 1 10 13120 8944 8946 3733 3734 := CountSegment.join atomic_8945 atomic_8946
private theorem within_069_109_agreement : RowAgreement 8944 8946 := RowAgreement.join atomic_8945_agreement atomic_8946_agreement
private def within_069_110 : CountSegment 1 10 13120 8946 8948 3734 3735 := CountSegment.join atomic_8947 atomic_8948
private theorem within_069_110_agreement : RowAgreement 8946 8948 := RowAgreement.join atomic_8947_agreement atomic_8948_agreement
private def within_069_111 : CountSegment 1 10 13120 8944 8948 3733 3735 := CountSegment.join within_069_109 within_069_110
private theorem within_069_111_agreement : RowAgreement 8944 8948 := RowAgreement.join within_069_109_agreement within_069_110_agreement
private def within_069_112 : CountSegment 1 10 13120 8948 8950 3735 3737 := CountSegment.join atomic_8949 atomic_8950
private theorem within_069_112_agreement : RowAgreement 8948 8950 := RowAgreement.join atomic_8949_agreement atomic_8950_agreement
private def within_069_113 : CountSegment 1 10 13120 8950 8952 3737 3738 := CountSegment.join atomic_8951 atomic_8952
private theorem within_069_113_agreement : RowAgreement 8950 8952 := RowAgreement.join atomic_8951_agreement atomic_8952_agreement
private def within_069_114 : CountSegment 1 10 13120 8948 8952 3735 3738 := CountSegment.join within_069_112 within_069_113
private theorem within_069_114_agreement : RowAgreement 8948 8952 := RowAgreement.join within_069_112_agreement within_069_113_agreement
private def within_069_115 : CountSegment 1 10 13120 8944 8952 3733 3738 := CountSegment.join within_069_111 within_069_114
private theorem within_069_115_agreement : RowAgreement 8944 8952 := RowAgreement.join within_069_111_agreement within_069_114_agreement
private def within_069_116 : CountSegment 1 10 13120 8952 8954 3738 3739 := CountSegment.join atomic_8953 atomic_8954
private theorem within_069_116_agreement : RowAgreement 8952 8954 := RowAgreement.join atomic_8953_agreement atomic_8954_agreement
private def within_069_117 : CountSegment 1 10 13120 8954 8956 3739 3739 := CountSegment.join atomic_8955 atomic_8956
private theorem within_069_117_agreement : RowAgreement 8954 8956 := RowAgreement.join atomic_8955_agreement atomic_8956_agreement
private def within_069_118 : CountSegment 1 10 13120 8952 8956 3738 3739 := CountSegment.join within_069_116 within_069_117
private theorem within_069_118_agreement : RowAgreement 8952 8956 := RowAgreement.join within_069_116_agreement within_069_117_agreement
private def within_069_119 : CountSegment 1 10 13120 8956 8958 3739 3741 := CountSegment.join atomic_8957 atomic_8958
private theorem within_069_119_agreement : RowAgreement 8956 8958 := RowAgreement.join atomic_8957_agreement atomic_8958_agreement
private def within_069_120 : CountSegment 1 10 13120 8958 8960 3741 3741 := CountSegment.join atomic_8959 atomic_8960
private theorem within_069_120_agreement : RowAgreement 8958 8960 := RowAgreement.join atomic_8959_agreement atomic_8960_agreement
private def within_069_121 : CountSegment 1 10 13120 8956 8960 3739 3741 := CountSegment.join within_069_119 within_069_120
private theorem within_069_121_agreement : RowAgreement 8956 8960 := RowAgreement.join within_069_119_agreement within_069_120_agreement
private def within_069_122 : CountSegment 1 10 13120 8952 8960 3738 3741 := CountSegment.join within_069_118 within_069_121
private theorem within_069_122_agreement : RowAgreement 8952 8960 := RowAgreement.join within_069_118_agreement within_069_121_agreement
private def within_069_123 : CountSegment 1 10 13120 8944 8960 3733 3741 := CountSegment.join within_069_115 within_069_122
private theorem within_069_123_agreement : RowAgreement 8944 8960 := RowAgreement.join within_069_115_agreement within_069_122_agreement
private def within_069_124 : CountSegment 1 10 13120 8928 8960 3730 3741 := CountSegment.join within_069_108 within_069_123
private theorem within_069_124_agreement : RowAgreement 8928 8960 := RowAgreement.join within_069_108_agreement within_069_123_agreement
private def within_069_125 : CountSegment 1 10 13120 8896 8960 3715 3741 := CountSegment.join within_069_093 within_069_124
private theorem within_069_125_agreement : RowAgreement 8896 8960 := RowAgreement.join within_069_093_agreement within_069_124_agreement
private def within_069_126 : CountSegment 1 10 13120 8832 8960 3694 3741 := CountSegment.join within_069_062 within_069_125
private theorem within_069_126_agreement : RowAgreement 8832 8960 := RowAgreement.join within_069_062_agreement within_069_125_agreement
def segment_069 : CountSegment 1 10 13120 8832 8960 3694 3741 := within_069_126
theorem agreement_069 : RowAgreement 8832 8960 := within_069_126_agreement
end ProjectEulerSolutions.P58Numerical
