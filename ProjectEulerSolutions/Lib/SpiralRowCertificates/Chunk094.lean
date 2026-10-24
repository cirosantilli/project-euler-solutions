import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_12033 : RowCertificate 12033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12033 1) := by change ¬ Nat.Prime 579196423; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44553571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12033 2) := by change Nat.Prime 579172357; exact ProjectEulerSolutions.P58PrimeNodes.p_579172357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12033 3) := by change ¬ Nat.Prime 579148291; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30481489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12033 : row 12033 = 1 := (row_12033.checked).trans rfl
theorem model_count_12033 : modelRow 12033 = 1 := by decide
theorem modelrow_agrees_12033 : modelRow 12033 = row 12033 := model_count_12033.trans row_count_12033.symm
def row_12034 : RowCertificate 12034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12034 1) := by change ¬ Nat.Prime 579292693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82756099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12034 2) := by change ¬ Nat.Prime 579268625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115853725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12034 3) := by change ¬ Nat.Prime 579244557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193081519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12034 : row 12034 = 0 := (row_12034.checked).trans rfl
theorem model_count_12034 : modelRow 12034 = 0 := by decide
theorem modelrow_agrees_12034 : modelRow 12034 = row 12034 := model_count_12034.trans row_count_12034.symm
def row_12035 : RowCertificate 12035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12035 1) := by change ¬ Nat.Prime 579388971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193129657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12035 2) := by change Nat.Prime 579364901; exact ProjectEulerSolutions.P58PrimeNodes.p_579364901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12035 3) := by change ¬ Nat.Prime 579340831; exact PrattCertificates.composite_of_product (factor := 607) (cofactor := 954433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12035 : row 12035 = 1 := (row_12035.checked).trans rfl
theorem model_count_12035 : modelRow 12035 = 1 := by decide
theorem modelrow_agrees_12035 : modelRow 12035 = row 12035 := model_count_12035.trans row_count_12035.symm
def row_12036 : RowCertificate 12036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12036 1) := by change ¬ Nat.Prime 579485257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44575789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12036 2) := by change ¬ Nat.Prime 579461185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115892237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12036 3) := by change ¬ Nat.Prime 579437113; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7334647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12036 : row 12036 = 0 := (row_12036.checked).trans rfl
theorem model_count_12036 : modelRow 12036 = 0 := by decide
theorem modelrow_agrees_12036 : modelRow 12036 = row 12036 := model_count_12036.trans row_count_12036.symm
def row_12037 : RowCertificate 12037 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12037 1) := by change Nat.Prime 579581551; exact ProjectEulerSolutions.P58PrimeNodes.p_579581551.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12037 2) := by change ¬ Nat.Prime 579557477; exact PrattCertificates.composite_of_product (factor := 5737) (cofactor := 101021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12037 3) := by change ¬ Nat.Prime 579533403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193177801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12037 : row 12037 = 1 := (row_12037.checked).trans rfl
theorem model_count_12037 : modelRow 12037 = 1 := by decide
theorem modelrow_agrees_12037 : modelRow 12037 = row 12037 := model_count_12037.trans row_count_12037.symm
def row_12038 : RowCertificate 12038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12038 1) := by change ¬ Nat.Prime 579677853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193225951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12038 2) := by change ¬ Nat.Prime 579653777; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34097281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12038 3) := by change ¬ Nat.Prime 579629701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82804243) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12038 : row 12038 = 0 := (row_12038.checked).trans rfl
theorem model_count_12038 : modelRow 12038 = 0 := by decide
theorem modelrow_agrees_12038 : modelRow 12038 = row 12038 := model_count_12038.trans row_count_12038.symm
def row_12039 : RowCertificate 12039 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12039 1) := by change Nat.Prime 579774163; exact ProjectEulerSolutions.P58PrimeNodes.p_579774163.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12039 2) := by change ¬ Nat.Prime 579750085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115950017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12039 3) := by change ¬ Nat.Prime 579726007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82818001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12039 : row 12039 = 1 := (row_12039.checked).trans rfl
theorem model_count_12039 : modelRow 12039 = 1 := by decide
theorem modelrow_agrees_12039 : modelRow 12039 = row 12039 := model_count_12039.trans row_count_12039.symm
def row_12040 : RowCertificate 12040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12040 1) := by change ¬ Nat.Prime 579870481; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30519499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12040 2) := by change ¬ Nat.Prime 579846401; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 1446001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12040 3) := by change ¬ Nat.Prime 579822321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193274107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12040 : row 12040 = 0 := (row_12040.checked).trans rfl
theorem model_count_12040 : modelRow 12040 = 0 := by decide
theorem modelrow_agrees_12040 : modelRow 12040 = row 12040 := model_count_12040.trans row_count_12040.symm
def row_12041 : RowCertificate 12041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12041 1) := by change ¬ Nat.Prime 579966807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193322269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12041 2) := by change ¬ Nat.Prime 579942725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115988545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12041 3) := by change ¬ Nat.Prime 579918643; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18707053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12041 : row 12041 = 0 := (row_12041.checked).trans rfl
theorem model_count_12041 : modelRow 12041 = 0 := by decide
theorem modelrow_agrees_12041 : modelRow 12041 = row 12041 := model_count_12041.trans row_count_12041.symm
def row_12042 : RowCertificate 12042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12042 1) := by change ¬ Nat.Prime 580063141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82866163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12042 2) := by change ¬ Nat.Prime 580039057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44618389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12042 3) := by change Nat.Prime 580014973; exact ProjectEulerSolutions.P58PrimeNodes.p_580014973.prime
    exact decide_eq_true prime
theorem row_count_12042 : row 12042 = 1 := (row_12042.checked).trans rfl
theorem model_count_12042 : modelRow 12042 = 1 := by decide
theorem modelrow_agrees_12042 : modelRow 12042 = row 12042 := model_count_12042.trans row_count_12042.symm
def row_12043 : RowCertificate 12043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12043 1) := by change ¬ Nat.Prime 580159483; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13492081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12043 2) := by change Nat.Prime 580135397; exact ProjectEulerSolutions.P58PrimeNodes.p_580135397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12043 3) := by change ¬ Nat.Prime 580111311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193370437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12043 : row 12043 = 1 := (row_12043.checked).trans rfl
theorem model_count_12043 : modelRow 12043 = 1 := by decide
theorem modelrow_agrees_12043 : modelRow 12043 = row 12043 := model_count_12043.trans row_count_12043.symm
def row_12044 : RowCertificate 12044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12044 1) := by change ¬ Nat.Prime 580255833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193418611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12044 2) := by change ¬ Nat.Prime 580231745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116046349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12044 3) := by change ¬ Nat.Prime 580207657; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 1269601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12044 : row 12044 = 0 := (row_12044.checked).trans rfl
theorem model_count_12044 : modelRow 12044 = 0 := by decide
theorem modelrow_agrees_12044 : modelRow 12044 = row 12044 := model_count_12044.trans row_count_12044.symm
def row_12045 : RowCertificate 12045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12045 1) := by change ¬ Nat.Prime 580352191; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8661973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12045 2) := by change ¬ Nat.Prime 580328101; exact PrattCertificates.composite_of_product (factor := 2861) (cofactor := 202841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12045 3) := by change ¬ Nat.Prime 580304011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82900573) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12045 : row 12045 = 0 := (row_12045.checked).trans rfl
theorem model_count_12045 : modelRow 12045 = 0 := by decide
theorem modelrow_agrees_12045 : modelRow 12045 = row 12045 := model_count_12045.trans row_count_12045.symm
def row_12046 : RowCertificate 12046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12046 1) := by change ¬ Nat.Prime 580448557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44649889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12046 2) := by change ¬ Nat.Prime 580424465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116084893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12046 3) := by change ¬ Nat.Prime 580400373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193466791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12046 : row 12046 = 0 := (row_12046.checked).trans rfl
theorem model_count_12046 : modelRow 12046 = 0 := by decide
theorem modelrow_agrees_12046 : modelRow 12046 = row 12046 := model_count_12046.trans row_count_12046.symm
def row_12047 : RowCertificate 12047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12047 1) := by change ¬ Nat.Prime 580544931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193514977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12047 2) := by change ¬ Nat.Prime 580520837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44655449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12047 3) := by change ¬ Nat.Prime 580496743; exact PrattCertificates.composite_of_product (factor := 9931) (cofactor := 58453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12047 : row 12047 = 0 := (row_12047.checked).trans rfl
theorem model_count_12047 : modelRow 12047 = 0 := by decide
theorem modelrow_agrees_12047 : modelRow 12047 = row 12047 := model_count_12047.trans row_count_12047.symm
def row_12048 : RowCertificate 12048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12048 1) := by change ¬ Nat.Prime 580641313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82948759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12048 2) := by change Nat.Prime 580617217; exact ProjectEulerSolutions.P58PrimeNodes.p_580617217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12048 3) := by change ¬ Nat.Prime 580593121; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 1192183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12048 : row 12048 = 1 := (row_12048.checked).trans rfl
theorem model_count_12048 : modelRow 12048 = 1 := by decide
theorem modelrow_agrees_12048 : modelRow 12048 = row 12048 := model_count_12048.trans row_count_12048.symm
def row_12049 : RowCertificate 12049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12049 1) := by change ¬ Nat.Prime 580737703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82962529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12049 2) := by change ¬ Nat.Prime 580713605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116142721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12049 3) := by change ¬ Nat.Prime 580689507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193563169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12049 : row 12049 = 0 := (row_12049.checked).trans rfl
theorem model_count_12049 : modelRow 12049 = 0 := by decide
theorem modelrow_agrees_12049 : modelRow 12049 = row 12049 := model_count_12049.trans row_count_12049.symm
def row_12050 : RowCertificate 12050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12050 1) := by change ¬ Nat.Prime 580834101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193611367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12050 2) := by change ¬ Nat.Prime 580810001; exact PrattCertificates.composite_of_product (factor := 15473) (cofactor := 37537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12050 3) := by change ¬ Nat.Prime 580785901; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30567679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12050 : row 12050 = 0 := (row_12050.checked).trans rfl
theorem model_count_12050 : modelRow 12050 = 0 := by decide
theorem modelrow_agrees_12050 : modelRow 12050 = row 12050 := model_count_12050.trans row_count_12050.symm
def row_12051 : RowCertificate 12051 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12051 1) := by change Nat.Prime 580930507; exact ProjectEulerSolutions.P58PrimeNodes.p_580930507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12051 2) := by change ¬ Nat.Prime 580906405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116181281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12051 3) := by change Nat.Prime 580882303; exact ProjectEulerSolutions.P58PrimeNodes.p_580882303.prime
    exact decide_eq_true prime
theorem row_count_12051 : row 12051 = 2 := (row_12051.checked).trans rfl
theorem model_count_12051 : modelRow 12051 = 2 := by decide
theorem modelrow_agrees_12051 : modelRow 12051 = row 12051 := model_count_12051.trans row_count_12051.symm
def row_12052 : RowCertificate 12052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12052 1) := by change ¬ Nat.Prime 581026921; exact PrattCertificates.composite_of_product (factor := 751) (cofactor := 773671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12052 2) := by change ¬ Nat.Prime 581002817; exact PrattCertificates.composite_of_product (factor := 821) (cofactor := 707677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12052 3) := by change ¬ Nat.Prime 580978713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193659571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12052 : row 12052 = 0 := (row_12052.checked).trans rfl
theorem model_count_12052 : modelRow 12052 = 0 := by decide
theorem modelrow_agrees_12052 : modelRow 12052 = row 12052 := model_count_12052.trans row_count_12052.symm
def row_12053 : RowCertificate 12053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12053 1) := by change ¬ Nat.Prime 581123343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193707781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12053 2) := by change ¬ Nat.Prime 581099237; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9526217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12053 3) := by change ¬ Nat.Prime 581075131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83010733) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12053 : row 12053 = 0 := (row_12053.checked).trans rfl
theorem model_count_12053 : modelRow 12053 = 0 := by decide
theorem modelrow_agrees_12053 : modelRow 12053 = row 12053 := model_count_12053.trans row_count_12053.symm
def row_12054 : RowCertificate 12054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12054 1) := by change ¬ Nat.Prime 581219773; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9528193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12054 2) := by change ¬ Nat.Prime 581195665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116239133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12054 3) := by change Nat.Prime 581171557; exact ProjectEulerSolutions.P58PrimeNodes.p_581171557.prime
    exact decide_eq_true prime
theorem row_count_12054 : row 12054 = 1 := (row_12054.checked).trans rfl
theorem model_count_12054 : modelRow 12054 = 1 := by decide
theorem modelrow_agrees_12054 : modelRow 12054 = row 12054 := model_count_12054.trans row_count_12054.symm
def row_12055 : RowCertificate 12055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12055 1) := by change ¬ Nat.Prime 581316211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83045173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12055 2) := by change ¬ Nat.Prime 581292101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44714777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12055 3) := by change ¬ Nat.Prime 581267991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193755997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12055 : row 12055 = 0 := (row_12055.checked).trans rfl
theorem model_count_12055 : modelRow 12055 = 0 := by decide
theorem modelrow_agrees_12055 : modelRow 12055 = row 12055 := model_count_12055.trans row_count_12055.symm
def row_12056 : RowCertificate 12056 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12056 1) := by change ¬ Nat.Prime 581412657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193804219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12056 2) := by change ¬ Nat.Prime 581388545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116277709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12056 3) := by change ¬ Nat.Prime 581364433; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44720341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12056 : row 12056 = 0 := (row_12056.checked).trans rfl
theorem model_count_12056 : modelRow 12056 = 0 := by decide
theorem modelrow_agrees_12056 : modelRow 12056 = row 12056 := model_count_12056.trans row_count_12056.symm
def row_12057 : RowCertificate 12057 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12057 1) := by change Nat.Prime 581509111; exact ProjectEulerSolutions.P58PrimeNodes.p_581509111.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12057 2) := by change ¬ Nat.Prime 581484997; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 562909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12057 3) := by change ¬ Nat.Prime 581460883; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15715159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12057 : row 12057 = 1 := (row_12057.checked).trans rfl
theorem model_count_12057 : modelRow 12057 = 1 := by decide
theorem modelrow_agrees_12057 : modelRow 12057 = row 12057 := model_count_12057.trans row_count_12057.symm
def row_12058 : RowCertificate 12058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12058 1) := by change ¬ Nat.Prime 581605573; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13525711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12058 2) := by change ¬ Nat.Prime 581581457; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20054533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12058 3) := by change ¬ Nat.Prime 581557341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193852447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12058 : row 12058 = 0 := (row_12058.checked).trans rfl
theorem model_count_12058 : modelRow 12058 = 0 := by decide
theorem modelrow_agrees_12058 : modelRow 12058 = row 12058 := model_count_12058.trans row_count_12058.symm
def row_12059 : RowCertificate 12059 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12059 1) := by change ¬ Nat.Prime 581702043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193900681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12059 2) := by change ¬ Nat.Prime 581677925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116335585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12059 3) := by change ¬ Nat.Prime 581653807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83093401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12059 : row 12059 = 0 := (row_12059.checked).trans rfl
theorem model_count_12059 : modelRow 12059 = 0 := by decide
theorem modelrow_agrees_12059 : modelRow 12059 = row 12059 := model_count_12059.trans row_count_12059.symm
def row_12060 : RowCertificate 12060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12060 1) := by change ¬ Nat.Prime 581798521; exact PrattCertificates.composite_of_product (factor := 5923) (cofactor := 98227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12060 2) := by change ¬ Nat.Prime 581774401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44751877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12060 3) := by change ¬ Nat.Prime 581750281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83107183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12060 : row 12060 = 0 := (row_12060.checked).trans rfl
theorem model_count_12060 : modelRow 12060 = 0 := by decide
theorem modelrow_agrees_12060 : modelRow 12060 = row 12060 := model_count_12060.trans row_count_12060.symm
def row_12061 : RowCertificate 12061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12061 1) := by change ¬ Nat.Prime 581895007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30626053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12061 2) := by change ¬ Nat.Prime 581870885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116374177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12061 3) := by change ¬ Nat.Prime 581846763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193948921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12061 : row 12061 = 0 := (row_12061.checked).trans rfl
theorem model_count_12061 : modelRow 12061 = 0 := by decide
theorem modelrow_agrees_12061 : modelRow 12061 = row 12061 := model_count_12061.trans row_count_12061.symm
def row_12062 : RowCertificate 12062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12062 1) := by change ¬ Nat.Prime 581991501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193997167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12062 2) := by change Nat.Prime 581967377; exact ProjectEulerSolutions.P58PrimeNodes.p_581967377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12062 3) := by change ¬ Nat.Prime 581943253; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18772363) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12062 : row 12062 = 1 := (row_12062.checked).trans rfl
theorem model_count_12062 : modelRow 12062 = 1 := by decide
theorem modelrow_agrees_12062 : modelRow 12062 = row 12062 := model_count_12062.trans row_count_12062.symm
def row_12063 : RowCertificate 12063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12063 1) := by change ¬ Nat.Prime 582088003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83155429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12063 2) := by change ¬ Nat.Prime 582063877; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1859629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12063 3) := by change Nat.Prime 582039751; exact ProjectEulerSolutions.P58PrimeNodes.p_582039751.prime
    exact decide_eq_true prime
theorem row_count_12063 : row 12063 = 1 := (row_12063.checked).trans rfl
theorem model_count_12063 : modelRow 12063 = 1 := by decide
theorem modelrow_agrees_12063 : modelRow 12063 = row 12063 := model_count_12063.trans row_count_12063.symm
def row_12064 : RowCertificate 12064 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12064 1) := by change Nat.Prime 582184513; exact ProjectEulerSolutions.P58PrimeNodes.p_582184513.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12064 2) := by change ¬ Nat.Prime 582160385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116432077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12064 3) := by change ¬ Nat.Prime 582136257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194045419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12064 : row 12064 = 1 := (row_12064.checked).trans rfl
theorem model_count_12064 : modelRow 12064 = 1 := by decide
theorem modelrow_agrees_12064 : modelRow 12064 = row 12064 := model_count_12064.trans row_count_12064.symm
def row_12065 : RowCertificate 12065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12065 1) := by change ¬ Nat.Prime 582281031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194093677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12065 2) := by change ¬ Nat.Prime 582256901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15736673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12065 3) := by change ¬ Nat.Prime 582232771; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13540297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12065 : row 12065 = 0 := (row_12065.checked).trans rfl
theorem model_count_12065 : modelRow 12065 = 0 := by decide
theorem modelrow_agrees_12065 : modelRow 12065 = row 12065 := model_count_12065.trans row_count_12065.symm
def row_12066 : RowCertificate 12066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12066 1) := by change ¬ Nat.Prime 582377557; exact PrattCertificates.composite_of_product (factor := 2731) (cofactor := 213247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12066 2) := by change ¬ Nat.Prime 582353425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116470685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12066 3) := by change ¬ Nat.Prime 582329293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83189899) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12066 : row 12066 = 0 := (row_12066.checked).trans rfl
theorem model_count_12066 : modelRow 12066 = 0 := by decide
theorem modelrow_agrees_12066 : modelRow 12066 = row 12066 := model_count_12066.trans row_count_12066.symm
def row_12067 : RowCertificate 12067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12067 1) := by change ¬ Nat.Prime 582474091; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15742543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12067 2) := by change Nat.Prime 582449957; exact ProjectEulerSolutions.P58PrimeNodes.p_582449957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12067 3) := by change ¬ Nat.Prime 582425823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194141941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12067 : row 12067 = 1 := (row_12067.checked).trans rfl
theorem model_count_12067 : modelRow 12067 = 1 := by decide
theorem modelrow_agrees_12067 : modelRow 12067 = row 12067 := model_count_12067.trans row_count_12067.symm
def row_12068 : RowCertificate 12068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12068 1) := by change ¬ Nat.Prime 582570633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194190211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12068 2) := by change ¬ Nat.Prime 582546497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44811269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12068 3) := by change ¬ Nat.Prime 582522361; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 1728553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12068 : row 12068 = 0 := (row_12068.checked).trans rfl
theorem model_count_12068 : modelRow 12068 = 0 := by decide
theorem modelrow_agrees_12068 : modelRow 12068 = row 12068 := model_count_12068.trans row_count_12068.symm
def row_12069 : RowCertificate 12069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12069 1) := by change ¬ Nat.Prime 582667183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83238169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12069 2) := by change ¬ Nat.Prime 582643045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116528609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12069 3) := by change ¬ Nat.Prime 582618907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44816839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12069 : row 12069 = 0 := (row_12069.checked).trans rfl
theorem model_count_12069 : modelRow 12069 = 0 := by decide
theorem modelrow_agrees_12069 : modelRow 12069 = row 12069 := model_count_12069.trans row_count_12069.symm
def row_12070 : RowCertificate 12070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12070 1) := by change ¬ Nat.Prime 582763741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83251963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12070 2) := by change ¬ Nat.Prime 582739601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20094469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12070 3) := by change ¬ Nat.Prime 582715461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194238487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12070 : row 12070 = 0 := (row_12070.checked).trans rfl
theorem model_count_12070 : modelRow 12070 = 0 := by decide
theorem modelrow_agrees_12070 : modelRow 12070 = row 12070 := model_count_12070.trans row_count_12070.symm
def row_12071 : RowCertificate 12071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12071 1) := by change ¬ Nat.Prime 582860307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194286769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12071 2) := by change ¬ Nat.Prime 582836165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116567233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12071 3) := by change ¬ Nat.Prime 582812023; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30674317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12071 : row 12071 = 0 := (row_12071.checked).trans rfl
theorem model_count_12071 : modelRow 12071 = 0 := by decide
theorem modelrow_agrees_12071 : modelRow 12071 = row 12071 := model_count_12071.trans row_count_12071.symm
def row_12072 : RowCertificate 12072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12072 1) := by change ¬ Nat.Prime 582956881; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44842837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12072 2) := by change ¬ Nat.Prime 582932737; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34290161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12072 3) := by change ¬ Nat.Prime 582908593; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18803503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12072 : row 12072 = 0 := (row_12072.checked).trans rfl
theorem model_count_12072 : modelRow 12072 = 0 := by decide
theorem modelrow_agrees_12072 : modelRow 12072 = row 12072 := model_count_12072.trans row_count_12072.symm
def row_12073 : RowCertificate 12073 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12073 1) := by change Nat.Prime 583053463; exact ProjectEulerSolutions.P58PrimeNodes.p_583053463.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12073 2) := by change ¬ Nat.Prime 583029317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44848409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12073 3) := by change ¬ Nat.Prime 583005171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194335057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12073 : row 12073 = 1 := (row_12073.checked).trans rfl
theorem model_count_12073 : modelRow 12073 = 1 := by decide
theorem modelrow_agrees_12073 : modelRow 12073 = row 12073 := model_count_12073.trans row_count_12073.symm
def row_12074 : RowCertificate 12074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12074 1) := by change ¬ Nat.Prime 583150053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194383351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12074 2) := by change ¬ Nat.Prime 583125905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116625181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12074 3) := by change ¬ Nat.Prime 583101757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83300251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12074 : row 12074 = 0 := (row_12074.checked).trans rfl
theorem model_count_12074 : modelRow 12074 = 0 := by decide
theorem modelrow_agrees_12074 : modelRow 12074 = row 12074 := model_count_12074.trans row_count_12074.symm
def row_12075 : RowCertificate 12075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12075 1) := by change ¬ Nat.Prime 583246651; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44865127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12075 2) := by change ¬ Nat.Prime 583222501; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 3714793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12075 3) := by change ¬ Nat.Prime 583198351; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8704453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12075 : row 12075 = 0 := (row_12075.checked).trans rfl
theorem model_count_12075 : modelRow 12075 = 0 := by decide
theorem modelrow_agrees_12075 : modelRow 12075 = row 12075 := model_count_12075.trans row_count_12075.symm
def row_12076 : RowCertificate 12076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12076 1) := by change ¬ Nat.Prime 583343257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83334751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12076 2) := by change ¬ Nat.Prime 583319105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116663821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12076 3) := by change ¬ Nat.Prime 583294953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194431651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12076 : row 12076 = 0 := (row_12076.checked).trans rfl
theorem model_count_12076 : modelRow 12076 = 0 := by decide
theorem modelrow_agrees_12076 : modelRow 12076 = row 12076 := model_count_12076.trans row_count_12076.symm
def row_12077 : RowCertificate 12077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12077 1) := by change ¬ Nat.Prime 583439871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194479957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12077 2) := by change ¬ Nat.Prime 583415717; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 1469561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12077 3) := by change Nat.Prime 583391563; exact ProjectEulerSolutions.P58PrimeNodes.p_583391563.prime
    exact decide_eq_true prime
theorem row_count_12077 : row 12077 = 1 := (row_12077.checked).trans rfl
theorem model_count_12077 : modelRow 12077 = 1 := by decide
theorem modelrow_agrees_12077 : modelRow 12077 = row 12077 := model_count_12077.trans row_count_12077.symm
def row_12078 : RowCertificate 12078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12078 1) := by change ¬ Nat.Prime 583536493; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30712447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12078 2) := by change Nat.Prime 583512337; exact ProjectEulerSolutions.P58PrimeNodes.p_583512337.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12078 3) := by change Nat.Prime 583488181; exact ProjectEulerSolutions.P58PrimeNodes.p_583488181.prime
    exact decide_eq_true prime
theorem row_count_12078 : row 12078 = 2 := (row_12078.checked).trans rfl
theorem model_count_12078 : modelRow 12078 = 2 := by decide
theorem modelrow_agrees_12078 : modelRow 12078 = row 12078 := model_count_12078.trans row_count_12078.symm
def row_12079 : RowCertificate 12079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12079 1) := by change ¬ Nat.Prime 583633123; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 5666341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12079 2) := by change ¬ Nat.Prime 583608965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116721793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12079 3) := by change ¬ Nat.Prime 583584807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194528269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12079 : row 12079 = 0 := (row_12079.checked).trans rfl
theorem model_count_12079 : modelRow 12079 = 0 := by decide
theorem modelrow_agrees_12079 : modelRow 12079 = row 12079 := model_count_12079.trans row_count_12079.symm
def row_12080 : RowCertificate 12080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12080 1) := by change ¬ Nat.Prime 583729761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194576587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12080 2) := by change ¬ Nat.Prime 583705601; exact PrattCertificates.composite_of_product (factor := 13037) (cofactor := 44773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12080 3) := by change ¬ Nat.Prime 583681441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83383063) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12080 : row 12080 = 0 := (row_12080.checked).trans rfl
theorem model_count_12080 : modelRow 12080 = 0 := by decide
theorem modelrow_agrees_12080 : modelRow 12080 = row 12080 := model_count_12080.trans row_count_12080.symm
def row_12081 : RowCertificate 12081 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12081 1) := by change Nat.Prime 583826407; exact ProjectEulerSolutions.P58PrimeNodes.p_583826407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12081 2) := by change ¬ Nat.Prime 583802245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116760449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12081 3) := by change ¬ Nat.Prime 583778083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83396869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12081 : row 12081 = 1 := (row_12081.checked).trans rfl
theorem model_count_12081 : modelRow 12081 = 1 := by decide
theorem modelrow_agrees_12081 : modelRow 12081 = row 12081 := model_count_12081.trans row_count_12081.symm
def row_12082 : RowCertificate 12082 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12082 1) := by change Nat.Prime 583923061; exact ProjectEulerSolutions.P58PrimeNodes.p_583923061.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12082 2) := by change ¬ Nat.Prime 583898897; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 2422817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12082 3) := by change ¬ Nat.Prime 583874733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194624911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12082 : row 12082 = 1 := (row_12082.checked).trans rfl
theorem model_count_12082 : modelRow 12082 = 1 := by decide
theorem modelrow_agrees_12082 : modelRow 12082 = row 12082 := model_count_12082.trans row_count_12082.symm
def row_12083 : RowCertificate 12083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12083 1) := by change ¬ Nat.Prime 584019723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194673241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12083 2) := by change ¬ Nat.Prime 583995557; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 623261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12083 3) := by change ¬ Nat.Prime 583971391; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 3719563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12083 : row 12083 = 0 := (row_12083.checked).trans rfl
theorem model_count_12083 : modelRow 12083 = 0 := by decide
theorem modelrow_agrees_12083 : modelRow 12083 = row 12083 := model_count_12083.trans row_count_12083.symm
def row_12084 : RowCertificate 12084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12084 1) := by change ¬ Nat.Prime 584116393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83445199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12084 2) := by change ¬ Nat.Prime 584092225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116818445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12084 3) := by change Nat.Prime 584068057; exact ProjectEulerSolutions.P58PrimeNodes.p_584068057.prime
    exact decide_eq_true prime
theorem row_count_12084 : row 12084 = 1 := (row_12084.checked).trans rfl
theorem model_count_12084 : modelRow 12084 = 1 := by decide
theorem modelrow_agrees_12084 : modelRow 12084 = row 12084 := model_count_12084.trans row_count_12084.symm
def row_12085 : RowCertificate 12085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12085 1) := by change ¬ Nat.Prime 584213071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44939467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12085 2) := by change ¬ Nat.Prime 584188901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34364053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12085 3) := by change ¬ Nat.Prime 584164731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194721577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12085 : row 12085 = 0 := (row_12085.checked).trans rfl
theorem model_count_12085 : modelRow 12085 = 0 := by decide
theorem modelrow_agrees_12085 : modelRow 12085 = row 12085 := model_count_12085.trans row_count_12085.symm
def row_12086 : RowCertificate 12086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12086 1) := by change ¬ Nat.Prime 584309757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194769919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12086 2) := by change ¬ Nat.Prime 584285585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116857117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12086 3) := by change ¬ Nat.Prime 584261413; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15790849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12086 : row 12086 = 0 := (row_12086.checked).trans rfl
theorem model_count_12086 : modelRow 12086 = 0 := by decide
theorem modelrow_agrees_12086 : modelRow 12086 = row 12086 := model_count_12086.trans row_count_12086.symm
def row_12087 : RowCertificate 12087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12087 1) := by change ¬ Nat.Prime 584406451; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18851821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12087 2) := by change ¬ Nat.Prime 584382277; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20151113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12087 3) := by change ¬ Nat.Prime 584358103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83479729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12087 : row 12087 = 0 := (row_12087.checked).trans rfl
theorem model_count_12087 : modelRow 12087 = 0 := by decide
theorem modelrow_agrees_12087 : modelRow 12087 = row 12087 := model_count_12087.trans row_count_12087.symm
def row_12088 : RowCertificate 12088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12088 1) := by change ¬ Nat.Prime 584503153; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44961781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12088 2) := by change Nat.Prime 584478977; exact ProjectEulerSolutions.P58PrimeNodes.p_584478977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12088 3) := by change ¬ Nat.Prime 584454801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194818267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12088 : row 12088 = 1 := (row_12088.checked).trans rfl
theorem model_count_12088 : modelRow 12088 = 1 := by decide
theorem modelrow_agrees_12088 : modelRow 12088 = row 12088 := model_count_12088.trans row_count_12088.symm
def row_12089 : RowCertificate 12089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12089 1) := by change ¬ Nat.Prime 584599863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194866621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12089 2) := by change ¬ Nat.Prime 584575685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116915137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12089 3) := by change ¬ Nat.Prime 584551507; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 366031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12089 : row 12089 = 0 := (row_12089.checked).trans rfl
theorem model_count_12089 : modelRow 12089 = 0 := by decide
theorem modelrow_agrees_12089 : modelRow 12089 = row 12089 := model_count_12089.trans row_count_12089.symm
def row_12090 : RowCertificate 12090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12090 1) := by change ¬ Nat.Prime 584696581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83528083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12090 2) := by change ¬ Nat.Prime 584672401; exact PrattCertificates.composite_of_product (factor := 16333) (cofactor := 35797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12090 3) := by change ¬ Nat.Prime 584648221; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30770959) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12090 : row 12090 = 0 := (row_12090.checked).trans rfl
theorem model_count_12090 : modelRow 12090 = 0 := by decide
theorem modelrow_agrees_12090 : modelRow 12090 = row 12090 := model_count_12090.trans row_count_12090.symm
def row_12091 : RowCertificate 12091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12091 1) := by change ¬ Nat.Prime 584793307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83541901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12091 2) := by change ¬ Nat.Prime 584769125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116953825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12091 3) := by change ¬ Nat.Prime 584744943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194914981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12091 : row 12091 = 0 := (row_12091.checked).trans rfl
theorem model_count_12091 : modelRow 12091 = 0 := by decide
theorem modelrow_agrees_12091 : modelRow 12091 = row 12091 := model_count_12091.trans row_count_12091.symm
def row_12092 : RowCertificate 12092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12092 1) := by change ¬ Nat.Prime 584890041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194963347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12092 2) := by change Nat.Prime 584865857; exact ProjectEulerSolutions.P58PrimeNodes.p_584865857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12092 3) := by change ¬ Nat.Prime 584841673; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44987821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12092 : row 12092 = 1 := (row_12092.checked).trans rfl
theorem model_count_12092 : modelRow 12092 = 1 := by decide
theorem modelrow_agrees_12092 : modelRow 12092 = row 12092 := model_count_12092.trans row_count_12092.symm
def row_12093 : RowCertificate 12093 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12093 1) := by change Nat.Prime 584986783; exact ProjectEulerSolutions.P58PrimeNodes.p_584986783.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12093 2) := by change ¬ Nat.Prime 584962597; exact PrattCertificates.composite_of_product (factor := 18413) (cofactor := 31769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12093 3) := by change ¬ Nat.Prime 584938411; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18868981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12093 : row 12093 = 1 := (row_12093.checked).trans rfl
theorem model_count_12093 : modelRow 12093 = 1 := by decide
theorem modelrow_agrees_12093 : modelRow 12093 = row 12093 := model_count_12093.trans row_count_12093.symm
def row_12094 : RowCertificate 12094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12094 1) := by change ¬ Nat.Prime 585083533; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 6031789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12094 2) := by change ¬ Nat.Prime 585059345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117011869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12094 3) := by change ¬ Nat.Prime 585035157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195011719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12094 : row 12094 = 0 := (row_12094.checked).trans rfl
theorem model_count_12094 : modelRow 12094 = 0 := by decide
theorem modelrow_agrees_12094 : modelRow 12094 = row 12094 := model_count_12094.trans row_count_12094.symm
def row_12095 : RowCertificate 12095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12095 1) := by change ¬ Nat.Prime 585180291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195060097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12095 2) := by change ¬ Nat.Prime 585156101; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 8015837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12095 3) := by change ¬ Nat.Prime 585131911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83590273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12095 : row 12095 = 0 := (row_12095.checked).trans rfl
theorem model_count_12095 : modelRow 12095 = 0 := by decide
theorem modelrow_agrees_12095 : modelRow 12095 = row 12095 := model_count_12095.trans row_count_12095.symm
def row_12096 : RowCertificate 12096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12096 1) := by change ¬ Nat.Prime 585277057; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3876007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12096 2) := by change ¬ Nat.Prime 585252865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117050573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12096 3) := by change Nat.Prime 585228673; exact ProjectEulerSolutions.P58PrimeNodes.p_585228673.prime
    exact decide_eq_true prime
theorem row_count_12096 : row 12096 = 1 := (row_12096.checked).trans rfl
theorem model_count_12096 : modelRow 12096 = 1 := by decide
theorem modelrow_agrees_12096 : modelRow 12096 = row 12096 := model_count_12096.trans row_count_12096.symm
def row_12097 : RowCertificate 12097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12097 1) := by change ¬ Nat.Prime 585373831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83624833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12097 2) := by change ¬ Nat.Prime 585349637; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 473201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12097 3) := by change ¬ Nat.Prime 585325443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195108481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12097 : row 12097 = 0 := (row_12097.checked).trans rfl
theorem model_count_12097 : modelRow 12097 = 0 := by decide
theorem modelrow_agrees_12097 : modelRow 12097 = row 12097 := model_count_12097.trans row_count_12097.symm
def row_12098 : RowCertificate 12098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12098 1) := by change ¬ Nat.Prime 585470613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195156871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12098 2) := by change Nat.Prime 585446417; exact ProjectEulerSolutions.P58PrimeNodes.p_585446417.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12098 3) := by change Nat.Prime 585422221; exact ProjectEulerSolutions.P58PrimeNodes.p_585422221.prime
    exact decide_eq_true prime
theorem row_count_12098 : row 12098 = 2 := (row_12098.checked).trans rfl
theorem model_count_12098 : modelRow 12098 = 2 := by decide
theorem modelrow_agrees_12098 : modelRow 12098 = row 12098 := model_count_12098.trans row_count_12098.symm
def row_12099 : RowCertificate 12099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12099 1) := by change ¬ Nat.Prime 585567403; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30819337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12099 2) := by change ¬ Nat.Prime 585543205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117108641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12099 3) := by change ¬ Nat.Prime 585519007; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7411633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12099 : row 12099 = 0 := (row_12099.checked).trans rfl
theorem model_count_12099 : modelRow 12099 = 0 := by decide
theorem modelrow_agrees_12099 : modelRow 12099 = row 12099 := model_count_12099.trans row_count_12099.symm
def row_12100 : RowCertificate 12100 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12100 1) := by change Nat.Prime 585664201; exact ProjectEulerSolutions.P58PrimeNodes.p_585664201.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12100 2) := by change ¬ Nat.Prime 585640001; exact PrattCertificates.composite_of_product (factor := 23981) (cofactor := 24421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12100 3) := by change ¬ Nat.Prime 585615801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195205267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12100 : row 12100 = 1 := (row_12100.checked).trans rfl
theorem model_count_12100 : modelRow 12100 = 1 := by decide
theorem modelrow_agrees_12100 : modelRow 12100 = row 12100 := model_count_12100.trans row_count_12100.symm
def row_12101 : RowCertificate 12101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12101 1) := by change ¬ Nat.Prime 585761007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195253669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12101 2) := by change ¬ Nat.Prime 585736805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117147361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12101 3) := by change ¬ Nat.Prime 585712603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83673229) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12101 : row 12101 = 0 := (row_12101.checked).trans rfl
theorem model_count_12101 : modelRow 12101 = 0 := by decide
theorem modelrow_agrees_12101 : modelRow 12101 = row 12101 := model_count_12101.trans row_count_12101.symm
def row_12102 : RowCertificate 12102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12102 1) := by change ¬ Nat.Prime 585857821; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 1545799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12102 2) := by change ¬ Nat.Prime 585833617; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34460801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12102 3) := by change ¬ Nat.Prime 585809413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83687059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12102 : row 12102 = 0 := (row_12102.checked).trans rfl
theorem model_count_12102 : modelRow 12102 = 0 := by decide
theorem modelrow_agrees_12102 : modelRow 12102 = row 12102 := model_count_12102.trans row_count_12102.symm
def row_12103 : RowCertificate 12103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12103 1) := by change ¬ Nat.Prime 585954643; exact PrattCertificates.composite_of_product (factor := 9109) (cofactor := 64327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12103 2) := by change ¬ Nat.Prime 585930437; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9605417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12103 3) := by change ¬ Nat.Prime 585906231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195302077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12103 : row 12103 = 0 := (row_12103.checked).trans rfl
theorem model_count_12103 : modelRow 12103 = 0 := by decide
theorem modelrow_agrees_12103 : modelRow 12103 = row 12103 := model_count_12103.trans row_count_12103.symm
def row_12104 : RowCertificate 12104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12104 1) := by change ¬ Nat.Prime 586051473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195350491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12104 2) := by change ¬ Nat.Prime 586027265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117205453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12104 3) := by change Nat.Prime 586003057; exact ProjectEulerSolutions.P58PrimeNodes.p_586003057.prime
    exact decide_eq_true prime
theorem row_count_12104 : row 12104 = 1 := (row_12104.checked).trans rfl
theorem model_count_12104 : modelRow 12104 = 1 := by decide
theorem modelrow_agrees_12104 : modelRow 12104 = row 12104 := model_count_12104.trans row_count_12104.symm
def row_12105 : RowCertificate 12105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12105 1) := by change ¬ Nat.Prime 586148311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83735473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12105 2) := by change ¬ Nat.Prime 586124101; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 1015813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12105 3) := by change ¬ Nat.Prime 586099891; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45084607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12105 : row 12105 = 0 := (row_12105.checked).trans rfl
theorem model_count_12105 : modelRow 12105 = 0 := by decide
theorem modelrow_agrees_12105 : modelRow 12105 = row 12105 := model_count_12105.trans row_count_12105.symm
def row_12106 : RowCertificate 12106 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12106 1) := by change Nat.Prime 586245157; exact ProjectEulerSolutions.P58PrimeNodes.p_586245157.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12106 2) := by change ¬ Nat.Prime 586220945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117244189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12106 3) := by change ¬ Nat.Prime 586196733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195398911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12106 : row 12106 = 1 := (row_12106.checked).trans rfl
theorem model_count_12106 : modelRow 12106 = 1 := by decide
theorem modelrow_agrees_12106 : modelRow 12106 = row 12106 := model_count_12106.trans row_count_12106.symm
def row_12107 : RowCertificate 12107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12107 1) := by change ¬ Nat.Prime 586342011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195447337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12107 2) := by change ¬ Nat.Prime 586317797; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45101369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12107 3) := by change ¬ Nat.Prime 586293583; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30857557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12107 : row 12107 = 0 := (row_12107.checked).trans rfl
theorem model_count_12107 : modelRow 12107 = 0 := by decide
theorem modelrow_agrees_12107 : modelRow 12107 = row 12107 := model_count_12107.trans row_count_12107.symm
def row_12108 : RowCertificate 12108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12108 1) := by change ¬ Nat.Prime 586438873; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18917383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12108 2) := by change ¬ Nat.Prime 586414657; exact PrattCertificates.composite_of_product (factor := 5441) (cofactor := 107777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12108 3) := by change ¬ Nat.Prime 586390441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83770063) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12108 : row 12108 = 0 := (row_12108.checked).trans rfl
theorem model_count_12108 : modelRow 12108 = 0 := by decide
theorem modelrow_agrees_12108 : modelRow 12108 = row 12108 := model_count_12108.trans row_count_12108.symm
def row_12109 : RowCertificate 12109 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12109 1) := by change Nat.Prime 586535743; exact ProjectEulerSolutions.P58PrimeNodes.p_586535743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12109 2) := by change ¬ Nat.Prime 586511525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117302305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12109 3) := by change ¬ Nat.Prime 586487307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195495769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12109 : row 12109 = 1 := (row_12109.checked).trans rfl
theorem model_count_12109 : modelRow 12109 = 1 := by decide
theorem modelrow_agrees_12109 : modelRow 12109 = row 12109 := model_count_12109.trans row_count_12109.symm
def row_12110 : RowCertificate 12110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12110 1) := by change ¬ Nat.Prime 586632621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195544207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12110 2) := by change ¬ Nat.Prime 586608401; exact PrattCertificates.composite_of_product (factor := 2617) (cofactor := 224153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12110 3) := by change Nat.Prime 586584181; exact ProjectEulerSolutions.P58PrimeNodes.p_586584181.prime
    exact decide_eq_true prime
theorem row_count_12110 : row 12110 = 1 := (row_12110.checked).trans rfl
theorem model_count_12110 : modelRow 12110 = 1 := by decide
theorem modelrow_agrees_12110 : modelRow 12110 = row 12110 := model_count_12110.trans row_count_12110.symm
def row_12111 : RowCertificate 12111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12111 1) := by change ¬ Nat.Prime 586729507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83818501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12111 2) := by change ¬ Nat.Prime 586705285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117341057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12111 3) := by change Nat.Prime 586681063; exact ProjectEulerSolutions.P58PrimeNodes.p_586681063.prime
    exact decide_eq_true prime
theorem row_count_12111 : row 12111 = 1 := (row_12111.checked).trans rfl
theorem model_count_12111 : modelRow 12111 = 1 := by decide
theorem modelrow_agrees_12111 : modelRow 12111 = row 12111 := model_count_12111.trans row_count_12111.symm
def row_12112 : RowCertificate 12112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12112 1) := by change ¬ Nat.Prime 586826401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83832343) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12112 2) := by change ¬ Nat.Prime 586802177; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45138629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12112 3) := by change ¬ Nat.Prime 586777953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195592651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12112 : row 12112 = 0 := (row_12112.checked).trans rfl
theorem model_count_12112 : modelRow 12112 = 0 := by decide
theorem modelrow_agrees_12112 : modelRow 12112 = row 12112 := model_count_12112.trans row_count_12112.symm
def row_12113 : RowCertificate 12113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12113 1) := by change ¬ Nat.Prime 586923303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195641101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12113 2) := by change ¬ Nat.Prime 586899077; exact PrattCertificates.composite_of_product (factor := 1889) (cofactor := 310693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12113 3) := by change Nat.Prime 586874851; exact ProjectEulerSolutions.P58PrimeNodes.p_586874851.prime
    exact decide_eq_true prime
theorem row_count_12113 : row 12113 = 1 := (row_12113.checked).trans rfl
theorem model_count_12113 : modelRow 12113 = 1 := by decide
theorem modelrow_agrees_12113 : modelRow 12113 = row 12113 := model_count_12113.trans row_count_12113.symm
def row_12114 : RowCertificate 12114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12114 1) := by change ¬ Nat.Prime 587020213; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45155401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12114 2) := by change ¬ Nat.Prime 586995985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117399197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12114 3) := by change ¬ Nat.Prime 586971757; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 8040709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12114 : row 12114 = 0 := (row_12114.checked).trans rfl
theorem model_count_12114 : modelRow 12114 = 0 := by decide
theorem modelrow_agrees_12114 : modelRow 12114 = row 12114 := model_count_12114.trans row_count_12114.symm
def row_12115 : RowCertificate 12115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12115 1) := by change ¬ Nat.Prime 587117131; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9624871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12115 2) := by change ¬ Nat.Prime 587092901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 5812801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12115 3) := by change ¬ Nat.Prime 587068671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195689557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12115 : row 12115 = 0 := (row_12115.checked).trans rfl
theorem model_count_12115 : modelRow 12115 = 0 := by decide
theorem modelrow_agrees_12115 : modelRow 12115 = row 12115 := model_count_12115.trans row_count_12115.symm
def row_12116 : RowCertificate 12116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12116 1) := by change ¬ Nat.Prime 587214057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195738019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12116 2) := by change ¬ Nat.Prime 587189825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117437965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12116 3) := by change ¬ Nat.Prime 587165593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83880799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12116 : row 12116 = 0 := (row_12116.checked).trans rfl
theorem model_count_12116 : modelRow 12116 = 0 := by decide
theorem modelrow_agrees_12116 : modelRow 12116 = row 12116 := model_count_12116.trans row_count_12116.symm
def row_12117 : RowCertificate 12117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12117 1) := by change ¬ Nat.Prime 587310991; exact PrattCertificates.composite_of_product (factor := 1873) (cofactor := 313567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12117 2) := by change Nat.Prime 587286757; exact ProjectEulerSolutions.P58PrimeNodes.p_587286757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12117 3) := by change ¬ Nat.Prime 587262523; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 3042811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12117 : row 12117 = 1 := (row_12117.checked).trans rfl
theorem model_count_12117 : modelRow 12117 = 1 := by decide
theorem modelrow_agrees_12117 : modelRow 12117 = row 12117 := model_count_12117.trans row_count_12117.symm
def row_12118 : RowCertificate 12118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12118 1) := by change ¬ Nat.Prime 587407933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83915419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12118 2) := by change ¬ Nat.Prime 587383697; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 990529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12118 3) := by change ¬ Nat.Prime 587359461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195786487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12118 : row 12118 = 0 := (row_12118.checked).trans rfl
theorem model_count_12118 : modelRow 12118 = 0 := by decide
theorem modelrow_agrees_12118 : modelRow 12118 = row 12118 := model_count_12118.trans row_count_12118.symm
def row_12119 : RowCertificate 12119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12119 1) := by change ¬ Nat.Prime 587504883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195834961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12119 2) := by change ¬ Nat.Prime 587480645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117496129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12119 3) := by change ¬ Nat.Prime 587456407; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 4625641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12119 : row 12119 = 0 := (row_12119.checked).trans rfl
theorem model_count_12119 : modelRow 12119 = 0 := by decide
theorem modelrow_agrees_12119 : modelRow 12119 = row 12119 := model_count_12119.trans row_count_12119.symm
def row_12120 : RowCertificate 12120 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12120 1) := by change Nat.Prime 587601841; exact ProjectEulerSolutions.P58PrimeNodes.p_587601841.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12120 2) := by change ¬ Nat.Prime 587577601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45198277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12120 3) := by change Nat.Prime 587553361; exact ProjectEulerSolutions.P58PrimeNodes.p_587553361.prime
    exact decide_eq_true prime
theorem row_count_12120 : row 12120 = 2 := (row_12120.checked).trans rfl
theorem model_count_12120 : modelRow 12120 = 2 := by decide
theorem modelrow_agrees_12120 : modelRow 12120 = row 12120 := model_count_12120.trans row_count_12120.symm
def row_12121 : RowCertificate 12121 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12121 1) := by change Nat.Prime 587698807; exact ProjectEulerSolutions.P58PrimeNodes.p_587698807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12121 2) := by change ¬ Nat.Prime 587674565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117534913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12121 3) := by change ¬ Nat.Prime 587650323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195883441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12121 : row 12121 = 1 := (row_12121.checked).trans rfl
theorem model_count_12121 : modelRow 12121 = 1 := by decide
theorem modelrow_agrees_12121 : modelRow 12121 = row 12121 := model_count_12121.trans row_count_12121.symm
def row_12122 : RowCertificate 12122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12122 1) := by change ¬ Nat.Prime 587795781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195931927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12122 2) := by change ¬ Nat.Prime 587771537; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 11090029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12122 3) := by change ¬ Nat.Prime 587747293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83963899) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12122 : row 12122 = 0 := (row_12122.checked).trans rfl
theorem model_count_12122 : modelRow 12122 = 0 := by decide
theorem modelrow_agrees_12122 : modelRow 12122 = row 12122 := model_count_12122.trans row_count_12122.symm
def row_12123 : RowCertificate 12123 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12123 1) := by change Nat.Prime 587892763; exact ProjectEulerSolutions.P58PrimeNodes.p_587892763.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12123 2) := by change ¬ Nat.Prime 587868517; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34580501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12123 3) := by change ¬ Nat.Prime 587844271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83977753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12123 : row 12123 = 1 := (row_12123.checked).trans rfl
theorem model_count_12123 : modelRow 12123 = 1 := by decide
theorem modelrow_agrees_12123 : modelRow 12123 = row 12123 := model_count_12123.trans row_count_12123.symm
def row_12124 : RowCertificate 12124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12124 1) := by change ¬ Nat.Prime 587989753; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45229981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12124 2) := by change ¬ Nat.Prime 587965505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117593101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12124 3) := by change ¬ Nat.Prime 587941257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195980419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12124 : row 12124 = 0 := (row_12124.checked).trans rfl
theorem model_count_12124 : modelRow 12124 = 0 := by decide
theorem modelrow_agrees_12124 : modelRow 12124 = row 12124 := model_count_12124.trans row_count_12124.symm
def row_12125 : RowCertificate 12125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12125 1) := by change ¬ Nat.Prime 588086751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196028917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12125 2) := by change ¬ Nat.Prime 588062501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45235577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12125 3) := by change Nat.Prime 588038251; exact ProjectEulerSolutions.P58PrimeNodes.p_588038251.prime
    exact decide_eq_true prime
theorem row_count_12125 : row 12125 = 1 := (row_12125.checked).trans rfl
theorem model_count_12125 : modelRow 12125 = 1 := by decide
theorem modelrow_agrees_12125 : modelRow 12125 = row 12125 := model_count_12125.trans row_count_12125.symm
def row_12126 : RowCertificate 12126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12126 1) := by change ¬ Nat.Prime 588183757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84026251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12126 2) := by change ¬ Nat.Prime 588159505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117631901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12126 3) := by change ¬ Nat.Prime 588135253; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30954487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12126 : row 12126 = 0 := (row_12126.checked).trans rfl
theorem model_count_12126 : modelRow 12126 = 0 := by decide
theorem modelrow_agrees_12126 : modelRow 12126 = row 12126 := model_count_12126.trans row_count_12126.symm
def row_12127 : RowCertificate 12127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12127 1) := by change ¬ Nat.Prime 588280771; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45252367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12127 2) := by change ¬ Nat.Prime 588256517; exact PrattCertificates.composite_of_product (factor := 1861) (cofactor := 316097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12127 3) := by change ¬ Nat.Prime 588232263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196077421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12127 : row 12127 = 0 := (row_12127.checked).trans rfl
theorem model_count_12127 : modelRow 12127 = 0 := by decide
theorem modelrow_agrees_12127 : modelRow 12127 = row 12127 := model_count_12127.trans row_count_12127.symm
def row_12128 : RowCertificate 12128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12128 1) := by change ¬ Nat.Prime 588377793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196125931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12128 2) := by change ¬ Nat.Prime 588353537; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20288053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12128 3) := by change ¬ Nat.Prime 588329281; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30964699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12128 : row 12128 = 0 := (row_12128.checked).trans rfl
theorem model_count_12128 : modelRow 12128 = 0 := by decide
theorem modelrow_agrees_12128 : modelRow 12128 = row 12128 := model_count_12128.trans row_count_12128.symm
def row_12129 : RowCertificate 12129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12129 1) := by change ¬ Nat.Prime 588474823; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13685461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12129 2) := by change ¬ Nat.Prime 588450565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117690113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12129 3) := by change ¬ Nat.Prime 588426307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84060901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12129 : row 12129 = 0 := (row_12129.checked).trans rfl
theorem model_count_12129 : modelRow 12129 = 0 := by decide
theorem modelrow_agrees_12129 : modelRow 12129 = row 12129 := model_count_12129.trans row_count_12129.symm
def row_12130 : RowCertificate 12130 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12130 1) := by change Nat.Prime 588571861; exact ProjectEulerSolutions.P58PrimeNodes.p_588571861.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12130 2) := by change Nat.Prime 588547601; exact ProjectEulerSolutions.P58PrimeNodes.p_588547601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12130 3) := by change ¬ Nat.Prime 588523341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196174447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12130 : row 12130 = 2 := (row_12130.checked).trans rfl
theorem model_count_12130 : modelRow 12130 = 2 := by decide
theorem modelrow_agrees_12130 : modelRow 12130 = row 12130 := model_count_12130.trans row_count_12130.symm
def row_12131 : RowCertificate 12131 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12131 1) := by change ¬ Nat.Prime 588668907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196222969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12131 2) := by change ¬ Nat.Prime 588644645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117728929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12131 3) := by change ¬ Nat.Prime 588620383; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45278491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12131 : row 12131 = 0 := (row_12131.checked).trans rfl
theorem model_count_12131 : modelRow 12131 = 0 := by decide
theorem modelrow_agrees_12131 : modelRow 12131 = row 12131 := model_count_12131.trans row_count_12131.symm
def row_12132 : RowCertificate 12132 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12132 1) := by change ¬ Nat.Prime 588765961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84109423) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12132 2) := by change ¬ Nat.Prime 588741697; exact PrattCertificates.composite_of_product (factor := 1877) (cofactor := 313661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12132 3) := by change Nat.Prime 588717433; exact ProjectEulerSolutions.P58PrimeNodes.p_588717433.prime
    exact decide_eq_true prime
theorem row_count_12132 : row 12132 = 1 := (row_12132.checked).trans rfl
theorem model_count_12132 : modelRow 12132 = 1 := by decide
theorem modelrow_agrees_12132 : modelRow 12132 = row 12132 := model_count_12132.trans row_count_12132.symm
def row_12133 : RowCertificate 12133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12133 1) := by change ¬ Nat.Prime 588863023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84123289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12133 2) := by change ¬ Nat.Prime 588838757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45295289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12133 3) := by change ¬ Nat.Prime 588814491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196271497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12133 : row 12133 = 0 := (row_12133.checked).trans rfl
theorem model_count_12133 : modelRow 12133 = 0 := by decide
theorem modelrow_agrees_12133 : modelRow 12133 = row 12133 := model_count_12133.trans row_count_12133.symm
def row_12134 : RowCertificate 12134 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12134 1) := by change ¬ Nat.Prime 588960093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196320031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12134 2) := by change ¬ Nat.Prime 588935825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117787165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12134 3) := by change ¬ Nat.Prime 588911557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45300889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12134 : row 12134 = 0 := (row_12134.checked).trans rfl
theorem model_count_12134 : modelRow 12134 = 0 := by decide
theorem modelrow_agrees_12134 : modelRow 12134 = row 12134 := model_count_12134.trans row_count_12134.symm
def row_12135 : RowCertificate 12135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12135 1) := by change ¬ Nat.Prime 589057171; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31003009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12135 2) := by change Nat.Prime 589032901; exact ProjectEulerSolutions.P58PrimeNodes.p_589032901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12135 3) := by change Nat.Prime 589008631; exact ProjectEulerSolutions.P58PrimeNodes.p_589008631.prime
    exact decide_eq_true prime
theorem row_count_12135 : row 12135 = 2 := (row_12135.checked).trans rfl
theorem model_count_12135 : modelRow 12135 = 2 := by decide
theorem modelrow_agrees_12135 : modelRow 12135 = row 12135 := model_count_12135.trans row_count_12135.symm
def row_12136 : RowCertificate 12136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12136 1) := by change ¬ Nat.Prime 589154257; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 810391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12136 2) := by change ¬ Nat.Prime 589129985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117825997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12136 3) := by change ¬ Nat.Prime 589105713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196368571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12136 : row 12136 = 0 := (row_12136.checked).trans rfl
theorem model_count_12136 : modelRow 12136 = 0 := by decide
theorem modelrow_agrees_12136 : modelRow 12136 = row 12136 := model_count_12136.trans row_count_12136.symm
def row_12137 : RowCertificate 12137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12137 1) := by change ¬ Nat.Prime 589251351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196417117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12137 2) := by change ¬ Nat.Prime 589227077; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 5405753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12137 3) := by change ¬ Nat.Prime 589202803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84171829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12137 : row 12137 = 0 := (row_12137.checked).trans rfl
theorem model_count_12137 : modelRow 12137 = 0 := by decide
theorem modelrow_agrees_12137 : modelRow 12137 = row 12137 := model_count_12137.trans row_count_12137.symm
def row_12138 : RowCertificate 12138 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12138 1) := by change ¬ Nat.Prime 589348453; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7460107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12138 2) := by change ¬ Nat.Prime 589324177; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45332629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12138 3) := by change Nat.Prime 589299901; exact ProjectEulerSolutions.P58PrimeNodes.p_589299901.prime
    exact decide_eq_true prime
theorem row_count_12138 : row 12138 = 1 := (row_12138.checked).trans rfl
theorem model_count_12138 : modelRow 12138 = 1 := by decide
theorem modelrow_agrees_12138 : modelRow 12138 = row 12138 := model_count_12138.trans row_count_12138.symm
def row_12139 : RowCertificate 12139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12139 1) := by change ¬ Nat.Prime 589445563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84206509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12139 2) := by change ¬ Nat.Prime 589421285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117884257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12139 3) := by change ¬ Nat.Prime 589397007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196465669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12139 : row 12139 = 0 := (row_12139.checked).trans rfl
theorem model_count_12139 : modelRow 12139 = 0 := by decide
theorem modelrow_agrees_12139 : modelRow 12139 = row 12139 := model_count_12139.trans row_count_12139.symm
def row_12140 : RowCertificate 12140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12140 1) := by change ¬ Nat.Prime 589542681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196514227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12140 2) := by change ¬ Nat.Prime 589518401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34677553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12140 3) := by change ¬ Nat.Prime 589494121; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 2793811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12140 : row 12140 = 0 := (row_12140.checked).trans rfl
theorem model_count_12140 : modelRow 12140 = 0 := by decide
theorem modelrow_agrees_12140 : modelRow 12140 = row 12140 := model_count_12140.trans row_count_12140.symm
def row_12141 : RowCertificate 12141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12141 1) := by change ¬ Nat.Prime 589639807; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15936211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12141 2) := by change ¬ Nat.Prime 589615525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117923105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12141 3) := by change Nat.Prime 589591243; exact ProjectEulerSolutions.P58PrimeNodes.p_589591243.prime
    exact decide_eq_true prime
theorem row_count_12141 : row 12141 = 1 := (row_12141.checked).trans rfl
theorem model_count_12141 : modelRow 12141 = 1 := by decide
theorem modelrow_agrees_12141 : modelRow 12141 = row 12141 := model_count_12141.trans row_count_12141.symm
def row_12142 : RowCertificate 12142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12142 1) := by change ¬ Nat.Prime 589736941; exact PrattCertificates.composite_of_product (factor := 2437) (cofactor := 241993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12142 2) := by change Nat.Prime 589712657; exact ProjectEulerSolutions.P58PrimeNodes.p_589712657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12142 3) := by change ¬ Nat.Prime 589688373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196562791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12142 : row 12142 = 1 := (row_12142.checked).trans rfl
theorem model_count_12142 : modelRow 12142 = 1 := by decide
theorem modelrow_agrees_12142 : modelRow 12142 = row 12142 := model_count_12142.trans row_count_12142.symm
def row_12143 : RowCertificate 12143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12143 1) := by change ¬ Nat.Prime 589834083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196611361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12143 2) := by change ¬ Nat.Prime 589809797; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 4305181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12143 3) := by change ¬ Nat.Prime 589785511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84255073) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12143 : row 12143 = 0 := (row_12143.checked).trans rfl
theorem model_count_12143 : modelRow 12143 = 0 := by decide
theorem modelrow_agrees_12143 : modelRow 12143 = row 12143 := model_count_12143.trans row_count_12143.symm
def row_12144 : RowCertificate 12144 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12144 1) := by change ¬ Nat.Prime 589931233; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13719331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12144 2) := by change ¬ Nat.Prime 589906945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117981389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12144 3) := by change ¬ Nat.Prime 589882657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84268951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12144 : row 12144 = 0 := (row_12144.checked).trans rfl
theorem model_count_12144 : modelRow 12144 = 0 := by decide
theorem modelrow_agrees_12144 : modelRow 12144 = row 12144 := model_count_12144.trans row_count_12144.symm
def row_12145 : RowCertificate 12145 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12145 1) := by change Nat.Prime 590028391; exact ProjectEulerSolutions.P58PrimeNodes.p_590028391.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12145 2) := by change ¬ Nat.Prime 590004101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20344969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12145 3) := by change ¬ Nat.Prime 589979811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196659937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12145 : row 12145 = 1 := (row_12145.checked).trans rfl
theorem model_count_12145 : modelRow 12145 = 1 := by decide
theorem modelrow_agrees_12145 : modelRow 12145 = row 12145 := model_count_12145.trans row_count_12145.symm
def row_12146 : RowCertificate 12146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12146 1) := by change ¬ Nat.Prime 590125557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196708519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12146 2) := by change ¬ Nat.Prime 590101265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118020253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12146 3) := by change ¬ Nat.Prime 590076973; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9673393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12146 : row 12146 = 0 := (row_12146.checked).trans rfl
theorem model_count_12146 : modelRow 12146 = 0 := by decide
theorem modelrow_agrees_12146 : modelRow 12146 = row 12146 := model_count_12146.trans row_count_12146.symm
def row_12147 : RowCertificate 12147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12147 1) := by change ¬ Nat.Prime 590222731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84317533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12147 2) := by change ¬ Nat.Prime 590198437; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1691113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12147 3) := by change ¬ Nat.Prime 590174143; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45398011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12147 : row 12147 = 0 := (row_12147.checked).trans rfl
theorem model_count_12147 : modelRow 12147 = 0 := by decide
theorem modelrow_agrees_12147 : modelRow 12147 = row 12147 := model_count_12147.trans row_count_12147.symm
def row_12148 : RowCertificate 12148 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12148 1) := by change Nat.Prime 590319913; exact ProjectEulerSolutions.P58PrimeNodes.p_590319913.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12148 2) := by change ¬ Nat.Prime 590295617; exact PrattCertificates.composite_of_product (factor := 2897) (cofactor := 203761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12148 3) := by change ¬ Nat.Prime 590271321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196757107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12148 : row 12148 = 1 := (row_12148.checked).trans rfl
theorem model_count_12148 : modelRow 12148 = 1 := by decide
theorem modelrow_agrees_12148 : modelRow 12148 = row 12148 := model_count_12148.trans row_count_12148.symm
def row_12149 : RowCertificate 12149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12149 1) := by change ¬ Nat.Prime 590417103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196805701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12149 2) := by change ¬ Nat.Prime 590392805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118078561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12149 3) := by change Nat.Prime 590368507; exact ProjectEulerSolutions.P58PrimeNodes.p_590368507.prime
    exact decide_eq_true prime
theorem row_count_12149 : row 12149 = 1 := (row_12149.checked).trans rfl
theorem model_count_12149 : modelRow 12149 = 1 := by decide
theorem modelrow_agrees_12149 : modelRow 12149 = row 12149 := model_count_12149.trans row_count_12149.symm
def row_12150 : RowCertificate 12150 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12150 1) := by change ¬ Nat.Prime 590514301; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45424177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12150 2) := by change ¬ Nat.Prime 590490001; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 3413237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12150 3) := by change ¬ Nat.Prime 590465701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84352243) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12150 : row 12150 = 0 := (row_12150.checked).trans rfl
theorem model_count_12150 : modelRow 12150 = 0 := by decide
theorem modelrow_agrees_12150 : modelRow 12150 = row 12150 := model_count_12150.trans row_count_12150.symm
def row_12151 : RowCertificate 12151 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12151 1) := by change ¬ Nat.Prime 590611507; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 3263047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12151 2) := by change ¬ Nat.Prime 590587205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118117441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12151 3) := by change ¬ Nat.Prime 590562903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196854301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12151 : row 12151 = 0 := (row_12151.checked).trans rfl
theorem model_count_12151 : modelRow 12151 = 0 := by decide
theorem modelrow_agrees_12151 : modelRow 12151 = row 12151 := model_count_12151.trans row_count_12151.symm
def row_12152 : RowCertificate 12152 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12152 1) := by change ¬ Nat.Prime 590708721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196902907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12152 2) := by change ¬ Nat.Prime 590684417; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14406937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12152 3) := by change Nat.Prime 590660113; exact ProjectEulerSolutions.P58PrimeNodes.p_590660113.prime
    exact decide_eq_true prime
theorem row_count_12152 : row 12152 = 1 := (row_12152.checked).trans rfl
theorem model_count_12152 : modelRow 12152 = 1 := by decide
theorem modelrow_agrees_12152 : modelRow 12152 = row 12152 := model_count_12152.trans row_count_12152.symm
def row_12153 : RowCertificate 12153 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12153 1) := by change ¬ Nat.Prime 590805943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84400849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12153 2) := by change ¬ Nat.Prime 590781637; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34751861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12153 3) := by change ¬ Nat.Prime 590757331; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 1609693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12153 : row 12153 = 0 := (row_12153.checked).trans rfl
theorem model_count_12153 : modelRow 12153 = 0 := by decide
theorem modelrow_agrees_12153 : modelRow 12153 = row 12153 := model_count_12153.trans row_count_12153.symm
def row_12154 : RowCertificate 12154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12154 1) := by change ¬ Nat.Prime 590903173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84414739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12154 2) := by change ¬ Nat.Prime 590878865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118175773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12154 3) := by change ¬ Nat.Prime 590854557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 196951519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12154 : row 12154 = 0 := (row_12154.checked).trans rfl
theorem model_count_12154 : modelRow 12154 = 0 := by decide
theorem modelrow_agrees_12154 : modelRow 12154 = row 12154 := model_count_12154.trans row_count_12154.symm
def row_12155 : RowCertificate 12155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12155 1) := by change ¬ Nat.Prime 591000411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197000137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12155 2) := by change ¬ Nat.Prime 590976101; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 5229877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12155 3) := by change ¬ Nat.Prime 590951791; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 19062961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12155 : row 12155 = 0 := (row_12155.checked).trans rfl
theorem model_count_12155 : modelRow 12155 = 0 := by decide
theorem modelrow_agrees_12155 : modelRow 12155 = row 12155 := model_count_12155.trans row_count_12155.symm
def row_12156 : RowCertificate 12156 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12156 1) := by change ¬ Nat.Prime 591097657; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31110403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12156 2) := by change ¬ Nat.Prime 591073345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118214669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12156 3) := by change ¬ Nat.Prime 591049033; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 6093289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12156 : row 12156 = 0 := (row_12156.checked).trans rfl
theorem model_count_12156 : modelRow 12156 = 0 := by decide
theorem modelrow_agrees_12156 : modelRow 12156 = row 12156 := model_count_12156.trans row_count_12156.symm
def row_12157 : RowCertificate 12157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12157 1) := by change ¬ Nat.Prime 591194911; exact PrattCertificates.composite_of_product (factor := 3313) (cofactor := 178447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12157 2) := by change ¬ Nat.Prime 591170597; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34774741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12157 3) := by change ¬ Nat.Prime 591146283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197048761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12157 : row 12157 = 0 := (row_12157.checked).trans rfl
theorem model_count_12157 : modelRow 12157 = 0 := by decide
theorem modelrow_agrees_12157 : modelRow 12157 = row 12157 := model_count_12157.trans row_count_12157.symm
def row_12158 : RowCertificate 12158 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12158 1) := by change ¬ Nat.Prime 591292173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197097391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12158 2) := by change ¬ Nat.Prime 591267857; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 2134541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12158 3) := by change ¬ Nat.Prime 591243541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84463363) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12158 : row 12158 = 0 := (row_12158.checked).trans rfl
theorem model_count_12158 : modelRow 12158 = 0 := by decide
theorem modelrow_agrees_12158 : modelRow 12158 = row 12158 := model_count_12158.trans row_count_12158.symm
def row_12159 : RowCertificate 12159 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12159 1) := by change ¬ Nat.Prime 591389443; exact PrattCertificates.composite_of_product (factor := 13003) (cofactor := 45481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12159 2) := by change ¬ Nat.Prime 591365125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118273025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12159 3) := by change ¬ Nat.Prime 591340807; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 8100559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12159 : row 12159 = 0 := (row_12159.checked).trans rfl
theorem model_count_12159 : modelRow 12159 = 0 := by decide
theorem modelrow_agrees_12159 : modelRow 12159 = row 12159 := model_count_12159.trans row_count_12159.symm
def row_12160 : RowCertificate 12160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12160 1) := by change ¬ Nat.Prime 591486721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84498103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12160 2) := by change ¬ Nat.Prime 591462401; exact PrattCertificates.composite_of_product (factor := 1493) (cofactor := 396157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12160 3) := by change ¬ Nat.Prime 591438081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197146027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12160 : row 12160 = 0 := (row_12160.checked).trans rfl
theorem model_count_12160 : modelRow 12160 = 0 := by decide
theorem modelrow_agrees_12160 : modelRow 12160 = row 12160 := model_count_12160.trans row_count_12160.symm
private def atomic_12033 : CountSegment 1 10 13120 12032 12033 4866 4867 := CountSegment.single (first := 12032) (initial := 4866) row_count_12033 (by decide)
private theorem atomic_12033_agreement : RowAgreement 12032 12033 := RowAgreement.single (first := 12032) modelrow_agrees_12033
private def atomic_12034 : CountSegment 1 10 13120 12033 12034 4867 4867 := CountSegment.single (first := 12033) (initial := 4867) row_count_12034 (by decide)
private theorem atomic_12034_agreement : RowAgreement 12033 12034 := RowAgreement.single (first := 12033) modelrow_agrees_12034
private def atomic_12035 : CountSegment 1 10 13120 12034 12035 4867 4868 := CountSegment.single (first := 12034) (initial := 4867) row_count_12035 (by decide)
private theorem atomic_12035_agreement : RowAgreement 12034 12035 := RowAgreement.single (first := 12034) modelrow_agrees_12035
private def atomic_12036 : CountSegment 1 10 13120 12035 12036 4868 4868 := CountSegment.single (first := 12035) (initial := 4868) row_count_12036 (by decide)
private theorem atomic_12036_agreement : RowAgreement 12035 12036 := RowAgreement.single (first := 12035) modelrow_agrees_12036
private def atomic_12037 : CountSegment 1 10 13120 12036 12037 4868 4869 := CountSegment.single (first := 12036) (initial := 4868) row_count_12037 (by decide)
private theorem atomic_12037_agreement : RowAgreement 12036 12037 := RowAgreement.single (first := 12036) modelrow_agrees_12037
private def atomic_12038 : CountSegment 1 10 13120 12037 12038 4869 4869 := CountSegment.single (first := 12037) (initial := 4869) row_count_12038 (by decide)
private theorem atomic_12038_agreement : RowAgreement 12037 12038 := RowAgreement.single (first := 12037) modelrow_agrees_12038
private def atomic_12039 : CountSegment 1 10 13120 12038 12039 4869 4870 := CountSegment.single (first := 12038) (initial := 4869) row_count_12039 (by decide)
private theorem atomic_12039_agreement : RowAgreement 12038 12039 := RowAgreement.single (first := 12038) modelrow_agrees_12039
private def atomic_12040 : CountSegment 1 10 13120 12039 12040 4870 4870 := CountSegment.single (first := 12039) (initial := 4870) row_count_12040 (by decide)
private theorem atomic_12040_agreement : RowAgreement 12039 12040 := RowAgreement.single (first := 12039) modelrow_agrees_12040
private def atomic_12041 : CountSegment 1 10 13120 12040 12041 4870 4870 := CountSegment.single (first := 12040) (initial := 4870) row_count_12041 (by decide)
private theorem atomic_12041_agreement : RowAgreement 12040 12041 := RowAgreement.single (first := 12040) modelrow_agrees_12041
private def atomic_12042 : CountSegment 1 10 13120 12041 12042 4870 4871 := CountSegment.single (first := 12041) (initial := 4870) row_count_12042 (by decide)
private theorem atomic_12042_agreement : RowAgreement 12041 12042 := RowAgreement.single (first := 12041) modelrow_agrees_12042
private def atomic_12043 : CountSegment 1 10 13120 12042 12043 4871 4872 := CountSegment.single (first := 12042) (initial := 4871) row_count_12043 (by decide)
private theorem atomic_12043_agreement : RowAgreement 12042 12043 := RowAgreement.single (first := 12042) modelrow_agrees_12043
private def atomic_12044 : CountSegment 1 10 13120 12043 12044 4872 4872 := CountSegment.single (first := 12043) (initial := 4872) row_count_12044 (by decide)
private theorem atomic_12044_agreement : RowAgreement 12043 12044 := RowAgreement.single (first := 12043) modelrow_agrees_12044
private def atomic_12045 : CountSegment 1 10 13120 12044 12045 4872 4872 := CountSegment.single (first := 12044) (initial := 4872) row_count_12045 (by decide)
private theorem atomic_12045_agreement : RowAgreement 12044 12045 := RowAgreement.single (first := 12044) modelrow_agrees_12045
private def atomic_12046 : CountSegment 1 10 13120 12045 12046 4872 4872 := CountSegment.single (first := 12045) (initial := 4872) row_count_12046 (by decide)
private theorem atomic_12046_agreement : RowAgreement 12045 12046 := RowAgreement.single (first := 12045) modelrow_agrees_12046
private def atomic_12047 : CountSegment 1 10 13120 12046 12047 4872 4872 := CountSegment.single (first := 12046) (initial := 4872) row_count_12047 (by decide)
private theorem atomic_12047_agreement : RowAgreement 12046 12047 := RowAgreement.single (first := 12046) modelrow_agrees_12047
private def atomic_12048 : CountSegment 1 10 13120 12047 12048 4872 4873 := CountSegment.single (first := 12047) (initial := 4872) row_count_12048 (by decide)
private theorem atomic_12048_agreement : RowAgreement 12047 12048 := RowAgreement.single (first := 12047) modelrow_agrees_12048
private def atomic_12049 : CountSegment 1 10 13120 12048 12049 4873 4873 := CountSegment.single (first := 12048) (initial := 4873) row_count_12049 (by decide)
private theorem atomic_12049_agreement : RowAgreement 12048 12049 := RowAgreement.single (first := 12048) modelrow_agrees_12049
private def atomic_12050 : CountSegment 1 10 13120 12049 12050 4873 4873 := CountSegment.single (first := 12049) (initial := 4873) row_count_12050 (by decide)
private theorem atomic_12050_agreement : RowAgreement 12049 12050 := RowAgreement.single (first := 12049) modelrow_agrees_12050
private def atomic_12051 : CountSegment 1 10 13120 12050 12051 4873 4875 := CountSegment.single (first := 12050) (initial := 4873) row_count_12051 (by decide)
private theorem atomic_12051_agreement : RowAgreement 12050 12051 := RowAgreement.single (first := 12050) modelrow_agrees_12051
private def atomic_12052 : CountSegment 1 10 13120 12051 12052 4875 4875 := CountSegment.single (first := 12051) (initial := 4875) row_count_12052 (by decide)
private theorem atomic_12052_agreement : RowAgreement 12051 12052 := RowAgreement.single (first := 12051) modelrow_agrees_12052
private def atomic_12053 : CountSegment 1 10 13120 12052 12053 4875 4875 := CountSegment.single (first := 12052) (initial := 4875) row_count_12053 (by decide)
private theorem atomic_12053_agreement : RowAgreement 12052 12053 := RowAgreement.single (first := 12052) modelrow_agrees_12053
private def atomic_12054 : CountSegment 1 10 13120 12053 12054 4875 4876 := CountSegment.single (first := 12053) (initial := 4875) row_count_12054 (by decide)
private theorem atomic_12054_agreement : RowAgreement 12053 12054 := RowAgreement.single (first := 12053) modelrow_agrees_12054
private def atomic_12055 : CountSegment 1 10 13120 12054 12055 4876 4876 := CountSegment.single (first := 12054) (initial := 4876) row_count_12055 (by decide)
private theorem atomic_12055_agreement : RowAgreement 12054 12055 := RowAgreement.single (first := 12054) modelrow_agrees_12055
private def atomic_12056 : CountSegment 1 10 13120 12055 12056 4876 4876 := CountSegment.single (first := 12055) (initial := 4876) row_count_12056 (by decide)
private theorem atomic_12056_agreement : RowAgreement 12055 12056 := RowAgreement.single (first := 12055) modelrow_agrees_12056
private def atomic_12057 : CountSegment 1 10 13120 12056 12057 4876 4877 := CountSegment.single (first := 12056) (initial := 4876) row_count_12057 (by decide)
private theorem atomic_12057_agreement : RowAgreement 12056 12057 := RowAgreement.single (first := 12056) modelrow_agrees_12057
private def atomic_12058 : CountSegment 1 10 13120 12057 12058 4877 4877 := CountSegment.single (first := 12057) (initial := 4877) row_count_12058 (by decide)
private theorem atomic_12058_agreement : RowAgreement 12057 12058 := RowAgreement.single (first := 12057) modelrow_agrees_12058
private def atomic_12059 : CountSegment 1 10 13120 12058 12059 4877 4877 := CountSegment.single (first := 12058) (initial := 4877) row_count_12059 (by decide)
private theorem atomic_12059_agreement : RowAgreement 12058 12059 := RowAgreement.single (first := 12058) modelrow_agrees_12059
private def atomic_12060 : CountSegment 1 10 13120 12059 12060 4877 4877 := CountSegment.single (first := 12059) (initial := 4877) row_count_12060 (by decide)
private theorem atomic_12060_agreement : RowAgreement 12059 12060 := RowAgreement.single (first := 12059) modelrow_agrees_12060
private def atomic_12061 : CountSegment 1 10 13120 12060 12061 4877 4877 := CountSegment.single (first := 12060) (initial := 4877) row_count_12061 (by decide)
private theorem atomic_12061_agreement : RowAgreement 12060 12061 := RowAgreement.single (first := 12060) modelrow_agrees_12061
private def atomic_12062 : CountSegment 1 10 13120 12061 12062 4877 4878 := CountSegment.single (first := 12061) (initial := 4877) row_count_12062 (by decide)
private theorem atomic_12062_agreement : RowAgreement 12061 12062 := RowAgreement.single (first := 12061) modelrow_agrees_12062
private def atomic_12063 : CountSegment 1 10 13120 12062 12063 4878 4879 := CountSegment.single (first := 12062) (initial := 4878) row_count_12063 (by decide)
private theorem atomic_12063_agreement : RowAgreement 12062 12063 := RowAgreement.single (first := 12062) modelrow_agrees_12063
private def atomic_12064 : CountSegment 1 10 13120 12063 12064 4879 4880 := CountSegment.single (first := 12063) (initial := 4879) row_count_12064 (by decide)
private theorem atomic_12064_agreement : RowAgreement 12063 12064 := RowAgreement.single (first := 12063) modelrow_agrees_12064
private def atomic_12065 : CountSegment 1 10 13120 12064 12065 4880 4880 := CountSegment.single (first := 12064) (initial := 4880) row_count_12065 (by decide)
private theorem atomic_12065_agreement : RowAgreement 12064 12065 := RowAgreement.single (first := 12064) modelrow_agrees_12065
private def atomic_12066 : CountSegment 1 10 13120 12065 12066 4880 4880 := CountSegment.single (first := 12065) (initial := 4880) row_count_12066 (by decide)
private theorem atomic_12066_agreement : RowAgreement 12065 12066 := RowAgreement.single (first := 12065) modelrow_agrees_12066
private def atomic_12067 : CountSegment 1 10 13120 12066 12067 4880 4881 := CountSegment.single (first := 12066) (initial := 4880) row_count_12067 (by decide)
private theorem atomic_12067_agreement : RowAgreement 12066 12067 := RowAgreement.single (first := 12066) modelrow_agrees_12067
private def atomic_12068 : CountSegment 1 10 13120 12067 12068 4881 4881 := CountSegment.single (first := 12067) (initial := 4881) row_count_12068 (by decide)
private theorem atomic_12068_agreement : RowAgreement 12067 12068 := RowAgreement.single (first := 12067) modelrow_agrees_12068
private def atomic_12069 : CountSegment 1 10 13120 12068 12069 4881 4881 := CountSegment.single (first := 12068) (initial := 4881) row_count_12069 (by decide)
private theorem atomic_12069_agreement : RowAgreement 12068 12069 := RowAgreement.single (first := 12068) modelrow_agrees_12069
private def atomic_12070 : CountSegment 1 10 13120 12069 12070 4881 4881 := CountSegment.single (first := 12069) (initial := 4881) row_count_12070 (by decide)
private theorem atomic_12070_agreement : RowAgreement 12069 12070 := RowAgreement.single (first := 12069) modelrow_agrees_12070
private def atomic_12071 : CountSegment 1 10 13120 12070 12071 4881 4881 := CountSegment.single (first := 12070) (initial := 4881) row_count_12071 (by decide)
private theorem atomic_12071_agreement : RowAgreement 12070 12071 := RowAgreement.single (first := 12070) modelrow_agrees_12071
private def atomic_12072 : CountSegment 1 10 13120 12071 12072 4881 4881 := CountSegment.single (first := 12071) (initial := 4881) row_count_12072 (by decide)
private theorem atomic_12072_agreement : RowAgreement 12071 12072 := RowAgreement.single (first := 12071) modelrow_agrees_12072
private def atomic_12073 : CountSegment 1 10 13120 12072 12073 4881 4882 := CountSegment.single (first := 12072) (initial := 4881) row_count_12073 (by decide)
private theorem atomic_12073_agreement : RowAgreement 12072 12073 := RowAgreement.single (first := 12072) modelrow_agrees_12073
private def atomic_12074 : CountSegment 1 10 13120 12073 12074 4882 4882 := CountSegment.single (first := 12073) (initial := 4882) row_count_12074 (by decide)
private theorem atomic_12074_agreement : RowAgreement 12073 12074 := RowAgreement.single (first := 12073) modelrow_agrees_12074
private def atomic_12075 : CountSegment 1 10 13120 12074 12075 4882 4882 := CountSegment.single (first := 12074) (initial := 4882) row_count_12075 (by decide)
private theorem atomic_12075_agreement : RowAgreement 12074 12075 := RowAgreement.single (first := 12074) modelrow_agrees_12075
private def atomic_12076 : CountSegment 1 10 13120 12075 12076 4882 4882 := CountSegment.single (first := 12075) (initial := 4882) row_count_12076 (by decide)
private theorem atomic_12076_agreement : RowAgreement 12075 12076 := RowAgreement.single (first := 12075) modelrow_agrees_12076
private def atomic_12077 : CountSegment 1 10 13120 12076 12077 4882 4883 := CountSegment.single (first := 12076) (initial := 4882) row_count_12077 (by decide)
private theorem atomic_12077_agreement : RowAgreement 12076 12077 := RowAgreement.single (first := 12076) modelrow_agrees_12077
private def atomic_12078 : CountSegment 1 10 13120 12077 12078 4883 4885 := CountSegment.single (first := 12077) (initial := 4883) row_count_12078 (by decide)
private theorem atomic_12078_agreement : RowAgreement 12077 12078 := RowAgreement.single (first := 12077) modelrow_agrees_12078
private def atomic_12079 : CountSegment 1 10 13120 12078 12079 4885 4885 := CountSegment.single (first := 12078) (initial := 4885) row_count_12079 (by decide)
private theorem atomic_12079_agreement : RowAgreement 12078 12079 := RowAgreement.single (first := 12078) modelrow_agrees_12079
private def atomic_12080 : CountSegment 1 10 13120 12079 12080 4885 4885 := CountSegment.single (first := 12079) (initial := 4885) row_count_12080 (by decide)
private theorem atomic_12080_agreement : RowAgreement 12079 12080 := RowAgreement.single (first := 12079) modelrow_agrees_12080
private def atomic_12081 : CountSegment 1 10 13120 12080 12081 4885 4886 := CountSegment.single (first := 12080) (initial := 4885) row_count_12081 (by decide)
private theorem atomic_12081_agreement : RowAgreement 12080 12081 := RowAgreement.single (first := 12080) modelrow_agrees_12081
private def atomic_12082 : CountSegment 1 10 13120 12081 12082 4886 4887 := CountSegment.single (first := 12081) (initial := 4886) row_count_12082 (by decide)
private theorem atomic_12082_agreement : RowAgreement 12081 12082 := RowAgreement.single (first := 12081) modelrow_agrees_12082
private def atomic_12083 : CountSegment 1 10 13120 12082 12083 4887 4887 := CountSegment.single (first := 12082) (initial := 4887) row_count_12083 (by decide)
private theorem atomic_12083_agreement : RowAgreement 12082 12083 := RowAgreement.single (first := 12082) modelrow_agrees_12083
private def atomic_12084 : CountSegment 1 10 13120 12083 12084 4887 4888 := CountSegment.single (first := 12083) (initial := 4887) row_count_12084 (by decide)
private theorem atomic_12084_agreement : RowAgreement 12083 12084 := RowAgreement.single (first := 12083) modelrow_agrees_12084
private def atomic_12085 : CountSegment 1 10 13120 12084 12085 4888 4888 := CountSegment.single (first := 12084) (initial := 4888) row_count_12085 (by decide)
private theorem atomic_12085_agreement : RowAgreement 12084 12085 := RowAgreement.single (first := 12084) modelrow_agrees_12085
private def atomic_12086 : CountSegment 1 10 13120 12085 12086 4888 4888 := CountSegment.single (first := 12085) (initial := 4888) row_count_12086 (by decide)
private theorem atomic_12086_agreement : RowAgreement 12085 12086 := RowAgreement.single (first := 12085) modelrow_agrees_12086
private def atomic_12087 : CountSegment 1 10 13120 12086 12087 4888 4888 := CountSegment.single (first := 12086) (initial := 4888) row_count_12087 (by decide)
private theorem atomic_12087_agreement : RowAgreement 12086 12087 := RowAgreement.single (first := 12086) modelrow_agrees_12087
private def atomic_12088 : CountSegment 1 10 13120 12087 12088 4888 4889 := CountSegment.single (first := 12087) (initial := 4888) row_count_12088 (by decide)
private theorem atomic_12088_agreement : RowAgreement 12087 12088 := RowAgreement.single (first := 12087) modelrow_agrees_12088
private def atomic_12089 : CountSegment 1 10 13120 12088 12089 4889 4889 := CountSegment.single (first := 12088) (initial := 4889) row_count_12089 (by decide)
private theorem atomic_12089_agreement : RowAgreement 12088 12089 := RowAgreement.single (first := 12088) modelrow_agrees_12089
private def atomic_12090 : CountSegment 1 10 13120 12089 12090 4889 4889 := CountSegment.single (first := 12089) (initial := 4889) row_count_12090 (by decide)
private theorem atomic_12090_agreement : RowAgreement 12089 12090 := RowAgreement.single (first := 12089) modelrow_agrees_12090
private def atomic_12091 : CountSegment 1 10 13120 12090 12091 4889 4889 := CountSegment.single (first := 12090) (initial := 4889) row_count_12091 (by decide)
private theorem atomic_12091_agreement : RowAgreement 12090 12091 := RowAgreement.single (first := 12090) modelrow_agrees_12091
private def atomic_12092 : CountSegment 1 10 13120 12091 12092 4889 4890 := CountSegment.single (first := 12091) (initial := 4889) row_count_12092 (by decide)
private theorem atomic_12092_agreement : RowAgreement 12091 12092 := RowAgreement.single (first := 12091) modelrow_agrees_12092
private def atomic_12093 : CountSegment 1 10 13120 12092 12093 4890 4891 := CountSegment.single (first := 12092) (initial := 4890) row_count_12093 (by decide)
private theorem atomic_12093_agreement : RowAgreement 12092 12093 := RowAgreement.single (first := 12092) modelrow_agrees_12093
private def atomic_12094 : CountSegment 1 10 13120 12093 12094 4891 4891 := CountSegment.single (first := 12093) (initial := 4891) row_count_12094 (by decide)
private theorem atomic_12094_agreement : RowAgreement 12093 12094 := RowAgreement.single (first := 12093) modelrow_agrees_12094
private def atomic_12095 : CountSegment 1 10 13120 12094 12095 4891 4891 := CountSegment.single (first := 12094) (initial := 4891) row_count_12095 (by decide)
private theorem atomic_12095_agreement : RowAgreement 12094 12095 := RowAgreement.single (first := 12094) modelrow_agrees_12095
private def atomic_12096 : CountSegment 1 10 13120 12095 12096 4891 4892 := CountSegment.single (first := 12095) (initial := 4891) row_count_12096 (by decide)
private theorem atomic_12096_agreement : RowAgreement 12095 12096 := RowAgreement.single (first := 12095) modelrow_agrees_12096
private def atomic_12097 : CountSegment 1 10 13120 12096 12097 4892 4892 := CountSegment.single (first := 12096) (initial := 4892) row_count_12097 (by decide)
private theorem atomic_12097_agreement : RowAgreement 12096 12097 := RowAgreement.single (first := 12096) modelrow_agrees_12097
private def atomic_12098 : CountSegment 1 10 13120 12097 12098 4892 4894 := CountSegment.single (first := 12097) (initial := 4892) row_count_12098 (by decide)
private theorem atomic_12098_agreement : RowAgreement 12097 12098 := RowAgreement.single (first := 12097) modelrow_agrees_12098
private def atomic_12099 : CountSegment 1 10 13120 12098 12099 4894 4894 := CountSegment.single (first := 12098) (initial := 4894) row_count_12099 (by decide)
private theorem atomic_12099_agreement : RowAgreement 12098 12099 := RowAgreement.single (first := 12098) modelrow_agrees_12099
private def atomic_12100 : CountSegment 1 10 13120 12099 12100 4894 4895 := CountSegment.single (first := 12099) (initial := 4894) row_count_12100 (by decide)
private theorem atomic_12100_agreement : RowAgreement 12099 12100 := RowAgreement.single (first := 12099) modelrow_agrees_12100
private def atomic_12101 : CountSegment 1 10 13120 12100 12101 4895 4895 := CountSegment.single (first := 12100) (initial := 4895) row_count_12101 (by decide)
private theorem atomic_12101_agreement : RowAgreement 12100 12101 := RowAgreement.single (first := 12100) modelrow_agrees_12101
private def atomic_12102 : CountSegment 1 10 13120 12101 12102 4895 4895 := CountSegment.single (first := 12101) (initial := 4895) row_count_12102 (by decide)
private theorem atomic_12102_agreement : RowAgreement 12101 12102 := RowAgreement.single (first := 12101) modelrow_agrees_12102
private def atomic_12103 : CountSegment 1 10 13120 12102 12103 4895 4895 := CountSegment.single (first := 12102) (initial := 4895) row_count_12103 (by decide)
private theorem atomic_12103_agreement : RowAgreement 12102 12103 := RowAgreement.single (first := 12102) modelrow_agrees_12103
private def atomic_12104 : CountSegment 1 10 13120 12103 12104 4895 4896 := CountSegment.single (first := 12103) (initial := 4895) row_count_12104 (by decide)
private theorem atomic_12104_agreement : RowAgreement 12103 12104 := RowAgreement.single (first := 12103) modelrow_agrees_12104
private def atomic_12105 : CountSegment 1 10 13120 12104 12105 4896 4896 := CountSegment.single (first := 12104) (initial := 4896) row_count_12105 (by decide)
private theorem atomic_12105_agreement : RowAgreement 12104 12105 := RowAgreement.single (first := 12104) modelrow_agrees_12105
private def atomic_12106 : CountSegment 1 10 13120 12105 12106 4896 4897 := CountSegment.single (first := 12105) (initial := 4896) row_count_12106 (by decide)
private theorem atomic_12106_agreement : RowAgreement 12105 12106 := RowAgreement.single (first := 12105) modelrow_agrees_12106
private def atomic_12107 : CountSegment 1 10 13120 12106 12107 4897 4897 := CountSegment.single (first := 12106) (initial := 4897) row_count_12107 (by decide)
private theorem atomic_12107_agreement : RowAgreement 12106 12107 := RowAgreement.single (first := 12106) modelrow_agrees_12107
private def atomic_12108 : CountSegment 1 10 13120 12107 12108 4897 4897 := CountSegment.single (first := 12107) (initial := 4897) row_count_12108 (by decide)
private theorem atomic_12108_agreement : RowAgreement 12107 12108 := RowAgreement.single (first := 12107) modelrow_agrees_12108
private def atomic_12109 : CountSegment 1 10 13120 12108 12109 4897 4898 := CountSegment.single (first := 12108) (initial := 4897) row_count_12109 (by decide)
private theorem atomic_12109_agreement : RowAgreement 12108 12109 := RowAgreement.single (first := 12108) modelrow_agrees_12109
private def atomic_12110 : CountSegment 1 10 13120 12109 12110 4898 4899 := CountSegment.single (first := 12109) (initial := 4898) row_count_12110 (by decide)
private theorem atomic_12110_agreement : RowAgreement 12109 12110 := RowAgreement.single (first := 12109) modelrow_agrees_12110
private def atomic_12111 : CountSegment 1 10 13120 12110 12111 4899 4900 := CountSegment.single (first := 12110) (initial := 4899) row_count_12111 (by decide)
private theorem atomic_12111_agreement : RowAgreement 12110 12111 := RowAgreement.single (first := 12110) modelrow_agrees_12111
private def atomic_12112 : CountSegment 1 10 13120 12111 12112 4900 4900 := CountSegment.single (first := 12111) (initial := 4900) row_count_12112 (by decide)
private theorem atomic_12112_agreement : RowAgreement 12111 12112 := RowAgreement.single (first := 12111) modelrow_agrees_12112
private def atomic_12113 : CountSegment 1 10 13120 12112 12113 4900 4901 := CountSegment.single (first := 12112) (initial := 4900) row_count_12113 (by decide)
private theorem atomic_12113_agreement : RowAgreement 12112 12113 := RowAgreement.single (first := 12112) modelrow_agrees_12113
private def atomic_12114 : CountSegment 1 10 13120 12113 12114 4901 4901 := CountSegment.single (first := 12113) (initial := 4901) row_count_12114 (by decide)
private theorem atomic_12114_agreement : RowAgreement 12113 12114 := RowAgreement.single (first := 12113) modelrow_agrees_12114
private def atomic_12115 : CountSegment 1 10 13120 12114 12115 4901 4901 := CountSegment.single (first := 12114) (initial := 4901) row_count_12115 (by decide)
private theorem atomic_12115_agreement : RowAgreement 12114 12115 := RowAgreement.single (first := 12114) modelrow_agrees_12115
private def atomic_12116 : CountSegment 1 10 13120 12115 12116 4901 4901 := CountSegment.single (first := 12115) (initial := 4901) row_count_12116 (by decide)
private theorem atomic_12116_agreement : RowAgreement 12115 12116 := RowAgreement.single (first := 12115) modelrow_agrees_12116
private def atomic_12117 : CountSegment 1 10 13120 12116 12117 4901 4902 := CountSegment.single (first := 12116) (initial := 4901) row_count_12117 (by decide)
private theorem atomic_12117_agreement : RowAgreement 12116 12117 := RowAgreement.single (first := 12116) modelrow_agrees_12117
private def atomic_12118 : CountSegment 1 10 13120 12117 12118 4902 4902 := CountSegment.single (first := 12117) (initial := 4902) row_count_12118 (by decide)
private theorem atomic_12118_agreement : RowAgreement 12117 12118 := RowAgreement.single (first := 12117) modelrow_agrees_12118
private def atomic_12119 : CountSegment 1 10 13120 12118 12119 4902 4902 := CountSegment.single (first := 12118) (initial := 4902) row_count_12119 (by decide)
private theorem atomic_12119_agreement : RowAgreement 12118 12119 := RowAgreement.single (first := 12118) modelrow_agrees_12119
private def atomic_12120 : CountSegment 1 10 13120 12119 12120 4902 4904 := CountSegment.single (first := 12119) (initial := 4902) row_count_12120 (by decide)
private theorem atomic_12120_agreement : RowAgreement 12119 12120 := RowAgreement.single (first := 12119) modelrow_agrees_12120
private def atomic_12121 : CountSegment 1 10 13120 12120 12121 4904 4905 := CountSegment.single (first := 12120) (initial := 4904) row_count_12121 (by decide)
private theorem atomic_12121_agreement : RowAgreement 12120 12121 := RowAgreement.single (first := 12120) modelrow_agrees_12121
private def atomic_12122 : CountSegment 1 10 13120 12121 12122 4905 4905 := CountSegment.single (first := 12121) (initial := 4905) row_count_12122 (by decide)
private theorem atomic_12122_agreement : RowAgreement 12121 12122 := RowAgreement.single (first := 12121) modelrow_agrees_12122
private def atomic_12123 : CountSegment 1 10 13120 12122 12123 4905 4906 := CountSegment.single (first := 12122) (initial := 4905) row_count_12123 (by decide)
private theorem atomic_12123_agreement : RowAgreement 12122 12123 := RowAgreement.single (first := 12122) modelrow_agrees_12123
private def atomic_12124 : CountSegment 1 10 13120 12123 12124 4906 4906 := CountSegment.single (first := 12123) (initial := 4906) row_count_12124 (by decide)
private theorem atomic_12124_agreement : RowAgreement 12123 12124 := RowAgreement.single (first := 12123) modelrow_agrees_12124
private def atomic_12125 : CountSegment 1 10 13120 12124 12125 4906 4907 := CountSegment.single (first := 12124) (initial := 4906) row_count_12125 (by decide)
private theorem atomic_12125_agreement : RowAgreement 12124 12125 := RowAgreement.single (first := 12124) modelrow_agrees_12125
private def atomic_12126 : CountSegment 1 10 13120 12125 12126 4907 4907 := CountSegment.single (first := 12125) (initial := 4907) row_count_12126 (by decide)
private theorem atomic_12126_agreement : RowAgreement 12125 12126 := RowAgreement.single (first := 12125) modelrow_agrees_12126
private def atomic_12127 : CountSegment 1 10 13120 12126 12127 4907 4907 := CountSegment.single (first := 12126) (initial := 4907) row_count_12127 (by decide)
private theorem atomic_12127_agreement : RowAgreement 12126 12127 := RowAgreement.single (first := 12126) modelrow_agrees_12127
private def atomic_12128 : CountSegment 1 10 13120 12127 12128 4907 4907 := CountSegment.single (first := 12127) (initial := 4907) row_count_12128 (by decide)
private theorem atomic_12128_agreement : RowAgreement 12127 12128 := RowAgreement.single (first := 12127) modelrow_agrees_12128
private def atomic_12129 : CountSegment 1 10 13120 12128 12129 4907 4907 := CountSegment.single (first := 12128) (initial := 4907) row_count_12129 (by decide)
private theorem atomic_12129_agreement : RowAgreement 12128 12129 := RowAgreement.single (first := 12128) modelrow_agrees_12129
private def atomic_12130 : CountSegment 1 10 13120 12129 12130 4907 4909 := CountSegment.single (first := 12129) (initial := 4907) row_count_12130 (by decide)
private theorem atomic_12130_agreement : RowAgreement 12129 12130 := RowAgreement.single (first := 12129) modelrow_agrees_12130
private def atomic_12131 : CountSegment 1 10 13120 12130 12131 4909 4909 := CountSegment.single (first := 12130) (initial := 4909) row_count_12131 (by decide)
private theorem atomic_12131_agreement : RowAgreement 12130 12131 := RowAgreement.single (first := 12130) modelrow_agrees_12131
private def atomic_12132 : CountSegment 1 10 13120 12131 12132 4909 4910 := CountSegment.single (first := 12131) (initial := 4909) row_count_12132 (by decide)
private theorem atomic_12132_agreement : RowAgreement 12131 12132 := RowAgreement.single (first := 12131) modelrow_agrees_12132
private def atomic_12133 : CountSegment 1 10 13120 12132 12133 4910 4910 := CountSegment.single (first := 12132) (initial := 4910) row_count_12133 (by decide)
private theorem atomic_12133_agreement : RowAgreement 12132 12133 := RowAgreement.single (first := 12132) modelrow_agrees_12133
private def atomic_12134 : CountSegment 1 10 13120 12133 12134 4910 4910 := CountSegment.single (first := 12133) (initial := 4910) row_count_12134 (by decide)
private theorem atomic_12134_agreement : RowAgreement 12133 12134 := RowAgreement.single (first := 12133) modelrow_agrees_12134
private def atomic_12135 : CountSegment 1 10 13120 12134 12135 4910 4912 := CountSegment.single (first := 12134) (initial := 4910) row_count_12135 (by decide)
private theorem atomic_12135_agreement : RowAgreement 12134 12135 := RowAgreement.single (first := 12134) modelrow_agrees_12135
private def atomic_12136 : CountSegment 1 10 13120 12135 12136 4912 4912 := CountSegment.single (first := 12135) (initial := 4912) row_count_12136 (by decide)
private theorem atomic_12136_agreement : RowAgreement 12135 12136 := RowAgreement.single (first := 12135) modelrow_agrees_12136
private def atomic_12137 : CountSegment 1 10 13120 12136 12137 4912 4912 := CountSegment.single (first := 12136) (initial := 4912) row_count_12137 (by decide)
private theorem atomic_12137_agreement : RowAgreement 12136 12137 := RowAgreement.single (first := 12136) modelrow_agrees_12137
private def atomic_12138 : CountSegment 1 10 13120 12137 12138 4912 4913 := CountSegment.single (first := 12137) (initial := 4912) row_count_12138 (by decide)
private theorem atomic_12138_agreement : RowAgreement 12137 12138 := RowAgreement.single (first := 12137) modelrow_agrees_12138
private def atomic_12139 : CountSegment 1 10 13120 12138 12139 4913 4913 := CountSegment.single (first := 12138) (initial := 4913) row_count_12139 (by decide)
private theorem atomic_12139_agreement : RowAgreement 12138 12139 := RowAgreement.single (first := 12138) modelrow_agrees_12139
private def atomic_12140 : CountSegment 1 10 13120 12139 12140 4913 4913 := CountSegment.single (first := 12139) (initial := 4913) row_count_12140 (by decide)
private theorem atomic_12140_agreement : RowAgreement 12139 12140 := RowAgreement.single (first := 12139) modelrow_agrees_12140
private def atomic_12141 : CountSegment 1 10 13120 12140 12141 4913 4914 := CountSegment.single (first := 12140) (initial := 4913) row_count_12141 (by decide)
private theorem atomic_12141_agreement : RowAgreement 12140 12141 := RowAgreement.single (first := 12140) modelrow_agrees_12141
private def atomic_12142 : CountSegment 1 10 13120 12141 12142 4914 4915 := CountSegment.single (first := 12141) (initial := 4914) row_count_12142 (by decide)
private theorem atomic_12142_agreement : RowAgreement 12141 12142 := RowAgreement.single (first := 12141) modelrow_agrees_12142
private def atomic_12143 : CountSegment 1 10 13120 12142 12143 4915 4915 := CountSegment.single (first := 12142) (initial := 4915) row_count_12143 (by decide)
private theorem atomic_12143_agreement : RowAgreement 12142 12143 := RowAgreement.single (first := 12142) modelrow_agrees_12143
private def atomic_12144 : CountSegment 1 10 13120 12143 12144 4915 4915 := CountSegment.single (first := 12143) (initial := 4915) row_count_12144 (by decide)
private theorem atomic_12144_agreement : RowAgreement 12143 12144 := RowAgreement.single (first := 12143) modelrow_agrees_12144
private def atomic_12145 : CountSegment 1 10 13120 12144 12145 4915 4916 := CountSegment.single (first := 12144) (initial := 4915) row_count_12145 (by decide)
private theorem atomic_12145_agreement : RowAgreement 12144 12145 := RowAgreement.single (first := 12144) modelrow_agrees_12145
private def atomic_12146 : CountSegment 1 10 13120 12145 12146 4916 4916 := CountSegment.single (first := 12145) (initial := 4916) row_count_12146 (by decide)
private theorem atomic_12146_agreement : RowAgreement 12145 12146 := RowAgreement.single (first := 12145) modelrow_agrees_12146
private def atomic_12147 : CountSegment 1 10 13120 12146 12147 4916 4916 := CountSegment.single (first := 12146) (initial := 4916) row_count_12147 (by decide)
private theorem atomic_12147_agreement : RowAgreement 12146 12147 := RowAgreement.single (first := 12146) modelrow_agrees_12147
private def atomic_12148 : CountSegment 1 10 13120 12147 12148 4916 4917 := CountSegment.single (first := 12147) (initial := 4916) row_count_12148 (by decide)
private theorem atomic_12148_agreement : RowAgreement 12147 12148 := RowAgreement.single (first := 12147) modelrow_agrees_12148
private def atomic_12149 : CountSegment 1 10 13120 12148 12149 4917 4918 := CountSegment.single (first := 12148) (initial := 4917) row_count_12149 (by decide)
private theorem atomic_12149_agreement : RowAgreement 12148 12149 := RowAgreement.single (first := 12148) modelrow_agrees_12149
private def atomic_12150 : CountSegment 1 10 13120 12149 12150 4918 4918 := CountSegment.single (first := 12149) (initial := 4918) row_count_12150 (by decide)
private theorem atomic_12150_agreement : RowAgreement 12149 12150 := RowAgreement.single (first := 12149) modelrow_agrees_12150
private def atomic_12151 : CountSegment 1 10 13120 12150 12151 4918 4918 := CountSegment.single (first := 12150) (initial := 4918) row_count_12151 (by decide)
private theorem atomic_12151_agreement : RowAgreement 12150 12151 := RowAgreement.single (first := 12150) modelrow_agrees_12151
private def atomic_12152 : CountSegment 1 10 13120 12151 12152 4918 4919 := CountSegment.single (first := 12151) (initial := 4918) row_count_12152 (by decide)
private theorem atomic_12152_agreement : RowAgreement 12151 12152 := RowAgreement.single (first := 12151) modelrow_agrees_12152
private def atomic_12153 : CountSegment 1 10 13120 12152 12153 4919 4919 := CountSegment.single (first := 12152) (initial := 4919) row_count_12153 (by decide)
private theorem atomic_12153_agreement : RowAgreement 12152 12153 := RowAgreement.single (first := 12152) modelrow_agrees_12153
private def atomic_12154 : CountSegment 1 10 13120 12153 12154 4919 4919 := CountSegment.single (first := 12153) (initial := 4919) row_count_12154 (by decide)
private theorem atomic_12154_agreement : RowAgreement 12153 12154 := RowAgreement.single (first := 12153) modelrow_agrees_12154
private def atomic_12155 : CountSegment 1 10 13120 12154 12155 4919 4919 := CountSegment.single (first := 12154) (initial := 4919) row_count_12155 (by decide)
private theorem atomic_12155_agreement : RowAgreement 12154 12155 := RowAgreement.single (first := 12154) modelrow_agrees_12155
private def atomic_12156 : CountSegment 1 10 13120 12155 12156 4919 4919 := CountSegment.single (first := 12155) (initial := 4919) row_count_12156 (by decide)
private theorem atomic_12156_agreement : RowAgreement 12155 12156 := RowAgreement.single (first := 12155) modelrow_agrees_12156
private def atomic_12157 : CountSegment 1 10 13120 12156 12157 4919 4919 := CountSegment.single (first := 12156) (initial := 4919) row_count_12157 (by decide)
private theorem atomic_12157_agreement : RowAgreement 12156 12157 := RowAgreement.single (first := 12156) modelrow_agrees_12157
private def atomic_12158 : CountSegment 1 10 13120 12157 12158 4919 4919 := CountSegment.single (first := 12157) (initial := 4919) row_count_12158 (by decide)
private theorem atomic_12158_agreement : RowAgreement 12157 12158 := RowAgreement.single (first := 12157) modelrow_agrees_12158
private def atomic_12159 : CountSegment 1 10 13120 12158 12159 4919 4919 := CountSegment.single (first := 12158) (initial := 4919) row_count_12159 (by decide)
private theorem atomic_12159_agreement : RowAgreement 12158 12159 := RowAgreement.single (first := 12158) modelrow_agrees_12159
private def atomic_12160 : CountSegment 1 10 13120 12159 12160 4919 4919 := CountSegment.single (first := 12159) (initial := 4919) row_count_12160 (by decide)
private theorem atomic_12160_agreement : RowAgreement 12159 12160 := RowAgreement.single (first := 12159) modelrow_agrees_12160
private def within_094_000 : CountSegment 1 10 13120 12032 12034 4866 4867 := CountSegment.join atomic_12033 atomic_12034
private theorem within_094_000_agreement : RowAgreement 12032 12034 := RowAgreement.join atomic_12033_agreement atomic_12034_agreement
private def within_094_001 : CountSegment 1 10 13120 12034 12036 4867 4868 := CountSegment.join atomic_12035 atomic_12036
private theorem within_094_001_agreement : RowAgreement 12034 12036 := RowAgreement.join atomic_12035_agreement atomic_12036_agreement
private def within_094_002 : CountSegment 1 10 13120 12032 12036 4866 4868 := CountSegment.join within_094_000 within_094_001
private theorem within_094_002_agreement : RowAgreement 12032 12036 := RowAgreement.join within_094_000_agreement within_094_001_agreement
private def within_094_003 : CountSegment 1 10 13120 12036 12038 4868 4869 := CountSegment.join atomic_12037 atomic_12038
private theorem within_094_003_agreement : RowAgreement 12036 12038 := RowAgreement.join atomic_12037_agreement atomic_12038_agreement
private def within_094_004 : CountSegment 1 10 13120 12038 12040 4869 4870 := CountSegment.join atomic_12039 atomic_12040
private theorem within_094_004_agreement : RowAgreement 12038 12040 := RowAgreement.join atomic_12039_agreement atomic_12040_agreement
private def within_094_005 : CountSegment 1 10 13120 12036 12040 4868 4870 := CountSegment.join within_094_003 within_094_004
private theorem within_094_005_agreement : RowAgreement 12036 12040 := RowAgreement.join within_094_003_agreement within_094_004_agreement
private def within_094_006 : CountSegment 1 10 13120 12032 12040 4866 4870 := CountSegment.join within_094_002 within_094_005
private theorem within_094_006_agreement : RowAgreement 12032 12040 := RowAgreement.join within_094_002_agreement within_094_005_agreement
private def within_094_007 : CountSegment 1 10 13120 12040 12042 4870 4871 := CountSegment.join atomic_12041 atomic_12042
private theorem within_094_007_agreement : RowAgreement 12040 12042 := RowAgreement.join atomic_12041_agreement atomic_12042_agreement
private def within_094_008 : CountSegment 1 10 13120 12042 12044 4871 4872 := CountSegment.join atomic_12043 atomic_12044
private theorem within_094_008_agreement : RowAgreement 12042 12044 := RowAgreement.join atomic_12043_agreement atomic_12044_agreement
private def within_094_009 : CountSegment 1 10 13120 12040 12044 4870 4872 := CountSegment.join within_094_007 within_094_008
private theorem within_094_009_agreement : RowAgreement 12040 12044 := RowAgreement.join within_094_007_agreement within_094_008_agreement
private def within_094_010 : CountSegment 1 10 13120 12044 12046 4872 4872 := CountSegment.join atomic_12045 atomic_12046
private theorem within_094_010_agreement : RowAgreement 12044 12046 := RowAgreement.join atomic_12045_agreement atomic_12046_agreement
private def within_094_011 : CountSegment 1 10 13120 12046 12048 4872 4873 := CountSegment.join atomic_12047 atomic_12048
private theorem within_094_011_agreement : RowAgreement 12046 12048 := RowAgreement.join atomic_12047_agreement atomic_12048_agreement
private def within_094_012 : CountSegment 1 10 13120 12044 12048 4872 4873 := CountSegment.join within_094_010 within_094_011
private theorem within_094_012_agreement : RowAgreement 12044 12048 := RowAgreement.join within_094_010_agreement within_094_011_agreement
private def within_094_013 : CountSegment 1 10 13120 12040 12048 4870 4873 := CountSegment.join within_094_009 within_094_012
private theorem within_094_013_agreement : RowAgreement 12040 12048 := RowAgreement.join within_094_009_agreement within_094_012_agreement
private def within_094_014 : CountSegment 1 10 13120 12032 12048 4866 4873 := CountSegment.join within_094_006 within_094_013
private theorem within_094_014_agreement : RowAgreement 12032 12048 := RowAgreement.join within_094_006_agreement within_094_013_agreement
private def within_094_015 : CountSegment 1 10 13120 12048 12050 4873 4873 := CountSegment.join atomic_12049 atomic_12050
private theorem within_094_015_agreement : RowAgreement 12048 12050 := RowAgreement.join atomic_12049_agreement atomic_12050_agreement
private def within_094_016 : CountSegment 1 10 13120 12050 12052 4873 4875 := CountSegment.join atomic_12051 atomic_12052
private theorem within_094_016_agreement : RowAgreement 12050 12052 := RowAgreement.join atomic_12051_agreement atomic_12052_agreement
private def within_094_017 : CountSegment 1 10 13120 12048 12052 4873 4875 := CountSegment.join within_094_015 within_094_016
private theorem within_094_017_agreement : RowAgreement 12048 12052 := RowAgreement.join within_094_015_agreement within_094_016_agreement
private def within_094_018 : CountSegment 1 10 13120 12052 12054 4875 4876 := CountSegment.join atomic_12053 atomic_12054
private theorem within_094_018_agreement : RowAgreement 12052 12054 := RowAgreement.join atomic_12053_agreement atomic_12054_agreement
private def within_094_019 : CountSegment 1 10 13120 12054 12056 4876 4876 := CountSegment.join atomic_12055 atomic_12056
private theorem within_094_019_agreement : RowAgreement 12054 12056 := RowAgreement.join atomic_12055_agreement atomic_12056_agreement
private def within_094_020 : CountSegment 1 10 13120 12052 12056 4875 4876 := CountSegment.join within_094_018 within_094_019
private theorem within_094_020_agreement : RowAgreement 12052 12056 := RowAgreement.join within_094_018_agreement within_094_019_agreement
private def within_094_021 : CountSegment 1 10 13120 12048 12056 4873 4876 := CountSegment.join within_094_017 within_094_020
private theorem within_094_021_agreement : RowAgreement 12048 12056 := RowAgreement.join within_094_017_agreement within_094_020_agreement
private def within_094_022 : CountSegment 1 10 13120 12056 12058 4876 4877 := CountSegment.join atomic_12057 atomic_12058
private theorem within_094_022_agreement : RowAgreement 12056 12058 := RowAgreement.join atomic_12057_agreement atomic_12058_agreement
private def within_094_023 : CountSegment 1 10 13120 12058 12060 4877 4877 := CountSegment.join atomic_12059 atomic_12060
private theorem within_094_023_agreement : RowAgreement 12058 12060 := RowAgreement.join atomic_12059_agreement atomic_12060_agreement
private def within_094_024 : CountSegment 1 10 13120 12056 12060 4876 4877 := CountSegment.join within_094_022 within_094_023
private theorem within_094_024_agreement : RowAgreement 12056 12060 := RowAgreement.join within_094_022_agreement within_094_023_agreement
private def within_094_025 : CountSegment 1 10 13120 12060 12062 4877 4878 := CountSegment.join atomic_12061 atomic_12062
private theorem within_094_025_agreement : RowAgreement 12060 12062 := RowAgreement.join atomic_12061_agreement atomic_12062_agreement
private def within_094_026 : CountSegment 1 10 13120 12062 12064 4878 4880 := CountSegment.join atomic_12063 atomic_12064
private theorem within_094_026_agreement : RowAgreement 12062 12064 := RowAgreement.join atomic_12063_agreement atomic_12064_agreement
private def within_094_027 : CountSegment 1 10 13120 12060 12064 4877 4880 := CountSegment.join within_094_025 within_094_026
private theorem within_094_027_agreement : RowAgreement 12060 12064 := RowAgreement.join within_094_025_agreement within_094_026_agreement
private def within_094_028 : CountSegment 1 10 13120 12056 12064 4876 4880 := CountSegment.join within_094_024 within_094_027
private theorem within_094_028_agreement : RowAgreement 12056 12064 := RowAgreement.join within_094_024_agreement within_094_027_agreement
private def within_094_029 : CountSegment 1 10 13120 12048 12064 4873 4880 := CountSegment.join within_094_021 within_094_028
private theorem within_094_029_agreement : RowAgreement 12048 12064 := RowAgreement.join within_094_021_agreement within_094_028_agreement
private def within_094_030 : CountSegment 1 10 13120 12032 12064 4866 4880 := CountSegment.join within_094_014 within_094_029
private theorem within_094_030_agreement : RowAgreement 12032 12064 := RowAgreement.join within_094_014_agreement within_094_029_agreement
private def within_094_031 : CountSegment 1 10 13120 12064 12066 4880 4880 := CountSegment.join atomic_12065 atomic_12066
private theorem within_094_031_agreement : RowAgreement 12064 12066 := RowAgreement.join atomic_12065_agreement atomic_12066_agreement
private def within_094_032 : CountSegment 1 10 13120 12066 12068 4880 4881 := CountSegment.join atomic_12067 atomic_12068
private theorem within_094_032_agreement : RowAgreement 12066 12068 := RowAgreement.join atomic_12067_agreement atomic_12068_agreement
private def within_094_033 : CountSegment 1 10 13120 12064 12068 4880 4881 := CountSegment.join within_094_031 within_094_032
private theorem within_094_033_agreement : RowAgreement 12064 12068 := RowAgreement.join within_094_031_agreement within_094_032_agreement
private def within_094_034 : CountSegment 1 10 13120 12068 12070 4881 4881 := CountSegment.join atomic_12069 atomic_12070
private theorem within_094_034_agreement : RowAgreement 12068 12070 := RowAgreement.join atomic_12069_agreement atomic_12070_agreement
private def within_094_035 : CountSegment 1 10 13120 12070 12072 4881 4881 := CountSegment.join atomic_12071 atomic_12072
private theorem within_094_035_agreement : RowAgreement 12070 12072 := RowAgreement.join atomic_12071_agreement atomic_12072_agreement
private def within_094_036 : CountSegment 1 10 13120 12068 12072 4881 4881 := CountSegment.join within_094_034 within_094_035
private theorem within_094_036_agreement : RowAgreement 12068 12072 := RowAgreement.join within_094_034_agreement within_094_035_agreement
private def within_094_037 : CountSegment 1 10 13120 12064 12072 4880 4881 := CountSegment.join within_094_033 within_094_036
private theorem within_094_037_agreement : RowAgreement 12064 12072 := RowAgreement.join within_094_033_agreement within_094_036_agreement
private def within_094_038 : CountSegment 1 10 13120 12072 12074 4881 4882 := CountSegment.join atomic_12073 atomic_12074
private theorem within_094_038_agreement : RowAgreement 12072 12074 := RowAgreement.join atomic_12073_agreement atomic_12074_agreement
private def within_094_039 : CountSegment 1 10 13120 12074 12076 4882 4882 := CountSegment.join atomic_12075 atomic_12076
private theorem within_094_039_agreement : RowAgreement 12074 12076 := RowAgreement.join atomic_12075_agreement atomic_12076_agreement
private def within_094_040 : CountSegment 1 10 13120 12072 12076 4881 4882 := CountSegment.join within_094_038 within_094_039
private theorem within_094_040_agreement : RowAgreement 12072 12076 := RowAgreement.join within_094_038_agreement within_094_039_agreement
private def within_094_041 : CountSegment 1 10 13120 12076 12078 4882 4885 := CountSegment.join atomic_12077 atomic_12078
private theorem within_094_041_agreement : RowAgreement 12076 12078 := RowAgreement.join atomic_12077_agreement atomic_12078_agreement
private def within_094_042 : CountSegment 1 10 13120 12078 12080 4885 4885 := CountSegment.join atomic_12079 atomic_12080
private theorem within_094_042_agreement : RowAgreement 12078 12080 := RowAgreement.join atomic_12079_agreement atomic_12080_agreement
private def within_094_043 : CountSegment 1 10 13120 12076 12080 4882 4885 := CountSegment.join within_094_041 within_094_042
private theorem within_094_043_agreement : RowAgreement 12076 12080 := RowAgreement.join within_094_041_agreement within_094_042_agreement
private def within_094_044 : CountSegment 1 10 13120 12072 12080 4881 4885 := CountSegment.join within_094_040 within_094_043
private theorem within_094_044_agreement : RowAgreement 12072 12080 := RowAgreement.join within_094_040_agreement within_094_043_agreement
private def within_094_045 : CountSegment 1 10 13120 12064 12080 4880 4885 := CountSegment.join within_094_037 within_094_044
private theorem within_094_045_agreement : RowAgreement 12064 12080 := RowAgreement.join within_094_037_agreement within_094_044_agreement
private def within_094_046 : CountSegment 1 10 13120 12080 12082 4885 4887 := CountSegment.join atomic_12081 atomic_12082
private theorem within_094_046_agreement : RowAgreement 12080 12082 := RowAgreement.join atomic_12081_agreement atomic_12082_agreement
private def within_094_047 : CountSegment 1 10 13120 12082 12084 4887 4888 := CountSegment.join atomic_12083 atomic_12084
private theorem within_094_047_agreement : RowAgreement 12082 12084 := RowAgreement.join atomic_12083_agreement atomic_12084_agreement
private def within_094_048 : CountSegment 1 10 13120 12080 12084 4885 4888 := CountSegment.join within_094_046 within_094_047
private theorem within_094_048_agreement : RowAgreement 12080 12084 := RowAgreement.join within_094_046_agreement within_094_047_agreement
private def within_094_049 : CountSegment 1 10 13120 12084 12086 4888 4888 := CountSegment.join atomic_12085 atomic_12086
private theorem within_094_049_agreement : RowAgreement 12084 12086 := RowAgreement.join atomic_12085_agreement atomic_12086_agreement
private def within_094_050 : CountSegment 1 10 13120 12086 12088 4888 4889 := CountSegment.join atomic_12087 atomic_12088
private theorem within_094_050_agreement : RowAgreement 12086 12088 := RowAgreement.join atomic_12087_agreement atomic_12088_agreement
private def within_094_051 : CountSegment 1 10 13120 12084 12088 4888 4889 := CountSegment.join within_094_049 within_094_050
private theorem within_094_051_agreement : RowAgreement 12084 12088 := RowAgreement.join within_094_049_agreement within_094_050_agreement
private def within_094_052 : CountSegment 1 10 13120 12080 12088 4885 4889 := CountSegment.join within_094_048 within_094_051
private theorem within_094_052_agreement : RowAgreement 12080 12088 := RowAgreement.join within_094_048_agreement within_094_051_agreement
private def within_094_053 : CountSegment 1 10 13120 12088 12090 4889 4889 := CountSegment.join atomic_12089 atomic_12090
private theorem within_094_053_agreement : RowAgreement 12088 12090 := RowAgreement.join atomic_12089_agreement atomic_12090_agreement
private def within_094_054 : CountSegment 1 10 13120 12090 12092 4889 4890 := CountSegment.join atomic_12091 atomic_12092
private theorem within_094_054_agreement : RowAgreement 12090 12092 := RowAgreement.join atomic_12091_agreement atomic_12092_agreement
private def within_094_055 : CountSegment 1 10 13120 12088 12092 4889 4890 := CountSegment.join within_094_053 within_094_054
private theorem within_094_055_agreement : RowAgreement 12088 12092 := RowAgreement.join within_094_053_agreement within_094_054_agreement
private def within_094_056 : CountSegment 1 10 13120 12092 12094 4890 4891 := CountSegment.join atomic_12093 atomic_12094
private theorem within_094_056_agreement : RowAgreement 12092 12094 := RowAgreement.join atomic_12093_agreement atomic_12094_agreement
private def within_094_057 : CountSegment 1 10 13120 12094 12096 4891 4892 := CountSegment.join atomic_12095 atomic_12096
private theorem within_094_057_agreement : RowAgreement 12094 12096 := RowAgreement.join atomic_12095_agreement atomic_12096_agreement
private def within_094_058 : CountSegment 1 10 13120 12092 12096 4890 4892 := CountSegment.join within_094_056 within_094_057
private theorem within_094_058_agreement : RowAgreement 12092 12096 := RowAgreement.join within_094_056_agreement within_094_057_agreement
private def within_094_059 : CountSegment 1 10 13120 12088 12096 4889 4892 := CountSegment.join within_094_055 within_094_058
private theorem within_094_059_agreement : RowAgreement 12088 12096 := RowAgreement.join within_094_055_agreement within_094_058_agreement
private def within_094_060 : CountSegment 1 10 13120 12080 12096 4885 4892 := CountSegment.join within_094_052 within_094_059
private theorem within_094_060_agreement : RowAgreement 12080 12096 := RowAgreement.join within_094_052_agreement within_094_059_agreement
private def within_094_061 : CountSegment 1 10 13120 12064 12096 4880 4892 := CountSegment.join within_094_045 within_094_060
private theorem within_094_061_agreement : RowAgreement 12064 12096 := RowAgreement.join within_094_045_agreement within_094_060_agreement
private def within_094_062 : CountSegment 1 10 13120 12032 12096 4866 4892 := CountSegment.join within_094_030 within_094_061
private theorem within_094_062_agreement : RowAgreement 12032 12096 := RowAgreement.join within_094_030_agreement within_094_061_agreement
private def within_094_063 : CountSegment 1 10 13120 12096 12098 4892 4894 := CountSegment.join atomic_12097 atomic_12098
private theorem within_094_063_agreement : RowAgreement 12096 12098 := RowAgreement.join atomic_12097_agreement atomic_12098_agreement
private def within_094_064 : CountSegment 1 10 13120 12098 12100 4894 4895 := CountSegment.join atomic_12099 atomic_12100
private theorem within_094_064_agreement : RowAgreement 12098 12100 := RowAgreement.join atomic_12099_agreement atomic_12100_agreement
private def within_094_065 : CountSegment 1 10 13120 12096 12100 4892 4895 := CountSegment.join within_094_063 within_094_064
private theorem within_094_065_agreement : RowAgreement 12096 12100 := RowAgreement.join within_094_063_agreement within_094_064_agreement
private def within_094_066 : CountSegment 1 10 13120 12100 12102 4895 4895 := CountSegment.join atomic_12101 atomic_12102
private theorem within_094_066_agreement : RowAgreement 12100 12102 := RowAgreement.join atomic_12101_agreement atomic_12102_agreement
private def within_094_067 : CountSegment 1 10 13120 12102 12104 4895 4896 := CountSegment.join atomic_12103 atomic_12104
private theorem within_094_067_agreement : RowAgreement 12102 12104 := RowAgreement.join atomic_12103_agreement atomic_12104_agreement
private def within_094_068 : CountSegment 1 10 13120 12100 12104 4895 4896 := CountSegment.join within_094_066 within_094_067
private theorem within_094_068_agreement : RowAgreement 12100 12104 := RowAgreement.join within_094_066_agreement within_094_067_agreement
private def within_094_069 : CountSegment 1 10 13120 12096 12104 4892 4896 := CountSegment.join within_094_065 within_094_068
private theorem within_094_069_agreement : RowAgreement 12096 12104 := RowAgreement.join within_094_065_agreement within_094_068_agreement
private def within_094_070 : CountSegment 1 10 13120 12104 12106 4896 4897 := CountSegment.join atomic_12105 atomic_12106
private theorem within_094_070_agreement : RowAgreement 12104 12106 := RowAgreement.join atomic_12105_agreement atomic_12106_agreement
private def within_094_071 : CountSegment 1 10 13120 12106 12108 4897 4897 := CountSegment.join atomic_12107 atomic_12108
private theorem within_094_071_agreement : RowAgreement 12106 12108 := RowAgreement.join atomic_12107_agreement atomic_12108_agreement
private def within_094_072 : CountSegment 1 10 13120 12104 12108 4896 4897 := CountSegment.join within_094_070 within_094_071
private theorem within_094_072_agreement : RowAgreement 12104 12108 := RowAgreement.join within_094_070_agreement within_094_071_agreement
private def within_094_073 : CountSegment 1 10 13120 12108 12110 4897 4899 := CountSegment.join atomic_12109 atomic_12110
private theorem within_094_073_agreement : RowAgreement 12108 12110 := RowAgreement.join atomic_12109_agreement atomic_12110_agreement
private def within_094_074 : CountSegment 1 10 13120 12110 12112 4899 4900 := CountSegment.join atomic_12111 atomic_12112
private theorem within_094_074_agreement : RowAgreement 12110 12112 := RowAgreement.join atomic_12111_agreement atomic_12112_agreement
private def within_094_075 : CountSegment 1 10 13120 12108 12112 4897 4900 := CountSegment.join within_094_073 within_094_074
private theorem within_094_075_agreement : RowAgreement 12108 12112 := RowAgreement.join within_094_073_agreement within_094_074_agreement
private def within_094_076 : CountSegment 1 10 13120 12104 12112 4896 4900 := CountSegment.join within_094_072 within_094_075
private theorem within_094_076_agreement : RowAgreement 12104 12112 := RowAgreement.join within_094_072_agreement within_094_075_agreement
private def within_094_077 : CountSegment 1 10 13120 12096 12112 4892 4900 := CountSegment.join within_094_069 within_094_076
private theorem within_094_077_agreement : RowAgreement 12096 12112 := RowAgreement.join within_094_069_agreement within_094_076_agreement
private def within_094_078 : CountSegment 1 10 13120 12112 12114 4900 4901 := CountSegment.join atomic_12113 atomic_12114
private theorem within_094_078_agreement : RowAgreement 12112 12114 := RowAgreement.join atomic_12113_agreement atomic_12114_agreement
private def within_094_079 : CountSegment 1 10 13120 12114 12116 4901 4901 := CountSegment.join atomic_12115 atomic_12116
private theorem within_094_079_agreement : RowAgreement 12114 12116 := RowAgreement.join atomic_12115_agreement atomic_12116_agreement
private def within_094_080 : CountSegment 1 10 13120 12112 12116 4900 4901 := CountSegment.join within_094_078 within_094_079
private theorem within_094_080_agreement : RowAgreement 12112 12116 := RowAgreement.join within_094_078_agreement within_094_079_agreement
private def within_094_081 : CountSegment 1 10 13120 12116 12118 4901 4902 := CountSegment.join atomic_12117 atomic_12118
private theorem within_094_081_agreement : RowAgreement 12116 12118 := RowAgreement.join atomic_12117_agreement atomic_12118_agreement
private def within_094_082 : CountSegment 1 10 13120 12118 12120 4902 4904 := CountSegment.join atomic_12119 atomic_12120
private theorem within_094_082_agreement : RowAgreement 12118 12120 := RowAgreement.join atomic_12119_agreement atomic_12120_agreement
private def within_094_083 : CountSegment 1 10 13120 12116 12120 4901 4904 := CountSegment.join within_094_081 within_094_082
private theorem within_094_083_agreement : RowAgreement 12116 12120 := RowAgreement.join within_094_081_agreement within_094_082_agreement
private def within_094_084 : CountSegment 1 10 13120 12112 12120 4900 4904 := CountSegment.join within_094_080 within_094_083
private theorem within_094_084_agreement : RowAgreement 12112 12120 := RowAgreement.join within_094_080_agreement within_094_083_agreement
private def within_094_085 : CountSegment 1 10 13120 12120 12122 4904 4905 := CountSegment.join atomic_12121 atomic_12122
private theorem within_094_085_agreement : RowAgreement 12120 12122 := RowAgreement.join atomic_12121_agreement atomic_12122_agreement
private def within_094_086 : CountSegment 1 10 13120 12122 12124 4905 4906 := CountSegment.join atomic_12123 atomic_12124
private theorem within_094_086_agreement : RowAgreement 12122 12124 := RowAgreement.join atomic_12123_agreement atomic_12124_agreement
private def within_094_087 : CountSegment 1 10 13120 12120 12124 4904 4906 := CountSegment.join within_094_085 within_094_086
private theorem within_094_087_agreement : RowAgreement 12120 12124 := RowAgreement.join within_094_085_agreement within_094_086_agreement
private def within_094_088 : CountSegment 1 10 13120 12124 12126 4906 4907 := CountSegment.join atomic_12125 atomic_12126
private theorem within_094_088_agreement : RowAgreement 12124 12126 := RowAgreement.join atomic_12125_agreement atomic_12126_agreement
private def within_094_089 : CountSegment 1 10 13120 12126 12128 4907 4907 := CountSegment.join atomic_12127 atomic_12128
private theorem within_094_089_agreement : RowAgreement 12126 12128 := RowAgreement.join atomic_12127_agreement atomic_12128_agreement
private def within_094_090 : CountSegment 1 10 13120 12124 12128 4906 4907 := CountSegment.join within_094_088 within_094_089
private theorem within_094_090_agreement : RowAgreement 12124 12128 := RowAgreement.join within_094_088_agreement within_094_089_agreement
private def within_094_091 : CountSegment 1 10 13120 12120 12128 4904 4907 := CountSegment.join within_094_087 within_094_090
private theorem within_094_091_agreement : RowAgreement 12120 12128 := RowAgreement.join within_094_087_agreement within_094_090_agreement
private def within_094_092 : CountSegment 1 10 13120 12112 12128 4900 4907 := CountSegment.join within_094_084 within_094_091
private theorem within_094_092_agreement : RowAgreement 12112 12128 := RowAgreement.join within_094_084_agreement within_094_091_agreement
private def within_094_093 : CountSegment 1 10 13120 12096 12128 4892 4907 := CountSegment.join within_094_077 within_094_092
private theorem within_094_093_agreement : RowAgreement 12096 12128 := RowAgreement.join within_094_077_agreement within_094_092_agreement
private def within_094_094 : CountSegment 1 10 13120 12128 12130 4907 4909 := CountSegment.join atomic_12129 atomic_12130
private theorem within_094_094_agreement : RowAgreement 12128 12130 := RowAgreement.join atomic_12129_agreement atomic_12130_agreement
private def within_094_095 : CountSegment 1 10 13120 12130 12132 4909 4910 := CountSegment.join atomic_12131 atomic_12132
private theorem within_094_095_agreement : RowAgreement 12130 12132 := RowAgreement.join atomic_12131_agreement atomic_12132_agreement
private def within_094_096 : CountSegment 1 10 13120 12128 12132 4907 4910 := CountSegment.join within_094_094 within_094_095
private theorem within_094_096_agreement : RowAgreement 12128 12132 := RowAgreement.join within_094_094_agreement within_094_095_agreement
private def within_094_097 : CountSegment 1 10 13120 12132 12134 4910 4910 := CountSegment.join atomic_12133 atomic_12134
private theorem within_094_097_agreement : RowAgreement 12132 12134 := RowAgreement.join atomic_12133_agreement atomic_12134_agreement
private def within_094_098 : CountSegment 1 10 13120 12134 12136 4910 4912 := CountSegment.join atomic_12135 atomic_12136
private theorem within_094_098_agreement : RowAgreement 12134 12136 := RowAgreement.join atomic_12135_agreement atomic_12136_agreement
private def within_094_099 : CountSegment 1 10 13120 12132 12136 4910 4912 := CountSegment.join within_094_097 within_094_098
private theorem within_094_099_agreement : RowAgreement 12132 12136 := RowAgreement.join within_094_097_agreement within_094_098_agreement
private def within_094_100 : CountSegment 1 10 13120 12128 12136 4907 4912 := CountSegment.join within_094_096 within_094_099
private theorem within_094_100_agreement : RowAgreement 12128 12136 := RowAgreement.join within_094_096_agreement within_094_099_agreement
private def within_094_101 : CountSegment 1 10 13120 12136 12138 4912 4913 := CountSegment.join atomic_12137 atomic_12138
private theorem within_094_101_agreement : RowAgreement 12136 12138 := RowAgreement.join atomic_12137_agreement atomic_12138_agreement
private def within_094_102 : CountSegment 1 10 13120 12138 12140 4913 4913 := CountSegment.join atomic_12139 atomic_12140
private theorem within_094_102_agreement : RowAgreement 12138 12140 := RowAgreement.join atomic_12139_agreement atomic_12140_agreement
private def within_094_103 : CountSegment 1 10 13120 12136 12140 4912 4913 := CountSegment.join within_094_101 within_094_102
private theorem within_094_103_agreement : RowAgreement 12136 12140 := RowAgreement.join within_094_101_agreement within_094_102_agreement
private def within_094_104 : CountSegment 1 10 13120 12140 12142 4913 4915 := CountSegment.join atomic_12141 atomic_12142
private theorem within_094_104_agreement : RowAgreement 12140 12142 := RowAgreement.join atomic_12141_agreement atomic_12142_agreement
private def within_094_105 : CountSegment 1 10 13120 12142 12144 4915 4915 := CountSegment.join atomic_12143 atomic_12144
private theorem within_094_105_agreement : RowAgreement 12142 12144 := RowAgreement.join atomic_12143_agreement atomic_12144_agreement
private def within_094_106 : CountSegment 1 10 13120 12140 12144 4913 4915 := CountSegment.join within_094_104 within_094_105
private theorem within_094_106_agreement : RowAgreement 12140 12144 := RowAgreement.join within_094_104_agreement within_094_105_agreement
private def within_094_107 : CountSegment 1 10 13120 12136 12144 4912 4915 := CountSegment.join within_094_103 within_094_106
private theorem within_094_107_agreement : RowAgreement 12136 12144 := RowAgreement.join within_094_103_agreement within_094_106_agreement
private def within_094_108 : CountSegment 1 10 13120 12128 12144 4907 4915 := CountSegment.join within_094_100 within_094_107
private theorem within_094_108_agreement : RowAgreement 12128 12144 := RowAgreement.join within_094_100_agreement within_094_107_agreement
private def within_094_109 : CountSegment 1 10 13120 12144 12146 4915 4916 := CountSegment.join atomic_12145 atomic_12146
private theorem within_094_109_agreement : RowAgreement 12144 12146 := RowAgreement.join atomic_12145_agreement atomic_12146_agreement
private def within_094_110 : CountSegment 1 10 13120 12146 12148 4916 4917 := CountSegment.join atomic_12147 atomic_12148
private theorem within_094_110_agreement : RowAgreement 12146 12148 := RowAgreement.join atomic_12147_agreement atomic_12148_agreement
private def within_094_111 : CountSegment 1 10 13120 12144 12148 4915 4917 := CountSegment.join within_094_109 within_094_110
private theorem within_094_111_agreement : RowAgreement 12144 12148 := RowAgreement.join within_094_109_agreement within_094_110_agreement
private def within_094_112 : CountSegment 1 10 13120 12148 12150 4917 4918 := CountSegment.join atomic_12149 atomic_12150
private theorem within_094_112_agreement : RowAgreement 12148 12150 := RowAgreement.join atomic_12149_agreement atomic_12150_agreement
private def within_094_113 : CountSegment 1 10 13120 12150 12152 4918 4919 := CountSegment.join atomic_12151 atomic_12152
private theorem within_094_113_agreement : RowAgreement 12150 12152 := RowAgreement.join atomic_12151_agreement atomic_12152_agreement
private def within_094_114 : CountSegment 1 10 13120 12148 12152 4917 4919 := CountSegment.join within_094_112 within_094_113
private theorem within_094_114_agreement : RowAgreement 12148 12152 := RowAgreement.join within_094_112_agreement within_094_113_agreement
private def within_094_115 : CountSegment 1 10 13120 12144 12152 4915 4919 := CountSegment.join within_094_111 within_094_114
private theorem within_094_115_agreement : RowAgreement 12144 12152 := RowAgreement.join within_094_111_agreement within_094_114_agreement
private def within_094_116 : CountSegment 1 10 13120 12152 12154 4919 4919 := CountSegment.join atomic_12153 atomic_12154
private theorem within_094_116_agreement : RowAgreement 12152 12154 := RowAgreement.join atomic_12153_agreement atomic_12154_agreement
private def within_094_117 : CountSegment 1 10 13120 12154 12156 4919 4919 := CountSegment.join atomic_12155 atomic_12156
private theorem within_094_117_agreement : RowAgreement 12154 12156 := RowAgreement.join atomic_12155_agreement atomic_12156_agreement
private def within_094_118 : CountSegment 1 10 13120 12152 12156 4919 4919 := CountSegment.join within_094_116 within_094_117
private theorem within_094_118_agreement : RowAgreement 12152 12156 := RowAgreement.join within_094_116_agreement within_094_117_agreement
private def within_094_119 : CountSegment 1 10 13120 12156 12158 4919 4919 := CountSegment.join atomic_12157 atomic_12158
private theorem within_094_119_agreement : RowAgreement 12156 12158 := RowAgreement.join atomic_12157_agreement atomic_12158_agreement
private def within_094_120 : CountSegment 1 10 13120 12158 12160 4919 4919 := CountSegment.join atomic_12159 atomic_12160
private theorem within_094_120_agreement : RowAgreement 12158 12160 := RowAgreement.join atomic_12159_agreement atomic_12160_agreement
private def within_094_121 : CountSegment 1 10 13120 12156 12160 4919 4919 := CountSegment.join within_094_119 within_094_120
private theorem within_094_121_agreement : RowAgreement 12156 12160 := RowAgreement.join within_094_119_agreement within_094_120_agreement
private def within_094_122 : CountSegment 1 10 13120 12152 12160 4919 4919 := CountSegment.join within_094_118 within_094_121
private theorem within_094_122_agreement : RowAgreement 12152 12160 := RowAgreement.join within_094_118_agreement within_094_121_agreement
private def within_094_123 : CountSegment 1 10 13120 12144 12160 4915 4919 := CountSegment.join within_094_115 within_094_122
private theorem within_094_123_agreement : RowAgreement 12144 12160 := RowAgreement.join within_094_115_agreement within_094_122_agreement
private def within_094_124 : CountSegment 1 10 13120 12128 12160 4907 4919 := CountSegment.join within_094_108 within_094_123
private theorem within_094_124_agreement : RowAgreement 12128 12160 := RowAgreement.join within_094_108_agreement within_094_123_agreement
private def within_094_125 : CountSegment 1 10 13120 12096 12160 4892 4919 := CountSegment.join within_094_093 within_094_124
private theorem within_094_125_agreement : RowAgreement 12096 12160 := RowAgreement.join within_094_093_agreement within_094_124_agreement
private def within_094_126 : CountSegment 1 10 13120 12032 12160 4866 4919 := CountSegment.join within_094_062 within_094_125
private theorem within_094_126_agreement : RowAgreement 12032 12160 := RowAgreement.join within_094_062_agreement within_094_125_agreement
def segment_094 : CountSegment 1 10 13120 12032 12160 4866 4919 := within_094_126
theorem agreement_094 : RowAgreement 12032 12160 := within_094_126_agreement
end ProjectEulerSolutions.P58Numerical
