import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_4865 : RowCertificate 4865 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4865 1) := by change ¬ Nat.Prime 94682631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31560877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4865 2) := by change ¬ Nat.Prime 94672901; exact PrattCertificates.composite_of_product (factor := 1193) (cofactor := 79357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4865 3) := by change ¬ Nat.Prime 94663171; exact PrattCertificates.composite_of_product (factor := 7417) (cofactor := 12763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4865 : row 4865 = 0 := (row_4865.checked).trans rfl
theorem model_count_4865 : modelRow 4865 = 0 := by decide
theorem modelrow_agrees_4865 : modelRow 4865 = row 4865 := model_count_4865.trans row_count_4865.symm
def row_4866 : RowCertificate 4866 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4866 1) := by change ¬ Nat.Prime 94721557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13531651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4866 2) := by change ¬ Nat.Prime 94711825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 18942365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4866 3) := by change ¬ Nat.Prime 94702093; exact PrattCertificates.composite_of_product (factor := 4831) (cofactor := 19603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4866 : row 4866 = 0 := (row_4866.checked).trans rfl
theorem model_count_4866 : modelRow 4866 = 0 := by decide
theorem modelrow_agrees_4866 : modelRow 4866 = row 4866 := model_count_4866.trans row_count_4866.symm
def row_4867 : RowCertificate 4867 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4867 1) := by change ¬ Nat.Prime 94760491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13537213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4867 2) := by change ¬ Nat.Prime 94750757; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 869273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4867 3) := by change ¬ Nat.Prime 94741023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31580341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4867 : row 4867 = 0 := (row_4867.checked).trans rfl
theorem model_count_4867 : modelRow 4867 = 0 := by decide
theorem modelrow_agrees_4867 : modelRow 4867 = row 4867 := model_count_4867.trans row_count_4867.symm
def row_4868 : RowCertificate 4868 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4868 1) := by change ¬ Nat.Prime 94799433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31599811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4868 2) := by change ¬ Nat.Prime 94789697; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1298489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4868 3) := by change ¬ Nat.Prime 94779961; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 4988419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4868 : row 4868 = 0 := (row_4868.checked).trans rfl
theorem model_count_4868 : modelRow 4868 = 0 := by decide
theorem modelrow_agrees_4868 : modelRow 4868 = row 4868 := model_count_4868.trans row_count_4868.symm
def row_4869 : RowCertificate 4869 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4869 1) := by change ¬ Nat.Prime 94838383; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 920761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4869 2) := by change ¬ Nat.Prime 94828645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 18965729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4869 3) := by change ¬ Nat.Prime 94818907; exact PrattCertificates.composite_of_product (factor := 6421) (cofactor := 14767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4869 : row 4869 = 0 := (row_4869.checked).trans rfl
theorem model_count_4869 : modelRow 4869 = 0 := by decide
theorem modelrow_agrees_4869 : modelRow 4869 = row 4869 := model_count_4869.trans row_count_4869.symm
def row_4870 : RowCertificate 4870 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4870 1) := by change ¬ Nat.Prime 94877341; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7298257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4870 2) := by change ¬ Nat.Prime 94867601; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 414269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4870 3) := by change ¬ Nat.Prime 94857861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31619287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4870 : row 4870 = 0 := (row_4870.checked).trans rfl
theorem model_count_4870 : modelRow 4870 = 0 := by decide
theorem modelrow_agrees_4870 : modelRow 4870 = row 4870 := model_count_4870.trans row_count_4870.symm
def row_4871 : RowCertificate 4871 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4871 1) := by change ¬ Nat.Prime 94916307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31638769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4871 2) := by change ¬ Nat.Prime 94906565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 18981313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4871 3) := by change ¬ Nat.Prime 94896823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13556689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4871 : row 4871 = 0 := (row_4871.checked).trans rfl
theorem model_count_4871 : modelRow 4871 = 0 := by decide
theorem modelrow_agrees_4871 : modelRow 4871 = row 4871 := model_count_4871.trans row_count_4871.symm
def row_4872 : RowCertificate 4872 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4872 1) := by change ¬ Nat.Prime 94955281; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1417243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4872 2) := by change Nat.Prime 94945537; exact ProjectEulerSolutions.P58PrimeNodes.p_94945537.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4872 3) := by change Nat.Prime 94935793; exact ProjectEulerSolutions.P58PrimeNodes.p_94935793.prime
    exact decide_eq_true prime
theorem row_count_4872 : row 4872 = 2 := (row_4872.checked).trans rfl
theorem model_count_4872 : modelRow 4872 = 2 := by decide
theorem modelrow_agrees_4872 : modelRow 4872 = row 4872 := model_count_4872.trans row_count_4872.symm
def row_4873 : RowCertificate 4873 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4873 1) := by change ¬ Nat.Prime 94994263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13570609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4873 2) := by change ¬ Nat.Prime 94984517; exact PrattCertificates.composite_of_product (factor := 4273) (cofactor := 22229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4873 3) := by change ¬ Nat.Prime 94974771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31658257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4873 : row 4873 = 0 := (row_4873.checked).trans rfl
theorem model_count_4873 : modelRow 4873 = 0 := by decide
theorem modelrow_agrees_4873 : modelRow 4873 = row 4873 := model_count_4873.trans row_count_4873.symm
def row_4874 : RowCertificate 4874 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4874 1) := by change ¬ Nat.Prime 95033253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31677751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4874 2) := by change ¬ Nat.Prime 95023505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19004701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4874 3) := by change Nat.Prime 95013757; exact ProjectEulerSolutions.P58PrimeNodes.p_95013757.prime
    exact decide_eq_true prime
theorem row_count_4874 : row 4874 = 1 := (row_4874.checked).trans rfl
theorem model_count_4874 : modelRow 4874 = 1 := by decide
theorem modelrow_agrees_4874 : modelRow 4874 = row 4874 := model_count_4874.trans row_count_4874.symm
def row_4875 : RowCertificate 4875 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4875 1) := by change Nat.Prime 95072251; exact ProjectEulerSolutions.P58PrimeNodes.p_95072251.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4875 2) := by change ¬ Nat.Prime 95062501; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 369893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4875 3) := by change Nat.Prime 95052751; exact ProjectEulerSolutions.P58PrimeNodes.p_95052751.prime
    exact decide_eq_true prime
theorem row_count_4875 : row 4875 = 2 := (row_4875.checked).trans rfl
theorem model_count_4875 : modelRow 4875 = 2 := by decide
theorem modelrow_agrees_4875 : modelRow 4875 = row 4875 := model_count_4875.trans row_count_4875.symm
def row_4876 : RowCertificate 4876 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4876 1) := by change ¬ Nat.Prime 95111257; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1419571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4876 2) := by change ¬ Nat.Prime 95101505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19020301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4876 3) := by change ¬ Nat.Prime 95091753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31697251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4876 : row 4876 = 0 := (row_4876.checked).trans rfl
theorem model_count_4876 : modelRow 4876 = 0 := by decide
theorem modelrow_agrees_4876 : modelRow 4876 = row 4876 := model_count_4876.trans row_count_4876.symm
def row_4877 : RowCertificate 4877 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4877 1) := by change ¬ Nat.Prime 95150271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31716757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4877 2) := by change ¬ Nat.Prime 95140517; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5596501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4877 3) := by change ¬ Nat.Prime 95130763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13590109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4877 : row 4877 = 0 := (row_4877.checked).trans rfl
theorem model_count_4877 : modelRow 4877 = 0 := by decide
theorem modelrow_agrees_4877 : modelRow 4877 = row 4877 := model_count_4877.trans row_count_4877.symm
def row_4878 : RowCertificate 4878 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4878 1) := by change Nat.Prime 95189293; exact ProjectEulerSolutions.P58PrimeNodes.p_95189293.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4878 2) := by change ¬ Nat.Prime 95179537; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3282053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4878 3) := by change ¬ Nat.Prime 95169781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13595683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4878 : row 4878 = 1 := (row_4878.checked).trans rfl
theorem model_count_4878 : modelRow 4878 = 1 := by decide
theorem modelrow_agrees_4878 : modelRow 4878 = row 4878 := model_count_4878.trans row_count_4878.symm
def row_4879 : RowCertificate 4879 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4879 1) := by change ¬ Nat.Prime 95228323; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5012017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4879 2) := by change ¬ Nat.Prime 95218565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19043713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4879 3) := by change ¬ Nat.Prime 95208807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31736269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4879 : row 4879 = 0 := (row_4879.checked).trans rfl
theorem model_count_4879 : modelRow 4879 = 0 := by decide
theorem modelrow_agrees_4879 : modelRow 4879 = row 4879 := model_count_4879.trans row_count_4879.symm
def row_4880 : RowCertificate 4880 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4880 1) := by change ¬ Nat.Prime 95267361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31755787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4880 2) := by change Nat.Prime 95257601; exact ProjectEulerSolutions.P58PrimeNodes.p_95257601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4880 3) := by change ¬ Nat.Prime 95247841; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7326757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4880 : row 4880 = 1 := (row_4880.checked).trans rfl
theorem model_count_4880 : modelRow 4880 = 1 := by decide
theorem modelrow_agrees_4880 : modelRow 4880 = row 4880 := model_count_4880.trans row_count_4880.symm
def row_4881 : RowCertificate 4881 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4881 1) := by change ¬ Nat.Prime 95306407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13615201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4881 2) := by change ¬ Nat.Prime 95296645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19059329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4881 3) := by change ¬ Nat.Prime 95286883; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 982339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4881 : row 4881 = 0 := (row_4881.checked).trans rfl
theorem model_count_4881 : modelRow 4881 = 0 := by decide
theorem modelrow_agrees_4881 : modelRow 4881 = row 4881 := model_count_4881.trans row_count_4881.symm
def row_4882 : RowCertificate 4882 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4882 1) := by change ¬ Nat.Prime 95345461; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 874729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4882 2) := by change ¬ Nat.Prime 95335697; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 695881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4882 3) := by change ¬ Nat.Prime 95325933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31775311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4882 : row 4882 = 0 := (row_4882.checked).trans rfl
theorem model_count_4882 : modelRow 4882 = 0 := by decide
theorem modelrow_agrees_4882 : modelRow 4882 = row 4882 := model_count_4882.trans row_count_4882.symm
def row_4883 : RowCertificate 4883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4883 1) := by change ¬ Nat.Prime 95384523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31794841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4883 2) := by change ¬ Nat.Prime 95374757; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 354553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4883 3) := by change ¬ Nat.Prime 95364991; exact PrattCertificates.composite_of_product (factor := 1669) (cofactor := 57139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4883 : row 4883 = 0 := (row_4883.checked).trans rfl
theorem model_count_4883 : modelRow 4883 = 0 := by decide
theorem modelrow_agrees_4883 : modelRow 4883 = row 4883 := model_count_4883.trans row_count_4883.symm
def row_4884 : RowCertificate 4884 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4884 1) := by change Nat.Prime 95423593; exact ProjectEulerSolutions.P58PrimeNodes.p_95423593.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4884 2) := by change ¬ Nat.Prime 95413825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19082765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4884 3) := by change ¬ Nat.Prime 95404057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13629151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4884 : row 4884 = 1 := (row_4884.checked).trans rfl
theorem model_count_4884 : modelRow 4884 = 1 := by decide
theorem modelrow_agrees_4884 : modelRow 4884 = row 4884 := model_count_4884.trans row_count_4884.symm
def row_4885 : RowCertificate 4885 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4885 1) := by change ¬ Nat.Prime 95462671; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 3079441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4885 2) := by change Nat.Prime 95452901; exact ProjectEulerSolutions.P58PrimeNodes.p_95452901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4885 3) := by change ¬ Nat.Prime 95443131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31814377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4885 : row 4885 = 1 := (row_4885.checked).trans rfl
theorem model_count_4885 : modelRow 4885 = 1 := by decide
theorem modelrow_agrees_4885 : modelRow 4885 = row 4885 := model_count_4885.trans row_count_4885.symm
def row_4886 : RowCertificate 4886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4886 1) := by change ¬ Nat.Prime 95501757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31833919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4886 2) := by change ¬ Nat.Prime 95491985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19098397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4886 3) := by change ¬ Nat.Prime 95482213; exact PrattCertificates.composite_of_product (factor := 5407) (cofactor := 17659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4886 : row 4886 = 0 := (row_4886.checked).trans rfl
theorem model_count_4886 : modelRow 4886 = 0 := by decide
theorem modelrow_agrees_4886 : modelRow 4886 = row 4886 := model_count_4886.trans row_count_4886.symm
def row_4887 : RowCertificate 4887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4887 1) := by change ¬ Nat.Prime 95540851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13648693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4887 2) := by change ¬ Nat.Prime 95531077; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2581921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4887 3) := by change ¬ Nat.Prime 95521303; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5027437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4887 : row 4887 = 0 := (row_4887.checked).trans rfl
theorem model_count_4887 : modelRow 4887 = 0 := by decide
theorem modelrow_agrees_4887 : modelRow 4887 = row 4887 := model_count_4887.trans row_count_4887.symm
def row_4888 : RowCertificate 4888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4888 1) := by change ¬ Nat.Prime 95579953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13654279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4888 2) := by change ¬ Nat.Prime 95570177; exact PrattCertificates.composite_of_product (factor := 1889) (cofactor := 50593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4888 3) := by change ¬ Nat.Prime 95560401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31853467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4888 : row 4888 = 0 := (row_4888.checked).trans rfl
theorem model_count_4888 : modelRow 4888 = 0 := by decide
theorem modelrow_agrees_4888 : modelRow 4888 = row 4888 := model_count_4888.trans row_count_4888.symm
def row_4889 : RowCertificate 4889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4889 1) := by change ¬ Nat.Prime 95619063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31873021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4889 2) := by change ¬ Nat.Prime 95609285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19121857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4889 3) := by change ¬ Nat.Prime 95599507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5031553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4889 : row 4889 = 0 := (row_4889.checked).trans rfl
theorem model_count_4889 : modelRow 4889 = 0 := by decide
theorem modelrow_agrees_4889 : modelRow 4889 = row 4889 := model_count_4889.trans row_count_4889.symm
def row_4890 : RowCertificate 4890 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4890 1) := by change Nat.Prime 95658181; exact ProjectEulerSolutions.P58PrimeNodes.p_95658181.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4890 2) := by change ¬ Nat.Prime 95648401; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 220897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4890 3) := by change ¬ Nat.Prime 95638621; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7356817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4890 : row 4890 = 1 := (row_4890.checked).trans rfl
theorem model_count_4890 : modelRow 4890 = 1 := by decide
theorem modelrow_agrees_4890 : modelRow 4890 = row 4890 := model_count_4890.trans row_count_4890.symm
def row_4891 : RowCertificate 4891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4891 1) := by change ¬ Nat.Prime 95697307; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 633757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4891 2) := by change ¬ Nat.Prime 95687525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19137505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4891 3) := by change ¬ Nat.Prime 95677743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31892581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4891 : row 4891 = 0 := (row_4891.checked).trans rfl
theorem model_count_4891 : modelRow 4891 = 0 := by decide
theorem modelrow_agrees_4891 : modelRow 4891 = row 4891 := model_count_4891.trans row_count_4891.symm
def row_4892 : RowCertificate 4892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4892 1) := by change ¬ Nat.Prime 95736441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31912147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4892 2) := by change ¬ Nat.Prime 95726657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7363589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4892 3) := by change ¬ Nat.Prime 95716873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13673839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4892 : row 4892 = 0 := (row_4892.checked).trans rfl
theorem model_count_4892 : modelRow 4892 = 0 := by decide
theorem modelrow_agrees_4892 : modelRow 4892 = row 4892 := model_count_4892.trans row_count_4892.symm
def row_4893 : RowCertificate 4893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4893 1) := by change ¬ Nat.Prime 95775583; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 256771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4893 2) := by change Nat.Prime 95765797; exact ProjectEulerSolutions.P58PrimeNodes.p_95765797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4893 3) := by change ¬ Nat.Prime 95756011; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7365847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4893 : row 4893 = 1 := (row_4893.checked).trans rfl
theorem model_count_4893 : modelRow 4893 = 1 := by decide
theorem modelrow_agrees_4893 : modelRow 4893 = row 4893 := model_count_4893.trans row_count_4893.symm
def row_4894 : RowCertificate 4894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4894 1) := by change ¬ Nat.Prime 95814733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13687819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4894 2) := by change ¬ Nat.Prime 95804945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19160989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4894 3) := by change ¬ Nat.Prime 95795157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31931719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4894 : row 4894 = 0 := (row_4894.checked).trans rfl
theorem model_count_4894 : modelRow 4894 = 0 := by decide
theorem modelrow_agrees_4894 : modelRow 4894 = row 4894 := model_count_4894.trans row_count_4894.symm
def row_4895 : RowCertificate 4895 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4895 1) := by change ¬ Nat.Prime 95853891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31951297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4895 2) := by change ¬ Nat.Prime 95844101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3304969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4895 3) := by change Nat.Prime 95834311; exact ProjectEulerSolutions.P58PrimeNodes.p_95834311.prime
    exact decide_eq_true prime
theorem row_count_4895 : row 4895 = 1 := (row_4895.checked).trans rfl
theorem model_count_4895 : modelRow 4895 = 1 := by decide
theorem modelrow_agrees_4895 : modelRow 4895 = row 4895 := model_count_4895.trans row_count_4895.symm
def row_4896 : RowCertificate 4896 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4896 1) := by change ¬ Nat.Prime 95893057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7376389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4896 2) := by change ¬ Nat.Prime 95883265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19176653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4896 3) := by change ¬ Nat.Prime 95873473; exact PrattCertificates.composite_of_product (factor := 8803) (cofactor := 10891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4896 : row 4896 = 0 := (row_4896.checked).trans rfl
theorem model_count_4896 : modelRow 4896 = 0 := by decide
theorem modelrow_agrees_4896 : modelRow 4896 = row 4896 := model_count_4896.trans row_count_4896.symm
def row_4897 : RowCertificate 4897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4897 1) := by change ¬ Nat.Prime 95932231; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2592763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4897 2) := by change ¬ Nat.Prime 95922437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7378649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4897 3) := by change ¬ Nat.Prime 95912643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31970881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4897 : row 4897 = 0 := (row_4897.checked).trans rfl
theorem model_count_4897 : modelRow 4897 = 0 := by decide
theorem modelrow_agrees_4897 : modelRow 4897 = row 4897 := model_count_4897.trans row_count_4897.symm
def row_4898 : RowCertificate 4898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4898 1) := by change ¬ Nat.Prime 95971413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 31990471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4898 2) := by change ¬ Nat.Prime 95961617; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5644801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4898 3) := by change ¬ Nat.Prime 95951821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13707403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4898 : row 4898 = 0 := (row_4898.checked).trans rfl
theorem model_count_4898 : modelRow 4898 = 0 := by decide
theorem modelrow_agrees_4898 : modelRow 4898 = row 4898 := model_count_4898.trans row_count_4898.symm
def row_4899 : RowCertificate 4899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4899 1) := by change ¬ Nat.Prime 96010603; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7385431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4899 2) := by change ¬ Nat.Prime 96000805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19200161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4899 3) := by change ¬ Nat.Prime 95991007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13713001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4899 : row 4899 = 0 := (row_4899.checked).trans rfl
theorem model_count_4899 : modelRow 4899 = 0 := by decide
theorem modelrow_agrees_4899 : modelRow 4899 = row 4899 := model_count_4899.trans row_count_4899.symm
def row_4900 : RowCertificate 4900 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4900 1) := by change Nat.Prime 96049801; exact ProjectEulerSolutions.P58PrimeNodes.p_96049801.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4900 2) := by change ¬ Nat.Prime 96040001; exact PrattCertificates.composite_of_product (factor := 9661) (cofactor := 9941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4900 3) := by change ¬ Nat.Prime 96030201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32010067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4900 : row 4900 = 1 := (row_4900.checked).trans rfl
theorem model_count_4900 : modelRow 4900 = 1 := by decide
theorem modelrow_agrees_4900 : modelRow 4900 = row 4900 := model_count_4900.trans row_count_4900.symm
def row_4901 : RowCertificate 4901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4901 1) := by change ¬ Nat.Prime 96089007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32029669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4901 2) := by change ¬ Nat.Prime 96079205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19215841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4901 3) := by change ¬ Nat.Prime 96069403; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 3099013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4901 : row 4901 = 0 := (row_4901.checked).trans rfl
theorem model_count_4901 : modelRow 4901 = 0 := by decide
theorem modelrow_agrees_4901 : modelRow 4901 = row 4901 := model_count_4901.trans row_count_4901.symm
def row_4902 : RowCertificate 4902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4902 1) := by change ¬ Nat.Prime 96128221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13732603) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4902 2) := by change Nat.Prime 96118417; exact ProjectEulerSolutions.P58PrimeNodes.p_96118417.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4902 3) := by change Nat.Prime 96108613; exact ProjectEulerSolutions.P58PrimeNodes.p_96108613.prime
    exact decide_eq_true prime
theorem row_count_4902 : row 4902 = 2 := (row_4902.checked).trans rfl
theorem model_count_4902 : modelRow 4902 = 2 := by decide
theorem modelrow_agrees_4902 : modelRow 4902 = row 4902 := model_count_4902.trans row_count_4902.symm
def row_4903 : RowCertificate 4903 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4903 1) := by change ¬ Nat.Prime 96167443; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 313249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4903 2) := by change Nat.Prime 96157637; exact ProjectEulerSolutions.P58PrimeNodes.p_96157637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4903 3) := by change ¬ Nat.Prime 96147831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32049277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4903 : row 4903 = 1 := (row_4903.checked).trans rfl
theorem model_count_4903 : modelRow 4903 = 1 := by decide
theorem modelrow_agrees_4903 : modelRow 4903 = row 4903 := model_count_4903.trans row_count_4903.symm
def row_4904 : RowCertificate 4904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4904 1) := by change ¬ Nat.Prime 96206673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32068891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4904 2) := by change ¬ Nat.Prime 96196865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19239373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4904 3) := by change ¬ Nat.Prime 96187057; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1576837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4904 : row 4904 = 0 := (row_4904.checked).trans rfl
theorem model_count_4904 : modelRow 4904 = 0 := by decide
theorem modelrow_agrees_4904 : modelRow 4904 = row 4904 := model_count_4904.trans row_count_4904.symm
def row_4905 : RowCertificate 4905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4905 1) := by change ¬ Nat.Prime 96245911; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2238277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4905 2) := by change ¬ Nat.Prime 96236101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7402777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4905 3) := by change ¬ Nat.Prime 96226291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13746613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4905 : row 4905 = 0 := (row_4905.checked).trans rfl
theorem model_count_4905 : modelRow 4905 = 0 := by decide
theorem modelrow_agrees_4905 : modelRow 4905 = row 4905 := model_count_4905.trans row_count_4905.symm
def row_4906 : RowCertificate 4906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4906 1) := by change ¬ Nat.Prime 96285157; exact PrattCertificates.composite_of_product (factor := 967) (cofactor := 99571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4906 2) := by change ¬ Nat.Prime 96275345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19255069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4906 3) := by change ¬ Nat.Prime 96265533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32088511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4906 : row 4906 = 0 := (row_4906.checked).trans rfl
theorem model_count_4906 : modelRow 4906 = 0 := by decide
theorem modelrow_agrees_4906 : modelRow 4906 = row 4906 := model_count_4906.trans row_count_4906.symm
def row_4907 : RowCertificate 4907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4907 1) := by change ¬ Nat.Prime 96324411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32108137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4907 2) := by change ¬ Nat.Prime 96314597; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3321193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4907 3) := by change ¬ Nat.Prime 96304783; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 127219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4907 : row 4907 = 0 := (row_4907.checked).trans rfl
theorem model_count_4907 : modelRow 4907 = 0 := by decide
theorem modelrow_agrees_4907 : modelRow 4907 = row 4907 := model_count_4907.trans row_count_4907.symm
def row_4908 : RowCertificate 4908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4908 1) := by change ¬ Nat.Prime 96363673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13766239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4908 2) := by change ¬ Nat.Prime 96353857; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 852689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4908 3) := by change ¬ Nat.Prime 96344041; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5070739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4908 : row 4908 = 0 := (row_4908.checked).trans rfl
theorem model_count_4908 : modelRow 4908 = 0 := by decide
theorem modelrow_agrees_4908 : modelRow 4908 = row 4908 := model_count_4908.trans row_count_4908.symm
def row_4909 : RowCertificate 4909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4909 1) := by change ¬ Nat.Prime 96402943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13771849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4909 2) := by change ¬ Nat.Prime 96393125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19278625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4909 3) := by change ¬ Nat.Prime 96383307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32127769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4909 : row 4909 = 0 := (row_4909.checked).trans rfl
theorem model_count_4909 : modelRow 4909 = 0 := by decide
theorem modelrow_agrees_4909 : modelRow 4909 = row 4909 := model_count_4909.trans row_count_4909.symm
def row_4910 : RowCertificate 4910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4910 1) := by change ¬ Nat.Prime 96442221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32147407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4910 2) := by change ¬ Nat.Prime 96432401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7417877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4910 3) := by change ¬ Nat.Prime 96422581; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1439143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4910 : row 4910 = 0 := (row_4910.checked).trans rfl
theorem model_count_4910 : modelRow 4910 = 0 := by decide
theorem modelrow_agrees_4910 : modelRow 4910 = row 4910 := model_count_4910.trans row_count_4910.symm
def row_4911 : RowCertificate 4911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4911 1) := by change ¬ Nat.Prime 96481507; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 533047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4911 2) := by change ¬ Nat.Prime 96471685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19294337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4911 3) := by change ¬ Nat.Prime 96461863; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 3111673) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4911 : row 4911 = 0 := (row_4911.checked).trans rfl
theorem model_count_4911 : modelRow 4911 = 0 := by decide
theorem modelrow_agrees_4911 : modelRow 4911 = row 4911 := model_count_4911.trans row_count_4911.symm
def row_4912 : RowCertificate 4912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4912 1) := by change ¬ Nat.Prime 96520801; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7424677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4912 2) := by change ¬ Nat.Prime 96510977; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 1084393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4912 3) := by change ¬ Nat.Prime 96501153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32167051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4912 : row 4912 = 0 := (row_4912.checked).trans rfl
theorem model_count_4912 : modelRow 4912 = 0 := by decide
theorem modelrow_agrees_4912 : modelRow 4912 = row 4912 := model_count_4912.trans row_count_4912.symm
def row_4913 : RowCertificate 4913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4913 1) := by change ¬ Nat.Prime 96560103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32186701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4913 2) := by change Nat.Prime 96550277; exact ProjectEulerSolutions.P58PrimeNodes.p_96550277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4913 3) := by change ¬ Nat.Prime 96540451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13791493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4913 : row 4913 = 1 := (row_4913.checked).trans rfl
theorem model_count_4913 : modelRow 4913 = 1 := by decide
theorem modelrow_agrees_4913 : modelRow 4913 = row 4913 := model_count_4913.trans row_count_4913.symm
def row_4914 : RowCertificate 4914 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4914 1) := by change Nat.Prime 96599413; exact ProjectEulerSolutions.P58PrimeNodes.p_96599413.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4914 2) := by change ¬ Nat.Prime 96589585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19317917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4914 3) := by change ¬ Nat.Prime 96579757; exact PrattCertificates.composite_of_product (factor := 1831) (cofactor := 52747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4914 : row 4914 = 1 := (row_4914.checked).trans rfl
theorem model_count_4914 : modelRow 4914 = 1 := by decide
theorem modelrow_agrees_4914 : modelRow 4914 = row 4914 := model_count_4914.trans row_count_4914.symm
def row_4915 : RowCertificate 4915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4915 1) := by change ¬ Nat.Prime 96638731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13805533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4915 2) := by change ¬ Nat.Prime 96628901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5684053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4915 3) := by change ¬ Nat.Prime 96619071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32206357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4915 : row 4915 = 0 := (row_4915.checked).trans rfl
theorem model_count_4915 : modelRow 4915 = 0 := by decide
theorem modelrow_agrees_4915 : modelRow 4915 = row 4915 := model_count_4915.trans row_count_4915.symm
def row_4916 : RowCertificate 4916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4916 1) := by change ¬ Nat.Prime 96678057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32226019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4916 2) := by change ¬ Nat.Prime 96668225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19333645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4916 3) := by change ¬ Nat.Prime 96658393; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7435261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4916 : row 4916 = 0 := (row_4916.checked).trans rfl
theorem model_count_4916 : modelRow 4916 = 0 := by decide
theorem modelrow_agrees_4916 : modelRow 4916 = row 4916 := model_count_4916.trans row_count_4916.symm
def row_4917 : RowCertificate 4917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4917 1) := by change ¬ Nat.Prime 96717391; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5090389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4917 2) := by change ¬ Nat.Prime 96707557; exact PrattCertificates.composite_of_product (factor := 2281) (cofactor := 42397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4917 3) := by change ¬ Nat.Prime 96697723; exact PrattCertificates.composite_of_product (factor := 2437) (cofactor := 39679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4917 : row 4917 = 0 := (row_4917.checked).trans rfl
theorem model_count_4917 : modelRow 4917 = 0 := by decide
theorem modelrow_agrees_4917 : modelRow 4917 = row 4917 := model_count_4917.trans row_count_4917.symm
def row_4918 : RowCertificate 4918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4918 1) := by change ¬ Nat.Prime 96756733; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 132001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4918 2) := by change ¬ Nat.Prime 96746897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7442069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4918 3) := by change ¬ Nat.Prime 96737061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32245687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4918 : row 4918 = 0 := (row_4918.checked).trans rfl
theorem model_count_4918 : modelRow 4918 = 0 := by decide
theorem modelrow_agrees_4918 : modelRow 4918 = row 4918 := model_count_4918.trans row_count_4918.symm
def row_4919 : RowCertificate 4919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4919 1) := by change ¬ Nat.Prime 96796083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32265361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4919 2) := by change ¬ Nat.Prime 96786245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19357249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4919 3) := by change ¬ Nat.Prime 96776407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13825201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4919 : row 4919 = 0 := (row_4919.checked).trans rfl
theorem model_count_4919 : modelRow 4919 = 0 := by decide
theorem modelrow_agrees_4919 : modelRow 4919 = row 4919 := model_count_4919.trans row_count_4919.symm
def row_4920 : RowCertificate 4920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4920 1) := by change ¬ Nat.Prime 96835441; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2251987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4920 2) := by change ¬ Nat.Prime 96825601; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 248909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4920 3) := by change ¬ Nat.Prime 96815761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13830823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4920 : row 4920 = 0 := (row_4920.checked).trans rfl
theorem model_count_4920 : modelRow 4920 = 0 := by decide
theorem modelrow_agrees_4920 : modelRow 4920 = row 4920 := model_count_4920.trans row_count_4920.symm
def row_4921 : RowCertificate 4921 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4921 1) := by change Nat.Prime 96874807; exact ProjectEulerSolutions.P58PrimeNodes.p_96874807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4921 2) := by change ¬ Nat.Prime 96864965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19372993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4921 3) := by change ¬ Nat.Prime 96855123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32285041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4921 : row 4921 = 1 := (row_4921.checked).trans rfl
theorem model_count_4921 : modelRow 4921 = 1 := by decide
theorem modelrow_agrees_4921 : modelRow 4921 = row 4921 := model_count_4921.trans row_count_4921.symm
def row_4922 : RowCertificate 4922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4922 1) := by change ¬ Nat.Prime 96914181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32304727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4922 2) := by change Nat.Prime 96904337; exact ProjectEulerSolutions.P58PrimeNodes.p_96904337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4922 3) := by change ¬ Nat.Prime 96894493; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 486907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4922 : row 4922 = 1 := (row_4922.checked).trans rfl
theorem model_count_4922 : modelRow 4922 = 1 := by decide
theorem modelrow_agrees_4922 : modelRow 4922 = row 4922 := model_count_4922.trans row_count_4922.symm
def row_4923 : RowCertificate 4923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4923 1) := by change ¬ Nat.Prime 96953563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13850509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4923 2) := by change ¬ Nat.Prime 96943717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7457209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4923 3) := by change Nat.Prime 96933871; exact ProjectEulerSolutions.P58PrimeNodes.p_96933871.prime
    exact decide_eq_true prime
theorem row_count_4923 : row 4923 = 1 := (row_4923.checked).trans rfl
theorem model_count_4923 : modelRow 4923 = 1 := by decide
theorem modelrow_agrees_4923 : modelRow 4923 = row 4923 := model_count_4923.trans row_count_4923.symm
def row_4924 : RowCertificate 4924 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4924 1) := by change Nat.Prime 96992953; exact ProjectEulerSolutions.P58PrimeNodes.p_96992953.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4924 2) := by change ¬ Nat.Prime 96983105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19396621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4924 3) := by change ¬ Nat.Prime 96973257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32324419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4924 : row 4924 = 1 := (row_4924.checked).trans rfl
theorem model_count_4924 : modelRow 4924 = 1 := by decide
theorem modelrow_agrees_4924 : modelRow 4924 = row 4924 := model_count_4924.trans row_count_4924.symm
def row_4925 : RowCertificate 4925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4925 1) := by change ¬ Nat.Prime 97032351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32344117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4925 2) := by change ¬ Nat.Prime 97022501; exact PrattCertificates.composite_of_product (factor := 8117) (cofactor := 11953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4925 3) := by change ¬ Nat.Prime 97012651; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5105929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4925 : row 4925 = 0 := (row_4925.checked).trans rfl
theorem model_count_4925 : modelRow 4925 = 0 := by decide
theorem modelrow_agrees_4925 : modelRow 4925 = row 4925 := model_count_4925.trans row_count_4925.symm
def row_4926 : RowCertificate 4926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4926 1) := by change ¬ Nat.Prime 97071757; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 3131347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4926 2) := by change ¬ Nat.Prime 97061905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19412381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4926 3) := by change ¬ Nat.Prime 97052053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13864579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4926 : row 4926 = 0 := (row_4926.checked).trans rfl
theorem model_count_4926 : modelRow 4926 = 0 := by decide
theorem modelrow_agrees_4926 : modelRow 4926 = row 4926 := model_count_4926.trans row_count_4926.symm
def row_4927 : RowCertificate 4927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4927 1) := by change ¬ Nat.Prime 97111171; exact PrattCertificates.composite_of_product (factor := 4021) (cofactor := 24151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4927 2) := by change ¬ Nat.Prime 97101317; exact PrattCertificates.composite_of_product (factor := 761) (cofactor := 127597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4927 3) := by change ¬ Nat.Prime 97091463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32363821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4927 : row 4927 = 0 := (row_4927.checked).trans rfl
theorem model_count_4927 : modelRow 4927 = 0 := by decide
theorem modelrow_agrees_4927 : modelRow 4927 = row 4927 := model_count_4927.trans row_count_4927.symm
def row_4928 : RowCertificate 4928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4928 1) := by change ¬ Nat.Prime 97150593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32383531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4928 2) := by change ¬ Nat.Prime 97140737; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5714161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4928 3) := by change ¬ Nat.Prime 97130881; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 891109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4928 : row 4928 = 0 := (row_4928.checked).trans rfl
theorem model_count_4928 : modelRow 4928 = 0 := by decide
theorem modelrow_agrees_4928 : modelRow 4928 = row 4928 := model_count_4928.trans row_count_4928.symm
def row_4929 : RowCertificate 4929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4929 1) := by change ¬ Nat.Prime 97190023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13884289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4929 2) := by change ¬ Nat.Prime 97180165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19436033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4929 3) := by change ¬ Nat.Prime 97170307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7474639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4929 : row 4929 = 0 := (row_4929.checked).trans rfl
theorem model_count_4929 : modelRow 4929 = 0 := by decide
theorem modelrow_agrees_4929 : modelRow 4929 = row 4929 := model_count_4929.trans row_count_4929.symm
def row_4930 : RowCertificate 4930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4930 1) := by change ¬ Nat.Prime 97229461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13889923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4930 2) := by change Nat.Prime 97219601; exact ProjectEulerSolutions.P58PrimeNodes.p_97219601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4930 3) := by change ¬ Nat.Prime 97209741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32403247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4930 : row 4930 = 1 := (row_4930.checked).trans rfl
theorem model_count_4930 : modelRow 4930 = 1 := by decide
theorem modelrow_agrees_4930 : modelRow 4930 = row 4930 := model_count_4930.trans row_count_4930.symm
def row_4931 : RowCertificate 4931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4931 1) := by change ¬ Nat.Prime 97268907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32422969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4931 2) := by change ¬ Nat.Prime 97259045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19451809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4931 3) := by change Nat.Prime 97249183; exact ProjectEulerSolutions.P58PrimeNodes.p_97249183.prime
    exact decide_eq_true prime
theorem row_count_4931 : row 4931 = 1 := (row_4931.checked).trans rfl
theorem model_count_4931 : modelRow 4931 = 1 := by decide
theorem modelrow_agrees_4931 : modelRow 4931 = row 4931 := model_count_4931.trans row_count_4931.symm
def row_4932 : RowCertificate 4932 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4932 1) := by change Nat.Prime 97308361; exact ProjectEulerSolutions.P58PrimeNodes.p_97308361.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4932 2) := by change ¬ Nat.Prime 97298497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5723441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4932 3) := by change ¬ Nat.Prime 97288633; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7483741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4932 : row 4932 = 1 := (row_4932.checked).trans rfl
theorem model_count_4932 : modelRow 4932 = 1 := by decide
theorem modelrow_agrees_4932 : modelRow 4932 = row 4932 := model_count_4932.trans row_count_4932.symm
def row_4933 : RowCertificate 4933 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4933 1) := by change Nat.Prime 97347823; exact ProjectEulerSolutions.P58PrimeNodes.p_97347823.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4933 2) := by change ¬ Nat.Prime 97337957; exact PrattCertificates.composite_of_product (factor := 7829) (cofactor := 12433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4933 3) := by change ¬ Nat.Prime 97328091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32442697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4933 : row 4933 = 1 := (row_4933.checked).trans rfl
theorem model_count_4933 : modelRow 4933 = 1 := by decide
theorem modelrow_agrees_4933 : modelRow 4933 = row 4933 := model_count_4933.trans row_count_4933.symm
def row_4934 : RowCertificate 4934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4934 1) := by change ¬ Nat.Prime 97387293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32462431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4934 2) := by change ¬ Nat.Prime 97377425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19475485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4934 3) := by change ¬ Nat.Prime 97367557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13909651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4934 : row 4934 = 0 := (row_4934.checked).trans rfl
theorem model_count_4934 : modelRow 4934 = 0 := by decide
theorem modelrow_agrees_4934 : modelRow 4934 = row 4934 := model_count_4934.trans row_count_4934.symm
def row_4935 : RowCertificate 4935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4935 1) := by change ¬ Nat.Prime 97426771; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7494367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4935 2) := by change ¬ Nat.Prime 97416901; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 91129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4935 3) := by change ¬ Nat.Prime 97407031; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 238159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4935 : row 4935 = 0 := (row_4935.checked).trans rfl
theorem model_count_4935 : modelRow 4935 = 0 := by decide
theorem modelrow_agrees_4935 : modelRow 4935 = row 4935 := model_count_4935.trans row_count_4935.symm
def row_4936 : RowCertificate 4936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4936 1) := by change ¬ Nat.Prime 97466257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13923751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4936 2) := by change ¬ Nat.Prime 97456385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19491277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4936 3) := by change ¬ Nat.Prime 97446513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32482171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4936 : row 4936 = 0 := (row_4936.checked).trans rfl
theorem model_count_4936 : modelRow 4936 = 0 := by decide
theorem modelrow_agrees_4936 : modelRow 4936 = row 4936 := model_count_4936.trans row_count_4936.symm
def row_4937 : RowCertificate 4937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4937 1) := by change ¬ Nat.Prime 97505751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32501917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4937 2) := by change Nat.Prime 97495877; exact ProjectEulerSolutions.P58PrimeNodes.p_97495877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4937 3) := by change ¬ Nat.Prime 97486003; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 894367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4937 : row 4937 = 1 := (row_4937.checked).trans rfl
theorem model_count_4937 : modelRow 4937 = 1 := by decide
theorem modelrow_agrees_4937 : modelRow 4937 = row 4937 := model_count_4937.trans row_count_4937.symm
def row_4938 : RowCertificate 4938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4938 1) := by change ¬ Nat.Prime 97545253; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7503481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4938 2) := by change Nat.Prime 97535377; exact ProjectEulerSolutions.P58PrimeNodes.p_97535377.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4938 3) := by change Nat.Prime 97525501; exact ProjectEulerSolutions.P58PrimeNodes.p_97525501.prime
    exact decide_eq_true prime
theorem row_count_4938 : row 4938 = 2 := (row_4938.checked).trans rfl
theorem model_count_4938 : modelRow 4938 = 2 := by decide
theorem modelrow_agrees_4938 : modelRow 4938 = row 4938 := model_count_4938.trans row_count_4938.symm
def row_4939 : RowCertificate 4939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4939 1) := by change ¬ Nat.Prime 97584763; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1456489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4939 2) := by change ¬ Nat.Prime 97574885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19514977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4939 3) := by change ¬ Nat.Prime 97565007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32521669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4939 : row 4939 = 0 := (row_4939.checked).trans rfl
theorem model_count_4939 : modelRow 4939 = 0 := by decide
theorem modelrow_agrees_4939 : modelRow 4939 = row 4939 := model_count_4939.trans row_count_4939.symm
def row_4940 : RowCertificate 4940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4940 1) := by change ¬ Nat.Prime 97624281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32541427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4940 2) := by change Nat.Prime 97614401; exact ProjectEulerSolutions.P58PrimeNodes.p_97614401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4940 3) := by change ¬ Nat.Prime 97604521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13943503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4940 : row 4940 = 1 := (row_4940.checked).trans rfl
theorem model_count_4940 : modelRow 4940 = 1 := by decide
theorem modelrow_agrees_4940 : modelRow 4940 = row 4940 := model_count_4940.trans row_count_4940.symm
def row_4941 : RowCertificate 4941 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4941 1) := by change Nat.Prime 97663807; exact ProjectEulerSolutions.P58PrimeNodes.p_97663807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4941 2) := by change ¬ Nat.Prime 97653925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19530785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4941 3) := by change ¬ Nat.Prime 97644043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13949149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4941 : row 4941 = 1 := (row_4941.checked).trans rfl
theorem model_count_4941 : modelRow 4941 = 1 := by decide
theorem modelrow_agrees_4941 : modelRow 4941 = row 4941 := model_count_4941.trans row_count_4941.symm
def row_4942 : RowCertificate 4942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4942 1) := by change ¬ Nat.Prime 97703341; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 599407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4942 2) := by change ¬ Nat.Prime 97693457; exact PrattCertificates.composite_of_product (factor := 3637) (cofactor := 26861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4942 3) := by change ¬ Nat.Prime 97683573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32561191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4942 : row 4942 = 0 := (row_4942.checked).trans rfl
theorem model_count_4942 : modelRow 4942 = 0 := by decide
theorem modelrow_agrees_4942 : modelRow 4942 = row 4942 := model_count_4942.trans row_count_4942.symm
def row_4943 : RowCertificate 4943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4943 1) := by change ¬ Nat.Prime 97742883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32580961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4943 2) := by change ¬ Nat.Prime 97732997; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 896633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4943 3) := by change Nat.Prime 97723111; exact ProjectEulerSolutions.P58PrimeNodes.p_97723111.prime
    exact decide_eq_true prime
theorem row_count_4943 : row 4943 = 1 := (row_4943.checked).trans rfl
theorem model_count_4943 : modelRow 4943 = 1 := by decide
theorem modelrow_agrees_4943 : modelRow 4943 = row 4943 := model_count_4943.trans row_count_4943.symm
def row_4944 : RowCertificate 4944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4944 1) := by change ¬ Nat.Prime 97782433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13968919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4944 2) := by change ¬ Nat.Prime 97772545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19554509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4944 3) := by change ¬ Nat.Prime 97762657; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5145403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4944 : row 4944 = 0 := (row_4944.checked).trans rfl
theorem model_count_4944 : modelRow 4944 = 0 := by decide
theorem modelrow_agrees_4944 : modelRow 4944 = row 4944 := model_count_4944.trans row_count_4944.symm
def row_4945 : RowCertificate 4945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4945 1) := by change ¬ Nat.Prime 97821991; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 246403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4945 2) := by change ¬ Nat.Prime 97812101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 5753653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4945 3) := by change ¬ Nat.Prime 97802211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32600737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4945 : row 4945 = 0 := (row_4945.checked).trans rfl
theorem model_count_4945 : modelRow 4945 = 0 := by decide
theorem modelrow_agrees_4945 : modelRow 4945 = row 4945 := model_count_4945.trans row_count_4945.symm
def row_4946 : RowCertificate 4946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4946 1) := by change ¬ Nat.Prime 97861557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32620519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4946 2) := by change ¬ Nat.Prime 97851665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19570333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4946 3) := by change ¬ Nat.Prime 97841773; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5149567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4946 : row 4946 = 0 := (row_4946.checked).trans rfl
theorem model_count_4946 : modelRow 4946 = 0 := by decide
theorem modelrow_agrees_4946 : modelRow 4946 = row 4946 := model_count_4946.trans row_count_4946.symm
def row_4947 : RowCertificate 4947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4947 1) := by change ¬ Nat.Prime 97901131; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 3158101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4947 2) := by change Nat.Prime 97891237; exact ProjectEulerSolutions.P58PrimeNodes.p_97891237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4947 3) := by change ¬ Nat.Prime 97881343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 13983049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4947 : row 4947 = 1 := (row_4947.checked).trans rfl
theorem model_count_4947 : modelRow 4947 = 1 := by decide
theorem modelrow_agrees_4947 : modelRow 4947 = row 4947 := model_count_4947.trans row_count_4947.symm
def row_4948 : RowCertificate 4948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4948 1) := by change ¬ Nat.Prime 97940713; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7533901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4948 2) := by change Nat.Prime 97930817; exact ProjectEulerSolutions.P58PrimeNodes.p_97930817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4948 3) := by change ¬ Nat.Prime 97920921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32640307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4948 : row 4948 = 1 := (row_4948.checked).trans rfl
theorem model_count_4948 : modelRow 4948 = 1 := by decide
theorem modelrow_agrees_4948 : modelRow 4948 = row 4948 := model_count_4948.trans row_count_4948.symm
def row_4949 : RowCertificate 4949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4949 1) := by change ¬ Nat.Prime 97980303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32660101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4949 2) := by change ¬ Nat.Prime 97970405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19594081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4949 3) := by change Nat.Prime 97960507; exact ProjectEulerSolutions.P58PrimeNodes.p_97960507.prime
    exact decide_eq_true prime
theorem row_count_4949 : row 4949 = 1 := (row_4949.checked).trans rfl
theorem model_count_4949 : modelRow 4949 = 1 := by decide
theorem modelrow_agrees_4949 : modelRow 4949 = row 4949 := model_count_4949.trans row_count_4949.symm
def row_4950 : RowCertificate 4950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4950 1) := by change ¬ Nat.Prime 98019901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14002843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4950 2) := by change Nat.Prime 98010001; exact ProjectEulerSolutions.P58PrimeNodes.p_98010001.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4950 3) := by change Nat.Prime 98000101; exact ProjectEulerSolutions.P58PrimeNodes.p_98000101.prime
    exact decide_eq_true prime
theorem row_count_4950 : row 4950 = 2 := (row_4950.checked).trans rfl
theorem model_count_4950 : modelRow 4950 = 2 := by decide
theorem modelrow_agrees_4950 : modelRow 4950 = row 4950 := model_count_4950.trans row_count_4950.symm
def row_4951 : RowCertificate 4951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4951 1) := by change ¬ Nat.Prime 98059507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14008501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4951 2) := by change ¬ Nat.Prime 98049605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19609921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4951 3) := by change ¬ Nat.Prime 98039703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32679901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4951 : row 4951 = 0 := (row_4951.checked).trans rfl
theorem model_count_4951 : modelRow 4951 = 0 := by decide
theorem modelrow_agrees_4951 : modelRow 4951 = row 4951 := model_count_4951.trans row_count_4951.symm
def row_4952 : RowCertificate 4952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4952 1) := by change ¬ Nat.Prime 98099121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32699707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4952 2) := by change Nat.Prime 98089217; exact ProjectEulerSolutions.P58PrimeNodes.p_98089217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4952 3) := by change ¬ Nat.Prime 98079313; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 624709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4952 : row 4952 = 1 := (row_4952.checked).trans rfl
theorem model_count_4952 : modelRow 4952 = 1 := by decide
theorem modelrow_agrees_4952 : modelRow 4952 = row 4952 := model_count_4952.trans row_count_4952.symm
def row_4953 : RowCertificate 4953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4953 1) := by change ¬ Nat.Prime 98138743; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5165197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4953 2) := by change ¬ Nat.Prime 98128837; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3383753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4953 3) := by change ¬ Nat.Prime 98118931; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2651863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4953 : row 4953 = 0 := (row_4953.checked).trans rfl
theorem model_count_4953 : modelRow 4953 = 0 := by decide
theorem modelrow_agrees_4953 : modelRow 4953 = row 4953 := model_count_4953.trans row_count_4953.symm
def row_4954 : RowCertificate 4954 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4954 1) := by change Nat.Prime 98178373; exact ProjectEulerSolutions.P58PrimeNodes.p_98178373.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4954 2) := by change ¬ Nat.Prime 98168465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19633693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4954 3) := by change ¬ Nat.Prime 98158557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32719519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4954 : row 4954 = 1 := (row_4954.checked).trans rfl
theorem model_count_4954 : modelRow 4954 = 1 := by decide
theorem modelrow_agrees_4954 : modelRow 4954 = row 4954 := model_count_4954.trans row_count_4954.symm
def row_4955 : RowCertificate 4955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4955 1) := by change ¬ Nat.Prime 98218011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32739337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4955 2) := by change ¬ Nat.Prime 98208101; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2654273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4955 3) := by change ¬ Nat.Prime 98198191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14028313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4955 : row 4955 = 0 := (row_4955.checked).trans rfl
theorem model_count_4955 : modelRow 4955 = 0 := by decide
theorem modelrow_agrees_4955 : modelRow 4955 = row 4955 := model_count_4955.trans row_count_4955.symm
def row_4956 : RowCertificate 4956 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4956 1) := by change Nat.Prime 98257657; exact ProjectEulerSolutions.P58PrimeNodes.p_98257657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4956 2) := by change ¬ Nat.Prime 98247745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19649549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4956 3) := by change ¬ Nat.Prime 98237833; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 706747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4956 : row 4956 = 1 := (row_4956.checked).trans rfl
theorem model_count_4956 : modelRow 4956 = 1 := by decide
theorem modelrow_agrees_4956 : modelRow 4956 = row 4956 := model_count_4956.trans row_count_4956.symm
def row_4957 : RowCertificate 4957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4957 1) := by change ¬ Nat.Prime 98297311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14042473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4957 2) := by change ¬ Nat.Prime 98287397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7560569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4957 3) := by change ¬ Nat.Prime 98277483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32759161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4957 : row 4957 = 0 := (row_4957.checked).trans rfl
theorem model_count_4957 : modelRow 4957 = 0 := by decide
theorem modelrow_agrees_4957 : modelRow 4957 = row 4957 := model_count_4957.trans row_count_4957.symm
def row_4958 : RowCertificate 4958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4958 1) := by change ¬ Nat.Prime 98336973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32778991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4958 2) := by change ¬ Nat.Prime 98327057; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 1013681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4958 3) := by change ¬ Nat.Prime 98317141; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7562857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4958 : row 4958 = 0 := (row_4958.checked).trans rfl
theorem model_count_4958 : modelRow 4958 = 0 := by decide
theorem modelrow_agrees_4958 : modelRow 4958 = row 4958 := model_count_4958.trans row_count_4958.symm
def row_4959 : RowCertificate 4959 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4959 1) := by change Nat.Prime 98376643; exact ProjectEulerSolutions.P58PrimeNodes.p_98376643.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4959 2) := by change ¬ Nat.Prime 98366725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19673345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4959 3) := by change ¬ Nat.Prime 98356807; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 314239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4959 : row 4959 = 1 := (row_4959.checked).trans rfl
theorem model_count_4959 : modelRow 4959 = 1 := by decide
theorem modelrow_agrees_4959 : modelRow 4959 = row 4959 := model_count_4959.trans row_count_4959.symm
def row_4960 : RowCertificate 4960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4960 1) := by change ¬ Nat.Prime 98416321; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 215353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4960 2) := by change ¬ Nat.Prime 98406401; exact PrattCertificates.composite_of_product (factor := 6553) (cofactor := 15017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4960 3) := by change ¬ Nat.Prime 98396481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32798827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4960 : row 4960 = 0 := (row_4960.checked).trans rfl
theorem model_count_4960 : modelRow 4960 = 0 := by decide
theorem modelrow_agrees_4960 : modelRow 4960 = row 4960 := model_count_4960.trans row_count_4960.symm
def row_4961 : RowCertificate 4961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4961 1) := by change ¬ Nat.Prime 98456007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32818669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4961 2) := by change ¬ Nat.Prime 98446085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19689217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4961 3) := by change ¬ Nat.Prime 98436163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14062309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4961 : row 4961 = 0 := (row_4961.checked).trans rfl
theorem model_count_4961 : modelRow 4961 = 0 := by decide
theorem modelrow_agrees_4961 : modelRow 4961 = row 4961 := model_count_4961.trans row_count_4961.symm
def row_4962 : RowCertificate 4962 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4962 1) := by change Nat.Prime 98495701; exact ProjectEulerSolutions.P58PrimeNodes.p_98495701.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4962 2) := by change ¬ Nat.Prime 98485777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7575829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4962 3) := by change ¬ Nat.Prime 98475853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14067979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4962 : row 4962 = 1 := (row_4962.checked).trans rfl
theorem model_count_4962 : modelRow 4962 = 1 := by decide
theorem modelrow_agrees_4962 : modelRow 4962 = row 4962 := model_count_4962.trans row_count_4962.symm
def row_4963 : RowCertificate 4963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4963 1) := by change ¬ Nat.Prime 98535403; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2663119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4963 2) := by change ¬ Nat.Prime 98525477; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 279109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4963 3) := by change ¬ Nat.Prime 98515551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32838517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4963 : row 4963 = 0 := (row_4963.checked).trans rfl
theorem model_count_4963 : modelRow 4963 = 0 := by decide
theorem modelrow_agrees_4963 : modelRow 4963 = row 4963 := model_count_4963.trans row_count_4963.symm
def row_4964 : RowCertificate 4964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4964 1) := by change ¬ Nat.Prime 98575113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32858371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4964 2) := by change ¬ Nat.Prime 98565185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19713037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4964 3) := by change ¬ Nat.Prime 98555257; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 510649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4964 : row 4964 = 0 := (row_4964.checked).trans rfl
theorem model_count_4964 : modelRow 4964 = 0 := by decide
theorem modelrow_agrees_4964 : modelRow 4964 = row 4964 := model_count_4964.trans row_count_4964.symm
def row_4965 : RowCertificate 4965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4965 1) := by change ¬ Nat.Prime 98614831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14087833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4965 2) := by change ¬ Nat.Prime 98604901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3400169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4965 3) := by change ¬ Nat.Prime 98594971; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 5189209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4965 : row 4965 = 0 := (row_4965.checked).trans rfl
theorem model_count_4965 : modelRow 4965 = 0 := by decide
theorem modelrow_agrees_4965 : modelRow 4965 = row 4965 := model_count_4965.trans row_count_4965.symm
def row_4966 : RowCertificate 4966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4966 1) := by change ¬ Nat.Prime 98654557; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 88321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4966 2) := by change ¬ Nat.Prime 98644625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19728925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4966 3) := by change ¬ Nat.Prime 98634693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32878231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4966 : row 4966 = 0 := (row_4966.checked).trans rfl
theorem model_count_4966 : modelRow 4966 = 0 := by decide
theorem modelrow_agrees_4966 : modelRow 4966 = row 4966 := model_count_4966.trans row_count_4966.symm
def row_4967 : RowCertificate 4967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4967 1) := by change ¬ Nat.Prime 98694291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32898097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4967 2) := by change ¬ Nat.Prime 98684357; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1861969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4967 3) := by change ¬ Nat.Prime 98674423; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 653473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4967 : row 4967 = 0 := (row_4967.checked).trans rfl
theorem model_count_4967 : modelRow 4967 = 0 := by decide
theorem modelrow_agrees_4967 : modelRow 4967 = row 4967 := model_count_4967.trans row_count_4967.symm
def row_4968 : RowCertificate 4968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4968 1) := by change ¬ Nat.Prime 98734033; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1352521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4968 2) := by change ¬ Nat.Prime 98724097; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 545437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4968 3) := by change ¬ Nat.Prime 98714161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14102023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4968 : row 4968 = 0 := (row_4968.checked).trans rfl
theorem model_count_4968 : modelRow 4968 = 0 := by decide
theorem modelrow_agrees_4968 : modelRow 4968 = row 4968 := model_count_4968.trans row_count_4968.symm
def row_4969 : RowCertificate 4969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4969 1) := by change ¬ Nat.Prime 98773783; exact PrattCertificates.composite_of_product (factor := 7933) (cofactor := 12451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4969 2) := by change ¬ Nat.Prime 98763845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19752769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4969 3) := by change ¬ Nat.Prime 98753907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32917969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4969 : row 4969 = 0 := (row_4969.checked).trans rfl
theorem model_count_4969 : modelRow 4969 = 0 := by decide
theorem modelrow_agrees_4969 : modelRow 4969 = row 4969 := model_count_4969.trans row_count_4969.symm
def row_4970 : RowCertificate 4970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4970 1) := by change ¬ Nat.Prime 98813541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32937847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4970 2) := by change ¬ Nat.Prime 98803601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7600277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4970 3) := by change ¬ Nat.Prime 98793661; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2297527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4970 : row 4970 = 0 := (row_4970.checked).trans rfl
theorem model_count_4970 : modelRow 4970 = 0 := by decide
theorem modelrow_agrees_4970 : modelRow 4970 = row 4970 := model_count_4970.trans row_count_4970.symm
def row_4971 : RowCertificate 4971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4971 1) := by change ¬ Nat.Prime 98853307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14121901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4971 2) := by change ¬ Nat.Prime 98843365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19768673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4971 3) := by change ¬ Nat.Prime 98833423; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7602571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4971 : row 4971 = 0 := (row_4971.checked).trans rfl
theorem model_count_4971 : modelRow 4971 = 0 := by decide
theorem modelrow_agrees_4971 : modelRow 4971 = row 4971 := model_count_4971.trans row_count_4971.symm
def row_4972 : RowCertificate 4972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4972 1) := by change ¬ Nat.Prime 98893081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14127583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4972 2) := by change ¬ Nat.Prime 98883137; exact PrattCertificates.composite_of_product (factor := 653) (cofactor := 151429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4972 3) := by change ¬ Nat.Prime 98873193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32957731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4972 : row 4972 = 0 := (row_4972.checked).trans rfl
theorem model_count_4972 : modelRow 4972 = 0 := by decide
theorem modelrow_agrees_4972 : modelRow 4972 = row 4972 := model_count_4972.trans row_count_4972.symm
def row_4973 : RowCertificate 4973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4973 1) := by change ¬ Nat.Prime 98932863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32977621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4973 2) := by change ¬ Nat.Prime 98922917; exact PrattCertificates.composite_of_product (factor := 2801) (cofactor := 35317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4973 3) := by change ¬ Nat.Prime 98912971; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 3190741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4973 : row 4973 = 0 := (row_4973.checked).trans rfl
theorem model_count_4973 : modelRow 4973 = 0 := by decide
theorem modelrow_agrees_4973 : modelRow 4973 = row 4973 := model_count_4973.trans row_count_4973.symm
def row_4974 : RowCertificate 4974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4974 1) := by change ¬ Nat.Prime 98972653; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7613281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4974 2) := by change ¬ Nat.Prime 98962705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19792541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4974 3) := by change Nat.Prime 98952757; exact ProjectEulerSolutions.P58PrimeNodes.p_98952757.prime
    exact decide_eq_true prime
theorem row_count_4974 : row 4974 = 1 := (row_4974.checked).trans rfl
theorem model_count_4974 : modelRow 4974 = 1 := by decide
theorem modelrow_agrees_4974 : modelRow 4974 = row 4974 := model_count_4974.trans row_count_4974.symm
def row_4975 : RowCertificate 4975 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4975 1) := by change Nat.Prime 99012451; exact ProjectEulerSolutions.P58PrimeNodes.p_99012451.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4975 2) := by change ¬ Nat.Prime 99002501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7615577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4975 3) := by change ¬ Nat.Prime 98992551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 32997517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4975 : row 4975 = 1 := (row_4975.checked).trans rfl
theorem model_count_4975 : modelRow 4975 = 1 := by decide
theorem modelrow_agrees_4975 : modelRow 4975 = row 4975 := model_count_4975.trans row_count_4975.symm
def row_4976 : RowCertificate 4976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4976 1) := by change ¬ Nat.Prime 99052257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33017419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4976 2) := by change ¬ Nat.Prime 99042305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19808461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4976 3) := by change ¬ Nat.Prime 99032353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14147479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4976 : row 4976 = 0 := (row_4976.checked).trans rfl
theorem model_count_4976 : modelRow 4976 = 0 := by decide
theorem modelrow_agrees_4976 : modelRow 4976 = row 4976 := model_count_4976.trans row_count_4976.symm
def row_4977 : RowCertificate 4977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4977 1) := by change ¬ Nat.Prime 99092071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7622467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4977 2) := by change ¬ Nat.Prime 99082117; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 2416637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4977 3) := by change ¬ Nat.Prime 99072163; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1478689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4977 : row 4977 = 0 := (row_4977.checked).trans rfl
theorem model_count_4977 : modelRow 4977 = 0 := by decide
theorem modelrow_agrees_4977 : modelRow 4977 = row 4977 := model_count_4977.trans row_count_4977.symm
def row_4978 : RowCertificate 4978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4978 1) := by change ¬ Nat.Prime 99131893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14161699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4978 2) := by change Nat.Prime 99121937; exact ProjectEulerSolutions.P58PrimeNodes.p_99121937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4978 3) := by change ¬ Nat.Prime 99111981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33037327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4978 : row 4978 = 1 := (row_4978.checked).trans rfl
theorem model_count_4978 : modelRow 4978 = 1 := by decide
theorem modelrow_agrees_4978 : modelRow 4978 = row 4978 := model_count_4978.trans row_count_4978.symm
def row_4979 : RowCertificate 4979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4979 1) := by change ¬ Nat.Prime 99171723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33057241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4979 2) := by change ¬ Nat.Prime 99161765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19832353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4979 3) := by change Nat.Prime 99151807; exact ProjectEulerSolutions.P58PrimeNodes.p_99151807.prime
    exact decide_eq_true prime
theorem row_count_4979 : row 4979 = 1 := (row_4979.checked).trans rfl
theorem model_count_4979 : modelRow 4979 = 1 := by decide
theorem modelrow_agrees_4979 : modelRow 4979 = row 4979 := model_count_4979.trans row_count_4979.symm
def row_4980 : RowCertificate 4980 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4980 1) := by change Nat.Prime 99211561; exact ProjectEulerSolutions.P58PrimeNodes.p_99211561.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4980 2) := by change ¬ Nat.Prime 99201601; exact PrattCertificates.composite_of_product (factor := 3697) (cofactor := 26833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4980 3) := by change ¬ Nat.Prime 99191641; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 249853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4980 : row 4980 = 1 := (row_4980.checked).trans rfl
theorem model_count_4980 : modelRow 4980 = 1 := by decide
theorem modelrow_agrees_4980 : modelRow 4980 = row 4980 := model_count_4980.trans row_count_4980.symm
def row_4981 : RowCertificate 4981 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4981 1) := by change Nat.Prime 99251407; exact ProjectEulerSolutions.P58PrimeNodes.p_99251407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4981 2) := by change ¬ Nat.Prime 99241445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19848289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4981 3) := by change ¬ Nat.Prime 99231483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33077161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4981 : row 4981 = 1 := (row_4981.checked).trans rfl
theorem model_count_4981 : modelRow 4981 = 1 := by decide
theorem modelrow_agrees_4981 : modelRow 4981 = row 4981 := model_count_4981.trans row_count_4981.symm
def row_4982 : RowCertificate 4982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4982 1) := by change ¬ Nat.Prime 99291261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33097087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4982 2) := by change ¬ Nat.Prime 99281297; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 3423493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4982 3) := by change ¬ Nat.Prime 99271333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14181619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4982 : row 4982 = 0 := (row_4982.checked).trans rfl
theorem model_count_4982 : modelRow 4982 = 0 := by decide
theorem modelrow_agrees_4982 : modelRow 4982 = row 4982 := model_count_4982.trans row_count_4982.symm
def row_4983 : RowCertificate 4983 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4983 1) := by change Nat.Prime 99331123; exact ProjectEulerSolutions.P58PrimeNodes.p_99331123.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4983 2) := by change ¬ Nat.Prime 99321157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7640089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4983 3) := by change ¬ Nat.Prime 99311191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14187313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4983 : row 4983 = 1 := (row_4983.checked).trans rfl
theorem model_count_4983 : modelRow 4983 = 1 := by decide
theorem modelrow_agrees_4983 : modelRow 4983 = row 4983 := model_count_4983.trans row_count_4983.symm
def row_4984 : RowCertificate 4984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4984 1) := by change ¬ Nat.Prime 99370993; exact PrattCertificates.composite_of_product (factor := 7699) (cofactor := 12907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4984 2) := by change ¬ Nat.Prime 99361025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19872205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4984 3) := by change ¬ Nat.Prime 99351057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33117019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4984 : row 4984 = 0 := (row_4984.checked).trans rfl
theorem model_count_4984 : modelRow 4984 = 0 := by decide
theorem modelrow_agrees_4984 : modelRow 4984 = row 4984 := model_count_4984.trans row_count_4984.symm
def row_4985 : RowCertificate 4985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4985 1) := by change ¬ Nat.Prime 99410871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33136957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4985 2) := by change Nat.Prime 99400901; exact ProjectEulerSolutions.P58PrimeNodes.p_99400901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4985 3) := by change ¬ Nat.Prime 99390931; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 2311417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4985 : row 4985 = 1 := (row_4985.checked).trans rfl
theorem model_count_4985 : modelRow 4985 = 1 := by decide
theorem modelrow_agrees_4985 : modelRow 4985 = row 4985 := model_count_4985.trans row_count_4985.symm
def row_4986 : RowCertificate 4986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4986 1) := by change ¬ Nat.Prime 99450757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14207251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4986 2) := by change ¬ Nat.Prime 99440785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19888157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4986 3) := by change Nat.Prime 99430813; exact ProjectEulerSolutions.P58PrimeNodes.p_99430813.prime
    exact decide_eq_true prime
theorem row_count_4986 : row 4986 = 1 := (row_4986.checked).trans rfl
theorem model_count_4986 : modelRow 4986 = 1 := by decide
theorem modelrow_agrees_4986 : modelRow 4986 = row 4986 := model_count_4986.trans row_count_4986.symm
def row_4987 : RowCertificate 4987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4987 1) := by change ¬ Nat.Prime 99490651; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7653127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4987 2) := by change ¬ Nat.Prime 99480677; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1362749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4987 3) := by change ¬ Nat.Prime 99470703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33156901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4987 : row 4987 = 0 := (row_4987.checked).trans rfl
theorem model_count_4987 : modelRow 4987 = 0 := by decide
theorem modelrow_agrees_4987 : modelRow 4987 = row 4987 := model_count_4987.trans row_count_4987.symm
def row_4988 : RowCertificate 4988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4988 1) := by change ¬ Nat.Prime 99530553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33176851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4988 2) := by change ¬ Nat.Prime 99520577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7655429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4988 3) := by change Nat.Prime 99510601; exact ProjectEulerSolutions.P58PrimeNodes.p_99510601.prime
    exact decide_eq_true prime
theorem row_count_4988 : row 4988 = 1 := (row_4988.checked).trans rfl
theorem model_count_4988 : modelRow 4988 = 1 := by decide
theorem modelrow_agrees_4988 : modelRow 4988 = row 4988 := model_count_4988.trans row_count_4988.symm
def row_4989 : RowCertificate 4989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4989 1) := by change ¬ Nat.Prime 99570463; exact PrattCertificates.composite_of_product (factor := 3307) (cofactor := 30109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4989 2) := by change ¬ Nat.Prime 99560485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19912097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4989 3) := by change ¬ Nat.Prime 99550507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14221501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4989 : row 4989 = 0 := (row_4989.checked).trans rfl
theorem model_count_4989 : modelRow 4989 = 0 := by decide
theorem modelrow_agrees_4989 : modelRow 4989 = row 4989 := model_count_4989.trans row_count_4989.symm
def row_4990 : RowCertificate 4990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4990 1) := by change ¬ Nat.Prime 99610381; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 7662337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4990 2) := by change Nat.Prime 99600401; exact ProjectEulerSolutions.P58PrimeNodes.p_99600401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4990 3) := by change ¬ Nat.Prime 99590421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33196807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4990 : row 4990 = 1 := (row_4990.checked).trans rfl
theorem model_count_4990 : modelRow 4990 = 1 := by decide
theorem modelrow_agrees_4990 : modelRow 4990 = row 4990 := model_count_4990.trans row_count_4990.symm
def row_4991 : RowCertificate 4991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4991 1) := by change ¬ Nat.Prime 99650307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 33216769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4991 2) := by change ¬ Nat.Prime 99640325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 19928065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4991 3) := by change ¬ Nat.Prime 99630343; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 435067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4991 : row 4991 = 0 := (row_4991.checked).trans rfl
theorem model_count_4991 : modelRow 4991 = 0 := by decide
theorem modelrow_agrees_4991 : modelRow 4991 = row 4991 := model_count_4991.trans row_count_4991.symm
def row_4992 : RowCertificate 4992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4992 1) := by change ¬ Nat.Prime 99690241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 14241463) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4992 2) := by change ¬ Nat.Prime 99680257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2694061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4992 3) := by change ¬ Nat.Prime 99670273; exact PrattCertificates.composite_of_product (factor := 3181) (cofactor := 31333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4992 : row 4992 = 0 := (row_4992.checked).trans rfl
theorem model_count_4992 : modelRow 4992 = 0 := by decide
theorem modelrow_agrees_4992 : modelRow 4992 = row 4992 := model_count_4992.trans row_count_4992.symm
private def atomic_4865 : CountSegment 1 10 13120 4864 4865 2192 2192 := CountSegment.single (first := 4864) (initial := 2192) row_count_4865 (by decide)
private theorem atomic_4865_agreement : RowAgreement 4864 4865 := RowAgreement.single (first := 4864) modelrow_agrees_4865
private def atomic_4866 : CountSegment 1 10 13120 4865 4866 2192 2192 := CountSegment.single (first := 4865) (initial := 2192) row_count_4866 (by decide)
private theorem atomic_4866_agreement : RowAgreement 4865 4866 := RowAgreement.single (first := 4865) modelrow_agrees_4866
private def atomic_4867 : CountSegment 1 10 13120 4866 4867 2192 2192 := CountSegment.single (first := 4866) (initial := 2192) row_count_4867 (by decide)
private theorem atomic_4867_agreement : RowAgreement 4866 4867 := RowAgreement.single (first := 4866) modelrow_agrees_4867
private def atomic_4868 : CountSegment 1 10 13120 4867 4868 2192 2192 := CountSegment.single (first := 4867) (initial := 2192) row_count_4868 (by decide)
private theorem atomic_4868_agreement : RowAgreement 4867 4868 := RowAgreement.single (first := 4867) modelrow_agrees_4868
private def atomic_4869 : CountSegment 1 10 13120 4868 4869 2192 2192 := CountSegment.single (first := 4868) (initial := 2192) row_count_4869 (by decide)
private theorem atomic_4869_agreement : RowAgreement 4868 4869 := RowAgreement.single (first := 4868) modelrow_agrees_4869
private def atomic_4870 : CountSegment 1 10 13120 4869 4870 2192 2192 := CountSegment.single (first := 4869) (initial := 2192) row_count_4870 (by decide)
private theorem atomic_4870_agreement : RowAgreement 4869 4870 := RowAgreement.single (first := 4869) modelrow_agrees_4870
private def atomic_4871 : CountSegment 1 10 13120 4870 4871 2192 2192 := CountSegment.single (first := 4870) (initial := 2192) row_count_4871 (by decide)
private theorem atomic_4871_agreement : RowAgreement 4870 4871 := RowAgreement.single (first := 4870) modelrow_agrees_4871
private def atomic_4872 : CountSegment 1 10 13120 4871 4872 2192 2194 := CountSegment.single (first := 4871) (initial := 2192) row_count_4872 (by decide)
private theorem atomic_4872_agreement : RowAgreement 4871 4872 := RowAgreement.single (first := 4871) modelrow_agrees_4872
private def atomic_4873 : CountSegment 1 10 13120 4872 4873 2194 2194 := CountSegment.single (first := 4872) (initial := 2194) row_count_4873 (by decide)
private theorem atomic_4873_agreement : RowAgreement 4872 4873 := RowAgreement.single (first := 4872) modelrow_agrees_4873
private def atomic_4874 : CountSegment 1 10 13120 4873 4874 2194 2195 := CountSegment.single (first := 4873) (initial := 2194) row_count_4874 (by decide)
private theorem atomic_4874_agreement : RowAgreement 4873 4874 := RowAgreement.single (first := 4873) modelrow_agrees_4874
private def atomic_4875 : CountSegment 1 10 13120 4874 4875 2195 2197 := CountSegment.single (first := 4874) (initial := 2195) row_count_4875 (by decide)
private theorem atomic_4875_agreement : RowAgreement 4874 4875 := RowAgreement.single (first := 4874) modelrow_agrees_4875
private def atomic_4876 : CountSegment 1 10 13120 4875 4876 2197 2197 := CountSegment.single (first := 4875) (initial := 2197) row_count_4876 (by decide)
private theorem atomic_4876_agreement : RowAgreement 4875 4876 := RowAgreement.single (first := 4875) modelrow_agrees_4876
private def atomic_4877 : CountSegment 1 10 13120 4876 4877 2197 2197 := CountSegment.single (first := 4876) (initial := 2197) row_count_4877 (by decide)
private theorem atomic_4877_agreement : RowAgreement 4876 4877 := RowAgreement.single (first := 4876) modelrow_agrees_4877
private def atomic_4878 : CountSegment 1 10 13120 4877 4878 2197 2198 := CountSegment.single (first := 4877) (initial := 2197) row_count_4878 (by decide)
private theorem atomic_4878_agreement : RowAgreement 4877 4878 := RowAgreement.single (first := 4877) modelrow_agrees_4878
private def atomic_4879 : CountSegment 1 10 13120 4878 4879 2198 2198 := CountSegment.single (first := 4878) (initial := 2198) row_count_4879 (by decide)
private theorem atomic_4879_agreement : RowAgreement 4878 4879 := RowAgreement.single (first := 4878) modelrow_agrees_4879
private def atomic_4880 : CountSegment 1 10 13120 4879 4880 2198 2199 := CountSegment.single (first := 4879) (initial := 2198) row_count_4880 (by decide)
private theorem atomic_4880_agreement : RowAgreement 4879 4880 := RowAgreement.single (first := 4879) modelrow_agrees_4880
private def atomic_4881 : CountSegment 1 10 13120 4880 4881 2199 2199 := CountSegment.single (first := 4880) (initial := 2199) row_count_4881 (by decide)
private theorem atomic_4881_agreement : RowAgreement 4880 4881 := RowAgreement.single (first := 4880) modelrow_agrees_4881
private def atomic_4882 : CountSegment 1 10 13120 4881 4882 2199 2199 := CountSegment.single (first := 4881) (initial := 2199) row_count_4882 (by decide)
private theorem atomic_4882_agreement : RowAgreement 4881 4882 := RowAgreement.single (first := 4881) modelrow_agrees_4882
private def atomic_4883 : CountSegment 1 10 13120 4882 4883 2199 2199 := CountSegment.single (first := 4882) (initial := 2199) row_count_4883 (by decide)
private theorem atomic_4883_agreement : RowAgreement 4882 4883 := RowAgreement.single (first := 4882) modelrow_agrees_4883
private def atomic_4884 : CountSegment 1 10 13120 4883 4884 2199 2200 := CountSegment.single (first := 4883) (initial := 2199) row_count_4884 (by decide)
private theorem atomic_4884_agreement : RowAgreement 4883 4884 := RowAgreement.single (first := 4883) modelrow_agrees_4884
private def atomic_4885 : CountSegment 1 10 13120 4884 4885 2200 2201 := CountSegment.single (first := 4884) (initial := 2200) row_count_4885 (by decide)
private theorem atomic_4885_agreement : RowAgreement 4884 4885 := RowAgreement.single (first := 4884) modelrow_agrees_4885
private def atomic_4886 : CountSegment 1 10 13120 4885 4886 2201 2201 := CountSegment.single (first := 4885) (initial := 2201) row_count_4886 (by decide)
private theorem atomic_4886_agreement : RowAgreement 4885 4886 := RowAgreement.single (first := 4885) modelrow_agrees_4886
private def atomic_4887 : CountSegment 1 10 13120 4886 4887 2201 2201 := CountSegment.single (first := 4886) (initial := 2201) row_count_4887 (by decide)
private theorem atomic_4887_agreement : RowAgreement 4886 4887 := RowAgreement.single (first := 4886) modelrow_agrees_4887
private def atomic_4888 : CountSegment 1 10 13120 4887 4888 2201 2201 := CountSegment.single (first := 4887) (initial := 2201) row_count_4888 (by decide)
private theorem atomic_4888_agreement : RowAgreement 4887 4888 := RowAgreement.single (first := 4887) modelrow_agrees_4888
private def atomic_4889 : CountSegment 1 10 13120 4888 4889 2201 2201 := CountSegment.single (first := 4888) (initial := 2201) row_count_4889 (by decide)
private theorem atomic_4889_agreement : RowAgreement 4888 4889 := RowAgreement.single (first := 4888) modelrow_agrees_4889
private def atomic_4890 : CountSegment 1 10 13120 4889 4890 2201 2202 := CountSegment.single (first := 4889) (initial := 2201) row_count_4890 (by decide)
private theorem atomic_4890_agreement : RowAgreement 4889 4890 := RowAgreement.single (first := 4889) modelrow_agrees_4890
private def atomic_4891 : CountSegment 1 10 13120 4890 4891 2202 2202 := CountSegment.single (first := 4890) (initial := 2202) row_count_4891 (by decide)
private theorem atomic_4891_agreement : RowAgreement 4890 4891 := RowAgreement.single (first := 4890) modelrow_agrees_4891
private def atomic_4892 : CountSegment 1 10 13120 4891 4892 2202 2202 := CountSegment.single (first := 4891) (initial := 2202) row_count_4892 (by decide)
private theorem atomic_4892_agreement : RowAgreement 4891 4892 := RowAgreement.single (first := 4891) modelrow_agrees_4892
private def atomic_4893 : CountSegment 1 10 13120 4892 4893 2202 2203 := CountSegment.single (first := 4892) (initial := 2202) row_count_4893 (by decide)
private theorem atomic_4893_agreement : RowAgreement 4892 4893 := RowAgreement.single (first := 4892) modelrow_agrees_4893
private def atomic_4894 : CountSegment 1 10 13120 4893 4894 2203 2203 := CountSegment.single (first := 4893) (initial := 2203) row_count_4894 (by decide)
private theorem atomic_4894_agreement : RowAgreement 4893 4894 := RowAgreement.single (first := 4893) modelrow_agrees_4894
private def atomic_4895 : CountSegment 1 10 13120 4894 4895 2203 2204 := CountSegment.single (first := 4894) (initial := 2203) row_count_4895 (by decide)
private theorem atomic_4895_agreement : RowAgreement 4894 4895 := RowAgreement.single (first := 4894) modelrow_agrees_4895
private def atomic_4896 : CountSegment 1 10 13120 4895 4896 2204 2204 := CountSegment.single (first := 4895) (initial := 2204) row_count_4896 (by decide)
private theorem atomic_4896_agreement : RowAgreement 4895 4896 := RowAgreement.single (first := 4895) modelrow_agrees_4896
private def atomic_4897 : CountSegment 1 10 13120 4896 4897 2204 2204 := CountSegment.single (first := 4896) (initial := 2204) row_count_4897 (by decide)
private theorem atomic_4897_agreement : RowAgreement 4896 4897 := RowAgreement.single (first := 4896) modelrow_agrees_4897
private def atomic_4898 : CountSegment 1 10 13120 4897 4898 2204 2204 := CountSegment.single (first := 4897) (initial := 2204) row_count_4898 (by decide)
private theorem atomic_4898_agreement : RowAgreement 4897 4898 := RowAgreement.single (first := 4897) modelrow_agrees_4898
private def atomic_4899 : CountSegment 1 10 13120 4898 4899 2204 2204 := CountSegment.single (first := 4898) (initial := 2204) row_count_4899 (by decide)
private theorem atomic_4899_agreement : RowAgreement 4898 4899 := RowAgreement.single (first := 4898) modelrow_agrees_4899
private def atomic_4900 : CountSegment 1 10 13120 4899 4900 2204 2205 := CountSegment.single (first := 4899) (initial := 2204) row_count_4900 (by decide)
private theorem atomic_4900_agreement : RowAgreement 4899 4900 := RowAgreement.single (first := 4899) modelrow_agrees_4900
private def atomic_4901 : CountSegment 1 10 13120 4900 4901 2205 2205 := CountSegment.single (first := 4900) (initial := 2205) row_count_4901 (by decide)
private theorem atomic_4901_agreement : RowAgreement 4900 4901 := RowAgreement.single (first := 4900) modelrow_agrees_4901
private def atomic_4902 : CountSegment 1 10 13120 4901 4902 2205 2207 := CountSegment.single (first := 4901) (initial := 2205) row_count_4902 (by decide)
private theorem atomic_4902_agreement : RowAgreement 4901 4902 := RowAgreement.single (first := 4901) modelrow_agrees_4902
private def atomic_4903 : CountSegment 1 10 13120 4902 4903 2207 2208 := CountSegment.single (first := 4902) (initial := 2207) row_count_4903 (by decide)
private theorem atomic_4903_agreement : RowAgreement 4902 4903 := RowAgreement.single (first := 4902) modelrow_agrees_4903
private def atomic_4904 : CountSegment 1 10 13120 4903 4904 2208 2208 := CountSegment.single (first := 4903) (initial := 2208) row_count_4904 (by decide)
private theorem atomic_4904_agreement : RowAgreement 4903 4904 := RowAgreement.single (first := 4903) modelrow_agrees_4904
private def atomic_4905 : CountSegment 1 10 13120 4904 4905 2208 2208 := CountSegment.single (first := 4904) (initial := 2208) row_count_4905 (by decide)
private theorem atomic_4905_agreement : RowAgreement 4904 4905 := RowAgreement.single (first := 4904) modelrow_agrees_4905
private def atomic_4906 : CountSegment 1 10 13120 4905 4906 2208 2208 := CountSegment.single (first := 4905) (initial := 2208) row_count_4906 (by decide)
private theorem atomic_4906_agreement : RowAgreement 4905 4906 := RowAgreement.single (first := 4905) modelrow_agrees_4906
private def atomic_4907 : CountSegment 1 10 13120 4906 4907 2208 2208 := CountSegment.single (first := 4906) (initial := 2208) row_count_4907 (by decide)
private theorem atomic_4907_agreement : RowAgreement 4906 4907 := RowAgreement.single (first := 4906) modelrow_agrees_4907
private def atomic_4908 : CountSegment 1 10 13120 4907 4908 2208 2208 := CountSegment.single (first := 4907) (initial := 2208) row_count_4908 (by decide)
private theorem atomic_4908_agreement : RowAgreement 4907 4908 := RowAgreement.single (first := 4907) modelrow_agrees_4908
private def atomic_4909 : CountSegment 1 10 13120 4908 4909 2208 2208 := CountSegment.single (first := 4908) (initial := 2208) row_count_4909 (by decide)
private theorem atomic_4909_agreement : RowAgreement 4908 4909 := RowAgreement.single (first := 4908) modelrow_agrees_4909
private def atomic_4910 : CountSegment 1 10 13120 4909 4910 2208 2208 := CountSegment.single (first := 4909) (initial := 2208) row_count_4910 (by decide)
private theorem atomic_4910_agreement : RowAgreement 4909 4910 := RowAgreement.single (first := 4909) modelrow_agrees_4910
private def atomic_4911 : CountSegment 1 10 13120 4910 4911 2208 2208 := CountSegment.single (first := 4910) (initial := 2208) row_count_4911 (by decide)
private theorem atomic_4911_agreement : RowAgreement 4910 4911 := RowAgreement.single (first := 4910) modelrow_agrees_4911
private def atomic_4912 : CountSegment 1 10 13120 4911 4912 2208 2208 := CountSegment.single (first := 4911) (initial := 2208) row_count_4912 (by decide)
private theorem atomic_4912_agreement : RowAgreement 4911 4912 := RowAgreement.single (first := 4911) modelrow_agrees_4912
private def atomic_4913 : CountSegment 1 10 13120 4912 4913 2208 2209 := CountSegment.single (first := 4912) (initial := 2208) row_count_4913 (by decide)
private theorem atomic_4913_agreement : RowAgreement 4912 4913 := RowAgreement.single (first := 4912) modelrow_agrees_4913
private def atomic_4914 : CountSegment 1 10 13120 4913 4914 2209 2210 := CountSegment.single (first := 4913) (initial := 2209) row_count_4914 (by decide)
private theorem atomic_4914_agreement : RowAgreement 4913 4914 := RowAgreement.single (first := 4913) modelrow_agrees_4914
private def atomic_4915 : CountSegment 1 10 13120 4914 4915 2210 2210 := CountSegment.single (first := 4914) (initial := 2210) row_count_4915 (by decide)
private theorem atomic_4915_agreement : RowAgreement 4914 4915 := RowAgreement.single (first := 4914) modelrow_agrees_4915
private def atomic_4916 : CountSegment 1 10 13120 4915 4916 2210 2210 := CountSegment.single (first := 4915) (initial := 2210) row_count_4916 (by decide)
private theorem atomic_4916_agreement : RowAgreement 4915 4916 := RowAgreement.single (first := 4915) modelrow_agrees_4916
private def atomic_4917 : CountSegment 1 10 13120 4916 4917 2210 2210 := CountSegment.single (first := 4916) (initial := 2210) row_count_4917 (by decide)
private theorem atomic_4917_agreement : RowAgreement 4916 4917 := RowAgreement.single (first := 4916) modelrow_agrees_4917
private def atomic_4918 : CountSegment 1 10 13120 4917 4918 2210 2210 := CountSegment.single (first := 4917) (initial := 2210) row_count_4918 (by decide)
private theorem atomic_4918_agreement : RowAgreement 4917 4918 := RowAgreement.single (first := 4917) modelrow_agrees_4918
private def atomic_4919 : CountSegment 1 10 13120 4918 4919 2210 2210 := CountSegment.single (first := 4918) (initial := 2210) row_count_4919 (by decide)
private theorem atomic_4919_agreement : RowAgreement 4918 4919 := RowAgreement.single (first := 4918) modelrow_agrees_4919
private def atomic_4920 : CountSegment 1 10 13120 4919 4920 2210 2210 := CountSegment.single (first := 4919) (initial := 2210) row_count_4920 (by decide)
private theorem atomic_4920_agreement : RowAgreement 4919 4920 := RowAgreement.single (first := 4919) modelrow_agrees_4920
private def atomic_4921 : CountSegment 1 10 13120 4920 4921 2210 2211 := CountSegment.single (first := 4920) (initial := 2210) row_count_4921 (by decide)
private theorem atomic_4921_agreement : RowAgreement 4920 4921 := RowAgreement.single (first := 4920) modelrow_agrees_4921
private def atomic_4922 : CountSegment 1 10 13120 4921 4922 2211 2212 := CountSegment.single (first := 4921) (initial := 2211) row_count_4922 (by decide)
private theorem atomic_4922_agreement : RowAgreement 4921 4922 := RowAgreement.single (first := 4921) modelrow_agrees_4922
private def atomic_4923 : CountSegment 1 10 13120 4922 4923 2212 2213 := CountSegment.single (first := 4922) (initial := 2212) row_count_4923 (by decide)
private theorem atomic_4923_agreement : RowAgreement 4922 4923 := RowAgreement.single (first := 4922) modelrow_agrees_4923
private def atomic_4924 : CountSegment 1 10 13120 4923 4924 2213 2214 := CountSegment.single (first := 4923) (initial := 2213) row_count_4924 (by decide)
private theorem atomic_4924_agreement : RowAgreement 4923 4924 := RowAgreement.single (first := 4923) modelrow_agrees_4924
private def atomic_4925 : CountSegment 1 10 13120 4924 4925 2214 2214 := CountSegment.single (first := 4924) (initial := 2214) row_count_4925 (by decide)
private theorem atomic_4925_agreement : RowAgreement 4924 4925 := RowAgreement.single (first := 4924) modelrow_agrees_4925
private def atomic_4926 : CountSegment 1 10 13120 4925 4926 2214 2214 := CountSegment.single (first := 4925) (initial := 2214) row_count_4926 (by decide)
private theorem atomic_4926_agreement : RowAgreement 4925 4926 := RowAgreement.single (first := 4925) modelrow_agrees_4926
private def atomic_4927 : CountSegment 1 10 13120 4926 4927 2214 2214 := CountSegment.single (first := 4926) (initial := 2214) row_count_4927 (by decide)
private theorem atomic_4927_agreement : RowAgreement 4926 4927 := RowAgreement.single (first := 4926) modelrow_agrees_4927
private def atomic_4928 : CountSegment 1 10 13120 4927 4928 2214 2214 := CountSegment.single (first := 4927) (initial := 2214) row_count_4928 (by decide)
private theorem atomic_4928_agreement : RowAgreement 4927 4928 := RowAgreement.single (first := 4927) modelrow_agrees_4928
private def atomic_4929 : CountSegment 1 10 13120 4928 4929 2214 2214 := CountSegment.single (first := 4928) (initial := 2214) row_count_4929 (by decide)
private theorem atomic_4929_agreement : RowAgreement 4928 4929 := RowAgreement.single (first := 4928) modelrow_agrees_4929
private def atomic_4930 : CountSegment 1 10 13120 4929 4930 2214 2215 := CountSegment.single (first := 4929) (initial := 2214) row_count_4930 (by decide)
private theorem atomic_4930_agreement : RowAgreement 4929 4930 := RowAgreement.single (first := 4929) modelrow_agrees_4930
private def atomic_4931 : CountSegment 1 10 13120 4930 4931 2215 2216 := CountSegment.single (first := 4930) (initial := 2215) row_count_4931 (by decide)
private theorem atomic_4931_agreement : RowAgreement 4930 4931 := RowAgreement.single (first := 4930) modelrow_agrees_4931
private def atomic_4932 : CountSegment 1 10 13120 4931 4932 2216 2217 := CountSegment.single (first := 4931) (initial := 2216) row_count_4932 (by decide)
private theorem atomic_4932_agreement : RowAgreement 4931 4932 := RowAgreement.single (first := 4931) modelrow_agrees_4932
private def atomic_4933 : CountSegment 1 10 13120 4932 4933 2217 2218 := CountSegment.single (first := 4932) (initial := 2217) row_count_4933 (by decide)
private theorem atomic_4933_agreement : RowAgreement 4932 4933 := RowAgreement.single (first := 4932) modelrow_agrees_4933
private def atomic_4934 : CountSegment 1 10 13120 4933 4934 2218 2218 := CountSegment.single (first := 4933) (initial := 2218) row_count_4934 (by decide)
private theorem atomic_4934_agreement : RowAgreement 4933 4934 := RowAgreement.single (first := 4933) modelrow_agrees_4934
private def atomic_4935 : CountSegment 1 10 13120 4934 4935 2218 2218 := CountSegment.single (first := 4934) (initial := 2218) row_count_4935 (by decide)
private theorem atomic_4935_agreement : RowAgreement 4934 4935 := RowAgreement.single (first := 4934) modelrow_agrees_4935
private def atomic_4936 : CountSegment 1 10 13120 4935 4936 2218 2218 := CountSegment.single (first := 4935) (initial := 2218) row_count_4936 (by decide)
private theorem atomic_4936_agreement : RowAgreement 4935 4936 := RowAgreement.single (first := 4935) modelrow_agrees_4936
private def atomic_4937 : CountSegment 1 10 13120 4936 4937 2218 2219 := CountSegment.single (first := 4936) (initial := 2218) row_count_4937 (by decide)
private theorem atomic_4937_agreement : RowAgreement 4936 4937 := RowAgreement.single (first := 4936) modelrow_agrees_4937
private def atomic_4938 : CountSegment 1 10 13120 4937 4938 2219 2221 := CountSegment.single (first := 4937) (initial := 2219) row_count_4938 (by decide)
private theorem atomic_4938_agreement : RowAgreement 4937 4938 := RowAgreement.single (first := 4937) modelrow_agrees_4938
private def atomic_4939 : CountSegment 1 10 13120 4938 4939 2221 2221 := CountSegment.single (first := 4938) (initial := 2221) row_count_4939 (by decide)
private theorem atomic_4939_agreement : RowAgreement 4938 4939 := RowAgreement.single (first := 4938) modelrow_agrees_4939
private def atomic_4940 : CountSegment 1 10 13120 4939 4940 2221 2222 := CountSegment.single (first := 4939) (initial := 2221) row_count_4940 (by decide)
private theorem atomic_4940_agreement : RowAgreement 4939 4940 := RowAgreement.single (first := 4939) modelrow_agrees_4940
private def atomic_4941 : CountSegment 1 10 13120 4940 4941 2222 2223 := CountSegment.single (first := 4940) (initial := 2222) row_count_4941 (by decide)
private theorem atomic_4941_agreement : RowAgreement 4940 4941 := RowAgreement.single (first := 4940) modelrow_agrees_4941
private def atomic_4942 : CountSegment 1 10 13120 4941 4942 2223 2223 := CountSegment.single (first := 4941) (initial := 2223) row_count_4942 (by decide)
private theorem atomic_4942_agreement : RowAgreement 4941 4942 := RowAgreement.single (first := 4941) modelrow_agrees_4942
private def atomic_4943 : CountSegment 1 10 13120 4942 4943 2223 2224 := CountSegment.single (first := 4942) (initial := 2223) row_count_4943 (by decide)
private theorem atomic_4943_agreement : RowAgreement 4942 4943 := RowAgreement.single (first := 4942) modelrow_agrees_4943
private def atomic_4944 : CountSegment 1 10 13120 4943 4944 2224 2224 := CountSegment.single (first := 4943) (initial := 2224) row_count_4944 (by decide)
private theorem atomic_4944_agreement : RowAgreement 4943 4944 := RowAgreement.single (first := 4943) modelrow_agrees_4944
private def atomic_4945 : CountSegment 1 10 13120 4944 4945 2224 2224 := CountSegment.single (first := 4944) (initial := 2224) row_count_4945 (by decide)
private theorem atomic_4945_agreement : RowAgreement 4944 4945 := RowAgreement.single (first := 4944) modelrow_agrees_4945
private def atomic_4946 : CountSegment 1 10 13120 4945 4946 2224 2224 := CountSegment.single (first := 4945) (initial := 2224) row_count_4946 (by decide)
private theorem atomic_4946_agreement : RowAgreement 4945 4946 := RowAgreement.single (first := 4945) modelrow_agrees_4946
private def atomic_4947 : CountSegment 1 10 13120 4946 4947 2224 2225 := CountSegment.single (first := 4946) (initial := 2224) row_count_4947 (by decide)
private theorem atomic_4947_agreement : RowAgreement 4946 4947 := RowAgreement.single (first := 4946) modelrow_agrees_4947
private def atomic_4948 : CountSegment 1 10 13120 4947 4948 2225 2226 := CountSegment.single (first := 4947) (initial := 2225) row_count_4948 (by decide)
private theorem atomic_4948_agreement : RowAgreement 4947 4948 := RowAgreement.single (first := 4947) modelrow_agrees_4948
private def atomic_4949 : CountSegment 1 10 13120 4948 4949 2226 2227 := CountSegment.single (first := 4948) (initial := 2226) row_count_4949 (by decide)
private theorem atomic_4949_agreement : RowAgreement 4948 4949 := RowAgreement.single (first := 4948) modelrow_agrees_4949
private def atomic_4950 : CountSegment 1 10 13120 4949 4950 2227 2229 := CountSegment.single (first := 4949) (initial := 2227) row_count_4950 (by decide)
private theorem atomic_4950_agreement : RowAgreement 4949 4950 := RowAgreement.single (first := 4949) modelrow_agrees_4950
private def atomic_4951 : CountSegment 1 10 13120 4950 4951 2229 2229 := CountSegment.single (first := 4950) (initial := 2229) row_count_4951 (by decide)
private theorem atomic_4951_agreement : RowAgreement 4950 4951 := RowAgreement.single (first := 4950) modelrow_agrees_4951
private def atomic_4952 : CountSegment 1 10 13120 4951 4952 2229 2230 := CountSegment.single (first := 4951) (initial := 2229) row_count_4952 (by decide)
private theorem atomic_4952_agreement : RowAgreement 4951 4952 := RowAgreement.single (first := 4951) modelrow_agrees_4952
private def atomic_4953 : CountSegment 1 10 13120 4952 4953 2230 2230 := CountSegment.single (first := 4952) (initial := 2230) row_count_4953 (by decide)
private theorem atomic_4953_agreement : RowAgreement 4952 4953 := RowAgreement.single (first := 4952) modelrow_agrees_4953
private def atomic_4954 : CountSegment 1 10 13120 4953 4954 2230 2231 := CountSegment.single (first := 4953) (initial := 2230) row_count_4954 (by decide)
private theorem atomic_4954_agreement : RowAgreement 4953 4954 := RowAgreement.single (first := 4953) modelrow_agrees_4954
private def atomic_4955 : CountSegment 1 10 13120 4954 4955 2231 2231 := CountSegment.single (first := 4954) (initial := 2231) row_count_4955 (by decide)
private theorem atomic_4955_agreement : RowAgreement 4954 4955 := RowAgreement.single (first := 4954) modelrow_agrees_4955
private def atomic_4956 : CountSegment 1 10 13120 4955 4956 2231 2232 := CountSegment.single (first := 4955) (initial := 2231) row_count_4956 (by decide)
private theorem atomic_4956_agreement : RowAgreement 4955 4956 := RowAgreement.single (first := 4955) modelrow_agrees_4956
private def atomic_4957 : CountSegment 1 10 13120 4956 4957 2232 2232 := CountSegment.single (first := 4956) (initial := 2232) row_count_4957 (by decide)
private theorem atomic_4957_agreement : RowAgreement 4956 4957 := RowAgreement.single (first := 4956) modelrow_agrees_4957
private def atomic_4958 : CountSegment 1 10 13120 4957 4958 2232 2232 := CountSegment.single (first := 4957) (initial := 2232) row_count_4958 (by decide)
private theorem atomic_4958_agreement : RowAgreement 4957 4958 := RowAgreement.single (first := 4957) modelrow_agrees_4958
private def atomic_4959 : CountSegment 1 10 13120 4958 4959 2232 2233 := CountSegment.single (first := 4958) (initial := 2232) row_count_4959 (by decide)
private theorem atomic_4959_agreement : RowAgreement 4958 4959 := RowAgreement.single (first := 4958) modelrow_agrees_4959
private def atomic_4960 : CountSegment 1 10 13120 4959 4960 2233 2233 := CountSegment.single (first := 4959) (initial := 2233) row_count_4960 (by decide)
private theorem atomic_4960_agreement : RowAgreement 4959 4960 := RowAgreement.single (first := 4959) modelrow_agrees_4960
private def atomic_4961 : CountSegment 1 10 13120 4960 4961 2233 2233 := CountSegment.single (first := 4960) (initial := 2233) row_count_4961 (by decide)
private theorem atomic_4961_agreement : RowAgreement 4960 4961 := RowAgreement.single (first := 4960) modelrow_agrees_4961
private def atomic_4962 : CountSegment 1 10 13120 4961 4962 2233 2234 := CountSegment.single (first := 4961) (initial := 2233) row_count_4962 (by decide)
private theorem atomic_4962_agreement : RowAgreement 4961 4962 := RowAgreement.single (first := 4961) modelrow_agrees_4962
private def atomic_4963 : CountSegment 1 10 13120 4962 4963 2234 2234 := CountSegment.single (first := 4962) (initial := 2234) row_count_4963 (by decide)
private theorem atomic_4963_agreement : RowAgreement 4962 4963 := RowAgreement.single (first := 4962) modelrow_agrees_4963
private def atomic_4964 : CountSegment 1 10 13120 4963 4964 2234 2234 := CountSegment.single (first := 4963) (initial := 2234) row_count_4964 (by decide)
private theorem atomic_4964_agreement : RowAgreement 4963 4964 := RowAgreement.single (first := 4963) modelrow_agrees_4964
private def atomic_4965 : CountSegment 1 10 13120 4964 4965 2234 2234 := CountSegment.single (first := 4964) (initial := 2234) row_count_4965 (by decide)
private theorem atomic_4965_agreement : RowAgreement 4964 4965 := RowAgreement.single (first := 4964) modelrow_agrees_4965
private def atomic_4966 : CountSegment 1 10 13120 4965 4966 2234 2234 := CountSegment.single (first := 4965) (initial := 2234) row_count_4966 (by decide)
private theorem atomic_4966_agreement : RowAgreement 4965 4966 := RowAgreement.single (first := 4965) modelrow_agrees_4966
private def atomic_4967 : CountSegment 1 10 13120 4966 4967 2234 2234 := CountSegment.single (first := 4966) (initial := 2234) row_count_4967 (by decide)
private theorem atomic_4967_agreement : RowAgreement 4966 4967 := RowAgreement.single (first := 4966) modelrow_agrees_4967
private def atomic_4968 : CountSegment 1 10 13120 4967 4968 2234 2234 := CountSegment.single (first := 4967) (initial := 2234) row_count_4968 (by decide)
private theorem atomic_4968_agreement : RowAgreement 4967 4968 := RowAgreement.single (first := 4967) modelrow_agrees_4968
private def atomic_4969 : CountSegment 1 10 13120 4968 4969 2234 2234 := CountSegment.single (first := 4968) (initial := 2234) row_count_4969 (by decide)
private theorem atomic_4969_agreement : RowAgreement 4968 4969 := RowAgreement.single (first := 4968) modelrow_agrees_4969
private def atomic_4970 : CountSegment 1 10 13120 4969 4970 2234 2234 := CountSegment.single (first := 4969) (initial := 2234) row_count_4970 (by decide)
private theorem atomic_4970_agreement : RowAgreement 4969 4970 := RowAgreement.single (first := 4969) modelrow_agrees_4970
private def atomic_4971 : CountSegment 1 10 13120 4970 4971 2234 2234 := CountSegment.single (first := 4970) (initial := 2234) row_count_4971 (by decide)
private theorem atomic_4971_agreement : RowAgreement 4970 4971 := RowAgreement.single (first := 4970) modelrow_agrees_4971
private def atomic_4972 : CountSegment 1 10 13120 4971 4972 2234 2234 := CountSegment.single (first := 4971) (initial := 2234) row_count_4972 (by decide)
private theorem atomic_4972_agreement : RowAgreement 4971 4972 := RowAgreement.single (first := 4971) modelrow_agrees_4972
private def atomic_4973 : CountSegment 1 10 13120 4972 4973 2234 2234 := CountSegment.single (first := 4972) (initial := 2234) row_count_4973 (by decide)
private theorem atomic_4973_agreement : RowAgreement 4972 4973 := RowAgreement.single (first := 4972) modelrow_agrees_4973
private def atomic_4974 : CountSegment 1 10 13120 4973 4974 2234 2235 := CountSegment.single (first := 4973) (initial := 2234) row_count_4974 (by decide)
private theorem atomic_4974_agreement : RowAgreement 4973 4974 := RowAgreement.single (first := 4973) modelrow_agrees_4974
private def atomic_4975 : CountSegment 1 10 13120 4974 4975 2235 2236 := CountSegment.single (first := 4974) (initial := 2235) row_count_4975 (by decide)
private theorem atomic_4975_agreement : RowAgreement 4974 4975 := RowAgreement.single (first := 4974) modelrow_agrees_4975
private def atomic_4976 : CountSegment 1 10 13120 4975 4976 2236 2236 := CountSegment.single (first := 4975) (initial := 2236) row_count_4976 (by decide)
private theorem atomic_4976_agreement : RowAgreement 4975 4976 := RowAgreement.single (first := 4975) modelrow_agrees_4976
private def atomic_4977 : CountSegment 1 10 13120 4976 4977 2236 2236 := CountSegment.single (first := 4976) (initial := 2236) row_count_4977 (by decide)
private theorem atomic_4977_agreement : RowAgreement 4976 4977 := RowAgreement.single (first := 4976) modelrow_agrees_4977
private def atomic_4978 : CountSegment 1 10 13120 4977 4978 2236 2237 := CountSegment.single (first := 4977) (initial := 2236) row_count_4978 (by decide)
private theorem atomic_4978_agreement : RowAgreement 4977 4978 := RowAgreement.single (first := 4977) modelrow_agrees_4978
private def atomic_4979 : CountSegment 1 10 13120 4978 4979 2237 2238 := CountSegment.single (first := 4978) (initial := 2237) row_count_4979 (by decide)
private theorem atomic_4979_agreement : RowAgreement 4978 4979 := RowAgreement.single (first := 4978) modelrow_agrees_4979
private def atomic_4980 : CountSegment 1 10 13120 4979 4980 2238 2239 := CountSegment.single (first := 4979) (initial := 2238) row_count_4980 (by decide)
private theorem atomic_4980_agreement : RowAgreement 4979 4980 := RowAgreement.single (first := 4979) modelrow_agrees_4980
private def atomic_4981 : CountSegment 1 10 13120 4980 4981 2239 2240 := CountSegment.single (first := 4980) (initial := 2239) row_count_4981 (by decide)
private theorem atomic_4981_agreement : RowAgreement 4980 4981 := RowAgreement.single (first := 4980) modelrow_agrees_4981
private def atomic_4982 : CountSegment 1 10 13120 4981 4982 2240 2240 := CountSegment.single (first := 4981) (initial := 2240) row_count_4982 (by decide)
private theorem atomic_4982_agreement : RowAgreement 4981 4982 := RowAgreement.single (first := 4981) modelrow_agrees_4982
private def atomic_4983 : CountSegment 1 10 13120 4982 4983 2240 2241 := CountSegment.single (first := 4982) (initial := 2240) row_count_4983 (by decide)
private theorem atomic_4983_agreement : RowAgreement 4982 4983 := RowAgreement.single (first := 4982) modelrow_agrees_4983
private def atomic_4984 : CountSegment 1 10 13120 4983 4984 2241 2241 := CountSegment.single (first := 4983) (initial := 2241) row_count_4984 (by decide)
private theorem atomic_4984_agreement : RowAgreement 4983 4984 := RowAgreement.single (first := 4983) modelrow_agrees_4984
private def atomic_4985 : CountSegment 1 10 13120 4984 4985 2241 2242 := CountSegment.single (first := 4984) (initial := 2241) row_count_4985 (by decide)
private theorem atomic_4985_agreement : RowAgreement 4984 4985 := RowAgreement.single (first := 4984) modelrow_agrees_4985
private def atomic_4986 : CountSegment 1 10 13120 4985 4986 2242 2243 := CountSegment.single (first := 4985) (initial := 2242) row_count_4986 (by decide)
private theorem atomic_4986_agreement : RowAgreement 4985 4986 := RowAgreement.single (first := 4985) modelrow_agrees_4986
private def atomic_4987 : CountSegment 1 10 13120 4986 4987 2243 2243 := CountSegment.single (first := 4986) (initial := 2243) row_count_4987 (by decide)
private theorem atomic_4987_agreement : RowAgreement 4986 4987 := RowAgreement.single (first := 4986) modelrow_agrees_4987
private def atomic_4988 : CountSegment 1 10 13120 4987 4988 2243 2244 := CountSegment.single (first := 4987) (initial := 2243) row_count_4988 (by decide)
private theorem atomic_4988_agreement : RowAgreement 4987 4988 := RowAgreement.single (first := 4987) modelrow_agrees_4988
private def atomic_4989 : CountSegment 1 10 13120 4988 4989 2244 2244 := CountSegment.single (first := 4988) (initial := 2244) row_count_4989 (by decide)
private theorem atomic_4989_agreement : RowAgreement 4988 4989 := RowAgreement.single (first := 4988) modelrow_agrees_4989
private def atomic_4990 : CountSegment 1 10 13120 4989 4990 2244 2245 := CountSegment.single (first := 4989) (initial := 2244) row_count_4990 (by decide)
private theorem atomic_4990_agreement : RowAgreement 4989 4990 := RowAgreement.single (first := 4989) modelrow_agrees_4990
private def atomic_4991 : CountSegment 1 10 13120 4990 4991 2245 2245 := CountSegment.single (first := 4990) (initial := 2245) row_count_4991 (by decide)
private theorem atomic_4991_agreement : RowAgreement 4990 4991 := RowAgreement.single (first := 4990) modelrow_agrees_4991
private def atomic_4992 : CountSegment 1 10 13120 4991 4992 2245 2245 := CountSegment.single (first := 4991) (initial := 2245) row_count_4992 (by decide)
private theorem atomic_4992_agreement : RowAgreement 4991 4992 := RowAgreement.single (first := 4991) modelrow_agrees_4992
private def within_038_000 : CountSegment 1 10 13120 4864 4866 2192 2192 := CountSegment.join atomic_4865 atomic_4866
private theorem within_038_000_agreement : RowAgreement 4864 4866 := RowAgreement.join atomic_4865_agreement atomic_4866_agreement
private def within_038_001 : CountSegment 1 10 13120 4866 4868 2192 2192 := CountSegment.join atomic_4867 atomic_4868
private theorem within_038_001_agreement : RowAgreement 4866 4868 := RowAgreement.join atomic_4867_agreement atomic_4868_agreement
private def within_038_002 : CountSegment 1 10 13120 4864 4868 2192 2192 := CountSegment.join within_038_000 within_038_001
private theorem within_038_002_agreement : RowAgreement 4864 4868 := RowAgreement.join within_038_000_agreement within_038_001_agreement
private def within_038_003 : CountSegment 1 10 13120 4868 4870 2192 2192 := CountSegment.join atomic_4869 atomic_4870
private theorem within_038_003_agreement : RowAgreement 4868 4870 := RowAgreement.join atomic_4869_agreement atomic_4870_agreement
private def within_038_004 : CountSegment 1 10 13120 4870 4872 2192 2194 := CountSegment.join atomic_4871 atomic_4872
private theorem within_038_004_agreement : RowAgreement 4870 4872 := RowAgreement.join atomic_4871_agreement atomic_4872_agreement
private def within_038_005 : CountSegment 1 10 13120 4868 4872 2192 2194 := CountSegment.join within_038_003 within_038_004
private theorem within_038_005_agreement : RowAgreement 4868 4872 := RowAgreement.join within_038_003_agreement within_038_004_agreement
private def within_038_006 : CountSegment 1 10 13120 4864 4872 2192 2194 := CountSegment.join within_038_002 within_038_005
private theorem within_038_006_agreement : RowAgreement 4864 4872 := RowAgreement.join within_038_002_agreement within_038_005_agreement
private def within_038_007 : CountSegment 1 10 13120 4872 4874 2194 2195 := CountSegment.join atomic_4873 atomic_4874
private theorem within_038_007_agreement : RowAgreement 4872 4874 := RowAgreement.join atomic_4873_agreement atomic_4874_agreement
private def within_038_008 : CountSegment 1 10 13120 4874 4876 2195 2197 := CountSegment.join atomic_4875 atomic_4876
private theorem within_038_008_agreement : RowAgreement 4874 4876 := RowAgreement.join atomic_4875_agreement atomic_4876_agreement
private def within_038_009 : CountSegment 1 10 13120 4872 4876 2194 2197 := CountSegment.join within_038_007 within_038_008
private theorem within_038_009_agreement : RowAgreement 4872 4876 := RowAgreement.join within_038_007_agreement within_038_008_agreement
private def within_038_010 : CountSegment 1 10 13120 4876 4878 2197 2198 := CountSegment.join atomic_4877 atomic_4878
private theorem within_038_010_agreement : RowAgreement 4876 4878 := RowAgreement.join atomic_4877_agreement atomic_4878_agreement
private def within_038_011 : CountSegment 1 10 13120 4878 4880 2198 2199 := CountSegment.join atomic_4879 atomic_4880
private theorem within_038_011_agreement : RowAgreement 4878 4880 := RowAgreement.join atomic_4879_agreement atomic_4880_agreement
private def within_038_012 : CountSegment 1 10 13120 4876 4880 2197 2199 := CountSegment.join within_038_010 within_038_011
private theorem within_038_012_agreement : RowAgreement 4876 4880 := RowAgreement.join within_038_010_agreement within_038_011_agreement
private def within_038_013 : CountSegment 1 10 13120 4872 4880 2194 2199 := CountSegment.join within_038_009 within_038_012
private theorem within_038_013_agreement : RowAgreement 4872 4880 := RowAgreement.join within_038_009_agreement within_038_012_agreement
private def within_038_014 : CountSegment 1 10 13120 4864 4880 2192 2199 := CountSegment.join within_038_006 within_038_013
private theorem within_038_014_agreement : RowAgreement 4864 4880 := RowAgreement.join within_038_006_agreement within_038_013_agreement
private def within_038_015 : CountSegment 1 10 13120 4880 4882 2199 2199 := CountSegment.join atomic_4881 atomic_4882
private theorem within_038_015_agreement : RowAgreement 4880 4882 := RowAgreement.join atomic_4881_agreement atomic_4882_agreement
private def within_038_016 : CountSegment 1 10 13120 4882 4884 2199 2200 := CountSegment.join atomic_4883 atomic_4884
private theorem within_038_016_agreement : RowAgreement 4882 4884 := RowAgreement.join atomic_4883_agreement atomic_4884_agreement
private def within_038_017 : CountSegment 1 10 13120 4880 4884 2199 2200 := CountSegment.join within_038_015 within_038_016
private theorem within_038_017_agreement : RowAgreement 4880 4884 := RowAgreement.join within_038_015_agreement within_038_016_agreement
private def within_038_018 : CountSegment 1 10 13120 4884 4886 2200 2201 := CountSegment.join atomic_4885 atomic_4886
private theorem within_038_018_agreement : RowAgreement 4884 4886 := RowAgreement.join atomic_4885_agreement atomic_4886_agreement
private def within_038_019 : CountSegment 1 10 13120 4886 4888 2201 2201 := CountSegment.join atomic_4887 atomic_4888
private theorem within_038_019_agreement : RowAgreement 4886 4888 := RowAgreement.join atomic_4887_agreement atomic_4888_agreement
private def within_038_020 : CountSegment 1 10 13120 4884 4888 2200 2201 := CountSegment.join within_038_018 within_038_019
private theorem within_038_020_agreement : RowAgreement 4884 4888 := RowAgreement.join within_038_018_agreement within_038_019_agreement
private def within_038_021 : CountSegment 1 10 13120 4880 4888 2199 2201 := CountSegment.join within_038_017 within_038_020
private theorem within_038_021_agreement : RowAgreement 4880 4888 := RowAgreement.join within_038_017_agreement within_038_020_agreement
private def within_038_022 : CountSegment 1 10 13120 4888 4890 2201 2202 := CountSegment.join atomic_4889 atomic_4890
private theorem within_038_022_agreement : RowAgreement 4888 4890 := RowAgreement.join atomic_4889_agreement atomic_4890_agreement
private def within_038_023 : CountSegment 1 10 13120 4890 4892 2202 2202 := CountSegment.join atomic_4891 atomic_4892
private theorem within_038_023_agreement : RowAgreement 4890 4892 := RowAgreement.join atomic_4891_agreement atomic_4892_agreement
private def within_038_024 : CountSegment 1 10 13120 4888 4892 2201 2202 := CountSegment.join within_038_022 within_038_023
private theorem within_038_024_agreement : RowAgreement 4888 4892 := RowAgreement.join within_038_022_agreement within_038_023_agreement
private def within_038_025 : CountSegment 1 10 13120 4892 4894 2202 2203 := CountSegment.join atomic_4893 atomic_4894
private theorem within_038_025_agreement : RowAgreement 4892 4894 := RowAgreement.join atomic_4893_agreement atomic_4894_agreement
private def within_038_026 : CountSegment 1 10 13120 4894 4896 2203 2204 := CountSegment.join atomic_4895 atomic_4896
private theorem within_038_026_agreement : RowAgreement 4894 4896 := RowAgreement.join atomic_4895_agreement atomic_4896_agreement
private def within_038_027 : CountSegment 1 10 13120 4892 4896 2202 2204 := CountSegment.join within_038_025 within_038_026
private theorem within_038_027_agreement : RowAgreement 4892 4896 := RowAgreement.join within_038_025_agreement within_038_026_agreement
private def within_038_028 : CountSegment 1 10 13120 4888 4896 2201 2204 := CountSegment.join within_038_024 within_038_027
private theorem within_038_028_agreement : RowAgreement 4888 4896 := RowAgreement.join within_038_024_agreement within_038_027_agreement
private def within_038_029 : CountSegment 1 10 13120 4880 4896 2199 2204 := CountSegment.join within_038_021 within_038_028
private theorem within_038_029_agreement : RowAgreement 4880 4896 := RowAgreement.join within_038_021_agreement within_038_028_agreement
private def within_038_030 : CountSegment 1 10 13120 4864 4896 2192 2204 := CountSegment.join within_038_014 within_038_029
private theorem within_038_030_agreement : RowAgreement 4864 4896 := RowAgreement.join within_038_014_agreement within_038_029_agreement
private def within_038_031 : CountSegment 1 10 13120 4896 4898 2204 2204 := CountSegment.join atomic_4897 atomic_4898
private theorem within_038_031_agreement : RowAgreement 4896 4898 := RowAgreement.join atomic_4897_agreement atomic_4898_agreement
private def within_038_032 : CountSegment 1 10 13120 4898 4900 2204 2205 := CountSegment.join atomic_4899 atomic_4900
private theorem within_038_032_agreement : RowAgreement 4898 4900 := RowAgreement.join atomic_4899_agreement atomic_4900_agreement
private def within_038_033 : CountSegment 1 10 13120 4896 4900 2204 2205 := CountSegment.join within_038_031 within_038_032
private theorem within_038_033_agreement : RowAgreement 4896 4900 := RowAgreement.join within_038_031_agreement within_038_032_agreement
private def within_038_034 : CountSegment 1 10 13120 4900 4902 2205 2207 := CountSegment.join atomic_4901 atomic_4902
private theorem within_038_034_agreement : RowAgreement 4900 4902 := RowAgreement.join atomic_4901_agreement atomic_4902_agreement
private def within_038_035 : CountSegment 1 10 13120 4902 4904 2207 2208 := CountSegment.join atomic_4903 atomic_4904
private theorem within_038_035_agreement : RowAgreement 4902 4904 := RowAgreement.join atomic_4903_agreement atomic_4904_agreement
private def within_038_036 : CountSegment 1 10 13120 4900 4904 2205 2208 := CountSegment.join within_038_034 within_038_035
private theorem within_038_036_agreement : RowAgreement 4900 4904 := RowAgreement.join within_038_034_agreement within_038_035_agreement
private def within_038_037 : CountSegment 1 10 13120 4896 4904 2204 2208 := CountSegment.join within_038_033 within_038_036
private theorem within_038_037_agreement : RowAgreement 4896 4904 := RowAgreement.join within_038_033_agreement within_038_036_agreement
private def within_038_038 : CountSegment 1 10 13120 4904 4906 2208 2208 := CountSegment.join atomic_4905 atomic_4906
private theorem within_038_038_agreement : RowAgreement 4904 4906 := RowAgreement.join atomic_4905_agreement atomic_4906_agreement
private def within_038_039 : CountSegment 1 10 13120 4906 4908 2208 2208 := CountSegment.join atomic_4907 atomic_4908
private theorem within_038_039_agreement : RowAgreement 4906 4908 := RowAgreement.join atomic_4907_agreement atomic_4908_agreement
private def within_038_040 : CountSegment 1 10 13120 4904 4908 2208 2208 := CountSegment.join within_038_038 within_038_039
private theorem within_038_040_agreement : RowAgreement 4904 4908 := RowAgreement.join within_038_038_agreement within_038_039_agreement
private def within_038_041 : CountSegment 1 10 13120 4908 4910 2208 2208 := CountSegment.join atomic_4909 atomic_4910
private theorem within_038_041_agreement : RowAgreement 4908 4910 := RowAgreement.join atomic_4909_agreement atomic_4910_agreement
private def within_038_042 : CountSegment 1 10 13120 4910 4912 2208 2208 := CountSegment.join atomic_4911 atomic_4912
private theorem within_038_042_agreement : RowAgreement 4910 4912 := RowAgreement.join atomic_4911_agreement atomic_4912_agreement
private def within_038_043 : CountSegment 1 10 13120 4908 4912 2208 2208 := CountSegment.join within_038_041 within_038_042
private theorem within_038_043_agreement : RowAgreement 4908 4912 := RowAgreement.join within_038_041_agreement within_038_042_agreement
private def within_038_044 : CountSegment 1 10 13120 4904 4912 2208 2208 := CountSegment.join within_038_040 within_038_043
private theorem within_038_044_agreement : RowAgreement 4904 4912 := RowAgreement.join within_038_040_agreement within_038_043_agreement
private def within_038_045 : CountSegment 1 10 13120 4896 4912 2204 2208 := CountSegment.join within_038_037 within_038_044
private theorem within_038_045_agreement : RowAgreement 4896 4912 := RowAgreement.join within_038_037_agreement within_038_044_agreement
private def within_038_046 : CountSegment 1 10 13120 4912 4914 2208 2210 := CountSegment.join atomic_4913 atomic_4914
private theorem within_038_046_agreement : RowAgreement 4912 4914 := RowAgreement.join atomic_4913_agreement atomic_4914_agreement
private def within_038_047 : CountSegment 1 10 13120 4914 4916 2210 2210 := CountSegment.join atomic_4915 atomic_4916
private theorem within_038_047_agreement : RowAgreement 4914 4916 := RowAgreement.join atomic_4915_agreement atomic_4916_agreement
private def within_038_048 : CountSegment 1 10 13120 4912 4916 2208 2210 := CountSegment.join within_038_046 within_038_047
private theorem within_038_048_agreement : RowAgreement 4912 4916 := RowAgreement.join within_038_046_agreement within_038_047_agreement
private def within_038_049 : CountSegment 1 10 13120 4916 4918 2210 2210 := CountSegment.join atomic_4917 atomic_4918
private theorem within_038_049_agreement : RowAgreement 4916 4918 := RowAgreement.join atomic_4917_agreement atomic_4918_agreement
private def within_038_050 : CountSegment 1 10 13120 4918 4920 2210 2210 := CountSegment.join atomic_4919 atomic_4920
private theorem within_038_050_agreement : RowAgreement 4918 4920 := RowAgreement.join atomic_4919_agreement atomic_4920_agreement
private def within_038_051 : CountSegment 1 10 13120 4916 4920 2210 2210 := CountSegment.join within_038_049 within_038_050
private theorem within_038_051_agreement : RowAgreement 4916 4920 := RowAgreement.join within_038_049_agreement within_038_050_agreement
private def within_038_052 : CountSegment 1 10 13120 4912 4920 2208 2210 := CountSegment.join within_038_048 within_038_051
private theorem within_038_052_agreement : RowAgreement 4912 4920 := RowAgreement.join within_038_048_agreement within_038_051_agreement
private def within_038_053 : CountSegment 1 10 13120 4920 4922 2210 2212 := CountSegment.join atomic_4921 atomic_4922
private theorem within_038_053_agreement : RowAgreement 4920 4922 := RowAgreement.join atomic_4921_agreement atomic_4922_agreement
private def within_038_054 : CountSegment 1 10 13120 4922 4924 2212 2214 := CountSegment.join atomic_4923 atomic_4924
private theorem within_038_054_agreement : RowAgreement 4922 4924 := RowAgreement.join atomic_4923_agreement atomic_4924_agreement
private def within_038_055 : CountSegment 1 10 13120 4920 4924 2210 2214 := CountSegment.join within_038_053 within_038_054
private theorem within_038_055_agreement : RowAgreement 4920 4924 := RowAgreement.join within_038_053_agreement within_038_054_agreement
private def within_038_056 : CountSegment 1 10 13120 4924 4926 2214 2214 := CountSegment.join atomic_4925 atomic_4926
private theorem within_038_056_agreement : RowAgreement 4924 4926 := RowAgreement.join atomic_4925_agreement atomic_4926_agreement
private def within_038_057 : CountSegment 1 10 13120 4926 4928 2214 2214 := CountSegment.join atomic_4927 atomic_4928
private theorem within_038_057_agreement : RowAgreement 4926 4928 := RowAgreement.join atomic_4927_agreement atomic_4928_agreement
private def within_038_058 : CountSegment 1 10 13120 4924 4928 2214 2214 := CountSegment.join within_038_056 within_038_057
private theorem within_038_058_agreement : RowAgreement 4924 4928 := RowAgreement.join within_038_056_agreement within_038_057_agreement
private def within_038_059 : CountSegment 1 10 13120 4920 4928 2210 2214 := CountSegment.join within_038_055 within_038_058
private theorem within_038_059_agreement : RowAgreement 4920 4928 := RowAgreement.join within_038_055_agreement within_038_058_agreement
private def within_038_060 : CountSegment 1 10 13120 4912 4928 2208 2214 := CountSegment.join within_038_052 within_038_059
private theorem within_038_060_agreement : RowAgreement 4912 4928 := RowAgreement.join within_038_052_agreement within_038_059_agreement
private def within_038_061 : CountSegment 1 10 13120 4896 4928 2204 2214 := CountSegment.join within_038_045 within_038_060
private theorem within_038_061_agreement : RowAgreement 4896 4928 := RowAgreement.join within_038_045_agreement within_038_060_agreement
private def within_038_062 : CountSegment 1 10 13120 4864 4928 2192 2214 := CountSegment.join within_038_030 within_038_061
private theorem within_038_062_agreement : RowAgreement 4864 4928 := RowAgreement.join within_038_030_agreement within_038_061_agreement
private def within_038_063 : CountSegment 1 10 13120 4928 4930 2214 2215 := CountSegment.join atomic_4929 atomic_4930
private theorem within_038_063_agreement : RowAgreement 4928 4930 := RowAgreement.join atomic_4929_agreement atomic_4930_agreement
private def within_038_064 : CountSegment 1 10 13120 4930 4932 2215 2217 := CountSegment.join atomic_4931 atomic_4932
private theorem within_038_064_agreement : RowAgreement 4930 4932 := RowAgreement.join atomic_4931_agreement atomic_4932_agreement
private def within_038_065 : CountSegment 1 10 13120 4928 4932 2214 2217 := CountSegment.join within_038_063 within_038_064
private theorem within_038_065_agreement : RowAgreement 4928 4932 := RowAgreement.join within_038_063_agreement within_038_064_agreement
private def within_038_066 : CountSegment 1 10 13120 4932 4934 2217 2218 := CountSegment.join atomic_4933 atomic_4934
private theorem within_038_066_agreement : RowAgreement 4932 4934 := RowAgreement.join atomic_4933_agreement atomic_4934_agreement
private def within_038_067 : CountSegment 1 10 13120 4934 4936 2218 2218 := CountSegment.join atomic_4935 atomic_4936
private theorem within_038_067_agreement : RowAgreement 4934 4936 := RowAgreement.join atomic_4935_agreement atomic_4936_agreement
private def within_038_068 : CountSegment 1 10 13120 4932 4936 2217 2218 := CountSegment.join within_038_066 within_038_067
private theorem within_038_068_agreement : RowAgreement 4932 4936 := RowAgreement.join within_038_066_agreement within_038_067_agreement
private def within_038_069 : CountSegment 1 10 13120 4928 4936 2214 2218 := CountSegment.join within_038_065 within_038_068
private theorem within_038_069_agreement : RowAgreement 4928 4936 := RowAgreement.join within_038_065_agreement within_038_068_agreement
private def within_038_070 : CountSegment 1 10 13120 4936 4938 2218 2221 := CountSegment.join atomic_4937 atomic_4938
private theorem within_038_070_agreement : RowAgreement 4936 4938 := RowAgreement.join atomic_4937_agreement atomic_4938_agreement
private def within_038_071 : CountSegment 1 10 13120 4938 4940 2221 2222 := CountSegment.join atomic_4939 atomic_4940
private theorem within_038_071_agreement : RowAgreement 4938 4940 := RowAgreement.join atomic_4939_agreement atomic_4940_agreement
private def within_038_072 : CountSegment 1 10 13120 4936 4940 2218 2222 := CountSegment.join within_038_070 within_038_071
private theorem within_038_072_agreement : RowAgreement 4936 4940 := RowAgreement.join within_038_070_agreement within_038_071_agreement
private def within_038_073 : CountSegment 1 10 13120 4940 4942 2222 2223 := CountSegment.join atomic_4941 atomic_4942
private theorem within_038_073_agreement : RowAgreement 4940 4942 := RowAgreement.join atomic_4941_agreement atomic_4942_agreement
private def within_038_074 : CountSegment 1 10 13120 4942 4944 2223 2224 := CountSegment.join atomic_4943 atomic_4944
private theorem within_038_074_agreement : RowAgreement 4942 4944 := RowAgreement.join atomic_4943_agreement atomic_4944_agreement
private def within_038_075 : CountSegment 1 10 13120 4940 4944 2222 2224 := CountSegment.join within_038_073 within_038_074
private theorem within_038_075_agreement : RowAgreement 4940 4944 := RowAgreement.join within_038_073_agreement within_038_074_agreement
private def within_038_076 : CountSegment 1 10 13120 4936 4944 2218 2224 := CountSegment.join within_038_072 within_038_075
private theorem within_038_076_agreement : RowAgreement 4936 4944 := RowAgreement.join within_038_072_agreement within_038_075_agreement
private def within_038_077 : CountSegment 1 10 13120 4928 4944 2214 2224 := CountSegment.join within_038_069 within_038_076
private theorem within_038_077_agreement : RowAgreement 4928 4944 := RowAgreement.join within_038_069_agreement within_038_076_agreement
private def within_038_078 : CountSegment 1 10 13120 4944 4946 2224 2224 := CountSegment.join atomic_4945 atomic_4946
private theorem within_038_078_agreement : RowAgreement 4944 4946 := RowAgreement.join atomic_4945_agreement atomic_4946_agreement
private def within_038_079 : CountSegment 1 10 13120 4946 4948 2224 2226 := CountSegment.join atomic_4947 atomic_4948
private theorem within_038_079_agreement : RowAgreement 4946 4948 := RowAgreement.join atomic_4947_agreement atomic_4948_agreement
private def within_038_080 : CountSegment 1 10 13120 4944 4948 2224 2226 := CountSegment.join within_038_078 within_038_079
private theorem within_038_080_agreement : RowAgreement 4944 4948 := RowAgreement.join within_038_078_agreement within_038_079_agreement
private def within_038_081 : CountSegment 1 10 13120 4948 4950 2226 2229 := CountSegment.join atomic_4949 atomic_4950
private theorem within_038_081_agreement : RowAgreement 4948 4950 := RowAgreement.join atomic_4949_agreement atomic_4950_agreement
private def within_038_082 : CountSegment 1 10 13120 4950 4952 2229 2230 := CountSegment.join atomic_4951 atomic_4952
private theorem within_038_082_agreement : RowAgreement 4950 4952 := RowAgreement.join atomic_4951_agreement atomic_4952_agreement
private def within_038_083 : CountSegment 1 10 13120 4948 4952 2226 2230 := CountSegment.join within_038_081 within_038_082
private theorem within_038_083_agreement : RowAgreement 4948 4952 := RowAgreement.join within_038_081_agreement within_038_082_agreement
private def within_038_084 : CountSegment 1 10 13120 4944 4952 2224 2230 := CountSegment.join within_038_080 within_038_083
private theorem within_038_084_agreement : RowAgreement 4944 4952 := RowAgreement.join within_038_080_agreement within_038_083_agreement
private def within_038_085 : CountSegment 1 10 13120 4952 4954 2230 2231 := CountSegment.join atomic_4953 atomic_4954
private theorem within_038_085_agreement : RowAgreement 4952 4954 := RowAgreement.join atomic_4953_agreement atomic_4954_agreement
private def within_038_086 : CountSegment 1 10 13120 4954 4956 2231 2232 := CountSegment.join atomic_4955 atomic_4956
private theorem within_038_086_agreement : RowAgreement 4954 4956 := RowAgreement.join atomic_4955_agreement atomic_4956_agreement
private def within_038_087 : CountSegment 1 10 13120 4952 4956 2230 2232 := CountSegment.join within_038_085 within_038_086
private theorem within_038_087_agreement : RowAgreement 4952 4956 := RowAgreement.join within_038_085_agreement within_038_086_agreement
private def within_038_088 : CountSegment 1 10 13120 4956 4958 2232 2232 := CountSegment.join atomic_4957 atomic_4958
private theorem within_038_088_agreement : RowAgreement 4956 4958 := RowAgreement.join atomic_4957_agreement atomic_4958_agreement
private def within_038_089 : CountSegment 1 10 13120 4958 4960 2232 2233 := CountSegment.join atomic_4959 atomic_4960
private theorem within_038_089_agreement : RowAgreement 4958 4960 := RowAgreement.join atomic_4959_agreement atomic_4960_agreement
private def within_038_090 : CountSegment 1 10 13120 4956 4960 2232 2233 := CountSegment.join within_038_088 within_038_089
private theorem within_038_090_agreement : RowAgreement 4956 4960 := RowAgreement.join within_038_088_agreement within_038_089_agreement
private def within_038_091 : CountSegment 1 10 13120 4952 4960 2230 2233 := CountSegment.join within_038_087 within_038_090
private theorem within_038_091_agreement : RowAgreement 4952 4960 := RowAgreement.join within_038_087_agreement within_038_090_agreement
private def within_038_092 : CountSegment 1 10 13120 4944 4960 2224 2233 := CountSegment.join within_038_084 within_038_091
private theorem within_038_092_agreement : RowAgreement 4944 4960 := RowAgreement.join within_038_084_agreement within_038_091_agreement
private def within_038_093 : CountSegment 1 10 13120 4928 4960 2214 2233 := CountSegment.join within_038_077 within_038_092
private theorem within_038_093_agreement : RowAgreement 4928 4960 := RowAgreement.join within_038_077_agreement within_038_092_agreement
private def within_038_094 : CountSegment 1 10 13120 4960 4962 2233 2234 := CountSegment.join atomic_4961 atomic_4962
private theorem within_038_094_agreement : RowAgreement 4960 4962 := RowAgreement.join atomic_4961_agreement atomic_4962_agreement
private def within_038_095 : CountSegment 1 10 13120 4962 4964 2234 2234 := CountSegment.join atomic_4963 atomic_4964
private theorem within_038_095_agreement : RowAgreement 4962 4964 := RowAgreement.join atomic_4963_agreement atomic_4964_agreement
private def within_038_096 : CountSegment 1 10 13120 4960 4964 2233 2234 := CountSegment.join within_038_094 within_038_095
private theorem within_038_096_agreement : RowAgreement 4960 4964 := RowAgreement.join within_038_094_agreement within_038_095_agreement
private def within_038_097 : CountSegment 1 10 13120 4964 4966 2234 2234 := CountSegment.join atomic_4965 atomic_4966
private theorem within_038_097_agreement : RowAgreement 4964 4966 := RowAgreement.join atomic_4965_agreement atomic_4966_agreement
private def within_038_098 : CountSegment 1 10 13120 4966 4968 2234 2234 := CountSegment.join atomic_4967 atomic_4968
private theorem within_038_098_agreement : RowAgreement 4966 4968 := RowAgreement.join atomic_4967_agreement atomic_4968_agreement
private def within_038_099 : CountSegment 1 10 13120 4964 4968 2234 2234 := CountSegment.join within_038_097 within_038_098
private theorem within_038_099_agreement : RowAgreement 4964 4968 := RowAgreement.join within_038_097_agreement within_038_098_agreement
private def within_038_100 : CountSegment 1 10 13120 4960 4968 2233 2234 := CountSegment.join within_038_096 within_038_099
private theorem within_038_100_agreement : RowAgreement 4960 4968 := RowAgreement.join within_038_096_agreement within_038_099_agreement
private def within_038_101 : CountSegment 1 10 13120 4968 4970 2234 2234 := CountSegment.join atomic_4969 atomic_4970
private theorem within_038_101_agreement : RowAgreement 4968 4970 := RowAgreement.join atomic_4969_agreement atomic_4970_agreement
private def within_038_102 : CountSegment 1 10 13120 4970 4972 2234 2234 := CountSegment.join atomic_4971 atomic_4972
private theorem within_038_102_agreement : RowAgreement 4970 4972 := RowAgreement.join atomic_4971_agreement atomic_4972_agreement
private def within_038_103 : CountSegment 1 10 13120 4968 4972 2234 2234 := CountSegment.join within_038_101 within_038_102
private theorem within_038_103_agreement : RowAgreement 4968 4972 := RowAgreement.join within_038_101_agreement within_038_102_agreement
private def within_038_104 : CountSegment 1 10 13120 4972 4974 2234 2235 := CountSegment.join atomic_4973 atomic_4974
private theorem within_038_104_agreement : RowAgreement 4972 4974 := RowAgreement.join atomic_4973_agreement atomic_4974_agreement
private def within_038_105 : CountSegment 1 10 13120 4974 4976 2235 2236 := CountSegment.join atomic_4975 atomic_4976
private theorem within_038_105_agreement : RowAgreement 4974 4976 := RowAgreement.join atomic_4975_agreement atomic_4976_agreement
private def within_038_106 : CountSegment 1 10 13120 4972 4976 2234 2236 := CountSegment.join within_038_104 within_038_105
private theorem within_038_106_agreement : RowAgreement 4972 4976 := RowAgreement.join within_038_104_agreement within_038_105_agreement
private def within_038_107 : CountSegment 1 10 13120 4968 4976 2234 2236 := CountSegment.join within_038_103 within_038_106
private theorem within_038_107_agreement : RowAgreement 4968 4976 := RowAgreement.join within_038_103_agreement within_038_106_agreement
private def within_038_108 : CountSegment 1 10 13120 4960 4976 2233 2236 := CountSegment.join within_038_100 within_038_107
private theorem within_038_108_agreement : RowAgreement 4960 4976 := RowAgreement.join within_038_100_agreement within_038_107_agreement
private def within_038_109 : CountSegment 1 10 13120 4976 4978 2236 2237 := CountSegment.join atomic_4977 atomic_4978
private theorem within_038_109_agreement : RowAgreement 4976 4978 := RowAgreement.join atomic_4977_agreement atomic_4978_agreement
private def within_038_110 : CountSegment 1 10 13120 4978 4980 2237 2239 := CountSegment.join atomic_4979 atomic_4980
private theorem within_038_110_agreement : RowAgreement 4978 4980 := RowAgreement.join atomic_4979_agreement atomic_4980_agreement
private def within_038_111 : CountSegment 1 10 13120 4976 4980 2236 2239 := CountSegment.join within_038_109 within_038_110
private theorem within_038_111_agreement : RowAgreement 4976 4980 := RowAgreement.join within_038_109_agreement within_038_110_agreement
private def within_038_112 : CountSegment 1 10 13120 4980 4982 2239 2240 := CountSegment.join atomic_4981 atomic_4982
private theorem within_038_112_agreement : RowAgreement 4980 4982 := RowAgreement.join atomic_4981_agreement atomic_4982_agreement
private def within_038_113 : CountSegment 1 10 13120 4982 4984 2240 2241 := CountSegment.join atomic_4983 atomic_4984
private theorem within_038_113_agreement : RowAgreement 4982 4984 := RowAgreement.join atomic_4983_agreement atomic_4984_agreement
private def within_038_114 : CountSegment 1 10 13120 4980 4984 2239 2241 := CountSegment.join within_038_112 within_038_113
private theorem within_038_114_agreement : RowAgreement 4980 4984 := RowAgreement.join within_038_112_agreement within_038_113_agreement
private def within_038_115 : CountSegment 1 10 13120 4976 4984 2236 2241 := CountSegment.join within_038_111 within_038_114
private theorem within_038_115_agreement : RowAgreement 4976 4984 := RowAgreement.join within_038_111_agreement within_038_114_agreement
private def within_038_116 : CountSegment 1 10 13120 4984 4986 2241 2243 := CountSegment.join atomic_4985 atomic_4986
private theorem within_038_116_agreement : RowAgreement 4984 4986 := RowAgreement.join atomic_4985_agreement atomic_4986_agreement
private def within_038_117 : CountSegment 1 10 13120 4986 4988 2243 2244 := CountSegment.join atomic_4987 atomic_4988
private theorem within_038_117_agreement : RowAgreement 4986 4988 := RowAgreement.join atomic_4987_agreement atomic_4988_agreement
private def within_038_118 : CountSegment 1 10 13120 4984 4988 2241 2244 := CountSegment.join within_038_116 within_038_117
private theorem within_038_118_agreement : RowAgreement 4984 4988 := RowAgreement.join within_038_116_agreement within_038_117_agreement
private def within_038_119 : CountSegment 1 10 13120 4988 4990 2244 2245 := CountSegment.join atomic_4989 atomic_4990
private theorem within_038_119_agreement : RowAgreement 4988 4990 := RowAgreement.join atomic_4989_agreement atomic_4990_agreement
private def within_038_120 : CountSegment 1 10 13120 4990 4992 2245 2245 := CountSegment.join atomic_4991 atomic_4992
private theorem within_038_120_agreement : RowAgreement 4990 4992 := RowAgreement.join atomic_4991_agreement atomic_4992_agreement
private def within_038_121 : CountSegment 1 10 13120 4988 4992 2244 2245 := CountSegment.join within_038_119 within_038_120
private theorem within_038_121_agreement : RowAgreement 4988 4992 := RowAgreement.join within_038_119_agreement within_038_120_agreement
private def within_038_122 : CountSegment 1 10 13120 4984 4992 2241 2245 := CountSegment.join within_038_118 within_038_121
private theorem within_038_122_agreement : RowAgreement 4984 4992 := RowAgreement.join within_038_118_agreement within_038_121_agreement
private def within_038_123 : CountSegment 1 10 13120 4976 4992 2236 2245 := CountSegment.join within_038_115 within_038_122
private theorem within_038_123_agreement : RowAgreement 4976 4992 := RowAgreement.join within_038_115_agreement within_038_122_agreement
private def within_038_124 : CountSegment 1 10 13120 4960 4992 2233 2245 := CountSegment.join within_038_108 within_038_123
private theorem within_038_124_agreement : RowAgreement 4960 4992 := RowAgreement.join within_038_108_agreement within_038_123_agreement
private def within_038_125 : CountSegment 1 10 13120 4928 4992 2214 2245 := CountSegment.join within_038_093 within_038_124
private theorem within_038_125_agreement : RowAgreement 4928 4992 := RowAgreement.join within_038_093_agreement within_038_124_agreement
private def within_038_126 : CountSegment 1 10 13120 4864 4992 2192 2245 := CountSegment.join within_038_062 within_038_125
private theorem within_038_126_agreement : RowAgreement 4864 4992 := RowAgreement.join within_038_062_agreement within_038_125_agreement
def segment_038 : CountSegment 1 10 13120 4864 4992 2192 2245 := within_038_126
theorem agreement_038 : RowAgreement 4864 4992 := within_038_126_agreement
end ProjectEulerSolutions.P58Numerical
