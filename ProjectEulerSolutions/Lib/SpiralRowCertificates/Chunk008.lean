import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1025 : RowCertificate 1025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1025 1) := by change ¬ Nat.Prime 4204551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1401517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1025 2) := by change Nat.Prime 4202501; exact ProjectEulerSolutions.P58PrimeNodes.p_4202501.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1025 3) := by change Nat.Prime 4200451; exact ProjectEulerSolutions.P58PrimeNodes.p_4200451.prime
    exact decide_eq_true prime
theorem row_count_1025 : row 1025 = 2 := (row_1025.checked).trans rfl
theorem model_count_1025 : modelRow 1025 = 2 := by decide
theorem modelrow_agrees_1025 : modelRow 1025 = row 1025 := model_count_1025.trans row_count_1025.symm
def row_1026 : RowCertificate 1026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1026 1) := by change ¬ Nat.Prime 4212757; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 57709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1026 2) := by change ¬ Nat.Prime 4210705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 842141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1026 3) := by change ¬ Nat.Prime 4208653; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 135763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1026 : row 1026 = 0 := (row_1026.checked).trans rfl
theorem model_count_1026 : modelRow 1026 = 0 := by decide
theorem modelrow_agrees_1026 : modelRow 1026 = row 1026 := model_count_1026.trans row_count_1026.symm
def row_1027 : RowCertificate 1027 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1027 1) := by change Nat.Prime 4220971; exact ProjectEulerSolutions.P58PrimeNodes.p_4220971.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1027 2) := by change Nat.Prime 4218917; exact ProjectEulerSolutions.P58PrimeNodes.p_4218917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1027 3) := by change ¬ Nat.Prime 4216863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1405621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1027 : row 1027 = 2 := (row_1027.checked).trans rfl
theorem model_count_1027 : modelRow 1027 = 2 := by decide
theorem modelrow_agrees_1027 : modelRow 1027 = row 1027 := model_count_1027.trans row_count_1027.symm
def row_1028 : RowCertificate 1028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1028 1) := by change ¬ Nat.Prime 4229193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1409731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1028 2) := by change Nat.Prime 4227137; exact ProjectEulerSolutions.P58PrimeNodes.p_4227137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1028 3) := by change ¬ Nat.Prime 4225081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 603583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1028 : row 1028 = 1 := (row_1028.checked).trans rfl
theorem model_count_1028 : modelRow 1028 = 1 := by decide
theorem modelrow_agrees_1028 : modelRow 1028 = row 1028 := model_count_1028.trans row_count_1028.symm
def row_1029 : RowCertificate 1029 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1029 1) := by change Nat.Prime 4237423; exact ProjectEulerSolutions.P58PrimeNodes.p_4237423.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1029 2) := by change ¬ Nat.Prime 4235365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 847073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1029 3) := by change ¬ Nat.Prime 4233307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 325639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1029 : row 1029 = 1 := (row_1029.checked).trans rfl
theorem model_count_1029 : modelRow 1029 = 1 := by decide
theorem modelrow_agrees_1029 : modelRow 1029 = row 1029 := model_count_1029.trans row_count_1029.symm
def row_1030 : RowCertificate 1030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1030 1) := by change ¬ Nat.Prime 4245661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 606523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1030 2) := by change ¬ Nat.Prime 4243601; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 10909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1030 3) := by change ¬ Nat.Prime 4241541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1413847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1030 : row 1030 = 0 := (row_1030.checked).trans rfl
theorem model_count_1030 : modelRow 1030 = 0 := by decide
theorem modelrow_agrees_1030 : modelRow 1030 = row 1030 := model_count_1030.trans row_count_1030.symm
def row_1031 : RowCertificate 1031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1031 1) := by change ¬ Nat.Prime 4253907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1417969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1031 2) := by change ¬ Nat.Prime 4251845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 850369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1031 3) := by change ¬ Nat.Prime 4249783; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 114859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1031 : row 1031 = 0 := (row_1031.checked).trans rfl
theorem model_count_1031 : modelRow 1031 = 0 := by decide
theorem modelrow_agrees_1031 : modelRow 1031 = row 1031 := model_count_1031.trans row_count_1031.symm
def row_1032 : RowCertificate 1032 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1032 1) := by change Nat.Prime 4262161; exact ProjectEulerSolutions.P58PrimeNodes.p_4262161.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1032 2) := by change Nat.Prime 4260097; exact ProjectEulerSolutions.P58PrimeNodes.p_4260097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1032 3) := by change ¬ Nat.Prime 4258033; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 327541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1032 : row 1032 = 2 := (row_1032.checked).trans rfl
theorem model_count_1032 : modelRow 1032 = 2 := by decide
theorem modelrow_agrees_1032 : modelRow 1032 = row 1032 := model_count_1032.trans row_count_1032.symm
def row_1033 : RowCertificate 1033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1033 1) := by change ¬ Nat.Prime 4270423; exact PrattCertificates.composite_of_product (factor := 1423) (cofactor := 3001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1033 2) := by change ¬ Nat.Prime 4268357; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 115361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1033 3) := by change ¬ Nat.Prime 4266291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1422097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1033 : row 1033 = 0 := (row_1033.checked).trans rfl
theorem model_count_1033 : modelRow 1033 = 0 := by decide
theorem modelrow_agrees_1033 : modelRow 1033 = row 1033 := model_count_1033.trans row_count_1033.symm
def row_1034 : RowCertificate 1034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1034 1) := by change ¬ Nat.Prime 4278693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1426231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1034 2) := by change ¬ Nat.Prime 4276625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 855325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1034 3) := by change ¬ Nat.Prime 4274557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 610651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1034 : row 1034 = 0 := (row_1034.checked).trans rfl
theorem model_count_1034 : modelRow 1034 = 0 := by decide
theorem modelrow_agrees_1034 : modelRow 1034 = row 1034 := model_count_1034.trans row_count_1034.symm
def row_1035 : RowCertificate 1035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1035 1) := by change ¬ Nat.Prime 4286971; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 329767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1035 2) := by change ¬ Nat.Prime 4284901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 252053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1035 3) := by change ¬ Nat.Prime 4282831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 611833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1035 : row 1035 = 0 := (row_1035.checked).trans rfl
theorem model_count_1035 : modelRow 1035 = 0 := by decide
theorem modelrow_agrees_1035 : modelRow 1035 = row 1035 := model_count_1035.trans row_count_1035.symm
def row_1036 : RowCertificate 1036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1036 1) := by change ¬ Nat.Prime 4295257; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 44281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1036 2) := by change ¬ Nat.Prime 4293185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 858637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1036 3) := by change ¬ Nat.Prime 4291113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1430371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1036 : row 1036 = 0 := (row_1036.checked).trans rfl
theorem model_count_1036 : modelRow 1036 = 0 := by decide
theorem modelrow_agrees_1036 : modelRow 1036 = row 1036 := model_count_1036.trans row_count_1036.symm
def row_1037 : RowCertificate 1037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1037 1) := by change ¬ Nat.Prime 4303551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1434517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1037 2) := by change Nat.Prime 4301477; exact ProjectEulerSolutions.P58PrimeNodes.p_4301477.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1037 3) := by change Nat.Prime 4299403; exact ProjectEulerSolutions.P58PrimeNodes.p_4299403.prime
    exact decide_eq_true prime
theorem row_count_1037 : row 1037 = 2 := (row_1037.checked).trans rfl
theorem model_count_1037 : modelRow 1037 = 2 := by decide
theorem modelrow_agrees_1037 : modelRow 1037 = row 1037 := model_count_1037.trans row_count_1037.symm
def row_1038 : RowCertificate 1038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1038 1) := by change ¬ Nat.Prime 4311853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 615979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1038 2) := by change ¬ Nat.Prime 4309777; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 148613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1038 3) := by change Nat.Prime 4307701; exact ProjectEulerSolutions.P58PrimeNodes.p_4307701.prime
    exact decide_eq_true prime
theorem row_count_1038 : row 1038 = 1 := (row_1038.checked).trans rfl
theorem model_count_1038 : modelRow 1038 = 1 := by decide
theorem modelrow_agrees_1038 : modelRow 1038 = row 1038 := model_count_1038.trans row_count_1038.symm
def row_1039 : RowCertificate 1039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1039 1) := by change ¬ Nat.Prime 4320163; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 227377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1039 2) := by change ¬ Nat.Prime 4318085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 863617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1039 3) := by change ¬ Nat.Prime 4316007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1438669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1039 : row 1039 = 0 := (row_1039.checked).trans rfl
theorem model_count_1039 : modelRow 1039 = 0 := by decide
theorem modelrow_agrees_1039 : modelRow 1039 = row 1039 := model_count_1039.trans row_count_1039.symm
def row_1040 : RowCertificate 1040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1040 1) := by change ¬ Nat.Prime 4328481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1442827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1040 2) := by change Nat.Prime 4326401; exact ProjectEulerSolutions.P58PrimeNodes.p_4326401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1040 3) := by change Nat.Prime 4324321; exact ProjectEulerSolutions.P58PrimeNodes.p_4324321.prime
    exact decide_eq_true prime
theorem row_count_1040 : row 1040 = 2 := (row_1040.checked).trans rfl
theorem model_count_1040 : modelRow 1040 = 2 := by decide
theorem modelrow_agrees_1040 : modelRow 1040 = row 1040 := model_count_1040.trans row_count_1040.symm
def row_1041 : RowCertificate 1041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1041 1) := by change ¬ Nat.Prime 4336807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 228253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1041 2) := by change ¬ Nat.Prime 4334725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 866945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1041 3) := by change ¬ Nat.Prime 4332643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 618949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1041 : row 1041 = 0 := (row_1041.checked).trans rfl
theorem model_count_1041 : modelRow 1041 = 0 := by decide
theorem modelrow_agrees_1041 : modelRow 1041 = row 1041 := model_count_1041.trans row_count_1041.symm
def row_1042 : RowCertificate 1042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1042 1) := by change ¬ Nat.Prime 4345141; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 10321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1042 2) := by change Nat.Prime 4343057; exact ProjectEulerSolutions.P58PrimeNodes.p_4343057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1042 3) := by change ¬ Nat.Prime 4340973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1446991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1042 : row 1042 = 1 := (row_1042.checked).trans rfl
theorem model_count_1042 : modelRow 1042 = 1 := by decide
theorem modelrow_agrees_1042 : modelRow 1042 = row 1042 := model_count_1042.trans row_count_1042.symm
def row_1043 : RowCertificate 1043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1043 1) := by change ¬ Nat.Prime 4353483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1451161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1043 2) := by change Nat.Prime 4351397; exact ProjectEulerSolutions.P58PrimeNodes.p_4351397.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1043 3) := by change Nat.Prime 4349311; exact ProjectEulerSolutions.P58PrimeNodes.p_4349311.prime
    exact decide_eq_true prime
theorem row_count_1043 : row 1043 = 2 := (row_1043.checked).trans rfl
theorem model_count_1043 : modelRow 1043 = 2 := by decide
theorem modelrow_agrees_1043 : modelRow 1043 = row 1043 := model_count_1043.trans row_count_1043.symm
def row_1044 : RowCertificate 1044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1044 1) := by change ¬ Nat.Prime 4361833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 623119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1044 2) := by change ¬ Nat.Prime 4359745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 871949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1044 3) := by change ¬ Nat.Prime 4357657; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 71437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1044 : row 1044 = 0 := (row_1044.checked).trans rfl
theorem model_count_1044 : modelRow 1044 = 0 := by decide
theorem modelrow_agrees_1044 : modelRow 1044 = row 1044 := model_count_1044.trans row_count_1044.symm
def row_1045 : RowCertificate 1045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1045 1) := by change ¬ Nat.Prime 4370191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 624313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1045 2) := by change ¬ Nat.Prime 4368101; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 82417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1045 3) := by change ¬ Nat.Prime 4366011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1455337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1045 : row 1045 = 0 := (row_1045.checked).trans rfl
theorem model_count_1045 : modelRow 1045 = 0 := by decide
theorem modelrow_agrees_1045 : modelRow 1045 = row 1045 := model_count_1045.trans row_count_1045.symm
def row_1046 : RowCertificate 1046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1046 1) := by change ¬ Nat.Prime 4378557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1459519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1046 2) := by change ¬ Nat.Prime 4376465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 875293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1046 3) := by change Nat.Prime 4374373; exact ProjectEulerSolutions.P58PrimeNodes.p_4374373.prime
    exact decide_eq_true prime
theorem row_count_1046 : row 1046 = 1 := (row_1046.checked).trans rfl
theorem model_count_1046 : modelRow 1046 = 1 := by decide
theorem modelrow_agrees_1046 : modelRow 1046 = row 1046 := model_count_1046.trans row_count_1046.symm
def row_1047 : RowCertificate 1047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1047 1) := by change ¬ Nat.Prime 4386931; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 7603) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1047 2) := by change Nat.Prime 4384837; exact ProjectEulerSolutions.P58PrimeNodes.p_4384837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1047 3) := by change ¬ Nat.Prime 4382743; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 3067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1047 : row 1047 = 1 := (row_1047.checked).trans rfl
theorem model_count_1047 : modelRow 1047 = 1 := by decide
theorem modelrow_agrees_1047 : modelRow 1047 = row 1047 := model_count_1047.trans row_count_1047.symm
def row_1048 : RowCertificate 1048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1048 1) := by change ¬ Nat.Prime 4395313; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 338101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1048 2) := by change Nat.Prime 4393217; exact ProjectEulerSolutions.P58PrimeNodes.p_4393217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1048 3) := by change ¬ Nat.Prime 4391121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1463707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1048 : row 1048 = 1 := (row_1048.checked).trans rfl
theorem model_count_1048 : modelRow 1048 = 1 := by decide
theorem modelrow_agrees_1048 : modelRow 1048 = row 1048 := model_count_1048.trans row_count_1048.symm
def row_1049 : RowCertificate 1049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1049 1) := by change ¬ Nat.Prime 4403703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1467901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1049 2) := by change ¬ Nat.Prime 4401605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 880321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1049 3) := by change ¬ Nat.Prime 4399507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 628501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1049 : row 1049 = 0 := (row_1049.checked).trans rfl
theorem model_count_1049 : modelRow 1049 = 0 := by decide
theorem modelrow_agrees_1049 : modelRow 1049 = row 1049 := model_count_1049.trans row_count_1049.symm
def row_1050 : RowCertificate 1050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1050 1) := by change ¬ Nat.Prime 4412101; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 102607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1050 2) := by change ¬ Nat.Prime 4410001; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 152069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1050 3) := by change ¬ Nat.Prime 4407901; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 15913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1050 : row 1050 = 0 := (row_1050.checked).trans rfl
theorem model_count_1050 : modelRow 1050 = 0 := by decide
theorem modelrow_agrees_1050 : modelRow 1050 = row 1050 := model_count_1050.trans row_count_1050.symm
def row_1051 : RowCertificate 1051 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1051 1) := by change ¬ Nat.Prime 4420507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 631501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1051 2) := by change ¬ Nat.Prime 4418405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 883681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1051 3) := by change ¬ Nat.Prime 4416303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1472101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1051 : row 1051 = 0 := (row_1051.checked).trans rfl
theorem model_count_1051 : modelRow 1051 = 0 := by decide
theorem modelrow_agrees_1051 : modelRow 1051 = row 1051 := model_count_1051.trans row_count_1051.symm
def row_1052 : RowCertificate 1052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1052 1) := by change ¬ Nat.Prime 4428921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1476307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1052 2) := by change ¬ Nat.Prime 4426817; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 260401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1052 3) := by change ¬ Nat.Prime 4424713; exact PrattCertificates.composite_of_product (factor := 883) (cofactor := 5011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1052 : row 1052 = 0 := (row_1052.checked).trans rfl
theorem model_count_1052 : modelRow 1052 = 0 := by decide
theorem modelrow_agrees_1052 : modelRow 1052 = row 1052 := model_count_1052.trans row_count_1052.symm
def row_1053 : RowCertificate 1053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1053 1) := by change ¬ Nat.Prime 4437343; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 66229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1053 2) := by change Nat.Prime 4435237; exact ProjectEulerSolutions.P58PrimeNodes.p_4435237.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1053 3) := by change Nat.Prime 4433131; exact ProjectEulerSolutions.P58PrimeNodes.p_4433131.prime
    exact decide_eq_true prime
theorem row_count_1053 : row 1053 = 2 := (row_1053.checked).trans rfl
theorem model_count_1053 : modelRow 1053 = 2 := by decide
theorem modelrow_agrees_1053 : modelRow 1053 = row 1053 := model_count_1053.trans row_count_1053.symm
def row_1054 : RowCertificate 1054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1054 1) := by change ¬ Nat.Prime 4445773; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 60901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1054 2) := by change ¬ Nat.Prime 4443665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 888733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1054 3) := by change ¬ Nat.Prime 4441557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1480519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1054 : row 1054 = 0 := (row_1054.checked).trans rfl
theorem model_count_1054 : modelRow 1054 = 0 := by decide
theorem modelrow_agrees_1054 : modelRow 1054 = row 1054 := model_count_1054.trans row_count_1054.symm
def row_1055 : RowCertificate 1055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1055 1) := by change ¬ Nat.Prime 4454211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1484737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1055 2) := by change ¬ Nat.Prime 4452101; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 7993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1055 3) := by change ¬ Nat.Prime 4449991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 635713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1055 : row 1055 = 0 := (row_1055.checked).trans rfl
theorem model_count_1055 : modelRow 1055 = 0 := by decide
theorem modelrow_agrees_1055 : modelRow 1055 = row 1055 := model_count_1055.trans row_count_1055.symm
def row_1056 : RowCertificate 1056 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1056 1) := by change Nat.Prime 4462657; exact ProjectEulerSolutions.P58PrimeNodes.p_4462657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1056 2) := by change ¬ Nat.Prime 4460545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 892109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1056 3) := by change ¬ Nat.Prime 4458433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 636919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1056 : row 1056 = 1 := (row_1056.checked).trans rfl
theorem model_count_1056 : modelRow 1056 = 1 := by decide
theorem modelrow_agrees_1056 : modelRow 1056 = row 1056 := model_count_1056.trans row_count_1056.symm
def row_1057 : RowCertificate 1057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1057 1) := by change ¬ Nat.Prime 4471111; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 66733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1057 2) := by change ¬ Nat.Prime 4468997; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 343769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1057 3) := by change ¬ Nat.Prime 4466883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1488961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1057 : row 1057 = 0 := (row_1057.checked).trans rfl
theorem model_count_1057 : modelRow 1057 = 0 := by decide
theorem modelrow_agrees_1057 : modelRow 1057 = row 1057 := model_count_1057.trans row_count_1057.symm
def row_1058 : RowCertificate 1058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1058 1) := by change ¬ Nat.Prime 4479573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1493191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1058 2) := by change Nat.Prime 4477457; exact ProjectEulerSolutions.P58PrimeNodes.p_4477457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1058 3) := by change ¬ Nat.Prime 4475341; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 344257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1058 : row 1058 = 1 := (row_1058.checked).trans rfl
theorem model_count_1058 : modelRow 1058 = 1 := by decide
theorem modelrow_agrees_1058 : modelRow 1058 = row 1058 := model_count_1058.trans row_count_1058.symm
def row_1059 : RowCertificate 1059 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1059 1) := by change ¬ Nat.Prime 4488043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 641149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1059 2) := by change ¬ Nat.Prime 4485925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 897185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1059 3) := by change Nat.Prime 4483807; exact ProjectEulerSolutions.P58PrimeNodes.p_4483807.prime
    exact decide_eq_true prime
theorem row_count_1059 : row 1059 = 1 := (row_1059.checked).trans rfl
theorem model_count_1059 : modelRow 1059 = 1 := by decide
theorem modelrow_agrees_1059 : modelRow 1059 = row 1059 := model_count_1059.trans row_count_1059.symm
def row_1060 : RowCertificate 1060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1060 1) := by change ¬ Nat.Prime 4496521; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 236659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1060 2) := by change Nat.Prime 4494401; exact ProjectEulerSolutions.P58PrimeNodes.p_4494401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1060 3) := by change ¬ Nat.Prime 4492281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1497427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1060 : row 1060 = 1 := (row_1060.checked).trans rfl
theorem model_count_1060 : modelRow 1060 = 1 := by decide
theorem modelrow_agrees_1060 : modelRow 1060 = row 1060 := model_count_1060.trans row_count_1060.symm
def row_1061 : RowCertificate 1061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1061 1) := by change ¬ Nat.Prime 4505007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1501669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1061 2) := by change ¬ Nat.Prime 4502885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 900577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1061 3) := by change ¬ Nat.Prime 4500763; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 73783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1061 : row 1061 = 0 := (row_1061.checked).trans rfl
theorem model_count_1061 : modelRow 1061 = 0 := by decide
theorem modelrow_agrees_1061 : modelRow 1061 = row 1061 := model_count_1061.trans row_count_1061.symm
def row_1062 : RowCertificate 1062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1062 1) := by change ¬ Nat.Prime 4513501; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 21391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1062 2) := by change ¬ Nat.Prime 4511377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 347029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1062 3) := by change ¬ Nat.Prime 4509253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 644179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1062 : row 1062 = 0 := (row_1062.checked).trans rfl
theorem model_count_1062 : modelRow 1062 = 0 := by decide
theorem modelrow_agrees_1062 : modelRow 1062 = row 1062 := model_count_1062.trans row_count_1062.symm
def row_1063 : RowCertificate 1063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1063 1) := by change ¬ Nat.Prime 4522003; exact PrattCertificates.composite_of_product (factor := 1693) (cofactor := 2671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1063 2) := by change Nat.Prime 4519877; exact ProjectEulerSolutions.P58PrimeNodes.p_4519877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1063 3) := by change ¬ Nat.Prime 4517751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1505917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1063 : row 1063 = 1 := (row_1063.checked).trans rfl
theorem model_count_1063 : modelRow 1063 = 1 := by decide
theorem modelrow_agrees_1063 : modelRow 1063 = row 1063 := model_count_1063.trans row_count_1063.symm
def row_1064 : RowCertificate 1064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1064 1) := by change ¬ Nat.Prime 4530513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1510171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1064 2) := by change ¬ Nat.Prime 4528385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 905677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1064 3) := by change ¬ Nat.Prime 4526257; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 32563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1064 : row 1064 = 0 := (row_1064.checked).trans rfl
theorem model_count_1064 : modelRow 1064 = 0 := by decide
theorem modelrow_agrees_1064 : modelRow 1064 = row 1064 := model_count_1064.trans row_count_1064.symm
def row_1065 : RowCertificate 1065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1065 1) := by change ¬ Nat.Prime 4539031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 648433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1065 2) := by change ¬ Nat.Prime 4536901; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 30449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1065 3) := by change Nat.Prime 4534771; exact ProjectEulerSolutions.P58PrimeNodes.p_4534771.prime
    exact decide_eq_true prime
theorem row_count_1065 : row 1065 = 1 := (row_1065.checked).trans rfl
theorem model_count_1065 : modelRow 1065 = 1 := by decide
theorem modelrow_agrees_1065 : modelRow 1065 = row 1065 := model_count_1065.trans row_count_1065.symm
def row_1066 : RowCertificate 1066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1066 1) := by change ¬ Nat.Prime 4547557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 649651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1066 2) := by change ¬ Nat.Prime 4545425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 909085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1066 3) := by change ¬ Nat.Prime 4543293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1514431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1066 : row 1066 = 0 := (row_1066.checked).trans rfl
theorem model_count_1066 : modelRow 1066 = 0 := by decide
theorem modelrow_agrees_1066 : modelRow 1066 = row 1066 := model_count_1066.trans row_count_1066.symm
def row_1067 : RowCertificate 1067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1067 1) := by change ¬ Nat.Prime 4556091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1518697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1067 2) := by change ¬ Nat.Prime 4553957; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 157033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1067 3) := by change ¬ Nat.Prime 4551823; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 146833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1067 : row 1067 = 0 := (row_1067.checked).trans rfl
theorem model_count_1067 : modelRow 1067 = 0 := by decide
theorem modelrow_agrees_1067 : modelRow 1067 = row 1067 := model_count_1067.trans row_count_1067.symm
def row_1068 : RowCertificate 1068 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1068 1) := by change Nat.Prime 4564633; exact ProjectEulerSolutions.P58PrimeNodes.p_4564633.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1068 2) := by change Nat.Prime 4562497; exact ProjectEulerSolutions.P58PrimeNodes.p_4562497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1068 3) := by change ¬ Nat.Prime 4560361; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 350797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1068 : row 1068 = 2 := (row_1068.checked).trans rfl
theorem model_count_1068 : modelRow 1068 = 2 := by decide
theorem modelrow_agrees_1068 : modelRow 1068 = row 1068 := model_count_1068.trans row_count_1068.symm
def row_1069 : RowCertificate 1069 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1069 1) := by change Nat.Prime 4573183; exact ProjectEulerSolutions.P58PrimeNodes.p_4573183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1069 2) := by change ¬ Nat.Prime 4571045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 914209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1069 3) := by change ¬ Nat.Prime 4568907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1522969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1069 : row 1069 = 1 := (row_1069.checked).trans rfl
theorem model_count_1069 : modelRow 1069 = 1 := by decide
theorem modelrow_agrees_1069 : modelRow 1069 = row 1069 := model_count_1069.trans row_count_1069.symm
def row_1070 : RowCertificate 1070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1070 1) := by change ¬ Nat.Prime 4581741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1527247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1070 2) := by change ¬ Nat.Prime 4579601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 352277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1070 3) := by change ¬ Nat.Prime 4577461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 653923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1070 : row 1070 = 0 := (row_1070.checked).trans rfl
theorem model_count_1070 : modelRow 1070 = 0 := by decide
theorem modelrow_agrees_1070 : modelRow 1070 = row 1070 := model_count_1070.trans row_count_1070.symm
def row_1071 : RowCertificate 1071 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1071 1) := by change Nat.Prime 4590307; exact ProjectEulerSolutions.P58PrimeNodes.p_4590307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1071 2) := by change ¬ Nat.Prime 4588165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 917633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1071 3) := by change ¬ Nat.Prime 4586023; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 352771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1071 : row 1071 = 1 := (row_1071.checked).trans rfl
theorem model_count_1071 : modelRow 1071 = 1 := by decide
theorem modelrow_agrees_1071 : modelRow 1071 = row 1071 := model_count_1071.trans row_count_1071.symm
def row_1072 : RowCertificate 1072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1072 1) := by change ¬ Nat.Prime 4598881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 656983) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1072 2) := by change ¬ Nat.Prime 4596737; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 62969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1072 3) := by change ¬ Nat.Prime 4594593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1531531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1072 : row 1072 = 0 := (row_1072.checked).trans rfl
theorem model_count_1072 : modelRow 1072 = 0 := by decide
theorem modelrow_agrees_1072 : modelRow 1072 = row 1072 := model_count_1072.trans row_count_1072.symm
def row_1073 : RowCertificate 1073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1073 1) := by change ¬ Nat.Prime 4607463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1535821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1073 2) := by change ¬ Nat.Prime 4605317; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 270901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1073 3) := by change Nat.Prime 4603171; exact ProjectEulerSolutions.P58PrimeNodes.p_4603171.prime
    exact decide_eq_true prime
theorem row_count_1073 : row 1073 = 1 := (row_1073.checked).trans rfl
theorem model_count_1073 : modelRow 1073 = 1 := by decide
theorem modelrow_agrees_1073 : modelRow 1073 = row 1073 := model_count_1073.trans row_count_1073.symm
def row_1074 : RowCertificate 1074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1074 1) := by change ¬ Nat.Prime 4616053; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 355081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1074 2) := by change ¬ Nat.Prime 4613905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 922781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1074 3) := by change ¬ Nat.Prime 4611757; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 8431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1074 : row 1074 = 0 := (row_1074.checked).trans rfl
theorem model_count_1074 : modelRow 1074 = 0 := by decide
theorem modelrow_agrees_1074 : modelRow 1074 = row 1074 := model_count_1074.trans row_count_1074.symm
def row_1075 : RowCertificate 1075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1075 1) := by change ¬ Nat.Prime 4624651; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 13723) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1075 2) := by change ¬ Nat.Prime 4622501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 355577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1075 3) := by change ¬ Nat.Prime 4620351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1540117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1075 : row 1075 = 0 := (row_1075.checked).trans rfl
theorem model_count_1075 : modelRow 1075 = 0 := by decide
theorem modelrow_agrees_1075 : modelRow 1075 = row 1075 := model_count_1075.trans row_count_1075.symm
def row_1076 : RowCertificate 1076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1076 1) := by change ¬ Nat.Prime 4633257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1544419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1076 2) := by change ¬ Nat.Prime 4631105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 926221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1076 3) := by change ¬ Nat.Prime 4628953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 661279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1076 : row 1076 = 0 := (row_1076.checked).trans rfl
theorem model_count_1076 : modelRow 1076 = 0 := by decide
theorem modelrow_agrees_1076 : modelRow 1076 = row 1076 := model_count_1076.trans row_count_1076.symm
def row_1077 : RowCertificate 1077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1077 1) := by change ¬ Nat.Prime 4641871; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 357067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1077 2) := by change Nat.Prime 4639717; exact ProjectEulerSolutions.P58PrimeNodes.p_4639717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1077 3) := by change ¬ Nat.Prime 4637563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 662509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1077 : row 1077 = 1 := (row_1077.checked).trans rfl
theorem model_count_1077 : modelRow 1077 = 1 := by decide
theorem modelrow_agrees_1077 : modelRow 1077 = row 1077 := model_count_1077.trans row_count_1077.symm
def row_1078 : RowCertificate 1078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1078 1) := by change ¬ Nat.Prime 4650493; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 125689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1078 2) := by change ¬ Nat.Prime 4648337; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 47921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1078 3) := by change ¬ Nat.Prime 4646181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1548727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1078 : row 1078 = 0 := (row_1078.checked).trans rfl
theorem model_count_1078 : modelRow 1078 = 0 := by decide
theorem modelrow_agrees_1078 : modelRow 1078 = row 1078 := model_count_1078.trans row_count_1078.symm
def row_1079 : RowCertificate 1079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1079 1) := by change ¬ Nat.Prime 4659123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1553041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1079 2) := by change ¬ Nat.Prime 4656965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 931393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1079 3) := by change Nat.Prime 4654807; exact ProjectEulerSolutions.P58PrimeNodes.p_4654807.prime
    exact decide_eq_true prime
theorem row_count_1079 : row 1079 = 1 := (row_1079.checked).trans rfl
theorem model_count_1079 : modelRow 1079 = 1 := by decide
theorem modelrow_agrees_1079 : modelRow 1079 = row 1079 := model_count_1079.trans row_count_1079.symm
def row_1080 : RowCertificate 1080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1080 1) := by change ¬ Nat.Prime 4667761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 666823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1080 2) := by change ¬ Nat.Prime 4665601; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 13217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1080 3) := by change Nat.Prime 4663441; exact ProjectEulerSolutions.P58PrimeNodes.p_4663441.prime
    exact decide_eq_true prime
theorem row_count_1080 : row 1080 = 1 := (row_1080.checked).trans rfl
theorem model_count_1080 : modelRow 1080 = 1 := by decide
theorem modelrow_agrees_1080 : modelRow 1080 = row 1080 := model_count_1080.trans row_count_1080.symm
def row_1081 : RowCertificate 1081 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1081 1) := by change Nat.Prime 4676407; exact ProjectEulerSolutions.P58PrimeNodes.p_4676407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1081 2) := by change ¬ Nat.Prime 4674245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 934849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1081 3) := by change ¬ Nat.Prime 4672083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1557361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1081 : row 1081 = 1 := (row_1081.checked).trans rfl
theorem model_count_1081 : modelRow 1081 = 1 := by decide
theorem modelrow_agrees_1081 : modelRow 1081 = row 1081 := model_count_1081.trans row_count_1081.symm
def row_1082 : RowCertificate 1082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1082 1) := by change ¬ Nat.Prime 4685061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1561687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1082 2) := by change ¬ Nat.Prime 4682897; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 114217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1082 3) := by change Nat.Prime 4680733; exact ProjectEulerSolutions.P58PrimeNodes.p_4680733.prime
    exact decide_eq_true prime
theorem row_count_1082 : row 1082 = 1 := (row_1082.checked).trans rfl
theorem model_count_1082 : modelRow 1082 = 1 := by decide
theorem modelrow_agrees_1082 : modelRow 1082 = row 1082 := model_count_1082.trans row_count_1082.symm
def row_1083 : RowCertificate 1083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1083 1) := by change ¬ Nat.Prime 4693723; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 15289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1083 2) := by change ¬ Nat.Prime 4691557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 360889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1083 3) := by change ¬ Nat.Prime 4689391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 669913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1083 : row 1083 = 0 := (row_1083.checked).trans rfl
theorem model_count_1083 : modelRow 1083 = 0 := by decide
theorem modelrow_agrees_1083 : modelRow 1083 = row 1083 := model_count_1083.trans row_count_1083.symm
def row_1084 : RowCertificate 1084 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1084 1) := by change Nat.Prime 4702393; exact ProjectEulerSolutions.P58PrimeNodes.p_4702393.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1084 2) := by change ¬ Nat.Prime 4700225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 940045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1084 3) := by change ¬ Nat.Prime 4698057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1566019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1084 : row 1084 = 1 := (row_1084.checked).trans rfl
theorem model_count_1084 : modelRow 1084 = 1 := by decide
theorem modelrow_agrees_1084 : modelRow 1084 = row 1084 := model_count_1084.trans row_count_1084.symm
def row_1085 : RowCertificate 1085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1085 1) := by change ¬ Nat.Prime 4711071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1570357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1085 2) := by change ¬ Nat.Prime 4708901; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 52909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1085 3) := by change ¬ Nat.Prime 4706731; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 48523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1085 : row 1085 = 0 := (row_1085.checked).trans rfl
theorem model_count_1085 : modelRow 1085 = 0 := by decide
theorem modelrow_agrees_1085 : modelRow 1085 = row 1085 := model_count_1085.trans row_count_1085.symm
def row_1086 : RowCertificate 1086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1086 1) := by change ¬ Nat.Prime 4719757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 674251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1086 2) := by change ¬ Nat.Prime 4717585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 943517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1086 3) := by change Nat.Prime 4715413; exact ProjectEulerSolutions.P58PrimeNodes.p_4715413.prime
    exact decide_eq_true prime
theorem row_count_1086 : row 1086 = 1 := (row_1086.checked).trans rfl
theorem model_count_1086 : modelRow 1086 = 1 := by decide
theorem modelrow_agrees_1086 : modelRow 1086 = row 1086 := model_count_1086.trans row_count_1086.symm
def row_1087 : RowCertificate 1087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1087 1) := by change ¬ Nat.Prime 4728451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 675493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1087 2) := by change Nat.Prime 4726277; exact ProjectEulerSolutions.P58PrimeNodes.p_4726277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1087 3) := by change ¬ Nat.Prime 4724103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1574701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1087 : row 1087 = 1 := (row_1087.checked).trans rfl
theorem model_count_1087 : modelRow 1087 = 1 := by decide
theorem modelrow_agrees_1087 : modelRow 1087 = row 1087 := model_count_1087.trans row_count_1087.symm
def row_1088 : RowCertificate 1088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1088 1) := by change ¬ Nat.Prime 4737153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1579051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1088 2) := by change ¬ Nat.Prime 4734977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 364229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1088 3) := by change ¬ Nat.Prime 4732801; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 152671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1088 : row 1088 = 0 := (row_1088.checked).trans rfl
theorem model_count_1088 : modelRow 1088 = 0 := by decide
theorem modelrow_agrees_1088 : modelRow 1088 = row 1088 := model_count_1088.trans row_count_1088.symm
def row_1089 : RowCertificate 1089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1089 1) := by change ¬ Nat.Prime 4745863; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 37369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1089 2) := by change ¬ Nat.Prime 4743685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 948737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1089 3) := by change ¬ Nat.Prime 4741507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 249553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1089 : row 1089 = 0 := (row_1089.checked).trans rfl
theorem model_count_1089 : modelRow 1089 = 0 := by decide
theorem modelrow_agrees_1089 : modelRow 1089 = row 1089 := model_count_1089.trans row_count_1089.symm
def row_1090 : RowCertificate 1090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1090 1) := by change ¬ Nat.Prime 4754581; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 365737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1090 2) := by change ¬ Nat.Prime 4752401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 279553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1090 3) := by change ¬ Nat.Prime 4750221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1583407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1090 : row 1090 = 0 := (row_1090.checked).trans rfl
theorem model_count_1090 : modelRow 1090 = 0 := by decide
theorem modelrow_agrees_1090 : modelRow 1090 = row 1090 := model_count_1090.trans row_count_1090.symm
def row_1091 : RowCertificate 1091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1091 1) := by change ¬ Nat.Prime 4763307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1587769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1091 2) := by change ¬ Nat.Prime 4761125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 952225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1091 3) := by change ¬ Nat.Prime 4758943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 679849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1091 : row 1091 = 0 := (row_1091.checked).trans rfl
theorem model_count_1091 : modelRow 1091 = 0 := by decide
theorem modelrow_agrees_1091 : modelRow 1091 = row 1091 := model_count_1091.trans row_count_1091.symm
def row_1092 : RowCertificate 1092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1092 1) := by change ¬ Nat.Prime 4772041; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 19801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1092 2) := by change ¬ Nat.Prime 4769857; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 6301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1092 3) := by change Nat.Prime 4767673; exact ProjectEulerSolutions.P58PrimeNodes.p_4767673.prime
    exact decide_eq_true prime
theorem row_count_1092 : row 1092 = 1 := (row_1092.checked).trans rfl
theorem model_count_1092 : modelRow 1092 = 1 := by decide
theorem modelrow_agrees_1092 : modelRow 1092 = row 1092 := model_count_1092.trans row_count_1092.symm
def row_1093 : RowCertificate 1093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1093 1) := by change ¬ Nat.Prime 4780783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 682969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1093 2) := by change ¬ Nat.Prime 4778597; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 20509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1093 3) := by change ¬ Nat.Prime 4776411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1592137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1093 : row 1093 = 0 := (row_1093.checked).trans rfl
theorem model_count_1093 : modelRow 1093 = 0 := by decide
theorem modelrow_agrees_1093 : modelRow 1093 = row 1093 := model_count_1093.trans row_count_1093.symm
def row_1094 : RowCertificate 1094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1094 1) := by change ¬ Nat.Prime 4789533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1596511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1094 2) := by change ¬ Nat.Prime 4787345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 957469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1094 3) := by change ¬ Nat.Prime 4785157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 368089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1094 : row 1094 = 0 := (row_1094.checked).trans rfl
theorem model_count_1094 : modelRow 1094 = 0 := by decide
theorem modelrow_agrees_1094 : modelRow 1094 = row 1094 := model_count_1094.trans row_count_1094.symm
def row_1095 : RowCertificate 1095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1095 1) := by change ¬ Nat.Prime 4798291; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 21517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1095 2) := by change ¬ Nat.Prime 4796101; exact PrattCertificates.composite_of_product (factor := 569) (cofactor := 8429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1095 3) := by change ¬ Nat.Prime 4793911; exact PrattCertificates.composite_of_product (factor := 1447) (cofactor := 3313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1095 : row 1095 = 0 := (row_1095.checked).trans rfl
theorem model_count_1095 : modelRow 1095 = 0 := by decide
theorem modelrow_agrees_1095 : modelRow 1095 = row 1095 := model_count_1095.trans row_count_1095.symm
def row_1096 : RowCertificate 1096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1096 1) := by change ¬ Nat.Prime 4807057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 253003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1096 2) := by change ¬ Nat.Prime 4804865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 960973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1096 3) := by change ¬ Nat.Prime 4802673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1600891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1096 : row 1096 = 0 := (row_1096.checked).trans rfl
theorem model_count_1096 : modelRow 1096 = 0 := by decide
theorem modelrow_agrees_1096 : modelRow 1096 = row 1096 := model_count_1096.trans row_count_1096.symm
def row_1097 : RowCertificate 1097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1097 1) := by change ¬ Nat.Prime 4815831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1605277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1097 2) := by change ¬ Nat.Prime 4813637; exact PrattCertificates.composite_of_product (factor := 1721) (cofactor := 2797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1097 3) := by change ¬ Nat.Prime 4811443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 687349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1097 : row 1097 = 0 := (row_1097.checked).trans rfl
theorem model_count_1097 : modelRow 1097 = 0 := by decide
theorem modelrow_agrees_1097 : modelRow 1097 = row 1097 := model_count_1097.trans row_count_1097.symm
def row_1098 : RowCertificate 1098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1098 1) := by change ¬ Nat.Prime 4824613; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 253927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1098 2) := by change ¬ Nat.Prime 4822417; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 90989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1098 3) := by change ¬ Nat.Prime 4820221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 688603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1098 : row 1098 = 0 := (row_1098.checked).trans rfl
theorem model_count_1098 : modelRow 1098 = 0 := by decide
theorem modelrow_agrees_1098 : modelRow 1098 = row 1098 := model_count_1098.trans row_count_1098.symm
def row_1099 : RowCertificate 1099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1099 1) := by change ¬ Nat.Prime 4833403; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 66211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1099 2) := by change ¬ Nat.Prime 4831205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 966241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1099 3) := by change ¬ Nat.Prime 4829007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1609669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1099 : row 1099 = 0 := (row_1099.checked).trans rfl
theorem model_count_1099 : modelRow 1099 = 0 := by decide
theorem modelrow_agrees_1099 : modelRow 1099 = row 1099 := model_count_1099.trans row_count_1099.symm
def row_1100 : RowCertificate 1100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1100 1) := by change ¬ Nat.Prime 4842201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1614067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1100 2) := by change ¬ Nat.Prime 4840001; exact PrattCertificates.composite_of_product (factor := 1193) (cofactor := 4057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1100 3) := by change ¬ Nat.Prime 4837801; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 112507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1100 : row 1100 = 0 := (row_1100.checked).trans rfl
theorem model_count_1100 : modelRow 1100 = 0 := by decide
theorem modelrow_agrees_1100 : modelRow 1100 = row 1100 := model_count_1100.trans row_count_1100.symm
def row_1101 : RowCertificate 1101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1101 1) := by change ¬ Nat.Prime 4851007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 693001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1101 2) := by change ¬ Nat.Prime 4848805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 969761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1101 3) := by change Nat.Prime 4846603; exact ProjectEulerSolutions.P58PrimeNodes.p_4846603.prime
    exact decide_eq_true prime
theorem row_count_1101 : row 1101 = 1 := (row_1101.checked).trans rfl
theorem model_count_1101 : modelRow 1101 = 1 := by decide
theorem modelrow_agrees_1101 : modelRow 1101 = row 1101 := model_count_1101.trans row_count_1101.symm
def row_1102 : RowCertificate 1102 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1102 1) := by change Nat.Prime 4859821; exact ProjectEulerSolutions.P58PrimeNodes.p_4859821.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1102 2) := by change ¬ Nat.Prime 4857617; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 25169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1102 3) := by change ¬ Nat.Prime 4855413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1618471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1102 : row 1102 = 1 := (row_1102.checked).trans rfl
theorem model_count_1102 : modelRow 1102 = 1 := by decide
theorem modelrow_agrees_1102 : modelRow 1102 = row 1102 := model_count_1102.trans row_count_1102.symm
def row_1103 : RowCertificate 1103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1103 1) := by change ¬ Nat.Prime 4868643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1622881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1103 2) := by change ¬ Nat.Prime 4866437; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 286261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1103 3) := by change Nat.Prime 4864231; exact ProjectEulerSolutions.P58PrimeNodes.p_4864231.prime
    exact decide_eq_true prime
theorem row_count_1103 : row 1103 = 1 := (row_1103.checked).trans rfl
theorem model_count_1103 : modelRow 1103 = 1 := by decide
theorem modelrow_agrees_1103 : modelRow 1103 = row 1103 := model_count_1103.trans row_count_1103.symm
def row_1104 : RowCertificate 1104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1104 1) := by change ¬ Nat.Prime 4877473; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 4021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1104 2) := by change ¬ Nat.Prime 4875265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 975053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1104 3) := by change ¬ Nat.Prime 4873057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 696151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1104 : row 1104 = 0 := (row_1104.checked).trans rfl
theorem model_count_1104 : modelRow 1104 = 0 := by decide
theorem modelrow_agrees_1104 : modelRow 1104 = row 1104 := model_count_1104.trans row_count_1104.symm
def row_1105 : RowCertificate 1105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1105 1) := by change ¬ Nat.Prime 4886311; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 31123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1105 2) := by change Nat.Prime 4884101; exact ProjectEulerSolutions.P58PrimeNodes.p_4884101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1105 3) := by change ¬ Nat.Prime 4881891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1627297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1105 : row 1105 = 1 := (row_1105.checked).trans rfl
theorem model_count_1105 : modelRow 1105 = 1 := by decide
theorem modelrow_agrees_1105 : modelRow 1105 = row 1105 := model_count_1105.trans row_count_1105.symm
def row_1106 : RowCertificate 1106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1106 1) := by change ¬ Nat.Prime 4895157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1631719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1106 2) := by change ¬ Nat.Prime 4892945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 978589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1106 3) := by change ¬ Nat.Prime 4890733; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 257407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1106 : row 1106 = 0 := (row_1106.checked).trans rfl
theorem model_count_1106 : modelRow 1106 = 0 := by decide
theorem modelrow_agrees_1106 : modelRow 1106 = row 1106 := model_count_1106.trans row_count_1106.symm
def row_1107 : RowCertificate 1107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1107 1) := by change ¬ Nat.Prime 4904011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 700573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1107 2) := by change ¬ Nat.Prime 4901797; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 288341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1107 3) := by change ¬ Nat.Prime 4899583; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 376891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1107 : row 1107 = 0 := (row_1107.checked).trans rfl
theorem model_count_1107 : modelRow 1107 = 0 := by decide
theorem modelrow_agrees_1107 : modelRow 1107 = row 1107 := model_count_1107.trans row_count_1107.symm
def row_1108 : RowCertificate 1108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1108 1) := by change ¬ Nat.Prime 4912873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 701839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1108 2) := by change ¬ Nat.Prime 4910657; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 169333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1108 3) := by change ¬ Nat.Prime 4908441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1636147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1108 : row 1108 = 0 := (row_1108.checked).trans rfl
theorem model_count_1108 : modelRow 1108 = 0 := by decide
theorem modelrow_agrees_1108 : modelRow 1108 = row 1108 := model_count_1108.trans row_count_1108.symm
def row_1109 : RowCertificate 1109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1109 1) := by change ¬ Nat.Prime 4921743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1640581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1109 2) := by change ¬ Nat.Prime 4919525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 983905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1109 3) := by change ¬ Nat.Prime 4917307; exact PrattCertificates.composite_of_product (factor := 607) (cofactor := 8101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1109 : row 1109 = 0 := (row_1109.checked).trans rfl
theorem model_count_1109 : modelRow 1109 = 0 := by decide
theorem modelrow_agrees_1109 : modelRow 1109 = row 1109 := model_count_1109.trans row_count_1109.symm
def row_1110 : RowCertificate 1110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1110 1) := by change ¬ Nat.Prime 4930621; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 27241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1110 2) := by change ¬ Nat.Prime 4928401; exact PrattCertificates.composite_of_product (factor := 1889) (cofactor := 2609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1110 3) := by change ¬ Nat.Prime 4926181; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 378937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1110 : row 1110 = 0 := (row_1110.checked).trans rfl
theorem model_count_1110 : modelRow 1110 = 0 := by decide
theorem modelrow_agrees_1110 : modelRow 1110 = row 1110 := model_count_1110.trans row_count_1110.symm
def row_1111 : RowCertificate 1111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1111 1) := by change ¬ Nat.Prime 4939507; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 13033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1111 2) := by change ¬ Nat.Prime 4937285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 987457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1111 3) := by change ¬ Nat.Prime 4935063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1645021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1111 : row 1111 = 0 := (row_1111.checked).trans rfl
theorem model_count_1111 : modelRow 1111 = 0 := by decide
theorem modelrow_agrees_1111 : modelRow 1111 = row 1111 := model_count_1111.trans row_count_1111.symm
def row_1112 : RowCertificate 1112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1112 1) := by change ¬ Nat.Prime 4948401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1649467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1112 2) := by change Nat.Prime 4946177; exact ProjectEulerSolutions.P58PrimeNodes.p_4946177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1112 3) := by change ¬ Nat.Prime 4943953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 706279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1112 : row 1112 = 1 := (row_1112.checked).trans rfl
theorem model_count_1112 : modelRow 1112 = 1 := by decide
theorem modelrow_agrees_1112 : modelRow 1112 = row 1112 := model_count_1112.trans row_count_1112.symm
def row_1113 : RowCertificate 1113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1113 1) := by change ¬ Nat.Prime 4957303; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 381331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1113 2) := by change ¬ Nat.Prime 4955077; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 133921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1113 3) := by change ¬ Nat.Prime 4952851; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 45439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1113 : row 1113 = 0 := (row_1113.checked).trans rfl
theorem model_count_1113 : modelRow 1113 = 0 := by decide
theorem modelrow_agrees_1113 : modelRow 1113 = row 1113 := model_count_1113.trans row_count_1113.symm
def row_1114 : RowCertificate 1114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1114 1) := by change ¬ Nat.Prime 4966213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 709459) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1114 2) := by change ¬ Nat.Prime 4963985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 992797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1114 3) := by change ¬ Nat.Prime 4961757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1653919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1114 : row 1114 = 0 := (row_1114.checked).trans rfl
theorem model_count_1114 : modelRow 1114 = 0 := by decide
theorem modelrow_agrees_1114 : modelRow 1114 = row 1114 := model_count_1114.trans row_count_1114.symm
def row_1115 : RowCertificate 1115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1115 1) := by change ¬ Nat.Prime 4975131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1658377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1115 2) := by change ¬ Nat.Prime 4972901; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 14249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1115 3) := by change ¬ Nat.Prime 4970671; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 115597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1115 : row 1115 = 0 := (row_1115.checked).trans rfl
theorem model_count_1115 : modelRow 1115 = 0 := by decide
theorem modelrow_agrees_1115 : modelRow 1115 = row 1115 := model_count_1115.trans row_count_1115.symm
def row_1116 : RowCertificate 1116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1116 1) := by change ¬ Nat.Prime 4984057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 383389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1116 2) := by change ¬ Nat.Prime 4981825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 996365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1116 3) := by change ¬ Nat.Prime 4979593; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 25801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1116 : row 1116 = 0 := (row_1116.checked).trans rfl
theorem model_count_1116 : modelRow 1116 = 0 := by decide
theorem modelrow_agrees_1116 : modelRow 1116 = row 1116 := model_count_1116.trans row_count_1116.symm
def row_1117 : RowCertificate 1117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1117 1) := by change ¬ Nat.Prime 4992991; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 262789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1117 2) := by change ¬ Nat.Prime 4990757; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 18553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1117 3) := by change ¬ Nat.Prime 4988523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1662841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1117 : row 1117 = 0 := (row_1117.checked).trans rfl
theorem model_count_1117 : modelRow 1117 = 0 := by decide
theorem modelrow_agrees_1117 : modelRow 1117 = row 1117 := model_count_1117.trans row_count_1117.symm
def row_1118 : RowCertificate 1118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1118 1) := by change ¬ Nat.Prime 5001933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1667311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1118 2) := by change ¬ Nat.Prime 4999697; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 68489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1118 3) := by change ¬ Nat.Prime 4997461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 713923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1118 : row 1118 = 0 := (row_1118.checked).trans rfl
theorem model_count_1118 : modelRow 1118 = 0 := by decide
theorem modelrow_agrees_1118 : modelRow 1118 = row 1118 := model_count_1118.trans row_count_1118.symm
def row_1119 : RowCertificate 1119 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1119 1) := by change Nat.Prime 5010883; exact ProjectEulerSolutions.P58PrimeNodes.p_5010883.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1119 2) := by change ¬ Nat.Prime 5008645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1001729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1119 3) := by change ¬ Nat.Prime 5006407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 715201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1119 : row 1119 = 1 := (row_1119.checked).trans rfl
theorem model_count_1119 : modelRow 1119 = 1 := by decide
theorem modelrow_agrees_1119 : modelRow 1119 = row 1119 := model_count_1119.trans row_count_1119.symm
def row_1120 : RowCertificate 1120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1120 1) := by change ¬ Nat.Prime 5019841; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 74923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1120 2) := by change ¬ Nat.Prime 5017601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 295153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1120 3) := by change ¬ Nat.Prime 5015361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1671787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1120 : row 1120 = 0 := (row_1120.checked).trans rfl
theorem model_count_1120 : modelRow 1120 = 0 := by decide
theorem modelrow_agrees_1120 : modelRow 1120 = row 1120 := model_count_1120.trans row_count_1120.symm
def row_1121 : RowCertificate 1121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1121 1) := by change ¬ Nat.Prime 5028807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1676269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1121 2) := by change ¬ Nat.Prime 5026565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1005313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1121 3) := by change Nat.Prime 5024323; exact ProjectEulerSolutions.P58PrimeNodes.p_5024323.prime
    exact decide_eq_true prime
theorem row_count_1121 : row 1121 = 1 := (row_1121.checked).trans rfl
theorem model_count_1121 : modelRow 1121 = 1 := by decide
theorem modelrow_agrees_1121 : modelRow 1121 = row 1121 := model_count_1121.trans row_count_1121.symm
def row_1122 : RowCertificate 1122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1122 1) := by change ¬ Nat.Prime 5037781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 719683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1122 2) := by change ¬ Nat.Prime 5035537; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 387349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1122 3) := by change ¬ Nat.Prime 5033293; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 82513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1122 : row 1122 = 0 := (row_1122.checked).trans rfl
theorem model_count_1122 : modelRow 1122 = 0 := by decide
theorem modelrow_agrees_1122 : modelRow 1122 = row 1122 := model_count_1122.trans row_count_1122.symm
def row_1123 : RowCertificate 1123 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1123 1) := by change ¬ Nat.Prime 5046763; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 136399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1123 2) := by change ¬ Nat.Prime 5044517; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 123037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1123 3) := by change ¬ Nat.Prime 5042271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1680757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1123 : row 1123 = 0 := (row_1123.checked).trans rfl
theorem model_count_1123 : modelRow 1123 = 0 := by decide
theorem modelrow_agrees_1123 : modelRow 1123 = row 1123 := model_count_1123.trans row_count_1123.symm
def row_1124 : RowCertificate 1124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1124 1) := by change ¬ Nat.Prime 5055753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1685251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1124 2) := by change ¬ Nat.Prime 5053505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1010701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1124 3) := by change Nat.Prime 5051257; exact ProjectEulerSolutions.P58PrimeNodes.p_5051257.prime
    exact decide_eq_true prime
theorem row_count_1124 : row 1124 = 1 := (row_1124.checked).trans rfl
theorem model_count_1124 : modelRow 1124 = 1 := by decide
theorem modelrow_agrees_1124 : modelRow 1124 = row 1124 := model_count_1124.trans row_count_1124.symm
def row_1125 : RowCertificate 1125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1125 1) := by change ¬ Nat.Prime 5064751; exact PrattCertificates.composite_of_product (factor := 2053) (cofactor := 2467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1125 2) := by change ¬ Nat.Prime 5062501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 174569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1125 3) := by change ¬ Nat.Prime 5060251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 722893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1125 : row 1125 = 0 := (row_1125.checked).trans rfl
theorem model_count_1125 : modelRow 1125 = 0 := by decide
theorem modelrow_agrees_1125 : modelRow 1125 = row 1125 := model_count_1125.trans row_count_1125.symm
def row_1126 : RowCertificate 1126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1126 1) := by change ¬ Nat.Prime 5073757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 390289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1126 2) := by change ¬ Nat.Prime 5071505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1014301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1126 3) := by change ¬ Nat.Prime 5069253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1689751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1126 : row 1126 = 0 := (row_1126.checked).trans rfl
theorem model_count_1126 : modelRow 1126 = 0 := by decide
theorem modelrow_agrees_1126 : modelRow 1126 = row 1126 := model_count_1126.trans row_count_1126.symm
def row_1127 : RowCertificate 1127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1127 1) := by change ¬ Nat.Prime 5082771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1694257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1127 2) := by change ¬ Nat.Prime 5080517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 390809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1127 3) := by change ¬ Nat.Prime 5078263; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 267277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1127 : row 1127 = 0 := (row_1127.checked).trans rfl
theorem model_count_1127 : modelRow 1127 = 0 := by decide
theorem modelrow_agrees_1127 : modelRow 1127 = row 1127 := model_count_1127.trans row_count_1127.symm
def row_1128 : RowCertificate 1128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1128 1) := by change ¬ Nat.Prime 5091793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 727399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1128 2) := by change ¬ Nat.Prime 5089537; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 96029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1128 3) := by change Nat.Prime 5087281; exact ProjectEulerSolutions.P58PrimeNodes.p_5087281.prime
    exact decide_eq_true prime
theorem row_count_1128 : row 1128 = 1 := (row_1128.checked).trans rfl
theorem model_count_1128 : modelRow 1128 = 1 := by decide
theorem modelrow_agrees_1128 : modelRow 1128 = row 1128 := model_count_1128.trans row_count_1128.symm
def row_1129 : RowCertificate 1129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1129 1) := by change ¬ Nat.Prime 5100823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 728689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1129 2) := by change ¬ Nat.Prime 5098565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1019713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1129 3) := by change ¬ Nat.Prime 5096307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1698769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1129 : row 1129 = 0 := (row_1129.checked).trans rfl
theorem model_count_1129 : modelRow 1129 = 0 := by decide
theorem modelrow_agrees_1129 : modelRow 1129 = row 1129 := model_count_1129.trans row_count_1129.symm
def row_1130 : RowCertificate 1130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1130 1) := by change ¬ Nat.Prime 5109861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1703287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1130 2) := by change Nat.Prime 5107601; exact ProjectEulerSolutions.P58PrimeNodes.p_5107601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1130 3) := by change Nat.Prime 5105341; exact ProjectEulerSolutions.P58PrimeNodes.p_5105341.prime
    exact decide_eq_true prime
theorem row_count_1130 : row 1130 = 2 := (row_1130.checked).trans rfl
theorem model_count_1130 : modelRow 1130 = 2 := by decide
theorem modelrow_agrees_1130 : modelRow 1130 = row 1130 := model_count_1130.trans row_count_1130.symm
def row_1131 : RowCertificate 1131 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1131 1) := by change Nat.Prime 5118907; exact ProjectEulerSolutions.P58PrimeNodes.p_5118907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1131 2) := by change ¬ Nat.Prime 5116645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1023329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1131 3) := by change ¬ Nat.Prime 5114383; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 4951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1131 : row 1131 = 1 := (row_1131.checked).trans rfl
theorem model_count_1131 : modelRow 1131 = 1 := by decide
theorem modelrow_agrees_1131 : modelRow 1131 = row 1131 := model_count_1131.trans row_count_1131.symm
def row_1132 : RowCertificate 1132 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1132 1) := by change Nat.Prime 5127961; exact ProjectEulerSolutions.P58PrimeNodes.p_5127961.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1132 2) := by change ¬ Nat.Prime 5125697; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 125017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1132 3) := by change ¬ Nat.Prime 5123433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1707811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1132 : row 1132 = 1 := (row_1132.checked).trans rfl
theorem model_count_1132 : modelRow 1132 = 1 := by decide
theorem modelrow_agrees_1132 : modelRow 1132 = row 1132 := model_count_1132.trans row_count_1132.symm
def row_1133 : RowCertificate 1133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1133 1) := by change ¬ Nat.Prime 5137023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1712341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1133 2) := by change Nat.Prime 5134757; exact ProjectEulerSolutions.P58PrimeNodes.p_5134757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1133 3) := by change ¬ Nat.Prime 5132491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 733213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1133 : row 1133 = 1 := (row_1133.checked).trans rfl
theorem model_count_1133 : modelRow 1133 = 1 := by decide
theorem modelrow_agrees_1133 : modelRow 1133 = row 1133 := model_count_1133.trans row_count_1133.symm
def row_1134 : RowCertificate 1134 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1134 1) := by change ¬ Nat.Prime 5146093; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 270847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1134 2) := by change ¬ Nat.Prime 5143825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1028765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1134 3) := by change ¬ Nat.Prime 5141557; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 138961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1134 : row 1134 = 0 := (row_1134.checked).trans rfl
theorem model_count_1134 : modelRow 1134 = 0 := by decide
theorem modelrow_agrees_1134 : modelRow 1134 = row 1134 := model_count_1134.trans row_count_1134.symm
def row_1135 : RowCertificate 1135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1135 1) := by change ¬ Nat.Prime 5155171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 736453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1135 2) := by change ¬ Nat.Prime 5152901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 396377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1135 3) := by change ¬ Nat.Prime 5150631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1716877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1135 : row 1135 = 0 := (row_1135.checked).trans rfl
theorem model_count_1135 : modelRow 1135 = 0 := by decide
theorem modelrow_agrees_1135 : modelRow 1135 = row 1135 := model_count_1135.trans row_count_1135.symm
def row_1136 : RowCertificate 1136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1136 1) := by change ¬ Nat.Prime 5164257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1721419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1136 2) := by change ¬ Nat.Prime 5161985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1032397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1136 3) := by change ¬ Nat.Prime 5159713; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 396901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1136 : row 1136 = 0 := (row_1136.checked).trans rfl
theorem model_count_1136 : modelRow 1136 = 0 := by decide
theorem modelrow_agrees_1136 : modelRow 1136 = row 1136 := model_count_1136.trans row_count_1136.symm
def row_1137 : RowCertificate 1137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1137 1) := by change ¬ Nat.Prime 5173351; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 7687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1137 2) := by change ¬ Nat.Prime 5171077; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 304181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1137 3) := by change Nat.Prime 5168803; exact ProjectEulerSolutions.P58PrimeNodes.p_5168803.prime
    exact decide_eq_true prime
theorem row_count_1137 : row 1137 = 1 := (row_1137.checked).trans rfl
theorem model_count_1137 : modelRow 1137 = 1 := by decide
theorem modelrow_agrees_1137 : modelRow 1137 = row 1137 := model_count_1137.trans row_count_1137.symm
def row_1138 : RowCertificate 1138 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1138 1) := by change Nat.Prime 5182453; exact ProjectEulerSolutions.P58PrimeNodes.p_5182453.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1138 2) := by change ¬ Nat.Prime 5180177; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 18701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1138 3) := by change ¬ Nat.Prime 5177901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1725967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1138 : row 1138 = 1 := (row_1138.checked).trans rfl
theorem model_count_1138 : modelRow 1138 = 1 := by decide
theorem modelrow_agrees_1138 : modelRow 1138 = row 1138 := model_count_1138.trans row_count_1138.symm
def row_1139 : RowCertificate 1139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1139 1) := by change ¬ Nat.Prime 5191563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1730521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1139 2) := by change ¬ Nat.Prime 5189285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1037857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1139 3) := by change ¬ Nat.Prime 5187007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 741001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1139 : row 1139 = 0 := (row_1139.checked).trans rfl
theorem model_count_1139 : modelRow 1139 = 0 := by decide
theorem modelrow_agrees_1139 : modelRow 1139 = row 1139 := model_count_1139.trans row_count_1139.symm
def row_1140 : RowCertificate 1140 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1140 1) := by change Nat.Prime 5200681; exact ProjectEulerSolutions.P58PrimeNodes.p_5200681.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1140 2) := by change ¬ Nat.Prime 5198401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 399877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1140 3) := by change ¬ Nat.Prime 5196121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 742303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1140 : row 1140 = 1 := (row_1140.checked).trans rfl
theorem model_count_1140 : modelRow 1140 = 1 := by decide
theorem modelrow_agrees_1140 : modelRow 1140 = row 1140 := model_count_1140.trans row_count_1140.symm
def row_1141 : RowCertificate 1141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1141 1) := by change ¬ Nat.Prime 5209807; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 4957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1141 2) := by change ¬ Nat.Prime 5207525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1041505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1141 3) := by change ¬ Nat.Prime 5205243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1735081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1141 : row 1141 = 0 := (row_1141.checked).trans rfl
theorem model_count_1141 : modelRow 1141 = 0 := by decide
theorem modelrow_agrees_1141 : modelRow 1141 = row 1141 := model_count_1141.trans row_count_1141.symm
def row_1142 : RowCertificate 1142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1142 1) := by change ¬ Nat.Prime 5218941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1739647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1142 2) := by change ¬ Nat.Prime 5216657; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 9041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1142 3) := by change ¬ Nat.Prime 5214373; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 140929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1142 : row 1142 = 0 := (row_1142.checked).trans rfl
theorem model_count_1142 : modelRow 1142 = 0 := by decide
theorem modelrow_agrees_1142 : modelRow 1142 = row 1142 := model_count_1142.trans row_count_1142.symm
def row_1143 : RowCertificate 1143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1143 1) := by change ¬ Nat.Prime 5228083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 746869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1143 2) := by change Nat.Prime 5225797; exact ProjectEulerSolutions.P58PrimeNodes.p_5225797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1143 3) := by change ¬ Nat.Prime 5223511; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 121477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1143 : row 1143 = 1 := (row_1143.checked).trans rfl
theorem model_count_1143 : modelRow 1143 = 1 := by decide
theorem modelrow_agrees_1143 : modelRow 1143 = row 1143 := model_count_1143.trans row_count_1143.symm
def row_1144 : RowCertificate 1144 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1144 1) := by change ¬ Nat.Prime 5237233; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 168943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1144 2) := by change ¬ Nat.Prime 5234945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1046989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1144 3) := by change ¬ Nat.Prime 5232657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1744219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1144 : row 1144 = 0 := (row_1144.checked).trans rfl
theorem model_count_1144 : modelRow 1144 = 0 := by decide
theorem modelrow_agrees_1144 : modelRow 1144 = row 1144 := model_count_1144.trans row_count_1144.symm
def row_1145 : RowCertificate 1145 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1145 1) := by change ¬ Nat.Prime 5246391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1748797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1145 2) := by change ¬ Nat.Prime 5244101; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 71837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1145 3) := by change ¬ Nat.Prime 5241811; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 16747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1145 : row 1145 = 0 := (row_1145.checked).trans rfl
theorem model_count_1145 : modelRow 1145 = 0 := by decide
theorem modelrow_agrees_1145 : modelRow 1145 = row 1145 := model_count_1145.trans row_count_1145.symm
def row_1146 : RowCertificate 1146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1146 1) := by change ¬ Nat.Prime 5255557; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 54181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1146 2) := by change ¬ Nat.Prime 5253265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1050653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1146 3) := by change ¬ Nat.Prime 5250973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 750139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1146 : row 1146 = 0 := (row_1146.checked).trans rfl
theorem model_count_1146 : modelRow 1146 = 0 := by decide
theorem modelrow_agrees_1146 : modelRow 1146 = row 1146 := model_count_1146.trans row_count_1146.symm
def row_1147 : RowCertificate 1147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1147 1) := by change ¬ Nat.Prime 5264731; exact PrattCertificates.composite_of_product (factor := 823) (cofactor := 6397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1147 2) := by change Nat.Prime 5262437; exact ProjectEulerSolutions.P58PrimeNodes.p_5262437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1147 3) := by change ¬ Nat.Prime 5260143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1753381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1147 : row 1147 = 1 := (row_1147.checked).trans rfl
theorem model_count_1147 : modelRow 1147 = 1 := by decide
theorem modelrow_agrees_1147 : modelRow 1147 = row 1147 := model_count_1147.trans row_count_1147.symm
def row_1148 : RowCertificate 1148 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1148 1) := by change ¬ Nat.Prime 5273913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1757971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1148 2) := by change ¬ Nat.Prime 5271617; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 405509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1148 3) := by change ¬ Nat.Prime 5269321; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 26479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1148 : row 1148 = 0 := (row_1148.checked).trans rfl
theorem model_count_1148 : modelRow 1148 = 0 := by decide
theorem modelrow_agrees_1148 : modelRow 1148 = row 1148 := model_count_1148.trans row_count_1148.symm
def row_1149 : RowCertificate 1149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1149 1) := by change ¬ Nat.Prime 5283103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 754729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1149 2) := by change ¬ Nat.Prime 5280805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1056161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1149 3) := by change ¬ Nat.Prime 5278507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 406039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1149 : row 1149 = 0 := (row_1149.checked).trans rfl
theorem model_count_1149 : modelRow 1149 = 0 := by decide
theorem modelrow_agrees_1149 : modelRow 1149 = row 1149 := model_count_1149.trans row_count_1149.symm
def row_1150 : RowCertificate 1150 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1150 1) := by change ¬ Nat.Prime 5292301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 756043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1150 2) := by change ¬ Nat.Prime 5290001; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 142973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1150 3) := by change ¬ Nat.Prime 5287701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1762567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1150 : row 1150 = 0 := (row_1150.checked).trans rfl
theorem model_count_1150 : modelRow 1150 = 0 := by decide
theorem modelrow_agrees_1150 : modelRow 1150 = row 1150 := model_count_1150.trans row_count_1150.symm
def row_1151 : RowCertificate 1151 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1151 1) := by change ¬ Nat.Prime 5301507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1767169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1151 2) := by change ¬ Nat.Prime 5299205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1059841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1151 3) := by change Nat.Prime 5296903; exact ProjectEulerSolutions.P58PrimeNodes.p_5296903.prime
    exact decide_eq_true prime
theorem row_count_1151 : row 1151 = 1 := (row_1151.checked).trans rfl
theorem model_count_1151 : modelRow 1151 = 1 := by decide
theorem modelrow_agrees_1151 : modelRow 1151 = row 1151 := model_count_1151.trans row_count_1151.symm
def row_1152 : RowCertificate 1152 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1152 1) := by change ¬ Nat.Prime 5310721; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 408517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1152 2) := by change Nat.Prime 5308417; exact ProjectEulerSolutions.P58PrimeNodes.p_5308417.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1152 3) := by change Nat.Prime 5306113; exact ProjectEulerSolutions.P58PrimeNodes.p_5306113.prime
    exact decide_eq_true prime
theorem row_count_1152 : row 1152 = 2 := (row_1152.checked).trans rfl
theorem model_count_1152 : modelRow 1152 = 2 := by decide
theorem modelrow_agrees_1152 : modelRow 1152 = row 1152 := model_count_1152.trans row_count_1152.symm
private def atomic_1025 : CountSegment 1 10 13120 1024 1025 563 565 := CountSegment.single (first := 1024) (initial := 563) row_count_1025 (by decide)
private theorem atomic_1025_agreement : RowAgreement 1024 1025 := RowAgreement.single (first := 1024) modelrow_agrees_1025
private def atomic_1026 : CountSegment 1 10 13120 1025 1026 565 565 := CountSegment.single (first := 1025) (initial := 565) row_count_1026 (by decide)
private theorem atomic_1026_agreement : RowAgreement 1025 1026 := RowAgreement.single (first := 1025) modelrow_agrees_1026
private def atomic_1027 : CountSegment 1 10 13120 1026 1027 565 567 := CountSegment.single (first := 1026) (initial := 565) row_count_1027 (by decide)
private theorem atomic_1027_agreement : RowAgreement 1026 1027 := RowAgreement.single (first := 1026) modelrow_agrees_1027
private def atomic_1028 : CountSegment 1 10 13120 1027 1028 567 568 := CountSegment.single (first := 1027) (initial := 567) row_count_1028 (by decide)
private theorem atomic_1028_agreement : RowAgreement 1027 1028 := RowAgreement.single (first := 1027) modelrow_agrees_1028
private def atomic_1029 : CountSegment 1 10 13120 1028 1029 568 569 := CountSegment.single (first := 1028) (initial := 568) row_count_1029 (by decide)
private theorem atomic_1029_agreement : RowAgreement 1028 1029 := RowAgreement.single (first := 1028) modelrow_agrees_1029
private def atomic_1030 : CountSegment 1 10 13120 1029 1030 569 569 := CountSegment.single (first := 1029) (initial := 569) row_count_1030 (by decide)
private theorem atomic_1030_agreement : RowAgreement 1029 1030 := RowAgreement.single (first := 1029) modelrow_agrees_1030
private def atomic_1031 : CountSegment 1 10 13120 1030 1031 569 569 := CountSegment.single (first := 1030) (initial := 569) row_count_1031 (by decide)
private theorem atomic_1031_agreement : RowAgreement 1030 1031 := RowAgreement.single (first := 1030) modelrow_agrees_1031
private def atomic_1032 : CountSegment 1 10 13120 1031 1032 569 571 := CountSegment.single (first := 1031) (initial := 569) row_count_1032 (by decide)
private theorem atomic_1032_agreement : RowAgreement 1031 1032 := RowAgreement.single (first := 1031) modelrow_agrees_1032
private def atomic_1033 : CountSegment 1 10 13120 1032 1033 571 571 := CountSegment.single (first := 1032) (initial := 571) row_count_1033 (by decide)
private theorem atomic_1033_agreement : RowAgreement 1032 1033 := RowAgreement.single (first := 1032) modelrow_agrees_1033
private def atomic_1034 : CountSegment 1 10 13120 1033 1034 571 571 := CountSegment.single (first := 1033) (initial := 571) row_count_1034 (by decide)
private theorem atomic_1034_agreement : RowAgreement 1033 1034 := RowAgreement.single (first := 1033) modelrow_agrees_1034
private def atomic_1035 : CountSegment 1 10 13120 1034 1035 571 571 := CountSegment.single (first := 1034) (initial := 571) row_count_1035 (by decide)
private theorem atomic_1035_agreement : RowAgreement 1034 1035 := RowAgreement.single (first := 1034) modelrow_agrees_1035
private def atomic_1036 : CountSegment 1 10 13120 1035 1036 571 571 := CountSegment.single (first := 1035) (initial := 571) row_count_1036 (by decide)
private theorem atomic_1036_agreement : RowAgreement 1035 1036 := RowAgreement.single (first := 1035) modelrow_agrees_1036
private def atomic_1037 : CountSegment 1 10 13120 1036 1037 571 573 := CountSegment.single (first := 1036) (initial := 571) row_count_1037 (by decide)
private theorem atomic_1037_agreement : RowAgreement 1036 1037 := RowAgreement.single (first := 1036) modelrow_agrees_1037
private def atomic_1038 : CountSegment 1 10 13120 1037 1038 573 574 := CountSegment.single (first := 1037) (initial := 573) row_count_1038 (by decide)
private theorem atomic_1038_agreement : RowAgreement 1037 1038 := RowAgreement.single (first := 1037) modelrow_agrees_1038
private def atomic_1039 : CountSegment 1 10 13120 1038 1039 574 574 := CountSegment.single (first := 1038) (initial := 574) row_count_1039 (by decide)
private theorem atomic_1039_agreement : RowAgreement 1038 1039 := RowAgreement.single (first := 1038) modelrow_agrees_1039
private def atomic_1040 : CountSegment 1 10 13120 1039 1040 574 576 := CountSegment.single (first := 1039) (initial := 574) row_count_1040 (by decide)
private theorem atomic_1040_agreement : RowAgreement 1039 1040 := RowAgreement.single (first := 1039) modelrow_agrees_1040
private def atomic_1041 : CountSegment 1 10 13120 1040 1041 576 576 := CountSegment.single (first := 1040) (initial := 576) row_count_1041 (by decide)
private theorem atomic_1041_agreement : RowAgreement 1040 1041 := RowAgreement.single (first := 1040) modelrow_agrees_1041
private def atomic_1042 : CountSegment 1 10 13120 1041 1042 576 577 := CountSegment.single (first := 1041) (initial := 576) row_count_1042 (by decide)
private theorem atomic_1042_agreement : RowAgreement 1041 1042 := RowAgreement.single (first := 1041) modelrow_agrees_1042
private def atomic_1043 : CountSegment 1 10 13120 1042 1043 577 579 := CountSegment.single (first := 1042) (initial := 577) row_count_1043 (by decide)
private theorem atomic_1043_agreement : RowAgreement 1042 1043 := RowAgreement.single (first := 1042) modelrow_agrees_1043
private def atomic_1044 : CountSegment 1 10 13120 1043 1044 579 579 := CountSegment.single (first := 1043) (initial := 579) row_count_1044 (by decide)
private theorem atomic_1044_agreement : RowAgreement 1043 1044 := RowAgreement.single (first := 1043) modelrow_agrees_1044
private def atomic_1045 : CountSegment 1 10 13120 1044 1045 579 579 := CountSegment.single (first := 1044) (initial := 579) row_count_1045 (by decide)
private theorem atomic_1045_agreement : RowAgreement 1044 1045 := RowAgreement.single (first := 1044) modelrow_agrees_1045
private def atomic_1046 : CountSegment 1 10 13120 1045 1046 579 580 := CountSegment.single (first := 1045) (initial := 579) row_count_1046 (by decide)
private theorem atomic_1046_agreement : RowAgreement 1045 1046 := RowAgreement.single (first := 1045) modelrow_agrees_1046
private def atomic_1047 : CountSegment 1 10 13120 1046 1047 580 581 := CountSegment.single (first := 1046) (initial := 580) row_count_1047 (by decide)
private theorem atomic_1047_agreement : RowAgreement 1046 1047 := RowAgreement.single (first := 1046) modelrow_agrees_1047
private def atomic_1048 : CountSegment 1 10 13120 1047 1048 581 582 := CountSegment.single (first := 1047) (initial := 581) row_count_1048 (by decide)
private theorem atomic_1048_agreement : RowAgreement 1047 1048 := RowAgreement.single (first := 1047) modelrow_agrees_1048
private def atomic_1049 : CountSegment 1 10 13120 1048 1049 582 582 := CountSegment.single (first := 1048) (initial := 582) row_count_1049 (by decide)
private theorem atomic_1049_agreement : RowAgreement 1048 1049 := RowAgreement.single (first := 1048) modelrow_agrees_1049
private def atomic_1050 : CountSegment 1 10 13120 1049 1050 582 582 := CountSegment.single (first := 1049) (initial := 582) row_count_1050 (by decide)
private theorem atomic_1050_agreement : RowAgreement 1049 1050 := RowAgreement.single (first := 1049) modelrow_agrees_1050
private def atomic_1051 : CountSegment 1 10 13120 1050 1051 582 582 := CountSegment.single (first := 1050) (initial := 582) row_count_1051 (by decide)
private theorem atomic_1051_agreement : RowAgreement 1050 1051 := RowAgreement.single (first := 1050) modelrow_agrees_1051
private def atomic_1052 : CountSegment 1 10 13120 1051 1052 582 582 := CountSegment.single (first := 1051) (initial := 582) row_count_1052 (by decide)
private theorem atomic_1052_agreement : RowAgreement 1051 1052 := RowAgreement.single (first := 1051) modelrow_agrees_1052
private def atomic_1053 : CountSegment 1 10 13120 1052 1053 582 584 := CountSegment.single (first := 1052) (initial := 582) row_count_1053 (by decide)
private theorem atomic_1053_agreement : RowAgreement 1052 1053 := RowAgreement.single (first := 1052) modelrow_agrees_1053
private def atomic_1054 : CountSegment 1 10 13120 1053 1054 584 584 := CountSegment.single (first := 1053) (initial := 584) row_count_1054 (by decide)
private theorem atomic_1054_agreement : RowAgreement 1053 1054 := RowAgreement.single (first := 1053) modelrow_agrees_1054
private def atomic_1055 : CountSegment 1 10 13120 1054 1055 584 584 := CountSegment.single (first := 1054) (initial := 584) row_count_1055 (by decide)
private theorem atomic_1055_agreement : RowAgreement 1054 1055 := RowAgreement.single (first := 1054) modelrow_agrees_1055
private def atomic_1056 : CountSegment 1 10 13120 1055 1056 584 585 := CountSegment.single (first := 1055) (initial := 584) row_count_1056 (by decide)
private theorem atomic_1056_agreement : RowAgreement 1055 1056 := RowAgreement.single (first := 1055) modelrow_agrees_1056
private def atomic_1057 : CountSegment 1 10 13120 1056 1057 585 585 := CountSegment.single (first := 1056) (initial := 585) row_count_1057 (by decide)
private theorem atomic_1057_agreement : RowAgreement 1056 1057 := RowAgreement.single (first := 1056) modelrow_agrees_1057
private def atomic_1058 : CountSegment 1 10 13120 1057 1058 585 586 := CountSegment.single (first := 1057) (initial := 585) row_count_1058 (by decide)
private theorem atomic_1058_agreement : RowAgreement 1057 1058 := RowAgreement.single (first := 1057) modelrow_agrees_1058
private def atomic_1059 : CountSegment 1 10 13120 1058 1059 586 587 := CountSegment.single (first := 1058) (initial := 586) row_count_1059 (by decide)
private theorem atomic_1059_agreement : RowAgreement 1058 1059 := RowAgreement.single (first := 1058) modelrow_agrees_1059
private def atomic_1060 : CountSegment 1 10 13120 1059 1060 587 588 := CountSegment.single (first := 1059) (initial := 587) row_count_1060 (by decide)
private theorem atomic_1060_agreement : RowAgreement 1059 1060 := RowAgreement.single (first := 1059) modelrow_agrees_1060
private def atomic_1061 : CountSegment 1 10 13120 1060 1061 588 588 := CountSegment.single (first := 1060) (initial := 588) row_count_1061 (by decide)
private theorem atomic_1061_agreement : RowAgreement 1060 1061 := RowAgreement.single (first := 1060) modelrow_agrees_1061
private def atomic_1062 : CountSegment 1 10 13120 1061 1062 588 588 := CountSegment.single (first := 1061) (initial := 588) row_count_1062 (by decide)
private theorem atomic_1062_agreement : RowAgreement 1061 1062 := RowAgreement.single (first := 1061) modelrow_agrees_1062
private def atomic_1063 : CountSegment 1 10 13120 1062 1063 588 589 := CountSegment.single (first := 1062) (initial := 588) row_count_1063 (by decide)
private theorem atomic_1063_agreement : RowAgreement 1062 1063 := RowAgreement.single (first := 1062) modelrow_agrees_1063
private def atomic_1064 : CountSegment 1 10 13120 1063 1064 589 589 := CountSegment.single (first := 1063) (initial := 589) row_count_1064 (by decide)
private theorem atomic_1064_agreement : RowAgreement 1063 1064 := RowAgreement.single (first := 1063) modelrow_agrees_1064
private def atomic_1065 : CountSegment 1 10 13120 1064 1065 589 590 := CountSegment.single (first := 1064) (initial := 589) row_count_1065 (by decide)
private theorem atomic_1065_agreement : RowAgreement 1064 1065 := RowAgreement.single (first := 1064) modelrow_agrees_1065
private def atomic_1066 : CountSegment 1 10 13120 1065 1066 590 590 := CountSegment.single (first := 1065) (initial := 590) row_count_1066 (by decide)
private theorem atomic_1066_agreement : RowAgreement 1065 1066 := RowAgreement.single (first := 1065) modelrow_agrees_1066
private def atomic_1067 : CountSegment 1 10 13120 1066 1067 590 590 := CountSegment.single (first := 1066) (initial := 590) row_count_1067 (by decide)
private theorem atomic_1067_agreement : RowAgreement 1066 1067 := RowAgreement.single (first := 1066) modelrow_agrees_1067
private def atomic_1068 : CountSegment 1 10 13120 1067 1068 590 592 := CountSegment.single (first := 1067) (initial := 590) row_count_1068 (by decide)
private theorem atomic_1068_agreement : RowAgreement 1067 1068 := RowAgreement.single (first := 1067) modelrow_agrees_1068
private def atomic_1069 : CountSegment 1 10 13120 1068 1069 592 593 := CountSegment.single (first := 1068) (initial := 592) row_count_1069 (by decide)
private theorem atomic_1069_agreement : RowAgreement 1068 1069 := RowAgreement.single (first := 1068) modelrow_agrees_1069
private def atomic_1070 : CountSegment 1 10 13120 1069 1070 593 593 := CountSegment.single (first := 1069) (initial := 593) row_count_1070 (by decide)
private theorem atomic_1070_agreement : RowAgreement 1069 1070 := RowAgreement.single (first := 1069) modelrow_agrees_1070
private def atomic_1071 : CountSegment 1 10 13120 1070 1071 593 594 := CountSegment.single (first := 1070) (initial := 593) row_count_1071 (by decide)
private theorem atomic_1071_agreement : RowAgreement 1070 1071 := RowAgreement.single (first := 1070) modelrow_agrees_1071
private def atomic_1072 : CountSegment 1 10 13120 1071 1072 594 594 := CountSegment.single (first := 1071) (initial := 594) row_count_1072 (by decide)
private theorem atomic_1072_agreement : RowAgreement 1071 1072 := RowAgreement.single (first := 1071) modelrow_agrees_1072
private def atomic_1073 : CountSegment 1 10 13120 1072 1073 594 595 := CountSegment.single (first := 1072) (initial := 594) row_count_1073 (by decide)
private theorem atomic_1073_agreement : RowAgreement 1072 1073 := RowAgreement.single (first := 1072) modelrow_agrees_1073
private def atomic_1074 : CountSegment 1 10 13120 1073 1074 595 595 := CountSegment.single (first := 1073) (initial := 595) row_count_1074 (by decide)
private theorem atomic_1074_agreement : RowAgreement 1073 1074 := RowAgreement.single (first := 1073) modelrow_agrees_1074
private def atomic_1075 : CountSegment 1 10 13120 1074 1075 595 595 := CountSegment.single (first := 1074) (initial := 595) row_count_1075 (by decide)
private theorem atomic_1075_agreement : RowAgreement 1074 1075 := RowAgreement.single (first := 1074) modelrow_agrees_1075
private def atomic_1076 : CountSegment 1 10 13120 1075 1076 595 595 := CountSegment.single (first := 1075) (initial := 595) row_count_1076 (by decide)
private theorem atomic_1076_agreement : RowAgreement 1075 1076 := RowAgreement.single (first := 1075) modelrow_agrees_1076
private def atomic_1077 : CountSegment 1 10 13120 1076 1077 595 596 := CountSegment.single (first := 1076) (initial := 595) row_count_1077 (by decide)
private theorem atomic_1077_agreement : RowAgreement 1076 1077 := RowAgreement.single (first := 1076) modelrow_agrees_1077
private def atomic_1078 : CountSegment 1 10 13120 1077 1078 596 596 := CountSegment.single (first := 1077) (initial := 596) row_count_1078 (by decide)
private theorem atomic_1078_agreement : RowAgreement 1077 1078 := RowAgreement.single (first := 1077) modelrow_agrees_1078
private def atomic_1079 : CountSegment 1 10 13120 1078 1079 596 597 := CountSegment.single (first := 1078) (initial := 596) row_count_1079 (by decide)
private theorem atomic_1079_agreement : RowAgreement 1078 1079 := RowAgreement.single (first := 1078) modelrow_agrees_1079
private def atomic_1080 : CountSegment 1 10 13120 1079 1080 597 598 := CountSegment.single (first := 1079) (initial := 597) row_count_1080 (by decide)
private theorem atomic_1080_agreement : RowAgreement 1079 1080 := RowAgreement.single (first := 1079) modelrow_agrees_1080
private def atomic_1081 : CountSegment 1 10 13120 1080 1081 598 599 := CountSegment.single (first := 1080) (initial := 598) row_count_1081 (by decide)
private theorem atomic_1081_agreement : RowAgreement 1080 1081 := RowAgreement.single (first := 1080) modelrow_agrees_1081
private def atomic_1082 : CountSegment 1 10 13120 1081 1082 599 600 := CountSegment.single (first := 1081) (initial := 599) row_count_1082 (by decide)
private theorem atomic_1082_agreement : RowAgreement 1081 1082 := RowAgreement.single (first := 1081) modelrow_agrees_1082
private def atomic_1083 : CountSegment 1 10 13120 1082 1083 600 600 := CountSegment.single (first := 1082) (initial := 600) row_count_1083 (by decide)
private theorem atomic_1083_agreement : RowAgreement 1082 1083 := RowAgreement.single (first := 1082) modelrow_agrees_1083
private def atomic_1084 : CountSegment 1 10 13120 1083 1084 600 601 := CountSegment.single (first := 1083) (initial := 600) row_count_1084 (by decide)
private theorem atomic_1084_agreement : RowAgreement 1083 1084 := RowAgreement.single (first := 1083) modelrow_agrees_1084
private def atomic_1085 : CountSegment 1 10 13120 1084 1085 601 601 := CountSegment.single (first := 1084) (initial := 601) row_count_1085 (by decide)
private theorem atomic_1085_agreement : RowAgreement 1084 1085 := RowAgreement.single (first := 1084) modelrow_agrees_1085
private def atomic_1086 : CountSegment 1 10 13120 1085 1086 601 602 := CountSegment.single (first := 1085) (initial := 601) row_count_1086 (by decide)
private theorem atomic_1086_agreement : RowAgreement 1085 1086 := RowAgreement.single (first := 1085) modelrow_agrees_1086
private def atomic_1087 : CountSegment 1 10 13120 1086 1087 602 603 := CountSegment.single (first := 1086) (initial := 602) row_count_1087 (by decide)
private theorem atomic_1087_agreement : RowAgreement 1086 1087 := RowAgreement.single (first := 1086) modelrow_agrees_1087
private def atomic_1088 : CountSegment 1 10 13120 1087 1088 603 603 := CountSegment.single (first := 1087) (initial := 603) row_count_1088 (by decide)
private theorem atomic_1088_agreement : RowAgreement 1087 1088 := RowAgreement.single (first := 1087) modelrow_agrees_1088
private def atomic_1089 : CountSegment 1 10 13120 1088 1089 603 603 := CountSegment.single (first := 1088) (initial := 603) row_count_1089 (by decide)
private theorem atomic_1089_agreement : RowAgreement 1088 1089 := RowAgreement.single (first := 1088) modelrow_agrees_1089
private def atomic_1090 : CountSegment 1 10 13120 1089 1090 603 603 := CountSegment.single (first := 1089) (initial := 603) row_count_1090 (by decide)
private theorem atomic_1090_agreement : RowAgreement 1089 1090 := RowAgreement.single (first := 1089) modelrow_agrees_1090
private def atomic_1091 : CountSegment 1 10 13120 1090 1091 603 603 := CountSegment.single (first := 1090) (initial := 603) row_count_1091 (by decide)
private theorem atomic_1091_agreement : RowAgreement 1090 1091 := RowAgreement.single (first := 1090) modelrow_agrees_1091
private def atomic_1092 : CountSegment 1 10 13120 1091 1092 603 604 := CountSegment.single (first := 1091) (initial := 603) row_count_1092 (by decide)
private theorem atomic_1092_agreement : RowAgreement 1091 1092 := RowAgreement.single (first := 1091) modelrow_agrees_1092
private def atomic_1093 : CountSegment 1 10 13120 1092 1093 604 604 := CountSegment.single (first := 1092) (initial := 604) row_count_1093 (by decide)
private theorem atomic_1093_agreement : RowAgreement 1092 1093 := RowAgreement.single (first := 1092) modelrow_agrees_1093
private def atomic_1094 : CountSegment 1 10 13120 1093 1094 604 604 := CountSegment.single (first := 1093) (initial := 604) row_count_1094 (by decide)
private theorem atomic_1094_agreement : RowAgreement 1093 1094 := RowAgreement.single (first := 1093) modelrow_agrees_1094
private def atomic_1095 : CountSegment 1 10 13120 1094 1095 604 604 := CountSegment.single (first := 1094) (initial := 604) row_count_1095 (by decide)
private theorem atomic_1095_agreement : RowAgreement 1094 1095 := RowAgreement.single (first := 1094) modelrow_agrees_1095
private def atomic_1096 : CountSegment 1 10 13120 1095 1096 604 604 := CountSegment.single (first := 1095) (initial := 604) row_count_1096 (by decide)
private theorem atomic_1096_agreement : RowAgreement 1095 1096 := RowAgreement.single (first := 1095) modelrow_agrees_1096
private def atomic_1097 : CountSegment 1 10 13120 1096 1097 604 604 := CountSegment.single (first := 1096) (initial := 604) row_count_1097 (by decide)
private theorem atomic_1097_agreement : RowAgreement 1096 1097 := RowAgreement.single (first := 1096) modelrow_agrees_1097
private def atomic_1098 : CountSegment 1 10 13120 1097 1098 604 604 := CountSegment.single (first := 1097) (initial := 604) row_count_1098 (by decide)
private theorem atomic_1098_agreement : RowAgreement 1097 1098 := RowAgreement.single (first := 1097) modelrow_agrees_1098
private def atomic_1099 : CountSegment 1 10 13120 1098 1099 604 604 := CountSegment.single (first := 1098) (initial := 604) row_count_1099 (by decide)
private theorem atomic_1099_agreement : RowAgreement 1098 1099 := RowAgreement.single (first := 1098) modelrow_agrees_1099
private def atomic_1100 : CountSegment 1 10 13120 1099 1100 604 604 := CountSegment.single (first := 1099) (initial := 604) row_count_1100 (by decide)
private theorem atomic_1100_agreement : RowAgreement 1099 1100 := RowAgreement.single (first := 1099) modelrow_agrees_1100
private def atomic_1101 : CountSegment 1 10 13120 1100 1101 604 605 := CountSegment.single (first := 1100) (initial := 604) row_count_1101 (by decide)
private theorem atomic_1101_agreement : RowAgreement 1100 1101 := RowAgreement.single (first := 1100) modelrow_agrees_1101
private def atomic_1102 : CountSegment 1 10 13120 1101 1102 605 606 := CountSegment.single (first := 1101) (initial := 605) row_count_1102 (by decide)
private theorem atomic_1102_agreement : RowAgreement 1101 1102 := RowAgreement.single (first := 1101) modelrow_agrees_1102
private def atomic_1103 : CountSegment 1 10 13120 1102 1103 606 607 := CountSegment.single (first := 1102) (initial := 606) row_count_1103 (by decide)
private theorem atomic_1103_agreement : RowAgreement 1102 1103 := RowAgreement.single (first := 1102) modelrow_agrees_1103
private def atomic_1104 : CountSegment 1 10 13120 1103 1104 607 607 := CountSegment.single (first := 1103) (initial := 607) row_count_1104 (by decide)
private theorem atomic_1104_agreement : RowAgreement 1103 1104 := RowAgreement.single (first := 1103) modelrow_agrees_1104
private def atomic_1105 : CountSegment 1 10 13120 1104 1105 607 608 := CountSegment.single (first := 1104) (initial := 607) row_count_1105 (by decide)
private theorem atomic_1105_agreement : RowAgreement 1104 1105 := RowAgreement.single (first := 1104) modelrow_agrees_1105
private def atomic_1106 : CountSegment 1 10 13120 1105 1106 608 608 := CountSegment.single (first := 1105) (initial := 608) row_count_1106 (by decide)
private theorem atomic_1106_agreement : RowAgreement 1105 1106 := RowAgreement.single (first := 1105) modelrow_agrees_1106
private def atomic_1107 : CountSegment 1 10 13120 1106 1107 608 608 := CountSegment.single (first := 1106) (initial := 608) row_count_1107 (by decide)
private theorem atomic_1107_agreement : RowAgreement 1106 1107 := RowAgreement.single (first := 1106) modelrow_agrees_1107
private def atomic_1108 : CountSegment 1 10 13120 1107 1108 608 608 := CountSegment.single (first := 1107) (initial := 608) row_count_1108 (by decide)
private theorem atomic_1108_agreement : RowAgreement 1107 1108 := RowAgreement.single (first := 1107) modelrow_agrees_1108
private def atomic_1109 : CountSegment 1 10 13120 1108 1109 608 608 := CountSegment.single (first := 1108) (initial := 608) row_count_1109 (by decide)
private theorem atomic_1109_agreement : RowAgreement 1108 1109 := RowAgreement.single (first := 1108) modelrow_agrees_1109
private def atomic_1110 : CountSegment 1 10 13120 1109 1110 608 608 := CountSegment.single (first := 1109) (initial := 608) row_count_1110 (by decide)
private theorem atomic_1110_agreement : RowAgreement 1109 1110 := RowAgreement.single (first := 1109) modelrow_agrees_1110
private def atomic_1111 : CountSegment 1 10 13120 1110 1111 608 608 := CountSegment.single (first := 1110) (initial := 608) row_count_1111 (by decide)
private theorem atomic_1111_agreement : RowAgreement 1110 1111 := RowAgreement.single (first := 1110) modelrow_agrees_1111
private def atomic_1112 : CountSegment 1 10 13120 1111 1112 608 609 := CountSegment.single (first := 1111) (initial := 608) row_count_1112 (by decide)
private theorem atomic_1112_agreement : RowAgreement 1111 1112 := RowAgreement.single (first := 1111) modelrow_agrees_1112
private def atomic_1113 : CountSegment 1 10 13120 1112 1113 609 609 := CountSegment.single (first := 1112) (initial := 609) row_count_1113 (by decide)
private theorem atomic_1113_agreement : RowAgreement 1112 1113 := RowAgreement.single (first := 1112) modelrow_agrees_1113
private def atomic_1114 : CountSegment 1 10 13120 1113 1114 609 609 := CountSegment.single (first := 1113) (initial := 609) row_count_1114 (by decide)
private theorem atomic_1114_agreement : RowAgreement 1113 1114 := RowAgreement.single (first := 1113) modelrow_agrees_1114
private def atomic_1115 : CountSegment 1 10 13120 1114 1115 609 609 := CountSegment.single (first := 1114) (initial := 609) row_count_1115 (by decide)
private theorem atomic_1115_agreement : RowAgreement 1114 1115 := RowAgreement.single (first := 1114) modelrow_agrees_1115
private def atomic_1116 : CountSegment 1 10 13120 1115 1116 609 609 := CountSegment.single (first := 1115) (initial := 609) row_count_1116 (by decide)
private theorem atomic_1116_agreement : RowAgreement 1115 1116 := RowAgreement.single (first := 1115) modelrow_agrees_1116
private def atomic_1117 : CountSegment 1 10 13120 1116 1117 609 609 := CountSegment.single (first := 1116) (initial := 609) row_count_1117 (by decide)
private theorem atomic_1117_agreement : RowAgreement 1116 1117 := RowAgreement.single (first := 1116) modelrow_agrees_1117
private def atomic_1118 : CountSegment 1 10 13120 1117 1118 609 609 := CountSegment.single (first := 1117) (initial := 609) row_count_1118 (by decide)
private theorem atomic_1118_agreement : RowAgreement 1117 1118 := RowAgreement.single (first := 1117) modelrow_agrees_1118
private def atomic_1119 : CountSegment 1 10 13120 1118 1119 609 610 := CountSegment.single (first := 1118) (initial := 609) row_count_1119 (by decide)
private theorem atomic_1119_agreement : RowAgreement 1118 1119 := RowAgreement.single (first := 1118) modelrow_agrees_1119
private def atomic_1120 : CountSegment 1 10 13120 1119 1120 610 610 := CountSegment.single (first := 1119) (initial := 610) row_count_1120 (by decide)
private theorem atomic_1120_agreement : RowAgreement 1119 1120 := RowAgreement.single (first := 1119) modelrow_agrees_1120
private def atomic_1121 : CountSegment 1 10 13120 1120 1121 610 611 := CountSegment.single (first := 1120) (initial := 610) row_count_1121 (by decide)
private theorem atomic_1121_agreement : RowAgreement 1120 1121 := RowAgreement.single (first := 1120) modelrow_agrees_1121
private def atomic_1122 : CountSegment 1 10 13120 1121 1122 611 611 := CountSegment.single (first := 1121) (initial := 611) row_count_1122 (by decide)
private theorem atomic_1122_agreement : RowAgreement 1121 1122 := RowAgreement.single (first := 1121) modelrow_agrees_1122
private def atomic_1123 : CountSegment 1 10 13120 1122 1123 611 611 := CountSegment.single (first := 1122) (initial := 611) row_count_1123 (by decide)
private theorem atomic_1123_agreement : RowAgreement 1122 1123 := RowAgreement.single (first := 1122) modelrow_agrees_1123
private def atomic_1124 : CountSegment 1 10 13120 1123 1124 611 612 := CountSegment.single (first := 1123) (initial := 611) row_count_1124 (by decide)
private theorem atomic_1124_agreement : RowAgreement 1123 1124 := RowAgreement.single (first := 1123) modelrow_agrees_1124
private def atomic_1125 : CountSegment 1 10 13120 1124 1125 612 612 := CountSegment.single (first := 1124) (initial := 612) row_count_1125 (by decide)
private theorem atomic_1125_agreement : RowAgreement 1124 1125 := RowAgreement.single (first := 1124) modelrow_agrees_1125
private def atomic_1126 : CountSegment 1 10 13120 1125 1126 612 612 := CountSegment.single (first := 1125) (initial := 612) row_count_1126 (by decide)
private theorem atomic_1126_agreement : RowAgreement 1125 1126 := RowAgreement.single (first := 1125) modelrow_agrees_1126
private def atomic_1127 : CountSegment 1 10 13120 1126 1127 612 612 := CountSegment.single (first := 1126) (initial := 612) row_count_1127 (by decide)
private theorem atomic_1127_agreement : RowAgreement 1126 1127 := RowAgreement.single (first := 1126) modelrow_agrees_1127
private def atomic_1128 : CountSegment 1 10 13120 1127 1128 612 613 := CountSegment.single (first := 1127) (initial := 612) row_count_1128 (by decide)
private theorem atomic_1128_agreement : RowAgreement 1127 1128 := RowAgreement.single (first := 1127) modelrow_agrees_1128
private def atomic_1129 : CountSegment 1 10 13120 1128 1129 613 613 := CountSegment.single (first := 1128) (initial := 613) row_count_1129 (by decide)
private theorem atomic_1129_agreement : RowAgreement 1128 1129 := RowAgreement.single (first := 1128) modelrow_agrees_1129
private def atomic_1130 : CountSegment 1 10 13120 1129 1130 613 615 := CountSegment.single (first := 1129) (initial := 613) row_count_1130 (by decide)
private theorem atomic_1130_agreement : RowAgreement 1129 1130 := RowAgreement.single (first := 1129) modelrow_agrees_1130
private def atomic_1131 : CountSegment 1 10 13120 1130 1131 615 616 := CountSegment.single (first := 1130) (initial := 615) row_count_1131 (by decide)
private theorem atomic_1131_agreement : RowAgreement 1130 1131 := RowAgreement.single (first := 1130) modelrow_agrees_1131
private def atomic_1132 : CountSegment 1 10 13120 1131 1132 616 617 := CountSegment.single (first := 1131) (initial := 616) row_count_1132 (by decide)
private theorem atomic_1132_agreement : RowAgreement 1131 1132 := RowAgreement.single (first := 1131) modelrow_agrees_1132
private def atomic_1133 : CountSegment 1 10 13120 1132 1133 617 618 := CountSegment.single (first := 1132) (initial := 617) row_count_1133 (by decide)
private theorem atomic_1133_agreement : RowAgreement 1132 1133 := RowAgreement.single (first := 1132) modelrow_agrees_1133
private def atomic_1134 : CountSegment 1 10 13120 1133 1134 618 618 := CountSegment.single (first := 1133) (initial := 618) row_count_1134 (by decide)
private theorem atomic_1134_agreement : RowAgreement 1133 1134 := RowAgreement.single (first := 1133) modelrow_agrees_1134
private def atomic_1135 : CountSegment 1 10 13120 1134 1135 618 618 := CountSegment.single (first := 1134) (initial := 618) row_count_1135 (by decide)
private theorem atomic_1135_agreement : RowAgreement 1134 1135 := RowAgreement.single (first := 1134) modelrow_agrees_1135
private def atomic_1136 : CountSegment 1 10 13120 1135 1136 618 618 := CountSegment.single (first := 1135) (initial := 618) row_count_1136 (by decide)
private theorem atomic_1136_agreement : RowAgreement 1135 1136 := RowAgreement.single (first := 1135) modelrow_agrees_1136
private def atomic_1137 : CountSegment 1 10 13120 1136 1137 618 619 := CountSegment.single (first := 1136) (initial := 618) row_count_1137 (by decide)
private theorem atomic_1137_agreement : RowAgreement 1136 1137 := RowAgreement.single (first := 1136) modelrow_agrees_1137
private def atomic_1138 : CountSegment 1 10 13120 1137 1138 619 620 := CountSegment.single (first := 1137) (initial := 619) row_count_1138 (by decide)
private theorem atomic_1138_agreement : RowAgreement 1137 1138 := RowAgreement.single (first := 1137) modelrow_agrees_1138
private def atomic_1139 : CountSegment 1 10 13120 1138 1139 620 620 := CountSegment.single (first := 1138) (initial := 620) row_count_1139 (by decide)
private theorem atomic_1139_agreement : RowAgreement 1138 1139 := RowAgreement.single (first := 1138) modelrow_agrees_1139
private def atomic_1140 : CountSegment 1 10 13120 1139 1140 620 621 := CountSegment.single (first := 1139) (initial := 620) row_count_1140 (by decide)
private theorem atomic_1140_agreement : RowAgreement 1139 1140 := RowAgreement.single (first := 1139) modelrow_agrees_1140
private def atomic_1141 : CountSegment 1 10 13120 1140 1141 621 621 := CountSegment.single (first := 1140) (initial := 621) row_count_1141 (by decide)
private theorem atomic_1141_agreement : RowAgreement 1140 1141 := RowAgreement.single (first := 1140) modelrow_agrees_1141
private def atomic_1142 : CountSegment 1 10 13120 1141 1142 621 621 := CountSegment.single (first := 1141) (initial := 621) row_count_1142 (by decide)
private theorem atomic_1142_agreement : RowAgreement 1141 1142 := RowAgreement.single (first := 1141) modelrow_agrees_1142
private def atomic_1143 : CountSegment 1 10 13120 1142 1143 621 622 := CountSegment.single (first := 1142) (initial := 621) row_count_1143 (by decide)
private theorem atomic_1143_agreement : RowAgreement 1142 1143 := RowAgreement.single (first := 1142) modelrow_agrees_1143
private def atomic_1144 : CountSegment 1 10 13120 1143 1144 622 622 := CountSegment.single (first := 1143) (initial := 622) row_count_1144 (by decide)
private theorem atomic_1144_agreement : RowAgreement 1143 1144 := RowAgreement.single (first := 1143) modelrow_agrees_1144
private def atomic_1145 : CountSegment 1 10 13120 1144 1145 622 622 := CountSegment.single (first := 1144) (initial := 622) row_count_1145 (by decide)
private theorem atomic_1145_agreement : RowAgreement 1144 1145 := RowAgreement.single (first := 1144) modelrow_agrees_1145
private def atomic_1146 : CountSegment 1 10 13120 1145 1146 622 622 := CountSegment.single (first := 1145) (initial := 622) row_count_1146 (by decide)
private theorem atomic_1146_agreement : RowAgreement 1145 1146 := RowAgreement.single (first := 1145) modelrow_agrees_1146
private def atomic_1147 : CountSegment 1 10 13120 1146 1147 622 623 := CountSegment.single (first := 1146) (initial := 622) row_count_1147 (by decide)
private theorem atomic_1147_agreement : RowAgreement 1146 1147 := RowAgreement.single (first := 1146) modelrow_agrees_1147
private def atomic_1148 : CountSegment 1 10 13120 1147 1148 623 623 := CountSegment.single (first := 1147) (initial := 623) row_count_1148 (by decide)
private theorem atomic_1148_agreement : RowAgreement 1147 1148 := RowAgreement.single (first := 1147) modelrow_agrees_1148
private def atomic_1149 : CountSegment 1 10 13120 1148 1149 623 623 := CountSegment.single (first := 1148) (initial := 623) row_count_1149 (by decide)
private theorem atomic_1149_agreement : RowAgreement 1148 1149 := RowAgreement.single (first := 1148) modelrow_agrees_1149
private def atomic_1150 : CountSegment 1 10 13120 1149 1150 623 623 := CountSegment.single (first := 1149) (initial := 623) row_count_1150 (by decide)
private theorem atomic_1150_agreement : RowAgreement 1149 1150 := RowAgreement.single (first := 1149) modelrow_agrees_1150
private def atomic_1151 : CountSegment 1 10 13120 1150 1151 623 624 := CountSegment.single (first := 1150) (initial := 623) row_count_1151 (by decide)
private theorem atomic_1151_agreement : RowAgreement 1150 1151 := RowAgreement.single (first := 1150) modelrow_agrees_1151
private def atomic_1152 : CountSegment 1 10 13120 1151 1152 624 626 := CountSegment.single (first := 1151) (initial := 624) row_count_1152 (by decide)
private theorem atomic_1152_agreement : RowAgreement 1151 1152 := RowAgreement.single (first := 1151) modelrow_agrees_1152
private def within_008_000 : CountSegment 1 10 13120 1024 1026 563 565 := CountSegment.join atomic_1025 atomic_1026
private theorem within_008_000_agreement : RowAgreement 1024 1026 := RowAgreement.join atomic_1025_agreement atomic_1026_agreement
private def within_008_001 : CountSegment 1 10 13120 1026 1028 565 568 := CountSegment.join atomic_1027 atomic_1028
private theorem within_008_001_agreement : RowAgreement 1026 1028 := RowAgreement.join atomic_1027_agreement atomic_1028_agreement
private def within_008_002 : CountSegment 1 10 13120 1024 1028 563 568 := CountSegment.join within_008_000 within_008_001
private theorem within_008_002_agreement : RowAgreement 1024 1028 := RowAgreement.join within_008_000_agreement within_008_001_agreement
private def within_008_003 : CountSegment 1 10 13120 1028 1030 568 569 := CountSegment.join atomic_1029 atomic_1030
private theorem within_008_003_agreement : RowAgreement 1028 1030 := RowAgreement.join atomic_1029_agreement atomic_1030_agreement
private def within_008_004 : CountSegment 1 10 13120 1030 1032 569 571 := CountSegment.join atomic_1031 atomic_1032
private theorem within_008_004_agreement : RowAgreement 1030 1032 := RowAgreement.join atomic_1031_agreement atomic_1032_agreement
private def within_008_005 : CountSegment 1 10 13120 1028 1032 568 571 := CountSegment.join within_008_003 within_008_004
private theorem within_008_005_agreement : RowAgreement 1028 1032 := RowAgreement.join within_008_003_agreement within_008_004_agreement
private def within_008_006 : CountSegment 1 10 13120 1024 1032 563 571 := CountSegment.join within_008_002 within_008_005
private theorem within_008_006_agreement : RowAgreement 1024 1032 := RowAgreement.join within_008_002_agreement within_008_005_agreement
private def within_008_007 : CountSegment 1 10 13120 1032 1034 571 571 := CountSegment.join atomic_1033 atomic_1034
private theorem within_008_007_agreement : RowAgreement 1032 1034 := RowAgreement.join atomic_1033_agreement atomic_1034_agreement
private def within_008_008 : CountSegment 1 10 13120 1034 1036 571 571 := CountSegment.join atomic_1035 atomic_1036
private theorem within_008_008_agreement : RowAgreement 1034 1036 := RowAgreement.join atomic_1035_agreement atomic_1036_agreement
private def within_008_009 : CountSegment 1 10 13120 1032 1036 571 571 := CountSegment.join within_008_007 within_008_008
private theorem within_008_009_agreement : RowAgreement 1032 1036 := RowAgreement.join within_008_007_agreement within_008_008_agreement
private def within_008_010 : CountSegment 1 10 13120 1036 1038 571 574 := CountSegment.join atomic_1037 atomic_1038
private theorem within_008_010_agreement : RowAgreement 1036 1038 := RowAgreement.join atomic_1037_agreement atomic_1038_agreement
private def within_008_011 : CountSegment 1 10 13120 1038 1040 574 576 := CountSegment.join atomic_1039 atomic_1040
private theorem within_008_011_agreement : RowAgreement 1038 1040 := RowAgreement.join atomic_1039_agreement atomic_1040_agreement
private def within_008_012 : CountSegment 1 10 13120 1036 1040 571 576 := CountSegment.join within_008_010 within_008_011
private theorem within_008_012_agreement : RowAgreement 1036 1040 := RowAgreement.join within_008_010_agreement within_008_011_agreement
private def within_008_013 : CountSegment 1 10 13120 1032 1040 571 576 := CountSegment.join within_008_009 within_008_012
private theorem within_008_013_agreement : RowAgreement 1032 1040 := RowAgreement.join within_008_009_agreement within_008_012_agreement
private def within_008_014 : CountSegment 1 10 13120 1024 1040 563 576 := CountSegment.join within_008_006 within_008_013
private theorem within_008_014_agreement : RowAgreement 1024 1040 := RowAgreement.join within_008_006_agreement within_008_013_agreement
private def within_008_015 : CountSegment 1 10 13120 1040 1042 576 577 := CountSegment.join atomic_1041 atomic_1042
private theorem within_008_015_agreement : RowAgreement 1040 1042 := RowAgreement.join atomic_1041_agreement atomic_1042_agreement
private def within_008_016 : CountSegment 1 10 13120 1042 1044 577 579 := CountSegment.join atomic_1043 atomic_1044
private theorem within_008_016_agreement : RowAgreement 1042 1044 := RowAgreement.join atomic_1043_agreement atomic_1044_agreement
private def within_008_017 : CountSegment 1 10 13120 1040 1044 576 579 := CountSegment.join within_008_015 within_008_016
private theorem within_008_017_agreement : RowAgreement 1040 1044 := RowAgreement.join within_008_015_agreement within_008_016_agreement
private def within_008_018 : CountSegment 1 10 13120 1044 1046 579 580 := CountSegment.join atomic_1045 atomic_1046
private theorem within_008_018_agreement : RowAgreement 1044 1046 := RowAgreement.join atomic_1045_agreement atomic_1046_agreement
private def within_008_019 : CountSegment 1 10 13120 1046 1048 580 582 := CountSegment.join atomic_1047 atomic_1048
private theorem within_008_019_agreement : RowAgreement 1046 1048 := RowAgreement.join atomic_1047_agreement atomic_1048_agreement
private def within_008_020 : CountSegment 1 10 13120 1044 1048 579 582 := CountSegment.join within_008_018 within_008_019
private theorem within_008_020_agreement : RowAgreement 1044 1048 := RowAgreement.join within_008_018_agreement within_008_019_agreement
private def within_008_021 : CountSegment 1 10 13120 1040 1048 576 582 := CountSegment.join within_008_017 within_008_020
private theorem within_008_021_agreement : RowAgreement 1040 1048 := RowAgreement.join within_008_017_agreement within_008_020_agreement
private def within_008_022 : CountSegment 1 10 13120 1048 1050 582 582 := CountSegment.join atomic_1049 atomic_1050
private theorem within_008_022_agreement : RowAgreement 1048 1050 := RowAgreement.join atomic_1049_agreement atomic_1050_agreement
private def within_008_023 : CountSegment 1 10 13120 1050 1052 582 582 := CountSegment.join atomic_1051 atomic_1052
private theorem within_008_023_agreement : RowAgreement 1050 1052 := RowAgreement.join atomic_1051_agreement atomic_1052_agreement
private def within_008_024 : CountSegment 1 10 13120 1048 1052 582 582 := CountSegment.join within_008_022 within_008_023
private theorem within_008_024_agreement : RowAgreement 1048 1052 := RowAgreement.join within_008_022_agreement within_008_023_agreement
private def within_008_025 : CountSegment 1 10 13120 1052 1054 582 584 := CountSegment.join atomic_1053 atomic_1054
private theorem within_008_025_agreement : RowAgreement 1052 1054 := RowAgreement.join atomic_1053_agreement atomic_1054_agreement
private def within_008_026 : CountSegment 1 10 13120 1054 1056 584 585 := CountSegment.join atomic_1055 atomic_1056
private theorem within_008_026_agreement : RowAgreement 1054 1056 := RowAgreement.join atomic_1055_agreement atomic_1056_agreement
private def within_008_027 : CountSegment 1 10 13120 1052 1056 582 585 := CountSegment.join within_008_025 within_008_026
private theorem within_008_027_agreement : RowAgreement 1052 1056 := RowAgreement.join within_008_025_agreement within_008_026_agreement
private def within_008_028 : CountSegment 1 10 13120 1048 1056 582 585 := CountSegment.join within_008_024 within_008_027
private theorem within_008_028_agreement : RowAgreement 1048 1056 := RowAgreement.join within_008_024_agreement within_008_027_agreement
private def within_008_029 : CountSegment 1 10 13120 1040 1056 576 585 := CountSegment.join within_008_021 within_008_028
private theorem within_008_029_agreement : RowAgreement 1040 1056 := RowAgreement.join within_008_021_agreement within_008_028_agreement
private def within_008_030 : CountSegment 1 10 13120 1024 1056 563 585 := CountSegment.join within_008_014 within_008_029
private theorem within_008_030_agreement : RowAgreement 1024 1056 := RowAgreement.join within_008_014_agreement within_008_029_agreement
private def within_008_031 : CountSegment 1 10 13120 1056 1058 585 586 := CountSegment.join atomic_1057 atomic_1058
private theorem within_008_031_agreement : RowAgreement 1056 1058 := RowAgreement.join atomic_1057_agreement atomic_1058_agreement
private def within_008_032 : CountSegment 1 10 13120 1058 1060 586 588 := CountSegment.join atomic_1059 atomic_1060
private theorem within_008_032_agreement : RowAgreement 1058 1060 := RowAgreement.join atomic_1059_agreement atomic_1060_agreement
private def within_008_033 : CountSegment 1 10 13120 1056 1060 585 588 := CountSegment.join within_008_031 within_008_032
private theorem within_008_033_agreement : RowAgreement 1056 1060 := RowAgreement.join within_008_031_agreement within_008_032_agreement
private def within_008_034 : CountSegment 1 10 13120 1060 1062 588 588 := CountSegment.join atomic_1061 atomic_1062
private theorem within_008_034_agreement : RowAgreement 1060 1062 := RowAgreement.join atomic_1061_agreement atomic_1062_agreement
private def within_008_035 : CountSegment 1 10 13120 1062 1064 588 589 := CountSegment.join atomic_1063 atomic_1064
private theorem within_008_035_agreement : RowAgreement 1062 1064 := RowAgreement.join atomic_1063_agreement atomic_1064_agreement
private def within_008_036 : CountSegment 1 10 13120 1060 1064 588 589 := CountSegment.join within_008_034 within_008_035
private theorem within_008_036_agreement : RowAgreement 1060 1064 := RowAgreement.join within_008_034_agreement within_008_035_agreement
private def within_008_037 : CountSegment 1 10 13120 1056 1064 585 589 := CountSegment.join within_008_033 within_008_036
private theorem within_008_037_agreement : RowAgreement 1056 1064 := RowAgreement.join within_008_033_agreement within_008_036_agreement
private def within_008_038 : CountSegment 1 10 13120 1064 1066 589 590 := CountSegment.join atomic_1065 atomic_1066
private theorem within_008_038_agreement : RowAgreement 1064 1066 := RowAgreement.join atomic_1065_agreement atomic_1066_agreement
private def within_008_039 : CountSegment 1 10 13120 1066 1068 590 592 := CountSegment.join atomic_1067 atomic_1068
private theorem within_008_039_agreement : RowAgreement 1066 1068 := RowAgreement.join atomic_1067_agreement atomic_1068_agreement
private def within_008_040 : CountSegment 1 10 13120 1064 1068 589 592 := CountSegment.join within_008_038 within_008_039
private theorem within_008_040_agreement : RowAgreement 1064 1068 := RowAgreement.join within_008_038_agreement within_008_039_agreement
private def within_008_041 : CountSegment 1 10 13120 1068 1070 592 593 := CountSegment.join atomic_1069 atomic_1070
private theorem within_008_041_agreement : RowAgreement 1068 1070 := RowAgreement.join atomic_1069_agreement atomic_1070_agreement
private def within_008_042 : CountSegment 1 10 13120 1070 1072 593 594 := CountSegment.join atomic_1071 atomic_1072
private theorem within_008_042_agreement : RowAgreement 1070 1072 := RowAgreement.join atomic_1071_agreement atomic_1072_agreement
private def within_008_043 : CountSegment 1 10 13120 1068 1072 592 594 := CountSegment.join within_008_041 within_008_042
private theorem within_008_043_agreement : RowAgreement 1068 1072 := RowAgreement.join within_008_041_agreement within_008_042_agreement
private def within_008_044 : CountSegment 1 10 13120 1064 1072 589 594 := CountSegment.join within_008_040 within_008_043
private theorem within_008_044_agreement : RowAgreement 1064 1072 := RowAgreement.join within_008_040_agreement within_008_043_agreement
private def within_008_045 : CountSegment 1 10 13120 1056 1072 585 594 := CountSegment.join within_008_037 within_008_044
private theorem within_008_045_agreement : RowAgreement 1056 1072 := RowAgreement.join within_008_037_agreement within_008_044_agreement
private def within_008_046 : CountSegment 1 10 13120 1072 1074 594 595 := CountSegment.join atomic_1073 atomic_1074
private theorem within_008_046_agreement : RowAgreement 1072 1074 := RowAgreement.join atomic_1073_agreement atomic_1074_agreement
private def within_008_047 : CountSegment 1 10 13120 1074 1076 595 595 := CountSegment.join atomic_1075 atomic_1076
private theorem within_008_047_agreement : RowAgreement 1074 1076 := RowAgreement.join atomic_1075_agreement atomic_1076_agreement
private def within_008_048 : CountSegment 1 10 13120 1072 1076 594 595 := CountSegment.join within_008_046 within_008_047
private theorem within_008_048_agreement : RowAgreement 1072 1076 := RowAgreement.join within_008_046_agreement within_008_047_agreement
private def within_008_049 : CountSegment 1 10 13120 1076 1078 595 596 := CountSegment.join atomic_1077 atomic_1078
private theorem within_008_049_agreement : RowAgreement 1076 1078 := RowAgreement.join atomic_1077_agreement atomic_1078_agreement
private def within_008_050 : CountSegment 1 10 13120 1078 1080 596 598 := CountSegment.join atomic_1079 atomic_1080
private theorem within_008_050_agreement : RowAgreement 1078 1080 := RowAgreement.join atomic_1079_agreement atomic_1080_agreement
private def within_008_051 : CountSegment 1 10 13120 1076 1080 595 598 := CountSegment.join within_008_049 within_008_050
private theorem within_008_051_agreement : RowAgreement 1076 1080 := RowAgreement.join within_008_049_agreement within_008_050_agreement
private def within_008_052 : CountSegment 1 10 13120 1072 1080 594 598 := CountSegment.join within_008_048 within_008_051
private theorem within_008_052_agreement : RowAgreement 1072 1080 := RowAgreement.join within_008_048_agreement within_008_051_agreement
private def within_008_053 : CountSegment 1 10 13120 1080 1082 598 600 := CountSegment.join atomic_1081 atomic_1082
private theorem within_008_053_agreement : RowAgreement 1080 1082 := RowAgreement.join atomic_1081_agreement atomic_1082_agreement
private def within_008_054 : CountSegment 1 10 13120 1082 1084 600 601 := CountSegment.join atomic_1083 atomic_1084
private theorem within_008_054_agreement : RowAgreement 1082 1084 := RowAgreement.join atomic_1083_agreement atomic_1084_agreement
private def within_008_055 : CountSegment 1 10 13120 1080 1084 598 601 := CountSegment.join within_008_053 within_008_054
private theorem within_008_055_agreement : RowAgreement 1080 1084 := RowAgreement.join within_008_053_agreement within_008_054_agreement
private def within_008_056 : CountSegment 1 10 13120 1084 1086 601 602 := CountSegment.join atomic_1085 atomic_1086
private theorem within_008_056_agreement : RowAgreement 1084 1086 := RowAgreement.join atomic_1085_agreement atomic_1086_agreement
private def within_008_057 : CountSegment 1 10 13120 1086 1088 602 603 := CountSegment.join atomic_1087 atomic_1088
private theorem within_008_057_agreement : RowAgreement 1086 1088 := RowAgreement.join atomic_1087_agreement atomic_1088_agreement
private def within_008_058 : CountSegment 1 10 13120 1084 1088 601 603 := CountSegment.join within_008_056 within_008_057
private theorem within_008_058_agreement : RowAgreement 1084 1088 := RowAgreement.join within_008_056_agreement within_008_057_agreement
private def within_008_059 : CountSegment 1 10 13120 1080 1088 598 603 := CountSegment.join within_008_055 within_008_058
private theorem within_008_059_agreement : RowAgreement 1080 1088 := RowAgreement.join within_008_055_agreement within_008_058_agreement
private def within_008_060 : CountSegment 1 10 13120 1072 1088 594 603 := CountSegment.join within_008_052 within_008_059
private theorem within_008_060_agreement : RowAgreement 1072 1088 := RowAgreement.join within_008_052_agreement within_008_059_agreement
private def within_008_061 : CountSegment 1 10 13120 1056 1088 585 603 := CountSegment.join within_008_045 within_008_060
private theorem within_008_061_agreement : RowAgreement 1056 1088 := RowAgreement.join within_008_045_agreement within_008_060_agreement
private def within_008_062 : CountSegment 1 10 13120 1024 1088 563 603 := CountSegment.join within_008_030 within_008_061
private theorem within_008_062_agreement : RowAgreement 1024 1088 := RowAgreement.join within_008_030_agreement within_008_061_agreement
private def within_008_063 : CountSegment 1 10 13120 1088 1090 603 603 := CountSegment.join atomic_1089 atomic_1090
private theorem within_008_063_agreement : RowAgreement 1088 1090 := RowAgreement.join atomic_1089_agreement atomic_1090_agreement
private def within_008_064 : CountSegment 1 10 13120 1090 1092 603 604 := CountSegment.join atomic_1091 atomic_1092
private theorem within_008_064_agreement : RowAgreement 1090 1092 := RowAgreement.join atomic_1091_agreement atomic_1092_agreement
private def within_008_065 : CountSegment 1 10 13120 1088 1092 603 604 := CountSegment.join within_008_063 within_008_064
private theorem within_008_065_agreement : RowAgreement 1088 1092 := RowAgreement.join within_008_063_agreement within_008_064_agreement
private def within_008_066 : CountSegment 1 10 13120 1092 1094 604 604 := CountSegment.join atomic_1093 atomic_1094
private theorem within_008_066_agreement : RowAgreement 1092 1094 := RowAgreement.join atomic_1093_agreement atomic_1094_agreement
private def within_008_067 : CountSegment 1 10 13120 1094 1096 604 604 := CountSegment.join atomic_1095 atomic_1096
private theorem within_008_067_agreement : RowAgreement 1094 1096 := RowAgreement.join atomic_1095_agreement atomic_1096_agreement
private def within_008_068 : CountSegment 1 10 13120 1092 1096 604 604 := CountSegment.join within_008_066 within_008_067
private theorem within_008_068_agreement : RowAgreement 1092 1096 := RowAgreement.join within_008_066_agreement within_008_067_agreement
private def within_008_069 : CountSegment 1 10 13120 1088 1096 603 604 := CountSegment.join within_008_065 within_008_068
private theorem within_008_069_agreement : RowAgreement 1088 1096 := RowAgreement.join within_008_065_agreement within_008_068_agreement
private def within_008_070 : CountSegment 1 10 13120 1096 1098 604 604 := CountSegment.join atomic_1097 atomic_1098
private theorem within_008_070_agreement : RowAgreement 1096 1098 := RowAgreement.join atomic_1097_agreement atomic_1098_agreement
private def within_008_071 : CountSegment 1 10 13120 1098 1100 604 604 := CountSegment.join atomic_1099 atomic_1100
private theorem within_008_071_agreement : RowAgreement 1098 1100 := RowAgreement.join atomic_1099_agreement atomic_1100_agreement
private def within_008_072 : CountSegment 1 10 13120 1096 1100 604 604 := CountSegment.join within_008_070 within_008_071
private theorem within_008_072_agreement : RowAgreement 1096 1100 := RowAgreement.join within_008_070_agreement within_008_071_agreement
private def within_008_073 : CountSegment 1 10 13120 1100 1102 604 606 := CountSegment.join atomic_1101 atomic_1102
private theorem within_008_073_agreement : RowAgreement 1100 1102 := RowAgreement.join atomic_1101_agreement atomic_1102_agreement
private def within_008_074 : CountSegment 1 10 13120 1102 1104 606 607 := CountSegment.join atomic_1103 atomic_1104
private theorem within_008_074_agreement : RowAgreement 1102 1104 := RowAgreement.join atomic_1103_agreement atomic_1104_agreement
private def within_008_075 : CountSegment 1 10 13120 1100 1104 604 607 := CountSegment.join within_008_073 within_008_074
private theorem within_008_075_agreement : RowAgreement 1100 1104 := RowAgreement.join within_008_073_agreement within_008_074_agreement
private def within_008_076 : CountSegment 1 10 13120 1096 1104 604 607 := CountSegment.join within_008_072 within_008_075
private theorem within_008_076_agreement : RowAgreement 1096 1104 := RowAgreement.join within_008_072_agreement within_008_075_agreement
private def within_008_077 : CountSegment 1 10 13120 1088 1104 603 607 := CountSegment.join within_008_069 within_008_076
private theorem within_008_077_agreement : RowAgreement 1088 1104 := RowAgreement.join within_008_069_agreement within_008_076_agreement
private def within_008_078 : CountSegment 1 10 13120 1104 1106 607 608 := CountSegment.join atomic_1105 atomic_1106
private theorem within_008_078_agreement : RowAgreement 1104 1106 := RowAgreement.join atomic_1105_agreement atomic_1106_agreement
private def within_008_079 : CountSegment 1 10 13120 1106 1108 608 608 := CountSegment.join atomic_1107 atomic_1108
private theorem within_008_079_agreement : RowAgreement 1106 1108 := RowAgreement.join atomic_1107_agreement atomic_1108_agreement
private def within_008_080 : CountSegment 1 10 13120 1104 1108 607 608 := CountSegment.join within_008_078 within_008_079
private theorem within_008_080_agreement : RowAgreement 1104 1108 := RowAgreement.join within_008_078_agreement within_008_079_agreement
private def within_008_081 : CountSegment 1 10 13120 1108 1110 608 608 := CountSegment.join atomic_1109 atomic_1110
private theorem within_008_081_agreement : RowAgreement 1108 1110 := RowAgreement.join atomic_1109_agreement atomic_1110_agreement
private def within_008_082 : CountSegment 1 10 13120 1110 1112 608 609 := CountSegment.join atomic_1111 atomic_1112
private theorem within_008_082_agreement : RowAgreement 1110 1112 := RowAgreement.join atomic_1111_agreement atomic_1112_agreement
private def within_008_083 : CountSegment 1 10 13120 1108 1112 608 609 := CountSegment.join within_008_081 within_008_082
private theorem within_008_083_agreement : RowAgreement 1108 1112 := RowAgreement.join within_008_081_agreement within_008_082_agreement
private def within_008_084 : CountSegment 1 10 13120 1104 1112 607 609 := CountSegment.join within_008_080 within_008_083
private theorem within_008_084_agreement : RowAgreement 1104 1112 := RowAgreement.join within_008_080_agreement within_008_083_agreement
private def within_008_085 : CountSegment 1 10 13120 1112 1114 609 609 := CountSegment.join atomic_1113 atomic_1114
private theorem within_008_085_agreement : RowAgreement 1112 1114 := RowAgreement.join atomic_1113_agreement atomic_1114_agreement
private def within_008_086 : CountSegment 1 10 13120 1114 1116 609 609 := CountSegment.join atomic_1115 atomic_1116
private theorem within_008_086_agreement : RowAgreement 1114 1116 := RowAgreement.join atomic_1115_agreement atomic_1116_agreement
private def within_008_087 : CountSegment 1 10 13120 1112 1116 609 609 := CountSegment.join within_008_085 within_008_086
private theorem within_008_087_agreement : RowAgreement 1112 1116 := RowAgreement.join within_008_085_agreement within_008_086_agreement
private def within_008_088 : CountSegment 1 10 13120 1116 1118 609 609 := CountSegment.join atomic_1117 atomic_1118
private theorem within_008_088_agreement : RowAgreement 1116 1118 := RowAgreement.join atomic_1117_agreement atomic_1118_agreement
private def within_008_089 : CountSegment 1 10 13120 1118 1120 609 610 := CountSegment.join atomic_1119 atomic_1120
private theorem within_008_089_agreement : RowAgreement 1118 1120 := RowAgreement.join atomic_1119_agreement atomic_1120_agreement
private def within_008_090 : CountSegment 1 10 13120 1116 1120 609 610 := CountSegment.join within_008_088 within_008_089
private theorem within_008_090_agreement : RowAgreement 1116 1120 := RowAgreement.join within_008_088_agreement within_008_089_agreement
private def within_008_091 : CountSegment 1 10 13120 1112 1120 609 610 := CountSegment.join within_008_087 within_008_090
private theorem within_008_091_agreement : RowAgreement 1112 1120 := RowAgreement.join within_008_087_agreement within_008_090_agreement
private def within_008_092 : CountSegment 1 10 13120 1104 1120 607 610 := CountSegment.join within_008_084 within_008_091
private theorem within_008_092_agreement : RowAgreement 1104 1120 := RowAgreement.join within_008_084_agreement within_008_091_agreement
private def within_008_093 : CountSegment 1 10 13120 1088 1120 603 610 := CountSegment.join within_008_077 within_008_092
private theorem within_008_093_agreement : RowAgreement 1088 1120 := RowAgreement.join within_008_077_agreement within_008_092_agreement
private def within_008_094 : CountSegment 1 10 13120 1120 1122 610 611 := CountSegment.join atomic_1121 atomic_1122
private theorem within_008_094_agreement : RowAgreement 1120 1122 := RowAgreement.join atomic_1121_agreement atomic_1122_agreement
private def within_008_095 : CountSegment 1 10 13120 1122 1124 611 612 := CountSegment.join atomic_1123 atomic_1124
private theorem within_008_095_agreement : RowAgreement 1122 1124 := RowAgreement.join atomic_1123_agreement atomic_1124_agreement
private def within_008_096 : CountSegment 1 10 13120 1120 1124 610 612 := CountSegment.join within_008_094 within_008_095
private theorem within_008_096_agreement : RowAgreement 1120 1124 := RowAgreement.join within_008_094_agreement within_008_095_agreement
private def within_008_097 : CountSegment 1 10 13120 1124 1126 612 612 := CountSegment.join atomic_1125 atomic_1126
private theorem within_008_097_agreement : RowAgreement 1124 1126 := RowAgreement.join atomic_1125_agreement atomic_1126_agreement
private def within_008_098 : CountSegment 1 10 13120 1126 1128 612 613 := CountSegment.join atomic_1127 atomic_1128
private theorem within_008_098_agreement : RowAgreement 1126 1128 := RowAgreement.join atomic_1127_agreement atomic_1128_agreement
private def within_008_099 : CountSegment 1 10 13120 1124 1128 612 613 := CountSegment.join within_008_097 within_008_098
private theorem within_008_099_agreement : RowAgreement 1124 1128 := RowAgreement.join within_008_097_agreement within_008_098_agreement
private def within_008_100 : CountSegment 1 10 13120 1120 1128 610 613 := CountSegment.join within_008_096 within_008_099
private theorem within_008_100_agreement : RowAgreement 1120 1128 := RowAgreement.join within_008_096_agreement within_008_099_agreement
private def within_008_101 : CountSegment 1 10 13120 1128 1130 613 615 := CountSegment.join atomic_1129 atomic_1130
private theorem within_008_101_agreement : RowAgreement 1128 1130 := RowAgreement.join atomic_1129_agreement atomic_1130_agreement
private def within_008_102 : CountSegment 1 10 13120 1130 1132 615 617 := CountSegment.join atomic_1131 atomic_1132
private theorem within_008_102_agreement : RowAgreement 1130 1132 := RowAgreement.join atomic_1131_agreement atomic_1132_agreement
private def within_008_103 : CountSegment 1 10 13120 1128 1132 613 617 := CountSegment.join within_008_101 within_008_102
private theorem within_008_103_agreement : RowAgreement 1128 1132 := RowAgreement.join within_008_101_agreement within_008_102_agreement
private def within_008_104 : CountSegment 1 10 13120 1132 1134 617 618 := CountSegment.join atomic_1133 atomic_1134
private theorem within_008_104_agreement : RowAgreement 1132 1134 := RowAgreement.join atomic_1133_agreement atomic_1134_agreement
private def within_008_105 : CountSegment 1 10 13120 1134 1136 618 618 := CountSegment.join atomic_1135 atomic_1136
private theorem within_008_105_agreement : RowAgreement 1134 1136 := RowAgreement.join atomic_1135_agreement atomic_1136_agreement
private def within_008_106 : CountSegment 1 10 13120 1132 1136 617 618 := CountSegment.join within_008_104 within_008_105
private theorem within_008_106_agreement : RowAgreement 1132 1136 := RowAgreement.join within_008_104_agreement within_008_105_agreement
private def within_008_107 : CountSegment 1 10 13120 1128 1136 613 618 := CountSegment.join within_008_103 within_008_106
private theorem within_008_107_agreement : RowAgreement 1128 1136 := RowAgreement.join within_008_103_agreement within_008_106_agreement
private def within_008_108 : CountSegment 1 10 13120 1120 1136 610 618 := CountSegment.join within_008_100 within_008_107
private theorem within_008_108_agreement : RowAgreement 1120 1136 := RowAgreement.join within_008_100_agreement within_008_107_agreement
private def within_008_109 : CountSegment 1 10 13120 1136 1138 618 620 := CountSegment.join atomic_1137 atomic_1138
private theorem within_008_109_agreement : RowAgreement 1136 1138 := RowAgreement.join atomic_1137_agreement atomic_1138_agreement
private def within_008_110 : CountSegment 1 10 13120 1138 1140 620 621 := CountSegment.join atomic_1139 atomic_1140
private theorem within_008_110_agreement : RowAgreement 1138 1140 := RowAgreement.join atomic_1139_agreement atomic_1140_agreement
private def within_008_111 : CountSegment 1 10 13120 1136 1140 618 621 := CountSegment.join within_008_109 within_008_110
private theorem within_008_111_agreement : RowAgreement 1136 1140 := RowAgreement.join within_008_109_agreement within_008_110_agreement
private def within_008_112 : CountSegment 1 10 13120 1140 1142 621 621 := CountSegment.join atomic_1141 atomic_1142
private theorem within_008_112_agreement : RowAgreement 1140 1142 := RowAgreement.join atomic_1141_agreement atomic_1142_agreement
private def within_008_113 : CountSegment 1 10 13120 1142 1144 621 622 := CountSegment.join atomic_1143 atomic_1144
private theorem within_008_113_agreement : RowAgreement 1142 1144 := RowAgreement.join atomic_1143_agreement atomic_1144_agreement
private def within_008_114 : CountSegment 1 10 13120 1140 1144 621 622 := CountSegment.join within_008_112 within_008_113
private theorem within_008_114_agreement : RowAgreement 1140 1144 := RowAgreement.join within_008_112_agreement within_008_113_agreement
private def within_008_115 : CountSegment 1 10 13120 1136 1144 618 622 := CountSegment.join within_008_111 within_008_114
private theorem within_008_115_agreement : RowAgreement 1136 1144 := RowAgreement.join within_008_111_agreement within_008_114_agreement
private def within_008_116 : CountSegment 1 10 13120 1144 1146 622 622 := CountSegment.join atomic_1145 atomic_1146
private theorem within_008_116_agreement : RowAgreement 1144 1146 := RowAgreement.join atomic_1145_agreement atomic_1146_agreement
private def within_008_117 : CountSegment 1 10 13120 1146 1148 622 623 := CountSegment.join atomic_1147 atomic_1148
private theorem within_008_117_agreement : RowAgreement 1146 1148 := RowAgreement.join atomic_1147_agreement atomic_1148_agreement
private def within_008_118 : CountSegment 1 10 13120 1144 1148 622 623 := CountSegment.join within_008_116 within_008_117
private theorem within_008_118_agreement : RowAgreement 1144 1148 := RowAgreement.join within_008_116_agreement within_008_117_agreement
private def within_008_119 : CountSegment 1 10 13120 1148 1150 623 623 := CountSegment.join atomic_1149 atomic_1150
private theorem within_008_119_agreement : RowAgreement 1148 1150 := RowAgreement.join atomic_1149_agreement atomic_1150_agreement
private def within_008_120 : CountSegment 1 10 13120 1150 1152 623 626 := CountSegment.join atomic_1151 atomic_1152
private theorem within_008_120_agreement : RowAgreement 1150 1152 := RowAgreement.join atomic_1151_agreement atomic_1152_agreement
private def within_008_121 : CountSegment 1 10 13120 1148 1152 623 626 := CountSegment.join within_008_119 within_008_120
private theorem within_008_121_agreement : RowAgreement 1148 1152 := RowAgreement.join within_008_119_agreement within_008_120_agreement
private def within_008_122 : CountSegment 1 10 13120 1144 1152 622 626 := CountSegment.join within_008_118 within_008_121
private theorem within_008_122_agreement : RowAgreement 1144 1152 := RowAgreement.join within_008_118_agreement within_008_121_agreement
private def within_008_123 : CountSegment 1 10 13120 1136 1152 618 626 := CountSegment.join within_008_115 within_008_122
private theorem within_008_123_agreement : RowAgreement 1136 1152 := RowAgreement.join within_008_115_agreement within_008_122_agreement
private def within_008_124 : CountSegment 1 10 13120 1120 1152 610 626 := CountSegment.join within_008_108 within_008_123
private theorem within_008_124_agreement : RowAgreement 1120 1152 := RowAgreement.join within_008_108_agreement within_008_123_agreement
private def within_008_125 : CountSegment 1 10 13120 1088 1152 603 626 := CountSegment.join within_008_093 within_008_124
private theorem within_008_125_agreement : RowAgreement 1088 1152 := RowAgreement.join within_008_093_agreement within_008_124_agreement
private def within_008_126 : CountSegment 1 10 13120 1024 1152 563 626 := CountSegment.join within_008_062 within_008_125
private theorem within_008_126_agreement : RowAgreement 1024 1152 := RowAgreement.join within_008_062_agreement within_008_125_agreement
def segment_008 : CountSegment 1 10 13120 1024 1152 563 626 := within_008_126
theorem agreement_008 : RowAgreement 1024 1152 := within_008_126_agreement
end ProjectEulerSolutions.P58Numerical
