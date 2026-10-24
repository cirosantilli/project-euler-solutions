import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_6017 : RowCertificate 6017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6017 1) := by change ¬ Nat.Prime 144829191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48276397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6017 2) := by change ¬ Nat.Prime 144817157; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 314137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6017 3) := by change ¬ Nat.Prime 144805123; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 4671133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6017 : row 6017 = 0 := (row_6017.checked).trans rfl
theorem model_count_6017 : modelRow 6017 = 0 := by decide
theorem modelrow_agrees_6017 : modelRow 6017 = row 6017 := model_count_6017.trans row_count_6017.symm
def row_6018 : RowCertificate 6018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6018 1) := by change ¬ Nat.Prime 144877333; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1984621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6018 2) := by change Nat.Prime 144865297; exact ProjectEulerSolutions.P58PrimeNodes.p_144865297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6018 3) := by change ¬ Nat.Prime 144853261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20693323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6018 : row 6018 = 1 := (row_6018.checked).trans rfl
theorem model_count_6018 : modelRow 6018 = 1 := by decide
theorem modelrow_agrees_6018 : modelRow 6018 = row 6018 := model_count_6018.trans row_count_6018.symm
def row_6019 : RowCertificate 6019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6019 1) := by change ¬ Nat.Prime 144925483; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7627657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6019 2) := by change ¬ Nat.Prime 144913445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 28982689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6019 3) := by change ¬ Nat.Prime 144901407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48300469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6019 : row 6019 = 0 := (row_6019.checked).trans rfl
theorem model_count_6019 : modelRow 6019 = 0 := by decide
theorem modelrow_agrees_6019 : modelRow 6019 = row 6019 := model_count_6019.trans row_count_6019.symm
def row_6020 : RowCertificate 6020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6020 1) := by change ¬ Nat.Prime 144973641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48324547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6020 2) := by change ¬ Nat.Prime 144961601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8527153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6020 3) := by change ¬ Nat.Prime 144949561; exact PrattCertificates.composite_of_product (factor := 643) (cofactor := 225427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6020 : row 6020 = 0 := (row_6020.checked).trans rfl
theorem model_count_6020 : modelRow 6020 = 0 := by decide
theorem modelrow_agrees_6020 : modelRow 6020 = row 6020 := model_count_6020.trans row_count_6020.symm
def row_6021 : RowCertificate 6021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6021 1) := by change ¬ Nat.Prime 145021807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20717401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6021 2) := by change ¬ Nat.Prime 145009765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29001953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6021 3) := by change ¬ Nat.Prime 144997723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11153671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6021 : row 6021 = 0 := (row_6021.checked).trans rfl
theorem model_count_6021 : modelRow 6021 = 0 := by decide
theorem modelrow_agrees_6021 : modelRow 6021 = row 6021 := model_count_6021.trans row_count_6021.symm
def row_6022 : RowCertificate 6022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6022 1) := by change ¬ Nat.Prime 145069981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20724283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6022 2) := by change ¬ Nat.Prime 145057937; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 410929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6022 3) := by change ¬ Nat.Prime 145045893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48348631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6022 : row 6022 = 0 := (row_6022.checked).trans rfl
theorem model_count_6022 : modelRow 6022 = 0 := by decide
theorem modelrow_agrees_6022 : modelRow 6022 = row 6022 := model_count_6022.trans row_count_6022.symm
def row_6023 : RowCertificate 6023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6023 1) := by change ¬ Nat.Prime 145118163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48372721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6023 2) := by change ¬ Nat.Prime 145106117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11162009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6023 3) := by change ¬ Nat.Prime 145094071; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1142473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6023 : row 6023 = 0 := (row_6023.checked).trans rfl
theorem model_count_6023 : modelRow 6023 = 0 := by decide
theorem modelrow_agrees_6023 : modelRow 6023 = row 6023 := model_count_6023.trans row_count_6023.symm
def row_6024 : RowCertificate 6024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6024 1) := by change ¬ Nat.Prime 145166353; exact PrattCertificates.composite_of_product (factor := 2083) (cofactor := 69691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6024 2) := by change ¬ Nat.Prime 145154305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29030861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6024 3) := by change ¬ Nat.Prime 145142257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11164789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6024 : row 6024 = 0 := (row_6024.checked).trans rfl
theorem model_count_6024 : modelRow 6024 = 0 := by decide
theorem modelrow_agrees_6024 : modelRow 6024 = row 6024 := model_count_6024.trans row_count_6024.symm
def row_6025 : RowCertificate 6025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6025 1) := by change ¬ Nat.Prime 145214551; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1044709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6025 2) := by change ¬ Nat.Prime 145202501; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 1496933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6025 3) := by change ¬ Nat.Prime 145190451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48396817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6025 : row 6025 = 0 := (row_6025.checked).trans rfl
theorem model_count_6025 : modelRow 6025 = 0 := by decide
theorem modelrow_agrees_6025 : modelRow 6025 = row 6025 := model_count_6025.trans row_count_6025.symm
def row_6026 : RowCertificate 6026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6026 1) := by change ¬ Nat.Prime 145262757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48420919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6026 2) := by change ¬ Nat.Prime 145250705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29050141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6026 3) := by change ¬ Nat.Prime 145238653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20748379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6026 : row 6026 = 0 := (row_6026.checked).trans rfl
theorem model_count_6026 : modelRow 6026 = 0 := by decide
theorem modelrow_agrees_6026 : modelRow 6026 = row 6026 := model_count_6026.trans row_count_6026.symm
def row_6027 : RowCertificate 6027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6027 1) := by change ¬ Nat.Prime 145310971; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11177767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6027 2) := by change ¬ Nat.Prime 145298917; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 2741489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6027 3) := by change ¬ Nat.Prime 145286863; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7646677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6027 : row 6027 = 0 := (row_6027.checked).trans rfl
theorem model_count_6027 : modelRow 6027 = 0 := by decide
theorem modelrow_agrees_6027 : modelRow 6027 = row 6027 := model_count_6027.trans row_count_6027.symm
def row_6028 : RowCertificate 6028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6028 1) := by change ¬ Nat.Prime 145359193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20765599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6028 2) := by change ¬ Nat.Prime 145347137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11180549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6028 3) := by change ¬ Nat.Prime 145335081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48445027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6028 : row 6028 = 0 := (row_6028.checked).trans rfl
theorem model_count_6028 : modelRow 6028 = 0 := by decide
theorem modelrow_agrees_6028 : modelRow 6028 = row 6028 := model_count_6028.trans row_count_6028.symm
def row_6029 : RowCertificate 6029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6029 1) := by change ¬ Nat.Prime 145407423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48469141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6029 2) := by change ¬ Nat.Prime 145395365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29079073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6029 3) := by change ¬ Nat.Prime 145383307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7651753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6029 : row 6029 = 0 := (row_6029.checked).trans rfl
theorem model_count_6029 : modelRow 6029 = 0 := by decide
theorem modelrow_agrees_6029 : modelRow 6029 = row 6029 := model_count_6029.trans row_count_6029.symm
def row_6030 : RowCertificate 6030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6030 1) := by change ¬ Nat.Prime 145455661; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11188897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6030 2) := by change Nat.Prime 145443601; exact ProjectEulerSolutions.P58PrimeNodes.p_145443601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6030 3) := by change ¬ Nat.Prime 145431541; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 314107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6030 : row 6030 = 1 := (row_6030.checked).trans rfl
theorem model_count_6030 : modelRow 6030 = 1 := by decide
theorem modelrow_agrees_6030 : modelRow 6030 = row 6030 := model_count_6030.trans row_count_6030.symm
def row_6031 : RowCertificate 6031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6031 1) := by change ¬ Nat.Prime 145503907; exact PrattCertificates.composite_of_product (factor := 4801) (cofactor := 30307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6031 2) := by change ¬ Nat.Prime 145491845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29098369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6031 3) := by change ¬ Nat.Prime 145479783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48493261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6031 : row 6031 = 0 := (row_6031.checked).trans rfl
theorem model_count_6031 : modelRow 6031 = 0 := by decide
theorem modelrow_agrees_6031 : modelRow 6031 = row 6031 := model_count_6031.trans row_count_6031.symm
def row_6032 : RowCertificate 6032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6032 1) := by change ¬ Nat.Prime 145552161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48517387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6032 2) := by change Nat.Prime 145540097; exact ProjectEulerSolutions.P58PrimeNodes.p_145540097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6032 3) := by change ¬ Nat.Prime 145528033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20789719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6032 : row 6032 = 1 := (row_6032.checked).trans rfl
theorem model_count_6032 : modelRow 6032 = 1 := by decide
theorem modelrow_agrees_6032 : modelRow 6032 = row 6032 := model_count_6032.trans row_count_6032.symm
def row_6033 : RowCertificate 6033 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6033 1) := by change Nat.Prime 145600423; exact ProjectEulerSolutions.P58PrimeNodes.p_145600423.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6033 2) := by change ¬ Nat.Prime 145588357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8564021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6033 3) := by change ¬ Nat.Prime 145576291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20796613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6033 : row 6033 = 1 := (row_6033.checked).trans rfl
theorem model_count_6033 : modelRow 6033 = 1 := by decide
theorem modelrow_agrees_6033 : modelRow 6033 = row 6033 := model_count_6033.trans row_count_6033.symm
def row_6034 : RowCertificate 6034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6034 1) := by change ¬ Nat.Prime 145648693; exact PrattCertificates.composite_of_product (factor := 2281) (cofactor := 63853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6034 2) := by change ¬ Nat.Prime 145636625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29127325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6034 3) := by change ¬ Nat.Prime 145624557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48541519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6034 : row 6034 = 0 := (row_6034.checked).trans rfl
theorem model_count_6034 : modelRow 6034 = 0 := by decide
theorem modelrow_agrees_6034 : modelRow 6034 = row 6034 := model_count_6034.trans row_count_6034.symm
def row_6035 : RowCertificate 6035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6035 1) := by change ¬ Nat.Prime 145696971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48565657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6035 2) := by change ¬ Nat.Prime 145684901; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 1636909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6035 3) := by change ¬ Nat.Prime 145672831; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 117763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6035 : row 6035 = 0 := (row_6035.checked).trans rfl
theorem model_count_6035 : modelRow 6035 = 0 := by decide
theorem modelrow_agrees_6035 : modelRow 6035 = row 6035 := model_count_6035.trans row_count_6035.symm
def row_6036 : RowCertificate 6036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6036 1) := by change ¬ Nat.Prime 145745257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20820751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6036 2) := by change ¬ Nat.Prime 145733185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29146637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6036 3) := by change ¬ Nat.Prime 145721113; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 526069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6036 : row 6036 = 0 := (row_6036.checked).trans rfl
theorem model_count_6036 : modelRow 6036 = 0 := by decide
theorem modelrow_agrees_6036 : modelRow 6036 = row 6036 := model_count_6036.trans row_count_6036.symm
def row_6037 : RowCertificate 6037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6037 1) := by change ¬ Nat.Prime 145793551; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 755407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6037 2) := by change ¬ Nat.Prime 145781477; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8575381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6037 3) := by change ¬ Nat.Prime 145769403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48589801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6037 : row 6037 = 0 := (row_6037.checked).trans rfl
theorem model_count_6037 : modelRow 6037 = 0 := by decide
theorem modelrow_agrees_6037 : modelRow 6037 = row 6037 := model_count_6037.trans row_count_6037.symm
def row_6038 : RowCertificate 6038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6038 1) := by change ¬ Nat.Prime 145841853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48613951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6038 2) := by change ¬ Nat.Prime 145829777; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 5028613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6038 3) := by change Nat.Prime 145817701; exact ProjectEulerSolutions.P58PrimeNodes.p_145817701.prime
    exact decide_eq_true prime
theorem row_count_6038 : row 6038 = 1 := (row_6038.checked).trans rfl
theorem model_count_6038 : modelRow 6038 = 1 := by decide
theorem modelrow_agrees_6038 : modelRow 6038 = row 6038 := model_count_6038.trans row_count_6038.symm
def row_6039 : RowCertificate 6039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6039 1) := by change ¬ Nat.Prime 145890163; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 806023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6039 2) := by change ¬ Nat.Prime 145878085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29175617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6039 3) := by change ¬ Nat.Prime 145866007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20838001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6039 : row 6039 = 0 := (row_6039.checked).trans rfl
theorem model_count_6039 : modelRow 6039 = 0 := by decide
theorem modelrow_agrees_6039 : modelRow 6039 = row 6039 := model_count_6039.trans row_count_6039.symm
def row_6040 : RowCertificate 6040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6040 1) := by change ¬ Nat.Prime 145938481; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11226037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6040 2) := by change ¬ Nat.Prime 145926401; exact PrattCertificates.composite_of_product (factor := 1289) (cofactor := 113209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6040 3) := by change ¬ Nat.Prime 145914321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48638107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6040 : row 6040 = 0 := (row_6040.checked).trans rfl
theorem model_count_6040 : modelRow 6040 = 0 := by decide
theorem modelrow_agrees_6040 : modelRow 6040 = row 6040 := model_count_6040.trans row_count_6040.symm
def row_6041 : RowCertificate 6041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6041 1) := by change ¬ Nat.Prime 145986807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48662269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6041 2) := by change ¬ Nat.Prime 145974725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29194945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6041 3) := by change ¬ Nat.Prime 145962643; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 654541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6041 : row 6041 = 0 := (row_6041.checked).trans rfl
theorem model_count_6041 : modelRow 6041 = 0 := by decide
theorem modelrow_agrees_6041 : modelRow 6041 = row 6041 := model_count_6041.trans row_count_6041.symm
def row_6042 : RowCertificate 6042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6042 1) := by change ¬ Nat.Prime 146035141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20862163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6042 2) := by change Nat.Prime 146023057; exact ProjectEulerSolutions.P58PrimeNodes.p_146023057.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6042 3) := by change Nat.Prime 146010973; exact ProjectEulerSolutions.P58PrimeNodes.p_146010973.prime
    exact decide_eq_true prime
theorem row_count_6042 : row 6042 = 2 := (row_6042.checked).trans rfl
theorem model_count_6042 : modelRow 6042 = 2 := by decide
theorem modelrow_agrees_6042 : modelRow 6042 = row 6042 := model_count_6042.trans row_count_6042.symm
def row_6043 : RowCertificate 6043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6043 1) := by change ¬ Nat.Prime 146083483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20869069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6043 2) := by change ¬ Nat.Prime 146071397; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 3562717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6043 3) := by change ¬ Nat.Prime 146059311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48686437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6043 : row 6043 = 0 := (row_6043.checked).trans rfl
theorem model_count_6043 : modelRow 6043 = 0 := by decide
theorem modelrow_agrees_6043 : modelRow 6043 = row 6043 := model_count_6043.trans row_count_6043.symm
def row_6044 : RowCertificate 6044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6044 1) := by change ¬ Nat.Prime 146131833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48710611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6044 2) := by change ¬ Nat.Prime 146119745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29223949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6044 3) := by change Nat.Prime 146107657; exact ProjectEulerSolutions.P58PrimeNodes.p_146107657.prime
    exact decide_eq_true prime
theorem row_count_6044 : row 6044 = 1 := (row_6044.checked).trans rfl
theorem model_count_6044 : modelRow 6044 = 1 := by decide
theorem modelrow_agrees_6044 : modelRow 6044 = row 6044 := model_count_6044.trans row_count_6044.symm
def row_6045 : RowCertificate 6045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6045 1) := by change ¬ Nat.Prime 146180191; exact PrattCertificates.composite_of_product (factor := 3469) (cofactor := 42139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6045 2) := by change Nat.Prime 146168101; exact ProjectEulerSolutions.P58PrimeNodes.p_146168101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6045 3) := by change ¬ Nat.Prime 146156011; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 3398977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6045 : row 6045 = 1 := (row_6045.checked).trans rfl
theorem model_count_6045 : modelRow 6045 = 1 := by decide
theorem modelrow_agrees_6045 : modelRow 6045 = row 6045 := model_count_6045.trans row_count_6045.symm
def row_6046 : RowCertificate 6046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6046 1) := by change ¬ Nat.Prime 146228557; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 418993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6046 2) := by change ¬ Nat.Prime 146216465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29243293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6046 3) := by change ¬ Nat.Prime 146204373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48734791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6046 : row 6046 = 0 := (row_6046.checked).trans rfl
theorem model_count_6046 : modelRow 6046 = 0 := by decide
theorem modelrow_agrees_6046 : modelRow 6046 = row 6046 := model_count_6046.trans row_count_6046.symm
def row_6047 : RowCertificate 6047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6047 1) := by change ¬ Nat.Prime 146276931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48758977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6047 2) := by change Nat.Prime 146264837; exact ProjectEulerSolutions.P58PrimeNodes.p_146264837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6047 3) := by change ¬ Nat.Prime 146252743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20893249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6047 : row 6047 = 1 := (row_6047.checked).trans rfl
theorem model_count_6047 : modelRow 6047 = 1 := by decide
theorem modelrow_agrees_6047 : modelRow 6047 = row 6047 := model_count_6047.trans row_count_6047.symm
def row_6048 : RowCertificate 6048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6048 1) := by change ¬ Nat.Prime 146325313; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 102397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6048 2) := by change Nat.Prime 146313217; exact ProjectEulerSolutions.P58PrimeNodes.p_146313217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6048 3) := by change ¬ Nat.Prime 146301121; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7700059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6048 : row 6048 = 1 := (row_6048.checked).trans rfl
theorem model_count_6048 : modelRow 6048 = 1 := by decide
theorem modelrow_agrees_6048 : modelRow 6048 = row 6048 := model_count_6048.trans row_count_6048.symm
def row_6049 : RowCertificate 6049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6049 1) := by change ¬ Nat.Prime 146373703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20910529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6049 2) := by change ¬ Nat.Prime 146361605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29272321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6049 3) := by change ¬ Nat.Prime 146349507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48783169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6049 : row 6049 = 0 := (row_6049.checked).trans rfl
theorem model_count_6049 : modelRow 6049 = 0 := by decide
theorem modelrow_agrees_6049 : modelRow 6049 = row 6049 := model_count_6049.trans row_count_6049.symm
def row_6050 : RowCertificate 6050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6050 1) := by change ¬ Nat.Prime 146422101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48807367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6050 2) := by change ¬ Nat.Prime 146410001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8612353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6050 3) := by change ¬ Nat.Prime 146397901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11261377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6050 : row 6050 = 0 := (row_6050.checked).trans rfl
theorem model_count_6050 : modelRow 6050 = 0 := by decide
theorem modelrow_agrees_6050 : modelRow 6050 = row 6050 := model_count_6050.trans row_count_6050.symm
def row_6051 : RowCertificate 6051 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6051 1) := by change ¬ Nat.Prime 146470507; exact PrattCertificates.composite_of_product (factor := 1723) (cofactor := 85009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6051 2) := by change ¬ Nat.Prime 146458405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29291681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6051 3) := by change ¬ Nat.Prime 146446303; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 932779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6051 : row 6051 = 0 := (row_6051.checked).trans rfl
theorem model_count_6051 : modelRow 6051 = 0 := by decide
theorem modelrow_agrees_6051 : modelRow 6051 = row 6051 := model_count_6051.trans row_count_6051.symm
def row_6052 : RowCertificate 6052 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6052 1) := by change Nat.Prime 146518921; exact ProjectEulerSolutions.P58PrimeNodes.p_146518921.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6052 2) := by change ¬ Nat.Prime 146506817; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 3573337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6052 3) := by change ¬ Nat.Prime 146494713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48831571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6052 : row 6052 = 1 := (row_6052.checked).trans rfl
theorem model_count_6052 : modelRow 6052 = 1 := by decide
theorem modelrow_agrees_6052 : modelRow 6052 = row 6052 := model_count_6052.trans row_count_6052.symm
def row_6053 : RowCertificate 6053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6053 1) := by change ¬ Nat.Prime 146567343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48855781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6053 2) := by change ¬ Nat.Prime 146555237; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 1296949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6053 3) := by change ¬ Nat.Prime 146543131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20934733) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6053 : row 6053 = 0 := (row_6053.checked).trans rfl
theorem model_count_6053 : modelRow 6053 = 0 := by decide
theorem modelrow_agrees_6053 : modelRow 6053 = row 6053 := model_count_6053.trans row_count_6053.symm
def row_6054 : RowCertificate 6054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6054 1) := by change ¬ Nat.Prime 146615773; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 468421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6054 2) := by change ¬ Nat.Prime 146603665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29320733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6054 3) := by change ¬ Nat.Prime 146591557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20941651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6054 : row 6054 = 0 := (row_6054.checked).trans rfl
theorem model_count_6054 : modelRow 6054 = 0 := by decide
theorem modelrow_agrees_6054 : modelRow 6054 = row 6054 := model_count_6054.trans row_count_6054.symm
def row_6055 : RowCertificate 6055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6055 1) := by change ¬ Nat.Prime 146664211; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7719169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6055 2) := by change ¬ Nat.Prime 146652101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 5056969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6055 3) := by change ¬ Nat.Prime 146639991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48879997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6055 : row 6055 = 0 := (row_6055.checked).trans rfl
theorem model_count_6055 : modelRow 6055 = 0 := by decide
theorem modelrow_agrees_6055 : modelRow 6055 = row 6055 := model_count_6055.trans row_count_6055.symm
def row_6056 : RowCertificate 6056 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6056 1) := by change ¬ Nat.Prime 146712657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48904219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6056 2) := by change ¬ Nat.Prime 146700545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29340109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6056 3) := by change Nat.Prime 146688433; exact ProjectEulerSolutions.P58PrimeNodes.p_146688433.prime
    exact decide_eq_true prime
theorem row_count_6056 : row 6056 = 1 := (row_6056.checked).trans rfl
theorem model_count_6056 : modelRow 6056 = 1 := by decide
theorem modelrow_agrees_6056 : modelRow 6056 = row 6056 := model_count_6056.trans row_count_6056.symm
def row_6057 : RowCertificate 6057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6057 1) := by change ¬ Nat.Prime 146761111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20965873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6057 2) := by change ¬ Nat.Prime 146748997; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 2768849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6057 3) := by change Nat.Prime 146736883; exact ProjectEulerSolutions.P58PrimeNodes.p_146736883.prime
    exact decide_eq_true prime
theorem row_count_6057 : row 6057 = 1 := (row_6057.checked).trans rfl
theorem model_count_6057 : modelRow 6057 = 1 := by decide
theorem modelrow_agrees_6057 : modelRow 6057 = row 6057 := model_count_6057.trans row_count_6057.symm
def row_6058 : RowCertificate 6058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6058 1) := by change ¬ Nat.Prime 146809573; exact PrattCertificates.composite_of_product (factor := 1549) (cofactor := 94777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6058 2) := by change ¬ Nat.Prime 146797457; exact PrattCertificates.composite_of_product (factor := 1193) (cofactor := 123049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6058 3) := by change ¬ Nat.Prime 146785341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48928447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6058 : row 6058 = 0 := (row_6058.checked).trans rfl
theorem model_count_6058 : modelRow 6058 = 0 := by decide
theorem modelrow_agrees_6058 : modelRow 6058 = row 6058 := model_count_6058.trans row_count_6058.symm
def row_6059 : RowCertificate 6059 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6059 1) := by change ¬ Nat.Prime 146858043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48952681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6059 2) := by change ¬ Nat.Prime 146845925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29369185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6059 3) := by change ¬ Nat.Prime 146833807; exact PrattCertificates.composite_of_product (factor := 7621) (cofactor := 19267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6059 : row 6059 = 0 := (row_6059.checked).trans rfl
theorem model_count_6059 : modelRow 6059 = 0 := by decide
theorem modelrow_agrees_6059 : modelRow 6059 = row 6059 := model_count_6059.trans row_count_6059.symm
def row_6060 : RowCertificate 6060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6060 1) := by change ¬ Nat.Prime 146906521; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 901267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6060 2) := by change Nat.Prime 146894401; exact ProjectEulerSolutions.P58PrimeNodes.p_146894401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6060 3) := by change ¬ Nat.Prime 146882281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 20983183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6060 : row 6060 = 1 := (row_6060.checked).trans rfl
theorem model_count_6060 : modelRow 6060 = 1 := by decide
theorem modelrow_agrees_6060 : modelRow 6060 = row 6060 := model_count_6060.trans row_count_6060.symm
def row_6061 : RowCertificate 6061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6061 1) := by change ¬ Nat.Prime 146955007; exact PrattCertificates.composite_of_product (factor := 6373) (cofactor := 23059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6061 2) := by change ¬ Nat.Prime 146942885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29388577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6061 3) := by change ¬ Nat.Prime 146930763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 48976921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6061 : row 6061 = 0 := (row_6061.checked).trans rfl
theorem model_count_6061 : modelRow 6061 = 0 := by decide
theorem modelrow_agrees_6061 : modelRow 6061 = row 6061 := model_count_6061.trans row_count_6061.symm
def row_6062 : RowCertificate 6062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6062 1) := by change ¬ Nat.Prime 147003501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49001167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6062 2) := by change ¬ Nat.Prime 146991377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11307029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6062 3) := by change Nat.Prime 146979253; exact ProjectEulerSolutions.P58PrimeNodes.p_146979253.prime
    exact decide_eq_true prime
theorem row_count_6062 : row 6062 = 1 := (row_6062.checked).trans rfl
theorem model_count_6062 : modelRow 6062 = 1 := by decide
theorem modelrow_agrees_6062 : modelRow 6062 = row 6062 := model_count_6062.trans row_count_6062.symm
def row_6063 : RowCertificate 6063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6063 1) := by change ¬ Nat.Prime 147052003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21007429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6063 2) := by change Nat.Prime 147039877; exact ProjectEulerSolutions.P58PrimeNodes.p_147039877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6063 3) := by change ¬ Nat.Prime 147027751; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11309827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6063 : row 6063 = 1 := (row_6063.checked).trans rfl
theorem model_count_6063 : modelRow 6063 = 1 := by decide
theorem modelrow_agrees_6063 : modelRow 6063 = row 6063 := model_count_6063.trans row_count_6063.symm
def row_6064 : RowCertificate 6064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6064 1) := by change ¬ Nat.Prime 147100513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21014359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6064 2) := by change ¬ Nat.Prime 147088385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29417677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6064 3) := by change ¬ Nat.Prime 147076257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49025419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6064 : row 6064 = 0 := (row_6064.checked).trans rfl
theorem model_count_6064 : modelRow 6064 = 0 := by decide
theorem modelrow_agrees_6064 : modelRow 6064 = row 6064 := model_count_6064.trans row_count_6064.symm
def row_6065 : RowCertificate 6065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6065 1) := by change ¬ Nat.Prime 147149031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49049677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6065 2) := by change ¬ Nat.Prime 147136901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 3976673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6065 3) := by change ¬ Nat.Prime 147124771; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7743409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6065 : row 6065 = 0 := (row_6065.checked).trans rfl
theorem model_count_6065 : modelRow 6065 = 0 := by decide
theorem modelrow_agrees_6065 : modelRow 6065 = row 6065 := model_count_6065.trans row_count_6065.symm
def row_6066 : RowCertificate 6066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6066 1) := by change ¬ Nat.Prime 147197557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11322889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6066 2) := by change ¬ Nat.Prime 147185425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29437085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6066 3) := by change Nat.Prime 147173293; exact ProjectEulerSolutions.P58PrimeNodes.p_147173293.prime
    exact decide_eq_true prime
theorem row_count_6066 : row 6066 = 1 := (row_6066.checked).trans rfl
theorem model_count_6066 : modelRow 6066 = 1 := by decide
theorem modelrow_agrees_6066 : modelRow 6066 = row 6066 := model_count_6066.trans row_count_6066.symm
def row_6067 : RowCertificate 6067 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6067 1) := by change Nat.Prime 147246091; exact ProjectEulerSolutions.P58PrimeNodes.p_147246091.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6067 2) := by change ¬ Nat.Prime 147233957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11325689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6067 3) := by change ¬ Nat.Prime 147221823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49073941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6067 : row 6067 = 1 := (row_6067.checked).trans rfl
theorem model_count_6067 : modelRow 6067 = 1 := by decide
theorem modelrow_agrees_6067 : modelRow 6067 = row 6067 := model_count_6067.trans row_count_6067.symm
def row_6068 : RowCertificate 6068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6068 1) := by change ¬ Nat.Prime 147294633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49098211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6068 2) := by change ¬ Nat.Prime 147282497; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 524137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6068 3) := by change ¬ Nat.Prime 147270361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21038623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6068 : row 6068 = 0 := (row_6068.checked).trans rfl
theorem model_count_6068 : modelRow 6068 = 0 := by decide
theorem modelrow_agrees_6068 : modelRow 6068 = row 6068 := model_count_6068.trans row_count_6068.symm
def row_6069 : RowCertificate 6069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6069 1) := by change ¬ Nat.Prime 147343183; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11334091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6069 2) := by change ¬ Nat.Prime 147331045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29466209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6069 3) := by change ¬ Nat.Prime 147318907; exact PrattCertificates.composite_of_product (factor := 9127) (cofactor := 16141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6069 : row 6069 = 0 := (row_6069.checked).trans rfl
theorem model_count_6069 : modelRow 6069 = 0 := by decide
theorem modelrow_agrees_6069 : modelRow 6069 = row 6069 := model_count_6069.trans row_count_6069.symm
def row_6070 : RowCertificate 6070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6070 1) := by change ¬ Nat.Prime 147391741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21055963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6070 2) := by change Nat.Prime 147379601; exact ProjectEulerSolutions.P58PrimeNodes.p_147379601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6070 3) := by change ¬ Nat.Prime 147367461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49122487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6070 : row 6070 = 1 := (row_6070.checked).trans rfl
theorem model_count_6070 : modelRow 6070 = 1 := by decide
theorem modelrow_agrees_6070 : modelRow 6070 = row 6070 := model_count_6070.trans row_count_6070.symm
def row_6071 : RowCertificate 6071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6071 1) := by change ¬ Nat.Prime 147440307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49146769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6071 2) := by change ¬ Nat.Prime 147428165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29485633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6071 3) := by change ¬ Nat.Prime 147416023; exact PrattCertificates.composite_of_product (factor := 2719) (cofactor := 54217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6071 : row 6071 = 0 := (row_6071.checked).trans rfl
theorem model_count_6071 : modelRow 6071 = 0 := by decide
theorem modelrow_agrees_6071 : modelRow 6071 = row 6071 := model_count_6071.trans row_count_6071.symm
def row_6072 : RowCertificate 6072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6072 1) := by change ¬ Nat.Prime 147488881; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1353109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6072 2) := by change Nat.Prime 147476737; exact ProjectEulerSolutions.P58PrimeNodes.p_147476737.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6072 3) := by change Nat.Prime 147464593; exact ProjectEulerSolutions.P58PrimeNodes.p_147464593.prime
    exact decide_eq_true prime
theorem row_count_6072 : row 6072 = 2 := (row_6072.checked).trans rfl
theorem model_count_6072 : modelRow 6072 = 2 := by decide
theorem modelrow_agrees_6072 : modelRow 6072 = row 6072 := model_count_6072.trans row_count_6072.symm
def row_6073 : RowCertificate 6073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6073 1) := by change ¬ Nat.Prime 147537463; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 4759273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6073 2) := by change ¬ Nat.Prime 147525317; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 815057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6073 3) := by change ¬ Nat.Prime 147513171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49171057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6073 : row 6073 = 0 := (row_6073.checked).trans rfl
theorem model_count_6073 : modelRow 6073 = 0 := by decide
theorem modelrow_agrees_6073 : modelRow 6073 = row 6073 := model_count_6073.trans row_count_6073.symm
def row_6074 : RowCertificate 6074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6074 1) := by change ¬ Nat.Prime 147586053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49195351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6074 2) := by change ¬ Nat.Prime 147573905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29514781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6074 3) := by change ¬ Nat.Prime 147561757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21080251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6074 : row 6074 = 0 := (row_6074.checked).trans rfl
theorem model_count_6074 : modelRow 6074 = 0 := by decide
theorem modelrow_agrees_6074 : modelRow 6074 = row 6074 := model_count_6074.trans row_count_6074.symm
def row_6075 : RowCertificate 6075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6075 1) := by change ¬ Nat.Prime 147634651; exact PrattCertificates.composite_of_product (factor := 4729) (cofactor := 31219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6075 2) := by change ¬ Nat.Prime 147622501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11355577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6075 3) := by change ¬ Nat.Prime 147610351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21087193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6075 : row 6075 = 0 := (row_6075.checked).trans rfl
theorem model_count_6075 : modelRow 6075 = 0 := by decide
theorem modelrow_agrees_6075 : modelRow 6075 = row 6075 := model_count_6075.trans row_count_6075.symm
def row_6076 : RowCertificate 6076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6076 1) := by change ¬ Nat.Prime 147683257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7772803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6076 2) := by change ¬ Nat.Prime 147671105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29534221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6076 3) := by change ¬ Nat.Prime 147658953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49219651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6076 : row 6076 = 0 := (row_6076.checked).trans rfl
theorem model_count_6076 : modelRow 6076 = 0 := by decide
theorem modelrow_agrees_6076 : modelRow 6076 = row 6076 := model_count_6076.trans row_count_6076.symm
def row_6077 : RowCertificate 6077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6077 1) := by change ¬ Nat.Prime 147731871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49243957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6077 2) := by change Nat.Prime 147719717; exact ProjectEulerSolutions.P58PrimeNodes.p_147719717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6077 3) := by change ¬ Nat.Prime 147707563; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 201511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6077 : row 6077 = 1 := (row_6077.checked).trans rfl
theorem model_count_6077 : modelRow 6077 = 1 := by decide
theorem modelrow_agrees_6077 : modelRow 6077 = row 6077 := model_count_6077.trans row_count_6077.symm
def row_6078 : RowCertificate 6078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6078 1) := by change ¬ Nat.Prime 147780493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21111499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6078 2) := by change ¬ Nat.Prime 147768337; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 1078601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6078 3) := by change ¬ Nat.Prime 147756181; exact PrattCertificates.composite_of_product (factor := 2707) (cofactor := 54583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6078 : row 6078 = 0 := (row_6078.checked).trans rfl
theorem model_count_6078 : modelRow 6078 = 0 := by decide
theorem modelrow_agrees_6078 : modelRow 6078 = row 6078 := model_count_6078.trans row_count_6078.symm
def row_6079 : RowCertificate 6079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6079 1) := by change ¬ Nat.Prime 147829123; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11371471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6079 2) := by change ¬ Nat.Prime 147816965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29563393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6079 3) := by change ¬ Nat.Prime 147804807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49268269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6079 : row 6079 = 0 := (row_6079.checked).trans rfl
theorem model_count_6079 : modelRow 6079 = 0 := by decide
theorem modelrow_agrees_6079 : modelRow 6079 = row 6079 := model_count_6079.trans row_count_6079.symm
def row_6080 : RowCertificate 6080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6080 1) := by change ¬ Nat.Prime 147877761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49292587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6080 2) := by change ¬ Nat.Prime 147865601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11374277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6080 3) := by change Nat.Prime 147853441; exact ProjectEulerSolutions.P58PrimeNodes.p_147853441.prime
    exact decide_eq_true prime
theorem row_count_6080 : row 6080 = 1 := (row_6080.checked).trans rfl
theorem model_count_6080 : modelRow 6080 = 1 := by decide
theorem modelrow_agrees_6080 : modelRow 6080 = row 6080 := model_count_6080.trans row_count_6080.symm
def row_6081 : RowCertificate 6081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6081 1) := by change ¬ Nat.Prime 147926407; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 3998011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6081 2) := by change ¬ Nat.Prime 147914245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29582849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6081 3) := by change ¬ Nat.Prime 147902083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21128869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6081 : row 6081 = 0 := (row_6081.checked).trans rfl
theorem model_count_6081 : modelRow 6081 = 0 := by decide
theorem modelrow_agrees_6081 : modelRow 6081 = row 6081 := model_count_6081.trans row_count_6081.symm
def row_6082 : RowCertificate 6082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6082 1) := by change ¬ Nat.Prime 147975061; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11382697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6082 2) := by change ¬ Nat.Prime 147962897; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 2026889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6082 3) := by change ¬ Nat.Prime 147950733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49316911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6082 : row 6082 = 0 := (row_6082.checked).trans rfl
theorem model_count_6082 : modelRow 6082 = 0 := by decide
theorem modelrow_agrees_6082 : modelRow 6082 = row 6082 := model_count_6082.trans row_count_6082.symm
def row_6083 : RowCertificate 6083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6083 1) := by change ¬ Nat.Prime 148023723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49341241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6083 2) := by change ¬ Nat.Prime 148011557; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 92681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6083 3) := by change Nat.Prime 147999391; exact ProjectEulerSolutions.P58PrimeNodes.p_147999391.prime
    exact decide_eq_true prime
theorem row_count_6083 : row 6083 = 1 := (row_6083.checked).trans rfl
theorem model_count_6083 : modelRow 6083 = 1 := by decide
theorem modelrow_agrees_6083 : modelRow 6083 = row 6083 := model_count_6083.trans row_count_6083.symm
def row_6084 : RowCertificate 6084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6084 1) := by change ¬ Nat.Prime 148072393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21153199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6084 2) := by change ¬ Nat.Prime 148060225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29612045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6084 3) := by change ¬ Nat.Prime 148048057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7792003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6084 : row 6084 = 0 := (row_6084.checked).trans rfl
theorem model_count_6084 : modelRow 6084 = 0 := by decide
theorem modelrow_agrees_6084 : modelRow 6084 = row 6084 := model_count_6084.trans row_count_6084.symm
def row_6085 : RowCertificate 6085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6085 1) := by change ¬ Nat.Prime 148121071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21160153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6085 2) := by change ¬ Nat.Prime 148108901; exact PrattCertificates.composite_of_product (factor := 797) (cofactor := 185833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6085 3) := by change ¬ Nat.Prime 148096731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49365577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6085 : row 6085 = 0 := (row_6085.checked).trans rfl
theorem model_count_6085 : modelRow 6085 = 0 := by decide
theorem modelrow_agrees_6085 : modelRow 6085 = row 6085 := model_count_6085.trans row_count_6085.symm
def row_6086 : RowCertificate 6086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6086 1) := by change ¬ Nat.Prime 148169757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49389919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6086 2) := by change ¬ Nat.Prime 148157585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29631517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6086 3) := by change ¬ Nat.Prime 148145413; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11395801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6086 : row 6086 = 0 := (row_6086.checked).trans rfl
theorem model_count_6086 : modelRow 6086 = 0 := by decide
theorem modelrow_agrees_6086 : modelRow 6086 = row 6086 := model_count_6086.trans row_count_6086.symm
def row_6087 : RowCertificate 6087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6087 1) := by change ¬ Nat.Prime 148218451; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 135607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6087 2) := by change ¬ Nat.Prime 148206277; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 994673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6087 3) := by change Nat.Prime 148194103; exact ProjectEulerSolutions.P58PrimeNodes.p_148194103.prime
    exact decide_eq_true prime
theorem row_count_6087 : row 6087 = 1 := (row_6087.checked).trans rfl
theorem model_count_6087 : modelRow 6087 = 1 := by decide
theorem modelrow_agrees_6087 : modelRow 6087 = row 6087 := model_count_6087.trans row_count_6087.symm
def row_6088 : RowCertificate 6088 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6088 1) := by change Nat.Prime 148267153; exact ProjectEulerSolutions.P58PrimeNodes.p_148267153.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6088 2) := by change ¬ Nat.Prime 148254977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11404229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6088 3) := by change ¬ Nat.Prime 148242801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49414267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6088 : row 6088 = 1 := (row_6088.checked).trans rfl
theorem model_count_6088 : modelRow 6088 = 1 := by decide
theorem modelrow_agrees_6088 : modelRow 6088 = row 6088 := model_count_6088.trans row_count_6088.symm
def row_6089 : RowCertificate 6089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6089 1) := by change ¬ Nat.Prime 148315863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49438621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6089 2) := by change ¬ Nat.Prime 148303685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29660737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6089 3) := by change ¬ Nat.Prime 148291507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21184501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6089 : row 6089 = 0 := (row_6089.checked).trans rfl
theorem model_count_6089 : modelRow 6089 = 0 := by decide
theorem modelrow_agrees_6089 : modelRow 6089 = row 6089 := model_count_6089.trans row_count_6089.symm
def row_6090 : RowCertificate 6090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6090 1) := by change ¬ Nat.Prime 148364581; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 128677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6090 2) := by change ¬ Nat.Prime 148352401; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 352381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6090 3) := by change ¬ Nat.Prime 148340221; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 391399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6090 : row 6090 = 0 := (row_6090.checked).trans rfl
theorem model_count_6090 : modelRow 6090 = 0 := by decide
theorem modelrow_agrees_6090 : modelRow 6090 = row 6090 := model_count_6090.trans row_count_6090.symm
def row_6091 : RowCertificate 6091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6091 1) := by change ¬ Nat.Prime 148413307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21201901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6091 2) := by change ¬ Nat.Prime 148401125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29680225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6091 3) := by change ¬ Nat.Prime 148388943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49462981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6091 : row 6091 = 0 := (row_6091.checked).trans rfl
theorem model_count_6091 : modelRow 6091 = 0 := by decide
theorem modelrow_agrees_6091 : modelRow 6091 = row 6091 := model_count_6091.trans row_count_6091.symm
def row_6092 : RowCertificate 6092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6092 1) := by change ¬ Nat.Prime 148462041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49487347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6092 2) := by change Nat.Prime 148449857; exact ProjectEulerSolutions.P58PrimeNodes.p_148449857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6092 3) := by change ¬ Nat.Prime 148437673; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4011829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6092 : row 6092 = 1 := (row_6092.checked).trans rfl
theorem model_count_6092 : modelRow 6092 = 1 := by decide
theorem modelrow_agrees_6092 : modelRow 6092 = row 6092 := model_count_6092.trans row_count_6092.symm
def row_6093 : RowCertificate 6093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6093 1) := by change ¬ Nat.Prime 148510783; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7816357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6093 2) := by change ¬ Nat.Prime 148498597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11422969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6093 3) := by change Nat.Prime 148486411; exact ProjectEulerSolutions.P58PrimeNodes.p_148486411.prime
    exact decide_eq_true prime
theorem row_count_6093 : row 6093 = 1 := (row_6093.checked).trans rfl
theorem model_count_6093 : modelRow 6093 = 1 := by decide
theorem modelrow_agrees_6093 : modelRow 6093 = row 6093 := model_count_6093.trans row_count_6093.symm
def row_6094 : RowCertificate 6094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6094 1) := by change ¬ Nat.Prime 148559533; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 4792243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6094 2) := by change ¬ Nat.Prime 148547345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29709469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6094 3) := by change ¬ Nat.Prime 148535157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49511719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6094 : row 6094 = 0 := (row_6094.checked).trans rfl
theorem model_count_6094 : modelRow 6094 = 0 := by decide
theorem modelrow_agrees_6094 : modelRow 6094 = row 6094 := model_count_6094.trans row_count_6094.symm
def row_6095 : RowCertificate 6095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6095 1) := by change ¬ Nat.Prime 148608291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49536097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6095 2) := by change ¬ Nat.Prime 148596101; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 858937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6095 3) := by change ¬ Nat.Prime 148583911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21226273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6095 : row 6095 = 0 := (row_6095.checked).trans rfl
theorem model_count_6095 : modelRow 6095 = 0 := by decide
theorem modelrow_agrees_6095 : modelRow 6095 = row 6095 := model_count_6095.trans row_count_6095.symm
def row_6096 : RowCertificate 6096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6096 1) := by change ¬ Nat.Prime 148657057; exact PrattCertificates.composite_of_product (factor := 3469) (cofactor := 42853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6096 2) := by change ¬ Nat.Prime 148644865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29728973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6096 3) := by change ¬ Nat.Prime 148632673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21233239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6096 : row 6096 = 0 := (row_6096.checked).trans rfl
theorem model_count_6096 : modelRow 6096 = 0 := by decide
theorem modelrow_agrees_6096 : modelRow 6096 = row 6096 := model_count_6096.trans row_count_6096.symm
def row_6097 : RowCertificate 6097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6097 1) := by change ¬ Nat.Prime 148705831; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 405193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6097 2) := by change Nat.Prime 148693637; exact ProjectEulerSolutions.P58PrimeNodes.p_148693637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6097 3) := by change ¬ Nat.Prime 148681443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49560481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6097 : row 6097 = 1 := (row_6097.checked).trans rfl
theorem model_count_6097 : modelRow 6097 = 1 := by decide
theorem modelrow_agrees_6097 : modelRow 6097 = row 6097 := model_count_6097.trans row_count_6097.symm
def row_6098 : RowCertificate 6098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6098 1) := by change ¬ Nat.Prime 148754613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49584871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6098 2) := by change ¬ Nat.Prime 148742417; exact PrattCertificates.composite_of_product (factor := 1933) (cofactor := 76949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6098 3) := by change ¬ Nat.Prime 148730221; exact PrattCertificates.composite_of_product (factor := 9043) (cofactor := 16447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6098 : row 6098 = 0 := (row_6098.checked).trans rfl
theorem model_count_6098 : modelRow 6098 = 0 := by decide
theorem modelrow_agrees_6098 : modelRow 6098 = row 6098 := model_count_6098.trans row_count_6098.symm
def row_6099 : RowCertificate 6099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6099 1) := by change ¬ Nat.Prime 148803403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21257629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6099 2) := by change ¬ Nat.Prime 148791205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29758241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6099 3) := by change ¬ Nat.Prime 148779007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11444539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6099 : row 6099 = 0 := (row_6099.checked).trans rfl
theorem model_count_6099 : modelRow 6099 = 0 := by decide
theorem modelrow_agrees_6099 : modelRow 6099 = row 6099 := model_count_6099.trans row_count_6099.symm
def row_6100 : RowCertificate 6100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6100 1) := by change ¬ Nat.Prime 148852201; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1445167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6100 2) := by change ¬ Nat.Prime 148840001; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 1534433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6100 3) := by change ¬ Nat.Prime 148827801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49609267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6100 : row 6100 = 0 := (row_6100.checked).trans rfl
theorem model_count_6100 : modelRow 6100 = 0 := by decide
theorem modelrow_agrees_6100 : modelRow 6100 = row 6100 := model_count_6100.trans row_count_6100.symm
def row_6101 : RowCertificate 6101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6101 1) := by change ¬ Nat.Prime 148901007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49633669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6101 2) := by change ¬ Nat.Prime 148888805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29777761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6101 3) := by change Nat.Prime 148876603; exact ProjectEulerSolutions.P58PrimeNodes.p_148876603.prime
    exact decide_eq_true prime
theorem row_count_6101 : row 6101 = 1 := (row_6101.checked).trans rfl
theorem model_count_6101 : modelRow 6101 = 1 := by decide
theorem modelrow_agrees_6101 : modelRow 6101 = row 6101 := model_count_6101.trans row_count_6101.symm
def row_6102 : RowCertificate 6102 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6102 1) := by change Nat.Prime 148949821; exact ProjectEulerSolutions.P58PrimeNodes.p_148949821.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6102 2) := by change ¬ Nat.Prime 148937617; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4025341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6102 3) := by change ¬ Nat.Prime 148925413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21275059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6102 : row 6102 = 1 := (row_6102.checked).trans rfl
theorem model_count_6102 : modelRow 6102 = 1 := by decide
theorem modelrow_agrees_6102 : modelRow 6102 = row 6102 := model_count_6102.trans row_count_6102.symm
def row_6103 : RowCertificate 6103 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6103 1) := by change Nat.Prime 148998643; exact ProjectEulerSolutions.P58PrimeNodes.p_148998643.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6103 2) := by change ¬ Nat.Prime 148986437; exact PrattCertificates.composite_of_product (factor := 2237) (cofactor := 66601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6103 3) := by change ¬ Nat.Prime 148974231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49658077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6103 : row 6103 = 1 := (row_6103.checked).trans rfl
theorem model_count_6103 : modelRow 6103 = 1 := by decide
theorem modelrow_agrees_6103 : modelRow 6103 = row 6103 := model_count_6103.trans row_count_6103.symm
def row_6104 : RowCertificate 6104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6104 1) := by change ¬ Nat.Prime 149047473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49682491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6104 2) := by change ¬ Nat.Prime 149035265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29807053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6104 3) := by change ¬ Nat.Prime 149023057; exact PrattCertificates.composite_of_product (factor := 2707) (cofactor := 55051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6104 : row 6104 = 0 := (row_6104.checked).trans rfl
theorem model_count_6104 : modelRow 6104 = 0 := by decide
theorem modelrow_agrees_6104 : modelRow 6104 = row 6104 := model_count_6104.trans row_count_6104.symm
def row_6105 : RowCertificate 6105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6105 1) := by change ¬ Nat.Prime 149096311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21299473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6105 2) := by change ¬ Nat.Prime 149084101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8769653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6105 3) := by change ¬ Nat.Prime 149071891; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7845889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6105 : row 6105 = 0 := (row_6105.checked).trans rfl
theorem model_count_6105 : modelRow 6105 = 0 := by decide
theorem modelrow_agrees_6105 : modelRow 6105 = row 6105 := model_count_6105.trans row_count_6105.symm
def row_6106 : RowCertificate 6106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6106 1) := by change ¬ Nat.Prime 149145157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21306451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6106 2) := by change ¬ Nat.Prime 149132945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29826589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6106 3) := by change ¬ Nat.Prime 149120733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49706911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6106 : row 6106 = 0 := (row_6106.checked).trans rfl
theorem model_count_6106 : modelRow 6106 = 0 := by decide
theorem modelrow_agrees_6106 : modelRow 6106 = row 6106 := model_count_6106.trans row_count_6106.symm
def row_6107 : RowCertificate 6107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6107 1) := by change ¬ Nat.Prime 149194011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49731337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6107 2) := by change Nat.Prime 149181797; exact ProjectEulerSolutions.P58PrimeNodes.p_149181797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6107 3) := by change ¬ Nat.Prime 149169583; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 2445403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6107 : row 6107 = 1 := (row_6107.checked).trans rfl
theorem model_count_6107 : modelRow 6107 = 1 := by decide
theorem modelrow_agrees_6107 : modelRow 6107 = row 6107 := model_count_6107.trans row_count_6107.symm
def row_6108 : RowCertificate 6108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6108 1) := by change ¬ Nat.Prime 149242873; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11480221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6108 2) := by change ¬ Nat.Prime 149230657; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4033261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6108 3) := by change ¬ Nat.Prime 149218441; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 450811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6108 : row 6108 = 0 := (row_6108.checked).trans rfl
theorem model_count_6108 : modelRow 6108 = 0 := by decide
theorem modelrow_agrees_6108 : modelRow 6108 = row 6108 := model_count_6108.trans row_count_6108.symm
def row_6109 : RowCertificate 6109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6109 1) := by change ¬ Nat.Prime 149291743; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 3471901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6109 2) := by change ¬ Nat.Prime 149279525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29855905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6109 3) := by change ¬ Nat.Prime 149267307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49755769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6109 : row 6109 = 0 := (row_6109.checked).trans rfl
theorem model_count_6109 : modelRow 6109 = 0 := by decide
theorem modelrow_agrees_6109 : modelRow 6109 = row 6109 := model_count_6109.trans row_count_6109.symm
def row_6110 : RowCertificate 6110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6110 1) := by change ¬ Nat.Prime 149340621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49780207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6110 2) := by change ¬ Nat.Prime 149328401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 2817517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6110 3) := by change ¬ Nat.Prime 149316181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21330883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6110 : row 6110 = 0 := (row_6110.checked).trans rfl
theorem model_count_6110 : modelRow 6110 = 0 := by decide
theorem modelrow_agrees_6110 : modelRow 6110 = row 6110 := model_count_6110.trans row_count_6110.symm
def row_6111 : RowCertificate 6111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6111 1) := by change ¬ Nat.Prime 149389507; exact PrattCertificates.composite_of_product (factor := 4903) (cofactor := 30469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6111 2) := by change ¬ Nat.Prime 149377285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29875457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6111 3) := by change ¬ Nat.Prime 149365063; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 1890697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6111 : row 6111 = 0 := (row_6111.checked).trans rfl
theorem model_count_6111 : modelRow 6111 = 0 := by decide
theorem modelrow_agrees_6111 : modelRow 6111 = row 6111 := model_count_6111.trans row_count_6111.symm
def row_6112 : RowCertificate 6112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6112 1) := by change ¬ Nat.Prime 149438401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21348343) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6112 2) := by change Nat.Prime 149426177; exact ProjectEulerSolutions.P58PrimeNodes.p_149426177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6112 3) := by change ¬ Nat.Prime 149413953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49804651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6112 : row 6112 = 1 := (row_6112.checked).trans rfl
theorem model_count_6112 : modelRow 6112 = 1 := by decide
theorem modelrow_agrees_6112 : modelRow 6112 = row 6112 := model_count_6112.trans row_count_6112.symm
def row_6113 : RowCertificate 6113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6113 1) := by change ¬ Nat.Prime 149487303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49829101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6113 2) := by change ¬ Nat.Prime 149475077; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 5154313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6113 3) := by change ¬ Nat.Prime 149462851; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 670237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6113 : row 6113 = 0 := (row_6113.checked).trans rfl
theorem model_count_6113 : modelRow 6113 = 0 := by decide
theorem modelrow_agrees_6113 : modelRow 6113 = row 6113 := model_count_6113.trans row_count_6113.symm
def row_6114 : RowCertificate 6114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6114 1) := by change ¬ Nat.Prime 149536213; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7870327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6114 2) := by change ¬ Nat.Prime 149523985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29904797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6114 3) := by change Nat.Prime 149511757; exact ProjectEulerSolutions.P58PrimeNodes.p_149511757.prime
    exact decide_eq_true prime
theorem row_count_6114 : row 6114 = 1 := (row_6114.checked).trans rfl
theorem model_count_6114 : modelRow 6114 = 1 := by decide
theorem modelrow_agrees_6114 : modelRow 6114 = row 6114 := model_count_6114.trans row_count_6114.symm
def row_6115 : RowCertificate 6115 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6115 1) := by change Nat.Prime 149585131; exact ProjectEulerSolutions.P58PrimeNodes.p_149585131.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6115 2) := by change ¬ Nat.Prime 149572901; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 1091773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6115 3) := by change ¬ Nat.Prime 149560671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49853557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6115 : row 6115 = 1 := (row_6115.checked).trans rfl
theorem model_count_6115 : modelRow 6115 = 1 := by decide
theorem modelrow_agrees_6115 : modelRow 6115 = row 6115 := model_count_6115.trans row_count_6115.symm
def row_6116 : RowCertificate 6116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6116 1) := by change ¬ Nat.Prime 149634057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49878019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6116 2) := by change ¬ Nat.Prime 149621825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29924365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6116 3) := by change ¬ Nat.Prime 149609593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21372799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6116 : row 6116 = 0 := (row_6116.checked).trans rfl
theorem model_count_6116 : modelRow 6116 = 0 := by decide
theorem modelrow_agrees_6116 : modelRow 6116 = row 6116 := model_count_6116.trans row_count_6116.symm
def row_6117 : RowCertificate 6117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6117 1) := by change ¬ Nat.Prime 149682991; exact PrattCertificates.composite_of_product (factor := 1867) (cofactor := 80173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6117 2) := by change Nat.Prime 149670757; exact ProjectEulerSolutions.P58PrimeNodes.p_149670757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6117 3) := by change ¬ Nat.Prime 149658523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21379789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6117 : row 6117 = 1 := (row_6117.checked).trans rfl
theorem model_count_6117 : modelRow 6117 = 1 := by decide
theorem modelrow_agrees_6117 : modelRow 6117 = row 6117 := model_count_6117.trans row_count_6117.symm
def row_6118 : RowCertificate 6118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6118 1) := by change ¬ Nat.Prime 149731933; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11517841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6118 2) := by change ¬ Nat.Prime 149719697; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8807041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6118 3) := by change ¬ Nat.Prime 149707461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49902487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6118 : row 6118 = 0 := (row_6118.checked).trans rfl
theorem model_count_6118 : modelRow 6118 = 0 := by decide
theorem modelrow_agrees_6118 : modelRow 6118 = row 6118 := model_count_6118.trans row_count_6118.symm
def row_6119 : RowCertificate 6119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6119 1) := by change ¬ Nat.Prime 149780883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49926961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6119 2) := by change ¬ Nat.Prime 149768645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29953729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6119 3) := by change Nat.Prime 149756407; exact ProjectEulerSolutions.P58PrimeNodes.p_149756407.prime
    exact decide_eq_true prime
theorem row_count_6119 : row 6119 = 1 := (row_6119.checked).trans rfl
theorem model_count_6119 : modelRow 6119 = 1 := by decide
theorem modelrow_agrees_6119 : modelRow 6119 = row 6119 := model_count_6119.trans row_count_6119.symm
def row_6120 : RowCertificate 6120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6120 1) := by change ¬ Nat.Prime 149829841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21404263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6120 2) := by change ¬ Nat.Prime 149817601; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 776257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6120 3) := by change ¬ Nat.Prime 149805361; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 4832431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6120 : row 6120 = 0 := (row_6120.checked).trans rfl
theorem model_count_6120 : modelRow 6120 = 0 := by decide
theorem modelrow_agrees_6120 : modelRow 6120 = row 6120 := model_count_6120.trans row_count_6120.symm
def row_6121 : RowCertificate 6121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6121 1) := by change ¬ Nat.Prime 149878807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11529139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6121 2) := by change ¬ Nat.Prime 149866565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 29973313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6121 3) := by change ¬ Nat.Prime 149854323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49951441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6121 : row 6121 = 0 := (row_6121.checked).trans rfl
theorem model_count_6121 : modelRow 6121 = 0 := by decide
theorem modelrow_agrees_6121 : modelRow 6121 = row 6121 := model_count_6121.trans row_count_6121.symm
def row_6122 : RowCertificate 6122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6122 1) := by change ¬ Nat.Prime 149927781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 49975927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6122 2) := by change ¬ Nat.Prime 149915537; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8818561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6122 3) := by change ¬ Nat.Prime 149903293; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7889647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6122 : row 6122 = 0 := (row_6122.checked).trans rfl
theorem model_count_6122 : modelRow 6122 = 0 := by decide
theorem modelrow_agrees_6122 : modelRow 6122 = row 6122 := model_count_6122.trans row_count_6122.symm
def row_6123 : RowCertificate 6123 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6123 1) := by change ¬ Nat.Prime 149976763; exact PrattCertificates.composite_of_product (factor := 2887) (cofactor := 51949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6123 2) := by change ¬ Nat.Prime 149964517; exact PrattCertificates.composite_of_product (factor := 1741) (cofactor := 86137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6123 3) := by change ¬ Nat.Prime 149952271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21421753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6123 : row 6123 = 0 := (row_6123.checked).trans rfl
theorem model_count_6123 : modelRow 6123 = 0 := by decide
theorem modelrow_agrees_6123 : modelRow 6123 = row 6123 := model_count_6123.trans row_count_6123.symm
def row_6124 : RowCertificate 6124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6124 1) := by change ¬ Nat.Prime 150025753; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 3488971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6124 2) := by change ¬ Nat.Prime 150013505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30002701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6124 3) := by change ¬ Nat.Prime 150001257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50000419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6124 : row 6124 = 0 := (row_6124.checked).trans rfl
theorem model_count_6124 : modelRow 6124 = 0 := by decide
theorem modelrow_agrees_6124 : modelRow 6124 = row 6124 := model_count_6124.trans row_count_6124.symm
def row_6125 : RowCertificate 6125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6125 1) := by change ¬ Nat.Prime 150074751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50024917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6125 2) := by change ¬ Nat.Prime 150062501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 5174569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6125 3) := by change ¬ Nat.Prime 150050251; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11542327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6125 : row 6125 = 0 := (row_6125.checked).trans rfl
theorem model_count_6125 : modelRow 6125 = 0 := by decide
theorem modelrow_agrees_6125 : modelRow 6125 = row 6125 := model_count_6125.trans row_count_6125.symm
def row_6126 : RowCertificate 6126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6126 1) := by change ¬ Nat.Prime 150123757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21446251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6126 2) := by change ¬ Nat.Prime 150111505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30022301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6126 3) := by change Nat.Prime 150099253; exact ProjectEulerSolutions.P58PrimeNodes.p_150099253.prime
    exact decide_eq_true prime
theorem row_count_6126 : row 6126 = 1 := (row_6126.checked).trans rfl
theorem model_count_6126 : modelRow 6126 = 1 := by decide
theorem modelrow_agrees_6126 : modelRow 6126 = row 6126 := model_count_6126.trans row_count_6126.symm
def row_6127 : RowCertificate 6127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6127 1) := by change ¬ Nat.Prime 150172771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21453253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6127 2) := by change ¬ Nat.Prime 150160517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11550809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6127 3) := by change ¬ Nat.Prime 150148263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50049421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6127 : row 6127 = 0 := (row_6127.checked).trans rfl
theorem model_count_6127 : modelRow 6127 = 0 := by decide
theorem modelrow_agrees_6127 : modelRow 6127 = row 6127 := model_count_6127.trans row_count_6127.symm
def row_6128 : RowCertificate 6128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6128 1) := by change ¬ Nat.Prime 150221793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50073931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6128 2) := by change Nat.Prime 150209537; exact ProjectEulerSolutions.P58PrimeNodes.p_150209537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6128 3) := by change ¬ Nat.Prime 150197281; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11553637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6128 : row 6128 = 1 := (row_6128.checked).trans rfl
theorem model_count_6128 : modelRow 6128 = 1 := by decide
theorem modelrow_agrees_6128 : modelRow 6128 = row 6128 := model_count_6128.trans row_count_6128.symm
def row_6129 : RowCertificate 6129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6129 1) := by change ¬ Nat.Prime 150270823; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 957139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6129 2) := by change ¬ Nat.Prime 150258565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30051713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6129 3) := by change ¬ Nat.Prime 150246307; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4060711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6129 : row 6129 = 0 := (row_6129.checked).trans rfl
theorem model_count_6129 : modelRow 6129 = 0 := by decide
theorem modelrow_agrees_6129 : modelRow 6129 = row 6129 := model_count_6129.trans row_count_6129.symm
def row_6130 : RowCertificate 6130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6130 1) := by change ¬ Nat.Prime 150319861; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 446053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6130 2) := by change ¬ Nat.Prime 150307601; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 645097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6130 3) := by change ¬ Nat.Prime 150295341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50098447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6130 : row 6130 = 0 := (row_6130.checked).trans rfl
theorem model_count_6130 : modelRow 6130 = 0 := by decide
theorem modelrow_agrees_6130 : modelRow 6130 = row 6130 := model_count_6130.trans row_count_6130.symm
def row_6131 : RowCertificate 6131 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6131 1) := by change ¬ Nat.Prime 150368907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50122969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6131 2) := by change ¬ Nat.Prime 150356645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30071329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6131 3) := by change ¬ Nat.Prime 150344383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21477769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6131 : row 6131 = 0 := (row_6131.checked).trans rfl
theorem model_count_6131 : modelRow 6131 = 0 := by decide
theorem modelrow_agrees_6131 : modelRow 6131 = row 6131 := model_count_6131.trans row_count_6131.symm
def row_6132 : RowCertificate 6132 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6132 1) := by change Nat.Prime 150417961; exact ProjectEulerSolutions.P58PrimeNodes.p_150417961.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6132 2) := by change ¬ Nat.Prime 150405697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11569669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6132 3) := by change ¬ Nat.Prime 150393433; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 995983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6132 : row 6132 = 1 := (row_6132.checked).trans rfl
theorem model_count_6132 : modelRow 6132 = 1 := by decide
theorem modelrow_agrees_6132 : modelRow 6132 = row 6132 := model_count_6132.trans row_count_6132.symm
def row_6133 : RowCertificate 6133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6133 1) := by change ¬ Nat.Prime 150467023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21495289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6133 2) := by change ¬ Nat.Prime 150454757; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 2838769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6133 3) := by change ¬ Nat.Prime 150442491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50147497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6133 : row 6133 = 0 := (row_6133.checked).trans rfl
theorem model_count_6133 : modelRow 6133 = 0 := by decide
theorem modelrow_agrees_6133 : modelRow 6133 = row 6133 := model_count_6133.trans row_count_6133.symm
def row_6134 : RowCertificate 6134 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6134 1) := by change ¬ Nat.Prime 150516093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50172031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6134 2) := by change ¬ Nat.Prime 150503825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30100765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6134 3) := by change ¬ Nat.Prime 150491557; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 779749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6134 : row 6134 = 0 := (row_6134.checked).trans rfl
theorem model_count_6134 : modelRow 6134 = 0 := by decide
theorem modelrow_agrees_6134 : modelRow 6134 = row 6134 := model_count_6134.trans row_count_6134.symm
def row_6135 : RowCertificate 6135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6135 1) := by change ¬ Nat.Prime 150565171; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 4856941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6135 2) := by change ¬ Nat.Prime 150552901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 8856053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6135 3) := by change Nat.Prime 150540631; exact ProjectEulerSolutions.P58PrimeNodes.p_150540631.prime
    exact decide_eq_true prime
theorem row_count_6135 : row 6135 = 1 := (row_6135.checked).trans rfl
theorem model_count_6135 : modelRow 6135 = 1 := by decide
theorem modelrow_agrees_6135 : modelRow 6135 = row 6135 := model_count_6135.trans row_count_6135.symm
def row_6136 : RowCertificate 6136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6136 1) := by change ¬ Nat.Prime 150614257; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 2063209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6136 2) := by change ¬ Nat.Prime 150601985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30120397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6136 3) := by change ¬ Nat.Prime 150589713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50196571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6136 : row 6136 = 0 := (row_6136.checked).trans rfl
theorem model_count_6136 : modelRow 6136 = 0 := by decide
theorem modelrow_agrees_6136 : modelRow 6136 = row 6136 := model_count_6136.trans row_count_6136.symm
def row_6137 : RowCertificate 6137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6137 1) := by change ¬ Nat.Prime 150663351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50221117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6137 2) := by change ¬ Nat.Prime 150651077; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 959561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6137 3) := by change ¬ Nat.Prime 150638803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21519829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6137 : row 6137 = 0 := (row_6137.checked).trans rfl
theorem model_count_6137 : modelRow 6137 = 0 := by decide
theorem modelrow_agrees_6137 : modelRow 6137 = row 6137 := model_count_6137.trans row_count_6137.symm
def row_6138 : RowCertificate 6138 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6138 1) := by change Nat.Prime 150712453; exact ProjectEulerSolutions.P58PrimeNodes.p_150712453.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6138 2) := by change Nat.Prime 150700177; exact ProjectEulerSolutions.P58PrimeNodes.p_150700177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6138 3) := by change ¬ Nat.Prime 150687901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21526843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6138 : row 6138 = 2 := (row_6138.checked).trans rfl
theorem model_count_6138 : modelRow 6138 = 2 := by decide
theorem modelrow_agrees_6138 : modelRow 6138 = row 6138 := model_count_6138.trans row_count_6138.symm
def row_6139 : RowCertificate 6139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6139 1) := by change ¬ Nat.Prime 150761563; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 455473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6139 2) := by change ¬ Nat.Prime 150749285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30149857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6139 3) := by change ¬ Nat.Prime 150737007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50245669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6139 : row 6139 = 0 := (row_6139.checked).trans rfl
theorem model_count_6139 : modelRow 6139 = 0 := by decide
theorem modelrow_agrees_6139 : modelRow 6139 = row 6139 := model_count_6139.trans row_count_6139.symm
def row_6140 : RowCertificate 6140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6140 1) := by change ¬ Nat.Prime 150810681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50270227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6140 2) := by change ¬ Nat.Prime 150798401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11599877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6140 3) := by change ¬ Nat.Prime 150786121; exact PrattCertificates.composite_of_product (factor := 5851) (cofactor := 25771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6140 : row 6140 = 0 := (row_6140.checked).trans rfl
theorem model_count_6140 : modelRow 6140 = 0 := by decide
theorem modelrow_agrees_6140 : modelRow 6140 = row 6140 := model_count_6140.trans row_count_6140.symm
def row_6141 : RowCertificate 6141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6141 1) := by change ¬ Nat.Prime 150859807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21551401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6141 2) := by change ¬ Nat.Prime 150847525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30169505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6141 3) := by change ¬ Nat.Prime 150835243; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11602711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6141 : row 6141 = 0 := (row_6141.checked).trans rfl
theorem model_count_6141 : modelRow 6141 = 0 := by decide
theorem modelrow_agrees_6141 : modelRow 6141 = row 6141 := model_count_6141.trans row_count_6141.symm
def row_6142 : RowCertificate 6142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6142 1) := by change ¬ Nat.Prime 150908941; exact PrattCertificates.composite_of_product (factor := 1693) (cofactor := 89137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6142 2) := by change ¬ Nat.Prime 150896657; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 5203333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6142 3) := by change ¬ Nat.Prime 150884373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50294791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6142 : row 6142 = 0 := (row_6142.checked).trans rfl
theorem model_count_6142 : modelRow 6142 = 0 := by decide
theorem modelrow_agrees_6142 : modelRow 6142 = row 6142 := model_count_6142.trans row_count_6142.symm
def row_6143 : RowCertificate 6143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6143 1) := by change ¬ Nat.Prime 150958083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 50319361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6143 2) := by change ¬ Nat.Prime 150945797; exact PrattCertificates.composite_of_product (factor := 9161) (cofactor := 16477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6143 3) := by change ¬ Nat.Prime 150933511; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 7943869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6143 : row 6143 = 0 := (row_6143.checked).trans rfl
theorem model_count_6143 : modelRow 6143 = 0 := by decide
theorem modelrow_agrees_6143 : modelRow 6143 = row 6143 := model_count_6143.trans row_count_6143.symm
def row_6144 : RowCertificate 6144 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6144 1) := by change ¬ Nat.Prime 151007233; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 11615941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6144 2) := by change ¬ Nat.Prime 150994945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 30198989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6144 3) := by change ¬ Nat.Prime 150982657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 21568951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6144 : row 6144 = 0 := (row_6144.checked).trans rfl
theorem model_count_6144 : modelRow 6144 = 0 := by decide
theorem modelrow_agrees_6144 : modelRow 6144 = row 6144 := model_count_6144.trans row_count_6144.symm
private def atomic_6017 : CountSegment 1 10 13120 6016 6017 2654 2654 := CountSegment.single (first := 6016) (initial := 2654) row_count_6017 (by decide)
private theorem atomic_6017_agreement : RowAgreement 6016 6017 := RowAgreement.single (first := 6016) modelrow_agrees_6017
private def atomic_6018 : CountSegment 1 10 13120 6017 6018 2654 2655 := CountSegment.single (first := 6017) (initial := 2654) row_count_6018 (by decide)
private theorem atomic_6018_agreement : RowAgreement 6017 6018 := RowAgreement.single (first := 6017) modelrow_agrees_6018
private def atomic_6019 : CountSegment 1 10 13120 6018 6019 2655 2655 := CountSegment.single (first := 6018) (initial := 2655) row_count_6019 (by decide)
private theorem atomic_6019_agreement : RowAgreement 6018 6019 := RowAgreement.single (first := 6018) modelrow_agrees_6019
private def atomic_6020 : CountSegment 1 10 13120 6019 6020 2655 2655 := CountSegment.single (first := 6019) (initial := 2655) row_count_6020 (by decide)
private theorem atomic_6020_agreement : RowAgreement 6019 6020 := RowAgreement.single (first := 6019) modelrow_agrees_6020
private def atomic_6021 : CountSegment 1 10 13120 6020 6021 2655 2655 := CountSegment.single (first := 6020) (initial := 2655) row_count_6021 (by decide)
private theorem atomic_6021_agreement : RowAgreement 6020 6021 := RowAgreement.single (first := 6020) modelrow_agrees_6021
private def atomic_6022 : CountSegment 1 10 13120 6021 6022 2655 2655 := CountSegment.single (first := 6021) (initial := 2655) row_count_6022 (by decide)
private theorem atomic_6022_agreement : RowAgreement 6021 6022 := RowAgreement.single (first := 6021) modelrow_agrees_6022
private def atomic_6023 : CountSegment 1 10 13120 6022 6023 2655 2655 := CountSegment.single (first := 6022) (initial := 2655) row_count_6023 (by decide)
private theorem atomic_6023_agreement : RowAgreement 6022 6023 := RowAgreement.single (first := 6022) modelrow_agrees_6023
private def atomic_6024 : CountSegment 1 10 13120 6023 6024 2655 2655 := CountSegment.single (first := 6023) (initial := 2655) row_count_6024 (by decide)
private theorem atomic_6024_agreement : RowAgreement 6023 6024 := RowAgreement.single (first := 6023) modelrow_agrees_6024
private def atomic_6025 : CountSegment 1 10 13120 6024 6025 2655 2655 := CountSegment.single (first := 6024) (initial := 2655) row_count_6025 (by decide)
private theorem atomic_6025_agreement : RowAgreement 6024 6025 := RowAgreement.single (first := 6024) modelrow_agrees_6025
private def atomic_6026 : CountSegment 1 10 13120 6025 6026 2655 2655 := CountSegment.single (first := 6025) (initial := 2655) row_count_6026 (by decide)
private theorem atomic_6026_agreement : RowAgreement 6025 6026 := RowAgreement.single (first := 6025) modelrow_agrees_6026
private def atomic_6027 : CountSegment 1 10 13120 6026 6027 2655 2655 := CountSegment.single (first := 6026) (initial := 2655) row_count_6027 (by decide)
private theorem atomic_6027_agreement : RowAgreement 6026 6027 := RowAgreement.single (first := 6026) modelrow_agrees_6027
private def atomic_6028 : CountSegment 1 10 13120 6027 6028 2655 2655 := CountSegment.single (first := 6027) (initial := 2655) row_count_6028 (by decide)
private theorem atomic_6028_agreement : RowAgreement 6027 6028 := RowAgreement.single (first := 6027) modelrow_agrees_6028
private def atomic_6029 : CountSegment 1 10 13120 6028 6029 2655 2655 := CountSegment.single (first := 6028) (initial := 2655) row_count_6029 (by decide)
private theorem atomic_6029_agreement : RowAgreement 6028 6029 := RowAgreement.single (first := 6028) modelrow_agrees_6029
private def atomic_6030 : CountSegment 1 10 13120 6029 6030 2655 2656 := CountSegment.single (first := 6029) (initial := 2655) row_count_6030 (by decide)
private theorem atomic_6030_agreement : RowAgreement 6029 6030 := RowAgreement.single (first := 6029) modelrow_agrees_6030
private def atomic_6031 : CountSegment 1 10 13120 6030 6031 2656 2656 := CountSegment.single (first := 6030) (initial := 2656) row_count_6031 (by decide)
private theorem atomic_6031_agreement : RowAgreement 6030 6031 := RowAgreement.single (first := 6030) modelrow_agrees_6031
private def atomic_6032 : CountSegment 1 10 13120 6031 6032 2656 2657 := CountSegment.single (first := 6031) (initial := 2656) row_count_6032 (by decide)
private theorem atomic_6032_agreement : RowAgreement 6031 6032 := RowAgreement.single (first := 6031) modelrow_agrees_6032
private def atomic_6033 : CountSegment 1 10 13120 6032 6033 2657 2658 := CountSegment.single (first := 6032) (initial := 2657) row_count_6033 (by decide)
private theorem atomic_6033_agreement : RowAgreement 6032 6033 := RowAgreement.single (first := 6032) modelrow_agrees_6033
private def atomic_6034 : CountSegment 1 10 13120 6033 6034 2658 2658 := CountSegment.single (first := 6033) (initial := 2658) row_count_6034 (by decide)
private theorem atomic_6034_agreement : RowAgreement 6033 6034 := RowAgreement.single (first := 6033) modelrow_agrees_6034
private def atomic_6035 : CountSegment 1 10 13120 6034 6035 2658 2658 := CountSegment.single (first := 6034) (initial := 2658) row_count_6035 (by decide)
private theorem atomic_6035_agreement : RowAgreement 6034 6035 := RowAgreement.single (first := 6034) modelrow_agrees_6035
private def atomic_6036 : CountSegment 1 10 13120 6035 6036 2658 2658 := CountSegment.single (first := 6035) (initial := 2658) row_count_6036 (by decide)
private theorem atomic_6036_agreement : RowAgreement 6035 6036 := RowAgreement.single (first := 6035) modelrow_agrees_6036
private def atomic_6037 : CountSegment 1 10 13120 6036 6037 2658 2658 := CountSegment.single (first := 6036) (initial := 2658) row_count_6037 (by decide)
private theorem atomic_6037_agreement : RowAgreement 6036 6037 := RowAgreement.single (first := 6036) modelrow_agrees_6037
private def atomic_6038 : CountSegment 1 10 13120 6037 6038 2658 2659 := CountSegment.single (first := 6037) (initial := 2658) row_count_6038 (by decide)
private theorem atomic_6038_agreement : RowAgreement 6037 6038 := RowAgreement.single (first := 6037) modelrow_agrees_6038
private def atomic_6039 : CountSegment 1 10 13120 6038 6039 2659 2659 := CountSegment.single (first := 6038) (initial := 2659) row_count_6039 (by decide)
private theorem atomic_6039_agreement : RowAgreement 6038 6039 := RowAgreement.single (first := 6038) modelrow_agrees_6039
private def atomic_6040 : CountSegment 1 10 13120 6039 6040 2659 2659 := CountSegment.single (first := 6039) (initial := 2659) row_count_6040 (by decide)
private theorem atomic_6040_agreement : RowAgreement 6039 6040 := RowAgreement.single (first := 6039) modelrow_agrees_6040
private def atomic_6041 : CountSegment 1 10 13120 6040 6041 2659 2659 := CountSegment.single (first := 6040) (initial := 2659) row_count_6041 (by decide)
private theorem atomic_6041_agreement : RowAgreement 6040 6041 := RowAgreement.single (first := 6040) modelrow_agrees_6041
private def atomic_6042 : CountSegment 1 10 13120 6041 6042 2659 2661 := CountSegment.single (first := 6041) (initial := 2659) row_count_6042 (by decide)
private theorem atomic_6042_agreement : RowAgreement 6041 6042 := RowAgreement.single (first := 6041) modelrow_agrees_6042
private def atomic_6043 : CountSegment 1 10 13120 6042 6043 2661 2661 := CountSegment.single (first := 6042) (initial := 2661) row_count_6043 (by decide)
private theorem atomic_6043_agreement : RowAgreement 6042 6043 := RowAgreement.single (first := 6042) modelrow_agrees_6043
private def atomic_6044 : CountSegment 1 10 13120 6043 6044 2661 2662 := CountSegment.single (first := 6043) (initial := 2661) row_count_6044 (by decide)
private theorem atomic_6044_agreement : RowAgreement 6043 6044 := RowAgreement.single (first := 6043) modelrow_agrees_6044
private def atomic_6045 : CountSegment 1 10 13120 6044 6045 2662 2663 := CountSegment.single (first := 6044) (initial := 2662) row_count_6045 (by decide)
private theorem atomic_6045_agreement : RowAgreement 6044 6045 := RowAgreement.single (first := 6044) modelrow_agrees_6045
private def atomic_6046 : CountSegment 1 10 13120 6045 6046 2663 2663 := CountSegment.single (first := 6045) (initial := 2663) row_count_6046 (by decide)
private theorem atomic_6046_agreement : RowAgreement 6045 6046 := RowAgreement.single (first := 6045) modelrow_agrees_6046
private def atomic_6047 : CountSegment 1 10 13120 6046 6047 2663 2664 := CountSegment.single (first := 6046) (initial := 2663) row_count_6047 (by decide)
private theorem atomic_6047_agreement : RowAgreement 6046 6047 := RowAgreement.single (first := 6046) modelrow_agrees_6047
private def atomic_6048 : CountSegment 1 10 13120 6047 6048 2664 2665 := CountSegment.single (first := 6047) (initial := 2664) row_count_6048 (by decide)
private theorem atomic_6048_agreement : RowAgreement 6047 6048 := RowAgreement.single (first := 6047) modelrow_agrees_6048
private def atomic_6049 : CountSegment 1 10 13120 6048 6049 2665 2665 := CountSegment.single (first := 6048) (initial := 2665) row_count_6049 (by decide)
private theorem atomic_6049_agreement : RowAgreement 6048 6049 := RowAgreement.single (first := 6048) modelrow_agrees_6049
private def atomic_6050 : CountSegment 1 10 13120 6049 6050 2665 2665 := CountSegment.single (first := 6049) (initial := 2665) row_count_6050 (by decide)
private theorem atomic_6050_agreement : RowAgreement 6049 6050 := RowAgreement.single (first := 6049) modelrow_agrees_6050
private def atomic_6051 : CountSegment 1 10 13120 6050 6051 2665 2665 := CountSegment.single (first := 6050) (initial := 2665) row_count_6051 (by decide)
private theorem atomic_6051_agreement : RowAgreement 6050 6051 := RowAgreement.single (first := 6050) modelrow_agrees_6051
private def atomic_6052 : CountSegment 1 10 13120 6051 6052 2665 2666 := CountSegment.single (first := 6051) (initial := 2665) row_count_6052 (by decide)
private theorem atomic_6052_agreement : RowAgreement 6051 6052 := RowAgreement.single (first := 6051) modelrow_agrees_6052
private def atomic_6053 : CountSegment 1 10 13120 6052 6053 2666 2666 := CountSegment.single (first := 6052) (initial := 2666) row_count_6053 (by decide)
private theorem atomic_6053_agreement : RowAgreement 6052 6053 := RowAgreement.single (first := 6052) modelrow_agrees_6053
private def atomic_6054 : CountSegment 1 10 13120 6053 6054 2666 2666 := CountSegment.single (first := 6053) (initial := 2666) row_count_6054 (by decide)
private theorem atomic_6054_agreement : RowAgreement 6053 6054 := RowAgreement.single (first := 6053) modelrow_agrees_6054
private def atomic_6055 : CountSegment 1 10 13120 6054 6055 2666 2666 := CountSegment.single (first := 6054) (initial := 2666) row_count_6055 (by decide)
private theorem atomic_6055_agreement : RowAgreement 6054 6055 := RowAgreement.single (first := 6054) modelrow_agrees_6055
private def atomic_6056 : CountSegment 1 10 13120 6055 6056 2666 2667 := CountSegment.single (first := 6055) (initial := 2666) row_count_6056 (by decide)
private theorem atomic_6056_agreement : RowAgreement 6055 6056 := RowAgreement.single (first := 6055) modelrow_agrees_6056
private def atomic_6057 : CountSegment 1 10 13120 6056 6057 2667 2668 := CountSegment.single (first := 6056) (initial := 2667) row_count_6057 (by decide)
private theorem atomic_6057_agreement : RowAgreement 6056 6057 := RowAgreement.single (first := 6056) modelrow_agrees_6057
private def atomic_6058 : CountSegment 1 10 13120 6057 6058 2668 2668 := CountSegment.single (first := 6057) (initial := 2668) row_count_6058 (by decide)
private theorem atomic_6058_agreement : RowAgreement 6057 6058 := RowAgreement.single (first := 6057) modelrow_agrees_6058
private def atomic_6059 : CountSegment 1 10 13120 6058 6059 2668 2668 := CountSegment.single (first := 6058) (initial := 2668) row_count_6059 (by decide)
private theorem atomic_6059_agreement : RowAgreement 6058 6059 := RowAgreement.single (first := 6058) modelrow_agrees_6059
private def atomic_6060 : CountSegment 1 10 13120 6059 6060 2668 2669 := CountSegment.single (first := 6059) (initial := 2668) row_count_6060 (by decide)
private theorem atomic_6060_agreement : RowAgreement 6059 6060 := RowAgreement.single (first := 6059) modelrow_agrees_6060
private def atomic_6061 : CountSegment 1 10 13120 6060 6061 2669 2669 := CountSegment.single (first := 6060) (initial := 2669) row_count_6061 (by decide)
private theorem atomic_6061_agreement : RowAgreement 6060 6061 := RowAgreement.single (first := 6060) modelrow_agrees_6061
private def atomic_6062 : CountSegment 1 10 13120 6061 6062 2669 2670 := CountSegment.single (first := 6061) (initial := 2669) row_count_6062 (by decide)
private theorem atomic_6062_agreement : RowAgreement 6061 6062 := RowAgreement.single (first := 6061) modelrow_agrees_6062
private def atomic_6063 : CountSegment 1 10 13120 6062 6063 2670 2671 := CountSegment.single (first := 6062) (initial := 2670) row_count_6063 (by decide)
private theorem atomic_6063_agreement : RowAgreement 6062 6063 := RowAgreement.single (first := 6062) modelrow_agrees_6063
private def atomic_6064 : CountSegment 1 10 13120 6063 6064 2671 2671 := CountSegment.single (first := 6063) (initial := 2671) row_count_6064 (by decide)
private theorem atomic_6064_agreement : RowAgreement 6063 6064 := RowAgreement.single (first := 6063) modelrow_agrees_6064
private def atomic_6065 : CountSegment 1 10 13120 6064 6065 2671 2671 := CountSegment.single (first := 6064) (initial := 2671) row_count_6065 (by decide)
private theorem atomic_6065_agreement : RowAgreement 6064 6065 := RowAgreement.single (first := 6064) modelrow_agrees_6065
private def atomic_6066 : CountSegment 1 10 13120 6065 6066 2671 2672 := CountSegment.single (first := 6065) (initial := 2671) row_count_6066 (by decide)
private theorem atomic_6066_agreement : RowAgreement 6065 6066 := RowAgreement.single (first := 6065) modelrow_agrees_6066
private def atomic_6067 : CountSegment 1 10 13120 6066 6067 2672 2673 := CountSegment.single (first := 6066) (initial := 2672) row_count_6067 (by decide)
private theorem atomic_6067_agreement : RowAgreement 6066 6067 := RowAgreement.single (first := 6066) modelrow_agrees_6067
private def atomic_6068 : CountSegment 1 10 13120 6067 6068 2673 2673 := CountSegment.single (first := 6067) (initial := 2673) row_count_6068 (by decide)
private theorem atomic_6068_agreement : RowAgreement 6067 6068 := RowAgreement.single (first := 6067) modelrow_agrees_6068
private def atomic_6069 : CountSegment 1 10 13120 6068 6069 2673 2673 := CountSegment.single (first := 6068) (initial := 2673) row_count_6069 (by decide)
private theorem atomic_6069_agreement : RowAgreement 6068 6069 := RowAgreement.single (first := 6068) modelrow_agrees_6069
private def atomic_6070 : CountSegment 1 10 13120 6069 6070 2673 2674 := CountSegment.single (first := 6069) (initial := 2673) row_count_6070 (by decide)
private theorem atomic_6070_agreement : RowAgreement 6069 6070 := RowAgreement.single (first := 6069) modelrow_agrees_6070
private def atomic_6071 : CountSegment 1 10 13120 6070 6071 2674 2674 := CountSegment.single (first := 6070) (initial := 2674) row_count_6071 (by decide)
private theorem atomic_6071_agreement : RowAgreement 6070 6071 := RowAgreement.single (first := 6070) modelrow_agrees_6071
private def atomic_6072 : CountSegment 1 10 13120 6071 6072 2674 2676 := CountSegment.single (first := 6071) (initial := 2674) row_count_6072 (by decide)
private theorem atomic_6072_agreement : RowAgreement 6071 6072 := RowAgreement.single (first := 6071) modelrow_agrees_6072
private def atomic_6073 : CountSegment 1 10 13120 6072 6073 2676 2676 := CountSegment.single (first := 6072) (initial := 2676) row_count_6073 (by decide)
private theorem atomic_6073_agreement : RowAgreement 6072 6073 := RowAgreement.single (first := 6072) modelrow_agrees_6073
private def atomic_6074 : CountSegment 1 10 13120 6073 6074 2676 2676 := CountSegment.single (first := 6073) (initial := 2676) row_count_6074 (by decide)
private theorem atomic_6074_agreement : RowAgreement 6073 6074 := RowAgreement.single (first := 6073) modelrow_agrees_6074
private def atomic_6075 : CountSegment 1 10 13120 6074 6075 2676 2676 := CountSegment.single (first := 6074) (initial := 2676) row_count_6075 (by decide)
private theorem atomic_6075_agreement : RowAgreement 6074 6075 := RowAgreement.single (first := 6074) modelrow_agrees_6075
private def atomic_6076 : CountSegment 1 10 13120 6075 6076 2676 2676 := CountSegment.single (first := 6075) (initial := 2676) row_count_6076 (by decide)
private theorem atomic_6076_agreement : RowAgreement 6075 6076 := RowAgreement.single (first := 6075) modelrow_agrees_6076
private def atomic_6077 : CountSegment 1 10 13120 6076 6077 2676 2677 := CountSegment.single (first := 6076) (initial := 2676) row_count_6077 (by decide)
private theorem atomic_6077_agreement : RowAgreement 6076 6077 := RowAgreement.single (first := 6076) modelrow_agrees_6077
private def atomic_6078 : CountSegment 1 10 13120 6077 6078 2677 2677 := CountSegment.single (first := 6077) (initial := 2677) row_count_6078 (by decide)
private theorem atomic_6078_agreement : RowAgreement 6077 6078 := RowAgreement.single (first := 6077) modelrow_agrees_6078
private def atomic_6079 : CountSegment 1 10 13120 6078 6079 2677 2677 := CountSegment.single (first := 6078) (initial := 2677) row_count_6079 (by decide)
private theorem atomic_6079_agreement : RowAgreement 6078 6079 := RowAgreement.single (first := 6078) modelrow_agrees_6079
private def atomic_6080 : CountSegment 1 10 13120 6079 6080 2677 2678 := CountSegment.single (first := 6079) (initial := 2677) row_count_6080 (by decide)
private theorem atomic_6080_agreement : RowAgreement 6079 6080 := RowAgreement.single (first := 6079) modelrow_agrees_6080
private def atomic_6081 : CountSegment 1 10 13120 6080 6081 2678 2678 := CountSegment.single (first := 6080) (initial := 2678) row_count_6081 (by decide)
private theorem atomic_6081_agreement : RowAgreement 6080 6081 := RowAgreement.single (first := 6080) modelrow_agrees_6081
private def atomic_6082 : CountSegment 1 10 13120 6081 6082 2678 2678 := CountSegment.single (first := 6081) (initial := 2678) row_count_6082 (by decide)
private theorem atomic_6082_agreement : RowAgreement 6081 6082 := RowAgreement.single (first := 6081) modelrow_agrees_6082
private def atomic_6083 : CountSegment 1 10 13120 6082 6083 2678 2679 := CountSegment.single (first := 6082) (initial := 2678) row_count_6083 (by decide)
private theorem atomic_6083_agreement : RowAgreement 6082 6083 := RowAgreement.single (first := 6082) modelrow_agrees_6083
private def atomic_6084 : CountSegment 1 10 13120 6083 6084 2679 2679 := CountSegment.single (first := 6083) (initial := 2679) row_count_6084 (by decide)
private theorem atomic_6084_agreement : RowAgreement 6083 6084 := RowAgreement.single (first := 6083) modelrow_agrees_6084
private def atomic_6085 : CountSegment 1 10 13120 6084 6085 2679 2679 := CountSegment.single (first := 6084) (initial := 2679) row_count_6085 (by decide)
private theorem atomic_6085_agreement : RowAgreement 6084 6085 := RowAgreement.single (first := 6084) modelrow_agrees_6085
private def atomic_6086 : CountSegment 1 10 13120 6085 6086 2679 2679 := CountSegment.single (first := 6085) (initial := 2679) row_count_6086 (by decide)
private theorem atomic_6086_agreement : RowAgreement 6085 6086 := RowAgreement.single (first := 6085) modelrow_agrees_6086
private def atomic_6087 : CountSegment 1 10 13120 6086 6087 2679 2680 := CountSegment.single (first := 6086) (initial := 2679) row_count_6087 (by decide)
private theorem atomic_6087_agreement : RowAgreement 6086 6087 := RowAgreement.single (first := 6086) modelrow_agrees_6087
private def atomic_6088 : CountSegment 1 10 13120 6087 6088 2680 2681 := CountSegment.single (first := 6087) (initial := 2680) row_count_6088 (by decide)
private theorem atomic_6088_agreement : RowAgreement 6087 6088 := RowAgreement.single (first := 6087) modelrow_agrees_6088
private def atomic_6089 : CountSegment 1 10 13120 6088 6089 2681 2681 := CountSegment.single (first := 6088) (initial := 2681) row_count_6089 (by decide)
private theorem atomic_6089_agreement : RowAgreement 6088 6089 := RowAgreement.single (first := 6088) modelrow_agrees_6089
private def atomic_6090 : CountSegment 1 10 13120 6089 6090 2681 2681 := CountSegment.single (first := 6089) (initial := 2681) row_count_6090 (by decide)
private theorem atomic_6090_agreement : RowAgreement 6089 6090 := RowAgreement.single (first := 6089) modelrow_agrees_6090
private def atomic_6091 : CountSegment 1 10 13120 6090 6091 2681 2681 := CountSegment.single (first := 6090) (initial := 2681) row_count_6091 (by decide)
private theorem atomic_6091_agreement : RowAgreement 6090 6091 := RowAgreement.single (first := 6090) modelrow_agrees_6091
private def atomic_6092 : CountSegment 1 10 13120 6091 6092 2681 2682 := CountSegment.single (first := 6091) (initial := 2681) row_count_6092 (by decide)
private theorem atomic_6092_agreement : RowAgreement 6091 6092 := RowAgreement.single (first := 6091) modelrow_agrees_6092
private def atomic_6093 : CountSegment 1 10 13120 6092 6093 2682 2683 := CountSegment.single (first := 6092) (initial := 2682) row_count_6093 (by decide)
private theorem atomic_6093_agreement : RowAgreement 6092 6093 := RowAgreement.single (first := 6092) modelrow_agrees_6093
private def atomic_6094 : CountSegment 1 10 13120 6093 6094 2683 2683 := CountSegment.single (first := 6093) (initial := 2683) row_count_6094 (by decide)
private theorem atomic_6094_agreement : RowAgreement 6093 6094 := RowAgreement.single (first := 6093) modelrow_agrees_6094
private def atomic_6095 : CountSegment 1 10 13120 6094 6095 2683 2683 := CountSegment.single (first := 6094) (initial := 2683) row_count_6095 (by decide)
private theorem atomic_6095_agreement : RowAgreement 6094 6095 := RowAgreement.single (first := 6094) modelrow_agrees_6095
private def atomic_6096 : CountSegment 1 10 13120 6095 6096 2683 2683 := CountSegment.single (first := 6095) (initial := 2683) row_count_6096 (by decide)
private theorem atomic_6096_agreement : RowAgreement 6095 6096 := RowAgreement.single (first := 6095) modelrow_agrees_6096
private def atomic_6097 : CountSegment 1 10 13120 6096 6097 2683 2684 := CountSegment.single (first := 6096) (initial := 2683) row_count_6097 (by decide)
private theorem atomic_6097_agreement : RowAgreement 6096 6097 := RowAgreement.single (first := 6096) modelrow_agrees_6097
private def atomic_6098 : CountSegment 1 10 13120 6097 6098 2684 2684 := CountSegment.single (first := 6097) (initial := 2684) row_count_6098 (by decide)
private theorem atomic_6098_agreement : RowAgreement 6097 6098 := RowAgreement.single (first := 6097) modelrow_agrees_6098
private def atomic_6099 : CountSegment 1 10 13120 6098 6099 2684 2684 := CountSegment.single (first := 6098) (initial := 2684) row_count_6099 (by decide)
private theorem atomic_6099_agreement : RowAgreement 6098 6099 := RowAgreement.single (first := 6098) modelrow_agrees_6099
private def atomic_6100 : CountSegment 1 10 13120 6099 6100 2684 2684 := CountSegment.single (first := 6099) (initial := 2684) row_count_6100 (by decide)
private theorem atomic_6100_agreement : RowAgreement 6099 6100 := RowAgreement.single (first := 6099) modelrow_agrees_6100
private def atomic_6101 : CountSegment 1 10 13120 6100 6101 2684 2685 := CountSegment.single (first := 6100) (initial := 2684) row_count_6101 (by decide)
private theorem atomic_6101_agreement : RowAgreement 6100 6101 := RowAgreement.single (first := 6100) modelrow_agrees_6101
private def atomic_6102 : CountSegment 1 10 13120 6101 6102 2685 2686 := CountSegment.single (first := 6101) (initial := 2685) row_count_6102 (by decide)
private theorem atomic_6102_agreement : RowAgreement 6101 6102 := RowAgreement.single (first := 6101) modelrow_agrees_6102
private def atomic_6103 : CountSegment 1 10 13120 6102 6103 2686 2687 := CountSegment.single (first := 6102) (initial := 2686) row_count_6103 (by decide)
private theorem atomic_6103_agreement : RowAgreement 6102 6103 := RowAgreement.single (first := 6102) modelrow_agrees_6103
private def atomic_6104 : CountSegment 1 10 13120 6103 6104 2687 2687 := CountSegment.single (first := 6103) (initial := 2687) row_count_6104 (by decide)
private theorem atomic_6104_agreement : RowAgreement 6103 6104 := RowAgreement.single (first := 6103) modelrow_agrees_6104
private def atomic_6105 : CountSegment 1 10 13120 6104 6105 2687 2687 := CountSegment.single (first := 6104) (initial := 2687) row_count_6105 (by decide)
private theorem atomic_6105_agreement : RowAgreement 6104 6105 := RowAgreement.single (first := 6104) modelrow_agrees_6105
private def atomic_6106 : CountSegment 1 10 13120 6105 6106 2687 2687 := CountSegment.single (first := 6105) (initial := 2687) row_count_6106 (by decide)
private theorem atomic_6106_agreement : RowAgreement 6105 6106 := RowAgreement.single (first := 6105) modelrow_agrees_6106
private def atomic_6107 : CountSegment 1 10 13120 6106 6107 2687 2688 := CountSegment.single (first := 6106) (initial := 2687) row_count_6107 (by decide)
private theorem atomic_6107_agreement : RowAgreement 6106 6107 := RowAgreement.single (first := 6106) modelrow_agrees_6107
private def atomic_6108 : CountSegment 1 10 13120 6107 6108 2688 2688 := CountSegment.single (first := 6107) (initial := 2688) row_count_6108 (by decide)
private theorem atomic_6108_agreement : RowAgreement 6107 6108 := RowAgreement.single (first := 6107) modelrow_agrees_6108
private def atomic_6109 : CountSegment 1 10 13120 6108 6109 2688 2688 := CountSegment.single (first := 6108) (initial := 2688) row_count_6109 (by decide)
private theorem atomic_6109_agreement : RowAgreement 6108 6109 := RowAgreement.single (first := 6108) modelrow_agrees_6109
private def atomic_6110 : CountSegment 1 10 13120 6109 6110 2688 2688 := CountSegment.single (first := 6109) (initial := 2688) row_count_6110 (by decide)
private theorem atomic_6110_agreement : RowAgreement 6109 6110 := RowAgreement.single (first := 6109) modelrow_agrees_6110
private def atomic_6111 : CountSegment 1 10 13120 6110 6111 2688 2688 := CountSegment.single (first := 6110) (initial := 2688) row_count_6111 (by decide)
private theorem atomic_6111_agreement : RowAgreement 6110 6111 := RowAgreement.single (first := 6110) modelrow_agrees_6111
private def atomic_6112 : CountSegment 1 10 13120 6111 6112 2688 2689 := CountSegment.single (first := 6111) (initial := 2688) row_count_6112 (by decide)
private theorem atomic_6112_agreement : RowAgreement 6111 6112 := RowAgreement.single (first := 6111) modelrow_agrees_6112
private def atomic_6113 : CountSegment 1 10 13120 6112 6113 2689 2689 := CountSegment.single (first := 6112) (initial := 2689) row_count_6113 (by decide)
private theorem atomic_6113_agreement : RowAgreement 6112 6113 := RowAgreement.single (first := 6112) modelrow_agrees_6113
private def atomic_6114 : CountSegment 1 10 13120 6113 6114 2689 2690 := CountSegment.single (first := 6113) (initial := 2689) row_count_6114 (by decide)
private theorem atomic_6114_agreement : RowAgreement 6113 6114 := RowAgreement.single (first := 6113) modelrow_agrees_6114
private def atomic_6115 : CountSegment 1 10 13120 6114 6115 2690 2691 := CountSegment.single (first := 6114) (initial := 2690) row_count_6115 (by decide)
private theorem atomic_6115_agreement : RowAgreement 6114 6115 := RowAgreement.single (first := 6114) modelrow_agrees_6115
private def atomic_6116 : CountSegment 1 10 13120 6115 6116 2691 2691 := CountSegment.single (first := 6115) (initial := 2691) row_count_6116 (by decide)
private theorem atomic_6116_agreement : RowAgreement 6115 6116 := RowAgreement.single (first := 6115) modelrow_agrees_6116
private def atomic_6117 : CountSegment 1 10 13120 6116 6117 2691 2692 := CountSegment.single (first := 6116) (initial := 2691) row_count_6117 (by decide)
private theorem atomic_6117_agreement : RowAgreement 6116 6117 := RowAgreement.single (first := 6116) modelrow_agrees_6117
private def atomic_6118 : CountSegment 1 10 13120 6117 6118 2692 2692 := CountSegment.single (first := 6117) (initial := 2692) row_count_6118 (by decide)
private theorem atomic_6118_agreement : RowAgreement 6117 6118 := RowAgreement.single (first := 6117) modelrow_agrees_6118
private def atomic_6119 : CountSegment 1 10 13120 6118 6119 2692 2693 := CountSegment.single (first := 6118) (initial := 2692) row_count_6119 (by decide)
private theorem atomic_6119_agreement : RowAgreement 6118 6119 := RowAgreement.single (first := 6118) modelrow_agrees_6119
private def atomic_6120 : CountSegment 1 10 13120 6119 6120 2693 2693 := CountSegment.single (first := 6119) (initial := 2693) row_count_6120 (by decide)
private theorem atomic_6120_agreement : RowAgreement 6119 6120 := RowAgreement.single (first := 6119) modelrow_agrees_6120
private def atomic_6121 : CountSegment 1 10 13120 6120 6121 2693 2693 := CountSegment.single (first := 6120) (initial := 2693) row_count_6121 (by decide)
private theorem atomic_6121_agreement : RowAgreement 6120 6121 := RowAgreement.single (first := 6120) modelrow_agrees_6121
private def atomic_6122 : CountSegment 1 10 13120 6121 6122 2693 2693 := CountSegment.single (first := 6121) (initial := 2693) row_count_6122 (by decide)
private theorem atomic_6122_agreement : RowAgreement 6121 6122 := RowAgreement.single (first := 6121) modelrow_agrees_6122
private def atomic_6123 : CountSegment 1 10 13120 6122 6123 2693 2693 := CountSegment.single (first := 6122) (initial := 2693) row_count_6123 (by decide)
private theorem atomic_6123_agreement : RowAgreement 6122 6123 := RowAgreement.single (first := 6122) modelrow_agrees_6123
private def atomic_6124 : CountSegment 1 10 13120 6123 6124 2693 2693 := CountSegment.single (first := 6123) (initial := 2693) row_count_6124 (by decide)
private theorem atomic_6124_agreement : RowAgreement 6123 6124 := RowAgreement.single (first := 6123) modelrow_agrees_6124
private def atomic_6125 : CountSegment 1 10 13120 6124 6125 2693 2693 := CountSegment.single (first := 6124) (initial := 2693) row_count_6125 (by decide)
private theorem atomic_6125_agreement : RowAgreement 6124 6125 := RowAgreement.single (first := 6124) modelrow_agrees_6125
private def atomic_6126 : CountSegment 1 10 13120 6125 6126 2693 2694 := CountSegment.single (first := 6125) (initial := 2693) row_count_6126 (by decide)
private theorem atomic_6126_agreement : RowAgreement 6125 6126 := RowAgreement.single (first := 6125) modelrow_agrees_6126
private def atomic_6127 : CountSegment 1 10 13120 6126 6127 2694 2694 := CountSegment.single (first := 6126) (initial := 2694) row_count_6127 (by decide)
private theorem atomic_6127_agreement : RowAgreement 6126 6127 := RowAgreement.single (first := 6126) modelrow_agrees_6127
private def atomic_6128 : CountSegment 1 10 13120 6127 6128 2694 2695 := CountSegment.single (first := 6127) (initial := 2694) row_count_6128 (by decide)
private theorem atomic_6128_agreement : RowAgreement 6127 6128 := RowAgreement.single (first := 6127) modelrow_agrees_6128
private def atomic_6129 : CountSegment 1 10 13120 6128 6129 2695 2695 := CountSegment.single (first := 6128) (initial := 2695) row_count_6129 (by decide)
private theorem atomic_6129_agreement : RowAgreement 6128 6129 := RowAgreement.single (first := 6128) modelrow_agrees_6129
private def atomic_6130 : CountSegment 1 10 13120 6129 6130 2695 2695 := CountSegment.single (first := 6129) (initial := 2695) row_count_6130 (by decide)
private theorem atomic_6130_agreement : RowAgreement 6129 6130 := RowAgreement.single (first := 6129) modelrow_agrees_6130
private def atomic_6131 : CountSegment 1 10 13120 6130 6131 2695 2695 := CountSegment.single (first := 6130) (initial := 2695) row_count_6131 (by decide)
private theorem atomic_6131_agreement : RowAgreement 6130 6131 := RowAgreement.single (first := 6130) modelrow_agrees_6131
private def atomic_6132 : CountSegment 1 10 13120 6131 6132 2695 2696 := CountSegment.single (first := 6131) (initial := 2695) row_count_6132 (by decide)
private theorem atomic_6132_agreement : RowAgreement 6131 6132 := RowAgreement.single (first := 6131) modelrow_agrees_6132
private def atomic_6133 : CountSegment 1 10 13120 6132 6133 2696 2696 := CountSegment.single (first := 6132) (initial := 2696) row_count_6133 (by decide)
private theorem atomic_6133_agreement : RowAgreement 6132 6133 := RowAgreement.single (first := 6132) modelrow_agrees_6133
private def atomic_6134 : CountSegment 1 10 13120 6133 6134 2696 2696 := CountSegment.single (first := 6133) (initial := 2696) row_count_6134 (by decide)
private theorem atomic_6134_agreement : RowAgreement 6133 6134 := RowAgreement.single (first := 6133) modelrow_agrees_6134
private def atomic_6135 : CountSegment 1 10 13120 6134 6135 2696 2697 := CountSegment.single (first := 6134) (initial := 2696) row_count_6135 (by decide)
private theorem atomic_6135_agreement : RowAgreement 6134 6135 := RowAgreement.single (first := 6134) modelrow_agrees_6135
private def atomic_6136 : CountSegment 1 10 13120 6135 6136 2697 2697 := CountSegment.single (first := 6135) (initial := 2697) row_count_6136 (by decide)
private theorem atomic_6136_agreement : RowAgreement 6135 6136 := RowAgreement.single (first := 6135) modelrow_agrees_6136
private def atomic_6137 : CountSegment 1 10 13120 6136 6137 2697 2697 := CountSegment.single (first := 6136) (initial := 2697) row_count_6137 (by decide)
private theorem atomic_6137_agreement : RowAgreement 6136 6137 := RowAgreement.single (first := 6136) modelrow_agrees_6137
private def atomic_6138 : CountSegment 1 10 13120 6137 6138 2697 2699 := CountSegment.single (first := 6137) (initial := 2697) row_count_6138 (by decide)
private theorem atomic_6138_agreement : RowAgreement 6137 6138 := RowAgreement.single (first := 6137) modelrow_agrees_6138
private def atomic_6139 : CountSegment 1 10 13120 6138 6139 2699 2699 := CountSegment.single (first := 6138) (initial := 2699) row_count_6139 (by decide)
private theorem atomic_6139_agreement : RowAgreement 6138 6139 := RowAgreement.single (first := 6138) modelrow_agrees_6139
private def atomic_6140 : CountSegment 1 10 13120 6139 6140 2699 2699 := CountSegment.single (first := 6139) (initial := 2699) row_count_6140 (by decide)
private theorem atomic_6140_agreement : RowAgreement 6139 6140 := RowAgreement.single (first := 6139) modelrow_agrees_6140
private def atomic_6141 : CountSegment 1 10 13120 6140 6141 2699 2699 := CountSegment.single (first := 6140) (initial := 2699) row_count_6141 (by decide)
private theorem atomic_6141_agreement : RowAgreement 6140 6141 := RowAgreement.single (first := 6140) modelrow_agrees_6141
private def atomic_6142 : CountSegment 1 10 13120 6141 6142 2699 2699 := CountSegment.single (first := 6141) (initial := 2699) row_count_6142 (by decide)
private theorem atomic_6142_agreement : RowAgreement 6141 6142 := RowAgreement.single (first := 6141) modelrow_agrees_6142
private def atomic_6143 : CountSegment 1 10 13120 6142 6143 2699 2699 := CountSegment.single (first := 6142) (initial := 2699) row_count_6143 (by decide)
private theorem atomic_6143_agreement : RowAgreement 6142 6143 := RowAgreement.single (first := 6142) modelrow_agrees_6143
private def atomic_6144 : CountSegment 1 10 13120 6143 6144 2699 2699 := CountSegment.single (first := 6143) (initial := 2699) row_count_6144 (by decide)
private theorem atomic_6144_agreement : RowAgreement 6143 6144 := RowAgreement.single (first := 6143) modelrow_agrees_6144
private def within_047_000 : CountSegment 1 10 13120 6016 6018 2654 2655 := CountSegment.join atomic_6017 atomic_6018
private theorem within_047_000_agreement : RowAgreement 6016 6018 := RowAgreement.join atomic_6017_agreement atomic_6018_agreement
private def within_047_001 : CountSegment 1 10 13120 6018 6020 2655 2655 := CountSegment.join atomic_6019 atomic_6020
private theorem within_047_001_agreement : RowAgreement 6018 6020 := RowAgreement.join atomic_6019_agreement atomic_6020_agreement
private def within_047_002 : CountSegment 1 10 13120 6016 6020 2654 2655 := CountSegment.join within_047_000 within_047_001
private theorem within_047_002_agreement : RowAgreement 6016 6020 := RowAgreement.join within_047_000_agreement within_047_001_agreement
private def within_047_003 : CountSegment 1 10 13120 6020 6022 2655 2655 := CountSegment.join atomic_6021 atomic_6022
private theorem within_047_003_agreement : RowAgreement 6020 6022 := RowAgreement.join atomic_6021_agreement atomic_6022_agreement
private def within_047_004 : CountSegment 1 10 13120 6022 6024 2655 2655 := CountSegment.join atomic_6023 atomic_6024
private theorem within_047_004_agreement : RowAgreement 6022 6024 := RowAgreement.join atomic_6023_agreement atomic_6024_agreement
private def within_047_005 : CountSegment 1 10 13120 6020 6024 2655 2655 := CountSegment.join within_047_003 within_047_004
private theorem within_047_005_agreement : RowAgreement 6020 6024 := RowAgreement.join within_047_003_agreement within_047_004_agreement
private def within_047_006 : CountSegment 1 10 13120 6016 6024 2654 2655 := CountSegment.join within_047_002 within_047_005
private theorem within_047_006_agreement : RowAgreement 6016 6024 := RowAgreement.join within_047_002_agreement within_047_005_agreement
private def within_047_007 : CountSegment 1 10 13120 6024 6026 2655 2655 := CountSegment.join atomic_6025 atomic_6026
private theorem within_047_007_agreement : RowAgreement 6024 6026 := RowAgreement.join atomic_6025_agreement atomic_6026_agreement
private def within_047_008 : CountSegment 1 10 13120 6026 6028 2655 2655 := CountSegment.join atomic_6027 atomic_6028
private theorem within_047_008_agreement : RowAgreement 6026 6028 := RowAgreement.join atomic_6027_agreement atomic_6028_agreement
private def within_047_009 : CountSegment 1 10 13120 6024 6028 2655 2655 := CountSegment.join within_047_007 within_047_008
private theorem within_047_009_agreement : RowAgreement 6024 6028 := RowAgreement.join within_047_007_agreement within_047_008_agreement
private def within_047_010 : CountSegment 1 10 13120 6028 6030 2655 2656 := CountSegment.join atomic_6029 atomic_6030
private theorem within_047_010_agreement : RowAgreement 6028 6030 := RowAgreement.join atomic_6029_agreement atomic_6030_agreement
private def within_047_011 : CountSegment 1 10 13120 6030 6032 2656 2657 := CountSegment.join atomic_6031 atomic_6032
private theorem within_047_011_agreement : RowAgreement 6030 6032 := RowAgreement.join atomic_6031_agreement atomic_6032_agreement
private def within_047_012 : CountSegment 1 10 13120 6028 6032 2655 2657 := CountSegment.join within_047_010 within_047_011
private theorem within_047_012_agreement : RowAgreement 6028 6032 := RowAgreement.join within_047_010_agreement within_047_011_agreement
private def within_047_013 : CountSegment 1 10 13120 6024 6032 2655 2657 := CountSegment.join within_047_009 within_047_012
private theorem within_047_013_agreement : RowAgreement 6024 6032 := RowAgreement.join within_047_009_agreement within_047_012_agreement
private def within_047_014 : CountSegment 1 10 13120 6016 6032 2654 2657 := CountSegment.join within_047_006 within_047_013
private theorem within_047_014_agreement : RowAgreement 6016 6032 := RowAgreement.join within_047_006_agreement within_047_013_agreement
private def within_047_015 : CountSegment 1 10 13120 6032 6034 2657 2658 := CountSegment.join atomic_6033 atomic_6034
private theorem within_047_015_agreement : RowAgreement 6032 6034 := RowAgreement.join atomic_6033_agreement atomic_6034_agreement
private def within_047_016 : CountSegment 1 10 13120 6034 6036 2658 2658 := CountSegment.join atomic_6035 atomic_6036
private theorem within_047_016_agreement : RowAgreement 6034 6036 := RowAgreement.join atomic_6035_agreement atomic_6036_agreement
private def within_047_017 : CountSegment 1 10 13120 6032 6036 2657 2658 := CountSegment.join within_047_015 within_047_016
private theorem within_047_017_agreement : RowAgreement 6032 6036 := RowAgreement.join within_047_015_agreement within_047_016_agreement
private def within_047_018 : CountSegment 1 10 13120 6036 6038 2658 2659 := CountSegment.join atomic_6037 atomic_6038
private theorem within_047_018_agreement : RowAgreement 6036 6038 := RowAgreement.join atomic_6037_agreement atomic_6038_agreement
private def within_047_019 : CountSegment 1 10 13120 6038 6040 2659 2659 := CountSegment.join atomic_6039 atomic_6040
private theorem within_047_019_agreement : RowAgreement 6038 6040 := RowAgreement.join atomic_6039_agreement atomic_6040_agreement
private def within_047_020 : CountSegment 1 10 13120 6036 6040 2658 2659 := CountSegment.join within_047_018 within_047_019
private theorem within_047_020_agreement : RowAgreement 6036 6040 := RowAgreement.join within_047_018_agreement within_047_019_agreement
private def within_047_021 : CountSegment 1 10 13120 6032 6040 2657 2659 := CountSegment.join within_047_017 within_047_020
private theorem within_047_021_agreement : RowAgreement 6032 6040 := RowAgreement.join within_047_017_agreement within_047_020_agreement
private def within_047_022 : CountSegment 1 10 13120 6040 6042 2659 2661 := CountSegment.join atomic_6041 atomic_6042
private theorem within_047_022_agreement : RowAgreement 6040 6042 := RowAgreement.join atomic_6041_agreement atomic_6042_agreement
private def within_047_023 : CountSegment 1 10 13120 6042 6044 2661 2662 := CountSegment.join atomic_6043 atomic_6044
private theorem within_047_023_agreement : RowAgreement 6042 6044 := RowAgreement.join atomic_6043_agreement atomic_6044_agreement
private def within_047_024 : CountSegment 1 10 13120 6040 6044 2659 2662 := CountSegment.join within_047_022 within_047_023
private theorem within_047_024_agreement : RowAgreement 6040 6044 := RowAgreement.join within_047_022_agreement within_047_023_agreement
private def within_047_025 : CountSegment 1 10 13120 6044 6046 2662 2663 := CountSegment.join atomic_6045 atomic_6046
private theorem within_047_025_agreement : RowAgreement 6044 6046 := RowAgreement.join atomic_6045_agreement atomic_6046_agreement
private def within_047_026 : CountSegment 1 10 13120 6046 6048 2663 2665 := CountSegment.join atomic_6047 atomic_6048
private theorem within_047_026_agreement : RowAgreement 6046 6048 := RowAgreement.join atomic_6047_agreement atomic_6048_agreement
private def within_047_027 : CountSegment 1 10 13120 6044 6048 2662 2665 := CountSegment.join within_047_025 within_047_026
private theorem within_047_027_agreement : RowAgreement 6044 6048 := RowAgreement.join within_047_025_agreement within_047_026_agreement
private def within_047_028 : CountSegment 1 10 13120 6040 6048 2659 2665 := CountSegment.join within_047_024 within_047_027
private theorem within_047_028_agreement : RowAgreement 6040 6048 := RowAgreement.join within_047_024_agreement within_047_027_agreement
private def within_047_029 : CountSegment 1 10 13120 6032 6048 2657 2665 := CountSegment.join within_047_021 within_047_028
private theorem within_047_029_agreement : RowAgreement 6032 6048 := RowAgreement.join within_047_021_agreement within_047_028_agreement
private def within_047_030 : CountSegment 1 10 13120 6016 6048 2654 2665 := CountSegment.join within_047_014 within_047_029
private theorem within_047_030_agreement : RowAgreement 6016 6048 := RowAgreement.join within_047_014_agreement within_047_029_agreement
private def within_047_031 : CountSegment 1 10 13120 6048 6050 2665 2665 := CountSegment.join atomic_6049 atomic_6050
private theorem within_047_031_agreement : RowAgreement 6048 6050 := RowAgreement.join atomic_6049_agreement atomic_6050_agreement
private def within_047_032 : CountSegment 1 10 13120 6050 6052 2665 2666 := CountSegment.join atomic_6051 atomic_6052
private theorem within_047_032_agreement : RowAgreement 6050 6052 := RowAgreement.join atomic_6051_agreement atomic_6052_agreement
private def within_047_033 : CountSegment 1 10 13120 6048 6052 2665 2666 := CountSegment.join within_047_031 within_047_032
private theorem within_047_033_agreement : RowAgreement 6048 6052 := RowAgreement.join within_047_031_agreement within_047_032_agreement
private def within_047_034 : CountSegment 1 10 13120 6052 6054 2666 2666 := CountSegment.join atomic_6053 atomic_6054
private theorem within_047_034_agreement : RowAgreement 6052 6054 := RowAgreement.join atomic_6053_agreement atomic_6054_agreement
private def within_047_035 : CountSegment 1 10 13120 6054 6056 2666 2667 := CountSegment.join atomic_6055 atomic_6056
private theorem within_047_035_agreement : RowAgreement 6054 6056 := RowAgreement.join atomic_6055_agreement atomic_6056_agreement
private def within_047_036 : CountSegment 1 10 13120 6052 6056 2666 2667 := CountSegment.join within_047_034 within_047_035
private theorem within_047_036_agreement : RowAgreement 6052 6056 := RowAgreement.join within_047_034_agreement within_047_035_agreement
private def within_047_037 : CountSegment 1 10 13120 6048 6056 2665 2667 := CountSegment.join within_047_033 within_047_036
private theorem within_047_037_agreement : RowAgreement 6048 6056 := RowAgreement.join within_047_033_agreement within_047_036_agreement
private def within_047_038 : CountSegment 1 10 13120 6056 6058 2667 2668 := CountSegment.join atomic_6057 atomic_6058
private theorem within_047_038_agreement : RowAgreement 6056 6058 := RowAgreement.join atomic_6057_agreement atomic_6058_agreement
private def within_047_039 : CountSegment 1 10 13120 6058 6060 2668 2669 := CountSegment.join atomic_6059 atomic_6060
private theorem within_047_039_agreement : RowAgreement 6058 6060 := RowAgreement.join atomic_6059_agreement atomic_6060_agreement
private def within_047_040 : CountSegment 1 10 13120 6056 6060 2667 2669 := CountSegment.join within_047_038 within_047_039
private theorem within_047_040_agreement : RowAgreement 6056 6060 := RowAgreement.join within_047_038_agreement within_047_039_agreement
private def within_047_041 : CountSegment 1 10 13120 6060 6062 2669 2670 := CountSegment.join atomic_6061 atomic_6062
private theorem within_047_041_agreement : RowAgreement 6060 6062 := RowAgreement.join atomic_6061_agreement atomic_6062_agreement
private def within_047_042 : CountSegment 1 10 13120 6062 6064 2670 2671 := CountSegment.join atomic_6063 atomic_6064
private theorem within_047_042_agreement : RowAgreement 6062 6064 := RowAgreement.join atomic_6063_agreement atomic_6064_agreement
private def within_047_043 : CountSegment 1 10 13120 6060 6064 2669 2671 := CountSegment.join within_047_041 within_047_042
private theorem within_047_043_agreement : RowAgreement 6060 6064 := RowAgreement.join within_047_041_agreement within_047_042_agreement
private def within_047_044 : CountSegment 1 10 13120 6056 6064 2667 2671 := CountSegment.join within_047_040 within_047_043
private theorem within_047_044_agreement : RowAgreement 6056 6064 := RowAgreement.join within_047_040_agreement within_047_043_agreement
private def within_047_045 : CountSegment 1 10 13120 6048 6064 2665 2671 := CountSegment.join within_047_037 within_047_044
private theorem within_047_045_agreement : RowAgreement 6048 6064 := RowAgreement.join within_047_037_agreement within_047_044_agreement
private def within_047_046 : CountSegment 1 10 13120 6064 6066 2671 2672 := CountSegment.join atomic_6065 atomic_6066
private theorem within_047_046_agreement : RowAgreement 6064 6066 := RowAgreement.join atomic_6065_agreement atomic_6066_agreement
private def within_047_047 : CountSegment 1 10 13120 6066 6068 2672 2673 := CountSegment.join atomic_6067 atomic_6068
private theorem within_047_047_agreement : RowAgreement 6066 6068 := RowAgreement.join atomic_6067_agreement atomic_6068_agreement
private def within_047_048 : CountSegment 1 10 13120 6064 6068 2671 2673 := CountSegment.join within_047_046 within_047_047
private theorem within_047_048_agreement : RowAgreement 6064 6068 := RowAgreement.join within_047_046_agreement within_047_047_agreement
private def within_047_049 : CountSegment 1 10 13120 6068 6070 2673 2674 := CountSegment.join atomic_6069 atomic_6070
private theorem within_047_049_agreement : RowAgreement 6068 6070 := RowAgreement.join atomic_6069_agreement atomic_6070_agreement
private def within_047_050 : CountSegment 1 10 13120 6070 6072 2674 2676 := CountSegment.join atomic_6071 atomic_6072
private theorem within_047_050_agreement : RowAgreement 6070 6072 := RowAgreement.join atomic_6071_agreement atomic_6072_agreement
private def within_047_051 : CountSegment 1 10 13120 6068 6072 2673 2676 := CountSegment.join within_047_049 within_047_050
private theorem within_047_051_agreement : RowAgreement 6068 6072 := RowAgreement.join within_047_049_agreement within_047_050_agreement
private def within_047_052 : CountSegment 1 10 13120 6064 6072 2671 2676 := CountSegment.join within_047_048 within_047_051
private theorem within_047_052_agreement : RowAgreement 6064 6072 := RowAgreement.join within_047_048_agreement within_047_051_agreement
private def within_047_053 : CountSegment 1 10 13120 6072 6074 2676 2676 := CountSegment.join atomic_6073 atomic_6074
private theorem within_047_053_agreement : RowAgreement 6072 6074 := RowAgreement.join atomic_6073_agreement atomic_6074_agreement
private def within_047_054 : CountSegment 1 10 13120 6074 6076 2676 2676 := CountSegment.join atomic_6075 atomic_6076
private theorem within_047_054_agreement : RowAgreement 6074 6076 := RowAgreement.join atomic_6075_agreement atomic_6076_agreement
private def within_047_055 : CountSegment 1 10 13120 6072 6076 2676 2676 := CountSegment.join within_047_053 within_047_054
private theorem within_047_055_agreement : RowAgreement 6072 6076 := RowAgreement.join within_047_053_agreement within_047_054_agreement
private def within_047_056 : CountSegment 1 10 13120 6076 6078 2676 2677 := CountSegment.join atomic_6077 atomic_6078
private theorem within_047_056_agreement : RowAgreement 6076 6078 := RowAgreement.join atomic_6077_agreement atomic_6078_agreement
private def within_047_057 : CountSegment 1 10 13120 6078 6080 2677 2678 := CountSegment.join atomic_6079 atomic_6080
private theorem within_047_057_agreement : RowAgreement 6078 6080 := RowAgreement.join atomic_6079_agreement atomic_6080_agreement
private def within_047_058 : CountSegment 1 10 13120 6076 6080 2676 2678 := CountSegment.join within_047_056 within_047_057
private theorem within_047_058_agreement : RowAgreement 6076 6080 := RowAgreement.join within_047_056_agreement within_047_057_agreement
private def within_047_059 : CountSegment 1 10 13120 6072 6080 2676 2678 := CountSegment.join within_047_055 within_047_058
private theorem within_047_059_agreement : RowAgreement 6072 6080 := RowAgreement.join within_047_055_agreement within_047_058_agreement
private def within_047_060 : CountSegment 1 10 13120 6064 6080 2671 2678 := CountSegment.join within_047_052 within_047_059
private theorem within_047_060_agreement : RowAgreement 6064 6080 := RowAgreement.join within_047_052_agreement within_047_059_agreement
private def within_047_061 : CountSegment 1 10 13120 6048 6080 2665 2678 := CountSegment.join within_047_045 within_047_060
private theorem within_047_061_agreement : RowAgreement 6048 6080 := RowAgreement.join within_047_045_agreement within_047_060_agreement
private def within_047_062 : CountSegment 1 10 13120 6016 6080 2654 2678 := CountSegment.join within_047_030 within_047_061
private theorem within_047_062_agreement : RowAgreement 6016 6080 := RowAgreement.join within_047_030_agreement within_047_061_agreement
private def within_047_063 : CountSegment 1 10 13120 6080 6082 2678 2678 := CountSegment.join atomic_6081 atomic_6082
private theorem within_047_063_agreement : RowAgreement 6080 6082 := RowAgreement.join atomic_6081_agreement atomic_6082_agreement
private def within_047_064 : CountSegment 1 10 13120 6082 6084 2678 2679 := CountSegment.join atomic_6083 atomic_6084
private theorem within_047_064_agreement : RowAgreement 6082 6084 := RowAgreement.join atomic_6083_agreement atomic_6084_agreement
private def within_047_065 : CountSegment 1 10 13120 6080 6084 2678 2679 := CountSegment.join within_047_063 within_047_064
private theorem within_047_065_agreement : RowAgreement 6080 6084 := RowAgreement.join within_047_063_agreement within_047_064_agreement
private def within_047_066 : CountSegment 1 10 13120 6084 6086 2679 2679 := CountSegment.join atomic_6085 atomic_6086
private theorem within_047_066_agreement : RowAgreement 6084 6086 := RowAgreement.join atomic_6085_agreement atomic_6086_agreement
private def within_047_067 : CountSegment 1 10 13120 6086 6088 2679 2681 := CountSegment.join atomic_6087 atomic_6088
private theorem within_047_067_agreement : RowAgreement 6086 6088 := RowAgreement.join atomic_6087_agreement atomic_6088_agreement
private def within_047_068 : CountSegment 1 10 13120 6084 6088 2679 2681 := CountSegment.join within_047_066 within_047_067
private theorem within_047_068_agreement : RowAgreement 6084 6088 := RowAgreement.join within_047_066_agreement within_047_067_agreement
private def within_047_069 : CountSegment 1 10 13120 6080 6088 2678 2681 := CountSegment.join within_047_065 within_047_068
private theorem within_047_069_agreement : RowAgreement 6080 6088 := RowAgreement.join within_047_065_agreement within_047_068_agreement
private def within_047_070 : CountSegment 1 10 13120 6088 6090 2681 2681 := CountSegment.join atomic_6089 atomic_6090
private theorem within_047_070_agreement : RowAgreement 6088 6090 := RowAgreement.join atomic_6089_agreement atomic_6090_agreement
private def within_047_071 : CountSegment 1 10 13120 6090 6092 2681 2682 := CountSegment.join atomic_6091 atomic_6092
private theorem within_047_071_agreement : RowAgreement 6090 6092 := RowAgreement.join atomic_6091_agreement atomic_6092_agreement
private def within_047_072 : CountSegment 1 10 13120 6088 6092 2681 2682 := CountSegment.join within_047_070 within_047_071
private theorem within_047_072_agreement : RowAgreement 6088 6092 := RowAgreement.join within_047_070_agreement within_047_071_agreement
private def within_047_073 : CountSegment 1 10 13120 6092 6094 2682 2683 := CountSegment.join atomic_6093 atomic_6094
private theorem within_047_073_agreement : RowAgreement 6092 6094 := RowAgreement.join atomic_6093_agreement atomic_6094_agreement
private def within_047_074 : CountSegment 1 10 13120 6094 6096 2683 2683 := CountSegment.join atomic_6095 atomic_6096
private theorem within_047_074_agreement : RowAgreement 6094 6096 := RowAgreement.join atomic_6095_agreement atomic_6096_agreement
private def within_047_075 : CountSegment 1 10 13120 6092 6096 2682 2683 := CountSegment.join within_047_073 within_047_074
private theorem within_047_075_agreement : RowAgreement 6092 6096 := RowAgreement.join within_047_073_agreement within_047_074_agreement
private def within_047_076 : CountSegment 1 10 13120 6088 6096 2681 2683 := CountSegment.join within_047_072 within_047_075
private theorem within_047_076_agreement : RowAgreement 6088 6096 := RowAgreement.join within_047_072_agreement within_047_075_agreement
private def within_047_077 : CountSegment 1 10 13120 6080 6096 2678 2683 := CountSegment.join within_047_069 within_047_076
private theorem within_047_077_agreement : RowAgreement 6080 6096 := RowAgreement.join within_047_069_agreement within_047_076_agreement
private def within_047_078 : CountSegment 1 10 13120 6096 6098 2683 2684 := CountSegment.join atomic_6097 atomic_6098
private theorem within_047_078_agreement : RowAgreement 6096 6098 := RowAgreement.join atomic_6097_agreement atomic_6098_agreement
private def within_047_079 : CountSegment 1 10 13120 6098 6100 2684 2684 := CountSegment.join atomic_6099 atomic_6100
private theorem within_047_079_agreement : RowAgreement 6098 6100 := RowAgreement.join atomic_6099_agreement atomic_6100_agreement
private def within_047_080 : CountSegment 1 10 13120 6096 6100 2683 2684 := CountSegment.join within_047_078 within_047_079
private theorem within_047_080_agreement : RowAgreement 6096 6100 := RowAgreement.join within_047_078_agreement within_047_079_agreement
private def within_047_081 : CountSegment 1 10 13120 6100 6102 2684 2686 := CountSegment.join atomic_6101 atomic_6102
private theorem within_047_081_agreement : RowAgreement 6100 6102 := RowAgreement.join atomic_6101_agreement atomic_6102_agreement
private def within_047_082 : CountSegment 1 10 13120 6102 6104 2686 2687 := CountSegment.join atomic_6103 atomic_6104
private theorem within_047_082_agreement : RowAgreement 6102 6104 := RowAgreement.join atomic_6103_agreement atomic_6104_agreement
private def within_047_083 : CountSegment 1 10 13120 6100 6104 2684 2687 := CountSegment.join within_047_081 within_047_082
private theorem within_047_083_agreement : RowAgreement 6100 6104 := RowAgreement.join within_047_081_agreement within_047_082_agreement
private def within_047_084 : CountSegment 1 10 13120 6096 6104 2683 2687 := CountSegment.join within_047_080 within_047_083
private theorem within_047_084_agreement : RowAgreement 6096 6104 := RowAgreement.join within_047_080_agreement within_047_083_agreement
private def within_047_085 : CountSegment 1 10 13120 6104 6106 2687 2687 := CountSegment.join atomic_6105 atomic_6106
private theorem within_047_085_agreement : RowAgreement 6104 6106 := RowAgreement.join atomic_6105_agreement atomic_6106_agreement
private def within_047_086 : CountSegment 1 10 13120 6106 6108 2687 2688 := CountSegment.join atomic_6107 atomic_6108
private theorem within_047_086_agreement : RowAgreement 6106 6108 := RowAgreement.join atomic_6107_agreement atomic_6108_agreement
private def within_047_087 : CountSegment 1 10 13120 6104 6108 2687 2688 := CountSegment.join within_047_085 within_047_086
private theorem within_047_087_agreement : RowAgreement 6104 6108 := RowAgreement.join within_047_085_agreement within_047_086_agreement
private def within_047_088 : CountSegment 1 10 13120 6108 6110 2688 2688 := CountSegment.join atomic_6109 atomic_6110
private theorem within_047_088_agreement : RowAgreement 6108 6110 := RowAgreement.join atomic_6109_agreement atomic_6110_agreement
private def within_047_089 : CountSegment 1 10 13120 6110 6112 2688 2689 := CountSegment.join atomic_6111 atomic_6112
private theorem within_047_089_agreement : RowAgreement 6110 6112 := RowAgreement.join atomic_6111_agreement atomic_6112_agreement
private def within_047_090 : CountSegment 1 10 13120 6108 6112 2688 2689 := CountSegment.join within_047_088 within_047_089
private theorem within_047_090_agreement : RowAgreement 6108 6112 := RowAgreement.join within_047_088_agreement within_047_089_agreement
private def within_047_091 : CountSegment 1 10 13120 6104 6112 2687 2689 := CountSegment.join within_047_087 within_047_090
private theorem within_047_091_agreement : RowAgreement 6104 6112 := RowAgreement.join within_047_087_agreement within_047_090_agreement
private def within_047_092 : CountSegment 1 10 13120 6096 6112 2683 2689 := CountSegment.join within_047_084 within_047_091
private theorem within_047_092_agreement : RowAgreement 6096 6112 := RowAgreement.join within_047_084_agreement within_047_091_agreement
private def within_047_093 : CountSegment 1 10 13120 6080 6112 2678 2689 := CountSegment.join within_047_077 within_047_092
private theorem within_047_093_agreement : RowAgreement 6080 6112 := RowAgreement.join within_047_077_agreement within_047_092_agreement
private def within_047_094 : CountSegment 1 10 13120 6112 6114 2689 2690 := CountSegment.join atomic_6113 atomic_6114
private theorem within_047_094_agreement : RowAgreement 6112 6114 := RowAgreement.join atomic_6113_agreement atomic_6114_agreement
private def within_047_095 : CountSegment 1 10 13120 6114 6116 2690 2691 := CountSegment.join atomic_6115 atomic_6116
private theorem within_047_095_agreement : RowAgreement 6114 6116 := RowAgreement.join atomic_6115_agreement atomic_6116_agreement
private def within_047_096 : CountSegment 1 10 13120 6112 6116 2689 2691 := CountSegment.join within_047_094 within_047_095
private theorem within_047_096_agreement : RowAgreement 6112 6116 := RowAgreement.join within_047_094_agreement within_047_095_agreement
private def within_047_097 : CountSegment 1 10 13120 6116 6118 2691 2692 := CountSegment.join atomic_6117 atomic_6118
private theorem within_047_097_agreement : RowAgreement 6116 6118 := RowAgreement.join atomic_6117_agreement atomic_6118_agreement
private def within_047_098 : CountSegment 1 10 13120 6118 6120 2692 2693 := CountSegment.join atomic_6119 atomic_6120
private theorem within_047_098_agreement : RowAgreement 6118 6120 := RowAgreement.join atomic_6119_agreement atomic_6120_agreement
private def within_047_099 : CountSegment 1 10 13120 6116 6120 2691 2693 := CountSegment.join within_047_097 within_047_098
private theorem within_047_099_agreement : RowAgreement 6116 6120 := RowAgreement.join within_047_097_agreement within_047_098_agreement
private def within_047_100 : CountSegment 1 10 13120 6112 6120 2689 2693 := CountSegment.join within_047_096 within_047_099
private theorem within_047_100_agreement : RowAgreement 6112 6120 := RowAgreement.join within_047_096_agreement within_047_099_agreement
private def within_047_101 : CountSegment 1 10 13120 6120 6122 2693 2693 := CountSegment.join atomic_6121 atomic_6122
private theorem within_047_101_agreement : RowAgreement 6120 6122 := RowAgreement.join atomic_6121_agreement atomic_6122_agreement
private def within_047_102 : CountSegment 1 10 13120 6122 6124 2693 2693 := CountSegment.join atomic_6123 atomic_6124
private theorem within_047_102_agreement : RowAgreement 6122 6124 := RowAgreement.join atomic_6123_agreement atomic_6124_agreement
private def within_047_103 : CountSegment 1 10 13120 6120 6124 2693 2693 := CountSegment.join within_047_101 within_047_102
private theorem within_047_103_agreement : RowAgreement 6120 6124 := RowAgreement.join within_047_101_agreement within_047_102_agreement
private def within_047_104 : CountSegment 1 10 13120 6124 6126 2693 2694 := CountSegment.join atomic_6125 atomic_6126
private theorem within_047_104_agreement : RowAgreement 6124 6126 := RowAgreement.join atomic_6125_agreement atomic_6126_agreement
private def within_047_105 : CountSegment 1 10 13120 6126 6128 2694 2695 := CountSegment.join atomic_6127 atomic_6128
private theorem within_047_105_agreement : RowAgreement 6126 6128 := RowAgreement.join atomic_6127_agreement atomic_6128_agreement
private def within_047_106 : CountSegment 1 10 13120 6124 6128 2693 2695 := CountSegment.join within_047_104 within_047_105
private theorem within_047_106_agreement : RowAgreement 6124 6128 := RowAgreement.join within_047_104_agreement within_047_105_agreement
private def within_047_107 : CountSegment 1 10 13120 6120 6128 2693 2695 := CountSegment.join within_047_103 within_047_106
private theorem within_047_107_agreement : RowAgreement 6120 6128 := RowAgreement.join within_047_103_agreement within_047_106_agreement
private def within_047_108 : CountSegment 1 10 13120 6112 6128 2689 2695 := CountSegment.join within_047_100 within_047_107
private theorem within_047_108_agreement : RowAgreement 6112 6128 := RowAgreement.join within_047_100_agreement within_047_107_agreement
private def within_047_109 : CountSegment 1 10 13120 6128 6130 2695 2695 := CountSegment.join atomic_6129 atomic_6130
private theorem within_047_109_agreement : RowAgreement 6128 6130 := RowAgreement.join atomic_6129_agreement atomic_6130_agreement
private def within_047_110 : CountSegment 1 10 13120 6130 6132 2695 2696 := CountSegment.join atomic_6131 atomic_6132
private theorem within_047_110_agreement : RowAgreement 6130 6132 := RowAgreement.join atomic_6131_agreement atomic_6132_agreement
private def within_047_111 : CountSegment 1 10 13120 6128 6132 2695 2696 := CountSegment.join within_047_109 within_047_110
private theorem within_047_111_agreement : RowAgreement 6128 6132 := RowAgreement.join within_047_109_agreement within_047_110_agreement
private def within_047_112 : CountSegment 1 10 13120 6132 6134 2696 2696 := CountSegment.join atomic_6133 atomic_6134
private theorem within_047_112_agreement : RowAgreement 6132 6134 := RowAgreement.join atomic_6133_agreement atomic_6134_agreement
private def within_047_113 : CountSegment 1 10 13120 6134 6136 2696 2697 := CountSegment.join atomic_6135 atomic_6136
private theorem within_047_113_agreement : RowAgreement 6134 6136 := RowAgreement.join atomic_6135_agreement atomic_6136_agreement
private def within_047_114 : CountSegment 1 10 13120 6132 6136 2696 2697 := CountSegment.join within_047_112 within_047_113
private theorem within_047_114_agreement : RowAgreement 6132 6136 := RowAgreement.join within_047_112_agreement within_047_113_agreement
private def within_047_115 : CountSegment 1 10 13120 6128 6136 2695 2697 := CountSegment.join within_047_111 within_047_114
private theorem within_047_115_agreement : RowAgreement 6128 6136 := RowAgreement.join within_047_111_agreement within_047_114_agreement
private def within_047_116 : CountSegment 1 10 13120 6136 6138 2697 2699 := CountSegment.join atomic_6137 atomic_6138
private theorem within_047_116_agreement : RowAgreement 6136 6138 := RowAgreement.join atomic_6137_agreement atomic_6138_agreement
private def within_047_117 : CountSegment 1 10 13120 6138 6140 2699 2699 := CountSegment.join atomic_6139 atomic_6140
private theorem within_047_117_agreement : RowAgreement 6138 6140 := RowAgreement.join atomic_6139_agreement atomic_6140_agreement
private def within_047_118 : CountSegment 1 10 13120 6136 6140 2697 2699 := CountSegment.join within_047_116 within_047_117
private theorem within_047_118_agreement : RowAgreement 6136 6140 := RowAgreement.join within_047_116_agreement within_047_117_agreement
private def within_047_119 : CountSegment 1 10 13120 6140 6142 2699 2699 := CountSegment.join atomic_6141 atomic_6142
private theorem within_047_119_agreement : RowAgreement 6140 6142 := RowAgreement.join atomic_6141_agreement atomic_6142_agreement
private def within_047_120 : CountSegment 1 10 13120 6142 6144 2699 2699 := CountSegment.join atomic_6143 atomic_6144
private theorem within_047_120_agreement : RowAgreement 6142 6144 := RowAgreement.join atomic_6143_agreement atomic_6144_agreement
private def within_047_121 : CountSegment 1 10 13120 6140 6144 2699 2699 := CountSegment.join within_047_119 within_047_120
private theorem within_047_121_agreement : RowAgreement 6140 6144 := RowAgreement.join within_047_119_agreement within_047_120_agreement
private def within_047_122 : CountSegment 1 10 13120 6136 6144 2697 2699 := CountSegment.join within_047_118 within_047_121
private theorem within_047_122_agreement : RowAgreement 6136 6144 := RowAgreement.join within_047_118_agreement within_047_121_agreement
private def within_047_123 : CountSegment 1 10 13120 6128 6144 2695 2699 := CountSegment.join within_047_115 within_047_122
private theorem within_047_123_agreement : RowAgreement 6128 6144 := RowAgreement.join within_047_115_agreement within_047_122_agreement
private def within_047_124 : CountSegment 1 10 13120 6112 6144 2689 2699 := CountSegment.join within_047_108 within_047_123
private theorem within_047_124_agreement : RowAgreement 6112 6144 := RowAgreement.join within_047_108_agreement within_047_123_agreement
private def within_047_125 : CountSegment 1 10 13120 6080 6144 2678 2699 := CountSegment.join within_047_093 within_047_124
private theorem within_047_125_agreement : RowAgreement 6080 6144 := RowAgreement.join within_047_093_agreement within_047_124_agreement
private def within_047_126 : CountSegment 1 10 13120 6016 6144 2654 2699 := CountSegment.join within_047_062 within_047_125
private theorem within_047_126_agreement : RowAgreement 6016 6144 := RowAgreement.join within_047_062_agreement within_047_125_agreement
def segment_047 : CountSegment 1 10 13120 6016 6144 2654 2699 := within_047_126
theorem agreement_047 : RowAgreement 6016 6144 := within_047_126_agreement
end ProjectEulerSolutions.P58Numerical
