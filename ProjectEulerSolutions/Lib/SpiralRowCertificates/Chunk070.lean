import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_8961 : RowCertificate 8961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8961 1) := by change ¬ Nat.Prime 321216007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45888001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8961 2) := by change ¬ Nat.Prime 321198085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64239617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8961 3) := by change ¬ Nat.Prime 321180163; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2946607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8961 : row 8961 = 0 := (row_8961.checked).trans rfl
theorem model_count_8961 : modelRow 8961 = 0 := by decide
theorem modelrow_agrees_8961 : modelRow 8961 = row 8961 := model_count_8961.trans row_count_8961.symm
def row_8962 : RowCertificate 8962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8962 1) := by change ¬ Nat.Prime 321287701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45898243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8962 2) := by change ¬ Nat.Prime 321269777; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2156173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8962 3) := by change ¬ Nat.Prime 321251853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107083951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8962 : row 8962 = 0 := (row_8962.checked).trans rfl
theorem model_count_8962 : modelRow 8962 = 0 := by decide
theorem modelrow_agrees_8962 : modelRow 8962 = row 8962 := model_count_8962.trans row_count_8962.symm
def row_8963 : RowCertificate 8963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8963 1) := by change ¬ Nat.Prime 321359403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107119801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8963 2) := by change ¬ Nat.Prime 321341477; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 7837597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8963 3) := by change Nat.Prime 321323551; exact ProjectEulerSolutions.P58PrimeNodes.p_321323551.prime
    exact decide_eq_true prime
theorem row_count_8963 : row 8963 = 1 := (row_8963.checked).trans rfl
theorem model_count_8963 : modelRow 8963 = 1 := by decide
theorem modelrow_agrees_8963 : modelRow 8963 = row 8963 := model_count_8963.trans row_count_8963.symm
def row_8964 : RowCertificate 8964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8964 1) := by change ¬ Nat.Prime 321431113; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16917427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8964 2) := by change ¬ Nat.Prime 321413185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64282637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8964 3) := by change ¬ Nat.Prime 321395257; exact PrattCertificates.composite_of_product (factor := 3079) (cofactor := 104383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8964 : row 8964 = 0 := (row_8964.checked).trans rfl
theorem model_count_8964 : modelRow 8964 = 0 := by decide
theorem modelrow_agrees_8964 : modelRow 8964 = row 8964 := model_count_8964.trans row_count_8964.symm
def row_8965 : RowCertificate 8965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8965 1) := by change ¬ Nat.Prime 321502831; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24730987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8965 2) := by change Nat.Prime 321484901; exact ProjectEulerSolutions.P58PrimeNodes.p_321484901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8965 3) := by change ¬ Nat.Prime 321466971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107155657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8965 : row 8965 = 1 := (row_8965.checked).trans rfl
theorem model_count_8965 : modelRow 8965 = 1 := by decide
theorem modelrow_agrees_8965 : modelRow 8965 = row 8965 := model_count_8965.trans row_count_8965.symm
def row_8966 : RowCertificate 8966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8966 1) := by change ¬ Nat.Prime 321574557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107191519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8966 2) := by change ¬ Nat.Prime 321556625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64311325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8966 3) := by change ¬ Nat.Prime 321538693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45934099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8966 : row 8966 = 0 := (row_8966.checked).trans rfl
theorem model_count_8966 : modelRow 8966 = 0 := by decide
theorem modelrow_agrees_8966 : modelRow 8966 = row 8966 := model_count_8966.trans row_count_8966.symm
def row_8967 : RowCertificate 8967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8967 1) := by change ¬ Nat.Prime 321646291; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8693143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8967 2) := by change ¬ Nat.Prime 321628357; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11090633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8967 3) := by change Nat.Prime 321610423; exact ProjectEulerSolutions.P58PrimeNodes.p_321610423.prime
    exact decide_eq_true prime
theorem row_count_8967 : row 8967 = 1 := (row_8967.checked).trans rfl
theorem model_count_8967 : modelRow 8967 = 1 := by decide
theorem modelrow_agrees_8967 : modelRow 8967 = row 8967 := model_count_8967.trans row_count_8967.symm
def row_8968 : RowCertificate 8968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8968 1) := by change ¬ Nat.Prime 321718033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45959719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8968 2) := by change Nat.Prime 321700097; exact ProjectEulerSolutions.P58PrimeNodes.p_321700097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8968 3) := by change ¬ Nat.Prime 321682161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107227387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8968 : row 8968 = 1 := (row_8968.checked).trans rfl
theorem model_count_8968 : modelRow 8968 = 1 := by decide
theorem modelrow_agrees_8968 : modelRow 8968 = row 8968 := model_count_8968.trans row_count_8968.symm
def row_8969 : RowCertificate 8969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8969 1) := by change ¬ Nat.Prime 321789783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107263261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8969 2) := by change ¬ Nat.Prime 321771845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64354369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8969 3) := by change ¬ Nat.Prime 321753907; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7482649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8969 : row 8969 = 0 := (row_8969.checked).trans rfl
theorem model_count_8969 : modelRow 8969 = 0 := by decide
theorem modelrow_agrees_8969 : modelRow 8969 = row 8969 := model_count_8969.trans row_count_8969.symm
def row_8970 : RowCertificate 8970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8970 1) := by change ¬ Nat.Prime 321861541; exact PrattCertificates.composite_of_product (factor := 5011) (cofactor := 64231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8970 2) := by change Nat.Prime 321843601; exact ProjectEulerSolutions.P58PrimeNodes.p_321843601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8970 3) := by change ¬ Nat.Prime 321825661; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2952529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8970 : row 8970 = 1 := (row_8970.checked).trans rfl
theorem model_count_8970 : modelRow 8970 = 1 := by decide
theorem modelrow_agrees_8970 : modelRow 8970 = row 8970 := model_count_8970.trans row_count_8970.symm
def row_8971 : RowCertificate 8971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8971 1) := by change ¬ Nat.Prime 321933307; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 787123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8971 2) := by change ¬ Nat.Prime 321915365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64383073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8971 3) := by change ¬ Nat.Prime 321897423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107299141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8971 : row 8971 = 0 := (row_8971.checked).trans rfl
theorem model_count_8971 : modelRow 8971 = 0 := by decide
theorem modelrow_agrees_8971 : modelRow 8971 = row 8971 := model_count_8971.trans row_count_8971.symm
def row_8972 : RowCertificate 8972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8972 1) := by change ¬ Nat.Prime 322005081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107335027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8972 2) := by change ¬ Nat.Prime 321987137; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6075229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8972 3) := by change ¬ Nat.Prime 321969193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45995599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8972 : row 8972 = 0 := (row_8972.checked).trans rfl
theorem model_count_8972 : modelRow 8972 = 0 := by decide
theorem modelrow_agrees_8972 : modelRow 8972 = row 8972 := model_count_8972.trans row_count_8972.symm
def row_8973 : RowCertificate 8973 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8973 1) := by change Nat.Prime 322076863; exact ProjectEulerSolutions.P58PrimeNodes.p_322076863.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8973 2) := by change Nat.Prime 322058917; exact ProjectEulerSolutions.P58PrimeNodes.p_322058917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8973 3) := by change ¬ Nat.Prime 322040971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46005853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8973 : row 8973 = 2 := (row_8973.checked).trans rfl
theorem model_count_8973 : modelRow 8973 = 2 := by decide
theorem modelrow_agrees_8973 : modelRow 8973 = row 8973 := model_count_8973.trans row_count_8973.symm
def row_8974 : RowCertificate 8974 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8974 1) := by change Nat.Prime 322148653; exact ProjectEulerSolutions.P58PrimeNodes.p_322148653.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8974 2) := by change ¬ Nat.Prime 322130705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64426141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8974 3) := by change ¬ Nat.Prime 322112757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107370919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8974 : row 8974 = 1 := (row_8974.checked).trans rfl
theorem model_count_8974 : modelRow 8974 = 1 := by decide
theorem modelrow_agrees_8974 : modelRow 8974 = row 8974 := model_count_8974.trans row_count_8974.symm
def row_8975 : RowCertificate 8975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8975 1) := by change ¬ Nat.Prime 322220451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107406817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8975 2) := by change Nat.Prime 322202501; exact ProjectEulerSolutions.P58PrimeNodes.p_322202501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8975 3) := by change ¬ Nat.Prime 322184551; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24783427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8975 : row 8975 = 1 := (row_8975.checked).trans rfl
theorem model_count_8975 : modelRow 8975 = 1 := by decide
theorem modelrow_agrees_8975 : modelRow 8975 = row 8975 := model_count_8975.trans row_count_8975.symm
def row_8976 : RowCertificate 8976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8976 1) := by change ¬ Nat.Prime 322292257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46041751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8976 2) := by change ¬ Nat.Prime 322274305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64454861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8976 3) := by change ¬ Nat.Prime 322256353; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 744241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8976 : row 8976 = 0 := (row_8976.checked).trans rfl
theorem model_count_8976 : modelRow 8976 = 0 := by decide
theorem modelrow_agrees_8976 : modelRow 8976 = row 8976 := model_count_8976.trans row_count_8976.symm
def row_8977 : RowCertificate 8977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8977 1) := by change ¬ Nat.Prime 322364071; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10398841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8977 2) := by change Nat.Prime 322346117; exact ProjectEulerSolutions.P58PrimeNodes.p_322346117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8977 3) := by change ¬ Nat.Prime 322328163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107442721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8977 : row 8977 = 1 := (row_8977.checked).trans rfl
theorem model_count_8977 : modelRow 8977 = 1 := by decide
theorem modelrow_agrees_8977 : modelRow 8977 = row 8977 := model_count_8977.trans row_count_8977.symm
def row_8978 : RowCertificate 8978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8978 1) := by change ¬ Nat.Prime 322435893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107478631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8978 2) := by change ¬ Nat.Prime 322417937; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 18965761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8978 3) := by change ¬ Nat.Prime 322399981; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8713513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8978 : row 8978 = 0 := (row_8978.checked).trans rfl
theorem model_count_8978 : modelRow 8978 = 0 := by decide
theorem modelrow_agrees_8978 : modelRow 8978 = row 8978 := model_count_8978.trans row_count_8978.symm
def row_8979 : RowCertificate 8979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8979 1) := by change ¬ Nat.Prime 322507723; exact PrattCertificates.composite_of_product (factor := 3571) (cofactor := 90313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8979 2) := by change ¬ Nat.Prime 322489765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64497953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8979 3) := by change ¬ Nat.Prime 322471807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46067401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8979 : row 8979 = 0 := (row_8979.checked).trans rfl
theorem model_count_8979 : modelRow 8979 = 0 := by decide
theorem modelrow_agrees_8979 : modelRow 8979 = row 8979 := model_count_8979.trans row_count_8979.symm
def row_8980 : RowCertificate 8980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8980 1) := by change ¬ Nat.Prime 322579561; exact PrattCertificates.composite_of_product (factor := 2383) (cofactor := 135367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8980 2) := by change ¬ Nat.Prime 322561601; exact PrattCertificates.composite_of_product (factor := 3853) (cofactor := 83717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8980 3) := by change ¬ Nat.Prime 322543641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107514547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8980 : row 8980 = 0 := (row_8980.checked).trans rfl
theorem model_count_8980 : modelRow 8980 = 0 := by decide
theorem modelrow_agrees_8980 : modelRow 8980 = row 8980 := model_count_8980.trans row_count_8980.symm
def row_8981 : RowCertificate 8981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8981 1) := by change ¬ Nat.Prime 322651407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107550469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8981 2) := by change ¬ Nat.Prime 322633445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64526689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8981 3) := by change Nat.Prime 322615483; exact ProjectEulerSolutions.P58PrimeNodes.p_322615483.prime
    exact decide_eq_true prime
theorem row_count_8981 : row 8981 = 1 := (row_8981.checked).trans rfl
theorem model_count_8981 : modelRow 8981 = 1 := by decide
theorem modelrow_agrees_8981 : modelRow 8981 = row 8981 := model_count_8981.trans row_count_8981.symm
def row_8982 : RowCertificate 8982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8982 1) := by change ¬ Nat.Prime 322723261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46103323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8982 2) := by change Nat.Prime 322705297; exact ProjectEulerSolutions.P58PrimeNodes.p_322705297.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8982 3) := by change Nat.Prime 322687333; exact ProjectEulerSolutions.P58PrimeNodes.p_322687333.prime
    exact decide_eq_true prime
theorem row_count_8982 : row 8982 = 2 := (row_8982.checked).trans rfl
theorem model_count_8982 : modelRow 8982 = 2 := by decide
theorem modelrow_agrees_8982 : modelRow 8982 = row 8982 := model_count_8982.trans row_count_8982.symm
def row_8983 : RowCertificate 8983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8983 1) := by change ¬ Nat.Prime 322795123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46113589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8983 2) := by change ¬ Nat.Prime 322777157; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 268757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8983 3) := by change ¬ Nat.Prime 322759191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107586397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8983 : row 8983 = 0 := (row_8983.checked).trans rfl
theorem model_count_8983 : modelRow 8983 = 0 := by decide
theorem modelrow_agrees_8983 : modelRow 8983 = row 8983 := model_count_8983.trans row_count_8983.symm
def row_8984 : RowCertificate 8984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8984 1) := by change ¬ Nat.Prime 322866993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107622331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8984 2) := by change ¬ Nat.Prime 322849025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64569805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8984 3) := by change ¬ Nat.Prime 322831057; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7507699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8984 : row 8984 = 0 := (row_8984.checked).trans rfl
theorem model_count_8984 : modelRow 8984 = 0 := by decide
theorem modelrow_agrees_8984 : modelRow 8984 = row 8984 := model_count_8984.trans row_count_8984.symm
def row_8985 : RowCertificate 8985 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8985 1) := by change Nat.Prime 322938871; exact ProjectEulerSolutions.P58PrimeNodes.p_322938871.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8985 2) := by change ¬ Nat.Prime 322920901; exact PrattCertificates.composite_of_product (factor := 3373) (cofactor := 95737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8985 3) := by change ¬ Nat.Prime 322902931; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24838687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8985 : row 8985 = 1 := (row_8985.checked).trans rfl
theorem model_count_8985 : modelRow 8985 = 1 := by decide
theorem modelrow_agrees_8985 : modelRow 8985 = row 8985 := model_count_8985.trans row_count_8985.symm
def row_8986 : RowCertificate 8986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8986 1) := by change ¬ Nat.Prime 323010757; exact PrattCertificates.composite_of_product (factor := 811) (cofactor := 398287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8986 2) := by change ¬ Nat.Prime 322992785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64598557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8986 3) := by change ¬ Nat.Prime 322974813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107658271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8986 : row 8986 = 0 := (row_8986.checked).trans rfl
theorem model_count_8986 : modelRow 8986 = 0 := by decide
theorem modelrow_agrees_8986 : modelRow 8986 = row 8986 := model_count_8986.trans row_count_8986.symm
def row_8987 : RowCertificate 8987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8987 1) := by change ¬ Nat.Prime 323082651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107694217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8987 2) := by change ¬ Nat.Prime 323064677; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24851129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8987 3) := by change ¬ Nat.Prime 323046703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46149529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8987 : row 8987 = 0 := (row_8987.checked).trans rfl
theorem model_count_8987 : modelRow 8987 = 0 := by decide
theorem modelrow_agrees_8987 : modelRow 8987 = row 8987 := model_count_8987.trans row_count_8987.symm
def row_8988 : RowCertificate 8988 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8988 1) := by change Nat.Prime 323154553; exact ProjectEulerSolutions.P58PrimeNodes.p_323154553.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8988 2) := by change ¬ Nat.Prime 323136577; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8733421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8988 3) := by change ¬ Nat.Prime 323118601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24855277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8988 : row 8988 = 1 := (row_8988.checked).trans rfl
theorem model_count_8988 : modelRow 8988 = 1 := by decide
theorem modelrow_agrees_8988 : modelRow 8988 = row 8988 := model_count_8988.trans row_count_8988.symm
def row_8989 : RowCertificate 8989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8989 1) := by change ¬ Nat.Prime 323226463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46175209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8989 2) := by change ¬ Nat.Prime 323208485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64641697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8989 3) := by change ¬ Nat.Prime 323190507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107730169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8989 : row 8989 = 0 := (row_8989.checked).trans rfl
theorem model_count_8989 : modelRow 8989 = 0 := by decide
theorem modelrow_agrees_8989 : modelRow 8989 = row 8989 := model_count_8989.trans row_count_8989.symm
def row_8990 : RowCertificate 8990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8990 1) := by change ¬ Nat.Prime 323298381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107766127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8990 2) := by change ¬ Nat.Prime 323280401; exact PrattCertificates.composite_of_product (factor := 3613) (cofactor := 89477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8990 3) := by change Nat.Prime 323262421; exact ProjectEulerSolutions.P58PrimeNodes.p_323262421.prime
    exact decide_eq_true prime
theorem row_count_8990 : row 8990 = 1 := (row_8990.checked).trans rfl
theorem model_count_8990 : modelRow 8990 = 1 := by decide
theorem modelrow_agrees_8990 : modelRow 8990 = row 8990 := model_count_8990.trans row_count_8990.symm
def row_8991 : RowCertificate 8991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8991 1) := by change ¬ Nat.Prime 323370307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24874639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8991 2) := by change ¬ Nat.Prime 323352325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64670465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8991 3) := by change ¬ Nat.Prime 323334343; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17017597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8991 : row 8991 = 0 := (row_8991.checked).trans rfl
theorem model_count_8991 : modelRow 8991 = 0 := by decide
theorem modelrow_agrees_8991 : modelRow 8991 = row 8991 := model_count_8991.trans row_count_8991.symm
def row_8992 : RowCertificate 8992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8992 1) := by change ¬ Nat.Prime 323442241; exact PrattCertificates.composite_of_product (factor := 15901) (cofactor := 20341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8992 2) := by change ¬ Nat.Prime 323424257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24878789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8992 3) := by change ¬ Nat.Prime 323406273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107802091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8992 : row 8992 = 0 := (row_8992.checked).trans rfl
theorem model_count_8992 : modelRow 8992 = 0 := by decide
theorem modelrow_agrees_8992 : modelRow 8992 = row 8992 := model_count_8992.trans row_count_8992.symm
def row_8993 : RowCertificate 8993 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8993 1) := by change ¬ Nat.Prime 323514183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107838061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8993 2) := by change Nat.Prime 323496197; exact ProjectEulerSolutions.P58PrimeNodes.p_323496197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8993 3) := by change ¬ Nat.Prime 323478211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46211173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8993 : row 8993 = 1 := (row_8993.checked).trans rfl
theorem model_count_8993 : modelRow 8993 = 1 := by decide
theorem modelrow_agrees_8993 : modelRow 8993 = row 8993 := model_count_8993.trans row_count_8993.symm
def row_8994 : RowCertificate 8994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8994 1) := by change ¬ Nat.Prime 323586133; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24891241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8994 2) := by change ¬ Nat.Prime 323568145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64713629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8994 3) := by change ¬ Nat.Prime 323550157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46221451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8994 : row 8994 = 0 := (row_8994.checked).trans rfl
theorem model_count_8994 : modelRow 8994 = 0 := by decide
theorem modelrow_agrees_8994 : modelRow 8994 = row 8994 := model_count_8994.trans row_count_8994.symm
def row_8995 : RowCertificate 8995 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8995 1) := by change Nat.Prime 323658091; exact ProjectEulerSolutions.P58PrimeNodes.p_323658091.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8995 2) := by change ¬ Nat.Prime 323640101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19037653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8995 3) := by change ¬ Nat.Prime 323622111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107874037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8995 : row 8995 = 1 := (row_8995.checked).trans rfl
theorem model_count_8995 : modelRow 8995 = 1 := by decide
theorem modelrow_agrees_8995 : modelRow 8995 = row 8995 := model_count_8995.trans row_count_8995.symm
def row_8996 : RowCertificate 8996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8996 1) := by change ¬ Nat.Prime 323730057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107910019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8996 2) := by change ¬ Nat.Prime 323712065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64742413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8996 3) := by change ¬ Nat.Prime 323694073; exact PrattCertificates.composite_of_product (factor := 11197) (cofactor := 28909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8996 : row 8996 = 0 := (row_8996.checked).trans rfl
theorem model_count_8996 : modelRow 8996 = 0 := by decide
theorem modelrow_agrees_8996 : modelRow 8996 = row 8996 := model_count_8996.trans row_count_8996.symm
def row_8997 : RowCertificate 8997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8997 1) := by change ¬ Nat.Prime 323802031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46257433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8997 2) := by change ¬ Nat.Prime 323784037; exact PrattCertificates.composite_of_product (factor := 11801) (cofactor := 27437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8997 3) := by change ¬ Nat.Prime 323766043; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4832329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8997 : row 8997 = 0 := (row_8997.checked).trans rfl
theorem model_count_8997 : modelRow 8997 = 0 := by decide
theorem modelrow_agrees_8997 : modelRow 8997 = row 8997 := model_count_8997.trans row_count_8997.symm
def row_8998 : RowCertificate 8998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8998 1) := by change ¬ Nat.Prime 323874013; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1414297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8998 2) := by change Nat.Prime 323856017; exact ProjectEulerSolutions.P58PrimeNodes.p_323856017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8998 3) := by change ¬ Nat.Prime 323838021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107946007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8998 : row 8998 = 1 := (row_8998.checked).trans rfl
theorem model_count_8998 : modelRow 8998 = 1 := by decide
theorem modelrow_agrees_8998 : modelRow 8998 = row 8998 := model_count_8998.trans row_count_8998.symm
def row_8999 : RowCertificate 8999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8999 1) := by change ¬ Nat.Prime 323946003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107982001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8999 2) := by change ¬ Nat.Prime 323928005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64785601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8999 3) := by change ¬ Nat.Prime 323910007; exact PrattCertificates.composite_of_product (factor := 883) (cofactor := 366829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8999 : row 8999 = 0 := (row_8999.checked).trans rfl
theorem model_count_8999 : modelRow 8999 = 0 := by decide
theorem modelrow_agrees_8999 : modelRow 8999 = row 8999 := model_count_8999.trans row_count_8999.symm
def row_9000 : RowCertificate 9000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9000 1) := by change ¬ Nat.Prime 324018001; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17053579) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9000 2) := by change ¬ Nat.Prime 324000001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24923077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9000 3) := by change ¬ Nat.Prime 323982001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46283143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9000 : row 9000 = 0 := (row_9000.checked).trans rfl
theorem model_count_9000 : modelRow 9000 = 0 := by decide
theorem modelrow_agrees_9000 : modelRow 9000 = row 9000 := model_count_9000.trans row_count_9000.symm
def row_9001 : RowCertificate 9001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9001 1) := by change ¬ Nat.Prime 324090007; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1628593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9001 2) := by change ¬ Nat.Prime 324072005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64814401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9001 3) := by change ¬ Nat.Prime 324054003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108018001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9001 : row 9001 = 0 := (row_9001.checked).trans rfl
theorem model_count_9001 : modelRow 9001 = 0 := by decide
theorem modelrow_agrees_9001 : modelRow 9001 = row 9001 := model_count_9001.trans row_count_9001.symm
def row_9002 : RowCertificate 9002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9002 1) := by change ¬ Nat.Prime 324162021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108054007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9002 2) := by change ¬ Nat.Prime 324144017; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4440329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9002 3) := by change Nat.Prime 324126013; exact ProjectEulerSolutions.P58PrimeNodes.p_324126013.prime
    exact decide_eq_true prime
theorem row_count_9002 : row 9002 = 1 := (row_9002.checked).trans rfl
theorem model_count_9002 : modelRow 9002 = 1 := by decide
theorem modelrow_agrees_9002 : modelRow 9002 = row 9002 := model_count_9002.trans row_count_9002.symm
def row_9003 : RowCertificate 9003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9003 1) := by change ¬ Nat.Prime 324234043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46319149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9003 2) := by change ¬ Nat.Prime 324216037; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5315017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9003 3) := by change ¬ Nat.Prime 324198031; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10458001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9003 : row 9003 = 0 := (row_9003.checked).trans rfl
theorem model_count_9003 : modelRow 9003 = 0 := by decide
theorem modelrow_agrees_9003 : modelRow 9003 = row 9003 := model_count_9003.trans row_count_9003.symm
def row_9004 : RowCertificate 9004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9004 1) := by change ¬ Nat.Prime 324306073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46329439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9004 2) := by change ¬ Nat.Prime 324288065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64857613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9004 3) := by change ¬ Nat.Prime 324270057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108090019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9004 : row 9004 = 0 := (row_9004.checked).trans rfl
theorem model_count_9004 : modelRow 9004 = 0 := by decide
theorem modelrow_agrees_9004 : modelRow 9004 = row 9004 := model_count_9004.trans row_count_9004.symm
def row_9005 : RowCertificate 9005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9005 1) := by change ¬ Nat.Prime 324378111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108126037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9005 2) := by change ¬ Nat.Prime 324360101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24950777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9005 3) := by change ¬ Nat.Prime 324342091; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 317671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9005 : row 9005 = 0 := (row_9005.checked).trans rfl
theorem model_count_9005 : modelRow 9005 = 0 := by decide
theorem modelrow_agrees_9005 : modelRow 9005 = row 9005 := model_count_9005.trans row_count_9005.symm
def row_9006 : RowCertificate 9006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9006 1) := by change ¬ Nat.Prime 324450157; exact PrattCertificates.composite_of_product (factor := 15913) (cofactor := 20389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9006 2) := by change ¬ Nat.Prime 324432145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64886429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9006 3) := by change Nat.Prime 324414133; exact ProjectEulerSolutions.P58PrimeNodes.p_324414133.prime
    exact decide_eq_true prime
theorem row_count_9006 : row 9006 = 1 := (row_9006.checked).trans rfl
theorem model_count_9006 : modelRow 9006 = 1 := by decide
theorem modelrow_agrees_9006 : modelRow 9006 = row 9006 := model_count_9006.trans row_count_9006.symm
def row_9007 : RowCertificate 9007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9007 1) := by change ¬ Nat.Prime 324522211; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24963247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9007 2) := by change Nat.Prime 324504197; exact ProjectEulerSolutions.P58PrimeNodes.p_324504197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9007 3) := by change ¬ Nat.Prime 324486183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108162061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9007 : row 9007 = 1 := (row_9007.checked).trans rfl
theorem model_count_9007 : modelRow 9007 = 1 := by decide
theorem modelrow_agrees_9007 : modelRow 9007 = row 9007 := model_count_9007.trans row_count_9007.symm
def row_9008 : RowCertificate 9008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9008 1) := by change ¬ Nat.Prime 324594273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108198091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9008 2) := by change ¬ Nat.Prime 324576257; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19092721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9008 3) := by change ¬ Nat.Prime 324558241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46365463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9008 : row 9008 = 0 := (row_9008.checked).trans rfl
theorem model_count_9008 : modelRow 9008 = 0 := by decide
theorem modelrow_agrees_9008 : modelRow 9008 = row 9008 := model_count_9008.trans row_count_9008.symm
def row_9009 : RowCertificate 9009 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9009 1) := by change Nat.Prime 324666343; exact ProjectEulerSolutions.P58PrimeNodes.p_324666343.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9009 2) := by change ¬ Nat.Prime 324648325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64929665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9009 3) := by change ¬ Nat.Prime 324630307; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 442879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9009 : row 9009 = 1 := (row_9009.checked).trans rfl
theorem model_count_9009 : modelRow 9009 = 1 := by decide
theorem modelrow_agrees_9009 : modelRow 9009 = row 9009 := model_count_9009.trans row_count_9009.symm
def row_9010 : RowCertificate 9010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9010 1) := by change ¬ Nat.Prime 324738421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46391203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9010 2) := by change ¬ Nat.Prime 324720401; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3347633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9010 3) := by change ¬ Nat.Prime 324702381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108234127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9010 : row 9010 = 0 := (row_9010.checked).trans rfl
theorem model_count_9010 : modelRow 9010 = 0 := by decide
theorem modelrow_agrees_9010 : modelRow 9010 = row 9010 := model_count_9010.trans row_count_9010.symm
def row_9011 : RowCertificate 9011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9011 1) := by change ¬ Nat.Prime 324810507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108270169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9011 2) := by change ¬ Nat.Prime 324792485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64958497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9011 3) := by change ¬ Nat.Prime 324774463; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24982651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9011 : row 9011 = 0 := (row_9011.checked).trans rfl
theorem model_count_9011 : modelRow 9011 = 0 := by decide
theorem modelrow_agrees_9011 : modelRow 9011 = row 9011 := model_count_9011.trans row_count_9011.symm
def row_9012 : RowCertificate 9012 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9012 1) := by change Nat.Prime 324882601; exact ProjectEulerSolutions.P58PrimeNodes.p_324882601.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9012 2) := by change ¬ Nat.Prime 324864577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19109681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9012 3) := by change ¬ Nat.Prime 324846553; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17097187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9012 : row 9012 = 1 := (row_9012.checked).trans rfl
theorem model_count_9012 : modelRow 9012 = 1 := by decide
theorem modelrow_agrees_9012 : modelRow 9012 = row 9012 := model_count_9012.trans row_count_9012.symm
def row_9013 : RowCertificate 9013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9013 1) := by change ¬ Nat.Prime 324954703; exact PrattCertificates.composite_of_product (factor := 13597) (cofactor := 23899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9013 2) := by change ¬ Nat.Prime 324936677; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24995129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9013 3) := by change ¬ Nat.Prime 324918651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108306217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9013 : row 9013 = 0 := (row_9013.checked).trans rfl
theorem model_count_9013 : modelRow 9013 = 0 := by decide
theorem modelrow_agrees_9013 : modelRow 9013 = row 9013 := model_count_9013.trans row_count_9013.symm
def row_9014 : RowCertificate 9014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9014 1) := by change ¬ Nat.Prime 325026813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108342271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9014 2) := by change ¬ Nat.Prime 325008785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65001757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9014 3) := by change ¬ Nat.Prime 324990757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46427251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9014 : row 9014 = 0 := (row_9014.checked).trans rfl
theorem model_count_9014 : modelRow 9014 = 0 := by decide
theorem modelrow_agrees_9014 : modelRow 9014 = row 9014 := model_count_9014.trans row_count_9014.symm
def row_9015 : RowCertificate 9015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9015 1) := by change ¬ Nat.Prime 325098931; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2982559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9015 2) := by change ¬ Nat.Prime 325080901; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1684357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9015 3) := by change ¬ Nat.Prime 325062871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46437553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9015 : row 9015 = 0 := (row_9015.checked).trans rfl
theorem model_count_9015 : modelRow 9015 = 0 := by decide
theorem modelrow_agrees_9015 : modelRow 9015 = row 9015 := model_count_9015.trans row_count_9015.symm
def row_9016 : RowCertificate 9016 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9016 1) := by change Nat.Prime 325171057; exact ProjectEulerSolutions.P58PrimeNodes.p_325171057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9016 2) := by change ¬ Nat.Prime 325153025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65030605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9016 3) := by change ¬ Nat.Prime 325134993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108378331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9016 : row 9016 = 1 := (row_9016.checked).trans rfl
theorem model_count_9016 : modelRow 9016 = 1 := by decide
theorem modelrow_agrees_9016 : modelRow 9016 = row 9016 := model_count_9016.trans row_count_9016.symm
def row_9017 : RowCertificate 9017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9017 1) := by change ¬ Nat.Prime 325243191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108414397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9017 2) := by change Nat.Prime 325225157; exact ProjectEulerSolutions.P58PrimeNodes.p_325225157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9017 3) := by change ¬ Nat.Prime 325207123; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1685011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9017 : row 9017 = 1 := (row_9017.checked).trans rfl
theorem model_count_9017 : modelRow 9017 = 1 := by decide
theorem modelrow_agrees_9017 : modelRow 9017 = row 9017 := model_count_9017.trans row_count_9017.symm
def row_9018 : RowCertificate 9018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9018 1) := by change ¬ Nat.Prime 325315333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46473619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9018 2) := by change ¬ Nat.Prime 325297297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25022869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9018 3) := by change ¬ Nat.Prime 325279261; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4117459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9018 : row 9018 = 0 := (row_9018.checked).trans rfl
theorem model_count_9018 : modelRow 9018 = 0 := by decide
theorem modelrow_agrees_9018 : modelRow 9018 = row 9018 := model_count_9018.trans row_count_9018.symm
def row_9019 : RowCertificate 9019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9019 1) := by change ¬ Nat.Prime 325387483; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17125657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9019 2) := by change ¬ Nat.Prime 325369445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65073889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9019 3) := by change ¬ Nat.Prime 325351407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108450469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9019 : row 9019 = 0 := (row_9019.checked).trans rfl
theorem model_count_9019 : modelRow 9019 = 0 := by decide
theorem modelrow_agrees_9019 : modelRow 9019 = row 9019 := model_count_9019.trans row_count_9019.symm
def row_9020 : RowCertificate 9020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9020 1) := by change ¬ Nat.Prime 325459641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108486547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9020 2) := by change Nat.Prime 325441601; exact ProjectEulerSolutions.P58PrimeNodes.p_325441601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9020 3) := by change ¬ Nat.Prime 325423561; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4457857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9020 : row 9020 = 1 := (row_9020.checked).trans rfl
theorem model_count_9020 : modelRow 9020 = 1 := by decide
theorem modelrow_agrees_9020 : modelRow 9020 = row 9020 := model_count_9020.trans row_count_9020.symm
def row_9021 : RowCertificate 9021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9021 1) := by change ¬ Nat.Prime 325531807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17133253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9021 2) := by change ¬ Nat.Prime 325513765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65102753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9021 3) := by change ¬ Nat.Prime 325495723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46499389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9021 : row 9021 = 0 := (row_9021.checked).trans rfl
theorem model_count_9021 : modelRow 9021 = 0 := by decide
theorem modelrow_agrees_9021 : modelRow 9021 = row 9021 := model_count_9021.trans row_count_9021.symm
def row_9022 : RowCertificate 9022 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9022 1) := by change Nat.Prime 325603981; exact ProjectEulerSolutions.P58PrimeNodes.p_325603981.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9022 2) := by change ¬ Nat.Prime 325585937; exact PrattCertificates.composite_of_product (factor := 9437) (cofactor := 34501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9022 3) := by change ¬ Nat.Prime 325567893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108522631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9022 : row 9022 = 1 := (row_9022.checked).trans rfl
theorem model_count_9022 : modelRow 9022 = 1 := by decide
theorem modelrow_agrees_9022 : modelRow 9022 = row 9022 := model_count_9022.trans row_count_9022.symm
def row_9023 : RowCertificate 9023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9023 1) := by change ¬ Nat.Prime 325676163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108558721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9023 2) := by change Nat.Prime 325658117; exact ProjectEulerSolutions.P58PrimeNodes.p_325658117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9023 3) := by change ¬ Nat.Prime 325640071; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8801083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9023 : row 9023 = 1 := (row_9023.checked).trans rfl
theorem model_count_9023 : modelRow 9023 = 1 := by decide
theorem modelrow_agrees_9023 : modelRow 9023 = row 9023 := model_count_9023.trans row_count_9023.symm
def row_9024 : RowCertificate 9024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9024 1) := by change ¬ Nat.Prime 325748353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46535479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9024 2) := by change ¬ Nat.Prime 325730305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65146061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9024 3) := by change ¬ Nat.Prime 325712257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25054789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9024 : row 9024 = 0 := (row_9024.checked).trans rfl
theorem model_count_9024 : modelRow 9024 = 0 := by decide
theorem modelrow_agrees_9024 : modelRow 9024 = row 9024 := model_count_9024.trans row_count_9024.symm
def row_9025 : RowCertificate 9025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9025 1) := by change ¬ Nat.Prime 325820551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46545793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9025 2) := by change ¬ Nat.Prime 325802501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19164853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9025 3) := by change ¬ Nat.Prime 325784451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108594817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9025 : row 9025 = 0 := (row_9025.checked).trans rfl
theorem model_count_9025 : modelRow 9025 = 0 := by decide
theorem modelrow_agrees_9025 : modelRow 9025 = row 9025 := model_count_9025.trans row_count_9025.symm
def row_9026 : RowCertificate 9026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9026 1) := by change ¬ Nat.Prime 325892757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108630919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9026 2) := by change ¬ Nat.Prime 325874705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65174941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9026 3) := by change ¬ Nat.Prime 325856653; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1800313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9026 : row 9026 = 0 := (row_9026.checked).trans rfl
theorem model_count_9026 : modelRow 9026 = 0 := by decide
theorem modelrow_agrees_9026 : modelRow 9026 = row 9026 := model_count_9026.trans row_count_9026.symm
def row_9027 : RowCertificate 9027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9027 1) := by change ¬ Nat.Prime 325964971; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 542371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9027 2) := by change Nat.Prime 325946917; exact ProjectEulerSolutions.P58PrimeNodes.p_325946917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9027 3) := by change ¬ Nat.Prime 325928863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25071451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9027 : row 9027 = 1 := (row_9027.checked).trans rfl
theorem model_count_9027 : modelRow 9027 = 1 := by decide
theorem modelrow_agrees_9027 : modelRow 9027 = row 9027 := model_count_9027.trans row_count_9027.symm
def row_9028 : RowCertificate 9028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9028 1) := by change ¬ Nat.Prime 326037193; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 441187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9028 2) := by change ¬ Nat.Prime 326019137; exact PrattCertificates.composite_of_product (factor := 6101) (cofactor := 53437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9028 3) := by change ¬ Nat.Prime 326001081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108667027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9028 : row 9028 = 0 := (row_9028.checked).trans rfl
theorem model_count_9028 : modelRow 9028 = 0 := by decide
theorem modelrow_agrees_9028 : modelRow 9028 = row 9028 := model_count_9028.trans row_count_9028.symm
def row_9029 : RowCertificate 9029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9029 1) := by change ¬ Nat.Prime 326109423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108703141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9029 2) := by change ¬ Nat.Prime 326091365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65218273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9029 3) := by change ¬ Nat.Prime 326073307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46581901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9029 : row 9029 = 0 := (row_9029.checked).trans rfl
theorem model_count_9029 : modelRow 9029 = 0 := by decide
theorem modelrow_agrees_9029 : modelRow 9029 = row 9029 := model_count_9029.trans row_count_9029.symm
def row_9030 : RowCertificate 9030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9030 1) := by change ¬ Nat.Prime 326181661; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25090897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9030 2) := by change Nat.Prime 326163601; exact ProjectEulerSolutions.P58PrimeNodes.p_326163601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9030 3) := by change ¬ Nat.Prime 326145541; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1353301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9030 : row 9030 = 1 := (row_9030.checked).trans rfl
theorem model_count_9030 : modelRow 9030 = 1 := by decide
theorem modelrow_agrees_9030 : modelRow 9030 = row 9030 := model_count_9030.trans row_count_9030.symm
def row_9031 : RowCertificate 9031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9031 1) := by change ¬ Nat.Prime 326253907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46607701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9031 2) := by change ¬ Nat.Prime 326235845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65247169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9031 3) := by change ¬ Nat.Prime 326217783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108739261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9031 : row 9031 = 0 := (row_9031.checked).trans rfl
theorem model_count_9031 : modelRow 9031 = 0 := by decide
theorem modelrow_agrees_9031 : modelRow 9031 = row 9031 := model_count_9031.trans row_count_9031.symm
def row_9032 : RowCertificate 9032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9032 1) := by change ¬ Nat.Prime 326326161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108775387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9032 2) := by change ¬ Nat.Prime 326308097; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3364001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9032 3) := by change Nat.Prime 326290033; exact ProjectEulerSolutions.P58PrimeNodes.p_326290033.prime
    exact decide_eq_true prime
theorem row_count_9032 : row 9032 = 1 := (row_9032.checked).trans rfl
theorem model_count_9032 : modelRow 9032 = 1 := by decide
theorem modelrow_agrees_9032 : modelRow 9032 = row 9032 := model_count_9032.trans row_count_9032.symm
def row_9033 : RowCertificate 9033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9033 1) := by change ¬ Nat.Prime 326398423; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25107571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9033 2) := by change Nat.Prime 326380357; exact ProjectEulerSolutions.P58PrimeNodes.p_326380357.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9033 3) := by change Nat.Prime 326362291; exact ProjectEulerSolutions.P58PrimeNodes.p_326362291.prime
    exact decide_eq_true prime
theorem row_count_9033 : row 9033 = 2 := (row_9033.checked).trans rfl
theorem model_count_9033 : modelRow 9033 = 2 := by decide
theorem modelrow_agrees_9033 : modelRow 9033 = row 9033 := model_count_9033.trans row_count_9033.symm
def row_9034 : RowCertificate 9034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9034 1) := by change ¬ Nat.Prime 326470693; exact PrattCertificates.composite_of_product (factor := 1447) (cofactor := 225619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9034 2) := by change ¬ Nat.Prime 326452625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65290525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9034 3) := by change ¬ Nat.Prime 326434557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108811519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9034 : row 9034 = 0 := (row_9034.checked).trans rfl
theorem model_count_9034 : modelRow 9034 = 0 := by decide
theorem modelrow_agrees_9034 : modelRow 9034 = row 9034 := model_count_9034.trans row_count_9034.symm
def row_9035 : RowCertificate 9035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9035 1) := by change ¬ Nat.Prime 326542971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108847657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9035 2) := by change Nat.Prime 326524901; exact ProjectEulerSolutions.P58PrimeNodes.p_326524901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9035 3) := by change ¬ Nat.Prime 326506831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46643833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9035 : row 9035 = 1 := (row_9035.checked).trans rfl
theorem model_count_9035 : modelRow 9035 = 1 := by decide
theorem modelrow_agrees_9035 : modelRow 9035 = row 9035 := model_count_9035.trans row_count_9035.symm
def row_9036 : RowCertificate 9036 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9036 1) := by change Nat.Prime 326615257; exact ProjectEulerSolutions.P58PrimeNodes.p_326615257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9036 2) := by change ¬ Nat.Prime 326597185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65319437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9036 3) := by change ¬ Nat.Prime 326579113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46654159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9036 : row 9036 = 1 := (row_9036.checked).trans rfl
theorem model_count_9036 : modelRow 9036 = 1 := by decide
theorem modelrow_agrees_9036 : modelRow 9036 = row 9036 := model_count_9036.trans row_count_9036.symm
def row_9037 : RowCertificate 9037 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9037 1) := by change Nat.Prime 326687551; exact ProjectEulerSolutions.P58PrimeNodes.p_326687551.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9037 2) := by change ¬ Nat.Prime 326669477; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 775937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9037 3) := by change ¬ Nat.Prime 326651403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108883801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9037 : row 9037 = 1 := (row_9037.checked).trans rfl
theorem model_count_9037 : modelRow 9037 = 1 := by decide
theorem modelrow_agrees_9037 : modelRow 9037 = row 9037 := model_count_9037.trans row_count_9037.symm
def row_9038 : RowCertificate 9038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9038 1) := by change ¬ Nat.Prime 326759853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108919951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9038 2) := by change Nat.Prime 326741777; exact ProjectEulerSolutions.P58PrimeNodes.p_326741777.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9038 3) := by change Nat.Prime 326723701; exact ProjectEulerSolutions.P58PrimeNodes.p_326723701.prime
    exact decide_eq_true prime
theorem row_count_9038 : row 9038 = 2 := (row_9038.checked).trans rfl
theorem model_count_9038 : modelRow 9038 = 2 := by decide
theorem modelrow_agrees_9038 : modelRow 9038 = row 9038 := model_count_9038.trans row_count_9038.symm
def row_9039 : RowCertificate 9039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9039 1) := by change ¬ Nat.Prime 326832163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46690309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9039 2) := by change ¬ Nat.Prime 326814085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65362817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9039 3) := by change ¬ Nat.Prime 326796007; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3369031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9039 : row 9039 = 0 := (row_9039.checked).trans rfl
theorem model_count_9039 : modelRow 9039 = 0 := by decide
theorem modelrow_agrees_9039 : modelRow 9039 = row 9039 := model_count_9039.trans row_count_9039.symm
def row_9040 : RowCertificate 9040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9040 1) := by change ¬ Nat.Prime 326904481; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17205499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9040 2) := by change Nat.Prime 326886401; exact ProjectEulerSolutions.P58PrimeNodes.p_326886401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9040 3) := by change ¬ Nat.Prime 326868321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108956107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9040 : row 9040 = 1 := (row_9040.checked).trans rfl
theorem model_count_9040 : modelRow 9040 = 1 := by decide
theorem modelrow_agrees_9040 : modelRow 9040 = row 9040 := model_count_9040.trans row_count_9040.symm
def row_9041 : RowCertificate 9041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9041 1) := by change ¬ Nat.Prime 326976807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108992269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9041 2) := by change ¬ Nat.Prime 326958725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65391745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9041 3) := by change ¬ Nat.Prime 326940643; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3174181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9041 : row 9041 = 0 := (row_9041.checked).trans rfl
theorem model_count_9041 : modelRow 9041 = 0 := by decide
theorem modelrow_agrees_9041 : modelRow 9041 = row 9041 := model_count_9041.trans row_count_9041.symm
def row_9042 : RowCertificate 9042 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9042 1) := by change Nat.Prime 327049141; exact ProjectEulerSolutions.P58PrimeNodes.p_327049141.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9042 2) := by change ¬ Nat.Prime 327031057; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19237121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9042 3) := by change ¬ Nat.Prime 327012973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46716139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9042 : row 9042 = 1 := (row_9042.checked).trans rfl
theorem model_count_9042 : modelRow 9042 = 1 := by decide
theorem modelrow_agrees_9042 : modelRow 9042 = row 9042 := model_count_9042.trans row_count_9042.symm
def row_9043 : RowCertificate 9043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9043 1) := by change ¬ Nat.Prime 327121483; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25163191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9043 2) := by change ¬ Nat.Prime 327103397; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 292841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9043 3) := by change ¬ Nat.Prime 327085311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109028437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9043 : row 9043 = 0 := (row_9043.checked).trans rfl
theorem model_count_9043 : modelRow 9043 = 0 := by decide
theorem modelrow_agrees_9043 : modelRow 9043 = row 9043 := model_count_9043.trans row_count_9043.symm
def row_9044 : RowCertificate 9044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9044 1) := by change ¬ Nat.Prime 327193833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109064611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9044 2) := by change ¬ Nat.Prime 327175745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65435149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9044 3) := by change ¬ Nat.Prime 327157657; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2166607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9044 : row 9044 = 0 := (row_9044.checked).trans rfl
theorem model_count_9044 : modelRow 9044 = 0 := by decide
theorem modelrow_agrees_9044 : modelRow 9044 = row 9044 := model_count_9044.trans row_count_9044.symm
def row_9045 : RowCertificate 9045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9045 1) := by change ¬ Nat.Prime 327266191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46752313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9045 2) := by change ¬ Nat.Prime 327248101; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 7981661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9045 3) := by change Nat.Prime 327230011; exact ProjectEulerSolutions.P58PrimeNodes.p_327230011.prime
    exact decide_eq_true prime
theorem row_count_9045 : row 9045 = 1 := (row_9045.checked).trans rfl
theorem model_count_9045 : modelRow 9045 = 1 := by decide
theorem modelrow_agrees_9045 : modelRow 9045 = row 9045 := model_count_9045.trans row_count_9045.symm
def row_9046 : RowCertificate 9046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9046 1) := by change ¬ Nat.Prime 327338557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46762651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9046 2) := by change ¬ Nat.Prime 327320465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65464093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9046 3) := by change ¬ Nat.Prime 327302373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109100791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9046 : row 9046 = 0 := (row_9046.checked).trans rfl
theorem model_count_9046 : modelRow 9046 = 0 := by decide
theorem modelrow_agrees_9046 : modelRow 9046 = row 9046 := model_count_9046.trans row_count_9046.symm
def row_9047 : RowCertificate 9047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9047 1) := by change ¬ Nat.Prime 327410931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109136977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9047 2) := by change Nat.Prime 327392837; exact ProjectEulerSolutions.P58PrimeNodes.p_327392837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9047 3) := by change ¬ Nat.Prime 327374743; exact PrattCertificates.composite_of_product (factor := 3529) (cofactor := 92767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9047 : row 9047 = 1 := (row_9047.checked).trans rfl
theorem model_count_9047 : modelRow 9047 = 1 := by decide
theorem modelrow_agrees_9047 : modelRow 9047 = row 9047 := model_count_9047.trans row_count_9047.symm
def row_9048 : RowCertificate 9048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9048 1) := by change ¬ Nat.Prime 327483313; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7615891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9048 2) := by change ¬ Nat.Prime 327465217; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6178589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9048 3) := by change ¬ Nat.Prime 327447121; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17234059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9048 : row 9048 = 0 := (row_9048.checked).trans rfl
theorem model_count_9048 : modelRow 9048 = 0 := by decide
theorem modelrow_agrees_9048 : modelRow 9048 = row 9048 := model_count_9048.trans row_count_9048.symm
def row_9049 : RowCertificate 9049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9049 1) := by change ¬ Nat.Prime 327555703; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10566313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9049 2) := by change ¬ Nat.Prime 327537605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65507521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9049 3) := by change ¬ Nat.Prime 327519507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109173169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9049 : row 9049 = 0 := (row_9049.checked).trans rfl
theorem model_count_9049 : modelRow 9049 = 0 := by decide
theorem modelrow_agrees_9049 : modelRow 9049 = row 9049 := model_count_9049.trans row_count_9049.symm
def row_9050 : RowCertificate 9050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9050 1) := by change ¬ Nat.Prime 327628101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109209367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9050 2) := by change Nat.Prime 327610001; exact ProjectEulerSolutions.P58PrimeNodes.p_327610001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9050 3) := by change ¬ Nat.Prime 327591901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46798843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9050 : row 9050 = 1 := (row_9050.checked).trans rfl
theorem model_count_9050 : modelRow 9050 = 1 := by decide
theorem modelrow_agrees_9050 : modelRow 9050 = row 9050 := model_count_9050.trans row_count_9050.symm
def row_9051 : RowCertificate 9051 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9051 1) := by change ¬ Nat.Prime 327700507; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1469509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9051 2) := by change ¬ Nat.Prime 327682405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65536481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9051 3) := by change Nat.Prime 327664303; exact ProjectEulerSolutions.P58PrimeNodes.p_327664303.prime
    exact decide_eq_true prime
theorem row_count_9051 : row 9051 = 1 := (row_9051.checked).trans rfl
theorem model_count_9051 : modelRow 9051 = 1 := by decide
theorem modelrow_agrees_9051 : modelRow 9051 = row 9051 := model_count_9051.trans row_count_9051.symm
def row_9052 : RowCertificate 9052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9052 1) := by change ¬ Nat.Prime 327772921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46824703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9052 2) := by change ¬ Nat.Prime 327754817; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25211909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9052 3) := by change ¬ Nat.Prime 327736713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109245571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9052 : row 9052 = 0 := (row_9052.checked).trans rfl
theorem model_count_9052 : modelRow 9052 = 0 := by decide
theorem modelrow_agrees_9052 : modelRow 9052 = row 9052 := model_count_9052.trans row_count_9052.symm
def row_9053 : RowCertificate 9053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9053 1) := by change ¬ Nat.Prime 327845343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109281781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9053 2) := by change ¬ Nat.Prime 327827237; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5374217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9053 3) := by change ¬ Nat.Prime 327809131; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25216087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9053 : row 9053 = 0 := (row_9053.checked).trans rfl
theorem model_count_9053 : modelRow 9053 = 0 := by decide
theorem modelrow_agrees_9053 : modelRow 9053 = row 9053 := model_count_9053.trans row_count_9053.symm
def row_9054 : RowCertificate 9054 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9054 1) := by change Nat.Prime 327917773; exact ProjectEulerSolutions.P58PrimeNodes.p_327917773.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9054 2) := by change ¬ Nat.Prime 327899665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65579933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9054 3) := by change Nat.Prime 327881557; exact ProjectEulerSolutions.P58PrimeNodes.p_327881557.prime
    exact decide_eq_true prime
theorem row_count_9054 : row 9054 = 2 := (row_9054.checked).trans rfl
theorem model_count_9054 : modelRow 9054 = 2 := by decide
theorem modelrow_agrees_9054 : modelRow 9054 = row 9054 := model_count_9054.trans row_count_9054.symm
def row_9055 : RowCertificate 9055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9055 1) := by change ¬ Nat.Prime 327990211; exact PrattCertificates.composite_of_product (factor := 1669) (cofactor := 196519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9055 2) := by change ¬ Nat.Prime 327972101; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 1664833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9055 3) := by change ¬ Nat.Prime 327953991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109317997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9055 : row 9055 = 0 := (row_9055.checked).trans rfl
theorem model_count_9055 : modelRow 9055 = 0 := by decide
theorem modelrow_agrees_9055 : modelRow 9055 = row 9055 := model_count_9055.trans row_count_9055.symm
def row_9056 : RowCertificate 9056 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9056 1) := by change ¬ Nat.Prime 328062657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109354219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9056 2) := by change ¬ Nat.Prime 328044545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65608909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9056 3) := by change ¬ Nat.Prime 328026433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46860919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9056 : row 9056 = 0 := (row_9056.checked).trans rfl
theorem model_count_9056 : modelRow 9056 = 0 := by decide
theorem modelrow_agrees_9056 : modelRow 9056 = row 9056 := model_count_9056.trans row_count_9056.symm
def row_9057 : RowCertificate 9057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9057 1) := by change ¬ Nat.Prime 328135111; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17270269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9057 2) := by change ¬ Nat.Prime 328116997; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25239769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9057 3) := by change ¬ Nat.Prime 328098883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46871269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9057 : row 9057 = 0 := (row_9057.checked).trans rfl
theorem model_count_9057 : modelRow 9057 = 0 := by decide
theorem modelrow_agrees_9057 : modelRow 9057 = row 9057 := model_count_9057.trans row_count_9057.symm
def row_9058 : RowCertificate 9058 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9058 1) := by change Nat.Prime 328207573; exact ProjectEulerSolutions.P58PrimeNodes.p_328207573.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9058 2) := by change ¬ Nat.Prime 328189457; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 644773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9058 3) := by change ¬ Nat.Prime 328171341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109390447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9058 : row 9058 = 1 := (row_9058.checked).trans rfl
theorem model_count_9058 : modelRow 9058 = 1 := by decide
theorem modelrow_agrees_9058 : modelRow 9058 = row 9058 := model_count_9058.trans row_count_9058.symm
def row_9059 : RowCertificate 9059 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9059 1) := by change ¬ Nat.Prime 328280043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109426681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9059 2) := by change ¬ Nat.Prime 328261925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65652385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9059 3) := by change Nat.Prime 328243807; exact ProjectEulerSolutions.P58PrimeNodes.p_328243807.prime
    exact decide_eq_true prime
theorem row_count_9059 : row 9059 = 1 := (row_9059.checked).trans rfl
theorem model_count_9059 : modelRow 9059 = 1 := by decide
theorem modelrow_agrees_9059 : modelRow 9059 = row 9059 := model_count_9059.trans row_count_9059.symm
def row_9060 : RowCertificate 9060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9060 1) := by change ¬ Nat.Prime 328352521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46907503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9060 2) := by change Nat.Prime 328334401; exact ProjectEulerSolutions.P58PrimeNodes.p_328334401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9060 3) := by change ¬ Nat.Prime 328316281; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8873413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9060 : row 9060 = 1 := (row_9060.checked).trans rfl
theorem model_count_9060 : modelRow 9060 = 1 := by decide
theorem modelrow_agrees_9060 : modelRow 9060 = row 9060 := model_count_9060.trans row_count_9060.symm
def row_9061 : RowCertificate 9061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9061 1) := by change ¬ Nat.Prime 328425007; exact PrattCertificates.composite_of_product (factor := 3319) (cofactor := 98953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9061 2) := by change ¬ Nat.Prime 328406885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65681377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9061 3) := by change ¬ Nat.Prime 328388763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109462921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9061 : row 9061 = 0 := (row_9061.checked).trans rfl
theorem model_count_9061 : modelRow 9061 = 0 := by decide
theorem modelrow_agrees_9061 : modelRow 9061 = row 9061 := model_count_9061.trans row_count_9061.symm
def row_9062 : RowCertificate 9062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9062 1) := by change ¬ Nat.Prime 328497501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109499167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9062 2) := by change ¬ Nat.Prime 328479377; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8877821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9062 3) := by change ¬ Nat.Prime 328461253; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 780193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9062 : row 9062 = 0 := (row_9062.checked).trans rfl
theorem model_count_9062 : modelRow 9062 = 0 := by decide
theorem modelrow_agrees_9062 : modelRow 9062 = row 9062 := model_count_9062.trans row_count_9062.symm
def row_9063 : RowCertificate 9063 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9063 1) := by change Nat.Prime 328570003; exact ProjectEulerSolutions.P58PrimeNodes.p_328570003.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9063 2) := by change ¬ Nat.Prime 328551877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19326581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9063 3) := by change ¬ Nat.Prime 328533751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46933393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9063 : row 9063 = 1 := (row_9063.checked).trans rfl
theorem model_count_9063 : modelRow 9063 = 1 := by decide
theorem modelrow_agrees_9063 : modelRow 9063 = row 9063 := model_count_9063.trans row_count_9063.symm
def row_9064 : RowCertificate 9064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9064 1) := by change ¬ Nat.Prime 328642513; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 1212703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9064 2) := by change ¬ Nat.Prime 328624385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65724877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9064 3) := by change ¬ Nat.Prime 328606257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109535419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9064 : row 9064 = 0 := (row_9064.checked).trans rfl
theorem model_count_9064 : modelRow 9064 = 0 := by decide
theorem modelrow_agrees_9064 : modelRow 9064 = row 9064 := model_count_9064.trans row_count_9064.symm
def row_9065 : RowCertificate 9065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9065 1) := by change ¬ Nat.Prime 328715031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109571677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9065 2) := by change ¬ Nat.Prime 328696901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25284377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9065 3) := by change ¬ Nat.Prime 328678771; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10602541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9065 : row 9065 = 0 := (row_9065.checked).trans rfl
theorem model_count_9065 : modelRow 9065 = 0 := by decide
theorem modelrow_agrees_9065 : modelRow 9065 = row 9065 := model_count_9065.trans row_count_9065.symm
def row_9066 : RowCertificate 9066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9066 1) := by change ¬ Nat.Prime 328787557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46969651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9066 2) := by change ¬ Nat.Prime 328769425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65753885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9066 3) := by change ¬ Nat.Prime 328751293; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25288561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9066 : row 9066 = 0 := (row_9066.checked).trans rfl
theorem model_count_9066 : modelRow 9066 = 0 := by decide
theorem modelrow_agrees_9066 : modelRow 9066 = row 9066 := model_count_9066.trans row_count_9066.symm
def row_9067 : RowCertificate 9067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9067 1) := by change ¬ Nat.Prime 328860091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46980013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9067 2) := by change ¬ Nat.Prime 328841957; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2206993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9067 3) := by change ¬ Nat.Prime 328823823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109607941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9067 : row 9067 = 0 := (row_9067.checked).trans rfl
theorem model_count_9067 : modelRow 9067 = 0 := by decide
theorem modelrow_agrees_9067 : modelRow 9067 = row 9067 := model_count_9067.trans row_count_9067.symm
def row_9068 : RowCertificate 9068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9068 1) := by change ¬ Nat.Prime 328932633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109644211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9068 2) := by change ¬ Nat.Prime 328914497; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 8889581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9068 3) := by change Nat.Prime 328896361; exact ProjectEulerSolutions.P58PrimeNodes.p_328896361.prime
    exact decide_eq_true prime
theorem row_count_9068 : row 9068 = 1 := (row_9068.checked).trans rfl
theorem model_count_9068 : modelRow 9068 = 1 := by decide
theorem modelrow_agrees_9068 : modelRow 9068 = row 9068 := model_count_9068.trans row_count_9068.symm
def row_9069 : RowCertificate 9069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9069 1) := by change ¬ Nat.Prime 329005183; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25308091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9069 2) := by change ¬ Nat.Prime 328987045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65797409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9069 3) := by change ¬ Nat.Prime 328968907; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17314153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9069 : row 9069 = 0 := (row_9069.checked).trans rfl
theorem model_count_9069 : modelRow 9069 = 0 := by decide
theorem modelrow_agrees_9069 : modelRow 9069 = row 9069 := model_count_9069.trans row_count_9069.symm
def row_9070 : RowCertificate 9070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9070 1) := by change ¬ Nat.Prime 329077741; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10615411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9070 2) := by change ¬ Nat.Prime 329059601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25312277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9070 3) := by change ¬ Nat.Prime 329041461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109680487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9070 : row 9070 = 0 := (row_9070.checked).trans rfl
theorem model_count_9070 : modelRow 9070 = 0 := by decide
theorem modelrow_agrees_9070 : modelRow 9070 = row 9070 := model_count_9070.trans row_count_9070.symm
def row_9071 : RowCertificate 9071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9071 1) := by change ¬ Nat.Prime 329150307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109716769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9071 2) := by change ¬ Nat.Prime 329132165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65826433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9071 3) := by change ¬ Nat.Prime 329114023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47016289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9071 : row 9071 = 0 := (row_9071.checked).trans rfl
theorem model_count_9071 : modelRow 9071 = 0 := by decide
theorem modelrow_agrees_9071 : modelRow 9071 = row 9071 := model_count_9071.trans row_count_9071.symm
def row_9072 : RowCertificate 9072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9072 1) := by change ¬ Nat.Prime 329222881; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25324837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9072 2) := by change Nat.Prime 329204737; exact ProjectEulerSolutions.P58PrimeNodes.p_329204737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9072 3) := by change ¬ Nat.Prime 329186593; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 498013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9072 : row 9072 = 1 := (row_9072.checked).trans rfl
theorem model_count_9072 : modelRow 9072 = 1 := by decide
theorem modelrow_agrees_9072 : modelRow 9072 = row 9072 := model_count_9072.trans row_count_9072.symm
def row_9073 : RowCertificate 9073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9073 1) := by change ¬ Nat.Prime 329295463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47042209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9073 2) := by change Nat.Prime 329277317; exact ProjectEulerSolutions.P58PrimeNodes.p_329277317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9073 3) := by change ¬ Nat.Prime 329259171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109753057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9073 : row 9073 = 1 := (row_9073.checked).trans rfl
theorem model_count_9073 : modelRow 9073 = 1 := by decide
theorem modelrow_agrees_9073 : modelRow 9073 = row 9073 := model_count_9073.trans row_count_9073.symm
def row_9074 : RowCertificate 9074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9074 1) := by change ¬ Nat.Prime 329368053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109789351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9074 2) := by change ¬ Nat.Prime 329349905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65869981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9074 3) := by change Nat.Prime 329331757; exact ProjectEulerSolutions.P58PrimeNodes.p_329331757.prime
    exact decide_eq_true prime
theorem row_count_9074 : row 9074 = 1 := (row_9074.checked).trans rfl
theorem model_count_9074 : modelRow 9074 = 1 := by decide
theorem modelrow_agrees_9074 : modelRow 9074 = row 9074 := model_count_9074.trans row_count_9074.symm
def row_9075 : RowCertificate 9075 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9075 1) := by change Nat.Prime 329440651; exact ProjectEulerSolutions.P58PrimeNodes.p_329440651.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9075 2) := by change ¬ Nat.Prime 329422501; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4512637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9075 3) := by change Nat.Prime 329404351; exact ProjectEulerSolutions.P58PrimeNodes.p_329404351.prime
    exact decide_eq_true prime
theorem row_count_9075 : row 9075 = 2 := (row_9075.checked).trans rfl
theorem model_count_9075 : modelRow 9075 = 2 := by decide
theorem modelrow_agrees_9075 : modelRow 9075 = row 9075 := model_count_9075.trans row_count_9075.symm
def row_9076 : RowCertificate 9076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9076 1) := by change ¬ Nat.Prime 329513257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17342803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9076 2) := by change ¬ Nat.Prime 329495105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65899021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9076 3) := by change ¬ Nat.Prime 329476953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109825651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9076 : row 9076 = 0 := (row_9076.checked).trans rfl
theorem model_count_9076 : modelRow 9076 = 0 := by decide
theorem modelrow_agrees_9076 : modelRow 9076 = row 9076 := model_count_9076.trans row_count_9076.symm
def row_9077 : RowCertificate 9077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9077 1) := by change ¬ Nat.Prime 329585871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109861957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9077 2) := by change ¬ Nat.Prime 329567717; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8038237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9077 3) := by change ¬ Nat.Prime 329549563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47078509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9077 : row 9077 = 0 := (row_9077.checked).trans rfl
theorem model_count_9077 : modelRow 9077 = 0 := by decide
theorem modelrow_agrees_9077 : modelRow 9077 = row 9077 := model_count_9077.trans row_count_9077.symm
def row_9078 : RowCertificate 9078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9078 1) := by change ¬ Nat.Prime 329658493; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17350447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9078 2) := by change ¬ Nat.Prime 329640337; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25356949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9078 3) := by change ¬ Nat.Prime 329622181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47088883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9078 : row 9078 = 0 := (row_9078.checked).trans rfl
theorem model_count_9078 : modelRow 9078 = 0 := by decide
theorem modelrow_agrees_9078 : modelRow 9078 = row 9078 := model_count_9078.trans row_count_9078.symm
def row_9079 : RowCertificate 9079 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9079 1) := by change Nat.Prime 329731123; exact ProjectEulerSolutions.P58PrimeNodes.p_329731123.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9079 2) := by change ¬ Nat.Prime 329712965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65942593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9079 3) := by change ¬ Nat.Prime 329694807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109898269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9079 : row 9079 = 1 := (row_9079.checked).trans rfl
theorem model_count_9079 : modelRow 9079 = 1 := by decide
theorem modelrow_agrees_9079 : modelRow 9079 = row 9079 := model_count_9079.trans row_count_9079.symm
def row_9080 : RowCertificate 9080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9080 1) := by change ¬ Nat.Prime 329803761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109934587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9080 2) := by change ¬ Nat.Prime 329785601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19399153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9080 3) := by change ¬ Nat.Prime 329767441; exact PrattCertificates.composite_of_product (factor := 5683) (cofactor := 58027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9080 : row 9080 = 0 := (row_9080.checked).trans rfl
theorem model_count_9080 : modelRow 9080 = 0 := by decide
theorem modelrow_agrees_9080 : modelRow 9080 = row 9080 := model_count_9080.trans row_count_9080.symm
def row_9081 : RowCertificate 9081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9081 1) := by change ¬ Nat.Prime 329876407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47125201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9081 2) := by change ¬ Nat.Prime 329858245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65971649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9081 3) := by change ¬ Nat.Prime 329840083; exact PrattCertificates.composite_of_product (factor := 1627) (cofactor := 202729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9081 : row 9081 = 0 := (row_9081.checked).trans rfl
theorem model_count_9081 : modelRow 9081 = 0 := by decide
theorem modelrow_agrees_9081 : modelRow 9081 = row 9081 := model_count_9081.trans row_count_9081.symm
def row_9082 : RowCertificate 9082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9082 1) := by change ¬ Nat.Prime 329949061; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25380697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9082 2) := by change Nat.Prime 329930897; exact ProjectEulerSolutions.P58PrimeNodes.p_329930897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9082 3) := by change ¬ Nat.Prime 329912733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 109970911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9082 : row 9082 = 1 := (row_9082.checked).trans rfl
theorem model_count_9082 : modelRow 9082 = 1 := by decide
theorem modelrow_agrees_9082 : modelRow 9082 = row 9082 := model_count_9082.trans row_count_9082.symm
def row_9083 : RowCertificate 9083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9083 1) := by change ¬ Nat.Prime 330021723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 110007241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9083 2) := by change ¬ Nat.Prime 330003557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25384889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9083 3) := by change ¬ Nat.Prime 329985391; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1191283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9083 : row 9083 = 0 := (row_9083.checked).trans rfl
theorem model_count_9083 : modelRow 9083 = 0 := by decide
theorem modelrow_agrees_9083 : modelRow 9083 = row 9083 := model_count_9083.trans row_count_9083.symm
def row_9084 : RowCertificate 9084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9084 1) := by change ¬ Nat.Prime 330094393; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4521841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9084 2) := by change ¬ Nat.Prime 330076225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 66015245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9084 3) := by change ¬ Nat.Prime 330058057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47151151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9084 : row 9084 = 0 := (row_9084.checked).trans rfl
theorem model_count_9084 : modelRow 9084 = 0 := by decide
theorem modelrow_agrees_9084 : modelRow 9084 = row 9084 := model_count_9084.trans row_count_9084.symm
def row_9085 : RowCertificate 9085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9085 1) := by change ¬ Nat.Prime 330167071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25397467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9085 2) := by change ¬ Nat.Prime 330148901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3268801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9085 3) := by change ¬ Nat.Prime 330130731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 110043577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9085 : row 9085 = 0 := (row_9085.checked).trans rfl
theorem model_count_9085 : modelRow 9085 = 0 := by decide
theorem modelrow_agrees_9085 : modelRow 9085 = row 9085 := model_count_9085.trans row_count_9085.symm
def row_9086 : RowCertificate 9086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9086 1) := by change ¬ Nat.Prime 330239757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 110079919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9086 2) := by change ¬ Nat.Prime 330221585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 66044317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9086 3) := by change ¬ Nat.Prime 330203413; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17379127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9086 : row 9086 = 0 := (row_9086.checked).trans rfl
theorem model_count_9086 : modelRow 9086 = 0 := by decide
theorem modelrow_agrees_9086 : modelRow 9086 = row 9086 := model_count_9086.trans row_count_9086.symm
def row_9087 : RowCertificate 9087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9087 1) := by change ¬ Nat.Prime 330312451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47187493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9087 2) := by change Nat.Prime 330294277; exact ProjectEulerSolutions.P58PrimeNodes.p_330294277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9087 3) := by change ¬ Nat.Prime 330276103; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 997813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9087 : row 9087 = 1 := (row_9087.checked).trans rfl
theorem model_count_9087 : modelRow 9087 = 1 := by decide
theorem modelrow_agrees_9087 : modelRow 9087 = row 9087 := model_count_9087.trans row_count_9087.symm
def row_9088 : RowCertificate 9088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9088 1) := by change ¬ Nat.Prime 330385153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47197879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9088 2) := by change Nat.Prime 330366977; exact ProjectEulerSolutions.P58PrimeNodes.p_330366977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9088 3) := by change ¬ Nat.Prime 330348801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 110116267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9088 : row 9088 = 1 := (row_9088.checked).trans rfl
theorem model_count_9088 : modelRow 9088 = 1 := by decide
theorem modelrow_agrees_9088 : modelRow 9088 = row 9088 := model_count_9088.trans row_count_9088.symm
private def atomic_8961 : CountSegment 1 10 13120 8960 8961 3741 3741 := CountSegment.single (first := 8960) (initial := 3741) row_count_8961 (by decide)
private theorem atomic_8961_agreement : RowAgreement 8960 8961 := RowAgreement.single (first := 8960) modelrow_agrees_8961
private def atomic_8962 : CountSegment 1 10 13120 8961 8962 3741 3741 := CountSegment.single (first := 8961) (initial := 3741) row_count_8962 (by decide)
private theorem atomic_8962_agreement : RowAgreement 8961 8962 := RowAgreement.single (first := 8961) modelrow_agrees_8962
private def atomic_8963 : CountSegment 1 10 13120 8962 8963 3741 3742 := CountSegment.single (first := 8962) (initial := 3741) row_count_8963 (by decide)
private theorem atomic_8963_agreement : RowAgreement 8962 8963 := RowAgreement.single (first := 8962) modelrow_agrees_8963
private def atomic_8964 : CountSegment 1 10 13120 8963 8964 3742 3742 := CountSegment.single (first := 8963) (initial := 3742) row_count_8964 (by decide)
private theorem atomic_8964_agreement : RowAgreement 8963 8964 := RowAgreement.single (first := 8963) modelrow_agrees_8964
private def atomic_8965 : CountSegment 1 10 13120 8964 8965 3742 3743 := CountSegment.single (first := 8964) (initial := 3742) row_count_8965 (by decide)
private theorem atomic_8965_agreement : RowAgreement 8964 8965 := RowAgreement.single (first := 8964) modelrow_agrees_8965
private def atomic_8966 : CountSegment 1 10 13120 8965 8966 3743 3743 := CountSegment.single (first := 8965) (initial := 3743) row_count_8966 (by decide)
private theorem atomic_8966_agreement : RowAgreement 8965 8966 := RowAgreement.single (first := 8965) modelrow_agrees_8966
private def atomic_8967 : CountSegment 1 10 13120 8966 8967 3743 3744 := CountSegment.single (first := 8966) (initial := 3743) row_count_8967 (by decide)
private theorem atomic_8967_agreement : RowAgreement 8966 8967 := RowAgreement.single (first := 8966) modelrow_agrees_8967
private def atomic_8968 : CountSegment 1 10 13120 8967 8968 3744 3745 := CountSegment.single (first := 8967) (initial := 3744) row_count_8968 (by decide)
private theorem atomic_8968_agreement : RowAgreement 8967 8968 := RowAgreement.single (first := 8967) modelrow_agrees_8968
private def atomic_8969 : CountSegment 1 10 13120 8968 8969 3745 3745 := CountSegment.single (first := 8968) (initial := 3745) row_count_8969 (by decide)
private theorem atomic_8969_agreement : RowAgreement 8968 8969 := RowAgreement.single (first := 8968) modelrow_agrees_8969
private def atomic_8970 : CountSegment 1 10 13120 8969 8970 3745 3746 := CountSegment.single (first := 8969) (initial := 3745) row_count_8970 (by decide)
private theorem atomic_8970_agreement : RowAgreement 8969 8970 := RowAgreement.single (first := 8969) modelrow_agrees_8970
private def atomic_8971 : CountSegment 1 10 13120 8970 8971 3746 3746 := CountSegment.single (first := 8970) (initial := 3746) row_count_8971 (by decide)
private theorem atomic_8971_agreement : RowAgreement 8970 8971 := RowAgreement.single (first := 8970) modelrow_agrees_8971
private def atomic_8972 : CountSegment 1 10 13120 8971 8972 3746 3746 := CountSegment.single (first := 8971) (initial := 3746) row_count_8972 (by decide)
private theorem atomic_8972_agreement : RowAgreement 8971 8972 := RowAgreement.single (first := 8971) modelrow_agrees_8972
private def atomic_8973 : CountSegment 1 10 13120 8972 8973 3746 3748 := CountSegment.single (first := 8972) (initial := 3746) row_count_8973 (by decide)
private theorem atomic_8973_agreement : RowAgreement 8972 8973 := RowAgreement.single (first := 8972) modelrow_agrees_8973
private def atomic_8974 : CountSegment 1 10 13120 8973 8974 3748 3749 := CountSegment.single (first := 8973) (initial := 3748) row_count_8974 (by decide)
private theorem atomic_8974_agreement : RowAgreement 8973 8974 := RowAgreement.single (first := 8973) modelrow_agrees_8974
private def atomic_8975 : CountSegment 1 10 13120 8974 8975 3749 3750 := CountSegment.single (first := 8974) (initial := 3749) row_count_8975 (by decide)
private theorem atomic_8975_agreement : RowAgreement 8974 8975 := RowAgreement.single (first := 8974) modelrow_agrees_8975
private def atomic_8976 : CountSegment 1 10 13120 8975 8976 3750 3750 := CountSegment.single (first := 8975) (initial := 3750) row_count_8976 (by decide)
private theorem atomic_8976_agreement : RowAgreement 8975 8976 := RowAgreement.single (first := 8975) modelrow_agrees_8976
private def atomic_8977 : CountSegment 1 10 13120 8976 8977 3750 3751 := CountSegment.single (first := 8976) (initial := 3750) row_count_8977 (by decide)
private theorem atomic_8977_agreement : RowAgreement 8976 8977 := RowAgreement.single (first := 8976) modelrow_agrees_8977
private def atomic_8978 : CountSegment 1 10 13120 8977 8978 3751 3751 := CountSegment.single (first := 8977) (initial := 3751) row_count_8978 (by decide)
private theorem atomic_8978_agreement : RowAgreement 8977 8978 := RowAgreement.single (first := 8977) modelrow_agrees_8978
private def atomic_8979 : CountSegment 1 10 13120 8978 8979 3751 3751 := CountSegment.single (first := 8978) (initial := 3751) row_count_8979 (by decide)
private theorem atomic_8979_agreement : RowAgreement 8978 8979 := RowAgreement.single (first := 8978) modelrow_agrees_8979
private def atomic_8980 : CountSegment 1 10 13120 8979 8980 3751 3751 := CountSegment.single (first := 8979) (initial := 3751) row_count_8980 (by decide)
private theorem atomic_8980_agreement : RowAgreement 8979 8980 := RowAgreement.single (first := 8979) modelrow_agrees_8980
private def atomic_8981 : CountSegment 1 10 13120 8980 8981 3751 3752 := CountSegment.single (first := 8980) (initial := 3751) row_count_8981 (by decide)
private theorem atomic_8981_agreement : RowAgreement 8980 8981 := RowAgreement.single (first := 8980) modelrow_agrees_8981
private def atomic_8982 : CountSegment 1 10 13120 8981 8982 3752 3754 := CountSegment.single (first := 8981) (initial := 3752) row_count_8982 (by decide)
private theorem atomic_8982_agreement : RowAgreement 8981 8982 := RowAgreement.single (first := 8981) modelrow_agrees_8982
private def atomic_8983 : CountSegment 1 10 13120 8982 8983 3754 3754 := CountSegment.single (first := 8982) (initial := 3754) row_count_8983 (by decide)
private theorem atomic_8983_agreement : RowAgreement 8982 8983 := RowAgreement.single (first := 8982) modelrow_agrees_8983
private def atomic_8984 : CountSegment 1 10 13120 8983 8984 3754 3754 := CountSegment.single (first := 8983) (initial := 3754) row_count_8984 (by decide)
private theorem atomic_8984_agreement : RowAgreement 8983 8984 := RowAgreement.single (first := 8983) modelrow_agrees_8984
private def atomic_8985 : CountSegment 1 10 13120 8984 8985 3754 3755 := CountSegment.single (first := 8984) (initial := 3754) row_count_8985 (by decide)
private theorem atomic_8985_agreement : RowAgreement 8984 8985 := RowAgreement.single (first := 8984) modelrow_agrees_8985
private def atomic_8986 : CountSegment 1 10 13120 8985 8986 3755 3755 := CountSegment.single (first := 8985) (initial := 3755) row_count_8986 (by decide)
private theorem atomic_8986_agreement : RowAgreement 8985 8986 := RowAgreement.single (first := 8985) modelrow_agrees_8986
private def atomic_8987 : CountSegment 1 10 13120 8986 8987 3755 3755 := CountSegment.single (first := 8986) (initial := 3755) row_count_8987 (by decide)
private theorem atomic_8987_agreement : RowAgreement 8986 8987 := RowAgreement.single (first := 8986) modelrow_agrees_8987
private def atomic_8988 : CountSegment 1 10 13120 8987 8988 3755 3756 := CountSegment.single (first := 8987) (initial := 3755) row_count_8988 (by decide)
private theorem atomic_8988_agreement : RowAgreement 8987 8988 := RowAgreement.single (first := 8987) modelrow_agrees_8988
private def atomic_8989 : CountSegment 1 10 13120 8988 8989 3756 3756 := CountSegment.single (first := 8988) (initial := 3756) row_count_8989 (by decide)
private theorem atomic_8989_agreement : RowAgreement 8988 8989 := RowAgreement.single (first := 8988) modelrow_agrees_8989
private def atomic_8990 : CountSegment 1 10 13120 8989 8990 3756 3757 := CountSegment.single (first := 8989) (initial := 3756) row_count_8990 (by decide)
private theorem atomic_8990_agreement : RowAgreement 8989 8990 := RowAgreement.single (first := 8989) modelrow_agrees_8990
private def atomic_8991 : CountSegment 1 10 13120 8990 8991 3757 3757 := CountSegment.single (first := 8990) (initial := 3757) row_count_8991 (by decide)
private theorem atomic_8991_agreement : RowAgreement 8990 8991 := RowAgreement.single (first := 8990) modelrow_agrees_8991
private def atomic_8992 : CountSegment 1 10 13120 8991 8992 3757 3757 := CountSegment.single (first := 8991) (initial := 3757) row_count_8992 (by decide)
private theorem atomic_8992_agreement : RowAgreement 8991 8992 := RowAgreement.single (first := 8991) modelrow_agrees_8992
private def atomic_8993 : CountSegment 1 10 13120 8992 8993 3757 3758 := CountSegment.single (first := 8992) (initial := 3757) row_count_8993 (by decide)
private theorem atomic_8993_agreement : RowAgreement 8992 8993 := RowAgreement.single (first := 8992) modelrow_agrees_8993
private def atomic_8994 : CountSegment 1 10 13120 8993 8994 3758 3758 := CountSegment.single (first := 8993) (initial := 3758) row_count_8994 (by decide)
private theorem atomic_8994_agreement : RowAgreement 8993 8994 := RowAgreement.single (first := 8993) modelrow_agrees_8994
private def atomic_8995 : CountSegment 1 10 13120 8994 8995 3758 3759 := CountSegment.single (first := 8994) (initial := 3758) row_count_8995 (by decide)
private theorem atomic_8995_agreement : RowAgreement 8994 8995 := RowAgreement.single (first := 8994) modelrow_agrees_8995
private def atomic_8996 : CountSegment 1 10 13120 8995 8996 3759 3759 := CountSegment.single (first := 8995) (initial := 3759) row_count_8996 (by decide)
private theorem atomic_8996_agreement : RowAgreement 8995 8996 := RowAgreement.single (first := 8995) modelrow_agrees_8996
private def atomic_8997 : CountSegment 1 10 13120 8996 8997 3759 3759 := CountSegment.single (first := 8996) (initial := 3759) row_count_8997 (by decide)
private theorem atomic_8997_agreement : RowAgreement 8996 8997 := RowAgreement.single (first := 8996) modelrow_agrees_8997
private def atomic_8998 : CountSegment 1 10 13120 8997 8998 3759 3760 := CountSegment.single (first := 8997) (initial := 3759) row_count_8998 (by decide)
private theorem atomic_8998_agreement : RowAgreement 8997 8998 := RowAgreement.single (first := 8997) modelrow_agrees_8998
private def atomic_8999 : CountSegment 1 10 13120 8998 8999 3760 3760 := CountSegment.single (first := 8998) (initial := 3760) row_count_8999 (by decide)
private theorem atomic_8999_agreement : RowAgreement 8998 8999 := RowAgreement.single (first := 8998) modelrow_agrees_8999
private def atomic_9000 : CountSegment 1 10 13120 8999 9000 3760 3760 := CountSegment.single (first := 8999) (initial := 3760) row_count_9000 (by decide)
private theorem atomic_9000_agreement : RowAgreement 8999 9000 := RowAgreement.single (first := 8999) modelrow_agrees_9000
private def atomic_9001 : CountSegment 1 10 13120 9000 9001 3760 3760 := CountSegment.single (first := 9000) (initial := 3760) row_count_9001 (by decide)
private theorem atomic_9001_agreement : RowAgreement 9000 9001 := RowAgreement.single (first := 9000) modelrow_agrees_9001
private def atomic_9002 : CountSegment 1 10 13120 9001 9002 3760 3761 := CountSegment.single (first := 9001) (initial := 3760) row_count_9002 (by decide)
private theorem atomic_9002_agreement : RowAgreement 9001 9002 := RowAgreement.single (first := 9001) modelrow_agrees_9002
private def atomic_9003 : CountSegment 1 10 13120 9002 9003 3761 3761 := CountSegment.single (first := 9002) (initial := 3761) row_count_9003 (by decide)
private theorem atomic_9003_agreement : RowAgreement 9002 9003 := RowAgreement.single (first := 9002) modelrow_agrees_9003
private def atomic_9004 : CountSegment 1 10 13120 9003 9004 3761 3761 := CountSegment.single (first := 9003) (initial := 3761) row_count_9004 (by decide)
private theorem atomic_9004_agreement : RowAgreement 9003 9004 := RowAgreement.single (first := 9003) modelrow_agrees_9004
private def atomic_9005 : CountSegment 1 10 13120 9004 9005 3761 3761 := CountSegment.single (first := 9004) (initial := 3761) row_count_9005 (by decide)
private theorem atomic_9005_agreement : RowAgreement 9004 9005 := RowAgreement.single (first := 9004) modelrow_agrees_9005
private def atomic_9006 : CountSegment 1 10 13120 9005 9006 3761 3762 := CountSegment.single (first := 9005) (initial := 3761) row_count_9006 (by decide)
private theorem atomic_9006_agreement : RowAgreement 9005 9006 := RowAgreement.single (first := 9005) modelrow_agrees_9006
private def atomic_9007 : CountSegment 1 10 13120 9006 9007 3762 3763 := CountSegment.single (first := 9006) (initial := 3762) row_count_9007 (by decide)
private theorem atomic_9007_agreement : RowAgreement 9006 9007 := RowAgreement.single (first := 9006) modelrow_agrees_9007
private def atomic_9008 : CountSegment 1 10 13120 9007 9008 3763 3763 := CountSegment.single (first := 9007) (initial := 3763) row_count_9008 (by decide)
private theorem atomic_9008_agreement : RowAgreement 9007 9008 := RowAgreement.single (first := 9007) modelrow_agrees_9008
private def atomic_9009 : CountSegment 1 10 13120 9008 9009 3763 3764 := CountSegment.single (first := 9008) (initial := 3763) row_count_9009 (by decide)
private theorem atomic_9009_agreement : RowAgreement 9008 9009 := RowAgreement.single (first := 9008) modelrow_agrees_9009
private def atomic_9010 : CountSegment 1 10 13120 9009 9010 3764 3764 := CountSegment.single (first := 9009) (initial := 3764) row_count_9010 (by decide)
private theorem atomic_9010_agreement : RowAgreement 9009 9010 := RowAgreement.single (first := 9009) modelrow_agrees_9010
private def atomic_9011 : CountSegment 1 10 13120 9010 9011 3764 3764 := CountSegment.single (first := 9010) (initial := 3764) row_count_9011 (by decide)
private theorem atomic_9011_agreement : RowAgreement 9010 9011 := RowAgreement.single (first := 9010) modelrow_agrees_9011
private def atomic_9012 : CountSegment 1 10 13120 9011 9012 3764 3765 := CountSegment.single (first := 9011) (initial := 3764) row_count_9012 (by decide)
private theorem atomic_9012_agreement : RowAgreement 9011 9012 := RowAgreement.single (first := 9011) modelrow_agrees_9012
private def atomic_9013 : CountSegment 1 10 13120 9012 9013 3765 3765 := CountSegment.single (first := 9012) (initial := 3765) row_count_9013 (by decide)
private theorem atomic_9013_agreement : RowAgreement 9012 9013 := RowAgreement.single (first := 9012) modelrow_agrees_9013
private def atomic_9014 : CountSegment 1 10 13120 9013 9014 3765 3765 := CountSegment.single (first := 9013) (initial := 3765) row_count_9014 (by decide)
private theorem atomic_9014_agreement : RowAgreement 9013 9014 := RowAgreement.single (first := 9013) modelrow_agrees_9014
private def atomic_9015 : CountSegment 1 10 13120 9014 9015 3765 3765 := CountSegment.single (first := 9014) (initial := 3765) row_count_9015 (by decide)
private theorem atomic_9015_agreement : RowAgreement 9014 9015 := RowAgreement.single (first := 9014) modelrow_agrees_9015
private def atomic_9016 : CountSegment 1 10 13120 9015 9016 3765 3766 := CountSegment.single (first := 9015) (initial := 3765) row_count_9016 (by decide)
private theorem atomic_9016_agreement : RowAgreement 9015 9016 := RowAgreement.single (first := 9015) modelrow_agrees_9016
private def atomic_9017 : CountSegment 1 10 13120 9016 9017 3766 3767 := CountSegment.single (first := 9016) (initial := 3766) row_count_9017 (by decide)
private theorem atomic_9017_agreement : RowAgreement 9016 9017 := RowAgreement.single (first := 9016) modelrow_agrees_9017
private def atomic_9018 : CountSegment 1 10 13120 9017 9018 3767 3767 := CountSegment.single (first := 9017) (initial := 3767) row_count_9018 (by decide)
private theorem atomic_9018_agreement : RowAgreement 9017 9018 := RowAgreement.single (first := 9017) modelrow_agrees_9018
private def atomic_9019 : CountSegment 1 10 13120 9018 9019 3767 3767 := CountSegment.single (first := 9018) (initial := 3767) row_count_9019 (by decide)
private theorem atomic_9019_agreement : RowAgreement 9018 9019 := RowAgreement.single (first := 9018) modelrow_agrees_9019
private def atomic_9020 : CountSegment 1 10 13120 9019 9020 3767 3768 := CountSegment.single (first := 9019) (initial := 3767) row_count_9020 (by decide)
private theorem atomic_9020_agreement : RowAgreement 9019 9020 := RowAgreement.single (first := 9019) modelrow_agrees_9020
private def atomic_9021 : CountSegment 1 10 13120 9020 9021 3768 3768 := CountSegment.single (first := 9020) (initial := 3768) row_count_9021 (by decide)
private theorem atomic_9021_agreement : RowAgreement 9020 9021 := RowAgreement.single (first := 9020) modelrow_agrees_9021
private def atomic_9022 : CountSegment 1 10 13120 9021 9022 3768 3769 := CountSegment.single (first := 9021) (initial := 3768) row_count_9022 (by decide)
private theorem atomic_9022_agreement : RowAgreement 9021 9022 := RowAgreement.single (first := 9021) modelrow_agrees_9022
private def atomic_9023 : CountSegment 1 10 13120 9022 9023 3769 3770 := CountSegment.single (first := 9022) (initial := 3769) row_count_9023 (by decide)
private theorem atomic_9023_agreement : RowAgreement 9022 9023 := RowAgreement.single (first := 9022) modelrow_agrees_9023
private def atomic_9024 : CountSegment 1 10 13120 9023 9024 3770 3770 := CountSegment.single (first := 9023) (initial := 3770) row_count_9024 (by decide)
private theorem atomic_9024_agreement : RowAgreement 9023 9024 := RowAgreement.single (first := 9023) modelrow_agrees_9024
private def atomic_9025 : CountSegment 1 10 13120 9024 9025 3770 3770 := CountSegment.single (first := 9024) (initial := 3770) row_count_9025 (by decide)
private theorem atomic_9025_agreement : RowAgreement 9024 9025 := RowAgreement.single (first := 9024) modelrow_agrees_9025
private def atomic_9026 : CountSegment 1 10 13120 9025 9026 3770 3770 := CountSegment.single (first := 9025) (initial := 3770) row_count_9026 (by decide)
private theorem atomic_9026_agreement : RowAgreement 9025 9026 := RowAgreement.single (first := 9025) modelrow_agrees_9026
private def atomic_9027 : CountSegment 1 10 13120 9026 9027 3770 3771 := CountSegment.single (first := 9026) (initial := 3770) row_count_9027 (by decide)
private theorem atomic_9027_agreement : RowAgreement 9026 9027 := RowAgreement.single (first := 9026) modelrow_agrees_9027
private def atomic_9028 : CountSegment 1 10 13120 9027 9028 3771 3771 := CountSegment.single (first := 9027) (initial := 3771) row_count_9028 (by decide)
private theorem atomic_9028_agreement : RowAgreement 9027 9028 := RowAgreement.single (first := 9027) modelrow_agrees_9028
private def atomic_9029 : CountSegment 1 10 13120 9028 9029 3771 3771 := CountSegment.single (first := 9028) (initial := 3771) row_count_9029 (by decide)
private theorem atomic_9029_agreement : RowAgreement 9028 9029 := RowAgreement.single (first := 9028) modelrow_agrees_9029
private def atomic_9030 : CountSegment 1 10 13120 9029 9030 3771 3772 := CountSegment.single (first := 9029) (initial := 3771) row_count_9030 (by decide)
private theorem atomic_9030_agreement : RowAgreement 9029 9030 := RowAgreement.single (first := 9029) modelrow_agrees_9030
private def atomic_9031 : CountSegment 1 10 13120 9030 9031 3772 3772 := CountSegment.single (first := 9030) (initial := 3772) row_count_9031 (by decide)
private theorem atomic_9031_agreement : RowAgreement 9030 9031 := RowAgreement.single (first := 9030) modelrow_agrees_9031
private def atomic_9032 : CountSegment 1 10 13120 9031 9032 3772 3773 := CountSegment.single (first := 9031) (initial := 3772) row_count_9032 (by decide)
private theorem atomic_9032_agreement : RowAgreement 9031 9032 := RowAgreement.single (first := 9031) modelrow_agrees_9032
private def atomic_9033 : CountSegment 1 10 13120 9032 9033 3773 3775 := CountSegment.single (first := 9032) (initial := 3773) row_count_9033 (by decide)
private theorem atomic_9033_agreement : RowAgreement 9032 9033 := RowAgreement.single (first := 9032) modelrow_agrees_9033
private def atomic_9034 : CountSegment 1 10 13120 9033 9034 3775 3775 := CountSegment.single (first := 9033) (initial := 3775) row_count_9034 (by decide)
private theorem atomic_9034_agreement : RowAgreement 9033 9034 := RowAgreement.single (first := 9033) modelrow_agrees_9034
private def atomic_9035 : CountSegment 1 10 13120 9034 9035 3775 3776 := CountSegment.single (first := 9034) (initial := 3775) row_count_9035 (by decide)
private theorem atomic_9035_agreement : RowAgreement 9034 9035 := RowAgreement.single (first := 9034) modelrow_agrees_9035
private def atomic_9036 : CountSegment 1 10 13120 9035 9036 3776 3777 := CountSegment.single (first := 9035) (initial := 3776) row_count_9036 (by decide)
private theorem atomic_9036_agreement : RowAgreement 9035 9036 := RowAgreement.single (first := 9035) modelrow_agrees_9036
private def atomic_9037 : CountSegment 1 10 13120 9036 9037 3777 3778 := CountSegment.single (first := 9036) (initial := 3777) row_count_9037 (by decide)
private theorem atomic_9037_agreement : RowAgreement 9036 9037 := RowAgreement.single (first := 9036) modelrow_agrees_9037
private def atomic_9038 : CountSegment 1 10 13120 9037 9038 3778 3780 := CountSegment.single (first := 9037) (initial := 3778) row_count_9038 (by decide)
private theorem atomic_9038_agreement : RowAgreement 9037 9038 := RowAgreement.single (first := 9037) modelrow_agrees_9038
private def atomic_9039 : CountSegment 1 10 13120 9038 9039 3780 3780 := CountSegment.single (first := 9038) (initial := 3780) row_count_9039 (by decide)
private theorem atomic_9039_agreement : RowAgreement 9038 9039 := RowAgreement.single (first := 9038) modelrow_agrees_9039
private def atomic_9040 : CountSegment 1 10 13120 9039 9040 3780 3781 := CountSegment.single (first := 9039) (initial := 3780) row_count_9040 (by decide)
private theorem atomic_9040_agreement : RowAgreement 9039 9040 := RowAgreement.single (first := 9039) modelrow_agrees_9040
private def atomic_9041 : CountSegment 1 10 13120 9040 9041 3781 3781 := CountSegment.single (first := 9040) (initial := 3781) row_count_9041 (by decide)
private theorem atomic_9041_agreement : RowAgreement 9040 9041 := RowAgreement.single (first := 9040) modelrow_agrees_9041
private def atomic_9042 : CountSegment 1 10 13120 9041 9042 3781 3782 := CountSegment.single (first := 9041) (initial := 3781) row_count_9042 (by decide)
private theorem atomic_9042_agreement : RowAgreement 9041 9042 := RowAgreement.single (first := 9041) modelrow_agrees_9042
private def atomic_9043 : CountSegment 1 10 13120 9042 9043 3782 3782 := CountSegment.single (first := 9042) (initial := 3782) row_count_9043 (by decide)
private theorem atomic_9043_agreement : RowAgreement 9042 9043 := RowAgreement.single (first := 9042) modelrow_agrees_9043
private def atomic_9044 : CountSegment 1 10 13120 9043 9044 3782 3782 := CountSegment.single (first := 9043) (initial := 3782) row_count_9044 (by decide)
private theorem atomic_9044_agreement : RowAgreement 9043 9044 := RowAgreement.single (first := 9043) modelrow_agrees_9044
private def atomic_9045 : CountSegment 1 10 13120 9044 9045 3782 3783 := CountSegment.single (first := 9044) (initial := 3782) row_count_9045 (by decide)
private theorem atomic_9045_agreement : RowAgreement 9044 9045 := RowAgreement.single (first := 9044) modelrow_agrees_9045
private def atomic_9046 : CountSegment 1 10 13120 9045 9046 3783 3783 := CountSegment.single (first := 9045) (initial := 3783) row_count_9046 (by decide)
private theorem atomic_9046_agreement : RowAgreement 9045 9046 := RowAgreement.single (first := 9045) modelrow_agrees_9046
private def atomic_9047 : CountSegment 1 10 13120 9046 9047 3783 3784 := CountSegment.single (first := 9046) (initial := 3783) row_count_9047 (by decide)
private theorem atomic_9047_agreement : RowAgreement 9046 9047 := RowAgreement.single (first := 9046) modelrow_agrees_9047
private def atomic_9048 : CountSegment 1 10 13120 9047 9048 3784 3784 := CountSegment.single (first := 9047) (initial := 3784) row_count_9048 (by decide)
private theorem atomic_9048_agreement : RowAgreement 9047 9048 := RowAgreement.single (first := 9047) modelrow_agrees_9048
private def atomic_9049 : CountSegment 1 10 13120 9048 9049 3784 3784 := CountSegment.single (first := 9048) (initial := 3784) row_count_9049 (by decide)
private theorem atomic_9049_agreement : RowAgreement 9048 9049 := RowAgreement.single (first := 9048) modelrow_agrees_9049
private def atomic_9050 : CountSegment 1 10 13120 9049 9050 3784 3785 := CountSegment.single (first := 9049) (initial := 3784) row_count_9050 (by decide)
private theorem atomic_9050_agreement : RowAgreement 9049 9050 := RowAgreement.single (first := 9049) modelrow_agrees_9050
private def atomic_9051 : CountSegment 1 10 13120 9050 9051 3785 3786 := CountSegment.single (first := 9050) (initial := 3785) row_count_9051 (by decide)
private theorem atomic_9051_agreement : RowAgreement 9050 9051 := RowAgreement.single (first := 9050) modelrow_agrees_9051
private def atomic_9052 : CountSegment 1 10 13120 9051 9052 3786 3786 := CountSegment.single (first := 9051) (initial := 3786) row_count_9052 (by decide)
private theorem atomic_9052_agreement : RowAgreement 9051 9052 := RowAgreement.single (first := 9051) modelrow_agrees_9052
private def atomic_9053 : CountSegment 1 10 13120 9052 9053 3786 3786 := CountSegment.single (first := 9052) (initial := 3786) row_count_9053 (by decide)
private theorem atomic_9053_agreement : RowAgreement 9052 9053 := RowAgreement.single (first := 9052) modelrow_agrees_9053
private def atomic_9054 : CountSegment 1 10 13120 9053 9054 3786 3788 := CountSegment.single (first := 9053) (initial := 3786) row_count_9054 (by decide)
private theorem atomic_9054_agreement : RowAgreement 9053 9054 := RowAgreement.single (first := 9053) modelrow_agrees_9054
private def atomic_9055 : CountSegment 1 10 13120 9054 9055 3788 3788 := CountSegment.single (first := 9054) (initial := 3788) row_count_9055 (by decide)
private theorem atomic_9055_agreement : RowAgreement 9054 9055 := RowAgreement.single (first := 9054) modelrow_agrees_9055
private def atomic_9056 : CountSegment 1 10 13120 9055 9056 3788 3788 := CountSegment.single (first := 9055) (initial := 3788) row_count_9056 (by decide)
private theorem atomic_9056_agreement : RowAgreement 9055 9056 := RowAgreement.single (first := 9055) modelrow_agrees_9056
private def atomic_9057 : CountSegment 1 10 13120 9056 9057 3788 3788 := CountSegment.single (first := 9056) (initial := 3788) row_count_9057 (by decide)
private theorem atomic_9057_agreement : RowAgreement 9056 9057 := RowAgreement.single (first := 9056) modelrow_agrees_9057
private def atomic_9058 : CountSegment 1 10 13120 9057 9058 3788 3789 := CountSegment.single (first := 9057) (initial := 3788) row_count_9058 (by decide)
private theorem atomic_9058_agreement : RowAgreement 9057 9058 := RowAgreement.single (first := 9057) modelrow_agrees_9058
private def atomic_9059 : CountSegment 1 10 13120 9058 9059 3789 3790 := CountSegment.single (first := 9058) (initial := 3789) row_count_9059 (by decide)
private theorem atomic_9059_agreement : RowAgreement 9058 9059 := RowAgreement.single (first := 9058) modelrow_agrees_9059
private def atomic_9060 : CountSegment 1 10 13120 9059 9060 3790 3791 := CountSegment.single (first := 9059) (initial := 3790) row_count_9060 (by decide)
private theorem atomic_9060_agreement : RowAgreement 9059 9060 := RowAgreement.single (first := 9059) modelrow_agrees_9060
private def atomic_9061 : CountSegment 1 10 13120 9060 9061 3791 3791 := CountSegment.single (first := 9060) (initial := 3791) row_count_9061 (by decide)
private theorem atomic_9061_agreement : RowAgreement 9060 9061 := RowAgreement.single (first := 9060) modelrow_agrees_9061
private def atomic_9062 : CountSegment 1 10 13120 9061 9062 3791 3791 := CountSegment.single (first := 9061) (initial := 3791) row_count_9062 (by decide)
private theorem atomic_9062_agreement : RowAgreement 9061 9062 := RowAgreement.single (first := 9061) modelrow_agrees_9062
private def atomic_9063 : CountSegment 1 10 13120 9062 9063 3791 3792 := CountSegment.single (first := 9062) (initial := 3791) row_count_9063 (by decide)
private theorem atomic_9063_agreement : RowAgreement 9062 9063 := RowAgreement.single (first := 9062) modelrow_agrees_9063
private def atomic_9064 : CountSegment 1 10 13120 9063 9064 3792 3792 := CountSegment.single (first := 9063) (initial := 3792) row_count_9064 (by decide)
private theorem atomic_9064_agreement : RowAgreement 9063 9064 := RowAgreement.single (first := 9063) modelrow_agrees_9064
private def atomic_9065 : CountSegment 1 10 13120 9064 9065 3792 3792 := CountSegment.single (first := 9064) (initial := 3792) row_count_9065 (by decide)
private theorem atomic_9065_agreement : RowAgreement 9064 9065 := RowAgreement.single (first := 9064) modelrow_agrees_9065
private def atomic_9066 : CountSegment 1 10 13120 9065 9066 3792 3792 := CountSegment.single (first := 9065) (initial := 3792) row_count_9066 (by decide)
private theorem atomic_9066_agreement : RowAgreement 9065 9066 := RowAgreement.single (first := 9065) modelrow_agrees_9066
private def atomic_9067 : CountSegment 1 10 13120 9066 9067 3792 3792 := CountSegment.single (first := 9066) (initial := 3792) row_count_9067 (by decide)
private theorem atomic_9067_agreement : RowAgreement 9066 9067 := RowAgreement.single (first := 9066) modelrow_agrees_9067
private def atomic_9068 : CountSegment 1 10 13120 9067 9068 3792 3793 := CountSegment.single (first := 9067) (initial := 3792) row_count_9068 (by decide)
private theorem atomic_9068_agreement : RowAgreement 9067 9068 := RowAgreement.single (first := 9067) modelrow_agrees_9068
private def atomic_9069 : CountSegment 1 10 13120 9068 9069 3793 3793 := CountSegment.single (first := 9068) (initial := 3793) row_count_9069 (by decide)
private theorem atomic_9069_agreement : RowAgreement 9068 9069 := RowAgreement.single (first := 9068) modelrow_agrees_9069
private def atomic_9070 : CountSegment 1 10 13120 9069 9070 3793 3793 := CountSegment.single (first := 9069) (initial := 3793) row_count_9070 (by decide)
private theorem atomic_9070_agreement : RowAgreement 9069 9070 := RowAgreement.single (first := 9069) modelrow_agrees_9070
private def atomic_9071 : CountSegment 1 10 13120 9070 9071 3793 3793 := CountSegment.single (first := 9070) (initial := 3793) row_count_9071 (by decide)
private theorem atomic_9071_agreement : RowAgreement 9070 9071 := RowAgreement.single (first := 9070) modelrow_agrees_9071
private def atomic_9072 : CountSegment 1 10 13120 9071 9072 3793 3794 := CountSegment.single (first := 9071) (initial := 3793) row_count_9072 (by decide)
private theorem atomic_9072_agreement : RowAgreement 9071 9072 := RowAgreement.single (first := 9071) modelrow_agrees_9072
private def atomic_9073 : CountSegment 1 10 13120 9072 9073 3794 3795 := CountSegment.single (first := 9072) (initial := 3794) row_count_9073 (by decide)
private theorem atomic_9073_agreement : RowAgreement 9072 9073 := RowAgreement.single (first := 9072) modelrow_agrees_9073
private def atomic_9074 : CountSegment 1 10 13120 9073 9074 3795 3796 := CountSegment.single (first := 9073) (initial := 3795) row_count_9074 (by decide)
private theorem atomic_9074_agreement : RowAgreement 9073 9074 := RowAgreement.single (first := 9073) modelrow_agrees_9074
private def atomic_9075 : CountSegment 1 10 13120 9074 9075 3796 3798 := CountSegment.single (first := 9074) (initial := 3796) row_count_9075 (by decide)
private theorem atomic_9075_agreement : RowAgreement 9074 9075 := RowAgreement.single (first := 9074) modelrow_agrees_9075
private def atomic_9076 : CountSegment 1 10 13120 9075 9076 3798 3798 := CountSegment.single (first := 9075) (initial := 3798) row_count_9076 (by decide)
private theorem atomic_9076_agreement : RowAgreement 9075 9076 := RowAgreement.single (first := 9075) modelrow_agrees_9076
private def atomic_9077 : CountSegment 1 10 13120 9076 9077 3798 3798 := CountSegment.single (first := 9076) (initial := 3798) row_count_9077 (by decide)
private theorem atomic_9077_agreement : RowAgreement 9076 9077 := RowAgreement.single (first := 9076) modelrow_agrees_9077
private def atomic_9078 : CountSegment 1 10 13120 9077 9078 3798 3798 := CountSegment.single (first := 9077) (initial := 3798) row_count_9078 (by decide)
private theorem atomic_9078_agreement : RowAgreement 9077 9078 := RowAgreement.single (first := 9077) modelrow_agrees_9078
private def atomic_9079 : CountSegment 1 10 13120 9078 9079 3798 3799 := CountSegment.single (first := 9078) (initial := 3798) row_count_9079 (by decide)
private theorem atomic_9079_agreement : RowAgreement 9078 9079 := RowAgreement.single (first := 9078) modelrow_agrees_9079
private def atomic_9080 : CountSegment 1 10 13120 9079 9080 3799 3799 := CountSegment.single (first := 9079) (initial := 3799) row_count_9080 (by decide)
private theorem atomic_9080_agreement : RowAgreement 9079 9080 := RowAgreement.single (first := 9079) modelrow_agrees_9080
private def atomic_9081 : CountSegment 1 10 13120 9080 9081 3799 3799 := CountSegment.single (first := 9080) (initial := 3799) row_count_9081 (by decide)
private theorem atomic_9081_agreement : RowAgreement 9080 9081 := RowAgreement.single (first := 9080) modelrow_agrees_9081
private def atomic_9082 : CountSegment 1 10 13120 9081 9082 3799 3800 := CountSegment.single (first := 9081) (initial := 3799) row_count_9082 (by decide)
private theorem atomic_9082_agreement : RowAgreement 9081 9082 := RowAgreement.single (first := 9081) modelrow_agrees_9082
private def atomic_9083 : CountSegment 1 10 13120 9082 9083 3800 3800 := CountSegment.single (first := 9082) (initial := 3800) row_count_9083 (by decide)
private theorem atomic_9083_agreement : RowAgreement 9082 9083 := RowAgreement.single (first := 9082) modelrow_agrees_9083
private def atomic_9084 : CountSegment 1 10 13120 9083 9084 3800 3800 := CountSegment.single (first := 9083) (initial := 3800) row_count_9084 (by decide)
private theorem atomic_9084_agreement : RowAgreement 9083 9084 := RowAgreement.single (first := 9083) modelrow_agrees_9084
private def atomic_9085 : CountSegment 1 10 13120 9084 9085 3800 3800 := CountSegment.single (first := 9084) (initial := 3800) row_count_9085 (by decide)
private theorem atomic_9085_agreement : RowAgreement 9084 9085 := RowAgreement.single (first := 9084) modelrow_agrees_9085
private def atomic_9086 : CountSegment 1 10 13120 9085 9086 3800 3800 := CountSegment.single (first := 9085) (initial := 3800) row_count_9086 (by decide)
private theorem atomic_9086_agreement : RowAgreement 9085 9086 := RowAgreement.single (first := 9085) modelrow_agrees_9086
private def atomic_9087 : CountSegment 1 10 13120 9086 9087 3800 3801 := CountSegment.single (first := 9086) (initial := 3800) row_count_9087 (by decide)
private theorem atomic_9087_agreement : RowAgreement 9086 9087 := RowAgreement.single (first := 9086) modelrow_agrees_9087
private def atomic_9088 : CountSegment 1 10 13120 9087 9088 3801 3802 := CountSegment.single (first := 9087) (initial := 3801) row_count_9088 (by decide)
private theorem atomic_9088_agreement : RowAgreement 9087 9088 := RowAgreement.single (first := 9087) modelrow_agrees_9088
private def within_070_000 : CountSegment 1 10 13120 8960 8962 3741 3741 := CountSegment.join atomic_8961 atomic_8962
private theorem within_070_000_agreement : RowAgreement 8960 8962 := RowAgreement.join atomic_8961_agreement atomic_8962_agreement
private def within_070_001 : CountSegment 1 10 13120 8962 8964 3741 3742 := CountSegment.join atomic_8963 atomic_8964
private theorem within_070_001_agreement : RowAgreement 8962 8964 := RowAgreement.join atomic_8963_agreement atomic_8964_agreement
private def within_070_002 : CountSegment 1 10 13120 8960 8964 3741 3742 := CountSegment.join within_070_000 within_070_001
private theorem within_070_002_agreement : RowAgreement 8960 8964 := RowAgreement.join within_070_000_agreement within_070_001_agreement
private def within_070_003 : CountSegment 1 10 13120 8964 8966 3742 3743 := CountSegment.join atomic_8965 atomic_8966
private theorem within_070_003_agreement : RowAgreement 8964 8966 := RowAgreement.join atomic_8965_agreement atomic_8966_agreement
private def within_070_004 : CountSegment 1 10 13120 8966 8968 3743 3745 := CountSegment.join atomic_8967 atomic_8968
private theorem within_070_004_agreement : RowAgreement 8966 8968 := RowAgreement.join atomic_8967_agreement atomic_8968_agreement
private def within_070_005 : CountSegment 1 10 13120 8964 8968 3742 3745 := CountSegment.join within_070_003 within_070_004
private theorem within_070_005_agreement : RowAgreement 8964 8968 := RowAgreement.join within_070_003_agreement within_070_004_agreement
private def within_070_006 : CountSegment 1 10 13120 8960 8968 3741 3745 := CountSegment.join within_070_002 within_070_005
private theorem within_070_006_agreement : RowAgreement 8960 8968 := RowAgreement.join within_070_002_agreement within_070_005_agreement
private def within_070_007 : CountSegment 1 10 13120 8968 8970 3745 3746 := CountSegment.join atomic_8969 atomic_8970
private theorem within_070_007_agreement : RowAgreement 8968 8970 := RowAgreement.join atomic_8969_agreement atomic_8970_agreement
private def within_070_008 : CountSegment 1 10 13120 8970 8972 3746 3746 := CountSegment.join atomic_8971 atomic_8972
private theorem within_070_008_agreement : RowAgreement 8970 8972 := RowAgreement.join atomic_8971_agreement atomic_8972_agreement
private def within_070_009 : CountSegment 1 10 13120 8968 8972 3745 3746 := CountSegment.join within_070_007 within_070_008
private theorem within_070_009_agreement : RowAgreement 8968 8972 := RowAgreement.join within_070_007_agreement within_070_008_agreement
private def within_070_010 : CountSegment 1 10 13120 8972 8974 3746 3749 := CountSegment.join atomic_8973 atomic_8974
private theorem within_070_010_agreement : RowAgreement 8972 8974 := RowAgreement.join atomic_8973_agreement atomic_8974_agreement
private def within_070_011 : CountSegment 1 10 13120 8974 8976 3749 3750 := CountSegment.join atomic_8975 atomic_8976
private theorem within_070_011_agreement : RowAgreement 8974 8976 := RowAgreement.join atomic_8975_agreement atomic_8976_agreement
private def within_070_012 : CountSegment 1 10 13120 8972 8976 3746 3750 := CountSegment.join within_070_010 within_070_011
private theorem within_070_012_agreement : RowAgreement 8972 8976 := RowAgreement.join within_070_010_agreement within_070_011_agreement
private def within_070_013 : CountSegment 1 10 13120 8968 8976 3745 3750 := CountSegment.join within_070_009 within_070_012
private theorem within_070_013_agreement : RowAgreement 8968 8976 := RowAgreement.join within_070_009_agreement within_070_012_agreement
private def within_070_014 : CountSegment 1 10 13120 8960 8976 3741 3750 := CountSegment.join within_070_006 within_070_013
private theorem within_070_014_agreement : RowAgreement 8960 8976 := RowAgreement.join within_070_006_agreement within_070_013_agreement
private def within_070_015 : CountSegment 1 10 13120 8976 8978 3750 3751 := CountSegment.join atomic_8977 atomic_8978
private theorem within_070_015_agreement : RowAgreement 8976 8978 := RowAgreement.join atomic_8977_agreement atomic_8978_agreement
private def within_070_016 : CountSegment 1 10 13120 8978 8980 3751 3751 := CountSegment.join atomic_8979 atomic_8980
private theorem within_070_016_agreement : RowAgreement 8978 8980 := RowAgreement.join atomic_8979_agreement atomic_8980_agreement
private def within_070_017 : CountSegment 1 10 13120 8976 8980 3750 3751 := CountSegment.join within_070_015 within_070_016
private theorem within_070_017_agreement : RowAgreement 8976 8980 := RowAgreement.join within_070_015_agreement within_070_016_agreement
private def within_070_018 : CountSegment 1 10 13120 8980 8982 3751 3754 := CountSegment.join atomic_8981 atomic_8982
private theorem within_070_018_agreement : RowAgreement 8980 8982 := RowAgreement.join atomic_8981_agreement atomic_8982_agreement
private def within_070_019 : CountSegment 1 10 13120 8982 8984 3754 3754 := CountSegment.join atomic_8983 atomic_8984
private theorem within_070_019_agreement : RowAgreement 8982 8984 := RowAgreement.join atomic_8983_agreement atomic_8984_agreement
private def within_070_020 : CountSegment 1 10 13120 8980 8984 3751 3754 := CountSegment.join within_070_018 within_070_019
private theorem within_070_020_agreement : RowAgreement 8980 8984 := RowAgreement.join within_070_018_agreement within_070_019_agreement
private def within_070_021 : CountSegment 1 10 13120 8976 8984 3750 3754 := CountSegment.join within_070_017 within_070_020
private theorem within_070_021_agreement : RowAgreement 8976 8984 := RowAgreement.join within_070_017_agreement within_070_020_agreement
private def within_070_022 : CountSegment 1 10 13120 8984 8986 3754 3755 := CountSegment.join atomic_8985 atomic_8986
private theorem within_070_022_agreement : RowAgreement 8984 8986 := RowAgreement.join atomic_8985_agreement atomic_8986_agreement
private def within_070_023 : CountSegment 1 10 13120 8986 8988 3755 3756 := CountSegment.join atomic_8987 atomic_8988
private theorem within_070_023_agreement : RowAgreement 8986 8988 := RowAgreement.join atomic_8987_agreement atomic_8988_agreement
private def within_070_024 : CountSegment 1 10 13120 8984 8988 3754 3756 := CountSegment.join within_070_022 within_070_023
private theorem within_070_024_agreement : RowAgreement 8984 8988 := RowAgreement.join within_070_022_agreement within_070_023_agreement
private def within_070_025 : CountSegment 1 10 13120 8988 8990 3756 3757 := CountSegment.join atomic_8989 atomic_8990
private theorem within_070_025_agreement : RowAgreement 8988 8990 := RowAgreement.join atomic_8989_agreement atomic_8990_agreement
private def within_070_026 : CountSegment 1 10 13120 8990 8992 3757 3757 := CountSegment.join atomic_8991 atomic_8992
private theorem within_070_026_agreement : RowAgreement 8990 8992 := RowAgreement.join atomic_8991_agreement atomic_8992_agreement
private def within_070_027 : CountSegment 1 10 13120 8988 8992 3756 3757 := CountSegment.join within_070_025 within_070_026
private theorem within_070_027_agreement : RowAgreement 8988 8992 := RowAgreement.join within_070_025_agreement within_070_026_agreement
private def within_070_028 : CountSegment 1 10 13120 8984 8992 3754 3757 := CountSegment.join within_070_024 within_070_027
private theorem within_070_028_agreement : RowAgreement 8984 8992 := RowAgreement.join within_070_024_agreement within_070_027_agreement
private def within_070_029 : CountSegment 1 10 13120 8976 8992 3750 3757 := CountSegment.join within_070_021 within_070_028
private theorem within_070_029_agreement : RowAgreement 8976 8992 := RowAgreement.join within_070_021_agreement within_070_028_agreement
private def within_070_030 : CountSegment 1 10 13120 8960 8992 3741 3757 := CountSegment.join within_070_014 within_070_029
private theorem within_070_030_agreement : RowAgreement 8960 8992 := RowAgreement.join within_070_014_agreement within_070_029_agreement
private def within_070_031 : CountSegment 1 10 13120 8992 8994 3757 3758 := CountSegment.join atomic_8993 atomic_8994
private theorem within_070_031_agreement : RowAgreement 8992 8994 := RowAgreement.join atomic_8993_agreement atomic_8994_agreement
private def within_070_032 : CountSegment 1 10 13120 8994 8996 3758 3759 := CountSegment.join atomic_8995 atomic_8996
private theorem within_070_032_agreement : RowAgreement 8994 8996 := RowAgreement.join atomic_8995_agreement atomic_8996_agreement
private def within_070_033 : CountSegment 1 10 13120 8992 8996 3757 3759 := CountSegment.join within_070_031 within_070_032
private theorem within_070_033_agreement : RowAgreement 8992 8996 := RowAgreement.join within_070_031_agreement within_070_032_agreement
private def within_070_034 : CountSegment 1 10 13120 8996 8998 3759 3760 := CountSegment.join atomic_8997 atomic_8998
private theorem within_070_034_agreement : RowAgreement 8996 8998 := RowAgreement.join atomic_8997_agreement atomic_8998_agreement
private def within_070_035 : CountSegment 1 10 13120 8998 9000 3760 3760 := CountSegment.join atomic_8999 atomic_9000
private theorem within_070_035_agreement : RowAgreement 8998 9000 := RowAgreement.join atomic_8999_agreement atomic_9000_agreement
private def within_070_036 : CountSegment 1 10 13120 8996 9000 3759 3760 := CountSegment.join within_070_034 within_070_035
private theorem within_070_036_agreement : RowAgreement 8996 9000 := RowAgreement.join within_070_034_agreement within_070_035_agreement
private def within_070_037 : CountSegment 1 10 13120 8992 9000 3757 3760 := CountSegment.join within_070_033 within_070_036
private theorem within_070_037_agreement : RowAgreement 8992 9000 := RowAgreement.join within_070_033_agreement within_070_036_agreement
private def within_070_038 : CountSegment 1 10 13120 9000 9002 3760 3761 := CountSegment.join atomic_9001 atomic_9002
private theorem within_070_038_agreement : RowAgreement 9000 9002 := RowAgreement.join atomic_9001_agreement atomic_9002_agreement
private def within_070_039 : CountSegment 1 10 13120 9002 9004 3761 3761 := CountSegment.join atomic_9003 atomic_9004
private theorem within_070_039_agreement : RowAgreement 9002 9004 := RowAgreement.join atomic_9003_agreement atomic_9004_agreement
private def within_070_040 : CountSegment 1 10 13120 9000 9004 3760 3761 := CountSegment.join within_070_038 within_070_039
private theorem within_070_040_agreement : RowAgreement 9000 9004 := RowAgreement.join within_070_038_agreement within_070_039_agreement
private def within_070_041 : CountSegment 1 10 13120 9004 9006 3761 3762 := CountSegment.join atomic_9005 atomic_9006
private theorem within_070_041_agreement : RowAgreement 9004 9006 := RowAgreement.join atomic_9005_agreement atomic_9006_agreement
private def within_070_042 : CountSegment 1 10 13120 9006 9008 3762 3763 := CountSegment.join atomic_9007 atomic_9008
private theorem within_070_042_agreement : RowAgreement 9006 9008 := RowAgreement.join atomic_9007_agreement atomic_9008_agreement
private def within_070_043 : CountSegment 1 10 13120 9004 9008 3761 3763 := CountSegment.join within_070_041 within_070_042
private theorem within_070_043_agreement : RowAgreement 9004 9008 := RowAgreement.join within_070_041_agreement within_070_042_agreement
private def within_070_044 : CountSegment 1 10 13120 9000 9008 3760 3763 := CountSegment.join within_070_040 within_070_043
private theorem within_070_044_agreement : RowAgreement 9000 9008 := RowAgreement.join within_070_040_agreement within_070_043_agreement
private def within_070_045 : CountSegment 1 10 13120 8992 9008 3757 3763 := CountSegment.join within_070_037 within_070_044
private theorem within_070_045_agreement : RowAgreement 8992 9008 := RowAgreement.join within_070_037_agreement within_070_044_agreement
private def within_070_046 : CountSegment 1 10 13120 9008 9010 3763 3764 := CountSegment.join atomic_9009 atomic_9010
private theorem within_070_046_agreement : RowAgreement 9008 9010 := RowAgreement.join atomic_9009_agreement atomic_9010_agreement
private def within_070_047 : CountSegment 1 10 13120 9010 9012 3764 3765 := CountSegment.join atomic_9011 atomic_9012
private theorem within_070_047_agreement : RowAgreement 9010 9012 := RowAgreement.join atomic_9011_agreement atomic_9012_agreement
private def within_070_048 : CountSegment 1 10 13120 9008 9012 3763 3765 := CountSegment.join within_070_046 within_070_047
private theorem within_070_048_agreement : RowAgreement 9008 9012 := RowAgreement.join within_070_046_agreement within_070_047_agreement
private def within_070_049 : CountSegment 1 10 13120 9012 9014 3765 3765 := CountSegment.join atomic_9013 atomic_9014
private theorem within_070_049_agreement : RowAgreement 9012 9014 := RowAgreement.join atomic_9013_agreement atomic_9014_agreement
private def within_070_050 : CountSegment 1 10 13120 9014 9016 3765 3766 := CountSegment.join atomic_9015 atomic_9016
private theorem within_070_050_agreement : RowAgreement 9014 9016 := RowAgreement.join atomic_9015_agreement atomic_9016_agreement
private def within_070_051 : CountSegment 1 10 13120 9012 9016 3765 3766 := CountSegment.join within_070_049 within_070_050
private theorem within_070_051_agreement : RowAgreement 9012 9016 := RowAgreement.join within_070_049_agreement within_070_050_agreement
private def within_070_052 : CountSegment 1 10 13120 9008 9016 3763 3766 := CountSegment.join within_070_048 within_070_051
private theorem within_070_052_agreement : RowAgreement 9008 9016 := RowAgreement.join within_070_048_agreement within_070_051_agreement
private def within_070_053 : CountSegment 1 10 13120 9016 9018 3766 3767 := CountSegment.join atomic_9017 atomic_9018
private theorem within_070_053_agreement : RowAgreement 9016 9018 := RowAgreement.join atomic_9017_agreement atomic_9018_agreement
private def within_070_054 : CountSegment 1 10 13120 9018 9020 3767 3768 := CountSegment.join atomic_9019 atomic_9020
private theorem within_070_054_agreement : RowAgreement 9018 9020 := RowAgreement.join atomic_9019_agreement atomic_9020_agreement
private def within_070_055 : CountSegment 1 10 13120 9016 9020 3766 3768 := CountSegment.join within_070_053 within_070_054
private theorem within_070_055_agreement : RowAgreement 9016 9020 := RowAgreement.join within_070_053_agreement within_070_054_agreement
private def within_070_056 : CountSegment 1 10 13120 9020 9022 3768 3769 := CountSegment.join atomic_9021 atomic_9022
private theorem within_070_056_agreement : RowAgreement 9020 9022 := RowAgreement.join atomic_9021_agreement atomic_9022_agreement
private def within_070_057 : CountSegment 1 10 13120 9022 9024 3769 3770 := CountSegment.join atomic_9023 atomic_9024
private theorem within_070_057_agreement : RowAgreement 9022 9024 := RowAgreement.join atomic_9023_agreement atomic_9024_agreement
private def within_070_058 : CountSegment 1 10 13120 9020 9024 3768 3770 := CountSegment.join within_070_056 within_070_057
private theorem within_070_058_agreement : RowAgreement 9020 9024 := RowAgreement.join within_070_056_agreement within_070_057_agreement
private def within_070_059 : CountSegment 1 10 13120 9016 9024 3766 3770 := CountSegment.join within_070_055 within_070_058
private theorem within_070_059_agreement : RowAgreement 9016 9024 := RowAgreement.join within_070_055_agreement within_070_058_agreement
private def within_070_060 : CountSegment 1 10 13120 9008 9024 3763 3770 := CountSegment.join within_070_052 within_070_059
private theorem within_070_060_agreement : RowAgreement 9008 9024 := RowAgreement.join within_070_052_agreement within_070_059_agreement
private def within_070_061 : CountSegment 1 10 13120 8992 9024 3757 3770 := CountSegment.join within_070_045 within_070_060
private theorem within_070_061_agreement : RowAgreement 8992 9024 := RowAgreement.join within_070_045_agreement within_070_060_agreement
private def within_070_062 : CountSegment 1 10 13120 8960 9024 3741 3770 := CountSegment.join within_070_030 within_070_061
private theorem within_070_062_agreement : RowAgreement 8960 9024 := RowAgreement.join within_070_030_agreement within_070_061_agreement
private def within_070_063 : CountSegment 1 10 13120 9024 9026 3770 3770 := CountSegment.join atomic_9025 atomic_9026
private theorem within_070_063_agreement : RowAgreement 9024 9026 := RowAgreement.join atomic_9025_agreement atomic_9026_agreement
private def within_070_064 : CountSegment 1 10 13120 9026 9028 3770 3771 := CountSegment.join atomic_9027 atomic_9028
private theorem within_070_064_agreement : RowAgreement 9026 9028 := RowAgreement.join atomic_9027_agreement atomic_9028_agreement
private def within_070_065 : CountSegment 1 10 13120 9024 9028 3770 3771 := CountSegment.join within_070_063 within_070_064
private theorem within_070_065_agreement : RowAgreement 9024 9028 := RowAgreement.join within_070_063_agreement within_070_064_agreement
private def within_070_066 : CountSegment 1 10 13120 9028 9030 3771 3772 := CountSegment.join atomic_9029 atomic_9030
private theorem within_070_066_agreement : RowAgreement 9028 9030 := RowAgreement.join atomic_9029_agreement atomic_9030_agreement
private def within_070_067 : CountSegment 1 10 13120 9030 9032 3772 3773 := CountSegment.join atomic_9031 atomic_9032
private theorem within_070_067_agreement : RowAgreement 9030 9032 := RowAgreement.join atomic_9031_agreement atomic_9032_agreement
private def within_070_068 : CountSegment 1 10 13120 9028 9032 3771 3773 := CountSegment.join within_070_066 within_070_067
private theorem within_070_068_agreement : RowAgreement 9028 9032 := RowAgreement.join within_070_066_agreement within_070_067_agreement
private def within_070_069 : CountSegment 1 10 13120 9024 9032 3770 3773 := CountSegment.join within_070_065 within_070_068
private theorem within_070_069_agreement : RowAgreement 9024 9032 := RowAgreement.join within_070_065_agreement within_070_068_agreement
private def within_070_070 : CountSegment 1 10 13120 9032 9034 3773 3775 := CountSegment.join atomic_9033 atomic_9034
private theorem within_070_070_agreement : RowAgreement 9032 9034 := RowAgreement.join atomic_9033_agreement atomic_9034_agreement
private def within_070_071 : CountSegment 1 10 13120 9034 9036 3775 3777 := CountSegment.join atomic_9035 atomic_9036
private theorem within_070_071_agreement : RowAgreement 9034 9036 := RowAgreement.join atomic_9035_agreement atomic_9036_agreement
private def within_070_072 : CountSegment 1 10 13120 9032 9036 3773 3777 := CountSegment.join within_070_070 within_070_071
private theorem within_070_072_agreement : RowAgreement 9032 9036 := RowAgreement.join within_070_070_agreement within_070_071_agreement
private def within_070_073 : CountSegment 1 10 13120 9036 9038 3777 3780 := CountSegment.join atomic_9037 atomic_9038
private theorem within_070_073_agreement : RowAgreement 9036 9038 := RowAgreement.join atomic_9037_agreement atomic_9038_agreement
private def within_070_074 : CountSegment 1 10 13120 9038 9040 3780 3781 := CountSegment.join atomic_9039 atomic_9040
private theorem within_070_074_agreement : RowAgreement 9038 9040 := RowAgreement.join atomic_9039_agreement atomic_9040_agreement
private def within_070_075 : CountSegment 1 10 13120 9036 9040 3777 3781 := CountSegment.join within_070_073 within_070_074
private theorem within_070_075_agreement : RowAgreement 9036 9040 := RowAgreement.join within_070_073_agreement within_070_074_agreement
private def within_070_076 : CountSegment 1 10 13120 9032 9040 3773 3781 := CountSegment.join within_070_072 within_070_075
private theorem within_070_076_agreement : RowAgreement 9032 9040 := RowAgreement.join within_070_072_agreement within_070_075_agreement
private def within_070_077 : CountSegment 1 10 13120 9024 9040 3770 3781 := CountSegment.join within_070_069 within_070_076
private theorem within_070_077_agreement : RowAgreement 9024 9040 := RowAgreement.join within_070_069_agreement within_070_076_agreement
private def within_070_078 : CountSegment 1 10 13120 9040 9042 3781 3782 := CountSegment.join atomic_9041 atomic_9042
private theorem within_070_078_agreement : RowAgreement 9040 9042 := RowAgreement.join atomic_9041_agreement atomic_9042_agreement
private def within_070_079 : CountSegment 1 10 13120 9042 9044 3782 3782 := CountSegment.join atomic_9043 atomic_9044
private theorem within_070_079_agreement : RowAgreement 9042 9044 := RowAgreement.join atomic_9043_agreement atomic_9044_agreement
private def within_070_080 : CountSegment 1 10 13120 9040 9044 3781 3782 := CountSegment.join within_070_078 within_070_079
private theorem within_070_080_agreement : RowAgreement 9040 9044 := RowAgreement.join within_070_078_agreement within_070_079_agreement
private def within_070_081 : CountSegment 1 10 13120 9044 9046 3782 3783 := CountSegment.join atomic_9045 atomic_9046
private theorem within_070_081_agreement : RowAgreement 9044 9046 := RowAgreement.join atomic_9045_agreement atomic_9046_agreement
private def within_070_082 : CountSegment 1 10 13120 9046 9048 3783 3784 := CountSegment.join atomic_9047 atomic_9048
private theorem within_070_082_agreement : RowAgreement 9046 9048 := RowAgreement.join atomic_9047_agreement atomic_9048_agreement
private def within_070_083 : CountSegment 1 10 13120 9044 9048 3782 3784 := CountSegment.join within_070_081 within_070_082
private theorem within_070_083_agreement : RowAgreement 9044 9048 := RowAgreement.join within_070_081_agreement within_070_082_agreement
private def within_070_084 : CountSegment 1 10 13120 9040 9048 3781 3784 := CountSegment.join within_070_080 within_070_083
private theorem within_070_084_agreement : RowAgreement 9040 9048 := RowAgreement.join within_070_080_agreement within_070_083_agreement
private def within_070_085 : CountSegment 1 10 13120 9048 9050 3784 3785 := CountSegment.join atomic_9049 atomic_9050
private theorem within_070_085_agreement : RowAgreement 9048 9050 := RowAgreement.join atomic_9049_agreement atomic_9050_agreement
private def within_070_086 : CountSegment 1 10 13120 9050 9052 3785 3786 := CountSegment.join atomic_9051 atomic_9052
private theorem within_070_086_agreement : RowAgreement 9050 9052 := RowAgreement.join atomic_9051_agreement atomic_9052_agreement
private def within_070_087 : CountSegment 1 10 13120 9048 9052 3784 3786 := CountSegment.join within_070_085 within_070_086
private theorem within_070_087_agreement : RowAgreement 9048 9052 := RowAgreement.join within_070_085_agreement within_070_086_agreement
private def within_070_088 : CountSegment 1 10 13120 9052 9054 3786 3788 := CountSegment.join atomic_9053 atomic_9054
private theorem within_070_088_agreement : RowAgreement 9052 9054 := RowAgreement.join atomic_9053_agreement atomic_9054_agreement
private def within_070_089 : CountSegment 1 10 13120 9054 9056 3788 3788 := CountSegment.join atomic_9055 atomic_9056
private theorem within_070_089_agreement : RowAgreement 9054 9056 := RowAgreement.join atomic_9055_agreement atomic_9056_agreement
private def within_070_090 : CountSegment 1 10 13120 9052 9056 3786 3788 := CountSegment.join within_070_088 within_070_089
private theorem within_070_090_agreement : RowAgreement 9052 9056 := RowAgreement.join within_070_088_agreement within_070_089_agreement
private def within_070_091 : CountSegment 1 10 13120 9048 9056 3784 3788 := CountSegment.join within_070_087 within_070_090
private theorem within_070_091_agreement : RowAgreement 9048 9056 := RowAgreement.join within_070_087_agreement within_070_090_agreement
private def within_070_092 : CountSegment 1 10 13120 9040 9056 3781 3788 := CountSegment.join within_070_084 within_070_091
private theorem within_070_092_agreement : RowAgreement 9040 9056 := RowAgreement.join within_070_084_agreement within_070_091_agreement
private def within_070_093 : CountSegment 1 10 13120 9024 9056 3770 3788 := CountSegment.join within_070_077 within_070_092
private theorem within_070_093_agreement : RowAgreement 9024 9056 := RowAgreement.join within_070_077_agreement within_070_092_agreement
private def within_070_094 : CountSegment 1 10 13120 9056 9058 3788 3789 := CountSegment.join atomic_9057 atomic_9058
private theorem within_070_094_agreement : RowAgreement 9056 9058 := RowAgreement.join atomic_9057_agreement atomic_9058_agreement
private def within_070_095 : CountSegment 1 10 13120 9058 9060 3789 3791 := CountSegment.join atomic_9059 atomic_9060
private theorem within_070_095_agreement : RowAgreement 9058 9060 := RowAgreement.join atomic_9059_agreement atomic_9060_agreement
private def within_070_096 : CountSegment 1 10 13120 9056 9060 3788 3791 := CountSegment.join within_070_094 within_070_095
private theorem within_070_096_agreement : RowAgreement 9056 9060 := RowAgreement.join within_070_094_agreement within_070_095_agreement
private def within_070_097 : CountSegment 1 10 13120 9060 9062 3791 3791 := CountSegment.join atomic_9061 atomic_9062
private theorem within_070_097_agreement : RowAgreement 9060 9062 := RowAgreement.join atomic_9061_agreement atomic_9062_agreement
private def within_070_098 : CountSegment 1 10 13120 9062 9064 3791 3792 := CountSegment.join atomic_9063 atomic_9064
private theorem within_070_098_agreement : RowAgreement 9062 9064 := RowAgreement.join atomic_9063_agreement atomic_9064_agreement
private def within_070_099 : CountSegment 1 10 13120 9060 9064 3791 3792 := CountSegment.join within_070_097 within_070_098
private theorem within_070_099_agreement : RowAgreement 9060 9064 := RowAgreement.join within_070_097_agreement within_070_098_agreement
private def within_070_100 : CountSegment 1 10 13120 9056 9064 3788 3792 := CountSegment.join within_070_096 within_070_099
private theorem within_070_100_agreement : RowAgreement 9056 9064 := RowAgreement.join within_070_096_agreement within_070_099_agreement
private def within_070_101 : CountSegment 1 10 13120 9064 9066 3792 3792 := CountSegment.join atomic_9065 atomic_9066
private theorem within_070_101_agreement : RowAgreement 9064 9066 := RowAgreement.join atomic_9065_agreement atomic_9066_agreement
private def within_070_102 : CountSegment 1 10 13120 9066 9068 3792 3793 := CountSegment.join atomic_9067 atomic_9068
private theorem within_070_102_agreement : RowAgreement 9066 9068 := RowAgreement.join atomic_9067_agreement atomic_9068_agreement
private def within_070_103 : CountSegment 1 10 13120 9064 9068 3792 3793 := CountSegment.join within_070_101 within_070_102
private theorem within_070_103_agreement : RowAgreement 9064 9068 := RowAgreement.join within_070_101_agreement within_070_102_agreement
private def within_070_104 : CountSegment 1 10 13120 9068 9070 3793 3793 := CountSegment.join atomic_9069 atomic_9070
private theorem within_070_104_agreement : RowAgreement 9068 9070 := RowAgreement.join atomic_9069_agreement atomic_9070_agreement
private def within_070_105 : CountSegment 1 10 13120 9070 9072 3793 3794 := CountSegment.join atomic_9071 atomic_9072
private theorem within_070_105_agreement : RowAgreement 9070 9072 := RowAgreement.join atomic_9071_agreement atomic_9072_agreement
private def within_070_106 : CountSegment 1 10 13120 9068 9072 3793 3794 := CountSegment.join within_070_104 within_070_105
private theorem within_070_106_agreement : RowAgreement 9068 9072 := RowAgreement.join within_070_104_agreement within_070_105_agreement
private def within_070_107 : CountSegment 1 10 13120 9064 9072 3792 3794 := CountSegment.join within_070_103 within_070_106
private theorem within_070_107_agreement : RowAgreement 9064 9072 := RowAgreement.join within_070_103_agreement within_070_106_agreement
private def within_070_108 : CountSegment 1 10 13120 9056 9072 3788 3794 := CountSegment.join within_070_100 within_070_107
private theorem within_070_108_agreement : RowAgreement 9056 9072 := RowAgreement.join within_070_100_agreement within_070_107_agreement
private def within_070_109 : CountSegment 1 10 13120 9072 9074 3794 3796 := CountSegment.join atomic_9073 atomic_9074
private theorem within_070_109_agreement : RowAgreement 9072 9074 := RowAgreement.join atomic_9073_agreement atomic_9074_agreement
private def within_070_110 : CountSegment 1 10 13120 9074 9076 3796 3798 := CountSegment.join atomic_9075 atomic_9076
private theorem within_070_110_agreement : RowAgreement 9074 9076 := RowAgreement.join atomic_9075_agreement atomic_9076_agreement
private def within_070_111 : CountSegment 1 10 13120 9072 9076 3794 3798 := CountSegment.join within_070_109 within_070_110
private theorem within_070_111_agreement : RowAgreement 9072 9076 := RowAgreement.join within_070_109_agreement within_070_110_agreement
private def within_070_112 : CountSegment 1 10 13120 9076 9078 3798 3798 := CountSegment.join atomic_9077 atomic_9078
private theorem within_070_112_agreement : RowAgreement 9076 9078 := RowAgreement.join atomic_9077_agreement atomic_9078_agreement
private def within_070_113 : CountSegment 1 10 13120 9078 9080 3798 3799 := CountSegment.join atomic_9079 atomic_9080
private theorem within_070_113_agreement : RowAgreement 9078 9080 := RowAgreement.join atomic_9079_agreement atomic_9080_agreement
private def within_070_114 : CountSegment 1 10 13120 9076 9080 3798 3799 := CountSegment.join within_070_112 within_070_113
private theorem within_070_114_agreement : RowAgreement 9076 9080 := RowAgreement.join within_070_112_agreement within_070_113_agreement
private def within_070_115 : CountSegment 1 10 13120 9072 9080 3794 3799 := CountSegment.join within_070_111 within_070_114
private theorem within_070_115_agreement : RowAgreement 9072 9080 := RowAgreement.join within_070_111_agreement within_070_114_agreement
private def within_070_116 : CountSegment 1 10 13120 9080 9082 3799 3800 := CountSegment.join atomic_9081 atomic_9082
private theorem within_070_116_agreement : RowAgreement 9080 9082 := RowAgreement.join atomic_9081_agreement atomic_9082_agreement
private def within_070_117 : CountSegment 1 10 13120 9082 9084 3800 3800 := CountSegment.join atomic_9083 atomic_9084
private theorem within_070_117_agreement : RowAgreement 9082 9084 := RowAgreement.join atomic_9083_agreement atomic_9084_agreement
private def within_070_118 : CountSegment 1 10 13120 9080 9084 3799 3800 := CountSegment.join within_070_116 within_070_117
private theorem within_070_118_agreement : RowAgreement 9080 9084 := RowAgreement.join within_070_116_agreement within_070_117_agreement
private def within_070_119 : CountSegment 1 10 13120 9084 9086 3800 3800 := CountSegment.join atomic_9085 atomic_9086
private theorem within_070_119_agreement : RowAgreement 9084 9086 := RowAgreement.join atomic_9085_agreement atomic_9086_agreement
private def within_070_120 : CountSegment 1 10 13120 9086 9088 3800 3802 := CountSegment.join atomic_9087 atomic_9088
private theorem within_070_120_agreement : RowAgreement 9086 9088 := RowAgreement.join atomic_9087_agreement atomic_9088_agreement
private def within_070_121 : CountSegment 1 10 13120 9084 9088 3800 3802 := CountSegment.join within_070_119 within_070_120
private theorem within_070_121_agreement : RowAgreement 9084 9088 := RowAgreement.join within_070_119_agreement within_070_120_agreement
private def within_070_122 : CountSegment 1 10 13120 9080 9088 3799 3802 := CountSegment.join within_070_118 within_070_121
private theorem within_070_122_agreement : RowAgreement 9080 9088 := RowAgreement.join within_070_118_agreement within_070_121_agreement
private def within_070_123 : CountSegment 1 10 13120 9072 9088 3794 3802 := CountSegment.join within_070_115 within_070_122
private theorem within_070_123_agreement : RowAgreement 9072 9088 := RowAgreement.join within_070_115_agreement within_070_122_agreement
private def within_070_124 : CountSegment 1 10 13120 9056 9088 3788 3802 := CountSegment.join within_070_108 within_070_123
private theorem within_070_124_agreement : RowAgreement 9056 9088 := RowAgreement.join within_070_108_agreement within_070_123_agreement
private def within_070_125 : CountSegment 1 10 13120 9024 9088 3770 3802 := CountSegment.join within_070_093 within_070_124
private theorem within_070_125_agreement : RowAgreement 9024 9088 := RowAgreement.join within_070_093_agreement within_070_124_agreement
private def within_070_126 : CountSegment 1 10 13120 8960 9088 3741 3802 := CountSegment.join within_070_062 within_070_125
private theorem within_070_126_agreement : RowAgreement 8960 9088 := RowAgreement.join within_070_062_agreement within_070_125_agreement
def segment_070 : CountSegment 1 10 13120 8960 9088 3741 3802 := within_070_126
theorem agreement_070 : RowAgreement 8960 9088 := within_070_126_agreement
end ProjectEulerSolutions.P58Numerical
