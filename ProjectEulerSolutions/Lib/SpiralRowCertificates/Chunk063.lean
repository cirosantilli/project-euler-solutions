import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_8065 : RowCertificate 8065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8065 1) := by change ¬ Nat.Prime 260193031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37170433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8065 2) := by change Nat.Prime 260176901; exact ProjectEulerSolutions.P58PrimeNodes.p_260176901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8065 3) := by change ¬ Nat.Prime 260160771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86720257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8065 : row 8065 = 1 := (row_8065.checked).trans rfl
theorem model_count_8065 : modelRow 8065 = 1 := by decide
theorem modelrow_agrees_8065 : modelRow 8065 = row 8065 := model_count_8065.trans row_count_8065.symm
def row_8066 : RowCertificate 8066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8066 1) := by change ¬ Nat.Prime 260257557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86752519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8066 2) := by change ¬ Nat.Prime 260241425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52048285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8066 3) := by change ¬ Nat.Prime 260225293; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6051751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8066 : row 8066 = 0 := (row_8066.checked).trans rfl
theorem model_count_8066 : modelRow 8066 = 0 := by decide
theorem modelrow_agrees_8066 : modelRow 8066 = row 8066 := model_count_8066.trans row_count_8066.symm
def row_8067 : RowCertificate 8067 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8067 1) := by change Nat.Prime 260322091; exact ProjectEulerSolutions.P58PrimeNodes.p_260322091.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8067 2) := by change Nat.Prime 260305957; exact ProjectEulerSolutions.P58PrimeNodes.p_260305957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8067 3) := by change ¬ Nat.Prime 260289823; exact PrattCertificates.composite_of_product (factor := 14293) (cofactor := 18211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8067 : row 8067 = 2 := (row_8067.checked).trans rfl
theorem model_count_8067 : modelRow 8067 = 2 := by decide
theorem modelrow_agrees_8067 : modelRow 8067 = row 8067 := model_count_8067.trans row_count_8067.symm
def row_8068 : RowCertificate 8068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8068 1) := by change ¬ Nat.Prime 260386633; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20029741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8068 2) := by change ¬ Nat.Prime 260370497; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8978293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8068 3) := by change ¬ Nat.Prime 260354361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86784787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8068 : row 8068 = 0 := (row_8068.checked).trans rfl
theorem model_count_8068 : modelRow 8068 = 0 := by decide
theorem modelrow_agrees_8068 : modelRow 8068 = row 8068 := model_count_8068.trans row_count_8068.symm
def row_8069 : RowCertificate 8069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8069 1) := by change ¬ Nat.Prime 260451183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86817061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8069 2) := by change ¬ Nat.Prime 260435045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52087009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8069 3) := by change ¬ Nat.Prime 260418907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37202701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8069 : row 8069 = 0 := (row_8069.checked).trans rfl
theorem model_count_8069 : modelRow 8069 = 0 := by decide
theorem modelrow_agrees_8069 : modelRow 8069 = row 8069 := model_count_8069.trans row_count_8069.symm
def row_8070 : RowCertificate 8070 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8070 1) := by change Nat.Prime 260515741; exact ProjectEulerSolutions.P58PrimeNodes.p_260515741.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8070 2) := by change ¬ Nat.Prime 260499601; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 1322333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8070 3) := by change ¬ Nat.Prime 260483461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37211923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8070 : row 8070 = 1 := (row_8070.checked).trans rfl
theorem model_count_8070 : modelRow 8070 = 1 := by decide
theorem modelrow_agrees_8070 : modelRow 8070 = row 8070 := model_count_8070.trans row_count_8070.symm
def row_8071 : RowCertificate 8071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8071 1) := by change ¬ Nat.Prime 260580307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20044639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8071 2) := by change ¬ Nat.Prime 260564165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52112833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8071 3) := by change ¬ Nat.Prime 260548023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86849341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8071 : row 8071 = 0 := (row_8071.checked).trans rfl
theorem model_count_8071 : modelRow 8071 = 0 := by decide
theorem modelrow_agrees_8071 : modelRow 8071 = row 8071 := model_count_8071.trans row_count_8071.symm
def row_8072 : RowCertificate 8072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8072 1) := by change ¬ Nat.Prime 260644881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86881627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8072 2) := by change ¬ Nat.Prime 260628737; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2306449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8072 3) := by change ¬ Nat.Prime 260612593; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 208657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8072 : row 8072 = 0 := (row_8072.checked).trans rfl
theorem model_count_8072 : modelRow 8072 = 0 := by decide
theorem modelrow_agrees_8072 : modelRow 8072 = row 8072 := model_count_8072.trans row_count_8072.symm
def row_8073 : RowCertificate 8073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8073 1) := by change ¬ Nat.Prime 260709463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37244209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8073 2) := by change ¬ Nat.Prime 260693317; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15334901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8073 3) := by change ¬ Nat.Prime 260677171; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8408941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8073 : row 8073 = 0 := (row_8073.checked).trans rfl
theorem model_count_8073 : modelRow 8073 = 0 := by decide
theorem modelrow_agrees_8073 : modelRow 8073 = row 8073 := model_count_8073.trans row_count_8073.symm
def row_8074 : RowCertificate 8074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8074 1) := by change ¬ Nat.Prime 260774053; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2053339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8074 2) := by change ¬ Nat.Prime 260757905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52151581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8074 3) := by change ¬ Nat.Prime 260741757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86913919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8074 : row 8074 = 0 := (row_8074.checked).trans rfl
theorem model_count_8074 : modelRow 8074 = 0 := by decide
theorem modelrow_agrees_8074 : modelRow 8074 = row 8074 := model_count_8074.trans row_count_8074.symm
def row_8075 : RowCertificate 8075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8075 1) := by change ¬ Nat.Prime 260838651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86946217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8075 2) := by change ¬ Nat.Prime 260822501; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 2582401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8075 3) := by change ¬ Nat.Prime 260806351; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20062027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8075 : row 8075 = 0 := (row_8075.checked).trans rfl
theorem model_count_8075 : modelRow 8075 = 0 := by decide
theorem modelrow_agrees_8075 : modelRow 8075 = row 8075 := model_count_8075.trans row_count_8075.symm
def row_8076 : RowCertificate 8076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8076 1) := by change ¬ Nat.Prime 260903257; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 498859) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8076 2) := by change ¬ Nat.Prime 260887105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52177421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8076 3) := by change ¬ Nat.Prime 260870953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37267279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8076 : row 8076 = 0 := (row_8076.checked).trans rfl
theorem model_count_8076 : modelRow 8076 = 0 := by decide
theorem modelrow_agrees_8076 : modelRow 8076 = row 8076 := model_count_8076.trans row_count_8076.symm
def row_8077 : RowCertificate 8077 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8077 1) := by change Nat.Prime 260967871; exact ProjectEulerSolutions.P58PrimeNodes.p_260967871.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8077 2) := by change ¬ Nat.Prime 260951717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20073209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8077 3) := by change ¬ Nat.Prime 260935563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86978521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8077 : row 8077 = 1 := (row_8077.checked).trans rfl
theorem model_count_8077 : modelRow 8077 = 1 := by decide
theorem modelrow_agrees_8077 : modelRow 8077 = row 8077 := model_count_8077.trans row_count_8077.symm
def row_8078 : RowCertificate 8078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8078 1) := by change ¬ Nat.Prime 261032493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87010831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8078 2) := by change Nat.Prime 261016337; exact ProjectEulerSolutions.P58PrimeNodes.p_261016337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8078 3) := by change ¬ Nat.Prime 261000181; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20076937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8078 : row 8078 = 1 := (row_8078.checked).trans rfl
theorem model_count_8078 : modelRow 8078 = 1 := by decide
theorem modelrow_agrees_8078 : modelRow 8078 = row 8078 := model_count_8078.trans row_count_8078.symm
def row_8079 : RowCertificate 8079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8079 1) := by change ¬ Nat.Prime 261097123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37299589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8079 2) := by change ¬ Nat.Prime 261080965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52216193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8079 3) := by change ¬ Nat.Prime 261064807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13740253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8079 : row 8079 = 0 := (row_8079.checked).trans rfl
theorem model_count_8079 : modelRow 8079 = 0 := by decide
theorem modelrow_agrees_8079 : modelRow 8079 = row 8079 := model_count_8079.trans row_count_8079.symm
def row_8080 : RowCertificate 8080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8080 1) := by change ¬ Nat.Prime 261161761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37308823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8080 2) := by change ¬ Nat.Prime 261145601; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3577337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8080 3) := by change ¬ Nat.Prime 261129441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87043147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8080 : row 8080 = 0 := (row_8080.checked).trans rfl
theorem model_count_8080 : modelRow 8080 = 0 := by decide
theorem modelrow_agrees_8080 : modelRow 8080 = row 8080 := model_count_8080.trans row_count_8080.symm
def row_8081 : RowCertificate 8081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8081 1) := by change ¬ Nat.Prime 261226407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87075469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8081 2) := by change ¬ Nat.Prime 261210245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52242049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8081 3) := by change ¬ Nat.Prime 261194083; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13747057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8081 : row 8081 = 0 := (row_8081.checked).trans rfl
theorem model_count_8081 : modelRow 8081 = 0 := by decide
theorem modelrow_agrees_8081 : modelRow 8081 = row 8081 := model_count_8081.trans row_count_8081.symm
def row_8082 : RowCertificate 8082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8082 1) := by change ¬ Nat.Prime 261291061; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1141009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8082 2) := by change ¬ Nat.Prime 261274897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20098069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8082 3) := by change ¬ Nat.Prime 261258733; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2693389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8082 : row 8082 = 0 := (row_8082.checked).trans rfl
theorem model_count_8082 : modelRow 8082 = 0 := by decide
theorem modelrow_agrees_8082 : modelRow 8082 = row 8082 := model_count_8082.trans row_count_8082.symm
def row_8083 : RowCertificate 8083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8083 1) := by change ¬ Nat.Prime 261355723; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1880257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8083 2) := by change ¬ Nat.Prime 261339557; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 1121629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8083 3) := by change ¬ Nat.Prime 261323391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87107797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8083 : row 8083 = 0 := (row_8083.checked).trans rfl
theorem model_count_8083 : modelRow 8083 = 0 := by decide
theorem modelrow_agrees_8083 : modelRow 8083 = row 8083 := model_count_8083.trans row_count_8083.symm
def row_8084 : RowCertificate 8084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8084 1) := by change ¬ Nat.Prime 261420393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87140131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8084 2) := by change ¬ Nat.Prime 261404225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52280845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8084 3) := by change ¬ Nat.Prime 261388057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37341151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8084 : row 8084 = 0 := (row_8084.checked).trans rfl
theorem model_count_8084 : modelRow 8084 = 0 := by decide
theorem modelrow_agrees_8084 : modelRow 8084 = row 8084 := model_count_8084.trans row_count_8084.symm
def row_8085 : RowCertificate 8085 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8085 1) := by change Nat.Prime 261485071; exact ProjectEulerSolutions.P58PrimeNodes.p_261485071.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8085 2) := by change ¬ Nat.Prime 261468901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9016169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8085 3) := by change ¬ Nat.Prime 261452731; exact PrattCertificates.composite_of_product (factor := 9397) (cofactor := 27823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8085 : row 8085 = 1 := (row_8085.checked).trans rfl
theorem model_count_8085 : modelRow 8085 = 1 := by decide
theorem modelrow_agrees_8085 : modelRow 8085 = row 8085 := model_count_8085.trans row_count_8085.symm
def row_8086 : RowCertificate 8086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8086 1) := by change ¬ Nat.Prime 261549757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37364251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8086 2) := by change ¬ Nat.Prime 261533585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52306717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8086 3) := by change ¬ Nat.Prime 261517413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87172471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8086 : row 8086 = 0 := (row_8086.checked).trans rfl
theorem model_count_8086 : modelRow 8086 = 0 := by decide
theorem modelrow_agrees_8086 : modelRow 8086 = row 8086 := model_count_8086.trans row_count_8086.symm
def row_8087 : RowCertificate 8087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8087 1) := by change ¬ Nat.Prime 261614451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87204817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8087 2) := by change ¬ Nat.Prime 261598277; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2315029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8087 3) := by change Nat.Prime 261582103; exact ProjectEulerSolutions.P58PrimeNodes.p_261582103.prime
    exact decide_eq_true prime
theorem row_count_8087 : row 8087 = 1 := (row_8087.checked).trans rfl
theorem model_count_8087 : modelRow 8087 = 1 := by decide
theorem modelrow_agrees_8087 : modelRow 8087 = row 8087 := model_count_8087.trans row_count_8087.symm
def row_8088 : RowCertificate 8088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8088 1) := by change ¬ Nat.Prime 261679153; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13772587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8088 2) := by change ¬ Nat.Prime 261662977; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4289557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8088 3) := by change ¬ Nat.Prime 261646801; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20126677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8088 : row 8088 = 0 := (row_8088.checked).trans rfl
theorem model_count_8088 : modelRow 8088 = 0 := by decide
theorem modelrow_agrees_8088 : modelRow 8088 = row 8088 := model_count_8088.trans row_count_8088.symm
def row_8089 : RowCertificate 8089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8089 1) := by change ¬ Nat.Prime 261743863; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4290883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8089 2) := by change ¬ Nat.Prime 261727685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52345537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8089 3) := by change ¬ Nat.Prime 261711507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87237169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8089 : row 8089 = 0 := (row_8089.checked).trans rfl
theorem model_count_8089 : modelRow 8089 = 0 := by decide
theorem modelrow_agrees_8089 : modelRow 8089 = row 8089 := model_count_8089.trans row_count_8089.symm
def row_8090 : RowCertificate 8090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8090 1) := by change ¬ Nat.Prime 261808581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87269527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8090 2) := by change ¬ Nat.Prime 261792401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20137877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8090 3) := by change ¬ Nat.Prime 261776221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37396603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8090 : row 8090 = 0 := (row_8090.checked).trans rfl
theorem model_count_8090 : modelRow 8090 = 0 := by decide
theorem modelrow_agrees_8090 : modelRow 8090 = row 8090 := model_count_8090.trans row_count_8090.symm
def row_8091 : RowCertificate 8091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8091 1) := by change ¬ Nat.Prime 261873307; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 945391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8091 2) := by change ¬ Nat.Prime 261857125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52371425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8091 3) := by change ¬ Nat.Prime 261840943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37405849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8091 : row 8091 = 0 := (row_8091.checked).trans rfl
theorem model_count_8091 : modelRow 8091 = 0 := by decide
theorem modelrow_agrees_8091 : modelRow 8091 = row 8091 := model_count_8091.trans row_count_8091.symm
def row_8092 : RowCertificate 8092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8092 1) := by change ¬ Nat.Prime 261938041; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3909523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8092 2) := by change Nat.Prime 261921857; exact ProjectEulerSolutions.P58PrimeNodes.p_261921857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8092 3) := by change ¬ Nat.Prime 261905673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87301891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8092 : row 8092 = 1 := (row_8092.checked).trans rfl
theorem model_count_8092 : modelRow 8092 = 1 := by decide
theorem modelrow_agrees_8092 : modelRow 8092 = row 8092 := model_count_8092.trans row_count_8092.symm
def row_8093 : RowCertificate 8093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8093 1) := by change ¬ Nat.Prime 262002783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87334261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8093 2) := by change ¬ Nat.Prime 261986597; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6389917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8093 3) := by change Nat.Prime 261970411; exact ProjectEulerSolutions.P58PrimeNodes.p_261970411.prime
    exact decide_eq_true prime
theorem row_count_8093 : row 8093 = 1 := (row_8093.checked).trans rfl
theorem model_count_8093 : modelRow 8093 = 1 := by decide
theorem modelrow_agrees_8093 : modelRow 8093 = row 8093 := model_count_8093.trans row_count_8093.symm
def row_8094 : RowCertificate 8094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8094 1) := by change ¬ Nat.Prime 262067533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37438219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8094 2) := by change ¬ Nat.Prime 262051345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52410269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8094 3) := by change ¬ Nat.Prime 262035157; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8452747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8094 : row 8094 = 0 := (row_8094.checked).trans rfl
theorem model_count_8094 : modelRow 8094 = 0 := by decide
theorem modelrow_agrees_8094 : modelRow 8094 = row 8094 := model_count_8094.trans row_count_8094.symm
def row_8095 : RowCertificate 8095 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8095 1) := by change Nat.Prime 262132291; exact ProjectEulerSolutions.P58PrimeNodes.p_262132291.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8095 2) := by change ¬ Nat.Prime 262116101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20162777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8095 3) := by change ¬ Nat.Prime 262099911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87366637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8095 : row 8095 = 1 := (row_8095.checked).trans rfl
theorem model_count_8095 : modelRow 8095 = 1 := by decide
theorem modelrow_agrees_8095 : modelRow 8095 = row 8095 := model_count_8095.trans row_count_8095.symm
def row_8096 : RowCertificate 8096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8096 1) := by change ¬ Nat.Prime 262197057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87399019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8096 2) := by change ¬ Nat.Prime 262180865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52436173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8096 3) := by change ¬ Nat.Prime 262164673; exact PrattCertificates.composite_of_product (factor := 6217) (cofactor := 42169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8096 : row 8096 = 0 := (row_8096.checked).trans rfl
theorem model_count_8096 : modelRow 8096 = 0 := by decide
theorem modelrow_agrees_8096 : modelRow 8096 = row 8096 := model_count_8096.trans row_count_8096.symm
def row_8097 : RowCertificate 8097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8097 1) := by change ¬ Nat.Prime 262261831; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20173987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8097 2) := by change ¬ Nat.Prime 262245637; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9042953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8097 3) := by change ¬ Nat.Prime 262229443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37461349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8097 : row 8097 = 0 := (row_8097.checked).trans rfl
theorem model_count_8097 : modelRow 8097 = 0 := by decide
theorem modelrow_agrees_8097 : modelRow 8097 = row 8097 := model_count_8097.trans row_count_8097.symm
def row_8098 : RowCertificate 8098 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8098 1) := by change Nat.Prime 262326613; exact ProjectEulerSolutions.P58PrimeNodes.p_262326613.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8098 2) := by change ¬ Nat.Prime 262310417; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 743089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8098 3) := by change ¬ Nat.Prime 262294221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87431407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8098 : row 8098 = 1 := (row_8098.checked).trans rfl
theorem model_count_8098 : modelRow 8098 = 1 := by decide
theorem modelrow_agrees_8098 : modelRow 8098 = row 8098 := model_count_8098.trans row_count_8098.symm
def row_8099 : RowCertificate 8099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8099 1) := by change ¬ Nat.Prime 262391403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87463801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8099 2) := by change ¬ Nat.Prime 262375205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52475041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8099 3) := by change ¬ Nat.Prime 262359007; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3593959) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8099 : row 8099 = 0 := (row_8099.checked).trans rfl
theorem model_count_8099 : modelRow 8099 = 0 := by decide
theorem modelrow_agrees_8099 : modelRow 8099 = row 8099 := model_count_8099.trans row_count_8099.symm
def row_8100 : RowCertificate 8100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8100 1) := by change ¬ Nat.Prime 262456201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37493743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8100 2) := by change ¬ Nat.Prime 262440001; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7092973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8100 3) := by change ¬ Nat.Prime 262423801; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13811779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8100 : row 8100 = 0 := (row_8100.checked).trans rfl
theorem model_count_8100 : modelRow 8100 = 0 := by decide
theorem modelrow_agrees_8100 : modelRow 8100 = row 8100 := model_count_8100.trans row_count_8100.symm
def row_8101 : RowCertificate 8101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8101 1) := by change ¬ Nat.Prime 262521007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37503001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8101 2) := by change ¬ Nat.Prime 262504805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52500961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8101 3) := by change ¬ Nat.Prime 262488603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87496201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8101 : row 8101 = 0 := (row_8101.checked).trans rfl
theorem model_count_8101 : modelRow 8101 = 0 := by decide
theorem modelrow_agrees_8101 : modelRow 8101 = row 8101 := model_count_8101.trans row_count_8101.symm
def row_8102 : RowCertificate 8102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8102 1) := by change ¬ Nat.Prime 262585821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87528607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8102 2) := by change ¬ Nat.Prime 262569617; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6404137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8102 3) := by change Nat.Prime 262553413; exact ProjectEulerSolutions.P58PrimeNodes.p_262553413.prime
    exact decide_eq_true prime
theorem row_count_8102 : row 8102 = 1 := (row_8102.checked).trans rfl
theorem model_count_8102 : modelRow 8102 = 1 := by decide
theorem modelrow_agrees_8102 : modelRow 8102 = row 8102 := model_count_8102.trans row_count_8102.symm
def row_8103 : RowCertificate 8103 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8103 1) := by change Nat.Prime 262650643; exact ProjectEulerSolutions.P58PrimeNodes.p_262650643.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8103 2) := by change ¬ Nat.Prime 262634437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20202649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8103 3) := by change ¬ Nat.Prime 262618231; exact PrattCertificates.composite_of_product (factor := 1567) (cofactor := 167593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8103 : row 8103 = 1 := (row_8103.checked).trans rfl
theorem model_count_8103 : modelRow 8103 = 1 := by decide
theorem modelrow_agrees_8103 : modelRow 8103 = row 8103 := model_count_8103.trans row_count_8103.symm
def row_8104 : RowCertificate 8104 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8104 1) := by change Nat.Prime 262715473; exact ProjectEulerSolutions.P58PrimeNodes.p_262715473.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8104 2) := by change ¬ Nat.Prime 262699265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52539853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8104 3) := by change ¬ Nat.Prime 262683057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87561019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8104 : row 8104 = 1 := (row_8104.checked).trans rfl
theorem model_count_8104 : modelRow 8104 = 1 := by decide
theorem modelrow_agrees_8104 : modelRow 8104 = row 8104 := model_count_8104.trans row_count_8104.symm
def row_8105 : RowCertificate 8105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8105 1) := by change ¬ Nat.Prime 262780311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87593437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8105 2) := by change ¬ Nat.Prime 262764101; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 390437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8105 3) := by change ¬ Nat.Prime 262747891; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37535413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8105 : row 8105 = 0 := (row_8105.checked).trans rfl
theorem model_count_8105 : modelRow 8105 = 0 := by decide
theorem modelrow_agrees_8105 : modelRow 8105 = row 8105 := model_count_8105.trans row_count_8105.symm
def row_8106 : RowCertificate 8106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8106 1) := by change ¬ Nat.Prime 262845157; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4308937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8106 2) := by change ¬ Nat.Prime 262828945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52565789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8106 3) := by change ¬ Nat.Prime 262812733; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1320667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8106 : row 8106 = 0 := (row_8106.checked).trans rfl
theorem model_count_8106 : modelRow 8106 = 0 := by decide
theorem modelrow_agrees_8106 : modelRow 8106 = row 8106 := model_count_8106.trans row_count_8106.symm
def row_8107 : RowCertificate 8107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8107 1) := by change ¬ Nat.Prime 262910011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37558573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8107 2) := by change ¬ Nat.Prime 262893797; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15464341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8107 3) := by change ¬ Nat.Prime 262877583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87625861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8107 : row 8107 = 0 := (row_8107.checked).trans rfl
theorem model_count_8107 : modelRow 8107 = 0 := by decide
theorem modelrow_agrees_8107 : modelRow 8107 = row 8107 := model_count_8107.trans row_count_8107.symm
def row_8108 : RowCertificate 8108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8108 1) := by change ¬ Nat.Prime 262974873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87658291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8108 2) := by change ¬ Nat.Prime 262958657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20227589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8108 3) := by change ¬ Nat.Prime 262942441; exact PrattCertificates.composite_of_product (factor := 13789) (cofactor := 19069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8108 : row 8108 = 0 := (row_8108.checked).trans rfl
theorem model_count_8108 : modelRow 8108 = 0 := by decide
theorem modelrow_agrees_8108 : modelRow 8108 = row 8108 := model_count_8108.trans row_count_8108.symm
def row_8109 : RowCertificate 8109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8109 1) := by change ¬ Nat.Prime 263039743; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13844197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8109 2) := by change ¬ Nat.Prime 263023525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52604705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8109 3) := by change ¬ Nat.Prime 263007307; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6116449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8109 : row 8109 = 0 := (row_8109.checked).trans rfl
theorem model_count_8109 : modelRow 8109 = 0 := by decide
theorem modelrow_agrees_8109 : modelRow 8109 = row 8109 := model_count_8109.trans row_count_8109.symm
def row_8110 : RowCertificate 8110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8110 1) := by change ¬ Nat.Prime 263104621; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20238817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8110 2) := by change Nat.Prime 263088401; exact ProjectEulerSolutions.P58PrimeNodes.p_263088401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8110 3) := by change ¬ Nat.Prime 263072181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87690727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8110 : row 8110 = 1 := (row_8110.checked).trans rfl
theorem model_count_8110 : modelRow 8110 = 1 := by decide
theorem modelrow_agrees_8110 : modelRow 8110 = row 8110 := model_count_8110.trans row_count_8110.symm
def row_8111 : RowCertificate 8111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8111 1) := by change ¬ Nat.Prime 263169507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87723169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8111 2) := by change ¬ Nat.Prime 263153285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52630657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8111 3) := by change ¬ Nat.Prime 263137063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37591009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8111 : row 8111 = 0 := (row_8111.checked).trans rfl
theorem model_count_8111 : modelRow 8111 = 0 := by decide
theorem modelrow_agrees_8111 : modelRow 8111 = row 8111 := model_count_8111.trans row_count_8111.symm
def row_8112 : RowCertificate 8112 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8112 1) := by change Nat.Prime 263234401; exact ProjectEulerSolutions.P58PrimeNodes.p_263234401.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8112 2) := by change Nat.Prime 263218177; exact ProjectEulerSolutions.P58PrimeNodes.p_263218177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8112 3) := by change ¬ Nat.Prime 263201953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37600279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8112 : row 8112 = 2 := (row_8112.checked).trans rfl
theorem model_count_8112 : modelRow 8112 = 2 := by decide
theorem modelrow_agrees_8112 : modelRow 8112 = row 8112 := model_count_8112.trans row_count_8112.symm
def row_8113 : RowCertificate 8113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8113 1) := by change ¬ Nat.Prime 263299303; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 950539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8113 2) := by change Nat.Prime 263283077; exact ProjectEulerSolutions.P58PrimeNodes.p_263283077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8113 3) := by change ¬ Nat.Prime 263266851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87755617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8113 : row 8113 = 1 := (row_8113.checked).trans rfl
theorem model_count_8113 : modelRow 8113 = 1 := by decide
theorem modelrow_agrees_8113 : modelRow 8113 = row 8113 := model_count_8113.trans row_count_8113.symm
def row_8114 : RowCertificate 8114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8114 1) := by change ¬ Nat.Prime 263364213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87788071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8114 2) := by change ¬ Nat.Prime 263347985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52669597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8114 3) := by change ¬ Nat.Prime 263331757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20256289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8114 : row 8114 = 0 := (row_8114.checked).trans rfl
theorem model_count_8114 : modelRow 8114 = 0 := by decide
theorem modelrow_agrees_8114 : modelRow 8114 = row 8114 := model_count_8114.trans row_count_8114.symm
def row_8115 : RowCertificate 8115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8115 1) := by change ¬ Nat.Prime 263429131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37632733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8115 2) := by change Nat.Prime 263412901; exact ProjectEulerSolutions.P58PrimeNodes.p_263412901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8115 3) := by change Nat.Prime 263396671; exact ProjectEulerSolutions.P58PrimeNodes.p_263396671.prime
    exact decide_eq_true prime
theorem row_count_8115 : row 8115 = 2 := (row_8115.checked).trans rfl
theorem model_count_8115 : modelRow 8115 = 2 := by decide
theorem modelrow_agrees_8115 : modelRow 8115 = row 8115 := model_count_8115.trans row_count_8115.symm
def row_8116 : RowCertificate 8116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8116 1) := by change ¬ Nat.Prime 263494057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7121461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8116 2) := by change ¬ Nat.Prime 263477825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52695565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8116 3) := by change ¬ Nat.Prime 263461593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87820531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8116 : row 8116 = 0 := (row_8116.checked).trans rfl
theorem model_count_8116 : modelRow 8116 = 0 := by decide
theorem modelrow_agrees_8116 : modelRow 8116 = row 8116 := model_count_8116.trans row_count_8116.symm
def row_8117 : RowCertificate 8117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8117 1) := by change ¬ Nat.Prime 263558991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87852997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8117 2) := by change ¬ Nat.Prime 263542757; exact PrattCertificates.composite_of_product (factor := 2341) (cofactor := 112577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8117 3) := by change ¬ Nat.Prime 263526523; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20271271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8117 : row 8117 = 0 := (row_8117.checked).trans rfl
theorem model_count_8117 : modelRow 8117 = 0 := by decide
theorem modelrow_agrees_8117 : modelRow 8117 = row 8117 := model_count_8117.trans row_count_8117.symm
def row_8118 : RowCertificate 8118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8118 1) := by change ¬ Nat.Prime 263623933; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2075779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8118 2) := by change ¬ Nat.Prime 263607697; exact PrattCertificates.composite_of_product (factor := 617) (cofactor := 427241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8118 3) := by change ¬ Nat.Prime 263591461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37655923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8118 : row 8118 = 0 := (row_8118.checked).trans rfl
theorem model_count_8118 : modelRow 8118 = 0 := by decide
theorem modelrow_agrees_8118 : modelRow 8118 = row 8118 := model_count_8118.trans row_count_8118.symm
def row_8119 : RowCertificate 8119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8119 1) := by change ¬ Nat.Prime 263688883; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8506093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8119 2) := by change ¬ Nat.Prime 263672645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52734529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8119 3) := by change ¬ Nat.Prime 263656407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87885469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8119 : row 8119 = 0 := (row_8119.checked).trans rfl
theorem model_count_8119 : modelRow 8119 = 0 := by decide
theorem modelrow_agrees_8119 : modelRow 8119 = row 8119 := model_count_8119.trans row_count_8119.symm
def row_8120 : RowCertificate 8120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8120 1) := by change ¬ Nat.Prime 263753841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87917947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8120 2) := by change Nat.Prime 263737601; exact ProjectEulerSolutions.P58PrimeNodes.p_263737601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8120 3) := by change ¬ Nat.Prime 263721361; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4323301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8120 : row 8120 = 1 := (row_8120.checked).trans rfl
theorem model_count_8120 : modelRow 8120 = 1 := by decide
theorem modelrow_agrees_8120 : modelRow 8120 = row 8120 := model_count_8120.trans row_count_8120.symm
def row_8121 : RowCertificate 8121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8121 1) := by change ¬ Nat.Prime 263818807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37688401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8121 2) := by change ¬ Nat.Prime 263802565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52760513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8121 3) := by change ¬ Nat.Prime 263786323; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1618321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8121 : row 8121 = 0 := (row_8121.checked).trans rfl
theorem model_count_8121 : modelRow 8121 = 0 := by decide
theorem modelrow_agrees_8121 : modelRow 8121 = row 8121 := model_count_8121.trans row_count_8121.symm
def row_8122 : RowCertificate 8122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8122 1) := by change ¬ Nat.Prime 263883781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37697683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8122 2) := by change Nat.Prime 263867537; exact ProjectEulerSolutions.P58PrimeNodes.p_263867537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8122 3) := by change ¬ Nat.Prime 263851293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87950431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8122 : row 8122 = 1 := (row_8122.checked).trans rfl
theorem model_count_8122 : modelRow 8122 = 1 := by decide
theorem modelrow_agrees_8122 : modelRow 8122 = row 8122 := model_count_8122.trans row_count_8122.symm
def row_8123 : RowCertificate 8123 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8123 1) := by change ¬ Nat.Prime 263948763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 87982921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8123 2) := by change ¬ Nat.Prime 263932517; exact PrattCertificates.composite_of_product (factor := 2857) (cofactor := 92381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8123 3) := by change ¬ Nat.Prime 263916271; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 696349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8123 : row 8123 = 0 := (row_8123.checked).trans rfl
theorem model_count_8123 : modelRow 8123 = 0 := by decide
theorem modelrow_agrees_8123 : modelRow 8123 = row 8123 := model_count_8123.trans row_count_8123.symm
def row_8124 : RowCertificate 8124 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8124 1) := by change Nat.Prime 264013753; exact ProjectEulerSolutions.P58PrimeNodes.p_264013753.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8124 2) := by change ¬ Nat.Prime 263997505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52799501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8124 3) := by change ¬ Nat.Prime 263981257; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6139099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8124 : row 8124 = 1 := (row_8124.checked).trans rfl
theorem model_count_8124 : modelRow 8124 = 1 := by decide
theorem modelrow_agrees_8124 : modelRow 8124 = row 8124 := model_count_8124.trans row_count_8124.symm
def row_8125 : RowCertificate 8125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8125 1) := by change ¬ Nat.Prime 264078751; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3342769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8125 2) := by change Nat.Prime 264062501; exact ProjectEulerSolutions.P58PrimeNodes.p_264062501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8125 3) := by change ¬ Nat.Prime 264046251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88015417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8125 : row 8125 = 1 := (row_8125.checked).trans rfl
theorem model_count_8125 : modelRow 8125 = 1 := by decide
theorem modelrow_agrees_8125 : modelRow 8125 = row 8125 := model_count_8125.trans row_count_8125.symm
def row_8126 : RowCertificate 8126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8126 1) := by change ¬ Nat.Prime 264143757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88047919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8126 2) := by change ¬ Nat.Prime 264127505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52825501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8126 3) := by change ¬ Nat.Prime 264111253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37730179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8126 : row 8126 = 0 := (row_8126.checked).trans rfl
theorem model_count_8126 : modelRow 8126 = 0 := by decide
theorem modelrow_agrees_8126 : modelRow 8126 = row 8126 := model_count_8126.trans row_count_8126.symm
def row_8127 : RowCertificate 8127 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8127 1) := by change Nat.Prime 264208771; exact ProjectEulerSolutions.P58PrimeNodes.p_264208771.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8127 2) := by change Nat.Prime 264192517; exact ProjectEulerSolutions.P58PrimeNodes.p_264192517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8127 3) := by change ¬ Nat.Prime 264176263; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20321251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8127 : row 8127 = 2 := (row_8127.checked).trans rfl
theorem model_count_8127 : modelRow 8127 = 2 := by decide
theorem modelrow_agrees_8127 : modelRow 8127 = row 8127 := model_count_8127.trans row_count_8127.symm
def row_8128 : RowCertificate 8128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8128 1) := by change ¬ Nat.Prime 264273793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37753399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8128 2) := by change ¬ Nat.Prime 264257537; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15544561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8128 3) := by change ¬ Nat.Prime 264241281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88080427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8128 : row 8128 = 0 := (row_8128.checked).trans rfl
theorem model_count_8128 : modelRow 8128 = 0 := by decide
theorem modelrow_agrees_8128 : modelRow 8128 = row 8128 := model_count_8128.trans row_count_8128.symm
def row_8129 : RowCertificate 8129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8129 1) := by change ¬ Nat.Prime 264338823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88112941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8129 2) := by change ¬ Nat.Prime 264322565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52864513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8129 3) := by change Nat.Prime 264306307; exact ProjectEulerSolutions.P58PrimeNodes.p_264306307.prime
    exact decide_eq_true prime
theorem row_count_8129 : row 8129 = 1 := (row_8129.checked).trans rfl
theorem model_count_8129 : modelRow 8129 = 1 := by decide
theorem modelrow_agrees_8129 : modelRow 8129 = row 8129 := model_count_8129.trans row_count_8129.symm
def row_8130 : RowCertificate 8130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8130 1) := by change ¬ Nat.Prime 264403861; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6148927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8130 2) := by change Nat.Prime 264387601; exact ProjectEulerSolutions.P58PrimeNodes.p_264387601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8130 3) := by change ¬ Nat.Prime 264371341; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20336257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8130 : row 8130 = 1 := (row_8130.checked).trans rfl
theorem model_count_8130 : modelRow 8130 = 1 := by decide
theorem modelrow_agrees_8130 : modelRow 8130 = row 8130 := model_count_8130.trans row_count_8130.symm
def row_8131 : RowCertificate 8131 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8131 1) := by change Nat.Prime 264468907; exact ProjectEulerSolutions.P58PrimeNodes.p_264468907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8131 2) := by change ¬ Nat.Prime 264452645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52890529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8131 3) := by change ¬ Nat.Prime 264436383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88145461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8131 : row 8131 = 1 := (row_8131.checked).trans rfl
theorem model_count_8131 : modelRow 8131 = 1 := by decide
theorem modelrow_agrees_8131 : modelRow 8131 = row 8131 := model_count_8131.trans row_count_8131.symm
def row_8132 : RowCertificate 8132 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8132 1) := by change ¬ Nat.Prime 264533961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88177987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8132 2) := by change ¬ Nat.Prime 264517697; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 218069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8132 3) := by change ¬ Nat.Prime 264501433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37785919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8132 : row 8132 = 0 := (row_8132.checked).trans rfl
theorem model_count_8132 : modelRow 8132 = 0 := by decide
theorem modelrow_agrees_8132 : modelRow 8132 = row 8132 := model_count_8132.trans row_count_8132.symm
def row_8133 : RowCertificate 8133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8133 1) := by change ¬ Nat.Prime 264599023; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20353771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8133 2) := by change ¬ Nat.Prime 264582757; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 1931261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8133 3) := by change ¬ Nat.Prime 264566491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37795213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8133 : row 8133 = 0 := (row_8133.checked).trans rfl
theorem model_count_8133 : modelRow 8133 = 0 := by decide
theorem modelrow_agrees_8133 : modelRow 8133 = row 8133 := model_count_8133.trans row_count_8133.symm
def row_8134 : RowCertificate 8134 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8134 1) := by change Nat.Prime 264664093; exact ProjectEulerSolutions.P58PrimeNodes.p_264664093.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8134 2) := by change ¬ Nat.Prime 264647825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52929565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8134 3) := by change ¬ Nat.Prime 264631557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88210519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8134 : row 8134 = 1 := (row_8134.checked).trans rfl
theorem model_count_8134 : modelRow 8134 = 1 := by decide
theorem modelrow_agrees_8134 : modelRow 8134 = row 8134 := model_count_8134.trans row_count_8134.symm
def row_8135 : RowCertificate 8135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8135 1) := by change ¬ Nat.Prime 264729171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88243057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8135 2) := by change ¬ Nat.Prime 264712901; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 903457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8135 3) := by change ¬ Nat.Prime 264696631; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8538601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8135 : row 8135 = 0 := (row_8135.checked).trans rfl
theorem model_count_8135 : modelRow 8135 = 0 := by decide
theorem modelrow_agrees_8135 : modelRow 8135 = row 8135 := model_count_8135.trans row_count_8135.symm
def row_8136 : RowCertificate 8136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8136 1) := by change ¬ Nat.Prime 264794257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37827751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8136 2) := by change ¬ Nat.Prime 264777985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52955597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8136 3) := by change ¬ Nat.Prime 264761713; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13934827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8136 : row 8136 = 0 := (row_8136.checked).trans rfl
theorem model_count_8136 : modelRow 8136 = 0 := by decide
theorem modelrow_agrees_8136 : modelRow 8136 = row 8136 := model_count_8136.trans row_count_8136.symm
def row_8137 : RowCertificate 8137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8137 1) := by change ¬ Nat.Prime 264859351; exact PrattCertificates.composite_of_product (factor := 1483) (cofactor := 178597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8137 2) := by change ¬ Nat.Prime 264843077; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7157921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8137 3) := by change ¬ Nat.Prime 264826803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88275601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8137 : row 8137 = 0 := (row_8137.checked).trans rfl
theorem model_count_8137 : modelRow 8137 = 0 := by decide
theorem modelrow_agrees_8137 : modelRow 8137 = row 8137 := model_count_8137.trans row_count_8137.symm
def row_8138 : RowCertificate 8138 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8138 1) := by change ¬ Nat.Prime 264924453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88308151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8138 2) := by change ¬ Nat.Prime 264908177; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4342757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8138 3) := by change ¬ Nat.Prime 264891901; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13941679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8138 : row 8138 = 0 := (row_8138.checked).trans rfl
theorem model_count_8138 : modelRow 8138 = 0 := by decide
theorem modelrow_agrees_8138 : modelRow 8138 = row 8138 := model_count_8138.trans row_count_8138.symm
def row_8139 : RowCertificate 8139 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8139 1) := by change Nat.Prime 264989563; exact ProjectEulerSolutions.P58PrimeNodes.p_264989563.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8139 2) := by change ¬ Nat.Prime 264973285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52994657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8139 3) := by change ¬ Nat.Prime 264957007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37851001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8139 : row 8139 = 1 := (row_8139.checked).trans rfl
theorem model_count_8139 : modelRow 8139 = 1 := by decide
theorem modelrow_agrees_8139 : modelRow 8139 = row 8139 := model_count_8139.trans row_count_8139.symm
def row_8140 : RowCertificate 8140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8140 1) := by change ¬ Nat.Prime 265054681; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8550151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8140 2) := by change ¬ Nat.Prime 265038401; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 750817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8140 3) := by change ¬ Nat.Prime 265022121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88340707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8140 : row 8140 = 0 := (row_8140.checked).trans rfl
theorem model_count_8140 : modelRow 8140 = 0 := by decide
theorem modelrow_agrees_8140 : modelRow 8140 = row 8140 := model_count_8140.trans row_count_8140.symm
def row_8141 : RowCertificate 8141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8141 1) := by change ¬ Nat.Prime 265119807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88373269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8141 2) := by change ¬ Nat.Prime 265103525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53020705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8141 3) := by change Nat.Prime 265087243; exact ProjectEulerSolutions.P58PrimeNodes.p_265087243.prime
    exact decide_eq_true prime
theorem row_count_8141 : row 8141 = 1 := (row_8141.checked).trans rfl
theorem model_count_8141 : modelRow 8141 = 1 := by decide
theorem modelrow_agrees_8141 : modelRow 8141 = row 8141 := model_count_8141.trans row_count_8141.symm
def row_8142 : RowCertificate 8142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8142 1) := by change ¬ Nat.Prime 265184941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37883563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8142 2) := by change ¬ Nat.Prime 265168657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20397589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8142 3) := by change Nat.Prime 265152373; exact ProjectEulerSolutions.P58PrimeNodes.p_265152373.prime
    exact decide_eq_true prime
theorem row_count_8142 : row 8142 = 1 := (row_8142.checked).trans rfl
theorem model_count_8142 : modelRow 8142 = 1 := by decide
theorem modelrow_agrees_8142 : modelRow 8142 = row 8142 := model_count_8142.trans row_count_8142.symm
def row_8143 : RowCertificate 8143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8143 1) := by change ¬ Nat.Prime 265250083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37892869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8143 2) := by change ¬ Nat.Prime 265233797; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9145993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8143 3) := by change ¬ Nat.Prime 265217511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88405837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8143 : row 8143 = 0 := (row_8143.checked).trans rfl
theorem model_count_8143 : modelRow 8143 = 0 := by decide
theorem modelrow_agrees_8143 : modelRow 8143 = row 8143 := model_count_8143.trans row_count_8143.symm
def row_8144 : RowCertificate 8144 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8144 1) := by change ¬ Nat.Prime 265315233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88438411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8144 2) := by change ¬ Nat.Prime 265298945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53059789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8144 3) := by change ¬ Nat.Prime 265282657; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3634009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8144 : row 8144 = 0 := (row_8144.checked).trans rfl
theorem model_count_8144 : modelRow 8144 = 0 := by decide
theorem modelrow_agrees_8144 : modelRow 8144 = row 8144 := model_count_8144.trans row_count_8144.symm
def row_8145 : RowCertificate 8145 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8145 1) := by change ¬ Nat.Prime 265380391; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13967389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8145 2) := by change ¬ Nat.Prime 265364101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15609653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8145 3) := by change ¬ Nat.Prime 265347811; exact PrattCertificates.composite_of_product (factor := 3217) (cofactor := 82483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8145 : row 8145 = 0 := (row_8145.checked).trans rfl
theorem model_count_8145 : modelRow 8145 = 0 := by decide
theorem modelrow_agrees_8145 : modelRow 8145 = row 8145 := model_count_8145.trans row_count_8145.symm
def row_8146 : RowCertificate 8146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8146 1) := by change ¬ Nat.Prime 265445557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20418889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8146 2) := by change ¬ Nat.Prime 265429265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53085853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8146 3) := by change ¬ Nat.Prime 265412973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88470991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8146 : row 8146 = 0 := (row_8146.checked).trans rfl
theorem model_count_8146 : modelRow 8146 = 0 := by decide
theorem modelrow_agrees_8146 : modelRow 8146 = row 8146 := model_count_8146.trans row_count_8146.symm
def row_8147 : RowCertificate 8147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8147 1) := by change ¬ Nat.Prime 265510731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88503577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8147 2) := by change ¬ Nat.Prime 265494437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20422649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8147 3) := by change ¬ Nat.Prime 265478143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37925449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8147 : row 8147 = 0 := (row_8147.checked).trans rfl
theorem model_count_8147 : modelRow 8147 = 0 := by decide
theorem modelrow_agrees_8147 : modelRow 8147 = row 8147 := model_count_8147.trans row_count_8147.symm
def row_8148 : RowCertificate 8148 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8148 1) := by change ¬ Nat.Prime 265575913; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1376041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8148 2) := by change Nat.Prime 265559617; exact ProjectEulerSolutions.P58PrimeNodes.p_265559617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8148 3) := by change ¬ Nat.Prime 265543321; exact PrattCertificates.composite_of_product (factor := 2473) (cofactor := 107377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8148 : row 8148 = 1 := (row_8148.checked).trans rfl
theorem model_count_8148 : modelRow 8148 = 1 := by decide
theorem modelrow_agrees_8148 : modelRow 8148 = row 8148 := model_count_8148.trans row_count_8148.symm
def row_8149 : RowCertificate 8149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8149 1) := by change ¬ Nat.Prime 265641103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37948729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8149 2) := by change ¬ Nat.Prime 265624805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53124961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8149 3) := by change ¬ Nat.Prime 265608507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88536169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8149 : row 8149 = 0 := (row_8149.checked).trans rfl
theorem model_count_8149 : modelRow 8149 = 0 := by decide
theorem modelrow_agrees_8149 : modelRow 8149 = row 8149 := model_count_8149.trans row_count_8149.symm
def row_8150 : RowCertificate 8150 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8150 1) := by change ¬ Nat.Prime 265706301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88568767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8150 2) := by change ¬ Nat.Prime 265690001; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1692293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8150 3) := by change Nat.Prime 265673701; exact ProjectEulerSolutions.P58PrimeNodes.p_265673701.prime
    exact decide_eq_true prime
theorem row_count_8150 : row 8150 = 1 := (row_8150.checked).trans rfl
theorem model_count_8150 : modelRow 8150 = 1 := by decide
theorem modelrow_agrees_8150 : modelRow 8150 = row 8150 := model_count_8150.trans row_count_8150.symm
def row_8151 : RowCertificate 8151 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8151 1) := by change Nat.Prime 265771507; exact ProjectEulerSolutions.P58PrimeNodes.p_265771507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8151 2) := by change ¬ Nat.Prime 265755205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53151041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8151 3) := by change Nat.Prime 265738903; exact ProjectEulerSolutions.P58PrimeNodes.p_265738903.prime
    exact decide_eq_true prime
theorem row_count_8151 : row 8151 = 2 := (row_8151.checked).trans rfl
theorem model_count_8151 : modelRow 8151 = 2 := by decide
theorem modelrow_agrees_8151 : modelRow 8151 = row 8151 := model_count_8151.trans row_count_8151.symm
def row_8152 : RowCertificate 8152 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8152 1) := by change ¬ Nat.Prime 265836721; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2438869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8152 2) := by change Nat.Prime 265820417; exact ProjectEulerSolutions.P58PrimeNodes.p_265820417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8152 3) := by change ¬ Nat.Prime 265804113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88601371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8152 : row 8152 = 1 := (row_8152.checked).trans rfl
theorem model_count_8152 : modelRow 8152 = 1 := by decide
theorem modelrow_agrees_8152 : modelRow 8152 = row 8152 := model_count_8152.trans row_count_8152.symm
def row_8153 : RowCertificate 8153 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8153 1) := by change ¬ Nat.Prime 265901943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88633981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8153 2) := by change ¬ Nat.Prime 265885637; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3642269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8153 3) := by change ¬ Nat.Prime 265869331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37981333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8153 : row 8153 = 0 := (row_8153.checked).trans rfl
theorem model_count_8153 : modelRow 8153 = 0 := by decide
theorem modelrow_agrees_8153 : modelRow 8153 = row 8153 := model_count_8153.trans row_count_8153.symm
def row_8154 : RowCertificate 8154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8154 1) := by change ¬ Nat.Prime 265967173; exact PrattCertificates.composite_of_product (factor := 3067) (cofactor := 86719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8154 2) := by change ¬ Nat.Prime 265950865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53190173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8154 3) := by change ¬ Nat.Prime 265934557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37990651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8154 : row 8154 = 0 := (row_8154.checked).trans rfl
theorem model_count_8154 : modelRow 8154 = 0 := by decide
theorem modelrow_agrees_8154 : modelRow 8154 = row 8154 := model_count_8154.trans row_count_8154.symm
def row_8155 : RowCertificate 8155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8155 1) := by change ¬ Nat.Prime 266032411; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3970633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8155 2) := by change ¬ Nat.Prime 266016101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20462777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8155 3) := by change ¬ Nat.Prime 265999791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88666597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8155 : row 8155 = 0 := (row_8155.checked).trans rfl
theorem model_count_8155 : modelRow 8155 = 0 := by decide
theorem modelrow_agrees_8155 : modelRow 8155 = row 8155 := model_count_8155.trans row_count_8155.symm
def row_8156 : RowCertificate 8156 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8156 1) := by change ¬ Nat.Prime 266097657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88699219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8156 2) := by change ¬ Nat.Prime 266081345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53216269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8156 3) := by change ¬ Nat.Prime 266065033; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20466541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8156 : row 8156 = 0 := (row_8156.checked).trans rfl
theorem model_count_8156 : modelRow 8156 = 0 := by decide
theorem modelrow_agrees_8156 : modelRow 8156 = row 8156 := model_count_8156.trans row_count_8156.symm
def row_8157 : RowCertificate 8157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8157 1) := by change ¬ Nat.Prime 266162911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38023273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8157 2) := by change Nat.Prime 266146597; exact ProjectEulerSolutions.P58PrimeNodes.p_266146597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8157 3) := by change ¬ Nat.Prime 266130283; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14006857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8157 : row 8157 = 1 := (row_8157.checked).trans rfl
theorem model_count_8157 : modelRow 8157 = 1 := by decide
theorem modelrow_agrees_8157 : modelRow 8157 = row 8157 := model_count_8157.trans row_count_8157.symm
def row_8158 : RowCertificate 8158 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8158 1) := by change Nat.Prime 266228173; exact ProjectEulerSolutions.P58PrimeNodes.p_266228173.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8158 2) := by change ¬ Nat.Prime 266211857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15659521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8158 3) := by change ¬ Nat.Prime 266195541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88731847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8158 : row 8158 = 1 := (row_8158.checked).trans rfl
theorem model_count_8158 : modelRow 8158 = 1 := by decide
theorem modelrow_agrees_8158 : modelRow 8158 = row 8158 := model_count_8158.trans row_count_8158.symm
def row_8159 : RowCertificate 8159 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8159 1) := by change ¬ Nat.Prime 266293443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88764481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8159 2) := by change ¬ Nat.Prime 266277125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53255425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8159 3) := by change Nat.Prime 266260807; exact ProjectEulerSolutions.P58PrimeNodes.p_266260807.prime
    exact decide_eq_true prime
theorem row_count_8159 : row 8159 = 1 := (row_8159.checked).trans rfl
theorem model_count_8159 : modelRow 8159 = 1 := by decide
theorem modelrow_agrees_8159 : modelRow 8159 = row 8159 := model_count_8159.trans row_count_8159.symm
def row_8160 : RowCertificate 8160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8160 1) := by change ¬ Nat.Prime 266358721; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2586007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8160 2) := by change ¬ Nat.Prime 266342401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20487877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8160 3) := by change ¬ Nat.Prime 266326081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38046583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8160 : row 8160 = 0 := (row_8160.checked).trans rfl
theorem model_count_8160 : modelRow 8160 = 0 := by decide
theorem modelrow_agrees_8160 : modelRow 8160 = row 8160 := model_count_8160.trans row_count_8160.symm
def row_8161 : RowCertificate 8161 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8161 1) := by change Nat.Prime 266424007; exact ProjectEulerSolutions.P58PrimeNodes.p_266424007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8161 2) := by change ¬ Nat.Prime 266407685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53281537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8161 3) := by change ¬ Nat.Prime 266391363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88797121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8161 : row 8161 = 1 := (row_8161.checked).trans rfl
theorem model_count_8161 : modelRow 8161 = 1 := by decide
theorem modelrow_agrees_8161 : modelRow 8161 = row 8161 := model_count_8161.trans row_count_8161.symm
def row_8162 : RowCertificate 8162 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8162 1) := by change ¬ Nat.Prime 266489301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88829767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8162 2) := by change ¬ Nat.Prime 266472977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15674881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8162 3) := by change ¬ Nat.Prime 266456653; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 714361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8162 : row 8162 = 0 := (row_8162.checked).trans rfl
theorem model_count_8162 : modelRow 8162 = 0 := by decide
theorem modelrow_agrees_8162 : modelRow 8162 = row 8162 := model_count_8162.trans row_count_8162.symm
def row_8163 : RowCertificate 8163 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8163 1) := by change ¬ Nat.Prime 266554603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38079229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8163 2) := by change ¬ Nat.Prime 266538277; exact PrattCertificates.composite_of_product (factor := 7757) (cofactor := 34361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8163 3) := by change ¬ Nat.Prime 266521951; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 547273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8163 : row 8163 = 0 := (row_8163.checked).trans rfl
theorem model_count_8163 : modelRow 8163 = 0 := by decide
theorem modelrow_agrees_8163 : modelRow 8163 = row 8163 := model_count_8163.trans row_count_8163.symm
def row_8164 : RowCertificate 8164 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8164 1) := by change ¬ Nat.Prime 266619913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38088559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8164 2) := by change ¬ Nat.Prime 266603585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53320717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8164 3) := by change ¬ Nat.Prime 266587257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88862419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8164 : row 8164 = 0 := (row_8164.checked).trans rfl
theorem model_count_8164 : modelRow 8164 = 0 := by decide
theorem modelrow_agrees_8164 : modelRow 8164 = row 8164 := model_count_8164.trans row_count_8164.symm
def row_8165 : RowCertificate 8165 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8165 1) := by change ¬ Nat.Prime 266685231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88895077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8165 2) := by change Nat.Prime 266668901; exact ProjectEulerSolutions.P58PrimeNodes.p_266668901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8165 3) := by change ¬ Nat.Prime 266652571; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3375349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8165 : row 8165 = 1 := (row_8165.checked).trans rfl
theorem model_count_8165 : modelRow 8165 = 1 := by decide
theorem modelrow_agrees_8165 : modelRow 8165 = row 8165 := model_count_8165.trans row_count_8165.symm
def row_8166 : RowCertificate 8166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8166 1) := by change ¬ Nat.Prime 266750557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14039503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8166 2) := by change ¬ Nat.Prime 266734225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53346845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8166 3) := by change ¬ Nat.Prime 266717893; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20516761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8166 : row 8166 = 0 := (row_8166.checked).trans rfl
theorem model_count_8166 : modelRow 8166 = 0 := by decide
theorem modelrow_agrees_8166 : modelRow 8166 = row 8166 := model_count_8166.trans row_count_8166.symm
def row_8167 : RowCertificate 8167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8167 1) := by change ¬ Nat.Prime 266815891; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4374031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8167 2) := by change ¬ Nat.Prime 266799557; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 321833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8167 3) := by change ¬ Nat.Prime 266783223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88927741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8167 : row 8167 = 0 := (row_8167.checked).trans rfl
theorem model_count_8167 : modelRow 8167 = 0 := by decide
theorem modelrow_agrees_8167 : modelRow 8167 = row 8167 := model_count_8167.trans row_count_8167.symm
def row_8168 : RowCertificate 8168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8168 1) := by change ¬ Nat.Prime 266881233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88960411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8168 2) := by change ¬ Nat.Prime 266864897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20528069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8168 3) := by change ¬ Nat.Prime 266848561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38121223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8168 : row 8168 = 0 := (row_8168.checked).trans rfl
theorem model_count_8168 : modelRow 8168 = 0 := by decide
theorem modelrow_agrees_8168 : modelRow 8168 = row 8168 := model_count_8168.trans row_count_8168.symm
def row_8169 : RowCertificate 8169 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8169 1) := by change ¬ Nat.Prime 266946583; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1474843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8169 2) := by change ¬ Nat.Prime 266930245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53386049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8169 3) := by change ¬ Nat.Prime 266913907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20531839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8169 : row 8169 = 0 := (row_8169.checked).trans rfl
theorem model_count_8169 : modelRow 8169 = 0 := by decide
theorem modelrow_agrees_8169 : modelRow 8169 = row 8169 := model_count_8169.trans row_count_8169.symm
def row_8170 : RowCertificate 8170 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8170 1) := by change ¬ Nat.Prime 267011941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38144563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8170 2) := by change ¬ Nat.Prime 266995601; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 1948873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8170 3) := by change ¬ Nat.Prime 266979261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88993087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8170 : row 8170 = 0 := (row_8170.checked).trans rfl
theorem model_count_8170 : modelRow 8170 = 0 := by decide
theorem modelrow_agrees_8170 : modelRow 8170 = row 8170 := model_count_8170.trans row_count_8170.symm
def row_8171 : RowCertificate 8171 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8171 1) := by change ¬ Nat.Prime 267077307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89025769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8171 2) := by change ¬ Nat.Prime 267060965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53412193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8171 3) := by change ¬ Nat.Prime 267044623; exact PrattCertificates.composite_of_product (factor := 3931) (cofactor := 67933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8171 : row 8171 = 0 := (row_8171.checked).trans rfl
theorem model_count_8171 : modelRow 8171 = 0 := by decide
theorem modelrow_agrees_8171 : modelRow 8171 = row 8171 := model_count_8171.trans row_count_8171.symm
def row_8172 : RowCertificate 8172 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8172 1) := by change ¬ Nat.Prime 267142681; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20549437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8172 2) := by change ¬ Nat.Prime 267126337; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9211253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8172 3) := by change ¬ Nat.Prime 267109993; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7219189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8172 : row 8172 = 0 := (row_8172.checked).trans rfl
theorem model_count_8172 : modelRow 8172 = 0 := by decide
theorem modelrow_agrees_8172 : modelRow 8172 = row 8172 := model_count_8172.trans row_count_8172.symm
def row_8173 : RowCertificate 8173 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8173 1) := by change ¬ Nat.Prime 267208063; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6214141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8173 2) := by change ¬ Nat.Prime 267191717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20553209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8173 3) := by change ¬ Nat.Prime 267175371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89058457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8173 : row 8173 = 0 := (row_8173.checked).trans rfl
theorem model_count_8173 : modelRow 8173 = 0 := by decide
theorem modelrow_agrees_8173 : modelRow 8173 = row 8173 := model_count_8173.trans row_count_8173.symm
def row_8174 : RowCertificate 8174 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8174 1) := by change ¬ Nat.Prime 267273453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89091151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8174 2) := by change ¬ Nat.Prime 267257105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53451421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8174 3) := by change ¬ Nat.Prime 267240757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38177251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8174 : row 8174 = 0 := (row_8174.checked).trans rfl
theorem model_count_8174 : modelRow 8174 = 0 := by decide
theorem modelrow_agrees_8174 : modelRow 8174 = row 8174 := model_count_8174.trans row_count_8174.symm
def row_8175 : RowCertificate 8175 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8175 1) := by change ¬ Nat.Prime 267338851; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20564527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8175 2) := by change ¬ Nat.Prime 267322501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15724853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8175 3) := by change ¬ Nat.Prime 267306151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38186593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8175 : row 8175 = 0 := (row_8175.checked).trans rfl
theorem model_count_8175 : modelRow 8175 = 0 := by decide
theorem modelrow_agrees_8175 : modelRow 8175 = row 8175 := model_count_8175.trans row_count_8175.symm
def row_8176 : RowCertificate 8176 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8176 1) := by change Nat.Prime 267404257; exact ProjectEulerSolutions.P58PrimeNodes.p_267404257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8176 2) := by change ¬ Nat.Prime 267387905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53477581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8176 3) := by change ¬ Nat.Prime 267371553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89123851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8176 : row 8176 = 1 := (row_8176.checked).trans rfl
theorem model_count_8176 : modelRow 8176 = 1 := by decide
theorem modelrow_agrees_8176 : modelRow 8176 = row 8176 := model_count_8176.trans row_count_8176.symm
def row_8177 : RowCertificate 8177 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8177 1) := by change ¬ Nat.Prime 267469671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89156557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8177 2) := by change ¬ Nat.Prime 267453317; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5046289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8177 3) := by change ¬ Nat.Prime 267436963; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1167847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8177 : row 8177 = 0 := (row_8177.checked).trans rfl
theorem model_count_8177 : modelRow 8177 = 0 := by decide
theorem modelrow_agrees_8177 : modelRow 8177 = row 8177 := model_count_8177.trans row_count_8177.symm
def row_8178 : RowCertificate 8178 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8178 1) := by change ¬ Nat.Prime 267535093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38219299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8178 2) := by change ¬ Nat.Prime 267518737; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1703941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8178 3) := by change ¬ Nat.Prime 267502381; exact PrattCertificates.composite_of_product (factor := 10753) (cofactor := 24877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8178 : row 8178 = 0 := (row_8178.checked).trans rfl
theorem model_count_8178 : modelRow 8178 = 0 := by decide
theorem modelrow_agrees_8178 : modelRow 8178 = row 8178 := model_count_8178.trans row_count_8178.symm
def row_8179 : RowCertificate 8179 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8179 1) := by change ¬ Nat.Prime 267600523; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1641721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8179 2) := by change ¬ Nat.Prime 267584165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53516833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8179 3) := by change ¬ Nat.Prime 267567807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89189269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8179 : row 8179 = 0 := (row_8179.checked).trans rfl
theorem model_count_8179 : modelRow 8179 = 0 := by decide
theorem modelrow_agrees_8179 : modelRow 8179 = row 8179 := model_count_8179.trans row_count_8179.symm
def row_8180 : RowCertificate 8180 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8180 1) := by change ¬ Nat.Prime 267665961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89221987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8180 2) := by change ¬ Nat.Prime 267649601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7233773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8180 3) := by change ¬ Nat.Prime 267633241; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1925419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8180 : row 8180 = 0 := (row_8180.checked).trans rfl
theorem model_count_8180 : modelRow 8180 = 0 := by decide
theorem modelrow_agrees_8180 : modelRow 8180 = row 8180 := model_count_8180.trans row_count_8180.symm
def row_8181 : RowCertificate 8181 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8181 1) := by change ¬ Nat.Prime 267731407; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8636497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8181 2) := by change ¬ Nat.Prime 267715045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53543009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8181 3) := by change ¬ Nat.Prime 267698683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38242669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8181 : row 8181 = 0 := (row_8181.checked).trans rfl
theorem model_count_8181 : modelRow 8181 = 0 := by decide
theorem modelrow_agrees_8181 : modelRow 8181 = row 8181 := model_count_8181.trans row_count_8181.symm
def row_8182 : RowCertificate 8182 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8182 1) := by change ¬ Nat.Prime 267796861; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7237753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8182 2) := by change Nat.Prime 267780497; exact ProjectEulerSolutions.P58PrimeNodes.p_267780497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8182 3) := by change ¬ Nat.Prime 267764133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89254711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8182 : row 8182 = 1 := (row_8182.checked).trans rfl
theorem model_count_8182 : modelRow 8182 = 1 := by decide
theorem modelrow_agrees_8182 : modelRow 8182 = row 8182 := model_count_8182.trans row_count_8182.symm
def row_8183 : RowCertificate 8183 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8183 1) := by change ¬ Nat.Prime 267862323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89287441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8183 2) := by change ¬ Nat.Prime 267845957; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1169633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8183 3) := by change Nat.Prime 267829591; exact ProjectEulerSolutions.P58PrimeNodes.p_267829591.prime
    exact decide_eq_true prime
theorem row_count_8183 : row 8183 = 1 := (row_8183.checked).trans rfl
theorem model_count_8183 : modelRow 8183 = 1 := by decide
theorem modelrow_agrees_8183 : modelRow 8183 = row 8183 := model_count_8183.trans row_count_8183.symm
def row_8184 : RowCertificate 8184 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8184 1) := by change ¬ Nat.Prime 267927793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38275399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8184 2) := by change ¬ Nat.Prime 267911425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53582285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8184 3) := by change Nat.Prime 267895057; exact ProjectEulerSolutions.P58PrimeNodes.p_267895057.prime
    exact decide_eq_true prime
theorem row_count_8184 : row 8184 = 1 := (row_8184.checked).trans rfl
theorem model_count_8184 : modelRow 8184 = 1 := by decide
theorem modelrow_agrees_8184 : modelRow 8184 = row 8184 := model_count_8184.trans row_count_8184.symm
def row_8185 : RowCertificate 8185 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8185 1) := by change ¬ Nat.Prime 267993271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38284753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8185 2) := by change ¬ Nat.Prime 267976901; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2371477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8185 3) := by change ¬ Nat.Prime 267960531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89320177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8185 : row 8185 = 0 := (row_8185.checked).trans rfl
theorem model_count_8185 : modelRow 8185 = 0 := by decide
theorem modelrow_agrees_8185 : modelRow 8185 = row 8185 := model_count_8185.trans row_count_8185.symm
def row_8186 : RowCertificate 8186 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8186 1) := by change ¬ Nat.Prime 268058757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89352919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8186 2) := by change ¬ Nat.Prime 268042385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53608477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8186 3) := by change Nat.Prime 268026013; exact ProjectEulerSolutions.P58PrimeNodes.p_268026013.prime
    exact decide_eq_true prime
theorem row_count_8186 : row 8186 = 1 := (row_8186.checked).trans rfl
theorem model_count_8186 : modelRow 8186 = 1 := by decide
theorem modelrow_agrees_8186 : modelRow 8186 = row 8186 := model_count_8186.trans row_count_8186.symm
def row_8187 : RowCertificate 8187 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8187 1) := by change ¬ Nat.Prime 268124251; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 856627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8187 2) := by change Nat.Prime 268107877; exact ProjectEulerSolutions.P58PrimeNodes.p_268107877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8187 3) := by change ¬ Nat.Prime 268091503; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8648113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8187 : row 8187 = 1 := (row_8187.checked).trans rfl
theorem model_count_8187 : modelRow 8187 = 1 := by decide
theorem modelrow_agrees_8187 : modelRow 8187 = row 8187 := model_count_8187.trans row_count_8187.symm
def row_8188 : RowCertificate 8188 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8188 1) := by change ¬ Nat.Prime 268189753; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20629981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8188 2) := by change ¬ Nat.Prime 268173377; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 165437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8188 3) := by change ¬ Nat.Prime 268157001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89385667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8188 : row 8188 = 0 := (row_8188.checked).trans rfl
theorem model_count_8188 : modelRow 8188 = 0 := by decide
theorem modelrow_agrees_8188 : modelRow 8188 = row 8188 := model_count_8188.trans row_count_8188.symm
def row_8189 : RowCertificate 8189 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8189 1) := by change ¬ Nat.Prime 268255263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89418421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8189 2) := by change ¬ Nat.Prime 268238885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53647777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8189 3) := by change ¬ Nat.Prime 268222507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38317501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8189 : row 8189 = 0 := (row_8189.checked).trans rfl
theorem model_count_8189 : modelRow 8189 = 0 := by decide
theorem modelrow_agrees_8189 : modelRow 8189 = row 8189 := model_count_8189.trans row_count_8189.symm
def row_8190 : RowCertificate 8190 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8190 1) := by change ¬ Nat.Prime 268320781; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7251913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8190 2) := by change Nat.Prime 268304401; exact ProjectEulerSolutions.P58PrimeNodes.p_268304401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8190 3) := by change Nat.Prime 268288021; exact ProjectEulerSolutions.P58PrimeNodes.p_268288021.prime
    exact decide_eq_true prime
theorem row_count_8190 : row 8190 = 2 := (row_8190.checked).trans rfl
theorem model_count_8190 : modelRow 8190 = 2 := by decide
theorem modelrow_agrees_8190 : modelRow 8190 = row 8190 := model_count_8190.trans row_count_8190.symm
def row_8191 : RowCertificate 8191 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8191 1) := by change ¬ Nat.Prime 268386307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 38340901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8191 2) := by change ¬ Nat.Prime 268369925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53673985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8191 3) := by change ¬ Nat.Prime 268353543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89451181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8191 : row 8191 = 0 := (row_8191.checked).trans rfl
theorem model_count_8191 : modelRow 8191 = 0 := by decide
theorem modelrow_agrees_8191 : modelRow 8191 = row 8191 := model_count_8191.trans row_count_8191.symm
def row_8192 : RowCertificate 8192 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8192 1) := by change ¬ Nat.Prime 268451841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89483947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8192 2) := by change ¬ Nat.Prime 268435457; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15790321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8192 3) := by change ¬ Nat.Prime 268419073; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20647621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8192 : row 8192 = 0 := (row_8192.checked).trans rfl
theorem model_count_8192 : modelRow 8192 = 0 := by decide
theorem modelrow_agrees_8192 : modelRow 8192 = row 8192 := model_count_8192.trans row_count_8192.symm
private def atomic_8065 : CountSegment 1 10 13120 8064 8065 3415 3416 := CountSegment.single (first := 8064) (initial := 3415) row_count_8065 (by decide)
private theorem atomic_8065_agreement : RowAgreement 8064 8065 := RowAgreement.single (first := 8064) modelrow_agrees_8065
private def atomic_8066 : CountSegment 1 10 13120 8065 8066 3416 3416 := CountSegment.single (first := 8065) (initial := 3416) row_count_8066 (by decide)
private theorem atomic_8066_agreement : RowAgreement 8065 8066 := RowAgreement.single (first := 8065) modelrow_agrees_8066
private def atomic_8067 : CountSegment 1 10 13120 8066 8067 3416 3418 := CountSegment.single (first := 8066) (initial := 3416) row_count_8067 (by decide)
private theorem atomic_8067_agreement : RowAgreement 8066 8067 := RowAgreement.single (first := 8066) modelrow_agrees_8067
private def atomic_8068 : CountSegment 1 10 13120 8067 8068 3418 3418 := CountSegment.single (first := 8067) (initial := 3418) row_count_8068 (by decide)
private theorem atomic_8068_agreement : RowAgreement 8067 8068 := RowAgreement.single (first := 8067) modelrow_agrees_8068
private def atomic_8069 : CountSegment 1 10 13120 8068 8069 3418 3418 := CountSegment.single (first := 8068) (initial := 3418) row_count_8069 (by decide)
private theorem atomic_8069_agreement : RowAgreement 8068 8069 := RowAgreement.single (first := 8068) modelrow_agrees_8069
private def atomic_8070 : CountSegment 1 10 13120 8069 8070 3418 3419 := CountSegment.single (first := 8069) (initial := 3418) row_count_8070 (by decide)
private theorem atomic_8070_agreement : RowAgreement 8069 8070 := RowAgreement.single (first := 8069) modelrow_agrees_8070
private def atomic_8071 : CountSegment 1 10 13120 8070 8071 3419 3419 := CountSegment.single (first := 8070) (initial := 3419) row_count_8071 (by decide)
private theorem atomic_8071_agreement : RowAgreement 8070 8071 := RowAgreement.single (first := 8070) modelrow_agrees_8071
private def atomic_8072 : CountSegment 1 10 13120 8071 8072 3419 3419 := CountSegment.single (first := 8071) (initial := 3419) row_count_8072 (by decide)
private theorem atomic_8072_agreement : RowAgreement 8071 8072 := RowAgreement.single (first := 8071) modelrow_agrees_8072
private def atomic_8073 : CountSegment 1 10 13120 8072 8073 3419 3419 := CountSegment.single (first := 8072) (initial := 3419) row_count_8073 (by decide)
private theorem atomic_8073_agreement : RowAgreement 8072 8073 := RowAgreement.single (first := 8072) modelrow_agrees_8073
private def atomic_8074 : CountSegment 1 10 13120 8073 8074 3419 3419 := CountSegment.single (first := 8073) (initial := 3419) row_count_8074 (by decide)
private theorem atomic_8074_agreement : RowAgreement 8073 8074 := RowAgreement.single (first := 8073) modelrow_agrees_8074
private def atomic_8075 : CountSegment 1 10 13120 8074 8075 3419 3419 := CountSegment.single (first := 8074) (initial := 3419) row_count_8075 (by decide)
private theorem atomic_8075_agreement : RowAgreement 8074 8075 := RowAgreement.single (first := 8074) modelrow_agrees_8075
private def atomic_8076 : CountSegment 1 10 13120 8075 8076 3419 3419 := CountSegment.single (first := 8075) (initial := 3419) row_count_8076 (by decide)
private theorem atomic_8076_agreement : RowAgreement 8075 8076 := RowAgreement.single (first := 8075) modelrow_agrees_8076
private def atomic_8077 : CountSegment 1 10 13120 8076 8077 3419 3420 := CountSegment.single (first := 8076) (initial := 3419) row_count_8077 (by decide)
private theorem atomic_8077_agreement : RowAgreement 8076 8077 := RowAgreement.single (first := 8076) modelrow_agrees_8077
private def atomic_8078 : CountSegment 1 10 13120 8077 8078 3420 3421 := CountSegment.single (first := 8077) (initial := 3420) row_count_8078 (by decide)
private theorem atomic_8078_agreement : RowAgreement 8077 8078 := RowAgreement.single (first := 8077) modelrow_agrees_8078
private def atomic_8079 : CountSegment 1 10 13120 8078 8079 3421 3421 := CountSegment.single (first := 8078) (initial := 3421) row_count_8079 (by decide)
private theorem atomic_8079_agreement : RowAgreement 8078 8079 := RowAgreement.single (first := 8078) modelrow_agrees_8079
private def atomic_8080 : CountSegment 1 10 13120 8079 8080 3421 3421 := CountSegment.single (first := 8079) (initial := 3421) row_count_8080 (by decide)
private theorem atomic_8080_agreement : RowAgreement 8079 8080 := RowAgreement.single (first := 8079) modelrow_agrees_8080
private def atomic_8081 : CountSegment 1 10 13120 8080 8081 3421 3421 := CountSegment.single (first := 8080) (initial := 3421) row_count_8081 (by decide)
private theorem atomic_8081_agreement : RowAgreement 8080 8081 := RowAgreement.single (first := 8080) modelrow_agrees_8081
private def atomic_8082 : CountSegment 1 10 13120 8081 8082 3421 3421 := CountSegment.single (first := 8081) (initial := 3421) row_count_8082 (by decide)
private theorem atomic_8082_agreement : RowAgreement 8081 8082 := RowAgreement.single (first := 8081) modelrow_agrees_8082
private def atomic_8083 : CountSegment 1 10 13120 8082 8083 3421 3421 := CountSegment.single (first := 8082) (initial := 3421) row_count_8083 (by decide)
private theorem atomic_8083_agreement : RowAgreement 8082 8083 := RowAgreement.single (first := 8082) modelrow_agrees_8083
private def atomic_8084 : CountSegment 1 10 13120 8083 8084 3421 3421 := CountSegment.single (first := 8083) (initial := 3421) row_count_8084 (by decide)
private theorem atomic_8084_agreement : RowAgreement 8083 8084 := RowAgreement.single (first := 8083) modelrow_agrees_8084
private def atomic_8085 : CountSegment 1 10 13120 8084 8085 3421 3422 := CountSegment.single (first := 8084) (initial := 3421) row_count_8085 (by decide)
private theorem atomic_8085_agreement : RowAgreement 8084 8085 := RowAgreement.single (first := 8084) modelrow_agrees_8085
private def atomic_8086 : CountSegment 1 10 13120 8085 8086 3422 3422 := CountSegment.single (first := 8085) (initial := 3422) row_count_8086 (by decide)
private theorem atomic_8086_agreement : RowAgreement 8085 8086 := RowAgreement.single (first := 8085) modelrow_agrees_8086
private def atomic_8087 : CountSegment 1 10 13120 8086 8087 3422 3423 := CountSegment.single (first := 8086) (initial := 3422) row_count_8087 (by decide)
private theorem atomic_8087_agreement : RowAgreement 8086 8087 := RowAgreement.single (first := 8086) modelrow_agrees_8087
private def atomic_8088 : CountSegment 1 10 13120 8087 8088 3423 3423 := CountSegment.single (first := 8087) (initial := 3423) row_count_8088 (by decide)
private theorem atomic_8088_agreement : RowAgreement 8087 8088 := RowAgreement.single (first := 8087) modelrow_agrees_8088
private def atomic_8089 : CountSegment 1 10 13120 8088 8089 3423 3423 := CountSegment.single (first := 8088) (initial := 3423) row_count_8089 (by decide)
private theorem atomic_8089_agreement : RowAgreement 8088 8089 := RowAgreement.single (first := 8088) modelrow_agrees_8089
private def atomic_8090 : CountSegment 1 10 13120 8089 8090 3423 3423 := CountSegment.single (first := 8089) (initial := 3423) row_count_8090 (by decide)
private theorem atomic_8090_agreement : RowAgreement 8089 8090 := RowAgreement.single (first := 8089) modelrow_agrees_8090
private def atomic_8091 : CountSegment 1 10 13120 8090 8091 3423 3423 := CountSegment.single (first := 8090) (initial := 3423) row_count_8091 (by decide)
private theorem atomic_8091_agreement : RowAgreement 8090 8091 := RowAgreement.single (first := 8090) modelrow_agrees_8091
private def atomic_8092 : CountSegment 1 10 13120 8091 8092 3423 3424 := CountSegment.single (first := 8091) (initial := 3423) row_count_8092 (by decide)
private theorem atomic_8092_agreement : RowAgreement 8091 8092 := RowAgreement.single (first := 8091) modelrow_agrees_8092
private def atomic_8093 : CountSegment 1 10 13120 8092 8093 3424 3425 := CountSegment.single (first := 8092) (initial := 3424) row_count_8093 (by decide)
private theorem atomic_8093_agreement : RowAgreement 8092 8093 := RowAgreement.single (first := 8092) modelrow_agrees_8093
private def atomic_8094 : CountSegment 1 10 13120 8093 8094 3425 3425 := CountSegment.single (first := 8093) (initial := 3425) row_count_8094 (by decide)
private theorem atomic_8094_agreement : RowAgreement 8093 8094 := RowAgreement.single (first := 8093) modelrow_agrees_8094
private def atomic_8095 : CountSegment 1 10 13120 8094 8095 3425 3426 := CountSegment.single (first := 8094) (initial := 3425) row_count_8095 (by decide)
private theorem atomic_8095_agreement : RowAgreement 8094 8095 := RowAgreement.single (first := 8094) modelrow_agrees_8095
private def atomic_8096 : CountSegment 1 10 13120 8095 8096 3426 3426 := CountSegment.single (first := 8095) (initial := 3426) row_count_8096 (by decide)
private theorem atomic_8096_agreement : RowAgreement 8095 8096 := RowAgreement.single (first := 8095) modelrow_agrees_8096
private def atomic_8097 : CountSegment 1 10 13120 8096 8097 3426 3426 := CountSegment.single (first := 8096) (initial := 3426) row_count_8097 (by decide)
private theorem atomic_8097_agreement : RowAgreement 8096 8097 := RowAgreement.single (first := 8096) modelrow_agrees_8097
private def atomic_8098 : CountSegment 1 10 13120 8097 8098 3426 3427 := CountSegment.single (first := 8097) (initial := 3426) row_count_8098 (by decide)
private theorem atomic_8098_agreement : RowAgreement 8097 8098 := RowAgreement.single (first := 8097) modelrow_agrees_8098
private def atomic_8099 : CountSegment 1 10 13120 8098 8099 3427 3427 := CountSegment.single (first := 8098) (initial := 3427) row_count_8099 (by decide)
private theorem atomic_8099_agreement : RowAgreement 8098 8099 := RowAgreement.single (first := 8098) modelrow_agrees_8099
private def atomic_8100 : CountSegment 1 10 13120 8099 8100 3427 3427 := CountSegment.single (first := 8099) (initial := 3427) row_count_8100 (by decide)
private theorem atomic_8100_agreement : RowAgreement 8099 8100 := RowAgreement.single (first := 8099) modelrow_agrees_8100
private def atomic_8101 : CountSegment 1 10 13120 8100 8101 3427 3427 := CountSegment.single (first := 8100) (initial := 3427) row_count_8101 (by decide)
private theorem atomic_8101_agreement : RowAgreement 8100 8101 := RowAgreement.single (first := 8100) modelrow_agrees_8101
private def atomic_8102 : CountSegment 1 10 13120 8101 8102 3427 3428 := CountSegment.single (first := 8101) (initial := 3427) row_count_8102 (by decide)
private theorem atomic_8102_agreement : RowAgreement 8101 8102 := RowAgreement.single (first := 8101) modelrow_agrees_8102
private def atomic_8103 : CountSegment 1 10 13120 8102 8103 3428 3429 := CountSegment.single (first := 8102) (initial := 3428) row_count_8103 (by decide)
private theorem atomic_8103_agreement : RowAgreement 8102 8103 := RowAgreement.single (first := 8102) modelrow_agrees_8103
private def atomic_8104 : CountSegment 1 10 13120 8103 8104 3429 3430 := CountSegment.single (first := 8103) (initial := 3429) row_count_8104 (by decide)
private theorem atomic_8104_agreement : RowAgreement 8103 8104 := RowAgreement.single (first := 8103) modelrow_agrees_8104
private def atomic_8105 : CountSegment 1 10 13120 8104 8105 3430 3430 := CountSegment.single (first := 8104) (initial := 3430) row_count_8105 (by decide)
private theorem atomic_8105_agreement : RowAgreement 8104 8105 := RowAgreement.single (first := 8104) modelrow_agrees_8105
private def atomic_8106 : CountSegment 1 10 13120 8105 8106 3430 3430 := CountSegment.single (first := 8105) (initial := 3430) row_count_8106 (by decide)
private theorem atomic_8106_agreement : RowAgreement 8105 8106 := RowAgreement.single (first := 8105) modelrow_agrees_8106
private def atomic_8107 : CountSegment 1 10 13120 8106 8107 3430 3430 := CountSegment.single (first := 8106) (initial := 3430) row_count_8107 (by decide)
private theorem atomic_8107_agreement : RowAgreement 8106 8107 := RowAgreement.single (first := 8106) modelrow_agrees_8107
private def atomic_8108 : CountSegment 1 10 13120 8107 8108 3430 3430 := CountSegment.single (first := 8107) (initial := 3430) row_count_8108 (by decide)
private theorem atomic_8108_agreement : RowAgreement 8107 8108 := RowAgreement.single (first := 8107) modelrow_agrees_8108
private def atomic_8109 : CountSegment 1 10 13120 8108 8109 3430 3430 := CountSegment.single (first := 8108) (initial := 3430) row_count_8109 (by decide)
private theorem atomic_8109_agreement : RowAgreement 8108 8109 := RowAgreement.single (first := 8108) modelrow_agrees_8109
private def atomic_8110 : CountSegment 1 10 13120 8109 8110 3430 3431 := CountSegment.single (first := 8109) (initial := 3430) row_count_8110 (by decide)
private theorem atomic_8110_agreement : RowAgreement 8109 8110 := RowAgreement.single (first := 8109) modelrow_agrees_8110
private def atomic_8111 : CountSegment 1 10 13120 8110 8111 3431 3431 := CountSegment.single (first := 8110) (initial := 3431) row_count_8111 (by decide)
private theorem atomic_8111_agreement : RowAgreement 8110 8111 := RowAgreement.single (first := 8110) modelrow_agrees_8111
private def atomic_8112 : CountSegment 1 10 13120 8111 8112 3431 3433 := CountSegment.single (first := 8111) (initial := 3431) row_count_8112 (by decide)
private theorem atomic_8112_agreement : RowAgreement 8111 8112 := RowAgreement.single (first := 8111) modelrow_agrees_8112
private def atomic_8113 : CountSegment 1 10 13120 8112 8113 3433 3434 := CountSegment.single (first := 8112) (initial := 3433) row_count_8113 (by decide)
private theorem atomic_8113_agreement : RowAgreement 8112 8113 := RowAgreement.single (first := 8112) modelrow_agrees_8113
private def atomic_8114 : CountSegment 1 10 13120 8113 8114 3434 3434 := CountSegment.single (first := 8113) (initial := 3434) row_count_8114 (by decide)
private theorem atomic_8114_agreement : RowAgreement 8113 8114 := RowAgreement.single (first := 8113) modelrow_agrees_8114
private def atomic_8115 : CountSegment 1 10 13120 8114 8115 3434 3436 := CountSegment.single (first := 8114) (initial := 3434) row_count_8115 (by decide)
private theorem atomic_8115_agreement : RowAgreement 8114 8115 := RowAgreement.single (first := 8114) modelrow_agrees_8115
private def atomic_8116 : CountSegment 1 10 13120 8115 8116 3436 3436 := CountSegment.single (first := 8115) (initial := 3436) row_count_8116 (by decide)
private theorem atomic_8116_agreement : RowAgreement 8115 8116 := RowAgreement.single (first := 8115) modelrow_agrees_8116
private def atomic_8117 : CountSegment 1 10 13120 8116 8117 3436 3436 := CountSegment.single (first := 8116) (initial := 3436) row_count_8117 (by decide)
private theorem atomic_8117_agreement : RowAgreement 8116 8117 := RowAgreement.single (first := 8116) modelrow_agrees_8117
private def atomic_8118 : CountSegment 1 10 13120 8117 8118 3436 3436 := CountSegment.single (first := 8117) (initial := 3436) row_count_8118 (by decide)
private theorem atomic_8118_agreement : RowAgreement 8117 8118 := RowAgreement.single (first := 8117) modelrow_agrees_8118
private def atomic_8119 : CountSegment 1 10 13120 8118 8119 3436 3436 := CountSegment.single (first := 8118) (initial := 3436) row_count_8119 (by decide)
private theorem atomic_8119_agreement : RowAgreement 8118 8119 := RowAgreement.single (first := 8118) modelrow_agrees_8119
private def atomic_8120 : CountSegment 1 10 13120 8119 8120 3436 3437 := CountSegment.single (first := 8119) (initial := 3436) row_count_8120 (by decide)
private theorem atomic_8120_agreement : RowAgreement 8119 8120 := RowAgreement.single (first := 8119) modelrow_agrees_8120
private def atomic_8121 : CountSegment 1 10 13120 8120 8121 3437 3437 := CountSegment.single (first := 8120) (initial := 3437) row_count_8121 (by decide)
private theorem atomic_8121_agreement : RowAgreement 8120 8121 := RowAgreement.single (first := 8120) modelrow_agrees_8121
private def atomic_8122 : CountSegment 1 10 13120 8121 8122 3437 3438 := CountSegment.single (first := 8121) (initial := 3437) row_count_8122 (by decide)
private theorem atomic_8122_agreement : RowAgreement 8121 8122 := RowAgreement.single (first := 8121) modelrow_agrees_8122
private def atomic_8123 : CountSegment 1 10 13120 8122 8123 3438 3438 := CountSegment.single (first := 8122) (initial := 3438) row_count_8123 (by decide)
private theorem atomic_8123_agreement : RowAgreement 8122 8123 := RowAgreement.single (first := 8122) modelrow_agrees_8123
private def atomic_8124 : CountSegment 1 10 13120 8123 8124 3438 3439 := CountSegment.single (first := 8123) (initial := 3438) row_count_8124 (by decide)
private theorem atomic_8124_agreement : RowAgreement 8123 8124 := RowAgreement.single (first := 8123) modelrow_agrees_8124
private def atomic_8125 : CountSegment 1 10 13120 8124 8125 3439 3440 := CountSegment.single (first := 8124) (initial := 3439) row_count_8125 (by decide)
private theorem atomic_8125_agreement : RowAgreement 8124 8125 := RowAgreement.single (first := 8124) modelrow_agrees_8125
private def atomic_8126 : CountSegment 1 10 13120 8125 8126 3440 3440 := CountSegment.single (first := 8125) (initial := 3440) row_count_8126 (by decide)
private theorem atomic_8126_agreement : RowAgreement 8125 8126 := RowAgreement.single (first := 8125) modelrow_agrees_8126
private def atomic_8127 : CountSegment 1 10 13120 8126 8127 3440 3442 := CountSegment.single (first := 8126) (initial := 3440) row_count_8127 (by decide)
private theorem atomic_8127_agreement : RowAgreement 8126 8127 := RowAgreement.single (first := 8126) modelrow_agrees_8127
private def atomic_8128 : CountSegment 1 10 13120 8127 8128 3442 3442 := CountSegment.single (first := 8127) (initial := 3442) row_count_8128 (by decide)
private theorem atomic_8128_agreement : RowAgreement 8127 8128 := RowAgreement.single (first := 8127) modelrow_agrees_8128
private def atomic_8129 : CountSegment 1 10 13120 8128 8129 3442 3443 := CountSegment.single (first := 8128) (initial := 3442) row_count_8129 (by decide)
private theorem atomic_8129_agreement : RowAgreement 8128 8129 := RowAgreement.single (first := 8128) modelrow_agrees_8129
private def atomic_8130 : CountSegment 1 10 13120 8129 8130 3443 3444 := CountSegment.single (first := 8129) (initial := 3443) row_count_8130 (by decide)
private theorem atomic_8130_agreement : RowAgreement 8129 8130 := RowAgreement.single (first := 8129) modelrow_agrees_8130
private def atomic_8131 : CountSegment 1 10 13120 8130 8131 3444 3445 := CountSegment.single (first := 8130) (initial := 3444) row_count_8131 (by decide)
private theorem atomic_8131_agreement : RowAgreement 8130 8131 := RowAgreement.single (first := 8130) modelrow_agrees_8131
private def atomic_8132 : CountSegment 1 10 13120 8131 8132 3445 3445 := CountSegment.single (first := 8131) (initial := 3445) row_count_8132 (by decide)
private theorem atomic_8132_agreement : RowAgreement 8131 8132 := RowAgreement.single (first := 8131) modelrow_agrees_8132
private def atomic_8133 : CountSegment 1 10 13120 8132 8133 3445 3445 := CountSegment.single (first := 8132) (initial := 3445) row_count_8133 (by decide)
private theorem atomic_8133_agreement : RowAgreement 8132 8133 := RowAgreement.single (first := 8132) modelrow_agrees_8133
private def atomic_8134 : CountSegment 1 10 13120 8133 8134 3445 3446 := CountSegment.single (first := 8133) (initial := 3445) row_count_8134 (by decide)
private theorem atomic_8134_agreement : RowAgreement 8133 8134 := RowAgreement.single (first := 8133) modelrow_agrees_8134
private def atomic_8135 : CountSegment 1 10 13120 8134 8135 3446 3446 := CountSegment.single (first := 8134) (initial := 3446) row_count_8135 (by decide)
private theorem atomic_8135_agreement : RowAgreement 8134 8135 := RowAgreement.single (first := 8134) modelrow_agrees_8135
private def atomic_8136 : CountSegment 1 10 13120 8135 8136 3446 3446 := CountSegment.single (first := 8135) (initial := 3446) row_count_8136 (by decide)
private theorem atomic_8136_agreement : RowAgreement 8135 8136 := RowAgreement.single (first := 8135) modelrow_agrees_8136
private def atomic_8137 : CountSegment 1 10 13120 8136 8137 3446 3446 := CountSegment.single (first := 8136) (initial := 3446) row_count_8137 (by decide)
private theorem atomic_8137_agreement : RowAgreement 8136 8137 := RowAgreement.single (first := 8136) modelrow_agrees_8137
private def atomic_8138 : CountSegment 1 10 13120 8137 8138 3446 3446 := CountSegment.single (first := 8137) (initial := 3446) row_count_8138 (by decide)
private theorem atomic_8138_agreement : RowAgreement 8137 8138 := RowAgreement.single (first := 8137) modelrow_agrees_8138
private def atomic_8139 : CountSegment 1 10 13120 8138 8139 3446 3447 := CountSegment.single (first := 8138) (initial := 3446) row_count_8139 (by decide)
private theorem atomic_8139_agreement : RowAgreement 8138 8139 := RowAgreement.single (first := 8138) modelrow_agrees_8139
private def atomic_8140 : CountSegment 1 10 13120 8139 8140 3447 3447 := CountSegment.single (first := 8139) (initial := 3447) row_count_8140 (by decide)
private theorem atomic_8140_agreement : RowAgreement 8139 8140 := RowAgreement.single (first := 8139) modelrow_agrees_8140
private def atomic_8141 : CountSegment 1 10 13120 8140 8141 3447 3448 := CountSegment.single (first := 8140) (initial := 3447) row_count_8141 (by decide)
private theorem atomic_8141_agreement : RowAgreement 8140 8141 := RowAgreement.single (first := 8140) modelrow_agrees_8141
private def atomic_8142 : CountSegment 1 10 13120 8141 8142 3448 3449 := CountSegment.single (first := 8141) (initial := 3448) row_count_8142 (by decide)
private theorem atomic_8142_agreement : RowAgreement 8141 8142 := RowAgreement.single (first := 8141) modelrow_agrees_8142
private def atomic_8143 : CountSegment 1 10 13120 8142 8143 3449 3449 := CountSegment.single (first := 8142) (initial := 3449) row_count_8143 (by decide)
private theorem atomic_8143_agreement : RowAgreement 8142 8143 := RowAgreement.single (first := 8142) modelrow_agrees_8143
private def atomic_8144 : CountSegment 1 10 13120 8143 8144 3449 3449 := CountSegment.single (first := 8143) (initial := 3449) row_count_8144 (by decide)
private theorem atomic_8144_agreement : RowAgreement 8143 8144 := RowAgreement.single (first := 8143) modelrow_agrees_8144
private def atomic_8145 : CountSegment 1 10 13120 8144 8145 3449 3449 := CountSegment.single (first := 8144) (initial := 3449) row_count_8145 (by decide)
private theorem atomic_8145_agreement : RowAgreement 8144 8145 := RowAgreement.single (first := 8144) modelrow_agrees_8145
private def atomic_8146 : CountSegment 1 10 13120 8145 8146 3449 3449 := CountSegment.single (first := 8145) (initial := 3449) row_count_8146 (by decide)
private theorem atomic_8146_agreement : RowAgreement 8145 8146 := RowAgreement.single (first := 8145) modelrow_agrees_8146
private def atomic_8147 : CountSegment 1 10 13120 8146 8147 3449 3449 := CountSegment.single (first := 8146) (initial := 3449) row_count_8147 (by decide)
private theorem atomic_8147_agreement : RowAgreement 8146 8147 := RowAgreement.single (first := 8146) modelrow_agrees_8147
private def atomic_8148 : CountSegment 1 10 13120 8147 8148 3449 3450 := CountSegment.single (first := 8147) (initial := 3449) row_count_8148 (by decide)
private theorem atomic_8148_agreement : RowAgreement 8147 8148 := RowAgreement.single (first := 8147) modelrow_agrees_8148
private def atomic_8149 : CountSegment 1 10 13120 8148 8149 3450 3450 := CountSegment.single (first := 8148) (initial := 3450) row_count_8149 (by decide)
private theorem atomic_8149_agreement : RowAgreement 8148 8149 := RowAgreement.single (first := 8148) modelrow_agrees_8149
private def atomic_8150 : CountSegment 1 10 13120 8149 8150 3450 3451 := CountSegment.single (first := 8149) (initial := 3450) row_count_8150 (by decide)
private theorem atomic_8150_agreement : RowAgreement 8149 8150 := RowAgreement.single (first := 8149) modelrow_agrees_8150
private def atomic_8151 : CountSegment 1 10 13120 8150 8151 3451 3453 := CountSegment.single (first := 8150) (initial := 3451) row_count_8151 (by decide)
private theorem atomic_8151_agreement : RowAgreement 8150 8151 := RowAgreement.single (first := 8150) modelrow_agrees_8151
private def atomic_8152 : CountSegment 1 10 13120 8151 8152 3453 3454 := CountSegment.single (first := 8151) (initial := 3453) row_count_8152 (by decide)
private theorem atomic_8152_agreement : RowAgreement 8151 8152 := RowAgreement.single (first := 8151) modelrow_agrees_8152
private def atomic_8153 : CountSegment 1 10 13120 8152 8153 3454 3454 := CountSegment.single (first := 8152) (initial := 3454) row_count_8153 (by decide)
private theorem atomic_8153_agreement : RowAgreement 8152 8153 := RowAgreement.single (first := 8152) modelrow_agrees_8153
private def atomic_8154 : CountSegment 1 10 13120 8153 8154 3454 3454 := CountSegment.single (first := 8153) (initial := 3454) row_count_8154 (by decide)
private theorem atomic_8154_agreement : RowAgreement 8153 8154 := RowAgreement.single (first := 8153) modelrow_agrees_8154
private def atomic_8155 : CountSegment 1 10 13120 8154 8155 3454 3454 := CountSegment.single (first := 8154) (initial := 3454) row_count_8155 (by decide)
private theorem atomic_8155_agreement : RowAgreement 8154 8155 := RowAgreement.single (first := 8154) modelrow_agrees_8155
private def atomic_8156 : CountSegment 1 10 13120 8155 8156 3454 3454 := CountSegment.single (first := 8155) (initial := 3454) row_count_8156 (by decide)
private theorem atomic_8156_agreement : RowAgreement 8155 8156 := RowAgreement.single (first := 8155) modelrow_agrees_8156
private def atomic_8157 : CountSegment 1 10 13120 8156 8157 3454 3455 := CountSegment.single (first := 8156) (initial := 3454) row_count_8157 (by decide)
private theorem atomic_8157_agreement : RowAgreement 8156 8157 := RowAgreement.single (first := 8156) modelrow_agrees_8157
private def atomic_8158 : CountSegment 1 10 13120 8157 8158 3455 3456 := CountSegment.single (first := 8157) (initial := 3455) row_count_8158 (by decide)
private theorem atomic_8158_agreement : RowAgreement 8157 8158 := RowAgreement.single (first := 8157) modelrow_agrees_8158
private def atomic_8159 : CountSegment 1 10 13120 8158 8159 3456 3457 := CountSegment.single (first := 8158) (initial := 3456) row_count_8159 (by decide)
private theorem atomic_8159_agreement : RowAgreement 8158 8159 := RowAgreement.single (first := 8158) modelrow_agrees_8159
private def atomic_8160 : CountSegment 1 10 13120 8159 8160 3457 3457 := CountSegment.single (first := 8159) (initial := 3457) row_count_8160 (by decide)
private theorem atomic_8160_agreement : RowAgreement 8159 8160 := RowAgreement.single (first := 8159) modelrow_agrees_8160
private def atomic_8161 : CountSegment 1 10 13120 8160 8161 3457 3458 := CountSegment.single (first := 8160) (initial := 3457) row_count_8161 (by decide)
private theorem atomic_8161_agreement : RowAgreement 8160 8161 := RowAgreement.single (first := 8160) modelrow_agrees_8161
private def atomic_8162 : CountSegment 1 10 13120 8161 8162 3458 3458 := CountSegment.single (first := 8161) (initial := 3458) row_count_8162 (by decide)
private theorem atomic_8162_agreement : RowAgreement 8161 8162 := RowAgreement.single (first := 8161) modelrow_agrees_8162
private def atomic_8163 : CountSegment 1 10 13120 8162 8163 3458 3458 := CountSegment.single (first := 8162) (initial := 3458) row_count_8163 (by decide)
private theorem atomic_8163_agreement : RowAgreement 8162 8163 := RowAgreement.single (first := 8162) modelrow_agrees_8163
private def atomic_8164 : CountSegment 1 10 13120 8163 8164 3458 3458 := CountSegment.single (first := 8163) (initial := 3458) row_count_8164 (by decide)
private theorem atomic_8164_agreement : RowAgreement 8163 8164 := RowAgreement.single (first := 8163) modelrow_agrees_8164
private def atomic_8165 : CountSegment 1 10 13120 8164 8165 3458 3459 := CountSegment.single (first := 8164) (initial := 3458) row_count_8165 (by decide)
private theorem atomic_8165_agreement : RowAgreement 8164 8165 := RowAgreement.single (first := 8164) modelrow_agrees_8165
private def atomic_8166 : CountSegment 1 10 13120 8165 8166 3459 3459 := CountSegment.single (first := 8165) (initial := 3459) row_count_8166 (by decide)
private theorem atomic_8166_agreement : RowAgreement 8165 8166 := RowAgreement.single (first := 8165) modelrow_agrees_8166
private def atomic_8167 : CountSegment 1 10 13120 8166 8167 3459 3459 := CountSegment.single (first := 8166) (initial := 3459) row_count_8167 (by decide)
private theorem atomic_8167_agreement : RowAgreement 8166 8167 := RowAgreement.single (first := 8166) modelrow_agrees_8167
private def atomic_8168 : CountSegment 1 10 13120 8167 8168 3459 3459 := CountSegment.single (first := 8167) (initial := 3459) row_count_8168 (by decide)
private theorem atomic_8168_agreement : RowAgreement 8167 8168 := RowAgreement.single (first := 8167) modelrow_agrees_8168
private def atomic_8169 : CountSegment 1 10 13120 8168 8169 3459 3459 := CountSegment.single (first := 8168) (initial := 3459) row_count_8169 (by decide)
private theorem atomic_8169_agreement : RowAgreement 8168 8169 := RowAgreement.single (first := 8168) modelrow_agrees_8169
private def atomic_8170 : CountSegment 1 10 13120 8169 8170 3459 3459 := CountSegment.single (first := 8169) (initial := 3459) row_count_8170 (by decide)
private theorem atomic_8170_agreement : RowAgreement 8169 8170 := RowAgreement.single (first := 8169) modelrow_agrees_8170
private def atomic_8171 : CountSegment 1 10 13120 8170 8171 3459 3459 := CountSegment.single (first := 8170) (initial := 3459) row_count_8171 (by decide)
private theorem atomic_8171_agreement : RowAgreement 8170 8171 := RowAgreement.single (first := 8170) modelrow_agrees_8171
private def atomic_8172 : CountSegment 1 10 13120 8171 8172 3459 3459 := CountSegment.single (first := 8171) (initial := 3459) row_count_8172 (by decide)
private theorem atomic_8172_agreement : RowAgreement 8171 8172 := RowAgreement.single (first := 8171) modelrow_agrees_8172
private def atomic_8173 : CountSegment 1 10 13120 8172 8173 3459 3459 := CountSegment.single (first := 8172) (initial := 3459) row_count_8173 (by decide)
private theorem atomic_8173_agreement : RowAgreement 8172 8173 := RowAgreement.single (first := 8172) modelrow_agrees_8173
private def atomic_8174 : CountSegment 1 10 13120 8173 8174 3459 3459 := CountSegment.single (first := 8173) (initial := 3459) row_count_8174 (by decide)
private theorem atomic_8174_agreement : RowAgreement 8173 8174 := RowAgreement.single (first := 8173) modelrow_agrees_8174
private def atomic_8175 : CountSegment 1 10 13120 8174 8175 3459 3459 := CountSegment.single (first := 8174) (initial := 3459) row_count_8175 (by decide)
private theorem atomic_8175_agreement : RowAgreement 8174 8175 := RowAgreement.single (first := 8174) modelrow_agrees_8175
private def atomic_8176 : CountSegment 1 10 13120 8175 8176 3459 3460 := CountSegment.single (first := 8175) (initial := 3459) row_count_8176 (by decide)
private theorem atomic_8176_agreement : RowAgreement 8175 8176 := RowAgreement.single (first := 8175) modelrow_agrees_8176
private def atomic_8177 : CountSegment 1 10 13120 8176 8177 3460 3460 := CountSegment.single (first := 8176) (initial := 3460) row_count_8177 (by decide)
private theorem atomic_8177_agreement : RowAgreement 8176 8177 := RowAgreement.single (first := 8176) modelrow_agrees_8177
private def atomic_8178 : CountSegment 1 10 13120 8177 8178 3460 3460 := CountSegment.single (first := 8177) (initial := 3460) row_count_8178 (by decide)
private theorem atomic_8178_agreement : RowAgreement 8177 8178 := RowAgreement.single (first := 8177) modelrow_agrees_8178
private def atomic_8179 : CountSegment 1 10 13120 8178 8179 3460 3460 := CountSegment.single (first := 8178) (initial := 3460) row_count_8179 (by decide)
private theorem atomic_8179_agreement : RowAgreement 8178 8179 := RowAgreement.single (first := 8178) modelrow_agrees_8179
private def atomic_8180 : CountSegment 1 10 13120 8179 8180 3460 3460 := CountSegment.single (first := 8179) (initial := 3460) row_count_8180 (by decide)
private theorem atomic_8180_agreement : RowAgreement 8179 8180 := RowAgreement.single (first := 8179) modelrow_agrees_8180
private def atomic_8181 : CountSegment 1 10 13120 8180 8181 3460 3460 := CountSegment.single (first := 8180) (initial := 3460) row_count_8181 (by decide)
private theorem atomic_8181_agreement : RowAgreement 8180 8181 := RowAgreement.single (first := 8180) modelrow_agrees_8181
private def atomic_8182 : CountSegment 1 10 13120 8181 8182 3460 3461 := CountSegment.single (first := 8181) (initial := 3460) row_count_8182 (by decide)
private theorem atomic_8182_agreement : RowAgreement 8181 8182 := RowAgreement.single (first := 8181) modelrow_agrees_8182
private def atomic_8183 : CountSegment 1 10 13120 8182 8183 3461 3462 := CountSegment.single (first := 8182) (initial := 3461) row_count_8183 (by decide)
private theorem atomic_8183_agreement : RowAgreement 8182 8183 := RowAgreement.single (first := 8182) modelrow_agrees_8183
private def atomic_8184 : CountSegment 1 10 13120 8183 8184 3462 3463 := CountSegment.single (first := 8183) (initial := 3462) row_count_8184 (by decide)
private theorem atomic_8184_agreement : RowAgreement 8183 8184 := RowAgreement.single (first := 8183) modelrow_agrees_8184
private def atomic_8185 : CountSegment 1 10 13120 8184 8185 3463 3463 := CountSegment.single (first := 8184) (initial := 3463) row_count_8185 (by decide)
private theorem atomic_8185_agreement : RowAgreement 8184 8185 := RowAgreement.single (first := 8184) modelrow_agrees_8185
private def atomic_8186 : CountSegment 1 10 13120 8185 8186 3463 3464 := CountSegment.single (first := 8185) (initial := 3463) row_count_8186 (by decide)
private theorem atomic_8186_agreement : RowAgreement 8185 8186 := RowAgreement.single (first := 8185) modelrow_agrees_8186
private def atomic_8187 : CountSegment 1 10 13120 8186 8187 3464 3465 := CountSegment.single (first := 8186) (initial := 3464) row_count_8187 (by decide)
private theorem atomic_8187_agreement : RowAgreement 8186 8187 := RowAgreement.single (first := 8186) modelrow_agrees_8187
private def atomic_8188 : CountSegment 1 10 13120 8187 8188 3465 3465 := CountSegment.single (first := 8187) (initial := 3465) row_count_8188 (by decide)
private theorem atomic_8188_agreement : RowAgreement 8187 8188 := RowAgreement.single (first := 8187) modelrow_agrees_8188
private def atomic_8189 : CountSegment 1 10 13120 8188 8189 3465 3465 := CountSegment.single (first := 8188) (initial := 3465) row_count_8189 (by decide)
private theorem atomic_8189_agreement : RowAgreement 8188 8189 := RowAgreement.single (first := 8188) modelrow_agrees_8189
private def atomic_8190 : CountSegment 1 10 13120 8189 8190 3465 3467 := CountSegment.single (first := 8189) (initial := 3465) row_count_8190 (by decide)
private theorem atomic_8190_agreement : RowAgreement 8189 8190 := RowAgreement.single (first := 8189) modelrow_agrees_8190
private def atomic_8191 : CountSegment 1 10 13120 8190 8191 3467 3467 := CountSegment.single (first := 8190) (initial := 3467) row_count_8191 (by decide)
private theorem atomic_8191_agreement : RowAgreement 8190 8191 := RowAgreement.single (first := 8190) modelrow_agrees_8191
private def atomic_8192 : CountSegment 1 10 13120 8191 8192 3467 3467 := CountSegment.single (first := 8191) (initial := 3467) row_count_8192 (by decide)
private theorem atomic_8192_agreement : RowAgreement 8191 8192 := RowAgreement.single (first := 8191) modelrow_agrees_8192
private def within_063_000 : CountSegment 1 10 13120 8064 8066 3415 3416 := CountSegment.join atomic_8065 atomic_8066
private theorem within_063_000_agreement : RowAgreement 8064 8066 := RowAgreement.join atomic_8065_agreement atomic_8066_agreement
private def within_063_001 : CountSegment 1 10 13120 8066 8068 3416 3418 := CountSegment.join atomic_8067 atomic_8068
private theorem within_063_001_agreement : RowAgreement 8066 8068 := RowAgreement.join atomic_8067_agreement atomic_8068_agreement
private def within_063_002 : CountSegment 1 10 13120 8064 8068 3415 3418 := CountSegment.join within_063_000 within_063_001
private theorem within_063_002_agreement : RowAgreement 8064 8068 := RowAgreement.join within_063_000_agreement within_063_001_agreement
private def within_063_003 : CountSegment 1 10 13120 8068 8070 3418 3419 := CountSegment.join atomic_8069 atomic_8070
private theorem within_063_003_agreement : RowAgreement 8068 8070 := RowAgreement.join atomic_8069_agreement atomic_8070_agreement
private def within_063_004 : CountSegment 1 10 13120 8070 8072 3419 3419 := CountSegment.join atomic_8071 atomic_8072
private theorem within_063_004_agreement : RowAgreement 8070 8072 := RowAgreement.join atomic_8071_agreement atomic_8072_agreement
private def within_063_005 : CountSegment 1 10 13120 8068 8072 3418 3419 := CountSegment.join within_063_003 within_063_004
private theorem within_063_005_agreement : RowAgreement 8068 8072 := RowAgreement.join within_063_003_agreement within_063_004_agreement
private def within_063_006 : CountSegment 1 10 13120 8064 8072 3415 3419 := CountSegment.join within_063_002 within_063_005
private theorem within_063_006_agreement : RowAgreement 8064 8072 := RowAgreement.join within_063_002_agreement within_063_005_agreement
private def within_063_007 : CountSegment 1 10 13120 8072 8074 3419 3419 := CountSegment.join atomic_8073 atomic_8074
private theorem within_063_007_agreement : RowAgreement 8072 8074 := RowAgreement.join atomic_8073_agreement atomic_8074_agreement
private def within_063_008 : CountSegment 1 10 13120 8074 8076 3419 3419 := CountSegment.join atomic_8075 atomic_8076
private theorem within_063_008_agreement : RowAgreement 8074 8076 := RowAgreement.join atomic_8075_agreement atomic_8076_agreement
private def within_063_009 : CountSegment 1 10 13120 8072 8076 3419 3419 := CountSegment.join within_063_007 within_063_008
private theorem within_063_009_agreement : RowAgreement 8072 8076 := RowAgreement.join within_063_007_agreement within_063_008_agreement
private def within_063_010 : CountSegment 1 10 13120 8076 8078 3419 3421 := CountSegment.join atomic_8077 atomic_8078
private theorem within_063_010_agreement : RowAgreement 8076 8078 := RowAgreement.join atomic_8077_agreement atomic_8078_agreement
private def within_063_011 : CountSegment 1 10 13120 8078 8080 3421 3421 := CountSegment.join atomic_8079 atomic_8080
private theorem within_063_011_agreement : RowAgreement 8078 8080 := RowAgreement.join atomic_8079_agreement atomic_8080_agreement
private def within_063_012 : CountSegment 1 10 13120 8076 8080 3419 3421 := CountSegment.join within_063_010 within_063_011
private theorem within_063_012_agreement : RowAgreement 8076 8080 := RowAgreement.join within_063_010_agreement within_063_011_agreement
private def within_063_013 : CountSegment 1 10 13120 8072 8080 3419 3421 := CountSegment.join within_063_009 within_063_012
private theorem within_063_013_agreement : RowAgreement 8072 8080 := RowAgreement.join within_063_009_agreement within_063_012_agreement
private def within_063_014 : CountSegment 1 10 13120 8064 8080 3415 3421 := CountSegment.join within_063_006 within_063_013
private theorem within_063_014_agreement : RowAgreement 8064 8080 := RowAgreement.join within_063_006_agreement within_063_013_agreement
private def within_063_015 : CountSegment 1 10 13120 8080 8082 3421 3421 := CountSegment.join atomic_8081 atomic_8082
private theorem within_063_015_agreement : RowAgreement 8080 8082 := RowAgreement.join atomic_8081_agreement atomic_8082_agreement
private def within_063_016 : CountSegment 1 10 13120 8082 8084 3421 3421 := CountSegment.join atomic_8083 atomic_8084
private theorem within_063_016_agreement : RowAgreement 8082 8084 := RowAgreement.join atomic_8083_agreement atomic_8084_agreement
private def within_063_017 : CountSegment 1 10 13120 8080 8084 3421 3421 := CountSegment.join within_063_015 within_063_016
private theorem within_063_017_agreement : RowAgreement 8080 8084 := RowAgreement.join within_063_015_agreement within_063_016_agreement
private def within_063_018 : CountSegment 1 10 13120 8084 8086 3421 3422 := CountSegment.join atomic_8085 atomic_8086
private theorem within_063_018_agreement : RowAgreement 8084 8086 := RowAgreement.join atomic_8085_agreement atomic_8086_agreement
private def within_063_019 : CountSegment 1 10 13120 8086 8088 3422 3423 := CountSegment.join atomic_8087 atomic_8088
private theorem within_063_019_agreement : RowAgreement 8086 8088 := RowAgreement.join atomic_8087_agreement atomic_8088_agreement
private def within_063_020 : CountSegment 1 10 13120 8084 8088 3421 3423 := CountSegment.join within_063_018 within_063_019
private theorem within_063_020_agreement : RowAgreement 8084 8088 := RowAgreement.join within_063_018_agreement within_063_019_agreement
private def within_063_021 : CountSegment 1 10 13120 8080 8088 3421 3423 := CountSegment.join within_063_017 within_063_020
private theorem within_063_021_agreement : RowAgreement 8080 8088 := RowAgreement.join within_063_017_agreement within_063_020_agreement
private def within_063_022 : CountSegment 1 10 13120 8088 8090 3423 3423 := CountSegment.join atomic_8089 atomic_8090
private theorem within_063_022_agreement : RowAgreement 8088 8090 := RowAgreement.join atomic_8089_agreement atomic_8090_agreement
private def within_063_023 : CountSegment 1 10 13120 8090 8092 3423 3424 := CountSegment.join atomic_8091 atomic_8092
private theorem within_063_023_agreement : RowAgreement 8090 8092 := RowAgreement.join atomic_8091_agreement atomic_8092_agreement
private def within_063_024 : CountSegment 1 10 13120 8088 8092 3423 3424 := CountSegment.join within_063_022 within_063_023
private theorem within_063_024_agreement : RowAgreement 8088 8092 := RowAgreement.join within_063_022_agreement within_063_023_agreement
private def within_063_025 : CountSegment 1 10 13120 8092 8094 3424 3425 := CountSegment.join atomic_8093 atomic_8094
private theorem within_063_025_agreement : RowAgreement 8092 8094 := RowAgreement.join atomic_8093_agreement atomic_8094_agreement
private def within_063_026 : CountSegment 1 10 13120 8094 8096 3425 3426 := CountSegment.join atomic_8095 atomic_8096
private theorem within_063_026_agreement : RowAgreement 8094 8096 := RowAgreement.join atomic_8095_agreement atomic_8096_agreement
private def within_063_027 : CountSegment 1 10 13120 8092 8096 3424 3426 := CountSegment.join within_063_025 within_063_026
private theorem within_063_027_agreement : RowAgreement 8092 8096 := RowAgreement.join within_063_025_agreement within_063_026_agreement
private def within_063_028 : CountSegment 1 10 13120 8088 8096 3423 3426 := CountSegment.join within_063_024 within_063_027
private theorem within_063_028_agreement : RowAgreement 8088 8096 := RowAgreement.join within_063_024_agreement within_063_027_agreement
private def within_063_029 : CountSegment 1 10 13120 8080 8096 3421 3426 := CountSegment.join within_063_021 within_063_028
private theorem within_063_029_agreement : RowAgreement 8080 8096 := RowAgreement.join within_063_021_agreement within_063_028_agreement
private def within_063_030 : CountSegment 1 10 13120 8064 8096 3415 3426 := CountSegment.join within_063_014 within_063_029
private theorem within_063_030_agreement : RowAgreement 8064 8096 := RowAgreement.join within_063_014_agreement within_063_029_agreement
private def within_063_031 : CountSegment 1 10 13120 8096 8098 3426 3427 := CountSegment.join atomic_8097 atomic_8098
private theorem within_063_031_agreement : RowAgreement 8096 8098 := RowAgreement.join atomic_8097_agreement atomic_8098_agreement
private def within_063_032 : CountSegment 1 10 13120 8098 8100 3427 3427 := CountSegment.join atomic_8099 atomic_8100
private theorem within_063_032_agreement : RowAgreement 8098 8100 := RowAgreement.join atomic_8099_agreement atomic_8100_agreement
private def within_063_033 : CountSegment 1 10 13120 8096 8100 3426 3427 := CountSegment.join within_063_031 within_063_032
private theorem within_063_033_agreement : RowAgreement 8096 8100 := RowAgreement.join within_063_031_agreement within_063_032_agreement
private def within_063_034 : CountSegment 1 10 13120 8100 8102 3427 3428 := CountSegment.join atomic_8101 atomic_8102
private theorem within_063_034_agreement : RowAgreement 8100 8102 := RowAgreement.join atomic_8101_agreement atomic_8102_agreement
private def within_063_035 : CountSegment 1 10 13120 8102 8104 3428 3430 := CountSegment.join atomic_8103 atomic_8104
private theorem within_063_035_agreement : RowAgreement 8102 8104 := RowAgreement.join atomic_8103_agreement atomic_8104_agreement
private def within_063_036 : CountSegment 1 10 13120 8100 8104 3427 3430 := CountSegment.join within_063_034 within_063_035
private theorem within_063_036_agreement : RowAgreement 8100 8104 := RowAgreement.join within_063_034_agreement within_063_035_agreement
private def within_063_037 : CountSegment 1 10 13120 8096 8104 3426 3430 := CountSegment.join within_063_033 within_063_036
private theorem within_063_037_agreement : RowAgreement 8096 8104 := RowAgreement.join within_063_033_agreement within_063_036_agreement
private def within_063_038 : CountSegment 1 10 13120 8104 8106 3430 3430 := CountSegment.join atomic_8105 atomic_8106
private theorem within_063_038_agreement : RowAgreement 8104 8106 := RowAgreement.join atomic_8105_agreement atomic_8106_agreement
private def within_063_039 : CountSegment 1 10 13120 8106 8108 3430 3430 := CountSegment.join atomic_8107 atomic_8108
private theorem within_063_039_agreement : RowAgreement 8106 8108 := RowAgreement.join atomic_8107_agreement atomic_8108_agreement
private def within_063_040 : CountSegment 1 10 13120 8104 8108 3430 3430 := CountSegment.join within_063_038 within_063_039
private theorem within_063_040_agreement : RowAgreement 8104 8108 := RowAgreement.join within_063_038_agreement within_063_039_agreement
private def within_063_041 : CountSegment 1 10 13120 8108 8110 3430 3431 := CountSegment.join atomic_8109 atomic_8110
private theorem within_063_041_agreement : RowAgreement 8108 8110 := RowAgreement.join atomic_8109_agreement atomic_8110_agreement
private def within_063_042 : CountSegment 1 10 13120 8110 8112 3431 3433 := CountSegment.join atomic_8111 atomic_8112
private theorem within_063_042_agreement : RowAgreement 8110 8112 := RowAgreement.join atomic_8111_agreement atomic_8112_agreement
private def within_063_043 : CountSegment 1 10 13120 8108 8112 3430 3433 := CountSegment.join within_063_041 within_063_042
private theorem within_063_043_agreement : RowAgreement 8108 8112 := RowAgreement.join within_063_041_agreement within_063_042_agreement
private def within_063_044 : CountSegment 1 10 13120 8104 8112 3430 3433 := CountSegment.join within_063_040 within_063_043
private theorem within_063_044_agreement : RowAgreement 8104 8112 := RowAgreement.join within_063_040_agreement within_063_043_agreement
private def within_063_045 : CountSegment 1 10 13120 8096 8112 3426 3433 := CountSegment.join within_063_037 within_063_044
private theorem within_063_045_agreement : RowAgreement 8096 8112 := RowAgreement.join within_063_037_agreement within_063_044_agreement
private def within_063_046 : CountSegment 1 10 13120 8112 8114 3433 3434 := CountSegment.join atomic_8113 atomic_8114
private theorem within_063_046_agreement : RowAgreement 8112 8114 := RowAgreement.join atomic_8113_agreement atomic_8114_agreement
private def within_063_047 : CountSegment 1 10 13120 8114 8116 3434 3436 := CountSegment.join atomic_8115 atomic_8116
private theorem within_063_047_agreement : RowAgreement 8114 8116 := RowAgreement.join atomic_8115_agreement atomic_8116_agreement
private def within_063_048 : CountSegment 1 10 13120 8112 8116 3433 3436 := CountSegment.join within_063_046 within_063_047
private theorem within_063_048_agreement : RowAgreement 8112 8116 := RowAgreement.join within_063_046_agreement within_063_047_agreement
private def within_063_049 : CountSegment 1 10 13120 8116 8118 3436 3436 := CountSegment.join atomic_8117 atomic_8118
private theorem within_063_049_agreement : RowAgreement 8116 8118 := RowAgreement.join atomic_8117_agreement atomic_8118_agreement
private def within_063_050 : CountSegment 1 10 13120 8118 8120 3436 3437 := CountSegment.join atomic_8119 atomic_8120
private theorem within_063_050_agreement : RowAgreement 8118 8120 := RowAgreement.join atomic_8119_agreement atomic_8120_agreement
private def within_063_051 : CountSegment 1 10 13120 8116 8120 3436 3437 := CountSegment.join within_063_049 within_063_050
private theorem within_063_051_agreement : RowAgreement 8116 8120 := RowAgreement.join within_063_049_agreement within_063_050_agreement
private def within_063_052 : CountSegment 1 10 13120 8112 8120 3433 3437 := CountSegment.join within_063_048 within_063_051
private theorem within_063_052_agreement : RowAgreement 8112 8120 := RowAgreement.join within_063_048_agreement within_063_051_agreement
private def within_063_053 : CountSegment 1 10 13120 8120 8122 3437 3438 := CountSegment.join atomic_8121 atomic_8122
private theorem within_063_053_agreement : RowAgreement 8120 8122 := RowAgreement.join atomic_8121_agreement atomic_8122_agreement
private def within_063_054 : CountSegment 1 10 13120 8122 8124 3438 3439 := CountSegment.join atomic_8123 atomic_8124
private theorem within_063_054_agreement : RowAgreement 8122 8124 := RowAgreement.join atomic_8123_agreement atomic_8124_agreement
private def within_063_055 : CountSegment 1 10 13120 8120 8124 3437 3439 := CountSegment.join within_063_053 within_063_054
private theorem within_063_055_agreement : RowAgreement 8120 8124 := RowAgreement.join within_063_053_agreement within_063_054_agreement
private def within_063_056 : CountSegment 1 10 13120 8124 8126 3439 3440 := CountSegment.join atomic_8125 atomic_8126
private theorem within_063_056_agreement : RowAgreement 8124 8126 := RowAgreement.join atomic_8125_agreement atomic_8126_agreement
private def within_063_057 : CountSegment 1 10 13120 8126 8128 3440 3442 := CountSegment.join atomic_8127 atomic_8128
private theorem within_063_057_agreement : RowAgreement 8126 8128 := RowAgreement.join atomic_8127_agreement atomic_8128_agreement
private def within_063_058 : CountSegment 1 10 13120 8124 8128 3439 3442 := CountSegment.join within_063_056 within_063_057
private theorem within_063_058_agreement : RowAgreement 8124 8128 := RowAgreement.join within_063_056_agreement within_063_057_agreement
private def within_063_059 : CountSegment 1 10 13120 8120 8128 3437 3442 := CountSegment.join within_063_055 within_063_058
private theorem within_063_059_agreement : RowAgreement 8120 8128 := RowAgreement.join within_063_055_agreement within_063_058_agreement
private def within_063_060 : CountSegment 1 10 13120 8112 8128 3433 3442 := CountSegment.join within_063_052 within_063_059
private theorem within_063_060_agreement : RowAgreement 8112 8128 := RowAgreement.join within_063_052_agreement within_063_059_agreement
private def within_063_061 : CountSegment 1 10 13120 8096 8128 3426 3442 := CountSegment.join within_063_045 within_063_060
private theorem within_063_061_agreement : RowAgreement 8096 8128 := RowAgreement.join within_063_045_agreement within_063_060_agreement
private def within_063_062 : CountSegment 1 10 13120 8064 8128 3415 3442 := CountSegment.join within_063_030 within_063_061
private theorem within_063_062_agreement : RowAgreement 8064 8128 := RowAgreement.join within_063_030_agreement within_063_061_agreement
private def within_063_063 : CountSegment 1 10 13120 8128 8130 3442 3444 := CountSegment.join atomic_8129 atomic_8130
private theorem within_063_063_agreement : RowAgreement 8128 8130 := RowAgreement.join atomic_8129_agreement atomic_8130_agreement
private def within_063_064 : CountSegment 1 10 13120 8130 8132 3444 3445 := CountSegment.join atomic_8131 atomic_8132
private theorem within_063_064_agreement : RowAgreement 8130 8132 := RowAgreement.join atomic_8131_agreement atomic_8132_agreement
private def within_063_065 : CountSegment 1 10 13120 8128 8132 3442 3445 := CountSegment.join within_063_063 within_063_064
private theorem within_063_065_agreement : RowAgreement 8128 8132 := RowAgreement.join within_063_063_agreement within_063_064_agreement
private def within_063_066 : CountSegment 1 10 13120 8132 8134 3445 3446 := CountSegment.join atomic_8133 atomic_8134
private theorem within_063_066_agreement : RowAgreement 8132 8134 := RowAgreement.join atomic_8133_agreement atomic_8134_agreement
private def within_063_067 : CountSegment 1 10 13120 8134 8136 3446 3446 := CountSegment.join atomic_8135 atomic_8136
private theorem within_063_067_agreement : RowAgreement 8134 8136 := RowAgreement.join atomic_8135_agreement atomic_8136_agreement
private def within_063_068 : CountSegment 1 10 13120 8132 8136 3445 3446 := CountSegment.join within_063_066 within_063_067
private theorem within_063_068_agreement : RowAgreement 8132 8136 := RowAgreement.join within_063_066_agreement within_063_067_agreement
private def within_063_069 : CountSegment 1 10 13120 8128 8136 3442 3446 := CountSegment.join within_063_065 within_063_068
private theorem within_063_069_agreement : RowAgreement 8128 8136 := RowAgreement.join within_063_065_agreement within_063_068_agreement
private def within_063_070 : CountSegment 1 10 13120 8136 8138 3446 3446 := CountSegment.join atomic_8137 atomic_8138
private theorem within_063_070_agreement : RowAgreement 8136 8138 := RowAgreement.join atomic_8137_agreement atomic_8138_agreement
private def within_063_071 : CountSegment 1 10 13120 8138 8140 3446 3447 := CountSegment.join atomic_8139 atomic_8140
private theorem within_063_071_agreement : RowAgreement 8138 8140 := RowAgreement.join atomic_8139_agreement atomic_8140_agreement
private def within_063_072 : CountSegment 1 10 13120 8136 8140 3446 3447 := CountSegment.join within_063_070 within_063_071
private theorem within_063_072_agreement : RowAgreement 8136 8140 := RowAgreement.join within_063_070_agreement within_063_071_agreement
private def within_063_073 : CountSegment 1 10 13120 8140 8142 3447 3449 := CountSegment.join atomic_8141 atomic_8142
private theorem within_063_073_agreement : RowAgreement 8140 8142 := RowAgreement.join atomic_8141_agreement atomic_8142_agreement
private def within_063_074 : CountSegment 1 10 13120 8142 8144 3449 3449 := CountSegment.join atomic_8143 atomic_8144
private theorem within_063_074_agreement : RowAgreement 8142 8144 := RowAgreement.join atomic_8143_agreement atomic_8144_agreement
private def within_063_075 : CountSegment 1 10 13120 8140 8144 3447 3449 := CountSegment.join within_063_073 within_063_074
private theorem within_063_075_agreement : RowAgreement 8140 8144 := RowAgreement.join within_063_073_agreement within_063_074_agreement
private def within_063_076 : CountSegment 1 10 13120 8136 8144 3446 3449 := CountSegment.join within_063_072 within_063_075
private theorem within_063_076_agreement : RowAgreement 8136 8144 := RowAgreement.join within_063_072_agreement within_063_075_agreement
private def within_063_077 : CountSegment 1 10 13120 8128 8144 3442 3449 := CountSegment.join within_063_069 within_063_076
private theorem within_063_077_agreement : RowAgreement 8128 8144 := RowAgreement.join within_063_069_agreement within_063_076_agreement
private def within_063_078 : CountSegment 1 10 13120 8144 8146 3449 3449 := CountSegment.join atomic_8145 atomic_8146
private theorem within_063_078_agreement : RowAgreement 8144 8146 := RowAgreement.join atomic_8145_agreement atomic_8146_agreement
private def within_063_079 : CountSegment 1 10 13120 8146 8148 3449 3450 := CountSegment.join atomic_8147 atomic_8148
private theorem within_063_079_agreement : RowAgreement 8146 8148 := RowAgreement.join atomic_8147_agreement atomic_8148_agreement
private def within_063_080 : CountSegment 1 10 13120 8144 8148 3449 3450 := CountSegment.join within_063_078 within_063_079
private theorem within_063_080_agreement : RowAgreement 8144 8148 := RowAgreement.join within_063_078_agreement within_063_079_agreement
private def within_063_081 : CountSegment 1 10 13120 8148 8150 3450 3451 := CountSegment.join atomic_8149 atomic_8150
private theorem within_063_081_agreement : RowAgreement 8148 8150 := RowAgreement.join atomic_8149_agreement atomic_8150_agreement
private def within_063_082 : CountSegment 1 10 13120 8150 8152 3451 3454 := CountSegment.join atomic_8151 atomic_8152
private theorem within_063_082_agreement : RowAgreement 8150 8152 := RowAgreement.join atomic_8151_agreement atomic_8152_agreement
private def within_063_083 : CountSegment 1 10 13120 8148 8152 3450 3454 := CountSegment.join within_063_081 within_063_082
private theorem within_063_083_agreement : RowAgreement 8148 8152 := RowAgreement.join within_063_081_agreement within_063_082_agreement
private def within_063_084 : CountSegment 1 10 13120 8144 8152 3449 3454 := CountSegment.join within_063_080 within_063_083
private theorem within_063_084_agreement : RowAgreement 8144 8152 := RowAgreement.join within_063_080_agreement within_063_083_agreement
private def within_063_085 : CountSegment 1 10 13120 8152 8154 3454 3454 := CountSegment.join atomic_8153 atomic_8154
private theorem within_063_085_agreement : RowAgreement 8152 8154 := RowAgreement.join atomic_8153_agreement atomic_8154_agreement
private def within_063_086 : CountSegment 1 10 13120 8154 8156 3454 3454 := CountSegment.join atomic_8155 atomic_8156
private theorem within_063_086_agreement : RowAgreement 8154 8156 := RowAgreement.join atomic_8155_agreement atomic_8156_agreement
private def within_063_087 : CountSegment 1 10 13120 8152 8156 3454 3454 := CountSegment.join within_063_085 within_063_086
private theorem within_063_087_agreement : RowAgreement 8152 8156 := RowAgreement.join within_063_085_agreement within_063_086_agreement
private def within_063_088 : CountSegment 1 10 13120 8156 8158 3454 3456 := CountSegment.join atomic_8157 atomic_8158
private theorem within_063_088_agreement : RowAgreement 8156 8158 := RowAgreement.join atomic_8157_agreement atomic_8158_agreement
private def within_063_089 : CountSegment 1 10 13120 8158 8160 3456 3457 := CountSegment.join atomic_8159 atomic_8160
private theorem within_063_089_agreement : RowAgreement 8158 8160 := RowAgreement.join atomic_8159_agreement atomic_8160_agreement
private def within_063_090 : CountSegment 1 10 13120 8156 8160 3454 3457 := CountSegment.join within_063_088 within_063_089
private theorem within_063_090_agreement : RowAgreement 8156 8160 := RowAgreement.join within_063_088_agreement within_063_089_agreement
private def within_063_091 : CountSegment 1 10 13120 8152 8160 3454 3457 := CountSegment.join within_063_087 within_063_090
private theorem within_063_091_agreement : RowAgreement 8152 8160 := RowAgreement.join within_063_087_agreement within_063_090_agreement
private def within_063_092 : CountSegment 1 10 13120 8144 8160 3449 3457 := CountSegment.join within_063_084 within_063_091
private theorem within_063_092_agreement : RowAgreement 8144 8160 := RowAgreement.join within_063_084_agreement within_063_091_agreement
private def within_063_093 : CountSegment 1 10 13120 8128 8160 3442 3457 := CountSegment.join within_063_077 within_063_092
private theorem within_063_093_agreement : RowAgreement 8128 8160 := RowAgreement.join within_063_077_agreement within_063_092_agreement
private def within_063_094 : CountSegment 1 10 13120 8160 8162 3457 3458 := CountSegment.join atomic_8161 atomic_8162
private theorem within_063_094_agreement : RowAgreement 8160 8162 := RowAgreement.join atomic_8161_agreement atomic_8162_agreement
private def within_063_095 : CountSegment 1 10 13120 8162 8164 3458 3458 := CountSegment.join atomic_8163 atomic_8164
private theorem within_063_095_agreement : RowAgreement 8162 8164 := RowAgreement.join atomic_8163_agreement atomic_8164_agreement
private def within_063_096 : CountSegment 1 10 13120 8160 8164 3457 3458 := CountSegment.join within_063_094 within_063_095
private theorem within_063_096_agreement : RowAgreement 8160 8164 := RowAgreement.join within_063_094_agreement within_063_095_agreement
private def within_063_097 : CountSegment 1 10 13120 8164 8166 3458 3459 := CountSegment.join atomic_8165 atomic_8166
private theorem within_063_097_agreement : RowAgreement 8164 8166 := RowAgreement.join atomic_8165_agreement atomic_8166_agreement
private def within_063_098 : CountSegment 1 10 13120 8166 8168 3459 3459 := CountSegment.join atomic_8167 atomic_8168
private theorem within_063_098_agreement : RowAgreement 8166 8168 := RowAgreement.join atomic_8167_agreement atomic_8168_agreement
private def within_063_099 : CountSegment 1 10 13120 8164 8168 3458 3459 := CountSegment.join within_063_097 within_063_098
private theorem within_063_099_agreement : RowAgreement 8164 8168 := RowAgreement.join within_063_097_agreement within_063_098_agreement
private def within_063_100 : CountSegment 1 10 13120 8160 8168 3457 3459 := CountSegment.join within_063_096 within_063_099
private theorem within_063_100_agreement : RowAgreement 8160 8168 := RowAgreement.join within_063_096_agreement within_063_099_agreement
private def within_063_101 : CountSegment 1 10 13120 8168 8170 3459 3459 := CountSegment.join atomic_8169 atomic_8170
private theorem within_063_101_agreement : RowAgreement 8168 8170 := RowAgreement.join atomic_8169_agreement atomic_8170_agreement
private def within_063_102 : CountSegment 1 10 13120 8170 8172 3459 3459 := CountSegment.join atomic_8171 atomic_8172
private theorem within_063_102_agreement : RowAgreement 8170 8172 := RowAgreement.join atomic_8171_agreement atomic_8172_agreement
private def within_063_103 : CountSegment 1 10 13120 8168 8172 3459 3459 := CountSegment.join within_063_101 within_063_102
private theorem within_063_103_agreement : RowAgreement 8168 8172 := RowAgreement.join within_063_101_agreement within_063_102_agreement
private def within_063_104 : CountSegment 1 10 13120 8172 8174 3459 3459 := CountSegment.join atomic_8173 atomic_8174
private theorem within_063_104_agreement : RowAgreement 8172 8174 := RowAgreement.join atomic_8173_agreement atomic_8174_agreement
private def within_063_105 : CountSegment 1 10 13120 8174 8176 3459 3460 := CountSegment.join atomic_8175 atomic_8176
private theorem within_063_105_agreement : RowAgreement 8174 8176 := RowAgreement.join atomic_8175_agreement atomic_8176_agreement
private def within_063_106 : CountSegment 1 10 13120 8172 8176 3459 3460 := CountSegment.join within_063_104 within_063_105
private theorem within_063_106_agreement : RowAgreement 8172 8176 := RowAgreement.join within_063_104_agreement within_063_105_agreement
private def within_063_107 : CountSegment 1 10 13120 8168 8176 3459 3460 := CountSegment.join within_063_103 within_063_106
private theorem within_063_107_agreement : RowAgreement 8168 8176 := RowAgreement.join within_063_103_agreement within_063_106_agreement
private def within_063_108 : CountSegment 1 10 13120 8160 8176 3457 3460 := CountSegment.join within_063_100 within_063_107
private theorem within_063_108_agreement : RowAgreement 8160 8176 := RowAgreement.join within_063_100_agreement within_063_107_agreement
private def within_063_109 : CountSegment 1 10 13120 8176 8178 3460 3460 := CountSegment.join atomic_8177 atomic_8178
private theorem within_063_109_agreement : RowAgreement 8176 8178 := RowAgreement.join atomic_8177_agreement atomic_8178_agreement
private def within_063_110 : CountSegment 1 10 13120 8178 8180 3460 3460 := CountSegment.join atomic_8179 atomic_8180
private theorem within_063_110_agreement : RowAgreement 8178 8180 := RowAgreement.join atomic_8179_agreement atomic_8180_agreement
private def within_063_111 : CountSegment 1 10 13120 8176 8180 3460 3460 := CountSegment.join within_063_109 within_063_110
private theorem within_063_111_agreement : RowAgreement 8176 8180 := RowAgreement.join within_063_109_agreement within_063_110_agreement
private def within_063_112 : CountSegment 1 10 13120 8180 8182 3460 3461 := CountSegment.join atomic_8181 atomic_8182
private theorem within_063_112_agreement : RowAgreement 8180 8182 := RowAgreement.join atomic_8181_agreement atomic_8182_agreement
private def within_063_113 : CountSegment 1 10 13120 8182 8184 3461 3463 := CountSegment.join atomic_8183 atomic_8184
private theorem within_063_113_agreement : RowAgreement 8182 8184 := RowAgreement.join atomic_8183_agreement atomic_8184_agreement
private def within_063_114 : CountSegment 1 10 13120 8180 8184 3460 3463 := CountSegment.join within_063_112 within_063_113
private theorem within_063_114_agreement : RowAgreement 8180 8184 := RowAgreement.join within_063_112_agreement within_063_113_agreement
private def within_063_115 : CountSegment 1 10 13120 8176 8184 3460 3463 := CountSegment.join within_063_111 within_063_114
private theorem within_063_115_agreement : RowAgreement 8176 8184 := RowAgreement.join within_063_111_agreement within_063_114_agreement
private def within_063_116 : CountSegment 1 10 13120 8184 8186 3463 3464 := CountSegment.join atomic_8185 atomic_8186
private theorem within_063_116_agreement : RowAgreement 8184 8186 := RowAgreement.join atomic_8185_agreement atomic_8186_agreement
private def within_063_117 : CountSegment 1 10 13120 8186 8188 3464 3465 := CountSegment.join atomic_8187 atomic_8188
private theorem within_063_117_agreement : RowAgreement 8186 8188 := RowAgreement.join atomic_8187_agreement atomic_8188_agreement
private def within_063_118 : CountSegment 1 10 13120 8184 8188 3463 3465 := CountSegment.join within_063_116 within_063_117
private theorem within_063_118_agreement : RowAgreement 8184 8188 := RowAgreement.join within_063_116_agreement within_063_117_agreement
private def within_063_119 : CountSegment 1 10 13120 8188 8190 3465 3467 := CountSegment.join atomic_8189 atomic_8190
private theorem within_063_119_agreement : RowAgreement 8188 8190 := RowAgreement.join atomic_8189_agreement atomic_8190_agreement
private def within_063_120 : CountSegment 1 10 13120 8190 8192 3467 3467 := CountSegment.join atomic_8191 atomic_8192
private theorem within_063_120_agreement : RowAgreement 8190 8192 := RowAgreement.join atomic_8191_agreement atomic_8192_agreement
private def within_063_121 : CountSegment 1 10 13120 8188 8192 3465 3467 := CountSegment.join within_063_119 within_063_120
private theorem within_063_121_agreement : RowAgreement 8188 8192 := RowAgreement.join within_063_119_agreement within_063_120_agreement
private def within_063_122 : CountSegment 1 10 13120 8184 8192 3463 3467 := CountSegment.join within_063_118 within_063_121
private theorem within_063_122_agreement : RowAgreement 8184 8192 := RowAgreement.join within_063_118_agreement within_063_121_agreement
private def within_063_123 : CountSegment 1 10 13120 8176 8192 3460 3467 := CountSegment.join within_063_115 within_063_122
private theorem within_063_123_agreement : RowAgreement 8176 8192 := RowAgreement.join within_063_115_agreement within_063_122_agreement
private def within_063_124 : CountSegment 1 10 13120 8160 8192 3457 3467 := CountSegment.join within_063_108 within_063_123
private theorem within_063_124_agreement : RowAgreement 8160 8192 := RowAgreement.join within_063_108_agreement within_063_123_agreement
private def within_063_125 : CountSegment 1 10 13120 8128 8192 3442 3467 := CountSegment.join within_063_093 within_063_124
private theorem within_063_125_agreement : RowAgreement 8128 8192 := RowAgreement.join within_063_093_agreement within_063_124_agreement
private def within_063_126 : CountSegment 1 10 13120 8064 8192 3415 3467 := CountSegment.join within_063_062 within_063_125
private theorem within_063_126_agreement : RowAgreement 8064 8192 := RowAgreement.join within_063_062_agreement within_063_125_agreement
def segment_063 : CountSegment 1 10 13120 8064 8192 3415 3467 := within_063_126
theorem agreement_063 : RowAgreement 8064 8192 := within_063_126_agreement
end ProjectEulerSolutions.P58Numerical
