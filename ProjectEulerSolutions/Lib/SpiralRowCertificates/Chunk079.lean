import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_10113 : RowCertificate 10113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10113 1) := by change ¬ Nat.Prime 409111303; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 366259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10113 2) := by change ¬ Nat.Prime 409091077; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24064181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10113 3) := by change ¬ Nat.Prime 409070851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58438693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10113 : row 10113 = 0 := (row_10113.checked).trans rfl
theorem model_count_10113 : modelRow 10113 = 0 := by decide
theorem modelrow_agrees_10113 : modelRow 10113 = row 10113 := model_count_10113.trans row_count_10113.symm
def row_10114 : RowCertificate 10114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10114 1) := by change ¬ Nat.Prime 409192213; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11059249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10114 2) := by change ¬ Nat.Prime 409171985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81834397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10114 3) := by change ¬ Nat.Prime 409151757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136383919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10114 : row 10114 = 0 := (row_10114.checked).trans rfl
theorem model_count_10114 : modelRow 10114 = 0 := by decide
theorem modelrow_agrees_10114 : modelRow 10114 = row 10114 := model_count_10114.trans row_count_10114.symm
def row_10115 : RowCertificate 10115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10115 1) := by change ¬ Nat.Prime 409273131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136424377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10115 2) := by change ¬ Nat.Prime 409252901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14112169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10115 3) := by change ¬ Nat.Prime 409232671; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5605927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10115 : row 10115 = 0 := (row_10115.checked).trans rfl
theorem model_count_10115 : modelRow 10115 = 0 := by decide
theorem modelrow_agrees_10115 : modelRow 10115 = row 10115 := model_count_10115.trans row_count_10115.symm
def row_10116 : RowCertificate 10116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10116 1) := by change ¬ Nat.Prime 409354057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58479151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10116 2) := by change ¬ Nat.Prime 409333825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81866765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10116 3) := by change ¬ Nat.Prime 409313593; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31485661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10116 : row 10116 = 0 := (row_10116.checked).trans rfl
theorem model_count_10116 : modelRow 10116 = 0 := by decide
theorem modelrow_agrees_10116 : modelRow 10116 = row 10116 := model_count_10116.trans row_count_10116.symm
def row_10117 : RowCertificate 10117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10117 1) := by change ¬ Nat.Prime 409434991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58490713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10117 2) := by change ¬ Nat.Prime 409414757; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24083221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10117 3) := by change ¬ Nat.Prime 409394523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136464841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10117 : row 10117 = 0 := (row_10117.checked).trans rfl
theorem model_count_10117 : modelRow 10117 = 0 := by decide
theorem modelrow_agrees_10117 : modelRow 10117 = row 10117 := model_count_10117.trans row_count_10117.symm
def row_10118 : RowCertificate 10118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10118 1) := by change ¬ Nat.Prime 409515933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136505311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10118 2) := by change ¬ Nat.Prime 409495697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31499669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10118 3) := by change ¬ Nat.Prime 409475461; exact PrattCertificates.composite_of_product (factor := 631) (cofactor := 648931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10118 : row 10118 = 0 := (row_10118.checked).trans rfl
theorem model_count_10118 : modelRow 10118 = 0 := by decide
theorem modelrow_agrees_10118 : modelRow 10118 = row 10118 := model_count_10118.trans row_count_10118.symm
def row_10119 : RowCertificate 10119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10119 1) := by change ¬ Nat.Prime 409596883; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6714703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10119 2) := by change ¬ Nat.Prime 409576645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81915329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10119 3) := by change ¬ Nat.Prime 409556407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31504339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10119 : row 10119 = 0 := (row_10119.checked).trans rfl
theorem model_count_10119 : modelRow 10119 = 0 := by decide
theorem modelrow_agrees_10119 : modelRow 10119 = row 10119 := model_count_10119.trans row_count_10119.symm
def row_10120 : RowCertificate 10120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10120 1) := by change ¬ Nat.Prime 409677841; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2609413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10120 2) := by change Nat.Prime 409657601; exact ProjectEulerSolutions.P58PrimeNodes.p_409657601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10120 3) := by change ¬ Nat.Prime 409637361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136545787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10120 : row 10120 = 1 := (row_10120.checked).trans rfl
theorem model_count_10120 : modelRow 10120 = 1 := by decide
theorem modelrow_agrees_10120 : modelRow 10120 = row 10120 := model_count_10120.trans row_count_10120.symm
def row_10121 : RowCertificate 10121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10121 1) := by change ¬ Nat.Prime 409758807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136586269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10121 2) := by change ¬ Nat.Prime 409738565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81947713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10121 3) := by change ¬ Nat.Prime 409718323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58531189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10121 : row 10121 = 0 := (row_10121.checked).trans rfl
theorem model_count_10121 : modelRow 10121 = 0 := by decide
theorem modelrow_agrees_10121 : modelRow 10121 = row 10121 := model_count_10121.trans row_count_10121.symm
def row_10122 : RowCertificate 10122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10122 1) := by change ¬ Nat.Prime 409839781; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31526137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10122 2) := by change Nat.Prime 409819537; exact ProjectEulerSolutions.P58PrimeNodes.p_409819537.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10122 3) := by change Nat.Prime 409799293; exact ProjectEulerSolutions.P58PrimeNodes.p_409799293.prime
    exact decide_eq_true prime
theorem row_count_10122 : row 10122 = 2 := (row_10122.checked).trans rfl
theorem model_count_10122 : modelRow 10122 = 2 := by decide
theorem modelrow_agrees_10122 : modelRow 10122 = row 10122 := model_count_10122.trans row_count_10122.symm
def row_10123 : RowCertificate 10123 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10123 1) := by change ¬ Nat.Prime 409920763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58560109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10123 2) := by change ¬ Nat.Prime 409900517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31530809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10123 3) := by change ¬ Nat.Prime 409880271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136626757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10123 : row 10123 = 0 := (row_10123.checked).trans rfl
theorem model_count_10123 : modelRow 10123 = 0 := by decide
theorem modelrow_agrees_10123 : modelRow 10123 = row 10123 := model_count_10123.trans row_count_10123.symm
def row_10124 : RowCertificate 10124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10124 1) := by change ¬ Nat.Prime 410001753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136667251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10124 2) := by change ¬ Nat.Prime 409981505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81996301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10124 3) := by change ¬ Nat.Prime 409961257; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 5189383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10124 : row 10124 = 0 := (row_10124.checked).trans rfl
theorem model_count_10124 : modelRow 10124 = 0 := by decide
theorem modelrow_agrees_10124 : modelRow 10124 = row 10124 := model_count_10124.trans row_count_10124.symm
def row_10125 : RowCertificate 10125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10125 1) := by change ¬ Nat.Prime 410082751; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31544827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10125 2) := by change Nat.Prime 410062501; exact ProjectEulerSolutions.P58PrimeNodes.p_410062501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10125 3) := by change ¬ Nat.Prime 410042251; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11082223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10125 : row 10125 = 1 := (row_10125.checked).trans rfl
theorem model_count_10125 : modelRow 10125 = 1 := by decide
theorem modelrow_agrees_10125 : modelRow 10125 = row 10125 := model_count_10125.trans row_count_10125.symm
def row_10126 : RowCertificate 10126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10126 1) := by change ¬ Nat.Prime 410163757; exact PrattCertificates.composite_of_product (factor := 1327) (cofactor := 309091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10126 2) := by change ¬ Nat.Prime 410143505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82028701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10126 3) := by change ¬ Nat.Prime 410123253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136707751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10126 : row 10126 = 0 := (row_10126.checked).trans rfl
theorem model_count_10126 : modelRow 10126 = 0 := by decide
theorem modelrow_agrees_10126 : modelRow 10126 = row 10126 := model_count_10126.trans row_count_10126.symm
def row_10127 : RowCertificate 10127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10127 1) := by change ¬ Nat.Prime 410244771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136748257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10127 2) := by change ¬ Nat.Prime 410224517; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14145673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10127 3) := by change ¬ Nat.Prime 410204263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58600609) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10127 : row 10127 = 0 := (row_10127.checked).trans rfl
theorem model_count_10127 : modelRow 10127 = 0 := by decide
theorem modelrow_agrees_10127 : modelRow 10127 = row 10127 := model_count_10127.trans row_count_10127.symm
def row_10128 : RowCertificate 10128 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10128 1) := by change Nat.Prime 410325793; exact ProjectEulerSolutions.P58PrimeNodes.p_410325793.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10128 2) := by change Nat.Prime 410305537; exact ProjectEulerSolutions.P58PrimeNodes.p_410305537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10128 3) := by change ¬ Nat.Prime 410285281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58612183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10128 : row 10128 = 2 := (row_10128.checked).trans rfl
theorem model_count_10128 : modelRow 10128 = 2 := by decide
theorem modelrow_agrees_10128 : modelRow 10128 = row 10128 := model_count_10128.trans row_count_10128.symm
def row_10129 : RowCertificate 10129 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10129 1) := by change Nat.Prime 410406823; exact ProjectEulerSolutions.P58PrimeNodes.p_410406823.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10129 2) := by change ¬ Nat.Prime 410386565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82077313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10129 3) := by change ¬ Nat.Prime 410366307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136788769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10129 : row 10129 = 1 := (row_10129.checked).trans rfl
theorem model_count_10129 : modelRow 10129 = 1 := by decide
theorem modelrow_agrees_10129 : modelRow 10129 = row 10129 := model_count_10129.trans row_count_10129.symm
def row_10130 : RowCertificate 10130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10130 1) := by change ¬ Nat.Prime 410487861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136829287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10130 2) := by change ¬ Nat.Prime 410467601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24145153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10130 3) := by change ¬ Nat.Prime 410447341; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9545287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10130 : row 10130 = 0 := (row_10130.checked).trans rfl
theorem model_count_10130 : modelRow 10130 = 0 := by decide
theorem modelrow_agrees_10130 : modelRow 10130 = row 10130 := model_count_10130.trans row_count_10130.symm
def row_10131 : RowCertificate 10131 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10131 1) := by change ¬ Nat.Prime 410568907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58652701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10131 2) := by change ¬ Nat.Prime 410548645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82109729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10131 3) := by change ¬ Nat.Prime 410528383; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21606757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10131 : row 10131 = 0 := (row_10131.checked).trans rfl
theorem model_count_10131 : modelRow 10131 = 0 := by decide
theorem modelrow_agrees_10131 : modelRow 10131 = row 10131 := model_count_10131.trans row_count_10131.symm
def row_10132 : RowCertificate 10132 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10132 1) := by change ¬ Nat.Prime 410649961; exact PrattCertificates.composite_of_product (factor := 5779) (cofactor := 71059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10132 2) := by change ¬ Nat.Prime 410629697; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 669869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10132 3) := by change ¬ Nat.Prime 410609433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136869811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10132 : row 10132 = 0 := (row_10132.checked).trans rfl
theorem model_count_10132 : modelRow 10132 = 0 := by decide
theorem modelrow_agrees_10132 : modelRow 10132 = row 10132 := model_count_10132.trans row_count_10132.symm
def row_10133 : RowCertificate 10133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10133 1) := by change ¬ Nat.Prime 410731023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136910341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10133 2) := by change Nat.Prime 410710757; exact ProjectEulerSolutions.P58PrimeNodes.p_410710757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10133 3) := by change ¬ Nat.Prime 410690491; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21615289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10133 : row 10133 = 1 := (row_10133.checked).trans rfl
theorem model_count_10133 : modelRow 10133 = 1 := by decide
theorem modelrow_agrees_10133 : modelRow 10133 = row 10133 := model_count_10133.trans row_count_10133.symm
def row_10134 : RowCertificate 10134 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10134 1) := by change ¬ Nat.Prime 410812093; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13252003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10134 2) := by change ¬ Nat.Prime 410791825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82158365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10134 3) := by change ¬ Nat.Prime 410771557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58681651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10134 : row 10134 = 0 := (row_10134.checked).trans rfl
theorem model_count_10134 : modelRow 10134 = 0 := by decide
theorem modelrow_agrees_10134 : modelRow 10134 = row 10134 := model_count_10134.trans row_count_10134.symm
def row_10135 : RowCertificate 10135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10135 1) := by change ¬ Nat.Prime 410893171; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31607167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10135 2) := by change ¬ Nat.Prime 410872901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11104673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10135 3) := by change ¬ Nat.Prime 410852631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136950877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10135 : row 10135 = 0 := (row_10135.checked).trans rfl
theorem model_count_10135 : modelRow 10135 = 0 := by decide
theorem modelrow_agrees_10135 : modelRow 10135 = row 10135 := model_count_10135.trans row_count_10135.symm
def row_10136 : RowCertificate 10136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10136 1) := by change ¬ Nat.Prime 410974257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136991419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10136 2) := by change ¬ Nat.Prime 410953985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82190797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10136 3) := by change ¬ Nat.Prime 410933713; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 6133339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10136 : row 10136 = 0 := (row_10136.checked).trans rfl
theorem model_count_10136 : modelRow 10136 = 0 := by decide
theorem modelrow_agrees_10136 : modelRow 10136 = row 10136 := model_count_10136.trans row_count_10136.symm
def row_10137 : RowCertificate 10137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10137 1) := by change ¬ Nat.Prime 411055351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58722193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10137 2) := by change ¬ Nat.Prime 411035077; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1794913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10137 3) := by change ¬ Nat.Prime 411014803; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 556177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10137 : row 10137 = 0 := (row_10137.checked).trans rfl
theorem model_count_10137 : modelRow 10137 = 0 := by decide
theorem modelrow_agrees_10137 : modelRow 10137 = row 10137 := model_count_10137.trans row_count_10137.symm
def row_10138 : RowCertificate 10138 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10138 1) := by change ¬ Nat.Prime 411136453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58733779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10138 2) := by change ¬ Nat.Prime 411116177; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7756909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10138 3) := by change ¬ Nat.Prime 411095901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137031967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10138 : row 10138 = 0 := (row_10138.checked).trans rfl
theorem model_count_10138 : modelRow 10138 = 0 := by decide
theorem modelrow_agrees_10138 : modelRow 10138 = row 10138 := model_count_10138.trans row_count_10138.symm
def row_10139 : RowCertificate 10139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10139 1) := by change ¬ Nat.Prime 411217563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137072521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10139 2) := by change ¬ Nat.Prime 411197285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82239457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10139 3) := by change Nat.Prime 411177007; exact ProjectEulerSolutions.P58PrimeNodes.p_411177007.prime
    exact decide_eq_true prime
theorem row_count_10139 : row 10139 = 1 := (row_10139.checked).trans rfl
theorem model_count_10139 : modelRow 10139 = 1 := by decide
theorem modelrow_agrees_10139 : modelRow 10139 = row 10139 := model_count_10139.trans row_count_10139.symm
def row_10140 : RowCertificate 10140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10140 1) := by change ¬ Nat.Prime 411298681; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21647299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10140 2) := by change Nat.Prime 411278401; exact ProjectEulerSolutions.P58PrimeNodes.p_411278401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10140 3) := by change ¬ Nat.Prime 411258121; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13266391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10140 : row 10140 = 1 := (row_10140.checked).trans rfl
theorem model_count_10140 : modelRow 10140 = 1 := by decide
theorem modelrow_agrees_10140 : modelRow 10140 = row 10140 := model_count_10140.trans row_count_10140.symm
def row_10141 : RowCertificate 10141 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10141 1) := by change Nat.Prime 411379807; exact ProjectEulerSolutions.P58PrimeNodes.p_411379807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10141 2) := by change ¬ Nat.Prime 411359525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82271905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10141 3) := by change ¬ Nat.Prime 411339243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137113081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10141 : row 10141 = 1 := (row_10141.checked).trans rfl
theorem model_count_10141 : modelRow 10141 = 1 := by decide
theorem modelrow_agrees_10141 : modelRow 10141 = row 10141 := model_count_10141.trans row_count_10141.symm
def row_10142 : RowCertificate 10142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10142 1) := by change ¬ Nat.Prime 411460941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137153647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10142 2) := by change ¬ Nat.Prime 411440657; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 607741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10142 3) := by change ¬ Nat.Prime 411420373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58774339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10142 : row 10142 = 0 := (row_10142.checked).trans rfl
theorem model_count_10142 : modelRow 10142 = 0 := by decide
theorem modelrow_agrees_10142 : modelRow 10142 = row 10142 := model_count_10142.trans row_count_10142.symm
def row_10143 : RowCertificate 10143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10143 1) := by change ¬ Nat.Prime 411542083; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11122759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10143 2) := by change ¬ Nat.Prime 411521797; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10037117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10143 3) := by change ¬ Nat.Prime 411501511; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5637007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10143 : row 10143 = 0 := (row_10143.checked).trans rfl
theorem model_count_10143 : modelRow 10143 = 0 := by decide
theorem modelrow_agrees_10143 : modelRow 10143 = row 10143 := model_count_10143.trans row_count_10143.symm
def row_10144 : RowCertificate 10144 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10144 1) := by change ¬ Nat.Prime 411623233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58803319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10144 2) := by change ¬ Nat.Prime 411602945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82320589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10144 3) := by change ¬ Nat.Prime 411582657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137194219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10144 : row 10144 = 0 := (row_10144.checked).trans rfl
theorem model_count_10144 : modelRow 10144 = 0 := by decide
theorem modelrow_agrees_10144 : modelRow 10144 = row 10144 := model_count_10144.trans row_count_10144.symm
def row_10145 : RowCertificate 10145 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10145 1) := by change ¬ Nat.Prime 411704391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137234797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10145 2) := by change ¬ Nat.Prime 411684101; exact PrattCertificates.composite_of_product (factor := 3769) (cofactor := 109229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10145 3) := by change ¬ Nat.Prime 411663811; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31666447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10145 : row 10145 = 0 := (row_10145.checked).trans rfl
theorem model_count_10145 : modelRow 10145 = 0 := by decide
theorem modelrow_agrees_10145 : modelRow 10145 = row 10145 := model_count_10145.trans row_count_10145.symm
def row_10146 : RowCertificate 10146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10146 1) := by change ¬ Nat.Prime 411785557; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 595927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10146 2) := by change ¬ Nat.Prime 411765265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82353053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10146 3) := by change ¬ Nat.Prime 411744973; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 439429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10146 : row 10146 = 0 := (row_10146.checked).trans rfl
theorem model_count_10146 : modelRow 10146 = 0 := by decide
theorem modelrow_agrees_10146 : modelRow 10146 = row 10146 := model_count_10146.trans row_count_10146.symm
def row_10147 : RowCertificate 10147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10147 1) := by change ¬ Nat.Prime 411866731; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 671887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10147 2) := by change ¬ Nat.Prime 411846437; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24226261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10147 3) := by change ¬ Nat.Prime 411826143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137275381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10147 : row 10147 = 0 := (row_10147.checked).trans rfl
theorem model_count_10147 : modelRow 10147 = 0 := by decide
theorem modelrow_agrees_10147 : modelRow 10147 = row 10147 := model_count_10147.trans row_count_10147.symm
def row_10148 : RowCertificate 10148 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10148 1) := by change ¬ Nat.Prime 411947913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137315971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10148 2) := by change ¬ Nat.Prime 411927617; exact PrattCertificates.composite_of_product (factor := 1229) (cofactor := 335173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10148 3) := by change ¬ Nat.Prime 411907321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58843903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10148 : row 10148 = 0 := (row_10148.checked).trans rfl
theorem model_count_10148 : modelRow 10148 = 0 := by decide
theorem modelrow_agrees_10148 : modelRow 10148 = row 10148 := model_count_10148.trans row_count_10148.symm
def row_10149 : RowCertificate 10149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10149 1) := by change ¬ Nat.Prime 412029103; exact PrattCertificates.composite_of_product (factor := 2719) (cofactor := 151537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10149 2) := by change ¬ Nat.Prime 412008805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82401761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10149 3) := by change ¬ Nat.Prime 411988507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58855501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10149 : row 10149 = 0 := (row_10149.checked).trans rfl
theorem model_count_10149 : modelRow 10149 = 0 := by decide
theorem modelrow_agrees_10149 : modelRow 10149 = row 10149 := model_count_10149.trans row_count_10149.symm
def row_10150 : RowCertificate 10150 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10150 1) := by change ¬ Nat.Prime 412110301; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3244963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10150 2) := by change Nat.Prime 412090001; exact ProjectEulerSolutions.P58PrimeNodes.p_412090001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10150 3) := by change ¬ Nat.Prime 412069701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137356567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10150 : row 10150 = 1 := (row_10150.checked).trans rfl
theorem model_count_10150 : modelRow 10150 = 1 := by decide
theorem modelrow_agrees_10150 : modelRow 10150 = row 10150 := model_count_10150.trans row_count_10150.symm
def row_10151 : RowCertificate 10151 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10151 1) := by change ¬ Nat.Prime 412191507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137397169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10151 2) := by change ¬ Nat.Prime 412171205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82434241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10151 3) := by change Nat.Prime 412150903; exact ProjectEulerSolutions.P58PrimeNodes.p_412150903.prime
    exact decide_eq_true prime
theorem row_count_10151 : row 10151 = 1 := (row_10151.checked).trans rfl
theorem model_count_10151 : modelRow 10151 = 1 := by decide
theorem modelrow_agrees_10151 : modelRow 10151 = row 10151 := model_count_10151.trans row_count_10151.symm
def row_10152 : RowCertificate 10152 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10152 1) := by change ¬ Nat.Prime 412272721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58896103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10152 2) := by change ¬ Nat.Prime 412252417; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10054937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10152 3) := by change ¬ Nat.Prime 412232113; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21696427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10152 : row 10152 = 0 := (row_10152.checked).trans rfl
theorem model_count_10152 : modelRow 10152 = 0 := by decide
theorem modelrow_agrees_10152 : modelRow 10152 = row 10152 := model_count_10152.trans row_count_10152.symm
def row_10153 : RowCertificate 10153 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10153 1) := by change Nat.Prime 412353943; exact ProjectEulerSolutions.P58PrimeNodes.p_412353943.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10153 2) := by change Nat.Prime 412333637; exact ProjectEulerSolutions.P58PrimeNodes.p_412333637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10153 3) := by change ¬ Nat.Prime 412313331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137437777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10153 : row 10153 = 2 := (row_10153.checked).trans rfl
theorem model_count_10153 : modelRow 10153 = 2 := by decide
theorem modelrow_agrees_10153 : modelRow 10153 = row 10153 := model_count_10153.trans row_count_10153.symm
def row_10154 : RowCertificate 10154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10154 1) := by change ¬ Nat.Prime 412435173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137478391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10154 2) := by change ¬ Nat.Prime 412414865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82482973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10154 3) := by change Nat.Prime 412394557; exact ProjectEulerSolutions.P58PrimeNodes.p_412394557.prime
    exact decide_eq_true prime
theorem row_count_10154 : row 10154 = 1 := (row_10154.checked).trans rfl
theorem model_count_10154 : modelRow 10154 = 1 := by decide
theorem modelrow_agrees_10154 : modelRow 10154 = row 10154 := model_count_10154.trans row_count_10154.symm
def row_10155 : RowCertificate 10155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10155 1) := by change ¬ Nat.Prime 412516411; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13306981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10155 2) := by change ¬ Nat.Prime 412496101; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 1039033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10155 3) := by change ¬ Nat.Prime 412475791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58925113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10155 : row 10155 = 0 := (row_10155.checked).trans rfl
theorem model_count_10155 : modelRow 10155 = 0 := by decide
theorem modelrow_agrees_10155 : modelRow 10155 = row 10155 := model_count_10155.trans row_count_10155.symm
def row_10156 : RowCertificate 10156 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10156 1) := by change Nat.Prime 412597657; exact ProjectEulerSolutions.P58PrimeNodes.p_412597657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10156 2) := by change ¬ Nat.Prime 412577345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82515469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10156 3) := by change ¬ Nat.Prime 412557033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137519011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10156 : row 10156 = 1 := (row_10156.checked).trans rfl
theorem model_count_10156 : modelRow 10156 = 1 := by decide
theorem modelrow_agrees_10156 : modelRow 10156 = row 10156 := model_count_10156.trans row_count_10156.symm
def row_10157 : RowCertificate 10157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10157 1) := by change ¬ Nat.Prime 412678911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137559637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10157 2) := by change ¬ Nat.Prime 412658597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31742969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10157 3) := by change Nat.Prime 412638283; exact ProjectEulerSolutions.P58PrimeNodes.p_412638283.prime
    exact decide_eq_true prime
theorem row_count_10157 : row 10157 = 1 := (row_10157.checked).trans rfl
theorem model_count_10157 : modelRow 10157 = 1 := by decide
theorem modelrow_agrees_10157 : modelRow 10157 = row 10157 := model_count_10157.trans row_count_10157.symm
def row_10158 : RowCertificate 10158 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10158 1) := by change ¬ Nat.Prime 412760173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58965739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10158 2) := by change ¬ Nat.Prime 412739857; exact PrattCertificates.composite_of_product (factor := 1109) (cofactor := 372173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10158 3) := by change ¬ Nat.Prime 412719541; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31747657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10158 : row 10158 = 0 := (row_10158.checked).trans rfl
theorem model_count_10158 : modelRow 10158 = 0 := by decide
theorem modelrow_agrees_10158 : modelRow 10158 = row 10158 := model_count_10158.trans row_count_10158.symm
def row_10159 : RowCertificate 10159 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10159 1) := by change ¬ Nat.Prime 412841443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58977349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10159 2) := by change ¬ Nat.Prime 412821125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82564225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10159 3) := by change ¬ Nat.Prime 412800807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137600269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10159 : row 10159 = 0 := (row_10159.checked).trans rfl
theorem model_count_10159 : modelRow 10159 = 0 := by decide
theorem modelrow_agrees_10159 : modelRow 10159 = row 10159 := model_count_10159.trans row_count_10159.symm
def row_10160 : RowCertificate 10160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10160 1) := by change ¬ Nat.Prime 412922721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137640907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10160 2) := by change ¬ Nat.Prime 412902401; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1319177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10160 3) := by change ¬ Nat.Prime 412882081; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3787909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10160 : row 10160 = 0 := (row_10160.checked).trans rfl
theorem model_count_10160 : modelRow 10160 = 0 := by decide
theorem modelrow_agrees_10160 : modelRow 10160 = row 10160 := model_count_10160.trans row_count_10160.symm
def row_10161 : RowCertificate 10161 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10161 1) := by change ¬ Nat.Prime 413004007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31769539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10161 2) := by change ¬ Nat.Prime 412983685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82596737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10161 3) := by change ¬ Nat.Prime 412963363; exact PrattCertificates.composite_of_product (factor := 5449) (cofactor := 75787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10161 : row 10161 = 0 := (row_10161.checked).trans rfl
theorem model_count_10161 : modelRow 10161 = 0 := by decide
theorem modelrow_agrees_10161 : modelRow 10161 = row 10161 := model_count_10161.trans row_count_10161.symm
def row_10162 : RowCertificate 10162 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10162 1) := by change Nat.Prime 413085301; exact ProjectEulerSolutions.P58PrimeNodes.p_413085301.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10162 2) := by change ¬ Nat.Prime 413064977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31774229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10162 3) := by change ¬ Nat.Prime 413044653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137681551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10162 : row 10162 = 1 := (row_10162.checked).trans rfl
theorem model_count_10162 : modelRow 10162 = 1 := by decide
theorem modelrow_agrees_10162 : modelRow 10162 = row 10162 := model_count_10162.trans row_count_10162.symm
def row_10163 : RowCertificate 10163 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10163 1) := by change ¬ Nat.Prime 413166603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137722201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10163 2) := by change ¬ Nat.Prime 413146277; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 4642093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10163 3) := by change ¬ Nat.Prime 413125951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59017993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10163 : row 10163 = 0 := (row_10163.checked).trans rfl
theorem model_count_10163 : modelRow 10163 = 0 := by decide
theorem modelrow_agrees_10163 : modelRow 10163 = row 10163 := model_count_10163.trans row_count_10163.symm
def row_10164 : RowCertificate 10164 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10164 1) := by change ¬ Nat.Prime 413247913; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31788301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10164 2) := by change ¬ Nat.Prime 413227585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82645517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10164 3) := by change ¬ Nat.Prime 413207257; exact PrattCertificates.composite_of_product (factor := 1171) (cofactor := 352867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10164 : row 10164 = 0 := (row_10164.checked).trans rfl
theorem model_count_10164 : modelRow 10164 = 0 := by decide
theorem modelrow_agrees_10164 : modelRow 10164 = row 10164 := model_count_10164.trans row_count_10164.symm
def row_10165 : RowCertificate 10165 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10165 1) := by change ¬ Nat.Prime 413329231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59047033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10165 2) := by change ¬ Nat.Prime 413308901; exact PrattCertificates.composite_of_product (factor := 6269) (cofactor := 65929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10165 3) := by change ¬ Nat.Prime 413288571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137762857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10165 : row 10165 = 0 := (row_10165.checked).trans rfl
theorem model_count_10165 : modelRow 10165 = 0 := by decide
theorem modelrow_agrees_10165 : modelRow 10165 = row 10165 := model_count_10165.trans row_count_10165.symm
def row_10166 : RowCertificate 10166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10166 1) := by change ¬ Nat.Prime 413410557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137803519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10166 2) := by change ¬ Nat.Prime 413390225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82678045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10166 3) := by change ¬ Nat.Prime 413369893; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 1460671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10166 : row 10166 = 0 := (row_10166.checked).trans rfl
theorem model_count_10166 : modelRow 10166 = 0 := by decide
theorem modelrow_agrees_10166 : modelRow 10166 = row 10166 := model_count_10166.trans row_count_10166.symm
def row_10167 : RowCertificate 10167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10167 1) := by change ¬ Nat.Prime 413491891; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4262803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10167 2) := by change ¬ Nat.Prime 413471557; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 2390009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10167 3) := by change ¬ Nat.Prime 413451223; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 583147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10167 : row 10167 = 0 := (row_10167.checked).trans rfl
theorem model_count_10167 : modelRow 10167 = 0 := by decide
theorem modelrow_agrees_10167 : modelRow 10167 = row 10167 := model_count_10167.trans row_count_10167.symm
def row_10168 : RowCertificate 10168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10168 1) := by change ¬ Nat.Prime 413573233; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2975347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10168 2) := by change ¬ Nat.Prime 413552897; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24326641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10168 3) := by change ¬ Nat.Prime 413532561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137844187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10168 : row 10168 = 0 := (row_10168.checked).trans rfl
theorem model_count_10168 : modelRow 10168 = 0 := by decide
theorem modelrow_agrees_10168 : modelRow 10168 = row 10168 := model_count_10168.trans row_count_10168.symm
def row_10169 : RowCertificate 10169 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10169 1) := by change ¬ Nat.Prime 413654583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137884861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10169 2) := by change ¬ Nat.Prime 413634245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82726849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10169 3) := by change ¬ Nat.Prime 413613907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59087701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10169 : row 10169 = 0 := (row_10169.checked).trans rfl
theorem model_count_10169 : modelRow 10169 = 0 := by decide
theorem modelrow_agrees_10169 : modelRow 10169 = row 10169 := model_count_10169.trans row_count_10169.symm
def row_10170 : RowCertificate 10170 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10170 1) := by change Nat.Prime 413735941; exact ProjectEulerSolutions.P58PrimeNodes.p_413735941.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10170 2) := by change ¬ Nat.Prime 413715601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31824277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10170 3) := by change ¬ Nat.Prime 413695261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59099323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10170 : row 10170 = 1 := (row_10170.checked).trans rfl
theorem model_count_10170 : modelRow 10170 = 1 := by decide
theorem modelrow_agrees_10170 : modelRow 10170 = row 10170 := model_count_10170.trans row_count_10170.symm
def row_10171 : RowCertificate 10171 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10171 1) := by change ¬ Nat.Prime 413817307; exact PrattCertificates.composite_of_product (factor := 1471) (cofactor := 281317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10171 2) := by change ¬ Nat.Prime 413796965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82759393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10171 3) := by change ¬ Nat.Prime 413776623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137925541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10171 : row 10171 = 0 := (row_10171.checked).trans rfl
theorem model_count_10171 : modelRow 10171 = 0 := by decide
theorem modelrow_agrees_10171 : modelRow 10171 = row 10171 := model_count_10171.trans row_count_10171.symm
def row_10172 : RowCertificate 10172 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10172 1) := by change ¬ Nat.Prime 413898681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 137966227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10172 2) := by change ¬ Nat.Prime 413878337; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11185901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10172 3) := by change ¬ Nat.Prime 413857993; exact PrattCertificates.composite_of_product (factor := 2383) (cofactor := 173671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10172 : row 10172 = 0 := (row_10172.checked).trans rfl
theorem model_count_10172 : modelRow 10172 = 0 := by decide
theorem modelrow_agrees_10172 : modelRow 10172 = row 10172 := model_count_10172.trans row_count_10172.symm
def row_10173 : RowCertificate 10173 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10173 1) := by change ¬ Nat.Prime 413980063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59140009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10173 2) := by change ¬ Nat.Prime 413959717; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14274473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10173 3) := by change ¬ Nat.Prime 413939371; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9626497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10173 : row 10173 = 0 := (row_10173.checked).trans rfl
theorem model_count_10173 : modelRow 10173 = 0 := by decide
theorem modelrow_agrees_10173 : modelRow 10173 = row 10173 := model_count_10173.trans row_count_10173.symm
def row_10174 : RowCertificate 10174 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10174 1) := by change ¬ Nat.Prime 414061453; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31850881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10174 2) := by change ¬ Nat.Prime 414041105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82808221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10174 3) := by change ¬ Nat.Prime 414020757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138006919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10174 : row 10174 = 0 := (row_10174.checked).trans rfl
theorem model_count_10174 : modelRow 10174 = 0 := by decide
theorem modelrow_agrees_10174 : modelRow 10174 = row 10174 := model_count_10174.trans row_count_10174.symm
def row_10175 : RowCertificate 10175 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10175 1) := by change ¬ Nat.Prime 414142851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138047617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10175 2) := by change ¬ Nat.Prime 414122501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31855577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10175 3) := by change ¬ Nat.Prime 414102151; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2145607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10175 : row 10175 = 0 := (row_10175.checked).trans rfl
theorem model_count_10175 : modelRow 10175 = 0 := by decide
theorem modelrow_agrees_10175 : modelRow 10175 = row 10175 := model_count_10175.trans row_count_10175.symm
def row_10176 : RowCertificate 10176 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10176 1) := by change ¬ Nat.Prime 414224257; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2743207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10176 2) := by change ¬ Nat.Prime 414203905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82840781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10176 3) := by change ¬ Nat.Prime 414183553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59169079) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10176 : row 10176 = 0 := (row_10176.checked).trans rfl
theorem model_count_10176 : modelRow 10176 = 0 := by decide
theorem modelrow_agrees_10176 : modelRow 10176 = row 10176 := model_count_10176.trans row_count_10176.symm
def row_10177 : RowCertificate 10177 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10177 1) := by change ¬ Nat.Prime 414305671; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31869667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10177 2) := by change Nat.Prime 414285317; exact ProjectEulerSolutions.P58PrimeNodes.p_414285317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10177 3) := by change ¬ Nat.Prime 414264963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138088321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10177 : row 10177 = 1 := (row_10177.checked).trans rfl
theorem model_count_10177 : modelRow 10177 = 1 := by decide
theorem modelrow_agrees_10177 : modelRow 10177 = row 10177 := model_count_10177.trans row_count_10177.symm
def row_10178 : RowCertificate 10178 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10178 1) := by change ¬ Nat.Prime 414387093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138129031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10178 2) := by change ¬ Nat.Prime 414366737; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11199101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10178 3) := by change ¬ Nat.Prime 414346381; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 482359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10178 : row 10178 = 0 := (row_10178.checked).trans rfl
theorem model_count_10178 : modelRow 10178 = 0 := by decide
theorem modelrow_agrees_10178 : modelRow 10178 = row 10178 := model_count_10178.trans row_count_10178.symm
def row_10179 : RowCertificate 10179 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10179 1) := by change ¬ Nat.Prime 414468523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59209789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10179 2) := by change ¬ Nat.Prime 414448165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82889633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10179 3) := by change ¬ Nat.Prime 414427807; exact PrattCertificates.composite_of_product (factor := 2551) (cofactor := 162457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10179 : row 10179 = 0 := (row_10179.checked).trans rfl
theorem model_count_10179 : modelRow 10179 = 0 := by decide
theorem modelrow_agrees_10179 : modelRow 10179 = row 10179 := model_count_10179.trans row_count_10179.symm
def row_10180 : RowCertificate 10180 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10180 1) := by change ¬ Nat.Prime 414549961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59221423) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10180 2) := by change ¬ Nat.Prime 414529601; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 1779097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10180 3) := by change ¬ Nat.Prime 414509241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138169747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10180 : row 10180 = 0 := (row_10180.checked).trans rfl
theorem model_count_10180 : modelRow 10180 = 0 := by decide
theorem modelrow_agrees_10180 : modelRow 10180 = row 10180 := model_count_10180.trans row_count_10180.symm
def row_10181 : RowCertificate 10181 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10181 1) := by change ¬ Nat.Prime 414631407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138210469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10181 2) := by change ¬ Nat.Prime 414611045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82922209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10181 3) := by change ¬ Nat.Prime 414590683; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31891591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10181 : row 10181 = 0 := (row_10181.checked).trans rfl
theorem model_count_10181 : modelRow 10181 = 0 := by decide
theorem modelrow_agrees_10181 : modelRow 10181 = row 10181 := model_count_10181.trans row_count_10181.symm
def row_10182 : RowCertificate 10182 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10182 1) := by change ¬ Nat.Prime 414712861; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1188289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10182 2) := by change Nat.Prime 414692497; exact ProjectEulerSolutions.P58PrimeNodes.p_414692497.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10182 3) := by change Nat.Prime 414672133; exact ProjectEulerSolutions.P58PrimeNodes.p_414672133.prime
    exact decide_eq_true prime
theorem row_count_10182 : row 10182 = 2 := (row_10182.checked).trans rfl
theorem model_count_10182 : modelRow 10182 = 2 := by decide
theorem modelrow_agrees_10182 : modelRow 10182 = row 10182 := model_count_10182.trans row_count_10182.symm
def row_10183 : RowCertificate 10183 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10183 1) := by change Nat.Prime 414794323; exact ProjectEulerSolutions.P58PrimeNodes.p_414794323.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10183 2) := by change ¬ Nat.Prime 414773957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31905689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10183 3) := by change ¬ Nat.Prime 414753591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138251197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10183 : row 10183 = 1 := (row_10183.checked).trans rfl
theorem model_count_10183 : modelRow 10183 = 1 := by decide
theorem modelrow_agrees_10183 : modelRow 10183 = row 10183 := model_count_10183.trans row_count_10183.symm
def row_10184 : RowCertificate 10184 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10184 1) := by change ¬ Nat.Prime 414875793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138291931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10184 2) := by change ¬ Nat.Prime 414855425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82971085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10184 3) := by change ¬ Nat.Prime 414835057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59262151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10184 : row 10184 = 0 := (row_10184.checked).trans rfl
theorem model_count_10184 : modelRow 10184 = 0 := by decide
theorem modelrow_agrees_10184 : modelRow 10184 = row 10184 := model_count_10184.trans row_count_10184.symm
def row_10185 : RowCertificate 10185 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10185 1) := by change Nat.Prime 414957271; exact ProjectEulerSolutions.P58PrimeNodes.p_414957271.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10185 2) := by change ¬ Nat.Prime 414936901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24408053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10185 3) := by change Nat.Prime 414916531; exact ProjectEulerSolutions.P58PrimeNodes.p_414916531.prime
    exact decide_eq_true prime
theorem row_count_10185 : row 10185 = 2 := (row_10185.checked).trans rfl
theorem model_count_10185 : modelRow 10185 = 2 := by decide
theorem modelrow_agrees_10185 : modelRow 10185 = row 10185 := model_count_10185.trans row_count_10185.symm
def row_10186 : RowCertificate 10186 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10186 1) := by change ¬ Nat.Prime 415038757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59291251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10186 2) := by change ¬ Nat.Prime 415018385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83003677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10186 3) := by change ¬ Nat.Prime 414998013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138332671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10186 : row 10186 = 0 := (row_10186.checked).trans rfl
theorem model_count_10186 : modelRow 10186 = 0 := by decide
theorem modelrow_agrees_10186 : modelRow 10186 = row 10186 := model_count_10186.trans row_count_10186.symm
def row_10187 : RowCertificate 10187 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10187 1) := by change ¬ Nat.Prime 415120251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138373417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10187 2) := by change Nat.Prime 415099877; exact ProjectEulerSolutions.P58PrimeNodes.p_415099877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10187 3) := by change ¬ Nat.Prime 415079503; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2150671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10187 : row 10187 = 1 := (row_10187.checked).trans rfl
theorem model_count_10187 : modelRow 10187 = 1 := by decide
theorem modelrow_agrees_10187 : modelRow 10187 = row 10187 := model_count_10187.trans row_count_10187.symm
def row_10188 : RowCertificate 10188 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10188 1) := by change ¬ Nat.Prime 415201753; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11221669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10188 2) := by change ¬ Nat.Prime 415181377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31937029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10188 3) := by change ¬ Nat.Prime 415161001; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21850579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10188 : row 10188 = 0 := (row_10188.checked).trans rfl
theorem model_count_10188 : modelRow 10188 = 0 := by decide
theorem modelrow_agrees_10188 : modelRow 10188 = row 10188 := model_count_10188.trans row_count_10188.symm
def row_10189 : RowCertificate 10189 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10189 1) := by change Nat.Prime 415283263; exact ProjectEulerSolutions.P58PrimeNodes.p_415283263.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10189 2) := by change ¬ Nat.Prime 415262885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83052577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10189 3) := by change ¬ Nat.Prime 415242507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138414169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10189 : row 10189 = 1 := (row_10189.checked).trans rfl
theorem model_count_10189 : modelRow 10189 = 1 := by decide
theorem modelrow_agrees_10189 : modelRow 10189 = row 10189 := model_count_10189.trans row_count_10189.symm
def row_10190 : RowCertificate 10190 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10190 1) := by change ¬ Nat.Prime 415364781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138454927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10190 2) := by change Nat.Prime 415344401; exact ProjectEulerSolutions.P58PrimeNodes.p_415344401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10190 3) := by change ¬ Nat.Prime 415324021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59332003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10190 : row 10190 = 1 := (row_10190.checked).trans rfl
theorem model_count_10190 : modelRow 10190 = 1 := by decide
theorem modelrow_agrees_10190 : modelRow 10190 = row 10190 := model_count_10190.trans row_count_10190.symm
def row_10191 : RowCertificate 10191 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10191 1) := by change ¬ Nat.Prime 415446307; exact PrattCertificates.composite_of_product (factor := 2677) (cofactor := 155191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10191 2) := by change ¬ Nat.Prime 415425925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83085185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10191 3) := by change ¬ Nat.Prime 415405543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59343649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10191 : row 10191 = 0 := (row_10191.checked).trans rfl
theorem model_count_10191 : modelRow 10191 = 0 := by decide
theorem modelrow_agrees_10191 : modelRow 10191 = row 10191 := model_count_10191.trans row_count_10191.symm
def row_10192 : RowCertificate 10192 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10192 1) := by change Nat.Prime 415527841; exact ProjectEulerSolutions.P58PrimeNodes.p_415527841.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10192 2) := by change Nat.Prime 415507457; exact ProjectEulerSolutions.P58PrimeNodes.p_415507457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10192 3) := by change ¬ Nat.Prime 415487073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138495691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10192 : row 10192 = 2 := (row_10192.checked).trans rfl
theorem model_count_10192 : modelRow 10192 = 2 := by decide
theorem modelrow_agrees_10192 : modelRow 10192 = row 10192 := model_count_10192.trans row_count_10192.symm
def row_10193 : RowCertificate 10193 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10193 1) := by change ¬ Nat.Prime 415609383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138536461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10193 2) := by change ¬ Nat.Prime 415588997; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10136317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10193 3) := by change Nat.Prime 415568611; exact ProjectEulerSolutions.P58PrimeNodes.p_415568611.prime
    exact decide_eq_true prime
theorem row_count_10193 : row 10193 = 1 := (row_10193.checked).trans rfl
theorem model_count_10193 : modelRow 10193 = 1 := by decide
theorem modelrow_agrees_10193 : modelRow 10193 = row 10193 := model_count_10193.trans row_count_10193.symm
def row_10194 : RowCertificate 10194 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10194 1) := by change ¬ Nat.Prime 415690933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59384419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10194 2) := by change ¬ Nat.Prime 415670545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83134109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10194 3) := by change ¬ Nat.Prime 415650157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31973089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10194 : row 10194 = 0 := (row_10194.checked).trans rfl
theorem model_count_10194 : modelRow 10194 = 0 := by decide
theorem modelrow_agrees_10194 : modelRow 10194 = row 10194 := model_count_10194.trans row_count_10194.symm
def row_10195 : RowCertificate 10195 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10195 1) := by change ¬ Nat.Prime 415772491; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2991169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10195 2) := by change ¬ Nat.Prime 415752101; exact PrattCertificates.composite_of_product (factor := 10597) (cofactor := 39233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10195 3) := by change ¬ Nat.Prime 415731711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138577237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10195 : row 10195 = 0 := (row_10195.checked).trans rfl
theorem model_count_10195 : modelRow 10195 = 0 := by decide
theorem modelrow_agrees_10195 : modelRow 10195 = row 10195 := model_count_10195.trans row_count_10195.symm
def row_10196 : RowCertificate 10196 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10196 1) := by change ¬ Nat.Prime 415854057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138618019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10196 2) := by change ¬ Nat.Prime 415833665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83166733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10196 3) := by change ¬ Nat.Prime 415813273; exact PrattCertificates.composite_of_product (factor := 9967) (cofactor := 41719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10196 : row 10196 = 0 := (row_10196.checked).trans rfl
theorem model_count_10196 : modelRow 10196 = 0 := by decide
theorem modelrow_agrees_10196 : modelRow 10196 = row 10196 := model_count_10196.trans row_count_10196.symm
def row_10197 : RowCertificate 10197 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10197 1) := by change ¬ Nat.Prime 415935631; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21891349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10197 2) := by change ¬ Nat.Prime 415915237; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5697469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10197 3) := by change ¬ Nat.Prime 415894843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59413549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10197 : row 10197 = 0 := (row_10197.checked).trans rfl
theorem model_count_10197 : modelRow 10197 = 0 := by decide
theorem modelrow_agrees_10197 : modelRow 10197 = row 10197 := model_count_10197.trans row_count_10197.symm
def row_10198 : RowCertificate 10198 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10198 1) := by change ¬ Nat.Prime 416017213; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 1017157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10198 2) := by change ¬ Nat.Prime 415996817; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24470401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10198 3) := by change ¬ Nat.Prime 415976421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138658807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10198 : row 10198 = 0 := (row_10198.checked).trans rfl
theorem model_count_10198 : modelRow 10198 = 0 := by decide
theorem modelrow_agrees_10198 : modelRow 10198 = row 10198 := model_count_10198.trans row_count_10198.symm
def row_10199 : RowCertificate 10199 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10199 1) := by change ¬ Nat.Prime 416098803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138699601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10199 2) := by change ¬ Nat.Prime 416078405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83215681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10199 3) := by change ¬ Nat.Prime 416058007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11244811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10199 : row 10199 = 0 := (row_10199.checked).trans rfl
theorem model_count_10199 : modelRow 10199 = 0 := by decide
theorem modelrow_agrees_10199 : modelRow 10199 = row 10199 := model_count_10199.trans row_count_10199.symm
def row_10200 : RowCertificate 10200 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10200 1) := by change ¬ Nat.Prime 416180401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59454343) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10200 2) := by change Nat.Prime 416160001; exact ProjectEulerSolutions.P58PrimeNodes.p_416160001.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10200 3) := by change Nat.Prime 416139601; exact ProjectEulerSolutions.P58PrimeNodes.p_416139601.prime
    exact decide_eq_true prime
theorem row_count_10200 : row 10200 = 2 := (row_10200.checked).trans rfl
theorem model_count_10200 : modelRow 10200 = 2 := by decide
theorem modelrow_agrees_10200 : modelRow 10200 = row 10200 := model_count_10200.trans row_count_10200.symm
def row_10201 : RowCertificate 10201 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10201 1) := by change ¬ Nat.Prime 416262007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59466001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10201 2) := by change ¬ Nat.Prime 416241605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83248321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10201 3) := by change ¬ Nat.Prime 416221203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138740401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10201 : row 10201 = 0 := (row_10201.checked).trans rfl
theorem model_count_10201 : modelRow 10201 = 0 := by decide
theorem modelrow_agrees_10201 : modelRow 10201 = row 10201 := model_count_10201.trans row_count_10201.symm
def row_10202 : RowCertificate 10202 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10202 1) := by change ¬ Nat.Prime 416343621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138781207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10202 2) := by change ¬ Nat.Prime 416323217; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24489601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10202 3) := by change ¬ Nat.Prime 416302813; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13429123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10202 : row 10202 = 0 := (row_10202.checked).trans rfl
theorem model_count_10202 : modelRow 10202 = 0 := by decide
theorem modelrow_agrees_10202 : modelRow 10202 = row 10202 := model_count_10202.trans row_count_10202.symm
def row_10203 : RowCertificate 10203 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10203 1) := by change ¬ Nat.Prime 416425243; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32032711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10203 2) := by change Nat.Prime 416404837; exact ProjectEulerSolutions.P58PrimeNodes.p_416404837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10203 3) := by change ¬ Nat.Prime 416384431; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 6214693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10203 : row 10203 = 1 := (row_10203.checked).trans rfl
theorem model_count_10203 : modelRow 10203 = 1 := by decide
theorem modelrow_agrees_10203 : modelRow 10203 = row 10203 := model_count_10203.trans row_count_10203.symm
def row_10204 : RowCertificate 10204 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10204 1) := by change ¬ Nat.Prime 416506873; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 721849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10204 2) := by change ¬ Nat.Prime 416486465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83297293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10204 3) := by change ¬ Nat.Prime 416466057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138822019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10204 : row 10204 = 0 := (row_10204.checked).trans rfl
theorem model_count_10204 : modelRow 10204 = 0 := by decide
theorem modelrow_agrees_10204 : modelRow 10204 = row 10204 := model_count_10204.trans row_count_10204.symm
def row_10205 : RowCertificate 10205 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10205 1) := by change ¬ Nat.Prime 416588511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138862837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10205 2) := by change ¬ Nat.Prime 416568101; exact PrattCertificates.composite_of_product (factor := 1861) (cofactor := 223841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10205 3) := by change ¬ Nat.Prime 416547691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59506813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10205 : row 10205 = 0 := (row_10205.checked).trans rfl
theorem model_count_10205 : modelRow 10205 = 0 := by decide
theorem modelrow_agrees_10205 : modelRow 10205 = row 10205 := model_count_10205.trans row_count_10205.symm
def row_10206 : RowCertificate 10206 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10206 1) := by change ¬ Nat.Prime 416670157; exact PrattCertificates.composite_of_product (factor := 1549) (cofactor := 268993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10206 2) := by change ¬ Nat.Prime 416649745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83329949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10206 3) := by change ¬ Nat.Prime 416629333; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 550369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10206 : row 10206 = 0 := (row_10206.checked).trans rfl
theorem model_count_10206 : modelRow 10206 = 0 := by decide
theorem modelrow_agrees_10206 : modelRow 10206 = row 10206 := model_count_10206.trans row_count_10206.symm
def row_10207 : RowCertificate 10207 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10207 1) := by change ¬ Nat.Prime 416751811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59535973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10207 2) := by change Nat.Prime 416731397; exact ProjectEulerSolutions.P58PrimeNodes.p_416731397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10207 3) := by change ¬ Nat.Prime 416710983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138903661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10207 : row 10207 = 1 := (row_10207.checked).trans rfl
theorem model_count_10207 : modelRow 10207 = 1 := by decide
theorem modelrow_agrees_10207 : modelRow 10207 = row 10207 := model_count_10207.trans row_count_10207.symm
def row_10208 : RowCertificate 10208 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10208 1) := by change ¬ Nat.Prime 416833473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138944491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10208 2) := by change ¬ Nat.Prime 416813057; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3823973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10208 3) := by change ¬ Nat.Prime 416792641; exact PrattCertificates.composite_of_product (factor := 17431) (cofactor := 23911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10208 : row 10208 = 0 := (row_10208.checked).trans rfl
theorem model_count_10208 : modelRow 10208 = 0 := by decide
theorem modelrow_agrees_10208 : modelRow 10208 = row 10208 := model_count_10208.trans row_count_10208.symm
def row_10209 : RowCertificate 10209 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10209 1) := by change ¬ Nat.Prime 416915143; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9695701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10209 2) := by change ¬ Nat.Prime 416894725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83378945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10209 3) := by change ¬ Nat.Prime 416874307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21940753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10209 : row 10209 = 0 := (row_10209.checked).trans rfl
theorem model_count_10209 : modelRow 10209 = 0 := by decide
theorem modelrow_agrees_10209 : modelRow 10209 = row 10209 := model_count_10209.trans row_count_10209.symm
def row_10210 : RowCertificate 10210 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10210 1) := by change ¬ Nat.Prime 416996821; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 1473487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10210 2) := by change ¬ Nat.Prime 416976401; exact PrattCertificates.composite_of_product (factor := 3917) (cofactor := 106453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10210 3) := by change ¬ Nat.Prime 416955981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138985327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10210 : row 10210 = 0 := (row_10210.checked).trans rfl
theorem model_count_10210 : modelRow 10210 = 0 := by decide
theorem modelrow_agrees_10210 : modelRow 10210 = row 10210 := model_count_10210.trans row_count_10210.symm
def row_10211 : RowCertificate 10211 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10211 1) := by change ¬ Nat.Prime 417078507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139026169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10211 2) := by change ¬ Nat.Prime 417058085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83411617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10211 3) := by change ¬ Nat.Prime 417037663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59576809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10211 : row 10211 = 0 := (row_10211.checked).trans rfl
theorem model_count_10211 : modelRow 10211 = 0 := by decide
theorem modelrow_agrees_10211 : modelRow 10211 = row 10211 := model_count_10211.trans row_count_10211.symm
def row_10212 : RowCertificate 10212 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10212 1) := by change Nat.Prime 417160201; exact ProjectEulerSolutions.P58PrimeNodes.p_417160201.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10212 2) := by change ¬ Nat.Prime 417139777; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6838357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10212 3) := by change ¬ Nat.Prime 417119353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59588479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10212 : row 10212 = 1 := (row_10212.checked).trans rfl
theorem model_count_10212 : modelRow 10212 = 1 := by decide
theorem modelrow_agrees_10212 : modelRow 10212 = row 10212 := model_count_10212.trans row_count_10212.symm
def row_10213 : RowCertificate 10213 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10213 1) := by change ¬ Nat.Prime 417241903; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32095531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10213 2) := by change ¬ Nat.Prime 417221477; exact PrattCertificates.composite_of_product (factor := 3989) (cofactor := 104593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10213 3) := by change ¬ Nat.Prime 417201051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139067017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10213 : row 10213 = 0 := (row_10213.checked).trans rfl
theorem model_count_10213 : modelRow 10213 = 0 := by decide
theorem modelrow_agrees_10213 : modelRow 10213 = row 10213 := model_count_10213.trans row_count_10213.symm
def row_10214 : RowCertificate 10214 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10214 1) := by change ¬ Nat.Prime 417323613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139107871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10214 2) := by change ¬ Nat.Prime 417303185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83460637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10214 3) := by change ¬ Nat.Prime 417282757; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3285691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10214 : row 10214 = 0 := (row_10214.checked).trans rfl
theorem model_count_10214 : modelRow 10214 = 0 := by decide
theorem modelrow_agrees_10214 : modelRow 10214 = row 10214 := model_count_10214.trans row_count_10214.symm
def row_10215 : RowCertificate 10215 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10215 1) := by change ¬ Nat.Prime 417405331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59629333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10215 2) := by change ¬ Nat.Prime 417384901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24552053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10215 3) := by change ¬ Nat.Prime 417364471; exact PrattCertificates.composite_of_product (factor := 6883) (cofactor := 60637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10215 : row 10215 = 0 := (row_10215.checked).trans rfl
theorem model_count_10215 : modelRow 10215 = 0 := by decide
theorem modelrow_agrees_10215 : modelRow 10215 = row 10215 := model_count_10215.trans row_count_10215.symm
def row_10216 : RowCertificate 10216 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10216 1) := by change ¬ Nat.Prime 417487057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32114389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10216 2) := by change ¬ Nat.Prime 417466625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83493325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10216 3) := by change ¬ Nat.Prime 417446193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139148731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10216 : row 10216 = 0 := (row_10216.checked).trans rfl
theorem model_count_10216 : modelRow 10216 = 0 := by decide
theorem modelrow_agrees_10216 : modelRow 10216 = row 10216 := model_count_10216.trans row_count_10216.symm
def row_10217 : RowCertificate 10217 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10217 1) := by change ¬ Nat.Prime 417568791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139189597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10217 2) := by change ¬ Nat.Prime 417548357; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2306897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10217 3) := by change ¬ Nat.Prime 417527923; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2561521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10217 : row 10217 = 0 := (row_10217.checked).trans rfl
theorem model_count_10217 : modelRow 10217 = 0 := by decide
theorem modelrow_agrees_10217 : modelRow 10217 = row 10217 := model_count_10217.trans row_count_10217.symm
def row_10218 : RowCertificate 10218 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10218 1) := by change ¬ Nat.Prime 417650533; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21981607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10218 2) := by change ¬ Nat.Prime 417630097; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 4692473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10218 3) := by change ¬ Nat.Prime 417609661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59658523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10218 : row 10218 = 0 := (row_10218.checked).trans rfl
theorem model_count_10218 : modelRow 10218 = 0 := by decide
theorem modelrow_agrees_10218 : modelRow 10218 = row 10218 := model_count_10218.trans row_count_10218.symm
def row_10219 : RowCertificate 10219 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10219 1) := by change ¬ Nat.Prime 417732283; exact PrattCertificates.composite_of_product (factor := 13729) (cofactor := 30427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10219 2) := by change ¬ Nat.Prime 417711845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83542369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10219 3) := by change ¬ Nat.Prime 417691407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139230469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10219 : row 10219 = 0 := (row_10219.checked).trans rfl
theorem model_count_10219 : modelRow 10219 = 0 := by decide
theorem modelrow_agrees_10219 : modelRow 10219 = row 10219 := model_count_10219.trans row_count_10219.symm
def row_10220 : RowCertificate 10220 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10220 1) := by change ¬ Nat.Prime 417814041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139271347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10220 2) := by change Nat.Prime 417793601; exact ProjectEulerSolutions.P58PrimeNodes.p_417793601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10220 3) := by change ¬ Nat.Prime 417773161; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32136397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10220 : row 10220 = 1 := (row_10220.checked).trans rfl
theorem model_count_10220 : modelRow 10220 = 1 := by decide
theorem modelrow_agrees_10220 : modelRow 10220 = row 10220 := model_count_10220.trans row_count_10220.symm
def row_10221 : RowCertificate 10221 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10221 1) := by change ¬ Nat.Prime 417895807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59699401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10221 2) := by change ¬ Nat.Prime 417875365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83575073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10221 3) := by change ¬ Nat.Prime 417854923; exact PrattCertificates.composite_of_product (factor := 10993) (cofactor := 38011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10221 : row 10221 = 0 := (row_10221.checked).trans rfl
theorem model_count_10221 : modelRow 10221 = 0 := by decide
theorem modelrow_agrees_10221 : modelRow 10221 = row 10221 := model_count_10221.trans row_count_10221.symm
def row_10222 : RowCertificate 10222 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10222 1) := by change ¬ Nat.Prime 417977581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59711083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10222 2) := by change ¬ Nat.Prime 417957137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32150549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10222 3) := by change ¬ Nat.Prime 417936693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139312231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10222 : row 10222 = 0 := (row_10222.checked).trans rfl
theorem model_count_10222 : modelRow 10222 = 0 := by decide
theorem modelrow_agrees_10222 : modelRow 10222 = row 10222 := model_count_10222.trans row_count_10222.symm
def row_10223 : RowCertificate 10223 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10223 1) := by change ¬ Nat.Prime 418059363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139353121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10223 2) := by change ¬ Nat.Prime 418038917; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6853097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10223 3) := by change ¬ Nat.Prime 418018471; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32155267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10223 : row 10223 = 0 := (row_10223.checked).trans rfl
theorem model_count_10223 : modelRow 10223 = 0 := by decide
theorem modelrow_agrees_10223 : modelRow 10223 = row 10223 := model_count_10223.trans row_count_10223.symm
def row_10224 : RowCertificate 10224 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10224 1) := by change ¬ Nat.Prime 418141153; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6854773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10224 2) := by change ¬ Nat.Prime 418120705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83624141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10224 3) := by change Nat.Prime 418100257; exact ProjectEulerSolutions.P58PrimeNodes.p_418100257.prime
    exact decide_eq_true prime
theorem row_count_10224 : row 10224 = 1 := (row_10224.checked).trans rfl
theorem model_count_10224 : modelRow 10224 = 1 := by decide
theorem modelrow_agrees_10224 : modelRow 10224 = row 10224 := model_count_10224.trans row_count_10224.symm
def row_10225 : RowCertificate 10225 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10225 1) := by change ¬ Nat.Prime 418222951; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11303323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10225 2) := by change ¬ Nat.Prime 418202501; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10200061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10225 3) := by change ¬ Nat.Prime 418182051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139394017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10225 : row 10225 = 0 := (row_10225.checked).trans rfl
theorem model_count_10225 : modelRow 10225 = 0 := by decide
theorem modelrow_agrees_10225 : modelRow 10225 = row 10225 := model_count_10225.trans row_count_10225.symm
def row_10226 : RowCertificate 10226 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10226 1) := by change ¬ Nat.Prime 418304757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139434919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10226 2) := by change ¬ Nat.Prime 418284305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83656861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10226 3) := by change ¬ Nat.Prime 418263853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59751979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10226 : row 10226 = 0 := (row_10226.checked).trans rfl
theorem model_count_10226 : modelRow 10226 = 0 := by decide
theorem modelrow_agrees_10226 : modelRow 10226 = row 10226 := model_count_10226.trans row_count_10226.symm
def row_10227 : RowCertificate 10227 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10227 1) := by change ¬ Nat.Prime 418386571; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13496341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10227 2) := by change ¬ Nat.Prime 418366117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32182009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10227 3) := by change ¬ Nat.Prime 418345663; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 477019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10227 : row 10227 = 0 := (row_10227.checked).trans rfl
theorem model_count_10227 : modelRow 10227 = 0 := by decide
theorem modelrow_agrees_10227 : modelRow 10227 = row 10227 := model_count_10227.trans row_count_10227.symm
def row_10228 : RowCertificate 10228 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10228 1) := by change ¬ Nat.Prime 418468393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59781199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10228 2) := by change ¬ Nat.Prime 418447937; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 915641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10228 3) := by change ¬ Nat.Prime 418427481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139475827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10228 : row 10228 = 0 := (row_10228.checked).trans rfl
theorem model_count_10228 : modelRow 10228 = 0 := by decide
theorem modelrow_agrees_10228 : modelRow 10228 = row 10228 := model_count_10228.trans row_count_10228.symm
def row_10229 : RowCertificate 10229 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10229 1) := by change ¬ Nat.Prime 418550223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139516741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10229 2) := by change ¬ Nat.Prime 418529765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83705953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10229 3) := by change ¬ Nat.Prime 418509307; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 800209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10229 : row 10229 = 0 := (row_10229.checked).trans rfl
theorem model_count_10229 : modelRow 10229 = 0 := by decide
theorem modelrow_agrees_10229 : modelRow 10229 = row 10229 := model_count_10229.trans row_count_10229.symm
def row_10230 : RowCertificate 10230 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10230 1) := by change Nat.Prime 418632061; exact ProjectEulerSolutions.P58PrimeNodes.p_418632061.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10230 2) := by change ¬ Nat.Prime 418611601; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 1489721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10230 3) := by change ¬ Nat.Prime 418591141; exact PrattCertificates.composite_of_product (factor := 3583) (cofactor := 116827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10230 : row 10230 = 1 := (row_10230.checked).trans rfl
theorem model_count_10230 : modelRow 10230 = 1 := by decide
theorem modelrow_agrees_10230 : modelRow 10230 = row 10230 := model_count_10230.trans row_count_10230.symm
def row_10231 : RowCertificate 10231 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10231 1) := by change Nat.Prime 418713907; exact ProjectEulerSolutions.P58PrimeNodes.p_418713907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10231 2) := by change ¬ Nat.Prime 418693445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83738689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10231 3) := by change ¬ Nat.Prime 418672983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139557661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10231 : row 10231 = 1 := (row_10231.checked).trans rfl
theorem model_count_10231 : modelRow 10231 = 1 := by decide
theorem modelrow_agrees_10231 : modelRow 10231 = row 10231 := model_count_10231.trans row_count_10231.symm
def row_10232 : RowCertificate 10232 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10232 1) := by change ¬ Nat.Prime 418795761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139598587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10232 2) := by change ¬ Nat.Prime 418775297; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24633841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10232 3) := by change ¬ Nat.Prime 418754833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59822119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10232 : row 10232 = 0 := (row_10232.checked).trans rfl
theorem model_count_10232 : modelRow 10232 = 0 := by decide
theorem modelrow_agrees_10232 : modelRow 10232 = row 10232 := model_count_10232.trans row_count_10232.symm
def row_10233 : RowCertificate 10233 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10233 1) := by change ¬ Nat.Prime 418877623; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3298249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10233 2) := by change ¬ Nat.Prime 418857157; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 348757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10233 3) := by change ¬ Nat.Prime 418836691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59833813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10233 : row 10233 = 0 := (row_10233.checked).trans rfl
theorem model_count_10233 : modelRow 10233 = 0 := by decide
theorem modelrow_agrees_10233 : modelRow 10233 = row 10233 := model_count_10233.trans row_count_10233.symm
def row_10234 : RowCertificate 10234 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10234 1) := by change ¬ Nat.Prime 418959493; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1200457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10234 2) := by change ¬ Nat.Prime 418939025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83787805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10234 3) := by change ¬ Nat.Prime 418918557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139639519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10234 : row 10234 = 0 := (row_10234.checked).trans rfl
theorem model_count_10234 : modelRow 10234 = 0 := by decide
theorem modelrow_agrees_10234 : modelRow 10234 = row 10234 := model_count_10234.trans row_count_10234.symm
def row_10235 : RowCertificate 10235 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10235 1) := by change ¬ Nat.Prime 419041371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139680457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10235 2) := by change ¬ Nat.Prime 419020901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32232377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10235 3) := by change ¬ Nat.Prime 419000431; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1738591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10235 : row 10235 = 0 := (row_10235.checked).trans rfl
theorem model_count_10235 : modelRow 10235 = 0 := by decide
theorem modelrow_agrees_10235 : modelRow 10235 = row 10235 := model_count_10235.trans row_count_10235.symm
def row_10236 : RowCertificate 10236 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10236 1) := by change ¬ Nat.Prime 419123257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59874751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10236 2) := by change ¬ Nat.Prime 419102785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83820557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10236 3) := by change ¬ Nat.Prime 419082313; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32237101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10236 : row 10236 = 0 := (row_10236.checked).trans rfl
theorem model_count_10236 : modelRow 10236 = 0 := by decide
theorem modelrow_agrees_10236 : modelRow 10236 = row 10236 := model_count_10236.trans row_count_10236.symm
def row_10237 : RowCertificate 10237 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10237 1) := by change ¬ Nat.Prime 419205151; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22063429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10237 2) := by change ¬ Nat.Prime 419184677; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 2127841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10237 3) := by change ¬ Nat.Prime 419164203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139721401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10237 : row 10237 = 0 := (row_10237.checked).trans rfl
theorem model_count_10237 : modelRow 10237 = 0 := by decide
theorem modelrow_agrees_10237 : modelRow 10237 = row 10237 := model_count_10237.trans row_count_10237.symm
def row_10238 : RowCertificate 10238 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10238 1) := by change ¬ Nat.Prime 419287053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139762351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10238 2) := by change Nat.Prime 419266577; exact ProjectEulerSolutions.P58PrimeNodes.p_419266577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10238 3) := by change ¬ Nat.Prime 419246101; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 3016159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10238 : row 10238 = 1 := (row_10238.checked).trans rfl
theorem model_count_10238 : modelRow 10238 = 1 := by decide
theorem modelrow_agrees_10238 : modelRow 10238 = row 10238 := model_count_10238.trans row_count_10238.symm
def row_10239 : RowCertificate 10239 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10239 1) := by change ¬ Nat.Prime 419368963; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32259151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10239 2) := by change ¬ Nat.Prime 419348485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83869697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10239 3) := by change ¬ Nat.Prime 419328007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59904001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10239 : row 10239 = 0 := (row_10239.checked).trans rfl
theorem model_count_10239 : modelRow 10239 = 0 := by decide
theorem modelrow_agrees_10239 : modelRow 10239 = row 10239 := model_count_10239.trans row_count_10239.symm
def row_10240 : RowCertificate 10240 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10240 1) := by change Nat.Prime 419450881; exact ProjectEulerSolutions.P58PrimeNodes.p_419450881.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10240 2) := by change ¬ Nat.Prime 419430401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32263877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10240 3) := by change ¬ Nat.Prime 419409921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139803307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10240 : row 10240 = 1 := (row_10240.checked).trans rfl
theorem model_count_10240 : modelRow 10240 = 1 := by decide
theorem modelrow_agrees_10240 : modelRow 10240 = row 10240 := model_count_10240.trans row_count_10240.symm
private def atomic_10113 : CountSegment 1 10 13120 10112 10113 4181 4181 := CountSegment.single (first := 10112) (initial := 4181) row_count_10113 (by decide)
private theorem atomic_10113_agreement : RowAgreement 10112 10113 := RowAgreement.single (first := 10112) modelrow_agrees_10113
private def atomic_10114 : CountSegment 1 10 13120 10113 10114 4181 4181 := CountSegment.single (first := 10113) (initial := 4181) row_count_10114 (by decide)
private theorem atomic_10114_agreement : RowAgreement 10113 10114 := RowAgreement.single (first := 10113) modelrow_agrees_10114
private def atomic_10115 : CountSegment 1 10 13120 10114 10115 4181 4181 := CountSegment.single (first := 10114) (initial := 4181) row_count_10115 (by decide)
private theorem atomic_10115_agreement : RowAgreement 10114 10115 := RowAgreement.single (first := 10114) modelrow_agrees_10115
private def atomic_10116 : CountSegment 1 10 13120 10115 10116 4181 4181 := CountSegment.single (first := 10115) (initial := 4181) row_count_10116 (by decide)
private theorem atomic_10116_agreement : RowAgreement 10115 10116 := RowAgreement.single (first := 10115) modelrow_agrees_10116
private def atomic_10117 : CountSegment 1 10 13120 10116 10117 4181 4181 := CountSegment.single (first := 10116) (initial := 4181) row_count_10117 (by decide)
private theorem atomic_10117_agreement : RowAgreement 10116 10117 := RowAgreement.single (first := 10116) modelrow_agrees_10117
private def atomic_10118 : CountSegment 1 10 13120 10117 10118 4181 4181 := CountSegment.single (first := 10117) (initial := 4181) row_count_10118 (by decide)
private theorem atomic_10118_agreement : RowAgreement 10117 10118 := RowAgreement.single (first := 10117) modelrow_agrees_10118
private def atomic_10119 : CountSegment 1 10 13120 10118 10119 4181 4181 := CountSegment.single (first := 10118) (initial := 4181) row_count_10119 (by decide)
private theorem atomic_10119_agreement : RowAgreement 10118 10119 := RowAgreement.single (first := 10118) modelrow_agrees_10119
private def atomic_10120 : CountSegment 1 10 13120 10119 10120 4181 4182 := CountSegment.single (first := 10119) (initial := 4181) row_count_10120 (by decide)
private theorem atomic_10120_agreement : RowAgreement 10119 10120 := RowAgreement.single (first := 10119) modelrow_agrees_10120
private def atomic_10121 : CountSegment 1 10 13120 10120 10121 4182 4182 := CountSegment.single (first := 10120) (initial := 4182) row_count_10121 (by decide)
private theorem atomic_10121_agreement : RowAgreement 10120 10121 := RowAgreement.single (first := 10120) modelrow_agrees_10121
private def atomic_10122 : CountSegment 1 10 13120 10121 10122 4182 4184 := CountSegment.single (first := 10121) (initial := 4182) row_count_10122 (by decide)
private theorem atomic_10122_agreement : RowAgreement 10121 10122 := RowAgreement.single (first := 10121) modelrow_agrees_10122
private def atomic_10123 : CountSegment 1 10 13120 10122 10123 4184 4184 := CountSegment.single (first := 10122) (initial := 4184) row_count_10123 (by decide)
private theorem atomic_10123_agreement : RowAgreement 10122 10123 := RowAgreement.single (first := 10122) modelrow_agrees_10123
private def atomic_10124 : CountSegment 1 10 13120 10123 10124 4184 4184 := CountSegment.single (first := 10123) (initial := 4184) row_count_10124 (by decide)
private theorem atomic_10124_agreement : RowAgreement 10123 10124 := RowAgreement.single (first := 10123) modelrow_agrees_10124
private def atomic_10125 : CountSegment 1 10 13120 10124 10125 4184 4185 := CountSegment.single (first := 10124) (initial := 4184) row_count_10125 (by decide)
private theorem atomic_10125_agreement : RowAgreement 10124 10125 := RowAgreement.single (first := 10124) modelrow_agrees_10125
private def atomic_10126 : CountSegment 1 10 13120 10125 10126 4185 4185 := CountSegment.single (first := 10125) (initial := 4185) row_count_10126 (by decide)
private theorem atomic_10126_agreement : RowAgreement 10125 10126 := RowAgreement.single (first := 10125) modelrow_agrees_10126
private def atomic_10127 : CountSegment 1 10 13120 10126 10127 4185 4185 := CountSegment.single (first := 10126) (initial := 4185) row_count_10127 (by decide)
private theorem atomic_10127_agreement : RowAgreement 10126 10127 := RowAgreement.single (first := 10126) modelrow_agrees_10127
private def atomic_10128 : CountSegment 1 10 13120 10127 10128 4185 4187 := CountSegment.single (first := 10127) (initial := 4185) row_count_10128 (by decide)
private theorem atomic_10128_agreement : RowAgreement 10127 10128 := RowAgreement.single (first := 10127) modelrow_agrees_10128
private def atomic_10129 : CountSegment 1 10 13120 10128 10129 4187 4188 := CountSegment.single (first := 10128) (initial := 4187) row_count_10129 (by decide)
private theorem atomic_10129_agreement : RowAgreement 10128 10129 := RowAgreement.single (first := 10128) modelrow_agrees_10129
private def atomic_10130 : CountSegment 1 10 13120 10129 10130 4188 4188 := CountSegment.single (first := 10129) (initial := 4188) row_count_10130 (by decide)
private theorem atomic_10130_agreement : RowAgreement 10129 10130 := RowAgreement.single (first := 10129) modelrow_agrees_10130
private def atomic_10131 : CountSegment 1 10 13120 10130 10131 4188 4188 := CountSegment.single (first := 10130) (initial := 4188) row_count_10131 (by decide)
private theorem atomic_10131_agreement : RowAgreement 10130 10131 := RowAgreement.single (first := 10130) modelrow_agrees_10131
private def atomic_10132 : CountSegment 1 10 13120 10131 10132 4188 4188 := CountSegment.single (first := 10131) (initial := 4188) row_count_10132 (by decide)
private theorem atomic_10132_agreement : RowAgreement 10131 10132 := RowAgreement.single (first := 10131) modelrow_agrees_10132
private def atomic_10133 : CountSegment 1 10 13120 10132 10133 4188 4189 := CountSegment.single (first := 10132) (initial := 4188) row_count_10133 (by decide)
private theorem atomic_10133_agreement : RowAgreement 10132 10133 := RowAgreement.single (first := 10132) modelrow_agrees_10133
private def atomic_10134 : CountSegment 1 10 13120 10133 10134 4189 4189 := CountSegment.single (first := 10133) (initial := 4189) row_count_10134 (by decide)
private theorem atomic_10134_agreement : RowAgreement 10133 10134 := RowAgreement.single (first := 10133) modelrow_agrees_10134
private def atomic_10135 : CountSegment 1 10 13120 10134 10135 4189 4189 := CountSegment.single (first := 10134) (initial := 4189) row_count_10135 (by decide)
private theorem atomic_10135_agreement : RowAgreement 10134 10135 := RowAgreement.single (first := 10134) modelrow_agrees_10135
private def atomic_10136 : CountSegment 1 10 13120 10135 10136 4189 4189 := CountSegment.single (first := 10135) (initial := 4189) row_count_10136 (by decide)
private theorem atomic_10136_agreement : RowAgreement 10135 10136 := RowAgreement.single (first := 10135) modelrow_agrees_10136
private def atomic_10137 : CountSegment 1 10 13120 10136 10137 4189 4189 := CountSegment.single (first := 10136) (initial := 4189) row_count_10137 (by decide)
private theorem atomic_10137_agreement : RowAgreement 10136 10137 := RowAgreement.single (first := 10136) modelrow_agrees_10137
private def atomic_10138 : CountSegment 1 10 13120 10137 10138 4189 4189 := CountSegment.single (first := 10137) (initial := 4189) row_count_10138 (by decide)
private theorem atomic_10138_agreement : RowAgreement 10137 10138 := RowAgreement.single (first := 10137) modelrow_agrees_10138
private def atomic_10139 : CountSegment 1 10 13120 10138 10139 4189 4190 := CountSegment.single (first := 10138) (initial := 4189) row_count_10139 (by decide)
private theorem atomic_10139_agreement : RowAgreement 10138 10139 := RowAgreement.single (first := 10138) modelrow_agrees_10139
private def atomic_10140 : CountSegment 1 10 13120 10139 10140 4190 4191 := CountSegment.single (first := 10139) (initial := 4190) row_count_10140 (by decide)
private theorem atomic_10140_agreement : RowAgreement 10139 10140 := RowAgreement.single (first := 10139) modelrow_agrees_10140
private def atomic_10141 : CountSegment 1 10 13120 10140 10141 4191 4192 := CountSegment.single (first := 10140) (initial := 4191) row_count_10141 (by decide)
private theorem atomic_10141_agreement : RowAgreement 10140 10141 := RowAgreement.single (first := 10140) modelrow_agrees_10141
private def atomic_10142 : CountSegment 1 10 13120 10141 10142 4192 4192 := CountSegment.single (first := 10141) (initial := 4192) row_count_10142 (by decide)
private theorem atomic_10142_agreement : RowAgreement 10141 10142 := RowAgreement.single (first := 10141) modelrow_agrees_10142
private def atomic_10143 : CountSegment 1 10 13120 10142 10143 4192 4192 := CountSegment.single (first := 10142) (initial := 4192) row_count_10143 (by decide)
private theorem atomic_10143_agreement : RowAgreement 10142 10143 := RowAgreement.single (first := 10142) modelrow_agrees_10143
private def atomic_10144 : CountSegment 1 10 13120 10143 10144 4192 4192 := CountSegment.single (first := 10143) (initial := 4192) row_count_10144 (by decide)
private theorem atomic_10144_agreement : RowAgreement 10143 10144 := RowAgreement.single (first := 10143) modelrow_agrees_10144
private def atomic_10145 : CountSegment 1 10 13120 10144 10145 4192 4192 := CountSegment.single (first := 10144) (initial := 4192) row_count_10145 (by decide)
private theorem atomic_10145_agreement : RowAgreement 10144 10145 := RowAgreement.single (first := 10144) modelrow_agrees_10145
private def atomic_10146 : CountSegment 1 10 13120 10145 10146 4192 4192 := CountSegment.single (first := 10145) (initial := 4192) row_count_10146 (by decide)
private theorem atomic_10146_agreement : RowAgreement 10145 10146 := RowAgreement.single (first := 10145) modelrow_agrees_10146
private def atomic_10147 : CountSegment 1 10 13120 10146 10147 4192 4192 := CountSegment.single (first := 10146) (initial := 4192) row_count_10147 (by decide)
private theorem atomic_10147_agreement : RowAgreement 10146 10147 := RowAgreement.single (first := 10146) modelrow_agrees_10147
private def atomic_10148 : CountSegment 1 10 13120 10147 10148 4192 4192 := CountSegment.single (first := 10147) (initial := 4192) row_count_10148 (by decide)
private theorem atomic_10148_agreement : RowAgreement 10147 10148 := RowAgreement.single (first := 10147) modelrow_agrees_10148
private def atomic_10149 : CountSegment 1 10 13120 10148 10149 4192 4192 := CountSegment.single (first := 10148) (initial := 4192) row_count_10149 (by decide)
private theorem atomic_10149_agreement : RowAgreement 10148 10149 := RowAgreement.single (first := 10148) modelrow_agrees_10149
private def atomic_10150 : CountSegment 1 10 13120 10149 10150 4192 4193 := CountSegment.single (first := 10149) (initial := 4192) row_count_10150 (by decide)
private theorem atomic_10150_agreement : RowAgreement 10149 10150 := RowAgreement.single (first := 10149) modelrow_agrees_10150
private def atomic_10151 : CountSegment 1 10 13120 10150 10151 4193 4194 := CountSegment.single (first := 10150) (initial := 4193) row_count_10151 (by decide)
private theorem atomic_10151_agreement : RowAgreement 10150 10151 := RowAgreement.single (first := 10150) modelrow_agrees_10151
private def atomic_10152 : CountSegment 1 10 13120 10151 10152 4194 4194 := CountSegment.single (first := 10151) (initial := 4194) row_count_10152 (by decide)
private theorem atomic_10152_agreement : RowAgreement 10151 10152 := RowAgreement.single (first := 10151) modelrow_agrees_10152
private def atomic_10153 : CountSegment 1 10 13120 10152 10153 4194 4196 := CountSegment.single (first := 10152) (initial := 4194) row_count_10153 (by decide)
private theorem atomic_10153_agreement : RowAgreement 10152 10153 := RowAgreement.single (first := 10152) modelrow_agrees_10153
private def atomic_10154 : CountSegment 1 10 13120 10153 10154 4196 4197 := CountSegment.single (first := 10153) (initial := 4196) row_count_10154 (by decide)
private theorem atomic_10154_agreement : RowAgreement 10153 10154 := RowAgreement.single (first := 10153) modelrow_agrees_10154
private def atomic_10155 : CountSegment 1 10 13120 10154 10155 4197 4197 := CountSegment.single (first := 10154) (initial := 4197) row_count_10155 (by decide)
private theorem atomic_10155_agreement : RowAgreement 10154 10155 := RowAgreement.single (first := 10154) modelrow_agrees_10155
private def atomic_10156 : CountSegment 1 10 13120 10155 10156 4197 4198 := CountSegment.single (first := 10155) (initial := 4197) row_count_10156 (by decide)
private theorem atomic_10156_agreement : RowAgreement 10155 10156 := RowAgreement.single (first := 10155) modelrow_agrees_10156
private def atomic_10157 : CountSegment 1 10 13120 10156 10157 4198 4199 := CountSegment.single (first := 10156) (initial := 4198) row_count_10157 (by decide)
private theorem atomic_10157_agreement : RowAgreement 10156 10157 := RowAgreement.single (first := 10156) modelrow_agrees_10157
private def atomic_10158 : CountSegment 1 10 13120 10157 10158 4199 4199 := CountSegment.single (first := 10157) (initial := 4199) row_count_10158 (by decide)
private theorem atomic_10158_agreement : RowAgreement 10157 10158 := RowAgreement.single (first := 10157) modelrow_agrees_10158
private def atomic_10159 : CountSegment 1 10 13120 10158 10159 4199 4199 := CountSegment.single (first := 10158) (initial := 4199) row_count_10159 (by decide)
private theorem atomic_10159_agreement : RowAgreement 10158 10159 := RowAgreement.single (first := 10158) modelrow_agrees_10159
private def atomic_10160 : CountSegment 1 10 13120 10159 10160 4199 4199 := CountSegment.single (first := 10159) (initial := 4199) row_count_10160 (by decide)
private theorem atomic_10160_agreement : RowAgreement 10159 10160 := RowAgreement.single (first := 10159) modelrow_agrees_10160
private def atomic_10161 : CountSegment 1 10 13120 10160 10161 4199 4199 := CountSegment.single (first := 10160) (initial := 4199) row_count_10161 (by decide)
private theorem atomic_10161_agreement : RowAgreement 10160 10161 := RowAgreement.single (first := 10160) modelrow_agrees_10161
private def atomic_10162 : CountSegment 1 10 13120 10161 10162 4199 4200 := CountSegment.single (first := 10161) (initial := 4199) row_count_10162 (by decide)
private theorem atomic_10162_agreement : RowAgreement 10161 10162 := RowAgreement.single (first := 10161) modelrow_agrees_10162
private def atomic_10163 : CountSegment 1 10 13120 10162 10163 4200 4200 := CountSegment.single (first := 10162) (initial := 4200) row_count_10163 (by decide)
private theorem atomic_10163_agreement : RowAgreement 10162 10163 := RowAgreement.single (first := 10162) modelrow_agrees_10163
private def atomic_10164 : CountSegment 1 10 13120 10163 10164 4200 4200 := CountSegment.single (first := 10163) (initial := 4200) row_count_10164 (by decide)
private theorem atomic_10164_agreement : RowAgreement 10163 10164 := RowAgreement.single (first := 10163) modelrow_agrees_10164
private def atomic_10165 : CountSegment 1 10 13120 10164 10165 4200 4200 := CountSegment.single (first := 10164) (initial := 4200) row_count_10165 (by decide)
private theorem atomic_10165_agreement : RowAgreement 10164 10165 := RowAgreement.single (first := 10164) modelrow_agrees_10165
private def atomic_10166 : CountSegment 1 10 13120 10165 10166 4200 4200 := CountSegment.single (first := 10165) (initial := 4200) row_count_10166 (by decide)
private theorem atomic_10166_agreement : RowAgreement 10165 10166 := RowAgreement.single (first := 10165) modelrow_agrees_10166
private def atomic_10167 : CountSegment 1 10 13120 10166 10167 4200 4200 := CountSegment.single (first := 10166) (initial := 4200) row_count_10167 (by decide)
private theorem atomic_10167_agreement : RowAgreement 10166 10167 := RowAgreement.single (first := 10166) modelrow_agrees_10167
private def atomic_10168 : CountSegment 1 10 13120 10167 10168 4200 4200 := CountSegment.single (first := 10167) (initial := 4200) row_count_10168 (by decide)
private theorem atomic_10168_agreement : RowAgreement 10167 10168 := RowAgreement.single (first := 10167) modelrow_agrees_10168
private def atomic_10169 : CountSegment 1 10 13120 10168 10169 4200 4200 := CountSegment.single (first := 10168) (initial := 4200) row_count_10169 (by decide)
private theorem atomic_10169_agreement : RowAgreement 10168 10169 := RowAgreement.single (first := 10168) modelrow_agrees_10169
private def atomic_10170 : CountSegment 1 10 13120 10169 10170 4200 4201 := CountSegment.single (first := 10169) (initial := 4200) row_count_10170 (by decide)
private theorem atomic_10170_agreement : RowAgreement 10169 10170 := RowAgreement.single (first := 10169) modelrow_agrees_10170
private def atomic_10171 : CountSegment 1 10 13120 10170 10171 4201 4201 := CountSegment.single (first := 10170) (initial := 4201) row_count_10171 (by decide)
private theorem atomic_10171_agreement : RowAgreement 10170 10171 := RowAgreement.single (first := 10170) modelrow_agrees_10171
private def atomic_10172 : CountSegment 1 10 13120 10171 10172 4201 4201 := CountSegment.single (first := 10171) (initial := 4201) row_count_10172 (by decide)
private theorem atomic_10172_agreement : RowAgreement 10171 10172 := RowAgreement.single (first := 10171) modelrow_agrees_10172
private def atomic_10173 : CountSegment 1 10 13120 10172 10173 4201 4201 := CountSegment.single (first := 10172) (initial := 4201) row_count_10173 (by decide)
private theorem atomic_10173_agreement : RowAgreement 10172 10173 := RowAgreement.single (first := 10172) modelrow_agrees_10173
private def atomic_10174 : CountSegment 1 10 13120 10173 10174 4201 4201 := CountSegment.single (first := 10173) (initial := 4201) row_count_10174 (by decide)
private theorem atomic_10174_agreement : RowAgreement 10173 10174 := RowAgreement.single (first := 10173) modelrow_agrees_10174
private def atomic_10175 : CountSegment 1 10 13120 10174 10175 4201 4201 := CountSegment.single (first := 10174) (initial := 4201) row_count_10175 (by decide)
private theorem atomic_10175_agreement : RowAgreement 10174 10175 := RowAgreement.single (first := 10174) modelrow_agrees_10175
private def atomic_10176 : CountSegment 1 10 13120 10175 10176 4201 4201 := CountSegment.single (first := 10175) (initial := 4201) row_count_10176 (by decide)
private theorem atomic_10176_agreement : RowAgreement 10175 10176 := RowAgreement.single (first := 10175) modelrow_agrees_10176
private def atomic_10177 : CountSegment 1 10 13120 10176 10177 4201 4202 := CountSegment.single (first := 10176) (initial := 4201) row_count_10177 (by decide)
private theorem atomic_10177_agreement : RowAgreement 10176 10177 := RowAgreement.single (first := 10176) modelrow_agrees_10177
private def atomic_10178 : CountSegment 1 10 13120 10177 10178 4202 4202 := CountSegment.single (first := 10177) (initial := 4202) row_count_10178 (by decide)
private theorem atomic_10178_agreement : RowAgreement 10177 10178 := RowAgreement.single (first := 10177) modelrow_agrees_10178
private def atomic_10179 : CountSegment 1 10 13120 10178 10179 4202 4202 := CountSegment.single (first := 10178) (initial := 4202) row_count_10179 (by decide)
private theorem atomic_10179_agreement : RowAgreement 10178 10179 := RowAgreement.single (first := 10178) modelrow_agrees_10179
private def atomic_10180 : CountSegment 1 10 13120 10179 10180 4202 4202 := CountSegment.single (first := 10179) (initial := 4202) row_count_10180 (by decide)
private theorem atomic_10180_agreement : RowAgreement 10179 10180 := RowAgreement.single (first := 10179) modelrow_agrees_10180
private def atomic_10181 : CountSegment 1 10 13120 10180 10181 4202 4202 := CountSegment.single (first := 10180) (initial := 4202) row_count_10181 (by decide)
private theorem atomic_10181_agreement : RowAgreement 10180 10181 := RowAgreement.single (first := 10180) modelrow_agrees_10181
private def atomic_10182 : CountSegment 1 10 13120 10181 10182 4202 4204 := CountSegment.single (first := 10181) (initial := 4202) row_count_10182 (by decide)
private theorem atomic_10182_agreement : RowAgreement 10181 10182 := RowAgreement.single (first := 10181) modelrow_agrees_10182
private def atomic_10183 : CountSegment 1 10 13120 10182 10183 4204 4205 := CountSegment.single (first := 10182) (initial := 4204) row_count_10183 (by decide)
private theorem atomic_10183_agreement : RowAgreement 10182 10183 := RowAgreement.single (first := 10182) modelrow_agrees_10183
private def atomic_10184 : CountSegment 1 10 13120 10183 10184 4205 4205 := CountSegment.single (first := 10183) (initial := 4205) row_count_10184 (by decide)
private theorem atomic_10184_agreement : RowAgreement 10183 10184 := RowAgreement.single (first := 10183) modelrow_agrees_10184
private def atomic_10185 : CountSegment 1 10 13120 10184 10185 4205 4207 := CountSegment.single (first := 10184) (initial := 4205) row_count_10185 (by decide)
private theorem atomic_10185_agreement : RowAgreement 10184 10185 := RowAgreement.single (first := 10184) modelrow_agrees_10185
private def atomic_10186 : CountSegment 1 10 13120 10185 10186 4207 4207 := CountSegment.single (first := 10185) (initial := 4207) row_count_10186 (by decide)
private theorem atomic_10186_agreement : RowAgreement 10185 10186 := RowAgreement.single (first := 10185) modelrow_agrees_10186
private def atomic_10187 : CountSegment 1 10 13120 10186 10187 4207 4208 := CountSegment.single (first := 10186) (initial := 4207) row_count_10187 (by decide)
private theorem atomic_10187_agreement : RowAgreement 10186 10187 := RowAgreement.single (first := 10186) modelrow_agrees_10187
private def atomic_10188 : CountSegment 1 10 13120 10187 10188 4208 4208 := CountSegment.single (first := 10187) (initial := 4208) row_count_10188 (by decide)
private theorem atomic_10188_agreement : RowAgreement 10187 10188 := RowAgreement.single (first := 10187) modelrow_agrees_10188
private def atomic_10189 : CountSegment 1 10 13120 10188 10189 4208 4209 := CountSegment.single (first := 10188) (initial := 4208) row_count_10189 (by decide)
private theorem atomic_10189_agreement : RowAgreement 10188 10189 := RowAgreement.single (first := 10188) modelrow_agrees_10189
private def atomic_10190 : CountSegment 1 10 13120 10189 10190 4209 4210 := CountSegment.single (first := 10189) (initial := 4209) row_count_10190 (by decide)
private theorem atomic_10190_agreement : RowAgreement 10189 10190 := RowAgreement.single (first := 10189) modelrow_agrees_10190
private def atomic_10191 : CountSegment 1 10 13120 10190 10191 4210 4210 := CountSegment.single (first := 10190) (initial := 4210) row_count_10191 (by decide)
private theorem atomic_10191_agreement : RowAgreement 10190 10191 := RowAgreement.single (first := 10190) modelrow_agrees_10191
private def atomic_10192 : CountSegment 1 10 13120 10191 10192 4210 4212 := CountSegment.single (first := 10191) (initial := 4210) row_count_10192 (by decide)
private theorem atomic_10192_agreement : RowAgreement 10191 10192 := RowAgreement.single (first := 10191) modelrow_agrees_10192
private def atomic_10193 : CountSegment 1 10 13120 10192 10193 4212 4213 := CountSegment.single (first := 10192) (initial := 4212) row_count_10193 (by decide)
private theorem atomic_10193_agreement : RowAgreement 10192 10193 := RowAgreement.single (first := 10192) modelrow_agrees_10193
private def atomic_10194 : CountSegment 1 10 13120 10193 10194 4213 4213 := CountSegment.single (first := 10193) (initial := 4213) row_count_10194 (by decide)
private theorem atomic_10194_agreement : RowAgreement 10193 10194 := RowAgreement.single (first := 10193) modelrow_agrees_10194
private def atomic_10195 : CountSegment 1 10 13120 10194 10195 4213 4213 := CountSegment.single (first := 10194) (initial := 4213) row_count_10195 (by decide)
private theorem atomic_10195_agreement : RowAgreement 10194 10195 := RowAgreement.single (first := 10194) modelrow_agrees_10195
private def atomic_10196 : CountSegment 1 10 13120 10195 10196 4213 4213 := CountSegment.single (first := 10195) (initial := 4213) row_count_10196 (by decide)
private theorem atomic_10196_agreement : RowAgreement 10195 10196 := RowAgreement.single (first := 10195) modelrow_agrees_10196
private def atomic_10197 : CountSegment 1 10 13120 10196 10197 4213 4213 := CountSegment.single (first := 10196) (initial := 4213) row_count_10197 (by decide)
private theorem atomic_10197_agreement : RowAgreement 10196 10197 := RowAgreement.single (first := 10196) modelrow_agrees_10197
private def atomic_10198 : CountSegment 1 10 13120 10197 10198 4213 4213 := CountSegment.single (first := 10197) (initial := 4213) row_count_10198 (by decide)
private theorem atomic_10198_agreement : RowAgreement 10197 10198 := RowAgreement.single (first := 10197) modelrow_agrees_10198
private def atomic_10199 : CountSegment 1 10 13120 10198 10199 4213 4213 := CountSegment.single (first := 10198) (initial := 4213) row_count_10199 (by decide)
private theorem atomic_10199_agreement : RowAgreement 10198 10199 := RowAgreement.single (first := 10198) modelrow_agrees_10199
private def atomic_10200 : CountSegment 1 10 13120 10199 10200 4213 4215 := CountSegment.single (first := 10199) (initial := 4213) row_count_10200 (by decide)
private theorem atomic_10200_agreement : RowAgreement 10199 10200 := RowAgreement.single (first := 10199) modelrow_agrees_10200
private def atomic_10201 : CountSegment 1 10 13120 10200 10201 4215 4215 := CountSegment.single (first := 10200) (initial := 4215) row_count_10201 (by decide)
private theorem atomic_10201_agreement : RowAgreement 10200 10201 := RowAgreement.single (first := 10200) modelrow_agrees_10201
private def atomic_10202 : CountSegment 1 10 13120 10201 10202 4215 4215 := CountSegment.single (first := 10201) (initial := 4215) row_count_10202 (by decide)
private theorem atomic_10202_agreement : RowAgreement 10201 10202 := RowAgreement.single (first := 10201) modelrow_agrees_10202
private def atomic_10203 : CountSegment 1 10 13120 10202 10203 4215 4216 := CountSegment.single (first := 10202) (initial := 4215) row_count_10203 (by decide)
private theorem atomic_10203_agreement : RowAgreement 10202 10203 := RowAgreement.single (first := 10202) modelrow_agrees_10203
private def atomic_10204 : CountSegment 1 10 13120 10203 10204 4216 4216 := CountSegment.single (first := 10203) (initial := 4216) row_count_10204 (by decide)
private theorem atomic_10204_agreement : RowAgreement 10203 10204 := RowAgreement.single (first := 10203) modelrow_agrees_10204
private def atomic_10205 : CountSegment 1 10 13120 10204 10205 4216 4216 := CountSegment.single (first := 10204) (initial := 4216) row_count_10205 (by decide)
private theorem atomic_10205_agreement : RowAgreement 10204 10205 := RowAgreement.single (first := 10204) modelrow_agrees_10205
private def atomic_10206 : CountSegment 1 10 13120 10205 10206 4216 4216 := CountSegment.single (first := 10205) (initial := 4216) row_count_10206 (by decide)
private theorem atomic_10206_agreement : RowAgreement 10205 10206 := RowAgreement.single (first := 10205) modelrow_agrees_10206
private def atomic_10207 : CountSegment 1 10 13120 10206 10207 4216 4217 := CountSegment.single (first := 10206) (initial := 4216) row_count_10207 (by decide)
private theorem atomic_10207_agreement : RowAgreement 10206 10207 := RowAgreement.single (first := 10206) modelrow_agrees_10207
private def atomic_10208 : CountSegment 1 10 13120 10207 10208 4217 4217 := CountSegment.single (first := 10207) (initial := 4217) row_count_10208 (by decide)
private theorem atomic_10208_agreement : RowAgreement 10207 10208 := RowAgreement.single (first := 10207) modelrow_agrees_10208
private def atomic_10209 : CountSegment 1 10 13120 10208 10209 4217 4217 := CountSegment.single (first := 10208) (initial := 4217) row_count_10209 (by decide)
private theorem atomic_10209_agreement : RowAgreement 10208 10209 := RowAgreement.single (first := 10208) modelrow_agrees_10209
private def atomic_10210 : CountSegment 1 10 13120 10209 10210 4217 4217 := CountSegment.single (first := 10209) (initial := 4217) row_count_10210 (by decide)
private theorem atomic_10210_agreement : RowAgreement 10209 10210 := RowAgreement.single (first := 10209) modelrow_agrees_10210
private def atomic_10211 : CountSegment 1 10 13120 10210 10211 4217 4217 := CountSegment.single (first := 10210) (initial := 4217) row_count_10211 (by decide)
private theorem atomic_10211_agreement : RowAgreement 10210 10211 := RowAgreement.single (first := 10210) modelrow_agrees_10211
private def atomic_10212 : CountSegment 1 10 13120 10211 10212 4217 4218 := CountSegment.single (first := 10211) (initial := 4217) row_count_10212 (by decide)
private theorem atomic_10212_agreement : RowAgreement 10211 10212 := RowAgreement.single (first := 10211) modelrow_agrees_10212
private def atomic_10213 : CountSegment 1 10 13120 10212 10213 4218 4218 := CountSegment.single (first := 10212) (initial := 4218) row_count_10213 (by decide)
private theorem atomic_10213_agreement : RowAgreement 10212 10213 := RowAgreement.single (first := 10212) modelrow_agrees_10213
private def atomic_10214 : CountSegment 1 10 13120 10213 10214 4218 4218 := CountSegment.single (first := 10213) (initial := 4218) row_count_10214 (by decide)
private theorem atomic_10214_agreement : RowAgreement 10213 10214 := RowAgreement.single (first := 10213) modelrow_agrees_10214
private def atomic_10215 : CountSegment 1 10 13120 10214 10215 4218 4218 := CountSegment.single (first := 10214) (initial := 4218) row_count_10215 (by decide)
private theorem atomic_10215_agreement : RowAgreement 10214 10215 := RowAgreement.single (first := 10214) modelrow_agrees_10215
private def atomic_10216 : CountSegment 1 10 13120 10215 10216 4218 4218 := CountSegment.single (first := 10215) (initial := 4218) row_count_10216 (by decide)
private theorem atomic_10216_agreement : RowAgreement 10215 10216 := RowAgreement.single (first := 10215) modelrow_agrees_10216
private def atomic_10217 : CountSegment 1 10 13120 10216 10217 4218 4218 := CountSegment.single (first := 10216) (initial := 4218) row_count_10217 (by decide)
private theorem atomic_10217_agreement : RowAgreement 10216 10217 := RowAgreement.single (first := 10216) modelrow_agrees_10217
private def atomic_10218 : CountSegment 1 10 13120 10217 10218 4218 4218 := CountSegment.single (first := 10217) (initial := 4218) row_count_10218 (by decide)
private theorem atomic_10218_agreement : RowAgreement 10217 10218 := RowAgreement.single (first := 10217) modelrow_agrees_10218
private def atomic_10219 : CountSegment 1 10 13120 10218 10219 4218 4218 := CountSegment.single (first := 10218) (initial := 4218) row_count_10219 (by decide)
private theorem atomic_10219_agreement : RowAgreement 10218 10219 := RowAgreement.single (first := 10218) modelrow_agrees_10219
private def atomic_10220 : CountSegment 1 10 13120 10219 10220 4218 4219 := CountSegment.single (first := 10219) (initial := 4218) row_count_10220 (by decide)
private theorem atomic_10220_agreement : RowAgreement 10219 10220 := RowAgreement.single (first := 10219) modelrow_agrees_10220
private def atomic_10221 : CountSegment 1 10 13120 10220 10221 4219 4219 := CountSegment.single (first := 10220) (initial := 4219) row_count_10221 (by decide)
private theorem atomic_10221_agreement : RowAgreement 10220 10221 := RowAgreement.single (first := 10220) modelrow_agrees_10221
private def atomic_10222 : CountSegment 1 10 13120 10221 10222 4219 4219 := CountSegment.single (first := 10221) (initial := 4219) row_count_10222 (by decide)
private theorem atomic_10222_agreement : RowAgreement 10221 10222 := RowAgreement.single (first := 10221) modelrow_agrees_10222
private def atomic_10223 : CountSegment 1 10 13120 10222 10223 4219 4219 := CountSegment.single (first := 10222) (initial := 4219) row_count_10223 (by decide)
private theorem atomic_10223_agreement : RowAgreement 10222 10223 := RowAgreement.single (first := 10222) modelrow_agrees_10223
private def atomic_10224 : CountSegment 1 10 13120 10223 10224 4219 4220 := CountSegment.single (first := 10223) (initial := 4219) row_count_10224 (by decide)
private theorem atomic_10224_agreement : RowAgreement 10223 10224 := RowAgreement.single (first := 10223) modelrow_agrees_10224
private def atomic_10225 : CountSegment 1 10 13120 10224 10225 4220 4220 := CountSegment.single (first := 10224) (initial := 4220) row_count_10225 (by decide)
private theorem atomic_10225_agreement : RowAgreement 10224 10225 := RowAgreement.single (first := 10224) modelrow_agrees_10225
private def atomic_10226 : CountSegment 1 10 13120 10225 10226 4220 4220 := CountSegment.single (first := 10225) (initial := 4220) row_count_10226 (by decide)
private theorem atomic_10226_agreement : RowAgreement 10225 10226 := RowAgreement.single (first := 10225) modelrow_agrees_10226
private def atomic_10227 : CountSegment 1 10 13120 10226 10227 4220 4220 := CountSegment.single (first := 10226) (initial := 4220) row_count_10227 (by decide)
private theorem atomic_10227_agreement : RowAgreement 10226 10227 := RowAgreement.single (first := 10226) modelrow_agrees_10227
private def atomic_10228 : CountSegment 1 10 13120 10227 10228 4220 4220 := CountSegment.single (first := 10227) (initial := 4220) row_count_10228 (by decide)
private theorem atomic_10228_agreement : RowAgreement 10227 10228 := RowAgreement.single (first := 10227) modelrow_agrees_10228
private def atomic_10229 : CountSegment 1 10 13120 10228 10229 4220 4220 := CountSegment.single (first := 10228) (initial := 4220) row_count_10229 (by decide)
private theorem atomic_10229_agreement : RowAgreement 10228 10229 := RowAgreement.single (first := 10228) modelrow_agrees_10229
private def atomic_10230 : CountSegment 1 10 13120 10229 10230 4220 4221 := CountSegment.single (first := 10229) (initial := 4220) row_count_10230 (by decide)
private theorem atomic_10230_agreement : RowAgreement 10229 10230 := RowAgreement.single (first := 10229) modelrow_agrees_10230
private def atomic_10231 : CountSegment 1 10 13120 10230 10231 4221 4222 := CountSegment.single (first := 10230) (initial := 4221) row_count_10231 (by decide)
private theorem atomic_10231_agreement : RowAgreement 10230 10231 := RowAgreement.single (first := 10230) modelrow_agrees_10231
private def atomic_10232 : CountSegment 1 10 13120 10231 10232 4222 4222 := CountSegment.single (first := 10231) (initial := 4222) row_count_10232 (by decide)
private theorem atomic_10232_agreement : RowAgreement 10231 10232 := RowAgreement.single (first := 10231) modelrow_agrees_10232
private def atomic_10233 : CountSegment 1 10 13120 10232 10233 4222 4222 := CountSegment.single (first := 10232) (initial := 4222) row_count_10233 (by decide)
private theorem atomic_10233_agreement : RowAgreement 10232 10233 := RowAgreement.single (first := 10232) modelrow_agrees_10233
private def atomic_10234 : CountSegment 1 10 13120 10233 10234 4222 4222 := CountSegment.single (first := 10233) (initial := 4222) row_count_10234 (by decide)
private theorem atomic_10234_agreement : RowAgreement 10233 10234 := RowAgreement.single (first := 10233) modelrow_agrees_10234
private def atomic_10235 : CountSegment 1 10 13120 10234 10235 4222 4222 := CountSegment.single (first := 10234) (initial := 4222) row_count_10235 (by decide)
private theorem atomic_10235_agreement : RowAgreement 10234 10235 := RowAgreement.single (first := 10234) modelrow_agrees_10235
private def atomic_10236 : CountSegment 1 10 13120 10235 10236 4222 4222 := CountSegment.single (first := 10235) (initial := 4222) row_count_10236 (by decide)
private theorem atomic_10236_agreement : RowAgreement 10235 10236 := RowAgreement.single (first := 10235) modelrow_agrees_10236
private def atomic_10237 : CountSegment 1 10 13120 10236 10237 4222 4222 := CountSegment.single (first := 10236) (initial := 4222) row_count_10237 (by decide)
private theorem atomic_10237_agreement : RowAgreement 10236 10237 := RowAgreement.single (first := 10236) modelrow_agrees_10237
private def atomic_10238 : CountSegment 1 10 13120 10237 10238 4222 4223 := CountSegment.single (first := 10237) (initial := 4222) row_count_10238 (by decide)
private theorem atomic_10238_agreement : RowAgreement 10237 10238 := RowAgreement.single (first := 10237) modelrow_agrees_10238
private def atomic_10239 : CountSegment 1 10 13120 10238 10239 4223 4223 := CountSegment.single (first := 10238) (initial := 4223) row_count_10239 (by decide)
private theorem atomic_10239_agreement : RowAgreement 10238 10239 := RowAgreement.single (first := 10238) modelrow_agrees_10239
private def atomic_10240 : CountSegment 1 10 13120 10239 10240 4223 4224 := CountSegment.single (first := 10239) (initial := 4223) row_count_10240 (by decide)
private theorem atomic_10240_agreement : RowAgreement 10239 10240 := RowAgreement.single (first := 10239) modelrow_agrees_10240
private def within_079_000 : CountSegment 1 10 13120 10112 10114 4181 4181 := CountSegment.join atomic_10113 atomic_10114
private theorem within_079_000_agreement : RowAgreement 10112 10114 := RowAgreement.join atomic_10113_agreement atomic_10114_agreement
private def within_079_001 : CountSegment 1 10 13120 10114 10116 4181 4181 := CountSegment.join atomic_10115 atomic_10116
private theorem within_079_001_agreement : RowAgreement 10114 10116 := RowAgreement.join atomic_10115_agreement atomic_10116_agreement
private def within_079_002 : CountSegment 1 10 13120 10112 10116 4181 4181 := CountSegment.join within_079_000 within_079_001
private theorem within_079_002_agreement : RowAgreement 10112 10116 := RowAgreement.join within_079_000_agreement within_079_001_agreement
private def within_079_003 : CountSegment 1 10 13120 10116 10118 4181 4181 := CountSegment.join atomic_10117 atomic_10118
private theorem within_079_003_agreement : RowAgreement 10116 10118 := RowAgreement.join atomic_10117_agreement atomic_10118_agreement
private def within_079_004 : CountSegment 1 10 13120 10118 10120 4181 4182 := CountSegment.join atomic_10119 atomic_10120
private theorem within_079_004_agreement : RowAgreement 10118 10120 := RowAgreement.join atomic_10119_agreement atomic_10120_agreement
private def within_079_005 : CountSegment 1 10 13120 10116 10120 4181 4182 := CountSegment.join within_079_003 within_079_004
private theorem within_079_005_agreement : RowAgreement 10116 10120 := RowAgreement.join within_079_003_agreement within_079_004_agreement
private def within_079_006 : CountSegment 1 10 13120 10112 10120 4181 4182 := CountSegment.join within_079_002 within_079_005
private theorem within_079_006_agreement : RowAgreement 10112 10120 := RowAgreement.join within_079_002_agreement within_079_005_agreement
private def within_079_007 : CountSegment 1 10 13120 10120 10122 4182 4184 := CountSegment.join atomic_10121 atomic_10122
private theorem within_079_007_agreement : RowAgreement 10120 10122 := RowAgreement.join atomic_10121_agreement atomic_10122_agreement
private def within_079_008 : CountSegment 1 10 13120 10122 10124 4184 4184 := CountSegment.join atomic_10123 atomic_10124
private theorem within_079_008_agreement : RowAgreement 10122 10124 := RowAgreement.join atomic_10123_agreement atomic_10124_agreement
private def within_079_009 : CountSegment 1 10 13120 10120 10124 4182 4184 := CountSegment.join within_079_007 within_079_008
private theorem within_079_009_agreement : RowAgreement 10120 10124 := RowAgreement.join within_079_007_agreement within_079_008_agreement
private def within_079_010 : CountSegment 1 10 13120 10124 10126 4184 4185 := CountSegment.join atomic_10125 atomic_10126
private theorem within_079_010_agreement : RowAgreement 10124 10126 := RowAgreement.join atomic_10125_agreement atomic_10126_agreement
private def within_079_011 : CountSegment 1 10 13120 10126 10128 4185 4187 := CountSegment.join atomic_10127 atomic_10128
private theorem within_079_011_agreement : RowAgreement 10126 10128 := RowAgreement.join atomic_10127_agreement atomic_10128_agreement
private def within_079_012 : CountSegment 1 10 13120 10124 10128 4184 4187 := CountSegment.join within_079_010 within_079_011
private theorem within_079_012_agreement : RowAgreement 10124 10128 := RowAgreement.join within_079_010_agreement within_079_011_agreement
private def within_079_013 : CountSegment 1 10 13120 10120 10128 4182 4187 := CountSegment.join within_079_009 within_079_012
private theorem within_079_013_agreement : RowAgreement 10120 10128 := RowAgreement.join within_079_009_agreement within_079_012_agreement
private def within_079_014 : CountSegment 1 10 13120 10112 10128 4181 4187 := CountSegment.join within_079_006 within_079_013
private theorem within_079_014_agreement : RowAgreement 10112 10128 := RowAgreement.join within_079_006_agreement within_079_013_agreement
private def within_079_015 : CountSegment 1 10 13120 10128 10130 4187 4188 := CountSegment.join atomic_10129 atomic_10130
private theorem within_079_015_agreement : RowAgreement 10128 10130 := RowAgreement.join atomic_10129_agreement atomic_10130_agreement
private def within_079_016 : CountSegment 1 10 13120 10130 10132 4188 4188 := CountSegment.join atomic_10131 atomic_10132
private theorem within_079_016_agreement : RowAgreement 10130 10132 := RowAgreement.join atomic_10131_agreement atomic_10132_agreement
private def within_079_017 : CountSegment 1 10 13120 10128 10132 4187 4188 := CountSegment.join within_079_015 within_079_016
private theorem within_079_017_agreement : RowAgreement 10128 10132 := RowAgreement.join within_079_015_agreement within_079_016_agreement
private def within_079_018 : CountSegment 1 10 13120 10132 10134 4188 4189 := CountSegment.join atomic_10133 atomic_10134
private theorem within_079_018_agreement : RowAgreement 10132 10134 := RowAgreement.join atomic_10133_agreement atomic_10134_agreement
private def within_079_019 : CountSegment 1 10 13120 10134 10136 4189 4189 := CountSegment.join atomic_10135 atomic_10136
private theorem within_079_019_agreement : RowAgreement 10134 10136 := RowAgreement.join atomic_10135_agreement atomic_10136_agreement
private def within_079_020 : CountSegment 1 10 13120 10132 10136 4188 4189 := CountSegment.join within_079_018 within_079_019
private theorem within_079_020_agreement : RowAgreement 10132 10136 := RowAgreement.join within_079_018_agreement within_079_019_agreement
private def within_079_021 : CountSegment 1 10 13120 10128 10136 4187 4189 := CountSegment.join within_079_017 within_079_020
private theorem within_079_021_agreement : RowAgreement 10128 10136 := RowAgreement.join within_079_017_agreement within_079_020_agreement
private def within_079_022 : CountSegment 1 10 13120 10136 10138 4189 4189 := CountSegment.join atomic_10137 atomic_10138
private theorem within_079_022_agreement : RowAgreement 10136 10138 := RowAgreement.join atomic_10137_agreement atomic_10138_agreement
private def within_079_023 : CountSegment 1 10 13120 10138 10140 4189 4191 := CountSegment.join atomic_10139 atomic_10140
private theorem within_079_023_agreement : RowAgreement 10138 10140 := RowAgreement.join atomic_10139_agreement atomic_10140_agreement
private def within_079_024 : CountSegment 1 10 13120 10136 10140 4189 4191 := CountSegment.join within_079_022 within_079_023
private theorem within_079_024_agreement : RowAgreement 10136 10140 := RowAgreement.join within_079_022_agreement within_079_023_agreement
private def within_079_025 : CountSegment 1 10 13120 10140 10142 4191 4192 := CountSegment.join atomic_10141 atomic_10142
private theorem within_079_025_agreement : RowAgreement 10140 10142 := RowAgreement.join atomic_10141_agreement atomic_10142_agreement
private def within_079_026 : CountSegment 1 10 13120 10142 10144 4192 4192 := CountSegment.join atomic_10143 atomic_10144
private theorem within_079_026_agreement : RowAgreement 10142 10144 := RowAgreement.join atomic_10143_agreement atomic_10144_agreement
private def within_079_027 : CountSegment 1 10 13120 10140 10144 4191 4192 := CountSegment.join within_079_025 within_079_026
private theorem within_079_027_agreement : RowAgreement 10140 10144 := RowAgreement.join within_079_025_agreement within_079_026_agreement
private def within_079_028 : CountSegment 1 10 13120 10136 10144 4189 4192 := CountSegment.join within_079_024 within_079_027
private theorem within_079_028_agreement : RowAgreement 10136 10144 := RowAgreement.join within_079_024_agreement within_079_027_agreement
private def within_079_029 : CountSegment 1 10 13120 10128 10144 4187 4192 := CountSegment.join within_079_021 within_079_028
private theorem within_079_029_agreement : RowAgreement 10128 10144 := RowAgreement.join within_079_021_agreement within_079_028_agreement
private def within_079_030 : CountSegment 1 10 13120 10112 10144 4181 4192 := CountSegment.join within_079_014 within_079_029
private theorem within_079_030_agreement : RowAgreement 10112 10144 := RowAgreement.join within_079_014_agreement within_079_029_agreement
private def within_079_031 : CountSegment 1 10 13120 10144 10146 4192 4192 := CountSegment.join atomic_10145 atomic_10146
private theorem within_079_031_agreement : RowAgreement 10144 10146 := RowAgreement.join atomic_10145_agreement atomic_10146_agreement
private def within_079_032 : CountSegment 1 10 13120 10146 10148 4192 4192 := CountSegment.join atomic_10147 atomic_10148
private theorem within_079_032_agreement : RowAgreement 10146 10148 := RowAgreement.join atomic_10147_agreement atomic_10148_agreement
private def within_079_033 : CountSegment 1 10 13120 10144 10148 4192 4192 := CountSegment.join within_079_031 within_079_032
private theorem within_079_033_agreement : RowAgreement 10144 10148 := RowAgreement.join within_079_031_agreement within_079_032_agreement
private def within_079_034 : CountSegment 1 10 13120 10148 10150 4192 4193 := CountSegment.join atomic_10149 atomic_10150
private theorem within_079_034_agreement : RowAgreement 10148 10150 := RowAgreement.join atomic_10149_agreement atomic_10150_agreement
private def within_079_035 : CountSegment 1 10 13120 10150 10152 4193 4194 := CountSegment.join atomic_10151 atomic_10152
private theorem within_079_035_agreement : RowAgreement 10150 10152 := RowAgreement.join atomic_10151_agreement atomic_10152_agreement
private def within_079_036 : CountSegment 1 10 13120 10148 10152 4192 4194 := CountSegment.join within_079_034 within_079_035
private theorem within_079_036_agreement : RowAgreement 10148 10152 := RowAgreement.join within_079_034_agreement within_079_035_agreement
private def within_079_037 : CountSegment 1 10 13120 10144 10152 4192 4194 := CountSegment.join within_079_033 within_079_036
private theorem within_079_037_agreement : RowAgreement 10144 10152 := RowAgreement.join within_079_033_agreement within_079_036_agreement
private def within_079_038 : CountSegment 1 10 13120 10152 10154 4194 4197 := CountSegment.join atomic_10153 atomic_10154
private theorem within_079_038_agreement : RowAgreement 10152 10154 := RowAgreement.join atomic_10153_agreement atomic_10154_agreement
private def within_079_039 : CountSegment 1 10 13120 10154 10156 4197 4198 := CountSegment.join atomic_10155 atomic_10156
private theorem within_079_039_agreement : RowAgreement 10154 10156 := RowAgreement.join atomic_10155_agreement atomic_10156_agreement
private def within_079_040 : CountSegment 1 10 13120 10152 10156 4194 4198 := CountSegment.join within_079_038 within_079_039
private theorem within_079_040_agreement : RowAgreement 10152 10156 := RowAgreement.join within_079_038_agreement within_079_039_agreement
private def within_079_041 : CountSegment 1 10 13120 10156 10158 4198 4199 := CountSegment.join atomic_10157 atomic_10158
private theorem within_079_041_agreement : RowAgreement 10156 10158 := RowAgreement.join atomic_10157_agreement atomic_10158_agreement
private def within_079_042 : CountSegment 1 10 13120 10158 10160 4199 4199 := CountSegment.join atomic_10159 atomic_10160
private theorem within_079_042_agreement : RowAgreement 10158 10160 := RowAgreement.join atomic_10159_agreement atomic_10160_agreement
private def within_079_043 : CountSegment 1 10 13120 10156 10160 4198 4199 := CountSegment.join within_079_041 within_079_042
private theorem within_079_043_agreement : RowAgreement 10156 10160 := RowAgreement.join within_079_041_agreement within_079_042_agreement
private def within_079_044 : CountSegment 1 10 13120 10152 10160 4194 4199 := CountSegment.join within_079_040 within_079_043
private theorem within_079_044_agreement : RowAgreement 10152 10160 := RowAgreement.join within_079_040_agreement within_079_043_agreement
private def within_079_045 : CountSegment 1 10 13120 10144 10160 4192 4199 := CountSegment.join within_079_037 within_079_044
private theorem within_079_045_agreement : RowAgreement 10144 10160 := RowAgreement.join within_079_037_agreement within_079_044_agreement
private def within_079_046 : CountSegment 1 10 13120 10160 10162 4199 4200 := CountSegment.join atomic_10161 atomic_10162
private theorem within_079_046_agreement : RowAgreement 10160 10162 := RowAgreement.join atomic_10161_agreement atomic_10162_agreement
private def within_079_047 : CountSegment 1 10 13120 10162 10164 4200 4200 := CountSegment.join atomic_10163 atomic_10164
private theorem within_079_047_agreement : RowAgreement 10162 10164 := RowAgreement.join atomic_10163_agreement atomic_10164_agreement
private def within_079_048 : CountSegment 1 10 13120 10160 10164 4199 4200 := CountSegment.join within_079_046 within_079_047
private theorem within_079_048_agreement : RowAgreement 10160 10164 := RowAgreement.join within_079_046_agreement within_079_047_agreement
private def within_079_049 : CountSegment 1 10 13120 10164 10166 4200 4200 := CountSegment.join atomic_10165 atomic_10166
private theorem within_079_049_agreement : RowAgreement 10164 10166 := RowAgreement.join atomic_10165_agreement atomic_10166_agreement
private def within_079_050 : CountSegment 1 10 13120 10166 10168 4200 4200 := CountSegment.join atomic_10167 atomic_10168
private theorem within_079_050_agreement : RowAgreement 10166 10168 := RowAgreement.join atomic_10167_agreement atomic_10168_agreement
private def within_079_051 : CountSegment 1 10 13120 10164 10168 4200 4200 := CountSegment.join within_079_049 within_079_050
private theorem within_079_051_agreement : RowAgreement 10164 10168 := RowAgreement.join within_079_049_agreement within_079_050_agreement
private def within_079_052 : CountSegment 1 10 13120 10160 10168 4199 4200 := CountSegment.join within_079_048 within_079_051
private theorem within_079_052_agreement : RowAgreement 10160 10168 := RowAgreement.join within_079_048_agreement within_079_051_agreement
private def within_079_053 : CountSegment 1 10 13120 10168 10170 4200 4201 := CountSegment.join atomic_10169 atomic_10170
private theorem within_079_053_agreement : RowAgreement 10168 10170 := RowAgreement.join atomic_10169_agreement atomic_10170_agreement
private def within_079_054 : CountSegment 1 10 13120 10170 10172 4201 4201 := CountSegment.join atomic_10171 atomic_10172
private theorem within_079_054_agreement : RowAgreement 10170 10172 := RowAgreement.join atomic_10171_agreement atomic_10172_agreement
private def within_079_055 : CountSegment 1 10 13120 10168 10172 4200 4201 := CountSegment.join within_079_053 within_079_054
private theorem within_079_055_agreement : RowAgreement 10168 10172 := RowAgreement.join within_079_053_agreement within_079_054_agreement
private def within_079_056 : CountSegment 1 10 13120 10172 10174 4201 4201 := CountSegment.join atomic_10173 atomic_10174
private theorem within_079_056_agreement : RowAgreement 10172 10174 := RowAgreement.join atomic_10173_agreement atomic_10174_agreement
private def within_079_057 : CountSegment 1 10 13120 10174 10176 4201 4201 := CountSegment.join atomic_10175 atomic_10176
private theorem within_079_057_agreement : RowAgreement 10174 10176 := RowAgreement.join atomic_10175_agreement atomic_10176_agreement
private def within_079_058 : CountSegment 1 10 13120 10172 10176 4201 4201 := CountSegment.join within_079_056 within_079_057
private theorem within_079_058_agreement : RowAgreement 10172 10176 := RowAgreement.join within_079_056_agreement within_079_057_agreement
private def within_079_059 : CountSegment 1 10 13120 10168 10176 4200 4201 := CountSegment.join within_079_055 within_079_058
private theorem within_079_059_agreement : RowAgreement 10168 10176 := RowAgreement.join within_079_055_agreement within_079_058_agreement
private def within_079_060 : CountSegment 1 10 13120 10160 10176 4199 4201 := CountSegment.join within_079_052 within_079_059
private theorem within_079_060_agreement : RowAgreement 10160 10176 := RowAgreement.join within_079_052_agreement within_079_059_agreement
private def within_079_061 : CountSegment 1 10 13120 10144 10176 4192 4201 := CountSegment.join within_079_045 within_079_060
private theorem within_079_061_agreement : RowAgreement 10144 10176 := RowAgreement.join within_079_045_agreement within_079_060_agreement
private def within_079_062 : CountSegment 1 10 13120 10112 10176 4181 4201 := CountSegment.join within_079_030 within_079_061
private theorem within_079_062_agreement : RowAgreement 10112 10176 := RowAgreement.join within_079_030_agreement within_079_061_agreement
private def within_079_063 : CountSegment 1 10 13120 10176 10178 4201 4202 := CountSegment.join atomic_10177 atomic_10178
private theorem within_079_063_agreement : RowAgreement 10176 10178 := RowAgreement.join atomic_10177_agreement atomic_10178_agreement
private def within_079_064 : CountSegment 1 10 13120 10178 10180 4202 4202 := CountSegment.join atomic_10179 atomic_10180
private theorem within_079_064_agreement : RowAgreement 10178 10180 := RowAgreement.join atomic_10179_agreement atomic_10180_agreement
private def within_079_065 : CountSegment 1 10 13120 10176 10180 4201 4202 := CountSegment.join within_079_063 within_079_064
private theorem within_079_065_agreement : RowAgreement 10176 10180 := RowAgreement.join within_079_063_agreement within_079_064_agreement
private def within_079_066 : CountSegment 1 10 13120 10180 10182 4202 4204 := CountSegment.join atomic_10181 atomic_10182
private theorem within_079_066_agreement : RowAgreement 10180 10182 := RowAgreement.join atomic_10181_agreement atomic_10182_agreement
private def within_079_067 : CountSegment 1 10 13120 10182 10184 4204 4205 := CountSegment.join atomic_10183 atomic_10184
private theorem within_079_067_agreement : RowAgreement 10182 10184 := RowAgreement.join atomic_10183_agreement atomic_10184_agreement
private def within_079_068 : CountSegment 1 10 13120 10180 10184 4202 4205 := CountSegment.join within_079_066 within_079_067
private theorem within_079_068_agreement : RowAgreement 10180 10184 := RowAgreement.join within_079_066_agreement within_079_067_agreement
private def within_079_069 : CountSegment 1 10 13120 10176 10184 4201 4205 := CountSegment.join within_079_065 within_079_068
private theorem within_079_069_agreement : RowAgreement 10176 10184 := RowAgreement.join within_079_065_agreement within_079_068_agreement
private def within_079_070 : CountSegment 1 10 13120 10184 10186 4205 4207 := CountSegment.join atomic_10185 atomic_10186
private theorem within_079_070_agreement : RowAgreement 10184 10186 := RowAgreement.join atomic_10185_agreement atomic_10186_agreement
private def within_079_071 : CountSegment 1 10 13120 10186 10188 4207 4208 := CountSegment.join atomic_10187 atomic_10188
private theorem within_079_071_agreement : RowAgreement 10186 10188 := RowAgreement.join atomic_10187_agreement atomic_10188_agreement
private def within_079_072 : CountSegment 1 10 13120 10184 10188 4205 4208 := CountSegment.join within_079_070 within_079_071
private theorem within_079_072_agreement : RowAgreement 10184 10188 := RowAgreement.join within_079_070_agreement within_079_071_agreement
private def within_079_073 : CountSegment 1 10 13120 10188 10190 4208 4210 := CountSegment.join atomic_10189 atomic_10190
private theorem within_079_073_agreement : RowAgreement 10188 10190 := RowAgreement.join atomic_10189_agreement atomic_10190_agreement
private def within_079_074 : CountSegment 1 10 13120 10190 10192 4210 4212 := CountSegment.join atomic_10191 atomic_10192
private theorem within_079_074_agreement : RowAgreement 10190 10192 := RowAgreement.join atomic_10191_agreement atomic_10192_agreement
private def within_079_075 : CountSegment 1 10 13120 10188 10192 4208 4212 := CountSegment.join within_079_073 within_079_074
private theorem within_079_075_agreement : RowAgreement 10188 10192 := RowAgreement.join within_079_073_agreement within_079_074_agreement
private def within_079_076 : CountSegment 1 10 13120 10184 10192 4205 4212 := CountSegment.join within_079_072 within_079_075
private theorem within_079_076_agreement : RowAgreement 10184 10192 := RowAgreement.join within_079_072_agreement within_079_075_agreement
private def within_079_077 : CountSegment 1 10 13120 10176 10192 4201 4212 := CountSegment.join within_079_069 within_079_076
private theorem within_079_077_agreement : RowAgreement 10176 10192 := RowAgreement.join within_079_069_agreement within_079_076_agreement
private def within_079_078 : CountSegment 1 10 13120 10192 10194 4212 4213 := CountSegment.join atomic_10193 atomic_10194
private theorem within_079_078_agreement : RowAgreement 10192 10194 := RowAgreement.join atomic_10193_agreement atomic_10194_agreement
private def within_079_079 : CountSegment 1 10 13120 10194 10196 4213 4213 := CountSegment.join atomic_10195 atomic_10196
private theorem within_079_079_agreement : RowAgreement 10194 10196 := RowAgreement.join atomic_10195_agreement atomic_10196_agreement
private def within_079_080 : CountSegment 1 10 13120 10192 10196 4212 4213 := CountSegment.join within_079_078 within_079_079
private theorem within_079_080_agreement : RowAgreement 10192 10196 := RowAgreement.join within_079_078_agreement within_079_079_agreement
private def within_079_081 : CountSegment 1 10 13120 10196 10198 4213 4213 := CountSegment.join atomic_10197 atomic_10198
private theorem within_079_081_agreement : RowAgreement 10196 10198 := RowAgreement.join atomic_10197_agreement atomic_10198_agreement
private def within_079_082 : CountSegment 1 10 13120 10198 10200 4213 4215 := CountSegment.join atomic_10199 atomic_10200
private theorem within_079_082_agreement : RowAgreement 10198 10200 := RowAgreement.join atomic_10199_agreement atomic_10200_agreement
private def within_079_083 : CountSegment 1 10 13120 10196 10200 4213 4215 := CountSegment.join within_079_081 within_079_082
private theorem within_079_083_agreement : RowAgreement 10196 10200 := RowAgreement.join within_079_081_agreement within_079_082_agreement
private def within_079_084 : CountSegment 1 10 13120 10192 10200 4212 4215 := CountSegment.join within_079_080 within_079_083
private theorem within_079_084_agreement : RowAgreement 10192 10200 := RowAgreement.join within_079_080_agreement within_079_083_agreement
private def within_079_085 : CountSegment 1 10 13120 10200 10202 4215 4215 := CountSegment.join atomic_10201 atomic_10202
private theorem within_079_085_agreement : RowAgreement 10200 10202 := RowAgreement.join atomic_10201_agreement atomic_10202_agreement
private def within_079_086 : CountSegment 1 10 13120 10202 10204 4215 4216 := CountSegment.join atomic_10203 atomic_10204
private theorem within_079_086_agreement : RowAgreement 10202 10204 := RowAgreement.join atomic_10203_agreement atomic_10204_agreement
private def within_079_087 : CountSegment 1 10 13120 10200 10204 4215 4216 := CountSegment.join within_079_085 within_079_086
private theorem within_079_087_agreement : RowAgreement 10200 10204 := RowAgreement.join within_079_085_agreement within_079_086_agreement
private def within_079_088 : CountSegment 1 10 13120 10204 10206 4216 4216 := CountSegment.join atomic_10205 atomic_10206
private theorem within_079_088_agreement : RowAgreement 10204 10206 := RowAgreement.join atomic_10205_agreement atomic_10206_agreement
private def within_079_089 : CountSegment 1 10 13120 10206 10208 4216 4217 := CountSegment.join atomic_10207 atomic_10208
private theorem within_079_089_agreement : RowAgreement 10206 10208 := RowAgreement.join atomic_10207_agreement atomic_10208_agreement
private def within_079_090 : CountSegment 1 10 13120 10204 10208 4216 4217 := CountSegment.join within_079_088 within_079_089
private theorem within_079_090_agreement : RowAgreement 10204 10208 := RowAgreement.join within_079_088_agreement within_079_089_agreement
private def within_079_091 : CountSegment 1 10 13120 10200 10208 4215 4217 := CountSegment.join within_079_087 within_079_090
private theorem within_079_091_agreement : RowAgreement 10200 10208 := RowAgreement.join within_079_087_agreement within_079_090_agreement
private def within_079_092 : CountSegment 1 10 13120 10192 10208 4212 4217 := CountSegment.join within_079_084 within_079_091
private theorem within_079_092_agreement : RowAgreement 10192 10208 := RowAgreement.join within_079_084_agreement within_079_091_agreement
private def within_079_093 : CountSegment 1 10 13120 10176 10208 4201 4217 := CountSegment.join within_079_077 within_079_092
private theorem within_079_093_agreement : RowAgreement 10176 10208 := RowAgreement.join within_079_077_agreement within_079_092_agreement
private def within_079_094 : CountSegment 1 10 13120 10208 10210 4217 4217 := CountSegment.join atomic_10209 atomic_10210
private theorem within_079_094_agreement : RowAgreement 10208 10210 := RowAgreement.join atomic_10209_agreement atomic_10210_agreement
private def within_079_095 : CountSegment 1 10 13120 10210 10212 4217 4218 := CountSegment.join atomic_10211 atomic_10212
private theorem within_079_095_agreement : RowAgreement 10210 10212 := RowAgreement.join atomic_10211_agreement atomic_10212_agreement
private def within_079_096 : CountSegment 1 10 13120 10208 10212 4217 4218 := CountSegment.join within_079_094 within_079_095
private theorem within_079_096_agreement : RowAgreement 10208 10212 := RowAgreement.join within_079_094_agreement within_079_095_agreement
private def within_079_097 : CountSegment 1 10 13120 10212 10214 4218 4218 := CountSegment.join atomic_10213 atomic_10214
private theorem within_079_097_agreement : RowAgreement 10212 10214 := RowAgreement.join atomic_10213_agreement atomic_10214_agreement
private def within_079_098 : CountSegment 1 10 13120 10214 10216 4218 4218 := CountSegment.join atomic_10215 atomic_10216
private theorem within_079_098_agreement : RowAgreement 10214 10216 := RowAgreement.join atomic_10215_agreement atomic_10216_agreement
private def within_079_099 : CountSegment 1 10 13120 10212 10216 4218 4218 := CountSegment.join within_079_097 within_079_098
private theorem within_079_099_agreement : RowAgreement 10212 10216 := RowAgreement.join within_079_097_agreement within_079_098_agreement
private def within_079_100 : CountSegment 1 10 13120 10208 10216 4217 4218 := CountSegment.join within_079_096 within_079_099
private theorem within_079_100_agreement : RowAgreement 10208 10216 := RowAgreement.join within_079_096_agreement within_079_099_agreement
private def within_079_101 : CountSegment 1 10 13120 10216 10218 4218 4218 := CountSegment.join atomic_10217 atomic_10218
private theorem within_079_101_agreement : RowAgreement 10216 10218 := RowAgreement.join atomic_10217_agreement atomic_10218_agreement
private def within_079_102 : CountSegment 1 10 13120 10218 10220 4218 4219 := CountSegment.join atomic_10219 atomic_10220
private theorem within_079_102_agreement : RowAgreement 10218 10220 := RowAgreement.join atomic_10219_agreement atomic_10220_agreement
private def within_079_103 : CountSegment 1 10 13120 10216 10220 4218 4219 := CountSegment.join within_079_101 within_079_102
private theorem within_079_103_agreement : RowAgreement 10216 10220 := RowAgreement.join within_079_101_agreement within_079_102_agreement
private def within_079_104 : CountSegment 1 10 13120 10220 10222 4219 4219 := CountSegment.join atomic_10221 atomic_10222
private theorem within_079_104_agreement : RowAgreement 10220 10222 := RowAgreement.join atomic_10221_agreement atomic_10222_agreement
private def within_079_105 : CountSegment 1 10 13120 10222 10224 4219 4220 := CountSegment.join atomic_10223 atomic_10224
private theorem within_079_105_agreement : RowAgreement 10222 10224 := RowAgreement.join atomic_10223_agreement atomic_10224_agreement
private def within_079_106 : CountSegment 1 10 13120 10220 10224 4219 4220 := CountSegment.join within_079_104 within_079_105
private theorem within_079_106_agreement : RowAgreement 10220 10224 := RowAgreement.join within_079_104_agreement within_079_105_agreement
private def within_079_107 : CountSegment 1 10 13120 10216 10224 4218 4220 := CountSegment.join within_079_103 within_079_106
private theorem within_079_107_agreement : RowAgreement 10216 10224 := RowAgreement.join within_079_103_agreement within_079_106_agreement
private def within_079_108 : CountSegment 1 10 13120 10208 10224 4217 4220 := CountSegment.join within_079_100 within_079_107
private theorem within_079_108_agreement : RowAgreement 10208 10224 := RowAgreement.join within_079_100_agreement within_079_107_agreement
private def within_079_109 : CountSegment 1 10 13120 10224 10226 4220 4220 := CountSegment.join atomic_10225 atomic_10226
private theorem within_079_109_agreement : RowAgreement 10224 10226 := RowAgreement.join atomic_10225_agreement atomic_10226_agreement
private def within_079_110 : CountSegment 1 10 13120 10226 10228 4220 4220 := CountSegment.join atomic_10227 atomic_10228
private theorem within_079_110_agreement : RowAgreement 10226 10228 := RowAgreement.join atomic_10227_agreement atomic_10228_agreement
private def within_079_111 : CountSegment 1 10 13120 10224 10228 4220 4220 := CountSegment.join within_079_109 within_079_110
private theorem within_079_111_agreement : RowAgreement 10224 10228 := RowAgreement.join within_079_109_agreement within_079_110_agreement
private def within_079_112 : CountSegment 1 10 13120 10228 10230 4220 4221 := CountSegment.join atomic_10229 atomic_10230
private theorem within_079_112_agreement : RowAgreement 10228 10230 := RowAgreement.join atomic_10229_agreement atomic_10230_agreement
private def within_079_113 : CountSegment 1 10 13120 10230 10232 4221 4222 := CountSegment.join atomic_10231 atomic_10232
private theorem within_079_113_agreement : RowAgreement 10230 10232 := RowAgreement.join atomic_10231_agreement atomic_10232_agreement
private def within_079_114 : CountSegment 1 10 13120 10228 10232 4220 4222 := CountSegment.join within_079_112 within_079_113
private theorem within_079_114_agreement : RowAgreement 10228 10232 := RowAgreement.join within_079_112_agreement within_079_113_agreement
private def within_079_115 : CountSegment 1 10 13120 10224 10232 4220 4222 := CountSegment.join within_079_111 within_079_114
private theorem within_079_115_agreement : RowAgreement 10224 10232 := RowAgreement.join within_079_111_agreement within_079_114_agreement
private def within_079_116 : CountSegment 1 10 13120 10232 10234 4222 4222 := CountSegment.join atomic_10233 atomic_10234
private theorem within_079_116_agreement : RowAgreement 10232 10234 := RowAgreement.join atomic_10233_agreement atomic_10234_agreement
private def within_079_117 : CountSegment 1 10 13120 10234 10236 4222 4222 := CountSegment.join atomic_10235 atomic_10236
private theorem within_079_117_agreement : RowAgreement 10234 10236 := RowAgreement.join atomic_10235_agreement atomic_10236_agreement
private def within_079_118 : CountSegment 1 10 13120 10232 10236 4222 4222 := CountSegment.join within_079_116 within_079_117
private theorem within_079_118_agreement : RowAgreement 10232 10236 := RowAgreement.join within_079_116_agreement within_079_117_agreement
private def within_079_119 : CountSegment 1 10 13120 10236 10238 4222 4223 := CountSegment.join atomic_10237 atomic_10238
private theorem within_079_119_agreement : RowAgreement 10236 10238 := RowAgreement.join atomic_10237_agreement atomic_10238_agreement
private def within_079_120 : CountSegment 1 10 13120 10238 10240 4223 4224 := CountSegment.join atomic_10239 atomic_10240
private theorem within_079_120_agreement : RowAgreement 10238 10240 := RowAgreement.join atomic_10239_agreement atomic_10240_agreement
private def within_079_121 : CountSegment 1 10 13120 10236 10240 4222 4224 := CountSegment.join within_079_119 within_079_120
private theorem within_079_121_agreement : RowAgreement 10236 10240 := RowAgreement.join within_079_119_agreement within_079_120_agreement
private def within_079_122 : CountSegment 1 10 13120 10232 10240 4222 4224 := CountSegment.join within_079_118 within_079_121
private theorem within_079_122_agreement : RowAgreement 10232 10240 := RowAgreement.join within_079_118_agreement within_079_121_agreement
private def within_079_123 : CountSegment 1 10 13120 10224 10240 4220 4224 := CountSegment.join within_079_115 within_079_122
private theorem within_079_123_agreement : RowAgreement 10224 10240 := RowAgreement.join within_079_115_agreement within_079_122_agreement
private def within_079_124 : CountSegment 1 10 13120 10208 10240 4217 4224 := CountSegment.join within_079_108 within_079_123
private theorem within_079_124_agreement : RowAgreement 10208 10240 := RowAgreement.join within_079_108_agreement within_079_123_agreement
private def within_079_125 : CountSegment 1 10 13120 10176 10240 4201 4224 := CountSegment.join within_079_093 within_079_124
private theorem within_079_125_agreement : RowAgreement 10176 10240 := RowAgreement.join within_079_093_agreement within_079_124_agreement
private def within_079_126 : CountSegment 1 10 13120 10112 10240 4181 4224 := CountSegment.join within_079_062 within_079_125
private theorem within_079_126_agreement : RowAgreement 10112 10240 := RowAgreement.join within_079_062_agreement within_079_125_agreement
def segment_079 : CountSegment 1 10 13120 10112 10240 4181 4224 := within_079_126
theorem agreement_079 : RowAgreement 10112 10240 := within_079_126_agreement
end ProjectEulerSolutions.P58Numerical
