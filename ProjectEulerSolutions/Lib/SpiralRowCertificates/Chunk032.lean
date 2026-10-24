import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_4097 : RowCertificate 4097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4097 1) := by change ¬ Nat.Prime 67149831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22383277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4097 2) := by change Nat.Prime 67141637; exact ProjectEulerSolutions.P58PrimeNodes.p_67141637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4097 3) := by change ¬ Nat.Prime 67133443; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5164111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4097 : row 4097 = 1 := (row_4097.checked).trans rfl
theorem model_count_4097 : modelRow 4097 = 1 := by decide
theorem modelrow_agrees_4097 : modelRow 4097 = row 4097 := model_count_4097.trans row_count_4097.symm
def row_4098 : RowCertificate 4098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4098 1) := by change ¬ Nat.Prime 67182613; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3535927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4098 2) := by change Nat.Prime 67174417; exact ProjectEulerSolutions.P58PrimeNodes.p_67174417.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4098 3) := by change Nat.Prime 67166221; exact ProjectEulerSolutions.P58PrimeNodes.p_67166221.prime
    exact decide_eq_true prime
theorem row_count_4098 : row 4098 = 2 := (row_4098.checked).trans rfl
theorem model_count_4098 : modelRow 4098 = 2 := by decide
theorem modelrow_agrees_4098 : modelRow 4098 = row 4098 := model_count_4098.trans row_count_4098.symm
def row_4099 : RowCertificate 4099 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4099 1) := by change Nat.Prime 67215403; exact ProjectEulerSolutions.P58PrimeNodes.p_67215403.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4099 2) := by change ¬ Nat.Prime 67207205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13441441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4099 3) := by change ¬ Nat.Prime 67199007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22399669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4099 : row 4099 = 1 := (row_4099.checked).trans rfl
theorem model_count_4099 : modelRow 4099 = 1 := by decide
theorem modelrow_agrees_4099 : modelRow 4099 = row 4099 := model_count_4099.trans row_count_4099.symm
def row_4100 : RowCertificate 4100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4100 1) := by change ¬ Nat.Prime 67248201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22416067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4100 2) := by change ¬ Nat.Prime 67240001; exact PrattCertificates.composite_of_product (factor := 2789) (cofactor := 24109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4100 3) := by change ¬ Nat.Prime 67231801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9604543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4100 : row 4100 = 0 := (row_4100.checked).trans rfl
theorem model_count_4100 : modelRow 4100 = 0 := by decide
theorem modelrow_agrees_4100 : modelRow 4100 = row 4100 := model_count_4100.trans row_count_4100.symm
def row_4101 : RowCertificate 4101 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4101 1) := by change Nat.Prime 67281007; exact ProjectEulerSolutions.P58PrimeNodes.p_67281007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4101 2) := by change ¬ Nat.Prime 67272805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13454561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4101 3) := by change ¬ Nat.Prime 67264603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9609229) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4101 : row 4101 = 1 := (row_4101.checked).trans rfl
theorem model_count_4101 : modelRow 4101 = 1 := by decide
theorem modelrow_agrees_4101 : modelRow 4101 = row 4101 := model_count_4101.trans row_count_4101.symm
def row_4102 : RowCertificate 4102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4102 1) := by change ¬ Nat.Prime 67313821; exact PrattCertificates.composite_of_product (factor := 3583) (cofactor := 18787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4102 2) := by change ¬ Nat.Prime 67305617; exact PrattCertificates.composite_of_product (factor := 6197) (cofactor := 10861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4102 3) := by change ¬ Nat.Prime 67297413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22432471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4102 : row 4102 = 0 := (row_4102.checked).trans rfl
theorem model_count_4102 : modelRow 4102 = 0 := by decide
theorem modelrow_agrees_4102 : modelRow 4102 = row 4102 := model_count_4102.trans row_count_4102.symm
def row_4103 : RowCertificate 4103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4103 1) := by change ¬ Nat.Prime 67346643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22448881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4103 2) := by change Nat.Prime 67338437; exact ProjectEulerSolutions.P58PrimeNodes.p_67338437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4103 3) := by change ¬ Nat.Prime 67330231; exact PrattCertificates.composite_of_product (factor := 5059) (cofactor := 13309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4103 : row 4103 = 1 := (row_4103.checked).trans rfl
theorem model_count_4103 : modelRow 4103 = 1 := by decide
theorem modelrow_agrees_4103 : modelRow 4103 = row 4103 := model_count_4103.trans row_count_4103.symm
def row_4104 : RowCertificate 4104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4104 1) := by change ¬ Nat.Prime 67379473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9625639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4104 2) := by change ¬ Nat.Prime 67371265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13474253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4104 3) := by change ¬ Nat.Prime 67363057; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 42181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4104 : row 4104 = 0 := (row_4104.checked).trans rfl
theorem model_count_4104 : modelRow 4104 = 0 := by decide
theorem modelrow_agrees_4104 : modelRow 4104 = row 4104 := model_count_4104.trans row_count_4104.symm
def row_4105 : RowCertificate 4105 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4105 1) := by change Nat.Prime 67412311; exact ProjectEulerSolutions.P58PrimeNodes.p_67412311.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4105 2) := by change Nat.Prime 67404101; exact ProjectEulerSolutions.P58PrimeNodes.p_67404101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4105 3) := by change ¬ Nat.Prime 67395891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22465297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4105 : row 4105 = 2 := (row_4105.checked).trans rfl
theorem model_count_4105 : modelRow 4105 = 2 := by decide
theorem modelrow_agrees_4105 : modelRow 4105 = row 4105 := model_count_4105.trans row_count_4105.symm
def row_4106 : RowCertificate 4106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4106 1) := by change ¬ Nat.Prime 67445157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22481719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4106 2) := by change ¬ Nat.Prime 67436945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13487389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4106 3) := by change ¬ Nat.Prime 67428733; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1006399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4106 : row 4106 = 0 := (row_4106.checked).trans rfl
theorem model_count_4106 : modelRow 4106 = 0 := by decide
theorem modelrow_agrees_4106 : modelRow 4106 = row 4106 := model_count_4106.trans row_count_4106.symm
def row_4107 : RowCertificate 4107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4107 1) := by change ¬ Nat.Prime 67478011; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 349627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4107 2) := by change ¬ Nat.Prime 67469797; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 61729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4107 3) := by change ¬ Nat.Prime 67461583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9637369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4107 : row 4107 = 0 := (row_4107.checked).trans rfl
theorem model_count_4107 : modelRow 4107 = 0 := by decide
theorem modelrow_agrees_4107 : modelRow 4107 = row 4107 := model_count_4107.trans row_count_4107.symm
def row_4108 : RowCertificate 4108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4108 1) := by change ¬ Nat.Prime 67510873; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 200329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4108 2) := by change ¬ Nat.Prime 67502657; exact PrattCertificates.composite_of_product (factor := 2609) (cofactor := 25873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4108 3) := by change ¬ Nat.Prime 67494441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22498147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4108 : row 4108 = 0 := (row_4108.checked).trans rfl
theorem model_count_4108 : modelRow 4108 = 0 := by decide
theorem modelrow_agrees_4108 : modelRow 4108 = row 4108 := model_count_4108.trans row_count_4108.symm
def row_4109 : RowCertificate 4109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4109 1) := by change ¬ Nat.Prime 67543743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22514581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4109 2) := by change ¬ Nat.Prime 67535525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13507105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4109 3) := by change Nat.Prime 67527307; exact ProjectEulerSolutions.P58PrimeNodes.p_67527307.prime
    exact decide_eq_true prime
theorem row_count_4109 : row 4109 = 1 := (row_4109.checked).trans rfl
theorem model_count_4109 : modelRow 4109 = 1 := by decide
theorem modelrow_agrees_4109 : modelRow 4109 = row 4109 := model_count_4109.trans row_count_4109.symm
def row_4110 : RowCertificate 4110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4110 1) := by change ¬ Nat.Prime 67576621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9653803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4110 2) := by change ¬ Nat.Prime 67568401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1826173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4110 3) := by change ¬ Nat.Prime 67560181; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5196937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4110 : row 4110 = 0 := (row_4110.checked).trans rfl
theorem model_count_4110 : modelRow 4110 = 0 := by decide
theorem modelrow_agrees_4110 : modelRow 4110 = row 4110 := model_count_4110.trans row_count_4110.symm
def row_4111 : RowCertificate 4111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4111 1) := by change ¬ Nat.Prime 67609507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9658501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4111 2) := by change ¬ Nat.Prime 67601285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13520257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4111 3) := by change ¬ Nat.Prime 67593063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22531021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4111 : row 4111 = 0 := (row_4111.checked).trans rfl
theorem model_count_4111 : modelRow 4111 = 0 := by decide
theorem modelrow_agrees_4111 : modelRow 4111 = row 4111 := model_count_4111.trans row_count_4111.symm
def row_4112 : RowCertificate 4112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4112 1) := by change ¬ Nat.Prime 67642401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22547467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4112 2) := by change ¬ Nat.Prime 67634177; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5202629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4112 3) := by change ¬ Nat.Prime 67625953; exact PrattCertificates.composite_of_product (factor := 1321) (cofactor := 51193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4112 : row 4112 = 0 := (row_4112.checked).trans rfl
theorem model_count_4112 : modelRow 4112 = 0 := by decide
theorem modelrow_agrees_4112 : modelRow 4112 = row 4112 := model_count_4112.trans row_count_4112.symm
def row_4113 : RowCertificate 4113 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4113 1) := by change Nat.Prime 67675303; exact ProjectEulerSolutions.P58PrimeNodes.p_67675303.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4113 2) := by change Nat.Prime 67667077; exact ProjectEulerSolutions.P58PrimeNodes.p_67667077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4113 3) := by change ¬ Nat.Prime 67658851; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5204527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4113 : row 4113 = 2 := (row_4113.checked).trans rfl
theorem model_count_4113 : modelRow 4113 = 2 := by decide
theorem modelrow_agrees_4113 : modelRow 4113 = row 4113 := model_count_4113.trans row_count_4113.symm
def row_4114 : RowCertificate 4114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4114 1) := by change ¬ Nat.Prime 67708213; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 102433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4114 2) := by change ¬ Nat.Prime 67699985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13539997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4114 3) := by change ¬ Nat.Prime 67691757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22563919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4114 : row 4114 = 0 := (row_4114.checked).trans rfl
theorem model_count_4114 : modelRow 4114 = 0 := by decide
theorem modelrow_agrees_4114 : modelRow 4114 = row 4114 := model_count_4114.trans row_count_4114.symm
def row_4115 : RowCertificate 4115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4115 1) := by change ¬ Nat.Prime 67741131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22580377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4115 2) := by change Nat.Prime 67732901; exact ProjectEulerSolutions.P58PrimeNodes.p_67732901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4115 3) := by change ¬ Nat.Prime 67724671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9674953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4115 : row 4115 = 1 := (row_4115.checked).trans rfl
theorem model_count_4115 : modelRow 4115 = 1 := by decide
theorem modelrow_agrees_4115 : modelRow 4115 = row 4115 := model_count_4115.trans row_count_4115.symm
def row_4116 : RowCertificate 4116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4116 1) := by change ¬ Nat.Prime 67774057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5213389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4116 2) := by change ¬ Nat.Prime 67765825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13553165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4116 3) := by change Nat.Prime 67757593; exact ProjectEulerSolutions.P58PrimeNodes.p_67757593.prime
    exact decide_eq_true prime
theorem row_count_4116 : row 4116 = 1 := (row_4116.checked).trans rfl
theorem model_count_4116 : modelRow 4116 = 1 := by decide
theorem modelrow_agrees_4116 : modelRow 4116 = row 4116 := model_count_4116.trans row_count_4116.symm
def row_4117 : RowCertificate 4117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4117 1) := by change ¬ Nat.Prime 67806991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9686713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4117 2) := by change ¬ Nat.Prime 67798757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5215289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4117 3) := by change ¬ Nat.Prime 67790523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22596841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4117 : row 4117 = 0 := (row_4117.checked).trans rfl
theorem model_count_4117 : modelRow 4117 = 0 := by decide
theorem modelrow_agrees_4117 : modelRow 4117 = row 4117 := model_count_4117.trans row_count_4117.symm
def row_4118 : RowCertificate 4118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4118 1) := by change ¬ Nat.Prime 67839933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22613311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4118 2) := by change ¬ Nat.Prime 67831697; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 201281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4118 3) := by change ¬ Nat.Prime 67823461; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 534043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4118 : row 4118 = 0 := (row_4118.checked).trans rfl
theorem model_count_4118 : modelRow 4118 = 0 := by decide
theorem modelrow_agrees_4118 : modelRow 4118 = row 4118 := model_count_4118.trans row_count_4118.symm
def row_4119 : RowCertificate 4119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4119 1) := by change ¬ Nat.Prime 67872883; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5220991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4119 2) := by change ¬ Nat.Prime 67864645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13572929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4119 3) := by change Nat.Prime 67856407; exact ProjectEulerSolutions.P58PrimeNodes.p_67856407.prime
    exact decide_eq_true prime
theorem row_count_4119 : row 4119 = 1 := (row_4119.checked).trans rfl
theorem model_count_4119 : modelRow 4119 = 1 := by decide
theorem modelrow_agrees_4119 : modelRow 4119 = row 4119 := model_count_4119.trans row_count_4119.symm
def row_4120 : RowCertificate 4120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4120 1) := by change ¬ Nat.Prime 67905841; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2190511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4120 2) := by change ¬ Nat.Prime 67897601; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 264193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4120 3) := by change ¬ Nat.Prime 67889361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22629787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4120 : row 4120 = 0 := (row_4120.checked).trans rfl
theorem model_count_4120 : modelRow 4120 = 0 := by decide
theorem modelrow_agrees_4120 : modelRow 4120 = row 4120 := model_count_4120.trans row_count_4120.symm
def row_4121 : RowCertificate 4121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4121 1) := by change ¬ Nat.Prime 67938807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22646269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4121 2) := by change ¬ Nat.Prime 67930565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13586113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4121 3) := by change ¬ Nat.Prime 67922323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9703189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4121 : row 4121 = 0 := (row_4121.checked).trans rfl
theorem model_count_4121 : modelRow 4121 = 0 := by decide
theorem modelrow_agrees_4121 : modelRow 4121 = row 4121 := model_count_4121.trans row_count_4121.symm
def row_4122 : RowCertificate 4122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4122 1) := by change ¬ Nat.Prime 67971781; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 125641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4122 2) := by change ¬ Nat.Prime 67963537; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 291689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4122 3) := by change ¬ Nat.Prime 67955293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9707899) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4122 : row 4122 = 0 := (row_4122.checked).trans rfl
theorem model_count_4122 : modelRow 4122 = 0 := by decide
theorem modelrow_agrees_4122 : modelRow 4122 = row 4122 := model_count_4122.trans row_count_4122.symm
def row_4123 : RowCertificate 4123 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4123 1) := by change ¬ Nat.Prime 68004763; exact PrattCertificates.composite_of_product (factor := 631) (cofactor := 107773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4123 2) := by change ¬ Nat.Prime 67996517; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1114697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4123 3) := by change ¬ Nat.Prime 67988271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22662757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4123 : row 4123 = 0 := (row_4123.checked).trans rfl
theorem model_count_4123 : modelRow 4123 = 0 := by decide
theorem modelrow_agrees_4123 : modelRow 4123 = row 4123 := model_count_4123.trans row_count_4123.symm
def row_4124 : RowCertificate 4124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4124 1) := by change ¬ Nat.Prime 68037753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22679251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4124 2) := by change ¬ Nat.Prime 68029505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13605901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4124 3) := by change Nat.Prime 68021257; exact ProjectEulerSolutions.P58PrimeNodes.p_68021257.prime
    exact decide_eq_true prime
theorem row_count_4124 : row 4124 = 1 := (row_4124.checked).trans rfl
theorem model_count_4124 : modelRow 4124 = 1 := by decide
theorem modelrow_agrees_4124 : modelRow 4124 = row 4124 := model_count_4124.trans row_count_4124.symm
def row_4125 : RowCertificate 4125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4125 1) := by change ¬ Nat.Prime 68070751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9724393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4125 2) := by change ¬ Nat.Prime 68062501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5235577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4125 3) := by change ¬ Nat.Prime 68054251; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1582657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4125 : row 4125 = 0 := (row_4125.checked).trans rfl
theorem model_count_4125 : modelRow 4125 = 0 := by decide
theorem modelrow_agrees_4125 : modelRow 4125 = row 4125 := model_count_4125.trans row_count_4125.symm
def row_4126 : RowCertificate 4126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4126 1) := by change ¬ Nat.Prime 68103757; exact PrattCertificates.composite_of_product (factor := 2383) (cofactor := 28579) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4126 2) := by change ¬ Nat.Prime 68095505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13619101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4126 3) := by change ¬ Nat.Prime 68087253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22695751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4126 : row 4126 = 0 := (row_4126.checked).trans rfl
theorem model_count_4126 : modelRow 4126 = 0 := by decide
theorem modelrow_agrees_4126 : modelRow 4126 = row 4126 := model_count_4126.trans row_count_4126.symm
def row_4127 : RowCertificate 4127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4127 1) := by change ¬ Nat.Prime 68136771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22712257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4127 2) := by change Nat.Prime 68128517; exact ProjectEulerSolutions.P58PrimeNodes.p_68128517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4127 3) := by change ¬ Nat.Prime 68120263; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3585277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4127 : row 4127 = 1 := (row_4127.checked).trans rfl
theorem model_count_4127 : modelRow 4127 = 1 := by decide
theorem modelrow_agrees_4127 : modelRow 4127 = row 4127 := model_count_4127.trans row_count_4127.symm
def row_4128 : RowCertificate 4128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4128 1) := by change ¬ Nat.Prime 68169793; exact PrattCertificates.composite_of_product (factor := 3547) (cofactor := 19219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4128 2) := by change ¬ Nat.Prime 68161537; exact PrattCertificates.composite_of_product (factor := 3877) (cofactor := 17581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4128 3) := by change ¬ Nat.Prime 68153281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9736183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4128 : row 4128 = 0 := (row_4128.checked).trans rfl
theorem model_count_4128 : modelRow 4128 = 0 := by decide
theorem modelrow_agrees_4128 : modelRow 4128 = row 4128 := model_count_4128.trans row_count_4128.symm
def row_4129 : RowCertificate 4129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4129 1) := by change ¬ Nat.Prime 68202823; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5246371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4129 2) := by change ¬ Nat.Prime 68194565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13638913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4129 3) := by change ¬ Nat.Prime 68186307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22728769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4129 : row 4129 = 0 := (row_4129.checked).trans rfl
theorem model_count_4129 : modelRow 4129 = 0 := by decide
theorem modelrow_agrees_4129 : modelRow 4129 = row 4129 := model_count_4129.trans row_count_4129.symm
def row_4130 : RowCertificate 4130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4130 1) := by change ¬ Nat.Prime 68235861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22745287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4130 2) := by change ¬ Nat.Prime 68227601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5248277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4130 3) := by change Nat.Prime 68219341; exact ProjectEulerSolutions.P58PrimeNodes.p_68219341.prime
    exact decide_eq_true prime
theorem row_count_4130 : row 4130 = 1 := (row_4130.checked).trans rfl
theorem model_count_4130 : modelRow 4130 = 1 := by decide
theorem modelrow_agrees_4130 : modelRow 4130 = row 4130 := model_count_4130.trans row_count_4130.symm
def row_4131 : RowCertificate 4131 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4131 1) := by change ¬ Nat.Prime 68268907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9752701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4131 2) := by change ¬ Nat.Prime 68260645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13652129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4131 3) := by change ¬ Nat.Prime 68252383; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1844659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4131 : row 4131 = 0 := (row_4131.checked).trans rfl
theorem model_count_4131 : modelRow 4131 = 0 := by decide
theorem modelrow_agrees_4131 : modelRow 4131 = row 4131 := model_count_4131.trans row_count_4131.symm
def row_4132 : RowCertificate 4132 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4132 1) := by change ¬ Nat.Prime 68301961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9757423) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4132 2) := by change ¬ Nat.Prime 68293697; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 604369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4132 3) := by change ¬ Nat.Prime 68285433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22761811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4132 : row 4132 = 0 := (row_4132.checked).trans rfl
theorem model_count_4132 : modelRow 4132 = 0 := by decide
theorem modelrow_agrees_4132 : modelRow 4132 = row 4132 := model_count_4132.trans row_count_4132.symm
def row_4133 : RowCertificate 4133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4133 1) := by change ¬ Nat.Prime 68335023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22778341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4133 2) := by change ¬ Nat.Prime 68326757; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4019221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4133 3) := by change ¬ Nat.Prime 68318491; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 343309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4133 : row 4133 = 0 := (row_4133.checked).trans rfl
theorem model_count_4133 : modelRow 4133 = 0 := by decide
theorem modelrow_agrees_4133 : modelRow 4133 = row 4133 := model_count_4133.trans row_count_4133.symm
def row_4134 : RowCertificate 4134 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4134 1) := by change Nat.Prime 68368093; exact ProjectEulerSolutions.P58PrimeNodes.p_68368093.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4134 2) := by change ¬ Nat.Prime 68359825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13671965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4134 3) := by change Nat.Prime 68351557; exact ProjectEulerSolutions.P58PrimeNodes.p_68351557.prime
    exact decide_eq_true prime
theorem row_count_4134 : row 4134 = 2 := (row_4134.checked).trans rfl
theorem model_count_4134 : modelRow 4134 = 2 := by decide
theorem modelrow_agrees_4134 : modelRow 4134 = row 4134 := model_count_4134.trans row_count_4134.symm
def row_4135 : RowCertificate 4135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4135 1) := by change ¬ Nat.Prime 68401171; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1020913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4135 2) := by change Nat.Prime 68392901; exact ProjectEulerSolutions.P58PrimeNodes.p_68392901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4135 3) := by change ¬ Nat.Prime 68384631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22794877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4135 : row 4135 = 1 := (row_4135.checked).trans rfl
theorem model_count_4135 : modelRow 4135 = 1 := by decide
theorem modelrow_agrees_4135 : modelRow 4135 = row 4135 := model_count_4135.trans row_count_4135.symm
def row_4136 : RowCertificate 4136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4136 1) := by change ¬ Nat.Prime 68434257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22811419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4136 2) := by change ¬ Nat.Prime 68425985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13685197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4136 3) := by change ¬ Nat.Prime 68417713; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9773959) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4136 : row 4136 = 0 := (row_4136.checked).trans rfl
theorem model_count_4136 : modelRow 4136 = 0 := by decide
theorem modelrow_agrees_4136 : modelRow 4136 = row 4136 := model_count_4136.trans row_count_4136.symm
def row_4137 : RowCertificate 4137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4137 1) := by change ¬ Nat.Prime 68467351; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 539113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4137 2) := by change ¬ Nat.Prime 68459077; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 172441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4137 3) := by change ¬ Nat.Prime 68450803; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 56431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4137 : row 4137 = 0 := (row_4137.checked).trans rfl
theorem model_count_4137 : modelRow 4137 = 0 := by decide
theorem modelrow_agrees_4137 : modelRow 4137 = row 4137 := model_count_4137.trans row_count_4137.symm
def row_4138 : RowCertificate 4138 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4138 1) := by change ¬ Nat.Prime 68500453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9785779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4138 2) := by change ¬ Nat.Prime 68492177; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5268629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4138 3) := by change ¬ Nat.Prime 68483901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22827967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4138 : row 4138 = 0 := (row_4138.checked).trans rfl
theorem model_count_4138 : modelRow 4138 = 0 := by decide
theorem modelrow_agrees_4138 : modelRow 4138 = row 4138 := model_count_4138.trans row_count_4138.symm
def row_4139 : RowCertificate 4139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4139 1) := by change ¬ Nat.Prime 68533563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22844521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4139 2) := by change ¬ Nat.Prime 68525285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13705057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4139 3) := by change ¬ Nat.Prime 68517007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5270539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4139 : row 4139 = 0 := (row_4139.checked).trans rfl
theorem model_count_4139 : modelRow 4139 = 0 := by decide
theorem modelrow_agrees_4139 : modelRow 4139 = row 4139 := model_count_4139.trans row_count_4139.symm
def row_4140 : RowCertificate 4140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4140 1) := by change ¬ Nat.Prime 68566681; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 706873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4140 2) := by change ¬ Nat.Prime 68558401; exact PrattCertificates.composite_of_product (factor := 6317) (cofactor := 10853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4140 3) := by change Nat.Prime 68550121; exact ProjectEulerSolutions.P58PrimeNodes.p_68550121.prime
    exact decide_eq_true prime
theorem row_count_4140 : row 4140 = 1 := (row_4140.checked).trans rfl
theorem model_count_4140 : modelRow 4140 = 1 := by decide
theorem modelrow_agrees_4140 : modelRow 4140 = row 4140 := model_count_4140.trans row_count_4140.symm
def row_4141 : RowCertificate 4141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4141 1) := by change ¬ Nat.Prime 68599807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2212897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4141 2) := by change ¬ Nat.Prime 68591525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13718305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4141 3) := by change ¬ Nat.Prime 68583243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22861081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4141 : row 4141 = 0 := (row_4141.checked).trans rfl
theorem model_count_4141 : modelRow 4141 = 0 := by decide
theorem modelrow_agrees_4141 : modelRow 4141 = row 4141 := model_count_4141.trans row_count_4141.symm
def row_4142 : RowCertificate 4142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4142 1) := by change ¬ Nat.Prime 68632941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22877647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4142 2) := by change ¬ Nat.Prime 68624657; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 176413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4142 3) := by change ¬ Nat.Prime 68616373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9802339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4142 : row 4142 = 0 := (row_4142.checked).trans rfl
theorem model_count_4142 : modelRow 4142 = 0 := by decide
theorem modelrow_agrees_4142 : modelRow 4142 = row 4142 := model_count_4142.trans row_count_4142.symm
def row_4143 : RowCertificate 4143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4143 1) := by change ¬ Nat.Prime 68666083; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 666661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4143 2) := by change ¬ Nat.Prime 68657797; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5281369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4143 3) := by change ¬ Nat.Prime 68649511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9807073) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4143 : row 4143 = 0 := (row_4143.checked).trans rfl
theorem model_count_4143 : modelRow 4143 = 0 := by decide
theorem modelrow_agrees_4143 : modelRow 4143 = row 4143 := model_count_4143.trans row_count_4143.symm
def row_4144 : RowCertificate 4144 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4144 1) := by change Nat.Prime 68699233; exact ProjectEulerSolutions.P58PrimeNodes.p_68699233.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4144 2) := by change ¬ Nat.Prime 68690945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13738189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4144 3) := by change ¬ Nat.Prime 68682657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22894219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4144 : row 4144 = 1 := (row_4144.checked).trans rfl
theorem model_count_4144 : modelRow 4144 = 1 := by decide
theorem modelrow_agrees_4144 : modelRow 4144 = row 4144 := model_count_4144.trans row_count_4144.symm
def row_4145 : RowCertificate 4145 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4145 1) := by change ¬ Nat.Prime 68732391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22910797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4145 2) := by change Nat.Prime 68724101; exact ProjectEulerSolutions.P58PrimeNodes.p_68724101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4145 3) := by change ¬ Nat.Prime 68715811; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 42391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4145 : row 4145 = 1 := (row_4145.checked).trans rfl
theorem model_count_4145 : modelRow 4145 = 1 := by decide
theorem modelrow_agrees_4145 : modelRow 4145 = row 4145 := model_count_4145.trans row_count_4145.symm
def row_4146 : RowCertificate 4146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4146 1) := by change ¬ Nat.Prime 68765557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9823651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4146 2) := by change ¬ Nat.Prime 68757265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13751453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4146 3) := by change ¬ Nat.Prime 68748973; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3618367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4146 : row 4146 = 0 := (row_4146.checked).trans rfl
theorem model_count_4146 : modelRow 4146 = 0 := by decide
theorem modelrow_agrees_4146 : modelRow 4146 = row 4146 := model_count_4146.trans row_count_4146.symm
def row_4147 : RowCertificate 4147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4147 1) := by change ¬ Nat.Prime 68798731; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 184447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4147 2) := by change ¬ Nat.Prime 68790437; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1859201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4147 3) := by change ¬ Nat.Prime 68782143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22927381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4147 : row 4147 = 0 := (row_4147.checked).trans rfl
theorem model_count_4147 : modelRow 4147 = 0 := by decide
theorem modelrow_agrees_4147 : modelRow 4147 = row 4147 := model_count_4147.trans row_count_4147.symm
def row_4148 : RowCertificate 4148 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4148 1) := by change ¬ Nat.Prime 68831913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22943971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4148 2) := by change Nat.Prime 68823617; exact ProjectEulerSolutions.P58PrimeNodes.p_68823617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4148 3) := by change ¬ Nat.Prime 68815321; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3621859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4148 : row 4148 = 1 := (row_4148.checked).trans rfl
theorem model_count_4148 : modelRow 4148 = 1 := by decide
theorem modelrow_agrees_4148 : modelRow 4148 = row 4148 := model_count_4148.trans row_count_4148.symm
def row_4149 : RowCertificate 4149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4149 1) := by change ¬ Nat.Prime 68865103; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1861219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4149 2) := by change ¬ Nat.Prime 68856805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13771361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4149 3) := by change ¬ Nat.Prime 68848507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9835501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4149 : row 4149 = 0 := (row_4149.checked).trans rfl
theorem model_count_4149 : modelRow 4149 = 0 := by decide
theorem modelrow_agrees_4149 : modelRow 4149 = row 4149 := model_count_4149.trans row_count_4149.symm
def row_4150 : RowCertificate 4150 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4150 1) := by change Nat.Prime 68898301; exact ProjectEulerSolutions.P58PrimeNodes.p_68898301.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4150 2) := by change ¬ Nat.Prime 68890001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4052353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4150 3) := by change ¬ Nat.Prime 68881701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22960567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4150 : row 4150 = 1 := (row_4150.checked).trans rfl
theorem model_count_4150 : modelRow 4150 = 1 := by decide
theorem modelrow_agrees_4150 : modelRow 4150 = row 4150 := model_count_4150.trans row_count_4150.symm
def row_4151 : RowCertificate 4151 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4151 1) := by change ¬ Nat.Prime 68931507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22977169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4151 2) := by change ¬ Nat.Prime 68923205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13784641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4151 3) := by change ¬ Nat.Prime 68914903; exact PrattCertificates.composite_of_product (factor := 1531) (cofactor := 45013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4151 : row 4151 = 0 := (row_4151.checked).trans rfl
theorem model_count_4151 : modelRow 4151 = 0 := by decide
theorem modelrow_agrees_4151 : modelRow 4151 = row 4151 := model_count_4151.trans row_count_4151.symm
def row_4152 : RowCertificate 4152 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4152 1) := by change ¬ Nat.Prime 68964721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9852103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4152 2) := by change Nat.Prime 68956417; exact ProjectEulerSolutions.P58PrimeNodes.p_68956417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4152 3) := by change ¬ Nat.Prime 68948113; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5303701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4152 : row 4152 = 1 := (row_4152.checked).trans rfl
theorem model_count_4152 : modelRow 4152 = 1 := by decide
theorem modelrow_agrees_4152 : modelRow 4152 = row 4152 := model_count_4152.trans row_count_4152.symm
def row_4153 : RowCertificate 4153 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4153 1) := by change ¬ Nat.Prime 68997943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9856849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4153 2) := by change ¬ Nat.Prime 68989637; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2378953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4153 3) := by change ¬ Nat.Prime 68981331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22993777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4153 : row 4153 = 0 := (row_4153.checked).trans rfl
theorem model_count_4153 : modelRow 4153 = 0 := by decide
theorem modelrow_agrees_4153 : modelRow 4153 = row 4153 := model_count_4153.trans row_count_4153.symm
def row_4154 : RowCertificate 4154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4154 1) := by change ¬ Nat.Prime 69031173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23010391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4154 2) := by change ¬ Nat.Prime 69022865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13804573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4154 3) := by change ¬ Nat.Prime 69014557; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 131959) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4154 : row 4154 = 0 := (row_4154.checked).trans rfl
theorem model_count_4154 : modelRow 4154 = 0 := by decide
theorem modelrow_agrees_4154 : modelRow 4154 = row 4154 := model_count_4154.trans row_count_4154.symm
def row_4155 : RowCertificate 4155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4155 1) := by change ¬ Nat.Prime 69064411; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5312647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4155 2) := by change ¬ Nat.Prime 69056101; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 185137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4155 3) := by change ¬ Nat.Prime 69047791; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1131931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4155 : row 4155 = 0 := (row_4155.checked).trans rfl
theorem model_count_4155 : modelRow 4155 = 0 := by decide
theorem modelrow_agrees_4155 : modelRow 4155 = row 4155 := model_count_4155.trans row_count_4155.symm
def row_4156 : RowCertificate 4156 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4156 1) := by change ¬ Nat.Prime 69097657; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 149239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4156 2) := by change ¬ Nat.Prime 69089345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13817869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4156 3) := by change ¬ Nat.Prime 69081033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23027011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4156 : row 4156 = 0 := (row_4156.checked).trans rfl
theorem model_count_4156 : modelRow 4156 = 0 := by decide
theorem modelrow_agrees_4156 : modelRow 4156 = row 4156 := model_count_4156.trans row_count_4156.symm
def row_4157 : RowCertificate 4157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4157 1) := by change ¬ Nat.Prime 69130911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23043637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4157 2) := by change ¬ Nat.Prime 69122597; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1685917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4157 3) := by change ¬ Nat.Prime 69114283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9873469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4157 : row 4157 = 0 := (row_4157.checked).trans rfl
theorem model_count_4157 : modelRow 4157 = 0 := by decide
theorem modelrow_agrees_4157 : modelRow 4157 = row 4157 := model_count_4157.trans row_count_4157.symm
def row_4158 : RowCertificate 4158 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4158 1) := by change ¬ Nat.Prime 69164173; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5320321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4158 2) := by change ¬ Nat.Prime 69155857; exact PrattCertificates.composite_of_product (factor := 881) (cofactor := 78497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4158 3) := by change Nat.Prime 69147541; exact ProjectEulerSolutions.P58PrimeNodes.p_69147541.prime
    exact decide_eq_true prime
theorem row_count_4158 : row 4158 = 1 := (row_4158.checked).trans rfl
theorem model_count_4158 : modelRow 4158 = 1 := by decide
theorem modelrow_agrees_4158 : modelRow 4158 = row 4158 := model_count_4158.trans row_count_4158.symm
def row_4159 : RowCertificate 4159 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4159 1) := by change ¬ Nat.Prime 69197443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9885349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4159 2) := by change ¬ Nat.Prime 69189125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13837825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4159 3) := by change ¬ Nat.Prime 69180807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23060269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4159 : row 4159 = 0 := (row_4159.checked).trans rfl
theorem model_count_4159 : modelRow 4159 = 0 := by decide
theorem modelrow_agrees_4159 : modelRow 4159 = row 4159 := model_count_4159.trans row_count_4159.symm
def row_4160 : RowCertificate 4160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4160 1) := by change ¬ Nat.Prime 69230721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23076907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4160 2) := by change ¬ Nat.Prime 69222401; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 713633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4160 3) := by change ¬ Nat.Prime 69214081; exact PrattCertificates.composite_of_product (factor := 3253) (cofactor := 21277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4160 : row 4160 = 0 := (row_4160.checked).trans rfl
theorem model_count_4160 : modelRow 4160 = 0 := by decide
theorem modelrow_agrees_4160 : modelRow 4160 = row 4160 := model_count_4160.trans row_count_4160.symm
def row_4161 : RowCertificate 4161 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4161 1) := by change Nat.Prime 69264007; exact ProjectEulerSolutions.P58PrimeNodes.p_69264007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4161 2) := by change ¬ Nat.Prime 69255685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13851137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4161 3) := by change Nat.Prime 69247363; exact ProjectEulerSolutions.P58PrimeNodes.p_69247363.prime
    exact decide_eq_true prime
theorem row_count_4161 : row 4161 = 2 := (row_4161.checked).trans rfl
theorem model_count_4161 : modelRow 4161 = 2 := by decide
theorem modelrow_agrees_4161 : modelRow 4161 = row 4161 := model_count_4161.trans row_count_4161.symm
def row_4162 : RowCertificate 4162 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4162 1) := by change Nat.Prime 69297301; exact ProjectEulerSolutions.P58PrimeNodes.p_69297301.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4162 2) := by change Nat.Prime 69288977; exact ProjectEulerSolutions.P58PrimeNodes.p_69288977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4162 3) := by change ¬ Nat.Prime 69280653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23093551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4162 : row 4162 = 2 := (row_4162.checked).trans rfl
theorem model_count_4162 : modelRow 4162 = 2 := by decide
theorem modelrow_agrees_4162 : modelRow 4162 = row 4162 := model_count_4162.trans row_count_4162.symm
def row_4163 : RowCertificate 4163 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4163 1) := by change ¬ Nat.Prime 69330603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23110201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4163 2) := by change ¬ Nat.Prime 69322277; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4077781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4163 3) := by change ¬ Nat.Prime 69313951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9901993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4163 : row 4163 = 0 := (row_4163.checked).trans rfl
theorem model_count_4163 : modelRow 4163 = 0 := by decide
theorem modelrow_agrees_4163 : modelRow 4163 = row 4163 := model_count_4163.trans row_count_4163.symm
def row_4164 : RowCertificate 4164 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4164 1) := by change ¬ Nat.Prime 69363913; exact PrattCertificates.composite_of_product (factor := 3301) (cofactor := 21013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4164 2) := by change ¬ Nat.Prime 69355585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13871117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4164 3) := by change ¬ Nat.Prime 69347257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9906751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4164 : row 4164 = 0 := (row_4164.checked).trans rfl
theorem model_count_4164 : modelRow 4164 = 0 := by decide
theorem modelrow_agrees_4164 : modelRow 4164 = row 4164 := model_count_4164.trans row_count_4164.symm
def row_4165 : RowCertificate 4165 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4165 1) := by change ¬ Nat.Prime 69397231; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 950647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4165 2) := by change ¬ Nat.Prime 69388901; exact PrattCertificates.composite_of_product (factor := 1933) (cofactor := 35897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4165 3) := by change ¬ Nat.Prime 69380571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23126857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4165 : row 4165 = 0 := (row_4165.checked).trans rfl
theorem model_count_4165 : modelRow 4165 = 0 := by decide
theorem modelrow_agrees_4165 : modelRow 4165 = row 4165 := model_count_4165.trans row_count_4165.symm
def row_4166 : RowCertificate 4166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4166 1) := by change ¬ Nat.Prime 69430557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23143519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4166 2) := by change ¬ Nat.Prime 69422225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13884445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4166 3) := by change ¬ Nat.Prime 69413893; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 103141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4166 : row 4166 = 0 := (row_4166.checked).trans rfl
theorem model_count_4166 : modelRow 4166 = 0 := by decide
theorem modelrow_agrees_4166 : modelRow 4166 = row 4166 := model_count_4166.trans row_count_4166.symm
def row_4167 : RowCertificate 4167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4167 1) := by change ¬ Nat.Prime 69463891; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9923413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4167 2) := by change ¬ Nat.Prime 69455557; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4085621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4167 3) := by change ¬ Nat.Prime 69447223; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3655117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4167 : row 4167 = 0 := (row_4167.checked).trans rfl
theorem model_count_4167 : modelRow 4167 = 0 := by decide
theorem modelrow_agrees_4167 : modelRow 4167 = row 4167 := model_count_4167.trans row_count_4167.symm
def row_4168 : RowCertificate 4168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4168 1) := by change ¬ Nat.Prime 69497233; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5345941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4168 2) := by change ¬ Nat.Prime 69488897; exact PrattCertificates.composite_of_product (factor := 3797) (cofactor := 18301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4168 3) := by change ¬ Nat.Prime 69480561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23160187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4168 : row 4168 = 0 := (row_4168.checked).trans rfl
theorem model_count_4168 : modelRow 4168 = 0 := by decide
theorem modelrow_agrees_4168 : modelRow 4168 = row 4168 := model_count_4168.trans row_count_4168.symm
def row_4169 : RowCertificate 4169 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4169 1) := by change ¬ Nat.Prime 69530583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23176861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4169 2) := by change ¬ Nat.Prime 69522245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13904449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4169 3) := by change ¬ Nat.Prime 69513907; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1037521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4169 : row 4169 = 0 := (row_4169.checked).trans rfl
theorem model_count_4169 : modelRow 4169 = 0 := by decide
theorem modelrow_agrees_4169 : modelRow 4169 = row 4169 := model_count_4169.trans row_count_4169.symm
def row_4170 : RowCertificate 4170 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4170 1) := by change ¬ Nat.Prime 69563941; exact PrattCertificates.composite_of_product (factor := 2221) (cofactor := 31321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4170 2) := by change ¬ Nat.Prime 69555601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2398469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4170 3) := by change ¬ Nat.Prime 69547261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9935323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4170 : row 4170 = 0 := (row_4170.checked).trans rfl
theorem model_count_4170 : modelRow 4170 = 0 := by decide
theorem modelrow_agrees_4170 : modelRow 4170 = row 4170 := model_count_4170.trans row_count_4170.symm
def row_4171 : RowCertificate 4171 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4171 1) := by change ¬ Nat.Prime 69597307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5353639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4171 2) := by change ¬ Nat.Prime 69588965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13917793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4171 3) := by change ¬ Nat.Prime 69580623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23193541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4171 : row 4171 = 0 := (row_4171.checked).trans rfl
theorem model_count_4171 : modelRow 4171 = 0 := by decide
theorem modelrow_agrees_4171 : modelRow 4171 = row 4171 := model_count_4171.trans row_count_4171.symm
def row_4172 : RowCertificate 4172 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4172 1) := by change ¬ Nat.Prime 69630681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23210227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4172 2) := by change ¬ Nat.Prime 69622337; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1313629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4172 3) := by change ¬ Nat.Prime 69613993; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1141213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4172 : row 4172 = 0 := (row_4172.checked).trans rfl
theorem model_count_4172 : modelRow 4172 = 0 := by decide
theorem modelrow_agrees_4172 : modelRow 4172 = row 4172 := model_count_4172.trans row_count_4172.symm
def row_4173 : RowCertificate 4173 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4173 1) := by change ¬ Nat.Prime 69664063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9952009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4173 2) := by change ¬ Nat.Prime 69655717; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1141897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4173 3) := by change ¬ Nat.Prime 69647371; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1039513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4173 : row 4173 = 0 := (row_4173.checked).trans rfl
theorem model_count_4173 : modelRow 4173 = 0 := by decide
theorem modelrow_agrees_4173 : modelRow 4173 = row 4173 := model_count_4173.trans row_count_4173.symm
def row_4174 : RowCertificate 4174 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4174 1) := by change ¬ Nat.Prime 69697453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9956779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4174 2) := by change ¬ Nat.Prime 69689105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13937821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4174 3) := by change ¬ Nat.Prime 69680757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23226919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4174 : row 4174 = 0 := (row_4174.checked).trans rfl
theorem model_count_4174 : modelRow 4174 = 0 := by decide
theorem modelrow_agrees_4174 : modelRow 4174 = row 4174 := model_count_4174.trans row_count_4174.symm
def row_4175 : RowCertificate 4175 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4175 1) := by change ¬ Nat.Prime 69730851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23243617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4175 2) := by change Nat.Prime 69722501; exact ProjectEulerSolutions.P58PrimeNodes.p_69722501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4175 3) := by change ¬ Nat.Prime 69714151; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5362627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4175 : row 4175 = 1 := (row_4175.checked).trans rfl
theorem model_count_4175 : modelRow 4175 = 1 := by decide
theorem modelrow_agrees_4175 : modelRow 4175 = row 4175 := model_count_4175.trans row_count_4175.symm
def row_4176 : RowCertificate 4176 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4176 1) := by change ¬ Nat.Prime 69764257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3671803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4176 2) := by change ¬ Nat.Prime 69755905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13951181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4176 3) := by change ¬ Nat.Prime 69747553; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1885069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4176 : row 4176 = 0 := (row_4176.checked).trans rfl
theorem model_count_4176 : modelRow 4176 = 0 := by decide
theorem modelrow_agrees_4176 : modelRow 4176 = row 4176 := model_count_4176.trans row_count_4176.symm
def row_4177 : RowCertificate 4177 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4177 1) := by change ¬ Nat.Prime 69797671; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 92203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4177 2) := by change ¬ Nat.Prime 69789317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5368409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4177 3) := by change ¬ Nat.Prime 69780963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23260321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4177 : row 4177 = 0 := (row_4177.checked).trans rfl
theorem model_count_4177 : modelRow 4177 = 0 := by decide
theorem modelrow_agrees_4177 : modelRow 4177 = row 4177 := model_count_4177.trans row_count_4177.symm
def row_4178 : RowCertificate 4178 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4178 1) := by change ¬ Nat.Prime 69831093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23277031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4178 2) := by change ¬ Nat.Prime 69822737; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1887101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4178 3) := by change ¬ Nat.Prime 69814381; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9973483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4178 : row 4178 = 0 := (row_4178.checked).trans rfl
theorem model_count_4178 : modelRow 4178 = 0 := by decide
theorem modelrow_agrees_4178 : modelRow 4178 = row 4178 := model_count_4178.trans row_count_4178.symm
def row_4179 : RowCertificate 4179 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4179 1) := by change ¬ Nat.Prime 69864523; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 56479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4179 2) := by change ¬ Nat.Prime 69856165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13971233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4179 3) := by change ¬ Nat.Prime 69847807; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 190321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4179 : row 4179 = 0 := (row_4179.checked).trans rfl
theorem model_count_4179 : modelRow 4179 = 0 := by decide
theorem modelrow_agrees_4179 : modelRow 4179 = row 4179 := model_count_4179.trans row_count_4179.symm
def row_4180 : RowCertificate 4180 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4180 1) := by change ¬ Nat.Prime 69897961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9985423) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4180 2) := by change ¬ Nat.Prime 69889601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4111153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4180 3) := by change ¬ Nat.Prime 69881241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23293747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4180 : row 4180 = 0 := (row_4180.checked).trans rfl
theorem model_count_4180 : modelRow 4180 = 0 := by decide
theorem modelrow_agrees_4180 : modelRow 4180 = row 4180 := model_count_4180.trans row_count_4180.symm
def row_4181 : RowCertificate 4181 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4181 1) := by change ¬ Nat.Prime 69931407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23310469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4181 2) := by change ¬ Nat.Prime 69923045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13984609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4181 3) := by change ¬ Nat.Prime 69914683; exact PrattCertificates.composite_of_product (factor := 2311) (cofactor := 30253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4181 : row 4181 = 0 := (row_4181.checked).trans rfl
theorem model_count_4181 : modelRow 4181 = 0 := by decide
theorem modelrow_agrees_4181 : modelRow 4181 = row 4181 := model_count_4181.trans row_count_4181.symm
def row_4182 : RowCertificate 4182 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4182 1) := by change ¬ Nat.Prime 69964861; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2256931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4182 2) := by change ¬ Nat.Prime 69956497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5381269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4182 3) := by change ¬ Nat.Prime 69948133; exact PrattCertificates.composite_of_product (factor := 1747) (cofactor := 40039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4182 : row 4182 = 0 := (row_4182.checked).trans rfl
theorem model_count_4182 : modelRow 4182 = 0 := by decide
theorem modelrow_agrees_4182 : modelRow 4182 = row 4182 := model_count_4182.trans row_count_4182.symm
def row_4183 : RowCertificate 4183 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4183 1) := by change Nat.Prime 69998323; exact ProjectEulerSolutions.P58PrimeNodes.p_69998323.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4183 2) := by change ¬ Nat.Prime 69989957; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 305633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4183 3) := by change ¬ Nat.Prime 69981591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23327197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4183 : row 4183 = 1 := (row_4183.checked).trans rfl
theorem model_count_4183 : modelRow 4183 = 1 := by decide
theorem modelrow_agrees_4183 : modelRow 4183 = row 4183 := model_count_4183.trans row_count_4183.symm
def row_4184 : RowCertificate 4184 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4184 1) := by change ¬ Nat.Prime 70031793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23343931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4184 2) := by change ¬ Nat.Prime 70023425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14004685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4184 3) := by change ¬ Nat.Prime 70015057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10002151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4184 : row 4184 = 0 := (row_4184.checked).trans rfl
theorem model_count_4184 : modelRow 4184 = 0 := by decide
theorem modelrow_agrees_4184 : modelRow 4184 = row 4184 := model_count_4184.trans row_count_4184.symm
def row_4185 : RowCertificate 4185 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4185 1) := by change ¬ Nat.Prime 70065271; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1148611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4185 2) := by change ¬ Nat.Prime 70056901; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 252913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4185 3) := by change ¬ Nat.Prime 70048531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10006933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4185 : row 4185 = 0 := (row_4185.checked).trans rfl
theorem model_count_4185 : modelRow 4185 = 0 := by decide
theorem modelrow_agrees_4185 : modelRow 4185 = row 4185 := model_count_4185.trans row_count_4185.symm
def row_4186 : RowCertificate 4186 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4186 1) := by change ¬ Nat.Prime 70098757; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1894561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4186 2) := by change ¬ Nat.Prime 70090385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14018077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4186 3) := by change ¬ Nat.Prime 70082013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23360671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4186 : row 4186 = 0 := (row_4186.checked).trans rfl
theorem model_count_4186 : modelRow 4186 = 0 := by decide
theorem modelrow_agrees_4186 : modelRow 4186 = row 4186 := model_count_4186.trans row_count_4186.symm
def row_4187 : RowCertificate 4187 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4187 1) := by change ¬ Nat.Prime 70132251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23377417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4187 2) := by change ¬ Nat.Prime 70123877; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 29501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4187 3) := by change Nat.Prime 70115503; exact ProjectEulerSolutions.P58PrimeNodes.p_70115503.prime
    exact decide_eq_true prime
theorem row_count_4187 : row 4187 = 1 := (row_4187.checked).trans rfl
theorem model_count_4187 : modelRow 4187 = 1 := by decide
theorem modelrow_agrees_4187 : modelRow 4187 = row 4187 := model_count_4187.trans row_count_4187.symm
def row_4188 : RowCertificate 4188 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4188 1) := by change ¬ Nat.Prime 70165753; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10023679) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4188 2) := by change Nat.Prime 70157377; exact ProjectEulerSolutions.P58PrimeNodes.p_70157377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4188 3) := by change ¬ Nat.Prime 70149001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5396077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4188 : row 4188 = 1 := (row_4188.checked).trans rfl
theorem model_count_4188 : modelRow 4188 = 1 := by decide
theorem modelrow_agrees_4188 : modelRow 4188 = row 4188 := model_count_4188.trans row_count_4188.symm
def row_4189 : RowCertificate 4189 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4189 1) := by change ¬ Nat.Prime 70199263; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1632541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4189 2) := by change ¬ Nat.Prime 70190885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14038177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4189 3) := by change ¬ Nat.Prime 70182507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23394169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4189 : row 4189 = 0 := (row_4189.checked).trans rfl
theorem model_count_4189 : modelRow 4189 = 0 := by decide
theorem modelrow_agrees_4189 : modelRow 4189 = row 4189 := model_count_4189.trans row_count_4189.symm
def row_4190 : RowCertificate 4190 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4190 1) := by change ¬ Nat.Prime 70232781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23410927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4190 2) := by change ¬ Nat.Prime 70224401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5401877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4190 3) := by change Nat.Prime 70216021; exact ProjectEulerSolutions.P58PrimeNodes.p_70216021.prime
    exact decide_eq_true prime
theorem row_count_4190 : row 4190 = 1 := (row_4190.checked).trans rfl
theorem model_count_4190 : modelRow 4190 = 1 := by decide
theorem modelrow_agrees_4190 : modelRow 4190 = row 4190 := model_count_4190.trans row_count_4190.symm
def row_4191 : RowCertificate 4191 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4191 1) := by change ¬ Nat.Prime 70266307; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 505513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4191 2) := by change ¬ Nat.Prime 70257925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14051585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4191 3) := by change ¬ Nat.Prime 70249543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10035649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4191 : row 4191 = 0 := (row_4191.checked).trans rfl
theorem model_count_4191 : modelRow 4191 = 0 := by decide
theorem modelrow_agrees_4191 : modelRow 4191 = row 4191 := model_count_4191.trans row_count_4191.symm
def row_4192 : RowCertificate 4192 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4192 1) := by change Nat.Prime 70299841; exact ProjectEulerSolutions.P58PrimeNodes.p_70299841.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4192 2) := by change ¬ Nat.Prime 70291457; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 406309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4192 3) := by change ¬ Nat.Prime 70283073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23427691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4192 : row 4192 = 1 := (row_4192.checked).trans rfl
theorem model_count_4192 : modelRow 4192 = 1 := by decide
theorem modelrow_agrees_4192 : modelRow 4192 = row 4192 := model_count_4192.trans row_count_4192.symm
def row_4193 : RowCertificate 4193 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4193 1) := by change ¬ Nat.Prime 70333383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23444461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4193 2) := by change Nat.Prime 70324997; exact ProjectEulerSolutions.P58PrimeNodes.p_70324997.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4193 3) := by change Nat.Prime 70316611; exact ProjectEulerSolutions.P58PrimeNodes.p_70316611.prime
    exact decide_eq_true prime
theorem row_count_4193 : row 4193 = 2 := (row_4193.checked).trans rfl
theorem model_count_4193 : modelRow 4193 = 2 := by decide
theorem modelrow_agrees_4193 : modelRow 4193 = row 4193 := model_count_4193.trans row_count_4193.symm
def row_4194 : RowCertificate 4194 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4194 1) := by change ¬ Nat.Prime 70366933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10052419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4194 2) := by change ¬ Nat.Prime 70358545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14071709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4194 3) := by change Nat.Prime 70350157; exact ProjectEulerSolutions.P58PrimeNodes.p_70350157.prime
    exact decide_eq_true prime
theorem row_count_4194 : row 4194 = 1 := (row_4194.checked).trans rfl
theorem model_count_4194 : modelRow 4194 = 1 := by decide
theorem modelrow_agrees_4194 : modelRow 4194 = row 4194 := model_count_4194.trans row_count_4194.symm
def row_4195 : RowCertificate 4195 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4195 1) := by change ¬ Nat.Prime 70400491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10057213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4195 2) := by change ¬ Nat.Prime 70392101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5414777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4195 3) := by change ¬ Nat.Prime 70383711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23461237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4195 : row 4195 = 0 := (row_4195.checked).trans rfl
theorem model_count_4195 : modelRow 4195 = 0 := by decide
theorem modelrow_agrees_4195 : modelRow 4195 = row 4195 := model_count_4195.trans row_count_4195.symm
def row_4196 : RowCertificate 4196 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4196 1) := by change ¬ Nat.Prime 70434057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23478019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4196 2) := by change ¬ Nat.Prime 70425665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14085133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4196 3) := by change ¬ Nat.Prime 70417273; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1637611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4196 : row 4196 = 0 := (row_4196.checked).trans rfl
theorem model_count_4196 : modelRow 4196 = 0 := by decide
theorem modelrow_agrees_4196 : modelRow 4196 = row 4196 := model_count_4196.trans row_count_4196.symm
def row_4197 : RowCertificate 4197 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4197 1) := by change ¬ Nat.Prime 70467631; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5420587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4197 2) := by change ¬ Nat.Prime 70459237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4144661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4197 3) := by change Nat.Prime 70450843; exact ProjectEulerSolutions.P58PrimeNodes.p_70450843.prime
    exact decide_eq_true prime
theorem row_count_4197 : row 4197 = 1 := (row_4197.checked).trans rfl
theorem model_count_4197 : modelRow 4197 = 1 := by decide
theorem modelrow_agrees_4197 : modelRow 4197 = row 4197 := model_count_4197.trans row_count_4197.symm
def row_4198 : RowCertificate 4198 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4198 1) := by change ¬ Nat.Prime 70501213; exact PrattCertificates.composite_of_product (factor := 1543) (cofactor := 45691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4198 2) := by change ¬ Nat.Prime 70492817; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1719337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4198 3) := by change ¬ Nat.Prime 70484421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23494807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4198 : row 4198 = 0 := (row_4198.checked).trans rfl
theorem model_count_4198 : modelRow 4198 = 0 := by decide
theorem modelrow_agrees_4198 : modelRow 4198 = row 4198 := model_count_4198.trans row_count_4198.symm
def row_4199 : RowCertificate 4199 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4199 1) := by change ¬ Nat.Prime 70534803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23511601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4199 2) := by change ¬ Nat.Prime 70526405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14105281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4199 3) := by change ¬ Nat.Prime 70518007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10074001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4199 : row 4199 = 0 := (row_4199.checked).trans rfl
theorem model_count_4199 : modelRow 4199 = 0 := by decide
theorem modelrow_agrees_4199 : modelRow 4199 = row 4199 := model_count_4199.trans row_count_4199.symm
def row_4200 : RowCertificate 4200 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4200 1) := by change Nat.Prime 70568401; exact ProjectEulerSolutions.P58PrimeNodes.p_70568401.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4200 2) := by change ¬ Nat.Prime 70560001; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 792809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4200 3) := by change ¬ Nat.Prime 70551601; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 684967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4200 : row 4200 = 1 := (row_4200.checked).trans rfl
theorem model_count_4200 : modelRow 4200 = 1 := by decide
theorem modelrow_agrees_4200 : modelRow 4200 = row 4200 := model_count_4200.trans row_count_4200.symm
def row_4201 : RowCertificate 4201 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4201 1) := by change ¬ Nat.Prime 70602007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10086001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4201 2) := by change ¬ Nat.Prime 70593605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14118721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4201 3) := by change ¬ Nat.Prime 70585203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23528401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4201 : row 4201 = 0 := (row_4201.checked).trans rfl
theorem model_count_4201 : modelRow 4201 = 0 := by decide
theorem modelrow_agrees_4201 : modelRow 4201 = row 4201 := model_count_4201.trans row_count_4201.symm
def row_4202 : RowCertificate 4202 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4202 1) := by change ¬ Nat.Prime 70635621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23545207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4202 2) := by change ¬ Nat.Prime 70627217; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1332589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4202 3) := by change ¬ Nat.Prime 70618813; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 967381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4202 : row 4202 = 0 := (row_4202.checked).trans rfl
theorem model_count_4202 : modelRow 4202 = 0 := by decide
theorem modelrow_agrees_4202 : modelRow 4202 = row 4202 := model_count_4202.trans row_count_4202.symm
def row_4203 : RowCertificate 4203 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4203 1) := by change ¬ Nat.Prime 70669243; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2279653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4203 2) := by change ¬ Nat.Prime 70660837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5435449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4203 3) := by change ¬ Nat.Prime 70652431; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3718549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4203 : row 4203 = 0 := (row_4203.checked).trans rfl
theorem model_count_4203 : modelRow 4203 = 0 := by decide
theorem modelrow_agrees_4203 : modelRow 4203 = row 4203 := model_count_4203.trans row_count_4203.symm
def row_4204 : RowCertificate 4204 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4204 1) := by change Nat.Prime 70702873; exact ProjectEulerSolutions.P58PrimeNodes.p_70702873.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4204 2) := by change ¬ Nat.Prime 70694465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14138893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4204 3) := by change ¬ Nat.Prime 70686057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23562019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4204 : row 4204 = 1 := (row_4204.checked).trans rfl
theorem model_count_4204 : modelRow 4204 = 1 := by decide
theorem modelrow_agrees_4204 : modelRow 4204 = row 4204 := model_count_4204.trans row_count_4204.symm
def row_4205 : RowCertificate 4205 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4205 1) := by change ¬ Nat.Prime 70736511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23578837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4205 2) := by change ¬ Nat.Prime 70728101; exact PrattCertificates.composite_of_product (factor := 761) (cofactor := 92941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4205 3) := by change ¬ Nat.Prime 70719691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10102813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4205 : row 4205 = 0 := (row_4205.checked).trans rfl
theorem model_count_4205 : modelRow 4205 = 0 := by decide
theorem modelrow_agrees_4205 : modelRow 4205 = row 4205 := model_count_4205.trans row_count_4205.symm
def row_4206 : RowCertificate 4206 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4206 1) := by change ¬ Nat.Prime 70770157; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1056271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4206 2) := by change ¬ Nat.Prime 70761745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14152349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4206 3) := by change ¬ Nat.Prime 70753333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10107619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4206 : row 4206 = 0 := (row_4206.checked).trans rfl
theorem model_count_4206 : modelRow 4206 = 0 := by decide
theorem modelrow_agrees_4206 : modelRow 4206 = row 4206 := model_count_4206.trans row_count_4206.symm
def row_4207 : RowCertificate 4207 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4207 1) := by change ¬ Nat.Prime 70803811; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5446447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4207 2) := by change ¬ Nat.Prime 70795397; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1726717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4207 3) := by change ¬ Nat.Prime 70786983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23595661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4207 : row 4207 = 0 := (row_4207.checked).trans rfl
theorem model_count_4207 : modelRow 4207 = 0 := by decide
theorem modelrow_agrees_4207 : modelRow 4207 = row 4207 := model_count_4207.trans row_count_4207.symm
def row_4208 : RowCertificate 4208 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4208 1) := by change ¬ Nat.Prime 70837473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23612491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4208 2) := by change ¬ Nat.Prime 70829057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5448389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4208 3) := by change ¬ Nat.Prime 70820641; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 85429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4208 : row 4208 = 0 := (row_4208.checked).trans rfl
theorem model_count_4208 : modelRow 4208 = 0 := by decide
theorem modelrow_agrees_4208 : modelRow 4208 = row 4208 := model_count_4208.trans row_count_4208.symm
def row_4209 : RowCertificate 4209 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4209 1) := by change ¬ Nat.Prime 70871143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10124449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4209 2) := by change ¬ Nat.Prime 70862725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14172545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4209 3) := by change ¬ Nat.Prime 70854307; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 434689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4209 : row 4209 = 0 := (row_4209.checked).trans rfl
theorem model_count_4209 : modelRow 4209 = 0 := by decide
theorem modelrow_agrees_4209 : modelRow 4209 = row 4209 := model_count_4209.trans row_count_4209.symm
def row_4210 : RowCertificate 4210 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4210 1) := by change ¬ Nat.Prime 70904821; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5454217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4210 2) := by change Nat.Prime 70896401; exact ProjectEulerSolutions.P58PrimeNodes.p_70896401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4210 3) := by change ¬ Nat.Prime 70887981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23629327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4210 : row 4210 = 1 := (row_4210.checked).trans rfl
theorem model_count_4210 : modelRow 4210 = 1 := by decide
theorem modelrow_agrees_4210 : modelRow 4210 = row 4210 := model_count_4210.trans row_count_4210.symm
def row_4211 : RowCertificate 4211 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4211 1) := by change ¬ Nat.Prime 70938507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23646169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4211 2) := by change ¬ Nat.Prime 70930085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14186017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4211 3) := by change ¬ Nat.Prime 70921663; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1649341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4211 : row 4211 = 0 := (row_4211.checked).trans rfl
theorem model_count_4211 : modelRow 4211 = 0 := by decide
theorem modelrow_agrees_4211 : modelRow 4211 = row 4211 := model_count_4211.trans row_count_4211.symm
def row_4212 : RowCertificate 4212 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4212 1) := by change ¬ Nat.Prime 70972201; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3735379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4212 2) := by change Nat.Prime 70963777; exact ProjectEulerSolutions.P58PrimeNodes.p_70963777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4212 3) := by change ¬ Nat.Prime 70955353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10136479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4212 : row 4212 = 1 := (row_4212.checked).trans rfl
theorem model_count_4212 : modelRow 4212 = 1 := by decide
theorem modelrow_agrees_4212 : modelRow 4212 = row 4212 := model_count_4212.trans row_count_4212.symm
def row_4213 : RowCertificate 4213 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4213 1) := by change ¬ Nat.Prime 71005903; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2290513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4213 2) := by change ¬ Nat.Prime 70997477; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 651353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4213 3) := by change ¬ Nat.Prime 70989051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23663017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4213 : row 4213 = 0 := (row_4213.checked).trans rfl
theorem model_count_4213 : modelRow 4213 = 0 := by decide
theorem modelrow_agrees_4213 : modelRow 4213 = row 4213 := model_count_4213.trans row_count_4213.symm
def row_4214 : RowCertificate 4214 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4214 1) := by change ¬ Nat.Prime 71039613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23679871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4214 2) := by change ¬ Nat.Prime 71031185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14206237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4214 3) := by change ¬ Nat.Prime 71022757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5463289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4214 : row 4214 = 0 := (row_4214.checked).trans rfl
theorem model_count_4214 : modelRow 4214 = 0 := by decide
theorem modelrow_agrees_4214 : modelRow 4214 = row 4214 := model_count_4214.trans row_count_4214.symm
def row_4215 : RowCertificate 4215 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4215 1) := by change ¬ Nat.Prime 71073331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10153333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4215 2) := by change ¬ Nat.Prime 71064901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1920673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4215 3) := by change ¬ Nat.Prime 71056471; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 899449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4215 : row 4215 = 0 := (row_4215.checked).trans rfl
theorem model_count_4215 : modelRow 4215 = 0 := by decide
theorem modelrow_agrees_4215 : modelRow 4215 = row 4215 := model_count_4215.trans row_count_4215.symm
def row_4216 : RowCertificate 4216 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4216 1) := by change ¬ Nat.Prime 71107057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10158151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4216 2) := by change ¬ Nat.Prime 71098625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14219725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4216 3) := by change ¬ Nat.Prime 71090193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23696731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4216 : row 4216 = 0 := (row_4216.checked).trans rfl
theorem model_count_4216 : modelRow 4216 = 0 := by decide
theorem modelrow_agrees_4216 : modelRow 4216 = row 4216 := model_count_4216.trans row_count_4216.symm
def row_4217 : RowCertificate 4217 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4217 1) := by change ¬ Nat.Prime 71140791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23713597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4217 2) := by change Nat.Prime 71132357; exact ProjectEulerSolutions.P58PrimeNodes.p_71132357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4217 3) := by change ¬ Nat.Prime 71123923; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5471071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4217 : row 4217 = 1 := (row_4217.checked).trans rfl
theorem model_count_4217 : modelRow 4217 = 1 := by decide
theorem modelrow_agrees_4217 : modelRow 4217 = row 4217 := model_count_4217.trans row_count_4217.symm
def row_4218 : RowCertificate 4218 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4218 1) := by change ¬ Nat.Prime 71174533; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 512047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4218 2) := by change ¬ Nat.Prime 71166097; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4186241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4218 3) := by change Nat.Prime 71157661; exact ProjectEulerSolutions.P58PrimeNodes.p_71157661.prime
    exact decide_eq_true prime
theorem row_count_4218 : row 4218 = 1 := (row_4218.checked).trans rfl
theorem model_count_4218 : modelRow 4218 = 1 := by decide
theorem modelrow_agrees_4218 : modelRow 4218 = row 4218 := model_count_4218.trans row_count_4218.symm
def row_4219 : RowCertificate 4219 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4219 1) := by change ¬ Nat.Prime 71208283; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 653287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4219 2) := by change ¬ Nat.Prime 71199845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14239969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4219 3) := by change ¬ Nat.Prime 71191407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23730469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4219 : row 4219 = 0 := (row_4219.checked).trans rfl
theorem model_count_4219 : modelRow 4219 = 0 := by decide
theorem modelrow_agrees_4219 : modelRow 4219 = row 4219 := model_count_4219.trans row_count_4219.symm
def row_4220 : RowCertificate 4220 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4220 1) := by change ¬ Nat.Prime 71242041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23747347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4220 2) := by change ¬ Nat.Prime 71233601; exact PrattCertificates.composite_of_product (factor := 2069) (cofactor := 34429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4220 3) := by change ¬ Nat.Prime 71225161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10175023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4220 : row 4220 = 0 := (row_4220.checked).trans rfl
theorem model_count_4220 : modelRow 4220 = 0 := by decide
theorem modelrow_agrees_4220 : modelRow 4220 = row 4220 := model_count_4220.trans row_count_4220.symm
def row_4221 : RowCertificate 4221 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4221 1) := by change ¬ Nat.Prime 71275807; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 98041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4221 2) := by change ¬ Nat.Prime 71267365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14253473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4221 3) := by change ¬ Nat.Prime 71258923; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 123499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4221 : row 4221 = 0 := (row_4221.checked).trans rfl
theorem model_count_4221 : modelRow 4221 = 0 := by decide
theorem modelrow_agrees_4221 : modelRow 4221 = row 4221 := model_count_4221.trans row_count_4221.symm
def row_4222 : RowCertificate 4222 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4222 1) := by change ¬ Nat.Prime 71309581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10187083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4222 2) := by change ¬ Nat.Prime 71301137; exact PrattCertificates.composite_of_product (factor := 617) (cofactor := 115561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4222 3) := by change ¬ Nat.Prime 71292693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23764231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4222 : row 4222 = 0 := (row_4222.checked).trans rfl
theorem model_count_4222 : modelRow 4222 = 0 := by decide
theorem modelrow_agrees_4222 : modelRow 4222 = row 4222 := model_count_4222.trans row_count_4222.symm
def row_4223 : RowCertificate 4223 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4223 1) := by change ¬ Nat.Prime 71343363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23781121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4223 2) := by change ¬ Nat.Prime 71334917; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 174413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4223 3) := by change ¬ Nat.Prime 71326471; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 252037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4223 : row 4223 = 0 := (row_4223.checked).trans rfl
theorem model_count_4223 : modelRow 4223 = 0 := by decide
theorem modelrow_agrees_4223 : modelRow 4223 = row 4223 := model_count_4223.trans row_count_4223.symm
def row_4224 : RowCertificate 4224 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4224 1) := by change ¬ Nat.Prime 71377153; exact PrattCertificates.composite_of_product (factor := 1279) (cofactor := 55807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4224 2) := by change ¬ Nat.Prime 71368705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14273741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4224 3) := by change ¬ Nat.Prime 71360257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3755803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4224 : row 4224 = 0 := (row_4224.checked).trans rfl
theorem model_count_4224 : modelRow 4224 = 0 := by decide
theorem modelrow_agrees_4224 : modelRow 4224 = row 4224 := model_count_4224.trans row_count_4224.symm
private def atomic_4097 : CountSegment 1 10 13120 4096 4097 1886 1887 := CountSegment.single (first := 4096) (initial := 1886) row_count_4097 (by decide)
private theorem atomic_4097_agreement : RowAgreement 4096 4097 := RowAgreement.single (first := 4096) modelrow_agrees_4097
private def atomic_4098 : CountSegment 1 10 13120 4097 4098 1887 1889 := CountSegment.single (first := 4097) (initial := 1887) row_count_4098 (by decide)
private theorem atomic_4098_agreement : RowAgreement 4097 4098 := RowAgreement.single (first := 4097) modelrow_agrees_4098
private def atomic_4099 : CountSegment 1 10 13120 4098 4099 1889 1890 := CountSegment.single (first := 4098) (initial := 1889) row_count_4099 (by decide)
private theorem atomic_4099_agreement : RowAgreement 4098 4099 := RowAgreement.single (first := 4098) modelrow_agrees_4099
private def atomic_4100 : CountSegment 1 10 13120 4099 4100 1890 1890 := CountSegment.single (first := 4099) (initial := 1890) row_count_4100 (by decide)
private theorem atomic_4100_agreement : RowAgreement 4099 4100 := RowAgreement.single (first := 4099) modelrow_agrees_4100
private def atomic_4101 : CountSegment 1 10 13120 4100 4101 1890 1891 := CountSegment.single (first := 4100) (initial := 1890) row_count_4101 (by decide)
private theorem atomic_4101_agreement : RowAgreement 4100 4101 := RowAgreement.single (first := 4100) modelrow_agrees_4101
private def atomic_4102 : CountSegment 1 10 13120 4101 4102 1891 1891 := CountSegment.single (first := 4101) (initial := 1891) row_count_4102 (by decide)
private theorem atomic_4102_agreement : RowAgreement 4101 4102 := RowAgreement.single (first := 4101) modelrow_agrees_4102
private def atomic_4103 : CountSegment 1 10 13120 4102 4103 1891 1892 := CountSegment.single (first := 4102) (initial := 1891) row_count_4103 (by decide)
private theorem atomic_4103_agreement : RowAgreement 4102 4103 := RowAgreement.single (first := 4102) modelrow_agrees_4103
private def atomic_4104 : CountSegment 1 10 13120 4103 4104 1892 1892 := CountSegment.single (first := 4103) (initial := 1892) row_count_4104 (by decide)
private theorem atomic_4104_agreement : RowAgreement 4103 4104 := RowAgreement.single (first := 4103) modelrow_agrees_4104
private def atomic_4105 : CountSegment 1 10 13120 4104 4105 1892 1894 := CountSegment.single (first := 4104) (initial := 1892) row_count_4105 (by decide)
private theorem atomic_4105_agreement : RowAgreement 4104 4105 := RowAgreement.single (first := 4104) modelrow_agrees_4105
private def atomic_4106 : CountSegment 1 10 13120 4105 4106 1894 1894 := CountSegment.single (first := 4105) (initial := 1894) row_count_4106 (by decide)
private theorem atomic_4106_agreement : RowAgreement 4105 4106 := RowAgreement.single (first := 4105) modelrow_agrees_4106
private def atomic_4107 : CountSegment 1 10 13120 4106 4107 1894 1894 := CountSegment.single (first := 4106) (initial := 1894) row_count_4107 (by decide)
private theorem atomic_4107_agreement : RowAgreement 4106 4107 := RowAgreement.single (first := 4106) modelrow_agrees_4107
private def atomic_4108 : CountSegment 1 10 13120 4107 4108 1894 1894 := CountSegment.single (first := 4107) (initial := 1894) row_count_4108 (by decide)
private theorem atomic_4108_agreement : RowAgreement 4107 4108 := RowAgreement.single (first := 4107) modelrow_agrees_4108
private def atomic_4109 : CountSegment 1 10 13120 4108 4109 1894 1895 := CountSegment.single (first := 4108) (initial := 1894) row_count_4109 (by decide)
private theorem atomic_4109_agreement : RowAgreement 4108 4109 := RowAgreement.single (first := 4108) modelrow_agrees_4109
private def atomic_4110 : CountSegment 1 10 13120 4109 4110 1895 1895 := CountSegment.single (first := 4109) (initial := 1895) row_count_4110 (by decide)
private theorem atomic_4110_agreement : RowAgreement 4109 4110 := RowAgreement.single (first := 4109) modelrow_agrees_4110
private def atomic_4111 : CountSegment 1 10 13120 4110 4111 1895 1895 := CountSegment.single (first := 4110) (initial := 1895) row_count_4111 (by decide)
private theorem atomic_4111_agreement : RowAgreement 4110 4111 := RowAgreement.single (first := 4110) modelrow_agrees_4111
private def atomic_4112 : CountSegment 1 10 13120 4111 4112 1895 1895 := CountSegment.single (first := 4111) (initial := 1895) row_count_4112 (by decide)
private theorem atomic_4112_agreement : RowAgreement 4111 4112 := RowAgreement.single (first := 4111) modelrow_agrees_4112
private def atomic_4113 : CountSegment 1 10 13120 4112 4113 1895 1897 := CountSegment.single (first := 4112) (initial := 1895) row_count_4113 (by decide)
private theorem atomic_4113_agreement : RowAgreement 4112 4113 := RowAgreement.single (first := 4112) modelrow_agrees_4113
private def atomic_4114 : CountSegment 1 10 13120 4113 4114 1897 1897 := CountSegment.single (first := 4113) (initial := 1897) row_count_4114 (by decide)
private theorem atomic_4114_agreement : RowAgreement 4113 4114 := RowAgreement.single (first := 4113) modelrow_agrees_4114
private def atomic_4115 : CountSegment 1 10 13120 4114 4115 1897 1898 := CountSegment.single (first := 4114) (initial := 1897) row_count_4115 (by decide)
private theorem atomic_4115_agreement : RowAgreement 4114 4115 := RowAgreement.single (first := 4114) modelrow_agrees_4115
private def atomic_4116 : CountSegment 1 10 13120 4115 4116 1898 1899 := CountSegment.single (first := 4115) (initial := 1898) row_count_4116 (by decide)
private theorem atomic_4116_agreement : RowAgreement 4115 4116 := RowAgreement.single (first := 4115) modelrow_agrees_4116
private def atomic_4117 : CountSegment 1 10 13120 4116 4117 1899 1899 := CountSegment.single (first := 4116) (initial := 1899) row_count_4117 (by decide)
private theorem atomic_4117_agreement : RowAgreement 4116 4117 := RowAgreement.single (first := 4116) modelrow_agrees_4117
private def atomic_4118 : CountSegment 1 10 13120 4117 4118 1899 1899 := CountSegment.single (first := 4117) (initial := 1899) row_count_4118 (by decide)
private theorem atomic_4118_agreement : RowAgreement 4117 4118 := RowAgreement.single (first := 4117) modelrow_agrees_4118
private def atomic_4119 : CountSegment 1 10 13120 4118 4119 1899 1900 := CountSegment.single (first := 4118) (initial := 1899) row_count_4119 (by decide)
private theorem atomic_4119_agreement : RowAgreement 4118 4119 := RowAgreement.single (first := 4118) modelrow_agrees_4119
private def atomic_4120 : CountSegment 1 10 13120 4119 4120 1900 1900 := CountSegment.single (first := 4119) (initial := 1900) row_count_4120 (by decide)
private theorem atomic_4120_agreement : RowAgreement 4119 4120 := RowAgreement.single (first := 4119) modelrow_agrees_4120
private def atomic_4121 : CountSegment 1 10 13120 4120 4121 1900 1900 := CountSegment.single (first := 4120) (initial := 1900) row_count_4121 (by decide)
private theorem atomic_4121_agreement : RowAgreement 4120 4121 := RowAgreement.single (first := 4120) modelrow_agrees_4121
private def atomic_4122 : CountSegment 1 10 13120 4121 4122 1900 1900 := CountSegment.single (first := 4121) (initial := 1900) row_count_4122 (by decide)
private theorem atomic_4122_agreement : RowAgreement 4121 4122 := RowAgreement.single (first := 4121) modelrow_agrees_4122
private def atomic_4123 : CountSegment 1 10 13120 4122 4123 1900 1900 := CountSegment.single (first := 4122) (initial := 1900) row_count_4123 (by decide)
private theorem atomic_4123_agreement : RowAgreement 4122 4123 := RowAgreement.single (first := 4122) modelrow_agrees_4123
private def atomic_4124 : CountSegment 1 10 13120 4123 4124 1900 1901 := CountSegment.single (first := 4123) (initial := 1900) row_count_4124 (by decide)
private theorem atomic_4124_agreement : RowAgreement 4123 4124 := RowAgreement.single (first := 4123) modelrow_agrees_4124
private def atomic_4125 : CountSegment 1 10 13120 4124 4125 1901 1901 := CountSegment.single (first := 4124) (initial := 1901) row_count_4125 (by decide)
private theorem atomic_4125_agreement : RowAgreement 4124 4125 := RowAgreement.single (first := 4124) modelrow_agrees_4125
private def atomic_4126 : CountSegment 1 10 13120 4125 4126 1901 1901 := CountSegment.single (first := 4125) (initial := 1901) row_count_4126 (by decide)
private theorem atomic_4126_agreement : RowAgreement 4125 4126 := RowAgreement.single (first := 4125) modelrow_agrees_4126
private def atomic_4127 : CountSegment 1 10 13120 4126 4127 1901 1902 := CountSegment.single (first := 4126) (initial := 1901) row_count_4127 (by decide)
private theorem atomic_4127_agreement : RowAgreement 4126 4127 := RowAgreement.single (first := 4126) modelrow_agrees_4127
private def atomic_4128 : CountSegment 1 10 13120 4127 4128 1902 1902 := CountSegment.single (first := 4127) (initial := 1902) row_count_4128 (by decide)
private theorem atomic_4128_agreement : RowAgreement 4127 4128 := RowAgreement.single (first := 4127) modelrow_agrees_4128
private def atomic_4129 : CountSegment 1 10 13120 4128 4129 1902 1902 := CountSegment.single (first := 4128) (initial := 1902) row_count_4129 (by decide)
private theorem atomic_4129_agreement : RowAgreement 4128 4129 := RowAgreement.single (first := 4128) modelrow_agrees_4129
private def atomic_4130 : CountSegment 1 10 13120 4129 4130 1902 1903 := CountSegment.single (first := 4129) (initial := 1902) row_count_4130 (by decide)
private theorem atomic_4130_agreement : RowAgreement 4129 4130 := RowAgreement.single (first := 4129) modelrow_agrees_4130
private def atomic_4131 : CountSegment 1 10 13120 4130 4131 1903 1903 := CountSegment.single (first := 4130) (initial := 1903) row_count_4131 (by decide)
private theorem atomic_4131_agreement : RowAgreement 4130 4131 := RowAgreement.single (first := 4130) modelrow_agrees_4131
private def atomic_4132 : CountSegment 1 10 13120 4131 4132 1903 1903 := CountSegment.single (first := 4131) (initial := 1903) row_count_4132 (by decide)
private theorem atomic_4132_agreement : RowAgreement 4131 4132 := RowAgreement.single (first := 4131) modelrow_agrees_4132
private def atomic_4133 : CountSegment 1 10 13120 4132 4133 1903 1903 := CountSegment.single (first := 4132) (initial := 1903) row_count_4133 (by decide)
private theorem atomic_4133_agreement : RowAgreement 4132 4133 := RowAgreement.single (first := 4132) modelrow_agrees_4133
private def atomic_4134 : CountSegment 1 10 13120 4133 4134 1903 1905 := CountSegment.single (first := 4133) (initial := 1903) row_count_4134 (by decide)
private theorem atomic_4134_agreement : RowAgreement 4133 4134 := RowAgreement.single (first := 4133) modelrow_agrees_4134
private def atomic_4135 : CountSegment 1 10 13120 4134 4135 1905 1906 := CountSegment.single (first := 4134) (initial := 1905) row_count_4135 (by decide)
private theorem atomic_4135_agreement : RowAgreement 4134 4135 := RowAgreement.single (first := 4134) modelrow_agrees_4135
private def atomic_4136 : CountSegment 1 10 13120 4135 4136 1906 1906 := CountSegment.single (first := 4135) (initial := 1906) row_count_4136 (by decide)
private theorem atomic_4136_agreement : RowAgreement 4135 4136 := RowAgreement.single (first := 4135) modelrow_agrees_4136
private def atomic_4137 : CountSegment 1 10 13120 4136 4137 1906 1906 := CountSegment.single (first := 4136) (initial := 1906) row_count_4137 (by decide)
private theorem atomic_4137_agreement : RowAgreement 4136 4137 := RowAgreement.single (first := 4136) modelrow_agrees_4137
private def atomic_4138 : CountSegment 1 10 13120 4137 4138 1906 1906 := CountSegment.single (first := 4137) (initial := 1906) row_count_4138 (by decide)
private theorem atomic_4138_agreement : RowAgreement 4137 4138 := RowAgreement.single (first := 4137) modelrow_agrees_4138
private def atomic_4139 : CountSegment 1 10 13120 4138 4139 1906 1906 := CountSegment.single (first := 4138) (initial := 1906) row_count_4139 (by decide)
private theorem atomic_4139_agreement : RowAgreement 4138 4139 := RowAgreement.single (first := 4138) modelrow_agrees_4139
private def atomic_4140 : CountSegment 1 10 13120 4139 4140 1906 1907 := CountSegment.single (first := 4139) (initial := 1906) row_count_4140 (by decide)
private theorem atomic_4140_agreement : RowAgreement 4139 4140 := RowAgreement.single (first := 4139) modelrow_agrees_4140
private def atomic_4141 : CountSegment 1 10 13120 4140 4141 1907 1907 := CountSegment.single (first := 4140) (initial := 1907) row_count_4141 (by decide)
private theorem atomic_4141_agreement : RowAgreement 4140 4141 := RowAgreement.single (first := 4140) modelrow_agrees_4141
private def atomic_4142 : CountSegment 1 10 13120 4141 4142 1907 1907 := CountSegment.single (first := 4141) (initial := 1907) row_count_4142 (by decide)
private theorem atomic_4142_agreement : RowAgreement 4141 4142 := RowAgreement.single (first := 4141) modelrow_agrees_4142
private def atomic_4143 : CountSegment 1 10 13120 4142 4143 1907 1907 := CountSegment.single (first := 4142) (initial := 1907) row_count_4143 (by decide)
private theorem atomic_4143_agreement : RowAgreement 4142 4143 := RowAgreement.single (first := 4142) modelrow_agrees_4143
private def atomic_4144 : CountSegment 1 10 13120 4143 4144 1907 1908 := CountSegment.single (first := 4143) (initial := 1907) row_count_4144 (by decide)
private theorem atomic_4144_agreement : RowAgreement 4143 4144 := RowAgreement.single (first := 4143) modelrow_agrees_4144
private def atomic_4145 : CountSegment 1 10 13120 4144 4145 1908 1909 := CountSegment.single (first := 4144) (initial := 1908) row_count_4145 (by decide)
private theorem atomic_4145_agreement : RowAgreement 4144 4145 := RowAgreement.single (first := 4144) modelrow_agrees_4145
private def atomic_4146 : CountSegment 1 10 13120 4145 4146 1909 1909 := CountSegment.single (first := 4145) (initial := 1909) row_count_4146 (by decide)
private theorem atomic_4146_agreement : RowAgreement 4145 4146 := RowAgreement.single (first := 4145) modelrow_agrees_4146
private def atomic_4147 : CountSegment 1 10 13120 4146 4147 1909 1909 := CountSegment.single (first := 4146) (initial := 1909) row_count_4147 (by decide)
private theorem atomic_4147_agreement : RowAgreement 4146 4147 := RowAgreement.single (first := 4146) modelrow_agrees_4147
private def atomic_4148 : CountSegment 1 10 13120 4147 4148 1909 1910 := CountSegment.single (first := 4147) (initial := 1909) row_count_4148 (by decide)
private theorem atomic_4148_agreement : RowAgreement 4147 4148 := RowAgreement.single (first := 4147) modelrow_agrees_4148
private def atomic_4149 : CountSegment 1 10 13120 4148 4149 1910 1910 := CountSegment.single (first := 4148) (initial := 1910) row_count_4149 (by decide)
private theorem atomic_4149_agreement : RowAgreement 4148 4149 := RowAgreement.single (first := 4148) modelrow_agrees_4149
private def atomic_4150 : CountSegment 1 10 13120 4149 4150 1910 1911 := CountSegment.single (first := 4149) (initial := 1910) row_count_4150 (by decide)
private theorem atomic_4150_agreement : RowAgreement 4149 4150 := RowAgreement.single (first := 4149) modelrow_agrees_4150
private def atomic_4151 : CountSegment 1 10 13120 4150 4151 1911 1911 := CountSegment.single (first := 4150) (initial := 1911) row_count_4151 (by decide)
private theorem atomic_4151_agreement : RowAgreement 4150 4151 := RowAgreement.single (first := 4150) modelrow_agrees_4151
private def atomic_4152 : CountSegment 1 10 13120 4151 4152 1911 1912 := CountSegment.single (first := 4151) (initial := 1911) row_count_4152 (by decide)
private theorem atomic_4152_agreement : RowAgreement 4151 4152 := RowAgreement.single (first := 4151) modelrow_agrees_4152
private def atomic_4153 : CountSegment 1 10 13120 4152 4153 1912 1912 := CountSegment.single (first := 4152) (initial := 1912) row_count_4153 (by decide)
private theorem atomic_4153_agreement : RowAgreement 4152 4153 := RowAgreement.single (first := 4152) modelrow_agrees_4153
private def atomic_4154 : CountSegment 1 10 13120 4153 4154 1912 1912 := CountSegment.single (first := 4153) (initial := 1912) row_count_4154 (by decide)
private theorem atomic_4154_agreement : RowAgreement 4153 4154 := RowAgreement.single (first := 4153) modelrow_agrees_4154
private def atomic_4155 : CountSegment 1 10 13120 4154 4155 1912 1912 := CountSegment.single (first := 4154) (initial := 1912) row_count_4155 (by decide)
private theorem atomic_4155_agreement : RowAgreement 4154 4155 := RowAgreement.single (first := 4154) modelrow_agrees_4155
private def atomic_4156 : CountSegment 1 10 13120 4155 4156 1912 1912 := CountSegment.single (first := 4155) (initial := 1912) row_count_4156 (by decide)
private theorem atomic_4156_agreement : RowAgreement 4155 4156 := RowAgreement.single (first := 4155) modelrow_agrees_4156
private def atomic_4157 : CountSegment 1 10 13120 4156 4157 1912 1912 := CountSegment.single (first := 4156) (initial := 1912) row_count_4157 (by decide)
private theorem atomic_4157_agreement : RowAgreement 4156 4157 := RowAgreement.single (first := 4156) modelrow_agrees_4157
private def atomic_4158 : CountSegment 1 10 13120 4157 4158 1912 1913 := CountSegment.single (first := 4157) (initial := 1912) row_count_4158 (by decide)
private theorem atomic_4158_agreement : RowAgreement 4157 4158 := RowAgreement.single (first := 4157) modelrow_agrees_4158
private def atomic_4159 : CountSegment 1 10 13120 4158 4159 1913 1913 := CountSegment.single (first := 4158) (initial := 1913) row_count_4159 (by decide)
private theorem atomic_4159_agreement : RowAgreement 4158 4159 := RowAgreement.single (first := 4158) modelrow_agrees_4159
private def atomic_4160 : CountSegment 1 10 13120 4159 4160 1913 1913 := CountSegment.single (first := 4159) (initial := 1913) row_count_4160 (by decide)
private theorem atomic_4160_agreement : RowAgreement 4159 4160 := RowAgreement.single (first := 4159) modelrow_agrees_4160
private def atomic_4161 : CountSegment 1 10 13120 4160 4161 1913 1915 := CountSegment.single (first := 4160) (initial := 1913) row_count_4161 (by decide)
private theorem atomic_4161_agreement : RowAgreement 4160 4161 := RowAgreement.single (first := 4160) modelrow_agrees_4161
private def atomic_4162 : CountSegment 1 10 13120 4161 4162 1915 1917 := CountSegment.single (first := 4161) (initial := 1915) row_count_4162 (by decide)
private theorem atomic_4162_agreement : RowAgreement 4161 4162 := RowAgreement.single (first := 4161) modelrow_agrees_4162
private def atomic_4163 : CountSegment 1 10 13120 4162 4163 1917 1917 := CountSegment.single (first := 4162) (initial := 1917) row_count_4163 (by decide)
private theorem atomic_4163_agreement : RowAgreement 4162 4163 := RowAgreement.single (first := 4162) modelrow_agrees_4163
private def atomic_4164 : CountSegment 1 10 13120 4163 4164 1917 1917 := CountSegment.single (first := 4163) (initial := 1917) row_count_4164 (by decide)
private theorem atomic_4164_agreement : RowAgreement 4163 4164 := RowAgreement.single (first := 4163) modelrow_agrees_4164
private def atomic_4165 : CountSegment 1 10 13120 4164 4165 1917 1917 := CountSegment.single (first := 4164) (initial := 1917) row_count_4165 (by decide)
private theorem atomic_4165_agreement : RowAgreement 4164 4165 := RowAgreement.single (first := 4164) modelrow_agrees_4165
private def atomic_4166 : CountSegment 1 10 13120 4165 4166 1917 1917 := CountSegment.single (first := 4165) (initial := 1917) row_count_4166 (by decide)
private theorem atomic_4166_agreement : RowAgreement 4165 4166 := RowAgreement.single (first := 4165) modelrow_agrees_4166
private def atomic_4167 : CountSegment 1 10 13120 4166 4167 1917 1917 := CountSegment.single (first := 4166) (initial := 1917) row_count_4167 (by decide)
private theorem atomic_4167_agreement : RowAgreement 4166 4167 := RowAgreement.single (first := 4166) modelrow_agrees_4167
private def atomic_4168 : CountSegment 1 10 13120 4167 4168 1917 1917 := CountSegment.single (first := 4167) (initial := 1917) row_count_4168 (by decide)
private theorem atomic_4168_agreement : RowAgreement 4167 4168 := RowAgreement.single (first := 4167) modelrow_agrees_4168
private def atomic_4169 : CountSegment 1 10 13120 4168 4169 1917 1917 := CountSegment.single (first := 4168) (initial := 1917) row_count_4169 (by decide)
private theorem atomic_4169_agreement : RowAgreement 4168 4169 := RowAgreement.single (first := 4168) modelrow_agrees_4169
private def atomic_4170 : CountSegment 1 10 13120 4169 4170 1917 1917 := CountSegment.single (first := 4169) (initial := 1917) row_count_4170 (by decide)
private theorem atomic_4170_agreement : RowAgreement 4169 4170 := RowAgreement.single (first := 4169) modelrow_agrees_4170
private def atomic_4171 : CountSegment 1 10 13120 4170 4171 1917 1917 := CountSegment.single (first := 4170) (initial := 1917) row_count_4171 (by decide)
private theorem atomic_4171_agreement : RowAgreement 4170 4171 := RowAgreement.single (first := 4170) modelrow_agrees_4171
private def atomic_4172 : CountSegment 1 10 13120 4171 4172 1917 1917 := CountSegment.single (first := 4171) (initial := 1917) row_count_4172 (by decide)
private theorem atomic_4172_agreement : RowAgreement 4171 4172 := RowAgreement.single (first := 4171) modelrow_agrees_4172
private def atomic_4173 : CountSegment 1 10 13120 4172 4173 1917 1917 := CountSegment.single (first := 4172) (initial := 1917) row_count_4173 (by decide)
private theorem atomic_4173_agreement : RowAgreement 4172 4173 := RowAgreement.single (first := 4172) modelrow_agrees_4173
private def atomic_4174 : CountSegment 1 10 13120 4173 4174 1917 1917 := CountSegment.single (first := 4173) (initial := 1917) row_count_4174 (by decide)
private theorem atomic_4174_agreement : RowAgreement 4173 4174 := RowAgreement.single (first := 4173) modelrow_agrees_4174
private def atomic_4175 : CountSegment 1 10 13120 4174 4175 1917 1918 := CountSegment.single (first := 4174) (initial := 1917) row_count_4175 (by decide)
private theorem atomic_4175_agreement : RowAgreement 4174 4175 := RowAgreement.single (first := 4174) modelrow_agrees_4175
private def atomic_4176 : CountSegment 1 10 13120 4175 4176 1918 1918 := CountSegment.single (first := 4175) (initial := 1918) row_count_4176 (by decide)
private theorem atomic_4176_agreement : RowAgreement 4175 4176 := RowAgreement.single (first := 4175) modelrow_agrees_4176
private def atomic_4177 : CountSegment 1 10 13120 4176 4177 1918 1918 := CountSegment.single (first := 4176) (initial := 1918) row_count_4177 (by decide)
private theorem atomic_4177_agreement : RowAgreement 4176 4177 := RowAgreement.single (first := 4176) modelrow_agrees_4177
private def atomic_4178 : CountSegment 1 10 13120 4177 4178 1918 1918 := CountSegment.single (first := 4177) (initial := 1918) row_count_4178 (by decide)
private theorem atomic_4178_agreement : RowAgreement 4177 4178 := RowAgreement.single (first := 4177) modelrow_agrees_4178
private def atomic_4179 : CountSegment 1 10 13120 4178 4179 1918 1918 := CountSegment.single (first := 4178) (initial := 1918) row_count_4179 (by decide)
private theorem atomic_4179_agreement : RowAgreement 4178 4179 := RowAgreement.single (first := 4178) modelrow_agrees_4179
private def atomic_4180 : CountSegment 1 10 13120 4179 4180 1918 1918 := CountSegment.single (first := 4179) (initial := 1918) row_count_4180 (by decide)
private theorem atomic_4180_agreement : RowAgreement 4179 4180 := RowAgreement.single (first := 4179) modelrow_agrees_4180
private def atomic_4181 : CountSegment 1 10 13120 4180 4181 1918 1918 := CountSegment.single (first := 4180) (initial := 1918) row_count_4181 (by decide)
private theorem atomic_4181_agreement : RowAgreement 4180 4181 := RowAgreement.single (first := 4180) modelrow_agrees_4181
private def atomic_4182 : CountSegment 1 10 13120 4181 4182 1918 1918 := CountSegment.single (first := 4181) (initial := 1918) row_count_4182 (by decide)
private theorem atomic_4182_agreement : RowAgreement 4181 4182 := RowAgreement.single (first := 4181) modelrow_agrees_4182
private def atomic_4183 : CountSegment 1 10 13120 4182 4183 1918 1919 := CountSegment.single (first := 4182) (initial := 1918) row_count_4183 (by decide)
private theorem atomic_4183_agreement : RowAgreement 4182 4183 := RowAgreement.single (first := 4182) modelrow_agrees_4183
private def atomic_4184 : CountSegment 1 10 13120 4183 4184 1919 1919 := CountSegment.single (first := 4183) (initial := 1919) row_count_4184 (by decide)
private theorem atomic_4184_agreement : RowAgreement 4183 4184 := RowAgreement.single (first := 4183) modelrow_agrees_4184
private def atomic_4185 : CountSegment 1 10 13120 4184 4185 1919 1919 := CountSegment.single (first := 4184) (initial := 1919) row_count_4185 (by decide)
private theorem atomic_4185_agreement : RowAgreement 4184 4185 := RowAgreement.single (first := 4184) modelrow_agrees_4185
private def atomic_4186 : CountSegment 1 10 13120 4185 4186 1919 1919 := CountSegment.single (first := 4185) (initial := 1919) row_count_4186 (by decide)
private theorem atomic_4186_agreement : RowAgreement 4185 4186 := RowAgreement.single (first := 4185) modelrow_agrees_4186
private def atomic_4187 : CountSegment 1 10 13120 4186 4187 1919 1920 := CountSegment.single (first := 4186) (initial := 1919) row_count_4187 (by decide)
private theorem atomic_4187_agreement : RowAgreement 4186 4187 := RowAgreement.single (first := 4186) modelrow_agrees_4187
private def atomic_4188 : CountSegment 1 10 13120 4187 4188 1920 1921 := CountSegment.single (first := 4187) (initial := 1920) row_count_4188 (by decide)
private theorem atomic_4188_agreement : RowAgreement 4187 4188 := RowAgreement.single (first := 4187) modelrow_agrees_4188
private def atomic_4189 : CountSegment 1 10 13120 4188 4189 1921 1921 := CountSegment.single (first := 4188) (initial := 1921) row_count_4189 (by decide)
private theorem atomic_4189_agreement : RowAgreement 4188 4189 := RowAgreement.single (first := 4188) modelrow_agrees_4189
private def atomic_4190 : CountSegment 1 10 13120 4189 4190 1921 1922 := CountSegment.single (first := 4189) (initial := 1921) row_count_4190 (by decide)
private theorem atomic_4190_agreement : RowAgreement 4189 4190 := RowAgreement.single (first := 4189) modelrow_agrees_4190
private def atomic_4191 : CountSegment 1 10 13120 4190 4191 1922 1922 := CountSegment.single (first := 4190) (initial := 1922) row_count_4191 (by decide)
private theorem atomic_4191_agreement : RowAgreement 4190 4191 := RowAgreement.single (first := 4190) modelrow_agrees_4191
private def atomic_4192 : CountSegment 1 10 13120 4191 4192 1922 1923 := CountSegment.single (first := 4191) (initial := 1922) row_count_4192 (by decide)
private theorem atomic_4192_agreement : RowAgreement 4191 4192 := RowAgreement.single (first := 4191) modelrow_agrees_4192
private def atomic_4193 : CountSegment 1 10 13120 4192 4193 1923 1925 := CountSegment.single (first := 4192) (initial := 1923) row_count_4193 (by decide)
private theorem atomic_4193_agreement : RowAgreement 4192 4193 := RowAgreement.single (first := 4192) modelrow_agrees_4193
private def atomic_4194 : CountSegment 1 10 13120 4193 4194 1925 1926 := CountSegment.single (first := 4193) (initial := 1925) row_count_4194 (by decide)
private theorem atomic_4194_agreement : RowAgreement 4193 4194 := RowAgreement.single (first := 4193) modelrow_agrees_4194
private def atomic_4195 : CountSegment 1 10 13120 4194 4195 1926 1926 := CountSegment.single (first := 4194) (initial := 1926) row_count_4195 (by decide)
private theorem atomic_4195_agreement : RowAgreement 4194 4195 := RowAgreement.single (first := 4194) modelrow_agrees_4195
private def atomic_4196 : CountSegment 1 10 13120 4195 4196 1926 1926 := CountSegment.single (first := 4195) (initial := 1926) row_count_4196 (by decide)
private theorem atomic_4196_agreement : RowAgreement 4195 4196 := RowAgreement.single (first := 4195) modelrow_agrees_4196
private def atomic_4197 : CountSegment 1 10 13120 4196 4197 1926 1927 := CountSegment.single (first := 4196) (initial := 1926) row_count_4197 (by decide)
private theorem atomic_4197_agreement : RowAgreement 4196 4197 := RowAgreement.single (first := 4196) modelrow_agrees_4197
private def atomic_4198 : CountSegment 1 10 13120 4197 4198 1927 1927 := CountSegment.single (first := 4197) (initial := 1927) row_count_4198 (by decide)
private theorem atomic_4198_agreement : RowAgreement 4197 4198 := RowAgreement.single (first := 4197) modelrow_agrees_4198
private def atomic_4199 : CountSegment 1 10 13120 4198 4199 1927 1927 := CountSegment.single (first := 4198) (initial := 1927) row_count_4199 (by decide)
private theorem atomic_4199_agreement : RowAgreement 4198 4199 := RowAgreement.single (first := 4198) modelrow_agrees_4199
private def atomic_4200 : CountSegment 1 10 13120 4199 4200 1927 1928 := CountSegment.single (first := 4199) (initial := 1927) row_count_4200 (by decide)
private theorem atomic_4200_agreement : RowAgreement 4199 4200 := RowAgreement.single (first := 4199) modelrow_agrees_4200
private def atomic_4201 : CountSegment 1 10 13120 4200 4201 1928 1928 := CountSegment.single (first := 4200) (initial := 1928) row_count_4201 (by decide)
private theorem atomic_4201_agreement : RowAgreement 4200 4201 := RowAgreement.single (first := 4200) modelrow_agrees_4201
private def atomic_4202 : CountSegment 1 10 13120 4201 4202 1928 1928 := CountSegment.single (first := 4201) (initial := 1928) row_count_4202 (by decide)
private theorem atomic_4202_agreement : RowAgreement 4201 4202 := RowAgreement.single (first := 4201) modelrow_agrees_4202
private def atomic_4203 : CountSegment 1 10 13120 4202 4203 1928 1928 := CountSegment.single (first := 4202) (initial := 1928) row_count_4203 (by decide)
private theorem atomic_4203_agreement : RowAgreement 4202 4203 := RowAgreement.single (first := 4202) modelrow_agrees_4203
private def atomic_4204 : CountSegment 1 10 13120 4203 4204 1928 1929 := CountSegment.single (first := 4203) (initial := 1928) row_count_4204 (by decide)
private theorem atomic_4204_agreement : RowAgreement 4203 4204 := RowAgreement.single (first := 4203) modelrow_agrees_4204
private def atomic_4205 : CountSegment 1 10 13120 4204 4205 1929 1929 := CountSegment.single (first := 4204) (initial := 1929) row_count_4205 (by decide)
private theorem atomic_4205_agreement : RowAgreement 4204 4205 := RowAgreement.single (first := 4204) modelrow_agrees_4205
private def atomic_4206 : CountSegment 1 10 13120 4205 4206 1929 1929 := CountSegment.single (first := 4205) (initial := 1929) row_count_4206 (by decide)
private theorem atomic_4206_agreement : RowAgreement 4205 4206 := RowAgreement.single (first := 4205) modelrow_agrees_4206
private def atomic_4207 : CountSegment 1 10 13120 4206 4207 1929 1929 := CountSegment.single (first := 4206) (initial := 1929) row_count_4207 (by decide)
private theorem atomic_4207_agreement : RowAgreement 4206 4207 := RowAgreement.single (first := 4206) modelrow_agrees_4207
private def atomic_4208 : CountSegment 1 10 13120 4207 4208 1929 1929 := CountSegment.single (first := 4207) (initial := 1929) row_count_4208 (by decide)
private theorem atomic_4208_agreement : RowAgreement 4207 4208 := RowAgreement.single (first := 4207) modelrow_agrees_4208
private def atomic_4209 : CountSegment 1 10 13120 4208 4209 1929 1929 := CountSegment.single (first := 4208) (initial := 1929) row_count_4209 (by decide)
private theorem atomic_4209_agreement : RowAgreement 4208 4209 := RowAgreement.single (first := 4208) modelrow_agrees_4209
private def atomic_4210 : CountSegment 1 10 13120 4209 4210 1929 1930 := CountSegment.single (first := 4209) (initial := 1929) row_count_4210 (by decide)
private theorem atomic_4210_agreement : RowAgreement 4209 4210 := RowAgreement.single (first := 4209) modelrow_agrees_4210
private def atomic_4211 : CountSegment 1 10 13120 4210 4211 1930 1930 := CountSegment.single (first := 4210) (initial := 1930) row_count_4211 (by decide)
private theorem atomic_4211_agreement : RowAgreement 4210 4211 := RowAgreement.single (first := 4210) modelrow_agrees_4211
private def atomic_4212 : CountSegment 1 10 13120 4211 4212 1930 1931 := CountSegment.single (first := 4211) (initial := 1930) row_count_4212 (by decide)
private theorem atomic_4212_agreement : RowAgreement 4211 4212 := RowAgreement.single (first := 4211) modelrow_agrees_4212
private def atomic_4213 : CountSegment 1 10 13120 4212 4213 1931 1931 := CountSegment.single (first := 4212) (initial := 1931) row_count_4213 (by decide)
private theorem atomic_4213_agreement : RowAgreement 4212 4213 := RowAgreement.single (first := 4212) modelrow_agrees_4213
private def atomic_4214 : CountSegment 1 10 13120 4213 4214 1931 1931 := CountSegment.single (first := 4213) (initial := 1931) row_count_4214 (by decide)
private theorem atomic_4214_agreement : RowAgreement 4213 4214 := RowAgreement.single (first := 4213) modelrow_agrees_4214
private def atomic_4215 : CountSegment 1 10 13120 4214 4215 1931 1931 := CountSegment.single (first := 4214) (initial := 1931) row_count_4215 (by decide)
private theorem atomic_4215_agreement : RowAgreement 4214 4215 := RowAgreement.single (first := 4214) modelrow_agrees_4215
private def atomic_4216 : CountSegment 1 10 13120 4215 4216 1931 1931 := CountSegment.single (first := 4215) (initial := 1931) row_count_4216 (by decide)
private theorem atomic_4216_agreement : RowAgreement 4215 4216 := RowAgreement.single (first := 4215) modelrow_agrees_4216
private def atomic_4217 : CountSegment 1 10 13120 4216 4217 1931 1932 := CountSegment.single (first := 4216) (initial := 1931) row_count_4217 (by decide)
private theorem atomic_4217_agreement : RowAgreement 4216 4217 := RowAgreement.single (first := 4216) modelrow_agrees_4217
private def atomic_4218 : CountSegment 1 10 13120 4217 4218 1932 1933 := CountSegment.single (first := 4217) (initial := 1932) row_count_4218 (by decide)
private theorem atomic_4218_agreement : RowAgreement 4217 4218 := RowAgreement.single (first := 4217) modelrow_agrees_4218
private def atomic_4219 : CountSegment 1 10 13120 4218 4219 1933 1933 := CountSegment.single (first := 4218) (initial := 1933) row_count_4219 (by decide)
private theorem atomic_4219_agreement : RowAgreement 4218 4219 := RowAgreement.single (first := 4218) modelrow_agrees_4219
private def atomic_4220 : CountSegment 1 10 13120 4219 4220 1933 1933 := CountSegment.single (first := 4219) (initial := 1933) row_count_4220 (by decide)
private theorem atomic_4220_agreement : RowAgreement 4219 4220 := RowAgreement.single (first := 4219) modelrow_agrees_4220
private def atomic_4221 : CountSegment 1 10 13120 4220 4221 1933 1933 := CountSegment.single (first := 4220) (initial := 1933) row_count_4221 (by decide)
private theorem atomic_4221_agreement : RowAgreement 4220 4221 := RowAgreement.single (first := 4220) modelrow_agrees_4221
private def atomic_4222 : CountSegment 1 10 13120 4221 4222 1933 1933 := CountSegment.single (first := 4221) (initial := 1933) row_count_4222 (by decide)
private theorem atomic_4222_agreement : RowAgreement 4221 4222 := RowAgreement.single (first := 4221) modelrow_agrees_4222
private def atomic_4223 : CountSegment 1 10 13120 4222 4223 1933 1933 := CountSegment.single (first := 4222) (initial := 1933) row_count_4223 (by decide)
private theorem atomic_4223_agreement : RowAgreement 4222 4223 := RowAgreement.single (first := 4222) modelrow_agrees_4223
private def atomic_4224 : CountSegment 1 10 13120 4223 4224 1933 1933 := CountSegment.single (first := 4223) (initial := 1933) row_count_4224 (by decide)
private theorem atomic_4224_agreement : RowAgreement 4223 4224 := RowAgreement.single (first := 4223) modelrow_agrees_4224
private def within_032_000 : CountSegment 1 10 13120 4096 4098 1886 1889 := CountSegment.join atomic_4097 atomic_4098
private theorem within_032_000_agreement : RowAgreement 4096 4098 := RowAgreement.join atomic_4097_agreement atomic_4098_agreement
private def within_032_001 : CountSegment 1 10 13120 4098 4100 1889 1890 := CountSegment.join atomic_4099 atomic_4100
private theorem within_032_001_agreement : RowAgreement 4098 4100 := RowAgreement.join atomic_4099_agreement atomic_4100_agreement
private def within_032_002 : CountSegment 1 10 13120 4096 4100 1886 1890 := CountSegment.join within_032_000 within_032_001
private theorem within_032_002_agreement : RowAgreement 4096 4100 := RowAgreement.join within_032_000_agreement within_032_001_agreement
private def within_032_003 : CountSegment 1 10 13120 4100 4102 1890 1891 := CountSegment.join atomic_4101 atomic_4102
private theorem within_032_003_agreement : RowAgreement 4100 4102 := RowAgreement.join atomic_4101_agreement atomic_4102_agreement
private def within_032_004 : CountSegment 1 10 13120 4102 4104 1891 1892 := CountSegment.join atomic_4103 atomic_4104
private theorem within_032_004_agreement : RowAgreement 4102 4104 := RowAgreement.join atomic_4103_agreement atomic_4104_agreement
private def within_032_005 : CountSegment 1 10 13120 4100 4104 1890 1892 := CountSegment.join within_032_003 within_032_004
private theorem within_032_005_agreement : RowAgreement 4100 4104 := RowAgreement.join within_032_003_agreement within_032_004_agreement
private def within_032_006 : CountSegment 1 10 13120 4096 4104 1886 1892 := CountSegment.join within_032_002 within_032_005
private theorem within_032_006_agreement : RowAgreement 4096 4104 := RowAgreement.join within_032_002_agreement within_032_005_agreement
private def within_032_007 : CountSegment 1 10 13120 4104 4106 1892 1894 := CountSegment.join atomic_4105 atomic_4106
private theorem within_032_007_agreement : RowAgreement 4104 4106 := RowAgreement.join atomic_4105_agreement atomic_4106_agreement
private def within_032_008 : CountSegment 1 10 13120 4106 4108 1894 1894 := CountSegment.join atomic_4107 atomic_4108
private theorem within_032_008_agreement : RowAgreement 4106 4108 := RowAgreement.join atomic_4107_agreement atomic_4108_agreement
private def within_032_009 : CountSegment 1 10 13120 4104 4108 1892 1894 := CountSegment.join within_032_007 within_032_008
private theorem within_032_009_agreement : RowAgreement 4104 4108 := RowAgreement.join within_032_007_agreement within_032_008_agreement
private def within_032_010 : CountSegment 1 10 13120 4108 4110 1894 1895 := CountSegment.join atomic_4109 atomic_4110
private theorem within_032_010_agreement : RowAgreement 4108 4110 := RowAgreement.join atomic_4109_agreement atomic_4110_agreement
private def within_032_011 : CountSegment 1 10 13120 4110 4112 1895 1895 := CountSegment.join atomic_4111 atomic_4112
private theorem within_032_011_agreement : RowAgreement 4110 4112 := RowAgreement.join atomic_4111_agreement atomic_4112_agreement
private def within_032_012 : CountSegment 1 10 13120 4108 4112 1894 1895 := CountSegment.join within_032_010 within_032_011
private theorem within_032_012_agreement : RowAgreement 4108 4112 := RowAgreement.join within_032_010_agreement within_032_011_agreement
private def within_032_013 : CountSegment 1 10 13120 4104 4112 1892 1895 := CountSegment.join within_032_009 within_032_012
private theorem within_032_013_agreement : RowAgreement 4104 4112 := RowAgreement.join within_032_009_agreement within_032_012_agreement
private def within_032_014 : CountSegment 1 10 13120 4096 4112 1886 1895 := CountSegment.join within_032_006 within_032_013
private theorem within_032_014_agreement : RowAgreement 4096 4112 := RowAgreement.join within_032_006_agreement within_032_013_agreement
private def within_032_015 : CountSegment 1 10 13120 4112 4114 1895 1897 := CountSegment.join atomic_4113 atomic_4114
private theorem within_032_015_agreement : RowAgreement 4112 4114 := RowAgreement.join atomic_4113_agreement atomic_4114_agreement
private def within_032_016 : CountSegment 1 10 13120 4114 4116 1897 1899 := CountSegment.join atomic_4115 atomic_4116
private theorem within_032_016_agreement : RowAgreement 4114 4116 := RowAgreement.join atomic_4115_agreement atomic_4116_agreement
private def within_032_017 : CountSegment 1 10 13120 4112 4116 1895 1899 := CountSegment.join within_032_015 within_032_016
private theorem within_032_017_agreement : RowAgreement 4112 4116 := RowAgreement.join within_032_015_agreement within_032_016_agreement
private def within_032_018 : CountSegment 1 10 13120 4116 4118 1899 1899 := CountSegment.join atomic_4117 atomic_4118
private theorem within_032_018_agreement : RowAgreement 4116 4118 := RowAgreement.join atomic_4117_agreement atomic_4118_agreement
private def within_032_019 : CountSegment 1 10 13120 4118 4120 1899 1900 := CountSegment.join atomic_4119 atomic_4120
private theorem within_032_019_agreement : RowAgreement 4118 4120 := RowAgreement.join atomic_4119_agreement atomic_4120_agreement
private def within_032_020 : CountSegment 1 10 13120 4116 4120 1899 1900 := CountSegment.join within_032_018 within_032_019
private theorem within_032_020_agreement : RowAgreement 4116 4120 := RowAgreement.join within_032_018_agreement within_032_019_agreement
private def within_032_021 : CountSegment 1 10 13120 4112 4120 1895 1900 := CountSegment.join within_032_017 within_032_020
private theorem within_032_021_agreement : RowAgreement 4112 4120 := RowAgreement.join within_032_017_agreement within_032_020_agreement
private def within_032_022 : CountSegment 1 10 13120 4120 4122 1900 1900 := CountSegment.join atomic_4121 atomic_4122
private theorem within_032_022_agreement : RowAgreement 4120 4122 := RowAgreement.join atomic_4121_agreement atomic_4122_agreement
private def within_032_023 : CountSegment 1 10 13120 4122 4124 1900 1901 := CountSegment.join atomic_4123 atomic_4124
private theorem within_032_023_agreement : RowAgreement 4122 4124 := RowAgreement.join atomic_4123_agreement atomic_4124_agreement
private def within_032_024 : CountSegment 1 10 13120 4120 4124 1900 1901 := CountSegment.join within_032_022 within_032_023
private theorem within_032_024_agreement : RowAgreement 4120 4124 := RowAgreement.join within_032_022_agreement within_032_023_agreement
private def within_032_025 : CountSegment 1 10 13120 4124 4126 1901 1901 := CountSegment.join atomic_4125 atomic_4126
private theorem within_032_025_agreement : RowAgreement 4124 4126 := RowAgreement.join atomic_4125_agreement atomic_4126_agreement
private def within_032_026 : CountSegment 1 10 13120 4126 4128 1901 1902 := CountSegment.join atomic_4127 atomic_4128
private theorem within_032_026_agreement : RowAgreement 4126 4128 := RowAgreement.join atomic_4127_agreement atomic_4128_agreement
private def within_032_027 : CountSegment 1 10 13120 4124 4128 1901 1902 := CountSegment.join within_032_025 within_032_026
private theorem within_032_027_agreement : RowAgreement 4124 4128 := RowAgreement.join within_032_025_agreement within_032_026_agreement
private def within_032_028 : CountSegment 1 10 13120 4120 4128 1900 1902 := CountSegment.join within_032_024 within_032_027
private theorem within_032_028_agreement : RowAgreement 4120 4128 := RowAgreement.join within_032_024_agreement within_032_027_agreement
private def within_032_029 : CountSegment 1 10 13120 4112 4128 1895 1902 := CountSegment.join within_032_021 within_032_028
private theorem within_032_029_agreement : RowAgreement 4112 4128 := RowAgreement.join within_032_021_agreement within_032_028_agreement
private def within_032_030 : CountSegment 1 10 13120 4096 4128 1886 1902 := CountSegment.join within_032_014 within_032_029
private theorem within_032_030_agreement : RowAgreement 4096 4128 := RowAgreement.join within_032_014_agreement within_032_029_agreement
private def within_032_031 : CountSegment 1 10 13120 4128 4130 1902 1903 := CountSegment.join atomic_4129 atomic_4130
private theorem within_032_031_agreement : RowAgreement 4128 4130 := RowAgreement.join atomic_4129_agreement atomic_4130_agreement
private def within_032_032 : CountSegment 1 10 13120 4130 4132 1903 1903 := CountSegment.join atomic_4131 atomic_4132
private theorem within_032_032_agreement : RowAgreement 4130 4132 := RowAgreement.join atomic_4131_agreement atomic_4132_agreement
private def within_032_033 : CountSegment 1 10 13120 4128 4132 1902 1903 := CountSegment.join within_032_031 within_032_032
private theorem within_032_033_agreement : RowAgreement 4128 4132 := RowAgreement.join within_032_031_agreement within_032_032_agreement
private def within_032_034 : CountSegment 1 10 13120 4132 4134 1903 1905 := CountSegment.join atomic_4133 atomic_4134
private theorem within_032_034_agreement : RowAgreement 4132 4134 := RowAgreement.join atomic_4133_agreement atomic_4134_agreement
private def within_032_035 : CountSegment 1 10 13120 4134 4136 1905 1906 := CountSegment.join atomic_4135 atomic_4136
private theorem within_032_035_agreement : RowAgreement 4134 4136 := RowAgreement.join atomic_4135_agreement atomic_4136_agreement
private def within_032_036 : CountSegment 1 10 13120 4132 4136 1903 1906 := CountSegment.join within_032_034 within_032_035
private theorem within_032_036_agreement : RowAgreement 4132 4136 := RowAgreement.join within_032_034_agreement within_032_035_agreement
private def within_032_037 : CountSegment 1 10 13120 4128 4136 1902 1906 := CountSegment.join within_032_033 within_032_036
private theorem within_032_037_agreement : RowAgreement 4128 4136 := RowAgreement.join within_032_033_agreement within_032_036_agreement
private def within_032_038 : CountSegment 1 10 13120 4136 4138 1906 1906 := CountSegment.join atomic_4137 atomic_4138
private theorem within_032_038_agreement : RowAgreement 4136 4138 := RowAgreement.join atomic_4137_agreement atomic_4138_agreement
private def within_032_039 : CountSegment 1 10 13120 4138 4140 1906 1907 := CountSegment.join atomic_4139 atomic_4140
private theorem within_032_039_agreement : RowAgreement 4138 4140 := RowAgreement.join atomic_4139_agreement atomic_4140_agreement
private def within_032_040 : CountSegment 1 10 13120 4136 4140 1906 1907 := CountSegment.join within_032_038 within_032_039
private theorem within_032_040_agreement : RowAgreement 4136 4140 := RowAgreement.join within_032_038_agreement within_032_039_agreement
private def within_032_041 : CountSegment 1 10 13120 4140 4142 1907 1907 := CountSegment.join atomic_4141 atomic_4142
private theorem within_032_041_agreement : RowAgreement 4140 4142 := RowAgreement.join atomic_4141_agreement atomic_4142_agreement
private def within_032_042 : CountSegment 1 10 13120 4142 4144 1907 1908 := CountSegment.join atomic_4143 atomic_4144
private theorem within_032_042_agreement : RowAgreement 4142 4144 := RowAgreement.join atomic_4143_agreement atomic_4144_agreement
private def within_032_043 : CountSegment 1 10 13120 4140 4144 1907 1908 := CountSegment.join within_032_041 within_032_042
private theorem within_032_043_agreement : RowAgreement 4140 4144 := RowAgreement.join within_032_041_agreement within_032_042_agreement
private def within_032_044 : CountSegment 1 10 13120 4136 4144 1906 1908 := CountSegment.join within_032_040 within_032_043
private theorem within_032_044_agreement : RowAgreement 4136 4144 := RowAgreement.join within_032_040_agreement within_032_043_agreement
private def within_032_045 : CountSegment 1 10 13120 4128 4144 1902 1908 := CountSegment.join within_032_037 within_032_044
private theorem within_032_045_agreement : RowAgreement 4128 4144 := RowAgreement.join within_032_037_agreement within_032_044_agreement
private def within_032_046 : CountSegment 1 10 13120 4144 4146 1908 1909 := CountSegment.join atomic_4145 atomic_4146
private theorem within_032_046_agreement : RowAgreement 4144 4146 := RowAgreement.join atomic_4145_agreement atomic_4146_agreement
private def within_032_047 : CountSegment 1 10 13120 4146 4148 1909 1910 := CountSegment.join atomic_4147 atomic_4148
private theorem within_032_047_agreement : RowAgreement 4146 4148 := RowAgreement.join atomic_4147_agreement atomic_4148_agreement
private def within_032_048 : CountSegment 1 10 13120 4144 4148 1908 1910 := CountSegment.join within_032_046 within_032_047
private theorem within_032_048_agreement : RowAgreement 4144 4148 := RowAgreement.join within_032_046_agreement within_032_047_agreement
private def within_032_049 : CountSegment 1 10 13120 4148 4150 1910 1911 := CountSegment.join atomic_4149 atomic_4150
private theorem within_032_049_agreement : RowAgreement 4148 4150 := RowAgreement.join atomic_4149_agreement atomic_4150_agreement
private def within_032_050 : CountSegment 1 10 13120 4150 4152 1911 1912 := CountSegment.join atomic_4151 atomic_4152
private theorem within_032_050_agreement : RowAgreement 4150 4152 := RowAgreement.join atomic_4151_agreement atomic_4152_agreement
private def within_032_051 : CountSegment 1 10 13120 4148 4152 1910 1912 := CountSegment.join within_032_049 within_032_050
private theorem within_032_051_agreement : RowAgreement 4148 4152 := RowAgreement.join within_032_049_agreement within_032_050_agreement
private def within_032_052 : CountSegment 1 10 13120 4144 4152 1908 1912 := CountSegment.join within_032_048 within_032_051
private theorem within_032_052_agreement : RowAgreement 4144 4152 := RowAgreement.join within_032_048_agreement within_032_051_agreement
private def within_032_053 : CountSegment 1 10 13120 4152 4154 1912 1912 := CountSegment.join atomic_4153 atomic_4154
private theorem within_032_053_agreement : RowAgreement 4152 4154 := RowAgreement.join atomic_4153_agreement atomic_4154_agreement
private def within_032_054 : CountSegment 1 10 13120 4154 4156 1912 1912 := CountSegment.join atomic_4155 atomic_4156
private theorem within_032_054_agreement : RowAgreement 4154 4156 := RowAgreement.join atomic_4155_agreement atomic_4156_agreement
private def within_032_055 : CountSegment 1 10 13120 4152 4156 1912 1912 := CountSegment.join within_032_053 within_032_054
private theorem within_032_055_agreement : RowAgreement 4152 4156 := RowAgreement.join within_032_053_agreement within_032_054_agreement
private def within_032_056 : CountSegment 1 10 13120 4156 4158 1912 1913 := CountSegment.join atomic_4157 atomic_4158
private theorem within_032_056_agreement : RowAgreement 4156 4158 := RowAgreement.join atomic_4157_agreement atomic_4158_agreement
private def within_032_057 : CountSegment 1 10 13120 4158 4160 1913 1913 := CountSegment.join atomic_4159 atomic_4160
private theorem within_032_057_agreement : RowAgreement 4158 4160 := RowAgreement.join atomic_4159_agreement atomic_4160_agreement
private def within_032_058 : CountSegment 1 10 13120 4156 4160 1912 1913 := CountSegment.join within_032_056 within_032_057
private theorem within_032_058_agreement : RowAgreement 4156 4160 := RowAgreement.join within_032_056_agreement within_032_057_agreement
private def within_032_059 : CountSegment 1 10 13120 4152 4160 1912 1913 := CountSegment.join within_032_055 within_032_058
private theorem within_032_059_agreement : RowAgreement 4152 4160 := RowAgreement.join within_032_055_agreement within_032_058_agreement
private def within_032_060 : CountSegment 1 10 13120 4144 4160 1908 1913 := CountSegment.join within_032_052 within_032_059
private theorem within_032_060_agreement : RowAgreement 4144 4160 := RowAgreement.join within_032_052_agreement within_032_059_agreement
private def within_032_061 : CountSegment 1 10 13120 4128 4160 1902 1913 := CountSegment.join within_032_045 within_032_060
private theorem within_032_061_agreement : RowAgreement 4128 4160 := RowAgreement.join within_032_045_agreement within_032_060_agreement
private def within_032_062 : CountSegment 1 10 13120 4096 4160 1886 1913 := CountSegment.join within_032_030 within_032_061
private theorem within_032_062_agreement : RowAgreement 4096 4160 := RowAgreement.join within_032_030_agreement within_032_061_agreement
private def within_032_063 : CountSegment 1 10 13120 4160 4162 1913 1917 := CountSegment.join atomic_4161 atomic_4162
private theorem within_032_063_agreement : RowAgreement 4160 4162 := RowAgreement.join atomic_4161_agreement atomic_4162_agreement
private def within_032_064 : CountSegment 1 10 13120 4162 4164 1917 1917 := CountSegment.join atomic_4163 atomic_4164
private theorem within_032_064_agreement : RowAgreement 4162 4164 := RowAgreement.join atomic_4163_agreement atomic_4164_agreement
private def within_032_065 : CountSegment 1 10 13120 4160 4164 1913 1917 := CountSegment.join within_032_063 within_032_064
private theorem within_032_065_agreement : RowAgreement 4160 4164 := RowAgreement.join within_032_063_agreement within_032_064_agreement
private def within_032_066 : CountSegment 1 10 13120 4164 4166 1917 1917 := CountSegment.join atomic_4165 atomic_4166
private theorem within_032_066_agreement : RowAgreement 4164 4166 := RowAgreement.join atomic_4165_agreement atomic_4166_agreement
private def within_032_067 : CountSegment 1 10 13120 4166 4168 1917 1917 := CountSegment.join atomic_4167 atomic_4168
private theorem within_032_067_agreement : RowAgreement 4166 4168 := RowAgreement.join atomic_4167_agreement atomic_4168_agreement
private def within_032_068 : CountSegment 1 10 13120 4164 4168 1917 1917 := CountSegment.join within_032_066 within_032_067
private theorem within_032_068_agreement : RowAgreement 4164 4168 := RowAgreement.join within_032_066_agreement within_032_067_agreement
private def within_032_069 : CountSegment 1 10 13120 4160 4168 1913 1917 := CountSegment.join within_032_065 within_032_068
private theorem within_032_069_agreement : RowAgreement 4160 4168 := RowAgreement.join within_032_065_agreement within_032_068_agreement
private def within_032_070 : CountSegment 1 10 13120 4168 4170 1917 1917 := CountSegment.join atomic_4169 atomic_4170
private theorem within_032_070_agreement : RowAgreement 4168 4170 := RowAgreement.join atomic_4169_agreement atomic_4170_agreement
private def within_032_071 : CountSegment 1 10 13120 4170 4172 1917 1917 := CountSegment.join atomic_4171 atomic_4172
private theorem within_032_071_agreement : RowAgreement 4170 4172 := RowAgreement.join atomic_4171_agreement atomic_4172_agreement
private def within_032_072 : CountSegment 1 10 13120 4168 4172 1917 1917 := CountSegment.join within_032_070 within_032_071
private theorem within_032_072_agreement : RowAgreement 4168 4172 := RowAgreement.join within_032_070_agreement within_032_071_agreement
private def within_032_073 : CountSegment 1 10 13120 4172 4174 1917 1917 := CountSegment.join atomic_4173 atomic_4174
private theorem within_032_073_agreement : RowAgreement 4172 4174 := RowAgreement.join atomic_4173_agreement atomic_4174_agreement
private def within_032_074 : CountSegment 1 10 13120 4174 4176 1917 1918 := CountSegment.join atomic_4175 atomic_4176
private theorem within_032_074_agreement : RowAgreement 4174 4176 := RowAgreement.join atomic_4175_agreement atomic_4176_agreement
private def within_032_075 : CountSegment 1 10 13120 4172 4176 1917 1918 := CountSegment.join within_032_073 within_032_074
private theorem within_032_075_agreement : RowAgreement 4172 4176 := RowAgreement.join within_032_073_agreement within_032_074_agreement
private def within_032_076 : CountSegment 1 10 13120 4168 4176 1917 1918 := CountSegment.join within_032_072 within_032_075
private theorem within_032_076_agreement : RowAgreement 4168 4176 := RowAgreement.join within_032_072_agreement within_032_075_agreement
private def within_032_077 : CountSegment 1 10 13120 4160 4176 1913 1918 := CountSegment.join within_032_069 within_032_076
private theorem within_032_077_agreement : RowAgreement 4160 4176 := RowAgreement.join within_032_069_agreement within_032_076_agreement
private def within_032_078 : CountSegment 1 10 13120 4176 4178 1918 1918 := CountSegment.join atomic_4177 atomic_4178
private theorem within_032_078_agreement : RowAgreement 4176 4178 := RowAgreement.join atomic_4177_agreement atomic_4178_agreement
private def within_032_079 : CountSegment 1 10 13120 4178 4180 1918 1918 := CountSegment.join atomic_4179 atomic_4180
private theorem within_032_079_agreement : RowAgreement 4178 4180 := RowAgreement.join atomic_4179_agreement atomic_4180_agreement
private def within_032_080 : CountSegment 1 10 13120 4176 4180 1918 1918 := CountSegment.join within_032_078 within_032_079
private theorem within_032_080_agreement : RowAgreement 4176 4180 := RowAgreement.join within_032_078_agreement within_032_079_agreement
private def within_032_081 : CountSegment 1 10 13120 4180 4182 1918 1918 := CountSegment.join atomic_4181 atomic_4182
private theorem within_032_081_agreement : RowAgreement 4180 4182 := RowAgreement.join atomic_4181_agreement atomic_4182_agreement
private def within_032_082 : CountSegment 1 10 13120 4182 4184 1918 1919 := CountSegment.join atomic_4183 atomic_4184
private theorem within_032_082_agreement : RowAgreement 4182 4184 := RowAgreement.join atomic_4183_agreement atomic_4184_agreement
private def within_032_083 : CountSegment 1 10 13120 4180 4184 1918 1919 := CountSegment.join within_032_081 within_032_082
private theorem within_032_083_agreement : RowAgreement 4180 4184 := RowAgreement.join within_032_081_agreement within_032_082_agreement
private def within_032_084 : CountSegment 1 10 13120 4176 4184 1918 1919 := CountSegment.join within_032_080 within_032_083
private theorem within_032_084_agreement : RowAgreement 4176 4184 := RowAgreement.join within_032_080_agreement within_032_083_agreement
private def within_032_085 : CountSegment 1 10 13120 4184 4186 1919 1919 := CountSegment.join atomic_4185 atomic_4186
private theorem within_032_085_agreement : RowAgreement 4184 4186 := RowAgreement.join atomic_4185_agreement atomic_4186_agreement
private def within_032_086 : CountSegment 1 10 13120 4186 4188 1919 1921 := CountSegment.join atomic_4187 atomic_4188
private theorem within_032_086_agreement : RowAgreement 4186 4188 := RowAgreement.join atomic_4187_agreement atomic_4188_agreement
private def within_032_087 : CountSegment 1 10 13120 4184 4188 1919 1921 := CountSegment.join within_032_085 within_032_086
private theorem within_032_087_agreement : RowAgreement 4184 4188 := RowAgreement.join within_032_085_agreement within_032_086_agreement
private def within_032_088 : CountSegment 1 10 13120 4188 4190 1921 1922 := CountSegment.join atomic_4189 atomic_4190
private theorem within_032_088_agreement : RowAgreement 4188 4190 := RowAgreement.join atomic_4189_agreement atomic_4190_agreement
private def within_032_089 : CountSegment 1 10 13120 4190 4192 1922 1923 := CountSegment.join atomic_4191 atomic_4192
private theorem within_032_089_agreement : RowAgreement 4190 4192 := RowAgreement.join atomic_4191_agreement atomic_4192_agreement
private def within_032_090 : CountSegment 1 10 13120 4188 4192 1921 1923 := CountSegment.join within_032_088 within_032_089
private theorem within_032_090_agreement : RowAgreement 4188 4192 := RowAgreement.join within_032_088_agreement within_032_089_agreement
private def within_032_091 : CountSegment 1 10 13120 4184 4192 1919 1923 := CountSegment.join within_032_087 within_032_090
private theorem within_032_091_agreement : RowAgreement 4184 4192 := RowAgreement.join within_032_087_agreement within_032_090_agreement
private def within_032_092 : CountSegment 1 10 13120 4176 4192 1918 1923 := CountSegment.join within_032_084 within_032_091
private theorem within_032_092_agreement : RowAgreement 4176 4192 := RowAgreement.join within_032_084_agreement within_032_091_agreement
private def within_032_093 : CountSegment 1 10 13120 4160 4192 1913 1923 := CountSegment.join within_032_077 within_032_092
private theorem within_032_093_agreement : RowAgreement 4160 4192 := RowAgreement.join within_032_077_agreement within_032_092_agreement
private def within_032_094 : CountSegment 1 10 13120 4192 4194 1923 1926 := CountSegment.join atomic_4193 atomic_4194
private theorem within_032_094_agreement : RowAgreement 4192 4194 := RowAgreement.join atomic_4193_agreement atomic_4194_agreement
private def within_032_095 : CountSegment 1 10 13120 4194 4196 1926 1926 := CountSegment.join atomic_4195 atomic_4196
private theorem within_032_095_agreement : RowAgreement 4194 4196 := RowAgreement.join atomic_4195_agreement atomic_4196_agreement
private def within_032_096 : CountSegment 1 10 13120 4192 4196 1923 1926 := CountSegment.join within_032_094 within_032_095
private theorem within_032_096_agreement : RowAgreement 4192 4196 := RowAgreement.join within_032_094_agreement within_032_095_agreement
private def within_032_097 : CountSegment 1 10 13120 4196 4198 1926 1927 := CountSegment.join atomic_4197 atomic_4198
private theorem within_032_097_agreement : RowAgreement 4196 4198 := RowAgreement.join atomic_4197_agreement atomic_4198_agreement
private def within_032_098 : CountSegment 1 10 13120 4198 4200 1927 1928 := CountSegment.join atomic_4199 atomic_4200
private theorem within_032_098_agreement : RowAgreement 4198 4200 := RowAgreement.join atomic_4199_agreement atomic_4200_agreement
private def within_032_099 : CountSegment 1 10 13120 4196 4200 1926 1928 := CountSegment.join within_032_097 within_032_098
private theorem within_032_099_agreement : RowAgreement 4196 4200 := RowAgreement.join within_032_097_agreement within_032_098_agreement
private def within_032_100 : CountSegment 1 10 13120 4192 4200 1923 1928 := CountSegment.join within_032_096 within_032_099
private theorem within_032_100_agreement : RowAgreement 4192 4200 := RowAgreement.join within_032_096_agreement within_032_099_agreement
private def within_032_101 : CountSegment 1 10 13120 4200 4202 1928 1928 := CountSegment.join atomic_4201 atomic_4202
private theorem within_032_101_agreement : RowAgreement 4200 4202 := RowAgreement.join atomic_4201_agreement atomic_4202_agreement
private def within_032_102 : CountSegment 1 10 13120 4202 4204 1928 1929 := CountSegment.join atomic_4203 atomic_4204
private theorem within_032_102_agreement : RowAgreement 4202 4204 := RowAgreement.join atomic_4203_agreement atomic_4204_agreement
private def within_032_103 : CountSegment 1 10 13120 4200 4204 1928 1929 := CountSegment.join within_032_101 within_032_102
private theorem within_032_103_agreement : RowAgreement 4200 4204 := RowAgreement.join within_032_101_agreement within_032_102_agreement
private def within_032_104 : CountSegment 1 10 13120 4204 4206 1929 1929 := CountSegment.join atomic_4205 atomic_4206
private theorem within_032_104_agreement : RowAgreement 4204 4206 := RowAgreement.join atomic_4205_agreement atomic_4206_agreement
private def within_032_105 : CountSegment 1 10 13120 4206 4208 1929 1929 := CountSegment.join atomic_4207 atomic_4208
private theorem within_032_105_agreement : RowAgreement 4206 4208 := RowAgreement.join atomic_4207_agreement atomic_4208_agreement
private def within_032_106 : CountSegment 1 10 13120 4204 4208 1929 1929 := CountSegment.join within_032_104 within_032_105
private theorem within_032_106_agreement : RowAgreement 4204 4208 := RowAgreement.join within_032_104_agreement within_032_105_agreement
private def within_032_107 : CountSegment 1 10 13120 4200 4208 1928 1929 := CountSegment.join within_032_103 within_032_106
private theorem within_032_107_agreement : RowAgreement 4200 4208 := RowAgreement.join within_032_103_agreement within_032_106_agreement
private def within_032_108 : CountSegment 1 10 13120 4192 4208 1923 1929 := CountSegment.join within_032_100 within_032_107
private theorem within_032_108_agreement : RowAgreement 4192 4208 := RowAgreement.join within_032_100_agreement within_032_107_agreement
private def within_032_109 : CountSegment 1 10 13120 4208 4210 1929 1930 := CountSegment.join atomic_4209 atomic_4210
private theorem within_032_109_agreement : RowAgreement 4208 4210 := RowAgreement.join atomic_4209_agreement atomic_4210_agreement
private def within_032_110 : CountSegment 1 10 13120 4210 4212 1930 1931 := CountSegment.join atomic_4211 atomic_4212
private theorem within_032_110_agreement : RowAgreement 4210 4212 := RowAgreement.join atomic_4211_agreement atomic_4212_agreement
private def within_032_111 : CountSegment 1 10 13120 4208 4212 1929 1931 := CountSegment.join within_032_109 within_032_110
private theorem within_032_111_agreement : RowAgreement 4208 4212 := RowAgreement.join within_032_109_agreement within_032_110_agreement
private def within_032_112 : CountSegment 1 10 13120 4212 4214 1931 1931 := CountSegment.join atomic_4213 atomic_4214
private theorem within_032_112_agreement : RowAgreement 4212 4214 := RowAgreement.join atomic_4213_agreement atomic_4214_agreement
private def within_032_113 : CountSegment 1 10 13120 4214 4216 1931 1931 := CountSegment.join atomic_4215 atomic_4216
private theorem within_032_113_agreement : RowAgreement 4214 4216 := RowAgreement.join atomic_4215_agreement atomic_4216_agreement
private def within_032_114 : CountSegment 1 10 13120 4212 4216 1931 1931 := CountSegment.join within_032_112 within_032_113
private theorem within_032_114_agreement : RowAgreement 4212 4216 := RowAgreement.join within_032_112_agreement within_032_113_agreement
private def within_032_115 : CountSegment 1 10 13120 4208 4216 1929 1931 := CountSegment.join within_032_111 within_032_114
private theorem within_032_115_agreement : RowAgreement 4208 4216 := RowAgreement.join within_032_111_agreement within_032_114_agreement
private def within_032_116 : CountSegment 1 10 13120 4216 4218 1931 1933 := CountSegment.join atomic_4217 atomic_4218
private theorem within_032_116_agreement : RowAgreement 4216 4218 := RowAgreement.join atomic_4217_agreement atomic_4218_agreement
private def within_032_117 : CountSegment 1 10 13120 4218 4220 1933 1933 := CountSegment.join atomic_4219 atomic_4220
private theorem within_032_117_agreement : RowAgreement 4218 4220 := RowAgreement.join atomic_4219_agreement atomic_4220_agreement
private def within_032_118 : CountSegment 1 10 13120 4216 4220 1931 1933 := CountSegment.join within_032_116 within_032_117
private theorem within_032_118_agreement : RowAgreement 4216 4220 := RowAgreement.join within_032_116_agreement within_032_117_agreement
private def within_032_119 : CountSegment 1 10 13120 4220 4222 1933 1933 := CountSegment.join atomic_4221 atomic_4222
private theorem within_032_119_agreement : RowAgreement 4220 4222 := RowAgreement.join atomic_4221_agreement atomic_4222_agreement
private def within_032_120 : CountSegment 1 10 13120 4222 4224 1933 1933 := CountSegment.join atomic_4223 atomic_4224
private theorem within_032_120_agreement : RowAgreement 4222 4224 := RowAgreement.join atomic_4223_agreement atomic_4224_agreement
private def within_032_121 : CountSegment 1 10 13120 4220 4224 1933 1933 := CountSegment.join within_032_119 within_032_120
private theorem within_032_121_agreement : RowAgreement 4220 4224 := RowAgreement.join within_032_119_agreement within_032_120_agreement
private def within_032_122 : CountSegment 1 10 13120 4216 4224 1931 1933 := CountSegment.join within_032_118 within_032_121
private theorem within_032_122_agreement : RowAgreement 4216 4224 := RowAgreement.join within_032_118_agreement within_032_121_agreement
private def within_032_123 : CountSegment 1 10 13120 4208 4224 1929 1933 := CountSegment.join within_032_115 within_032_122
private theorem within_032_123_agreement : RowAgreement 4208 4224 := RowAgreement.join within_032_115_agreement within_032_122_agreement
private def within_032_124 : CountSegment 1 10 13120 4192 4224 1923 1933 := CountSegment.join within_032_108 within_032_123
private theorem within_032_124_agreement : RowAgreement 4192 4224 := RowAgreement.join within_032_108_agreement within_032_123_agreement
private def within_032_125 : CountSegment 1 10 13120 4160 4224 1913 1933 := CountSegment.join within_032_093 within_032_124
private theorem within_032_125_agreement : RowAgreement 4160 4224 := RowAgreement.join within_032_093_agreement within_032_124_agreement
private def within_032_126 : CountSegment 1 10 13120 4096 4224 1886 1933 := CountSegment.join within_032_062 within_032_125
private theorem within_032_126_agreement : RowAgreement 4096 4224 := RowAgreement.join within_032_062_agreement within_032_125_agreement
def segment_032 : CountSegment 1 10 13120 4096 4224 1886 1933 := within_032_126
theorem agreement_032 : RowAgreement 4096 4224 := within_032_126_agreement
end ProjectEulerSolutions.P58Numerical
