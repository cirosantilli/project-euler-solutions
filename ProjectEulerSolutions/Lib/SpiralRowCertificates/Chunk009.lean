import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1153 : RowCertificate 1153 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1153 1) := by change ¬ Nat.Prime 5319943; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 279997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1153 2) := by change ¬ Nat.Prime 5317637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 409049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1153 3) := by change ¬ Nat.Prime 5315331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1771777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1153 : row 1153 = 0 := (row_1153.checked).trans rfl
theorem model_count_1153 : modelRow 1153 = 0 := by decide
theorem modelrow_agrees_1153 : modelRow 1153 = row 1153 := model_count_1153.trans row_count_1153.symm
def row_1154 : RowCertificate 1154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1154 1) := by change ¬ Nat.Prime 5329173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1776391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1154 2) := by change ¬ Nat.Prime 5326865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1065373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1154 3) := by change ¬ Nat.Prime 5324557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 760651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1154 : row 1154 = 0 := (row_1154.checked).trans rfl
theorem model_count_1154 : modelRow 1154 = 0 := by decide
theorem modelrow_agrees_1154 : modelRow 1154 = row 1154 := model_count_1154.trans row_count_1154.symm
def row_1155 : RowCertificate 1155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1155 1) := by change ¬ Nat.Prime 5338411; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 410647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1155 2) := by change Nat.Prime 5336101; exact ProjectEulerSolutions.P58PrimeNodes.p_5336101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1155 3) := by change Nat.Prime 5333791; exact ProjectEulerSolutions.P58PrimeNodes.p_5333791.prime
    exact decide_eq_true prime
theorem row_count_1155 : row 1155 = 2 := (row_1155.checked).trans rfl
theorem model_count_1155 : modelRow 1155 = 2 := by decide
theorem modelrow_agrees_1155 : modelRow 1155 = row 1155 := model_count_1155.trans row_count_1155.symm
def row_1156 : RowCertificate 1156 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1156 1) := by change ¬ Nat.Prime 5347657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 763951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1156 2) := by change ¬ Nat.Prime 5345345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1069069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1156 3) := by change ¬ Nat.Prime 5343033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1781011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1156 : row 1156 = 0 := (row_1156.checked).trans rfl
theorem model_count_1156 : modelRow 1156 = 0 := by decide
theorem modelrow_agrees_1156 : modelRow 1156 = row 1156 := model_count_1156.trans row_count_1156.symm
def row_1157 : RowCertificate 1157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1157 1) := by change ¬ Nat.Prime 5356911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1785637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1157 2) := by change Nat.Prime 5354597; exact ProjectEulerSolutions.P58PrimeNodes.p_5354597.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1157 3) := by change Nat.Prime 5352283; exact ProjectEulerSolutions.P58PrimeNodes.p_5352283.prime
    exact decide_eq_true prime
theorem row_count_1157 : row 1157 = 2 := (row_1157.checked).trans rfl
theorem model_count_1157 : modelRow 1157 = 2 := by decide
theorem modelrow_agrees_1157 : modelRow 1157 = row 1157 := model_count_1157.trans row_count_1157.symm
def row_1158 : RowCertificate 1158 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1158 1) := by change ¬ Nat.Prime 5366173; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 6247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1158 2) := by change ¬ Nat.Prime 5363857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 315521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1158 3) := by change ¬ Nat.Prime 5361541; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 124687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1158 : row 1158 = 0 := (row_1158.checked).trans rfl
theorem model_count_1158 : modelRow 1158 = 0 := by decide
theorem modelrow_agrees_1158 : modelRow 1158 = row 1158 := model_count_1158.trans row_count_1158.symm
def row_1159 : RowCertificate 1159 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1159 1) := by change Nat.Prime 5375443; exact ProjectEulerSolutions.P58PrimeNodes.p_5375443.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1159 2) := by change ¬ Nat.Prime 5373125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1074625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1159 3) := by change ¬ Nat.Prime 5370807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1790269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1159 : row 1159 = 1 := (row_1159.checked).trans rfl
theorem model_count_1159 : modelRow 1159 = 1 := by decide
theorem modelrow_agrees_1159 : modelRow 1159 = row 1159 := model_count_1159.trans row_count_1159.symm
def row_1160 : RowCertificate 1160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1160 1) := by change ¬ Nat.Prime 5384721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1794907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1160 2) := by change Nat.Prime 5382401; exact ProjectEulerSolutions.P58PrimeNodes.p_5382401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1160 3) := by change ¬ Nat.Prime 5380081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 768583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1160 : row 1160 = 1 := (row_1160.checked).trans rfl
theorem model_count_1160 : modelRow 1160 = 1 := by decide
theorem modelrow_agrees_1160 : modelRow 1160 = row 1160 := model_count_1160.trans row_count_1160.symm
def row_1161 : RowCertificate 1161 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1161 1) := by change ¬ Nat.Prime 5394007; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 52369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1161 2) := by change ¬ Nat.Prime 5391685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1078337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1161 3) := by change ¬ Nat.Prime 5389363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 769909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1161 : row 1161 = 0 := (row_1161.checked).trans rfl
theorem model_count_1161 : modelRow 1161 = 0 := by decide
theorem modelrow_agrees_1161 : modelRow 1161 = row 1161 := model_count_1161.trans row_count_1161.symm
def row_1162 : RowCertificate 1162 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1162 1) := by change Nat.Prime 5403301; exact ProjectEulerSolutions.P58PrimeNodes.p_5403301.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1162 2) := by change ¬ Nat.Prime 5400977; exact PrattCertificates.composite_of_product (factor := 1433) (cofactor := 3769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1162 3) := by change ¬ Nat.Prime 5398653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1799551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1162 : row 1162 = 1 := (row_1162.checked).trans rfl
theorem model_count_1162 : modelRow 1162 = 1 := by decide
theorem modelrow_agrees_1162 : modelRow 1162 = row 1162 := model_count_1162.trans row_count_1162.symm
def row_1163 : RowCertificate 1163 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1163 1) := by change ¬ Nat.Prime 5412603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1804201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1163 2) := by change Nat.Prime 5410277; exact ProjectEulerSolutions.P58PrimeNodes.p_5410277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1163 3) := by change ¬ Nat.Prime 5407951; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 284629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1163 : row 1163 = 1 := (row_1163.checked).trans rfl
theorem model_count_1163 : modelRow 1163 = 1 := by decide
theorem modelrow_agrees_1163 : modelRow 1163 = row 1163 := model_count_1163.trans row_count_1163.symm
def row_1164 : RowCertificate 1164 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1164 1) := by change ¬ Nat.Prime 5421913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 774559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1164 2) := by change ¬ Nat.Prime 5419585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1083917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1164 3) := by change ¬ Nat.Prime 5417257; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 74209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1164 : row 1164 = 0 := (row_1164.checked).trans rfl
theorem model_count_1164 : modelRow 1164 = 0 := by decide
theorem modelrow_agrees_1164 : modelRow 1164 = row 1164 := model_count_1164.trans row_count_1164.symm
def row_1165 : RowCertificate 1165 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1165 1) := by change ¬ Nat.Prime 5431231; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 417787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1165 2) := by change Nat.Prime 5428901; exact ProjectEulerSolutions.P58PrimeNodes.p_5428901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1165 3) := by change ¬ Nat.Prime 5426571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1808857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1165 : row 1165 = 1 := (row_1165.checked).trans rfl
theorem model_count_1165 : modelRow 1165 = 1 := by decide
theorem modelrow_agrees_1165 : modelRow 1165 = row 1165 := model_count_1165.trans row_count_1165.symm
def row_1166 : RowCertificate 1166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1166 1) := by change ¬ Nat.Prime 5440557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1813519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1166 2) := by change ¬ Nat.Prime 5438225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1087645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1166 3) := by change ¬ Nat.Prime 5435893; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 89113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1166 : row 1166 = 0 := (row_1166.checked).trans rfl
theorem model_count_1166 : modelRow 1166 = 0 := by decide
theorem modelrow_agrees_1166 : modelRow 1166 = row 1166 := model_count_1166.trans row_count_1166.symm
def row_1167 : RowCertificate 1167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1167 1) := by change ¬ Nat.Prime 5449891; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 49999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1167 2) := by change ¬ Nat.Prime 5447557; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 30097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1167 3) := by change ¬ Nat.Prime 5445223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 777889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1167 : row 1167 = 0 := (row_1167.checked).trans rfl
theorem model_count_1167 : modelRow 1167 = 0 := by decide
theorem modelrow_agrees_1167 : modelRow 1167 = row 1167 := model_count_1167.trans row_count_1167.symm
def row_1168 : RowCertificate 1168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1168 1) := by change ¬ Nat.Prime 5459233; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 419941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1168 2) := by change Nat.Prime 5456897; exact ProjectEulerSolutions.P58PrimeNodes.p_5456897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1168 3) := by change ¬ Nat.Prime 5454561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1818187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1168 : row 1168 = 1 := (row_1168.checked).trans rfl
theorem model_count_1168 : modelRow 1168 = 1 := by decide
theorem modelrow_agrees_1168 : modelRow 1168 = row 1168 := model_count_1168.trans row_count_1168.symm
def row_1169 : RowCertificate 1169 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1169 1) := by change ¬ Nat.Prime 5468583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1822861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1169 2) := by change ¬ Nat.Prime 5466245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1093249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1169 3) := by change ¬ Nat.Prime 5463907; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 4999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1169 : row 1169 = 0 := (row_1169.checked).trans rfl
theorem model_count_1169 : modelRow 1169 = 0 := by decide
theorem modelrow_agrees_1169 : modelRow 1169 = row 1169 := model_count_1169.trans row_count_1169.symm
def row_1170 : RowCertificate 1170 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1170 1) := by change ¬ Nat.Prime 5477941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 782563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1170 2) := by change ¬ Nat.Prime 5475601; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 36749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1170 3) := by change Nat.Prime 5473261; exact ProjectEulerSolutions.P58PrimeNodes.p_5473261.prime
    exact decide_eq_true prime
theorem row_count_1170 : row 1170 = 1 := (row_1170.checked).trans rfl
theorem model_count_1170 : modelRow 1170 = 1 := by decide
theorem modelrow_agrees_1170 : modelRow 1170 = row 1170 := model_count_1170.trans row_count_1170.symm
def row_1171 : RowCertificate 1171 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1171 1) := by change ¬ Nat.Prime 5487307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 783901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1171 2) := by change ¬ Nat.Prime 5484965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1096993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1171 3) := by change ¬ Nat.Prime 5482623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1827541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1171 : row 1171 = 0 := (row_1171.checked).trans rfl
theorem model_count_1171 : modelRow 1171 = 0 := by decide
theorem modelrow_agrees_1171 : modelRow 1171 = row 1171 := model_count_1171.trans row_count_1171.symm
def row_1172 : RowCertificate 1172 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1172 1) := by change ¬ Nat.Prime 5496681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1832227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1172 2) := by change ¬ Nat.Prime 5494337; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 12689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1172 3) := by change ¬ Nat.Prime 5491993; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 422461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1172 : row 1172 = 0 := (row_1172.checked).trans rfl
theorem model_count_1172 : modelRow 1172 = 0 := by decide
theorem modelrow_agrees_1172 : modelRow 1172 = row 1172 := model_count_1172.trans row_count_1172.symm
def row_1173 : RowCertificate 1173 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1173 1) := by change ¬ Nat.Prime 5506063; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 69697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1173 2) := by change ¬ Nat.Prime 5503717; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 134237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1173 3) := by change Nat.Prime 5501371; exact ProjectEulerSolutions.P58PrimeNodes.p_5501371.prime
    exact decide_eq_true prime
theorem row_count_1173 : row 1173 = 1 := (row_1173.checked).trans rfl
theorem model_count_1173 : modelRow 1173 = 1 := by decide
theorem modelrow_agrees_1173 : modelRow 1173 = row 1173 := model_count_1173.trans row_count_1173.symm
def row_1174 : RowCertificate 1174 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1174 1) := by change ¬ Nat.Prime 5515453; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 290287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1174 2) := by change ¬ Nat.Prime 5513105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1102621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1174 3) := by change ¬ Nat.Prime 5510757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1836919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1174 : row 1174 = 0 := (row_1174.checked).trans rfl
theorem model_count_1174 : modelRow 1174 = 0 := by decide
theorem modelrow_agrees_1174 : modelRow 1174 = row 1174 := model_count_1174.trans row_count_1174.symm
def row_1175 : RowCertificate 1175 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1175 1) := by change ¬ Nat.Prime 5524851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1841617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1175 2) := by change ¬ Nat.Prime 5522501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 324853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1175 3) := by change ¬ Nat.Prime 5520151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 788593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1175 : row 1175 = 0 := (row_1175.checked).trans rfl
theorem model_count_1175 : modelRow 1175 = 0 := by decide
theorem modelrow_agrees_1175 : modelRow 1175 = row 1175 := model_count_1175.trans row_count_1175.symm
def row_1176 : RowCertificate 1176 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1176 1) := by change ¬ Nat.Prime 5534257; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 50773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1176 2) := by change ¬ Nat.Prime 5531905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1106381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1176 3) := by change ¬ Nat.Prime 5529553; exact PrattCertificates.composite_of_product (factor := 2269) (cofactor := 2437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1176 : row 1176 = 0 := (row_1176.checked).trans rfl
theorem model_count_1176 : modelRow 1176 = 0 := by decide
theorem modelrow_agrees_1176 : modelRow 1176 = row 1176 := model_count_1176.trans row_count_1176.symm
def row_1177 : RowCertificate 1177 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1177 1) := by change ¬ Nat.Prime 5543671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 791953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1177 2) := by change Nat.Prime 5541317; exact ProjectEulerSolutions.P58PrimeNodes.p_5541317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1177 3) := by change ¬ Nat.Prime 5538963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1846321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1177 : row 1177 = 1 := (row_1177.checked).trans rfl
theorem model_count_1177 : modelRow 1177 = 1 := by decide
theorem modelrow_agrees_1177 : modelRow 1177 = row 1177 := model_count_1177.trans row_count_1177.symm
def row_1178 : RowCertificate 1178 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1178 1) := by change ¬ Nat.Prime 5553093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1851031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1178 2) := by change ¬ Nat.Prime 5550737; exact PrattCertificates.composite_of_product (factor := 1217) (cofactor := 4561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1178 3) := by change ¬ Nat.Prime 5548381; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 11119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1178 : row 1178 = 0 := (row_1178.checked).trans rfl
theorem model_count_1178 : modelRow 1178 = 0 := by decide
theorem modelrow_agrees_1178 : modelRow 1178 = row 1178 := model_count_1178.trans row_count_1178.symm
def row_1179 : RowCertificate 1179 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1179 1) := by change ¬ Nat.Prime 5562523; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 129361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1179 2) := by change ¬ Nat.Prime 5560165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1112033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1179 3) := by change ¬ Nat.Prime 5557807; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 150211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1179 : row 1179 = 0 := (row_1179.checked).trans rfl
theorem model_count_1179 : modelRow 1179 = 0 := by decide
theorem modelrow_agrees_1179 : modelRow 1179 = row 1179 := model_count_1179.trans row_count_1179.symm
def row_1180 : RowCertificate 1180 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1180 1) := by change Nat.Prime 5571961; exact ProjectEulerSolutions.P58PrimeNodes.p_5571961.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1180 2) := by change Nat.Prime 5569601; exact ProjectEulerSolutions.P58PrimeNodes.p_5569601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1180 3) := by change ¬ Nat.Prime 5567241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1855747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1180 : row 1180 = 2 := (row_1180.checked).trans rfl
theorem model_count_1180 : modelRow 1180 = 2 := by decide
theorem modelrow_agrees_1180 : modelRow 1180 = row 1180 := model_count_1180.trans row_count_1180.symm
def row_1181 : RowCertificate 1181 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1181 1) := by change ¬ Nat.Prime 5581407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1860469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1181 2) := by change ¬ Nat.Prime 5579045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1115809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1181 3) := by change ¬ Nat.Prime 5576683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 796669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1181 : row 1181 = 0 := (row_1181.checked).trans rfl
theorem model_count_1181 : modelRow 1181 = 0 := by decide
theorem modelrow_agrees_1181 : modelRow 1181 = row 1181 := model_count_1181.trans row_count_1181.symm
def row_1182 : RowCertificate 1182 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1182 1) := by change Nat.Prime 5590861; exact ProjectEulerSolutions.P58PrimeNodes.p_5590861.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1182 2) := by change ¬ Nat.Prime 5588497; exact PrattCertificates.composite_of_product (factor := 857) (cofactor := 6521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1182 3) := by change ¬ Nat.Prime 5586133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 798019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1182 : row 1182 = 1 := (row_1182.checked).trans rfl
theorem model_count_1182 : modelRow 1182 = 1 := by decide
theorem modelrow_agrees_1182 : modelRow 1182 = row 1182 := model_count_1182.trans row_count_1182.symm
def row_1183 : RowCertificate 1183 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1183 1) := by change ¬ Nat.Prime 5600323; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 12757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1183 2) := by change ¬ Nat.Prime 5597957; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 193033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1183 3) := by change ¬ Nat.Prime 5595591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1865197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1183 : row 1183 = 0 := (row_1183.checked).trans rfl
theorem model_count_1183 : modelRow 1183 = 0 := by decide
theorem modelrow_agrees_1183 : modelRow 1183 = row 1183 := model_count_1183.trans row_count_1183.symm
def row_1184 : RowCertificate 1184 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1184 1) := by change ¬ Nat.Prime 5609793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1869931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1184 2) := by change ¬ Nat.Prime 5607425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1121485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1184 3) := by change ¬ Nat.Prime 5605057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 295003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1184 : row 1184 = 0 := (row_1184.checked).trans rfl
theorem model_count_1184 : modelRow 1184 = 0 := by decide
theorem modelrow_agrees_1184 : modelRow 1184 = row 1184 := model_count_1184.trans row_count_1184.symm
def row_1185 : RowCertificate 1185 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1185 1) := by change ¬ Nat.Prime 5619271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 802753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1185 2) := by change ¬ Nat.Prime 5616901; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 10781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1185 3) := by change ¬ Nat.Prime 5614531; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 431887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1185 : row 1185 = 0 := (row_1185.checked).trans rfl
theorem model_count_1185 : modelRow 1185 = 0 := by decide
theorem modelrow_agrees_1185 : modelRow 1185 = row 1185 := model_count_1185.trans row_count_1185.symm
def row_1186 : RowCertificate 1186 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1186 1) := by change Nat.Prime 5628757; exact ProjectEulerSolutions.P58PrimeNodes.p_5628757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1186 2) := by change ¬ Nat.Prime 5626385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1125277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1186 3) := by change ¬ Nat.Prime 5624013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1874671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1186 : row 1186 = 1 := (row_1186.checked).trans rfl
theorem model_count_1186 : modelRow 1186 = 1 := by decide
theorem modelrow_agrees_1186 : modelRow 1186 = row 1186 := model_count_1186.trans row_count_1186.symm
def row_1187 : RowCertificate 1187 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1187 1) := by change ¬ Nat.Prime 5638251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1879417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1187 2) := by change ¬ Nat.Prime 5635877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 433529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1187 3) := by change Nat.Prime 5633503; exact ProjectEulerSolutions.P58PrimeNodes.p_5633503.prime
    exact decide_eq_true prime
theorem row_count_1187 : row 1187 = 1 := (row_1187.checked).trans rfl
theorem model_count_1187 : modelRow 1187 = 1 := by decide
theorem modelrow_agrees_1187 : modelRow 1187 = row 1187 := model_count_1187.trans row_count_1187.symm
def row_1188 : RowCertificate 1188 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1188 1) := by change ¬ Nat.Prime 5647753; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 20389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1188 2) := by change ¬ Nat.Prime 5645377; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 332081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1188 3) := by change ¬ Nat.Prime 5643001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 806143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1188 : row 1188 = 0 := (row_1188.checked).trans rfl
theorem model_count_1188 : modelRow 1188 = 0 := by decide
theorem modelrow_agrees_1188 : modelRow 1188 = row 1188 := model_count_1188.trans row_count_1188.symm
def row_1189 : RowCertificate 1189 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1189 1) := by change ¬ Nat.Prime 5657263; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 152899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1189 2) := by change ¬ Nat.Prime 5654885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1130977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1189 3) := by change ¬ Nat.Prime 5652507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1884169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1189 : row 1189 = 0 := (row_1189.checked).trans rfl
theorem model_count_1189 : modelRow 1189 = 0 := by decide
theorem modelrow_agrees_1189 : modelRow 1189 = row 1189 := model_count_1189.trans row_count_1189.symm
def row_1190 : RowCertificate 1190 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1190 1) := by change ¬ Nat.Prime 5666781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1888927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1190 2) := by change Nat.Prime 5664401; exact ProjectEulerSolutions.P58PrimeNodes.p_5664401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1190 3) := by change ¬ Nat.Prime 5662021; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 9421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1190 : row 1190 = 1 := (row_1190.checked).trans rfl
theorem model_count_1190 : modelRow 1190 = 1 := by decide
theorem modelrow_agrees_1190 : modelRow 1190 = row 1190 := model_count_1190.trans row_count_1190.symm
def row_1191 : RowCertificate 1191 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1191 1) := by change ¬ Nat.Prime 5676307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 810901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1191 2) := by change ¬ Nat.Prime 5673925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1134785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1191 3) := by change ¬ Nat.Prime 5671543; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 182953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1191 : row 1191 = 0 := (row_1191.checked).trans rfl
theorem model_count_1191 : modelRow 1191 = 0 := by decide
theorem modelrow_agrees_1191 : modelRow 1191 = row 1191 := model_count_1191.trans row_count_1191.symm
def row_1192 : RowCertificate 1192 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1192 1) := by change ¬ Nat.Prime 5685841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 812263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1192 2) := by change ¬ Nat.Prime 5683457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 437189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1192 3) := by change ¬ Nat.Prime 5681073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1893691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1192 : row 1192 = 0 := (row_1192.checked).trans rfl
theorem model_count_1192 : modelRow 1192 = 0 := by decide
theorem modelrow_agrees_1192 : modelRow 1192 = row 1192 := model_count_1192.trans row_count_1192.symm
def row_1193 : RowCertificate 1193 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1193 1) := by change ¬ Nat.Prime 5695383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1898461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1193 2) := by change ¬ Nat.Prime 5692997; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 14197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1193 3) := by change Nat.Prime 5690611; exact ProjectEulerSolutions.P58PrimeNodes.p_5690611.prime
    exact decide_eq_true prime
theorem row_count_1193 : row 1193 = 1 := (row_1193.checked).trans rfl
theorem model_count_1193 : modelRow 1193 = 1 := by decide
theorem modelrow_agrees_1193 : modelRow 1193 = row 1193 := model_count_1193.trans row_count_1193.symm
def row_1194 : RowCertificate 1194 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1194 1) := by change ¬ Nat.Prime 5704933; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 438841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1194 2) := by change ¬ Nat.Prime 5702545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1140509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1194 3) := by change Nat.Prime 5700157; exact ProjectEulerSolutions.P58PrimeNodes.p_5700157.prime
    exact decide_eq_true prime
theorem row_count_1194 : row 1194 = 1 := (row_1194.checked).trans rfl
theorem model_count_1194 : modelRow 1194 = 1 := by decide
theorem modelrow_agrees_1194 : modelRow 1194 = row 1194 := model_count_1194.trans row_count_1194.symm
def row_1195 : RowCertificate 1195 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1195 1) := by change Nat.Prime 5714491; exact ProjectEulerSolutions.P58PrimeNodes.p_5714491.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1195 2) := by change ¬ Nat.Prime 5712101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 196969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1195 3) := by change ¬ Nat.Prime 5709711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1903237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1195 : row 1195 = 1 := (row_1195.checked).trans rfl
theorem model_count_1195 : modelRow 1195 = 1 := by decide
theorem modelrow_agrees_1195 : modelRow 1195 = row 1195 := model_count_1195.trans row_count_1195.symm
def row_1196 : RowCertificate 1196 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1196 1) := by change ¬ Nat.Prime 5724057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1908019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1196 2) := by change ¬ Nat.Prime 5721665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1144333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1196 3) := by change ¬ Nat.Prime 5719273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 817039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1196 : row 1196 = 0 := (row_1196.checked).trans rfl
theorem model_count_1196 : modelRow 1196 = 0 := by decide
theorem modelrow_agrees_1196 : modelRow 1196 = row 1196 := model_count_1196.trans row_count_1196.symm
def row_1197 : RowCertificate 1197 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1197 1) := by change ¬ Nat.Prime 5733631; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 154963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1197 2) := by change ¬ Nat.Prime 5731237; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 12541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1197 3) := by change ¬ Nat.Prime 5728843; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 72517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1197 : row 1197 = 0 := (row_1197.checked).trans rfl
theorem model_count_1197 : modelRow 1197 = 0 := by decide
theorem modelrow_agrees_1197 : modelRow 1197 = row 1197 := model_count_1197.trans row_count_1197.symm
def row_1198 : RowCertificate 1198 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1198 1) := by change ¬ Nat.Prime 5743213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 820459) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1198 2) := by change ¬ Nat.Prime 5740817; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 4157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1198 3) := by change ¬ Nat.Prime 5738421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1912807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1198 : row 1198 = 0 := (row_1198.checked).trans rfl
theorem model_count_1198 : modelRow 1198 = 0 := by decide
theorem modelrow_agrees_1198 : modelRow 1198 = row 1198 := model_count_1198.trans row_count_1198.symm
def row_1199 : RowCertificate 1199 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1199 1) := by change ¬ Nat.Prime 5752803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1917601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1199 2) := by change ¬ Nat.Prime 5750405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1150081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1199 3) := by change Nat.Prime 5748007; exact ProjectEulerSolutions.P58PrimeNodes.p_5748007.prime
    exact decide_eq_true prime
theorem row_count_1199 : row 1199 = 1 := (row_1199.checked).trans rfl
theorem model_count_1199 : modelRow 1199 = 1 := by decide
theorem modelrow_agrees_1199 : modelRow 1199 = row 1199 := model_count_1199.trans row_count_1199.symm
def row_1200 : RowCertificate 1200 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1200 1) := by change ¬ Nat.Prime 5762401; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 78937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1200 2) := by change ¬ Nat.Prime 5760001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 443077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1200 3) := by change ¬ Nat.Prime 5757601; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 13297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1200 : row 1200 = 0 := (row_1200.checked).trans rfl
theorem model_count_1200 : modelRow 1200 = 0 := by decide
theorem modelrow_agrees_1200 : modelRow 1200 = row 1200 := model_count_1200.trans row_count_1200.symm
def row_1201 : RowCertificate 1201 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1201 1) := by change Nat.Prime 5772007; exact ProjectEulerSolutions.P58PrimeNodes.p_5772007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1201 2) := by change ¬ Nat.Prime 5769605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1153921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1201 3) := by change ¬ Nat.Prime 5767203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1922401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1201 : row 1201 = 1 := (row_1201.checked).trans rfl
theorem model_count_1201 : modelRow 1201 = 1 := by decide
theorem modelrow_agrees_1201 : modelRow 1201 = row 1201 := model_count_1201.trans row_count_1201.symm
def row_1202 : RowCertificate 1202 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1202 1) := by change ¬ Nat.Prime 5781621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1927207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1202 2) := by change Nat.Prime 5779217; exact ProjectEulerSolutions.P58PrimeNodes.p_5779217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1202 3) := by change ¬ Nat.Prime 5776813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 825259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1202 : row 1202 = 1 := (row_1202.checked).trans rfl
theorem model_count_1202 : modelRow 1202 = 1 := by decide
theorem modelrow_agrees_1202 : modelRow 1202 = row 1202 := model_count_1202.trans row_count_1202.symm
def row_1203 : RowCertificate 1203 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1203 1) := by change ¬ Nat.Prime 5791243; exact PrattCertificates.composite_of_product (factor := 1777) (cofactor := 3259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1203 2) := by change Nat.Prime 5788837; exact ProjectEulerSolutions.P58PrimeNodes.p_5788837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1203 3) := by change ¬ Nat.Prime 5786431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 826633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1203 : row 1203 = 1 := (row_1203.checked).trans rfl
theorem model_count_1203 : modelRow 1203 = 1 := by decide
theorem modelrow_agrees_1203 : modelRow 1203 = row 1203 := model_count_1203.trans row_count_1203.symm
def row_1204 : RowCertificate 1204 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1204 1) := by change ¬ Nat.Prime 5800873; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 446221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1204 2) := by change ¬ Nat.Prime 5798465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1159693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1204 3) := by change ¬ Nat.Prime 5796057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1932019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1204 : row 1204 = 0 := (row_1204.checked).trans rfl
theorem model_count_1204 : modelRow 1204 = 0 := by decide
theorem modelrow_agrees_1204 : modelRow 1204 = row 1204 := model_count_1204.trans row_count_1204.symm
def row_1205 : RowCertificate 1205 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1205 1) := by change ¬ Nat.Prime 5810511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1936837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1205 2) := by change ¬ Nat.Prime 5808101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 446777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1205 3) := by change Nat.Prime 5805691; exact ProjectEulerSolutions.P58PrimeNodes.p_5805691.prime
    exact decide_eq_true prime
theorem row_count_1205 : row 1205 = 1 := (row_1205.checked).trans rfl
theorem model_count_1205 : modelRow 1205 = 1 := by decide
theorem modelrow_agrees_1205 : modelRow 1205 = row 1205 := model_count_1205.trans row_count_1205.symm
def row_1206 : RowCertificate 1206 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1206 1) := by change ¬ Nat.Prime 5820157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 831451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1206 2) := by change ¬ Nat.Prime 5817745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1163549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1206 3) := by change Nat.Prime 5815333; exact ProjectEulerSolutions.P58PrimeNodes.p_5815333.prime
    exact decide_eq_true prime
theorem row_count_1206 : row 1206 = 1 := (row_1206.checked).trans rfl
theorem model_count_1206 : modelRow 1206 = 1 := by decide
theorem modelrow_agrees_1206 : modelRow 1206 = row 1206 := model_count_1206.trans row_count_1206.symm
def row_1207 : RowCertificate 1207 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1207 1) := by change ¬ Nat.Prime 5829811; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 448447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1207 2) := by change ¬ Nat.Prime 5827397; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 57697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1207 3) := by change ¬ Nat.Prime 5824983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1941661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1207 : row 1207 = 0 := (row_1207.checked).trans rfl
theorem model_count_1207 : modelRow 1207 = 0 := by decide
theorem modelrow_agrees_1207 : modelRow 1207 = row 1207 := model_count_1207.trans row_count_1207.symm
def row_1208 : RowCertificate 1208 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1208 1) := by change ¬ Nat.Prime 5839473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1946491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1208 2) := by change ¬ Nat.Prime 5837057; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 5717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1208 3) := by change ¬ Nat.Prime 5834641; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 157693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1208 : row 1208 = 0 := (row_1208.checked).trans rfl
theorem model_count_1208 : modelRow 1208 = 0 := by decide
theorem modelrow_agrees_1208 : modelRow 1208 = row 1208 := model_count_1208.trans row_count_1208.symm
def row_1209 : RowCertificate 1209 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1209 1) := by change ¬ Nat.Prime 5849143; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 12799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1209 2) := by change ¬ Nat.Prime 5846725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1169345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1209 3) := by change ¬ Nat.Prime 5844307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 834901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1209 : row 1209 = 0 := (row_1209.checked).trans rfl
theorem model_count_1209 : modelRow 1209 = 0 := by decide
theorem modelrow_agrees_1209 : modelRow 1209 = row 1209 := model_count_1209.trans row_count_1209.symm
def row_1210 : RowCertificate 1210 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1210 1) := by change ¬ Nat.Prime 5858821; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 308359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1210 2) := by change Nat.Prime 5856401; exact ProjectEulerSolutions.P58PrimeNodes.p_5856401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1210 3) := by change ¬ Nat.Prime 5853981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1951327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1210 : row 1210 = 1 := (row_1210.checked).trans rfl
theorem model_count_1210 : modelRow 1210 = 1 := by decide
theorem modelrow_agrees_1210 : modelRow 1210 = row 1210 := model_count_1210.trans row_count_1210.symm
def row_1211 : RowCertificate 1211 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1211 1) := by change ¬ Nat.Prime 5868507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1956169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1211 2) := by change ¬ Nat.Prime 5866085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1173217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1211 3) := by change ¬ Nat.Prime 5863663; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 451051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1211 : row 1211 = 0 := (row_1211.checked).trans rfl
theorem model_count_1211 : modelRow 1211 = 0 := by decide
theorem modelrow_agrees_1211 : modelRow 1211 = row 1211 := model_count_1211.trans row_count_1211.symm
def row_1212 : RowCertificate 1212 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1212 1) := by change ¬ Nat.Prime 5878201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 839743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1212 2) := by change ¬ Nat.Prime 5875777; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 202613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1212 3) := by change ¬ Nat.Prime 5873353; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 189463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1212 : row 1212 = 0 := (row_1212.checked).trans rfl
theorem model_count_1212 : modelRow 1212 = 0 := by decide
theorem modelrow_agrees_1212 : modelRow 1212 = row 1212 := model_count_1212.trans row_count_1212.symm
def row_1213 : RowCertificate 1213 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1213 1) := by change ¬ Nat.Prime 5887903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 841129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1213 2) := by change ¬ Nat.Prime 5885477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 452729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1213 3) := by change ¬ Nat.Prime 5883051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1961017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1213 : row 1213 = 0 := (row_1213.checked).trans rfl
theorem model_count_1213 : modelRow 1213 = 0 := by decide
theorem modelrow_agrees_1213 : modelRow 1213 = row 1213 := model_count_1213.trans row_count_1213.symm
def row_1214 : RowCertificate 1214 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1214 1) := by change ¬ Nat.Prime 5897613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1965871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1214 2) := by change ¬ Nat.Prime 5895185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1179037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1214 3) := by change ¬ Nat.Prime 5892757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 453289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1214 : row 1214 = 0 := (row_1214.checked).trans rfl
theorem model_count_1214 : modelRow 1214 = 0 := by decide
theorem modelrow_agrees_1214 : modelRow 1214 = row 1214 := model_count_1214.trans row_count_1214.symm
def row_1215 : RowCertificate 1215 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1215 1) := by change Nat.Prime 5907331; exact ProjectEulerSolutions.P58PrimeNodes.p_5907331.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1215 2) := by change Nat.Prime 5904901; exact ProjectEulerSolutions.P58PrimeNodes.p_5904901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1215 3) := by change Nat.Prime 5902471; exact ProjectEulerSolutions.P58PrimeNodes.p_5902471.prime
    exact decide_eq_true prime
theorem row_count_1215 : row 1215 = 3 := (row_1215.checked).trans rfl
theorem model_count_1215 : modelRow 1215 = 3 := by decide
theorem modelrow_agrees_1215 : modelRow 1215 = row 1215 := model_count_1215.trans row_count_1215.symm
def row_1216 : RowCertificate 1216 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1216 1) := by change ¬ Nat.Prime 5917057; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 46591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1216 2) := by change ¬ Nat.Prime 5914625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1182925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1216 3) := by change ¬ Nat.Prime 5912193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1970731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1216 : row 1216 = 0 := (row_1216.checked).trans rfl
theorem model_count_1216 : modelRow 1216 = 0 := by decide
theorem modelrow_agrees_1216 : modelRow 1216 = row 1216 := model_count_1216.trans row_count_1216.symm
def row_1217 : RowCertificate 1217 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1217 1) := by change ¬ Nat.Prime 5926791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1975597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1217 2) := by change ¬ Nat.Prime 5924357; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 58657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1217 3) := by change ¬ Nat.Prime 5921923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 845989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1217 : row 1217 = 0 := (row_1217.checked).trans rfl
theorem model_count_1217 : modelRow 1217 = 0 := by decide
theorem modelrow_agrees_1217 : modelRow 1217 = row 1217 := model_count_1217.trans row_count_1217.symm
def row_1218 : RowCertificate 1218 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1218 1) := by change ¬ Nat.Prime 5936533; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 8821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1218 2) := by change ¬ Nat.Prime 5934097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 456469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1218 3) := by change Nat.Prime 5931661; exact ProjectEulerSolutions.P58PrimeNodes.p_5931661.prime
    exact decide_eq_true prime
theorem row_count_1218 : row 1218 = 1 := (row_1218.checked).trans rfl
theorem model_count_1218 : modelRow 1218 = 1 := by decide
theorem modelrow_agrees_1218 : modelRow 1218 = row 1218 := model_count_1218.trans row_count_1218.symm
def row_1219 : RowCertificate 1219 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1219 1) := by change ¬ Nat.Prime 5946283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 849469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1219 2) := by change ¬ Nat.Prime 5943845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1188769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1219 3) := by change ¬ Nat.Prime 5941407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1980469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1219 : row 1219 = 0 := (row_1219.checked).trans rfl
theorem model_count_1219 : modelRow 1219 = 0 := by decide
theorem modelrow_agrees_1219 : modelRow 1219 = row 1219 := model_count_1219.trans row_count_1219.symm
def row_1220 : RowCertificate 1220 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1220 1) := by change ¬ Nat.Prime 5956041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1985347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1220 2) := by change ¬ Nat.Prime 5953601; exact PrattCertificates.composite_of_product (factor := 1657) (cofactor := 3593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1220 3) := by change ¬ Nat.Prime 5951161; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 313219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1220 : row 1220 = 0 := (row_1220.checked).trans rfl
theorem model_count_1220 : modelRow 1220 = 0 := by decide
theorem modelrow_agrees_1220 : modelRow 1220 = row 1220 := model_count_1220.trans row_count_1220.symm
def row_1221 : RowCertificate 1221 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1221 1) := by change Nat.Prime 5965807; exact ProjectEulerSolutions.P58PrimeNodes.p_5965807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1221 2) := by change ¬ Nat.Prime 5963365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1192673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1221 3) := by change ¬ Nat.Prime 5960923; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 88969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1221 : row 1221 = 1 := (row_1221.checked).trans rfl
theorem model_count_1221 : modelRow 1221 = 1 := by decide
theorem modelrow_agrees_1221 : modelRow 1221 = row 1221 := model_count_1221.trans row_count_1221.symm
def row_1222 : RowCertificate 1222 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1222 1) := by change ¬ Nat.Prime 5975581; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 138967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1222 2) := by change ¬ Nat.Prime 5973137; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 351361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1222 3) := by change ¬ Nat.Prime 5970693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1990231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1222 : row 1222 = 0 := (row_1222.checked).trans rfl
theorem model_count_1222 : modelRow 1222 = 0 := by decide
theorem modelrow_agrees_1222 : modelRow 1222 = row 1222 := model_count_1222.trans row_count_1222.symm
def row_1223 : RowCertificate 1223 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1223 1) := by change ¬ Nat.Prime 5985363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1995121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1223 2) := by change ¬ Nat.Prime 5982917; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 5189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1223 3) := by change ¬ Nat.Prime 5980471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 854353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1223 : row 1223 = 0 := (row_1223.checked).trans rfl
theorem model_count_1223 : modelRow 1223 = 0 := by decide
theorem modelrow_agrees_1223 : modelRow 1223 = row 1223 := model_count_1223.trans row_count_1223.symm
def row_1224 : RowCertificate 1224 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1224 1) := by change ¬ Nat.Prime 5995153; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 39703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1224 2) := by change ¬ Nat.Prime 5992705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1198541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1224 3) := by change ¬ Nat.Prime 5990257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 855751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1224 : row 1224 = 0 := (row_1224.checked).trans rfl
theorem model_count_1224 : modelRow 1224 = 0 := by decide
theorem modelrow_agrees_1224 : modelRow 1224 = row 1224 := model_count_1224.trans row_count_1224.symm
def row_1225 : RowCertificate 1225 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1225 1) := by change Nat.Prime 6004951; exact ProjectEulerSolutions.P58PrimeNodes.p_6004951.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1225 2) := by change ¬ Nat.Prime 6002501; exact PrattCertificates.composite_of_product (factor := 2381) (cofactor := 2521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1225 3) := by change ¬ Nat.Prime 6000051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2000017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1225 : row 1225 = 1 := (row_1225.checked).trans rfl
theorem model_count_1225 : modelRow 1225 = 1 := by decide
theorem modelrow_agrees_1225 : modelRow 1225 = row 1225 := model_count_1225.trans row_count_1225.symm
def row_1226 : RowCertificate 1226 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1226 1) := by change ¬ Nat.Prime 6014757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2004919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1226 2) := by change ¬ Nat.Prime 6012305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1202461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1226 3) := by change Nat.Prime 6009853; exact ProjectEulerSolutions.P58PrimeNodes.p_6009853.prime
    exact decide_eq_true prime
theorem row_count_1226 : row 1226 = 1 := (row_1226.checked).trans rfl
theorem model_count_1226 : modelRow 1226 = 1 := by decide
theorem modelrow_agrees_1226 : modelRow 1226 = row 1226 := model_count_1226.trans row_count_1226.symm
def row_1227 : RowCertificate 1227 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1227 1) := by change ¬ Nat.Prime 6024571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 860653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1227 2) := by change Nat.Prime 6022117; exact ProjectEulerSolutions.P58PrimeNodes.p_6022117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1227 3) := by change ¬ Nat.Prime 6019663; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 463051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1227 : row 1227 = 1 := (row_1227.checked).trans rfl
theorem model_count_1227 : modelRow 1227 = 1 := by decide
theorem modelrow_agrees_1227 : modelRow 1227 = row 1227 := model_count_1227.trans row_count_1227.symm
def row_1228 : RowCertificate 1228 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1228 1) := by change Nat.Prime 6034393; exact ProjectEulerSolutions.P58PrimeNodes.p_6034393.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1228 2) := by change Nat.Prime 6031937; exact ProjectEulerSolutions.P58PrimeNodes.p_6031937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1228 3) := by change ¬ Nat.Prime 6029481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2009827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1228 : row 1228 = 2 := (row_1228.checked).trans rfl
theorem model_count_1228 : modelRow 1228 = 2 := by decide
theorem modelrow_agrees_1228 : modelRow 1228 = row 1228 := model_count_1228.trans row_count_1228.symm
def row_1229 : RowCertificate 1229 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1229 1) := by change ¬ Nat.Prime 6044223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2014741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1229 2) := by change ¬ Nat.Prime 6041765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1208353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1229 3) := by change ¬ Nat.Prime 6039307; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 140449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1229 : row 1229 = 0 := (row_1229.checked).trans rfl
theorem model_count_1229 : modelRow 1229 = 0 := by decide
theorem modelrow_agrees_1229 : modelRow 1229 = row 1229 := model_count_1229.trans row_count_1229.symm
def row_1230 : RowCertificate 1230 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1230 1) := by change ¬ Nat.Prime 6054061; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 465697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1230 2) := by change Nat.Prime 6051601; exact ProjectEulerSolutions.P58PrimeNodes.p_6051601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1230 3) := by change ¬ Nat.Prime 6049141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 864163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1230 : row 1230 = 1 := (row_1230.checked).trans rfl
theorem model_count_1230 : modelRow 1230 = 1 := by decide
theorem modelrow_agrees_1230 : modelRow 1230 = row 1230 := model_count_1230.trans row_count_1230.symm
def row_1231 : RowCertificate 1231 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1231 1) := by change ¬ Nat.Prime 6063907; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 319153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1231 2) := by change ¬ Nat.Prime 6061445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1212289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1231 3) := by change ¬ Nat.Prime 6058983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2019661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1231 : row 1231 = 0 := (row_1231.checked).trans rfl
theorem model_count_1231 : modelRow 1231 = 0 := by decide
theorem modelrow_agrees_1231 : modelRow 1231 = row 1231 := model_count_1231.trans row_count_1231.symm
def row_1232 : RowCertificate 1232 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1232 1) := by change ¬ Nat.Prime 6073761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2024587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1232 2) := by change Nat.Prime 6071297; exact ProjectEulerSolutions.P58PrimeNodes.p_6071297.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1232 3) := by change Nat.Prime 6068833; exact ProjectEulerSolutions.P58PrimeNodes.p_6068833.prime
    exact decide_eq_true prime
theorem row_count_1232 : row 1232 = 2 := (row_1232.checked).trans rfl
theorem model_count_1232 : modelRow 1232 = 2 := by decide
theorem modelrow_agrees_1232 : modelRow 1232 = row 1232 := model_count_1232.trans row_count_1232.symm
def row_1233 : RowCertificate 1233 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1233 1) := by change ¬ Nat.Prime 6083623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 869089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1233 2) := by change ¬ Nat.Prime 6081157; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 3469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1233 3) := by change Nat.Prime 6078691; exact ProjectEulerSolutions.P58PrimeNodes.p_6078691.prime
    exact decide_eq_true prime
theorem row_count_1233 : row 1233 = 1 := (row_1233.checked).trans rfl
theorem model_count_1233 : modelRow 1233 = 1 := by decide
theorem modelrow_agrees_1233 : modelRow 1233 = row 1233 := model_count_1233.trans row_count_1233.symm
def row_1234 : RowCertificate 1234 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1234 1) := by change ¬ Nat.Prime 6093493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 870499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1234 2) := by change ¬ Nat.Prime 6091025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1218205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1234 3) := by change ¬ Nat.Prime 6088557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2029519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1234 : row 1234 = 0 := (row_1234.checked).trans rfl
theorem model_count_1234 : modelRow 1234 = 0 := by decide
theorem modelrow_agrees_1234 : modelRow 1234 = row 1234 := model_count_1234.trans row_count_1234.symm
def row_1235 : RowCertificate 1235 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1235 1) := by change ¬ Nat.Prime 6103371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2034457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1235 2) := by change Nat.Prime 6100901; exact ProjectEulerSolutions.P58PrimeNodes.p_6100901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1235 3) := by change ¬ Nat.Prime 6098431; exact PrattCertificates.composite_of_product (factor := 1063) (cofactor := 5737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1235 : row 1235 = 1 := (row_1235.checked).trans rfl
theorem model_count_1235 : modelRow 1235 = 1 := by decide
theorem modelrow_agrees_1235 : modelRow 1235 = row 1235 := model_count_1235.trans row_count_1235.symm
def row_1236 : RowCertificate 1236 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1236 1) := by change ¬ Nat.Prime 6113257; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 77383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1236 2) := by change ¬ Nat.Prime 6110785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1222157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1236 3) := by change Nat.Prime 6108313; exact ProjectEulerSolutions.P58PrimeNodes.p_6108313.prime
    exact decide_eq_true prime
theorem row_count_1236 : row 1236 = 1 := (row_1236.checked).trans rfl
theorem model_count_1236 : modelRow 1236 = 1 := by decide
theorem modelrow_agrees_1236 : modelRow 1236 = row 1236 := model_count_1236.trans row_count_1236.symm
def row_1237 : RowCertificate 1237 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1237 1) := by change ¬ Nat.Prime 6123151; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 197521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1237 2) := by change ¬ Nat.Prime 6120677; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 56153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1237 3) := by change ¬ Nat.Prime 6118203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2039401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1237 : row 1237 = 0 := (row_1237.checked).trans rfl
theorem model_count_1237 : modelRow 1237 = 0 := by decide
theorem modelrow_agrees_1237 : modelRow 1237 = row 1237 := model_count_1237.trans row_count_1237.symm
def row_1238 : RowCertificate 1238 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1238 1) := by change ¬ Nat.Prime 6133053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2044351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1238 2) := by change ¬ Nat.Prime 6130577; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 21817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1238 3) := by change ¬ Nat.Prime 6128101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 875443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1238 : row 1238 = 0 := (row_1238.checked).trans rfl
theorem model_count_1238 : modelRow 1238 = 0 := by decide
theorem modelrow_agrees_1238 : modelRow 1238 = row 1238 := model_count_1238.trans row_count_1238.symm
def row_1239 : RowCertificate 1239 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1239 1) := by change Nat.Prime 6142963; exact ProjectEulerSolutions.P58PrimeNodes.p_6142963.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1239 2) := by change ¬ Nat.Prime 6140485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1228097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1239 3) := by change ¬ Nat.Prime 6138007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 323053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1239 : row 1239 = 1 := (row_1239.checked).trans rfl
theorem model_count_1239 : modelRow 1239 = 1 := by decide
theorem modelrow_agrees_1239 : modelRow 1239 = row 1239 := model_count_1239.trans row_count_1239.symm
def row_1240 : RowCertificate 1240 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1240 1) := by change ¬ Nat.Prime 6152881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 878983) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1240 2) := by change ¬ Nat.Prime 6150401; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 7013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1240 3) := by change ¬ Nat.Prime 6147921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2049307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1240 : row 1240 = 0 := (row_1240.checked).trans rfl
theorem model_count_1240 : modelRow 1240 = 0 := by decide
theorem modelrow_agrees_1240 : modelRow 1240 = row 1240 := model_count_1240.trans row_count_1240.symm
def row_1241 : RowCertificate 1241 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1241 1) := by change ¬ Nat.Prime 6162807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2054269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1241 2) := by change ¬ Nat.Prime 6160325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1232065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1241 3) := by change ¬ Nat.Prime 6157843; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 324097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1241 : row 1241 = 0 := (row_1241.checked).trans rfl
theorem model_count_1241 : modelRow 1241 = 0 := by decide
theorem modelrow_agrees_1241 : modelRow 1241 = row 1241 := model_count_1241.trans row_count_1241.symm
def row_1242 : RowCertificate 1242 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1242 1) := by change Nat.Prime 6172741; exact ProjectEulerSolutions.P58PrimeNodes.p_6172741.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1242 2) := by change ¬ Nat.Prime 6170257; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 39301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1242 3) := by change Nat.Prime 6167773; exact ProjectEulerSolutions.P58PrimeNodes.p_6167773.prime
    exact decide_eq_true prime
theorem row_count_1242 : row 1242 = 2 := (row_1242.checked).trans rfl
theorem model_count_1242 : modelRow 1242 = 2 := by decide
theorem modelrow_agrees_1242 : modelRow 1242 = row 1242 := model_count_1242.trans row_count_1242.symm
def row_1243 : RowCertificate 1243 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1243 1) := by change ¬ Nat.Prime 6182683; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 475591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1243 2) := by change ¬ Nat.Prime 6180197; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 363541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1243 3) := by change ¬ Nat.Prime 6177711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2059237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1243 : row 1243 = 0 := (row_1243.checked).trans rfl
theorem model_count_1243 : modelRow 1243 = 0 := by decide
theorem modelrow_agrees_1243 : modelRow 1243 = row 1243 := model_count_1243.trans row_count_1243.symm
def row_1244 : RowCertificate 1244 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1244 1) := by change ¬ Nat.Prime 6192633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2064211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1244 2) := by change ¬ Nat.Prime 6190145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1238029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1244 3) := by change ¬ Nat.Prime 6187657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 883951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1244 : row 1244 = 0 := (row_1244.checked).trans rfl
theorem model_count_1244 : modelRow 1244 = 0 := by decide
theorem modelrow_agrees_1244 : modelRow 1244 = row 1244 := model_count_1244.trans row_count_1244.symm
def row_1245 : RowCertificate 1245 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1245 1) := by change ¬ Nat.Prime 6202591; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 84967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1245 2) := by change ¬ Nat.Prime 6200101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 101641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1245 3) := by change ¬ Nat.Prime 6197611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 885373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1245 : row 1245 = 0 := (row_1245.checked).trans rfl
theorem model_count_1245 : modelRow 1245 = 0 := by decide
theorem modelrow_agrees_1245 : modelRow 1245 = row 1245 := model_count_1245.trans row_count_1245.symm
def row_1246 : RowCertificate 1246 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1246 1) := by change ¬ Nat.Prime 6212557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 477889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1246 2) := by change ¬ Nat.Prime 6210065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1242013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1246 3) := by change ¬ Nat.Prime 6207573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2069191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1246 : row 1246 = 0 := (row_1246.checked).trans rfl
theorem model_count_1246 : modelRow 1246 = 0 := by decide
theorem modelrow_agrees_1246 : modelRow 1246 = row 1246 := model_count_1246.trans row_count_1246.symm
def row_1247 : RowCertificate 1247 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1247 1) := by change ¬ Nat.Prime 6222531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2074177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1247 2) := by change ¬ Nat.Prime 6220037; exact PrattCertificates.composite_of_product (factor := 2341) (cofactor := 2657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1247 3) := by change Nat.Prime 6217543; exact ProjectEulerSolutions.P58PrimeNodes.p_6217543.prime
    exact decide_eq_true prime
theorem row_count_1247 : row 1247 = 1 := (row_1247.checked).trans rfl
theorem model_count_1247 : modelRow 1247 = 1 := by decide
theorem modelrow_agrees_1247 : modelRow 1247 = row 1247 := model_count_1247.trans row_count_1247.symm
def row_1248 : RowCertificate 1248 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1248 1) := by change ¬ Nat.Prime 6232513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 890359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1248 2) := by change Nat.Prime 6230017; exact ProjectEulerSolutions.P58PrimeNodes.p_6230017.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1248 3) := by change Nat.Prime 6227521; exact ProjectEulerSolutions.P58PrimeNodes.p_6227521.prime
    exact decide_eq_true prime
theorem row_count_1248 : row 1248 = 2 := (row_1248.checked).trans rfl
theorem model_count_1248 : modelRow 1248 = 2 := by decide
theorem modelrow_agrees_1248 : modelRow 1248 = row 1248 := model_count_1248.trans row_count_1248.symm
def row_1249 : RowCertificate 1249 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1249 1) := by change Nat.Prime 6242503; exact ProjectEulerSolutions.P58PrimeNodes.p_6242503.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1249 2) := by change ¬ Nat.Prime 6240005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1248001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1249 3) := by change ¬ Nat.Prime 6237507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2079169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1249 : row 1249 = 1 := (row_1249.checked).trans rfl
theorem model_count_1249 : modelRow 1249 = 1 := by decide
theorem modelrow_agrees_1249 : modelRow 1249 = row 1249 := model_count_1249.trans row_count_1249.symm
def row_1250 : RowCertificate 1250 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1250 1) := by change ¬ Nat.Prime 6252501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2084167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1250 2) := by change ¬ Nat.Prime 6250001; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 64433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1250 3) := by change ¬ Nat.Prime 6247501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 480577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1250 : row 1250 = 0 := (row_1250.checked).trans rfl
theorem model_count_1250 : modelRow 1250 = 0 := by decide
theorem modelrow_agrees_1250 : modelRow 1250 = row 1250 := model_count_1250.trans row_count_1250.symm
def row_1251 : RowCertificate 1251 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1251 1) := by change ¬ Nat.Prime 6262507; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 22129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1251 2) := by change ¬ Nat.Prime 6260005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1252001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1251 3) := by change ¬ Nat.Prime 6257503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 893929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1251 : row 1251 = 0 := (row_1251.checked).trans rfl
theorem model_count_1251 : modelRow 1251 = 0 := by decide
theorem modelrow_agrees_1251 : modelRow 1251 = row 1251 := model_count_1251.trans row_count_1251.symm
def row_1252 : RowCertificate 1252 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1252 1) := by change ¬ Nat.Prime 6272521; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 79399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1252 2) := by change ¬ Nat.Prime 6270017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 482309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1252 3) := by change ¬ Nat.Prime 6267513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2089171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1252 : row 1252 = 0 := (row_1252.checked).trans rfl
theorem model_count_1252 : modelRow 1252 = 0 := by decide
theorem modelrow_agrees_1252 : modelRow 1252 = row 1252 := model_count_1252.trans row_count_1252.symm
def row_1253 : RowCertificate 1253 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1253 1) := by change ¬ Nat.Prime 6282543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2094181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1253 2) := by change ¬ Nat.Prime 6280037; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 216553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1253 3) := by change ¬ Nat.Prime 6277531; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 482887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1253 : row 1253 = 0 := (row_1253.checked).trans rfl
theorem model_count_1253 : modelRow 1253 = 0 := by decide
theorem modelrow_agrees_1253 : modelRow 1253 = row 1253 := model_count_1253.trans row_count_1253.symm
def row_1254 : RowCertificate 1254 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1254 1) := by change ¬ Nat.Prime 6292573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 898939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1254 2) := by change ¬ Nat.Prime 6290065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1258013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1254 3) := by change ¬ Nat.Prime 6287557; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 15373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1254 : row 1254 = 0 := (row_1254.checked).trans rfl
theorem model_count_1254 : modelRow 1254 = 0 := by decide
theorem modelrow_agrees_1254 : modelRow 1254 = row 1254 := model_count_1254.trans row_count_1254.symm
def row_1255 : RowCertificate 1255 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1255 1) := by change ¬ Nat.Prime 6302611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 900373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1255 2) := by change ¬ Nat.Prime 6300101; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 170273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1255 3) := by change ¬ Nat.Prime 6297591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2099197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1255 : row 1255 = 0 := (row_1255.checked).trans rfl
theorem model_count_1255 : modelRow 1255 = 0 := by decide
theorem modelrow_agrees_1255 : modelRow 1255 = row 1255 := model_count_1255.trans row_count_1255.symm
def row_1256 : RowCertificate 1256 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1256 1) := by change ¬ Nat.Prime 6312657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2104219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1256 2) := by change ¬ Nat.Prime 6310145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1262029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1256 3) := by change Nat.Prime 6307633; exact ProjectEulerSolutions.P58PrimeNodes.p_6307633.prime
    exact decide_eq_true prime
theorem row_count_1256 : row 1256 = 1 := (row_1256.checked).trans rfl
theorem model_count_1256 : modelRow 1256 = 1 := by decide
theorem modelrow_agrees_1256 : modelRow 1256 = row 1256 := model_count_1256.trans row_count_1256.symm
def row_1257 : RowCertificate 1257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1257 1) := by change ¬ Nat.Prime 6322711; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 103651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1257 2) := by change ¬ Nat.Prime 6320197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 486169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1257 3) := by change Nat.Prime 6317683; exact ProjectEulerSolutions.P58PrimeNodes.p_6317683.prime
    exact decide_eq_true prime
theorem row_count_1257 : row 1257 = 1 := (row_1257.checked).trans rfl
theorem model_count_1257 : modelRow 1257 = 1 := by decide
theorem modelrow_agrees_1257 : modelRow 1257 = row 1257 := model_count_1257.trans row_count_1257.symm
def row_1258 : RowCertificate 1258 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1258 1) := by change ¬ Nat.Prime 6332773; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 204283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1258 2) := by change Nat.Prime 6330257; exact ProjectEulerSolutions.P58PrimeNodes.p_6330257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1258 3) := by change ¬ Nat.Prime 6327741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2109247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1258 : row 1258 = 1 := (row_1258.checked).trans rfl
theorem model_count_1258 : modelRow 1258 = 1 := by decide
theorem modelrow_agrees_1258 : modelRow 1258 = row 1258 := model_count_1258.trans row_count_1258.symm
def row_1259 : RowCertificate 1259 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1259 1) := by change ¬ Nat.Prime 6342843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2114281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1259 2) := by change ¬ Nat.Prime 6340325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1268065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1259 3) := by change ¬ Nat.Prime 6337807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 905401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1259 : row 1259 = 0 := (row_1259.checked).trans rfl
theorem model_count_1259 : modelRow 1259 = 0 := by decide
theorem modelrow_agrees_1259 : modelRow 1259 = row 1259 := model_count_1259.trans row_count_1259.symm
def row_1260 : RowCertificate 1260 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1260 1) := by change ¬ Nat.Prime 6352921; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 50023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1260 2) := by change ¬ Nat.Prime 6350401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 373553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1260 3) := by change ¬ Nat.Prime 6347881; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 334099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1260 : row 1260 = 0 := (row_1260.checked).trans rfl
theorem model_count_1260 : modelRow 1260 = 0 := by decide
theorem modelrow_agrees_1260 : modelRow 1260 = row 1260 := model_count_1260.trans row_count_1260.symm
def row_1261 : RowCertificate 1261 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1261 1) := by change ¬ Nat.Prime 6363007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 909001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1261 2) := by change ¬ Nat.Prime 6360485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1272097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1261 3) := by change ¬ Nat.Prime 6357963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2119321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1261 : row 1261 = 0 := (row_1261.checked).trans rfl
theorem model_count_1261 : modelRow 1261 = 0 := by decide
theorem modelrow_agrees_1261 : modelRow 1261 = row 1261 := model_count_1261.trans row_count_1261.symm
def row_1262 : RowCertificate 1262 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1262 1) := by change ¬ Nat.Prime 6373101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2124367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1262 2) := by change ¬ Nat.Prime 6370577; exact PrattCertificates.composite_of_product (factor := 1049) (cofactor := 6073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1262 3) := by change Nat.Prime 6368053; exact ProjectEulerSolutions.P58PrimeNodes.p_6368053.prime
    exact decide_eq_true prime
theorem row_count_1262 : row 1262 = 1 := (row_1262.checked).trans rfl
theorem model_count_1262 : modelRow 1262 = 1 := by decide
theorem modelrow_agrees_1262 : modelRow 1262 = row 1262 := model_count_1262.trans row_count_1262.symm
def row_1263 : RowCertificate 1263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1263 1) := by change ¬ Nat.Prime 6383203; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 172519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1263 2) := by change ¬ Nat.Prime 6380677; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 71693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1263 3) := by change ¬ Nat.Prime 6378151; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 490627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1263 : row 1263 = 0 := (row_1263.checked).trans rfl
theorem model_count_1263 : modelRow 1263 = 0 := by decide
theorem modelrow_agrees_1263 : modelRow 1263 = row 1263 := model_count_1263.trans row_count_1263.symm
def row_1264 : RowCertificate 1264 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1264 1) := by change ¬ Nat.Prime 6393313; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 62071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1264 2) := by change ¬ Nat.Prime 6390785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1278157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1264 3) := by change ¬ Nat.Prime 6388257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2129419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1264 : row 1264 = 0 := (row_1264.checked).trans rfl
theorem model_count_1264 : modelRow 1264 = 0 := by decide
theorem modelrow_agrees_1264 : modelRow 1264 = row 1264 := model_count_1264.trans row_count_1264.symm
def row_1265 : RowCertificate 1265 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1265 1) := by change ¬ Nat.Prime 6403431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2134477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1265 2) := by change ¬ Nat.Prime 6400901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 492377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1265 3) := by change ¬ Nat.Prime 6398371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 914053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1265 : row 1265 = 0 := (row_1265.checked).trans rfl
theorem model_count_1265 : modelRow 1265 = 0 := by decide
theorem modelrow_agrees_1265 : modelRow 1265 = row 1265 := model_count_1265.trans row_count_1265.symm
def row_1266 : RowCertificate 1266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1266 1) := by change ¬ Nat.Prime 6413557; exact PrattCertificates.composite_of_product (factor := 2083) (cofactor := 3079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1266 2) := by change ¬ Nat.Prime 6411025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1282205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1266 3) := by change ¬ Nat.Prime 6408493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 915499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1266 : row 1266 = 0 := (row_1266.checked).trans rfl
theorem model_count_1266 : modelRow 1266 = 0 := by decide
theorem modelrow_agrees_1266 : modelRow 1266 = row 1266 := model_count_1266.trans row_count_1266.symm
def row_1267 : RowCertificate 1267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1267 1) := by change ¬ Nat.Prime 6423691; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 338089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1267 2) := by change Nat.Prime 6421157; exact ProjectEulerSolutions.P58PrimeNodes.p_6421157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1267 3) := by change ¬ Nat.Prime 6418623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2139541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1267 : row 1267 = 1 := (row_1267.checked).trans rfl
theorem model_count_1267 : modelRow 1267 = 1 := by decide
theorem modelrow_agrees_1267 : modelRow 1267 = row 1267 := model_count_1267.trans row_count_1267.symm
def row_1268 : RowCertificate 1268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1268 1) := by change ¬ Nat.Prime 6433833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2144611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1268 2) := by change Nat.Prime 6431297; exact ProjectEulerSolutions.P58PrimeNodes.p_6431297.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1268 3) := by change Nat.Prime 6428761; exact ProjectEulerSolutions.P58PrimeNodes.p_6428761.prime
    exact decide_eq_true prime
theorem row_count_1268 : row 1268 = 2 := (row_1268.checked).trans rfl
theorem model_count_1268 : modelRow 1268 = 2 := by decide
theorem modelrow_agrees_1268 : modelRow 1268 = row 1268 := model_count_1268.trans row_count_1268.symm
def row_1269 : RowCertificate 1269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1269 1) := by change ¬ Nat.Prime 6443983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 920569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1269 2) := by change ¬ Nat.Prime 6441445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1288289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1269 3) := by change ¬ Nat.Prime 6438907; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 8713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1269 : row 1269 = 0 := (row_1269.checked).trans rfl
theorem model_count_1269 : modelRow 1269 = 0 := by decide
theorem modelrow_agrees_1269 : modelRow 1269 = row 1269 := model_count_1269.trans row_count_1269.symm
def row_1270 : RowCertificate 1270 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1270 1) := by change Nat.Prime 6454141; exact ProjectEulerSolutions.P58PrimeNodes.p_6454141.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1270 2) := by change ¬ Nat.Prime 6451601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 496277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1270 3) := by change ¬ Nat.Prime 6449061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2149687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1270 : row 1270 = 1 := (row_1270.checked).trans rfl
theorem model_count_1270 : modelRow 1270 = 1 := by decide
theorem modelrow_agrees_1270 : modelRow 1270 = row 1270 := model_count_1270.trans row_count_1270.symm
def row_1271 : RowCertificate 1271 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1271 1) := by change ¬ Nat.Prime 6464307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2154769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1271 2) := by change ¬ Nat.Prime 6461765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1292353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1271 3) := by change Nat.Prime 6459223; exact ProjectEulerSolutions.P58PrimeNodes.p_6459223.prime
    exact decide_eq_true prime
theorem row_count_1271 : row 1271 = 1 := (row_1271.checked).trans rfl
theorem model_count_1271 : modelRow 1271 = 1 := by decide
theorem modelrow_agrees_1271 : modelRow 1271 = row 1271 := model_count_1271.trans row_count_1271.symm
def row_1272 : RowCertificate 1272 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1272 1) := by change ¬ Nat.Prime 6474481; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 498037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1272 2) := by change ¬ Nat.Prime 6471937; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 66721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1272 3) := by change ¬ Nat.Prime 6469393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 924199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1272 : row 1272 = 0 := (row_1272.checked).trans rfl
theorem model_count_1272 : modelRow 1272 = 0 := by decide
theorem modelrow_agrees_1272 : modelRow 1272 = row 1272 := model_count_1272.trans row_count_1272.symm
def row_1273 : RowCertificate 1273 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1273 1) := by change ¬ Nat.Prime 6484663; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 88831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1273 2) := by change ¬ Nat.Prime 6482117; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 381301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1273 3) := by change ¬ Nat.Prime 6479571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2159857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1273 : row 1273 = 0 := (row_1273.checked).trans rfl
theorem model_count_1273 : modelRow 1273 = 0 := by decide
theorem modelrow_agrees_1273 : modelRow 1273 = row 1273 := model_count_1273.trans row_count_1273.symm
def row_1274 : RowCertificate 1274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1274 1) := by change ¬ Nat.Prime 6494853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2164951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1274 2) := by change ¬ Nat.Prime 6492305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1298461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1274 3) := by change ¬ Nat.Prime 6489757; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 209347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1274 : row 1274 = 0 := (row_1274.checked).trans rfl
theorem model_count_1274 : modelRow 1274 = 0 := by decide
theorem modelrow_agrees_1274 : modelRow 1274 = row 1274 := model_count_1274.trans row_count_1274.symm
def row_1275 : RowCertificate 1275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1275 1) := by change ¬ Nat.Prime 6505051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 929293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1275 2) := by change Nat.Prime 6502501; exact ProjectEulerSolutions.P58PrimeNodes.p_6502501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1275 3) := by change ¬ Nat.Prime 6499951; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 39877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1275 : row 1275 = 1 := (row_1275.checked).trans rfl
theorem model_count_1275 : modelRow 1275 = 1 := by decide
theorem modelrow_agrees_1275 : modelRow 1275 = row 1275 := model_count_1275.trans row_count_1275.symm
def row_1276 : RowCertificate 1276 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1276 1) := by change ¬ Nat.Prime 6515257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 930751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1276 2) := by change ¬ Nat.Prime 6512705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1302541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1276 3) := by change ¬ Nat.Prime 6510153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2170051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1276 : row 1276 = 0 := (row_1276.checked).trans rfl
theorem model_count_1276 : modelRow 1276 = 0 := by decide
theorem modelrow_agrees_1276 : modelRow 1276 = row 1276 := model_count_1276.trans row_count_1276.symm
def row_1277 : RowCertificate 1277 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1277 1) := by change ¬ Nat.Prime 6525471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2175157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1277 2) := by change ¬ Nat.Prime 6522917; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 383701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1277 3) := by change ¬ Nat.Prime 6520363; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 343177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1277 : row 1277 = 0 := (row_1277.checked).trans rfl
theorem model_count_1277 : modelRow 1277 = 0 := by decide
theorem modelrow_agrees_1277 : modelRow 1277 = row 1277 := model_count_1277.trans row_count_1277.symm
def row_1278 : RowCertificate 1278 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1278 1) := by change Nat.Prime 6535693; exact ProjectEulerSolutions.P58PrimeNodes.p_6535693.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1278 2) := by change ¬ Nat.Prime 6533137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 502549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1278 3) := by change Nat.Prime 6530581; exact ProjectEulerSolutions.P58PrimeNodes.p_6530581.prime
    exact decide_eq_true prime
theorem row_count_1278 : row 1278 = 2 := (row_1278.checked).trans rfl
theorem model_count_1278 : modelRow 1278 = 2 := by decide
theorem modelrow_agrees_1278 : modelRow 1278 = row 1278 := model_count_1278.trans row_count_1278.symm
def row_1279 : RowCertificate 1279 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1279 1) := by change Nat.Prime 6545923; exact ProjectEulerSolutions.P58PrimeNodes.p_6545923.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1279 2) := by change ¬ Nat.Prime 6543365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1308673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1279 3) := by change ¬ Nat.Prime 6540807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2180269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1279 : row 1279 = 1 := (row_1279.checked).trans rfl
theorem model_count_1279 : modelRow 1279 = 1 := by decide
theorem modelrow_agrees_1279 : modelRow 1279 = row 1279 := model_count_1279.trans row_count_1279.symm
def row_1280 : RowCertificate 1280 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1280 1) := by change ¬ Nat.Prime 6556161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2185387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1280 2) := by change ¬ Nat.Prime 6553601; exact PrattCertificates.composite_of_product (factor := 2141) (cofactor := 3061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1280 3) := by change ¬ Nat.Prime 6551041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 935863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1280 : row 1280 = 0 := (row_1280.checked).trans rfl
theorem model_count_1280 : modelRow 1280 = 0 := by decide
theorem modelrow_agrees_1280 : modelRow 1280 = row 1280 := model_count_1280.trans row_count_1280.symm
private def atomic_1153 : CountSegment 1 10 13120 1152 1153 626 626 := CountSegment.single (first := 1152) (initial := 626) row_count_1153 (by decide)
private theorem atomic_1153_agreement : RowAgreement 1152 1153 := RowAgreement.single (first := 1152) modelrow_agrees_1153
private def atomic_1154 : CountSegment 1 10 13120 1153 1154 626 626 := CountSegment.single (first := 1153) (initial := 626) row_count_1154 (by decide)
private theorem atomic_1154_agreement : RowAgreement 1153 1154 := RowAgreement.single (first := 1153) modelrow_agrees_1154
private def atomic_1155 : CountSegment 1 10 13120 1154 1155 626 628 := CountSegment.single (first := 1154) (initial := 626) row_count_1155 (by decide)
private theorem atomic_1155_agreement : RowAgreement 1154 1155 := RowAgreement.single (first := 1154) modelrow_agrees_1155
private def atomic_1156 : CountSegment 1 10 13120 1155 1156 628 628 := CountSegment.single (first := 1155) (initial := 628) row_count_1156 (by decide)
private theorem atomic_1156_agreement : RowAgreement 1155 1156 := RowAgreement.single (first := 1155) modelrow_agrees_1156
private def atomic_1157 : CountSegment 1 10 13120 1156 1157 628 630 := CountSegment.single (first := 1156) (initial := 628) row_count_1157 (by decide)
private theorem atomic_1157_agreement : RowAgreement 1156 1157 := RowAgreement.single (first := 1156) modelrow_agrees_1157
private def atomic_1158 : CountSegment 1 10 13120 1157 1158 630 630 := CountSegment.single (first := 1157) (initial := 630) row_count_1158 (by decide)
private theorem atomic_1158_agreement : RowAgreement 1157 1158 := RowAgreement.single (first := 1157) modelrow_agrees_1158
private def atomic_1159 : CountSegment 1 10 13120 1158 1159 630 631 := CountSegment.single (first := 1158) (initial := 630) row_count_1159 (by decide)
private theorem atomic_1159_agreement : RowAgreement 1158 1159 := RowAgreement.single (first := 1158) modelrow_agrees_1159
private def atomic_1160 : CountSegment 1 10 13120 1159 1160 631 632 := CountSegment.single (first := 1159) (initial := 631) row_count_1160 (by decide)
private theorem atomic_1160_agreement : RowAgreement 1159 1160 := RowAgreement.single (first := 1159) modelrow_agrees_1160
private def atomic_1161 : CountSegment 1 10 13120 1160 1161 632 632 := CountSegment.single (first := 1160) (initial := 632) row_count_1161 (by decide)
private theorem atomic_1161_agreement : RowAgreement 1160 1161 := RowAgreement.single (first := 1160) modelrow_agrees_1161
private def atomic_1162 : CountSegment 1 10 13120 1161 1162 632 633 := CountSegment.single (first := 1161) (initial := 632) row_count_1162 (by decide)
private theorem atomic_1162_agreement : RowAgreement 1161 1162 := RowAgreement.single (first := 1161) modelrow_agrees_1162
private def atomic_1163 : CountSegment 1 10 13120 1162 1163 633 634 := CountSegment.single (first := 1162) (initial := 633) row_count_1163 (by decide)
private theorem atomic_1163_agreement : RowAgreement 1162 1163 := RowAgreement.single (first := 1162) modelrow_agrees_1163
private def atomic_1164 : CountSegment 1 10 13120 1163 1164 634 634 := CountSegment.single (first := 1163) (initial := 634) row_count_1164 (by decide)
private theorem atomic_1164_agreement : RowAgreement 1163 1164 := RowAgreement.single (first := 1163) modelrow_agrees_1164
private def atomic_1165 : CountSegment 1 10 13120 1164 1165 634 635 := CountSegment.single (first := 1164) (initial := 634) row_count_1165 (by decide)
private theorem atomic_1165_agreement : RowAgreement 1164 1165 := RowAgreement.single (first := 1164) modelrow_agrees_1165
private def atomic_1166 : CountSegment 1 10 13120 1165 1166 635 635 := CountSegment.single (first := 1165) (initial := 635) row_count_1166 (by decide)
private theorem atomic_1166_agreement : RowAgreement 1165 1166 := RowAgreement.single (first := 1165) modelrow_agrees_1166
private def atomic_1167 : CountSegment 1 10 13120 1166 1167 635 635 := CountSegment.single (first := 1166) (initial := 635) row_count_1167 (by decide)
private theorem atomic_1167_agreement : RowAgreement 1166 1167 := RowAgreement.single (first := 1166) modelrow_agrees_1167
private def atomic_1168 : CountSegment 1 10 13120 1167 1168 635 636 := CountSegment.single (first := 1167) (initial := 635) row_count_1168 (by decide)
private theorem atomic_1168_agreement : RowAgreement 1167 1168 := RowAgreement.single (first := 1167) modelrow_agrees_1168
private def atomic_1169 : CountSegment 1 10 13120 1168 1169 636 636 := CountSegment.single (first := 1168) (initial := 636) row_count_1169 (by decide)
private theorem atomic_1169_agreement : RowAgreement 1168 1169 := RowAgreement.single (first := 1168) modelrow_agrees_1169
private def atomic_1170 : CountSegment 1 10 13120 1169 1170 636 637 := CountSegment.single (first := 1169) (initial := 636) row_count_1170 (by decide)
private theorem atomic_1170_agreement : RowAgreement 1169 1170 := RowAgreement.single (first := 1169) modelrow_agrees_1170
private def atomic_1171 : CountSegment 1 10 13120 1170 1171 637 637 := CountSegment.single (first := 1170) (initial := 637) row_count_1171 (by decide)
private theorem atomic_1171_agreement : RowAgreement 1170 1171 := RowAgreement.single (first := 1170) modelrow_agrees_1171
private def atomic_1172 : CountSegment 1 10 13120 1171 1172 637 637 := CountSegment.single (first := 1171) (initial := 637) row_count_1172 (by decide)
private theorem atomic_1172_agreement : RowAgreement 1171 1172 := RowAgreement.single (first := 1171) modelrow_agrees_1172
private def atomic_1173 : CountSegment 1 10 13120 1172 1173 637 638 := CountSegment.single (first := 1172) (initial := 637) row_count_1173 (by decide)
private theorem atomic_1173_agreement : RowAgreement 1172 1173 := RowAgreement.single (first := 1172) modelrow_agrees_1173
private def atomic_1174 : CountSegment 1 10 13120 1173 1174 638 638 := CountSegment.single (first := 1173) (initial := 638) row_count_1174 (by decide)
private theorem atomic_1174_agreement : RowAgreement 1173 1174 := RowAgreement.single (first := 1173) modelrow_agrees_1174
private def atomic_1175 : CountSegment 1 10 13120 1174 1175 638 638 := CountSegment.single (first := 1174) (initial := 638) row_count_1175 (by decide)
private theorem atomic_1175_agreement : RowAgreement 1174 1175 := RowAgreement.single (first := 1174) modelrow_agrees_1175
private def atomic_1176 : CountSegment 1 10 13120 1175 1176 638 638 := CountSegment.single (first := 1175) (initial := 638) row_count_1176 (by decide)
private theorem atomic_1176_agreement : RowAgreement 1175 1176 := RowAgreement.single (first := 1175) modelrow_agrees_1176
private def atomic_1177 : CountSegment 1 10 13120 1176 1177 638 639 := CountSegment.single (first := 1176) (initial := 638) row_count_1177 (by decide)
private theorem atomic_1177_agreement : RowAgreement 1176 1177 := RowAgreement.single (first := 1176) modelrow_agrees_1177
private def atomic_1178 : CountSegment 1 10 13120 1177 1178 639 639 := CountSegment.single (first := 1177) (initial := 639) row_count_1178 (by decide)
private theorem atomic_1178_agreement : RowAgreement 1177 1178 := RowAgreement.single (first := 1177) modelrow_agrees_1178
private def atomic_1179 : CountSegment 1 10 13120 1178 1179 639 639 := CountSegment.single (first := 1178) (initial := 639) row_count_1179 (by decide)
private theorem atomic_1179_agreement : RowAgreement 1178 1179 := RowAgreement.single (first := 1178) modelrow_agrees_1179
private def atomic_1180 : CountSegment 1 10 13120 1179 1180 639 641 := CountSegment.single (first := 1179) (initial := 639) row_count_1180 (by decide)
private theorem atomic_1180_agreement : RowAgreement 1179 1180 := RowAgreement.single (first := 1179) modelrow_agrees_1180
private def atomic_1181 : CountSegment 1 10 13120 1180 1181 641 641 := CountSegment.single (first := 1180) (initial := 641) row_count_1181 (by decide)
private theorem atomic_1181_agreement : RowAgreement 1180 1181 := RowAgreement.single (first := 1180) modelrow_agrees_1181
private def atomic_1182 : CountSegment 1 10 13120 1181 1182 641 642 := CountSegment.single (first := 1181) (initial := 641) row_count_1182 (by decide)
private theorem atomic_1182_agreement : RowAgreement 1181 1182 := RowAgreement.single (first := 1181) modelrow_agrees_1182
private def atomic_1183 : CountSegment 1 10 13120 1182 1183 642 642 := CountSegment.single (first := 1182) (initial := 642) row_count_1183 (by decide)
private theorem atomic_1183_agreement : RowAgreement 1182 1183 := RowAgreement.single (first := 1182) modelrow_agrees_1183
private def atomic_1184 : CountSegment 1 10 13120 1183 1184 642 642 := CountSegment.single (first := 1183) (initial := 642) row_count_1184 (by decide)
private theorem atomic_1184_agreement : RowAgreement 1183 1184 := RowAgreement.single (first := 1183) modelrow_agrees_1184
private def atomic_1185 : CountSegment 1 10 13120 1184 1185 642 642 := CountSegment.single (first := 1184) (initial := 642) row_count_1185 (by decide)
private theorem atomic_1185_agreement : RowAgreement 1184 1185 := RowAgreement.single (first := 1184) modelrow_agrees_1185
private def atomic_1186 : CountSegment 1 10 13120 1185 1186 642 643 := CountSegment.single (first := 1185) (initial := 642) row_count_1186 (by decide)
private theorem atomic_1186_agreement : RowAgreement 1185 1186 := RowAgreement.single (first := 1185) modelrow_agrees_1186
private def atomic_1187 : CountSegment 1 10 13120 1186 1187 643 644 := CountSegment.single (first := 1186) (initial := 643) row_count_1187 (by decide)
private theorem atomic_1187_agreement : RowAgreement 1186 1187 := RowAgreement.single (first := 1186) modelrow_agrees_1187
private def atomic_1188 : CountSegment 1 10 13120 1187 1188 644 644 := CountSegment.single (first := 1187) (initial := 644) row_count_1188 (by decide)
private theorem atomic_1188_agreement : RowAgreement 1187 1188 := RowAgreement.single (first := 1187) modelrow_agrees_1188
private def atomic_1189 : CountSegment 1 10 13120 1188 1189 644 644 := CountSegment.single (first := 1188) (initial := 644) row_count_1189 (by decide)
private theorem atomic_1189_agreement : RowAgreement 1188 1189 := RowAgreement.single (first := 1188) modelrow_agrees_1189
private def atomic_1190 : CountSegment 1 10 13120 1189 1190 644 645 := CountSegment.single (first := 1189) (initial := 644) row_count_1190 (by decide)
private theorem atomic_1190_agreement : RowAgreement 1189 1190 := RowAgreement.single (first := 1189) modelrow_agrees_1190
private def atomic_1191 : CountSegment 1 10 13120 1190 1191 645 645 := CountSegment.single (first := 1190) (initial := 645) row_count_1191 (by decide)
private theorem atomic_1191_agreement : RowAgreement 1190 1191 := RowAgreement.single (first := 1190) modelrow_agrees_1191
private def atomic_1192 : CountSegment 1 10 13120 1191 1192 645 645 := CountSegment.single (first := 1191) (initial := 645) row_count_1192 (by decide)
private theorem atomic_1192_agreement : RowAgreement 1191 1192 := RowAgreement.single (first := 1191) modelrow_agrees_1192
private def atomic_1193 : CountSegment 1 10 13120 1192 1193 645 646 := CountSegment.single (first := 1192) (initial := 645) row_count_1193 (by decide)
private theorem atomic_1193_agreement : RowAgreement 1192 1193 := RowAgreement.single (first := 1192) modelrow_agrees_1193
private def atomic_1194 : CountSegment 1 10 13120 1193 1194 646 647 := CountSegment.single (first := 1193) (initial := 646) row_count_1194 (by decide)
private theorem atomic_1194_agreement : RowAgreement 1193 1194 := RowAgreement.single (first := 1193) modelrow_agrees_1194
private def atomic_1195 : CountSegment 1 10 13120 1194 1195 647 648 := CountSegment.single (first := 1194) (initial := 647) row_count_1195 (by decide)
private theorem atomic_1195_agreement : RowAgreement 1194 1195 := RowAgreement.single (first := 1194) modelrow_agrees_1195
private def atomic_1196 : CountSegment 1 10 13120 1195 1196 648 648 := CountSegment.single (first := 1195) (initial := 648) row_count_1196 (by decide)
private theorem atomic_1196_agreement : RowAgreement 1195 1196 := RowAgreement.single (first := 1195) modelrow_agrees_1196
private def atomic_1197 : CountSegment 1 10 13120 1196 1197 648 648 := CountSegment.single (first := 1196) (initial := 648) row_count_1197 (by decide)
private theorem atomic_1197_agreement : RowAgreement 1196 1197 := RowAgreement.single (first := 1196) modelrow_agrees_1197
private def atomic_1198 : CountSegment 1 10 13120 1197 1198 648 648 := CountSegment.single (first := 1197) (initial := 648) row_count_1198 (by decide)
private theorem atomic_1198_agreement : RowAgreement 1197 1198 := RowAgreement.single (first := 1197) modelrow_agrees_1198
private def atomic_1199 : CountSegment 1 10 13120 1198 1199 648 649 := CountSegment.single (first := 1198) (initial := 648) row_count_1199 (by decide)
private theorem atomic_1199_agreement : RowAgreement 1198 1199 := RowAgreement.single (first := 1198) modelrow_agrees_1199
private def atomic_1200 : CountSegment 1 10 13120 1199 1200 649 649 := CountSegment.single (first := 1199) (initial := 649) row_count_1200 (by decide)
private theorem atomic_1200_agreement : RowAgreement 1199 1200 := RowAgreement.single (first := 1199) modelrow_agrees_1200
private def atomic_1201 : CountSegment 1 10 13120 1200 1201 649 650 := CountSegment.single (first := 1200) (initial := 649) row_count_1201 (by decide)
private theorem atomic_1201_agreement : RowAgreement 1200 1201 := RowAgreement.single (first := 1200) modelrow_agrees_1201
private def atomic_1202 : CountSegment 1 10 13120 1201 1202 650 651 := CountSegment.single (first := 1201) (initial := 650) row_count_1202 (by decide)
private theorem atomic_1202_agreement : RowAgreement 1201 1202 := RowAgreement.single (first := 1201) modelrow_agrees_1202
private def atomic_1203 : CountSegment 1 10 13120 1202 1203 651 652 := CountSegment.single (first := 1202) (initial := 651) row_count_1203 (by decide)
private theorem atomic_1203_agreement : RowAgreement 1202 1203 := RowAgreement.single (first := 1202) modelrow_agrees_1203
private def atomic_1204 : CountSegment 1 10 13120 1203 1204 652 652 := CountSegment.single (first := 1203) (initial := 652) row_count_1204 (by decide)
private theorem atomic_1204_agreement : RowAgreement 1203 1204 := RowAgreement.single (first := 1203) modelrow_agrees_1204
private def atomic_1205 : CountSegment 1 10 13120 1204 1205 652 653 := CountSegment.single (first := 1204) (initial := 652) row_count_1205 (by decide)
private theorem atomic_1205_agreement : RowAgreement 1204 1205 := RowAgreement.single (first := 1204) modelrow_agrees_1205
private def atomic_1206 : CountSegment 1 10 13120 1205 1206 653 654 := CountSegment.single (first := 1205) (initial := 653) row_count_1206 (by decide)
private theorem atomic_1206_agreement : RowAgreement 1205 1206 := RowAgreement.single (first := 1205) modelrow_agrees_1206
private def atomic_1207 : CountSegment 1 10 13120 1206 1207 654 654 := CountSegment.single (first := 1206) (initial := 654) row_count_1207 (by decide)
private theorem atomic_1207_agreement : RowAgreement 1206 1207 := RowAgreement.single (first := 1206) modelrow_agrees_1207
private def atomic_1208 : CountSegment 1 10 13120 1207 1208 654 654 := CountSegment.single (first := 1207) (initial := 654) row_count_1208 (by decide)
private theorem atomic_1208_agreement : RowAgreement 1207 1208 := RowAgreement.single (first := 1207) modelrow_agrees_1208
private def atomic_1209 : CountSegment 1 10 13120 1208 1209 654 654 := CountSegment.single (first := 1208) (initial := 654) row_count_1209 (by decide)
private theorem atomic_1209_agreement : RowAgreement 1208 1209 := RowAgreement.single (first := 1208) modelrow_agrees_1209
private def atomic_1210 : CountSegment 1 10 13120 1209 1210 654 655 := CountSegment.single (first := 1209) (initial := 654) row_count_1210 (by decide)
private theorem atomic_1210_agreement : RowAgreement 1209 1210 := RowAgreement.single (first := 1209) modelrow_agrees_1210
private def atomic_1211 : CountSegment 1 10 13120 1210 1211 655 655 := CountSegment.single (first := 1210) (initial := 655) row_count_1211 (by decide)
private theorem atomic_1211_agreement : RowAgreement 1210 1211 := RowAgreement.single (first := 1210) modelrow_agrees_1211
private def atomic_1212 : CountSegment 1 10 13120 1211 1212 655 655 := CountSegment.single (first := 1211) (initial := 655) row_count_1212 (by decide)
private theorem atomic_1212_agreement : RowAgreement 1211 1212 := RowAgreement.single (first := 1211) modelrow_agrees_1212
private def atomic_1213 : CountSegment 1 10 13120 1212 1213 655 655 := CountSegment.single (first := 1212) (initial := 655) row_count_1213 (by decide)
private theorem atomic_1213_agreement : RowAgreement 1212 1213 := RowAgreement.single (first := 1212) modelrow_agrees_1213
private def atomic_1214 : CountSegment 1 10 13120 1213 1214 655 655 := CountSegment.single (first := 1213) (initial := 655) row_count_1214 (by decide)
private theorem atomic_1214_agreement : RowAgreement 1213 1214 := RowAgreement.single (first := 1213) modelrow_agrees_1214
private def atomic_1215 : CountSegment 1 10 13120 1214 1215 655 658 := CountSegment.single (first := 1214) (initial := 655) row_count_1215 (by decide)
private theorem atomic_1215_agreement : RowAgreement 1214 1215 := RowAgreement.single (first := 1214) modelrow_agrees_1215
private def atomic_1216 : CountSegment 1 10 13120 1215 1216 658 658 := CountSegment.single (first := 1215) (initial := 658) row_count_1216 (by decide)
private theorem atomic_1216_agreement : RowAgreement 1215 1216 := RowAgreement.single (first := 1215) modelrow_agrees_1216
private def atomic_1217 : CountSegment 1 10 13120 1216 1217 658 658 := CountSegment.single (first := 1216) (initial := 658) row_count_1217 (by decide)
private theorem atomic_1217_agreement : RowAgreement 1216 1217 := RowAgreement.single (first := 1216) modelrow_agrees_1217
private def atomic_1218 : CountSegment 1 10 13120 1217 1218 658 659 := CountSegment.single (first := 1217) (initial := 658) row_count_1218 (by decide)
private theorem atomic_1218_agreement : RowAgreement 1217 1218 := RowAgreement.single (first := 1217) modelrow_agrees_1218
private def atomic_1219 : CountSegment 1 10 13120 1218 1219 659 659 := CountSegment.single (first := 1218) (initial := 659) row_count_1219 (by decide)
private theorem atomic_1219_agreement : RowAgreement 1218 1219 := RowAgreement.single (first := 1218) modelrow_agrees_1219
private def atomic_1220 : CountSegment 1 10 13120 1219 1220 659 659 := CountSegment.single (first := 1219) (initial := 659) row_count_1220 (by decide)
private theorem atomic_1220_agreement : RowAgreement 1219 1220 := RowAgreement.single (first := 1219) modelrow_agrees_1220
private def atomic_1221 : CountSegment 1 10 13120 1220 1221 659 660 := CountSegment.single (first := 1220) (initial := 659) row_count_1221 (by decide)
private theorem atomic_1221_agreement : RowAgreement 1220 1221 := RowAgreement.single (first := 1220) modelrow_agrees_1221
private def atomic_1222 : CountSegment 1 10 13120 1221 1222 660 660 := CountSegment.single (first := 1221) (initial := 660) row_count_1222 (by decide)
private theorem atomic_1222_agreement : RowAgreement 1221 1222 := RowAgreement.single (first := 1221) modelrow_agrees_1222
private def atomic_1223 : CountSegment 1 10 13120 1222 1223 660 660 := CountSegment.single (first := 1222) (initial := 660) row_count_1223 (by decide)
private theorem atomic_1223_agreement : RowAgreement 1222 1223 := RowAgreement.single (first := 1222) modelrow_agrees_1223
private def atomic_1224 : CountSegment 1 10 13120 1223 1224 660 660 := CountSegment.single (first := 1223) (initial := 660) row_count_1224 (by decide)
private theorem atomic_1224_agreement : RowAgreement 1223 1224 := RowAgreement.single (first := 1223) modelrow_agrees_1224
private def atomic_1225 : CountSegment 1 10 13120 1224 1225 660 661 := CountSegment.single (first := 1224) (initial := 660) row_count_1225 (by decide)
private theorem atomic_1225_agreement : RowAgreement 1224 1225 := RowAgreement.single (first := 1224) modelrow_agrees_1225
private def atomic_1226 : CountSegment 1 10 13120 1225 1226 661 662 := CountSegment.single (first := 1225) (initial := 661) row_count_1226 (by decide)
private theorem atomic_1226_agreement : RowAgreement 1225 1226 := RowAgreement.single (first := 1225) modelrow_agrees_1226
private def atomic_1227 : CountSegment 1 10 13120 1226 1227 662 663 := CountSegment.single (first := 1226) (initial := 662) row_count_1227 (by decide)
private theorem atomic_1227_agreement : RowAgreement 1226 1227 := RowAgreement.single (first := 1226) modelrow_agrees_1227
private def atomic_1228 : CountSegment 1 10 13120 1227 1228 663 665 := CountSegment.single (first := 1227) (initial := 663) row_count_1228 (by decide)
private theorem atomic_1228_agreement : RowAgreement 1227 1228 := RowAgreement.single (first := 1227) modelrow_agrees_1228
private def atomic_1229 : CountSegment 1 10 13120 1228 1229 665 665 := CountSegment.single (first := 1228) (initial := 665) row_count_1229 (by decide)
private theorem atomic_1229_agreement : RowAgreement 1228 1229 := RowAgreement.single (first := 1228) modelrow_agrees_1229
private def atomic_1230 : CountSegment 1 10 13120 1229 1230 665 666 := CountSegment.single (first := 1229) (initial := 665) row_count_1230 (by decide)
private theorem atomic_1230_agreement : RowAgreement 1229 1230 := RowAgreement.single (first := 1229) modelrow_agrees_1230
private def atomic_1231 : CountSegment 1 10 13120 1230 1231 666 666 := CountSegment.single (first := 1230) (initial := 666) row_count_1231 (by decide)
private theorem atomic_1231_agreement : RowAgreement 1230 1231 := RowAgreement.single (first := 1230) modelrow_agrees_1231
private def atomic_1232 : CountSegment 1 10 13120 1231 1232 666 668 := CountSegment.single (first := 1231) (initial := 666) row_count_1232 (by decide)
private theorem atomic_1232_agreement : RowAgreement 1231 1232 := RowAgreement.single (first := 1231) modelrow_agrees_1232
private def atomic_1233 : CountSegment 1 10 13120 1232 1233 668 669 := CountSegment.single (first := 1232) (initial := 668) row_count_1233 (by decide)
private theorem atomic_1233_agreement : RowAgreement 1232 1233 := RowAgreement.single (first := 1232) modelrow_agrees_1233
private def atomic_1234 : CountSegment 1 10 13120 1233 1234 669 669 := CountSegment.single (first := 1233) (initial := 669) row_count_1234 (by decide)
private theorem atomic_1234_agreement : RowAgreement 1233 1234 := RowAgreement.single (first := 1233) modelrow_agrees_1234
private def atomic_1235 : CountSegment 1 10 13120 1234 1235 669 670 := CountSegment.single (first := 1234) (initial := 669) row_count_1235 (by decide)
private theorem atomic_1235_agreement : RowAgreement 1234 1235 := RowAgreement.single (first := 1234) modelrow_agrees_1235
private def atomic_1236 : CountSegment 1 10 13120 1235 1236 670 671 := CountSegment.single (first := 1235) (initial := 670) row_count_1236 (by decide)
private theorem atomic_1236_agreement : RowAgreement 1235 1236 := RowAgreement.single (first := 1235) modelrow_agrees_1236
private def atomic_1237 : CountSegment 1 10 13120 1236 1237 671 671 := CountSegment.single (first := 1236) (initial := 671) row_count_1237 (by decide)
private theorem atomic_1237_agreement : RowAgreement 1236 1237 := RowAgreement.single (first := 1236) modelrow_agrees_1237
private def atomic_1238 : CountSegment 1 10 13120 1237 1238 671 671 := CountSegment.single (first := 1237) (initial := 671) row_count_1238 (by decide)
private theorem atomic_1238_agreement : RowAgreement 1237 1238 := RowAgreement.single (first := 1237) modelrow_agrees_1238
private def atomic_1239 : CountSegment 1 10 13120 1238 1239 671 672 := CountSegment.single (first := 1238) (initial := 671) row_count_1239 (by decide)
private theorem atomic_1239_agreement : RowAgreement 1238 1239 := RowAgreement.single (first := 1238) modelrow_agrees_1239
private def atomic_1240 : CountSegment 1 10 13120 1239 1240 672 672 := CountSegment.single (first := 1239) (initial := 672) row_count_1240 (by decide)
private theorem atomic_1240_agreement : RowAgreement 1239 1240 := RowAgreement.single (first := 1239) modelrow_agrees_1240
private def atomic_1241 : CountSegment 1 10 13120 1240 1241 672 672 := CountSegment.single (first := 1240) (initial := 672) row_count_1241 (by decide)
private theorem atomic_1241_agreement : RowAgreement 1240 1241 := RowAgreement.single (first := 1240) modelrow_agrees_1241
private def atomic_1242 : CountSegment 1 10 13120 1241 1242 672 674 := CountSegment.single (first := 1241) (initial := 672) row_count_1242 (by decide)
private theorem atomic_1242_agreement : RowAgreement 1241 1242 := RowAgreement.single (first := 1241) modelrow_agrees_1242
private def atomic_1243 : CountSegment 1 10 13120 1242 1243 674 674 := CountSegment.single (first := 1242) (initial := 674) row_count_1243 (by decide)
private theorem atomic_1243_agreement : RowAgreement 1242 1243 := RowAgreement.single (first := 1242) modelrow_agrees_1243
private def atomic_1244 : CountSegment 1 10 13120 1243 1244 674 674 := CountSegment.single (first := 1243) (initial := 674) row_count_1244 (by decide)
private theorem atomic_1244_agreement : RowAgreement 1243 1244 := RowAgreement.single (first := 1243) modelrow_agrees_1244
private def atomic_1245 : CountSegment 1 10 13120 1244 1245 674 674 := CountSegment.single (first := 1244) (initial := 674) row_count_1245 (by decide)
private theorem atomic_1245_agreement : RowAgreement 1244 1245 := RowAgreement.single (first := 1244) modelrow_agrees_1245
private def atomic_1246 : CountSegment 1 10 13120 1245 1246 674 674 := CountSegment.single (first := 1245) (initial := 674) row_count_1246 (by decide)
private theorem atomic_1246_agreement : RowAgreement 1245 1246 := RowAgreement.single (first := 1245) modelrow_agrees_1246
private def atomic_1247 : CountSegment 1 10 13120 1246 1247 674 675 := CountSegment.single (first := 1246) (initial := 674) row_count_1247 (by decide)
private theorem atomic_1247_agreement : RowAgreement 1246 1247 := RowAgreement.single (first := 1246) modelrow_agrees_1247
private def atomic_1248 : CountSegment 1 10 13120 1247 1248 675 677 := CountSegment.single (first := 1247) (initial := 675) row_count_1248 (by decide)
private theorem atomic_1248_agreement : RowAgreement 1247 1248 := RowAgreement.single (first := 1247) modelrow_agrees_1248
private def atomic_1249 : CountSegment 1 10 13120 1248 1249 677 678 := CountSegment.single (first := 1248) (initial := 677) row_count_1249 (by decide)
private theorem atomic_1249_agreement : RowAgreement 1248 1249 := RowAgreement.single (first := 1248) modelrow_agrees_1249
private def atomic_1250 : CountSegment 1 10 13120 1249 1250 678 678 := CountSegment.single (first := 1249) (initial := 678) row_count_1250 (by decide)
private theorem atomic_1250_agreement : RowAgreement 1249 1250 := RowAgreement.single (first := 1249) modelrow_agrees_1250
private def atomic_1251 : CountSegment 1 10 13120 1250 1251 678 678 := CountSegment.single (first := 1250) (initial := 678) row_count_1251 (by decide)
private theorem atomic_1251_agreement : RowAgreement 1250 1251 := RowAgreement.single (first := 1250) modelrow_agrees_1251
private def atomic_1252 : CountSegment 1 10 13120 1251 1252 678 678 := CountSegment.single (first := 1251) (initial := 678) row_count_1252 (by decide)
private theorem atomic_1252_agreement : RowAgreement 1251 1252 := RowAgreement.single (first := 1251) modelrow_agrees_1252
private def atomic_1253 : CountSegment 1 10 13120 1252 1253 678 678 := CountSegment.single (first := 1252) (initial := 678) row_count_1253 (by decide)
private theorem atomic_1253_agreement : RowAgreement 1252 1253 := RowAgreement.single (first := 1252) modelrow_agrees_1253
private def atomic_1254 : CountSegment 1 10 13120 1253 1254 678 678 := CountSegment.single (first := 1253) (initial := 678) row_count_1254 (by decide)
private theorem atomic_1254_agreement : RowAgreement 1253 1254 := RowAgreement.single (first := 1253) modelrow_agrees_1254
private def atomic_1255 : CountSegment 1 10 13120 1254 1255 678 678 := CountSegment.single (first := 1254) (initial := 678) row_count_1255 (by decide)
private theorem atomic_1255_agreement : RowAgreement 1254 1255 := RowAgreement.single (first := 1254) modelrow_agrees_1255
private def atomic_1256 : CountSegment 1 10 13120 1255 1256 678 679 := CountSegment.single (first := 1255) (initial := 678) row_count_1256 (by decide)
private theorem atomic_1256_agreement : RowAgreement 1255 1256 := RowAgreement.single (first := 1255) modelrow_agrees_1256
private def atomic_1257 : CountSegment 1 10 13120 1256 1257 679 680 := CountSegment.single (first := 1256) (initial := 679) row_count_1257 (by decide)
private theorem atomic_1257_agreement : RowAgreement 1256 1257 := RowAgreement.single (first := 1256) modelrow_agrees_1257
private def atomic_1258 : CountSegment 1 10 13120 1257 1258 680 681 := CountSegment.single (first := 1257) (initial := 680) row_count_1258 (by decide)
private theorem atomic_1258_agreement : RowAgreement 1257 1258 := RowAgreement.single (first := 1257) modelrow_agrees_1258
private def atomic_1259 : CountSegment 1 10 13120 1258 1259 681 681 := CountSegment.single (first := 1258) (initial := 681) row_count_1259 (by decide)
private theorem atomic_1259_agreement : RowAgreement 1258 1259 := RowAgreement.single (first := 1258) modelrow_agrees_1259
private def atomic_1260 : CountSegment 1 10 13120 1259 1260 681 681 := CountSegment.single (first := 1259) (initial := 681) row_count_1260 (by decide)
private theorem atomic_1260_agreement : RowAgreement 1259 1260 := RowAgreement.single (first := 1259) modelrow_agrees_1260
private def atomic_1261 : CountSegment 1 10 13120 1260 1261 681 681 := CountSegment.single (first := 1260) (initial := 681) row_count_1261 (by decide)
private theorem atomic_1261_agreement : RowAgreement 1260 1261 := RowAgreement.single (first := 1260) modelrow_agrees_1261
private def atomic_1262 : CountSegment 1 10 13120 1261 1262 681 682 := CountSegment.single (first := 1261) (initial := 681) row_count_1262 (by decide)
private theorem atomic_1262_agreement : RowAgreement 1261 1262 := RowAgreement.single (first := 1261) modelrow_agrees_1262
private def atomic_1263 : CountSegment 1 10 13120 1262 1263 682 682 := CountSegment.single (first := 1262) (initial := 682) row_count_1263 (by decide)
private theorem atomic_1263_agreement : RowAgreement 1262 1263 := RowAgreement.single (first := 1262) modelrow_agrees_1263
private def atomic_1264 : CountSegment 1 10 13120 1263 1264 682 682 := CountSegment.single (first := 1263) (initial := 682) row_count_1264 (by decide)
private theorem atomic_1264_agreement : RowAgreement 1263 1264 := RowAgreement.single (first := 1263) modelrow_agrees_1264
private def atomic_1265 : CountSegment 1 10 13120 1264 1265 682 682 := CountSegment.single (first := 1264) (initial := 682) row_count_1265 (by decide)
private theorem atomic_1265_agreement : RowAgreement 1264 1265 := RowAgreement.single (first := 1264) modelrow_agrees_1265
private def atomic_1266 : CountSegment 1 10 13120 1265 1266 682 682 := CountSegment.single (first := 1265) (initial := 682) row_count_1266 (by decide)
private theorem atomic_1266_agreement : RowAgreement 1265 1266 := RowAgreement.single (first := 1265) modelrow_agrees_1266
private def atomic_1267 : CountSegment 1 10 13120 1266 1267 682 683 := CountSegment.single (first := 1266) (initial := 682) row_count_1267 (by decide)
private theorem atomic_1267_agreement : RowAgreement 1266 1267 := RowAgreement.single (first := 1266) modelrow_agrees_1267
private def atomic_1268 : CountSegment 1 10 13120 1267 1268 683 685 := CountSegment.single (first := 1267) (initial := 683) row_count_1268 (by decide)
private theorem atomic_1268_agreement : RowAgreement 1267 1268 := RowAgreement.single (first := 1267) modelrow_agrees_1268
private def atomic_1269 : CountSegment 1 10 13120 1268 1269 685 685 := CountSegment.single (first := 1268) (initial := 685) row_count_1269 (by decide)
private theorem atomic_1269_agreement : RowAgreement 1268 1269 := RowAgreement.single (first := 1268) modelrow_agrees_1269
private def atomic_1270 : CountSegment 1 10 13120 1269 1270 685 686 := CountSegment.single (first := 1269) (initial := 685) row_count_1270 (by decide)
private theorem atomic_1270_agreement : RowAgreement 1269 1270 := RowAgreement.single (first := 1269) modelrow_agrees_1270
private def atomic_1271 : CountSegment 1 10 13120 1270 1271 686 687 := CountSegment.single (first := 1270) (initial := 686) row_count_1271 (by decide)
private theorem atomic_1271_agreement : RowAgreement 1270 1271 := RowAgreement.single (first := 1270) modelrow_agrees_1271
private def atomic_1272 : CountSegment 1 10 13120 1271 1272 687 687 := CountSegment.single (first := 1271) (initial := 687) row_count_1272 (by decide)
private theorem atomic_1272_agreement : RowAgreement 1271 1272 := RowAgreement.single (first := 1271) modelrow_agrees_1272
private def atomic_1273 : CountSegment 1 10 13120 1272 1273 687 687 := CountSegment.single (first := 1272) (initial := 687) row_count_1273 (by decide)
private theorem atomic_1273_agreement : RowAgreement 1272 1273 := RowAgreement.single (first := 1272) modelrow_agrees_1273
private def atomic_1274 : CountSegment 1 10 13120 1273 1274 687 687 := CountSegment.single (first := 1273) (initial := 687) row_count_1274 (by decide)
private theorem atomic_1274_agreement : RowAgreement 1273 1274 := RowAgreement.single (first := 1273) modelrow_agrees_1274
private def atomic_1275 : CountSegment 1 10 13120 1274 1275 687 688 := CountSegment.single (first := 1274) (initial := 687) row_count_1275 (by decide)
private theorem atomic_1275_agreement : RowAgreement 1274 1275 := RowAgreement.single (first := 1274) modelrow_agrees_1275
private def atomic_1276 : CountSegment 1 10 13120 1275 1276 688 688 := CountSegment.single (first := 1275) (initial := 688) row_count_1276 (by decide)
private theorem atomic_1276_agreement : RowAgreement 1275 1276 := RowAgreement.single (first := 1275) modelrow_agrees_1276
private def atomic_1277 : CountSegment 1 10 13120 1276 1277 688 688 := CountSegment.single (first := 1276) (initial := 688) row_count_1277 (by decide)
private theorem atomic_1277_agreement : RowAgreement 1276 1277 := RowAgreement.single (first := 1276) modelrow_agrees_1277
private def atomic_1278 : CountSegment 1 10 13120 1277 1278 688 690 := CountSegment.single (first := 1277) (initial := 688) row_count_1278 (by decide)
private theorem atomic_1278_agreement : RowAgreement 1277 1278 := RowAgreement.single (first := 1277) modelrow_agrees_1278
private def atomic_1279 : CountSegment 1 10 13120 1278 1279 690 691 := CountSegment.single (first := 1278) (initial := 690) row_count_1279 (by decide)
private theorem atomic_1279_agreement : RowAgreement 1278 1279 := RowAgreement.single (first := 1278) modelrow_agrees_1279
private def atomic_1280 : CountSegment 1 10 13120 1279 1280 691 691 := CountSegment.single (first := 1279) (initial := 691) row_count_1280 (by decide)
private theorem atomic_1280_agreement : RowAgreement 1279 1280 := RowAgreement.single (first := 1279) modelrow_agrees_1280
private def within_009_000 : CountSegment 1 10 13120 1152 1154 626 626 := CountSegment.join atomic_1153 atomic_1154
private theorem within_009_000_agreement : RowAgreement 1152 1154 := RowAgreement.join atomic_1153_agreement atomic_1154_agreement
private def within_009_001 : CountSegment 1 10 13120 1154 1156 626 628 := CountSegment.join atomic_1155 atomic_1156
private theorem within_009_001_agreement : RowAgreement 1154 1156 := RowAgreement.join atomic_1155_agreement atomic_1156_agreement
private def within_009_002 : CountSegment 1 10 13120 1152 1156 626 628 := CountSegment.join within_009_000 within_009_001
private theorem within_009_002_agreement : RowAgreement 1152 1156 := RowAgreement.join within_009_000_agreement within_009_001_agreement
private def within_009_003 : CountSegment 1 10 13120 1156 1158 628 630 := CountSegment.join atomic_1157 atomic_1158
private theorem within_009_003_agreement : RowAgreement 1156 1158 := RowAgreement.join atomic_1157_agreement atomic_1158_agreement
private def within_009_004 : CountSegment 1 10 13120 1158 1160 630 632 := CountSegment.join atomic_1159 atomic_1160
private theorem within_009_004_agreement : RowAgreement 1158 1160 := RowAgreement.join atomic_1159_agreement atomic_1160_agreement
private def within_009_005 : CountSegment 1 10 13120 1156 1160 628 632 := CountSegment.join within_009_003 within_009_004
private theorem within_009_005_agreement : RowAgreement 1156 1160 := RowAgreement.join within_009_003_agreement within_009_004_agreement
private def within_009_006 : CountSegment 1 10 13120 1152 1160 626 632 := CountSegment.join within_009_002 within_009_005
private theorem within_009_006_agreement : RowAgreement 1152 1160 := RowAgreement.join within_009_002_agreement within_009_005_agreement
private def within_009_007 : CountSegment 1 10 13120 1160 1162 632 633 := CountSegment.join atomic_1161 atomic_1162
private theorem within_009_007_agreement : RowAgreement 1160 1162 := RowAgreement.join atomic_1161_agreement atomic_1162_agreement
private def within_009_008 : CountSegment 1 10 13120 1162 1164 633 634 := CountSegment.join atomic_1163 atomic_1164
private theorem within_009_008_agreement : RowAgreement 1162 1164 := RowAgreement.join atomic_1163_agreement atomic_1164_agreement
private def within_009_009 : CountSegment 1 10 13120 1160 1164 632 634 := CountSegment.join within_009_007 within_009_008
private theorem within_009_009_agreement : RowAgreement 1160 1164 := RowAgreement.join within_009_007_agreement within_009_008_agreement
private def within_009_010 : CountSegment 1 10 13120 1164 1166 634 635 := CountSegment.join atomic_1165 atomic_1166
private theorem within_009_010_agreement : RowAgreement 1164 1166 := RowAgreement.join atomic_1165_agreement atomic_1166_agreement
private def within_009_011 : CountSegment 1 10 13120 1166 1168 635 636 := CountSegment.join atomic_1167 atomic_1168
private theorem within_009_011_agreement : RowAgreement 1166 1168 := RowAgreement.join atomic_1167_agreement atomic_1168_agreement
private def within_009_012 : CountSegment 1 10 13120 1164 1168 634 636 := CountSegment.join within_009_010 within_009_011
private theorem within_009_012_agreement : RowAgreement 1164 1168 := RowAgreement.join within_009_010_agreement within_009_011_agreement
private def within_009_013 : CountSegment 1 10 13120 1160 1168 632 636 := CountSegment.join within_009_009 within_009_012
private theorem within_009_013_agreement : RowAgreement 1160 1168 := RowAgreement.join within_009_009_agreement within_009_012_agreement
private def within_009_014 : CountSegment 1 10 13120 1152 1168 626 636 := CountSegment.join within_009_006 within_009_013
private theorem within_009_014_agreement : RowAgreement 1152 1168 := RowAgreement.join within_009_006_agreement within_009_013_agreement
private def within_009_015 : CountSegment 1 10 13120 1168 1170 636 637 := CountSegment.join atomic_1169 atomic_1170
private theorem within_009_015_agreement : RowAgreement 1168 1170 := RowAgreement.join atomic_1169_agreement atomic_1170_agreement
private def within_009_016 : CountSegment 1 10 13120 1170 1172 637 637 := CountSegment.join atomic_1171 atomic_1172
private theorem within_009_016_agreement : RowAgreement 1170 1172 := RowAgreement.join atomic_1171_agreement atomic_1172_agreement
private def within_009_017 : CountSegment 1 10 13120 1168 1172 636 637 := CountSegment.join within_009_015 within_009_016
private theorem within_009_017_agreement : RowAgreement 1168 1172 := RowAgreement.join within_009_015_agreement within_009_016_agreement
private def within_009_018 : CountSegment 1 10 13120 1172 1174 637 638 := CountSegment.join atomic_1173 atomic_1174
private theorem within_009_018_agreement : RowAgreement 1172 1174 := RowAgreement.join atomic_1173_agreement atomic_1174_agreement
private def within_009_019 : CountSegment 1 10 13120 1174 1176 638 638 := CountSegment.join atomic_1175 atomic_1176
private theorem within_009_019_agreement : RowAgreement 1174 1176 := RowAgreement.join atomic_1175_agreement atomic_1176_agreement
private def within_009_020 : CountSegment 1 10 13120 1172 1176 637 638 := CountSegment.join within_009_018 within_009_019
private theorem within_009_020_agreement : RowAgreement 1172 1176 := RowAgreement.join within_009_018_agreement within_009_019_agreement
private def within_009_021 : CountSegment 1 10 13120 1168 1176 636 638 := CountSegment.join within_009_017 within_009_020
private theorem within_009_021_agreement : RowAgreement 1168 1176 := RowAgreement.join within_009_017_agreement within_009_020_agreement
private def within_009_022 : CountSegment 1 10 13120 1176 1178 638 639 := CountSegment.join atomic_1177 atomic_1178
private theorem within_009_022_agreement : RowAgreement 1176 1178 := RowAgreement.join atomic_1177_agreement atomic_1178_agreement
private def within_009_023 : CountSegment 1 10 13120 1178 1180 639 641 := CountSegment.join atomic_1179 atomic_1180
private theorem within_009_023_agreement : RowAgreement 1178 1180 := RowAgreement.join atomic_1179_agreement atomic_1180_agreement
private def within_009_024 : CountSegment 1 10 13120 1176 1180 638 641 := CountSegment.join within_009_022 within_009_023
private theorem within_009_024_agreement : RowAgreement 1176 1180 := RowAgreement.join within_009_022_agreement within_009_023_agreement
private def within_009_025 : CountSegment 1 10 13120 1180 1182 641 642 := CountSegment.join atomic_1181 atomic_1182
private theorem within_009_025_agreement : RowAgreement 1180 1182 := RowAgreement.join atomic_1181_agreement atomic_1182_agreement
private def within_009_026 : CountSegment 1 10 13120 1182 1184 642 642 := CountSegment.join atomic_1183 atomic_1184
private theorem within_009_026_agreement : RowAgreement 1182 1184 := RowAgreement.join atomic_1183_agreement atomic_1184_agreement
private def within_009_027 : CountSegment 1 10 13120 1180 1184 641 642 := CountSegment.join within_009_025 within_009_026
private theorem within_009_027_agreement : RowAgreement 1180 1184 := RowAgreement.join within_009_025_agreement within_009_026_agreement
private def within_009_028 : CountSegment 1 10 13120 1176 1184 638 642 := CountSegment.join within_009_024 within_009_027
private theorem within_009_028_agreement : RowAgreement 1176 1184 := RowAgreement.join within_009_024_agreement within_009_027_agreement
private def within_009_029 : CountSegment 1 10 13120 1168 1184 636 642 := CountSegment.join within_009_021 within_009_028
private theorem within_009_029_agreement : RowAgreement 1168 1184 := RowAgreement.join within_009_021_agreement within_009_028_agreement
private def within_009_030 : CountSegment 1 10 13120 1152 1184 626 642 := CountSegment.join within_009_014 within_009_029
private theorem within_009_030_agreement : RowAgreement 1152 1184 := RowAgreement.join within_009_014_agreement within_009_029_agreement
private def within_009_031 : CountSegment 1 10 13120 1184 1186 642 643 := CountSegment.join atomic_1185 atomic_1186
private theorem within_009_031_agreement : RowAgreement 1184 1186 := RowAgreement.join atomic_1185_agreement atomic_1186_agreement
private def within_009_032 : CountSegment 1 10 13120 1186 1188 643 644 := CountSegment.join atomic_1187 atomic_1188
private theorem within_009_032_agreement : RowAgreement 1186 1188 := RowAgreement.join atomic_1187_agreement atomic_1188_agreement
private def within_009_033 : CountSegment 1 10 13120 1184 1188 642 644 := CountSegment.join within_009_031 within_009_032
private theorem within_009_033_agreement : RowAgreement 1184 1188 := RowAgreement.join within_009_031_agreement within_009_032_agreement
private def within_009_034 : CountSegment 1 10 13120 1188 1190 644 645 := CountSegment.join atomic_1189 atomic_1190
private theorem within_009_034_agreement : RowAgreement 1188 1190 := RowAgreement.join atomic_1189_agreement atomic_1190_agreement
private def within_009_035 : CountSegment 1 10 13120 1190 1192 645 645 := CountSegment.join atomic_1191 atomic_1192
private theorem within_009_035_agreement : RowAgreement 1190 1192 := RowAgreement.join atomic_1191_agreement atomic_1192_agreement
private def within_009_036 : CountSegment 1 10 13120 1188 1192 644 645 := CountSegment.join within_009_034 within_009_035
private theorem within_009_036_agreement : RowAgreement 1188 1192 := RowAgreement.join within_009_034_agreement within_009_035_agreement
private def within_009_037 : CountSegment 1 10 13120 1184 1192 642 645 := CountSegment.join within_009_033 within_009_036
private theorem within_009_037_agreement : RowAgreement 1184 1192 := RowAgreement.join within_009_033_agreement within_009_036_agreement
private def within_009_038 : CountSegment 1 10 13120 1192 1194 645 647 := CountSegment.join atomic_1193 atomic_1194
private theorem within_009_038_agreement : RowAgreement 1192 1194 := RowAgreement.join atomic_1193_agreement atomic_1194_agreement
private def within_009_039 : CountSegment 1 10 13120 1194 1196 647 648 := CountSegment.join atomic_1195 atomic_1196
private theorem within_009_039_agreement : RowAgreement 1194 1196 := RowAgreement.join atomic_1195_agreement atomic_1196_agreement
private def within_009_040 : CountSegment 1 10 13120 1192 1196 645 648 := CountSegment.join within_009_038 within_009_039
private theorem within_009_040_agreement : RowAgreement 1192 1196 := RowAgreement.join within_009_038_agreement within_009_039_agreement
private def within_009_041 : CountSegment 1 10 13120 1196 1198 648 648 := CountSegment.join atomic_1197 atomic_1198
private theorem within_009_041_agreement : RowAgreement 1196 1198 := RowAgreement.join atomic_1197_agreement atomic_1198_agreement
private def within_009_042 : CountSegment 1 10 13120 1198 1200 648 649 := CountSegment.join atomic_1199 atomic_1200
private theorem within_009_042_agreement : RowAgreement 1198 1200 := RowAgreement.join atomic_1199_agreement atomic_1200_agreement
private def within_009_043 : CountSegment 1 10 13120 1196 1200 648 649 := CountSegment.join within_009_041 within_009_042
private theorem within_009_043_agreement : RowAgreement 1196 1200 := RowAgreement.join within_009_041_agreement within_009_042_agreement
private def within_009_044 : CountSegment 1 10 13120 1192 1200 645 649 := CountSegment.join within_009_040 within_009_043
private theorem within_009_044_agreement : RowAgreement 1192 1200 := RowAgreement.join within_009_040_agreement within_009_043_agreement
private def within_009_045 : CountSegment 1 10 13120 1184 1200 642 649 := CountSegment.join within_009_037 within_009_044
private theorem within_009_045_agreement : RowAgreement 1184 1200 := RowAgreement.join within_009_037_agreement within_009_044_agreement
private def within_009_046 : CountSegment 1 10 13120 1200 1202 649 651 := CountSegment.join atomic_1201 atomic_1202
private theorem within_009_046_agreement : RowAgreement 1200 1202 := RowAgreement.join atomic_1201_agreement atomic_1202_agreement
private def within_009_047 : CountSegment 1 10 13120 1202 1204 651 652 := CountSegment.join atomic_1203 atomic_1204
private theorem within_009_047_agreement : RowAgreement 1202 1204 := RowAgreement.join atomic_1203_agreement atomic_1204_agreement
private def within_009_048 : CountSegment 1 10 13120 1200 1204 649 652 := CountSegment.join within_009_046 within_009_047
private theorem within_009_048_agreement : RowAgreement 1200 1204 := RowAgreement.join within_009_046_agreement within_009_047_agreement
private def within_009_049 : CountSegment 1 10 13120 1204 1206 652 654 := CountSegment.join atomic_1205 atomic_1206
private theorem within_009_049_agreement : RowAgreement 1204 1206 := RowAgreement.join atomic_1205_agreement atomic_1206_agreement
private def within_009_050 : CountSegment 1 10 13120 1206 1208 654 654 := CountSegment.join atomic_1207 atomic_1208
private theorem within_009_050_agreement : RowAgreement 1206 1208 := RowAgreement.join atomic_1207_agreement atomic_1208_agreement
private def within_009_051 : CountSegment 1 10 13120 1204 1208 652 654 := CountSegment.join within_009_049 within_009_050
private theorem within_009_051_agreement : RowAgreement 1204 1208 := RowAgreement.join within_009_049_agreement within_009_050_agreement
private def within_009_052 : CountSegment 1 10 13120 1200 1208 649 654 := CountSegment.join within_009_048 within_009_051
private theorem within_009_052_agreement : RowAgreement 1200 1208 := RowAgreement.join within_009_048_agreement within_009_051_agreement
private def within_009_053 : CountSegment 1 10 13120 1208 1210 654 655 := CountSegment.join atomic_1209 atomic_1210
private theorem within_009_053_agreement : RowAgreement 1208 1210 := RowAgreement.join atomic_1209_agreement atomic_1210_agreement
private def within_009_054 : CountSegment 1 10 13120 1210 1212 655 655 := CountSegment.join atomic_1211 atomic_1212
private theorem within_009_054_agreement : RowAgreement 1210 1212 := RowAgreement.join atomic_1211_agreement atomic_1212_agreement
private def within_009_055 : CountSegment 1 10 13120 1208 1212 654 655 := CountSegment.join within_009_053 within_009_054
private theorem within_009_055_agreement : RowAgreement 1208 1212 := RowAgreement.join within_009_053_agreement within_009_054_agreement
private def within_009_056 : CountSegment 1 10 13120 1212 1214 655 655 := CountSegment.join atomic_1213 atomic_1214
private theorem within_009_056_agreement : RowAgreement 1212 1214 := RowAgreement.join atomic_1213_agreement atomic_1214_agreement
private def within_009_057 : CountSegment 1 10 13120 1214 1216 655 658 := CountSegment.join atomic_1215 atomic_1216
private theorem within_009_057_agreement : RowAgreement 1214 1216 := RowAgreement.join atomic_1215_agreement atomic_1216_agreement
private def within_009_058 : CountSegment 1 10 13120 1212 1216 655 658 := CountSegment.join within_009_056 within_009_057
private theorem within_009_058_agreement : RowAgreement 1212 1216 := RowAgreement.join within_009_056_agreement within_009_057_agreement
private def within_009_059 : CountSegment 1 10 13120 1208 1216 654 658 := CountSegment.join within_009_055 within_009_058
private theorem within_009_059_agreement : RowAgreement 1208 1216 := RowAgreement.join within_009_055_agreement within_009_058_agreement
private def within_009_060 : CountSegment 1 10 13120 1200 1216 649 658 := CountSegment.join within_009_052 within_009_059
private theorem within_009_060_agreement : RowAgreement 1200 1216 := RowAgreement.join within_009_052_agreement within_009_059_agreement
private def within_009_061 : CountSegment 1 10 13120 1184 1216 642 658 := CountSegment.join within_009_045 within_009_060
private theorem within_009_061_agreement : RowAgreement 1184 1216 := RowAgreement.join within_009_045_agreement within_009_060_agreement
private def within_009_062 : CountSegment 1 10 13120 1152 1216 626 658 := CountSegment.join within_009_030 within_009_061
private theorem within_009_062_agreement : RowAgreement 1152 1216 := RowAgreement.join within_009_030_agreement within_009_061_agreement
private def within_009_063 : CountSegment 1 10 13120 1216 1218 658 659 := CountSegment.join atomic_1217 atomic_1218
private theorem within_009_063_agreement : RowAgreement 1216 1218 := RowAgreement.join atomic_1217_agreement atomic_1218_agreement
private def within_009_064 : CountSegment 1 10 13120 1218 1220 659 659 := CountSegment.join atomic_1219 atomic_1220
private theorem within_009_064_agreement : RowAgreement 1218 1220 := RowAgreement.join atomic_1219_agreement atomic_1220_agreement
private def within_009_065 : CountSegment 1 10 13120 1216 1220 658 659 := CountSegment.join within_009_063 within_009_064
private theorem within_009_065_agreement : RowAgreement 1216 1220 := RowAgreement.join within_009_063_agreement within_009_064_agreement
private def within_009_066 : CountSegment 1 10 13120 1220 1222 659 660 := CountSegment.join atomic_1221 atomic_1222
private theorem within_009_066_agreement : RowAgreement 1220 1222 := RowAgreement.join atomic_1221_agreement atomic_1222_agreement
private def within_009_067 : CountSegment 1 10 13120 1222 1224 660 660 := CountSegment.join atomic_1223 atomic_1224
private theorem within_009_067_agreement : RowAgreement 1222 1224 := RowAgreement.join atomic_1223_agreement atomic_1224_agreement
private def within_009_068 : CountSegment 1 10 13120 1220 1224 659 660 := CountSegment.join within_009_066 within_009_067
private theorem within_009_068_agreement : RowAgreement 1220 1224 := RowAgreement.join within_009_066_agreement within_009_067_agreement
private def within_009_069 : CountSegment 1 10 13120 1216 1224 658 660 := CountSegment.join within_009_065 within_009_068
private theorem within_009_069_agreement : RowAgreement 1216 1224 := RowAgreement.join within_009_065_agreement within_009_068_agreement
private def within_009_070 : CountSegment 1 10 13120 1224 1226 660 662 := CountSegment.join atomic_1225 atomic_1226
private theorem within_009_070_agreement : RowAgreement 1224 1226 := RowAgreement.join atomic_1225_agreement atomic_1226_agreement
private def within_009_071 : CountSegment 1 10 13120 1226 1228 662 665 := CountSegment.join atomic_1227 atomic_1228
private theorem within_009_071_agreement : RowAgreement 1226 1228 := RowAgreement.join atomic_1227_agreement atomic_1228_agreement
private def within_009_072 : CountSegment 1 10 13120 1224 1228 660 665 := CountSegment.join within_009_070 within_009_071
private theorem within_009_072_agreement : RowAgreement 1224 1228 := RowAgreement.join within_009_070_agreement within_009_071_agreement
private def within_009_073 : CountSegment 1 10 13120 1228 1230 665 666 := CountSegment.join atomic_1229 atomic_1230
private theorem within_009_073_agreement : RowAgreement 1228 1230 := RowAgreement.join atomic_1229_agreement atomic_1230_agreement
private def within_009_074 : CountSegment 1 10 13120 1230 1232 666 668 := CountSegment.join atomic_1231 atomic_1232
private theorem within_009_074_agreement : RowAgreement 1230 1232 := RowAgreement.join atomic_1231_agreement atomic_1232_agreement
private def within_009_075 : CountSegment 1 10 13120 1228 1232 665 668 := CountSegment.join within_009_073 within_009_074
private theorem within_009_075_agreement : RowAgreement 1228 1232 := RowAgreement.join within_009_073_agreement within_009_074_agreement
private def within_009_076 : CountSegment 1 10 13120 1224 1232 660 668 := CountSegment.join within_009_072 within_009_075
private theorem within_009_076_agreement : RowAgreement 1224 1232 := RowAgreement.join within_009_072_agreement within_009_075_agreement
private def within_009_077 : CountSegment 1 10 13120 1216 1232 658 668 := CountSegment.join within_009_069 within_009_076
private theorem within_009_077_agreement : RowAgreement 1216 1232 := RowAgreement.join within_009_069_agreement within_009_076_agreement
private def within_009_078 : CountSegment 1 10 13120 1232 1234 668 669 := CountSegment.join atomic_1233 atomic_1234
private theorem within_009_078_agreement : RowAgreement 1232 1234 := RowAgreement.join atomic_1233_agreement atomic_1234_agreement
private def within_009_079 : CountSegment 1 10 13120 1234 1236 669 671 := CountSegment.join atomic_1235 atomic_1236
private theorem within_009_079_agreement : RowAgreement 1234 1236 := RowAgreement.join atomic_1235_agreement atomic_1236_agreement
private def within_009_080 : CountSegment 1 10 13120 1232 1236 668 671 := CountSegment.join within_009_078 within_009_079
private theorem within_009_080_agreement : RowAgreement 1232 1236 := RowAgreement.join within_009_078_agreement within_009_079_agreement
private def within_009_081 : CountSegment 1 10 13120 1236 1238 671 671 := CountSegment.join atomic_1237 atomic_1238
private theorem within_009_081_agreement : RowAgreement 1236 1238 := RowAgreement.join atomic_1237_agreement atomic_1238_agreement
private def within_009_082 : CountSegment 1 10 13120 1238 1240 671 672 := CountSegment.join atomic_1239 atomic_1240
private theorem within_009_082_agreement : RowAgreement 1238 1240 := RowAgreement.join atomic_1239_agreement atomic_1240_agreement
private def within_009_083 : CountSegment 1 10 13120 1236 1240 671 672 := CountSegment.join within_009_081 within_009_082
private theorem within_009_083_agreement : RowAgreement 1236 1240 := RowAgreement.join within_009_081_agreement within_009_082_agreement
private def within_009_084 : CountSegment 1 10 13120 1232 1240 668 672 := CountSegment.join within_009_080 within_009_083
private theorem within_009_084_agreement : RowAgreement 1232 1240 := RowAgreement.join within_009_080_agreement within_009_083_agreement
private def within_009_085 : CountSegment 1 10 13120 1240 1242 672 674 := CountSegment.join atomic_1241 atomic_1242
private theorem within_009_085_agreement : RowAgreement 1240 1242 := RowAgreement.join atomic_1241_agreement atomic_1242_agreement
private def within_009_086 : CountSegment 1 10 13120 1242 1244 674 674 := CountSegment.join atomic_1243 atomic_1244
private theorem within_009_086_agreement : RowAgreement 1242 1244 := RowAgreement.join atomic_1243_agreement atomic_1244_agreement
private def within_009_087 : CountSegment 1 10 13120 1240 1244 672 674 := CountSegment.join within_009_085 within_009_086
private theorem within_009_087_agreement : RowAgreement 1240 1244 := RowAgreement.join within_009_085_agreement within_009_086_agreement
private def within_009_088 : CountSegment 1 10 13120 1244 1246 674 674 := CountSegment.join atomic_1245 atomic_1246
private theorem within_009_088_agreement : RowAgreement 1244 1246 := RowAgreement.join atomic_1245_agreement atomic_1246_agreement
private def within_009_089 : CountSegment 1 10 13120 1246 1248 674 677 := CountSegment.join atomic_1247 atomic_1248
private theorem within_009_089_agreement : RowAgreement 1246 1248 := RowAgreement.join atomic_1247_agreement atomic_1248_agreement
private def within_009_090 : CountSegment 1 10 13120 1244 1248 674 677 := CountSegment.join within_009_088 within_009_089
private theorem within_009_090_agreement : RowAgreement 1244 1248 := RowAgreement.join within_009_088_agreement within_009_089_agreement
private def within_009_091 : CountSegment 1 10 13120 1240 1248 672 677 := CountSegment.join within_009_087 within_009_090
private theorem within_009_091_agreement : RowAgreement 1240 1248 := RowAgreement.join within_009_087_agreement within_009_090_agreement
private def within_009_092 : CountSegment 1 10 13120 1232 1248 668 677 := CountSegment.join within_009_084 within_009_091
private theorem within_009_092_agreement : RowAgreement 1232 1248 := RowAgreement.join within_009_084_agreement within_009_091_agreement
private def within_009_093 : CountSegment 1 10 13120 1216 1248 658 677 := CountSegment.join within_009_077 within_009_092
private theorem within_009_093_agreement : RowAgreement 1216 1248 := RowAgreement.join within_009_077_agreement within_009_092_agreement
private def within_009_094 : CountSegment 1 10 13120 1248 1250 677 678 := CountSegment.join atomic_1249 atomic_1250
private theorem within_009_094_agreement : RowAgreement 1248 1250 := RowAgreement.join atomic_1249_agreement atomic_1250_agreement
private def within_009_095 : CountSegment 1 10 13120 1250 1252 678 678 := CountSegment.join atomic_1251 atomic_1252
private theorem within_009_095_agreement : RowAgreement 1250 1252 := RowAgreement.join atomic_1251_agreement atomic_1252_agreement
private def within_009_096 : CountSegment 1 10 13120 1248 1252 677 678 := CountSegment.join within_009_094 within_009_095
private theorem within_009_096_agreement : RowAgreement 1248 1252 := RowAgreement.join within_009_094_agreement within_009_095_agreement
private def within_009_097 : CountSegment 1 10 13120 1252 1254 678 678 := CountSegment.join atomic_1253 atomic_1254
private theorem within_009_097_agreement : RowAgreement 1252 1254 := RowAgreement.join atomic_1253_agreement atomic_1254_agreement
private def within_009_098 : CountSegment 1 10 13120 1254 1256 678 679 := CountSegment.join atomic_1255 atomic_1256
private theorem within_009_098_agreement : RowAgreement 1254 1256 := RowAgreement.join atomic_1255_agreement atomic_1256_agreement
private def within_009_099 : CountSegment 1 10 13120 1252 1256 678 679 := CountSegment.join within_009_097 within_009_098
private theorem within_009_099_agreement : RowAgreement 1252 1256 := RowAgreement.join within_009_097_agreement within_009_098_agreement
private def within_009_100 : CountSegment 1 10 13120 1248 1256 677 679 := CountSegment.join within_009_096 within_009_099
private theorem within_009_100_agreement : RowAgreement 1248 1256 := RowAgreement.join within_009_096_agreement within_009_099_agreement
private def within_009_101 : CountSegment 1 10 13120 1256 1258 679 681 := CountSegment.join atomic_1257 atomic_1258
private theorem within_009_101_agreement : RowAgreement 1256 1258 := RowAgreement.join atomic_1257_agreement atomic_1258_agreement
private def within_009_102 : CountSegment 1 10 13120 1258 1260 681 681 := CountSegment.join atomic_1259 atomic_1260
private theorem within_009_102_agreement : RowAgreement 1258 1260 := RowAgreement.join atomic_1259_agreement atomic_1260_agreement
private def within_009_103 : CountSegment 1 10 13120 1256 1260 679 681 := CountSegment.join within_009_101 within_009_102
private theorem within_009_103_agreement : RowAgreement 1256 1260 := RowAgreement.join within_009_101_agreement within_009_102_agreement
private def within_009_104 : CountSegment 1 10 13120 1260 1262 681 682 := CountSegment.join atomic_1261 atomic_1262
private theorem within_009_104_agreement : RowAgreement 1260 1262 := RowAgreement.join atomic_1261_agreement atomic_1262_agreement
private def within_009_105 : CountSegment 1 10 13120 1262 1264 682 682 := CountSegment.join atomic_1263 atomic_1264
private theorem within_009_105_agreement : RowAgreement 1262 1264 := RowAgreement.join atomic_1263_agreement atomic_1264_agreement
private def within_009_106 : CountSegment 1 10 13120 1260 1264 681 682 := CountSegment.join within_009_104 within_009_105
private theorem within_009_106_agreement : RowAgreement 1260 1264 := RowAgreement.join within_009_104_agreement within_009_105_agreement
private def within_009_107 : CountSegment 1 10 13120 1256 1264 679 682 := CountSegment.join within_009_103 within_009_106
private theorem within_009_107_agreement : RowAgreement 1256 1264 := RowAgreement.join within_009_103_agreement within_009_106_agreement
private def within_009_108 : CountSegment 1 10 13120 1248 1264 677 682 := CountSegment.join within_009_100 within_009_107
private theorem within_009_108_agreement : RowAgreement 1248 1264 := RowAgreement.join within_009_100_agreement within_009_107_agreement
private def within_009_109 : CountSegment 1 10 13120 1264 1266 682 682 := CountSegment.join atomic_1265 atomic_1266
private theorem within_009_109_agreement : RowAgreement 1264 1266 := RowAgreement.join atomic_1265_agreement atomic_1266_agreement
private def within_009_110 : CountSegment 1 10 13120 1266 1268 682 685 := CountSegment.join atomic_1267 atomic_1268
private theorem within_009_110_agreement : RowAgreement 1266 1268 := RowAgreement.join atomic_1267_agreement atomic_1268_agreement
private def within_009_111 : CountSegment 1 10 13120 1264 1268 682 685 := CountSegment.join within_009_109 within_009_110
private theorem within_009_111_agreement : RowAgreement 1264 1268 := RowAgreement.join within_009_109_agreement within_009_110_agreement
private def within_009_112 : CountSegment 1 10 13120 1268 1270 685 686 := CountSegment.join atomic_1269 atomic_1270
private theorem within_009_112_agreement : RowAgreement 1268 1270 := RowAgreement.join atomic_1269_agreement atomic_1270_agreement
private def within_009_113 : CountSegment 1 10 13120 1270 1272 686 687 := CountSegment.join atomic_1271 atomic_1272
private theorem within_009_113_agreement : RowAgreement 1270 1272 := RowAgreement.join atomic_1271_agreement atomic_1272_agreement
private def within_009_114 : CountSegment 1 10 13120 1268 1272 685 687 := CountSegment.join within_009_112 within_009_113
private theorem within_009_114_agreement : RowAgreement 1268 1272 := RowAgreement.join within_009_112_agreement within_009_113_agreement
private def within_009_115 : CountSegment 1 10 13120 1264 1272 682 687 := CountSegment.join within_009_111 within_009_114
private theorem within_009_115_agreement : RowAgreement 1264 1272 := RowAgreement.join within_009_111_agreement within_009_114_agreement
private def within_009_116 : CountSegment 1 10 13120 1272 1274 687 687 := CountSegment.join atomic_1273 atomic_1274
private theorem within_009_116_agreement : RowAgreement 1272 1274 := RowAgreement.join atomic_1273_agreement atomic_1274_agreement
private def within_009_117 : CountSegment 1 10 13120 1274 1276 687 688 := CountSegment.join atomic_1275 atomic_1276
private theorem within_009_117_agreement : RowAgreement 1274 1276 := RowAgreement.join atomic_1275_agreement atomic_1276_agreement
private def within_009_118 : CountSegment 1 10 13120 1272 1276 687 688 := CountSegment.join within_009_116 within_009_117
private theorem within_009_118_agreement : RowAgreement 1272 1276 := RowAgreement.join within_009_116_agreement within_009_117_agreement
private def within_009_119 : CountSegment 1 10 13120 1276 1278 688 690 := CountSegment.join atomic_1277 atomic_1278
private theorem within_009_119_agreement : RowAgreement 1276 1278 := RowAgreement.join atomic_1277_agreement atomic_1278_agreement
private def within_009_120 : CountSegment 1 10 13120 1278 1280 690 691 := CountSegment.join atomic_1279 atomic_1280
private theorem within_009_120_agreement : RowAgreement 1278 1280 := RowAgreement.join atomic_1279_agreement atomic_1280_agreement
private def within_009_121 : CountSegment 1 10 13120 1276 1280 688 691 := CountSegment.join within_009_119 within_009_120
private theorem within_009_121_agreement : RowAgreement 1276 1280 := RowAgreement.join within_009_119_agreement within_009_120_agreement
private def within_009_122 : CountSegment 1 10 13120 1272 1280 687 691 := CountSegment.join within_009_118 within_009_121
private theorem within_009_122_agreement : RowAgreement 1272 1280 := RowAgreement.join within_009_118_agreement within_009_121_agreement
private def within_009_123 : CountSegment 1 10 13120 1264 1280 682 691 := CountSegment.join within_009_115 within_009_122
private theorem within_009_123_agreement : RowAgreement 1264 1280 := RowAgreement.join within_009_115_agreement within_009_122_agreement
private def within_009_124 : CountSegment 1 10 13120 1248 1280 677 691 := CountSegment.join within_009_108 within_009_123
private theorem within_009_124_agreement : RowAgreement 1248 1280 := RowAgreement.join within_009_108_agreement within_009_123_agreement
private def within_009_125 : CountSegment 1 10 13120 1216 1280 658 691 := CountSegment.join within_009_093 within_009_124
private theorem within_009_125_agreement : RowAgreement 1216 1280 := RowAgreement.join within_009_093_agreement within_009_124_agreement
private def within_009_126 : CountSegment 1 10 13120 1152 1280 626 691 := CountSegment.join within_009_062 within_009_125
private theorem within_009_126_agreement : RowAgreement 1152 1280 := RowAgreement.join within_009_062_agreement within_009_125_agreement
def segment_009 : CountSegment 1 10 13120 1152 1280 626 691 := within_009_126
theorem agreement_009 : RowAgreement 1152 1280 := within_009_126_agreement
end ProjectEulerSolutions.P58Numerical
