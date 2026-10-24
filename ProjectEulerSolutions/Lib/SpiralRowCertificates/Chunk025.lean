import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3201 : RowCertificate 3201 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3201 1) := by change ¬ Nat.Prime 40992007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5856001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3201 2) := by change ¬ Nat.Prime 40985605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8197121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3201 3) := by change ¬ Nat.Prime 40979203; exact PrattCertificates.composite_of_product (factor := 5209) (cofactor := 7867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3201 : row 3201 = 0 := (row_3201.checked).trans rfl
theorem model_count_3201 : modelRow 3201 = 0 := by decide
theorem modelrow_agrees_3201 : modelRow 3201 = row 3201 := model_count_3201.trans row_count_3201.symm
def row_3202 : RowCertificate 3202 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3202 1) := by change ¬ Nat.Prime 41017621; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 117529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3202 2) := by change ¬ Nat.Prime 41011217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3154709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3202 3) := by change ¬ Nat.Prime 41004813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13668271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3202 : row 3202 = 0 := (row_3202.checked).trans rfl
theorem model_count_3202 : modelRow 3202 = 0 := by decide
theorem modelrow_agrees_3202 : modelRow 3202 = row 3202 := model_count_3202.trans row_count_3202.symm
def row_3203 : RowCertificate 3203 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3203 1) := by change ¬ Nat.Prime 41043243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13681081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3203 2) := by change ¬ Nat.Prime 41036837; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 89017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3203 3) := by change ¬ Nat.Prime 41030431; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3156187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3203 : row 3203 = 0 := (row_3203.checked).trans rfl
theorem model_count_3203 : modelRow 3203 = 0 := by decide
theorem modelrow_agrees_3203 : modelRow 3203 = row 3203 := model_count_3203.trans row_count_3203.symm
def row_3204 : RowCertificate 3204 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3204 1) := by change Nat.Prime 41068873; exact ProjectEulerSolutions.P58PrimeNodes.p_41068873.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3204 2) := by change ¬ Nat.Prime 41062465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8212493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3204 3) := by change ¬ Nat.Prime 41056057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5865151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3204 : row 3204 = 1 := (row_3204.checked).trans rfl
theorem model_count_3204 : modelRow 3204 = 1 := by decide
theorem modelrow_agrees_3204 : modelRow 3204 = row 3204 := model_count_3204.trans row_count_3204.symm
def row_3205 : RowCertificate 3205 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3205 1) := by change ¬ Nat.Prime 41094511; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2162869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3205 2) := by change ¬ Nat.Prime 41088101; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 146221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3205 3) := by change ¬ Nat.Prime 41081691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13693897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3205 : row 3205 = 0 := (row_3205.checked).trans rfl
theorem model_count_3205 : modelRow 3205 = 0 := by decide
theorem modelrow_agrees_3205 : modelRow 3205 = row 3205 := model_count_3205.trans row_count_3205.symm
def row_3206 : RowCertificate 3206 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3206 1) := by change ¬ Nat.Prime 41120157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13706719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3206 2) := by change ¬ Nat.Prime 41113745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8222749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3206 3) := by change ¬ Nat.Prime 41107333; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1326043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3206 : row 3206 = 0 := (row_3206.checked).trans rfl
theorem model_count_3206 : modelRow 3206 = 0 := by decide
theorem modelrow_agrees_3206 : modelRow 3206 = row 3206 := model_count_3206.trans row_count_3206.symm
def row_3207 : RowCertificate 3207 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3207 1) := by change ¬ Nat.Prime 41145811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5877973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3207 2) := by change ¬ Nat.Prime 41139397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3164569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3207 3) := by change ¬ Nat.Prime 41132983; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 956581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3207 : row 3207 = 0 := (row_3207.checked).trans rfl
theorem model_count_3207 : modelRow 3207 = 0 := by decide
theorem modelrow_agrees_3207 : modelRow 3207 = row 3207 := model_count_3207.trans row_count_3207.symm
def row_3208 : RowCertificate 3208 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3208 1) := by change ¬ Nat.Prime 41171473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5881639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3208 2) := by change ¬ Nat.Prime 41165057; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 674837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3208 3) := by change ¬ Nat.Prime 41158641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13719547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3208 : row 3208 = 0 := (row_3208.checked).trans rfl
theorem model_count_3208 : modelRow 3208 = 0 := by decide
theorem modelrow_agrees_3208 : modelRow 3208 = row 3208 := model_count_3208.trans row_count_3208.symm
def row_3209 : RowCertificate 3209 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3209 1) := by change ¬ Nat.Prime 41197143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13732381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3209 2) := by change ¬ Nat.Prime 41190725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8238145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3209 3) := by change Nat.Prime 41184307; exact ProjectEulerSolutions.P58PrimeNodes.p_41184307.prime
    exact decide_eq_true prime
theorem row_count_3209 : row 3209 = 1 := (row_3209.checked).trans rfl
theorem model_count_3209 : modelRow 3209 = 1 := by decide
theorem modelrow_agrees_3209 : modelRow 3209 = row 3209 := model_count_3209.trans row_count_3209.symm
def row_3210 : RowCertificate 3210 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3210 1) := by change Nat.Prime 41222821; exact ProjectEulerSolutions.P58PrimeNodes.p_41222821.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3210 2) := by change Nat.Prime 41216401; exact ProjectEulerSolutions.P58PrimeNodes.p_41216401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3210 3) := by change ¬ Nat.Prime 41209981; exact PrattCertificates.composite_of_product (factor := 1303) (cofactor := 31627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3210 : row 3210 = 2 := (row_3210.checked).trans rfl
theorem model_count_3210 : modelRow 3210 = 2 := by decide
theorem modelrow_agrees_3210 : modelRow 3210 = row 3210 := model_count_3210.trans row_count_3210.symm
def row_3211 : RowCertificate 3211 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3211 1) := by change ¬ Nat.Prime 41248507; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1330597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3211 2) := by change ¬ Nat.Prime 41242085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8248417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3211 3) := by change ¬ Nat.Prime 41235663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13745221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3211 : row 3211 = 0 := (row_3211.checked).trans rfl
theorem model_count_3211 : modelRow 3211 = 0 := by decide
theorem modelrow_agrees_3211 : modelRow 3211 = row 3211 := model_count_3211.trans row_count_3211.symm
def row_3212 : RowCertificate 3212 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3212 1) := by change ¬ Nat.Prime 41274201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13758067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3212 2) := by change ¬ Nat.Prime 41267777; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 425441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3212 3) := by change ¬ Nat.Prime 41261353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5894479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3212 : row 3212 = 0 := (row_3212.checked).trans rfl
theorem model_count_3212 : modelRow 3212 = 0 := by decide
theorem modelrow_agrees_3212 : modelRow 3212 = row 3212 := model_count_3212.trans row_count_3212.symm
def row_3213 : RowCertificate 3213 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3213 1) := by change Nat.Prime 41299903; exact ProjectEulerSolutions.P58PrimeNodes.p_41299903.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3213 2) := by change ¬ Nat.Prime 41293477; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1423913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3213 3) := by change ¬ Nat.Prime 41287051; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3175927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3213 : row 3213 = 1 := (row_3213.checked).trans rfl
theorem model_count_3213 : modelRow 3213 = 1 := by decide
theorem modelrow_agrees_3213 : modelRow 3213 = row 3213 := model_count_3213.trans row_count_3213.symm
def row_3214 : RowCertificate 3214 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3214 1) := by change ¬ Nat.Prime 41325613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5903659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3214 2) := by change ¬ Nat.Prime 41319185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8263837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3214 3) := by change ¬ Nat.Prime 41312757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13770919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3214 : row 3214 = 0 := (row_3214.checked).trans rfl
theorem model_count_3214 : modelRow 3214 = 0 := by decide
theorem modelrow_agrees_3214 : modelRow 3214 = row 3214 := model_count_3214.trans row_count_3214.symm
def row_3215 : RowCertificate 3215 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3215 1) := by change ¬ Nat.Prime 41351331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13783777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3215 2) := by change ¬ Nat.Prime 41344901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3180377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3215 3) := by change ¬ Nat.Prime 41338471; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2175709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3215 : row 3215 = 0 := (row_3215.checked).trans rfl
theorem model_count_3215 : modelRow 3215 = 0 := by decide
theorem modelrow_agrees_3215 : modelRow 3215 = row 3215 := model_count_3215.trans row_count_3215.symm
def row_3216 : RowCertificate 3216 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3216 1) := by change ¬ Nat.Prime 41377057; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 566809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3216 2) := by change ¬ Nat.Prime 41370625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8274125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3216 3) := by change ¬ Nat.Prime 41364193; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3181861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3216 : row 3216 = 0 := (row_3216.checked).trans rfl
theorem model_count_3216 : modelRow 3216 = 0 := by decide
theorem modelrow_agrees_3216 : modelRow 3216 = row 3216 := model_count_3216.trans row_count_3216.symm
def row_3217 : RowCertificate 3217 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3217 1) := by change Nat.Prime 41402791; exact ProjectEulerSolutions.P58PrimeNodes.p_41402791.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3217 2) := by change Nat.Prime 41396357; exact ProjectEulerSolutions.P58PrimeNodes.p_41396357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3217 3) := by change ¬ Nat.Prime 41389923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13796641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3217 : row 3217 = 2 := (row_3217.checked).trans rfl
theorem model_count_3217 : modelRow 3217 = 2 := by decide
theorem modelrow_agrees_3217 : modelRow 3217 = row 3217 := model_count_3217.trans row_count_3217.symm
def row_3218 : RowCertificate 3218 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3218 1) := by change ¬ Nat.Prime 41428533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13809511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3218 2) := by change ¬ Nat.Prime 41422097; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 781549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3218 3) := by change ¬ Nat.Prime 41415661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5916523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3218 : row 3218 = 0 := (row_3218.checked).trans rfl
theorem model_count_3218 : modelRow 3218 = 0 := by decide
theorem modelrow_agrees_3218 : modelRow 3218 = row 3218 := model_count_3218.trans row_count_3218.symm
def row_3219 : RowCertificate 3219 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3219 1) := by change ¬ Nat.Prime 41454283; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3188791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3219 2) := by change ¬ Nat.Prime 41447845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8289569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3219 3) := by change ¬ Nat.Prime 41441407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5920201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3219 : row 3219 = 0 := (row_3219.checked).trans rfl
theorem model_count_3219 : modelRow 3219 = 0 := by decide
theorem modelrow_agrees_3219 : modelRow 3219 = row 3219 := model_count_3219.trans row_count_3219.symm
def row_3220 : RowCertificate 3220 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3220 1) := by change Nat.Prime 41480041; exact ProjectEulerSolutions.P58PrimeNodes.p_41480041.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3220 2) := by change ¬ Nat.Prime 41473601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3190277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3220 3) := by change ¬ Nat.Prime 41467161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13822387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3220 : row 3220 = 1 := (row_3220.checked).trans rfl
theorem model_count_3220 : modelRow 3220 = 1 := by decide
theorem modelrow_agrees_3220 : modelRow 3220 = row 3220 := model_count_3220.trans row_count_3220.symm
def row_3221 : RowCertificate 3221 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3221 1) := by change ¬ Nat.Prime 41505807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13835269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3221 2) := by change ¬ Nat.Prime 41499365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8299873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3221 3) := by change Nat.Prime 41492923; exact ProjectEulerSolutions.P58PrimeNodes.p_41492923.prime
    exact decide_eq_true prime
theorem row_count_3221 : row 3221 = 1 := (row_3221.checked).trans rfl
theorem model_count_3221 : modelRow 3221 = 1 := by decide
theorem modelrow_agrees_3221 : modelRow 3221 = row 3221 := model_count_3221.trans row_count_3221.symm
def row_3222 : RowCertificate 3222 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3222 1) := by change ¬ Nat.Prime 41531581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5933083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3222 2) := by change ¬ Nat.Prime 41525137; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1122301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3222 3) := by change ¬ Nat.Prime 41518693; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 965551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3222 : row 3222 = 0 := (row_3222.checked).trans rfl
theorem model_count_3222 : modelRow 3222 = 0 := by decide
theorem modelrow_agrees_3222 : modelRow 3222 = row 3222 := model_count_3222.trans row_count_3222.symm
def row_3223 : RowCertificate 3223 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3223 1) := by change Nat.Prime 41557363; exact ProjectEulerSolutions.P58PrimeNodes.p_41557363.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3223 2) := by change ¬ Nat.Prime 41550917; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1013437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3223 3) := by change ¬ Nat.Prime 41544471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13848157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3223 : row 3223 = 1 := (row_3223.checked).trans rfl
theorem model_count_3223 : modelRow 3223 = 1 := by decide
theorem modelrow_agrees_3223 : modelRow 3223 = row 3223 := model_count_3223.trans row_count_3223.symm
def row_3224 : RowCertificate 3224 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3224 1) := by change ¬ Nat.Prime 41583153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13861051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3224 2) := by change ¬ Nat.Prime 41576705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8315341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3224 3) := by change Nat.Prime 41570257; exact ProjectEulerSolutions.P58PrimeNodes.p_41570257.prime
    exact decide_eq_true prime
theorem row_count_3224 : row 3224 = 1 := (row_3224.checked).trans rfl
theorem model_count_3224 : modelRow 3224 = 1 := by decide
theorem modelrow_agrees_3224 : modelRow 3224 = row 3224 := model_count_3224.trans row_count_3224.symm
def row_3225 : RowCertificate 3225 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3225 1) := by change Nat.Prime 41608951; exact ProjectEulerSolutions.P58PrimeNodes.p_41608951.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3225 2) := by change ¬ Nat.Prime 41602501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1434569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3225 3) := by change ¬ Nat.Prime 41596051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5942293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3225 : row 3225 = 1 := (row_3225.checked).trans rfl
theorem model_count_3225 : modelRow 3225 = 1 := by decide
theorem modelrow_agrees_3225 : modelRow 3225 = row 3225 := model_count_3225.trans row_count_3225.symm
def row_3226 : RowCertificate 3226 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3226 1) := by change ¬ Nat.Prime 41634757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2191303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3226 2) := by change ¬ Nat.Prime 41628305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8325661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3226 3) := by change ¬ Nat.Prime 41621853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13873951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3226 : row 3226 = 0 := (row_3226.checked).trans rfl
theorem model_count_3226 : modelRow 3226 = 0 := by decide
theorem modelrow_agrees_3226 : modelRow 3226 = row 3226 := model_count_3226.trans row_count_3226.symm
def row_3227 : RowCertificate 3227 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3227 1) := by change ¬ Nat.Prime 41660571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13886857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3227 2) := by change ¬ Nat.Prime 41654117; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 412417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3227 3) := by change ¬ Nat.Prime 41647663; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1343473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3227 : row 3227 = 0 := (row_3227.checked).trans rfl
theorem model_count_3227 : modelRow 3227 = 0 := by decide
theorem modelrow_agrees_3227 : modelRow 3227 = row 3227 := model_count_3227.trans row_count_3227.symm
def row_3228 : RowCertificate 3228 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3228 1) := by change ¬ Nat.Prime 41686393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5955199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3228 2) := by change ¬ Nat.Prime 41679937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3206149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3228 3) := by change Nat.Prime 41673481; exact ProjectEulerSolutions.P58PrimeNodes.p_41673481.prime
    exact decide_eq_true prime
theorem row_count_3228 : row 3228 = 1 := (row_3228.checked).trans rfl
theorem model_count_3228 : modelRow 3228 = 1 := by decide
theorem modelrow_agrees_3228 : modelRow 3228 = row 3228 := model_count_3228.trans row_count_3228.symm
def row_3229 : RowCertificate 3229 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3229 1) := by change ¬ Nat.Prime 41712223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5958889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3229 2) := by change ¬ Nat.Prime 41705765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8341153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3229 3) := by change ¬ Nat.Prime 41699307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13899769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3229 : row 3229 = 0 := (row_3229.checked).trans rfl
theorem model_count_3229 : modelRow 3229 = 0 := by decide
theorem modelrow_agrees_3229 : modelRow 3229 = row 3229 := model_count_3229.trans row_count_3229.symm
def row_3230 : RowCertificate 3230 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3230 1) := by change ¬ Nat.Prime 41738061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13912687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3230 2) := by change Nat.Prime 41731601; exact ProjectEulerSolutions.P58PrimeNodes.p_41731601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3230 3) := by change Nat.Prime 41725141; exact ProjectEulerSolutions.P58PrimeNodes.p_41725141.prime
    exact decide_eq_true prime
theorem row_count_3230 : row 3230 = 2 := (row_3230.checked).trans rfl
theorem model_count_3230 : modelRow 3230 = 2 := by decide
theorem modelrow_agrees_3230 : modelRow 3230 = row 3230 := model_count_3230.trans row_count_3230.symm
def row_3231 : RowCertificate 3231 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3231 1) := by change Nat.Prime 41763907; exact ProjectEulerSolutions.P58PrimeNodes.p_41763907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3231 2) := by change ¬ Nat.Prime 41757445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8351489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3231 3) := by change ¬ Nat.Prime 41750983; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 623149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3231 : row 3231 = 1 := (row_3231.checked).trans rfl
theorem model_count_3231 : modelRow 3231 = 1 := by decide
theorem modelrow_agrees_3231 : modelRow 3231 = row 3231 := model_count_3231.trans row_count_3231.symm
def row_3232 : RowCertificate 3232 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3232 1) := by change ¬ Nat.Prime 41789761; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3214597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3232 2) := by change ¬ Nat.Prime 41783297; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2457841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3232 3) := by change ¬ Nat.Prime 41776833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13925611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3232 : row 3232 = 0 := (row_3232.checked).trans rfl
theorem model_count_3232 : modelRow 3232 = 0 := by decide
theorem modelrow_agrees_3232 : modelRow 3232 = row 3232 := model_count_3232.trans row_count_3232.symm
def row_3233 : RowCertificate 3233 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3233 1) := by change ¬ Nat.Prime 41815623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13938541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3233 2) := by change ¬ Nat.Prime 41809157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3216089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3233 3) := by change ¬ Nat.Prime 41802691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5971813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3233 : row 3233 = 0 := (row_3233.checked).trans rfl
theorem model_count_3233 : modelRow 3233 = 0 := by decide
theorem modelrow_agrees_3233 : modelRow 3233 = row 3233 := model_count_3233.trans row_count_3233.symm
def row_3234 : RowCertificate 3234 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3234 1) := by change ¬ Nat.Prime 41841493; exact PrattCertificates.composite_of_product (factor := 4789) (cofactor := 8737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3234 2) := by change ¬ Nat.Prime 41835025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8367005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3234 3) := by change ¬ Nat.Prime 41828557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2201503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3234 : row 3234 = 0 := (row_3234.checked).trans rfl
theorem model_count_3234 : modelRow 3234 = 0 := by decide
theorem modelrow_agrees_3234 : modelRow 3234 = row 3234 := model_count_3234.trans row_count_3234.symm
def row_3235 : RowCertificate 3235 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3235 1) := by change ¬ Nat.Prime 41867371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5981053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3235 2) := by change ¬ Nat.Prime 41860901; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 573437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3235 3) := by change ¬ Nat.Prime 41854431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13951477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3235 : row 3235 = 0 := (row_3235.checked).trans rfl
theorem model_count_3235 : modelRow 3235 = 0 := by decide
theorem modelrow_agrees_3235 : modelRow 3235 = row 3235 := model_count_3235.trans row_count_3235.symm
def row_3236 : RowCertificate 3236 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3236 1) := by change ¬ Nat.Prime 41893257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13964419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3236 2) := by change ¬ Nat.Prime 41886785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8377357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3236 3) := by change ¬ Nat.Prime 41880313; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2204227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3236 : row 3236 = 0 := (row_3236.checked).trans rfl
theorem model_count_3236 : modelRow 3236 = 0 := by decide
theorem modelrow_agrees_3236 : modelRow 3236 = row 3236 := model_count_3236.trans row_count_3236.symm
def row_3237 : RowCertificate 3237 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3237 1) := by change ¬ Nat.Prime 41919151; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 210649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3237 2) := by change ¬ Nat.Prime 41912677; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 414977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3237 3) := by change ¬ Nat.Prime 41906203; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1351813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3237 : row 3237 = 0 := (row_3237.checked).trans rfl
theorem model_count_3237 : modelRow 3237 = 0 := by decide
theorem modelrow_agrees_3237 : modelRow 3237 = row 3237 := model_count_3237.trans row_count_3237.symm
def row_3238 : RowCertificate 3238 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3238 1) := by change ¬ Nat.Prime 41945053; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 384817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3238 2) := by change ¬ Nat.Prime 41938577; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 306121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3238 3) := by change ¬ Nat.Prime 41932101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13977367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3238 : row 3238 = 0 := (row_3238.checked).trans rfl
theorem model_count_3238 : modelRow 3238 = 0 := by decide
theorem modelrow_agrees_3238 : modelRow 3238 = row 3238 := model_count_3238.trans row_count_3238.symm
def row_3239 : RowCertificate 3239 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3239 1) := by change ¬ Nat.Prime 41970963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13990321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3239 2) := by change ¬ Nat.Prime 41964485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8392897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3239 3) := by change ¬ Nat.Prime 41958007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5994001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3239 : row 3239 = 0 := (row_3239.checked).trans rfl
theorem model_count_3239 : modelRow 3239 = 0 := by decide
theorem modelrow_agrees_3239 : modelRow 3239 = row 3239 := model_count_3239.trans row_count_3239.symm
def row_3240 : RowCertificate 3240 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3240 1) := by change ¬ Nat.Prime 41996881; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 29389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3240 2) := by change Nat.Prime 41990401; exact ProjectEulerSolutions.P58PrimeNodes.p_41990401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3240 3) := by change ¬ Nat.Prime 41983921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5997703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3240 : row 3240 = 1 := (row_3240.checked).trans rfl
theorem model_count_3240 : modelRow 3240 = 1 := by decide
theorem modelrow_agrees_3240 : modelRow 3240 = row 3240 := model_count_3240.trans row_count_3240.symm
def row_3241 : RowCertificate 3241 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3241 1) := by change Nat.Prime 42022807; exact ProjectEulerSolutions.P58PrimeNodes.p_42022807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3241 2) := by change ¬ Nat.Prime 42016325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8403265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3241 3) := by change ¬ Nat.Prime 42009843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14003281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3241 : row 3241 = 1 := (row_3241.checked).trans rfl
theorem model_count_3241 : modelRow 3241 = 1 := by decide
theorem modelrow_agrees_3241 : modelRow 3241 = row 3241 := model_count_3241.trans row_count_3241.symm
def row_3242 : RowCertificate 3242 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3242 1) := by change ¬ Nat.Prime 42048741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14016247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3242 2) := by change ¬ Nat.Prime 42042257; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1449733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3242 3) := by change ¬ Nat.Prime 42035773; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3233521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3242 : row 3242 = 0 := (row_3242.checked).trans rfl
theorem model_count_3242 : modelRow 3242 = 0 := by decide
theorem modelrow_agrees_3242 : modelRow 3242 = row 3242 := model_count_3242.trans row_count_3242.symm
def row_3243 : RowCertificate 3243 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3243 1) := by change ¬ Nat.Prime 42074683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6010669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3243 2) := by change ¬ Nat.Prime 42068197; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 69997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3243 3) := by change ¬ Nat.Prime 42061711; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1136803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3243 : row 3243 = 0 := (row_3243.checked).trans rfl
theorem model_count_3243 : modelRow 3243 = 0 := by decide
theorem modelrow_agrees_3243 : modelRow 3243 = row 3243 := model_count_3243.trans row_count_3243.symm
def row_3244 : RowCertificate 3244 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3244 1) := by change ¬ Nat.Prime 42100633; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 576721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3244 2) := by change ¬ Nat.Prime 42094145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8418829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3244 3) := by change ¬ Nat.Prime 42087657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14029219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3244 : row 3244 = 0 := (row_3244.checked).trans rfl
theorem model_count_3244 : modelRow 3244 = 0 := by decide
theorem modelrow_agrees_3244 : modelRow 3244 = row 3244 := model_count_3244.trans row_count_3244.symm
def row_3245 : RowCertificate 3245 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3245 1) := by change ¬ Nat.Prime 42126591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14042197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3245 2) := by change ¬ Nat.Prime 42120101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2477653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3245 3) := by change Nat.Prime 42113611; exact ProjectEulerSolutions.P58PrimeNodes.p_42113611.prime
    exact decide_eq_true prime
theorem row_count_3245 : row 3245 = 1 := (row_3245.checked).trans rfl
theorem model_count_3245 : modelRow 3245 = 1 := by decide
theorem modelrow_agrees_3245 : modelRow 3245 = row 3245 := model_count_3245.trans row_count_3245.symm
def row_3246 : RowCertificate 3246 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3246 1) := by change ¬ Nat.Prime 42152557; exact PrattCertificates.composite_of_product (factor := 1447) (cofactor := 29131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3246 2) := by change ¬ Nat.Prime 42146065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8429213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3246 3) := by change ¬ Nat.Prime 42139573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6019939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3246 : row 3246 = 0 := (row_3246.checked).trans rfl
theorem model_count_3246 : modelRow 3246 = 0 := by decide
theorem modelrow_agrees_3246 : modelRow 3246 = row 3246 := model_count_3246.trans row_count_3246.symm
def row_3247 : RowCertificate 3247 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3247 1) := by change ¬ Nat.Prime 42178531; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 386959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3247 2) := by change ¬ Nat.Prime 42172037; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 243769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3247 3) := by change ¬ Nat.Prime 42165543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14055181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3247 : row 3247 = 0 := (row_3247.checked).trans rfl
theorem model_count_3247 : modelRow 3247 = 0 := by decide
theorem modelrow_agrees_3247 : modelRow 3247 = row 3247 := model_count_3247.trans row_count_3247.symm
def row_3248 : RowCertificate 3248 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3248 1) := by change ¬ Nat.Prime 42204513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14068171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3248 2) := by change ¬ Nat.Prime 42198017; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 796189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3248 3) := by change Nat.Prime 42191521; exact ProjectEulerSolutions.P58PrimeNodes.p_42191521.prime
    exact decide_eq_true prime
theorem row_count_3248 : row 3248 = 1 := (row_3248.checked).trans rfl
theorem model_count_3248 : modelRow 3248 = 1 := by decide
theorem modelrow_agrees_3248 : modelRow 3248 = row 3248 := model_count_3248.trans row_count_3248.symm
def row_3249 : RowCertificate 3249 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3249 1) := by change ¬ Nat.Prime 42230503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6032929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3249 2) := by change ¬ Nat.Prime 42224005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8444801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3249 3) := by change Nat.Prime 42217507; exact ProjectEulerSolutions.P58PrimeNodes.p_42217507.prime
    exact decide_eq_true prime
theorem row_count_3249 : row 3249 = 1 := (row_3249.checked).trans rfl
theorem model_count_3249 : modelRow 3249 = 1 := by decide
theorem modelrow_agrees_3249 : modelRow 3249 = row 3249 := model_count_3249.trans row_count_3249.symm
def row_3250 : RowCertificate 3250 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3250 1) := by change ¬ Nat.Prime 42256501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6036643) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3250 2) := by change ¬ Nat.Prime 42250001; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 41381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3250 3) := by change ¬ Nat.Prime 42243501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14081167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3250 : row 3250 = 0 := (row_3250.checked).trans rfl
theorem model_count_3250 : modelRow 3250 = 0 := by decide
theorem modelrow_agrees_3250 : modelRow 3250 = row 3250 := model_count_3250.trans row_count_3250.symm
def row_3251 : RowCertificate 3251 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3251 1) := by change ¬ Nat.Prime 42282507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14094169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3251 2) := by change ¬ Nat.Prime 42276005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8455201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3251 3) := by change ¬ Nat.Prime 42269503; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1142419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3251 : row 3251 = 0 := (row_3251.checked).trans rfl
theorem model_count_3251 : modelRow 3251 = 0 := by decide
theorem modelrow_agrees_3251 : modelRow 3251 = row 3251 := model_count_3251.trans row_count_3251.symm
def row_3252 : RowCertificate 3252 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3252 1) := by change ¬ Nat.Prime 42308521; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1364791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3252 2) := by change ¬ Nat.Prime 42302017; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 63997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3252 3) := by change ¬ Nat.Prime 42295513; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3253501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3252 : row 3252 = 0 := (row_3252.checked).trans rfl
theorem model_count_3252 : modelRow 3252 = 0 := by decide
theorem modelrow_agrees_3252 : modelRow 3252 = row 3252 := model_count_3252.trans row_count_3252.symm
def row_3253 : RowCertificate 3253 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3253 1) := by change ¬ Nat.Prime 42334543; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 189841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3253 2) := by change ¬ Nat.Prime 42328037; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1144001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3253 3) := by change ¬ Nat.Prime 42321531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14107177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3253 : row 3253 = 0 := (row_3253.checked).trans rfl
theorem model_count_3253 : modelRow 3253 = 0 := by decide
theorem modelrow_agrees_3253 : modelRow 3253 = row 3253 := model_count_3253.trans row_count_3253.symm
def row_3254 : RowCertificate 3254 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3254 1) := by change ¬ Nat.Prime 42360573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14120191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3254 2) := by change ¬ Nat.Prime 42354065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8470813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3254 3) := by change ¬ Nat.Prime 42347557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6049651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3254 : row 3254 = 0 := (row_3254.checked).trans rfl
theorem model_count_3254 : modelRow 3254 = 0 := by decide
theorem modelrow_agrees_3254 : modelRow 3254 = row 3254 := model_count_3254.trans row_count_3254.symm
def row_3255 : RowCertificate 3255 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3255 1) := by change Nat.Prime 42386611; exact ProjectEulerSolutions.P58PrimeNodes.p_42386611.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3255 2) := by change ¬ Nat.Prime 42380101; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1033661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3255 3) := by change ¬ Nat.Prime 42373591; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3259507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3255 : row 3255 = 1 := (row_3255.checked).trans rfl
theorem model_count_3255 : modelRow 3255 = 1 := by decide
theorem modelrow_agrees_3255 : modelRow 3255 = row 3255 := model_count_3255.trans row_count_3255.symm
def row_3256 : RowCertificate 3256 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3256 1) := by change ¬ Nat.Prime 42412657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6058951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3256 2) := by change ¬ Nat.Prime 42406145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8481229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3256 3) := by change ¬ Nat.Prime 42399633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14133211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3256 : row 3256 = 0 := (row_3256.checked).trans rfl
theorem model_count_3256 : modelRow 3256 = 0 := by decide
theorem modelrow_agrees_3256 : modelRow 3256 = row 3256 := model_count_3256.trans row_count_3256.symm
def row_3257 : RowCertificate 3257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3257 1) := by change ¬ Nat.Prime 42438711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14146237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3257 2) := by change Nat.Prime 42432197; exact ProjectEulerSolutions.P58PrimeNodes.p_42432197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3257 3) := by change ¬ Nat.Prime 42425683; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 695503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3257 : row 3257 = 1 := (row_3257.checked).trans rfl
theorem model_count_3257 : modelRow 3257 = 1 := by decide
theorem modelrow_agrees_3257 : modelRow 3257 = row 3257 := model_count_3257.trans row_count_3257.symm
def row_3258 : RowCertificate 3258 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3258 1) := by change ¬ Nat.Prime 42464773; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3266521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3258 2) := by change ¬ Nat.Prime 42458257; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 696037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3258 3) := by change ¬ Nat.Prime 42451741; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1369411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3258 : row 3258 = 0 := (row_3258.checked).trans rfl
theorem model_count_3258 : modelRow 3258 = 0 := by decide
theorem modelrow_agrees_3258 : modelRow 3258 = row 3258 := model_count_3258.trans row_count_3258.symm
def row_3259 : RowCertificate 3259 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3259 1) := by change ¬ Nat.Prime 42490843; exact PrattCertificates.composite_of_product (factor := 883) (cofactor := 48121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3259 2) := by change ¬ Nat.Prime 42484325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8496865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3259 3) := by change ¬ Nat.Prime 42477807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14159269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3259 : row 3259 = 0 := (row_3259.checked).trans rfl
theorem model_count_3259 : modelRow 3259 = 0 := by decide
theorem modelrow_agrees_3259 : modelRow 3259 = row 3259 := model_count_3259.trans row_count_3259.symm
def row_3260 : RowCertificate 3260 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3260 1) := by change ¬ Nat.Prime 42516921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14172307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3260 2) := by change ¬ Nat.Prime 42510401; exact PrattCertificates.composite_of_product (factor := 2689) (cofactor := 15809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3260 3) := by change ¬ Nat.Prime 42503881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6071983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3260 : row 3260 = 0 := (row_3260.checked).trans rfl
theorem model_count_3260 : modelRow 3260 = 0 := by decide
theorem modelrow_agrees_3260 : modelRow 3260 = row 3260 := model_count_3260.trans row_count_3260.symm
def row_3261 : RowCertificate 3261 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3261 1) := by change ¬ Nat.Prime 42543007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3272539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3261 2) := by change ¬ Nat.Prime 42536485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8507297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3261 3) := by change ¬ Nat.Prime 42529963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6075709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3261 : row 3261 = 0 := (row_3261.checked).trans rfl
theorem model_count_3261 : modelRow 3261 = 0 := by decide
theorem modelrow_agrees_3261 : modelRow 3261 = row 3261 := model_count_3261.trans row_count_3261.symm
def row_3262 : RowCertificate 3262 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3262 1) := by change ¬ Nat.Prime 42569101; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2240479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3262 2) := by change ¬ Nat.Prime 42562577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2503681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3262 3) := by change ¬ Nat.Prime 42556053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14185351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3262 : row 3262 = 0 := (row_3262.checked).trans rfl
theorem model_count_3262 : modelRow 3262 = 0 := by decide
theorem modelrow_agrees_3262 : modelRow 3262 = row 3262 := model_count_3262.trans row_count_3262.symm
def row_3263 : RowCertificate 3263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3263 1) := by change ¬ Nat.Prime 42595203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14198401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3263 2) := by change ¬ Nat.Prime 42588677; exact PrattCertificates.composite_of_product (factor := 2297) (cofactor := 18541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3263 3) := by change ¬ Nat.Prime 42582151; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 282001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3263 : row 3263 = 0 := (row_3263.checked).trans rfl
theorem model_count_3263 : modelRow 3263 = 0 := by decide
theorem modelrow_agrees_3263 : modelRow 3263 = row 3263 := model_count_3263.trans row_count_3263.symm
def row_3264 : RowCertificate 3264 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3264 1) := by change ¬ Nat.Prime 42621313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6088759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3264 2) := by change ¬ Nat.Prime 42614785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8522957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3264 3) := by change ¬ Nat.Prime 42608257; exact PrattCertificates.composite_of_product (factor := 1759) (cofactor := 24223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3264 : row 3264 = 0 := (row_3264.checked).trans rfl
theorem model_count_3264 : modelRow 3264 = 0 := by decide
theorem modelrow_agrees_3264 : modelRow 3264 = row 3264 := model_count_3264.trans row_count_3264.symm
def row_3265 : RowCertificate 3265 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3265 1) := by change ¬ Nat.Prime 42647431; exact PrattCertificates.composite_of_product (factor := 1447) (cofactor := 29473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3265 2) := by change Nat.Prime 42640901; exact ProjectEulerSolutions.P58PrimeNodes.p_42640901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3265 3) := by change ¬ Nat.Prime 42634371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14211457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3265 : row 3265 = 1 := (row_3265.checked).trans rfl
theorem model_count_3265 : modelRow 3265 = 1 := by decide
theorem modelrow_agrees_3265 : modelRow 3265 = row 3265 := model_count_3265.trans row_count_3265.symm
def row_3266 : RowCertificate 3266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3266 1) := by change ¬ Nat.Prime 42673557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14224519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3266 2) := by change ¬ Nat.Prime 42667025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8533405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3266 3) := by change ¬ Nat.Prime 42660493; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 154009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3266 : row 3266 = 0 := (row_3266.checked).trans rfl
theorem model_count_3266 : modelRow 3266 = 0 := by decide
theorem modelrow_agrees_3266 : modelRow 3266 = row 3266 := model_count_3266.trans row_count_3266.symm
def row_3267 : RowCertificate 3267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3267 1) := by change ¬ Nat.Prime 42699691; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 440203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3267 2) := by change ¬ Nat.Prime 42693157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3284089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3267 3) := by change ¬ Nat.Prime 42686623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6098089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3267 : row 3267 = 0 := (row_3267.checked).trans rfl
theorem model_count_3267 : modelRow 3267 = 0 := by decide
theorem modelrow_agrees_3267 : modelRow 3267 = row 3267 := model_count_3267.trans row_count_3267.symm
def row_3268 : RowCertificate 3268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3268 1) := by change ¬ Nat.Prime 42725833; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 637699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3268 2) := by change Nat.Prime 42719297; exact ProjectEulerSolutions.P58PrimeNodes.p_42719297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3268 3) := by change ¬ Nat.Prime 42712761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14237587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3268 : row 3268 = 1 := (row_3268.checked).trans rfl
theorem model_count_3268 : modelRow 3268 = 1 := by decide
theorem modelrow_agrees_3268 : modelRow 3268 = row 3268 := model_count_3268.trans row_count_3268.symm
def row_3269 : RowCertificate 3269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3269 1) := by change ¬ Nat.Prime 42751983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14250661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3269 2) := by change ¬ Nat.Prime 42745445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8549089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3269 3) := by change Nat.Prime 42738907; exact ProjectEulerSolutions.P58PrimeNodes.p_42738907.prime
    exact decide_eq_true prime
theorem row_count_3269 : row 3269 = 1 := (row_3269.checked).trans rfl
theorem model_count_3269 : modelRow 3269 = 1 := by decide
theorem modelrow_agrees_3269 : modelRow 3269 = row 3269 := model_count_3269.trans row_count_3269.symm
def row_3270 : RowCertificate 3270 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3270 1) := by change ¬ Nat.Prime 42778141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6111163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3270 2) := by change Nat.Prime 42771601; exact ProjectEulerSolutions.P58PrimeNodes.p_42771601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3270 3) := by change Nat.Prime 42765061; exact ProjectEulerSolutions.P58PrimeNodes.p_42765061.prime
    exact decide_eq_true prime
theorem row_count_3270 : row 3270 = 2 := (row_3270.checked).trans rfl
theorem model_count_3270 : modelRow 3270 = 2 := by decide
theorem modelrow_agrees_3270 : modelRow 3270 = row 3270 := model_count_3270.trans row_count_3270.symm
def row_3271 : RowCertificate 3271 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3271 1) := by change ¬ Nat.Prime 42804307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6114901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3271 2) := by change ¬ Nat.Prime 42797765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8559553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3271 3) := by change ¬ Nat.Prime 42791223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14263741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3271 : row 3271 = 0 := (row_3271.checked).trans rfl
theorem model_count_3271 : modelRow 3271 = 0 := by decide
theorem modelrow_agrees_3271 : modelRow 3271 = row 3271 := model_count_3271.trans row_count_3271.symm
def row_3272 : RowCertificate 3272 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3272 1) := by change ¬ Nat.Prime 42830481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14276827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3272 2) := by change ¬ Nat.Prime 42823937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3294149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3272 3) := by change ¬ Nat.Prime 42817393; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2253547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3272 : row 3272 = 0 := (row_3272.checked).trans rfl
theorem model_count_3272 : modelRow 3272 = 0 := by decide
theorem modelrow_agrees_3272 : modelRow 3272 = row 3272 := model_count_3272.trans row_count_3272.symm
def row_3273 : RowCertificate 3273 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3273 1) := by change ¬ Nat.Prime 42856663; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1382473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3273 2) := by change Nat.Prime 42850117; exact ProjectEulerSolutions.P58PrimeNodes.p_42850117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3273 3) := by change ¬ Nat.Prime 42843571; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 415957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3273 : row 3273 = 1 := (row_3273.checked).trans rfl
theorem model_count_3273 : modelRow 3273 = 1 := by decide
theorem modelrow_agrees_3273 : modelRow 3273 = row 3273 := model_count_3273.trans row_count_3273.symm
def row_3274 : RowCertificate 3274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3274 1) := by change ¬ Nat.Prime 42882853; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3298681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3274 2) := by change ¬ Nat.Prime 42876305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8575261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3274 3) := by change ¬ Nat.Prime 42869757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14289919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3274 : row 3274 = 0 := (row_3274.checked).trans rfl
theorem model_count_3274 : modelRow 3274 = 0 := by decide
theorem modelrow_agrees_3274 : modelRow 3274 = row 3274 := model_count_3274.trans row_count_3274.symm
def row_3275 : RowCertificate 3275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3275 1) := by change ¬ Nat.Prime 42909051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14303017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3275 2) := by change Nat.Prime 42902501; exact ProjectEulerSolutions.P58PrimeNodes.p_42902501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3275 3) := by change ¬ Nat.Prime 42895951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6127993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3275 : row 3275 = 1 := (row_3275.checked).trans rfl
theorem model_count_3275 : modelRow 3275 = 1 := by decide
theorem modelrow_agrees_3275 : modelRow 3275 = row 3275 := model_count_3275.trans row_count_3275.symm
def row_3276 : RowCertificate 3276 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3276 1) := by change Nat.Prime 42935257; exact ProjectEulerSolutions.P58PrimeNodes.p_42935257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3276 2) := by change ¬ Nat.Prime 42928705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8585741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3276 3) := by change Nat.Prime 42922153; exact ProjectEulerSolutions.P58PrimeNodes.p_42922153.prime
    exact decide_eq_true prime
theorem row_count_3276 : row 3276 = 2 := (row_3276.checked).trans rfl
theorem model_count_3276 : modelRow 3276 = 2 := by decide
theorem modelrow_agrees_3276 : modelRow 3276 = row 3276 := model_count_3276.trans row_count_3276.symm
def row_3277 : RowCertificate 3277 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3277 1) := by change ¬ Nat.Prime 42961471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6137353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3277 2) := by change ¬ Nat.Prime 42954917; exact PrattCertificates.composite_of_product (factor := 881) (cofactor := 48757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3277 3) := by change ¬ Nat.Prime 42948363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14316121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3277 : row 3277 = 0 := (row_3277.checked).trans rfl
theorem model_count_3277 : modelRow 3277 = 0 := by decide
theorem modelrow_agrees_3277 : modelRow 3277 = row 3277 := model_count_3277.trans row_count_3277.symm
def row_3278 : RowCertificate 3278 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3278 1) := by change ¬ Nat.Prime 42987693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14329231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3278 2) := by change ¬ Nat.Prime 42981137; exact PrattCertificates.composite_of_product (factor := 1301) (cofactor := 33037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3278 3) := by change ¬ Nat.Prime 42974581; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3305737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3278 : row 3278 = 0 := (row_3278.checked).trans rfl
theorem model_count_3278 : modelRow 3278 = 0 := by decide
theorem modelrow_agrees_3278 : modelRow 3278 = row 3278 := model_count_3278.trans row_count_3278.symm
def row_3279 : RowCertificate 3279 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3279 1) := by change ¬ Nat.Prime 43013923; exact PrattCertificates.composite_of_product (factor := 3463) (cofactor := 12421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3279 2) := by change ¬ Nat.Prime 43007365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8601473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3279 3) := by change ¬ Nat.Prime 43000807; exact PrattCertificates.composite_of_product (factor := 1657) (cofactor := 25951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3279 : row 3279 = 0 := (row_3279.checked).trans rfl
theorem model_count_3279 : modelRow 3279 = 0 := by decide
theorem modelrow_agrees_3279 : modelRow 3279 = row 3279 := model_count_3279.trans row_count_3279.symm
def row_3280 : RowCertificate 3280 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3280 1) := by change ¬ Nat.Prime 43040161; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 443713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3280 2) := by change ¬ Nat.Prime 43033601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3310277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3280 3) := by change ¬ Nat.Prime 43027041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14342347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3280 : row 3280 = 0 := (row_3280.checked).trans rfl
theorem model_count_3280 : modelRow 3280 = 0 := by decide
theorem modelrow_agrees_3280 : modelRow 3280 = row 3280 := model_count_3280.trans row_count_3280.symm
def row_3281 : RowCertificate 3281 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3281 1) := by change ¬ Nat.Prime 43066407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14355469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3281 2) := by change ¬ Nat.Prime 43059845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8611969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3281 3) := by change ¬ Nat.Prime 43053283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6150469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3281 : row 3281 = 0 := (row_3281.checked).trans rfl
theorem model_count_3281 : modelRow 3281 = 0 := by decide
theorem modelrow_agrees_3281 : modelRow 3281 = row 3281 := model_count_3281.trans row_count_3281.symm
def row_3282 : RowCertificate 3282 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3282 1) := by change ¬ Nat.Prime 43092661; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 238081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3282 2) := by change ¬ Nat.Prime 43086097; exact PrattCertificates.composite_of_product (factor := 641) (cofactor := 67217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3282 3) := by change ¬ Nat.Prime 43079533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6154219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3282 : row 3282 = 0 := (row_3282.checked).trans rfl
theorem model_count_3282 : modelRow 3282 = 0 := by decide
theorem modelrow_agrees_3282 : modelRow 3282 = row 3282 := model_count_3282.trans row_count_3282.symm
def row_3283 : RowCertificate 3283 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3283 1) := by change ¬ Nat.Prime 43118923; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2269417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3283 2) := by change ¬ Nat.Prime 43112357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2536021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3283 3) := by change ¬ Nat.Prime 43105791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14368597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3283 : row 3283 = 0 := (row_3283.checked).trans rfl
theorem model_count_3283 : modelRow 3283 = 0 := by decide
theorem modelrow_agrees_3283 : modelRow 3283 = row 3283 := model_count_3283.trans row_count_3283.symm
def row_3284 : RowCertificate 3284 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3284 1) := by change ¬ Nat.Prime 43145193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14381731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3284 2) := by change ¬ Nat.Prime 43138625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8627725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3284 3) := by change Nat.Prime 43132057; exact ProjectEulerSolutions.P58PrimeNodes.p_43132057.prime
    exact decide_eq_true prime
theorem row_count_3284 : row 3284 = 1 := (row_3284.checked).trans rfl
theorem model_count_3284 : modelRow 3284 = 1 := by decide
theorem modelrow_agrees_3284 : modelRow 3284 = row 3284 := model_count_3284.trans row_count_3284.symm
def row_3285 : RowCertificate 3285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3285 1) := by change ¬ Nat.Prime 43171471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6167353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3285 2) := by change ¬ Nat.Prime 43164901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3320377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3285 3) := by change Nat.Prime 43158331; exact ProjectEulerSolutions.P58PrimeNodes.p_43158331.prime
    exact decide_eq_true prime
theorem row_count_3285 : row 3285 = 1 := (row_3285.checked).trans rfl
theorem model_count_3285 : modelRow 3285 = 1 := by decide
theorem modelrow_agrees_3285 : modelRow 3285 = row 3285 := model_count_3285.trans row_count_3285.symm
def row_3286 : RowCertificate 3286 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3286 1) := by change ¬ Nat.Prime 43197757; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1004599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3286 2) := by change ¬ Nat.Prime 43191185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8638237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3286 3) := by change ¬ Nat.Prime 43184613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14394871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3286 : row 3286 = 0 := (row_3286.checked).trans rfl
theorem model_count_3286 : modelRow 3286 = 0 := by decide
theorem modelrow_agrees_3286 : modelRow 3286 = row 3286 := model_count_3286.trans row_count_3286.symm
def row_3287 : RowCertificate 3287 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3287 1) := by change ¬ Nat.Prime 43224051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14408017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3287 2) := by change ¬ Nat.Prime 43217477; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 445541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3287 3) := by change Nat.Prime 43210903; exact ProjectEulerSolutions.P58PrimeNodes.p_43210903.prime
    exact decide_eq_true prime
theorem row_count_3287 : row 3287 = 1 := (row_3287.checked).trans rfl
theorem model_count_3287 : modelRow 3287 = 1 := by decide
theorem modelrow_agrees_3287 : modelRow 3287 = row 3287 := model_count_3287.trans row_count_3287.symm
def row_3288 : RowCertificate 3288 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3288 1) := by change Nat.Prime 43250353; exact ProjectEulerSolutions.P58PrimeNodes.p_43250353.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3288 2) := by change Nat.Prime 43243777; exact ProjectEulerSolutions.P58PrimeNodes.p_43243777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3288 3) := by change ¬ Nat.Prime 43237201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6176743) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3288 : row 3288 = 2 := (row_3288.checked).trans rfl
theorem model_count_3288 : modelRow 3288 = 2 := by decide
theorem modelrow_agrees_3288 : modelRow 3288 = row 3288 := model_count_3288.trans row_count_3288.symm
def row_3289 : RowCertificate 3289 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3289 1) := by change ¬ Nat.Prime 43276663; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 592831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3289 2) := by change ¬ Nat.Prime 43270085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8654017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3289 3) := by change ¬ Nat.Prime 43263507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14421169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3289 : row 3289 = 0 := (row_3289.checked).trans rfl
theorem model_count_3289 : modelRow 3289 = 0 := by decide
theorem modelrow_agrees_3289 : modelRow 3289 = row 3289 := model_count_3289.trans row_count_3289.symm
def row_3290 : RowCertificate 3290 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3290 1) := by change ¬ Nat.Prime 43302981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14434327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3290 2) := by change ¬ Nat.Prime 43296401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1170173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3290 3) := by change Nat.Prime 43289821; exact ProjectEulerSolutions.P58PrimeNodes.p_43289821.prime
    exact decide_eq_true prime
theorem row_count_3290 : row 3290 = 1 := (row_3290.checked).trans rfl
theorem model_count_3290 : modelRow 3290 = 1 := by decide
theorem modelrow_agrees_3290 : modelRow 3290 = row 3290 := model_count_3290.trans row_count_3290.symm
def row_3291 : RowCertificate 3291 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3291 1) := by change ¬ Nat.Prime 43329307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6189901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3291 2) := by change ¬ Nat.Prime 43322725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8664545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3291 3) := by change ¬ Nat.Prime 43316143; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3332011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3291 : row 3291 = 0 := (row_3291.checked).trans rfl
theorem model_count_3291 : modelRow 3291 = 0 := by decide
theorem modelrow_agrees_3291 : modelRow 3291 = row 3291 := model_count_3291.trans row_count_3291.symm
def row_3292 : RowCertificate 3292 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3292 1) := by change ¬ Nat.Prime 43355641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6193663) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3292 2) := by change ¬ Nat.Prime 43349057; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 147949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3292 3) := by change ¬ Nat.Prime 43342473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14447491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3292 : row 3292 = 0 := (row_3292.checked).trans rfl
theorem model_count_3292 : modelRow 3292 = 0 := by decide
theorem modelrow_agrees_3292 : modelRow 3292 = row 3292 := model_count_3292.trans row_count_3292.symm
def row_3293 : RowCertificate 3293 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3293 1) := by change ¬ Nat.Prime 43381983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14460661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3293 2) := by change ¬ Nat.Prime 43375397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3336569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3293 3) := by change ¬ Nat.Prime 43368811; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2282569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3293 : row 3293 = 0 := (row_3293.checked).trans rfl
theorem model_count_3293 : modelRow 3293 = 0 := by decide
theorem modelrow_agrees_3293 : modelRow 3293 = row 3293 := model_count_3293.trans row_count_3293.symm
def row_3294 : RowCertificate 3294 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3294 1) := by change ¬ Nat.Prime 43408333; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 131143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3294 2) := by change ¬ Nat.Prime 43401745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8680349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3294 3) := by change ¬ Nat.Prime 43395157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3338089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3294 : row 3294 = 0 := (row_3294.checked).trans rfl
theorem model_count_3294 : modelRow 3294 = 0 := by decide
theorem modelrow_agrees_3294 : modelRow 3294 = row 3294 := model_count_3294.trans row_count_3294.symm
def row_3295 : RowCertificate 3295 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3295 1) := by change Nat.Prime 43434691; exact ProjectEulerSolutions.P58PrimeNodes.p_43434691.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3295 2) := by change Nat.Prime 43428101; exact ProjectEulerSolutions.P58PrimeNodes.p_43428101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3295 3) := by change ¬ Nat.Prime 43421511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14473837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3295 : row 3295 = 2 := (row_3295.checked).trans rfl
theorem model_count_3295 : modelRow 3295 = 2 := by decide
theorem modelrow_agrees_3295 : modelRow 3295 = row 3295 := model_count_3295.trans row_count_3295.symm
def row_3296 : RowCertificate 3296 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3296 1) := by change ¬ Nat.Prime 43461057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14487019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3296 2) := by change ¬ Nat.Prime 43454465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8690893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3296 3) := by change ¬ Nat.Prime 43447873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6206839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3296 : row 3296 = 0 := (row_3296.checked).trans rfl
theorem model_count_3296 : modelRow 3296 = 0 := by decide
theorem modelrow_agrees_3296 : modelRow 3296 = row 3296 := model_count_3296.trans row_count_3296.symm
def row_3297 : RowCertificate 3297 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3297 1) := by change ¬ Nat.Prime 43487431; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3345187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3297 2) := by change ¬ Nat.Prime 43480837; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 43093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3297 3) := by change Nat.Prime 43474243; exact ProjectEulerSolutions.P58PrimeNodes.p_43474243.prime
    exact decide_eq_true prime
theorem row_count_3297 : row 3297 = 1 := (row_3297.checked).trans rfl
theorem model_count_3297 : modelRow 3297 = 1 := by decide
theorem modelrow_agrees_3297 : modelRow 3297 = row 3297 := model_count_3297.trans row_count_3297.symm
def row_3298 : RowCertificate 3298 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3298 1) := by change ¬ Nat.Prime 43513813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6216259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3298 2) := by change ¬ Nat.Prime 43507217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3346709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3298 3) := by change ¬ Nat.Prime 43500621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14500207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3298 : row 3298 = 0 := (row_3298.checked).trans rfl
theorem model_count_3298 : modelRow 3298 = 0 := by decide
theorem modelrow_agrees_3298 : modelRow 3298 = row 3298 := model_count_3298.trans row_count_3298.symm
def row_3299 : RowCertificate 3299 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3299 1) := by change ¬ Nat.Prime 43540203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14513401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3299 2) := by change ¬ Nat.Prime 43533605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8706721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3299 3) := by change ¬ Nat.Prime 43527007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1404097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3299 : row 3299 = 0 := (row_3299.checked).trans rfl
theorem model_count_3299 : modelRow 3299 = 0 := by decide
theorem modelrow_agrees_3299 : modelRow 3299 = row 3299 := model_count_3299.trans row_count_3299.symm
def row_3300 : RowCertificate 3300 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3300 1) := by change ¬ Nat.Prime 43566601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3351277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3300 2) := by change ¬ Nat.Prime 43560001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2562353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3300 3) := by change Nat.Prime 43553401; exact ProjectEulerSolutions.P58PrimeNodes.p_43553401.prime
    exact decide_eq_true prime
theorem row_count_3300 : row 3300 = 1 := (row_3300.checked).trans rfl
theorem model_count_3300 : modelRow 3300 = 1 := by decide
theorem modelrow_agrees_3300 : modelRow 3300 = row 3300 := model_count_3300.trans row_count_3300.symm
def row_3301 : RowCertificate 3301 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3301 1) := by change Nat.Prime 43593007; exact ProjectEulerSolutions.P58PrimeNodes.p_43593007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3301 2) := by change ¬ Nat.Prime 43586405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8717281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3301 3) := by change ¬ Nat.Prime 43579803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14526601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3301 : row 3301 = 1 := (row_3301.checked).trans rfl
theorem model_count_3301 : modelRow 3301 = 1 := by decide
theorem modelrow_agrees_3301 : modelRow 3301 = row 3301 := model_count_3301.trans row_count_3301.symm
def row_3302 : RowCertificate 3302 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3302 1) := by change ¬ Nat.Prime 43619421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14539807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3302 2) := by change Nat.Prime 43612817; exact ProjectEulerSolutions.P58PrimeNodes.p_43612817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3302 3) := by change ¬ Nat.Prime 43606213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6229459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3302 : row 3302 = 1 := (row_3302.checked).trans rfl
theorem model_count_3302 : modelRow 3302 = 1 := by decide
theorem modelrow_agrees_3302 : modelRow 3302 = row 3302 := model_count_3302.trans row_count_3302.symm
def row_3303 : RowCertificate 3303 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3303 1) := by change ¬ Nat.Prime 43645843; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 277999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3303 2) := by change ¬ Nat.Prime 43639237; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 38653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3303 3) := by change ¬ Nat.Prime 43632631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6233233) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3303 : row 3303 = 0 := (row_3303.checked).trans rfl
theorem model_count_3303 : modelRow 3303 = 0 := by decide
theorem modelrow_agrees_3303 : modelRow 3303 = row 3303 := model_count_3303.trans row_count_3303.symm
def row_3304 : RowCertificate 3304 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3304 1) := by change ¬ Nat.Prime 43672273; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1408783) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3304 2) := by change ¬ Nat.Prime 43665665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8733133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3304 3) := by change ¬ Nat.Prime 43659057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14553019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3304 : row 3304 = 0 := (row_3304.checked).trans rfl
theorem model_count_3304 : modelRow 3304 = 0 := by decide
theorem modelrow_agrees_3304 : modelRow 3304 = row 3304 := model_count_3304.trans row_count_3304.symm
def row_3305 : RowCertificate 3305 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3305 1) := by change ¬ Nat.Prime 43698711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14566237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3305 2) := by change ¬ Nat.Prime 43692101; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1065661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3305 3) := by change ¬ Nat.Prime 43685491; exact PrattCertificates.composite_of_product (factor := 3529) (cofactor := 12379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3305 : row 3305 = 0 := (row_3305.checked).trans rfl
theorem model_count_3305 : modelRow 3305 = 0 := by decide
theorem modelrow_agrees_3305 : modelRow 3305 = row 3305 := model_count_3305.trans row_count_3305.symm
def row_3306 : RowCertificate 3306 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3306 1) := by change ¬ Nat.Prime 43725157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6246451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3306 2) := by change ¬ Nat.Prime 43718545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8743709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3306 3) := by change ¬ Nat.Prime 43711933; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 289483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3306 : row 3306 = 0 := (row_3306.checked).trans rfl
theorem model_count_3306 : modelRow 3306 = 0 := by decide
theorem modelrow_agrees_3306 : modelRow 3306 = row 3306 := model_count_3306.trans row_count_3306.symm
def row_3307 : RowCertificate 3307 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3307 1) := by change Nat.Prime 43751611; exact ProjectEulerSolutions.P58PrimeNodes.p_43751611.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3307 2) := by change Nat.Prime 43744997; exact ProjectEulerSolutions.P58PrimeNodes.p_43744997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3307 3) := by change ¬ Nat.Prime 43738383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14579461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3307 : row 3307 = 2 := (row_3307.checked).trans rfl
theorem model_count_3307 : modelRow 3307 = 2 := by decide
theorem modelrow_agrees_3307 : modelRow 3307 = row 3307 := model_count_3307.trans row_count_3307.symm
def row_3308 : RowCertificate 3308 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3308 1) := by change ¬ Nat.Prime 43778073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14592691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3308 2) := by change ¬ Nat.Prime 43771457; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 599609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3308 3) := by change ¬ Nat.Prime 43764841; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1017787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3308 : row 3308 = 0 := (row_3308.checked).trans rfl
theorem model_count_3308 : modelRow 3308 = 0 := by decide
theorem modelrow_agrees_3308 : modelRow 3308 = row 3308 := model_count_3308.trans row_count_3308.symm
def row_3309 : RowCertificate 3309 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3309 1) := by change ¬ Nat.Prime 43804543; exact PrattCertificates.composite_of_product (factor := 811) (cofactor := 54013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3309 2) := by change ¬ Nat.Prime 43797925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8759585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3309 3) := by change ¬ Nat.Prime 43791307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6255901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3309 : row 3309 = 0 := (row_3309.checked).trans rfl
theorem model_count_3309 : modelRow 3309 = 0 := by decide
theorem modelrow_agrees_3309 : modelRow 3309 = row 3309 := model_count_3309.trans row_count_3309.symm
def row_3310 : RowCertificate 3310 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3310 1) := by change ¬ Nat.Prime 43831021; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3371617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3310 2) := by change ¬ Nat.Prime 43824401; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 492409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3310 3) := by change ¬ Nat.Prime 43817781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14605927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3310 : row 3310 = 0 := (row_3310.checked).trans rfl
theorem model_count_3310 : modelRow 3310 = 0 := by decide
theorem modelrow_agrees_3310 : modelRow 3310 = row 3310 := model_count_3310.trans row_count_3310.symm
def row_3311 : RowCertificate 3311 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3311 1) := by change ¬ Nat.Prime 43857507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14619169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3311 2) := by change ¬ Nat.Prime 43850885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8770177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3311 3) := by change ¬ Nat.Prime 43844263; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 81043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3311 : row 3311 = 0 := (row_3311.checked).trans rfl
theorem model_count_3311 : modelRow 3311 = 0 := by decide
theorem modelrow_agrees_3311 : modelRow 3311 = row 3311 := model_count_3311.trans row_count_3311.symm
def row_3312 : RowCertificate 3312 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3312 1) := by change ¬ Nat.Prime 43884001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6269143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3312 2) := by change ¬ Nat.Prime 43877377; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1513013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3312 3) := by change ¬ Nat.Prime 43870753; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2308987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3312 : row 3312 = 0 := (row_3312.checked).trans rfl
theorem model_count_3312 : modelRow 3312 = 0 := by decide
theorem modelrow_agrees_3312 : modelRow 3312 = row 3312 := model_count_3312.trans row_count_3312.symm
def row_3313 : RowCertificate 3313 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3313 1) := by change ¬ Nat.Prime 43910503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6272929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3313 2) := by change ¬ Nat.Prime 43903877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2582581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3313 3) := by change ¬ Nat.Prime 43897251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14632417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3313 : row 3313 = 0 := (row_3313.checked).trans rfl
theorem model_count_3313 : modelRow 3313 = 0 := by decide
theorem modelrow_agrees_3313 : modelRow 3313 = row 3313 := model_count_3313.trans row_count_3313.symm
def row_3314 : RowCertificate 3314 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3314 1) := by change ¬ Nat.Prime 43937013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14645671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3314 2) := by change ¬ Nat.Prime 43930385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8786077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3314 3) := by change Nat.Prime 43923757; exact ProjectEulerSolutions.P58PrimeNodes.p_43923757.prime
    exact decide_eq_true prime
theorem row_count_3314 : row 3314 = 1 := (row_3314.checked).trans rfl
theorem model_count_3314 : modelRow 3314 = 1 := by decide
theorem modelrow_agrees_3314 : modelRow 3314 = row 3314 := model_count_3314.trans row_count_3314.symm
def row_3315 : RowCertificate 3315 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3315 1) := by change ¬ Nat.Prime 43963531; exact PrattCertificates.composite_of_product (factor := 3049) (cofactor := 14419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3315 2) := by change ¬ Nat.Prime 43956901; exact PrattCertificates.composite_of_product (factor := 2161) (cofactor := 20341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3315 3) := by change ¬ Nat.Prime 43950271; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 316189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3315 : row 3315 = 0 := (row_3315.checked).trans rfl
theorem model_count_3315 : modelRow 3315 = 0 := by decide
theorem modelrow_agrees_3315 : modelRow 3315 = row 3315 := model_count_3315.trans row_count_3315.symm
def row_3316 : RowCertificate 3316 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3316 1) := by change Nat.Prime 43990057; exact ProjectEulerSolutions.P58PrimeNodes.p_43990057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3316 2) := by change ¬ Nat.Prime 43983425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8796685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3316 3) := by change ¬ Nat.Prime 43976793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14658931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3316 : row 3316 = 1 := (row_3316.checked).trans rfl
theorem model_count_3316 : modelRow 3316 = 1 := by decide
theorem modelrow_agrees_3316 : modelRow 3316 = row 3316 := model_count_3316.trans row_count_3316.symm
def row_3317 : RowCertificate 3317 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3317 1) := by change ¬ Nat.Prime 44016591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14672197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3317 2) := by change ¬ Nat.Prime 44009957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2588821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3317 3) := by change ¬ Nat.Prime 44003323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6286189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3317 : row 3317 = 0 := (row_3317.checked).trans rfl
theorem model_count_3317 : modelRow 3317 = 0 := by decide
theorem modelrow_agrees_3317 : modelRow 3317 = row 3317 := model_count_3317.trans row_count_3317.symm
def row_3318 : RowCertificate 3318 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3318 1) := by change Nat.Prime 44043133; exact ProjectEulerSolutions.P58PrimeNodes.p_44043133.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3318 2) := by change Nat.Prime 44036497; exact ProjectEulerSolutions.P58PrimeNodes.p_44036497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3318 3) := by change ¬ Nat.Prime 44029861; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 721801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3318 : row 3318 = 2 := (row_3318.checked).trans rfl
theorem model_count_3318 : modelRow 3318 = 2 := by decide
theorem modelrow_agrees_3318 : modelRow 3318 = row 3318 := model_count_3318.trans row_count_3318.symm
def row_3319 : RowCertificate 3319 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3319 1) := by change ¬ Nat.Prime 44069683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6295669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3319 2) := by change ¬ Nat.Prime 44063045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8812609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3319 3) := by change ¬ Nat.Prime 44056407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14685469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3319 : row 3319 = 0 := (row_3319.checked).trans rfl
theorem model_count_3319 : modelRow 3319 = 0 := by decide
theorem modelrow_agrees_3319 : modelRow 3319 = row 3319 := model_count_3319.trans row_count_3319.symm
def row_3320 : RowCertificate 3320 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3320 1) := by change ¬ Nat.Prime 44096241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14698747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3320 2) := by change ¬ Nat.Prime 44089601; exact PrattCertificates.composite_of_product (factor := 773) (cofactor := 57037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3320 3) := by change ¬ Nat.Prime 44082961; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3390997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3320 : row 3320 = 0 := (row_3320.checked).trans rfl
theorem model_count_3320 : modelRow 3320 = 0 := by decide
theorem modelrow_agrees_3320 : modelRow 3320 = row 3320 := model_count_3320.trans row_count_3320.symm
def row_3321 : RowCertificate 3321 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3321 1) := by change ¬ Nat.Prime 44122807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2322253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3321 2) := by change ¬ Nat.Prime 44116165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8823233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3321 3) := by change ¬ Nat.Prime 44109523; exact PrattCertificates.composite_of_product (factor := 2593) (cofactor := 17011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3321 : row 3321 = 0 := (row_3321.checked).trans rfl
theorem model_count_3321 : modelRow 3321 = 0 := by decide
theorem modelrow_agrees_3321 : modelRow 3321 = row 3321 := model_count_3321.trans row_count_3321.symm
def row_3322 : RowCertificate 3322 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3322 1) := by change Nat.Prime 44149381; exact ProjectEulerSolutions.P58PrimeNodes.p_44149381.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3322 2) := by change ¬ Nat.Prime 44142737; exact PrattCertificates.composite_of_product (factor := 3181) (cofactor := 13877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3322 3) := by change ¬ Nat.Prime 44136093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14712031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3322 : row 3322 = 1 := (row_3322.checked).trans rfl
theorem model_count_3322 : modelRow 3322 = 1 := by decide
theorem modelrow_agrees_3322 : modelRow 3322 = row 3322 := model_count_3322.trans row_count_3322.symm
def row_3323 : RowCertificate 3323 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3323 1) := by change ¬ Nat.Prime 44175963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14725321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3323 2) := by change Nat.Prime 44169317; exact ProjectEulerSolutions.P58PrimeNodes.p_44169317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3323 3) := by change ¬ Nat.Prime 44162671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6308953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3323 : row 3323 = 1 := (row_3323.checked).trans rfl
theorem model_count_3323 : modelRow 3323 = 1 := by decide
theorem modelrow_agrees_3323 : modelRow 3323 = row 3323 := model_count_3323.trans row_count_3323.symm
def row_3324 : RowCertificate 3324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3324 1) := by change ¬ Nat.Prime 44202553; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 429151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3324 2) := by change ¬ Nat.Prime 44195905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8839181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3324 3) := by change ¬ Nat.Prime 44189257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6312751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3324 : row 3324 = 0 := (row_3324.checked).trans rfl
theorem model_count_3324 : modelRow 3324 = 0 := by decide
theorem modelrow_agrees_3324 : modelRow 3324 = row 3324 := model_count_3324.trans row_count_3324.symm
def row_3325 : RowCertificate 3325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3325 1) := by change ¬ Nat.Prime 44229151; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 198337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3325 2) := by change ¬ Nat.Prime 44222501; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 27281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3325 3) := by change ¬ Nat.Prime 44215851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14738617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3325 : row 3325 = 0 := (row_3325.checked).trans rfl
theorem model_count_3325 : modelRow 3325 = 0 := by decide
theorem modelrow_agrees_3325 : modelRow 3325 = row 3325 := model_count_3325.trans row_count_3325.symm
def row_3326 : RowCertificate 3326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3326 1) := by change ¬ Nat.Prime 44255757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14751919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3326 2) := by change ¬ Nat.Prime 44249105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8849821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3326 3) := by change ¬ Nat.Prime 44242453; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 606061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3326 : row 3326 = 0 := (row_3326.checked).trans rfl
theorem model_count_3326 : modelRow 3326 = 0 := by decide
theorem modelrow_agrees_3326 : modelRow 3326 = row 3326 := model_count_3326.trans row_count_3326.symm
def row_3327 : RowCertificate 3327 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3327 1) := by change ¬ Nat.Prime 44282371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 6326053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3327 2) := by change ¬ Nat.Prime 44275717; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1196641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3327 3) := by change Nat.Prime 44269063; exact ProjectEulerSolutions.P58PrimeNodes.p_44269063.prime
    exact decide_eq_true prime
theorem row_count_3327 : row 3327 = 1 := (row_3327.checked).trans rfl
theorem model_count_3327 : modelRow 3327 = 1 := by decide
theorem modelrow_agrees_3327 : modelRow 3327 = row 3327 := model_count_3327.trans row_count_3327.symm
def row_3328 : RowCertificate 3328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3328 1) := by change ¬ Nat.Prime 44308993; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 64123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3328 2) := by change ¬ Nat.Prime 44302337; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 438637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3328 3) := by change ¬ Nat.Prime 44295681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 14765227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3328 : row 3328 = 0 := (row_3328.checked).trans rfl
theorem model_count_3328 : modelRow 3328 = 0 := by decide
theorem modelrow_agrees_3328 : modelRow 3328 = row 3328 := model_count_3328.trans row_count_3328.symm
private def atomic_3201 : CountSegment 1 10 13120 3200 3201 1542 1542 := CountSegment.single (first := 3200) (initial := 1542) row_count_3201 (by decide)
private theorem atomic_3201_agreement : RowAgreement 3200 3201 := RowAgreement.single (first := 3200) modelrow_agrees_3201
private def atomic_3202 : CountSegment 1 10 13120 3201 3202 1542 1542 := CountSegment.single (first := 3201) (initial := 1542) row_count_3202 (by decide)
private theorem atomic_3202_agreement : RowAgreement 3201 3202 := RowAgreement.single (first := 3201) modelrow_agrees_3202
private def atomic_3203 : CountSegment 1 10 13120 3202 3203 1542 1542 := CountSegment.single (first := 3202) (initial := 1542) row_count_3203 (by decide)
private theorem atomic_3203_agreement : RowAgreement 3202 3203 := RowAgreement.single (first := 3202) modelrow_agrees_3203
private def atomic_3204 : CountSegment 1 10 13120 3203 3204 1542 1543 := CountSegment.single (first := 3203) (initial := 1542) row_count_3204 (by decide)
private theorem atomic_3204_agreement : RowAgreement 3203 3204 := RowAgreement.single (first := 3203) modelrow_agrees_3204
private def atomic_3205 : CountSegment 1 10 13120 3204 3205 1543 1543 := CountSegment.single (first := 3204) (initial := 1543) row_count_3205 (by decide)
private theorem atomic_3205_agreement : RowAgreement 3204 3205 := RowAgreement.single (first := 3204) modelrow_agrees_3205
private def atomic_3206 : CountSegment 1 10 13120 3205 3206 1543 1543 := CountSegment.single (first := 3205) (initial := 1543) row_count_3206 (by decide)
private theorem atomic_3206_agreement : RowAgreement 3205 3206 := RowAgreement.single (first := 3205) modelrow_agrees_3206
private def atomic_3207 : CountSegment 1 10 13120 3206 3207 1543 1543 := CountSegment.single (first := 3206) (initial := 1543) row_count_3207 (by decide)
private theorem atomic_3207_agreement : RowAgreement 3206 3207 := RowAgreement.single (first := 3206) modelrow_agrees_3207
private def atomic_3208 : CountSegment 1 10 13120 3207 3208 1543 1543 := CountSegment.single (first := 3207) (initial := 1543) row_count_3208 (by decide)
private theorem atomic_3208_agreement : RowAgreement 3207 3208 := RowAgreement.single (first := 3207) modelrow_agrees_3208
private def atomic_3209 : CountSegment 1 10 13120 3208 3209 1543 1544 := CountSegment.single (first := 3208) (initial := 1543) row_count_3209 (by decide)
private theorem atomic_3209_agreement : RowAgreement 3208 3209 := RowAgreement.single (first := 3208) modelrow_agrees_3209
private def atomic_3210 : CountSegment 1 10 13120 3209 3210 1544 1546 := CountSegment.single (first := 3209) (initial := 1544) row_count_3210 (by decide)
private theorem atomic_3210_agreement : RowAgreement 3209 3210 := RowAgreement.single (first := 3209) modelrow_agrees_3210
private def atomic_3211 : CountSegment 1 10 13120 3210 3211 1546 1546 := CountSegment.single (first := 3210) (initial := 1546) row_count_3211 (by decide)
private theorem atomic_3211_agreement : RowAgreement 3210 3211 := RowAgreement.single (first := 3210) modelrow_agrees_3211
private def atomic_3212 : CountSegment 1 10 13120 3211 3212 1546 1546 := CountSegment.single (first := 3211) (initial := 1546) row_count_3212 (by decide)
private theorem atomic_3212_agreement : RowAgreement 3211 3212 := RowAgreement.single (first := 3211) modelrow_agrees_3212
private def atomic_3213 : CountSegment 1 10 13120 3212 3213 1546 1547 := CountSegment.single (first := 3212) (initial := 1546) row_count_3213 (by decide)
private theorem atomic_3213_agreement : RowAgreement 3212 3213 := RowAgreement.single (first := 3212) modelrow_agrees_3213
private def atomic_3214 : CountSegment 1 10 13120 3213 3214 1547 1547 := CountSegment.single (first := 3213) (initial := 1547) row_count_3214 (by decide)
private theorem atomic_3214_agreement : RowAgreement 3213 3214 := RowAgreement.single (first := 3213) modelrow_agrees_3214
private def atomic_3215 : CountSegment 1 10 13120 3214 3215 1547 1547 := CountSegment.single (first := 3214) (initial := 1547) row_count_3215 (by decide)
private theorem atomic_3215_agreement : RowAgreement 3214 3215 := RowAgreement.single (first := 3214) modelrow_agrees_3215
private def atomic_3216 : CountSegment 1 10 13120 3215 3216 1547 1547 := CountSegment.single (first := 3215) (initial := 1547) row_count_3216 (by decide)
private theorem atomic_3216_agreement : RowAgreement 3215 3216 := RowAgreement.single (first := 3215) modelrow_agrees_3216
private def atomic_3217 : CountSegment 1 10 13120 3216 3217 1547 1549 := CountSegment.single (first := 3216) (initial := 1547) row_count_3217 (by decide)
private theorem atomic_3217_agreement : RowAgreement 3216 3217 := RowAgreement.single (first := 3216) modelrow_agrees_3217
private def atomic_3218 : CountSegment 1 10 13120 3217 3218 1549 1549 := CountSegment.single (first := 3217) (initial := 1549) row_count_3218 (by decide)
private theorem atomic_3218_agreement : RowAgreement 3217 3218 := RowAgreement.single (first := 3217) modelrow_agrees_3218
private def atomic_3219 : CountSegment 1 10 13120 3218 3219 1549 1549 := CountSegment.single (first := 3218) (initial := 1549) row_count_3219 (by decide)
private theorem atomic_3219_agreement : RowAgreement 3218 3219 := RowAgreement.single (first := 3218) modelrow_agrees_3219
private def atomic_3220 : CountSegment 1 10 13120 3219 3220 1549 1550 := CountSegment.single (first := 3219) (initial := 1549) row_count_3220 (by decide)
private theorem atomic_3220_agreement : RowAgreement 3219 3220 := RowAgreement.single (first := 3219) modelrow_agrees_3220
private def atomic_3221 : CountSegment 1 10 13120 3220 3221 1550 1551 := CountSegment.single (first := 3220) (initial := 1550) row_count_3221 (by decide)
private theorem atomic_3221_agreement : RowAgreement 3220 3221 := RowAgreement.single (first := 3220) modelrow_agrees_3221
private def atomic_3222 : CountSegment 1 10 13120 3221 3222 1551 1551 := CountSegment.single (first := 3221) (initial := 1551) row_count_3222 (by decide)
private theorem atomic_3222_agreement : RowAgreement 3221 3222 := RowAgreement.single (first := 3221) modelrow_agrees_3222
private def atomic_3223 : CountSegment 1 10 13120 3222 3223 1551 1552 := CountSegment.single (first := 3222) (initial := 1551) row_count_3223 (by decide)
private theorem atomic_3223_agreement : RowAgreement 3222 3223 := RowAgreement.single (first := 3222) modelrow_agrees_3223
private def atomic_3224 : CountSegment 1 10 13120 3223 3224 1552 1553 := CountSegment.single (first := 3223) (initial := 1552) row_count_3224 (by decide)
private theorem atomic_3224_agreement : RowAgreement 3223 3224 := RowAgreement.single (first := 3223) modelrow_agrees_3224
private def atomic_3225 : CountSegment 1 10 13120 3224 3225 1553 1554 := CountSegment.single (first := 3224) (initial := 1553) row_count_3225 (by decide)
private theorem atomic_3225_agreement : RowAgreement 3224 3225 := RowAgreement.single (first := 3224) modelrow_agrees_3225
private def atomic_3226 : CountSegment 1 10 13120 3225 3226 1554 1554 := CountSegment.single (first := 3225) (initial := 1554) row_count_3226 (by decide)
private theorem atomic_3226_agreement : RowAgreement 3225 3226 := RowAgreement.single (first := 3225) modelrow_agrees_3226
private def atomic_3227 : CountSegment 1 10 13120 3226 3227 1554 1554 := CountSegment.single (first := 3226) (initial := 1554) row_count_3227 (by decide)
private theorem atomic_3227_agreement : RowAgreement 3226 3227 := RowAgreement.single (first := 3226) modelrow_agrees_3227
private def atomic_3228 : CountSegment 1 10 13120 3227 3228 1554 1555 := CountSegment.single (first := 3227) (initial := 1554) row_count_3228 (by decide)
private theorem atomic_3228_agreement : RowAgreement 3227 3228 := RowAgreement.single (first := 3227) modelrow_agrees_3228
private def atomic_3229 : CountSegment 1 10 13120 3228 3229 1555 1555 := CountSegment.single (first := 3228) (initial := 1555) row_count_3229 (by decide)
private theorem atomic_3229_agreement : RowAgreement 3228 3229 := RowAgreement.single (first := 3228) modelrow_agrees_3229
private def atomic_3230 : CountSegment 1 10 13120 3229 3230 1555 1557 := CountSegment.single (first := 3229) (initial := 1555) row_count_3230 (by decide)
private theorem atomic_3230_agreement : RowAgreement 3229 3230 := RowAgreement.single (first := 3229) modelrow_agrees_3230
private def atomic_3231 : CountSegment 1 10 13120 3230 3231 1557 1558 := CountSegment.single (first := 3230) (initial := 1557) row_count_3231 (by decide)
private theorem atomic_3231_agreement : RowAgreement 3230 3231 := RowAgreement.single (first := 3230) modelrow_agrees_3231
private def atomic_3232 : CountSegment 1 10 13120 3231 3232 1558 1558 := CountSegment.single (first := 3231) (initial := 1558) row_count_3232 (by decide)
private theorem atomic_3232_agreement : RowAgreement 3231 3232 := RowAgreement.single (first := 3231) modelrow_agrees_3232
private def atomic_3233 : CountSegment 1 10 13120 3232 3233 1558 1558 := CountSegment.single (first := 3232) (initial := 1558) row_count_3233 (by decide)
private theorem atomic_3233_agreement : RowAgreement 3232 3233 := RowAgreement.single (first := 3232) modelrow_agrees_3233
private def atomic_3234 : CountSegment 1 10 13120 3233 3234 1558 1558 := CountSegment.single (first := 3233) (initial := 1558) row_count_3234 (by decide)
private theorem atomic_3234_agreement : RowAgreement 3233 3234 := RowAgreement.single (first := 3233) modelrow_agrees_3234
private def atomic_3235 : CountSegment 1 10 13120 3234 3235 1558 1558 := CountSegment.single (first := 3234) (initial := 1558) row_count_3235 (by decide)
private theorem atomic_3235_agreement : RowAgreement 3234 3235 := RowAgreement.single (first := 3234) modelrow_agrees_3235
private def atomic_3236 : CountSegment 1 10 13120 3235 3236 1558 1558 := CountSegment.single (first := 3235) (initial := 1558) row_count_3236 (by decide)
private theorem atomic_3236_agreement : RowAgreement 3235 3236 := RowAgreement.single (first := 3235) modelrow_agrees_3236
private def atomic_3237 : CountSegment 1 10 13120 3236 3237 1558 1558 := CountSegment.single (first := 3236) (initial := 1558) row_count_3237 (by decide)
private theorem atomic_3237_agreement : RowAgreement 3236 3237 := RowAgreement.single (first := 3236) modelrow_agrees_3237
private def atomic_3238 : CountSegment 1 10 13120 3237 3238 1558 1558 := CountSegment.single (first := 3237) (initial := 1558) row_count_3238 (by decide)
private theorem atomic_3238_agreement : RowAgreement 3237 3238 := RowAgreement.single (first := 3237) modelrow_agrees_3238
private def atomic_3239 : CountSegment 1 10 13120 3238 3239 1558 1558 := CountSegment.single (first := 3238) (initial := 1558) row_count_3239 (by decide)
private theorem atomic_3239_agreement : RowAgreement 3238 3239 := RowAgreement.single (first := 3238) modelrow_agrees_3239
private def atomic_3240 : CountSegment 1 10 13120 3239 3240 1558 1559 := CountSegment.single (first := 3239) (initial := 1558) row_count_3240 (by decide)
private theorem atomic_3240_agreement : RowAgreement 3239 3240 := RowAgreement.single (first := 3239) modelrow_agrees_3240
private def atomic_3241 : CountSegment 1 10 13120 3240 3241 1559 1560 := CountSegment.single (first := 3240) (initial := 1559) row_count_3241 (by decide)
private theorem atomic_3241_agreement : RowAgreement 3240 3241 := RowAgreement.single (first := 3240) modelrow_agrees_3241
private def atomic_3242 : CountSegment 1 10 13120 3241 3242 1560 1560 := CountSegment.single (first := 3241) (initial := 1560) row_count_3242 (by decide)
private theorem atomic_3242_agreement : RowAgreement 3241 3242 := RowAgreement.single (first := 3241) modelrow_agrees_3242
private def atomic_3243 : CountSegment 1 10 13120 3242 3243 1560 1560 := CountSegment.single (first := 3242) (initial := 1560) row_count_3243 (by decide)
private theorem atomic_3243_agreement : RowAgreement 3242 3243 := RowAgreement.single (first := 3242) modelrow_agrees_3243
private def atomic_3244 : CountSegment 1 10 13120 3243 3244 1560 1560 := CountSegment.single (first := 3243) (initial := 1560) row_count_3244 (by decide)
private theorem atomic_3244_agreement : RowAgreement 3243 3244 := RowAgreement.single (first := 3243) modelrow_agrees_3244
private def atomic_3245 : CountSegment 1 10 13120 3244 3245 1560 1561 := CountSegment.single (first := 3244) (initial := 1560) row_count_3245 (by decide)
private theorem atomic_3245_agreement : RowAgreement 3244 3245 := RowAgreement.single (first := 3244) modelrow_agrees_3245
private def atomic_3246 : CountSegment 1 10 13120 3245 3246 1561 1561 := CountSegment.single (first := 3245) (initial := 1561) row_count_3246 (by decide)
private theorem atomic_3246_agreement : RowAgreement 3245 3246 := RowAgreement.single (first := 3245) modelrow_agrees_3246
private def atomic_3247 : CountSegment 1 10 13120 3246 3247 1561 1561 := CountSegment.single (first := 3246) (initial := 1561) row_count_3247 (by decide)
private theorem atomic_3247_agreement : RowAgreement 3246 3247 := RowAgreement.single (first := 3246) modelrow_agrees_3247
private def atomic_3248 : CountSegment 1 10 13120 3247 3248 1561 1562 := CountSegment.single (first := 3247) (initial := 1561) row_count_3248 (by decide)
private theorem atomic_3248_agreement : RowAgreement 3247 3248 := RowAgreement.single (first := 3247) modelrow_agrees_3248
private def atomic_3249 : CountSegment 1 10 13120 3248 3249 1562 1563 := CountSegment.single (first := 3248) (initial := 1562) row_count_3249 (by decide)
private theorem atomic_3249_agreement : RowAgreement 3248 3249 := RowAgreement.single (first := 3248) modelrow_agrees_3249
private def atomic_3250 : CountSegment 1 10 13120 3249 3250 1563 1563 := CountSegment.single (first := 3249) (initial := 1563) row_count_3250 (by decide)
private theorem atomic_3250_agreement : RowAgreement 3249 3250 := RowAgreement.single (first := 3249) modelrow_agrees_3250
private def atomic_3251 : CountSegment 1 10 13120 3250 3251 1563 1563 := CountSegment.single (first := 3250) (initial := 1563) row_count_3251 (by decide)
private theorem atomic_3251_agreement : RowAgreement 3250 3251 := RowAgreement.single (first := 3250) modelrow_agrees_3251
private def atomic_3252 : CountSegment 1 10 13120 3251 3252 1563 1563 := CountSegment.single (first := 3251) (initial := 1563) row_count_3252 (by decide)
private theorem atomic_3252_agreement : RowAgreement 3251 3252 := RowAgreement.single (first := 3251) modelrow_agrees_3252
private def atomic_3253 : CountSegment 1 10 13120 3252 3253 1563 1563 := CountSegment.single (first := 3252) (initial := 1563) row_count_3253 (by decide)
private theorem atomic_3253_agreement : RowAgreement 3252 3253 := RowAgreement.single (first := 3252) modelrow_agrees_3253
private def atomic_3254 : CountSegment 1 10 13120 3253 3254 1563 1563 := CountSegment.single (first := 3253) (initial := 1563) row_count_3254 (by decide)
private theorem atomic_3254_agreement : RowAgreement 3253 3254 := RowAgreement.single (first := 3253) modelrow_agrees_3254
private def atomic_3255 : CountSegment 1 10 13120 3254 3255 1563 1564 := CountSegment.single (first := 3254) (initial := 1563) row_count_3255 (by decide)
private theorem atomic_3255_agreement : RowAgreement 3254 3255 := RowAgreement.single (first := 3254) modelrow_agrees_3255
private def atomic_3256 : CountSegment 1 10 13120 3255 3256 1564 1564 := CountSegment.single (first := 3255) (initial := 1564) row_count_3256 (by decide)
private theorem atomic_3256_agreement : RowAgreement 3255 3256 := RowAgreement.single (first := 3255) modelrow_agrees_3256
private def atomic_3257 : CountSegment 1 10 13120 3256 3257 1564 1565 := CountSegment.single (first := 3256) (initial := 1564) row_count_3257 (by decide)
private theorem atomic_3257_agreement : RowAgreement 3256 3257 := RowAgreement.single (first := 3256) modelrow_agrees_3257
private def atomic_3258 : CountSegment 1 10 13120 3257 3258 1565 1565 := CountSegment.single (first := 3257) (initial := 1565) row_count_3258 (by decide)
private theorem atomic_3258_agreement : RowAgreement 3257 3258 := RowAgreement.single (first := 3257) modelrow_agrees_3258
private def atomic_3259 : CountSegment 1 10 13120 3258 3259 1565 1565 := CountSegment.single (first := 3258) (initial := 1565) row_count_3259 (by decide)
private theorem atomic_3259_agreement : RowAgreement 3258 3259 := RowAgreement.single (first := 3258) modelrow_agrees_3259
private def atomic_3260 : CountSegment 1 10 13120 3259 3260 1565 1565 := CountSegment.single (first := 3259) (initial := 1565) row_count_3260 (by decide)
private theorem atomic_3260_agreement : RowAgreement 3259 3260 := RowAgreement.single (first := 3259) modelrow_agrees_3260
private def atomic_3261 : CountSegment 1 10 13120 3260 3261 1565 1565 := CountSegment.single (first := 3260) (initial := 1565) row_count_3261 (by decide)
private theorem atomic_3261_agreement : RowAgreement 3260 3261 := RowAgreement.single (first := 3260) modelrow_agrees_3261
private def atomic_3262 : CountSegment 1 10 13120 3261 3262 1565 1565 := CountSegment.single (first := 3261) (initial := 1565) row_count_3262 (by decide)
private theorem atomic_3262_agreement : RowAgreement 3261 3262 := RowAgreement.single (first := 3261) modelrow_agrees_3262
private def atomic_3263 : CountSegment 1 10 13120 3262 3263 1565 1565 := CountSegment.single (first := 3262) (initial := 1565) row_count_3263 (by decide)
private theorem atomic_3263_agreement : RowAgreement 3262 3263 := RowAgreement.single (first := 3262) modelrow_agrees_3263
private def atomic_3264 : CountSegment 1 10 13120 3263 3264 1565 1565 := CountSegment.single (first := 3263) (initial := 1565) row_count_3264 (by decide)
private theorem atomic_3264_agreement : RowAgreement 3263 3264 := RowAgreement.single (first := 3263) modelrow_agrees_3264
private def atomic_3265 : CountSegment 1 10 13120 3264 3265 1565 1566 := CountSegment.single (first := 3264) (initial := 1565) row_count_3265 (by decide)
private theorem atomic_3265_agreement : RowAgreement 3264 3265 := RowAgreement.single (first := 3264) modelrow_agrees_3265
private def atomic_3266 : CountSegment 1 10 13120 3265 3266 1566 1566 := CountSegment.single (first := 3265) (initial := 1566) row_count_3266 (by decide)
private theorem atomic_3266_agreement : RowAgreement 3265 3266 := RowAgreement.single (first := 3265) modelrow_agrees_3266
private def atomic_3267 : CountSegment 1 10 13120 3266 3267 1566 1566 := CountSegment.single (first := 3266) (initial := 1566) row_count_3267 (by decide)
private theorem atomic_3267_agreement : RowAgreement 3266 3267 := RowAgreement.single (first := 3266) modelrow_agrees_3267
private def atomic_3268 : CountSegment 1 10 13120 3267 3268 1566 1567 := CountSegment.single (first := 3267) (initial := 1566) row_count_3268 (by decide)
private theorem atomic_3268_agreement : RowAgreement 3267 3268 := RowAgreement.single (first := 3267) modelrow_agrees_3268
private def atomic_3269 : CountSegment 1 10 13120 3268 3269 1567 1568 := CountSegment.single (first := 3268) (initial := 1567) row_count_3269 (by decide)
private theorem atomic_3269_agreement : RowAgreement 3268 3269 := RowAgreement.single (first := 3268) modelrow_agrees_3269
private def atomic_3270 : CountSegment 1 10 13120 3269 3270 1568 1570 := CountSegment.single (first := 3269) (initial := 1568) row_count_3270 (by decide)
private theorem atomic_3270_agreement : RowAgreement 3269 3270 := RowAgreement.single (first := 3269) modelrow_agrees_3270
private def atomic_3271 : CountSegment 1 10 13120 3270 3271 1570 1570 := CountSegment.single (first := 3270) (initial := 1570) row_count_3271 (by decide)
private theorem atomic_3271_agreement : RowAgreement 3270 3271 := RowAgreement.single (first := 3270) modelrow_agrees_3271
private def atomic_3272 : CountSegment 1 10 13120 3271 3272 1570 1570 := CountSegment.single (first := 3271) (initial := 1570) row_count_3272 (by decide)
private theorem atomic_3272_agreement : RowAgreement 3271 3272 := RowAgreement.single (first := 3271) modelrow_agrees_3272
private def atomic_3273 : CountSegment 1 10 13120 3272 3273 1570 1571 := CountSegment.single (first := 3272) (initial := 1570) row_count_3273 (by decide)
private theorem atomic_3273_agreement : RowAgreement 3272 3273 := RowAgreement.single (first := 3272) modelrow_agrees_3273
private def atomic_3274 : CountSegment 1 10 13120 3273 3274 1571 1571 := CountSegment.single (first := 3273) (initial := 1571) row_count_3274 (by decide)
private theorem atomic_3274_agreement : RowAgreement 3273 3274 := RowAgreement.single (first := 3273) modelrow_agrees_3274
private def atomic_3275 : CountSegment 1 10 13120 3274 3275 1571 1572 := CountSegment.single (first := 3274) (initial := 1571) row_count_3275 (by decide)
private theorem atomic_3275_agreement : RowAgreement 3274 3275 := RowAgreement.single (first := 3274) modelrow_agrees_3275
private def atomic_3276 : CountSegment 1 10 13120 3275 3276 1572 1574 := CountSegment.single (first := 3275) (initial := 1572) row_count_3276 (by decide)
private theorem atomic_3276_agreement : RowAgreement 3275 3276 := RowAgreement.single (first := 3275) modelrow_agrees_3276
private def atomic_3277 : CountSegment 1 10 13120 3276 3277 1574 1574 := CountSegment.single (first := 3276) (initial := 1574) row_count_3277 (by decide)
private theorem atomic_3277_agreement : RowAgreement 3276 3277 := RowAgreement.single (first := 3276) modelrow_agrees_3277
private def atomic_3278 : CountSegment 1 10 13120 3277 3278 1574 1574 := CountSegment.single (first := 3277) (initial := 1574) row_count_3278 (by decide)
private theorem atomic_3278_agreement : RowAgreement 3277 3278 := RowAgreement.single (first := 3277) modelrow_agrees_3278
private def atomic_3279 : CountSegment 1 10 13120 3278 3279 1574 1574 := CountSegment.single (first := 3278) (initial := 1574) row_count_3279 (by decide)
private theorem atomic_3279_agreement : RowAgreement 3278 3279 := RowAgreement.single (first := 3278) modelrow_agrees_3279
private def atomic_3280 : CountSegment 1 10 13120 3279 3280 1574 1574 := CountSegment.single (first := 3279) (initial := 1574) row_count_3280 (by decide)
private theorem atomic_3280_agreement : RowAgreement 3279 3280 := RowAgreement.single (first := 3279) modelrow_agrees_3280
private def atomic_3281 : CountSegment 1 10 13120 3280 3281 1574 1574 := CountSegment.single (first := 3280) (initial := 1574) row_count_3281 (by decide)
private theorem atomic_3281_agreement : RowAgreement 3280 3281 := RowAgreement.single (first := 3280) modelrow_agrees_3281
private def atomic_3282 : CountSegment 1 10 13120 3281 3282 1574 1574 := CountSegment.single (first := 3281) (initial := 1574) row_count_3282 (by decide)
private theorem atomic_3282_agreement : RowAgreement 3281 3282 := RowAgreement.single (first := 3281) modelrow_agrees_3282
private def atomic_3283 : CountSegment 1 10 13120 3282 3283 1574 1574 := CountSegment.single (first := 3282) (initial := 1574) row_count_3283 (by decide)
private theorem atomic_3283_agreement : RowAgreement 3282 3283 := RowAgreement.single (first := 3282) modelrow_agrees_3283
private def atomic_3284 : CountSegment 1 10 13120 3283 3284 1574 1575 := CountSegment.single (first := 3283) (initial := 1574) row_count_3284 (by decide)
private theorem atomic_3284_agreement : RowAgreement 3283 3284 := RowAgreement.single (first := 3283) modelrow_agrees_3284
private def atomic_3285 : CountSegment 1 10 13120 3284 3285 1575 1576 := CountSegment.single (first := 3284) (initial := 1575) row_count_3285 (by decide)
private theorem atomic_3285_agreement : RowAgreement 3284 3285 := RowAgreement.single (first := 3284) modelrow_agrees_3285
private def atomic_3286 : CountSegment 1 10 13120 3285 3286 1576 1576 := CountSegment.single (first := 3285) (initial := 1576) row_count_3286 (by decide)
private theorem atomic_3286_agreement : RowAgreement 3285 3286 := RowAgreement.single (first := 3285) modelrow_agrees_3286
private def atomic_3287 : CountSegment 1 10 13120 3286 3287 1576 1577 := CountSegment.single (first := 3286) (initial := 1576) row_count_3287 (by decide)
private theorem atomic_3287_agreement : RowAgreement 3286 3287 := RowAgreement.single (first := 3286) modelrow_agrees_3287
private def atomic_3288 : CountSegment 1 10 13120 3287 3288 1577 1579 := CountSegment.single (first := 3287) (initial := 1577) row_count_3288 (by decide)
private theorem atomic_3288_agreement : RowAgreement 3287 3288 := RowAgreement.single (first := 3287) modelrow_agrees_3288
private def atomic_3289 : CountSegment 1 10 13120 3288 3289 1579 1579 := CountSegment.single (first := 3288) (initial := 1579) row_count_3289 (by decide)
private theorem atomic_3289_agreement : RowAgreement 3288 3289 := RowAgreement.single (first := 3288) modelrow_agrees_3289
private def atomic_3290 : CountSegment 1 10 13120 3289 3290 1579 1580 := CountSegment.single (first := 3289) (initial := 1579) row_count_3290 (by decide)
private theorem atomic_3290_agreement : RowAgreement 3289 3290 := RowAgreement.single (first := 3289) modelrow_agrees_3290
private def atomic_3291 : CountSegment 1 10 13120 3290 3291 1580 1580 := CountSegment.single (first := 3290) (initial := 1580) row_count_3291 (by decide)
private theorem atomic_3291_agreement : RowAgreement 3290 3291 := RowAgreement.single (first := 3290) modelrow_agrees_3291
private def atomic_3292 : CountSegment 1 10 13120 3291 3292 1580 1580 := CountSegment.single (first := 3291) (initial := 1580) row_count_3292 (by decide)
private theorem atomic_3292_agreement : RowAgreement 3291 3292 := RowAgreement.single (first := 3291) modelrow_agrees_3292
private def atomic_3293 : CountSegment 1 10 13120 3292 3293 1580 1580 := CountSegment.single (first := 3292) (initial := 1580) row_count_3293 (by decide)
private theorem atomic_3293_agreement : RowAgreement 3292 3293 := RowAgreement.single (first := 3292) modelrow_agrees_3293
private def atomic_3294 : CountSegment 1 10 13120 3293 3294 1580 1580 := CountSegment.single (first := 3293) (initial := 1580) row_count_3294 (by decide)
private theorem atomic_3294_agreement : RowAgreement 3293 3294 := RowAgreement.single (first := 3293) modelrow_agrees_3294
private def atomic_3295 : CountSegment 1 10 13120 3294 3295 1580 1582 := CountSegment.single (first := 3294) (initial := 1580) row_count_3295 (by decide)
private theorem atomic_3295_agreement : RowAgreement 3294 3295 := RowAgreement.single (first := 3294) modelrow_agrees_3295
private def atomic_3296 : CountSegment 1 10 13120 3295 3296 1582 1582 := CountSegment.single (first := 3295) (initial := 1582) row_count_3296 (by decide)
private theorem atomic_3296_agreement : RowAgreement 3295 3296 := RowAgreement.single (first := 3295) modelrow_agrees_3296
private def atomic_3297 : CountSegment 1 10 13120 3296 3297 1582 1583 := CountSegment.single (first := 3296) (initial := 1582) row_count_3297 (by decide)
private theorem atomic_3297_agreement : RowAgreement 3296 3297 := RowAgreement.single (first := 3296) modelrow_agrees_3297
private def atomic_3298 : CountSegment 1 10 13120 3297 3298 1583 1583 := CountSegment.single (first := 3297) (initial := 1583) row_count_3298 (by decide)
private theorem atomic_3298_agreement : RowAgreement 3297 3298 := RowAgreement.single (first := 3297) modelrow_agrees_3298
private def atomic_3299 : CountSegment 1 10 13120 3298 3299 1583 1583 := CountSegment.single (first := 3298) (initial := 1583) row_count_3299 (by decide)
private theorem atomic_3299_agreement : RowAgreement 3298 3299 := RowAgreement.single (first := 3298) modelrow_agrees_3299
private def atomic_3300 : CountSegment 1 10 13120 3299 3300 1583 1584 := CountSegment.single (first := 3299) (initial := 1583) row_count_3300 (by decide)
private theorem atomic_3300_agreement : RowAgreement 3299 3300 := RowAgreement.single (first := 3299) modelrow_agrees_3300
private def atomic_3301 : CountSegment 1 10 13120 3300 3301 1584 1585 := CountSegment.single (first := 3300) (initial := 1584) row_count_3301 (by decide)
private theorem atomic_3301_agreement : RowAgreement 3300 3301 := RowAgreement.single (first := 3300) modelrow_agrees_3301
private def atomic_3302 : CountSegment 1 10 13120 3301 3302 1585 1586 := CountSegment.single (first := 3301) (initial := 1585) row_count_3302 (by decide)
private theorem atomic_3302_agreement : RowAgreement 3301 3302 := RowAgreement.single (first := 3301) modelrow_agrees_3302
private def atomic_3303 : CountSegment 1 10 13120 3302 3303 1586 1586 := CountSegment.single (first := 3302) (initial := 1586) row_count_3303 (by decide)
private theorem atomic_3303_agreement : RowAgreement 3302 3303 := RowAgreement.single (first := 3302) modelrow_agrees_3303
private def atomic_3304 : CountSegment 1 10 13120 3303 3304 1586 1586 := CountSegment.single (first := 3303) (initial := 1586) row_count_3304 (by decide)
private theorem atomic_3304_agreement : RowAgreement 3303 3304 := RowAgreement.single (first := 3303) modelrow_agrees_3304
private def atomic_3305 : CountSegment 1 10 13120 3304 3305 1586 1586 := CountSegment.single (first := 3304) (initial := 1586) row_count_3305 (by decide)
private theorem atomic_3305_agreement : RowAgreement 3304 3305 := RowAgreement.single (first := 3304) modelrow_agrees_3305
private def atomic_3306 : CountSegment 1 10 13120 3305 3306 1586 1586 := CountSegment.single (first := 3305) (initial := 1586) row_count_3306 (by decide)
private theorem atomic_3306_agreement : RowAgreement 3305 3306 := RowAgreement.single (first := 3305) modelrow_agrees_3306
private def atomic_3307 : CountSegment 1 10 13120 3306 3307 1586 1588 := CountSegment.single (first := 3306) (initial := 1586) row_count_3307 (by decide)
private theorem atomic_3307_agreement : RowAgreement 3306 3307 := RowAgreement.single (first := 3306) modelrow_agrees_3307
private def atomic_3308 : CountSegment 1 10 13120 3307 3308 1588 1588 := CountSegment.single (first := 3307) (initial := 1588) row_count_3308 (by decide)
private theorem atomic_3308_agreement : RowAgreement 3307 3308 := RowAgreement.single (first := 3307) modelrow_agrees_3308
private def atomic_3309 : CountSegment 1 10 13120 3308 3309 1588 1588 := CountSegment.single (first := 3308) (initial := 1588) row_count_3309 (by decide)
private theorem atomic_3309_agreement : RowAgreement 3308 3309 := RowAgreement.single (first := 3308) modelrow_agrees_3309
private def atomic_3310 : CountSegment 1 10 13120 3309 3310 1588 1588 := CountSegment.single (first := 3309) (initial := 1588) row_count_3310 (by decide)
private theorem atomic_3310_agreement : RowAgreement 3309 3310 := RowAgreement.single (first := 3309) modelrow_agrees_3310
private def atomic_3311 : CountSegment 1 10 13120 3310 3311 1588 1588 := CountSegment.single (first := 3310) (initial := 1588) row_count_3311 (by decide)
private theorem atomic_3311_agreement : RowAgreement 3310 3311 := RowAgreement.single (first := 3310) modelrow_agrees_3311
private def atomic_3312 : CountSegment 1 10 13120 3311 3312 1588 1588 := CountSegment.single (first := 3311) (initial := 1588) row_count_3312 (by decide)
private theorem atomic_3312_agreement : RowAgreement 3311 3312 := RowAgreement.single (first := 3311) modelrow_agrees_3312
private def atomic_3313 : CountSegment 1 10 13120 3312 3313 1588 1588 := CountSegment.single (first := 3312) (initial := 1588) row_count_3313 (by decide)
private theorem atomic_3313_agreement : RowAgreement 3312 3313 := RowAgreement.single (first := 3312) modelrow_agrees_3313
private def atomic_3314 : CountSegment 1 10 13120 3313 3314 1588 1589 := CountSegment.single (first := 3313) (initial := 1588) row_count_3314 (by decide)
private theorem atomic_3314_agreement : RowAgreement 3313 3314 := RowAgreement.single (first := 3313) modelrow_agrees_3314
private def atomic_3315 : CountSegment 1 10 13120 3314 3315 1589 1589 := CountSegment.single (first := 3314) (initial := 1589) row_count_3315 (by decide)
private theorem atomic_3315_agreement : RowAgreement 3314 3315 := RowAgreement.single (first := 3314) modelrow_agrees_3315
private def atomic_3316 : CountSegment 1 10 13120 3315 3316 1589 1590 := CountSegment.single (first := 3315) (initial := 1589) row_count_3316 (by decide)
private theorem atomic_3316_agreement : RowAgreement 3315 3316 := RowAgreement.single (first := 3315) modelrow_agrees_3316
private def atomic_3317 : CountSegment 1 10 13120 3316 3317 1590 1590 := CountSegment.single (first := 3316) (initial := 1590) row_count_3317 (by decide)
private theorem atomic_3317_agreement : RowAgreement 3316 3317 := RowAgreement.single (first := 3316) modelrow_agrees_3317
private def atomic_3318 : CountSegment 1 10 13120 3317 3318 1590 1592 := CountSegment.single (first := 3317) (initial := 1590) row_count_3318 (by decide)
private theorem atomic_3318_agreement : RowAgreement 3317 3318 := RowAgreement.single (first := 3317) modelrow_agrees_3318
private def atomic_3319 : CountSegment 1 10 13120 3318 3319 1592 1592 := CountSegment.single (first := 3318) (initial := 1592) row_count_3319 (by decide)
private theorem atomic_3319_agreement : RowAgreement 3318 3319 := RowAgreement.single (first := 3318) modelrow_agrees_3319
private def atomic_3320 : CountSegment 1 10 13120 3319 3320 1592 1592 := CountSegment.single (first := 3319) (initial := 1592) row_count_3320 (by decide)
private theorem atomic_3320_agreement : RowAgreement 3319 3320 := RowAgreement.single (first := 3319) modelrow_agrees_3320
private def atomic_3321 : CountSegment 1 10 13120 3320 3321 1592 1592 := CountSegment.single (first := 3320) (initial := 1592) row_count_3321 (by decide)
private theorem atomic_3321_agreement : RowAgreement 3320 3321 := RowAgreement.single (first := 3320) modelrow_agrees_3321
private def atomic_3322 : CountSegment 1 10 13120 3321 3322 1592 1593 := CountSegment.single (first := 3321) (initial := 1592) row_count_3322 (by decide)
private theorem atomic_3322_agreement : RowAgreement 3321 3322 := RowAgreement.single (first := 3321) modelrow_agrees_3322
private def atomic_3323 : CountSegment 1 10 13120 3322 3323 1593 1594 := CountSegment.single (first := 3322) (initial := 1593) row_count_3323 (by decide)
private theorem atomic_3323_agreement : RowAgreement 3322 3323 := RowAgreement.single (first := 3322) modelrow_agrees_3323
private def atomic_3324 : CountSegment 1 10 13120 3323 3324 1594 1594 := CountSegment.single (first := 3323) (initial := 1594) row_count_3324 (by decide)
private theorem atomic_3324_agreement : RowAgreement 3323 3324 := RowAgreement.single (first := 3323) modelrow_agrees_3324
private def atomic_3325 : CountSegment 1 10 13120 3324 3325 1594 1594 := CountSegment.single (first := 3324) (initial := 1594) row_count_3325 (by decide)
private theorem atomic_3325_agreement : RowAgreement 3324 3325 := RowAgreement.single (first := 3324) modelrow_agrees_3325
private def atomic_3326 : CountSegment 1 10 13120 3325 3326 1594 1594 := CountSegment.single (first := 3325) (initial := 1594) row_count_3326 (by decide)
private theorem atomic_3326_agreement : RowAgreement 3325 3326 := RowAgreement.single (first := 3325) modelrow_agrees_3326
private def atomic_3327 : CountSegment 1 10 13120 3326 3327 1594 1595 := CountSegment.single (first := 3326) (initial := 1594) row_count_3327 (by decide)
private theorem atomic_3327_agreement : RowAgreement 3326 3327 := RowAgreement.single (first := 3326) modelrow_agrees_3327
private def atomic_3328 : CountSegment 1 10 13120 3327 3328 1595 1595 := CountSegment.single (first := 3327) (initial := 1595) row_count_3328 (by decide)
private theorem atomic_3328_agreement : RowAgreement 3327 3328 := RowAgreement.single (first := 3327) modelrow_agrees_3328
private def within_025_000 : CountSegment 1 10 13120 3200 3202 1542 1542 := CountSegment.join atomic_3201 atomic_3202
private theorem within_025_000_agreement : RowAgreement 3200 3202 := RowAgreement.join atomic_3201_agreement atomic_3202_agreement
private def within_025_001 : CountSegment 1 10 13120 3202 3204 1542 1543 := CountSegment.join atomic_3203 atomic_3204
private theorem within_025_001_agreement : RowAgreement 3202 3204 := RowAgreement.join atomic_3203_agreement atomic_3204_agreement
private def within_025_002 : CountSegment 1 10 13120 3200 3204 1542 1543 := CountSegment.join within_025_000 within_025_001
private theorem within_025_002_agreement : RowAgreement 3200 3204 := RowAgreement.join within_025_000_agreement within_025_001_agreement
private def within_025_003 : CountSegment 1 10 13120 3204 3206 1543 1543 := CountSegment.join atomic_3205 atomic_3206
private theorem within_025_003_agreement : RowAgreement 3204 3206 := RowAgreement.join atomic_3205_agreement atomic_3206_agreement
private def within_025_004 : CountSegment 1 10 13120 3206 3208 1543 1543 := CountSegment.join atomic_3207 atomic_3208
private theorem within_025_004_agreement : RowAgreement 3206 3208 := RowAgreement.join atomic_3207_agreement atomic_3208_agreement
private def within_025_005 : CountSegment 1 10 13120 3204 3208 1543 1543 := CountSegment.join within_025_003 within_025_004
private theorem within_025_005_agreement : RowAgreement 3204 3208 := RowAgreement.join within_025_003_agreement within_025_004_agreement
private def within_025_006 : CountSegment 1 10 13120 3200 3208 1542 1543 := CountSegment.join within_025_002 within_025_005
private theorem within_025_006_agreement : RowAgreement 3200 3208 := RowAgreement.join within_025_002_agreement within_025_005_agreement
private def within_025_007 : CountSegment 1 10 13120 3208 3210 1543 1546 := CountSegment.join atomic_3209 atomic_3210
private theorem within_025_007_agreement : RowAgreement 3208 3210 := RowAgreement.join atomic_3209_agreement atomic_3210_agreement
private def within_025_008 : CountSegment 1 10 13120 3210 3212 1546 1546 := CountSegment.join atomic_3211 atomic_3212
private theorem within_025_008_agreement : RowAgreement 3210 3212 := RowAgreement.join atomic_3211_agreement atomic_3212_agreement
private def within_025_009 : CountSegment 1 10 13120 3208 3212 1543 1546 := CountSegment.join within_025_007 within_025_008
private theorem within_025_009_agreement : RowAgreement 3208 3212 := RowAgreement.join within_025_007_agreement within_025_008_agreement
private def within_025_010 : CountSegment 1 10 13120 3212 3214 1546 1547 := CountSegment.join atomic_3213 atomic_3214
private theorem within_025_010_agreement : RowAgreement 3212 3214 := RowAgreement.join atomic_3213_agreement atomic_3214_agreement
private def within_025_011 : CountSegment 1 10 13120 3214 3216 1547 1547 := CountSegment.join atomic_3215 atomic_3216
private theorem within_025_011_agreement : RowAgreement 3214 3216 := RowAgreement.join atomic_3215_agreement atomic_3216_agreement
private def within_025_012 : CountSegment 1 10 13120 3212 3216 1546 1547 := CountSegment.join within_025_010 within_025_011
private theorem within_025_012_agreement : RowAgreement 3212 3216 := RowAgreement.join within_025_010_agreement within_025_011_agreement
private def within_025_013 : CountSegment 1 10 13120 3208 3216 1543 1547 := CountSegment.join within_025_009 within_025_012
private theorem within_025_013_agreement : RowAgreement 3208 3216 := RowAgreement.join within_025_009_agreement within_025_012_agreement
private def within_025_014 : CountSegment 1 10 13120 3200 3216 1542 1547 := CountSegment.join within_025_006 within_025_013
private theorem within_025_014_agreement : RowAgreement 3200 3216 := RowAgreement.join within_025_006_agreement within_025_013_agreement
private def within_025_015 : CountSegment 1 10 13120 3216 3218 1547 1549 := CountSegment.join atomic_3217 atomic_3218
private theorem within_025_015_agreement : RowAgreement 3216 3218 := RowAgreement.join atomic_3217_agreement atomic_3218_agreement
private def within_025_016 : CountSegment 1 10 13120 3218 3220 1549 1550 := CountSegment.join atomic_3219 atomic_3220
private theorem within_025_016_agreement : RowAgreement 3218 3220 := RowAgreement.join atomic_3219_agreement atomic_3220_agreement
private def within_025_017 : CountSegment 1 10 13120 3216 3220 1547 1550 := CountSegment.join within_025_015 within_025_016
private theorem within_025_017_agreement : RowAgreement 3216 3220 := RowAgreement.join within_025_015_agreement within_025_016_agreement
private def within_025_018 : CountSegment 1 10 13120 3220 3222 1550 1551 := CountSegment.join atomic_3221 atomic_3222
private theorem within_025_018_agreement : RowAgreement 3220 3222 := RowAgreement.join atomic_3221_agreement atomic_3222_agreement
private def within_025_019 : CountSegment 1 10 13120 3222 3224 1551 1553 := CountSegment.join atomic_3223 atomic_3224
private theorem within_025_019_agreement : RowAgreement 3222 3224 := RowAgreement.join atomic_3223_agreement atomic_3224_agreement
private def within_025_020 : CountSegment 1 10 13120 3220 3224 1550 1553 := CountSegment.join within_025_018 within_025_019
private theorem within_025_020_agreement : RowAgreement 3220 3224 := RowAgreement.join within_025_018_agreement within_025_019_agreement
private def within_025_021 : CountSegment 1 10 13120 3216 3224 1547 1553 := CountSegment.join within_025_017 within_025_020
private theorem within_025_021_agreement : RowAgreement 3216 3224 := RowAgreement.join within_025_017_agreement within_025_020_agreement
private def within_025_022 : CountSegment 1 10 13120 3224 3226 1553 1554 := CountSegment.join atomic_3225 atomic_3226
private theorem within_025_022_agreement : RowAgreement 3224 3226 := RowAgreement.join atomic_3225_agreement atomic_3226_agreement
private def within_025_023 : CountSegment 1 10 13120 3226 3228 1554 1555 := CountSegment.join atomic_3227 atomic_3228
private theorem within_025_023_agreement : RowAgreement 3226 3228 := RowAgreement.join atomic_3227_agreement atomic_3228_agreement
private def within_025_024 : CountSegment 1 10 13120 3224 3228 1553 1555 := CountSegment.join within_025_022 within_025_023
private theorem within_025_024_agreement : RowAgreement 3224 3228 := RowAgreement.join within_025_022_agreement within_025_023_agreement
private def within_025_025 : CountSegment 1 10 13120 3228 3230 1555 1557 := CountSegment.join atomic_3229 atomic_3230
private theorem within_025_025_agreement : RowAgreement 3228 3230 := RowAgreement.join atomic_3229_agreement atomic_3230_agreement
private def within_025_026 : CountSegment 1 10 13120 3230 3232 1557 1558 := CountSegment.join atomic_3231 atomic_3232
private theorem within_025_026_agreement : RowAgreement 3230 3232 := RowAgreement.join atomic_3231_agreement atomic_3232_agreement
private def within_025_027 : CountSegment 1 10 13120 3228 3232 1555 1558 := CountSegment.join within_025_025 within_025_026
private theorem within_025_027_agreement : RowAgreement 3228 3232 := RowAgreement.join within_025_025_agreement within_025_026_agreement
private def within_025_028 : CountSegment 1 10 13120 3224 3232 1553 1558 := CountSegment.join within_025_024 within_025_027
private theorem within_025_028_agreement : RowAgreement 3224 3232 := RowAgreement.join within_025_024_agreement within_025_027_agreement
private def within_025_029 : CountSegment 1 10 13120 3216 3232 1547 1558 := CountSegment.join within_025_021 within_025_028
private theorem within_025_029_agreement : RowAgreement 3216 3232 := RowAgreement.join within_025_021_agreement within_025_028_agreement
private def within_025_030 : CountSegment 1 10 13120 3200 3232 1542 1558 := CountSegment.join within_025_014 within_025_029
private theorem within_025_030_agreement : RowAgreement 3200 3232 := RowAgreement.join within_025_014_agreement within_025_029_agreement
private def within_025_031 : CountSegment 1 10 13120 3232 3234 1558 1558 := CountSegment.join atomic_3233 atomic_3234
private theorem within_025_031_agreement : RowAgreement 3232 3234 := RowAgreement.join atomic_3233_agreement atomic_3234_agreement
private def within_025_032 : CountSegment 1 10 13120 3234 3236 1558 1558 := CountSegment.join atomic_3235 atomic_3236
private theorem within_025_032_agreement : RowAgreement 3234 3236 := RowAgreement.join atomic_3235_agreement atomic_3236_agreement
private def within_025_033 : CountSegment 1 10 13120 3232 3236 1558 1558 := CountSegment.join within_025_031 within_025_032
private theorem within_025_033_agreement : RowAgreement 3232 3236 := RowAgreement.join within_025_031_agreement within_025_032_agreement
private def within_025_034 : CountSegment 1 10 13120 3236 3238 1558 1558 := CountSegment.join atomic_3237 atomic_3238
private theorem within_025_034_agreement : RowAgreement 3236 3238 := RowAgreement.join atomic_3237_agreement atomic_3238_agreement
private def within_025_035 : CountSegment 1 10 13120 3238 3240 1558 1559 := CountSegment.join atomic_3239 atomic_3240
private theorem within_025_035_agreement : RowAgreement 3238 3240 := RowAgreement.join atomic_3239_agreement atomic_3240_agreement
private def within_025_036 : CountSegment 1 10 13120 3236 3240 1558 1559 := CountSegment.join within_025_034 within_025_035
private theorem within_025_036_agreement : RowAgreement 3236 3240 := RowAgreement.join within_025_034_agreement within_025_035_agreement
private def within_025_037 : CountSegment 1 10 13120 3232 3240 1558 1559 := CountSegment.join within_025_033 within_025_036
private theorem within_025_037_agreement : RowAgreement 3232 3240 := RowAgreement.join within_025_033_agreement within_025_036_agreement
private def within_025_038 : CountSegment 1 10 13120 3240 3242 1559 1560 := CountSegment.join atomic_3241 atomic_3242
private theorem within_025_038_agreement : RowAgreement 3240 3242 := RowAgreement.join atomic_3241_agreement atomic_3242_agreement
private def within_025_039 : CountSegment 1 10 13120 3242 3244 1560 1560 := CountSegment.join atomic_3243 atomic_3244
private theorem within_025_039_agreement : RowAgreement 3242 3244 := RowAgreement.join atomic_3243_agreement atomic_3244_agreement
private def within_025_040 : CountSegment 1 10 13120 3240 3244 1559 1560 := CountSegment.join within_025_038 within_025_039
private theorem within_025_040_agreement : RowAgreement 3240 3244 := RowAgreement.join within_025_038_agreement within_025_039_agreement
private def within_025_041 : CountSegment 1 10 13120 3244 3246 1560 1561 := CountSegment.join atomic_3245 atomic_3246
private theorem within_025_041_agreement : RowAgreement 3244 3246 := RowAgreement.join atomic_3245_agreement atomic_3246_agreement
private def within_025_042 : CountSegment 1 10 13120 3246 3248 1561 1562 := CountSegment.join atomic_3247 atomic_3248
private theorem within_025_042_agreement : RowAgreement 3246 3248 := RowAgreement.join atomic_3247_agreement atomic_3248_agreement
private def within_025_043 : CountSegment 1 10 13120 3244 3248 1560 1562 := CountSegment.join within_025_041 within_025_042
private theorem within_025_043_agreement : RowAgreement 3244 3248 := RowAgreement.join within_025_041_agreement within_025_042_agreement
private def within_025_044 : CountSegment 1 10 13120 3240 3248 1559 1562 := CountSegment.join within_025_040 within_025_043
private theorem within_025_044_agreement : RowAgreement 3240 3248 := RowAgreement.join within_025_040_agreement within_025_043_agreement
private def within_025_045 : CountSegment 1 10 13120 3232 3248 1558 1562 := CountSegment.join within_025_037 within_025_044
private theorem within_025_045_agreement : RowAgreement 3232 3248 := RowAgreement.join within_025_037_agreement within_025_044_agreement
private def within_025_046 : CountSegment 1 10 13120 3248 3250 1562 1563 := CountSegment.join atomic_3249 atomic_3250
private theorem within_025_046_agreement : RowAgreement 3248 3250 := RowAgreement.join atomic_3249_agreement atomic_3250_agreement
private def within_025_047 : CountSegment 1 10 13120 3250 3252 1563 1563 := CountSegment.join atomic_3251 atomic_3252
private theorem within_025_047_agreement : RowAgreement 3250 3252 := RowAgreement.join atomic_3251_agreement atomic_3252_agreement
private def within_025_048 : CountSegment 1 10 13120 3248 3252 1562 1563 := CountSegment.join within_025_046 within_025_047
private theorem within_025_048_agreement : RowAgreement 3248 3252 := RowAgreement.join within_025_046_agreement within_025_047_agreement
private def within_025_049 : CountSegment 1 10 13120 3252 3254 1563 1563 := CountSegment.join atomic_3253 atomic_3254
private theorem within_025_049_agreement : RowAgreement 3252 3254 := RowAgreement.join atomic_3253_agreement atomic_3254_agreement
private def within_025_050 : CountSegment 1 10 13120 3254 3256 1563 1564 := CountSegment.join atomic_3255 atomic_3256
private theorem within_025_050_agreement : RowAgreement 3254 3256 := RowAgreement.join atomic_3255_agreement atomic_3256_agreement
private def within_025_051 : CountSegment 1 10 13120 3252 3256 1563 1564 := CountSegment.join within_025_049 within_025_050
private theorem within_025_051_agreement : RowAgreement 3252 3256 := RowAgreement.join within_025_049_agreement within_025_050_agreement
private def within_025_052 : CountSegment 1 10 13120 3248 3256 1562 1564 := CountSegment.join within_025_048 within_025_051
private theorem within_025_052_agreement : RowAgreement 3248 3256 := RowAgreement.join within_025_048_agreement within_025_051_agreement
private def within_025_053 : CountSegment 1 10 13120 3256 3258 1564 1565 := CountSegment.join atomic_3257 atomic_3258
private theorem within_025_053_agreement : RowAgreement 3256 3258 := RowAgreement.join atomic_3257_agreement atomic_3258_agreement
private def within_025_054 : CountSegment 1 10 13120 3258 3260 1565 1565 := CountSegment.join atomic_3259 atomic_3260
private theorem within_025_054_agreement : RowAgreement 3258 3260 := RowAgreement.join atomic_3259_agreement atomic_3260_agreement
private def within_025_055 : CountSegment 1 10 13120 3256 3260 1564 1565 := CountSegment.join within_025_053 within_025_054
private theorem within_025_055_agreement : RowAgreement 3256 3260 := RowAgreement.join within_025_053_agreement within_025_054_agreement
private def within_025_056 : CountSegment 1 10 13120 3260 3262 1565 1565 := CountSegment.join atomic_3261 atomic_3262
private theorem within_025_056_agreement : RowAgreement 3260 3262 := RowAgreement.join atomic_3261_agreement atomic_3262_agreement
private def within_025_057 : CountSegment 1 10 13120 3262 3264 1565 1565 := CountSegment.join atomic_3263 atomic_3264
private theorem within_025_057_agreement : RowAgreement 3262 3264 := RowAgreement.join atomic_3263_agreement atomic_3264_agreement
private def within_025_058 : CountSegment 1 10 13120 3260 3264 1565 1565 := CountSegment.join within_025_056 within_025_057
private theorem within_025_058_agreement : RowAgreement 3260 3264 := RowAgreement.join within_025_056_agreement within_025_057_agreement
private def within_025_059 : CountSegment 1 10 13120 3256 3264 1564 1565 := CountSegment.join within_025_055 within_025_058
private theorem within_025_059_agreement : RowAgreement 3256 3264 := RowAgreement.join within_025_055_agreement within_025_058_agreement
private def within_025_060 : CountSegment 1 10 13120 3248 3264 1562 1565 := CountSegment.join within_025_052 within_025_059
private theorem within_025_060_agreement : RowAgreement 3248 3264 := RowAgreement.join within_025_052_agreement within_025_059_agreement
private def within_025_061 : CountSegment 1 10 13120 3232 3264 1558 1565 := CountSegment.join within_025_045 within_025_060
private theorem within_025_061_agreement : RowAgreement 3232 3264 := RowAgreement.join within_025_045_agreement within_025_060_agreement
private def within_025_062 : CountSegment 1 10 13120 3200 3264 1542 1565 := CountSegment.join within_025_030 within_025_061
private theorem within_025_062_agreement : RowAgreement 3200 3264 := RowAgreement.join within_025_030_agreement within_025_061_agreement
private def within_025_063 : CountSegment 1 10 13120 3264 3266 1565 1566 := CountSegment.join atomic_3265 atomic_3266
private theorem within_025_063_agreement : RowAgreement 3264 3266 := RowAgreement.join atomic_3265_agreement atomic_3266_agreement
private def within_025_064 : CountSegment 1 10 13120 3266 3268 1566 1567 := CountSegment.join atomic_3267 atomic_3268
private theorem within_025_064_agreement : RowAgreement 3266 3268 := RowAgreement.join atomic_3267_agreement atomic_3268_agreement
private def within_025_065 : CountSegment 1 10 13120 3264 3268 1565 1567 := CountSegment.join within_025_063 within_025_064
private theorem within_025_065_agreement : RowAgreement 3264 3268 := RowAgreement.join within_025_063_agreement within_025_064_agreement
private def within_025_066 : CountSegment 1 10 13120 3268 3270 1567 1570 := CountSegment.join atomic_3269 atomic_3270
private theorem within_025_066_agreement : RowAgreement 3268 3270 := RowAgreement.join atomic_3269_agreement atomic_3270_agreement
private def within_025_067 : CountSegment 1 10 13120 3270 3272 1570 1570 := CountSegment.join atomic_3271 atomic_3272
private theorem within_025_067_agreement : RowAgreement 3270 3272 := RowAgreement.join atomic_3271_agreement atomic_3272_agreement
private def within_025_068 : CountSegment 1 10 13120 3268 3272 1567 1570 := CountSegment.join within_025_066 within_025_067
private theorem within_025_068_agreement : RowAgreement 3268 3272 := RowAgreement.join within_025_066_agreement within_025_067_agreement
private def within_025_069 : CountSegment 1 10 13120 3264 3272 1565 1570 := CountSegment.join within_025_065 within_025_068
private theorem within_025_069_agreement : RowAgreement 3264 3272 := RowAgreement.join within_025_065_agreement within_025_068_agreement
private def within_025_070 : CountSegment 1 10 13120 3272 3274 1570 1571 := CountSegment.join atomic_3273 atomic_3274
private theorem within_025_070_agreement : RowAgreement 3272 3274 := RowAgreement.join atomic_3273_agreement atomic_3274_agreement
private def within_025_071 : CountSegment 1 10 13120 3274 3276 1571 1574 := CountSegment.join atomic_3275 atomic_3276
private theorem within_025_071_agreement : RowAgreement 3274 3276 := RowAgreement.join atomic_3275_agreement atomic_3276_agreement
private def within_025_072 : CountSegment 1 10 13120 3272 3276 1570 1574 := CountSegment.join within_025_070 within_025_071
private theorem within_025_072_agreement : RowAgreement 3272 3276 := RowAgreement.join within_025_070_agreement within_025_071_agreement
private def within_025_073 : CountSegment 1 10 13120 3276 3278 1574 1574 := CountSegment.join atomic_3277 atomic_3278
private theorem within_025_073_agreement : RowAgreement 3276 3278 := RowAgreement.join atomic_3277_agreement atomic_3278_agreement
private def within_025_074 : CountSegment 1 10 13120 3278 3280 1574 1574 := CountSegment.join atomic_3279 atomic_3280
private theorem within_025_074_agreement : RowAgreement 3278 3280 := RowAgreement.join atomic_3279_agreement atomic_3280_agreement
private def within_025_075 : CountSegment 1 10 13120 3276 3280 1574 1574 := CountSegment.join within_025_073 within_025_074
private theorem within_025_075_agreement : RowAgreement 3276 3280 := RowAgreement.join within_025_073_agreement within_025_074_agreement
private def within_025_076 : CountSegment 1 10 13120 3272 3280 1570 1574 := CountSegment.join within_025_072 within_025_075
private theorem within_025_076_agreement : RowAgreement 3272 3280 := RowAgreement.join within_025_072_agreement within_025_075_agreement
private def within_025_077 : CountSegment 1 10 13120 3264 3280 1565 1574 := CountSegment.join within_025_069 within_025_076
private theorem within_025_077_agreement : RowAgreement 3264 3280 := RowAgreement.join within_025_069_agreement within_025_076_agreement
private def within_025_078 : CountSegment 1 10 13120 3280 3282 1574 1574 := CountSegment.join atomic_3281 atomic_3282
private theorem within_025_078_agreement : RowAgreement 3280 3282 := RowAgreement.join atomic_3281_agreement atomic_3282_agreement
private def within_025_079 : CountSegment 1 10 13120 3282 3284 1574 1575 := CountSegment.join atomic_3283 atomic_3284
private theorem within_025_079_agreement : RowAgreement 3282 3284 := RowAgreement.join atomic_3283_agreement atomic_3284_agreement
private def within_025_080 : CountSegment 1 10 13120 3280 3284 1574 1575 := CountSegment.join within_025_078 within_025_079
private theorem within_025_080_agreement : RowAgreement 3280 3284 := RowAgreement.join within_025_078_agreement within_025_079_agreement
private def within_025_081 : CountSegment 1 10 13120 3284 3286 1575 1576 := CountSegment.join atomic_3285 atomic_3286
private theorem within_025_081_agreement : RowAgreement 3284 3286 := RowAgreement.join atomic_3285_agreement atomic_3286_agreement
private def within_025_082 : CountSegment 1 10 13120 3286 3288 1576 1579 := CountSegment.join atomic_3287 atomic_3288
private theorem within_025_082_agreement : RowAgreement 3286 3288 := RowAgreement.join atomic_3287_agreement atomic_3288_agreement
private def within_025_083 : CountSegment 1 10 13120 3284 3288 1575 1579 := CountSegment.join within_025_081 within_025_082
private theorem within_025_083_agreement : RowAgreement 3284 3288 := RowAgreement.join within_025_081_agreement within_025_082_agreement
private def within_025_084 : CountSegment 1 10 13120 3280 3288 1574 1579 := CountSegment.join within_025_080 within_025_083
private theorem within_025_084_agreement : RowAgreement 3280 3288 := RowAgreement.join within_025_080_agreement within_025_083_agreement
private def within_025_085 : CountSegment 1 10 13120 3288 3290 1579 1580 := CountSegment.join atomic_3289 atomic_3290
private theorem within_025_085_agreement : RowAgreement 3288 3290 := RowAgreement.join atomic_3289_agreement atomic_3290_agreement
private def within_025_086 : CountSegment 1 10 13120 3290 3292 1580 1580 := CountSegment.join atomic_3291 atomic_3292
private theorem within_025_086_agreement : RowAgreement 3290 3292 := RowAgreement.join atomic_3291_agreement atomic_3292_agreement
private def within_025_087 : CountSegment 1 10 13120 3288 3292 1579 1580 := CountSegment.join within_025_085 within_025_086
private theorem within_025_087_agreement : RowAgreement 3288 3292 := RowAgreement.join within_025_085_agreement within_025_086_agreement
private def within_025_088 : CountSegment 1 10 13120 3292 3294 1580 1580 := CountSegment.join atomic_3293 atomic_3294
private theorem within_025_088_agreement : RowAgreement 3292 3294 := RowAgreement.join atomic_3293_agreement atomic_3294_agreement
private def within_025_089 : CountSegment 1 10 13120 3294 3296 1580 1582 := CountSegment.join atomic_3295 atomic_3296
private theorem within_025_089_agreement : RowAgreement 3294 3296 := RowAgreement.join atomic_3295_agreement atomic_3296_agreement
private def within_025_090 : CountSegment 1 10 13120 3292 3296 1580 1582 := CountSegment.join within_025_088 within_025_089
private theorem within_025_090_agreement : RowAgreement 3292 3296 := RowAgreement.join within_025_088_agreement within_025_089_agreement
private def within_025_091 : CountSegment 1 10 13120 3288 3296 1579 1582 := CountSegment.join within_025_087 within_025_090
private theorem within_025_091_agreement : RowAgreement 3288 3296 := RowAgreement.join within_025_087_agreement within_025_090_agreement
private def within_025_092 : CountSegment 1 10 13120 3280 3296 1574 1582 := CountSegment.join within_025_084 within_025_091
private theorem within_025_092_agreement : RowAgreement 3280 3296 := RowAgreement.join within_025_084_agreement within_025_091_agreement
private def within_025_093 : CountSegment 1 10 13120 3264 3296 1565 1582 := CountSegment.join within_025_077 within_025_092
private theorem within_025_093_agreement : RowAgreement 3264 3296 := RowAgreement.join within_025_077_agreement within_025_092_agreement
private def within_025_094 : CountSegment 1 10 13120 3296 3298 1582 1583 := CountSegment.join atomic_3297 atomic_3298
private theorem within_025_094_agreement : RowAgreement 3296 3298 := RowAgreement.join atomic_3297_agreement atomic_3298_agreement
private def within_025_095 : CountSegment 1 10 13120 3298 3300 1583 1584 := CountSegment.join atomic_3299 atomic_3300
private theorem within_025_095_agreement : RowAgreement 3298 3300 := RowAgreement.join atomic_3299_agreement atomic_3300_agreement
private def within_025_096 : CountSegment 1 10 13120 3296 3300 1582 1584 := CountSegment.join within_025_094 within_025_095
private theorem within_025_096_agreement : RowAgreement 3296 3300 := RowAgreement.join within_025_094_agreement within_025_095_agreement
private def within_025_097 : CountSegment 1 10 13120 3300 3302 1584 1586 := CountSegment.join atomic_3301 atomic_3302
private theorem within_025_097_agreement : RowAgreement 3300 3302 := RowAgreement.join atomic_3301_agreement atomic_3302_agreement
private def within_025_098 : CountSegment 1 10 13120 3302 3304 1586 1586 := CountSegment.join atomic_3303 atomic_3304
private theorem within_025_098_agreement : RowAgreement 3302 3304 := RowAgreement.join atomic_3303_agreement atomic_3304_agreement
private def within_025_099 : CountSegment 1 10 13120 3300 3304 1584 1586 := CountSegment.join within_025_097 within_025_098
private theorem within_025_099_agreement : RowAgreement 3300 3304 := RowAgreement.join within_025_097_agreement within_025_098_agreement
private def within_025_100 : CountSegment 1 10 13120 3296 3304 1582 1586 := CountSegment.join within_025_096 within_025_099
private theorem within_025_100_agreement : RowAgreement 3296 3304 := RowAgreement.join within_025_096_agreement within_025_099_agreement
private def within_025_101 : CountSegment 1 10 13120 3304 3306 1586 1586 := CountSegment.join atomic_3305 atomic_3306
private theorem within_025_101_agreement : RowAgreement 3304 3306 := RowAgreement.join atomic_3305_agreement atomic_3306_agreement
private def within_025_102 : CountSegment 1 10 13120 3306 3308 1586 1588 := CountSegment.join atomic_3307 atomic_3308
private theorem within_025_102_agreement : RowAgreement 3306 3308 := RowAgreement.join atomic_3307_agreement atomic_3308_agreement
private def within_025_103 : CountSegment 1 10 13120 3304 3308 1586 1588 := CountSegment.join within_025_101 within_025_102
private theorem within_025_103_agreement : RowAgreement 3304 3308 := RowAgreement.join within_025_101_agreement within_025_102_agreement
private def within_025_104 : CountSegment 1 10 13120 3308 3310 1588 1588 := CountSegment.join atomic_3309 atomic_3310
private theorem within_025_104_agreement : RowAgreement 3308 3310 := RowAgreement.join atomic_3309_agreement atomic_3310_agreement
private def within_025_105 : CountSegment 1 10 13120 3310 3312 1588 1588 := CountSegment.join atomic_3311 atomic_3312
private theorem within_025_105_agreement : RowAgreement 3310 3312 := RowAgreement.join atomic_3311_agreement atomic_3312_agreement
private def within_025_106 : CountSegment 1 10 13120 3308 3312 1588 1588 := CountSegment.join within_025_104 within_025_105
private theorem within_025_106_agreement : RowAgreement 3308 3312 := RowAgreement.join within_025_104_agreement within_025_105_agreement
private def within_025_107 : CountSegment 1 10 13120 3304 3312 1586 1588 := CountSegment.join within_025_103 within_025_106
private theorem within_025_107_agreement : RowAgreement 3304 3312 := RowAgreement.join within_025_103_agreement within_025_106_agreement
private def within_025_108 : CountSegment 1 10 13120 3296 3312 1582 1588 := CountSegment.join within_025_100 within_025_107
private theorem within_025_108_agreement : RowAgreement 3296 3312 := RowAgreement.join within_025_100_agreement within_025_107_agreement
private def within_025_109 : CountSegment 1 10 13120 3312 3314 1588 1589 := CountSegment.join atomic_3313 atomic_3314
private theorem within_025_109_agreement : RowAgreement 3312 3314 := RowAgreement.join atomic_3313_agreement atomic_3314_agreement
private def within_025_110 : CountSegment 1 10 13120 3314 3316 1589 1590 := CountSegment.join atomic_3315 atomic_3316
private theorem within_025_110_agreement : RowAgreement 3314 3316 := RowAgreement.join atomic_3315_agreement atomic_3316_agreement
private def within_025_111 : CountSegment 1 10 13120 3312 3316 1588 1590 := CountSegment.join within_025_109 within_025_110
private theorem within_025_111_agreement : RowAgreement 3312 3316 := RowAgreement.join within_025_109_agreement within_025_110_agreement
private def within_025_112 : CountSegment 1 10 13120 3316 3318 1590 1592 := CountSegment.join atomic_3317 atomic_3318
private theorem within_025_112_agreement : RowAgreement 3316 3318 := RowAgreement.join atomic_3317_agreement atomic_3318_agreement
private def within_025_113 : CountSegment 1 10 13120 3318 3320 1592 1592 := CountSegment.join atomic_3319 atomic_3320
private theorem within_025_113_agreement : RowAgreement 3318 3320 := RowAgreement.join atomic_3319_agreement atomic_3320_agreement
private def within_025_114 : CountSegment 1 10 13120 3316 3320 1590 1592 := CountSegment.join within_025_112 within_025_113
private theorem within_025_114_agreement : RowAgreement 3316 3320 := RowAgreement.join within_025_112_agreement within_025_113_agreement
private def within_025_115 : CountSegment 1 10 13120 3312 3320 1588 1592 := CountSegment.join within_025_111 within_025_114
private theorem within_025_115_agreement : RowAgreement 3312 3320 := RowAgreement.join within_025_111_agreement within_025_114_agreement
private def within_025_116 : CountSegment 1 10 13120 3320 3322 1592 1593 := CountSegment.join atomic_3321 atomic_3322
private theorem within_025_116_agreement : RowAgreement 3320 3322 := RowAgreement.join atomic_3321_agreement atomic_3322_agreement
private def within_025_117 : CountSegment 1 10 13120 3322 3324 1593 1594 := CountSegment.join atomic_3323 atomic_3324
private theorem within_025_117_agreement : RowAgreement 3322 3324 := RowAgreement.join atomic_3323_agreement atomic_3324_agreement
private def within_025_118 : CountSegment 1 10 13120 3320 3324 1592 1594 := CountSegment.join within_025_116 within_025_117
private theorem within_025_118_agreement : RowAgreement 3320 3324 := RowAgreement.join within_025_116_agreement within_025_117_agreement
private def within_025_119 : CountSegment 1 10 13120 3324 3326 1594 1594 := CountSegment.join atomic_3325 atomic_3326
private theorem within_025_119_agreement : RowAgreement 3324 3326 := RowAgreement.join atomic_3325_agreement atomic_3326_agreement
private def within_025_120 : CountSegment 1 10 13120 3326 3328 1594 1595 := CountSegment.join atomic_3327 atomic_3328
private theorem within_025_120_agreement : RowAgreement 3326 3328 := RowAgreement.join atomic_3327_agreement atomic_3328_agreement
private def within_025_121 : CountSegment 1 10 13120 3324 3328 1594 1595 := CountSegment.join within_025_119 within_025_120
private theorem within_025_121_agreement : RowAgreement 3324 3328 := RowAgreement.join within_025_119_agreement within_025_120_agreement
private def within_025_122 : CountSegment 1 10 13120 3320 3328 1592 1595 := CountSegment.join within_025_118 within_025_121
private theorem within_025_122_agreement : RowAgreement 3320 3328 := RowAgreement.join within_025_118_agreement within_025_121_agreement
private def within_025_123 : CountSegment 1 10 13120 3312 3328 1588 1595 := CountSegment.join within_025_115 within_025_122
private theorem within_025_123_agreement : RowAgreement 3312 3328 := RowAgreement.join within_025_115_agreement within_025_122_agreement
private def within_025_124 : CountSegment 1 10 13120 3296 3328 1582 1595 := CountSegment.join within_025_108 within_025_123
private theorem within_025_124_agreement : RowAgreement 3296 3328 := RowAgreement.join within_025_108_agreement within_025_123_agreement
private def within_025_125 : CountSegment 1 10 13120 3264 3328 1565 1595 := CountSegment.join within_025_093 within_025_124
private theorem within_025_125_agreement : RowAgreement 3264 3328 := RowAgreement.join within_025_093_agreement within_025_124_agreement
private def within_025_126 : CountSegment 1 10 13120 3200 3328 1542 1595 := CountSegment.join within_025_062 within_025_125
private theorem within_025_126_agreement : RowAgreement 3200 3328 := RowAgreement.join within_025_062_agreement within_025_125_agreement
def segment_025 : CountSegment 1 10 13120 3200 3328 1542 1595 := within_025_126
theorem agreement_025 : RowAgreement 3200 3328 := within_025_126_agreement
end ProjectEulerSolutions.P58Numerical
