import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_4225 : RowCertificate 4225 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4225 1) := by change ¬ Nat.Prime 71410951; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 358849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4225 2) := by change ¬ Nat.Prime 71402501; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1347217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4225 3) := by change ¬ Nat.Prime 71394051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23798017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4225 : row 4225 = 0 := (row_4225.checked).trans rfl
theorem model_count_4225 : modelRow 4225 = 0 := by decide
theorem modelrow_agrees_4225 : modelRow 4225 = row 4225 := model_count_4225.trans row_count_4225.symm
def row_4226 : RowCertificate 4226 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4226 1) := by change ¬ Nat.Prime 71444757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23814919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4226 2) := by change ¬ Nat.Prime 71436305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14287261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4226 3) := by change ¬ Nat.Prime 71427853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10203979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4226 : row 4226 = 0 := (row_4226.checked).trans rfl
theorem model_count_4226 : modelRow 4226 = 0 := by decide
theorem modelrow_agrees_4226 : modelRow 4226 = row 4226 := model_count_4226.trans row_count_4226.symm
def row_4227 : RowCertificate 4227 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4227 1) := by change ¬ Nat.Prime 71478571; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 338761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4227 2) := by change Nat.Prime 71470117; exact ProjectEulerSolutions.P58PrimeNodes.p_71470117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4227 3) := by change ¬ Nat.Prime 71461663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10208809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4227 : row 4227 = 1 := (row_4227.checked).trans rfl
theorem model_count_4227 : modelRow 4227 = 1 := by decide
theorem modelrow_agrees_4227 : modelRow 4227 = row 4227 := model_count_4227.trans row_count_4227.symm
def row_4228 : RowCertificate 4228 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4228 1) := by change ¬ Nat.Prime 71512393; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 656077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4228 2) := by change ¬ Nat.Prime 71503937; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2465653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4228 3) := by change ¬ Nat.Prime 71495481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23831827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4228 : row 4228 = 0 := (row_4228.checked).trans rfl
theorem model_count_4228 : modelRow 4228 = 0 := by decide
theorem modelrow_agrees_4228 : modelRow 4228 = row 4228 := model_count_4228.trans row_count_4228.symm
def row_4229 : RowCertificate 4229 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4229 1) := by change ¬ Nat.Prime 71546223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23848741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4229 2) := by change ¬ Nat.Prime 71537765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14307553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4229 3) := by change ¬ Nat.Prime 71529307; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2307397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4229 : row 4229 = 0 := (row_4229.checked).trans rfl
theorem model_count_4229 : modelRow 4229 = 0 := by decide
theorem modelrow_agrees_4229 : modelRow 4229 = row 4229 := model_count_4229.trans row_count_4229.symm
def row_4230 : RowCertificate 4230 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4230 1) := by change ¬ Nat.Prime 71580061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10225723) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4230 2) := by change ¬ Nat.Prime 71571601; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 633377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4230 3) := by change ¬ Nat.Prime 71563141; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5504857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4230 : row 4230 = 0 := (row_4230.checked).trans rfl
theorem model_count_4230 : modelRow 4230 = 0 := by decide
theorem modelrow_agrees_4230 : modelRow 4230 = row 4230 := model_count_4230.trans row_count_4230.symm
def row_4231 : RowCertificate 4231 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4231 1) := by change ¬ Nat.Prime 71613907; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3769153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4231 2) := by change ¬ Nat.Prime 71605445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14321089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4231 3) := by change ¬ Nat.Prime 71596983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23865661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4231 : row 4231 = 0 := (row_4231.checked).trans rfl
theorem model_count_4231 : modelRow 4231 = 0 := by decide
theorem modelrow_agrees_4231 : modelRow 4231 = row 4231 := model_count_4231.trans row_count_4231.symm
def row_4232 : RowCertificate 4232 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4232 1) := by change ¬ Nat.Prime 71647761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23882587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4232 2) := by change ¬ Nat.Prime 71639297; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 159553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4232 3) := by change Nat.Prime 71630833; exact ProjectEulerSolutions.P58PrimeNodes.p_71630833.prime
    exact decide_eq_true prime
theorem row_count_4232 : row 4232 = 1 := (row_4232.checked).trans rfl
theorem model_count_4232 : modelRow 4232 = 1 := by decide
theorem modelrow_agrees_4232 : modelRow 4232 = row 4232 := model_count_4232.trans row_count_4232.symm
def row_4233 : RowCertificate 4233 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4233 1) := by change ¬ Nat.Prime 71681623; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5513971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4233 2) := by change ¬ Nat.Prime 71673157; exact PrattCertificates.composite_of_product (factor := 6841) (cofactor := 10477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4233 3) := by change ¬ Nat.Prime 71664691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10237813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4233 : row 4233 = 0 := (row_4233.checked).trans rfl
theorem model_count_4233 : modelRow 4233 = 0 := by decide
theorem modelrow_agrees_4233 : modelRow 4233 = row 4233 := model_count_4233.trans row_count_4233.symm
def row_4234 : RowCertificate 4234 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4234 1) := by change ¬ Nat.Prime 71715493; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2313403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4234 2) := by change ¬ Nat.Prime 71707025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14341405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4234 3) := by change ¬ Nat.Prime 71698557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23899519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4234 : row 4234 = 0 := (row_4234.checked).trans rfl
theorem model_count_4234 : modelRow 4234 = 0 := by decide
theorem modelrow_agrees_4234 : modelRow 4234 = row 4234 := model_count_4234.trans row_count_4234.symm
def row_4235 : RowCertificate 4235 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4235 1) := by change ¬ Nat.Prime 71749371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23916457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4235 2) := by change ¬ Nat.Prime 71740901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4220053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4235 3) := by change Nat.Prime 71732431; exact ProjectEulerSolutions.P58PrimeNodes.p_71732431.prime
    exact decide_eq_true prime
theorem row_count_4235 : row 4235 = 1 := (row_4235.checked).trans rfl
theorem model_count_4235 : modelRow 4235 = 1 := by decide
theorem modelrow_agrees_4235 : modelRow 4235 = row 4235 := model_count_4235.trans row_count_4235.symm
def row_4236 : RowCertificate 4236 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4236 1) := by change ¬ Nat.Prime 71783257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10254751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4236 2) := by change ¬ Nat.Prime 71774785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14354957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4236 3) := by change ¬ Nat.Prime 71766313; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1071139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4236 : row 4236 = 0 := (row_4236.checked).trans rfl
theorem model_count_4236 : modelRow 4236 = 0 := by decide
theorem modelrow_agrees_4236 : modelRow 4236 = row 4236 := model_count_4236.trans row_count_4236.symm
def row_4237 : RowCertificate 4237 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4237 1) := by change ¬ Nat.Prime 71817151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10259593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4237 2) := by change ¬ Nat.Prime 71808677; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 710977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4237 3) := by change ¬ Nat.Prime 71800203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23933401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4237 : row 4237 = 0 := (row_4237.checked).trans rfl
theorem model_count_4237 : modelRow 4237 = 0 := by decide
theorem modelrow_agrees_4237 : modelRow 4237 = row 4237 := model_count_4237.trans row_count_4237.symm
def row_4238 : RowCertificate 4238 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4238 1) := by change ¬ Nat.Prime 71851053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23950351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4238 2) := by change ¬ Nat.Prime 71842577; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 229529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4238 3) := by change Nat.Prime 71834101; exact ProjectEulerSolutions.P58PrimeNodes.p_71834101.prime
    exact decide_eq_true prime
theorem row_count_4238 : row 4238 = 1 := (row_4238.checked).trans rfl
theorem model_count_4238 : modelRow 4238 = 1 := by decide
theorem modelrow_agrees_4238 : modelRow 4238 = row 4238 := model_count_4238.trans row_count_4238.symm
def row_4239 : RowCertificate 4239 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4239 1) := by change Nat.Prime 71884963; exact ProjectEulerSolutions.P58PrimeNodes.p_71884963.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4239 2) := by change ¬ Nat.Prime 71876485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14375297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4239 3) := by change ¬ Nat.Prime 71868007; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1671349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4239 : row 4239 = 1 := (row_4239.checked).trans rfl
theorem model_count_4239 : modelRow 4239 = 1 := by decide
theorem modelrow_agrees_4239 : modelRow 4239 = row 4239 := model_count_4239.trans row_count_4239.symm
def row_4240 : RowCertificate 4240 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4240 1) := by change Nat.Prime 71918881; exact ProjectEulerSolutions.P58PrimeNodes.p_71918881.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4240 2) := by change ¬ Nat.Prime 71910401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2479669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4240 3) := by change ¬ Nat.Prime 71901921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23967307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4240 : row 4240 = 1 := (row_4240.checked).trans rfl
theorem model_count_4240 : modelRow 4240 = 1 := by decide
theorem modelrow_agrees_4240 : modelRow 4240 = row 4240 := model_count_4240.trans row_count_4240.symm
def row_4241 : RowCertificate 4241 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4241 1) := by change ¬ Nat.Prime 71952807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 23984269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4241 2) := by change ¬ Nat.Prime 71944325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14388865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4241 3) := by change ¬ Nat.Prime 71935843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10276549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4241 : row 4241 = 0 := (row_4241.checked).trans rfl
theorem model_count_4241 : modelRow 4241 = 0 := by decide
theorem modelrow_agrees_4241 : modelRow 4241 = row 4241 := model_count_4241.trans row_count_4241.symm
def row_4242 : RowCertificate 4242 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4242 1) := by change Nat.Prime 71986741; exact ProjectEulerSolutions.P58PrimeNodes.p_71986741.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4242 2) := by change ¬ Nat.Prime 71978257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5536789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4242 3) := by change ¬ Nat.Prime 71969773; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1945129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4242 : row 4242 = 1 := (row_4242.checked).trans rfl
theorem model_count_4242 : modelRow 4242 = 1 := by decide
theorem modelrow_agrees_4242 : modelRow 4242 = row 4242 := model_count_4242.trans row_count_4242.symm
def row_4243 : RowCertificate 4243 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4243 1) := by change ¬ Nat.Prime 72020683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10288669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4243 2) := by change ¬ Nat.Prime 72012197; exact PrattCertificates.composite_of_product (factor := 4409) (cofactor := 16333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4243 3) := by change ¬ Nat.Prime 72003711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24001237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4243 : row 4243 = 0 := (row_4243.checked).trans rfl
theorem model_count_4243 : modelRow 4243 = 0 := by decide
theorem modelrow_agrees_4243 : modelRow 4243 = row 4243 := model_count_4243.trans row_count_4243.symm
def row_4244 : RowCertificate 4244 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4244 1) := by change ¬ Nat.Prime 72054633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24018211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4244 2) := by change ¬ Nat.Prime 72046145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14409229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4244 3) := by change ¬ Nat.Prime 72037657; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 82141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4244 : row 4244 = 0 := (row_4244.checked).trans rfl
theorem model_count_4244 : modelRow 4244 = 0 := by decide
theorem modelrow_agrees_4244 : modelRow 4244 = row 4244 := model_count_4244.trans row_count_4244.symm
def row_4245 : RowCertificate 4245 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4245 1) := by change ¬ Nat.Prime 72088591; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 459163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4245 2) := by change ¬ Nat.Prime 72080101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1181641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4245 3) := by change ¬ Nat.Prime 72071611; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 567493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4245 : row 4245 = 0 := (row_4245.checked).trans rfl
theorem model_count_4245 : modelRow 4245 = 0 := by decide
theorem modelrow_agrees_4245 : modelRow 4245 = row 4245 := model_count_4245.trans row_count_4245.symm
def row_4246 : RowCertificate 4246 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4246 1) := by change ¬ Nat.Prime 72122557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5547889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4246 2) := by change ¬ Nat.Prime 72114065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14422813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4246 3) := by change ¬ Nat.Prime 72105573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24035191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4246 : row 4246 = 0 := (row_4246.checked).trans rfl
theorem model_count_4246 : modelRow 4246 = 0 := by decide
theorem modelrow_agrees_4246 : modelRow 4246 = row 4246 := model_count_4246.trans row_count_4246.symm
def row_4247 : RowCertificate 4247 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4247 1) := by change ¬ Nat.Prime 72156531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24052177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4247 2) := by change ¬ Nat.Prime 72148037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5549849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4247 3) := by change ¬ Nat.Prime 72139543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10305649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4247 : row 4247 = 0 := (row_4247.checked).trans rfl
theorem model_count_4247 : modelRow 4247 = 0 := by decide
theorem modelrow_agrees_4247 : modelRow 4247 = row 4247 := model_count_4247.trans row_count_4247.symm
def row_4248 : RowCertificate 4248 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4248 1) := by change Nat.Prime 72190513; exact ProjectEulerSolutions.P58PrimeNodes.p_72190513.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4248 2) := by change ¬ Nat.Prime 72182017; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4246001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4248 3) := by change ¬ Nat.Prime 72173521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10310503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4248 : row 4248 = 1 := (row_4248.checked).trans rfl
theorem model_count_4248 : modelRow 4248 = 1 := by decide
theorem modelrow_agrees_4248 : modelRow 4248 = row 4248 := model_count_4248.trans row_count_4248.symm
def row_4249 : RowCertificate 4249 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4249 1) := by change ¬ Nat.Prime 72224503; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5555731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4249 2) := by change ¬ Nat.Prime 72216005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14443201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4249 3) := by change ¬ Nat.Prime 72207507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24069169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4249 : row 4249 = 0 := (row_4249.checked).trans rfl
theorem model_count_4249 : modelRow 4249 = 0 := by decide
theorem modelrow_agrees_4249 : modelRow 4249 = row 4249 := model_count_4249.trans row_count_4249.symm
def row_4250 : RowCertificate 4250 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4250 1) := by change ¬ Nat.Prime 72258501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24086167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4250 2) := by change Nat.Prime 72250001; exact ProjectEulerSolutions.P58PrimeNodes.p_72250001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4250 3) := by change ¬ Nat.Prime 72241501; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2330371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4250 : row 4250 = 1 := (row_4250.checked).trans rfl
theorem model_count_4250 : modelRow 4250 = 1 := by decide
theorem modelrow_agrees_4250 : modelRow 4250 = row 4250 := model_count_4250.trans row_count_4250.symm
def row_4251 : RowCertificate 4251 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4251 1) := by change ¬ Nat.Prime 72292507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10327501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4251 2) := by change ¬ Nat.Prime 72284005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14456801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4251 3) := by change Nat.Prime 72275503; exact ProjectEulerSolutions.P58PrimeNodes.p_72275503.prime
    exact decide_eq_true prime
theorem row_count_4251 : row 4251 = 1 := (row_4251.checked).trans rfl
theorem model_count_4251 : modelRow 4251 = 1 := by decide
theorem modelrow_agrees_4251 : modelRow 4251 = row 4251 := model_count_4251.trans row_count_4251.symm
def row_4252 : RowCertificate 4252 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4252 1) := by change ¬ Nat.Prime 72326521; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3806659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4252 2) := by change ¬ Nat.Prime 72318017; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4254001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4252 3) := by change ¬ Nat.Prime 72309513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24103171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4252 : row 4252 = 0 := (row_4252.checked).trans rfl
theorem model_count_4252 : modelRow 4252 = 0 := by decide
theorem modelrow_agrees_4252 : modelRow 4252 = row 4252 := model_count_4252.trans row_count_4252.symm
def row_4253 : RowCertificate 4253 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4253 1) := by change ¬ Nat.Prime 72360543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24120181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4253 2) := by change ¬ Nat.Prime 72352037; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 460841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4253 3) := by change ¬ Nat.Prime 72343531; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5564887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4253 : row 4253 = 0 := (row_4253.checked).trans rfl
theorem model_count_4253 : modelRow 4253 = 0 := by decide
theorem modelrow_agrees_4253 : modelRow 4253 = row 4253 := model_count_4253.trans row_count_4253.symm
def row_4254 : RowCertificate 4254 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4254 1) := by change ¬ Nat.Prime 72394573; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 916387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4254 2) := by change ¬ Nat.Prime 72386065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14477213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4254 3) := by change ¬ Nat.Prime 72377557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10339651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4254 : row 4254 = 0 := (row_4254.checked).trans rfl
theorem model_count_4254 : modelRow 4254 = 0 := by decide
theorem modelrow_agrees_4254 : modelRow 4254 = row 4254 := model_count_4254.trans row_count_4254.symm
def row_4255 : RowCertificate 4255 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4255 1) := by change Nat.Prime 72428611; exact ProjectEulerSolutions.P58PrimeNodes.p_72428611.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4255 2) := by change ¬ Nat.Prime 72420101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5570777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4255 3) := by change ¬ Nat.Prime 72411591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24137197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4255 : row 4255 = 1 := (row_4255.checked).trans rfl
theorem model_count_4255 : modelRow 4255 = 1 := by decide
theorem modelrow_agrees_4255 : modelRow 4255 = row 4255 := model_count_4255.trans row_count_4255.symm
def row_4256 : RowCertificate 4256 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4256 1) := by change ¬ Nat.Prime 72462657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24154219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4256 2) := by change ¬ Nat.Prime 72454145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14490829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4256 3) := by change ¬ Nat.Prime 72445633; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5572741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4256 : row 4256 = 0 := (row_4256.checked).trans rfl
theorem model_count_4256 : modelRow 4256 = 0 := by decide
theorem modelrow_agrees_4256 : modelRow 4256 = row 4256 := model_count_4256.trans row_count_4256.symm
def row_4257 : RowCertificate 4257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4257 1) := by change ¬ Nat.Prime 72496711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10356673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4257 2) := by change ¬ Nat.Prime 72488197; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2499593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4257 3) := by change ¬ Nat.Prime 72479683; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 325021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4257 : row 4257 = 0 := (row_4257.checked).trans rfl
theorem model_count_4257 : modelRow 4257 = 0 := by decide
theorem modelrow_agrees_4257 : modelRow 4257 = row 4257 := model_count_4257.trans row_count_4257.symm
def row_4258 : RowCertificate 4258 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4258 1) := by change ¬ Nat.Prime 72530773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10361539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4258 2) := by change ¬ Nat.Prime 72522257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1960061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4258 3) := by change ¬ Nat.Prime 72513741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24171247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4258 : row 4258 = 0 := (row_4258.checked).trans rfl
theorem model_count_4258 : modelRow 4258 = 0 := by decide
theorem modelrow_agrees_4258 : modelRow 4258 = row 4258 := model_count_4258.trans row_count_4258.symm
def row_4259 : RowCertificate 4259 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4259 1) := by change ¬ Nat.Prime 72564843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24188281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4259 2) := by change ¬ Nat.Prime 72556325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14511265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4259 3) := by change Nat.Prime 72547807; exact ProjectEulerSolutions.P58PrimeNodes.p_72547807.prime
    exact decide_eq_true prime
theorem row_count_4259 : row 4259 = 1 := (row_4259.checked).trans rfl
theorem model_count_4259 : modelRow 4259 = 1 := by decide
theorem modelrow_agrees_4259 : modelRow 4259 = row 4259 := model_count_4259.trans row_count_4259.symm
def row_4260 : RowCertificate 4260 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4260 1) := by change ¬ Nat.Prime 72598921; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1962133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4260 2) := by change ¬ Nat.Prime 72590401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5583877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4260 3) := by change ¬ Nat.Prime 72581881; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3820099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4260 : row 4260 = 0 := (row_4260.checked).trans rfl
theorem model_count_4260 : modelRow 4260 = 0 := by decide
theorem modelrow_agrees_4260 : modelRow 4260 = row 4260 := model_count_4260.trans row_count_4260.symm
def row_4261 : RowCertificate 4261 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4261 1) := by change Nat.Prime 72633007; exact ProjectEulerSolutions.P58PrimeNodes.p_72633007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4261 2) := by change ¬ Nat.Prime 72624485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14524897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4261 3) := by change ¬ Nat.Prime 72615963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24205321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4261 : row 4261 = 1 := (row_4261.checked).trans rfl
theorem model_count_4261 : modelRow 4261 = 1 := by decide
theorem modelrow_agrees_4261 : modelRow 4261 = row 4261 := model_count_4261.trans row_count_4261.symm
def row_4262 : RowCertificate 4262 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4262 1) := by change ¬ Nat.Prime 72667101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24222367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4262 2) := by change ¬ Nat.Prime 72658577; exact PrattCertificates.composite_of_product (factor := 8237) (cofactor := 8821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4262 3) := by change ¬ Nat.Prime 72650053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10378579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4262 : row 4262 = 0 := (row_4262.checked).trans rfl
theorem model_count_4262 : modelRow 4262 = 0 := by decide
theorem modelrow_agrees_4262 : modelRow 4262 = row 4262 := model_count_4262.trans row_count_4262.symm
def row_4263 : RowCertificate 4263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4263 1) := by change ¬ Nat.Prime 72701203; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1191823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4263 2) := by change ¬ Nat.Prime 72692677; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 401617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4263 3) := by change Nat.Prime 72684151; exact ProjectEulerSolutions.P58PrimeNodes.p_72684151.prime
    exact decide_eq_true prime
theorem row_count_4263 : row 4263 = 1 := (row_4263.checked).trans rfl
theorem model_count_4263 : modelRow 4263 = 1 := by decide
theorem modelrow_agrees_4263 : modelRow 4263 = row 4263 := model_count_4263.trans row_count_4263.symm
def row_4264 : RowCertificate 4264 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4264 1) := by change ¬ Nat.Prime 72735313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10390759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4264 2) := by change ¬ Nat.Prime 72726785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14545357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4264 3) := by change ¬ Nat.Prime 72718257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24239419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4264 : row 4264 = 0 := (row_4264.checked).trans rfl
theorem model_count_4264 : modelRow 4264 = 0 := by decide
theorem modelrow_agrees_4264 : modelRow 4264 = row 4264 := model_count_4264.trans row_count_4264.symm
def row_4265 : RowCertificate 4265 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4265 1) := by change ¬ Nat.Prime 72769431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24256477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4265 2) := by change ¬ Nat.Prime 72760901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4280053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4265 3) := by change ¬ Nat.Prime 72752371; exact PrattCertificates.composite_of_product (factor := 3331) (cofactor := 21841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4265 : row 4265 = 0 := (row_4265.checked).trans rfl
theorem model_count_4265 : modelRow 4265 = 0 := by decide
theorem modelrow_agrees_4265 : modelRow 4265 = row 4265 := model_count_4265.trans row_count_4265.symm
def row_4266 : RowCertificate 4266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4266 1) := by change ¬ Nat.Prime 72803557; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 997309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4266 2) := by change ¬ Nat.Prime 72795025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14559005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4266 3) := by change ¬ Nat.Prime 72786493; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5598961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4266 : row 4266 = 0 := (row_4266.checked).trans rfl
theorem model_count_4266 : modelRow 4266 = 0 := by decide
theorem modelrow_agrees_4266 : modelRow 4266 = row 4266 := model_count_4266.trans row_count_4266.symm
def row_4267 : RowCertificate 4267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4267 1) := by change ¬ Nat.Prime 72837691; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 446857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4267 2) := by change ¬ Nat.Prime 72829157; exact PrattCertificates.composite_of_product (factor := 2789) (cofactor := 26113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4267 3) := by change ¬ Nat.Prime 72820623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24273541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4267 : row 4267 = 0 := (row_4267.checked).trans rfl
theorem model_count_4267 : modelRow 4267 = 0 := by decide
theorem modelrow_agrees_4267 : modelRow 4267 = row 4267 := model_count_4267.trans row_count_4267.symm
def row_4268 : RowCertificate 4268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4268 1) := by change ¬ Nat.Prime 72871833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24290611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4268 2) := by change ¬ Nat.Prime 72863297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5604869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4268 3) := by change ¬ Nat.Prime 72854761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10407823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4268 : row 4268 = 0 := (row_4268.checked).trans rfl
theorem model_count_4268 : modelRow 4268 = 0 := by decide
theorem modelrow_agrees_4268 : modelRow 4268 = row 4268 := model_count_4268.trans row_count_4268.symm
def row_4269 : RowCertificate 4269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4269 1) := by change ¬ Nat.Prime 72905983; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3837157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4269 2) := by change ¬ Nat.Prime 72897445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14579489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4269 3) := by change ¬ Nat.Prime 72888907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10412701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4269 : row 4269 = 0 := (row_4269.checked).trans rfl
theorem model_count_4269 : modelRow 4269 = 0 := by decide
theorem modelrow_agrees_4269 : modelRow 4269 = row 4269 := model_count_4269.trans row_count_4269.symm
def row_4270 : RowCertificate 4270 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4270 1) := by change Nat.Prime 72940141; exact ProjectEulerSolutions.P58PrimeNodes.p_72940141.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4270 2) := by change Nat.Prime 72931601; exact ProjectEulerSolutions.P58PrimeNodes.p_72931601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4270 3) := by change ¬ Nat.Prime 72923061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24307687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4270 : row 4270 = 2 := (row_4270.checked).trans rfl
theorem model_count_4270 : modelRow 4270 = 2 := by decide
theorem modelrow_agrees_4270 : modelRow 4270 = row 4270 := model_count_4270.trans row_count_4270.symm
def row_4271 : RowCertificate 4271 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4271 1) := by change ¬ Nat.Prime 72974307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24324769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4271 2) := by change ¬ Nat.Prime 72965765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14593153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4271 3) := by change Nat.Prime 72957223; exact ProjectEulerSolutions.P58PrimeNodes.p_72957223.prime
    exact decide_eq_true prime
theorem row_count_4271 : row 4271 = 1 := (row_4271.checked).trans rfl
theorem model_count_4271 : modelRow 4271 = 1 := by decide
theorem modelrow_agrees_4271 : modelRow 4271 = row 4271 := model_count_4271.trans row_count_4271.symm
def row_4272 : RowCertificate 4272 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4272 1) := by change ¬ Nat.Prime 73008481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10429783) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4272 2) := by change ¬ Nat.Prime 72999937; exact PrattCertificates.composite_of_product (factor := 1289) (cofactor := 56633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4272 3) := by change Nat.Prime 72991393; exact ProjectEulerSolutions.P58PrimeNodes.p_72991393.prime
    exact decide_eq_true prime
theorem row_count_4272 : row 4272 = 1 := (row_4272.checked).trans rfl
theorem model_count_4272 : modelRow 4272 = 1 := by decide
theorem modelrow_agrees_4272 : modelRow 4272 = row 4272 := model_count_4272.trans row_count_4272.symm
def row_4273 : RowCertificate 4273 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4273 1) := by change ¬ Nat.Prime 73042663; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1090189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4273 2) := by change ¬ Nat.Prime 73034117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5618009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4273 3) := by change ¬ Nat.Prime 73025571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24341857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4273 : row 4273 = 0 := (row_4273.checked).trans rfl
theorem model_count_4273 : modelRow 4273 = 0 := by decide
theorem modelrow_agrees_4273 : modelRow 4273 = row 4273 := model_count_4273.trans row_count_4273.symm
def row_4274 : RowCertificate 4274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4274 1) := by change ¬ Nat.Prime 73076853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24358951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4274 2) := by change ¬ Nat.Prime 73068305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14613661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4274 3) := by change ¬ Nat.Prime 73059757; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 670273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4274 : row 4274 = 0 := (row_4274.checked).trans rfl
theorem model_count_4274 : modelRow 4274 = 0 := by decide
theorem modelrow_agrees_4274 : modelRow 4274 = row 4274 := model_count_4274.trans row_count_4274.symm
def row_4275 : RowCertificate 4275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4275 1) := by change ¬ Nat.Prime 73111051; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5623927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4275 2) := by change Nat.Prime 73102501; exact ProjectEulerSolutions.P58PrimeNodes.p_73102501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4275 3) := by change ¬ Nat.Prime 73093951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10441993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4275 : row 4275 = 1 := (row_4275.checked).trans rfl
theorem model_count_4275 : modelRow 4275 = 1 := by decide
theorem modelrow_agrees_4275 : modelRow 4275 = row 4275 := model_count_4275.trans row_count_4275.symm
def row_4276 : RowCertificate 4276 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4276 1) := by change Nat.Prime 73145257; exact ProjectEulerSolutions.P58PrimeNodes.p_73145257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4276 2) := by change ¬ Nat.Prime 73136705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14627341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4276 3) := by change ¬ Nat.Prime 73128153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24376051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4276 : row 4276 = 1 := (row_4276.checked).trans rfl
theorem model_count_4276 : modelRow 4276 = 1 := by decide
theorem modelrow_agrees_4276 : modelRow 4276 = row 4276 := model_count_4276.trans row_count_4276.symm
def row_4277 : RowCertificate 4277 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4277 1) := by change ¬ Nat.Prime 73179471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24393157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4277 2) := by change Nat.Prime 73170917; exact ProjectEulerSolutions.P58PrimeNodes.p_73170917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4277 3) := by change ¬ Nat.Prime 73162363; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1199383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4277 : row 4277 = 1 := (row_4277.checked).trans rfl
theorem model_count_4277 : modelRow 4277 = 1 := by decide
theorem modelrow_agrees_4277 : modelRow 4277 = row 4277 := model_count_4277.trans row_count_4277.symm
def row_4278 : RowCertificate 4278 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4278 1) := by change ¬ Nat.Prime 73213693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10459099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4278 2) := by change ¬ Nat.Prime 73205137; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1381229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4278 3) := by change ¬ Nat.Prime 73196581; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 926539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4278 : row 4278 = 0 := (row_4278.checked).trans rfl
theorem model_count_4278 : modelRow 4278 = 0 := by decide
theorem modelrow_agrees_4278 : modelRow 4278 = row 4278 := model_count_4278.trans row_count_4278.symm
def row_4279 : RowCertificate 4279 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4279 1) := by change ¬ Nat.Prime 73247923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10463989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4279 2) := by change ¬ Nat.Prime 73239365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14647873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4279 3) := by change ¬ Nat.Prime 73230807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24410269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4279 : row 4279 = 0 := (row_4279.checked).trans rfl
theorem model_count_4279 : modelRow 4279 = 0 := by decide
theorem modelrow_agrees_4279 : modelRow 4279 = row 4279 := model_count_4279.trans row_count_4279.symm
def row_4280 : RowCertificate 4280 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4280 1) := by change ¬ Nat.Prime 73282161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24427387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4280 2) := by change ¬ Nat.Prime 73273601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1787161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4280 3) := by change Nat.Prime 73265041; exact ProjectEulerSolutions.P58PrimeNodes.p_73265041.prime
    exact decide_eq_true prime
theorem row_count_4280 : row 4280 = 1 := (row_4280.checked).trans rfl
theorem model_count_4280 : modelRow 4280 = 1 := by decide
theorem modelrow_agrees_4280 : modelRow 4280 = row 4280 := model_count_4280.trans row_count_4280.symm
def row_4281 : RowCertificate 4281 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4281 1) := by change Nat.Prime 73316407; exact ProjectEulerSolutions.P58PrimeNodes.p_73316407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4281 2) := by change ¬ Nat.Prime 73307845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14661569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4281 3) := by change ¬ Nat.Prime 73299283; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3857857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4281 : row 4281 = 1 := (row_4281.checked).trans rfl
theorem model_count_4281 : modelRow 4281 = 1 := by decide
theorem modelrow_agrees_4281 : modelRow 4281 = row 4281 := model_count_4281.trans row_count_4281.symm
def row_4282 : RowCertificate 4282 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4282 1) := by change Nat.Prime 73350661; exact ProjectEulerSolutions.P58PrimeNodes.p_73350661.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4282 2) := by change ¬ Nat.Prime 73342097; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4314241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4282 3) := by change ¬ Nat.Prime 73333533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24444511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4282 : row 4282 = 1 := (row_4282.checked).trans rfl
theorem model_count_4282 : modelRow 4282 = 1 := by decide
theorem modelrow_agrees_4282 : modelRow 4282 = row 4282 := model_count_4282.trans row_count_4282.symm
def row_4283 : RowCertificate 4283 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4283 1) := by change ¬ Nat.Prime 73384923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24461641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4283 2) := by change ¬ Nat.Prime 73376357; exact PrattCertificates.composite_of_product (factor := 941) (cofactor := 77977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4283 3) := by change ¬ Nat.Prime 73367791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10481113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4283 : row 4283 = 0 := (row_4283.checked).trans rfl
theorem model_count_4283 : modelRow 4283 = 0 := by decide
theorem modelrow_agrees_4283 : modelRow 4283 = row 4283 := model_count_4283.trans row_count_4283.symm
def row_4284 : RowCertificate 4284 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4284 1) := by change Nat.Prime 73419193; exact ProjectEulerSolutions.P58PrimeNodes.p_73419193.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4284 2) := by change ¬ Nat.Prime 73410625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14682125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4284 3) := by change ¬ Nat.Prime 73402057; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 320533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4284 : row 4284 = 1 := (row_4284.checked).trans rfl
theorem model_count_4284 : modelRow 4284 = 1 := by decide
theorem modelrow_agrees_4284 : modelRow 4284 = row 4284 := model_count_4284.trans row_count_4284.symm
def row_4285 : RowCertificate 4285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4285 1) := by change ¬ Nat.Prime 73453471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10493353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4285 2) := by change ¬ Nat.Prime 73444901; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 424537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4285 3) := by change ¬ Nat.Prime 73436331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24478777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4285 : row 4285 = 0 := (row_4285.checked).trans rfl
theorem model_count_4285 : modelRow 4285 = 0 := by decide
theorem modelrow_agrees_4285 : modelRow 4285 = row 4285 := model_count_4285.trans row_count_4285.symm
def row_4286 : RowCertificate 4286 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4286 1) := by change ¬ Nat.Prime 73487757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24495919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4286 2) := by change ¬ Nat.Prime 73479185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14695837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4286 3) := by change ¬ Nat.Prime 73470613; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 757429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4286 : row 4286 = 0 := (row_4286.checked).trans rfl
theorem model_count_4286 : modelRow 4286 = 0 := by decide
theorem modelrow_agrees_4286 : modelRow 4286 = row 4286 := model_count_4286.trans row_count_4286.symm
def row_4287 : RowCertificate 4287 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4287 1) := by change ¬ Nat.Prime 73522051; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 486901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4287 2) := by change ¬ Nat.Prime 73513477; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 160861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4287 3) := by change ¬ Nat.Prime 73504903; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1986619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4287 : row 4287 = 0 := (row_4287.checked).trans rfl
theorem model_count_4287 : modelRow 4287 = 0 := by decide
theorem modelrow_agrees_4287 : modelRow 4287 = row 4287 := model_count_4287.trans row_count_4287.symm
def row_4288 : RowCertificate 4288 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4288 1) := by change ¬ Nat.Prime 73556353; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5658181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4288 2) := by change Nat.Prime 73547777; exact ProjectEulerSolutions.P58PrimeNodes.p_73547777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4288 3) := by change ¬ Nat.Prime 73539201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24513067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4288 : row 4288 = 1 := (row_4288.checked).trans rfl
theorem model_count_4288 : modelRow 4288 = 1 := by decide
theorem modelrow_agrees_4288 : modelRow 4288 = row 4288 := model_count_4288.trans row_count_4288.symm
def row_4289 : RowCertificate 4289 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4289 1) := by change ¬ Nat.Prime 73590663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24530221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4289 2) := by change ¬ Nat.Prime 73582085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14716417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4289 3) := by change ¬ Nat.Prime 73573507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10510501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4289 : row 4289 = 0 := (row_4289.checked).trans rfl
theorem model_count_4289 : modelRow 4289 = 0 := by decide
theorem modelrow_agrees_4289 : modelRow 4289 = row 4289 := model_count_4289.trans row_count_4289.symm
def row_4290 : RowCertificate 4290 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4290 1) := by change ¬ Nat.Prime 73624981; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3874999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4290 2) := by change ¬ Nat.Prime 73616401; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 321469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4290 3) := by change ¬ Nat.Prime 73607821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10515403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4290 : row 4290 = 0 := (row_4290.checked).trans rfl
theorem model_count_4290 : modelRow 4290 = 0 := by decide
theorem modelrow_agrees_4290 : modelRow 4290 = row 4290 := model_count_4290.trans row_count_4290.symm
def row_4291 : RowCertificate 4291 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4291 1) := by change ¬ Nat.Prime 73659307; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 42019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4291 2) := by change ¬ Nat.Prime 73650725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14730145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4291 3) := by change ¬ Nat.Prime 73642143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24547381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4291 : row 4291 = 0 := (row_4291.checked).trans rfl
theorem model_count_4291 : modelRow 4291 = 0 := by decide
theorem modelrow_agrees_4291 : modelRow 4291 = row 4291 := model_count_4291.trans row_count_4291.symm
def row_4292 : RowCertificate 4292 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4292 1) := by change ¬ Nat.Prime 73693641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24564547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4292 2) := by change Nat.Prime 73685057; exact ProjectEulerSolutions.P58PrimeNodes.p_73685057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4292 3) := by change ¬ Nat.Prime 73676473; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5667421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4292 : row 4292 = 1 := (row_4292.checked).trans rfl
theorem model_count_4292 : modelRow 4292 = 1 := by decide
theorem modelrow_agrees_4292 : modelRow 4292 = row 4292 := model_count_4292.trans row_count_4292.symm
def row_4293 : RowCertificate 4293 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4293 1) := by change ¬ Nat.Prime 73727983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10532569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4293 2) := by change ¬ Nat.Prime 73719397; exact PrattCertificates.composite_of_product (factor := 7537) (cofactor := 9781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4293 3) := by change ¬ Nat.Prime 73710811; exact PrattCertificates.composite_of_product (factor := 6277) (cofactor := 11743) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4293 : row 4293 = 0 := (row_4293.checked).trans rfl
theorem model_count_4293 : modelRow 4293 = 0 := by decide
theorem modelrow_agrees_4293 : modelRow 4293 = row 4293 := model_count_4293.trans row_count_4293.symm
def row_4294 : RowCertificate 4294 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4294 1) := by change ¬ Nat.Prime 73762333; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 70183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4294 2) := by change ¬ Nat.Prime 73753745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14750749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4294 3) := by change ¬ Nat.Prime 73745157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24581719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4294 : row 4294 = 0 := (row_4294.checked).trans rfl
theorem model_count_4294 : modelRow 4294 = 0 := by decide
theorem modelrow_agrees_4294 : modelRow 4294 = row 4294 := model_count_4294.trans row_count_4294.symm
def row_4295 : RowCertificate 4295 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4295 1) := by change ¬ Nat.Prime 73796691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24598897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4295 2) := by change ¬ Nat.Prime 73788101; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1994273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4295 3) := by change ¬ Nat.Prime 73779511; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5675347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4295 : row 4295 = 0 := (row_4295.checked).trans rfl
theorem model_count_4295 : modelRow 4295 = 0 := by decide
theorem modelrow_agrees_4295 : modelRow 4295 = row 4295 := model_count_4295.trans row_count_4295.symm
def row_4296 : RowCertificate 4296 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4296 1) := by change ¬ Nat.Prime 73831057; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2381647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4296 2) := by change ¬ Nat.Prime 73822465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14764493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4296 3) := by change ¬ Nat.Prime 73813873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10544839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4296 : row 4296 = 0 := (row_4296.checked).trans rfl
theorem model_count_4296 : modelRow 4296 = 0 := by decide
theorem modelrow_agrees_4296 : modelRow 4296 = row 4296 := model_count_4296.trans row_count_4296.symm
def row_4297 : RowCertificate 4297 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4297 1) := by change ¬ Nat.Prime 73865431; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1996363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4297 2) := by change ¬ Nat.Prime 73856837; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 539101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4297 3) := by change ¬ Nat.Prime 73848243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24616081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4297 : row 4297 = 0 := (row_4297.checked).trans rfl
theorem model_count_4297 : modelRow 4297 = 0 := by decide
theorem modelrow_agrees_4297 : modelRow 4297 = row 4297 := model_count_4297.trans row_count_4297.symm
def row_4298 : RowCertificate 4298 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4298 1) := by change ¬ Nat.Prime 73899813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24633271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4298 2) := by change ¬ Nat.Prime 73891217; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2547973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4298 3) := by change ¬ Nat.Prime 73882621; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3888559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4298 : row 4298 = 0 := (row_4298.checked).trans rfl
theorem model_count_4298 : modelRow 4298 = 0 := by decide
theorem modelrow_agrees_4298 : modelRow 4298 = row 4298 := model_count_4298.trans row_count_4298.symm
def row_4299 : RowCertificate 4299 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4299 1) := by change ¬ Nat.Prime 73934203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10562029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4299 2) := by change ¬ Nat.Prime 73925605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14785121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4299 3) := by change ¬ Nat.Prime 73917007; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 762031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4299 : row 4299 = 0 := (row_4299.checked).trans rfl
theorem model_count_4299 : modelRow 4299 = 0 := by decide
theorem modelrow_agrees_4299 : modelRow 4299 = row 4299 := model_count_4299.trans row_count_4299.symm
def row_4300 : RowCertificate 4300 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4300 1) := by change ¬ Nat.Prime 73968601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10566943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4300 2) := by change ¬ Nat.Prime 73960001; exact PrattCertificates.composite_of_product (factor := 3109) (cofactor := 23789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4300 3) := by change ¬ Nat.Prime 73951401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24650467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4300 : row 4300 = 0 := (row_4300.checked).trans rfl
theorem model_count_4300 : modelRow 4300 = 0 := by decide
theorem modelrow_agrees_4300 : modelRow 4300 = row 4300 := model_count_4300.trans row_count_4300.symm
def row_4301 : RowCertificate 4301 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4301 1) := by change ¬ Nat.Prime 74003007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24667669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4301 2) := by change ¬ Nat.Prime 73994405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14798881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4301 3) := by change Nat.Prime 73985803; exact ProjectEulerSolutions.P58PrimeNodes.p_73985803.prime
    exact decide_eq_true prime
theorem row_count_4301 : row 4301 = 1 := (row_4301.checked).trans rfl
theorem model_count_4301 : modelRow 4301 = 1 := by decide
theorem modelrow_agrees_4301 : modelRow 4301 = row 4301 := model_count_4301.trans row_count_4301.symm
def row_4302 : RowCertificate 4302 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4302 1) := by change ¬ Nat.Prime 74037421; exact PrattCertificates.composite_of_product (factor := 3343) (cofactor := 22147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4302 2) := by change ¬ Nat.Prime 74028817; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 383569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4302 3) := by change Nat.Prime 74020213; exact ProjectEulerSolutions.P58PrimeNodes.p_74020213.prime
    exact decide_eq_true prime
theorem row_count_4302 : row 4302 = 1 := (row_4302.checked).trans rfl
theorem model_count_4302 : modelRow 4302 = 1 := by decide
theorem modelrow_agrees_4302 : modelRow 4302 = row 4302 := model_count_4302.trans row_count_4302.symm
def row_4303 : RowCertificate 4303 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4303 1) := by change ¬ Nat.Prime 74071843; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1722601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4303 2) := by change ¬ Nat.Prime 74063237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4356661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4303 3) := by change ¬ Nat.Prime 74054631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24684877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4303 : row 4303 = 0 := (row_4303.checked).trans rfl
theorem model_count_4303 : modelRow 4303 = 0 := by decide
theorem modelrow_agrees_4303 : modelRow 4303 = row 4303 := model_count_4303.trans row_count_4303.symm
def row_4304 : RowCertificate 4304 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4304 1) := by change ¬ Nat.Prime 74106273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24702091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4304 2) := by change ¬ Nat.Prime 74097665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14819533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4304 3) := by change ¬ Nat.Prime 74089057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10584151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4304 : row 4304 = 0 := (row_4304.checked).trans rfl
theorem model_count_4304 : modelRow 4304 = 0 := by decide
theorem modelrow_agrees_4304 : modelRow 4304 = row 4304 := model_count_4304.trans row_count_4304.symm
def row_4305 : RowCertificate 4305 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4305 1) := by change ¬ Nat.Prime 74140711; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2003803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4305 2) := by change Nat.Prime 74132101; exact ProjectEulerSolutions.P58PrimeNodes.p_74132101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4305 3) := by change ¬ Nat.Prime 74123491; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5701807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4305 : row 4305 = 1 := (row_4305.checked).trans rfl
theorem model_count_4305 : modelRow 4305 = 1 := by decide
theorem modelrow_agrees_4305 : modelRow 4305 = row 4305 := model_count_4305.trans row_count_4305.symm
def row_4306 : RowCertificate 4306 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4306 1) := by change ¬ Nat.Prime 74175157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10596451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4306 2) := by change ¬ Nat.Prime 74166545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14833309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4306 3) := by change ¬ Nat.Prime 74157933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24719311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4306 : row 4306 = 0 := (row_4306.checked).trans rfl
theorem model_count_4306 : modelRow 4306 = 0 := by decide
theorem modelrow_agrees_4306 : modelRow 4306 = row 4306 := model_count_4306.trans row_count_4306.symm
def row_4307 : RowCertificate 4307 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4307 1) := by change ¬ Nat.Prime 74209611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24736537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4307 2) := by change ¬ Nat.Prime 74200997; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5707769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4307 3) := by change ¬ Nat.Prime 74192383; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1107349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4307 : row 4307 = 0 := (row_4307.checked).trans rfl
theorem model_count_4307 : modelRow 4307 = 0 := by decide
theorem modelrow_agrees_4307 : modelRow 4307 = row 4307 := model_count_4307.trans row_count_4307.symm
def row_4308 : RowCertificate 4308 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4308 1) := by change ¬ Nat.Prime 74244073; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 584599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4308 2) := by change ¬ Nat.Prime 74235457; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1400669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4308 3) := by change ¬ Nat.Prime 74226841; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5709757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4308 : row 4308 = 0 := (row_4308.checked).trans rfl
theorem model_count_4308 : modelRow 4308 = 0 := by decide
theorem modelrow_agrees_4308 : modelRow 4308 = row 4308 := model_count_4308.trans row_count_4308.symm
def row_4309 : RowCertificate 4309 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4309 1) := by change ¬ Nat.Prime 74278543; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3909397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4309 2) := by change ¬ Nat.Prime 74269925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14853985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4309 3) := by change ¬ Nat.Prime 74261307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24753769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4309 : row 4309 = 0 := (row_4309.checked).trans rfl
theorem model_count_4309 : modelRow 4309 = 0 := by decide
theorem modelrow_agrees_4309 : modelRow 4309 = row 4309 := model_count_4309.trans row_count_4309.symm
def row_4310 : RowCertificate 4310 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4310 1) := by change ¬ Nat.Prime 74313021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24771007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4310 2) := by change ¬ Nat.Prime 74304401; exact PrattCertificates.composite_of_product (factor := 6173) (cofactor := 12037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4310 3) := by change ¬ Nat.Prime 74295781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10613683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4310 : row 4310 = 0 := (row_4310.checked).trans rfl
theorem model_count_4310 : modelRow 4310 = 0 := by decide
theorem modelrow_agrees_4310 : modelRow 4310 = row 4310 := model_count_4310.trans row_count_4310.symm
def row_4311 : RowCertificate 4311 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4311 1) := by change ¬ Nat.Prime 74347507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5719039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4311 2) := by change ¬ Nat.Prime 74338885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14867777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4311 3) := by change ¬ Nat.Prime 74330263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10618609) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4311 : row 4311 = 0 := (row_4311.checked).trans rfl
theorem model_count_4311 : modelRow 4311 = 0 := by decide
theorem modelrow_agrees_4311 : modelRow 4311 = row 4311 := model_count_4311.trans row_count_4311.symm
def row_4312 : RowCertificate 4312 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4312 1) := by change Nat.Prime 74382001; exact ProjectEulerSolutions.P58PrimeNodes.p_74382001.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4312 2) := by change ¬ Nat.Prime 74373377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5721029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4312 3) := by change ¬ Nat.Prime 74364753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24788251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4312 : row 4312 = 1 := (row_4312.checked).trans rfl
theorem model_count_4312 : modelRow 4312 = 1 := by decide
theorem modelrow_agrees_4312 : modelRow 4312 = row 4312 := model_count_4312.trans row_count_4312.symm
def row_4313 : RowCertificate 4313 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4313 1) := by change ¬ Nat.Prime 74416503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24805501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4313 2) := by change Nat.Prime 74407877; exact ProjectEulerSolutions.P58PrimeNodes.p_74407877.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4313 3) := by change Nat.Prime 74399251; exact ProjectEulerSolutions.P58PrimeNodes.p_74399251.prime
    exact decide_eq_true prime
theorem row_count_4313 : row 4313 = 2 := (row_4313.checked).trans rfl
theorem model_count_4313 : modelRow 4313 = 2 := by decide
theorem modelrow_agrees_4313 : modelRow 4313 = row 4313 := model_count_4313.trans row_count_4313.symm
def row_4314 : RowCertificate 4314 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4314 1) := by change ¬ Nat.Prime 74451013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10635859) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4314 2) := by change ¬ Nat.Prime 74442385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14888477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4314 3) := by change Nat.Prime 74433757; exact ProjectEulerSolutions.P58PrimeNodes.p_74433757.prime
    exact decide_eq_true prime
theorem row_count_4314 : row 4314 = 1 := (row_4314.checked).trans rfl
theorem model_count_4314 : modelRow 4314 = 1 := by decide
theorem modelrow_agrees_4314 : modelRow 4314 = row 4314 := model_count_4314.trans row_count_4314.symm
def row_4315 : RowCertificate 4315 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4315 1) := by change ¬ Nat.Prime 74485531; exact PrattCertificates.composite_of_product (factor := 5779) (cofactor := 12889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4315 2) := by change ¬ Nat.Prime 74476901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2568169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4315 3) := by change ¬ Nat.Prime 74468271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24822757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4315 : row 4315 = 0 := (row_4315.checked).trans rfl
theorem model_count_4315 : modelRow 4315 = 0 := by decide
theorem modelrow_agrees_4315 : modelRow 4315 = row 4315 := model_count_4315.trans row_count_4315.symm
def row_4316 : RowCertificate 4316 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4316 1) := by change ¬ Nat.Prime 74520057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24840019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4316 2) := by change ¬ Nat.Prime 74511425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14902285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4316 3) := by change ¬ Nat.Prime 74502793; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2013589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4316 : row 4316 = 0 := (row_4316.checked).trans rfl
theorem model_count_4316 : modelRow 4316 = 0 := by decide
theorem modelrow_agrees_4316 : modelRow 4316 = row 4316 := model_count_4316.trans row_count_4316.symm
def row_4317 : RowCertificate 4317 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4317 1) := by change ¬ Nat.Prime 74554591; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 943729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4317 2) := by change Nat.Prime 74545957; exact ProjectEulerSolutions.P58PrimeNodes.p_74545957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4317 3) := by change ¬ Nat.Prime 74537323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10648189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4317 : row 4317 = 1 := (row_4317.checked).trans rfl
theorem model_count_4317 : modelRow 4317 = 1 := by decide
theorem modelrow_agrees_4317 : modelRow 4317 = row 4317 := model_count_4317.trans row_count_4317.symm
def row_4318 : RowCertificate 4318 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4318 1) := by change ¬ Nat.Prime 74589133; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1734631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4318 2) := by change ¬ Nat.Prime 74580497; exact PrattCertificates.composite_of_product (factor := 5557) (cofactor := 13421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4318 3) := by change ¬ Nat.Prime 74571861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24857287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4318 : row 4318 = 0 := (row_4318.checked).trans rfl
theorem model_count_4318 : modelRow 4318 = 0 := by decide
theorem modelrow_agrees_4318 : modelRow 4318 = row 4318 := model_count_4318.trans row_count_4318.symm
def row_4319 : RowCertificate 4319 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4319 1) := by change ¬ Nat.Prime 74623683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24874561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4319 2) := by change ¬ Nat.Prime 74615045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14923009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4319 3) := by change ¬ Nat.Prime 74606407; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3926653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4319 : row 4319 = 0 := (row_4319.checked).trans rfl
theorem model_count_4319 : modelRow 4319 = 0 := by decide
theorem modelrow_agrees_4319 : modelRow 4319 = row 4319 := model_count_4319.trans row_count_4319.symm
def row_4320 : RowCertificate 4320 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4320 1) := by change ¬ Nat.Prime 74658241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10665463) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4320 2) := by change ¬ Nat.Prime 74649601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5742277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4320 3) := by change ¬ Nat.Prime 74640961; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 494311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4320 : row 4320 = 0 := (row_4320.checked).trans rfl
theorem model_count_4320 : modelRow 4320 = 0 := by decide
theorem modelrow_agrees_4320 : modelRow 4320 = row 4320 := model_count_4320.trans row_count_4320.symm
def row_4321 : RowCertificate 4321 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4321 1) := by change ¬ Nat.Prime 74692807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10670401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4321 2) := by change ¬ Nat.Prime 74684165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14936833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4321 3) := by change ¬ Nat.Prime 74675523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24891841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4321 : row 4321 = 0 := (row_4321.checked).trans rfl
theorem model_count_4321 : modelRow 4321 = 0 := by decide
theorem modelrow_agrees_4321 : modelRow 4321 = row 4321 := model_count_4321.trans row_count_4321.symm
def row_4322 : RowCertificate 4322 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4322 1) := by change ¬ Nat.Prime 74727381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24909127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4322 2) := by change ¬ Nat.Prime 74718737; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 685493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4322 3) := by change ¬ Nat.Prime 74710093; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2410003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4322 : row 4322 = 0 := (row_4322.checked).trans rfl
theorem model_count_4322 : modelRow 4322 = 0 := by decide
theorem modelrow_agrees_4322 : modelRow 4322 = row 4322 := model_count_4322.trans row_count_4322.symm
def row_4323 : RowCertificate 4323 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4323 1) := by change Nat.Prime 74761963; exact ProjectEulerSolutions.P58PrimeNodes.p_74761963.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4323 2) := by change ¬ Nat.Prime 74753317; exact PrattCertificates.composite_of_product (factor := 1741) (cofactor := 42937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4323 3) := by change Nat.Prime 74744671; exact ProjectEulerSolutions.P58PrimeNodes.p_74744671.prime
    exact decide_eq_true prime
theorem row_count_4323 : row 4323 = 2 := (row_4323.checked).trans rfl
theorem model_count_4323 : modelRow 4323 = 2 := by decide
theorem modelrow_agrees_4323 : modelRow 4323 = row 4323 := model_count_4323.trans row_count_4323.symm
def row_4324 : RowCertificate 4324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4324 1) := by change ¬ Nat.Prime 74796553; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5753581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4324 2) := by change ¬ Nat.Prime 74787905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14957581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4324 3) := by change ¬ Nat.Prime 74779257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24926419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4324 : row 4324 = 0 := (row_4324.checked).trans rfl
theorem model_count_4324 : modelRow 4324 = 0 := by decide
theorem modelrow_agrees_4324 : modelRow 4324 = row 4324 := model_count_4324.trans row_count_4324.symm
def row_4325 : RowCertificate 4325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4325 1) := by change ¬ Nat.Prime 74831151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24943717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4325 2) := by change ¬ Nat.Prime 74822501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5755577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4325 3) := by change ¬ Nat.Prime 74813851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10687693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4325 : row 4325 = 0 := (row_4325.checked).trans rfl
theorem model_count_4325 : modelRow 4325 = 0 := by decide
theorem modelrow_agrees_4325 : modelRow 4325 = row 4325 := model_count_4325.trans row_count_4325.symm
def row_4326 : RowCertificate 4326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4326 1) := by change ¬ Nat.Prime 74865757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3940303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4326 2) := by change ¬ Nat.Prime 74857105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14971421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4326 3) := by change Nat.Prime 74848453; exact ProjectEulerSolutions.P58PrimeNodes.p_74848453.prime
    exact decide_eq_true prime
theorem row_count_4326 : row 4326 = 1 := (row_4326.checked).trans rfl
theorem model_count_4326 : modelRow 4326 = 1 := by decide
theorem modelrow_agrees_4326 : modelRow 4326 = row 4326 := model_count_4326.trans row_count_4326.symm
def row_4327 : RowCertificate 4327 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4327 1) := by change ¬ Nat.Prime 74900371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10700053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4327 2) := by change ¬ Nat.Prime 74891717; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2582473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4327 3) := by change ¬ Nat.Prime 74883063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24961021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4327 : row 4327 = 0 := (row_4327.checked).trans rfl
theorem model_count_4327 : modelRow 4327 = 0 := by decide
theorem modelrow_agrees_4327 : modelRow 4327 = row 4327 := model_count_4327.trans row_count_4327.symm
def row_4328 : RowCertificate 4328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4328 1) := by change ¬ Nat.Prime 74934993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24978331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4328 2) := by change Nat.Prime 74926337; exact ProjectEulerSolutions.P58PrimeNodes.p_74926337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4328 3) := by change ¬ Nat.Prime 74917681; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 589903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4328 : row 4328 = 1 := (row_4328.checked).trans rfl
theorem model_count_4328 : modelRow 4328 = 1 := by decide
theorem modelrow_agrees_4328 : modelRow 4328 = row 4328 := model_count_4328.trans row_count_4328.symm
def row_4329 : RowCertificate 4329 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4329 1) := by change Nat.Prime 74969623; exact ProjectEulerSolutions.P58PrimeNodes.p_74969623.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4329 2) := by change ¬ Nat.Prime 74960965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 14992193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4329 3) := by change ¬ Nat.Prime 74952307; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 336109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4329 : row 4329 = 1 := (row_4329.checked).trans rfl
theorem model_count_4329 : modelRow 4329 = 1 := by decide
theorem modelrow_agrees_4329 : modelRow 4329 = row 4329 := model_count_4329.trans row_count_4329.symm
def row_4330 : RowCertificate 4330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4330 1) := by change ¬ Nat.Prime 75004261; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 539599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4330 2) := by change ¬ Nat.Prime 74995601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1829161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4330 3) := by change ¬ Nat.Prime 74986941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 24995647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4330 : row 4330 = 0 := (row_4330.checked).trans rfl
theorem model_count_4330 : modelRow 4330 = 0 := by decide
theorem modelrow_agrees_4330 : modelRow 4330 = row 4330 := model_count_4330.trans row_count_4330.symm
def row_4331 : RowCertificate 4331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4331 1) := by change ¬ Nat.Prime 75038907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25012969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4331 2) := by change ¬ Nat.Prime 75030245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15006049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4331 3) := by change ¬ Nat.Prime 75021583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10717369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4331 : row 4331 = 0 := (row_4331.checked).trans rfl
theorem model_count_4331 : modelRow 4331 = 0 := by decide
theorem modelrow_agrees_4331 : modelRow 4331 = row 4331 := model_count_4331.trans row_count_4331.symm
def row_4332 : RowCertificate 4332 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4332 1) := by change Nat.Prime 75073561; exact ProjectEulerSolutions.P58PrimeNodes.p_75073561.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4332 2) := by change ¬ Nat.Prime 75064897; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 2028781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4332 3) := by change ¬ Nat.Prime 75056233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10722319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4332 : row 4332 = 1 := (row_4332.checked).trans rfl
theorem model_count_4332 : modelRow 4332 = 1 := by decide
theorem modelrow_agrees_4332 : modelRow 4332 = row 4332 := model_count_4332.trans row_count_4332.symm
def row_4333 : RowCertificate 4333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4333 1) := by change ¬ Nat.Prime 75108223; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 950737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4333 2) := by change ¬ Nat.Prime 75099557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5776889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4333 3) := by change ¬ Nat.Prime 75090891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25030297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4333 : row 4333 = 0 := (row_4333.checked).trans rfl
theorem model_count_4333 : modelRow 4333 = 0 := by decide
theorem modelrow_agrees_4333 : modelRow 4333 = row 4333 := model_count_4333.trans row_count_4333.symm
def row_4334 : RowCertificate 4334 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4334 1) := by change ¬ Nat.Prime 75142893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25047631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4334 2) := by change ¬ Nat.Prime 75134225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15026845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4334 3) := by change ¬ Nat.Prime 75125557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5778889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4334 : row 4334 = 0 := (row_4334.checked).trans rfl
theorem model_count_4334 : modelRow 4334 = 0 := by decide
theorem modelrow_agrees_4334 : modelRow 4334 = row 4334 := model_count_4334.trans row_count_4334.symm
def row_4335 : RowCertificate 4335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4335 1) := by change ¬ Nat.Prime 75177571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10739653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4335 2) := by change Nat.Prime 75168901; exact ProjectEulerSolutions.P58PrimeNodes.p_75168901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4335 3) := by change Nat.Prime 75160231; exact ProjectEulerSolutions.P58PrimeNodes.p_75160231.prime
    exact decide_eq_true prime
theorem row_count_4335 : row 4335 = 2 := (row_4335.checked).trans rfl
theorem model_count_4335 : modelRow 4335 = 2 := by decide
theorem modelrow_agrees_4335 : modelRow 4335 = row 4335 := model_count_4335.trans row_count_4335.symm
def row_4336 : RowCertificate 4336 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4336 1) := by change ¬ Nat.Prime 75212257; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 1122571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4336 2) := by change ¬ Nat.Prime 75203585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15040717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4336 3) := by change ¬ Nat.Prime 75194913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25064971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4336 : row 4336 = 0 := (row_4336.checked).trans rfl
theorem model_count_4336 : modelRow 4336 = 0 := by decide
theorem modelrow_agrees_4336 : modelRow 4336 = row 4336 := model_count_4336.trans row_count_4336.symm
def row_4337 : RowCertificate 4337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4337 1) := by change ¬ Nat.Prime 75246951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25082317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4337 2) := by change ¬ Nat.Prime 75238277; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4425781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4337 3) := by change Nat.Prime 75229603; exact ProjectEulerSolutions.P58PrimeNodes.p_75229603.prime
    exact decide_eq_true prime
theorem row_count_4337 : row 4337 = 1 := (row_4337.checked).trans rfl
theorem model_count_4337 : modelRow 4337 = 1 := by decide
theorem modelrow_agrees_4337 : modelRow 4337 = row 4337 := model_count_4337.trans row_count_4337.symm
def row_4338 : RowCertificate 4338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4338 1) := by change ¬ Nat.Prime 75281653; exact PrattCertificates.composite_of_product (factor := 6217) (cofactor := 12109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4338 2) := by change ¬ Nat.Prime 75272977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5790229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4338 3) := by change ¬ Nat.Prime 75264301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10752043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4338 : row 4338 = 0 := (row_4338.checked).trans rfl
theorem model_count_4338 : modelRow 4338 = 0 := by decide
theorem modelrow_agrees_4338 : modelRow 4338 = row 4338 := model_count_4338.trans row_count_4338.symm
def row_4339 : RowCertificate 4339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4339 1) := by change ¬ Nat.Prime 75316363; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 1031731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4339 2) := by change ¬ Nat.Prime 75307685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15061537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4339 3) := by change ¬ Nat.Prime 75299007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25099669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4339 : row 4339 = 0 := (row_4339.checked).trans rfl
theorem model_count_4339 : modelRow 4339 = 0 := by decide
theorem modelrow_agrees_4339 : modelRow 4339 = row 4339 := model_count_4339.trans row_count_4339.symm
def row_4340 : RowCertificate 4340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4340 1) := by change ¬ Nat.Prime 75351081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25117027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4340 2) := by change Nat.Prime 75342401; exact ProjectEulerSolutions.P58PrimeNodes.p_75342401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4340 3) := by change ¬ Nat.Prime 75333721; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1751947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4340 : row 4340 = 1 := (row_4340.checked).trans rfl
theorem model_count_4340 : modelRow 4340 = 1 := by decide
theorem modelrow_agrees_4340 : modelRow 4340 = row 4340 := model_count_4340.trans row_count_4340.symm
def row_4341 : RowCertificate 4341 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4341 1) := by change ¬ Nat.Prime 75385807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10769401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4341 2) := by change ¬ Nat.Prime 75377125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15075425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4341 3) := by change Nat.Prime 75368443; exact ProjectEulerSolutions.P58PrimeNodes.p_75368443.prime
    exact decide_eq_true prime
theorem row_count_4341 : row 4341 = 1 := (row_4341.checked).trans rfl
theorem model_count_4341 : modelRow 4341 = 1 := by decide
theorem modelrow_agrees_4341 : modelRow 4341 = row 4341 := model_count_4341.trans row_count_4341.symm
def row_4342 : RowCertificate 4342 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4342 1) := by change ¬ Nat.Prime 75420541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10774363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4342 2) := by change Nat.Prime 75411857; exact ProjectEulerSolutions.P58PrimeNodes.p_75411857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4342 3) := by change ¬ Nat.Prime 75403173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25134391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4342 : row 4342 = 1 := (row_4342.checked).trans rfl
theorem model_count_4342 : modelRow 4342 = 1 := by decide
theorem modelrow_agrees_4342 : modelRow 4342 = row 4342 := model_count_4342.trans row_count_4342.symm
def row_4343 : RowCertificate 4343 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4343 1) := by change ¬ Nat.Prime 75455283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25151761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4343 2) := by change ¬ Nat.Prime 75446597; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 667669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4343 3) := by change ¬ Nat.Prime 75437911; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2433481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4343 : row 4343 = 0 := (row_4343.checked).trans rfl
theorem model_count_4343 : modelRow 4343 = 0 := by decide
theorem modelrow_agrees_4343 : modelRow 4343 = row 4343 := model_count_4343.trans row_count_4343.symm
def row_4344 : RowCertificate 4344 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4344 1) := by change Nat.Prime 75490033; exact ProjectEulerSolutions.P58PrimeNodes.p_75490033.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4344 2) := by change ¬ Nat.Prime 75481345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15096269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4344 3) := by change ¬ Nat.Prime 75472657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5805589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4344 : row 4344 = 1 := (row_4344.checked).trans rfl
theorem model_count_4344 : modelRow 4344 = 1 := by decide
theorem modelrow_agrees_4344 : modelRow 4344 = row 4344 := model_count_4344.trans row_count_4344.symm
def row_4345 : RowCertificate 4345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4345 1) := by change ¬ Nat.Prime 75524791; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3974989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4345 2) := by change ¬ Nat.Prime 75516101; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 280729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4345 3) := by change ¬ Nat.Prime 75507411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25169137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4345 : row 4345 = 0 := (row_4345.checked).trans rfl
theorem model_count_4345 : modelRow 4345 = 0 := by decide
theorem modelrow_agrees_4345 : modelRow 4345 = row 4345 := model_count_4345.trans row_count_4345.symm
def row_4346 : RowCertificate 4346 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4346 1) := by change ¬ Nat.Prime 75559557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25186519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4346 2) := by change ¬ Nat.Prime 75550865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15110173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4346 3) := by change ¬ Nat.Prime 75542173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10791739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4346 : row 4346 = 0 := (row_4346.checked).trans rfl
theorem model_count_4346 : modelRow 4346 = 0 := by decide
theorem modelrow_agrees_4346 : modelRow 4346 = row 4346 := model_count_4346.trans row_count_4346.symm
def row_4347 : RowCertificate 4347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4347 1) := by change ¬ Nat.Prime 75594331; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3978649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4347 2) := by change Nat.Prime 75585637; exact ProjectEulerSolutions.P58PrimeNodes.p_75585637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4347 3) := by change ¬ Nat.Prime 75576943; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5813611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4347 : row 4347 = 1 := (row_4347.checked).trans rfl
theorem model_count_4347 : modelRow 4347 = 1 := by decide
theorem modelrow_agrees_4347 : modelRow 4347 = row 4347 := model_count_4347.trans row_count_4347.symm
def row_4348 : RowCertificate 4348 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4348 1) := by change ¬ Nat.Prime 75629113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10804159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4348 2) := by change ¬ Nat.Prime 75620417; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 748717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4348 3) := by change ¬ Nat.Prime 75611721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25203907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4348 : row 4348 = 0 := (row_4348.checked).trans rfl
theorem model_count_4348 : modelRow 4348 = 0 := by decide
theorem modelrow_agrees_4348 : modelRow 4348 = row 4348 := model_count_4348.trans row_count_4348.symm
def row_4349 : RowCertificate 4349 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4349 1) := by change ¬ Nat.Prime 75663903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25221301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4349 2) := by change ¬ Nat.Prime 75655205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15131041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4349 3) := by change ¬ Nat.Prime 75646507; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 464089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4349 : row 4349 = 0 := (row_4349.checked).trans rfl
theorem model_count_4349 : modelRow 4349 = 0 := by decide
theorem modelrow_agrees_4349 : modelRow 4349 = row 4349 := model_count_4349.trans row_count_4349.symm
def row_4350 : RowCertificate 4350 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4350 1) := by change ¬ Nat.Prime 75698701; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5822977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4350 2) := by change ¬ Nat.Prime 75690001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 4452353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4350 3) := by change ¬ Nat.Prime 75681301; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 190633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4350 : row 4350 = 0 := (row_4350.checked).trans rfl
theorem model_count_4350 : modelRow 4350 = 0 := by decide
theorem modelrow_agrees_4350 : modelRow 4350 = row 4350 := model_count_4350.trans row_count_4350.symm
def row_4351 : RowCertificate 4351 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4351 1) := by change Nat.Prime 75733507; exact ProjectEulerSolutions.P58PrimeNodes.p_75733507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4351 2) := by change ¬ Nat.Prime 75724805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 15144961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4351 3) := by change ¬ Nat.Prime 75716103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25238701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4351 : row 4351 = 1 := (row_4351.checked).trans rfl
theorem model_count_4351 : modelRow 4351 = 1 := by decide
theorem modelrow_agrees_4351 : modelRow 4351 = row 4351 := model_count_4351.trans row_count_4351.symm
def row_4352 : RowCertificate 4352 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4352 1) := by change ¬ Nat.Prime 75768321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 25256107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4352 2) := by change ¬ Nat.Prime 75759617; exact PrattCertificates.composite_of_product (factor := 6121) (cofactor := 12377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4352 3) := by change ¬ Nat.Prime 75750913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 10821559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4352 : row 4352 = 0 := (row_4352.checked).trans rfl
theorem model_count_4352 : modelRow 4352 = 0 := by decide
theorem modelrow_agrees_4352 : modelRow 4352 = row 4352 := model_count_4352.trans row_count_4352.symm
private def atomic_4225 : CountSegment 1 10 13120 4224 4225 1933 1933 := CountSegment.single (first := 4224) (initial := 1933) row_count_4225 (by decide)
private theorem atomic_4225_agreement : RowAgreement 4224 4225 := RowAgreement.single (first := 4224) modelrow_agrees_4225
private def atomic_4226 : CountSegment 1 10 13120 4225 4226 1933 1933 := CountSegment.single (first := 4225) (initial := 1933) row_count_4226 (by decide)
private theorem atomic_4226_agreement : RowAgreement 4225 4226 := RowAgreement.single (first := 4225) modelrow_agrees_4226
private def atomic_4227 : CountSegment 1 10 13120 4226 4227 1933 1934 := CountSegment.single (first := 4226) (initial := 1933) row_count_4227 (by decide)
private theorem atomic_4227_agreement : RowAgreement 4226 4227 := RowAgreement.single (first := 4226) modelrow_agrees_4227
private def atomic_4228 : CountSegment 1 10 13120 4227 4228 1934 1934 := CountSegment.single (first := 4227) (initial := 1934) row_count_4228 (by decide)
private theorem atomic_4228_agreement : RowAgreement 4227 4228 := RowAgreement.single (first := 4227) modelrow_agrees_4228
private def atomic_4229 : CountSegment 1 10 13120 4228 4229 1934 1934 := CountSegment.single (first := 4228) (initial := 1934) row_count_4229 (by decide)
private theorem atomic_4229_agreement : RowAgreement 4228 4229 := RowAgreement.single (first := 4228) modelrow_agrees_4229
private def atomic_4230 : CountSegment 1 10 13120 4229 4230 1934 1934 := CountSegment.single (first := 4229) (initial := 1934) row_count_4230 (by decide)
private theorem atomic_4230_agreement : RowAgreement 4229 4230 := RowAgreement.single (first := 4229) modelrow_agrees_4230
private def atomic_4231 : CountSegment 1 10 13120 4230 4231 1934 1934 := CountSegment.single (first := 4230) (initial := 1934) row_count_4231 (by decide)
private theorem atomic_4231_agreement : RowAgreement 4230 4231 := RowAgreement.single (first := 4230) modelrow_agrees_4231
private def atomic_4232 : CountSegment 1 10 13120 4231 4232 1934 1935 := CountSegment.single (first := 4231) (initial := 1934) row_count_4232 (by decide)
private theorem atomic_4232_agreement : RowAgreement 4231 4232 := RowAgreement.single (first := 4231) modelrow_agrees_4232
private def atomic_4233 : CountSegment 1 10 13120 4232 4233 1935 1935 := CountSegment.single (first := 4232) (initial := 1935) row_count_4233 (by decide)
private theorem atomic_4233_agreement : RowAgreement 4232 4233 := RowAgreement.single (first := 4232) modelrow_agrees_4233
private def atomic_4234 : CountSegment 1 10 13120 4233 4234 1935 1935 := CountSegment.single (first := 4233) (initial := 1935) row_count_4234 (by decide)
private theorem atomic_4234_agreement : RowAgreement 4233 4234 := RowAgreement.single (first := 4233) modelrow_agrees_4234
private def atomic_4235 : CountSegment 1 10 13120 4234 4235 1935 1936 := CountSegment.single (first := 4234) (initial := 1935) row_count_4235 (by decide)
private theorem atomic_4235_agreement : RowAgreement 4234 4235 := RowAgreement.single (first := 4234) modelrow_agrees_4235
private def atomic_4236 : CountSegment 1 10 13120 4235 4236 1936 1936 := CountSegment.single (first := 4235) (initial := 1936) row_count_4236 (by decide)
private theorem atomic_4236_agreement : RowAgreement 4235 4236 := RowAgreement.single (first := 4235) modelrow_agrees_4236
private def atomic_4237 : CountSegment 1 10 13120 4236 4237 1936 1936 := CountSegment.single (first := 4236) (initial := 1936) row_count_4237 (by decide)
private theorem atomic_4237_agreement : RowAgreement 4236 4237 := RowAgreement.single (first := 4236) modelrow_agrees_4237
private def atomic_4238 : CountSegment 1 10 13120 4237 4238 1936 1937 := CountSegment.single (first := 4237) (initial := 1936) row_count_4238 (by decide)
private theorem atomic_4238_agreement : RowAgreement 4237 4238 := RowAgreement.single (first := 4237) modelrow_agrees_4238
private def atomic_4239 : CountSegment 1 10 13120 4238 4239 1937 1938 := CountSegment.single (first := 4238) (initial := 1937) row_count_4239 (by decide)
private theorem atomic_4239_agreement : RowAgreement 4238 4239 := RowAgreement.single (first := 4238) modelrow_agrees_4239
private def atomic_4240 : CountSegment 1 10 13120 4239 4240 1938 1939 := CountSegment.single (first := 4239) (initial := 1938) row_count_4240 (by decide)
private theorem atomic_4240_agreement : RowAgreement 4239 4240 := RowAgreement.single (first := 4239) modelrow_agrees_4240
private def atomic_4241 : CountSegment 1 10 13120 4240 4241 1939 1939 := CountSegment.single (first := 4240) (initial := 1939) row_count_4241 (by decide)
private theorem atomic_4241_agreement : RowAgreement 4240 4241 := RowAgreement.single (first := 4240) modelrow_agrees_4241
private def atomic_4242 : CountSegment 1 10 13120 4241 4242 1939 1940 := CountSegment.single (first := 4241) (initial := 1939) row_count_4242 (by decide)
private theorem atomic_4242_agreement : RowAgreement 4241 4242 := RowAgreement.single (first := 4241) modelrow_agrees_4242
private def atomic_4243 : CountSegment 1 10 13120 4242 4243 1940 1940 := CountSegment.single (first := 4242) (initial := 1940) row_count_4243 (by decide)
private theorem atomic_4243_agreement : RowAgreement 4242 4243 := RowAgreement.single (first := 4242) modelrow_agrees_4243
private def atomic_4244 : CountSegment 1 10 13120 4243 4244 1940 1940 := CountSegment.single (first := 4243) (initial := 1940) row_count_4244 (by decide)
private theorem atomic_4244_agreement : RowAgreement 4243 4244 := RowAgreement.single (first := 4243) modelrow_agrees_4244
private def atomic_4245 : CountSegment 1 10 13120 4244 4245 1940 1940 := CountSegment.single (first := 4244) (initial := 1940) row_count_4245 (by decide)
private theorem atomic_4245_agreement : RowAgreement 4244 4245 := RowAgreement.single (first := 4244) modelrow_agrees_4245
private def atomic_4246 : CountSegment 1 10 13120 4245 4246 1940 1940 := CountSegment.single (first := 4245) (initial := 1940) row_count_4246 (by decide)
private theorem atomic_4246_agreement : RowAgreement 4245 4246 := RowAgreement.single (first := 4245) modelrow_agrees_4246
private def atomic_4247 : CountSegment 1 10 13120 4246 4247 1940 1940 := CountSegment.single (first := 4246) (initial := 1940) row_count_4247 (by decide)
private theorem atomic_4247_agreement : RowAgreement 4246 4247 := RowAgreement.single (first := 4246) modelrow_agrees_4247
private def atomic_4248 : CountSegment 1 10 13120 4247 4248 1940 1941 := CountSegment.single (first := 4247) (initial := 1940) row_count_4248 (by decide)
private theorem atomic_4248_agreement : RowAgreement 4247 4248 := RowAgreement.single (first := 4247) modelrow_agrees_4248
private def atomic_4249 : CountSegment 1 10 13120 4248 4249 1941 1941 := CountSegment.single (first := 4248) (initial := 1941) row_count_4249 (by decide)
private theorem atomic_4249_agreement : RowAgreement 4248 4249 := RowAgreement.single (first := 4248) modelrow_agrees_4249
private def atomic_4250 : CountSegment 1 10 13120 4249 4250 1941 1942 := CountSegment.single (first := 4249) (initial := 1941) row_count_4250 (by decide)
private theorem atomic_4250_agreement : RowAgreement 4249 4250 := RowAgreement.single (first := 4249) modelrow_agrees_4250
private def atomic_4251 : CountSegment 1 10 13120 4250 4251 1942 1943 := CountSegment.single (first := 4250) (initial := 1942) row_count_4251 (by decide)
private theorem atomic_4251_agreement : RowAgreement 4250 4251 := RowAgreement.single (first := 4250) modelrow_agrees_4251
private def atomic_4252 : CountSegment 1 10 13120 4251 4252 1943 1943 := CountSegment.single (first := 4251) (initial := 1943) row_count_4252 (by decide)
private theorem atomic_4252_agreement : RowAgreement 4251 4252 := RowAgreement.single (first := 4251) modelrow_agrees_4252
private def atomic_4253 : CountSegment 1 10 13120 4252 4253 1943 1943 := CountSegment.single (first := 4252) (initial := 1943) row_count_4253 (by decide)
private theorem atomic_4253_agreement : RowAgreement 4252 4253 := RowAgreement.single (first := 4252) modelrow_agrees_4253
private def atomic_4254 : CountSegment 1 10 13120 4253 4254 1943 1943 := CountSegment.single (first := 4253) (initial := 1943) row_count_4254 (by decide)
private theorem atomic_4254_agreement : RowAgreement 4253 4254 := RowAgreement.single (first := 4253) modelrow_agrees_4254
private def atomic_4255 : CountSegment 1 10 13120 4254 4255 1943 1944 := CountSegment.single (first := 4254) (initial := 1943) row_count_4255 (by decide)
private theorem atomic_4255_agreement : RowAgreement 4254 4255 := RowAgreement.single (first := 4254) modelrow_agrees_4255
private def atomic_4256 : CountSegment 1 10 13120 4255 4256 1944 1944 := CountSegment.single (first := 4255) (initial := 1944) row_count_4256 (by decide)
private theorem atomic_4256_agreement : RowAgreement 4255 4256 := RowAgreement.single (first := 4255) modelrow_agrees_4256
private def atomic_4257 : CountSegment 1 10 13120 4256 4257 1944 1944 := CountSegment.single (first := 4256) (initial := 1944) row_count_4257 (by decide)
private theorem atomic_4257_agreement : RowAgreement 4256 4257 := RowAgreement.single (first := 4256) modelrow_agrees_4257
private def atomic_4258 : CountSegment 1 10 13120 4257 4258 1944 1944 := CountSegment.single (first := 4257) (initial := 1944) row_count_4258 (by decide)
private theorem atomic_4258_agreement : RowAgreement 4257 4258 := RowAgreement.single (first := 4257) modelrow_agrees_4258
private def atomic_4259 : CountSegment 1 10 13120 4258 4259 1944 1945 := CountSegment.single (first := 4258) (initial := 1944) row_count_4259 (by decide)
private theorem atomic_4259_agreement : RowAgreement 4258 4259 := RowAgreement.single (first := 4258) modelrow_agrees_4259
private def atomic_4260 : CountSegment 1 10 13120 4259 4260 1945 1945 := CountSegment.single (first := 4259) (initial := 1945) row_count_4260 (by decide)
private theorem atomic_4260_agreement : RowAgreement 4259 4260 := RowAgreement.single (first := 4259) modelrow_agrees_4260
private def atomic_4261 : CountSegment 1 10 13120 4260 4261 1945 1946 := CountSegment.single (first := 4260) (initial := 1945) row_count_4261 (by decide)
private theorem atomic_4261_agreement : RowAgreement 4260 4261 := RowAgreement.single (first := 4260) modelrow_agrees_4261
private def atomic_4262 : CountSegment 1 10 13120 4261 4262 1946 1946 := CountSegment.single (first := 4261) (initial := 1946) row_count_4262 (by decide)
private theorem atomic_4262_agreement : RowAgreement 4261 4262 := RowAgreement.single (first := 4261) modelrow_agrees_4262
private def atomic_4263 : CountSegment 1 10 13120 4262 4263 1946 1947 := CountSegment.single (first := 4262) (initial := 1946) row_count_4263 (by decide)
private theorem atomic_4263_agreement : RowAgreement 4262 4263 := RowAgreement.single (first := 4262) modelrow_agrees_4263
private def atomic_4264 : CountSegment 1 10 13120 4263 4264 1947 1947 := CountSegment.single (first := 4263) (initial := 1947) row_count_4264 (by decide)
private theorem atomic_4264_agreement : RowAgreement 4263 4264 := RowAgreement.single (first := 4263) modelrow_agrees_4264
private def atomic_4265 : CountSegment 1 10 13120 4264 4265 1947 1947 := CountSegment.single (first := 4264) (initial := 1947) row_count_4265 (by decide)
private theorem atomic_4265_agreement : RowAgreement 4264 4265 := RowAgreement.single (first := 4264) modelrow_agrees_4265
private def atomic_4266 : CountSegment 1 10 13120 4265 4266 1947 1947 := CountSegment.single (first := 4265) (initial := 1947) row_count_4266 (by decide)
private theorem atomic_4266_agreement : RowAgreement 4265 4266 := RowAgreement.single (first := 4265) modelrow_agrees_4266
private def atomic_4267 : CountSegment 1 10 13120 4266 4267 1947 1947 := CountSegment.single (first := 4266) (initial := 1947) row_count_4267 (by decide)
private theorem atomic_4267_agreement : RowAgreement 4266 4267 := RowAgreement.single (first := 4266) modelrow_agrees_4267
private def atomic_4268 : CountSegment 1 10 13120 4267 4268 1947 1947 := CountSegment.single (first := 4267) (initial := 1947) row_count_4268 (by decide)
private theorem atomic_4268_agreement : RowAgreement 4267 4268 := RowAgreement.single (first := 4267) modelrow_agrees_4268
private def atomic_4269 : CountSegment 1 10 13120 4268 4269 1947 1947 := CountSegment.single (first := 4268) (initial := 1947) row_count_4269 (by decide)
private theorem atomic_4269_agreement : RowAgreement 4268 4269 := RowAgreement.single (first := 4268) modelrow_agrees_4269
private def atomic_4270 : CountSegment 1 10 13120 4269 4270 1947 1949 := CountSegment.single (first := 4269) (initial := 1947) row_count_4270 (by decide)
private theorem atomic_4270_agreement : RowAgreement 4269 4270 := RowAgreement.single (first := 4269) modelrow_agrees_4270
private def atomic_4271 : CountSegment 1 10 13120 4270 4271 1949 1950 := CountSegment.single (first := 4270) (initial := 1949) row_count_4271 (by decide)
private theorem atomic_4271_agreement : RowAgreement 4270 4271 := RowAgreement.single (first := 4270) modelrow_agrees_4271
private def atomic_4272 : CountSegment 1 10 13120 4271 4272 1950 1951 := CountSegment.single (first := 4271) (initial := 1950) row_count_4272 (by decide)
private theorem atomic_4272_agreement : RowAgreement 4271 4272 := RowAgreement.single (first := 4271) modelrow_agrees_4272
private def atomic_4273 : CountSegment 1 10 13120 4272 4273 1951 1951 := CountSegment.single (first := 4272) (initial := 1951) row_count_4273 (by decide)
private theorem atomic_4273_agreement : RowAgreement 4272 4273 := RowAgreement.single (first := 4272) modelrow_agrees_4273
private def atomic_4274 : CountSegment 1 10 13120 4273 4274 1951 1951 := CountSegment.single (first := 4273) (initial := 1951) row_count_4274 (by decide)
private theorem atomic_4274_agreement : RowAgreement 4273 4274 := RowAgreement.single (first := 4273) modelrow_agrees_4274
private def atomic_4275 : CountSegment 1 10 13120 4274 4275 1951 1952 := CountSegment.single (first := 4274) (initial := 1951) row_count_4275 (by decide)
private theorem atomic_4275_agreement : RowAgreement 4274 4275 := RowAgreement.single (first := 4274) modelrow_agrees_4275
private def atomic_4276 : CountSegment 1 10 13120 4275 4276 1952 1953 := CountSegment.single (first := 4275) (initial := 1952) row_count_4276 (by decide)
private theorem atomic_4276_agreement : RowAgreement 4275 4276 := RowAgreement.single (first := 4275) modelrow_agrees_4276
private def atomic_4277 : CountSegment 1 10 13120 4276 4277 1953 1954 := CountSegment.single (first := 4276) (initial := 1953) row_count_4277 (by decide)
private theorem atomic_4277_agreement : RowAgreement 4276 4277 := RowAgreement.single (first := 4276) modelrow_agrees_4277
private def atomic_4278 : CountSegment 1 10 13120 4277 4278 1954 1954 := CountSegment.single (first := 4277) (initial := 1954) row_count_4278 (by decide)
private theorem atomic_4278_agreement : RowAgreement 4277 4278 := RowAgreement.single (first := 4277) modelrow_agrees_4278
private def atomic_4279 : CountSegment 1 10 13120 4278 4279 1954 1954 := CountSegment.single (first := 4278) (initial := 1954) row_count_4279 (by decide)
private theorem atomic_4279_agreement : RowAgreement 4278 4279 := RowAgreement.single (first := 4278) modelrow_agrees_4279
private def atomic_4280 : CountSegment 1 10 13120 4279 4280 1954 1955 := CountSegment.single (first := 4279) (initial := 1954) row_count_4280 (by decide)
private theorem atomic_4280_agreement : RowAgreement 4279 4280 := RowAgreement.single (first := 4279) modelrow_agrees_4280
private def atomic_4281 : CountSegment 1 10 13120 4280 4281 1955 1956 := CountSegment.single (first := 4280) (initial := 1955) row_count_4281 (by decide)
private theorem atomic_4281_agreement : RowAgreement 4280 4281 := RowAgreement.single (first := 4280) modelrow_agrees_4281
private def atomic_4282 : CountSegment 1 10 13120 4281 4282 1956 1957 := CountSegment.single (first := 4281) (initial := 1956) row_count_4282 (by decide)
private theorem atomic_4282_agreement : RowAgreement 4281 4282 := RowAgreement.single (first := 4281) modelrow_agrees_4282
private def atomic_4283 : CountSegment 1 10 13120 4282 4283 1957 1957 := CountSegment.single (first := 4282) (initial := 1957) row_count_4283 (by decide)
private theorem atomic_4283_agreement : RowAgreement 4282 4283 := RowAgreement.single (first := 4282) modelrow_agrees_4283
private def atomic_4284 : CountSegment 1 10 13120 4283 4284 1957 1958 := CountSegment.single (first := 4283) (initial := 1957) row_count_4284 (by decide)
private theorem atomic_4284_agreement : RowAgreement 4283 4284 := RowAgreement.single (first := 4283) modelrow_agrees_4284
private def atomic_4285 : CountSegment 1 10 13120 4284 4285 1958 1958 := CountSegment.single (first := 4284) (initial := 1958) row_count_4285 (by decide)
private theorem atomic_4285_agreement : RowAgreement 4284 4285 := RowAgreement.single (first := 4284) modelrow_agrees_4285
private def atomic_4286 : CountSegment 1 10 13120 4285 4286 1958 1958 := CountSegment.single (first := 4285) (initial := 1958) row_count_4286 (by decide)
private theorem atomic_4286_agreement : RowAgreement 4285 4286 := RowAgreement.single (first := 4285) modelrow_agrees_4286
private def atomic_4287 : CountSegment 1 10 13120 4286 4287 1958 1958 := CountSegment.single (first := 4286) (initial := 1958) row_count_4287 (by decide)
private theorem atomic_4287_agreement : RowAgreement 4286 4287 := RowAgreement.single (first := 4286) modelrow_agrees_4287
private def atomic_4288 : CountSegment 1 10 13120 4287 4288 1958 1959 := CountSegment.single (first := 4287) (initial := 1958) row_count_4288 (by decide)
private theorem atomic_4288_agreement : RowAgreement 4287 4288 := RowAgreement.single (first := 4287) modelrow_agrees_4288
private def atomic_4289 : CountSegment 1 10 13120 4288 4289 1959 1959 := CountSegment.single (first := 4288) (initial := 1959) row_count_4289 (by decide)
private theorem atomic_4289_agreement : RowAgreement 4288 4289 := RowAgreement.single (first := 4288) modelrow_agrees_4289
private def atomic_4290 : CountSegment 1 10 13120 4289 4290 1959 1959 := CountSegment.single (first := 4289) (initial := 1959) row_count_4290 (by decide)
private theorem atomic_4290_agreement : RowAgreement 4289 4290 := RowAgreement.single (first := 4289) modelrow_agrees_4290
private def atomic_4291 : CountSegment 1 10 13120 4290 4291 1959 1959 := CountSegment.single (first := 4290) (initial := 1959) row_count_4291 (by decide)
private theorem atomic_4291_agreement : RowAgreement 4290 4291 := RowAgreement.single (first := 4290) modelrow_agrees_4291
private def atomic_4292 : CountSegment 1 10 13120 4291 4292 1959 1960 := CountSegment.single (first := 4291) (initial := 1959) row_count_4292 (by decide)
private theorem atomic_4292_agreement : RowAgreement 4291 4292 := RowAgreement.single (first := 4291) modelrow_agrees_4292
private def atomic_4293 : CountSegment 1 10 13120 4292 4293 1960 1960 := CountSegment.single (first := 4292) (initial := 1960) row_count_4293 (by decide)
private theorem atomic_4293_agreement : RowAgreement 4292 4293 := RowAgreement.single (first := 4292) modelrow_agrees_4293
private def atomic_4294 : CountSegment 1 10 13120 4293 4294 1960 1960 := CountSegment.single (first := 4293) (initial := 1960) row_count_4294 (by decide)
private theorem atomic_4294_agreement : RowAgreement 4293 4294 := RowAgreement.single (first := 4293) modelrow_agrees_4294
private def atomic_4295 : CountSegment 1 10 13120 4294 4295 1960 1960 := CountSegment.single (first := 4294) (initial := 1960) row_count_4295 (by decide)
private theorem atomic_4295_agreement : RowAgreement 4294 4295 := RowAgreement.single (first := 4294) modelrow_agrees_4295
private def atomic_4296 : CountSegment 1 10 13120 4295 4296 1960 1960 := CountSegment.single (first := 4295) (initial := 1960) row_count_4296 (by decide)
private theorem atomic_4296_agreement : RowAgreement 4295 4296 := RowAgreement.single (first := 4295) modelrow_agrees_4296
private def atomic_4297 : CountSegment 1 10 13120 4296 4297 1960 1960 := CountSegment.single (first := 4296) (initial := 1960) row_count_4297 (by decide)
private theorem atomic_4297_agreement : RowAgreement 4296 4297 := RowAgreement.single (first := 4296) modelrow_agrees_4297
private def atomic_4298 : CountSegment 1 10 13120 4297 4298 1960 1960 := CountSegment.single (first := 4297) (initial := 1960) row_count_4298 (by decide)
private theorem atomic_4298_agreement : RowAgreement 4297 4298 := RowAgreement.single (first := 4297) modelrow_agrees_4298
private def atomic_4299 : CountSegment 1 10 13120 4298 4299 1960 1960 := CountSegment.single (first := 4298) (initial := 1960) row_count_4299 (by decide)
private theorem atomic_4299_agreement : RowAgreement 4298 4299 := RowAgreement.single (first := 4298) modelrow_agrees_4299
private def atomic_4300 : CountSegment 1 10 13120 4299 4300 1960 1960 := CountSegment.single (first := 4299) (initial := 1960) row_count_4300 (by decide)
private theorem atomic_4300_agreement : RowAgreement 4299 4300 := RowAgreement.single (first := 4299) modelrow_agrees_4300
private def atomic_4301 : CountSegment 1 10 13120 4300 4301 1960 1961 := CountSegment.single (first := 4300) (initial := 1960) row_count_4301 (by decide)
private theorem atomic_4301_agreement : RowAgreement 4300 4301 := RowAgreement.single (first := 4300) modelrow_agrees_4301
private def atomic_4302 : CountSegment 1 10 13120 4301 4302 1961 1962 := CountSegment.single (first := 4301) (initial := 1961) row_count_4302 (by decide)
private theorem atomic_4302_agreement : RowAgreement 4301 4302 := RowAgreement.single (first := 4301) modelrow_agrees_4302
private def atomic_4303 : CountSegment 1 10 13120 4302 4303 1962 1962 := CountSegment.single (first := 4302) (initial := 1962) row_count_4303 (by decide)
private theorem atomic_4303_agreement : RowAgreement 4302 4303 := RowAgreement.single (first := 4302) modelrow_agrees_4303
private def atomic_4304 : CountSegment 1 10 13120 4303 4304 1962 1962 := CountSegment.single (first := 4303) (initial := 1962) row_count_4304 (by decide)
private theorem atomic_4304_agreement : RowAgreement 4303 4304 := RowAgreement.single (first := 4303) modelrow_agrees_4304
private def atomic_4305 : CountSegment 1 10 13120 4304 4305 1962 1963 := CountSegment.single (first := 4304) (initial := 1962) row_count_4305 (by decide)
private theorem atomic_4305_agreement : RowAgreement 4304 4305 := RowAgreement.single (first := 4304) modelrow_agrees_4305
private def atomic_4306 : CountSegment 1 10 13120 4305 4306 1963 1963 := CountSegment.single (first := 4305) (initial := 1963) row_count_4306 (by decide)
private theorem atomic_4306_agreement : RowAgreement 4305 4306 := RowAgreement.single (first := 4305) modelrow_agrees_4306
private def atomic_4307 : CountSegment 1 10 13120 4306 4307 1963 1963 := CountSegment.single (first := 4306) (initial := 1963) row_count_4307 (by decide)
private theorem atomic_4307_agreement : RowAgreement 4306 4307 := RowAgreement.single (first := 4306) modelrow_agrees_4307
private def atomic_4308 : CountSegment 1 10 13120 4307 4308 1963 1963 := CountSegment.single (first := 4307) (initial := 1963) row_count_4308 (by decide)
private theorem atomic_4308_agreement : RowAgreement 4307 4308 := RowAgreement.single (first := 4307) modelrow_agrees_4308
private def atomic_4309 : CountSegment 1 10 13120 4308 4309 1963 1963 := CountSegment.single (first := 4308) (initial := 1963) row_count_4309 (by decide)
private theorem atomic_4309_agreement : RowAgreement 4308 4309 := RowAgreement.single (first := 4308) modelrow_agrees_4309
private def atomic_4310 : CountSegment 1 10 13120 4309 4310 1963 1963 := CountSegment.single (first := 4309) (initial := 1963) row_count_4310 (by decide)
private theorem atomic_4310_agreement : RowAgreement 4309 4310 := RowAgreement.single (first := 4309) modelrow_agrees_4310
private def atomic_4311 : CountSegment 1 10 13120 4310 4311 1963 1963 := CountSegment.single (first := 4310) (initial := 1963) row_count_4311 (by decide)
private theorem atomic_4311_agreement : RowAgreement 4310 4311 := RowAgreement.single (first := 4310) modelrow_agrees_4311
private def atomic_4312 : CountSegment 1 10 13120 4311 4312 1963 1964 := CountSegment.single (first := 4311) (initial := 1963) row_count_4312 (by decide)
private theorem atomic_4312_agreement : RowAgreement 4311 4312 := RowAgreement.single (first := 4311) modelrow_agrees_4312
private def atomic_4313 : CountSegment 1 10 13120 4312 4313 1964 1966 := CountSegment.single (first := 4312) (initial := 1964) row_count_4313 (by decide)
private theorem atomic_4313_agreement : RowAgreement 4312 4313 := RowAgreement.single (first := 4312) modelrow_agrees_4313
private def atomic_4314 : CountSegment 1 10 13120 4313 4314 1966 1967 := CountSegment.single (first := 4313) (initial := 1966) row_count_4314 (by decide)
private theorem atomic_4314_agreement : RowAgreement 4313 4314 := RowAgreement.single (first := 4313) modelrow_agrees_4314
private def atomic_4315 : CountSegment 1 10 13120 4314 4315 1967 1967 := CountSegment.single (first := 4314) (initial := 1967) row_count_4315 (by decide)
private theorem atomic_4315_agreement : RowAgreement 4314 4315 := RowAgreement.single (first := 4314) modelrow_agrees_4315
private def atomic_4316 : CountSegment 1 10 13120 4315 4316 1967 1967 := CountSegment.single (first := 4315) (initial := 1967) row_count_4316 (by decide)
private theorem atomic_4316_agreement : RowAgreement 4315 4316 := RowAgreement.single (first := 4315) modelrow_agrees_4316
private def atomic_4317 : CountSegment 1 10 13120 4316 4317 1967 1968 := CountSegment.single (first := 4316) (initial := 1967) row_count_4317 (by decide)
private theorem atomic_4317_agreement : RowAgreement 4316 4317 := RowAgreement.single (first := 4316) modelrow_agrees_4317
private def atomic_4318 : CountSegment 1 10 13120 4317 4318 1968 1968 := CountSegment.single (first := 4317) (initial := 1968) row_count_4318 (by decide)
private theorem atomic_4318_agreement : RowAgreement 4317 4318 := RowAgreement.single (first := 4317) modelrow_agrees_4318
private def atomic_4319 : CountSegment 1 10 13120 4318 4319 1968 1968 := CountSegment.single (first := 4318) (initial := 1968) row_count_4319 (by decide)
private theorem atomic_4319_agreement : RowAgreement 4318 4319 := RowAgreement.single (first := 4318) modelrow_agrees_4319
private def atomic_4320 : CountSegment 1 10 13120 4319 4320 1968 1968 := CountSegment.single (first := 4319) (initial := 1968) row_count_4320 (by decide)
private theorem atomic_4320_agreement : RowAgreement 4319 4320 := RowAgreement.single (first := 4319) modelrow_agrees_4320
private def atomic_4321 : CountSegment 1 10 13120 4320 4321 1968 1968 := CountSegment.single (first := 4320) (initial := 1968) row_count_4321 (by decide)
private theorem atomic_4321_agreement : RowAgreement 4320 4321 := RowAgreement.single (first := 4320) modelrow_agrees_4321
private def atomic_4322 : CountSegment 1 10 13120 4321 4322 1968 1968 := CountSegment.single (first := 4321) (initial := 1968) row_count_4322 (by decide)
private theorem atomic_4322_agreement : RowAgreement 4321 4322 := RowAgreement.single (first := 4321) modelrow_agrees_4322
private def atomic_4323 : CountSegment 1 10 13120 4322 4323 1968 1970 := CountSegment.single (first := 4322) (initial := 1968) row_count_4323 (by decide)
private theorem atomic_4323_agreement : RowAgreement 4322 4323 := RowAgreement.single (first := 4322) modelrow_agrees_4323
private def atomic_4324 : CountSegment 1 10 13120 4323 4324 1970 1970 := CountSegment.single (first := 4323) (initial := 1970) row_count_4324 (by decide)
private theorem atomic_4324_agreement : RowAgreement 4323 4324 := RowAgreement.single (first := 4323) modelrow_agrees_4324
private def atomic_4325 : CountSegment 1 10 13120 4324 4325 1970 1970 := CountSegment.single (first := 4324) (initial := 1970) row_count_4325 (by decide)
private theorem atomic_4325_agreement : RowAgreement 4324 4325 := RowAgreement.single (first := 4324) modelrow_agrees_4325
private def atomic_4326 : CountSegment 1 10 13120 4325 4326 1970 1971 := CountSegment.single (first := 4325) (initial := 1970) row_count_4326 (by decide)
private theorem atomic_4326_agreement : RowAgreement 4325 4326 := RowAgreement.single (first := 4325) modelrow_agrees_4326
private def atomic_4327 : CountSegment 1 10 13120 4326 4327 1971 1971 := CountSegment.single (first := 4326) (initial := 1971) row_count_4327 (by decide)
private theorem atomic_4327_agreement : RowAgreement 4326 4327 := RowAgreement.single (first := 4326) modelrow_agrees_4327
private def atomic_4328 : CountSegment 1 10 13120 4327 4328 1971 1972 := CountSegment.single (first := 4327) (initial := 1971) row_count_4328 (by decide)
private theorem atomic_4328_agreement : RowAgreement 4327 4328 := RowAgreement.single (first := 4327) modelrow_agrees_4328
private def atomic_4329 : CountSegment 1 10 13120 4328 4329 1972 1973 := CountSegment.single (first := 4328) (initial := 1972) row_count_4329 (by decide)
private theorem atomic_4329_agreement : RowAgreement 4328 4329 := RowAgreement.single (first := 4328) modelrow_agrees_4329
private def atomic_4330 : CountSegment 1 10 13120 4329 4330 1973 1973 := CountSegment.single (first := 4329) (initial := 1973) row_count_4330 (by decide)
private theorem atomic_4330_agreement : RowAgreement 4329 4330 := RowAgreement.single (first := 4329) modelrow_agrees_4330
private def atomic_4331 : CountSegment 1 10 13120 4330 4331 1973 1973 := CountSegment.single (first := 4330) (initial := 1973) row_count_4331 (by decide)
private theorem atomic_4331_agreement : RowAgreement 4330 4331 := RowAgreement.single (first := 4330) modelrow_agrees_4331
private def atomic_4332 : CountSegment 1 10 13120 4331 4332 1973 1974 := CountSegment.single (first := 4331) (initial := 1973) row_count_4332 (by decide)
private theorem atomic_4332_agreement : RowAgreement 4331 4332 := RowAgreement.single (first := 4331) modelrow_agrees_4332
private def atomic_4333 : CountSegment 1 10 13120 4332 4333 1974 1974 := CountSegment.single (first := 4332) (initial := 1974) row_count_4333 (by decide)
private theorem atomic_4333_agreement : RowAgreement 4332 4333 := RowAgreement.single (first := 4332) modelrow_agrees_4333
private def atomic_4334 : CountSegment 1 10 13120 4333 4334 1974 1974 := CountSegment.single (first := 4333) (initial := 1974) row_count_4334 (by decide)
private theorem atomic_4334_agreement : RowAgreement 4333 4334 := RowAgreement.single (first := 4333) modelrow_agrees_4334
private def atomic_4335 : CountSegment 1 10 13120 4334 4335 1974 1976 := CountSegment.single (first := 4334) (initial := 1974) row_count_4335 (by decide)
private theorem atomic_4335_agreement : RowAgreement 4334 4335 := RowAgreement.single (first := 4334) modelrow_agrees_4335
private def atomic_4336 : CountSegment 1 10 13120 4335 4336 1976 1976 := CountSegment.single (first := 4335) (initial := 1976) row_count_4336 (by decide)
private theorem atomic_4336_agreement : RowAgreement 4335 4336 := RowAgreement.single (first := 4335) modelrow_agrees_4336
private def atomic_4337 : CountSegment 1 10 13120 4336 4337 1976 1977 := CountSegment.single (first := 4336) (initial := 1976) row_count_4337 (by decide)
private theorem atomic_4337_agreement : RowAgreement 4336 4337 := RowAgreement.single (first := 4336) modelrow_agrees_4337
private def atomic_4338 : CountSegment 1 10 13120 4337 4338 1977 1977 := CountSegment.single (first := 4337) (initial := 1977) row_count_4338 (by decide)
private theorem atomic_4338_agreement : RowAgreement 4337 4338 := RowAgreement.single (first := 4337) modelrow_agrees_4338
private def atomic_4339 : CountSegment 1 10 13120 4338 4339 1977 1977 := CountSegment.single (first := 4338) (initial := 1977) row_count_4339 (by decide)
private theorem atomic_4339_agreement : RowAgreement 4338 4339 := RowAgreement.single (first := 4338) modelrow_agrees_4339
private def atomic_4340 : CountSegment 1 10 13120 4339 4340 1977 1978 := CountSegment.single (first := 4339) (initial := 1977) row_count_4340 (by decide)
private theorem atomic_4340_agreement : RowAgreement 4339 4340 := RowAgreement.single (first := 4339) modelrow_agrees_4340
private def atomic_4341 : CountSegment 1 10 13120 4340 4341 1978 1979 := CountSegment.single (first := 4340) (initial := 1978) row_count_4341 (by decide)
private theorem atomic_4341_agreement : RowAgreement 4340 4341 := RowAgreement.single (first := 4340) modelrow_agrees_4341
private def atomic_4342 : CountSegment 1 10 13120 4341 4342 1979 1980 := CountSegment.single (first := 4341) (initial := 1979) row_count_4342 (by decide)
private theorem atomic_4342_agreement : RowAgreement 4341 4342 := RowAgreement.single (first := 4341) modelrow_agrees_4342
private def atomic_4343 : CountSegment 1 10 13120 4342 4343 1980 1980 := CountSegment.single (first := 4342) (initial := 1980) row_count_4343 (by decide)
private theorem atomic_4343_agreement : RowAgreement 4342 4343 := RowAgreement.single (first := 4342) modelrow_agrees_4343
private def atomic_4344 : CountSegment 1 10 13120 4343 4344 1980 1981 := CountSegment.single (first := 4343) (initial := 1980) row_count_4344 (by decide)
private theorem atomic_4344_agreement : RowAgreement 4343 4344 := RowAgreement.single (first := 4343) modelrow_agrees_4344
private def atomic_4345 : CountSegment 1 10 13120 4344 4345 1981 1981 := CountSegment.single (first := 4344) (initial := 1981) row_count_4345 (by decide)
private theorem atomic_4345_agreement : RowAgreement 4344 4345 := RowAgreement.single (first := 4344) modelrow_agrees_4345
private def atomic_4346 : CountSegment 1 10 13120 4345 4346 1981 1981 := CountSegment.single (first := 4345) (initial := 1981) row_count_4346 (by decide)
private theorem atomic_4346_agreement : RowAgreement 4345 4346 := RowAgreement.single (first := 4345) modelrow_agrees_4346
private def atomic_4347 : CountSegment 1 10 13120 4346 4347 1981 1982 := CountSegment.single (first := 4346) (initial := 1981) row_count_4347 (by decide)
private theorem atomic_4347_agreement : RowAgreement 4346 4347 := RowAgreement.single (first := 4346) modelrow_agrees_4347
private def atomic_4348 : CountSegment 1 10 13120 4347 4348 1982 1982 := CountSegment.single (first := 4347) (initial := 1982) row_count_4348 (by decide)
private theorem atomic_4348_agreement : RowAgreement 4347 4348 := RowAgreement.single (first := 4347) modelrow_agrees_4348
private def atomic_4349 : CountSegment 1 10 13120 4348 4349 1982 1982 := CountSegment.single (first := 4348) (initial := 1982) row_count_4349 (by decide)
private theorem atomic_4349_agreement : RowAgreement 4348 4349 := RowAgreement.single (first := 4348) modelrow_agrees_4349
private def atomic_4350 : CountSegment 1 10 13120 4349 4350 1982 1982 := CountSegment.single (first := 4349) (initial := 1982) row_count_4350 (by decide)
private theorem atomic_4350_agreement : RowAgreement 4349 4350 := RowAgreement.single (first := 4349) modelrow_agrees_4350
private def atomic_4351 : CountSegment 1 10 13120 4350 4351 1982 1983 := CountSegment.single (first := 4350) (initial := 1982) row_count_4351 (by decide)
private theorem atomic_4351_agreement : RowAgreement 4350 4351 := RowAgreement.single (first := 4350) modelrow_agrees_4351
private def atomic_4352 : CountSegment 1 10 13120 4351 4352 1983 1983 := CountSegment.single (first := 4351) (initial := 1983) row_count_4352 (by decide)
private theorem atomic_4352_agreement : RowAgreement 4351 4352 := RowAgreement.single (first := 4351) modelrow_agrees_4352
private def within_033_000 : CountSegment 1 10 13120 4224 4226 1933 1933 := CountSegment.join atomic_4225 atomic_4226
private theorem within_033_000_agreement : RowAgreement 4224 4226 := RowAgreement.join atomic_4225_agreement atomic_4226_agreement
private def within_033_001 : CountSegment 1 10 13120 4226 4228 1933 1934 := CountSegment.join atomic_4227 atomic_4228
private theorem within_033_001_agreement : RowAgreement 4226 4228 := RowAgreement.join atomic_4227_agreement atomic_4228_agreement
private def within_033_002 : CountSegment 1 10 13120 4224 4228 1933 1934 := CountSegment.join within_033_000 within_033_001
private theorem within_033_002_agreement : RowAgreement 4224 4228 := RowAgreement.join within_033_000_agreement within_033_001_agreement
private def within_033_003 : CountSegment 1 10 13120 4228 4230 1934 1934 := CountSegment.join atomic_4229 atomic_4230
private theorem within_033_003_agreement : RowAgreement 4228 4230 := RowAgreement.join atomic_4229_agreement atomic_4230_agreement
private def within_033_004 : CountSegment 1 10 13120 4230 4232 1934 1935 := CountSegment.join atomic_4231 atomic_4232
private theorem within_033_004_agreement : RowAgreement 4230 4232 := RowAgreement.join atomic_4231_agreement atomic_4232_agreement
private def within_033_005 : CountSegment 1 10 13120 4228 4232 1934 1935 := CountSegment.join within_033_003 within_033_004
private theorem within_033_005_agreement : RowAgreement 4228 4232 := RowAgreement.join within_033_003_agreement within_033_004_agreement
private def within_033_006 : CountSegment 1 10 13120 4224 4232 1933 1935 := CountSegment.join within_033_002 within_033_005
private theorem within_033_006_agreement : RowAgreement 4224 4232 := RowAgreement.join within_033_002_agreement within_033_005_agreement
private def within_033_007 : CountSegment 1 10 13120 4232 4234 1935 1935 := CountSegment.join atomic_4233 atomic_4234
private theorem within_033_007_agreement : RowAgreement 4232 4234 := RowAgreement.join atomic_4233_agreement atomic_4234_agreement
private def within_033_008 : CountSegment 1 10 13120 4234 4236 1935 1936 := CountSegment.join atomic_4235 atomic_4236
private theorem within_033_008_agreement : RowAgreement 4234 4236 := RowAgreement.join atomic_4235_agreement atomic_4236_agreement
private def within_033_009 : CountSegment 1 10 13120 4232 4236 1935 1936 := CountSegment.join within_033_007 within_033_008
private theorem within_033_009_agreement : RowAgreement 4232 4236 := RowAgreement.join within_033_007_agreement within_033_008_agreement
private def within_033_010 : CountSegment 1 10 13120 4236 4238 1936 1937 := CountSegment.join atomic_4237 atomic_4238
private theorem within_033_010_agreement : RowAgreement 4236 4238 := RowAgreement.join atomic_4237_agreement atomic_4238_agreement
private def within_033_011 : CountSegment 1 10 13120 4238 4240 1937 1939 := CountSegment.join atomic_4239 atomic_4240
private theorem within_033_011_agreement : RowAgreement 4238 4240 := RowAgreement.join atomic_4239_agreement atomic_4240_agreement
private def within_033_012 : CountSegment 1 10 13120 4236 4240 1936 1939 := CountSegment.join within_033_010 within_033_011
private theorem within_033_012_agreement : RowAgreement 4236 4240 := RowAgreement.join within_033_010_agreement within_033_011_agreement
private def within_033_013 : CountSegment 1 10 13120 4232 4240 1935 1939 := CountSegment.join within_033_009 within_033_012
private theorem within_033_013_agreement : RowAgreement 4232 4240 := RowAgreement.join within_033_009_agreement within_033_012_agreement
private def within_033_014 : CountSegment 1 10 13120 4224 4240 1933 1939 := CountSegment.join within_033_006 within_033_013
private theorem within_033_014_agreement : RowAgreement 4224 4240 := RowAgreement.join within_033_006_agreement within_033_013_agreement
private def within_033_015 : CountSegment 1 10 13120 4240 4242 1939 1940 := CountSegment.join atomic_4241 atomic_4242
private theorem within_033_015_agreement : RowAgreement 4240 4242 := RowAgreement.join atomic_4241_agreement atomic_4242_agreement
private def within_033_016 : CountSegment 1 10 13120 4242 4244 1940 1940 := CountSegment.join atomic_4243 atomic_4244
private theorem within_033_016_agreement : RowAgreement 4242 4244 := RowAgreement.join atomic_4243_agreement atomic_4244_agreement
private def within_033_017 : CountSegment 1 10 13120 4240 4244 1939 1940 := CountSegment.join within_033_015 within_033_016
private theorem within_033_017_agreement : RowAgreement 4240 4244 := RowAgreement.join within_033_015_agreement within_033_016_agreement
private def within_033_018 : CountSegment 1 10 13120 4244 4246 1940 1940 := CountSegment.join atomic_4245 atomic_4246
private theorem within_033_018_agreement : RowAgreement 4244 4246 := RowAgreement.join atomic_4245_agreement atomic_4246_agreement
private def within_033_019 : CountSegment 1 10 13120 4246 4248 1940 1941 := CountSegment.join atomic_4247 atomic_4248
private theorem within_033_019_agreement : RowAgreement 4246 4248 := RowAgreement.join atomic_4247_agreement atomic_4248_agreement
private def within_033_020 : CountSegment 1 10 13120 4244 4248 1940 1941 := CountSegment.join within_033_018 within_033_019
private theorem within_033_020_agreement : RowAgreement 4244 4248 := RowAgreement.join within_033_018_agreement within_033_019_agreement
private def within_033_021 : CountSegment 1 10 13120 4240 4248 1939 1941 := CountSegment.join within_033_017 within_033_020
private theorem within_033_021_agreement : RowAgreement 4240 4248 := RowAgreement.join within_033_017_agreement within_033_020_agreement
private def within_033_022 : CountSegment 1 10 13120 4248 4250 1941 1942 := CountSegment.join atomic_4249 atomic_4250
private theorem within_033_022_agreement : RowAgreement 4248 4250 := RowAgreement.join atomic_4249_agreement atomic_4250_agreement
private def within_033_023 : CountSegment 1 10 13120 4250 4252 1942 1943 := CountSegment.join atomic_4251 atomic_4252
private theorem within_033_023_agreement : RowAgreement 4250 4252 := RowAgreement.join atomic_4251_agreement atomic_4252_agreement
private def within_033_024 : CountSegment 1 10 13120 4248 4252 1941 1943 := CountSegment.join within_033_022 within_033_023
private theorem within_033_024_agreement : RowAgreement 4248 4252 := RowAgreement.join within_033_022_agreement within_033_023_agreement
private def within_033_025 : CountSegment 1 10 13120 4252 4254 1943 1943 := CountSegment.join atomic_4253 atomic_4254
private theorem within_033_025_agreement : RowAgreement 4252 4254 := RowAgreement.join atomic_4253_agreement atomic_4254_agreement
private def within_033_026 : CountSegment 1 10 13120 4254 4256 1943 1944 := CountSegment.join atomic_4255 atomic_4256
private theorem within_033_026_agreement : RowAgreement 4254 4256 := RowAgreement.join atomic_4255_agreement atomic_4256_agreement
private def within_033_027 : CountSegment 1 10 13120 4252 4256 1943 1944 := CountSegment.join within_033_025 within_033_026
private theorem within_033_027_agreement : RowAgreement 4252 4256 := RowAgreement.join within_033_025_agreement within_033_026_agreement
private def within_033_028 : CountSegment 1 10 13120 4248 4256 1941 1944 := CountSegment.join within_033_024 within_033_027
private theorem within_033_028_agreement : RowAgreement 4248 4256 := RowAgreement.join within_033_024_agreement within_033_027_agreement
private def within_033_029 : CountSegment 1 10 13120 4240 4256 1939 1944 := CountSegment.join within_033_021 within_033_028
private theorem within_033_029_agreement : RowAgreement 4240 4256 := RowAgreement.join within_033_021_agreement within_033_028_agreement
private def within_033_030 : CountSegment 1 10 13120 4224 4256 1933 1944 := CountSegment.join within_033_014 within_033_029
private theorem within_033_030_agreement : RowAgreement 4224 4256 := RowAgreement.join within_033_014_agreement within_033_029_agreement
private def within_033_031 : CountSegment 1 10 13120 4256 4258 1944 1944 := CountSegment.join atomic_4257 atomic_4258
private theorem within_033_031_agreement : RowAgreement 4256 4258 := RowAgreement.join atomic_4257_agreement atomic_4258_agreement
private def within_033_032 : CountSegment 1 10 13120 4258 4260 1944 1945 := CountSegment.join atomic_4259 atomic_4260
private theorem within_033_032_agreement : RowAgreement 4258 4260 := RowAgreement.join atomic_4259_agreement atomic_4260_agreement
private def within_033_033 : CountSegment 1 10 13120 4256 4260 1944 1945 := CountSegment.join within_033_031 within_033_032
private theorem within_033_033_agreement : RowAgreement 4256 4260 := RowAgreement.join within_033_031_agreement within_033_032_agreement
private def within_033_034 : CountSegment 1 10 13120 4260 4262 1945 1946 := CountSegment.join atomic_4261 atomic_4262
private theorem within_033_034_agreement : RowAgreement 4260 4262 := RowAgreement.join atomic_4261_agreement atomic_4262_agreement
private def within_033_035 : CountSegment 1 10 13120 4262 4264 1946 1947 := CountSegment.join atomic_4263 atomic_4264
private theorem within_033_035_agreement : RowAgreement 4262 4264 := RowAgreement.join atomic_4263_agreement atomic_4264_agreement
private def within_033_036 : CountSegment 1 10 13120 4260 4264 1945 1947 := CountSegment.join within_033_034 within_033_035
private theorem within_033_036_agreement : RowAgreement 4260 4264 := RowAgreement.join within_033_034_agreement within_033_035_agreement
private def within_033_037 : CountSegment 1 10 13120 4256 4264 1944 1947 := CountSegment.join within_033_033 within_033_036
private theorem within_033_037_agreement : RowAgreement 4256 4264 := RowAgreement.join within_033_033_agreement within_033_036_agreement
private def within_033_038 : CountSegment 1 10 13120 4264 4266 1947 1947 := CountSegment.join atomic_4265 atomic_4266
private theorem within_033_038_agreement : RowAgreement 4264 4266 := RowAgreement.join atomic_4265_agreement atomic_4266_agreement
private def within_033_039 : CountSegment 1 10 13120 4266 4268 1947 1947 := CountSegment.join atomic_4267 atomic_4268
private theorem within_033_039_agreement : RowAgreement 4266 4268 := RowAgreement.join atomic_4267_agreement atomic_4268_agreement
private def within_033_040 : CountSegment 1 10 13120 4264 4268 1947 1947 := CountSegment.join within_033_038 within_033_039
private theorem within_033_040_agreement : RowAgreement 4264 4268 := RowAgreement.join within_033_038_agreement within_033_039_agreement
private def within_033_041 : CountSegment 1 10 13120 4268 4270 1947 1949 := CountSegment.join atomic_4269 atomic_4270
private theorem within_033_041_agreement : RowAgreement 4268 4270 := RowAgreement.join atomic_4269_agreement atomic_4270_agreement
private def within_033_042 : CountSegment 1 10 13120 4270 4272 1949 1951 := CountSegment.join atomic_4271 atomic_4272
private theorem within_033_042_agreement : RowAgreement 4270 4272 := RowAgreement.join atomic_4271_agreement atomic_4272_agreement
private def within_033_043 : CountSegment 1 10 13120 4268 4272 1947 1951 := CountSegment.join within_033_041 within_033_042
private theorem within_033_043_agreement : RowAgreement 4268 4272 := RowAgreement.join within_033_041_agreement within_033_042_agreement
private def within_033_044 : CountSegment 1 10 13120 4264 4272 1947 1951 := CountSegment.join within_033_040 within_033_043
private theorem within_033_044_agreement : RowAgreement 4264 4272 := RowAgreement.join within_033_040_agreement within_033_043_agreement
private def within_033_045 : CountSegment 1 10 13120 4256 4272 1944 1951 := CountSegment.join within_033_037 within_033_044
private theorem within_033_045_agreement : RowAgreement 4256 4272 := RowAgreement.join within_033_037_agreement within_033_044_agreement
private def within_033_046 : CountSegment 1 10 13120 4272 4274 1951 1951 := CountSegment.join atomic_4273 atomic_4274
private theorem within_033_046_agreement : RowAgreement 4272 4274 := RowAgreement.join atomic_4273_agreement atomic_4274_agreement
private def within_033_047 : CountSegment 1 10 13120 4274 4276 1951 1953 := CountSegment.join atomic_4275 atomic_4276
private theorem within_033_047_agreement : RowAgreement 4274 4276 := RowAgreement.join atomic_4275_agreement atomic_4276_agreement
private def within_033_048 : CountSegment 1 10 13120 4272 4276 1951 1953 := CountSegment.join within_033_046 within_033_047
private theorem within_033_048_agreement : RowAgreement 4272 4276 := RowAgreement.join within_033_046_agreement within_033_047_agreement
private def within_033_049 : CountSegment 1 10 13120 4276 4278 1953 1954 := CountSegment.join atomic_4277 atomic_4278
private theorem within_033_049_agreement : RowAgreement 4276 4278 := RowAgreement.join atomic_4277_agreement atomic_4278_agreement
private def within_033_050 : CountSegment 1 10 13120 4278 4280 1954 1955 := CountSegment.join atomic_4279 atomic_4280
private theorem within_033_050_agreement : RowAgreement 4278 4280 := RowAgreement.join atomic_4279_agreement atomic_4280_agreement
private def within_033_051 : CountSegment 1 10 13120 4276 4280 1953 1955 := CountSegment.join within_033_049 within_033_050
private theorem within_033_051_agreement : RowAgreement 4276 4280 := RowAgreement.join within_033_049_agreement within_033_050_agreement
private def within_033_052 : CountSegment 1 10 13120 4272 4280 1951 1955 := CountSegment.join within_033_048 within_033_051
private theorem within_033_052_agreement : RowAgreement 4272 4280 := RowAgreement.join within_033_048_agreement within_033_051_agreement
private def within_033_053 : CountSegment 1 10 13120 4280 4282 1955 1957 := CountSegment.join atomic_4281 atomic_4282
private theorem within_033_053_agreement : RowAgreement 4280 4282 := RowAgreement.join atomic_4281_agreement atomic_4282_agreement
private def within_033_054 : CountSegment 1 10 13120 4282 4284 1957 1958 := CountSegment.join atomic_4283 atomic_4284
private theorem within_033_054_agreement : RowAgreement 4282 4284 := RowAgreement.join atomic_4283_agreement atomic_4284_agreement
private def within_033_055 : CountSegment 1 10 13120 4280 4284 1955 1958 := CountSegment.join within_033_053 within_033_054
private theorem within_033_055_agreement : RowAgreement 4280 4284 := RowAgreement.join within_033_053_agreement within_033_054_agreement
private def within_033_056 : CountSegment 1 10 13120 4284 4286 1958 1958 := CountSegment.join atomic_4285 atomic_4286
private theorem within_033_056_agreement : RowAgreement 4284 4286 := RowAgreement.join atomic_4285_agreement atomic_4286_agreement
private def within_033_057 : CountSegment 1 10 13120 4286 4288 1958 1959 := CountSegment.join atomic_4287 atomic_4288
private theorem within_033_057_agreement : RowAgreement 4286 4288 := RowAgreement.join atomic_4287_agreement atomic_4288_agreement
private def within_033_058 : CountSegment 1 10 13120 4284 4288 1958 1959 := CountSegment.join within_033_056 within_033_057
private theorem within_033_058_agreement : RowAgreement 4284 4288 := RowAgreement.join within_033_056_agreement within_033_057_agreement
private def within_033_059 : CountSegment 1 10 13120 4280 4288 1955 1959 := CountSegment.join within_033_055 within_033_058
private theorem within_033_059_agreement : RowAgreement 4280 4288 := RowAgreement.join within_033_055_agreement within_033_058_agreement
private def within_033_060 : CountSegment 1 10 13120 4272 4288 1951 1959 := CountSegment.join within_033_052 within_033_059
private theorem within_033_060_agreement : RowAgreement 4272 4288 := RowAgreement.join within_033_052_agreement within_033_059_agreement
private def within_033_061 : CountSegment 1 10 13120 4256 4288 1944 1959 := CountSegment.join within_033_045 within_033_060
private theorem within_033_061_agreement : RowAgreement 4256 4288 := RowAgreement.join within_033_045_agreement within_033_060_agreement
private def within_033_062 : CountSegment 1 10 13120 4224 4288 1933 1959 := CountSegment.join within_033_030 within_033_061
private theorem within_033_062_agreement : RowAgreement 4224 4288 := RowAgreement.join within_033_030_agreement within_033_061_agreement
private def within_033_063 : CountSegment 1 10 13120 4288 4290 1959 1959 := CountSegment.join atomic_4289 atomic_4290
private theorem within_033_063_agreement : RowAgreement 4288 4290 := RowAgreement.join atomic_4289_agreement atomic_4290_agreement
private def within_033_064 : CountSegment 1 10 13120 4290 4292 1959 1960 := CountSegment.join atomic_4291 atomic_4292
private theorem within_033_064_agreement : RowAgreement 4290 4292 := RowAgreement.join atomic_4291_agreement atomic_4292_agreement
private def within_033_065 : CountSegment 1 10 13120 4288 4292 1959 1960 := CountSegment.join within_033_063 within_033_064
private theorem within_033_065_agreement : RowAgreement 4288 4292 := RowAgreement.join within_033_063_agreement within_033_064_agreement
private def within_033_066 : CountSegment 1 10 13120 4292 4294 1960 1960 := CountSegment.join atomic_4293 atomic_4294
private theorem within_033_066_agreement : RowAgreement 4292 4294 := RowAgreement.join atomic_4293_agreement atomic_4294_agreement
private def within_033_067 : CountSegment 1 10 13120 4294 4296 1960 1960 := CountSegment.join atomic_4295 atomic_4296
private theorem within_033_067_agreement : RowAgreement 4294 4296 := RowAgreement.join atomic_4295_agreement atomic_4296_agreement
private def within_033_068 : CountSegment 1 10 13120 4292 4296 1960 1960 := CountSegment.join within_033_066 within_033_067
private theorem within_033_068_agreement : RowAgreement 4292 4296 := RowAgreement.join within_033_066_agreement within_033_067_agreement
private def within_033_069 : CountSegment 1 10 13120 4288 4296 1959 1960 := CountSegment.join within_033_065 within_033_068
private theorem within_033_069_agreement : RowAgreement 4288 4296 := RowAgreement.join within_033_065_agreement within_033_068_agreement
private def within_033_070 : CountSegment 1 10 13120 4296 4298 1960 1960 := CountSegment.join atomic_4297 atomic_4298
private theorem within_033_070_agreement : RowAgreement 4296 4298 := RowAgreement.join atomic_4297_agreement atomic_4298_agreement
private def within_033_071 : CountSegment 1 10 13120 4298 4300 1960 1960 := CountSegment.join atomic_4299 atomic_4300
private theorem within_033_071_agreement : RowAgreement 4298 4300 := RowAgreement.join atomic_4299_agreement atomic_4300_agreement
private def within_033_072 : CountSegment 1 10 13120 4296 4300 1960 1960 := CountSegment.join within_033_070 within_033_071
private theorem within_033_072_agreement : RowAgreement 4296 4300 := RowAgreement.join within_033_070_agreement within_033_071_agreement
private def within_033_073 : CountSegment 1 10 13120 4300 4302 1960 1962 := CountSegment.join atomic_4301 atomic_4302
private theorem within_033_073_agreement : RowAgreement 4300 4302 := RowAgreement.join atomic_4301_agreement atomic_4302_agreement
private def within_033_074 : CountSegment 1 10 13120 4302 4304 1962 1962 := CountSegment.join atomic_4303 atomic_4304
private theorem within_033_074_agreement : RowAgreement 4302 4304 := RowAgreement.join atomic_4303_agreement atomic_4304_agreement
private def within_033_075 : CountSegment 1 10 13120 4300 4304 1960 1962 := CountSegment.join within_033_073 within_033_074
private theorem within_033_075_agreement : RowAgreement 4300 4304 := RowAgreement.join within_033_073_agreement within_033_074_agreement
private def within_033_076 : CountSegment 1 10 13120 4296 4304 1960 1962 := CountSegment.join within_033_072 within_033_075
private theorem within_033_076_agreement : RowAgreement 4296 4304 := RowAgreement.join within_033_072_agreement within_033_075_agreement
private def within_033_077 : CountSegment 1 10 13120 4288 4304 1959 1962 := CountSegment.join within_033_069 within_033_076
private theorem within_033_077_agreement : RowAgreement 4288 4304 := RowAgreement.join within_033_069_agreement within_033_076_agreement
private def within_033_078 : CountSegment 1 10 13120 4304 4306 1962 1963 := CountSegment.join atomic_4305 atomic_4306
private theorem within_033_078_agreement : RowAgreement 4304 4306 := RowAgreement.join atomic_4305_agreement atomic_4306_agreement
private def within_033_079 : CountSegment 1 10 13120 4306 4308 1963 1963 := CountSegment.join atomic_4307 atomic_4308
private theorem within_033_079_agreement : RowAgreement 4306 4308 := RowAgreement.join atomic_4307_agreement atomic_4308_agreement
private def within_033_080 : CountSegment 1 10 13120 4304 4308 1962 1963 := CountSegment.join within_033_078 within_033_079
private theorem within_033_080_agreement : RowAgreement 4304 4308 := RowAgreement.join within_033_078_agreement within_033_079_agreement
private def within_033_081 : CountSegment 1 10 13120 4308 4310 1963 1963 := CountSegment.join atomic_4309 atomic_4310
private theorem within_033_081_agreement : RowAgreement 4308 4310 := RowAgreement.join atomic_4309_agreement atomic_4310_agreement
private def within_033_082 : CountSegment 1 10 13120 4310 4312 1963 1964 := CountSegment.join atomic_4311 atomic_4312
private theorem within_033_082_agreement : RowAgreement 4310 4312 := RowAgreement.join atomic_4311_agreement atomic_4312_agreement
private def within_033_083 : CountSegment 1 10 13120 4308 4312 1963 1964 := CountSegment.join within_033_081 within_033_082
private theorem within_033_083_agreement : RowAgreement 4308 4312 := RowAgreement.join within_033_081_agreement within_033_082_agreement
private def within_033_084 : CountSegment 1 10 13120 4304 4312 1962 1964 := CountSegment.join within_033_080 within_033_083
private theorem within_033_084_agreement : RowAgreement 4304 4312 := RowAgreement.join within_033_080_agreement within_033_083_agreement
private def within_033_085 : CountSegment 1 10 13120 4312 4314 1964 1967 := CountSegment.join atomic_4313 atomic_4314
private theorem within_033_085_agreement : RowAgreement 4312 4314 := RowAgreement.join atomic_4313_agreement atomic_4314_agreement
private def within_033_086 : CountSegment 1 10 13120 4314 4316 1967 1967 := CountSegment.join atomic_4315 atomic_4316
private theorem within_033_086_agreement : RowAgreement 4314 4316 := RowAgreement.join atomic_4315_agreement atomic_4316_agreement
private def within_033_087 : CountSegment 1 10 13120 4312 4316 1964 1967 := CountSegment.join within_033_085 within_033_086
private theorem within_033_087_agreement : RowAgreement 4312 4316 := RowAgreement.join within_033_085_agreement within_033_086_agreement
private def within_033_088 : CountSegment 1 10 13120 4316 4318 1967 1968 := CountSegment.join atomic_4317 atomic_4318
private theorem within_033_088_agreement : RowAgreement 4316 4318 := RowAgreement.join atomic_4317_agreement atomic_4318_agreement
private def within_033_089 : CountSegment 1 10 13120 4318 4320 1968 1968 := CountSegment.join atomic_4319 atomic_4320
private theorem within_033_089_agreement : RowAgreement 4318 4320 := RowAgreement.join atomic_4319_agreement atomic_4320_agreement
private def within_033_090 : CountSegment 1 10 13120 4316 4320 1967 1968 := CountSegment.join within_033_088 within_033_089
private theorem within_033_090_agreement : RowAgreement 4316 4320 := RowAgreement.join within_033_088_agreement within_033_089_agreement
private def within_033_091 : CountSegment 1 10 13120 4312 4320 1964 1968 := CountSegment.join within_033_087 within_033_090
private theorem within_033_091_agreement : RowAgreement 4312 4320 := RowAgreement.join within_033_087_agreement within_033_090_agreement
private def within_033_092 : CountSegment 1 10 13120 4304 4320 1962 1968 := CountSegment.join within_033_084 within_033_091
private theorem within_033_092_agreement : RowAgreement 4304 4320 := RowAgreement.join within_033_084_agreement within_033_091_agreement
private def within_033_093 : CountSegment 1 10 13120 4288 4320 1959 1968 := CountSegment.join within_033_077 within_033_092
private theorem within_033_093_agreement : RowAgreement 4288 4320 := RowAgreement.join within_033_077_agreement within_033_092_agreement
private def within_033_094 : CountSegment 1 10 13120 4320 4322 1968 1968 := CountSegment.join atomic_4321 atomic_4322
private theorem within_033_094_agreement : RowAgreement 4320 4322 := RowAgreement.join atomic_4321_agreement atomic_4322_agreement
private def within_033_095 : CountSegment 1 10 13120 4322 4324 1968 1970 := CountSegment.join atomic_4323 atomic_4324
private theorem within_033_095_agreement : RowAgreement 4322 4324 := RowAgreement.join atomic_4323_agreement atomic_4324_agreement
private def within_033_096 : CountSegment 1 10 13120 4320 4324 1968 1970 := CountSegment.join within_033_094 within_033_095
private theorem within_033_096_agreement : RowAgreement 4320 4324 := RowAgreement.join within_033_094_agreement within_033_095_agreement
private def within_033_097 : CountSegment 1 10 13120 4324 4326 1970 1971 := CountSegment.join atomic_4325 atomic_4326
private theorem within_033_097_agreement : RowAgreement 4324 4326 := RowAgreement.join atomic_4325_agreement atomic_4326_agreement
private def within_033_098 : CountSegment 1 10 13120 4326 4328 1971 1972 := CountSegment.join atomic_4327 atomic_4328
private theorem within_033_098_agreement : RowAgreement 4326 4328 := RowAgreement.join atomic_4327_agreement atomic_4328_agreement
private def within_033_099 : CountSegment 1 10 13120 4324 4328 1970 1972 := CountSegment.join within_033_097 within_033_098
private theorem within_033_099_agreement : RowAgreement 4324 4328 := RowAgreement.join within_033_097_agreement within_033_098_agreement
private def within_033_100 : CountSegment 1 10 13120 4320 4328 1968 1972 := CountSegment.join within_033_096 within_033_099
private theorem within_033_100_agreement : RowAgreement 4320 4328 := RowAgreement.join within_033_096_agreement within_033_099_agreement
private def within_033_101 : CountSegment 1 10 13120 4328 4330 1972 1973 := CountSegment.join atomic_4329 atomic_4330
private theorem within_033_101_agreement : RowAgreement 4328 4330 := RowAgreement.join atomic_4329_agreement atomic_4330_agreement
private def within_033_102 : CountSegment 1 10 13120 4330 4332 1973 1974 := CountSegment.join atomic_4331 atomic_4332
private theorem within_033_102_agreement : RowAgreement 4330 4332 := RowAgreement.join atomic_4331_agreement atomic_4332_agreement
private def within_033_103 : CountSegment 1 10 13120 4328 4332 1972 1974 := CountSegment.join within_033_101 within_033_102
private theorem within_033_103_agreement : RowAgreement 4328 4332 := RowAgreement.join within_033_101_agreement within_033_102_agreement
private def within_033_104 : CountSegment 1 10 13120 4332 4334 1974 1974 := CountSegment.join atomic_4333 atomic_4334
private theorem within_033_104_agreement : RowAgreement 4332 4334 := RowAgreement.join atomic_4333_agreement atomic_4334_agreement
private def within_033_105 : CountSegment 1 10 13120 4334 4336 1974 1976 := CountSegment.join atomic_4335 atomic_4336
private theorem within_033_105_agreement : RowAgreement 4334 4336 := RowAgreement.join atomic_4335_agreement atomic_4336_agreement
private def within_033_106 : CountSegment 1 10 13120 4332 4336 1974 1976 := CountSegment.join within_033_104 within_033_105
private theorem within_033_106_agreement : RowAgreement 4332 4336 := RowAgreement.join within_033_104_agreement within_033_105_agreement
private def within_033_107 : CountSegment 1 10 13120 4328 4336 1972 1976 := CountSegment.join within_033_103 within_033_106
private theorem within_033_107_agreement : RowAgreement 4328 4336 := RowAgreement.join within_033_103_agreement within_033_106_agreement
private def within_033_108 : CountSegment 1 10 13120 4320 4336 1968 1976 := CountSegment.join within_033_100 within_033_107
private theorem within_033_108_agreement : RowAgreement 4320 4336 := RowAgreement.join within_033_100_agreement within_033_107_agreement
private def within_033_109 : CountSegment 1 10 13120 4336 4338 1976 1977 := CountSegment.join atomic_4337 atomic_4338
private theorem within_033_109_agreement : RowAgreement 4336 4338 := RowAgreement.join atomic_4337_agreement atomic_4338_agreement
private def within_033_110 : CountSegment 1 10 13120 4338 4340 1977 1978 := CountSegment.join atomic_4339 atomic_4340
private theorem within_033_110_agreement : RowAgreement 4338 4340 := RowAgreement.join atomic_4339_agreement atomic_4340_agreement
private def within_033_111 : CountSegment 1 10 13120 4336 4340 1976 1978 := CountSegment.join within_033_109 within_033_110
private theorem within_033_111_agreement : RowAgreement 4336 4340 := RowAgreement.join within_033_109_agreement within_033_110_agreement
private def within_033_112 : CountSegment 1 10 13120 4340 4342 1978 1980 := CountSegment.join atomic_4341 atomic_4342
private theorem within_033_112_agreement : RowAgreement 4340 4342 := RowAgreement.join atomic_4341_agreement atomic_4342_agreement
private def within_033_113 : CountSegment 1 10 13120 4342 4344 1980 1981 := CountSegment.join atomic_4343 atomic_4344
private theorem within_033_113_agreement : RowAgreement 4342 4344 := RowAgreement.join atomic_4343_agreement atomic_4344_agreement
private def within_033_114 : CountSegment 1 10 13120 4340 4344 1978 1981 := CountSegment.join within_033_112 within_033_113
private theorem within_033_114_agreement : RowAgreement 4340 4344 := RowAgreement.join within_033_112_agreement within_033_113_agreement
private def within_033_115 : CountSegment 1 10 13120 4336 4344 1976 1981 := CountSegment.join within_033_111 within_033_114
private theorem within_033_115_agreement : RowAgreement 4336 4344 := RowAgreement.join within_033_111_agreement within_033_114_agreement
private def within_033_116 : CountSegment 1 10 13120 4344 4346 1981 1981 := CountSegment.join atomic_4345 atomic_4346
private theorem within_033_116_agreement : RowAgreement 4344 4346 := RowAgreement.join atomic_4345_agreement atomic_4346_agreement
private def within_033_117 : CountSegment 1 10 13120 4346 4348 1981 1982 := CountSegment.join atomic_4347 atomic_4348
private theorem within_033_117_agreement : RowAgreement 4346 4348 := RowAgreement.join atomic_4347_agreement atomic_4348_agreement
private def within_033_118 : CountSegment 1 10 13120 4344 4348 1981 1982 := CountSegment.join within_033_116 within_033_117
private theorem within_033_118_agreement : RowAgreement 4344 4348 := RowAgreement.join within_033_116_agreement within_033_117_agreement
private def within_033_119 : CountSegment 1 10 13120 4348 4350 1982 1982 := CountSegment.join atomic_4349 atomic_4350
private theorem within_033_119_agreement : RowAgreement 4348 4350 := RowAgreement.join atomic_4349_agreement atomic_4350_agreement
private def within_033_120 : CountSegment 1 10 13120 4350 4352 1982 1983 := CountSegment.join atomic_4351 atomic_4352
private theorem within_033_120_agreement : RowAgreement 4350 4352 := RowAgreement.join atomic_4351_agreement atomic_4352_agreement
private def within_033_121 : CountSegment 1 10 13120 4348 4352 1982 1983 := CountSegment.join within_033_119 within_033_120
private theorem within_033_121_agreement : RowAgreement 4348 4352 := RowAgreement.join within_033_119_agreement within_033_120_agreement
private def within_033_122 : CountSegment 1 10 13120 4344 4352 1981 1983 := CountSegment.join within_033_118 within_033_121
private theorem within_033_122_agreement : RowAgreement 4344 4352 := RowAgreement.join within_033_118_agreement within_033_121_agreement
private def within_033_123 : CountSegment 1 10 13120 4336 4352 1976 1983 := CountSegment.join within_033_115 within_033_122
private theorem within_033_123_agreement : RowAgreement 4336 4352 := RowAgreement.join within_033_115_agreement within_033_122_agreement
private def within_033_124 : CountSegment 1 10 13120 4320 4352 1968 1983 := CountSegment.join within_033_108 within_033_123
private theorem within_033_124_agreement : RowAgreement 4320 4352 := RowAgreement.join within_033_108_agreement within_033_123_agreement
private def within_033_125 : CountSegment 1 10 13120 4288 4352 1959 1983 := CountSegment.join within_033_093 within_033_124
private theorem within_033_125_agreement : RowAgreement 4288 4352 := RowAgreement.join within_033_093_agreement within_033_124_agreement
private def within_033_126 : CountSegment 1 10 13120 4224 4352 1933 1983 := CountSegment.join within_033_062 within_033_125
private theorem within_033_126_agreement : RowAgreement 4224 4352 := RowAgreement.join within_033_062_agreement within_033_125_agreement
def segment_033 : CountSegment 1 10 13120 4224 4352 1933 1983 := within_033_126
theorem agreement_033 : RowAgreement 4224 4352 := within_033_126_agreement
end ProjectEulerSolutions.P58Numerical
