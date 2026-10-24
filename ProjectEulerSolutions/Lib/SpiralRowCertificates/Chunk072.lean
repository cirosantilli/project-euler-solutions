import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9217 : RowCertificate 9217 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9217 1) := by change ¬ Nat.Prime 339830791; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2675833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9217 2) := by change ¬ Nat.Prime 339812357; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3007189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9217 3) := by change ¬ Nat.Prime 339793923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113264641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9217 : row 9217 = 0 := (row_9217.checked).trans rfl
theorem model_count_9217 : modelRow 9217 = 0 := by decide
theorem modelrow_agrees_9217 : modelRow 9217 = row 9217 := model_count_9217.trans row_count_9217.symm
def row_9218 : RowCertificate 9218 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9218 1) := by change ¬ Nat.Prime 339904533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113301511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9218 2) := by change ¬ Nat.Prime 339886097; exact PrattCertificates.composite_of_product (factor := 2657) (cofactor := 127921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9218 3) := by change ¬ Nat.Prime 339867661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48552523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9218 : row 9218 = 0 := (row_9218.checked).trans rfl
theorem model_count_9218 : modelRow 9218 = 0 := by decide
theorem modelrow_agrees_9218 : modelRow 9218 = row 9218 := model_count_9218.trans row_count_9218.symm
def row_9219 : RowCertificate 9219 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9219 1) := by change Nat.Prime 339978283; exact ProjectEulerSolutions.P58PrimeNodes.p_339978283.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9219 2) := by change ¬ Nat.Prime 339959845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 67991969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9219 3) := by change ¬ Nat.Prime 339941407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26149339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9219 : row 9219 = 1 := (row_9219.checked).trans rfl
theorem model_count_9219 : modelRow 9219 = 1 := by decide
theorem modelrow_agrees_9219 : modelRow 9219 = row 9219 := model_count_9219.trans row_count_9219.symm
def row_9220 : RowCertificate 9220 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9220 1) := by change ¬ Nat.Prime 340052041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48578863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9220 2) := by change ¬ Nat.Prime 340033601; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 785297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9220 3) := by change ¬ Nat.Prime 340015161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113338387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9220 : row 9220 = 0 := (row_9220.checked).trans rfl
theorem model_count_9220 : modelRow 9220 = 0 := by decide
theorem modelrow_agrees_9220 : modelRow 9220 = row 9220 := model_count_9220.trans row_count_9220.symm
def row_9221 : RowCertificate 9221 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9221 1) := by change ¬ Nat.Prime 340125807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113375269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9221 2) := by change ¬ Nat.Prime 340107365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68021473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9221 3) := by change ¬ Nat.Prime 340088923; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17899417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9221 : row 9221 = 0 := (row_9221.checked).trans rfl
theorem model_count_9221 : modelRow 9221 = 0 := by decide
theorem modelrow_agrees_9221 : modelRow 9221 = row 9221 := model_count_9221.trans row_count_9221.symm
def row_9222 : RowCertificate 9222 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9222 1) := by change ¬ Nat.Prime 340199581; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2447479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9222 2) := by change Nat.Prime 340181137; exact ProjectEulerSolutions.P58PrimeNodes.p_340181137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9222 3) := by change ¬ Nat.Prime 340162693; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26166361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9222 : row 9222 = 1 := (row_9222.checked).trans rfl
theorem model_count_9222 : modelRow 9222 = 1 := by decide
theorem modelrow_agrees_9222 : modelRow 9222 = row 9222 := model_count_9222.trans row_count_9222.symm
def row_9223 : RowCertificate 9223 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9223 1) := by change Nat.Prime 340273363; exact ProjectEulerSolutions.P58PrimeNodes.p_340273363.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9223 2) := by change ¬ Nat.Prime 340254917; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 628937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9223 3) := by change ¬ Nat.Prime 340236471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113412157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9223 : row 9223 = 1 := (row_9223.checked).trans rfl
theorem model_count_9223 : modelRow 9223 = 1 := by decide
theorem modelrow_agrees_9223 : modelRow 9223 = row 9223 := model_count_9223.trans row_count_9223.symm
def row_9224 : RowCertificate 9224 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9224 1) := by change ¬ Nat.Prime 340347153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113449051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9224 2) := by change ¬ Nat.Prime 340328705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68065741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9224 3) := by change ¬ Nat.Prime 340310257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48615751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9224 : row 9224 = 0 := (row_9224.checked).trans rfl
theorem model_count_9224 : modelRow 9224 = 0 := by decide
theorem modelrow_agrees_9224 : modelRow 9224 = row 9224 := model_count_9224.trans row_count_9224.symm
def row_9225 : RowCertificate 9225 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9225 1) := by change ¬ Nat.Prime 340420951; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26186227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9225 2) := by change Nat.Prime 340402501; exact ProjectEulerSolutions.P58PrimeNodes.p_340402501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9225 3) := by change ¬ Nat.Prime 340384051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48626293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9225 : row 9225 = 1 := (row_9225.checked).trans rfl
theorem model_count_9225 : modelRow 9225 = 1 := by decide
theorem modelrow_agrees_9225 : modelRow 9225 = row 9225 := model_count_9225.trans row_count_9225.symm
def row_9226 : RowCertificate 9226 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9226 1) := by change ¬ Nat.Prime 340494757; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9202561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9226 2) := by change ¬ Nat.Prime 340476305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68095261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9226 3) := by change ¬ Nat.Prime 340457853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113485951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9226 : row 9226 = 0 := (row_9226.checked).trans rfl
theorem model_count_9226 : modelRow 9226 = 0 := by decide
theorem modelrow_agrees_9226 : modelRow 9226 = row 9226 := model_count_9226.trans row_count_9226.symm
def row_9227 : RowCertificate 9227 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9227 1) := by change ¬ Nat.Prime 340568571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113522857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9227 2) := by change ¬ Nat.Prime 340550117; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3124313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9227 3) := by change ¬ Nat.Prime 340531663; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7919341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9227 : row 9227 = 0 := (row_9227.checked).trans rfl
theorem model_count_9227 : modelRow 9227 = 0 := by decide
theorem modelrow_agrees_9227 : modelRow 9227 = row 9227 := model_count_9227.trans row_count_9227.symm
def row_9228 : RowCertificate 9228 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9228 1) := by change ¬ Nat.Prime 340642393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48663199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9228 2) := by change ¬ Nat.Prime 340623937; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11745653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9228 3) := by change ¬ Nat.Prime 340605481; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 555637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9228 : row 9228 = 0 := (row_9228.checked).trans rfl
theorem model_count_9228 : modelRow 9228 = 0 := by decide
theorem modelrow_agrees_9228 : modelRow 9228 = row 9228 := model_count_9228.trans row_count_9228.symm
def row_9229 : RowCertificate 9229 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9229 1) := by change ¬ Nat.Prime 340716223; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 833047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9229 2) := by change ¬ Nat.Prime 340697765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68139553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9229 3) := by change ¬ Nat.Prime 340679307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113559769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9229 : row 9229 = 0 := (row_9229.checked).trans rfl
theorem model_count_9229 : modelRow 9229 = 0 := by decide
theorem modelrow_agrees_9229 : modelRow 9229 = row 9229 := model_count_9229.trans row_count_9229.symm
def row_9230 : RowCertificate 9230 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9230 1) := by change ¬ Nat.Prime 340790061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113596687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9230 2) := by change ¬ Nat.Prime 340771601; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 574657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9230 3) := by change ¬ Nat.Prime 340753141; exact PrattCertificates.composite_of_product (factor := 3229) (cofactor := 105529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9230 : row 9230 = 0 := (row_9230.checked).trans rfl
theorem model_count_9230 : modelRow 9230 = 0 := by decide
theorem modelrow_agrees_9230 : modelRow 9230 = row 9230 := model_count_9230.trans row_count_9230.symm
def row_9231 : RowCertificate 9231 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9231 1) := by change ¬ Nat.Prime 340863907; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5087521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9231 2) := by change ¬ Nat.Prime 340845445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68169089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9231 3) := by change ¬ Nat.Prime 340826983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48689569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9231 : row 9231 = 0 := (row_9231.checked).trans rfl
theorem model_count_9231 : modelRow 9231 = 0 := by decide
theorem modelrow_agrees_9231 : modelRow 9231 = row 9231 := model_count_9231.trans row_count_9231.symm
def row_9232 : RowCertificate 9232 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9232 1) := by change Nat.Prime 340937761; exact ProjectEulerSolutions.P58PrimeNodes.p_340937761.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9232 2) := by change Nat.Prime 340919297; exact ProjectEulerSolutions.P58PrimeNodes.p_340919297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9232 3) := by change ¬ Nat.Prime 340900833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113633611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9232 : row 9232 = 2 := (row_9232.checked).trans rfl
theorem model_count_9232 : modelRow 9232 = 2 := by decide
theorem modelrow_agrees_9232 : modelRow 9232 = row 9232 := model_count_9232.trans row_count_9232.symm
def row_9233 : RowCertificate 9233 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9233 1) := by change ¬ Nat.Prime 341011623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113670541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9233 2) := by change ¬ Nat.Prime 340993157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20058421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9233 3) := by change ¬ Nat.Prime 340974691; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3515203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9233 : row 9233 = 0 := (row_9233.checked).trans rfl
theorem model_count_9233 : modelRow 9233 = 0 := by decide
theorem modelrow_agrees_9233 : modelRow 9233 = row 9233 := model_count_9233.trans row_count_9233.symm
def row_9234 : RowCertificate 9234 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9234 1) := by change ¬ Nat.Prime 341085493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48726499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9234 2) := by change ¬ Nat.Prime 341067025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68213405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9234 3) := by change Nat.Prime 341048557; exact ProjectEulerSolutions.P58PrimeNodes.p_341048557.prime
    exact decide_eq_true prime
theorem row_count_9234 : row 9234 = 1 := (row_9234.checked).trans rfl
theorem model_count_9234 : modelRow 9234 = 1 := by decide
theorem modelrow_agrees_9234 : modelRow 9234 = row 9234 := model_count_9234.trans row_count_9234.symm
def row_9235 : RowCertificate 9235 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9235 1) := by change ¬ Nat.Prime 341159371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48737053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9235 2) := by change ¬ Nat.Prime 341140901; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 654781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9235 3) := by change ¬ Nat.Prime 341122431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113707477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9235 : row 9235 = 0 := (row_9235.checked).trans rfl
theorem model_count_9235 : modelRow 9235 = 0 := by decide
theorem modelrow_agrees_9235 : modelRow 9235 = row 9235 := model_count_9235.trans row_count_9235.symm
def row_9236 : RowCertificate 9236 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9236 1) := by change ¬ Nat.Prime 341233257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113744419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9236 2) := by change ¬ Nat.Prime 341214785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68242957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9236 3) := by change ¬ Nat.Prime 341196313; exact PrattCertificates.composite_of_product (factor := 6343) (cofactor := 53791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9236 : row 9236 = 0 := (row_9236.checked).trans rfl
theorem model_count_9236 : modelRow 9236 = 0 := by decide
theorem modelrow_agrees_9236 : modelRow 9236 = row 9236 := model_count_9236.trans row_count_9236.symm
def row_9237 : RowCertificate 9237 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9237 1) := by change Nat.Prime 341307151; exact ProjectEulerSolutions.P58PrimeNodes.p_341307151.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9237 2) := by change ¬ Nat.Prime 341288677; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6439409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9237 3) := by change ¬ Nat.Prime 341270203; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9223519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9237 : row 9237 = 1 := (row_9237.checked).trans rfl
theorem model_count_9237 : modelRow 9237 = 1 := by decide
theorem modelrow_agrees_9237 : modelRow 9237 = row 9237 := model_count_9237.trans row_count_9237.symm
def row_9238 : RowCertificate 9238 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9238 1) := by change ¬ Nat.Prime 341381053; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26260081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9238 2) := by change ¬ Nat.Prime 341362577; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 760273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9238 3) := by change ¬ Nat.Prime 341344101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113781367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9238 : row 9238 = 0 := (row_9238.checked).trans rfl
theorem model_count_9238 : modelRow 9238 = 0 := by decide
theorem modelrow_agrees_9238 : modelRow 9238 = row 9238 := model_count_9238.trans row_count_9238.symm
def row_9239 : RowCertificate 9239 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9239 1) := by change ¬ Nat.Prime 341454963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113818321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9239 2) := by change ¬ Nat.Prime 341436485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68287297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9239 3) := by change ¬ Nat.Prime 341418007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48774001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9239 : row 9239 = 0 := (row_9239.checked).trans rfl
theorem model_count_9239 : modelRow 9239 = 0 := by decide
theorem modelrow_agrees_9239 : modelRow 9239 = row 9239 := model_count_9239.trans row_count_9239.symm
def row_9240 : RowCertificate 9240 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9240 1) := by change ¬ Nat.Prime 341528881; exact PrattCertificates.composite_of_product (factor := 787) (cofactor := 433963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9240 2) := by change Nat.Prime 341510401; exact ProjectEulerSolutions.P58PrimeNodes.p_341510401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9240 3) := by change ¬ Nat.Prime 341491921; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 17973259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9240 : row 9240 = 1 := (row_9240.checked).trans rfl
theorem model_count_9240 : modelRow 9240 = 1 := by decide
theorem modelrow_agrees_9240 : modelRow 9240 = row 9240 := model_count_9240.trans row_count_9240.symm
def row_9241 : RowCertificate 9241 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9241 1) := by change ¬ Nat.Prime 341602807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48800401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9241 2) := by change ¬ Nat.Prime 341584325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68316865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9241 3) := by change ¬ Nat.Prime 341565843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113855281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9241 : row 9241 = 0 := (row_9241.checked).trans rfl
theorem model_count_9241 : modelRow 9241 = 0 := by decide
theorem modelrow_agrees_9241 : modelRow 9241 = row 9241 := model_count_9241.trans row_count_9241.symm
def row_9242 : RowCertificate 9242 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9242 1) := by change ¬ Nat.Prime 341676741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113892247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9242 2) := by change ¬ Nat.Prime 341658257; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 412133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9242 3) := by change ¬ Nat.Prime 341639773; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7945111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9242 : row 9242 = 0 := (row_9242.checked).trans rfl
theorem model_count_9242 : modelRow 9242 = 0 := by decide
theorem modelrow_agrees_9242 : modelRow 9242 = row 9242 := model_count_9242.trans row_count_9242.symm
def row_9243 : RowCertificate 9243 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9243 1) := by change Nat.Prime 341750683; exact ProjectEulerSolutions.P58PrimeNodes.p_341750683.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9243 2) := by change Nat.Prime 341732197; exact ProjectEulerSolutions.P58PrimeNodes.p_341732197.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9243 3) := by change Nat.Prime 341713711; exact ProjectEulerSolutions.P58PrimeNodes.p_341713711.prime
    exact decide_eq_true prime
theorem row_count_9243 : row 9243 = 3 := (row_9243.checked).trans rfl
theorem model_count_9243 : modelRow 9243 = 3 := by decide
theorem modelrow_agrees_9243 : modelRow 9243 = row 9243 := model_count_9243.trans row_count_9243.symm
def row_9244 : RowCertificate 9244 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9244 1) := by change Nat.Prime 341824633; exact ProjectEulerSolutions.P58PrimeNodes.p_341824633.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9244 2) := by change ¬ Nat.Prime 341806145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68361229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9244 3) := by change ¬ Nat.Prime 341787657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113929219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9244 : row 9244 = 1 := (row_9244.checked).trans rfl
theorem model_count_9244 : modelRow 9244 = 1 := by decide
theorem modelrow_agrees_9244 : modelRow 9244 = row 9244 := model_count_9244.trans row_count_9244.symm
def row_9245 : RowCertificate 9245 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9245 1) := by change ¬ Nat.Prime 341898591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113966197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9245 2) := by change ¬ Nat.Prime 341880101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11788969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9245 3) := by change ¬ Nat.Prime 341861611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48837373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9245 : row 9245 = 0 := (row_9245.checked).trans rfl
theorem model_count_9245 : modelRow 9245 = 0 := by decide
theorem modelrow_agrees_9245 : modelRow 9245 = row 9245 := model_count_9245.trans row_count_9245.symm
def row_9246 : RowCertificate 9246 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9246 1) := by change Nat.Prime 341972557; exact ProjectEulerSolutions.P58PrimeNodes.p_341972557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9246 2) := by change ¬ Nat.Prime 341954065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68390813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9246 3) := by change ¬ Nat.Prime 341935573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48847939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9246 : row 9246 = 1 := (row_9246.checked).trans rfl
theorem model_count_9246 : modelRow 9246 = 1 := by decide
theorem modelrow_agrees_9246 : modelRow 9246 = row 9246 := model_count_9246.trans row_count_9246.symm
def row_9247 : RowCertificate 9247 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9247 1) := by change ¬ Nat.Prime 342046531; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18002449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9247 2) := by change ¬ Nat.Prime 342028037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26309849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9247 3) := by change ¬ Nat.Prime 342009543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114003181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9247 : row 9247 = 0 := (row_9247.checked).trans rfl
theorem model_count_9247 : modelRow 9247 = 0 := by decide
theorem modelrow_agrees_9247 : modelRow 9247 = row 9247 := model_count_9247.trans row_count_9247.symm
def row_9248 : RowCertificate 9248 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9248 1) := by change ¬ Nat.Prime 342120513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114040171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9248 2) := by change ¬ Nat.Prime 342102017; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4686329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9248 3) := by change ¬ Nat.Prime 342083521; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26314117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9248 : row 9248 = 0 := (row_9248.checked).trans rfl
theorem model_count_9248 : modelRow 9248 = 0 := by decide
theorem modelrow_agrees_9248 : modelRow 9248 = row 9248 := model_count_9248.trans row_count_9248.symm
def row_9249 : RowCertificate 9249 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9249 1) := by change ¬ Nat.Prime 342194503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48884929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9249 2) := by change ¬ Nat.Prime 342176005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68435201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9249 3) := by change Nat.Prime 342157507; exact ProjectEulerSolutions.P58PrimeNodes.p_342157507.prime
    exact decide_eq_true prime
theorem row_count_9249 : row 9249 = 1 := (row_9249.checked).trans rfl
theorem model_count_9249 : modelRow 9249 = 1 := by decide
theorem modelrow_agrees_9249 : modelRow 9249 = row 9249 := model_count_9249.trans row_count_9249.symm
def row_9250 : RowCertificate 9250 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9250 1) := by change Nat.Prime 342268501; exact ProjectEulerSolutions.P58PrimeNodes.p_342268501.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9250 2) := by change ¬ Nat.Prime 342250001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20132353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9250 3) := by change ¬ Nat.Prime 342231501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114077167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9250 : row 9250 = 1 := (row_9250.checked).trans rfl
theorem model_count_9250 : modelRow 9250 = 1 := by decide
theorem modelrow_agrees_9250 : modelRow 9250 = row 9250 := model_count_9250.trans row_count_9250.symm
def row_9251 : RowCertificate 9251 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9251 1) := by change ¬ Nat.Prime 342342507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114114169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9251 2) := by change ¬ Nat.Prime 342324005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68464801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9251 3) := by change ¬ Nat.Prime 342305503; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11042113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9251 : row 9251 = 0 := (row_9251.checked).trans rfl
theorem model_count_9251 : modelRow 9251 = 0 := by decide
theorem modelrow_agrees_9251 : modelRow 9251 = row 9251 := model_count_9251.trans row_count_9251.symm
def row_9252 : RowCertificate 9252 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9252 1) := by change ¬ Nat.Prime 342416521; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 1034491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9252 2) := by change ¬ Nat.Prime 342398017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26338309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9252 3) := by change ¬ Nat.Prime 342379513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48911359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9252 : row 9252 = 0 := (row_9252.checked).trans rfl
theorem model_count_9252 : modelRow 9252 = 0 := by decide
theorem modelrow_agrees_9252 : modelRow 9252 = row 9252 := model_count_9252.trans row_count_9252.symm
def row_9253 : RowCertificate 9253 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9253 1) := by change Nat.Prime 342490543; exact ProjectEulerSolutions.P58PrimeNodes.p_342490543.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9253 2) := by change ¬ Nat.Prime 342472037; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9256001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9253 3) := by change ¬ Nat.Prime 342453531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114151177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9253 : row 9253 = 1 := (row_9253.checked).trans rfl
theorem model_count_9253 : modelRow 9253 = 1 := by decide
theorem modelrow_agrees_9253 : modelRow 9253 = row 9253 := model_count_9253.trans row_count_9253.symm
def row_9254 : RowCertificate 9254 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9254 1) := by change ¬ Nat.Prime 342564573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114188191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9254 2) := by change ¬ Nat.Prime 342546065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68509213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9254 3) := by change Nat.Prime 342527557; exact ProjectEulerSolutions.P58PrimeNodes.p_342527557.prime
    exact decide_eq_true prime
theorem row_count_9254 : row 9254 = 1 := (row_9254.checked).trans rfl
theorem model_count_9254 : modelRow 9254 = 1 := by decide
theorem modelrow_agrees_9254 : modelRow 9254 = row 9254 := model_count_9254.trans row_count_9254.symm
def row_9255 : RowCertificate 9255 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9255 1) := by change ¬ Nat.Prime 342638611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48948373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9255 2) := by change ¬ Nat.Prime 342620101; exact PrattCertificates.composite_of_product (factor := 2557) (cofactor := 133993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9255 3) := by change ¬ Nat.Prime 342601591; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4336729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9255 : row 9255 = 0 := (row_9255.checked).trans rfl
theorem model_count_9255 : modelRow 9255 = 0 := by decide
theorem modelrow_agrees_9255 : modelRow 9255 = row 9255 := model_count_9255.trans row_count_9255.symm
def row_9256 : RowCertificate 9256 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9256 1) := by change ¬ Nat.Prime 342712657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48958951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9256 2) := by change ¬ Nat.Prime 342694145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68538829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9256 3) := by change ¬ Nat.Prime 342675633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114225211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9256 : row 9256 = 0 := (row_9256.checked).trans rfl
theorem model_count_9256 : modelRow 9256 = 0 := by decide
theorem modelrow_agrees_9256 : modelRow 9256 = row 9256 := model_count_9256.trans row_count_9256.symm
def row_9257 : RowCertificate 9257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9257 1) := by change ¬ Nat.Prime 342786711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114262237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9257 2) := by change ¬ Nat.Prime 342768197; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11819593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9257 3) := by change ¬ Nat.Prime 342749683; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18039457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9257 : row 9257 = 0 := (row_9257.checked).trans rfl
theorem model_count_9257 : modelRow 9257 = 0 := by decide
theorem modelrow_agrees_9257 : modelRow 9257 = row 9257 := model_count_9257.trans row_count_9257.symm
def row_9258 : RowCertificate 9258 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9258 1) := by change Nat.Prime 342860773; exact ProjectEulerSolutions.P58PrimeNodes.p_342860773.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9258 2) := by change ¬ Nat.Prime 342842257; exact PrattCertificates.composite_of_product (factor := 14009) (cofactor := 24473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9258 3) := by change ¬ Nat.Prime 342823741; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26371057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9258 : row 9258 = 1 := (row_9258.checked).trans rfl
theorem model_count_9258 : modelRow 9258 = 1 := by decide
theorem modelrow_agrees_9258 : modelRow 9258 = row 9258 := model_count_9258.trans row_count_9258.symm
def row_9259 : RowCertificate 9259 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9259 1) := by change ¬ Nat.Prime 342934843; exact PrattCertificates.composite_of_product (factor := 2953) (cofactor := 116131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9259 2) := by change ¬ Nat.Prime 342916325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68583265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9259 3) := by change ¬ Nat.Prime 342897807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114299269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9259 : row 9259 = 0 := (row_9259.checked).trans rfl
theorem model_count_9259 : modelRow 9259 = 0 := by decide
theorem modelrow_agrees_9259 : modelRow 9259 = row 9259 := model_count_9259.trans row_count_9259.symm
def row_9260 : RowCertificate 9260 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9260 1) := by change ¬ Nat.Prime 343008921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114336307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9260 2) := by change ¬ Nat.Prime 342990401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26383877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9260 3) := by change ¬ Nat.Prime 342971881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48995983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9260 : row 9260 = 0 := (row_9260.checked).trans rfl
theorem model_count_9260 : modelRow 9260 = 0 := by decide
theorem modelrow_agrees_9260 : modelRow 9260 = row 9260 := model_count_9260.trans row_count_9260.symm
def row_9261 : RowCertificate 9261 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9261 1) := by change ¬ Nat.Prime 343083007; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2701441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9261 2) := by change ¬ Nat.Prime 343064485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68612897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9261 3) := by change ¬ Nat.Prime 343045963; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26388151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9261 : row 9261 = 0 := (row_9261.checked).trans rfl
theorem model_count_9261 : modelRow 9261 = 0 := by decide
theorem modelrow_agrees_9261 : modelRow 9261 = row 9261 := model_count_9261.trans row_count_9261.symm
def row_9262 : RowCertificate 9262 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9262 1) := by change ¬ Nat.Prime 343157101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49022443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9262 2) := by change Nat.Prime 343138577; exact ProjectEulerSolutions.P58PrimeNodes.p_343138577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9262 3) := by change ¬ Nat.Prime 343120053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114373351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9262 : row 9262 = 1 := (row_9262.checked).trans rfl
theorem model_count_9262 : modelRow 9262 = 1 := by decide
theorem modelrow_agrees_9262 : modelRow 9262 = row 9262 := model_count_9262.trans row_count_9262.symm
def row_9263 : RowCertificate 9263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9263 1) := by change ¬ Nat.Prime 343231203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114410401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9263 2) := by change ¬ Nat.Prime 343212677; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20188981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9263 3) := by change Nat.Prime 343194151; exact ProjectEulerSolutions.P58PrimeNodes.p_343194151.prime
    exact decide_eq_true prime
theorem row_count_9263 : row 9263 = 1 := (row_9263.checked).trans rfl
theorem model_count_9263 : modelRow 9263 = 1 := by decide
theorem modelrow_agrees_9263 : modelRow 9263 = row 9263 := model_count_9263.trans row_count_9263.symm
def row_9264 : RowCertificate 9264 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9264 1) := by change ¬ Nat.Prime 343305313; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26408101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9264 2) := by change ¬ Nat.Prime 343286785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68657357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9264 3) := by change Nat.Prime 343268257; exact ProjectEulerSolutions.P58PrimeNodes.p_343268257.prime
    exact decide_eq_true prime
theorem row_count_9264 : row 9264 = 1 := (row_9264.checked).trans rfl
theorem model_count_9264 : modelRow 9264 = 1 := by decide
theorem modelrow_agrees_9264 : modelRow 9264 = row 9264 := model_count_9264.trans row_count_9264.symm
def row_9265 : RowCertificate 9265 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9265 1) := by change ¬ Nat.Prime 343379431; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5629171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9265 2) := by change ¬ Nat.Prime 343360901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26412377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9265 3) := by change ¬ Nat.Prime 343342371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114447457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9265 : row 9265 = 0 := (row_9265.checked).trans rfl
theorem model_count_9265 : modelRow 9265 = 0 := by decide
theorem modelrow_agrees_9265 : modelRow 9265 = row 9265 := model_count_9265.trans row_count_9265.symm
def row_9266 : RowCertificate 9266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9266 1) := by change ¬ Nat.Prime 343453557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114484519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9266 2) := by change ¬ Nat.Prime 343435025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68687005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9266 3) := by change ¬ Nat.Prime 343416493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49059499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9266 : row 9266 = 0 := (row_9266.checked).trans rfl
theorem model_count_9266 : modelRow 9266 = 0 := by decide
theorem modelrow_agrees_9266 : modelRow 9266 = row 9266 := model_count_9266.trans row_count_9266.symm
def row_9267 : RowCertificate 9267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9267 1) := by change ¬ Nat.Prime 343527691; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26425207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9267 2) := by change ¬ Nat.Prime 343509157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20206421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9267 3) := by change ¬ Nat.Prime 343490623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49070089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9267 : row 9267 = 0 := (row_9267.checked).trans rfl
theorem model_count_9267 : modelRow 9267 = 0 := by decide
theorem modelrow_agrees_9267 : modelRow 9267 = row 9267 := model_count_9267.trans row_count_9267.symm
def row_9268 : RowCertificate 9268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9268 1) := by change ¬ Nat.Prime 343601833; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18084307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9268 2) := by change ¬ Nat.Prime 343583297; exact PrattCertificates.composite_of_product (factor := 3413) (cofactor := 100669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9268 3) := by change ¬ Nat.Prime 343564761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114521587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9268 : row 9268 = 0 := (row_9268.checked).trans rfl
theorem model_count_9268 : modelRow 9268 = 0 := by decide
theorem modelrow_agrees_9268 : modelRow 9268 = row 9268 := model_count_9268.trans row_count_9268.symm
def row_9269 : RowCertificate 9269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9269 1) := by change ¬ Nat.Prime 343675983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114558661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9269 2) := by change ¬ Nat.Prime 343657445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68731489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9269 3) := by change Nat.Prime 343638907; exact ProjectEulerSolutions.P58PrimeNodes.p_343638907.prime
    exact decide_eq_true prime
theorem row_count_9269 : row 9269 = 1 := (row_9269.checked).trans rfl
theorem model_count_9269 : modelRow 9269 = 1 := by decide
theorem modelrow_agrees_9269 : modelRow 9269 = row 9269 := model_count_9269.trans row_count_9269.symm
def row_9270 : RowCertificate 9270 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9270 1) := by change ¬ Nat.Prime 343750141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49107163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9270 2) := by change Nat.Prime 343731601; exact ProjectEulerSolutions.P58PrimeNodes.p_343731601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9270 3) := by change ¬ Nat.Prime 343713061; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 7993327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9270 : row 9270 = 1 := (row_9270.checked).trans rfl
theorem model_count_9270 : modelRow 9270 = 1 := by decide
theorem modelrow_agrees_9270 : modelRow 9270 = row 9270 := model_count_9270.trans row_count_9270.symm
def row_9271 : RowCertificate 9271 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9271 1) := by change Nat.Prime 343824307; exact ProjectEulerSolutions.P58PrimeNodes.p_343824307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9271 2) := by change ¬ Nat.Prime 343805765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68761153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9271 3) := by change ¬ Nat.Prime 343787223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114595741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9271 : row 9271 = 1 := (row_9271.checked).trans rfl
theorem model_count_9271 : modelRow 9271 = 1 := by decide
theorem modelrow_agrees_9271 : modelRow 9271 = row 9271 := model_count_9271.trans row_count_9271.symm
def row_9272 : RowCertificate 9272 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9272 1) := by change ¬ Nat.Prime 343898481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114632827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9272 2) := by change ¬ Nat.Prime 343879937; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 367001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9272 3) := by change ¬ Nat.Prime 343861393; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11092303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9272 : row 9272 = 0 := (row_9272.checked).trans rfl
theorem model_count_9272 : modelRow 9272 = 0 := by decide
theorem modelrow_agrees_9272 : modelRow 9272 = row 9272 := model_count_9272.trans row_count_9272.symm
def row_9273 : RowCertificate 9273 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9273 1) := by change Nat.Prime 343972663; exact ProjectEulerSolutions.P58PrimeNodes.p_343972663.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9273 2) := by change ¬ Nat.Prime 343954117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26458009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9273 3) := by change ¬ Nat.Prime 343935571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49133653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9273 : row 9273 = 1 := (row_9273.checked).trans rfl
theorem model_count_9273 : modelRow 9273 = 1 := by decide
theorem modelrow_agrees_9273 : modelRow 9273 = row 9273 := model_count_9273.trans row_count_9273.symm
def row_9274 : RowCertificate 9274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9274 1) := by change ¬ Nat.Prime 344046853; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1542811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9274 2) := by change ¬ Nat.Prime 344028305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68805661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9274 3) := by change ¬ Nat.Prime 344009757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114669919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9274 : row 9274 = 0 := (row_9274.checked).trans rfl
theorem model_count_9274 : modelRow 9274 = 0 := by decide
theorem modelrow_agrees_9274 : modelRow 9274 = row 9274 := model_count_9274.trans row_count_9274.symm
def row_9275 : RowCertificate 9275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9275 1) := by change ¬ Nat.Prime 344121051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114707017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9275 2) := by change ¬ Nat.Prime 344102501; exact PrattCertificates.composite_of_product (factor := 6101) (cofactor := 56401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9275 3) := by change Nat.Prime 344083951; exact ProjectEulerSolutions.P58PrimeNodes.p_344083951.prime
    exact decide_eq_true prime
theorem row_count_9275 : row 9275 = 1 := (row_9275.checked).trans rfl
theorem model_count_9275 : modelRow 9275 = 1 := by decide
theorem modelrow_agrees_9275 : modelRow 9275 = row 9275 := model_count_9275.trans row_count_9275.symm
def row_9276 : RowCertificate 9276 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9276 1) := by change ¬ Nat.Prime 344195257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49170751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9276 2) := by change ¬ Nat.Prime 344176705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68835341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9276 3) := by change ¬ Nat.Prime 344158153; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18113587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9276 : row 9276 = 0 := (row_9276.checked).trans rfl
theorem model_count_9276 : modelRow 9276 = 0 := by decide
theorem modelrow_agrees_9276 : modelRow 9276 = row 9276 := model_count_9276.trans row_count_9276.symm
def row_9277 : RowCertificate 9277 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9277 1) := by change ¬ Nat.Prime 344269471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49181353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9277 2) := by change ¬ Nat.Prime 344250917; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2192681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9277 3) := by change ¬ Nat.Prime 344232363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114744121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9277 : row 9277 = 0 := (row_9277.checked).trans rfl
theorem model_count_9277 : modelRow 9277 = 0 := by decide
theorem modelrow_agrees_9277 : modelRow 9277 = row 9277 := model_count_9277.trans row_count_9277.symm
def row_9278 : RowCertificate 9278 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9278 1) := by change ¬ Nat.Prime 344343693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114781231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9278 2) := by change ¬ Nat.Prime 344325137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26486549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9278 3) := by change ¬ Nat.Prime 344306581; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18121399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9278 : row 9278 = 0 := (row_9278.checked).trans rfl
theorem model_count_9278 : modelRow 9278 = 0 := by decide
theorem modelrow_agrees_9278 : modelRow 9278 = row 9278 := model_count_9278.trans row_count_9278.symm
def row_9279 : RowCertificate 9279 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9279 1) := by change Nat.Prime 344417923; exact ProjectEulerSolutions.P58PrimeNodes.p_344417923.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9279 2) := by change ¬ Nat.Prime 344399365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68879873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9279 3) := by change ¬ Nat.Prime 344380807; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5645587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9279 : row 9279 = 1 := (row_9279.checked).trans rfl
theorem model_count_9279 : modelRow 9279 = 1 := by decide
theorem modelrow_agrees_9279 : modelRow 9279 = row 9279 := model_count_9279.trans row_count_9279.symm
def row_9280 : RowCertificate 9280 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9280 1) := by change ¬ Nat.Prime 344492161; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26499397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9280 2) := by change ¬ Nat.Prime 344473601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20263153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9280 3) := by change ¬ Nat.Prime 344455041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114818347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9280 : row 9280 = 0 := (row_9280.checked).trans rfl
theorem model_count_9280 : modelRow 9280 = 0 := by decide
theorem modelrow_agrees_9280 : modelRow 9280 = row 9280 := model_count_9280.trans row_count_9280.symm
def row_9281 : RowCertificate 9281 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9281 1) := by change ¬ Nat.Prime 344566407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114855469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9281 2) := by change ¬ Nat.Prime 344547845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68909569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9281 3) := by change ¬ Nat.Prime 344529283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49218469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9281 : row 9281 = 0 := (row_9281.checked).trans rfl
theorem model_count_9281 : modelRow 9281 = 0 := by decide
theorem modelrow_agrees_9281 : modelRow 9281 = row 9281 := model_count_9281.trans row_count_9281.symm
def row_9282 : RowCertificate 9282 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9282 1) := by change Nat.Prime 344640661; exact ProjectEulerSolutions.P58PrimeNodes.p_344640661.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9282 2) := by change ¬ Nat.Prime 344622097; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8405417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9282 3) := by change ¬ Nat.Prime 344603533; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11116243) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9282 : row 9282 = 1 := (row_9282.checked).trans rfl
theorem model_count_9282 : modelRow 9282 = 1 := by decide
theorem modelrow_agrees_9282 : modelRow 9282 = row 9282 := model_count_9282.trans row_count_9282.symm
def row_9283 : RowCertificate 9283 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9283 1) := by change ¬ Nat.Prime 344714923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49244989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9283 2) := by change Nat.Prime 344696357; exact ProjectEulerSolutions.P58PrimeNodes.p_344696357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9283 3) := by change ¬ Nat.Prime 344677791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114892597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9283 : row 9283 = 1 := (row_9283.checked).trans rfl
theorem model_count_9283 : modelRow 9283 = 1 := by decide
theorem modelrow_agrees_9283 : modelRow 9283 = row 9283 := model_count_9283.trans row_count_9283.symm
def row_9284 : RowCertificate 9284 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9284 1) := by change ¬ Nat.Prime 344789193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114929731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9284 2) := by change ¬ Nat.Prime 344770625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68954125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9284 3) := by change ¬ Nat.Prime 344752057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26519389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9284 : row 9284 = 0 := (row_9284.checked).trans rfl
theorem model_count_9284 : modelRow 9284 = 0 := by decide
theorem modelrow_agrees_9284 : modelRow 9284 = row 9284 := model_count_9284.trans row_count_9284.symm
def row_9285 : RowCertificate 9285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9285 1) := by change ¬ Nat.Prime 344863471; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18150709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9285 2) := by change ¬ Nat.Prime 344844901; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 196717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9285 3) := by change ¬ Nat.Prime 344826331; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8019217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9285 : row 9285 = 0 := (row_9285.checked).trans rfl
theorem model_count_9285 : modelRow 9285 = 0 := by decide
theorem modelrow_agrees_9285 : modelRow 9285 = row 9285 := model_count_9285.trans row_count_9285.symm
def row_9286 : RowCertificate 9286 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9286 1) := by change ¬ Nat.Prime 344937757; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1431277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9286 2) := by change ¬ Nat.Prime 344919185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 68983837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9286 3) := by change ¬ Nat.Prime 344900613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114966871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9286 : row 9286 = 0 := (row_9286.checked).trans rfl
theorem model_count_9286 : modelRow 9286 = 0 := by decide
theorem modelrow_agrees_9286 : modelRow 9286 = row 9286 := model_count_9286.trans row_count_9286.symm
def row_9287 : RowCertificate 9287 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9287 1) := by change ¬ Nat.Prime 345012051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115004017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9287 2) := by change ¬ Nat.Prime 344993477; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3415777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9287 3) := by change ¬ Nat.Prime 344974903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49282129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9287 : row 9287 = 0 := (row_9287.checked).trans rfl
theorem model_count_9287 : modelRow 9287 = 0 := by decide
theorem modelrow_agrees_9287 : modelRow 9287 = row 9287 := model_count_9287.trans row_count_9287.symm
def row_9288 : RowCertificate 9288 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9288 1) := by change Nat.Prime 345086353; exact ProjectEulerSolutions.P58PrimeNodes.p_345086353.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9288 2) := by change Nat.Prime 345067777; exact ProjectEulerSolutions.P58PrimeNodes.p_345067777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9288 3) := by change ¬ Nat.Prime 345049201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49292743) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9288 : row 9288 = 2 := (row_9288.checked).trans rfl
theorem model_count_9288 : modelRow 9288 = 2 := by decide
theorem modelrow_agrees_9288 : modelRow 9288 = row 9288 := model_count_9288.trans row_count_9288.symm
def row_9289 : RowCertificate 9289 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9289 1) := by change ¬ Nat.Prime 345160663; exact PrattCertificates.composite_of_product (factor := 2113) (cofactor := 163351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9289 2) := by change ¬ Nat.Prime 345142085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69028417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9289 3) := by change ¬ Nat.Prime 345123507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115041169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9289 : row 9289 = 0 := (row_9289.checked).trans rfl
theorem model_count_9289 : modelRow 9289 = 0 := by decide
theorem modelrow_agrees_9289 : modelRow 9289 = row 9289 := model_count_9289.trans row_count_9289.symm
def row_9290 : RowCertificate 9290 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9290 1) := by change ¬ Nat.Prime 345234981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115078327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9290 2) := by change ¬ Nat.Prime 345216401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9330173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9290 3) := by change ¬ Nat.Prime 345197821; exact PrattCertificates.composite_of_product (factor := 3943) (cofactor := 87547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9290 : row 9290 = 0 := (row_9290.checked).trans rfl
theorem model_count_9290 : modelRow 9290 = 0 := by decide
theorem modelrow_agrees_9290 : modelRow 9290 = row 9290 := model_count_9290.trans row_count_9290.symm
def row_9291 : RowCertificate 9291 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9291 1) := by change ¬ Nat.Prime 345309307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49329901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9291 2) := by change ¬ Nat.Prime 345290725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69058145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9291 3) := by change Nat.Prime 345272143; exact ProjectEulerSolutions.P58PrimeNodes.p_345272143.prime
    exact decide_eq_true prime
theorem row_count_9291 : row 9291 = 1 := (row_9291.checked).trans rfl
theorem model_count_9291 : modelRow 9291 = 1 := by decide
theorem modelrow_agrees_9291 : modelRow 9291 = row 9291 := model_count_9291.trans row_count_9291.symm
def row_9292 : RowCertificate 9292 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9292 1) := by change ¬ Nat.Prime 345383641; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9334693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9292 2) := by change Nat.Prime 345365057; exact ProjectEulerSolutions.P58PrimeNodes.p_345365057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9292 3) := by change ¬ Nat.Prime 345346473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115115491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9292 : row 9292 = 1 := (row_9292.checked).trans rfl
theorem model_count_9292 : modelRow 9292 = 1 := by decide
theorem modelrow_agrees_9292 : modelRow 9292 = row 9292 := model_count_9292.trans row_count_9292.symm
def row_9293 : RowCertificate 9293 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9293 1) := by change ¬ Nat.Prime 345457983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115152661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9293 2) := by change Nat.Prime 345439397; exact ProjectEulerSolutions.P58PrimeNodes.p_345439397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9293 3) := by change ¬ Nat.Prime 345420811; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 287611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9293 : row 9293 = 1 := (row_9293.checked).trans rfl
theorem model_count_9293 : modelRow 9293 = 1 := by decide
theorem modelrow_agrees_9293 : modelRow 9293 = row 9293 := model_count_9293.trans row_count_9293.symm
def row_9294 : RowCertificate 9294 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9294 1) := by change ¬ Nat.Prime 345532333; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5157199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9294 2) := by change ¬ Nat.Prime 345513745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69102749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9294 3) := by change ¬ Nat.Prime 345495157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49356451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9294 : row 9294 = 0 := (row_9294.checked).trans rfl
theorem model_count_9294 : modelRow 9294 = 0 := by decide
theorem modelrow_agrees_9294 : modelRow 9294 = row 9294 := model_count_9294.trans row_count_9294.symm
def row_9295 : RowCertificate 9295 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9295 1) := by change Nat.Prime 345606691; exact ProjectEulerSolutions.P58PrimeNodes.p_345606691.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9295 2) := by change ¬ Nat.Prime 345588101; exact PrattCertificates.composite_of_product (factor := 14537) (cofactor := 23773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9295 3) := by change ¬ Nat.Prime 345569511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115189837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9295 : row 9295 = 1 := (row_9295.checked).trans rfl
theorem model_count_9295 : modelRow 9295 = 1 := by decide
theorem modelrow_agrees_9295 : modelRow 9295 = row 9295 := model_count_9295.trans row_count_9295.symm
def row_9296 : RowCertificate 9296 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9296 1) := by change ¬ Nat.Prime 345681057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115227019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9296 2) := by change ¬ Nat.Prime 345662465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69132493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9296 3) := by change ¬ Nat.Prime 345643873; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5666293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9296 : row 9296 = 0 := (row_9296.checked).trans rfl
theorem model_count_9296 : modelRow 9296 = 0 := by decide
theorem modelrow_agrees_9296 : modelRow 9296 = row 9296 := model_count_9296.trans row_count_9296.symm
def row_9297 : RowCertificate 9297 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9297 1) := by change ¬ Nat.Prime 345755431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49393633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9297 2) := by change ¬ Nat.Prime 345736837; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20337461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9297 3) := by change ¬ Nat.Prime 345718243; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26593711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9297 : row 9297 = 0 := (row_9297.checked).trans rfl
theorem model_count_9297 : modelRow 9297 = 0 := by decide
theorem modelrow_agrees_9297 : modelRow 9297 = row 9297 := model_count_9297.trans row_count_9297.symm
def row_9298 : RowCertificate 9298 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9298 1) := by change ¬ Nat.Prime 345829813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49404259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9298 2) := by change ¬ Nat.Prime 345811217; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 871061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9298 3) := by change ¬ Nat.Prime 345792621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115264207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9298 : row 9298 = 0 := (row_9298.checked).trans rfl
theorem model_count_9298 : modelRow 9298 = 0 := by decide
theorem modelrow_agrees_9298 : modelRow 9298 = row 9298 := model_count_9298.trans row_count_9298.symm
def row_9299 : RowCertificate 9299 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9299 1) := by change ¬ Nat.Prime 345904203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115301401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9299 2) := by change ¬ Nat.Prime 345885605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69177121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9299 3) := by change Nat.Prime 345867007; exact ProjectEulerSolutions.P58PrimeNodes.p_345867007.prime
    exact decide_eq_true prime
theorem row_count_9299 : row 9299 = 1 := (row_9299.checked).trans rfl
theorem model_count_9299 : modelRow 9299 = 1 := by decide
theorem modelrow_agrees_9299 : modelRow 9299 = row 9299 := model_count_9299.trans row_count_9299.symm
def row_9300 : RowCertificate 9300 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9300 1) := by change ¬ Nat.Prime 345978601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9350773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9300 2) := by change Nat.Prime 345960001; exact ProjectEulerSolutions.P58PrimeNodes.p_345960001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9300 3) := by change ¬ Nat.Prime 345941401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26610877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9300 : row 9300 = 1 := (row_9300.checked).trans rfl
theorem model_count_9300 : modelRow 9300 = 1 := by decide
theorem modelrow_agrees_9300 : modelRow 9300 = row 9300 := model_count_9300.trans row_count_9300.symm
def row_9301 : RowCertificate 9301 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9301 1) := by change ¬ Nat.Prime 346053007; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 693493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9301 2) := by change ¬ Nat.Prime 346034405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69206881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9301 3) := by change ¬ Nat.Prime 346015803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115338601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9301 : row 9301 = 0 := (row_9301.checked).trans rfl
theorem model_count_9301 : modelRow 9301 = 0 := by decide
theorem modelrow_agrees_9301 : modelRow 9301 = row 9301 := model_count_9301.trans row_count_9301.symm
def row_9302 : RowCertificate 9302 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9302 1) := by change ¬ Nat.Prime 346127421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115375807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9302 2) := by change ¬ Nat.Prime 346108817; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 2000629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9302 3) := by change ¬ Nat.Prime 346090213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49441459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9302 : row 9302 = 0 := (row_9302.checked).trans rfl
theorem model_count_9302 : modelRow 9302 = 0 := by decide
theorem modelrow_agrees_9302 : modelRow 9302 = row 9302 := model_count_9302.trans row_count_9302.symm
def row_9303 : RowCertificate 9303 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9303 1) := by change ¬ Nat.Prime 346201843; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26630911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9303 2) := by change ¬ Nat.Prime 346183237; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11937353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9303 3) := by change ¬ Nat.Prime 346164631; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11166601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9303 : row 9303 = 0 := (row_9303.checked).trans rfl
theorem model_count_9303 : modelRow 9303 = 0 := by decide
theorem modelrow_agrees_9303 : modelRow 9303 = row 9303 := model_count_9303.trans row_count_9303.symm
def row_9304 : RowCertificate 9304 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9304 1) := by change ¬ Nat.Prime 346276273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49468039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9304 2) := by change ¬ Nat.Prime 346257665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69251533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9304 3) := by change ¬ Nat.Prime 346239057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115413019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9304 : row 9304 = 0 := (row_9304.checked).trans rfl
theorem model_count_9304 : modelRow 9304 = 0 := by decide
theorem modelrow_agrees_9304 : modelRow 9304 = row 9304 := model_count_9304.trans row_count_9304.symm
def row_9305 : RowCertificate 9305 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9305 1) := by change ¬ Nat.Prime 346350711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115450237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9305 2) := by change Nat.Prime 346332101; exact ProjectEulerSolutions.P58PrimeNodes.p_346332101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9305 3) := by change ¬ Nat.Prime 346313491; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 788869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9305 : row 9305 = 1 := (row_9305.checked).trans rfl
theorem model_count_9305 : modelRow 9305 = 1 := by decide
theorem modelrow_agrees_9305 : modelRow 9305 = row 9305 := model_count_9305.trans row_count_9305.symm
def row_9306 : RowCertificate 9306 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9306 1) := by change ¬ Nat.Prime 346425157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26648089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9306 2) := by change ¬ Nat.Prime 346406545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69281309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9306 3) := by change ¬ Nat.Prime 346387933; exact PrattCertificates.composite_of_product (factor := 1663) (cofactor := 208291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9306 : row 9306 = 0 := (row_9306.checked).trans rfl
theorem model_count_9306 : modelRow 9306 = 0 := by decide
theorem modelrow_agrees_9306 : modelRow 9306 = row 9306 := model_count_9306.trans row_count_9306.symm
def row_9307 : RowCertificate 9307 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9307 1) := by change Nat.Prime 346499611; exact ProjectEulerSolutions.P58PrimeNodes.p_346499611.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9307 2) := by change ¬ Nat.Prime 346480997; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 1182529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9307 3) := by change ¬ Nat.Prime 346462383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115487461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9307 : row 9307 = 1 := (row_9307.checked).trans rfl
theorem model_count_9307 : modelRow 9307 = 1 := by decide
theorem modelrow_agrees_9307 : modelRow 9307 = row 9307 := model_count_9307.trans row_count_9307.symm
def row_9308 : RowCertificate 9308 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9308 1) := by change ¬ Nat.Prime 346574073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115524691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9308 2) := by change ¬ Nat.Prime 346555457; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5681237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9308 3) := by change ¬ Nat.Prime 346536841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49505263) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9308 : row 9308 = 0 := (row_9308.checked).trans rfl
theorem model_count_9308 : modelRow 9308 = 0 := by decide
theorem modelrow_agrees_9308 : modelRow 9308 = row 9308 := model_count_9308.trans row_count_9308.symm
def row_9309 : RowCertificate 9309 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9309 1) := by change ¬ Nat.Prime 346648543; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5682763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9309 2) := by change ¬ Nat.Prime 346629925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69325985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9309 3) := by change ¬ Nat.Prime 346611307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49515901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9309 : row 9309 = 0 := (row_9309.checked).trans rfl
theorem model_count_9309 : modelRow 9309 = 0 := by decide
theorem modelrow_agrees_9309 : modelRow 9309 = row 9309 := model_count_9309.trans row_count_9309.symm
def row_9310 : RowCertificate 9310 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9310 1) := by change ¬ Nat.Prime 346723021; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4388899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9310 2) := by change Nat.Prime 346704401; exact ProjectEulerSolutions.P58PrimeNodes.p_346704401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9310 3) := by change ¬ Nat.Prime 346685781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115561927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9310 : row 9310 = 1 := (row_9310.checked).trans rfl
theorem model_count_9310 : modelRow 9310 = 1 := by decide
theorem modelrow_agrees_9310 : modelRow 9310 = row 9310 := model_count_9310.trans row_count_9310.symm
def row_9311 : RowCertificate 9311 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9311 1) := by change ¬ Nat.Prime 346797507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115599169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9311 2) := by change ¬ Nat.Prime 346778885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69355777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9311 3) := by change ¬ Nat.Prime 346760263; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9371899) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9311 : row 9311 = 0 := (row_9311.checked).trans rfl
theorem model_count_9311 : modelRow 9311 = 0 := by decide
theorem modelrow_agrees_9311 : modelRow 9311 = row 9311 := model_count_9311.trans row_count_9311.symm
def row_9312 : RowCertificate 9312 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9312 1) := by change ¬ Nat.Prime 346872001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49553143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9312 2) := by change ¬ Nat.Prime 346853377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26681029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9312 3) := by change ¬ Nat.Prime 346834753; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4751161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9312 : row 9312 = 0 := (row_9312.checked).trans rfl
theorem model_count_9312 : modelRow 9312 = 0 := by decide
theorem modelrow_agrees_9312 : modelRow 9312 = row 9312 := model_count_9312.trans row_count_9312.symm
def row_9313 : RowCertificate 9313 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9313 1) := by change Nat.Prime 346946503; exact ProjectEulerSolutions.P58PrimeNodes.p_346946503.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9313 2) := by change ¬ Nat.Prime 346927877; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6545809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9313 3) := by change ¬ Nat.Prime 346909251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115636417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9313 : row 9313 = 1 := (row_9313.checked).trans rfl
theorem model_count_9313 : modelRow 9313 = 1 := by decide
theorem modelrow_agrees_9313 : modelRow 9313 = row 9313 := model_count_9313.trans row_count_9313.symm
def row_9314 : RowCertificate 9314 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9314 1) := by change ¬ Nat.Prime 347021013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115673671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9314 2) := by change ¬ Nat.Prime 347002385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69400477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9314 3) := by change ¬ Nat.Prime 346983757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18262303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9314 : row 9314 = 0 := (row_9314.checked).trans rfl
theorem model_count_9314 : modelRow 9314 = 0 := by decide
theorem modelrow_agrees_9314 : modelRow 9314 = row 9314 := model_count_9314.trans row_count_9314.symm
def row_9315 : RowCertificate 9315 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9315 1) := by change ¬ Nat.Prime 347095531; exact PrattCertificates.composite_of_product (factor := 12163) (cofactor := 28537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9315 2) := by change ¬ Nat.Prime 347076901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 11968169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9315 3) := by change ¬ Nat.Prime 347058271; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49579753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9315 : row 9315 = 0 := (row_9315.checked).trans rfl
theorem model_count_9315 : modelRow 9315 = 0 := by decide
theorem modelrow_agrees_9315 : modelRow 9315 = row 9315 := model_count_9315.trans row_count_9315.symm
def row_9316 : RowCertificate 9316 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9316 1) := by change ¬ Nat.Prime 347170057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26705389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9316 2) := by change ¬ Nat.Prime 347151425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69430285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9316 3) := by change ¬ Nat.Prime 347132793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115710931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9316 : row 9316 = 0 := (row_9316.checked).trans rfl
theorem model_count_9316 : modelRow 9316 = 0 := by decide
theorem modelrow_agrees_9316 : modelRow 9316 = row 9316 := model_count_9316.trans row_count_9316.symm
def row_9317 : RowCertificate 9317 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9317 1) := by change ¬ Nat.Prime 347244591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115748197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9317 2) := by change ¬ Nat.Prime 347225957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26709689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9317 3) := by change ¬ Nat.Prime 347207323; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 1226881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9317 : row 9317 = 0 := (row_9317.checked).trans rfl
theorem model_count_9317 : modelRow 9317 = 0 := by decide
theorem modelrow_agrees_9317 : modelRow 9317 = row 9317 := model_count_9317.trans row_count_9317.symm
def row_9318 : RowCertificate 9318 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9318 1) := by change ¬ Nat.Prime 347319133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49617019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9318 2) := by change ¬ Nat.Prime 347300497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20429441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9318 3) := by change Nat.Prime 347281861; exact ProjectEulerSolutions.P58PrimeNodes.p_347281861.prime
    exact decide_eq_true prime
theorem row_count_9318 : row 9318 = 1 := (row_9318.checked).trans rfl
theorem model_count_9318 : modelRow 9318 = 1 := by decide
theorem modelrow_agrees_9318 : modelRow 9318 = row 9318 := model_count_9318.trans row_count_9318.symm
def row_9319 : RowCertificate 9319 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9319 1) := by change ¬ Nat.Prime 347393683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49627669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9319 2) := by change ¬ Nat.Prime 347375045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69475009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9319 3) := by change ¬ Nat.Prime 347356407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115785469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9319 : row 9319 = 0 := (row_9319.checked).trans rfl
theorem model_count_9319 : modelRow 9319 = 0 := by decide
theorem modelrow_agrees_9319 : modelRow 9319 = row 9319 := model_count_9319.trans row_count_9319.symm
def row_9320 : RowCertificate 9320 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9320 1) := by change ¬ Nat.Prime 347468241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115822747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9320 2) := by change ¬ Nat.Prime 347449601; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1800257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9320 3) := by change ¬ Nat.Prime 347430961; exact PrattCertificates.composite_of_product (factor := 15679) (cofactor := 22159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9320 : row 9320 = 0 := (row_9320.checked).trans rfl
theorem model_count_9320 : modelRow 9320 = 0 := by decide
theorem modelrow_agrees_9320 : modelRow 9320 = row 9320 := model_count_9320.trans row_count_9320.symm
def row_9321 : RowCertificate 9321 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9321 1) := by change Nat.Prime 347542807; exact ProjectEulerSolutions.P58PrimeNodes.p_347542807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9321 2) := by change ¬ Nat.Prime 347524165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69504833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9321 3) := by change ¬ Nat.Prime 347505523; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 278227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9321 : row 9321 = 1 := (row_9321.checked).trans rfl
theorem model_count_9321 : modelRow 9321 = 1 := by decide
theorem modelrow_agrees_9321 : modelRow 9321 = row 9321 := model_count_9321.trans row_count_9321.symm
def row_9322 : RowCertificate 9322 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9322 1) := by change Nat.Prime 347617381; exact ProjectEulerSolutions.P58PrimeNodes.p_347617381.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9322 2) := by change Nat.Prime 347598737; exact ProjectEulerSolutions.P58PrimeNodes.p_347598737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9322 3) := by change ¬ Nat.Prime 347580093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115860031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9322 : row 9322 = 2 := (row_9322.checked).trans rfl
theorem model_count_9322 : modelRow 9322 = 2 := by decide
theorem modelrow_agrees_9322 : modelRow 9322 = row 9322 := model_count_9322.trans row_count_9322.symm
def row_9323 : RowCertificate 9323 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9323 1) := by change ¬ Nat.Prime 347691963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115897321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9323 2) := by change ¬ Nat.Prime 347673317; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8479837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9323 3) := by change ¬ Nat.Prime 347654671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49664953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9323 : row 9323 = 0 := (row_9323.checked).trans rfl
theorem model_count_9323 : modelRow 9323 = 0 := by decide
theorem modelrow_agrees_9323 : modelRow 9323 = row 9323 := model_count_9323.trans row_count_9323.symm
def row_9324 : RowCertificate 9324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9324 1) := by change ¬ Nat.Prime 347766553; exact PrattCertificates.composite_of_product (factor := 9241) (cofactor := 37633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9324 2) := by change ¬ Nat.Prime 347747905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69549581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9324 3) := by change ¬ Nat.Prime 347729257; exact PrattCertificates.composite_of_product (factor := 2749) (cofactor := 126493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9324 : row 9324 = 0 := (row_9324.checked).trans rfl
theorem model_count_9324 : modelRow 9324 = 0 := by decide
theorem modelrow_agrees_9324 : modelRow 9324 = row 9324 := model_count_9324.trans row_count_9324.symm
def row_9325 : RowCertificate 9325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9325 1) := by change ¬ Nat.Prime 347841151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49691593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9325 2) := by change ¬ Nat.Prime 347822501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26755577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9325 3) := by change ¬ Nat.Prime 347803851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115934617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9325 : row 9325 = 0 := (row_9325.checked).trans rfl
theorem model_count_9325 : modelRow 9325 = 0 := by decide
theorem modelrow_agrees_9325 : modelRow 9325 = row 9325 := model_count_9325.trans row_count_9325.symm
def row_9326 : RowCertificate 9326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9326 1) := by change ¬ Nat.Prime 347915757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115971919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9326 2) := by change ¬ Nat.Prime 347897105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69579421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9326 3) := by change ¬ Nat.Prime 347878453; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26759881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9326 : row 9326 = 0 := (row_9326.checked).trans rfl
theorem model_count_9326 : modelRow 9326 = 0 := by decide
theorem modelrow_agrees_9326 : modelRow 9326 = row 9326 := model_count_9326.trans row_count_9326.symm
def row_9327 : RowCertificate 9327 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9327 1) := by change Nat.Prime 347990371; exact ProjectEulerSolutions.P58PrimeNodes.p_347990371.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9327 2) := by change ¬ Nat.Prime 347971717; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9404641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9327 3) := by change Nat.Prime 347953063; exact ProjectEulerSolutions.P58PrimeNodes.p_347953063.prime
    exact decide_eq_true prime
theorem row_count_9327 : row 9327 = 2 := (row_9327.checked).trans rfl
theorem model_count_9327 : modelRow 9327 = 2 := by decide
theorem modelrow_agrees_9327 : modelRow 9327 = row 9327 := model_count_9327.trans row_count_9327.symm
def row_9328 : RowCertificate 9328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9328 1) := by change ¬ Nat.Prime 348064993; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11227903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9328 2) := by change ¬ Nat.Prime 348046337; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3910633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9328 3) := by change ¬ Nat.Prime 348027681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116009227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9328 : row 9328 = 0 := (row_9328.checked).trans rfl
theorem model_count_9328 : modelRow 9328 = 0 := by decide
theorem modelrow_agrees_9328 : modelRow 9328 = row 9328 := model_count_9328.trans row_count_9328.symm
def row_9329 : RowCertificate 9329 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9329 1) := by change ¬ Nat.Prime 348139623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116046541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9329 2) := by change ¬ Nat.Prime 348120965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69624193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9329 3) := by change ¬ Nat.Prime 348102307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49728901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9329 : row 9329 = 0 := (row_9329.checked).trans rfl
theorem model_count_9329 : modelRow 9329 = 0 := by decide
theorem modelrow_agrees_9329 : modelRow 9329 = row 9329 := model_count_9329.trans row_count_9329.symm
def row_9330 : RowCertificate 9330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9330 1) := by change ¬ Nat.Prime 348214261; exact PrattCertificates.composite_of_product (factor := 6151) (cofactor := 56611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9330 2) := by change ¬ Nat.Prime 348195601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26784277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9330 3) := by change ¬ Nat.Prime 348176941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49739563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9330 : row 9330 = 0 := (row_9330.checked).trans rfl
theorem model_count_9330 : modelRow 9330 = 0 := by decide
theorem modelrow_agrees_9330 : modelRow 9330 = row 9330 := model_count_9330.trans row_count_9330.symm
def row_9331 : RowCertificate 9331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9331 1) := by change ¬ Nat.Prime 348288907; exact PrattCertificates.composite_of_product (factor := 907) (cofactor := 384001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9331 2) := by change ¬ Nat.Prime 348270245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69654049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9331 3) := by change ¬ Nat.Prime 348251583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116083861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9331 : row 9331 = 0 := (row_9331.checked).trans rfl
theorem model_count_9331 : modelRow 9331 = 0 := by decide
theorem modelrow_agrees_9331 : modelRow 9331 = row 9331 := model_count_9331.trans row_count_9331.symm
def row_9332 : RowCertificate 9332 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9332 1) := by change ¬ Nat.Prime 348363561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116121187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9332 2) := by change ¬ Nat.Prime 348344897; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12011893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9332 3) := by change ¬ Nat.Prime 348326233; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5198899) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9332 : row 9332 = 0 := (row_9332.checked).trans rfl
theorem model_count_9332 : modelRow 9332 = 0 := by decide
theorem modelrow_agrees_9332 : modelRow 9332 = row 9332 := model_count_9332.trans row_count_9332.symm
def row_9333 : RowCertificate 9333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9333 1) := by change ¬ Nat.Prime 348438223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49776889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9333 2) := by change Nat.Prime 348419557; exact ProjectEulerSolutions.P58PrimeNodes.p_348419557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9333 3) := by change ¬ Nat.Prime 348400891; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18336889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9333 : row 9333 = 1 := (row_9333.checked).trans rfl
theorem model_count_9333 : modelRow 9333 = 1 := by decide
theorem modelrow_agrees_9333 : modelRow 9333 = row 9333 := model_count_9333.trans row_count_9333.symm
def row_9334 : RowCertificate 9334 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9334 1) := by change ¬ Nat.Prime 348512893; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8104951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9334 2) := by change ¬ Nat.Prime 348494225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69698845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9334 3) := by change ¬ Nat.Prime 348475557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116158519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9334 : row 9334 = 0 := (row_9334.checked).trans rfl
theorem model_count_9334 : modelRow 9334 = 0 := by decide
theorem modelrow_agrees_9334 : modelRow 9334 = row 9334 := model_count_9334.trans row_count_9334.symm
def row_9335 : RowCertificate 9335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9335 1) := by change ¬ Nat.Prime 348587571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116195857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9335 2) := by change ¬ Nat.Prime 348568901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20504053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9335 3) := by change ¬ Nat.Prime 348550231; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18344749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9335 : row 9335 = 0 := (row_9335.checked).trans rfl
theorem model_count_9335 : modelRow 9335 = 0 := by decide
theorem modelrow_agrees_9335 : modelRow 9335 = row 9335 := model_count_9335.trans row_count_9335.symm
def row_9336 : RowCertificate 9336 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9336 1) := by change Nat.Prime 348662257; exact ProjectEulerSolutions.P58PrimeNodes.p_348662257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9336 2) := by change ¬ Nat.Prime 348643585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69728717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9336 3) := by change ¬ Nat.Prime 348624913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49803559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9336 : row 9336 = 1 := (row_9336.checked).trans rfl
theorem model_count_9336 : modelRow 9336 = 1 := by decide
theorem modelrow_agrees_9336 : modelRow 9336 = row 9336 := model_count_9336.trans row_count_9336.symm
def row_9337 : RowCertificate 9337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9337 1) := by change ¬ Nat.Prime 348736951; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9425323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9337 2) := by change Nat.Prime 348718277; exact ProjectEulerSolutions.P58PrimeNodes.p_348718277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9337 3) := by change ¬ Nat.Prime 348699603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116233201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9337 : row 9337 = 1 := (row_9337.checked).trans rfl
theorem model_count_9337 : modelRow 9337 = 1 := by decide
theorem modelrow_agrees_9337 : modelRow 9337 = row 9337 := model_count_9337.trans row_count_9337.symm
def row_9338 : RowCertificate 9338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9338 1) := by change ¬ Nat.Prime 348811653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116270551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9338 2) := by change ¬ Nat.Prime 348792977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26830229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9338 3) := by change ¬ Nat.Prime 348774301; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1259113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9338 : row 9338 = 0 := (row_9338.checked).trans rfl
theorem model_count_9338 : modelRow 9338 = 0 := by decide
theorem modelrow_agrees_9338 : modelRow 9338 = row 9338 := model_count_9338.trans row_count_9338.symm
def row_9339 : RowCertificate 9339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9339 1) := by change ¬ Nat.Prime 348886363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49840909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9339 2) := by change ¬ Nat.Prime 348867685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69773537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9339 3) := by change ¬ Nat.Prime 348849007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26834539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9339 : row 9339 = 0 := (row_9339.checked).trans rfl
theorem model_count_9339 : modelRow 9339 = 0 := by decide
theorem modelrow_agrees_9339 : modelRow 9339 = row 9339 := model_count_9339.trans row_count_9339.symm
def row_9340 : RowCertificate 9340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9340 1) := by change ¬ Nat.Prime 348961081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49851583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9340 2) := by change ¬ Nat.Prime 348942401; exact PrattCertificates.composite_of_product (factor := 16493) (cofactor := 21157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9340 3) := by change ¬ Nat.Prime 348923721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116307907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9340 : row 9340 = 0 := (row_9340.checked).trans rfl
theorem model_count_9340 : modelRow 9340 = 0 := by decide
theorem modelrow_agrees_9340 : modelRow 9340 = row 9340 := model_count_9340.trans row_count_9340.symm
def row_9341 : RowCertificate 9341 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9341 1) := by change ¬ Nat.Prime 349035807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116345269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9341 2) := by change ¬ Nat.Prime 349017125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69803425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9341 3) := by change ¬ Nat.Prime 348998443; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1115011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9341 : row 9341 = 0 := (row_9341.checked).trans rfl
theorem model_count_9341 : modelRow 9341 = 0 := by decide
theorem modelrow_agrees_9341 : modelRow 9341 = row 9341 := model_count_9341.trans row_count_9341.symm
def row_9342 : RowCertificate 9342 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9342 1) := by change ¬ Nat.Prime 349110541; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26854657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9342 2) := by change Nat.Prime 349091857; exact ProjectEulerSolutions.P58PrimeNodes.p_349091857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9342 3) := by change ¬ Nat.Prime 349073173; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1524337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9342 : row 9342 = 1 := (row_9342.checked).trans rfl
theorem model_count_9342 : modelRow 9342 = 1 := by decide
theorem modelrow_agrees_9342 : modelRow 9342 = row 9342 := model_count_9342.trans row_count_9342.symm
def row_9343 : RowCertificate 9343 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9343 1) := by change ¬ Nat.Prime 349185283; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2142241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9343 2) := by change ¬ Nat.Prime 349166597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26858969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9343 3) := by change ¬ Nat.Prime 349147911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116382637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9343 : row 9343 = 0 := (row_9343.checked).trans rfl
theorem model_count_9343 : modelRow 9343 = 0 := by decide
theorem modelrow_agrees_9343 : modelRow 9343 = row 9343 := model_count_9343.trans row_count_9343.symm
def row_9344 : RowCertificate 9344 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9344 1) := by change ¬ Nat.Prime 349260033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116420011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9344 2) := by change ¬ Nat.Prime 349241345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69848269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9344 3) := by change ¬ Nat.Prime 349222657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49888951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9344 : row 9344 = 0 := (row_9344.checked).trans rfl
theorem model_count_9344 : modelRow 9344 = 0 := by decide
theorem modelrow_agrees_9344 : modelRow 9344 = row 9344 := model_count_9344.trans row_count_9344.symm
private def atomic_9217 : CountSegment 1 10 13120 9216 9217 3845 3845 := CountSegment.single (first := 9216) (initial := 3845) row_count_9217 (by decide)
private theorem atomic_9217_agreement : RowAgreement 9216 9217 := RowAgreement.single (first := 9216) modelrow_agrees_9217
private def atomic_9218 : CountSegment 1 10 13120 9217 9218 3845 3845 := CountSegment.single (first := 9217) (initial := 3845) row_count_9218 (by decide)
private theorem atomic_9218_agreement : RowAgreement 9217 9218 := RowAgreement.single (first := 9217) modelrow_agrees_9218
private def atomic_9219 : CountSegment 1 10 13120 9218 9219 3845 3846 := CountSegment.single (first := 9218) (initial := 3845) row_count_9219 (by decide)
private theorem atomic_9219_agreement : RowAgreement 9218 9219 := RowAgreement.single (first := 9218) modelrow_agrees_9219
private def atomic_9220 : CountSegment 1 10 13120 9219 9220 3846 3846 := CountSegment.single (first := 9219) (initial := 3846) row_count_9220 (by decide)
private theorem atomic_9220_agreement : RowAgreement 9219 9220 := RowAgreement.single (first := 9219) modelrow_agrees_9220
private def atomic_9221 : CountSegment 1 10 13120 9220 9221 3846 3846 := CountSegment.single (first := 9220) (initial := 3846) row_count_9221 (by decide)
private theorem atomic_9221_agreement : RowAgreement 9220 9221 := RowAgreement.single (first := 9220) modelrow_agrees_9221
private def atomic_9222 : CountSegment 1 10 13120 9221 9222 3846 3847 := CountSegment.single (first := 9221) (initial := 3846) row_count_9222 (by decide)
private theorem atomic_9222_agreement : RowAgreement 9221 9222 := RowAgreement.single (first := 9221) modelrow_agrees_9222
private def atomic_9223 : CountSegment 1 10 13120 9222 9223 3847 3848 := CountSegment.single (first := 9222) (initial := 3847) row_count_9223 (by decide)
private theorem atomic_9223_agreement : RowAgreement 9222 9223 := RowAgreement.single (first := 9222) modelrow_agrees_9223
private def atomic_9224 : CountSegment 1 10 13120 9223 9224 3848 3848 := CountSegment.single (first := 9223) (initial := 3848) row_count_9224 (by decide)
private theorem atomic_9224_agreement : RowAgreement 9223 9224 := RowAgreement.single (first := 9223) modelrow_agrees_9224
private def atomic_9225 : CountSegment 1 10 13120 9224 9225 3848 3849 := CountSegment.single (first := 9224) (initial := 3848) row_count_9225 (by decide)
private theorem atomic_9225_agreement : RowAgreement 9224 9225 := RowAgreement.single (first := 9224) modelrow_agrees_9225
private def atomic_9226 : CountSegment 1 10 13120 9225 9226 3849 3849 := CountSegment.single (first := 9225) (initial := 3849) row_count_9226 (by decide)
private theorem atomic_9226_agreement : RowAgreement 9225 9226 := RowAgreement.single (first := 9225) modelrow_agrees_9226
private def atomic_9227 : CountSegment 1 10 13120 9226 9227 3849 3849 := CountSegment.single (first := 9226) (initial := 3849) row_count_9227 (by decide)
private theorem atomic_9227_agreement : RowAgreement 9226 9227 := RowAgreement.single (first := 9226) modelrow_agrees_9227
private def atomic_9228 : CountSegment 1 10 13120 9227 9228 3849 3849 := CountSegment.single (first := 9227) (initial := 3849) row_count_9228 (by decide)
private theorem atomic_9228_agreement : RowAgreement 9227 9228 := RowAgreement.single (first := 9227) modelrow_agrees_9228
private def atomic_9229 : CountSegment 1 10 13120 9228 9229 3849 3849 := CountSegment.single (first := 9228) (initial := 3849) row_count_9229 (by decide)
private theorem atomic_9229_agreement : RowAgreement 9228 9229 := RowAgreement.single (first := 9228) modelrow_agrees_9229
private def atomic_9230 : CountSegment 1 10 13120 9229 9230 3849 3849 := CountSegment.single (first := 9229) (initial := 3849) row_count_9230 (by decide)
private theorem atomic_9230_agreement : RowAgreement 9229 9230 := RowAgreement.single (first := 9229) modelrow_agrees_9230
private def atomic_9231 : CountSegment 1 10 13120 9230 9231 3849 3849 := CountSegment.single (first := 9230) (initial := 3849) row_count_9231 (by decide)
private theorem atomic_9231_agreement : RowAgreement 9230 9231 := RowAgreement.single (first := 9230) modelrow_agrees_9231
private def atomic_9232 : CountSegment 1 10 13120 9231 9232 3849 3851 := CountSegment.single (first := 9231) (initial := 3849) row_count_9232 (by decide)
private theorem atomic_9232_agreement : RowAgreement 9231 9232 := RowAgreement.single (first := 9231) modelrow_agrees_9232
private def atomic_9233 : CountSegment 1 10 13120 9232 9233 3851 3851 := CountSegment.single (first := 9232) (initial := 3851) row_count_9233 (by decide)
private theorem atomic_9233_agreement : RowAgreement 9232 9233 := RowAgreement.single (first := 9232) modelrow_agrees_9233
private def atomic_9234 : CountSegment 1 10 13120 9233 9234 3851 3852 := CountSegment.single (first := 9233) (initial := 3851) row_count_9234 (by decide)
private theorem atomic_9234_agreement : RowAgreement 9233 9234 := RowAgreement.single (first := 9233) modelrow_agrees_9234
private def atomic_9235 : CountSegment 1 10 13120 9234 9235 3852 3852 := CountSegment.single (first := 9234) (initial := 3852) row_count_9235 (by decide)
private theorem atomic_9235_agreement : RowAgreement 9234 9235 := RowAgreement.single (first := 9234) modelrow_agrees_9235
private def atomic_9236 : CountSegment 1 10 13120 9235 9236 3852 3852 := CountSegment.single (first := 9235) (initial := 3852) row_count_9236 (by decide)
private theorem atomic_9236_agreement : RowAgreement 9235 9236 := RowAgreement.single (first := 9235) modelrow_agrees_9236
private def atomic_9237 : CountSegment 1 10 13120 9236 9237 3852 3853 := CountSegment.single (first := 9236) (initial := 3852) row_count_9237 (by decide)
private theorem atomic_9237_agreement : RowAgreement 9236 9237 := RowAgreement.single (first := 9236) modelrow_agrees_9237
private def atomic_9238 : CountSegment 1 10 13120 9237 9238 3853 3853 := CountSegment.single (first := 9237) (initial := 3853) row_count_9238 (by decide)
private theorem atomic_9238_agreement : RowAgreement 9237 9238 := RowAgreement.single (first := 9237) modelrow_agrees_9238
private def atomic_9239 : CountSegment 1 10 13120 9238 9239 3853 3853 := CountSegment.single (first := 9238) (initial := 3853) row_count_9239 (by decide)
private theorem atomic_9239_agreement : RowAgreement 9238 9239 := RowAgreement.single (first := 9238) modelrow_agrees_9239
private def atomic_9240 : CountSegment 1 10 13120 9239 9240 3853 3854 := CountSegment.single (first := 9239) (initial := 3853) row_count_9240 (by decide)
private theorem atomic_9240_agreement : RowAgreement 9239 9240 := RowAgreement.single (first := 9239) modelrow_agrees_9240
private def atomic_9241 : CountSegment 1 10 13120 9240 9241 3854 3854 := CountSegment.single (first := 9240) (initial := 3854) row_count_9241 (by decide)
private theorem atomic_9241_agreement : RowAgreement 9240 9241 := RowAgreement.single (first := 9240) modelrow_agrees_9241
private def atomic_9242 : CountSegment 1 10 13120 9241 9242 3854 3854 := CountSegment.single (first := 9241) (initial := 3854) row_count_9242 (by decide)
private theorem atomic_9242_agreement : RowAgreement 9241 9242 := RowAgreement.single (first := 9241) modelrow_agrees_9242
private def atomic_9243 : CountSegment 1 10 13120 9242 9243 3854 3857 := CountSegment.single (first := 9242) (initial := 3854) row_count_9243 (by decide)
private theorem atomic_9243_agreement : RowAgreement 9242 9243 := RowAgreement.single (first := 9242) modelrow_agrees_9243
private def atomic_9244 : CountSegment 1 10 13120 9243 9244 3857 3858 := CountSegment.single (first := 9243) (initial := 3857) row_count_9244 (by decide)
private theorem atomic_9244_agreement : RowAgreement 9243 9244 := RowAgreement.single (first := 9243) modelrow_agrees_9244
private def atomic_9245 : CountSegment 1 10 13120 9244 9245 3858 3858 := CountSegment.single (first := 9244) (initial := 3858) row_count_9245 (by decide)
private theorem atomic_9245_agreement : RowAgreement 9244 9245 := RowAgreement.single (first := 9244) modelrow_agrees_9245
private def atomic_9246 : CountSegment 1 10 13120 9245 9246 3858 3859 := CountSegment.single (first := 9245) (initial := 3858) row_count_9246 (by decide)
private theorem atomic_9246_agreement : RowAgreement 9245 9246 := RowAgreement.single (first := 9245) modelrow_agrees_9246
private def atomic_9247 : CountSegment 1 10 13120 9246 9247 3859 3859 := CountSegment.single (first := 9246) (initial := 3859) row_count_9247 (by decide)
private theorem atomic_9247_agreement : RowAgreement 9246 9247 := RowAgreement.single (first := 9246) modelrow_agrees_9247
private def atomic_9248 : CountSegment 1 10 13120 9247 9248 3859 3859 := CountSegment.single (first := 9247) (initial := 3859) row_count_9248 (by decide)
private theorem atomic_9248_agreement : RowAgreement 9247 9248 := RowAgreement.single (first := 9247) modelrow_agrees_9248
private def atomic_9249 : CountSegment 1 10 13120 9248 9249 3859 3860 := CountSegment.single (first := 9248) (initial := 3859) row_count_9249 (by decide)
private theorem atomic_9249_agreement : RowAgreement 9248 9249 := RowAgreement.single (first := 9248) modelrow_agrees_9249
private def atomic_9250 : CountSegment 1 10 13120 9249 9250 3860 3861 := CountSegment.single (first := 9249) (initial := 3860) row_count_9250 (by decide)
private theorem atomic_9250_agreement : RowAgreement 9249 9250 := RowAgreement.single (first := 9249) modelrow_agrees_9250
private def atomic_9251 : CountSegment 1 10 13120 9250 9251 3861 3861 := CountSegment.single (first := 9250) (initial := 3861) row_count_9251 (by decide)
private theorem atomic_9251_agreement : RowAgreement 9250 9251 := RowAgreement.single (first := 9250) modelrow_agrees_9251
private def atomic_9252 : CountSegment 1 10 13120 9251 9252 3861 3861 := CountSegment.single (first := 9251) (initial := 3861) row_count_9252 (by decide)
private theorem atomic_9252_agreement : RowAgreement 9251 9252 := RowAgreement.single (first := 9251) modelrow_agrees_9252
private def atomic_9253 : CountSegment 1 10 13120 9252 9253 3861 3862 := CountSegment.single (first := 9252) (initial := 3861) row_count_9253 (by decide)
private theorem atomic_9253_agreement : RowAgreement 9252 9253 := RowAgreement.single (first := 9252) modelrow_agrees_9253
private def atomic_9254 : CountSegment 1 10 13120 9253 9254 3862 3863 := CountSegment.single (first := 9253) (initial := 3862) row_count_9254 (by decide)
private theorem atomic_9254_agreement : RowAgreement 9253 9254 := RowAgreement.single (first := 9253) modelrow_agrees_9254
private def atomic_9255 : CountSegment 1 10 13120 9254 9255 3863 3863 := CountSegment.single (first := 9254) (initial := 3863) row_count_9255 (by decide)
private theorem atomic_9255_agreement : RowAgreement 9254 9255 := RowAgreement.single (first := 9254) modelrow_agrees_9255
private def atomic_9256 : CountSegment 1 10 13120 9255 9256 3863 3863 := CountSegment.single (first := 9255) (initial := 3863) row_count_9256 (by decide)
private theorem atomic_9256_agreement : RowAgreement 9255 9256 := RowAgreement.single (first := 9255) modelrow_agrees_9256
private def atomic_9257 : CountSegment 1 10 13120 9256 9257 3863 3863 := CountSegment.single (first := 9256) (initial := 3863) row_count_9257 (by decide)
private theorem atomic_9257_agreement : RowAgreement 9256 9257 := RowAgreement.single (first := 9256) modelrow_agrees_9257
private def atomic_9258 : CountSegment 1 10 13120 9257 9258 3863 3864 := CountSegment.single (first := 9257) (initial := 3863) row_count_9258 (by decide)
private theorem atomic_9258_agreement : RowAgreement 9257 9258 := RowAgreement.single (first := 9257) modelrow_agrees_9258
private def atomic_9259 : CountSegment 1 10 13120 9258 9259 3864 3864 := CountSegment.single (first := 9258) (initial := 3864) row_count_9259 (by decide)
private theorem atomic_9259_agreement : RowAgreement 9258 9259 := RowAgreement.single (first := 9258) modelrow_agrees_9259
private def atomic_9260 : CountSegment 1 10 13120 9259 9260 3864 3864 := CountSegment.single (first := 9259) (initial := 3864) row_count_9260 (by decide)
private theorem atomic_9260_agreement : RowAgreement 9259 9260 := RowAgreement.single (first := 9259) modelrow_agrees_9260
private def atomic_9261 : CountSegment 1 10 13120 9260 9261 3864 3864 := CountSegment.single (first := 9260) (initial := 3864) row_count_9261 (by decide)
private theorem atomic_9261_agreement : RowAgreement 9260 9261 := RowAgreement.single (first := 9260) modelrow_agrees_9261
private def atomic_9262 : CountSegment 1 10 13120 9261 9262 3864 3865 := CountSegment.single (first := 9261) (initial := 3864) row_count_9262 (by decide)
private theorem atomic_9262_agreement : RowAgreement 9261 9262 := RowAgreement.single (first := 9261) modelrow_agrees_9262
private def atomic_9263 : CountSegment 1 10 13120 9262 9263 3865 3866 := CountSegment.single (first := 9262) (initial := 3865) row_count_9263 (by decide)
private theorem atomic_9263_agreement : RowAgreement 9262 9263 := RowAgreement.single (first := 9262) modelrow_agrees_9263
private def atomic_9264 : CountSegment 1 10 13120 9263 9264 3866 3867 := CountSegment.single (first := 9263) (initial := 3866) row_count_9264 (by decide)
private theorem atomic_9264_agreement : RowAgreement 9263 9264 := RowAgreement.single (first := 9263) modelrow_agrees_9264
private def atomic_9265 : CountSegment 1 10 13120 9264 9265 3867 3867 := CountSegment.single (first := 9264) (initial := 3867) row_count_9265 (by decide)
private theorem atomic_9265_agreement : RowAgreement 9264 9265 := RowAgreement.single (first := 9264) modelrow_agrees_9265
private def atomic_9266 : CountSegment 1 10 13120 9265 9266 3867 3867 := CountSegment.single (first := 9265) (initial := 3867) row_count_9266 (by decide)
private theorem atomic_9266_agreement : RowAgreement 9265 9266 := RowAgreement.single (first := 9265) modelrow_agrees_9266
private def atomic_9267 : CountSegment 1 10 13120 9266 9267 3867 3867 := CountSegment.single (first := 9266) (initial := 3867) row_count_9267 (by decide)
private theorem atomic_9267_agreement : RowAgreement 9266 9267 := RowAgreement.single (first := 9266) modelrow_agrees_9267
private def atomic_9268 : CountSegment 1 10 13120 9267 9268 3867 3867 := CountSegment.single (first := 9267) (initial := 3867) row_count_9268 (by decide)
private theorem atomic_9268_agreement : RowAgreement 9267 9268 := RowAgreement.single (first := 9267) modelrow_agrees_9268
private def atomic_9269 : CountSegment 1 10 13120 9268 9269 3867 3868 := CountSegment.single (first := 9268) (initial := 3867) row_count_9269 (by decide)
private theorem atomic_9269_agreement : RowAgreement 9268 9269 := RowAgreement.single (first := 9268) modelrow_agrees_9269
private def atomic_9270 : CountSegment 1 10 13120 9269 9270 3868 3869 := CountSegment.single (first := 9269) (initial := 3868) row_count_9270 (by decide)
private theorem atomic_9270_agreement : RowAgreement 9269 9270 := RowAgreement.single (first := 9269) modelrow_agrees_9270
private def atomic_9271 : CountSegment 1 10 13120 9270 9271 3869 3870 := CountSegment.single (first := 9270) (initial := 3869) row_count_9271 (by decide)
private theorem atomic_9271_agreement : RowAgreement 9270 9271 := RowAgreement.single (first := 9270) modelrow_agrees_9271
private def atomic_9272 : CountSegment 1 10 13120 9271 9272 3870 3870 := CountSegment.single (first := 9271) (initial := 3870) row_count_9272 (by decide)
private theorem atomic_9272_agreement : RowAgreement 9271 9272 := RowAgreement.single (first := 9271) modelrow_agrees_9272
private def atomic_9273 : CountSegment 1 10 13120 9272 9273 3870 3871 := CountSegment.single (first := 9272) (initial := 3870) row_count_9273 (by decide)
private theorem atomic_9273_agreement : RowAgreement 9272 9273 := RowAgreement.single (first := 9272) modelrow_agrees_9273
private def atomic_9274 : CountSegment 1 10 13120 9273 9274 3871 3871 := CountSegment.single (first := 9273) (initial := 3871) row_count_9274 (by decide)
private theorem atomic_9274_agreement : RowAgreement 9273 9274 := RowAgreement.single (first := 9273) modelrow_agrees_9274
private def atomic_9275 : CountSegment 1 10 13120 9274 9275 3871 3872 := CountSegment.single (first := 9274) (initial := 3871) row_count_9275 (by decide)
private theorem atomic_9275_agreement : RowAgreement 9274 9275 := RowAgreement.single (first := 9274) modelrow_agrees_9275
private def atomic_9276 : CountSegment 1 10 13120 9275 9276 3872 3872 := CountSegment.single (first := 9275) (initial := 3872) row_count_9276 (by decide)
private theorem atomic_9276_agreement : RowAgreement 9275 9276 := RowAgreement.single (first := 9275) modelrow_agrees_9276
private def atomic_9277 : CountSegment 1 10 13120 9276 9277 3872 3872 := CountSegment.single (first := 9276) (initial := 3872) row_count_9277 (by decide)
private theorem atomic_9277_agreement : RowAgreement 9276 9277 := RowAgreement.single (first := 9276) modelrow_agrees_9277
private def atomic_9278 : CountSegment 1 10 13120 9277 9278 3872 3872 := CountSegment.single (first := 9277) (initial := 3872) row_count_9278 (by decide)
private theorem atomic_9278_agreement : RowAgreement 9277 9278 := RowAgreement.single (first := 9277) modelrow_agrees_9278
private def atomic_9279 : CountSegment 1 10 13120 9278 9279 3872 3873 := CountSegment.single (first := 9278) (initial := 3872) row_count_9279 (by decide)
private theorem atomic_9279_agreement : RowAgreement 9278 9279 := RowAgreement.single (first := 9278) modelrow_agrees_9279
private def atomic_9280 : CountSegment 1 10 13120 9279 9280 3873 3873 := CountSegment.single (first := 9279) (initial := 3873) row_count_9280 (by decide)
private theorem atomic_9280_agreement : RowAgreement 9279 9280 := RowAgreement.single (first := 9279) modelrow_agrees_9280
private def atomic_9281 : CountSegment 1 10 13120 9280 9281 3873 3873 := CountSegment.single (first := 9280) (initial := 3873) row_count_9281 (by decide)
private theorem atomic_9281_agreement : RowAgreement 9280 9281 := RowAgreement.single (first := 9280) modelrow_agrees_9281
private def atomic_9282 : CountSegment 1 10 13120 9281 9282 3873 3874 := CountSegment.single (first := 9281) (initial := 3873) row_count_9282 (by decide)
private theorem atomic_9282_agreement : RowAgreement 9281 9282 := RowAgreement.single (first := 9281) modelrow_agrees_9282
private def atomic_9283 : CountSegment 1 10 13120 9282 9283 3874 3875 := CountSegment.single (first := 9282) (initial := 3874) row_count_9283 (by decide)
private theorem atomic_9283_agreement : RowAgreement 9282 9283 := RowAgreement.single (first := 9282) modelrow_agrees_9283
private def atomic_9284 : CountSegment 1 10 13120 9283 9284 3875 3875 := CountSegment.single (first := 9283) (initial := 3875) row_count_9284 (by decide)
private theorem atomic_9284_agreement : RowAgreement 9283 9284 := RowAgreement.single (first := 9283) modelrow_agrees_9284
private def atomic_9285 : CountSegment 1 10 13120 9284 9285 3875 3875 := CountSegment.single (first := 9284) (initial := 3875) row_count_9285 (by decide)
private theorem atomic_9285_agreement : RowAgreement 9284 9285 := RowAgreement.single (first := 9284) modelrow_agrees_9285
private def atomic_9286 : CountSegment 1 10 13120 9285 9286 3875 3875 := CountSegment.single (first := 9285) (initial := 3875) row_count_9286 (by decide)
private theorem atomic_9286_agreement : RowAgreement 9285 9286 := RowAgreement.single (first := 9285) modelrow_agrees_9286
private def atomic_9287 : CountSegment 1 10 13120 9286 9287 3875 3875 := CountSegment.single (first := 9286) (initial := 3875) row_count_9287 (by decide)
private theorem atomic_9287_agreement : RowAgreement 9286 9287 := RowAgreement.single (first := 9286) modelrow_agrees_9287
private def atomic_9288 : CountSegment 1 10 13120 9287 9288 3875 3877 := CountSegment.single (first := 9287) (initial := 3875) row_count_9288 (by decide)
private theorem atomic_9288_agreement : RowAgreement 9287 9288 := RowAgreement.single (first := 9287) modelrow_agrees_9288
private def atomic_9289 : CountSegment 1 10 13120 9288 9289 3877 3877 := CountSegment.single (first := 9288) (initial := 3877) row_count_9289 (by decide)
private theorem atomic_9289_agreement : RowAgreement 9288 9289 := RowAgreement.single (first := 9288) modelrow_agrees_9289
private def atomic_9290 : CountSegment 1 10 13120 9289 9290 3877 3877 := CountSegment.single (first := 9289) (initial := 3877) row_count_9290 (by decide)
private theorem atomic_9290_agreement : RowAgreement 9289 9290 := RowAgreement.single (first := 9289) modelrow_agrees_9290
private def atomic_9291 : CountSegment 1 10 13120 9290 9291 3877 3878 := CountSegment.single (first := 9290) (initial := 3877) row_count_9291 (by decide)
private theorem atomic_9291_agreement : RowAgreement 9290 9291 := RowAgreement.single (first := 9290) modelrow_agrees_9291
private def atomic_9292 : CountSegment 1 10 13120 9291 9292 3878 3879 := CountSegment.single (first := 9291) (initial := 3878) row_count_9292 (by decide)
private theorem atomic_9292_agreement : RowAgreement 9291 9292 := RowAgreement.single (first := 9291) modelrow_agrees_9292
private def atomic_9293 : CountSegment 1 10 13120 9292 9293 3879 3880 := CountSegment.single (first := 9292) (initial := 3879) row_count_9293 (by decide)
private theorem atomic_9293_agreement : RowAgreement 9292 9293 := RowAgreement.single (first := 9292) modelrow_agrees_9293
private def atomic_9294 : CountSegment 1 10 13120 9293 9294 3880 3880 := CountSegment.single (first := 9293) (initial := 3880) row_count_9294 (by decide)
private theorem atomic_9294_agreement : RowAgreement 9293 9294 := RowAgreement.single (first := 9293) modelrow_agrees_9294
private def atomic_9295 : CountSegment 1 10 13120 9294 9295 3880 3881 := CountSegment.single (first := 9294) (initial := 3880) row_count_9295 (by decide)
private theorem atomic_9295_agreement : RowAgreement 9294 9295 := RowAgreement.single (first := 9294) modelrow_agrees_9295
private def atomic_9296 : CountSegment 1 10 13120 9295 9296 3881 3881 := CountSegment.single (first := 9295) (initial := 3881) row_count_9296 (by decide)
private theorem atomic_9296_agreement : RowAgreement 9295 9296 := RowAgreement.single (first := 9295) modelrow_agrees_9296
private def atomic_9297 : CountSegment 1 10 13120 9296 9297 3881 3881 := CountSegment.single (first := 9296) (initial := 3881) row_count_9297 (by decide)
private theorem atomic_9297_agreement : RowAgreement 9296 9297 := RowAgreement.single (first := 9296) modelrow_agrees_9297
private def atomic_9298 : CountSegment 1 10 13120 9297 9298 3881 3881 := CountSegment.single (first := 9297) (initial := 3881) row_count_9298 (by decide)
private theorem atomic_9298_agreement : RowAgreement 9297 9298 := RowAgreement.single (first := 9297) modelrow_agrees_9298
private def atomic_9299 : CountSegment 1 10 13120 9298 9299 3881 3882 := CountSegment.single (first := 9298) (initial := 3881) row_count_9299 (by decide)
private theorem atomic_9299_agreement : RowAgreement 9298 9299 := RowAgreement.single (first := 9298) modelrow_agrees_9299
private def atomic_9300 : CountSegment 1 10 13120 9299 9300 3882 3883 := CountSegment.single (first := 9299) (initial := 3882) row_count_9300 (by decide)
private theorem atomic_9300_agreement : RowAgreement 9299 9300 := RowAgreement.single (first := 9299) modelrow_agrees_9300
private def atomic_9301 : CountSegment 1 10 13120 9300 9301 3883 3883 := CountSegment.single (first := 9300) (initial := 3883) row_count_9301 (by decide)
private theorem atomic_9301_agreement : RowAgreement 9300 9301 := RowAgreement.single (first := 9300) modelrow_agrees_9301
private def atomic_9302 : CountSegment 1 10 13120 9301 9302 3883 3883 := CountSegment.single (first := 9301) (initial := 3883) row_count_9302 (by decide)
private theorem atomic_9302_agreement : RowAgreement 9301 9302 := RowAgreement.single (first := 9301) modelrow_agrees_9302
private def atomic_9303 : CountSegment 1 10 13120 9302 9303 3883 3883 := CountSegment.single (first := 9302) (initial := 3883) row_count_9303 (by decide)
private theorem atomic_9303_agreement : RowAgreement 9302 9303 := RowAgreement.single (first := 9302) modelrow_agrees_9303
private def atomic_9304 : CountSegment 1 10 13120 9303 9304 3883 3883 := CountSegment.single (first := 9303) (initial := 3883) row_count_9304 (by decide)
private theorem atomic_9304_agreement : RowAgreement 9303 9304 := RowAgreement.single (first := 9303) modelrow_agrees_9304
private def atomic_9305 : CountSegment 1 10 13120 9304 9305 3883 3884 := CountSegment.single (first := 9304) (initial := 3883) row_count_9305 (by decide)
private theorem atomic_9305_agreement : RowAgreement 9304 9305 := RowAgreement.single (first := 9304) modelrow_agrees_9305
private def atomic_9306 : CountSegment 1 10 13120 9305 9306 3884 3884 := CountSegment.single (first := 9305) (initial := 3884) row_count_9306 (by decide)
private theorem atomic_9306_agreement : RowAgreement 9305 9306 := RowAgreement.single (first := 9305) modelrow_agrees_9306
private def atomic_9307 : CountSegment 1 10 13120 9306 9307 3884 3885 := CountSegment.single (first := 9306) (initial := 3884) row_count_9307 (by decide)
private theorem atomic_9307_agreement : RowAgreement 9306 9307 := RowAgreement.single (first := 9306) modelrow_agrees_9307
private def atomic_9308 : CountSegment 1 10 13120 9307 9308 3885 3885 := CountSegment.single (first := 9307) (initial := 3885) row_count_9308 (by decide)
private theorem atomic_9308_agreement : RowAgreement 9307 9308 := RowAgreement.single (first := 9307) modelrow_agrees_9308
private def atomic_9309 : CountSegment 1 10 13120 9308 9309 3885 3885 := CountSegment.single (first := 9308) (initial := 3885) row_count_9309 (by decide)
private theorem atomic_9309_agreement : RowAgreement 9308 9309 := RowAgreement.single (first := 9308) modelrow_agrees_9309
private def atomic_9310 : CountSegment 1 10 13120 9309 9310 3885 3886 := CountSegment.single (first := 9309) (initial := 3885) row_count_9310 (by decide)
private theorem atomic_9310_agreement : RowAgreement 9309 9310 := RowAgreement.single (first := 9309) modelrow_agrees_9310
private def atomic_9311 : CountSegment 1 10 13120 9310 9311 3886 3886 := CountSegment.single (first := 9310) (initial := 3886) row_count_9311 (by decide)
private theorem atomic_9311_agreement : RowAgreement 9310 9311 := RowAgreement.single (first := 9310) modelrow_agrees_9311
private def atomic_9312 : CountSegment 1 10 13120 9311 9312 3886 3886 := CountSegment.single (first := 9311) (initial := 3886) row_count_9312 (by decide)
private theorem atomic_9312_agreement : RowAgreement 9311 9312 := RowAgreement.single (first := 9311) modelrow_agrees_9312
private def atomic_9313 : CountSegment 1 10 13120 9312 9313 3886 3887 := CountSegment.single (first := 9312) (initial := 3886) row_count_9313 (by decide)
private theorem atomic_9313_agreement : RowAgreement 9312 9313 := RowAgreement.single (first := 9312) modelrow_agrees_9313
private def atomic_9314 : CountSegment 1 10 13120 9313 9314 3887 3887 := CountSegment.single (first := 9313) (initial := 3887) row_count_9314 (by decide)
private theorem atomic_9314_agreement : RowAgreement 9313 9314 := RowAgreement.single (first := 9313) modelrow_agrees_9314
private def atomic_9315 : CountSegment 1 10 13120 9314 9315 3887 3887 := CountSegment.single (first := 9314) (initial := 3887) row_count_9315 (by decide)
private theorem atomic_9315_agreement : RowAgreement 9314 9315 := RowAgreement.single (first := 9314) modelrow_agrees_9315
private def atomic_9316 : CountSegment 1 10 13120 9315 9316 3887 3887 := CountSegment.single (first := 9315) (initial := 3887) row_count_9316 (by decide)
private theorem atomic_9316_agreement : RowAgreement 9315 9316 := RowAgreement.single (first := 9315) modelrow_agrees_9316
private def atomic_9317 : CountSegment 1 10 13120 9316 9317 3887 3887 := CountSegment.single (first := 9316) (initial := 3887) row_count_9317 (by decide)
private theorem atomic_9317_agreement : RowAgreement 9316 9317 := RowAgreement.single (first := 9316) modelrow_agrees_9317
private def atomic_9318 : CountSegment 1 10 13120 9317 9318 3887 3888 := CountSegment.single (first := 9317) (initial := 3887) row_count_9318 (by decide)
private theorem atomic_9318_agreement : RowAgreement 9317 9318 := RowAgreement.single (first := 9317) modelrow_agrees_9318
private def atomic_9319 : CountSegment 1 10 13120 9318 9319 3888 3888 := CountSegment.single (first := 9318) (initial := 3888) row_count_9319 (by decide)
private theorem atomic_9319_agreement : RowAgreement 9318 9319 := RowAgreement.single (first := 9318) modelrow_agrees_9319
private def atomic_9320 : CountSegment 1 10 13120 9319 9320 3888 3888 := CountSegment.single (first := 9319) (initial := 3888) row_count_9320 (by decide)
private theorem atomic_9320_agreement : RowAgreement 9319 9320 := RowAgreement.single (first := 9319) modelrow_agrees_9320
private def atomic_9321 : CountSegment 1 10 13120 9320 9321 3888 3889 := CountSegment.single (first := 9320) (initial := 3888) row_count_9321 (by decide)
private theorem atomic_9321_agreement : RowAgreement 9320 9321 := RowAgreement.single (first := 9320) modelrow_agrees_9321
private def atomic_9322 : CountSegment 1 10 13120 9321 9322 3889 3891 := CountSegment.single (first := 9321) (initial := 3889) row_count_9322 (by decide)
private theorem atomic_9322_agreement : RowAgreement 9321 9322 := RowAgreement.single (first := 9321) modelrow_agrees_9322
private def atomic_9323 : CountSegment 1 10 13120 9322 9323 3891 3891 := CountSegment.single (first := 9322) (initial := 3891) row_count_9323 (by decide)
private theorem atomic_9323_agreement : RowAgreement 9322 9323 := RowAgreement.single (first := 9322) modelrow_agrees_9323
private def atomic_9324 : CountSegment 1 10 13120 9323 9324 3891 3891 := CountSegment.single (first := 9323) (initial := 3891) row_count_9324 (by decide)
private theorem atomic_9324_agreement : RowAgreement 9323 9324 := RowAgreement.single (first := 9323) modelrow_agrees_9324
private def atomic_9325 : CountSegment 1 10 13120 9324 9325 3891 3891 := CountSegment.single (first := 9324) (initial := 3891) row_count_9325 (by decide)
private theorem atomic_9325_agreement : RowAgreement 9324 9325 := RowAgreement.single (first := 9324) modelrow_agrees_9325
private def atomic_9326 : CountSegment 1 10 13120 9325 9326 3891 3891 := CountSegment.single (first := 9325) (initial := 3891) row_count_9326 (by decide)
private theorem atomic_9326_agreement : RowAgreement 9325 9326 := RowAgreement.single (first := 9325) modelrow_agrees_9326
private def atomic_9327 : CountSegment 1 10 13120 9326 9327 3891 3893 := CountSegment.single (first := 9326) (initial := 3891) row_count_9327 (by decide)
private theorem atomic_9327_agreement : RowAgreement 9326 9327 := RowAgreement.single (first := 9326) modelrow_agrees_9327
private def atomic_9328 : CountSegment 1 10 13120 9327 9328 3893 3893 := CountSegment.single (first := 9327) (initial := 3893) row_count_9328 (by decide)
private theorem atomic_9328_agreement : RowAgreement 9327 9328 := RowAgreement.single (first := 9327) modelrow_agrees_9328
private def atomic_9329 : CountSegment 1 10 13120 9328 9329 3893 3893 := CountSegment.single (first := 9328) (initial := 3893) row_count_9329 (by decide)
private theorem atomic_9329_agreement : RowAgreement 9328 9329 := RowAgreement.single (first := 9328) modelrow_agrees_9329
private def atomic_9330 : CountSegment 1 10 13120 9329 9330 3893 3893 := CountSegment.single (first := 9329) (initial := 3893) row_count_9330 (by decide)
private theorem atomic_9330_agreement : RowAgreement 9329 9330 := RowAgreement.single (first := 9329) modelrow_agrees_9330
private def atomic_9331 : CountSegment 1 10 13120 9330 9331 3893 3893 := CountSegment.single (first := 9330) (initial := 3893) row_count_9331 (by decide)
private theorem atomic_9331_agreement : RowAgreement 9330 9331 := RowAgreement.single (first := 9330) modelrow_agrees_9331
private def atomic_9332 : CountSegment 1 10 13120 9331 9332 3893 3893 := CountSegment.single (first := 9331) (initial := 3893) row_count_9332 (by decide)
private theorem atomic_9332_agreement : RowAgreement 9331 9332 := RowAgreement.single (first := 9331) modelrow_agrees_9332
private def atomic_9333 : CountSegment 1 10 13120 9332 9333 3893 3894 := CountSegment.single (first := 9332) (initial := 3893) row_count_9333 (by decide)
private theorem atomic_9333_agreement : RowAgreement 9332 9333 := RowAgreement.single (first := 9332) modelrow_agrees_9333
private def atomic_9334 : CountSegment 1 10 13120 9333 9334 3894 3894 := CountSegment.single (first := 9333) (initial := 3894) row_count_9334 (by decide)
private theorem atomic_9334_agreement : RowAgreement 9333 9334 := RowAgreement.single (first := 9333) modelrow_agrees_9334
private def atomic_9335 : CountSegment 1 10 13120 9334 9335 3894 3894 := CountSegment.single (first := 9334) (initial := 3894) row_count_9335 (by decide)
private theorem atomic_9335_agreement : RowAgreement 9334 9335 := RowAgreement.single (first := 9334) modelrow_agrees_9335
private def atomic_9336 : CountSegment 1 10 13120 9335 9336 3894 3895 := CountSegment.single (first := 9335) (initial := 3894) row_count_9336 (by decide)
private theorem atomic_9336_agreement : RowAgreement 9335 9336 := RowAgreement.single (first := 9335) modelrow_agrees_9336
private def atomic_9337 : CountSegment 1 10 13120 9336 9337 3895 3896 := CountSegment.single (first := 9336) (initial := 3895) row_count_9337 (by decide)
private theorem atomic_9337_agreement : RowAgreement 9336 9337 := RowAgreement.single (first := 9336) modelrow_agrees_9337
private def atomic_9338 : CountSegment 1 10 13120 9337 9338 3896 3896 := CountSegment.single (first := 9337) (initial := 3896) row_count_9338 (by decide)
private theorem atomic_9338_agreement : RowAgreement 9337 9338 := RowAgreement.single (first := 9337) modelrow_agrees_9338
private def atomic_9339 : CountSegment 1 10 13120 9338 9339 3896 3896 := CountSegment.single (first := 9338) (initial := 3896) row_count_9339 (by decide)
private theorem atomic_9339_agreement : RowAgreement 9338 9339 := RowAgreement.single (first := 9338) modelrow_agrees_9339
private def atomic_9340 : CountSegment 1 10 13120 9339 9340 3896 3896 := CountSegment.single (first := 9339) (initial := 3896) row_count_9340 (by decide)
private theorem atomic_9340_agreement : RowAgreement 9339 9340 := RowAgreement.single (first := 9339) modelrow_agrees_9340
private def atomic_9341 : CountSegment 1 10 13120 9340 9341 3896 3896 := CountSegment.single (first := 9340) (initial := 3896) row_count_9341 (by decide)
private theorem atomic_9341_agreement : RowAgreement 9340 9341 := RowAgreement.single (first := 9340) modelrow_agrees_9341
private def atomic_9342 : CountSegment 1 10 13120 9341 9342 3896 3897 := CountSegment.single (first := 9341) (initial := 3896) row_count_9342 (by decide)
private theorem atomic_9342_agreement : RowAgreement 9341 9342 := RowAgreement.single (first := 9341) modelrow_agrees_9342
private def atomic_9343 : CountSegment 1 10 13120 9342 9343 3897 3897 := CountSegment.single (first := 9342) (initial := 3897) row_count_9343 (by decide)
private theorem atomic_9343_agreement : RowAgreement 9342 9343 := RowAgreement.single (first := 9342) modelrow_agrees_9343
private def atomic_9344 : CountSegment 1 10 13120 9343 9344 3897 3897 := CountSegment.single (first := 9343) (initial := 3897) row_count_9344 (by decide)
private theorem atomic_9344_agreement : RowAgreement 9343 9344 := RowAgreement.single (first := 9343) modelrow_agrees_9344
private def within_072_000 : CountSegment 1 10 13120 9216 9218 3845 3845 := CountSegment.join atomic_9217 atomic_9218
private theorem within_072_000_agreement : RowAgreement 9216 9218 := RowAgreement.join atomic_9217_agreement atomic_9218_agreement
private def within_072_001 : CountSegment 1 10 13120 9218 9220 3845 3846 := CountSegment.join atomic_9219 atomic_9220
private theorem within_072_001_agreement : RowAgreement 9218 9220 := RowAgreement.join atomic_9219_agreement atomic_9220_agreement
private def within_072_002 : CountSegment 1 10 13120 9216 9220 3845 3846 := CountSegment.join within_072_000 within_072_001
private theorem within_072_002_agreement : RowAgreement 9216 9220 := RowAgreement.join within_072_000_agreement within_072_001_agreement
private def within_072_003 : CountSegment 1 10 13120 9220 9222 3846 3847 := CountSegment.join atomic_9221 atomic_9222
private theorem within_072_003_agreement : RowAgreement 9220 9222 := RowAgreement.join atomic_9221_agreement atomic_9222_agreement
private def within_072_004 : CountSegment 1 10 13120 9222 9224 3847 3848 := CountSegment.join atomic_9223 atomic_9224
private theorem within_072_004_agreement : RowAgreement 9222 9224 := RowAgreement.join atomic_9223_agreement atomic_9224_agreement
private def within_072_005 : CountSegment 1 10 13120 9220 9224 3846 3848 := CountSegment.join within_072_003 within_072_004
private theorem within_072_005_agreement : RowAgreement 9220 9224 := RowAgreement.join within_072_003_agreement within_072_004_agreement
private def within_072_006 : CountSegment 1 10 13120 9216 9224 3845 3848 := CountSegment.join within_072_002 within_072_005
private theorem within_072_006_agreement : RowAgreement 9216 9224 := RowAgreement.join within_072_002_agreement within_072_005_agreement
private def within_072_007 : CountSegment 1 10 13120 9224 9226 3848 3849 := CountSegment.join atomic_9225 atomic_9226
private theorem within_072_007_agreement : RowAgreement 9224 9226 := RowAgreement.join atomic_9225_agreement atomic_9226_agreement
private def within_072_008 : CountSegment 1 10 13120 9226 9228 3849 3849 := CountSegment.join atomic_9227 atomic_9228
private theorem within_072_008_agreement : RowAgreement 9226 9228 := RowAgreement.join atomic_9227_agreement atomic_9228_agreement
private def within_072_009 : CountSegment 1 10 13120 9224 9228 3848 3849 := CountSegment.join within_072_007 within_072_008
private theorem within_072_009_agreement : RowAgreement 9224 9228 := RowAgreement.join within_072_007_agreement within_072_008_agreement
private def within_072_010 : CountSegment 1 10 13120 9228 9230 3849 3849 := CountSegment.join atomic_9229 atomic_9230
private theorem within_072_010_agreement : RowAgreement 9228 9230 := RowAgreement.join atomic_9229_agreement atomic_9230_agreement
private def within_072_011 : CountSegment 1 10 13120 9230 9232 3849 3851 := CountSegment.join atomic_9231 atomic_9232
private theorem within_072_011_agreement : RowAgreement 9230 9232 := RowAgreement.join atomic_9231_agreement atomic_9232_agreement
private def within_072_012 : CountSegment 1 10 13120 9228 9232 3849 3851 := CountSegment.join within_072_010 within_072_011
private theorem within_072_012_agreement : RowAgreement 9228 9232 := RowAgreement.join within_072_010_agreement within_072_011_agreement
private def within_072_013 : CountSegment 1 10 13120 9224 9232 3848 3851 := CountSegment.join within_072_009 within_072_012
private theorem within_072_013_agreement : RowAgreement 9224 9232 := RowAgreement.join within_072_009_agreement within_072_012_agreement
private def within_072_014 : CountSegment 1 10 13120 9216 9232 3845 3851 := CountSegment.join within_072_006 within_072_013
private theorem within_072_014_agreement : RowAgreement 9216 9232 := RowAgreement.join within_072_006_agreement within_072_013_agreement
private def within_072_015 : CountSegment 1 10 13120 9232 9234 3851 3852 := CountSegment.join atomic_9233 atomic_9234
private theorem within_072_015_agreement : RowAgreement 9232 9234 := RowAgreement.join atomic_9233_agreement atomic_9234_agreement
private def within_072_016 : CountSegment 1 10 13120 9234 9236 3852 3852 := CountSegment.join atomic_9235 atomic_9236
private theorem within_072_016_agreement : RowAgreement 9234 9236 := RowAgreement.join atomic_9235_agreement atomic_9236_agreement
private def within_072_017 : CountSegment 1 10 13120 9232 9236 3851 3852 := CountSegment.join within_072_015 within_072_016
private theorem within_072_017_agreement : RowAgreement 9232 9236 := RowAgreement.join within_072_015_agreement within_072_016_agreement
private def within_072_018 : CountSegment 1 10 13120 9236 9238 3852 3853 := CountSegment.join atomic_9237 atomic_9238
private theorem within_072_018_agreement : RowAgreement 9236 9238 := RowAgreement.join atomic_9237_agreement atomic_9238_agreement
private def within_072_019 : CountSegment 1 10 13120 9238 9240 3853 3854 := CountSegment.join atomic_9239 atomic_9240
private theorem within_072_019_agreement : RowAgreement 9238 9240 := RowAgreement.join atomic_9239_agreement atomic_9240_agreement
private def within_072_020 : CountSegment 1 10 13120 9236 9240 3852 3854 := CountSegment.join within_072_018 within_072_019
private theorem within_072_020_agreement : RowAgreement 9236 9240 := RowAgreement.join within_072_018_agreement within_072_019_agreement
private def within_072_021 : CountSegment 1 10 13120 9232 9240 3851 3854 := CountSegment.join within_072_017 within_072_020
private theorem within_072_021_agreement : RowAgreement 9232 9240 := RowAgreement.join within_072_017_agreement within_072_020_agreement
private def within_072_022 : CountSegment 1 10 13120 9240 9242 3854 3854 := CountSegment.join atomic_9241 atomic_9242
private theorem within_072_022_agreement : RowAgreement 9240 9242 := RowAgreement.join atomic_9241_agreement atomic_9242_agreement
private def within_072_023 : CountSegment 1 10 13120 9242 9244 3854 3858 := CountSegment.join atomic_9243 atomic_9244
private theorem within_072_023_agreement : RowAgreement 9242 9244 := RowAgreement.join atomic_9243_agreement atomic_9244_agreement
private def within_072_024 : CountSegment 1 10 13120 9240 9244 3854 3858 := CountSegment.join within_072_022 within_072_023
private theorem within_072_024_agreement : RowAgreement 9240 9244 := RowAgreement.join within_072_022_agreement within_072_023_agreement
private def within_072_025 : CountSegment 1 10 13120 9244 9246 3858 3859 := CountSegment.join atomic_9245 atomic_9246
private theorem within_072_025_agreement : RowAgreement 9244 9246 := RowAgreement.join atomic_9245_agreement atomic_9246_agreement
private def within_072_026 : CountSegment 1 10 13120 9246 9248 3859 3859 := CountSegment.join atomic_9247 atomic_9248
private theorem within_072_026_agreement : RowAgreement 9246 9248 := RowAgreement.join atomic_9247_agreement atomic_9248_agreement
private def within_072_027 : CountSegment 1 10 13120 9244 9248 3858 3859 := CountSegment.join within_072_025 within_072_026
private theorem within_072_027_agreement : RowAgreement 9244 9248 := RowAgreement.join within_072_025_agreement within_072_026_agreement
private def within_072_028 : CountSegment 1 10 13120 9240 9248 3854 3859 := CountSegment.join within_072_024 within_072_027
private theorem within_072_028_agreement : RowAgreement 9240 9248 := RowAgreement.join within_072_024_agreement within_072_027_agreement
private def within_072_029 : CountSegment 1 10 13120 9232 9248 3851 3859 := CountSegment.join within_072_021 within_072_028
private theorem within_072_029_agreement : RowAgreement 9232 9248 := RowAgreement.join within_072_021_agreement within_072_028_agreement
private def within_072_030 : CountSegment 1 10 13120 9216 9248 3845 3859 := CountSegment.join within_072_014 within_072_029
private theorem within_072_030_agreement : RowAgreement 9216 9248 := RowAgreement.join within_072_014_agreement within_072_029_agreement
private def within_072_031 : CountSegment 1 10 13120 9248 9250 3859 3861 := CountSegment.join atomic_9249 atomic_9250
private theorem within_072_031_agreement : RowAgreement 9248 9250 := RowAgreement.join atomic_9249_agreement atomic_9250_agreement
private def within_072_032 : CountSegment 1 10 13120 9250 9252 3861 3861 := CountSegment.join atomic_9251 atomic_9252
private theorem within_072_032_agreement : RowAgreement 9250 9252 := RowAgreement.join atomic_9251_agreement atomic_9252_agreement
private def within_072_033 : CountSegment 1 10 13120 9248 9252 3859 3861 := CountSegment.join within_072_031 within_072_032
private theorem within_072_033_agreement : RowAgreement 9248 9252 := RowAgreement.join within_072_031_agreement within_072_032_agreement
private def within_072_034 : CountSegment 1 10 13120 9252 9254 3861 3863 := CountSegment.join atomic_9253 atomic_9254
private theorem within_072_034_agreement : RowAgreement 9252 9254 := RowAgreement.join atomic_9253_agreement atomic_9254_agreement
private def within_072_035 : CountSegment 1 10 13120 9254 9256 3863 3863 := CountSegment.join atomic_9255 atomic_9256
private theorem within_072_035_agreement : RowAgreement 9254 9256 := RowAgreement.join atomic_9255_agreement atomic_9256_agreement
private def within_072_036 : CountSegment 1 10 13120 9252 9256 3861 3863 := CountSegment.join within_072_034 within_072_035
private theorem within_072_036_agreement : RowAgreement 9252 9256 := RowAgreement.join within_072_034_agreement within_072_035_agreement
private def within_072_037 : CountSegment 1 10 13120 9248 9256 3859 3863 := CountSegment.join within_072_033 within_072_036
private theorem within_072_037_agreement : RowAgreement 9248 9256 := RowAgreement.join within_072_033_agreement within_072_036_agreement
private def within_072_038 : CountSegment 1 10 13120 9256 9258 3863 3864 := CountSegment.join atomic_9257 atomic_9258
private theorem within_072_038_agreement : RowAgreement 9256 9258 := RowAgreement.join atomic_9257_agreement atomic_9258_agreement
private def within_072_039 : CountSegment 1 10 13120 9258 9260 3864 3864 := CountSegment.join atomic_9259 atomic_9260
private theorem within_072_039_agreement : RowAgreement 9258 9260 := RowAgreement.join atomic_9259_agreement atomic_9260_agreement
private def within_072_040 : CountSegment 1 10 13120 9256 9260 3863 3864 := CountSegment.join within_072_038 within_072_039
private theorem within_072_040_agreement : RowAgreement 9256 9260 := RowAgreement.join within_072_038_agreement within_072_039_agreement
private def within_072_041 : CountSegment 1 10 13120 9260 9262 3864 3865 := CountSegment.join atomic_9261 atomic_9262
private theorem within_072_041_agreement : RowAgreement 9260 9262 := RowAgreement.join atomic_9261_agreement atomic_9262_agreement
private def within_072_042 : CountSegment 1 10 13120 9262 9264 3865 3867 := CountSegment.join atomic_9263 atomic_9264
private theorem within_072_042_agreement : RowAgreement 9262 9264 := RowAgreement.join atomic_9263_agreement atomic_9264_agreement
private def within_072_043 : CountSegment 1 10 13120 9260 9264 3864 3867 := CountSegment.join within_072_041 within_072_042
private theorem within_072_043_agreement : RowAgreement 9260 9264 := RowAgreement.join within_072_041_agreement within_072_042_agreement
private def within_072_044 : CountSegment 1 10 13120 9256 9264 3863 3867 := CountSegment.join within_072_040 within_072_043
private theorem within_072_044_agreement : RowAgreement 9256 9264 := RowAgreement.join within_072_040_agreement within_072_043_agreement
private def within_072_045 : CountSegment 1 10 13120 9248 9264 3859 3867 := CountSegment.join within_072_037 within_072_044
private theorem within_072_045_agreement : RowAgreement 9248 9264 := RowAgreement.join within_072_037_agreement within_072_044_agreement
private def within_072_046 : CountSegment 1 10 13120 9264 9266 3867 3867 := CountSegment.join atomic_9265 atomic_9266
private theorem within_072_046_agreement : RowAgreement 9264 9266 := RowAgreement.join atomic_9265_agreement atomic_9266_agreement
private def within_072_047 : CountSegment 1 10 13120 9266 9268 3867 3867 := CountSegment.join atomic_9267 atomic_9268
private theorem within_072_047_agreement : RowAgreement 9266 9268 := RowAgreement.join atomic_9267_agreement atomic_9268_agreement
private def within_072_048 : CountSegment 1 10 13120 9264 9268 3867 3867 := CountSegment.join within_072_046 within_072_047
private theorem within_072_048_agreement : RowAgreement 9264 9268 := RowAgreement.join within_072_046_agreement within_072_047_agreement
private def within_072_049 : CountSegment 1 10 13120 9268 9270 3867 3869 := CountSegment.join atomic_9269 atomic_9270
private theorem within_072_049_agreement : RowAgreement 9268 9270 := RowAgreement.join atomic_9269_agreement atomic_9270_agreement
private def within_072_050 : CountSegment 1 10 13120 9270 9272 3869 3870 := CountSegment.join atomic_9271 atomic_9272
private theorem within_072_050_agreement : RowAgreement 9270 9272 := RowAgreement.join atomic_9271_agreement atomic_9272_agreement
private def within_072_051 : CountSegment 1 10 13120 9268 9272 3867 3870 := CountSegment.join within_072_049 within_072_050
private theorem within_072_051_agreement : RowAgreement 9268 9272 := RowAgreement.join within_072_049_agreement within_072_050_agreement
private def within_072_052 : CountSegment 1 10 13120 9264 9272 3867 3870 := CountSegment.join within_072_048 within_072_051
private theorem within_072_052_agreement : RowAgreement 9264 9272 := RowAgreement.join within_072_048_agreement within_072_051_agreement
private def within_072_053 : CountSegment 1 10 13120 9272 9274 3870 3871 := CountSegment.join atomic_9273 atomic_9274
private theorem within_072_053_agreement : RowAgreement 9272 9274 := RowAgreement.join atomic_9273_agreement atomic_9274_agreement
private def within_072_054 : CountSegment 1 10 13120 9274 9276 3871 3872 := CountSegment.join atomic_9275 atomic_9276
private theorem within_072_054_agreement : RowAgreement 9274 9276 := RowAgreement.join atomic_9275_agreement atomic_9276_agreement
private def within_072_055 : CountSegment 1 10 13120 9272 9276 3870 3872 := CountSegment.join within_072_053 within_072_054
private theorem within_072_055_agreement : RowAgreement 9272 9276 := RowAgreement.join within_072_053_agreement within_072_054_agreement
private def within_072_056 : CountSegment 1 10 13120 9276 9278 3872 3872 := CountSegment.join atomic_9277 atomic_9278
private theorem within_072_056_agreement : RowAgreement 9276 9278 := RowAgreement.join atomic_9277_agreement atomic_9278_agreement
private def within_072_057 : CountSegment 1 10 13120 9278 9280 3872 3873 := CountSegment.join atomic_9279 atomic_9280
private theorem within_072_057_agreement : RowAgreement 9278 9280 := RowAgreement.join atomic_9279_agreement atomic_9280_agreement
private def within_072_058 : CountSegment 1 10 13120 9276 9280 3872 3873 := CountSegment.join within_072_056 within_072_057
private theorem within_072_058_agreement : RowAgreement 9276 9280 := RowAgreement.join within_072_056_agreement within_072_057_agreement
private def within_072_059 : CountSegment 1 10 13120 9272 9280 3870 3873 := CountSegment.join within_072_055 within_072_058
private theorem within_072_059_agreement : RowAgreement 9272 9280 := RowAgreement.join within_072_055_agreement within_072_058_agreement
private def within_072_060 : CountSegment 1 10 13120 9264 9280 3867 3873 := CountSegment.join within_072_052 within_072_059
private theorem within_072_060_agreement : RowAgreement 9264 9280 := RowAgreement.join within_072_052_agreement within_072_059_agreement
private def within_072_061 : CountSegment 1 10 13120 9248 9280 3859 3873 := CountSegment.join within_072_045 within_072_060
private theorem within_072_061_agreement : RowAgreement 9248 9280 := RowAgreement.join within_072_045_agreement within_072_060_agreement
private def within_072_062 : CountSegment 1 10 13120 9216 9280 3845 3873 := CountSegment.join within_072_030 within_072_061
private theorem within_072_062_agreement : RowAgreement 9216 9280 := RowAgreement.join within_072_030_agreement within_072_061_agreement
private def within_072_063 : CountSegment 1 10 13120 9280 9282 3873 3874 := CountSegment.join atomic_9281 atomic_9282
private theorem within_072_063_agreement : RowAgreement 9280 9282 := RowAgreement.join atomic_9281_agreement atomic_9282_agreement
private def within_072_064 : CountSegment 1 10 13120 9282 9284 3874 3875 := CountSegment.join atomic_9283 atomic_9284
private theorem within_072_064_agreement : RowAgreement 9282 9284 := RowAgreement.join atomic_9283_agreement atomic_9284_agreement
private def within_072_065 : CountSegment 1 10 13120 9280 9284 3873 3875 := CountSegment.join within_072_063 within_072_064
private theorem within_072_065_agreement : RowAgreement 9280 9284 := RowAgreement.join within_072_063_agreement within_072_064_agreement
private def within_072_066 : CountSegment 1 10 13120 9284 9286 3875 3875 := CountSegment.join atomic_9285 atomic_9286
private theorem within_072_066_agreement : RowAgreement 9284 9286 := RowAgreement.join atomic_9285_agreement atomic_9286_agreement
private def within_072_067 : CountSegment 1 10 13120 9286 9288 3875 3877 := CountSegment.join atomic_9287 atomic_9288
private theorem within_072_067_agreement : RowAgreement 9286 9288 := RowAgreement.join atomic_9287_agreement atomic_9288_agreement
private def within_072_068 : CountSegment 1 10 13120 9284 9288 3875 3877 := CountSegment.join within_072_066 within_072_067
private theorem within_072_068_agreement : RowAgreement 9284 9288 := RowAgreement.join within_072_066_agreement within_072_067_agreement
private def within_072_069 : CountSegment 1 10 13120 9280 9288 3873 3877 := CountSegment.join within_072_065 within_072_068
private theorem within_072_069_agreement : RowAgreement 9280 9288 := RowAgreement.join within_072_065_agreement within_072_068_agreement
private def within_072_070 : CountSegment 1 10 13120 9288 9290 3877 3877 := CountSegment.join atomic_9289 atomic_9290
private theorem within_072_070_agreement : RowAgreement 9288 9290 := RowAgreement.join atomic_9289_agreement atomic_9290_agreement
private def within_072_071 : CountSegment 1 10 13120 9290 9292 3877 3879 := CountSegment.join atomic_9291 atomic_9292
private theorem within_072_071_agreement : RowAgreement 9290 9292 := RowAgreement.join atomic_9291_agreement atomic_9292_agreement
private def within_072_072 : CountSegment 1 10 13120 9288 9292 3877 3879 := CountSegment.join within_072_070 within_072_071
private theorem within_072_072_agreement : RowAgreement 9288 9292 := RowAgreement.join within_072_070_agreement within_072_071_agreement
private def within_072_073 : CountSegment 1 10 13120 9292 9294 3879 3880 := CountSegment.join atomic_9293 atomic_9294
private theorem within_072_073_agreement : RowAgreement 9292 9294 := RowAgreement.join atomic_9293_agreement atomic_9294_agreement
private def within_072_074 : CountSegment 1 10 13120 9294 9296 3880 3881 := CountSegment.join atomic_9295 atomic_9296
private theorem within_072_074_agreement : RowAgreement 9294 9296 := RowAgreement.join atomic_9295_agreement atomic_9296_agreement
private def within_072_075 : CountSegment 1 10 13120 9292 9296 3879 3881 := CountSegment.join within_072_073 within_072_074
private theorem within_072_075_agreement : RowAgreement 9292 9296 := RowAgreement.join within_072_073_agreement within_072_074_agreement
private def within_072_076 : CountSegment 1 10 13120 9288 9296 3877 3881 := CountSegment.join within_072_072 within_072_075
private theorem within_072_076_agreement : RowAgreement 9288 9296 := RowAgreement.join within_072_072_agreement within_072_075_agreement
private def within_072_077 : CountSegment 1 10 13120 9280 9296 3873 3881 := CountSegment.join within_072_069 within_072_076
private theorem within_072_077_agreement : RowAgreement 9280 9296 := RowAgreement.join within_072_069_agreement within_072_076_agreement
private def within_072_078 : CountSegment 1 10 13120 9296 9298 3881 3881 := CountSegment.join atomic_9297 atomic_9298
private theorem within_072_078_agreement : RowAgreement 9296 9298 := RowAgreement.join atomic_9297_agreement atomic_9298_agreement
private def within_072_079 : CountSegment 1 10 13120 9298 9300 3881 3883 := CountSegment.join atomic_9299 atomic_9300
private theorem within_072_079_agreement : RowAgreement 9298 9300 := RowAgreement.join atomic_9299_agreement atomic_9300_agreement
private def within_072_080 : CountSegment 1 10 13120 9296 9300 3881 3883 := CountSegment.join within_072_078 within_072_079
private theorem within_072_080_agreement : RowAgreement 9296 9300 := RowAgreement.join within_072_078_agreement within_072_079_agreement
private def within_072_081 : CountSegment 1 10 13120 9300 9302 3883 3883 := CountSegment.join atomic_9301 atomic_9302
private theorem within_072_081_agreement : RowAgreement 9300 9302 := RowAgreement.join atomic_9301_agreement atomic_9302_agreement
private def within_072_082 : CountSegment 1 10 13120 9302 9304 3883 3883 := CountSegment.join atomic_9303 atomic_9304
private theorem within_072_082_agreement : RowAgreement 9302 9304 := RowAgreement.join atomic_9303_agreement atomic_9304_agreement
private def within_072_083 : CountSegment 1 10 13120 9300 9304 3883 3883 := CountSegment.join within_072_081 within_072_082
private theorem within_072_083_agreement : RowAgreement 9300 9304 := RowAgreement.join within_072_081_agreement within_072_082_agreement
private def within_072_084 : CountSegment 1 10 13120 9296 9304 3881 3883 := CountSegment.join within_072_080 within_072_083
private theorem within_072_084_agreement : RowAgreement 9296 9304 := RowAgreement.join within_072_080_agreement within_072_083_agreement
private def within_072_085 : CountSegment 1 10 13120 9304 9306 3883 3884 := CountSegment.join atomic_9305 atomic_9306
private theorem within_072_085_agreement : RowAgreement 9304 9306 := RowAgreement.join atomic_9305_agreement atomic_9306_agreement
private def within_072_086 : CountSegment 1 10 13120 9306 9308 3884 3885 := CountSegment.join atomic_9307 atomic_9308
private theorem within_072_086_agreement : RowAgreement 9306 9308 := RowAgreement.join atomic_9307_agreement atomic_9308_agreement
private def within_072_087 : CountSegment 1 10 13120 9304 9308 3883 3885 := CountSegment.join within_072_085 within_072_086
private theorem within_072_087_agreement : RowAgreement 9304 9308 := RowAgreement.join within_072_085_agreement within_072_086_agreement
private def within_072_088 : CountSegment 1 10 13120 9308 9310 3885 3886 := CountSegment.join atomic_9309 atomic_9310
private theorem within_072_088_agreement : RowAgreement 9308 9310 := RowAgreement.join atomic_9309_agreement atomic_9310_agreement
private def within_072_089 : CountSegment 1 10 13120 9310 9312 3886 3886 := CountSegment.join atomic_9311 atomic_9312
private theorem within_072_089_agreement : RowAgreement 9310 9312 := RowAgreement.join atomic_9311_agreement atomic_9312_agreement
private def within_072_090 : CountSegment 1 10 13120 9308 9312 3885 3886 := CountSegment.join within_072_088 within_072_089
private theorem within_072_090_agreement : RowAgreement 9308 9312 := RowAgreement.join within_072_088_agreement within_072_089_agreement
private def within_072_091 : CountSegment 1 10 13120 9304 9312 3883 3886 := CountSegment.join within_072_087 within_072_090
private theorem within_072_091_agreement : RowAgreement 9304 9312 := RowAgreement.join within_072_087_agreement within_072_090_agreement
private def within_072_092 : CountSegment 1 10 13120 9296 9312 3881 3886 := CountSegment.join within_072_084 within_072_091
private theorem within_072_092_agreement : RowAgreement 9296 9312 := RowAgreement.join within_072_084_agreement within_072_091_agreement
private def within_072_093 : CountSegment 1 10 13120 9280 9312 3873 3886 := CountSegment.join within_072_077 within_072_092
private theorem within_072_093_agreement : RowAgreement 9280 9312 := RowAgreement.join within_072_077_agreement within_072_092_agreement
private def within_072_094 : CountSegment 1 10 13120 9312 9314 3886 3887 := CountSegment.join atomic_9313 atomic_9314
private theorem within_072_094_agreement : RowAgreement 9312 9314 := RowAgreement.join atomic_9313_agreement atomic_9314_agreement
private def within_072_095 : CountSegment 1 10 13120 9314 9316 3887 3887 := CountSegment.join atomic_9315 atomic_9316
private theorem within_072_095_agreement : RowAgreement 9314 9316 := RowAgreement.join atomic_9315_agreement atomic_9316_agreement
private def within_072_096 : CountSegment 1 10 13120 9312 9316 3886 3887 := CountSegment.join within_072_094 within_072_095
private theorem within_072_096_agreement : RowAgreement 9312 9316 := RowAgreement.join within_072_094_agreement within_072_095_agreement
private def within_072_097 : CountSegment 1 10 13120 9316 9318 3887 3888 := CountSegment.join atomic_9317 atomic_9318
private theorem within_072_097_agreement : RowAgreement 9316 9318 := RowAgreement.join atomic_9317_agreement atomic_9318_agreement
private def within_072_098 : CountSegment 1 10 13120 9318 9320 3888 3888 := CountSegment.join atomic_9319 atomic_9320
private theorem within_072_098_agreement : RowAgreement 9318 9320 := RowAgreement.join atomic_9319_agreement atomic_9320_agreement
private def within_072_099 : CountSegment 1 10 13120 9316 9320 3887 3888 := CountSegment.join within_072_097 within_072_098
private theorem within_072_099_agreement : RowAgreement 9316 9320 := RowAgreement.join within_072_097_agreement within_072_098_agreement
private def within_072_100 : CountSegment 1 10 13120 9312 9320 3886 3888 := CountSegment.join within_072_096 within_072_099
private theorem within_072_100_agreement : RowAgreement 9312 9320 := RowAgreement.join within_072_096_agreement within_072_099_agreement
private def within_072_101 : CountSegment 1 10 13120 9320 9322 3888 3891 := CountSegment.join atomic_9321 atomic_9322
private theorem within_072_101_agreement : RowAgreement 9320 9322 := RowAgreement.join atomic_9321_agreement atomic_9322_agreement
private def within_072_102 : CountSegment 1 10 13120 9322 9324 3891 3891 := CountSegment.join atomic_9323 atomic_9324
private theorem within_072_102_agreement : RowAgreement 9322 9324 := RowAgreement.join atomic_9323_agreement atomic_9324_agreement
private def within_072_103 : CountSegment 1 10 13120 9320 9324 3888 3891 := CountSegment.join within_072_101 within_072_102
private theorem within_072_103_agreement : RowAgreement 9320 9324 := RowAgreement.join within_072_101_agreement within_072_102_agreement
private def within_072_104 : CountSegment 1 10 13120 9324 9326 3891 3891 := CountSegment.join atomic_9325 atomic_9326
private theorem within_072_104_agreement : RowAgreement 9324 9326 := RowAgreement.join atomic_9325_agreement atomic_9326_agreement
private def within_072_105 : CountSegment 1 10 13120 9326 9328 3891 3893 := CountSegment.join atomic_9327 atomic_9328
private theorem within_072_105_agreement : RowAgreement 9326 9328 := RowAgreement.join atomic_9327_agreement atomic_9328_agreement
private def within_072_106 : CountSegment 1 10 13120 9324 9328 3891 3893 := CountSegment.join within_072_104 within_072_105
private theorem within_072_106_agreement : RowAgreement 9324 9328 := RowAgreement.join within_072_104_agreement within_072_105_agreement
private def within_072_107 : CountSegment 1 10 13120 9320 9328 3888 3893 := CountSegment.join within_072_103 within_072_106
private theorem within_072_107_agreement : RowAgreement 9320 9328 := RowAgreement.join within_072_103_agreement within_072_106_agreement
private def within_072_108 : CountSegment 1 10 13120 9312 9328 3886 3893 := CountSegment.join within_072_100 within_072_107
private theorem within_072_108_agreement : RowAgreement 9312 9328 := RowAgreement.join within_072_100_agreement within_072_107_agreement
private def within_072_109 : CountSegment 1 10 13120 9328 9330 3893 3893 := CountSegment.join atomic_9329 atomic_9330
private theorem within_072_109_agreement : RowAgreement 9328 9330 := RowAgreement.join atomic_9329_agreement atomic_9330_agreement
private def within_072_110 : CountSegment 1 10 13120 9330 9332 3893 3893 := CountSegment.join atomic_9331 atomic_9332
private theorem within_072_110_agreement : RowAgreement 9330 9332 := RowAgreement.join atomic_9331_agreement atomic_9332_agreement
private def within_072_111 : CountSegment 1 10 13120 9328 9332 3893 3893 := CountSegment.join within_072_109 within_072_110
private theorem within_072_111_agreement : RowAgreement 9328 9332 := RowAgreement.join within_072_109_agreement within_072_110_agreement
private def within_072_112 : CountSegment 1 10 13120 9332 9334 3893 3894 := CountSegment.join atomic_9333 atomic_9334
private theorem within_072_112_agreement : RowAgreement 9332 9334 := RowAgreement.join atomic_9333_agreement atomic_9334_agreement
private def within_072_113 : CountSegment 1 10 13120 9334 9336 3894 3895 := CountSegment.join atomic_9335 atomic_9336
private theorem within_072_113_agreement : RowAgreement 9334 9336 := RowAgreement.join atomic_9335_agreement atomic_9336_agreement
private def within_072_114 : CountSegment 1 10 13120 9332 9336 3893 3895 := CountSegment.join within_072_112 within_072_113
private theorem within_072_114_agreement : RowAgreement 9332 9336 := RowAgreement.join within_072_112_agreement within_072_113_agreement
private def within_072_115 : CountSegment 1 10 13120 9328 9336 3893 3895 := CountSegment.join within_072_111 within_072_114
private theorem within_072_115_agreement : RowAgreement 9328 9336 := RowAgreement.join within_072_111_agreement within_072_114_agreement
private def within_072_116 : CountSegment 1 10 13120 9336 9338 3895 3896 := CountSegment.join atomic_9337 atomic_9338
private theorem within_072_116_agreement : RowAgreement 9336 9338 := RowAgreement.join atomic_9337_agreement atomic_9338_agreement
private def within_072_117 : CountSegment 1 10 13120 9338 9340 3896 3896 := CountSegment.join atomic_9339 atomic_9340
private theorem within_072_117_agreement : RowAgreement 9338 9340 := RowAgreement.join atomic_9339_agreement atomic_9340_agreement
private def within_072_118 : CountSegment 1 10 13120 9336 9340 3895 3896 := CountSegment.join within_072_116 within_072_117
private theorem within_072_118_agreement : RowAgreement 9336 9340 := RowAgreement.join within_072_116_agreement within_072_117_agreement
private def within_072_119 : CountSegment 1 10 13120 9340 9342 3896 3897 := CountSegment.join atomic_9341 atomic_9342
private theorem within_072_119_agreement : RowAgreement 9340 9342 := RowAgreement.join atomic_9341_agreement atomic_9342_agreement
private def within_072_120 : CountSegment 1 10 13120 9342 9344 3897 3897 := CountSegment.join atomic_9343 atomic_9344
private theorem within_072_120_agreement : RowAgreement 9342 9344 := RowAgreement.join atomic_9343_agreement atomic_9344_agreement
private def within_072_121 : CountSegment 1 10 13120 9340 9344 3896 3897 := CountSegment.join within_072_119 within_072_120
private theorem within_072_121_agreement : RowAgreement 9340 9344 := RowAgreement.join within_072_119_agreement within_072_120_agreement
private def within_072_122 : CountSegment 1 10 13120 9336 9344 3895 3897 := CountSegment.join within_072_118 within_072_121
private theorem within_072_122_agreement : RowAgreement 9336 9344 := RowAgreement.join within_072_118_agreement within_072_121_agreement
private def within_072_123 : CountSegment 1 10 13120 9328 9344 3893 3897 := CountSegment.join within_072_115 within_072_122
private theorem within_072_123_agreement : RowAgreement 9328 9344 := RowAgreement.join within_072_115_agreement within_072_122_agreement
private def within_072_124 : CountSegment 1 10 13120 9312 9344 3886 3897 := CountSegment.join within_072_108 within_072_123
private theorem within_072_124_agreement : RowAgreement 9312 9344 := RowAgreement.join within_072_108_agreement within_072_123_agreement
private def within_072_125 : CountSegment 1 10 13120 9280 9344 3873 3897 := CountSegment.join within_072_093 within_072_124
private theorem within_072_125_agreement : RowAgreement 9280 9344 := RowAgreement.join within_072_093_agreement within_072_124_agreement
private def within_072_126 : CountSegment 1 10 13120 9216 9344 3845 3897 := CountSegment.join within_072_062 within_072_125
private theorem within_072_126_agreement : RowAgreement 9216 9344 := RowAgreement.join within_072_062_agreement within_072_125_agreement
def segment_072 : CountSegment 1 10 13120 9216 9344 3845 3897 := within_072_126
theorem agreement_072 : RowAgreement 9216 9344 := within_072_126_agreement
end ProjectEulerSolutions.P58Numerical
