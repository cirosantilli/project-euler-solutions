import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_8321 : RowCertificate 8321 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8321 1) := by change ¬ Nat.Prime 276972807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92324269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8321 2) := by change ¬ Nat.Prime 276956165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55391233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8321 3) := by change ¬ Nat.Prime 276939523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39562789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8321 : row 8321 = 0 := (row_8321.checked).trans rfl
theorem model_count_8321 : modelRow 8321 = 0 := by decide
theorem modelrow_agrees_8321 : modelRow 8321 = row 8321 := model_count_8321.trans row_count_8321.symm
def row_8322 : RowCertificate 8322 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8322 1) := by change ¬ Nat.Prime 277039381; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1149541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8322 2) := by change ¬ Nat.Prime 277022737; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7487101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8322 3) := by change ¬ Nat.Prime 277006093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39572299) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8322 : row 8322 = 0 := (row_8322.checked).trans rfl
theorem model_count_8322 : modelRow 8322 = 0 := by decide
theorem modelrow_agrees_8322 : modelRow 8322 = row 8322 := model_count_8322.trans row_count_8322.symm
def row_8323 : RowCertificate 8323 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8323 1) := by change ¬ Nat.Prime 277105963; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 598501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8323 2) := by change ¬ Nat.Prime 277089317; exact PrattCertificates.composite_of_product (factor := 16001) (cofactor := 17317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8323 3) := by change ¬ Nat.Prime 277072671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92357557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8323 : row 8323 = 0 := (row_8323.checked).trans rfl
theorem model_count_8323 : modelRow 8323 = 0 := by decide
theorem modelrow_agrees_8323 : modelRow 8323 = row 8323 := model_count_8323.trans row_count_8323.symm
def row_8324 : RowCertificate 8324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8324 1) := by change ¬ Nat.Prime 277172553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92390851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8324 2) := by change ¬ Nat.Prime 277155905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55431181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8324 3) := by change ¬ Nat.Prime 277139257; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6445099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8324 : row 8324 = 0 := (row_8324.checked).trans rfl
theorem model_count_8324 : modelRow 8324 = 0 := by decide
theorem modelrow_agrees_8324 : modelRow 8324 = row 8324 := model_count_8324.trans row_count_8324.symm
def row_8325 : RowCertificate 8325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8325 1) := by change ¬ Nat.Prime 277239151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39605593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8325 2) := by change Nat.Prime 277222501; exact ProjectEulerSolutions.P58PrimeNodes.p_277222501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8325 3) := by change ¬ Nat.Prime 277205851; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21323527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8325 : row 8325 = 1 := (row_8325.checked).trans rfl
theorem model_count_8325 : modelRow 8325 = 1 := by decide
theorem modelrow_agrees_8325 : modelRow 8325 = row 8325 := model_count_8325.trans row_count_8325.symm
def row_8326 : RowCertificate 8326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8326 1) := by change ¬ Nat.Prime 277305757; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8945347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8326 2) := by change ¬ Nat.Prime 277289105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55457821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8326 3) := by change ¬ Nat.Prime 277272453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92424151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8326 : row 8326 = 0 := (row_8326.checked).trans rfl
theorem model_count_8326 : modelRow 8326 = 0 := by decide
theorem modelrow_agrees_8326 : modelRow 8326 = row 8326 := model_count_8326.trans row_count_8326.symm
def row_8327 : RowCertificate 8327 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8327 1) := by change ¬ Nat.Prime 277372371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92457457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8327 2) := by change ¬ Nat.Prime 277355717; exact PrattCertificates.composite_of_product (factor := 1433) (cofactor := 193549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8327 3) := by change ¬ Nat.Prime 277339063; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4139389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8327 : row 8327 = 0 := (row_8327.checked).trans rfl
theorem model_count_8327 : modelRow 8327 = 0 := by decide
theorem modelrow_agrees_8327 : modelRow 8327 = row 8327 := model_count_8327.trans row_count_8327.symm
def row_8328 : RowCertificate 8328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8328 1) := by change ¬ Nat.Prime 277438993; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21341461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8328 2) := by change ¬ Nat.Prime 277422337; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16318961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8328 3) := by change ¬ Nat.Prime 277405681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39629383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8328 : row 8328 = 0 := (row_8328.checked).trans rfl
theorem model_count_8328 : modelRow 8328 = 0 := by decide
theorem modelrow_agrees_8328 : modelRow 8328 = row 8328 := model_count_8328.trans row_count_8328.symm
def row_8329 : RowCertificate 8329 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8329 1) := by change Nat.Prime 277505623; exact ProjectEulerSolutions.P58PrimeNodes.p_277505623.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8329 2) := by change ¬ Nat.Prime 277488965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55497793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8329 3) := by change ¬ Nat.Prime 277472307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92490769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8329 : row 8329 = 1 := (row_8329.checked).trans rfl
theorem model_count_8329 : modelRow 8329 = 1 := by decide
theorem modelrow_agrees_8329 : modelRow 8329 = row 8329 := model_count_8329.trans row_count_8329.symm
def row_8330 : RowCertificate 8330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8330 1) := by change ¬ Nat.Prime 277572261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92524087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8330 2) := by change ¬ Nat.Prime 277555601; exact PrattCertificates.composite_of_product (factor := 2129) (cofactor := 130369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8330 3) := by change ¬ Nat.Prime 277538941; exact PrattCertificates.composite_of_product (factor := 1447) (cofactor := 191803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8330 : row 8330 = 0 := (row_8330.checked).trans rfl
theorem model_count_8330 : modelRow 8330 = 0 := by decide
theorem modelrow_agrees_8330 : modelRow 8330 = row 8330 := model_count_8330.trans row_count_8330.symm
def row_8331 : RowCertificate 8331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8331 1) := by change ¬ Nat.Prime 277638907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39662701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8331 2) := by change ¬ Nat.Prime 277622245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55524449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8331 3) := by change Nat.Prime 277605583; exact ProjectEulerSolutions.P58PrimeNodes.p_277605583.prime
    exact decide_eq_true prime
theorem row_count_8331 : row 8331 = 1 := (row_8331.checked).trans rfl
theorem model_count_8331 : modelRow 8331 = 1 := by decide
theorem modelrow_agrees_8331 : modelRow 8331 = row 8331 := model_count_8331.trans row_count_8331.symm
def row_8332 : RowCertificate 8332 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8332 1) := by change ¬ Nat.Prime 277705561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39672223) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8332 2) := by change ¬ Nat.Prime 277688897; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16334641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8332 3) := by change ¬ Nat.Prime 277672233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92557411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8332 : row 8332 = 0 := (row_8332.checked).trans rfl
theorem model_count_8332 : modelRow 8332 = 0 := by decide
theorem modelrow_agrees_8332 : modelRow 8332 = row 8332 := model_count_8332.trans row_count_8332.symm
def row_8333 : RowCertificate 8333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8333 1) := by change ¬ Nat.Prime 277772223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92590741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8333 2) := by change ¬ Nat.Prime 277755557; exact PrattCertificates.composite_of_product (factor := 10429) (cofactor := 26633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8333 3) := by change Nat.Prime 277738891; exact ProjectEulerSolutions.P58PrimeNodes.p_277738891.prime
    exact decide_eq_true prime
theorem row_count_8333 : row 8333 = 1 := (row_8333.checked).trans rfl
theorem model_count_8333 : modelRow 8333 = 1 := by decide
theorem modelrow_agrees_8333 : modelRow 8333 = row 8333 := model_count_8333.trans row_count_8333.symm
def row_8334 : RowCertificate 8334 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8334 1) := by change ¬ Nat.Prime 277838893; exact PrattCertificates.composite_of_product (factor := 571) (cofactor := 486583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8334 2) := by change ¬ Nat.Prime 277822225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55564445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8334 3) := by change Nat.Prime 277805557; exact ProjectEulerSolutions.P58PrimeNodes.p_277805557.prime
    exact decide_eq_true prime
theorem row_count_8334 : row 8334 = 1 := (row_8334.checked).trans rfl
theorem model_count_8334 : modelRow 8334 = 1 := by decide
theorem modelrow_agrees_8334 : modelRow 8334 = row 8334 := model_count_8334.trans row_count_8334.symm
def row_8335 : RowCertificate 8335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8335 1) := by change ¬ Nat.Prime 277905571; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14626609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8335 2) := by change ¬ Nat.Prime 277888901; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1769993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8335 3) := by change ¬ Nat.Prime 277872231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92624077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8335 : row 8335 = 0 := (row_8335.checked).trans rfl
theorem model_count_8335 : modelRow 8335 = 0 := by decide
theorem modelrow_agrees_8335 : modelRow 8335 = row 8335 := model_count_8335.trans row_count_8335.symm
def row_8336 : RowCertificate 8336 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8336 1) := by change ¬ Nat.Prime 277972257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92657419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8336 2) := by change ¬ Nat.Prime 277955585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55591117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8336 3) := by change ¬ Nat.Prime 277938913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39705559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8336 : row 8336 = 0 := (row_8336.checked).trans rfl
theorem model_count_8336 : modelRow 8336 = 0 := by decide
theorem modelrow_agrees_8336 : modelRow 8336 = row 8336 := model_count_8336.trans row_count_8336.symm
def row_8337 : RowCertificate 8337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8337 1) := by change ¬ Nat.Prime 278038951; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14633629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8337 2) := by change ¬ Nat.Prime 278022277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21386329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8337 3) := by change ¬ Nat.Prime 278005603; exact PrattCertificates.composite_of_product (factor := 4243) (cofactor := 65521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8337 : row 8337 = 0 := (row_8337.checked).trans rfl
theorem model_count_8337 : modelRow 8337 = 0 := by decide
theorem modelrow_agrees_8337 : modelRow 8337 = row 8337 := model_count_8337.trans row_count_8337.symm
def row_8338 : RowCertificate 8338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8338 1) := by change ¬ Nat.Prime 278105653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39729379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8338 2) := by change Nat.Prime 278088977; exact ProjectEulerSolutions.P58PrimeNodes.p_278088977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8338 3) := by change ¬ Nat.Prime 278072301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92690767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8338 : row 8338 = 1 := (row_8338.checked).trans rfl
theorem model_count_8338 : modelRow 8338 = 1 := by decide
theorem modelrow_agrees_8338 : modelRow 8338 = row 8338 := model_count_8338.trans row_count_8338.symm
def row_8339 : RowCertificate 8339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8339 1) := by change ¬ Nat.Prime 278172363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92724121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8339 2) := by change ¬ Nat.Prime 278155685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55631137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8339 3) := by change ¬ Nat.Prime 278139007; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6468349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8339 : row 8339 = 0 := (row_8339.checked).trans rfl
theorem model_count_8339 : modelRow 8339 = 0 := by decide
theorem modelrow_agrees_8339 : modelRow 8339 = row 8339 := model_count_8339.trans row_count_8339.symm
def row_8340 : RowCertificate 8340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8340 1) := by change ¬ Nat.Prime 278239081; exact PrattCertificates.composite_of_product (factor := 4327) (cofactor := 64303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8340 2) := by change ¬ Nat.Prime 278222401; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1004413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8340 3) := by change Nat.Prime 278205721; exact ProjectEulerSolutions.P58PrimeNodes.p_278205721.prime
    exact decide_eq_true prime
theorem row_count_8340 : row 8340 = 1 := (row_8340.checked).trans rfl
theorem model_count_8340 : modelRow 8340 = 1 := by decide
theorem modelrow_agrees_8340 : modelRow 8340 = row 8340 := model_count_8340.trans row_count_8340.symm
def row_8341 : RowCertificate 8341 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8341 1) := by change ¬ Nat.Prime 278305807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21408139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8341 2) := by change ¬ Nat.Prime 278289125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55657825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8341 3) := by change ¬ Nat.Prime 278272443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92757481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8341 : row 8341 = 0 := (row_8341.checked).trans rfl
theorem model_count_8341 : modelRow 8341 = 0 := by decide
theorem modelrow_agrees_8341 : modelRow 8341 = row 8341 := model_count_8341.trans row_count_8341.symm
def row_8342 : RowCertificate 8342 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8342 1) := by change ¬ Nat.Prime 278372541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92790847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8342 2) := by change ¬ Nat.Prime 278355857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21411989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8342 3) := by change ¬ Nat.Prime 278339173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39762739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8342 : row 8342 = 0 := (row_8342.checked).trans rfl
theorem model_count_8342 : modelRow 8342 = 0 := by decide
theorem modelrow_agrees_8342 : modelRow 8342 = row 8342 := model_count_8342.trans row_count_8342.symm
def row_8343 : RowCertificate 8343 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8343 1) := by change Nat.Prime 278439283; exact ProjectEulerSolutions.P58PrimeNodes.p_278439283.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8343 2) := by change ¬ Nat.Prime 278422597; exact PrattCertificates.composite_of_product (factor := 1637) (cofactor := 170081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8343 3) := by change ¬ Nat.Prime 278405911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39772273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8343 : row 8343 = 1 := (row_8343.checked).trans rfl
theorem model_count_8343 : modelRow 8343 = 1 := by decide
theorem modelrow_agrees_8343 : modelRow 8343 = row 8343 := model_count_8343.trans row_count_8343.symm
def row_8344 : RowCertificate 8344 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8344 1) := by change ¬ Nat.Prime 278506033; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21423541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8344 2) := by change ¬ Nat.Prime 278489345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55697869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8344 3) := by change ¬ Nat.Prime 278472657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92824219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8344 : row 8344 = 0 := (row_8344.checked).trans rfl
theorem model_count_8344 : modelRow 8344 = 0 := by decide
theorem modelrow_agrees_8344 : modelRow 8344 = row 8344 := model_count_8344.trans row_count_8344.symm
def row_8345 : RowCertificate 8345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8345 1) := by change ¬ Nat.Prime 278572791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92857597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8345 2) := by change ¬ Nat.Prime 278556101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16385653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8345 3) := by change ¬ Nat.Prime 278539411; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14659969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8345 : row 8345 = 0 := (row_8345.checked).trans rfl
theorem model_count_8345 : modelRow 8345 = 0 := by decide
theorem modelrow_agrees_8345 : modelRow 8345 = row 8345 := model_count_8345.trans row_count_8345.symm
def row_8346 : RowCertificate 8346 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8346 1) := by change ¬ Nat.Prime 278639557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39805651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8346 2) := by change ¬ Nat.Prime 278622865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55724573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8346 3) := by change Nat.Prime 278606173; exact ProjectEulerSolutions.P58PrimeNodes.p_278606173.prime
    exact decide_eq_true prime
theorem row_count_8346 : row 8346 = 1 := (row_8346.checked).trans rfl
theorem model_count_8346 : modelRow 8346 = 1 := by decide
theorem modelrow_agrees_8346 : modelRow 8346 = row 8346 := model_count_8346.trans row_count_8346.symm
def row_8347 : RowCertificate 8347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8347 1) := by change ¬ Nat.Prime 278706331; exact PrattCertificates.composite_of_product (factor := 2161) (cofactor := 128971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8347 2) := by change ¬ Nat.Prime 278689637; exact PrattCertificates.composite_of_product (factor := 7577) (cofactor := 36781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8347 3) := by change ¬ Nat.Prime 278672943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92890981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8347 : row 8347 = 0 := (row_8347.checked).trans rfl
theorem model_count_8347 : modelRow 8347 = 0 := by decide
theorem modelrow_agrees_8347 : modelRow 8347 = row 8347 := model_count_8347.trans row_count_8347.symm
def row_8348 : RowCertificate 8348 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8348 1) := by change ¬ Nat.Prime 278773113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92924371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8348 2) := by change ¬ Nat.Prime 278756417; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6798937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8348 3) := by change ¬ Nat.Prime 278739721; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21441517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8348 : row 8348 = 0 := (row_8348.checked).trans rfl
theorem model_count_8348 : modelRow 8348 = 0 := by decide
theorem modelrow_agrees_8348 : modelRow 8348 = row 8348 := model_count_8348.trans row_count_8348.symm
def row_8349 : RowCertificate 8349 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8349 1) := by change Nat.Prime 278839903; exact ProjectEulerSolutions.P58PrimeNodes.p_278839903.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8349 2) := by change ¬ Nat.Prime 278823205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55764641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8349 3) := by change ¬ Nat.Prime 278806507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39829501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8349 : row 8349 = 1 := (row_8349.checked).trans rfl
theorem model_count_8349 : modelRow 8349 = 1 := by decide
theorem modelrow_agrees_8349 : modelRow 8349 = row 8349 := model_count_8349.trans row_count_8349.symm
def row_8350 : RowCertificate 8350 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8350 1) := by change ¬ Nat.Prime 278906701; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4572241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8350 2) := by change ¬ Nat.Prime 278890001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21453077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8350 3) := by change ¬ Nat.Prime 278873301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92957767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8350 : row 8350 = 0 := (row_8350.checked).trans rfl
theorem model_count_8350 : modelRow 8350 = 0 := by decide
theorem modelrow_agrees_8350 : modelRow 8350 = row 8350 := model_count_8350.trans row_count_8350.symm
def row_8351 : RowCertificate 8351 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8351 1) := by change ¬ Nat.Prime 278973507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92991169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8351 2) := by change ¬ Nat.Prime 278956805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55791361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8351 3) := by change ¬ Nat.Prime 278940103; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21456931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8351 : row 8351 = 0 := (row_8351.checked).trans rfl
theorem model_count_8351 : modelRow 8351 = 0 := by decide
theorem modelrow_agrees_8351 : modelRow 8351 = row 8351 := model_count_8351.trans row_count_8351.symm
def row_8352 : RowCertificate 8352 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8352 1) := by change ¬ Nat.Prime 279040321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39862903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8352 2) := by change Nat.Prime 279023617; exact ProjectEulerSolutions.P58PrimeNodes.p_279023617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8352 3) := by change ¬ Nat.Prime 279006913; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9000223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8352 : row 8352 = 1 := (row_8352.checked).trans rfl
theorem model_count_8352 : modelRow 8352 = 1 := by decide
theorem modelrow_agrees_8352 : modelRow 8352 = row 8352 := model_count_8352.trans row_count_8352.symm
def row_8353 : RowCertificate 8353 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8353 1) := by change ¬ Nat.Prime 279107143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39872449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8353 2) := by change ¬ Nat.Prime 279090437; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2877221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8353 3) := by change ¬ Nat.Prime 279073731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93024577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8353 : row 8353 = 0 := (row_8353.checked).trans rfl
theorem model_count_8353 : modelRow 8353 = 0 := by decide
theorem modelrow_agrees_8353 : modelRow 8353 = row 8353 := model_count_8353.trans row_count_8353.symm
def row_8354 : RowCertificate 8354 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8354 1) := by change ¬ Nat.Prime 279173973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93057991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8354 2) := by change ¬ Nat.Prime 279157265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55831453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8354 3) := by change ¬ Nat.Prime 279140557; exact PrattCertificates.composite_of_product (factor := 16369) (cofactor := 17053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8354 : row 8354 = 0 := (row_8354.checked).trans rfl
theorem model_count_8354 : modelRow 8354 = 0 := by decide
theorem modelrow_agrees_8354 : modelRow 8354 = row 8354 := model_count_8354.trans row_count_8354.symm
def row_8355 : RowCertificate 8355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8355 1) := by change ¬ Nat.Prime 279240811; exact PrattCertificates.composite_of_product (factor := 907) (cofactor := 307873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8355 2) := by change ¬ Nat.Prime 279224101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21478777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8355 3) := by change ¬ Nat.Prime 279207391; exact PrattCertificates.composite_of_product (factor := 9739) (cofactor := 28669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8355 : row 8355 = 0 := (row_8355.checked).trans rfl
theorem model_count_8355 : modelRow 8355 = 0 := by decide
theorem modelrow_agrees_8355 : modelRow 8355 = row 8355 := model_count_8355.trans row_count_8355.symm
def row_8356 : RowCertificate 8356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8356 1) := by change ¬ Nat.Prime 279307657; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14700403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8356 2) := by change ¬ Nat.Prime 279290945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55858189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8356 3) := by change ¬ Nat.Prime 279274233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93091411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8356 : row 8356 = 0 := (row_8356.checked).trans rfl
theorem model_count_8356 : modelRow 8356 = 0 := by decide
theorem modelrow_agrees_8356 : modelRow 8356 = row 8356 := model_count_8356.trans row_count_8356.symm
def row_8357 : RowCertificate 8357 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8357 1) := by change ¬ Nat.Prime 279374511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93124837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8357 2) := by change ¬ Nat.Prime 279357797; exact PrattCertificates.composite_of_product (factor := 1277) (cofactor := 218761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8357 3) := by change ¬ Nat.Prime 279341083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39905869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8357 : row 8357 = 0 := (row_8357.checked).trans rfl
theorem model_count_8357 : modelRow 8357 = 0 := by decide
theorem modelrow_agrees_8357 : modelRow 8357 = row 8357 := model_count_8357.trans row_count_8357.symm
def row_8358 : RowCertificate 8358 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8358 1) := by change Nat.Prime 279441373; exact ProjectEulerSolutions.P58PrimeNodes.p_279441373.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8358 2) := by change ¬ Nat.Prime 279424657; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9635333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8358 3) := by change ¬ Nat.Prime 279407941; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 683149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8358 : row 8358 = 1 := (row_8358.checked).trans rfl
theorem model_count_8358 : modelRow 8358 = 1 := by decide
theorem modelrow_agrees_8358 : modelRow 8358 = row 8358 := model_count_8358.trans row_count_8358.symm
def row_8359 : RowCertificate 8359 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8359 1) := by change ¬ Nat.Prime 279508243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39929749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8359 2) := by change ¬ Nat.Prime 279491525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55898305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8359 3) := by change ¬ Nat.Prime 279474807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93158269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8359 : row 8359 = 0 := (row_8359.checked).trans rfl
theorem model_count_8359 : modelRow 8359 = 0 := by decide
theorem modelrow_agrees_8359 : modelRow 8359 = row 8359 := model_count_8359.trans row_count_8359.symm
def row_8360 : RowCertificate 8360 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8360 1) := by change ¬ Nat.Prime 279575121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93191707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8360 2) := by change Nat.Prime 279558401; exact ProjectEulerSolutions.P58PrimeNodes.p_279558401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8360 3) := by change ¬ Nat.Prime 279541681; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2881873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8360 : row 8360 = 1 := (row_8360.checked).trans rfl
theorem model_count_8360 : modelRow 8360 = 1 := by decide
theorem modelrow_agrees_8360 : modelRow 8360 = row 8360 := model_count_8360.trans row_count_8360.symm
def row_8361 : RowCertificate 8361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8361 1) := by change ¬ Nat.Prime 279642007; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2565523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8361 2) := by change ¬ Nat.Prime 279625285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55925057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8361 3) := by change ¬ Nat.Prime 279608563; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21508351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8361 : row 8361 = 0 := (row_8361.checked).trans rfl
theorem model_count_8361 : modelRow 8361 = 0 := by decide
theorem modelrow_agrees_8361 : modelRow 8361 = row 8361 := model_count_8361.trans row_count_8361.symm
def row_8362 : RowCertificate 8362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8362 1) := by change ¬ Nat.Prime 279708901; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3540619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8362 2) := by change ¬ Nat.Prime 279692177; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16452481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8362 3) := by change ¬ Nat.Prime 279675453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93225151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8362 : row 8362 = 0 := (row_8362.checked).trans rfl
theorem model_count_8362 : modelRow 8362 = 0 := by decide
theorem modelrow_agrees_8362 : modelRow 8362 = row 8362 := model_count_8362.trans row_count_8362.symm
def row_8363 : RowCertificate 8363 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8363 1) := by change ¬ Nat.Prime 279775803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93258601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8363 2) := by change ¬ Nat.Prime 279759077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21519929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8363 3) := by change ¬ Nat.Prime 279742351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39963193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8363 : row 8363 = 0 := (row_8363.checked).trans rfl
theorem model_count_8363 : modelRow 8363 = 0 := by decide
theorem modelrow_agrees_8363 : modelRow 8363 = row 8363 := model_count_8363.trans row_count_8363.symm
def row_8364 : RowCertificate 8364 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8364 1) := by change ¬ Nat.Prime 279842713; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1853263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8364 2) := by change ¬ Nat.Prime 279825985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55965197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8364 3) := by change ¬ Nat.Prime 279809257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39972751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8364 : row 8364 = 0 := (row_8364.checked).trans rfl
theorem model_count_8364 : modelRow 8364 = 0 := by decide
theorem modelrow_agrees_8364 : modelRow 8364 = row 8364 := model_count_8364.trans row_count_8364.symm
def row_8365 : RowCertificate 8365 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8365 1) := by change ¬ Nat.Prime 279909631; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1717237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8365 2) := by change ¬ Nat.Prime 279892901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7564673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8365 3) := by change ¬ Nat.Prime 279876171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93292057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8365 : row 8365 = 0 := (row_8365.checked).trans rfl
theorem model_count_8365 : modelRow 8365 = 0 := by decide
theorem modelrow_agrees_8365 : modelRow 8365 = row 8365 := model_count_8365.trans row_count_8365.symm
def row_8366 : RowCertificate 8366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8366 1) := by change ¬ Nat.Prime 279976557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93325519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8366 2) := by change ¬ Nat.Prime 279959825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55991965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8366 3) := by change ¬ Nat.Prime 279943093; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14733847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8366 : row 8366 = 0 := (row_8366.checked).trans rfl
theorem model_count_8366 : modelRow 8366 = 0 := by decide
theorem modelrow_agrees_8366 : modelRow 8366 = row 8366 := model_count_8366.trans row_count_8366.symm
def row_8367 : RowCertificate 8367 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8367 1) := by change ¬ Nat.Prime 280043491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40006213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8367 2) := by change Nat.Prime 280026757; exact ProjectEulerSolutions.P58PrimeNodes.p_280026757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8367 3) := by change ¬ Nat.Prime 280010023; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6511861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8367 : row 8367 = 1 := (row_8367.checked).trans rfl
theorem model_count_8367 : modelRow 8367 = 1 := by decide
theorem modelrow_agrees_8367 : modelRow 8367 = row 8367 := model_count_8367.trans row_count_8367.symm
def row_8368 : RowCertificate 8368 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8368 1) := by change ¬ Nat.Prime 280110433; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1011229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8368 2) := by change ¬ Nat.Prime 280093697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21545669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8368 3) := by change ¬ Nat.Prime 280076961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93358987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8368 : row 8368 = 0 := (row_8368.checked).trans rfl
theorem model_count_8368 : modelRow 8368 = 0 := by decide
theorem modelrow_agrees_8368 : modelRow 8368 = row 8368 := model_count_8368.trans row_count_8368.symm
def row_8369 : RowCertificate 8369 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8369 1) := by change ¬ Nat.Prime 280177383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93392461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8369 2) := by change ¬ Nat.Prime 280160645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56032129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8369 3) := by change ¬ Nat.Prime 280143907; exact PrattCertificates.composite_of_product (factor := 8581) (cofactor := 32647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8369 : row 8369 = 0 := (row_8369.checked).trans rfl
theorem model_count_8369 : modelRow 8369 = 0 := by decide
theorem modelrow_agrees_8369 : modelRow 8369 = row 8369 := model_count_8369.trans row_count_8369.symm
def row_8370 : RowCertificate 8370 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8370 1) := by change ¬ Nat.Prime 280244341; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21557257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8370 2) := by change Nat.Prime 280227601; exact ProjectEulerSolutions.P58PrimeNodes.p_280227601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8370 3) := by change ¬ Nat.Prime 280210861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40030123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8370 : row 8370 = 1 := (row_8370.checked).trans rfl
theorem model_count_8370 : modelRow 8370 = 1 := by decide
theorem modelrow_agrees_8370 : modelRow 8370 = row 8370 := model_count_8370.trans row_count_8370.symm
def row_8371 : RowCertificate 8371 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8371 1) := by change ¬ Nat.Prime 280311307; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2721469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8371 2) := by change ¬ Nat.Prime 280294565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56058913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8371 3) := by change ¬ Nat.Prime 280277823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93425941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8371 : row 8371 = 0 := (row_8371.checked).trans rfl
theorem model_count_8371 : modelRow 8371 = 0 := by decide
theorem modelrow_agrees_8371 : modelRow 8371 = row 8371 := model_count_8371.trans row_count_8371.symm
def row_8372 : RowCertificate 8372 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8372 1) := by change ¬ Nat.Prime 280378281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93459427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8372 2) := by change ¬ Nat.Prime 280361537; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3840569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8372 3) := by change ¬ Nat.Prime 280344793; exact PrattCertificates.composite_of_product (factor := 4801) (cofactor := 58393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8372 : row 8372 = 0 := (row_8372.checked).trans rfl
theorem model_count_8372 : modelRow 8372 = 0 := by decide
theorem modelrow_agrees_8372 : modelRow 8372 = row 8372 := model_count_8372.trans row_count_8372.symm
def row_8373 : RowCertificate 8373 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8373 1) := by change ¬ Nat.Prime 280445263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40063609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8373 2) := by change Nat.Prime 280428517; exact ProjectEulerSolutions.P58PrimeNodes.p_280428517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8373 3) := by change ¬ Nat.Prime 280411771; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9045541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8373 : row 8373 = 1 := (row_8373.checked).trans rfl
theorem model_count_8373 : modelRow 8373 = 1 := by decide
theorem modelrow_agrees_8373 : modelRow 8373 = row 8373 := model_count_8373.trans row_count_8373.symm
def row_8374 : RowCertificate 8374 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8374 1) := by change ¬ Nat.Prime 280512253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40073179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8374 2) := by change ¬ Nat.Prime 280495505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56099101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8374 3) := by change ¬ Nat.Prime 280478757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93492919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8374 : row 8374 = 0 := (row_8374.checked).trans rfl
theorem model_count_8374 : modelRow 8374 = 0 := by decide
theorem modelrow_agrees_8374 : modelRow 8374 = row 8374 := model_count_8374.trans row_count_8374.symm
def row_8375 : RowCertificate 8375 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8375 1) := by change ¬ Nat.Prime 280579251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93526417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8375 2) := by change ¬ Nat.Prime 280562501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9674569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8375 3) := by change Nat.Prime 280545751; exact ProjectEulerSolutions.P58PrimeNodes.p_280545751.prime
    exact decide_eq_true prime
theorem row_count_8375 : row 8375 = 1 := (row_8375.checked).trans rfl
theorem model_count_8375 : modelRow 8375 = 1 := by decide
theorem modelrow_agrees_8375 : modelRow 8375 = row 8375 := model_count_8375.trans row_count_8375.symm
def row_8376 : RowCertificate 8376 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8376 1) := by change Nat.Prime 280646257; exact ProjectEulerSolutions.P58PrimeNodes.p_280646257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8376 2) := by change ¬ Nat.Prime 280629505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56125901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8376 3) := by change Nat.Prime 280612753; exact ProjectEulerSolutions.P58PrimeNodes.p_280612753.prime
    exact decide_eq_true prime
theorem row_count_8376 : row 8376 = 2 := (row_8376.checked).trans rfl
theorem model_count_8376 : modelRow 8376 = 2 := by decide
theorem modelrow_agrees_8376 : modelRow 8376 = row 8376 := model_count_8376.trans row_count_8376.symm
def row_8377 : RowCertificate 8377 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8377 1) := by change Nat.Prime 280713271; exact ProjectEulerSolutions.P58PrimeNodes.p_280713271.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8377 2) := by change Nat.Prime 280696517; exact ProjectEulerSolutions.P58PrimeNodes.p_280696517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8377 3) := by change ¬ Nat.Prime 280679763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93559921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8377 : row 8377 = 2 := (row_8377.checked).trans rfl
theorem model_count_8377 : modelRow 8377 = 2 := by decide
theorem modelrow_agrees_8377 : modelRow 8377 = row 8377 := model_count_8377.trans row_count_8377.symm
def row_8378 : RowCertificate 8378 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8378 1) := by change ¬ Nat.Prime 280780293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93593431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8378 2) := by change ¬ Nat.Prime 280763537; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 2779837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8378 3) := by change ¬ Nat.Prime 280746781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40106683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8378 : row 8378 = 0 := (row_8378.checked).trans rfl
theorem model_count_8378 : modelRow 8378 = 0 := by decide
theorem modelrow_agrees_8378 : modelRow 8378 = row 8378 := model_count_8378.trans row_count_8378.symm
def row_8379 : RowCertificate 8379 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8379 1) := by change Nat.Prime 280847323; exact ProjectEulerSolutions.P58PrimeNodes.p_280847323.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8379 2) := by change ¬ Nat.Prime 280830565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56166113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8379 3) := by change ¬ Nat.Prime 280813807; exact PrattCertificates.composite_of_product (factor := 13879) (cofactor := 20233) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8379 : row 8379 = 1 := (row_8379.checked).trans rfl
theorem model_count_8379 : modelRow 8379 = 1 := by decide
theorem modelrow_agrees_8379 : modelRow 8379 = row 8379 := model_count_8379.trans row_count_8379.symm
def row_8380 : RowCertificate 8380 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8380 1) := by change ¬ Nat.Prime 280914361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40130623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8380 2) := by change ¬ Nat.Prime 280897601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6851161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8380 3) := by change ¬ Nat.Prime 280880841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93626947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8380 : row 8380 = 0 := (row_8380.checked).trans rfl
theorem model_count_8380 : modelRow 8380 = 0 := by decide
theorem modelrow_agrees_8380 : modelRow 8380 = row 8380 := model_count_8380.trans row_count_8380.symm
def row_8381 : RowCertificate 8381 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8381 1) := by change ¬ Nat.Prime 280981407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93660469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8381 2) := by change ¬ Nat.Prime 280964645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56192929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8381 3) := by change ¬ Nat.Prime 280947883; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4605703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8381 : row 8381 = 0 := (row_8381.checked).trans rfl
theorem model_count_8381 : modelRow 8381 = 0 := by decide
theorem modelrow_agrees_8381 : modelRow 8381 = row 8381 := model_count_8381.trans row_count_8381.symm
def row_8382 : RowCertificate 8382 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8382 1) := by change ¬ Nat.Prime 281048461; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1260307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8382 2) := by change ¬ Nat.Prime 281031697; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4607077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8382 3) := by change ¬ Nat.Prime 281014933; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6535231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8382 : row 8382 = 0 := (row_8382.checked).trans rfl
theorem model_count_8382 : modelRow 8382 = 0 := by decide
theorem modelrow_agrees_8382 : modelRow 8382 = row 8382 := model_count_8382.trans row_count_8382.symm
def row_8383 : RowCertificate 8383 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8383 1) := by change ¬ Nat.Prime 281115523; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21624271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8383 2) := by change ¬ Nat.Prime 281098757; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16535221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8383 3) := by change ¬ Nat.Prime 281081991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93693997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8383 : row 8383 = 0 := (row_8383.checked).trans rfl
theorem model_count_8383 : modelRow 8383 = 0 := by decide
theorem modelrow_agrees_8383 : modelRow 8383 = row 8383 := model_count_8383.trans row_count_8383.symm
def row_8384 : RowCertificate 8384 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8384 1) := by change ¬ Nat.Prime 281182593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93727531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8384 2) := by change ¬ Nat.Prime 281165825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56233165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8384 3) := by change ¬ Nat.Prime 281149057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40164151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8384 : row 8384 = 0 := (row_8384.checked).trans rfl
theorem model_count_8384 : modelRow 8384 = 0 := by decide
theorem modelrow_agrees_8384 : modelRow 8384 = row 8384 := model_count_8384.trans row_count_8384.symm
def row_8385 : RowCertificate 8385 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8385 1) := by change Nat.Prime 281249671; exact ProjectEulerSolutions.P58PrimeNodes.p_281249671.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8385 2) := by change Nat.Prime 281232901; exact ProjectEulerSolutions.P58PrimeNodes.p_281232901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8385 3) := by change ¬ Nat.Prime 281216131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40173733) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8385 : row 8385 = 2 := (row_8385.checked).trans rfl
theorem model_count_8385 : modelRow 8385 = 2 := by decide
theorem modelrow_agrees_8385 : modelRow 8385 = row 8385 := model_count_8385.trans row_count_8385.symm
def row_8386 : RowCertificate 8386 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8386 1) := by change Nat.Prime 281316757; exact ProjectEulerSolutions.P58PrimeNodes.p_281316757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8386 2) := by change ¬ Nat.Prime 281299985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56259997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8386 3) := by change ¬ Nat.Prime 281283213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93761071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8386 : row 8386 = 1 := (row_8386.checked).trans rfl
theorem model_count_8386 : modelRow 8386 = 1 := by decide
theorem modelrow_agrees_8386 : modelRow 8386 = row 8386 := model_count_8386.trans row_count_8386.symm
def row_8387 : RowCertificate 8387 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8387 1) := by change ¬ Nat.Prime 281383851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93794617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8387 2) := by change ¬ Nat.Prime 281367077; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9702313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8387 3) := by change ¬ Nat.Prime 281350303; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21642331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8387 : row 8387 = 0 := (row_8387.checked).trans rfl
theorem model_count_8387 : modelRow 8387 = 0 := by decide
theorem modelrow_agrees_8387 : modelRow 8387 = row 8387 := model_count_8387.trans row_count_8387.symm
def row_8388 : RowCertificate 8388 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8388 1) := by change ¬ Nat.Prime 281450953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40207279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8388 2) := by change ¬ Nat.Prime 281434177; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 2786477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8388 3) := by change ¬ Nat.Prime 281417401; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 615793) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8388 : row 8388 = 0 := (row_8388.checked).trans rfl
theorem model_count_8388 : modelRow 8388 = 0 := by decide
theorem modelrow_agrees_8388 : modelRow 8388 = row 8388 := model_count_8388.trans row_count_8388.symm
def row_8389 : RowCertificate 8389 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8389 1) := by change ¬ Nat.Prime 281518063; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 279007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8389 2) := by change ¬ Nat.Prime 281501285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56300257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8389 3) := by change ¬ Nat.Prime 281484507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93828169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8389 : row 8389 = 0 := (row_8389.checked).trans rfl
theorem model_count_8389 : modelRow 8389 = 0 := by decide
theorem modelrow_agrees_8389 : modelRow 8389 = row 8389 := model_count_8389.trans row_count_8389.symm
def row_8390 : RowCertificate 8390 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8390 1) := by change ¬ Nat.Prime 281585181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93861727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8390 2) := by change Nat.Prime 281568401; exact ProjectEulerSolutions.P58PrimeNodes.p_281568401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8390 3) := by change ¬ Nat.Prime 281551621; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21657817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8390 : row 8390 = 1 := (row_8390.checked).trans rfl
theorem model_count_8390 : modelRow 8390 = 1 := by decide
theorem modelrow_agrees_8390 : modelRow 8390 = row 8390 := model_count_8390.trans row_count_8390.symm
def row_8391 : RowCertificate 8391 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8391 1) := by change ¬ Nat.Prime 281652307; exact PrattCertificates.composite_of_product (factor := 4153) (cofactor := 67819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8391 2) := by change ¬ Nat.Prime 281635525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56327105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8391 3) := by change ¬ Nat.Prime 281618743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40231249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8391 : row 8391 = 0 := (row_8391.checked).trans rfl
theorem model_count_8391 : modelRow 8391 = 0 := by decide
theorem modelrow_agrees_8391 : modelRow 8391 = row 8391 := model_count_8391.trans row_count_8391.symm
def row_8392 : RowCertificate 8392 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8392 1) := by change ¬ Nat.Prime 281719441; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14827339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8392 2) := by change Nat.Prime 281702657; exact ProjectEulerSolutions.P58PrimeNodes.p_281702657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8392 3) := by change ¬ Nat.Prime 281685873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93895291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8392 : row 8392 = 1 := (row_8392.checked).trans rfl
theorem model_count_8392 : modelRow 8392 = 1 := by decide
theorem modelrow_agrees_8392 : modelRow 8392 = row 8392 := model_count_8392.trans row_count_8392.symm
def row_8393 : RowCertificate 8393 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8393 1) := by change ¬ Nat.Prime 281786583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93928861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8393 2) := by change ¬ Nat.Prime 281769797; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4619177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8393 3) := by change ¬ Nat.Prime 281753011; exact PrattCertificates.composite_of_product (factor := 607) (cofactor := 464173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8393 : row 8393 = 0 := (row_8393.checked).trans rfl
theorem model_count_8393 : modelRow 8393 = 0 := by decide
theorem modelrow_agrees_8393 : modelRow 8393 = row 8393 := model_count_8393.trans row_count_8393.symm
def row_8394 : RowCertificate 8394 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8394 1) := by change ¬ Nat.Prime 281853733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40264819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8394 2) := by change ¬ Nat.Prime 281836945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56367389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8394 3) := by change ¬ Nat.Prime 281820157; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7616761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8394 : row 8394 = 0 := (row_8394.checked).trans rfl
theorem model_count_8394 : modelRow 8394 = 0 := by decide
theorem modelrow_agrees_8394 : modelRow 8394 = row 8394 := model_count_8394.trans row_count_8394.symm
def row_8395 : RowCertificate 8395 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8395 1) := by change ¬ Nat.Prime 281920891; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40274413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8395 2) := by change Nat.Prime 281904101; exact ProjectEulerSolutions.P58PrimeNodes.p_281904101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8395 3) := by change ¬ Nat.Prime 281887311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93962437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8395 : row 8395 = 1 := (row_8395.checked).trans rfl
theorem model_count_8395 : modelRow 8395 = 1 := by decide
theorem modelrow_agrees_8395 : modelRow 8395 = row 8395 := model_count_8395.trans row_count_8395.symm
def row_8396 : RowCertificate 8396 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8396 1) := by change ¬ Nat.Prime 281988057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93996019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8396 2) := by change ¬ Nat.Prime 281971265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56394253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8396 3) := by change ¬ Nat.Prime 281954473; exact PrattCertificates.composite_of_product (factor := 2857) (cofactor := 98689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8396 : row 8396 = 0 := (row_8396.checked).trans rfl
theorem model_count_8396 : modelRow 8396 = 0 := by decide
theorem modelrow_agrees_8396 : modelRow 8396 = row 8396 := model_count_8396.trans row_count_8396.symm
def row_8397 : RowCertificate 8397 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8397 1) := by change ¬ Nat.Prime 282055231; exact PrattCertificates.composite_of_product (factor := 2143) (cofactor := 131617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8397 2) := by change Nat.Prime 282038437; exact ProjectEulerSolutions.P58PrimeNodes.p_282038437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8397 3) := by change ¬ Nat.Prime 282021643; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1867693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8397 : row 8397 = 1 := (row_8397.checked).trans rfl
theorem model_count_8397 : modelRow 8397 = 1 := by decide
theorem modelrow_agrees_8397 : modelRow 8397 = row 8397 := model_count_8397.trans row_count_8397.symm
def row_8398 : RowCertificate 8398 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8398 1) := by change ¬ Nat.Prime 282122413; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9100723) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8398 2) := by change ¬ Nat.Prime 282105617; exact PrattCertificates.composite_of_product (factor := 1097) (cofactor := 257161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8398 3) := by change ¬ Nat.Prime 282088821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94029607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8398 : row 8398 = 0 := (row_8398.checked).trans rfl
theorem model_count_8398 : modelRow 8398 = 0 := by decide
theorem modelrow_agrees_8398 : modelRow 8398 = row 8398 := model_count_8398.trans row_count_8398.symm
def row_8399 : RowCertificate 8399 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8399 1) := by change ¬ Nat.Prime 282189603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94063201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8399 2) := by change ¬ Nat.Prime 282172805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56434561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8399 3) := by change ¬ Nat.Prime 282156007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40308001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8399 : row 8399 = 0 := (row_8399.checked).trans rfl
theorem model_count_8399 : modelRow 8399 = 0 := by decide
theorem modelrow_agrees_8399 : modelRow 8399 = row 8399 := model_count_8399.trans row_count_8399.symm
def row_8400 : RowCertificate 8400 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8400 1) := by change ¬ Nat.Prime 282256801; exact PrattCertificates.composite_of_product (factor := 1999) (cofactor := 141199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8400 2) := by change ¬ Nat.Prime 282240001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16602353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8400 3) := by change ¬ Nat.Prime 282223201; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21709477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8400 : row 8400 = 0 := (row_8400.checked).trans rfl
theorem model_count_8400 : modelRow 8400 = 0 := by decide
theorem modelrow_agrees_8400 : modelRow 8400 = row 8400 := model_count_8400.trans row_count_8400.symm
def row_8401 : RowCertificate 8401 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8401 1) := by change ¬ Nat.Prime 282324007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40332001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8401 2) := by change ¬ Nat.Prime 282307205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56461441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8401 3) := by change ¬ Nat.Prime 282290403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94096801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8401 : row 8401 = 0 := (row_8401.checked).trans rfl
theorem model_count_8401 : modelRow 8401 = 0 := by decide
theorem modelrow_agrees_8401 : modelRow 8401 = row 8401 := model_count_8401.trans row_count_8401.symm
def row_8402 : RowCertificate 8402 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8402 1) := by change ¬ Nat.Prime 282391221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94130407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8402 2) := by change ¬ Nat.Prime 282374417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21721109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8402 3) := by change ¬ Nat.Prime 282357613; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14860927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8402 : row 8402 = 0 := (row_8402.checked).trans rfl
theorem model_count_8402 : modelRow 8402 = 0 := by decide
theorem modelrow_agrees_8402 : modelRow 8402 = row 8402 := model_count_8402.trans row_count_8402.symm
def row_8403 : RowCertificate 8403 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8403 1) := by change ¬ Nat.Prime 282458443; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6568801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8403 2) := by change ¬ Nat.Prime 282441637; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1171957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8403 3) := by change ¬ Nat.Prime 282424831; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21724987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8403 : row 8403 = 0 := (row_8403.checked).trans rfl
theorem model_count_8403 : modelRow 8403 = 0 := by decide
theorem modelrow_agrees_8403 : modelRow 8403 = row 8403 := model_count_8403.trans row_count_8403.symm
def row_8404 : RowCertificate 8404 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8404 1) := by change ¬ Nat.Prime 282525673; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7635829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8404 2) := by change ¬ Nat.Prime 282508865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56501773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8404 3) := by change ¬ Nat.Prime 282492057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94164019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8404 : row 8404 = 0 := (row_8404.checked).trans rfl
theorem model_count_8404 : modelRow 8404 = 0 := by decide
theorem modelrow_agrees_8404 : modelRow 8404 = row 8404 := model_count_8404.trans row_count_8404.symm
def row_8405 : RowCertificate 8405 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8405 1) := by change ¬ Nat.Prime 282592911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94197637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8405 2) := by change ¬ Nat.Prime 282576101; exact PrattCertificates.composite_of_product (factor := 1861) (cofactor := 151841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8405 3) := by change ¬ Nat.Prime 282559291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40365613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8405 : row 8405 = 0 := (row_8405.checked).trans rfl
theorem model_count_8405 : modelRow 8405 = 0 := by decide
theorem modelrow_agrees_8405 : modelRow 8405 = row 8405 := model_count_8405.trans row_count_8405.symm
def row_8406 : RowCertificate 8406 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8406 1) := by change ¬ Nat.Prime 282660157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21743089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8406 2) := by change ¬ Nat.Prime 282643345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56528669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8406 3) := by change ¬ Nat.Prime 282626533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40375219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8406 : row 8406 = 0 := (row_8406.checked).trans rfl
theorem model_count_8406 : modelRow 8406 = 0 := by decide
theorem modelrow_agrees_8406 : modelRow 8406 = row 8406 := model_count_8406.trans row_count_8406.symm
def row_8407 : RowCertificate 8407 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8407 1) := by change ¬ Nat.Prime 282727411; exact PrattCertificates.composite_of_product (factor := 1171) (cofactor := 241441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8407 2) := by change ¬ Nat.Prime 282710597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21746969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8407 3) := by change ¬ Nat.Prime 282693783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94231261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8407 : row 8407 = 0 := (row_8407.checked).trans rfl
theorem model_count_8407 : modelRow 8407 = 0 := by decide
theorem modelrow_agrees_8407 : modelRow 8407 = row 8407 := model_count_8407.trans row_count_8407.symm
def row_8408 : RowCertificate 8408 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8408 1) := by change ¬ Nat.Prime 282794673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94264891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8408 2) := by change Nat.Prime 282777857; exact ProjectEulerSolutions.P58PrimeNodes.p_282777857.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8408 3) := by change Nat.Prime 282761041; exact ProjectEulerSolutions.P58PrimeNodes.p_282761041.prime
    exact decide_eq_true prime
theorem row_count_8408 : row 8408 = 2 := (row_8408.checked).trans rfl
theorem model_count_8408 : modelRow 8408 = 2 := by decide
theorem modelrow_agrees_8408 : modelRow 8408 = row 8408 := model_count_8408.trans row_count_8408.symm
def row_8409 : RowCertificate 8409 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8409 1) := by change ¬ Nat.Prime 282861943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40408849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8409 2) := by change ¬ Nat.Prime 282845125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56569025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8409 3) := by change Nat.Prime 282828307; exact ProjectEulerSolutions.P58PrimeNodes.p_282828307.prime
    exact decide_eq_true prime
theorem row_count_8409 : row 8409 = 1 := (row_8409.checked).trans rfl
theorem model_count_8409 : modelRow 8409 = 1 := by decide
theorem modelrow_agrees_8409 : modelRow 8409 = row 8409 := model_count_8409.trans row_count_8409.symm
def row_8410 : RowCertificate 8410 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8410 1) := by change ¬ Nat.Prime 282929221; exact PrattCertificates.composite_of_product (factor := 12979) (cofactor := 21799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8410 2) := by change ¬ Nat.Prime 282912401; exact PrattCertificates.composite_of_product (factor := 641) (cofactor := 441361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8410 3) := by change ¬ Nat.Prime 282895581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94298527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8410 : row 8410 = 0 := (row_8410.checked).trans rfl
theorem model_count_8410 : modelRow 8410 = 0 := by decide
theorem modelrow_agrees_8410 : modelRow 8410 = row 8410 := model_count_8410.trans row_count_8410.symm
def row_8411 : RowCertificate 8411 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8411 1) := by change ¬ Nat.Prime 282996507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94332169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8411 2) := by change ¬ Nat.Prime 282979685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56595937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8411 3) := by change ¬ Nat.Prime 282962863; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 428083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8411 : row 8411 = 0 := (row_8411.checked).trans rfl
theorem model_count_8411 : modelRow 8411 = 0 := by decide
theorem modelrow_agrees_8411 : modelRow 8411 = row 8411 := model_count_8411.trans row_count_8411.symm
def row_8412 : RowCertificate 8412 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8412 1) := by change ¬ Nat.Prime 283063801; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7650373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8412 2) := by change ¬ Nat.Prime 283046977; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5340509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8412 3) := by change ¬ Nat.Prime 283030153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40432879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8412 : row 8412 = 0 := (row_8412.checked).trans rfl
theorem model_count_8412 : modelRow 8412 = 0 := by decide
theorem modelrow_agrees_8412 : modelRow 8412 = row 8412 := model_count_8412.trans row_count_8412.symm
def row_8413 : RowCertificate 8413 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8413 1) := by change ¬ Nat.Prime 283131103; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14901637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8413 2) := by change ¬ Nat.Prime 283114277; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16653781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8413 3) := by change ¬ Nat.Prime 283097451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94365817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8413 : row 8413 = 0 := (row_8413.checked).trans rfl
theorem model_count_8413 : modelRow 8413 = 0 := by decide
theorem modelrow_agrees_8413 : modelRow 8413 = row 8413 := model_count_8413.trans row_count_8413.symm
def row_8414 : RowCertificate 8414 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8414 1) := by change ¬ Nat.Prime 283198413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94399471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8414 2) := by change ¬ Nat.Prime 283181585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56636317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8414 3) := by change ¬ Nat.Prime 283164757; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9134347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8414 : row 8414 = 0 := (row_8414.checked).trans rfl
theorem model_count_8414 : modelRow 8414 = 0 := by decide
theorem modelrow_agrees_8414 : modelRow 8414 = row 8414 := model_count_8414.trans row_count_8414.symm
def row_8415 : RowCertificate 8415 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8415 1) := by change ¬ Nat.Prime 283265731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40466533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8415 2) := by change ¬ Nat.Prime 283248901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21788377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8415 3) := by change ¬ Nat.Prime 283232071; exact PrattCertificates.composite_of_product (factor := 5869) (cofactor := 48259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8415 : row 8415 = 0 := (row_8415.checked).trans rfl
theorem model_count_8415 : modelRow 8415 = 0 := by decide
theorem modelrow_agrees_8415 : modelRow 8415 = row 8415 := model_count_8415.trans row_count_8415.symm
def row_8416 : RowCertificate 8416 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8416 1) := by change ¬ Nat.Prime 283333057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40476151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8416 2) := by change ¬ Nat.Prime 283316225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56663245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8416 3) := by change ¬ Nat.Prime 283299393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94433131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8416 : row 8416 = 0 := (row_8416.checked).trans rfl
theorem model_count_8416 : modelRow 8416 = 0 := by decide
theorem modelrow_agrees_8416 : modelRow 8416 = row 8416 := model_count_8416.trans row_count_8416.symm
def row_8417 : RowCertificate 8417 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8417 1) := by change ¬ Nat.Prime 283400391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94466797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8417 2) := by change ¬ Nat.Prime 283383557; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16669621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8417 3) := by change Nat.Prime 283366723; exact ProjectEulerSolutions.P58PrimeNodes.p_283366723.prime
    exact decide_eq_true prime
theorem row_count_8417 : row 8417 = 1 := (row_8417.checked).trans rfl
theorem model_count_8417 : modelRow 8417 = 1 := by decide
theorem modelrow_agrees_8417 : modelRow 8417 = row 8417 := model_count_8417.trans row_count_8417.symm
def row_8418 : RowCertificate 8418 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8418 1) := by change Nat.Prime 283467733; exact ProjectEulerSolutions.P58PrimeNodes.p_283467733.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8418 2) := by change ¬ Nat.Prime 283450897; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3882889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8418 3) := by change ¬ Nat.Prime 283434061; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2751787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8418 : row 8418 = 1 := (row_8418.checked).trans rfl
theorem model_count_8418 : modelRow 8418 = 1 := by decide
theorem modelrow_agrees_8418 : modelRow 8418 = row 8418 := model_count_8418.trans row_count_8418.symm
def row_8419 : RowCertificate 8419 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8419 1) := by change ¬ Nat.Prime 283535083; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21810391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8419 2) := by change ¬ Nat.Prime 283518245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56703649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8419 3) := by change ¬ Nat.Prime 283501407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94500469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8419 : row 8419 = 0 := (row_8419.checked).trans rfl
theorem model_count_8419 : modelRow 8419 = 0 := by decide
theorem modelrow_agrees_8419 : modelRow 8419 = row 8419 := model_count_8419.trans row_count_8419.symm
def row_8420 : RowCertificate 8420 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8420 1) := by change ¬ Nat.Prime 283602441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94534147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8420 2) := by change ¬ Nat.Prime 283585601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21814277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8420 3) := by change ¬ Nat.Prime 283568761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40509823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8420 : row 8420 = 0 := (row_8420.checked).trans rfl
theorem model_count_8420 : modelRow 8420 = 0 := by decide
theorem modelrow_agrees_8420 : modelRow 8420 = row 8420 := model_count_8420.trans row_count_8420.symm
def row_8421 : RowCertificate 8421 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8421 1) := by change ¬ Nat.Prime 283669807; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2924431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8421 2) := by change ¬ Nat.Prime 283652965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56730593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8421 3) := by change ¬ Nat.Prime 283636123; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14928217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8421 : row 8421 = 0 := (row_8421.checked).trans rfl
theorem model_count_8421 : modelRow 8421 = 0 := by decide
theorem modelrow_agrees_8421 : modelRow 8421 = row 8421 := model_count_8421.trans row_count_8421.symm
def row_8422 : RowCertificate 8422 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8422 1) := by change ¬ Nat.Prime 283737181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40533883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8422 2) := by change ¬ Nat.Prime 283720337; exact PrattCertificates.composite_of_product (factor := 5393) (cofactor := 52609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8422 3) := by change ¬ Nat.Prime 283703493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94567831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8422 : row 8422 = 0 := (row_8422.checked).trans rfl
theorem model_count_8422 : modelRow 8422 = 0 := by decide
theorem modelrow_agrees_8422 : modelRow 8422 = row 8422 := model_count_8422.trans row_count_8422.symm
def row_8423 : RowCertificate 8423 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8423 1) := by change ¬ Nat.Prime 283804563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94601521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8423 2) := by change Nat.Prime 283787717; exact ProjectEulerSolutions.P58PrimeNodes.p_283787717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8423 3) := by change ¬ Nat.Prime 283770871; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14935309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8423 : row 8423 = 1 := (row_8423.checked).trans rfl
theorem model_count_8423 : modelRow 8423 = 1 := by decide
theorem modelrow_agrees_8423 : modelRow 8423 = row 8423 := model_count_8423.trans row_count_8423.symm
def row_8424 : RowCertificate 8424 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8424 1) := by change ¬ Nat.Prime 283871953; exact PrattCertificates.composite_of_product (factor := 907) (cofactor := 312979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8424 2) := by change ¬ Nat.Prime 283855105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56771021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8424 3) := by change ¬ Nat.Prime 283838257; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1741339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8424 : row 8424 = 0 := (row_8424.checked).trans rfl
theorem model_count_8424 : modelRow 8424 = 0 := by decide
theorem modelrow_agrees_8424 : modelRow 8424 = row 8424 := model_count_8424.trans row_count_8424.symm
def row_8425 : RowCertificate 8425 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8425 1) := by change ¬ Nat.Prime 283939351; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3594169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8425 2) := by change ¬ Nat.Prime 283922501; exact PrattCertificates.composite_of_product (factor := 2969) (cofactor := 95629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8425 3) := by change ¬ Nat.Prime 283905651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94635217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8425 : row 8425 = 0 := (row_8425.checked).trans rfl
theorem model_count_8425 : modelRow 8425 = 0 := by decide
theorem modelrow_agrees_8425 : modelRow 8425 = row 8425 := model_count_8425.trans row_count_8425.symm
def row_8426 : RowCertificate 8426 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8426 1) := by change ¬ Nat.Prime 284006757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94668919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8426 2) := by change ¬ Nat.Prime 283989905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56797981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8426 3) := by change ¬ Nat.Prime 283973053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40567579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8426 : row 8426 = 0 := (row_8426.checked).trans rfl
theorem model_count_8426 : modelRow 8426 = 0 := by decide
theorem modelrow_agrees_8426 : modelRow 8426 = row 8426 := model_count_8426.trans row_count_8426.symm
def row_8427 : RowCertificate 8427 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8427 1) := by change ¬ Nat.Prime 284074171; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4239913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8427 2) := by change Nat.Prime 284057317; exact ProjectEulerSolutions.P58PrimeNodes.p_284057317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8427 3) := by change ¬ Nat.Prime 284040463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40577209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8427 : row 8427 = 1 := (row_8427.checked).trans rfl
theorem model_count_8427 : modelRow 8427 = 1 := by decide
theorem modelrow_agrees_8427 : modelRow 8427 = row 8427 := model_count_8427.trans row_count_8427.symm
def row_8428 : RowCertificate 8428 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8428 1) := by change Nat.Prime 284141593; exact ProjectEulerSolutions.P58PrimeNodes.p_284141593.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8428 2) := by change ¬ Nat.Prime 284124737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21855749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8428 3) := by change ¬ Nat.Prime 284107881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94702627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8428 : row 8428 = 1 := (row_8428.checked).trans rfl
theorem model_count_8428 : modelRow 8428 = 1 := by decide
theorem modelrow_agrees_8428 : modelRow 8428 = row 8428 := model_count_8428.trans row_count_8428.symm
def row_8429 : RowCertificate 8429 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8429 1) := by change ¬ Nat.Prime 284209023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94736341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8429 2) := by change ¬ Nat.Prime 284192165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56838433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8429 3) := by change ¬ Nat.Prime 284175307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21859639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8429 : row 8429 = 0 := (row_8429.checked).trans rfl
theorem model_count_8429 : modelRow 8429 = 0 := by decide
theorem modelrow_agrees_8429 : modelRow 8429 = row 8429 := model_count_8429.trans row_count_8429.symm
def row_8430 : RowCertificate 8430 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8430 1) := by change ¬ Nat.Prime 284276461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40610923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8430 2) := by change ¬ Nat.Prime 284259601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16721153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8430 3) := by change ¬ Nat.Prime 284242741; exact PrattCertificates.composite_of_product (factor := 13711) (cofactor := 20731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8430 : row 8430 = 0 := (row_8430.checked).trans rfl
theorem model_count_8430 : modelRow 8430 = 0 := by decide
theorem modelrow_agrees_8430 : modelRow 8430 = row 8430 := model_count_8430.trans row_count_8430.symm
def row_8431 : RowCertificate 8431 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8431 1) := by change ¬ Nat.Prime 284343907; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6612649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8431 2) := by change ¬ Nat.Prime 284327045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56865409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8431 3) := by change ¬ Nat.Prime 284310183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94770061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8431 : row 8431 = 0 := (row_8431.checked).trans rfl
theorem model_count_8431 : modelRow 8431 = 0 := by decide
theorem modelrow_agrees_8431 : modelRow 8431 = row 8431 := model_count_8431.trans row_count_8431.symm
def row_8432 : RowCertificate 8432 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8432 1) := by change ¬ Nat.Prime 284411361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94803787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8432 2) := by change Nat.Prime 284394497; exact ProjectEulerSolutions.P58PrimeNodes.p_284394497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8432 3) := by change ¬ Nat.Prime 284377633; exact PrattCertificates.composite_of_product (factor := 4801) (cofactor := 59233) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8432 : row 8432 = 1 := (row_8432.checked).trans rfl
theorem model_count_8432 : modelRow 8432 = 1 := by decide
theorem modelrow_agrees_8432 : modelRow 8432 = row 8432 := model_count_8432.trans row_count_8432.symm
def row_8433 : RowCertificate 8433 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8433 1) := by change Nat.Prime 284478823; exact ProjectEulerSolutions.P58PrimeNodes.p_284478823.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8433 2) := by change ¬ Nat.Prime 284461957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21881689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8433 3) := by change ¬ Nat.Prime 284445091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40635013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8433 : row 8433 = 1 := (row_8433.checked).trans rfl
theorem model_count_8433 : modelRow 8433 = 1 := by decide
theorem modelrow_agrees_8433 : modelRow 8433 = row 8433 := model_count_8433.trans row_count_8433.symm
def row_8434 : RowCertificate 8434 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8434 1) := by change Nat.Prime 284546293; exact ProjectEulerSolutions.P58PrimeNodes.p_284546293.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8434 2) := by change ¬ Nat.Prime 284529425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56905885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8434 3) := by change ¬ Nat.Prime 284512557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94837519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8434 : row 8434 = 1 := (row_8434.checked).trans rfl
theorem model_count_8434 : modelRow 8434 = 1 := by decide
theorem modelrow_agrees_8434 : modelRow 8434 = row 8434 := model_count_8434.trans row_count_8434.symm
def row_8435 : RowCertificate 8435 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8435 1) := by change ¬ Nat.Prime 284613771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94871257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8435 2) := by change Nat.Prime 284596901; exact ProjectEulerSolutions.P58PrimeNodes.p_284596901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8435 3) := by change ¬ Nat.Prime 284580031; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9180001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8435 : row 8435 = 1 := (row_8435.checked).trans rfl
theorem model_count_8435 : modelRow 8435 = 1 := by decide
theorem modelrow_agrees_8435 : modelRow 8435 = row 8435 := model_count_8435.trans row_count_8435.symm
def row_8436 : RowCertificate 8436 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8436 1) := by change ¬ Nat.Prime 284681257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40668751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8436 2) := by change ¬ Nat.Prime 284664385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56932877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8436 3) := by change ¬ Nat.Prime 284647513; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3899281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8436 : row 8436 = 0 := (row_8436.checked).trans rfl
theorem model_count_8436 : modelRow 8436 = 0 := by decide
theorem modelrow_agrees_8436 : modelRow 8436 = row 8436 := model_count_8436.trans row_count_8436.symm
def row_8437 : RowCertificate 8437 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8437 1) := by change ¬ Nat.Prime 284748751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40678393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8437 2) := by change ¬ Nat.Prime 284731877; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 1645849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8437 3) := by change ¬ Nat.Prime 284715003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94905001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8437 : row 8437 = 0 := (row_8437.checked).trans rfl
theorem model_count_8437 : modelRow 8437 = 0 := by decide
theorem modelrow_agrees_8437 : modelRow 8437 = row 8437 := model_count_8437.trans row_count_8437.symm
def row_8438 : RowCertificate 8438 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8438 1) := by change ¬ Nat.Prime 284816253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94938751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8438 2) := by change ¬ Nat.Prime 284799377; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3199993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8438 3) := by change ¬ Nat.Prime 284782501; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1475557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8438 : row 8438 = 0 := (row_8438.checked).trans rfl
theorem model_count_8438 : modelRow 8438 = 0 := by decide
theorem modelrow_agrees_8438 : modelRow 8438 = row 8438 := model_count_8438.trans row_count_8438.symm
def row_8439 : RowCertificate 8439 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8439 1) := by change ¬ Nat.Prime 284883763; exact PrattCertificates.composite_of_product (factor := 1123) (cofactor := 253681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8439 2) := by change ¬ Nat.Prime 284866885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56973377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8439 3) := by change ¬ Nat.Prime 284850007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21911539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8439 : row 8439 = 0 := (row_8439.checked).trans rfl
theorem model_count_8439 : modelRow 8439 = 0 := by decide
theorem modelrow_agrees_8439 : modelRow 8439 = row 8439 := model_count_8439.trans row_count_8439.symm
def row_8440 : RowCertificate 8440 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8440 1) := by change Nat.Prime 284951281; exact ProjectEulerSolutions.P58PrimeNodes.p_284951281.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8440 2) := by change Nat.Prime 284934401; exact ProjectEulerSolutions.P58PrimeNodes.p_284934401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8440 3) := by change ¬ Nat.Prime 284917521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94972507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8440 : row 8440 = 2 := (row_8440.checked).trans rfl
theorem model_count_8440 : modelRow 8440 = 2 := by decide
theorem modelrow_agrees_8440 : modelRow 8440 = row 8440 := model_count_8440.trans row_count_8440.symm
def row_8441 : RowCertificate 8441 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8441 1) := by change ¬ Nat.Prime 285018807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 95006269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8441 2) := by change ¬ Nat.Prime 285001925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 57000385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8441 3) := by change ¬ Nat.Prime 284985043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40712149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8441 : row 8441 = 0 := (row_8441.checked).trans rfl
theorem model_count_8441 : modelRow 8441 = 0 := by decide
theorem modelrow_agrees_8441 : modelRow 8441 = row 8441 := model_count_8441.trans row_count_8441.symm
def row_8442 : RowCertificate 8442 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8442 1) := by change Nat.Prime 285086341; exact ProjectEulerSolutions.P58PrimeNodes.p_285086341.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8442 2) := by change ¬ Nat.Prime 285069457; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5378669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8442 3) := by change ¬ Nat.Prime 285052573; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21927121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8442 : row 8442 = 1 := (row_8442.checked).trans rfl
theorem model_count_8442 : modelRow 8442 = 1 := by decide
theorem modelrow_agrees_8442 : modelRow 8442 = row 8442 := model_count_8442.trans row_count_8442.symm
def row_8443 : RowCertificate 8443 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8443 1) := by change ¬ Nat.Prime 285153883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40736269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8443 2) := by change ¬ Nat.Prime 285136997; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4674377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8443 3) := by change ¬ Nat.Prime 285120111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 95040037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8443 : row 8443 = 0 := (row_8443.checked).trans rfl
theorem model_count_8443 : modelRow 8443 = 0 := by decide
theorem modelrow_agrees_8443 : modelRow 8443 = row 8443 := model_count_8443.trans row_count_8443.symm
def row_8444 : RowCertificate 8444 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8444 1) := by change ¬ Nat.Prime 285221433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 95073811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8444 2) := by change ¬ Nat.Prime 285204545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 57040909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8444 3) := by change Nat.Prime 285187657; exact ProjectEulerSolutions.P58PrimeNodes.p_285187657.prime
    exact decide_eq_true prime
theorem row_count_8444 : row 8444 = 1 := (row_8444.checked).trans rfl
theorem model_count_8444 : modelRow 8444 = 1 := by decide
theorem modelrow_agrees_8444 : modelRow 8444 = row 8444 := model_count_8444.trans row_count_8444.symm
def row_8445 : RowCertificate 8445 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8445 1) := by change ¬ Nat.Prime 285288991; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21945307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8445 2) := by change ¬ Nat.Prime 285272101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9836969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8445 3) := by change ¬ Nat.Prime 285255211; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9201781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8445 : row 8445 = 0 := (row_8445.checked).trans rfl
theorem model_count_8445 : modelRow 8445 = 0 := by decide
theorem modelrow_agrees_8445 : modelRow 8445 = row 8445 := model_count_8445.trans row_count_8445.symm
def row_8446 : RowCertificate 8446 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8446 1) := by change ¬ Nat.Prime 285356557; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 6636199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8446 2) := by change ¬ Nat.Prime 285339665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 57067933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8446 3) := by change ¬ Nat.Prime 285322773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 95107591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8446 : row 8446 = 0 := (row_8446.checked).trans rfl
theorem model_count_8446 : modelRow 8446 = 0 := by decide
theorem modelrow_agrees_8446 : modelRow 8446 = row 8446 := model_count_8446.trans row_count_8446.symm
def row_8447 : RowCertificate 8447 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8447 1) := by change ¬ Nat.Prime 285424131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 95141377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8447 2) := by change ¬ Nat.Prime 285407237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 16788661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8447 3) := by change ¬ Nat.Prime 285390343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40770049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8447 : row 8447 = 0 := (row_8447.checked).trans rfl
theorem model_count_8447 : modelRow 8447 = 0 := by decide
theorem modelrow_agrees_8447 : modelRow 8447 = row 8447 := model_count_8447.trans row_count_8447.symm
def row_8448 : RowCertificate 8448 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8448 1) := by change ¬ Nat.Prime 285491713; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 21960901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8448 2) := by change Nat.Prime 285474817; exact ProjectEulerSolutions.P58PrimeNodes.p_285474817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8448 3) := by change ¬ Nat.Prime 285457921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40779703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8448 : row 8448 = 1 := (row_8448.checked).trans rfl
theorem model_count_8448 : modelRow 8448 = 1 := by decide
theorem modelrow_agrees_8448 : modelRow 8448 = row 8448 := model_count_8448.trans row_count_8448.symm
private def atomic_8321 : CountSegment 1 10 13120 8320 8321 3513 3513 := CountSegment.single (first := 8320) (initial := 3513) row_count_8321 (by decide)
private theorem atomic_8321_agreement : RowAgreement 8320 8321 := RowAgreement.single (first := 8320) modelrow_agrees_8321
private def atomic_8322 : CountSegment 1 10 13120 8321 8322 3513 3513 := CountSegment.single (first := 8321) (initial := 3513) row_count_8322 (by decide)
private theorem atomic_8322_agreement : RowAgreement 8321 8322 := RowAgreement.single (first := 8321) modelrow_agrees_8322
private def atomic_8323 : CountSegment 1 10 13120 8322 8323 3513 3513 := CountSegment.single (first := 8322) (initial := 3513) row_count_8323 (by decide)
private theorem atomic_8323_agreement : RowAgreement 8322 8323 := RowAgreement.single (first := 8322) modelrow_agrees_8323
private def atomic_8324 : CountSegment 1 10 13120 8323 8324 3513 3513 := CountSegment.single (first := 8323) (initial := 3513) row_count_8324 (by decide)
private theorem atomic_8324_agreement : RowAgreement 8323 8324 := RowAgreement.single (first := 8323) modelrow_agrees_8324
private def atomic_8325 : CountSegment 1 10 13120 8324 8325 3513 3514 := CountSegment.single (first := 8324) (initial := 3513) row_count_8325 (by decide)
private theorem atomic_8325_agreement : RowAgreement 8324 8325 := RowAgreement.single (first := 8324) modelrow_agrees_8325
private def atomic_8326 : CountSegment 1 10 13120 8325 8326 3514 3514 := CountSegment.single (first := 8325) (initial := 3514) row_count_8326 (by decide)
private theorem atomic_8326_agreement : RowAgreement 8325 8326 := RowAgreement.single (first := 8325) modelrow_agrees_8326
private def atomic_8327 : CountSegment 1 10 13120 8326 8327 3514 3514 := CountSegment.single (first := 8326) (initial := 3514) row_count_8327 (by decide)
private theorem atomic_8327_agreement : RowAgreement 8326 8327 := RowAgreement.single (first := 8326) modelrow_agrees_8327
private def atomic_8328 : CountSegment 1 10 13120 8327 8328 3514 3514 := CountSegment.single (first := 8327) (initial := 3514) row_count_8328 (by decide)
private theorem atomic_8328_agreement : RowAgreement 8327 8328 := RowAgreement.single (first := 8327) modelrow_agrees_8328
private def atomic_8329 : CountSegment 1 10 13120 8328 8329 3514 3515 := CountSegment.single (first := 8328) (initial := 3514) row_count_8329 (by decide)
private theorem atomic_8329_agreement : RowAgreement 8328 8329 := RowAgreement.single (first := 8328) modelrow_agrees_8329
private def atomic_8330 : CountSegment 1 10 13120 8329 8330 3515 3515 := CountSegment.single (first := 8329) (initial := 3515) row_count_8330 (by decide)
private theorem atomic_8330_agreement : RowAgreement 8329 8330 := RowAgreement.single (first := 8329) modelrow_agrees_8330
private def atomic_8331 : CountSegment 1 10 13120 8330 8331 3515 3516 := CountSegment.single (first := 8330) (initial := 3515) row_count_8331 (by decide)
private theorem atomic_8331_agreement : RowAgreement 8330 8331 := RowAgreement.single (first := 8330) modelrow_agrees_8331
private def atomic_8332 : CountSegment 1 10 13120 8331 8332 3516 3516 := CountSegment.single (first := 8331) (initial := 3516) row_count_8332 (by decide)
private theorem atomic_8332_agreement : RowAgreement 8331 8332 := RowAgreement.single (first := 8331) modelrow_agrees_8332
private def atomic_8333 : CountSegment 1 10 13120 8332 8333 3516 3517 := CountSegment.single (first := 8332) (initial := 3516) row_count_8333 (by decide)
private theorem atomic_8333_agreement : RowAgreement 8332 8333 := RowAgreement.single (first := 8332) modelrow_agrees_8333
private def atomic_8334 : CountSegment 1 10 13120 8333 8334 3517 3518 := CountSegment.single (first := 8333) (initial := 3517) row_count_8334 (by decide)
private theorem atomic_8334_agreement : RowAgreement 8333 8334 := RowAgreement.single (first := 8333) modelrow_agrees_8334
private def atomic_8335 : CountSegment 1 10 13120 8334 8335 3518 3518 := CountSegment.single (first := 8334) (initial := 3518) row_count_8335 (by decide)
private theorem atomic_8335_agreement : RowAgreement 8334 8335 := RowAgreement.single (first := 8334) modelrow_agrees_8335
private def atomic_8336 : CountSegment 1 10 13120 8335 8336 3518 3518 := CountSegment.single (first := 8335) (initial := 3518) row_count_8336 (by decide)
private theorem atomic_8336_agreement : RowAgreement 8335 8336 := RowAgreement.single (first := 8335) modelrow_agrees_8336
private def atomic_8337 : CountSegment 1 10 13120 8336 8337 3518 3518 := CountSegment.single (first := 8336) (initial := 3518) row_count_8337 (by decide)
private theorem atomic_8337_agreement : RowAgreement 8336 8337 := RowAgreement.single (first := 8336) modelrow_agrees_8337
private def atomic_8338 : CountSegment 1 10 13120 8337 8338 3518 3519 := CountSegment.single (first := 8337) (initial := 3518) row_count_8338 (by decide)
private theorem atomic_8338_agreement : RowAgreement 8337 8338 := RowAgreement.single (first := 8337) modelrow_agrees_8338
private def atomic_8339 : CountSegment 1 10 13120 8338 8339 3519 3519 := CountSegment.single (first := 8338) (initial := 3519) row_count_8339 (by decide)
private theorem atomic_8339_agreement : RowAgreement 8338 8339 := RowAgreement.single (first := 8338) modelrow_agrees_8339
private def atomic_8340 : CountSegment 1 10 13120 8339 8340 3519 3520 := CountSegment.single (first := 8339) (initial := 3519) row_count_8340 (by decide)
private theorem atomic_8340_agreement : RowAgreement 8339 8340 := RowAgreement.single (first := 8339) modelrow_agrees_8340
private def atomic_8341 : CountSegment 1 10 13120 8340 8341 3520 3520 := CountSegment.single (first := 8340) (initial := 3520) row_count_8341 (by decide)
private theorem atomic_8341_agreement : RowAgreement 8340 8341 := RowAgreement.single (first := 8340) modelrow_agrees_8341
private def atomic_8342 : CountSegment 1 10 13120 8341 8342 3520 3520 := CountSegment.single (first := 8341) (initial := 3520) row_count_8342 (by decide)
private theorem atomic_8342_agreement : RowAgreement 8341 8342 := RowAgreement.single (first := 8341) modelrow_agrees_8342
private def atomic_8343 : CountSegment 1 10 13120 8342 8343 3520 3521 := CountSegment.single (first := 8342) (initial := 3520) row_count_8343 (by decide)
private theorem atomic_8343_agreement : RowAgreement 8342 8343 := RowAgreement.single (first := 8342) modelrow_agrees_8343
private def atomic_8344 : CountSegment 1 10 13120 8343 8344 3521 3521 := CountSegment.single (first := 8343) (initial := 3521) row_count_8344 (by decide)
private theorem atomic_8344_agreement : RowAgreement 8343 8344 := RowAgreement.single (first := 8343) modelrow_agrees_8344
private def atomic_8345 : CountSegment 1 10 13120 8344 8345 3521 3521 := CountSegment.single (first := 8344) (initial := 3521) row_count_8345 (by decide)
private theorem atomic_8345_agreement : RowAgreement 8344 8345 := RowAgreement.single (first := 8344) modelrow_agrees_8345
private def atomic_8346 : CountSegment 1 10 13120 8345 8346 3521 3522 := CountSegment.single (first := 8345) (initial := 3521) row_count_8346 (by decide)
private theorem atomic_8346_agreement : RowAgreement 8345 8346 := RowAgreement.single (first := 8345) modelrow_agrees_8346
private def atomic_8347 : CountSegment 1 10 13120 8346 8347 3522 3522 := CountSegment.single (first := 8346) (initial := 3522) row_count_8347 (by decide)
private theorem atomic_8347_agreement : RowAgreement 8346 8347 := RowAgreement.single (first := 8346) modelrow_agrees_8347
private def atomic_8348 : CountSegment 1 10 13120 8347 8348 3522 3522 := CountSegment.single (first := 8347) (initial := 3522) row_count_8348 (by decide)
private theorem atomic_8348_agreement : RowAgreement 8347 8348 := RowAgreement.single (first := 8347) modelrow_agrees_8348
private def atomic_8349 : CountSegment 1 10 13120 8348 8349 3522 3523 := CountSegment.single (first := 8348) (initial := 3522) row_count_8349 (by decide)
private theorem atomic_8349_agreement : RowAgreement 8348 8349 := RowAgreement.single (first := 8348) modelrow_agrees_8349
private def atomic_8350 : CountSegment 1 10 13120 8349 8350 3523 3523 := CountSegment.single (first := 8349) (initial := 3523) row_count_8350 (by decide)
private theorem atomic_8350_agreement : RowAgreement 8349 8350 := RowAgreement.single (first := 8349) modelrow_agrees_8350
private def atomic_8351 : CountSegment 1 10 13120 8350 8351 3523 3523 := CountSegment.single (first := 8350) (initial := 3523) row_count_8351 (by decide)
private theorem atomic_8351_agreement : RowAgreement 8350 8351 := RowAgreement.single (first := 8350) modelrow_agrees_8351
private def atomic_8352 : CountSegment 1 10 13120 8351 8352 3523 3524 := CountSegment.single (first := 8351) (initial := 3523) row_count_8352 (by decide)
private theorem atomic_8352_agreement : RowAgreement 8351 8352 := RowAgreement.single (first := 8351) modelrow_agrees_8352
private def atomic_8353 : CountSegment 1 10 13120 8352 8353 3524 3524 := CountSegment.single (first := 8352) (initial := 3524) row_count_8353 (by decide)
private theorem atomic_8353_agreement : RowAgreement 8352 8353 := RowAgreement.single (first := 8352) modelrow_agrees_8353
private def atomic_8354 : CountSegment 1 10 13120 8353 8354 3524 3524 := CountSegment.single (first := 8353) (initial := 3524) row_count_8354 (by decide)
private theorem atomic_8354_agreement : RowAgreement 8353 8354 := RowAgreement.single (first := 8353) modelrow_agrees_8354
private def atomic_8355 : CountSegment 1 10 13120 8354 8355 3524 3524 := CountSegment.single (first := 8354) (initial := 3524) row_count_8355 (by decide)
private theorem atomic_8355_agreement : RowAgreement 8354 8355 := RowAgreement.single (first := 8354) modelrow_agrees_8355
private def atomic_8356 : CountSegment 1 10 13120 8355 8356 3524 3524 := CountSegment.single (first := 8355) (initial := 3524) row_count_8356 (by decide)
private theorem atomic_8356_agreement : RowAgreement 8355 8356 := RowAgreement.single (first := 8355) modelrow_agrees_8356
private def atomic_8357 : CountSegment 1 10 13120 8356 8357 3524 3524 := CountSegment.single (first := 8356) (initial := 3524) row_count_8357 (by decide)
private theorem atomic_8357_agreement : RowAgreement 8356 8357 := RowAgreement.single (first := 8356) modelrow_agrees_8357
private def atomic_8358 : CountSegment 1 10 13120 8357 8358 3524 3525 := CountSegment.single (first := 8357) (initial := 3524) row_count_8358 (by decide)
private theorem atomic_8358_agreement : RowAgreement 8357 8358 := RowAgreement.single (first := 8357) modelrow_agrees_8358
private def atomic_8359 : CountSegment 1 10 13120 8358 8359 3525 3525 := CountSegment.single (first := 8358) (initial := 3525) row_count_8359 (by decide)
private theorem atomic_8359_agreement : RowAgreement 8358 8359 := RowAgreement.single (first := 8358) modelrow_agrees_8359
private def atomic_8360 : CountSegment 1 10 13120 8359 8360 3525 3526 := CountSegment.single (first := 8359) (initial := 3525) row_count_8360 (by decide)
private theorem atomic_8360_agreement : RowAgreement 8359 8360 := RowAgreement.single (first := 8359) modelrow_agrees_8360
private def atomic_8361 : CountSegment 1 10 13120 8360 8361 3526 3526 := CountSegment.single (first := 8360) (initial := 3526) row_count_8361 (by decide)
private theorem atomic_8361_agreement : RowAgreement 8360 8361 := RowAgreement.single (first := 8360) modelrow_agrees_8361
private def atomic_8362 : CountSegment 1 10 13120 8361 8362 3526 3526 := CountSegment.single (first := 8361) (initial := 3526) row_count_8362 (by decide)
private theorem atomic_8362_agreement : RowAgreement 8361 8362 := RowAgreement.single (first := 8361) modelrow_agrees_8362
private def atomic_8363 : CountSegment 1 10 13120 8362 8363 3526 3526 := CountSegment.single (first := 8362) (initial := 3526) row_count_8363 (by decide)
private theorem atomic_8363_agreement : RowAgreement 8362 8363 := RowAgreement.single (first := 8362) modelrow_agrees_8363
private def atomic_8364 : CountSegment 1 10 13120 8363 8364 3526 3526 := CountSegment.single (first := 8363) (initial := 3526) row_count_8364 (by decide)
private theorem atomic_8364_agreement : RowAgreement 8363 8364 := RowAgreement.single (first := 8363) modelrow_agrees_8364
private def atomic_8365 : CountSegment 1 10 13120 8364 8365 3526 3526 := CountSegment.single (first := 8364) (initial := 3526) row_count_8365 (by decide)
private theorem atomic_8365_agreement : RowAgreement 8364 8365 := RowAgreement.single (first := 8364) modelrow_agrees_8365
private def atomic_8366 : CountSegment 1 10 13120 8365 8366 3526 3526 := CountSegment.single (first := 8365) (initial := 3526) row_count_8366 (by decide)
private theorem atomic_8366_agreement : RowAgreement 8365 8366 := RowAgreement.single (first := 8365) modelrow_agrees_8366
private def atomic_8367 : CountSegment 1 10 13120 8366 8367 3526 3527 := CountSegment.single (first := 8366) (initial := 3526) row_count_8367 (by decide)
private theorem atomic_8367_agreement : RowAgreement 8366 8367 := RowAgreement.single (first := 8366) modelrow_agrees_8367
private def atomic_8368 : CountSegment 1 10 13120 8367 8368 3527 3527 := CountSegment.single (first := 8367) (initial := 3527) row_count_8368 (by decide)
private theorem atomic_8368_agreement : RowAgreement 8367 8368 := RowAgreement.single (first := 8367) modelrow_agrees_8368
private def atomic_8369 : CountSegment 1 10 13120 8368 8369 3527 3527 := CountSegment.single (first := 8368) (initial := 3527) row_count_8369 (by decide)
private theorem atomic_8369_agreement : RowAgreement 8368 8369 := RowAgreement.single (first := 8368) modelrow_agrees_8369
private def atomic_8370 : CountSegment 1 10 13120 8369 8370 3527 3528 := CountSegment.single (first := 8369) (initial := 3527) row_count_8370 (by decide)
private theorem atomic_8370_agreement : RowAgreement 8369 8370 := RowAgreement.single (first := 8369) modelrow_agrees_8370
private def atomic_8371 : CountSegment 1 10 13120 8370 8371 3528 3528 := CountSegment.single (first := 8370) (initial := 3528) row_count_8371 (by decide)
private theorem atomic_8371_agreement : RowAgreement 8370 8371 := RowAgreement.single (first := 8370) modelrow_agrees_8371
private def atomic_8372 : CountSegment 1 10 13120 8371 8372 3528 3528 := CountSegment.single (first := 8371) (initial := 3528) row_count_8372 (by decide)
private theorem atomic_8372_agreement : RowAgreement 8371 8372 := RowAgreement.single (first := 8371) modelrow_agrees_8372
private def atomic_8373 : CountSegment 1 10 13120 8372 8373 3528 3529 := CountSegment.single (first := 8372) (initial := 3528) row_count_8373 (by decide)
private theorem atomic_8373_agreement : RowAgreement 8372 8373 := RowAgreement.single (first := 8372) modelrow_agrees_8373
private def atomic_8374 : CountSegment 1 10 13120 8373 8374 3529 3529 := CountSegment.single (first := 8373) (initial := 3529) row_count_8374 (by decide)
private theorem atomic_8374_agreement : RowAgreement 8373 8374 := RowAgreement.single (first := 8373) modelrow_agrees_8374
private def atomic_8375 : CountSegment 1 10 13120 8374 8375 3529 3530 := CountSegment.single (first := 8374) (initial := 3529) row_count_8375 (by decide)
private theorem atomic_8375_agreement : RowAgreement 8374 8375 := RowAgreement.single (first := 8374) modelrow_agrees_8375
private def atomic_8376 : CountSegment 1 10 13120 8375 8376 3530 3532 := CountSegment.single (first := 8375) (initial := 3530) row_count_8376 (by decide)
private theorem atomic_8376_agreement : RowAgreement 8375 8376 := RowAgreement.single (first := 8375) modelrow_agrees_8376
private def atomic_8377 : CountSegment 1 10 13120 8376 8377 3532 3534 := CountSegment.single (first := 8376) (initial := 3532) row_count_8377 (by decide)
private theorem atomic_8377_agreement : RowAgreement 8376 8377 := RowAgreement.single (first := 8376) modelrow_agrees_8377
private def atomic_8378 : CountSegment 1 10 13120 8377 8378 3534 3534 := CountSegment.single (first := 8377) (initial := 3534) row_count_8378 (by decide)
private theorem atomic_8378_agreement : RowAgreement 8377 8378 := RowAgreement.single (first := 8377) modelrow_agrees_8378
private def atomic_8379 : CountSegment 1 10 13120 8378 8379 3534 3535 := CountSegment.single (first := 8378) (initial := 3534) row_count_8379 (by decide)
private theorem atomic_8379_agreement : RowAgreement 8378 8379 := RowAgreement.single (first := 8378) modelrow_agrees_8379
private def atomic_8380 : CountSegment 1 10 13120 8379 8380 3535 3535 := CountSegment.single (first := 8379) (initial := 3535) row_count_8380 (by decide)
private theorem atomic_8380_agreement : RowAgreement 8379 8380 := RowAgreement.single (first := 8379) modelrow_agrees_8380
private def atomic_8381 : CountSegment 1 10 13120 8380 8381 3535 3535 := CountSegment.single (first := 8380) (initial := 3535) row_count_8381 (by decide)
private theorem atomic_8381_agreement : RowAgreement 8380 8381 := RowAgreement.single (first := 8380) modelrow_agrees_8381
private def atomic_8382 : CountSegment 1 10 13120 8381 8382 3535 3535 := CountSegment.single (first := 8381) (initial := 3535) row_count_8382 (by decide)
private theorem atomic_8382_agreement : RowAgreement 8381 8382 := RowAgreement.single (first := 8381) modelrow_agrees_8382
private def atomic_8383 : CountSegment 1 10 13120 8382 8383 3535 3535 := CountSegment.single (first := 8382) (initial := 3535) row_count_8383 (by decide)
private theorem atomic_8383_agreement : RowAgreement 8382 8383 := RowAgreement.single (first := 8382) modelrow_agrees_8383
private def atomic_8384 : CountSegment 1 10 13120 8383 8384 3535 3535 := CountSegment.single (first := 8383) (initial := 3535) row_count_8384 (by decide)
private theorem atomic_8384_agreement : RowAgreement 8383 8384 := RowAgreement.single (first := 8383) modelrow_agrees_8384
private def atomic_8385 : CountSegment 1 10 13120 8384 8385 3535 3537 := CountSegment.single (first := 8384) (initial := 3535) row_count_8385 (by decide)
private theorem atomic_8385_agreement : RowAgreement 8384 8385 := RowAgreement.single (first := 8384) modelrow_agrees_8385
private def atomic_8386 : CountSegment 1 10 13120 8385 8386 3537 3538 := CountSegment.single (first := 8385) (initial := 3537) row_count_8386 (by decide)
private theorem atomic_8386_agreement : RowAgreement 8385 8386 := RowAgreement.single (first := 8385) modelrow_agrees_8386
private def atomic_8387 : CountSegment 1 10 13120 8386 8387 3538 3538 := CountSegment.single (first := 8386) (initial := 3538) row_count_8387 (by decide)
private theorem atomic_8387_agreement : RowAgreement 8386 8387 := RowAgreement.single (first := 8386) modelrow_agrees_8387
private def atomic_8388 : CountSegment 1 10 13120 8387 8388 3538 3538 := CountSegment.single (first := 8387) (initial := 3538) row_count_8388 (by decide)
private theorem atomic_8388_agreement : RowAgreement 8387 8388 := RowAgreement.single (first := 8387) modelrow_agrees_8388
private def atomic_8389 : CountSegment 1 10 13120 8388 8389 3538 3538 := CountSegment.single (first := 8388) (initial := 3538) row_count_8389 (by decide)
private theorem atomic_8389_agreement : RowAgreement 8388 8389 := RowAgreement.single (first := 8388) modelrow_agrees_8389
private def atomic_8390 : CountSegment 1 10 13120 8389 8390 3538 3539 := CountSegment.single (first := 8389) (initial := 3538) row_count_8390 (by decide)
private theorem atomic_8390_agreement : RowAgreement 8389 8390 := RowAgreement.single (first := 8389) modelrow_agrees_8390
private def atomic_8391 : CountSegment 1 10 13120 8390 8391 3539 3539 := CountSegment.single (first := 8390) (initial := 3539) row_count_8391 (by decide)
private theorem atomic_8391_agreement : RowAgreement 8390 8391 := RowAgreement.single (first := 8390) modelrow_agrees_8391
private def atomic_8392 : CountSegment 1 10 13120 8391 8392 3539 3540 := CountSegment.single (first := 8391) (initial := 3539) row_count_8392 (by decide)
private theorem atomic_8392_agreement : RowAgreement 8391 8392 := RowAgreement.single (first := 8391) modelrow_agrees_8392
private def atomic_8393 : CountSegment 1 10 13120 8392 8393 3540 3540 := CountSegment.single (first := 8392) (initial := 3540) row_count_8393 (by decide)
private theorem atomic_8393_agreement : RowAgreement 8392 8393 := RowAgreement.single (first := 8392) modelrow_agrees_8393
private def atomic_8394 : CountSegment 1 10 13120 8393 8394 3540 3540 := CountSegment.single (first := 8393) (initial := 3540) row_count_8394 (by decide)
private theorem atomic_8394_agreement : RowAgreement 8393 8394 := RowAgreement.single (first := 8393) modelrow_agrees_8394
private def atomic_8395 : CountSegment 1 10 13120 8394 8395 3540 3541 := CountSegment.single (first := 8394) (initial := 3540) row_count_8395 (by decide)
private theorem atomic_8395_agreement : RowAgreement 8394 8395 := RowAgreement.single (first := 8394) modelrow_agrees_8395
private def atomic_8396 : CountSegment 1 10 13120 8395 8396 3541 3541 := CountSegment.single (first := 8395) (initial := 3541) row_count_8396 (by decide)
private theorem atomic_8396_agreement : RowAgreement 8395 8396 := RowAgreement.single (first := 8395) modelrow_agrees_8396
private def atomic_8397 : CountSegment 1 10 13120 8396 8397 3541 3542 := CountSegment.single (first := 8396) (initial := 3541) row_count_8397 (by decide)
private theorem atomic_8397_agreement : RowAgreement 8396 8397 := RowAgreement.single (first := 8396) modelrow_agrees_8397
private def atomic_8398 : CountSegment 1 10 13120 8397 8398 3542 3542 := CountSegment.single (first := 8397) (initial := 3542) row_count_8398 (by decide)
private theorem atomic_8398_agreement : RowAgreement 8397 8398 := RowAgreement.single (first := 8397) modelrow_agrees_8398
private def atomic_8399 : CountSegment 1 10 13120 8398 8399 3542 3542 := CountSegment.single (first := 8398) (initial := 3542) row_count_8399 (by decide)
private theorem atomic_8399_agreement : RowAgreement 8398 8399 := RowAgreement.single (first := 8398) modelrow_agrees_8399
private def atomic_8400 : CountSegment 1 10 13120 8399 8400 3542 3542 := CountSegment.single (first := 8399) (initial := 3542) row_count_8400 (by decide)
private theorem atomic_8400_agreement : RowAgreement 8399 8400 := RowAgreement.single (first := 8399) modelrow_agrees_8400
private def atomic_8401 : CountSegment 1 10 13120 8400 8401 3542 3542 := CountSegment.single (first := 8400) (initial := 3542) row_count_8401 (by decide)
private theorem atomic_8401_agreement : RowAgreement 8400 8401 := RowAgreement.single (first := 8400) modelrow_agrees_8401
private def atomic_8402 : CountSegment 1 10 13120 8401 8402 3542 3542 := CountSegment.single (first := 8401) (initial := 3542) row_count_8402 (by decide)
private theorem atomic_8402_agreement : RowAgreement 8401 8402 := RowAgreement.single (first := 8401) modelrow_agrees_8402
private def atomic_8403 : CountSegment 1 10 13120 8402 8403 3542 3542 := CountSegment.single (first := 8402) (initial := 3542) row_count_8403 (by decide)
private theorem atomic_8403_agreement : RowAgreement 8402 8403 := RowAgreement.single (first := 8402) modelrow_agrees_8403
private def atomic_8404 : CountSegment 1 10 13120 8403 8404 3542 3542 := CountSegment.single (first := 8403) (initial := 3542) row_count_8404 (by decide)
private theorem atomic_8404_agreement : RowAgreement 8403 8404 := RowAgreement.single (first := 8403) modelrow_agrees_8404
private def atomic_8405 : CountSegment 1 10 13120 8404 8405 3542 3542 := CountSegment.single (first := 8404) (initial := 3542) row_count_8405 (by decide)
private theorem atomic_8405_agreement : RowAgreement 8404 8405 := RowAgreement.single (first := 8404) modelrow_agrees_8405
private def atomic_8406 : CountSegment 1 10 13120 8405 8406 3542 3542 := CountSegment.single (first := 8405) (initial := 3542) row_count_8406 (by decide)
private theorem atomic_8406_agreement : RowAgreement 8405 8406 := RowAgreement.single (first := 8405) modelrow_agrees_8406
private def atomic_8407 : CountSegment 1 10 13120 8406 8407 3542 3542 := CountSegment.single (first := 8406) (initial := 3542) row_count_8407 (by decide)
private theorem atomic_8407_agreement : RowAgreement 8406 8407 := RowAgreement.single (first := 8406) modelrow_agrees_8407
private def atomic_8408 : CountSegment 1 10 13120 8407 8408 3542 3544 := CountSegment.single (first := 8407) (initial := 3542) row_count_8408 (by decide)
private theorem atomic_8408_agreement : RowAgreement 8407 8408 := RowAgreement.single (first := 8407) modelrow_agrees_8408
private def atomic_8409 : CountSegment 1 10 13120 8408 8409 3544 3545 := CountSegment.single (first := 8408) (initial := 3544) row_count_8409 (by decide)
private theorem atomic_8409_agreement : RowAgreement 8408 8409 := RowAgreement.single (first := 8408) modelrow_agrees_8409
private def atomic_8410 : CountSegment 1 10 13120 8409 8410 3545 3545 := CountSegment.single (first := 8409) (initial := 3545) row_count_8410 (by decide)
private theorem atomic_8410_agreement : RowAgreement 8409 8410 := RowAgreement.single (first := 8409) modelrow_agrees_8410
private def atomic_8411 : CountSegment 1 10 13120 8410 8411 3545 3545 := CountSegment.single (first := 8410) (initial := 3545) row_count_8411 (by decide)
private theorem atomic_8411_agreement : RowAgreement 8410 8411 := RowAgreement.single (first := 8410) modelrow_agrees_8411
private def atomic_8412 : CountSegment 1 10 13120 8411 8412 3545 3545 := CountSegment.single (first := 8411) (initial := 3545) row_count_8412 (by decide)
private theorem atomic_8412_agreement : RowAgreement 8411 8412 := RowAgreement.single (first := 8411) modelrow_agrees_8412
private def atomic_8413 : CountSegment 1 10 13120 8412 8413 3545 3545 := CountSegment.single (first := 8412) (initial := 3545) row_count_8413 (by decide)
private theorem atomic_8413_agreement : RowAgreement 8412 8413 := RowAgreement.single (first := 8412) modelrow_agrees_8413
private def atomic_8414 : CountSegment 1 10 13120 8413 8414 3545 3545 := CountSegment.single (first := 8413) (initial := 3545) row_count_8414 (by decide)
private theorem atomic_8414_agreement : RowAgreement 8413 8414 := RowAgreement.single (first := 8413) modelrow_agrees_8414
private def atomic_8415 : CountSegment 1 10 13120 8414 8415 3545 3545 := CountSegment.single (first := 8414) (initial := 3545) row_count_8415 (by decide)
private theorem atomic_8415_agreement : RowAgreement 8414 8415 := RowAgreement.single (first := 8414) modelrow_agrees_8415
private def atomic_8416 : CountSegment 1 10 13120 8415 8416 3545 3545 := CountSegment.single (first := 8415) (initial := 3545) row_count_8416 (by decide)
private theorem atomic_8416_agreement : RowAgreement 8415 8416 := RowAgreement.single (first := 8415) modelrow_agrees_8416
private def atomic_8417 : CountSegment 1 10 13120 8416 8417 3545 3546 := CountSegment.single (first := 8416) (initial := 3545) row_count_8417 (by decide)
private theorem atomic_8417_agreement : RowAgreement 8416 8417 := RowAgreement.single (first := 8416) modelrow_agrees_8417
private def atomic_8418 : CountSegment 1 10 13120 8417 8418 3546 3547 := CountSegment.single (first := 8417) (initial := 3546) row_count_8418 (by decide)
private theorem atomic_8418_agreement : RowAgreement 8417 8418 := RowAgreement.single (first := 8417) modelrow_agrees_8418
private def atomic_8419 : CountSegment 1 10 13120 8418 8419 3547 3547 := CountSegment.single (first := 8418) (initial := 3547) row_count_8419 (by decide)
private theorem atomic_8419_agreement : RowAgreement 8418 8419 := RowAgreement.single (first := 8418) modelrow_agrees_8419
private def atomic_8420 : CountSegment 1 10 13120 8419 8420 3547 3547 := CountSegment.single (first := 8419) (initial := 3547) row_count_8420 (by decide)
private theorem atomic_8420_agreement : RowAgreement 8419 8420 := RowAgreement.single (first := 8419) modelrow_agrees_8420
private def atomic_8421 : CountSegment 1 10 13120 8420 8421 3547 3547 := CountSegment.single (first := 8420) (initial := 3547) row_count_8421 (by decide)
private theorem atomic_8421_agreement : RowAgreement 8420 8421 := RowAgreement.single (first := 8420) modelrow_agrees_8421
private def atomic_8422 : CountSegment 1 10 13120 8421 8422 3547 3547 := CountSegment.single (first := 8421) (initial := 3547) row_count_8422 (by decide)
private theorem atomic_8422_agreement : RowAgreement 8421 8422 := RowAgreement.single (first := 8421) modelrow_agrees_8422
private def atomic_8423 : CountSegment 1 10 13120 8422 8423 3547 3548 := CountSegment.single (first := 8422) (initial := 3547) row_count_8423 (by decide)
private theorem atomic_8423_agreement : RowAgreement 8422 8423 := RowAgreement.single (first := 8422) modelrow_agrees_8423
private def atomic_8424 : CountSegment 1 10 13120 8423 8424 3548 3548 := CountSegment.single (first := 8423) (initial := 3548) row_count_8424 (by decide)
private theorem atomic_8424_agreement : RowAgreement 8423 8424 := RowAgreement.single (first := 8423) modelrow_agrees_8424
private def atomic_8425 : CountSegment 1 10 13120 8424 8425 3548 3548 := CountSegment.single (first := 8424) (initial := 3548) row_count_8425 (by decide)
private theorem atomic_8425_agreement : RowAgreement 8424 8425 := RowAgreement.single (first := 8424) modelrow_agrees_8425
private def atomic_8426 : CountSegment 1 10 13120 8425 8426 3548 3548 := CountSegment.single (first := 8425) (initial := 3548) row_count_8426 (by decide)
private theorem atomic_8426_agreement : RowAgreement 8425 8426 := RowAgreement.single (first := 8425) modelrow_agrees_8426
private def atomic_8427 : CountSegment 1 10 13120 8426 8427 3548 3549 := CountSegment.single (first := 8426) (initial := 3548) row_count_8427 (by decide)
private theorem atomic_8427_agreement : RowAgreement 8426 8427 := RowAgreement.single (first := 8426) modelrow_agrees_8427
private def atomic_8428 : CountSegment 1 10 13120 8427 8428 3549 3550 := CountSegment.single (first := 8427) (initial := 3549) row_count_8428 (by decide)
private theorem atomic_8428_agreement : RowAgreement 8427 8428 := RowAgreement.single (first := 8427) modelrow_agrees_8428
private def atomic_8429 : CountSegment 1 10 13120 8428 8429 3550 3550 := CountSegment.single (first := 8428) (initial := 3550) row_count_8429 (by decide)
private theorem atomic_8429_agreement : RowAgreement 8428 8429 := RowAgreement.single (first := 8428) modelrow_agrees_8429
private def atomic_8430 : CountSegment 1 10 13120 8429 8430 3550 3550 := CountSegment.single (first := 8429) (initial := 3550) row_count_8430 (by decide)
private theorem atomic_8430_agreement : RowAgreement 8429 8430 := RowAgreement.single (first := 8429) modelrow_agrees_8430
private def atomic_8431 : CountSegment 1 10 13120 8430 8431 3550 3550 := CountSegment.single (first := 8430) (initial := 3550) row_count_8431 (by decide)
private theorem atomic_8431_agreement : RowAgreement 8430 8431 := RowAgreement.single (first := 8430) modelrow_agrees_8431
private def atomic_8432 : CountSegment 1 10 13120 8431 8432 3550 3551 := CountSegment.single (first := 8431) (initial := 3550) row_count_8432 (by decide)
private theorem atomic_8432_agreement : RowAgreement 8431 8432 := RowAgreement.single (first := 8431) modelrow_agrees_8432
private def atomic_8433 : CountSegment 1 10 13120 8432 8433 3551 3552 := CountSegment.single (first := 8432) (initial := 3551) row_count_8433 (by decide)
private theorem atomic_8433_agreement : RowAgreement 8432 8433 := RowAgreement.single (first := 8432) modelrow_agrees_8433
private def atomic_8434 : CountSegment 1 10 13120 8433 8434 3552 3553 := CountSegment.single (first := 8433) (initial := 3552) row_count_8434 (by decide)
private theorem atomic_8434_agreement : RowAgreement 8433 8434 := RowAgreement.single (first := 8433) modelrow_agrees_8434
private def atomic_8435 : CountSegment 1 10 13120 8434 8435 3553 3554 := CountSegment.single (first := 8434) (initial := 3553) row_count_8435 (by decide)
private theorem atomic_8435_agreement : RowAgreement 8434 8435 := RowAgreement.single (first := 8434) modelrow_agrees_8435
private def atomic_8436 : CountSegment 1 10 13120 8435 8436 3554 3554 := CountSegment.single (first := 8435) (initial := 3554) row_count_8436 (by decide)
private theorem atomic_8436_agreement : RowAgreement 8435 8436 := RowAgreement.single (first := 8435) modelrow_agrees_8436
private def atomic_8437 : CountSegment 1 10 13120 8436 8437 3554 3554 := CountSegment.single (first := 8436) (initial := 3554) row_count_8437 (by decide)
private theorem atomic_8437_agreement : RowAgreement 8436 8437 := RowAgreement.single (first := 8436) modelrow_agrees_8437
private def atomic_8438 : CountSegment 1 10 13120 8437 8438 3554 3554 := CountSegment.single (first := 8437) (initial := 3554) row_count_8438 (by decide)
private theorem atomic_8438_agreement : RowAgreement 8437 8438 := RowAgreement.single (first := 8437) modelrow_agrees_8438
private def atomic_8439 : CountSegment 1 10 13120 8438 8439 3554 3554 := CountSegment.single (first := 8438) (initial := 3554) row_count_8439 (by decide)
private theorem atomic_8439_agreement : RowAgreement 8438 8439 := RowAgreement.single (first := 8438) modelrow_agrees_8439
private def atomic_8440 : CountSegment 1 10 13120 8439 8440 3554 3556 := CountSegment.single (first := 8439) (initial := 3554) row_count_8440 (by decide)
private theorem atomic_8440_agreement : RowAgreement 8439 8440 := RowAgreement.single (first := 8439) modelrow_agrees_8440
private def atomic_8441 : CountSegment 1 10 13120 8440 8441 3556 3556 := CountSegment.single (first := 8440) (initial := 3556) row_count_8441 (by decide)
private theorem atomic_8441_agreement : RowAgreement 8440 8441 := RowAgreement.single (first := 8440) modelrow_agrees_8441
private def atomic_8442 : CountSegment 1 10 13120 8441 8442 3556 3557 := CountSegment.single (first := 8441) (initial := 3556) row_count_8442 (by decide)
private theorem atomic_8442_agreement : RowAgreement 8441 8442 := RowAgreement.single (first := 8441) modelrow_agrees_8442
private def atomic_8443 : CountSegment 1 10 13120 8442 8443 3557 3557 := CountSegment.single (first := 8442) (initial := 3557) row_count_8443 (by decide)
private theorem atomic_8443_agreement : RowAgreement 8442 8443 := RowAgreement.single (first := 8442) modelrow_agrees_8443
private def atomic_8444 : CountSegment 1 10 13120 8443 8444 3557 3558 := CountSegment.single (first := 8443) (initial := 3557) row_count_8444 (by decide)
private theorem atomic_8444_agreement : RowAgreement 8443 8444 := RowAgreement.single (first := 8443) modelrow_agrees_8444
private def atomic_8445 : CountSegment 1 10 13120 8444 8445 3558 3558 := CountSegment.single (first := 8444) (initial := 3558) row_count_8445 (by decide)
private theorem atomic_8445_agreement : RowAgreement 8444 8445 := RowAgreement.single (first := 8444) modelrow_agrees_8445
private def atomic_8446 : CountSegment 1 10 13120 8445 8446 3558 3558 := CountSegment.single (first := 8445) (initial := 3558) row_count_8446 (by decide)
private theorem atomic_8446_agreement : RowAgreement 8445 8446 := RowAgreement.single (first := 8445) modelrow_agrees_8446
private def atomic_8447 : CountSegment 1 10 13120 8446 8447 3558 3558 := CountSegment.single (first := 8446) (initial := 3558) row_count_8447 (by decide)
private theorem atomic_8447_agreement : RowAgreement 8446 8447 := RowAgreement.single (first := 8446) modelrow_agrees_8447
private def atomic_8448 : CountSegment 1 10 13120 8447 8448 3558 3559 := CountSegment.single (first := 8447) (initial := 3558) row_count_8448 (by decide)
private theorem atomic_8448_agreement : RowAgreement 8447 8448 := RowAgreement.single (first := 8447) modelrow_agrees_8448
private def within_065_000 : CountSegment 1 10 13120 8320 8322 3513 3513 := CountSegment.join atomic_8321 atomic_8322
private theorem within_065_000_agreement : RowAgreement 8320 8322 := RowAgreement.join atomic_8321_agreement atomic_8322_agreement
private def within_065_001 : CountSegment 1 10 13120 8322 8324 3513 3513 := CountSegment.join atomic_8323 atomic_8324
private theorem within_065_001_agreement : RowAgreement 8322 8324 := RowAgreement.join atomic_8323_agreement atomic_8324_agreement
private def within_065_002 : CountSegment 1 10 13120 8320 8324 3513 3513 := CountSegment.join within_065_000 within_065_001
private theorem within_065_002_agreement : RowAgreement 8320 8324 := RowAgreement.join within_065_000_agreement within_065_001_agreement
private def within_065_003 : CountSegment 1 10 13120 8324 8326 3513 3514 := CountSegment.join atomic_8325 atomic_8326
private theorem within_065_003_agreement : RowAgreement 8324 8326 := RowAgreement.join atomic_8325_agreement atomic_8326_agreement
private def within_065_004 : CountSegment 1 10 13120 8326 8328 3514 3514 := CountSegment.join atomic_8327 atomic_8328
private theorem within_065_004_agreement : RowAgreement 8326 8328 := RowAgreement.join atomic_8327_agreement atomic_8328_agreement
private def within_065_005 : CountSegment 1 10 13120 8324 8328 3513 3514 := CountSegment.join within_065_003 within_065_004
private theorem within_065_005_agreement : RowAgreement 8324 8328 := RowAgreement.join within_065_003_agreement within_065_004_agreement
private def within_065_006 : CountSegment 1 10 13120 8320 8328 3513 3514 := CountSegment.join within_065_002 within_065_005
private theorem within_065_006_agreement : RowAgreement 8320 8328 := RowAgreement.join within_065_002_agreement within_065_005_agreement
private def within_065_007 : CountSegment 1 10 13120 8328 8330 3514 3515 := CountSegment.join atomic_8329 atomic_8330
private theorem within_065_007_agreement : RowAgreement 8328 8330 := RowAgreement.join atomic_8329_agreement atomic_8330_agreement
private def within_065_008 : CountSegment 1 10 13120 8330 8332 3515 3516 := CountSegment.join atomic_8331 atomic_8332
private theorem within_065_008_agreement : RowAgreement 8330 8332 := RowAgreement.join atomic_8331_agreement atomic_8332_agreement
private def within_065_009 : CountSegment 1 10 13120 8328 8332 3514 3516 := CountSegment.join within_065_007 within_065_008
private theorem within_065_009_agreement : RowAgreement 8328 8332 := RowAgreement.join within_065_007_agreement within_065_008_agreement
private def within_065_010 : CountSegment 1 10 13120 8332 8334 3516 3518 := CountSegment.join atomic_8333 atomic_8334
private theorem within_065_010_agreement : RowAgreement 8332 8334 := RowAgreement.join atomic_8333_agreement atomic_8334_agreement
private def within_065_011 : CountSegment 1 10 13120 8334 8336 3518 3518 := CountSegment.join atomic_8335 atomic_8336
private theorem within_065_011_agreement : RowAgreement 8334 8336 := RowAgreement.join atomic_8335_agreement atomic_8336_agreement
private def within_065_012 : CountSegment 1 10 13120 8332 8336 3516 3518 := CountSegment.join within_065_010 within_065_011
private theorem within_065_012_agreement : RowAgreement 8332 8336 := RowAgreement.join within_065_010_agreement within_065_011_agreement
private def within_065_013 : CountSegment 1 10 13120 8328 8336 3514 3518 := CountSegment.join within_065_009 within_065_012
private theorem within_065_013_agreement : RowAgreement 8328 8336 := RowAgreement.join within_065_009_agreement within_065_012_agreement
private def within_065_014 : CountSegment 1 10 13120 8320 8336 3513 3518 := CountSegment.join within_065_006 within_065_013
private theorem within_065_014_agreement : RowAgreement 8320 8336 := RowAgreement.join within_065_006_agreement within_065_013_agreement
private def within_065_015 : CountSegment 1 10 13120 8336 8338 3518 3519 := CountSegment.join atomic_8337 atomic_8338
private theorem within_065_015_agreement : RowAgreement 8336 8338 := RowAgreement.join atomic_8337_agreement atomic_8338_agreement
private def within_065_016 : CountSegment 1 10 13120 8338 8340 3519 3520 := CountSegment.join atomic_8339 atomic_8340
private theorem within_065_016_agreement : RowAgreement 8338 8340 := RowAgreement.join atomic_8339_agreement atomic_8340_agreement
private def within_065_017 : CountSegment 1 10 13120 8336 8340 3518 3520 := CountSegment.join within_065_015 within_065_016
private theorem within_065_017_agreement : RowAgreement 8336 8340 := RowAgreement.join within_065_015_agreement within_065_016_agreement
private def within_065_018 : CountSegment 1 10 13120 8340 8342 3520 3520 := CountSegment.join atomic_8341 atomic_8342
private theorem within_065_018_agreement : RowAgreement 8340 8342 := RowAgreement.join atomic_8341_agreement atomic_8342_agreement
private def within_065_019 : CountSegment 1 10 13120 8342 8344 3520 3521 := CountSegment.join atomic_8343 atomic_8344
private theorem within_065_019_agreement : RowAgreement 8342 8344 := RowAgreement.join atomic_8343_agreement atomic_8344_agreement
private def within_065_020 : CountSegment 1 10 13120 8340 8344 3520 3521 := CountSegment.join within_065_018 within_065_019
private theorem within_065_020_agreement : RowAgreement 8340 8344 := RowAgreement.join within_065_018_agreement within_065_019_agreement
private def within_065_021 : CountSegment 1 10 13120 8336 8344 3518 3521 := CountSegment.join within_065_017 within_065_020
private theorem within_065_021_agreement : RowAgreement 8336 8344 := RowAgreement.join within_065_017_agreement within_065_020_agreement
private def within_065_022 : CountSegment 1 10 13120 8344 8346 3521 3522 := CountSegment.join atomic_8345 atomic_8346
private theorem within_065_022_agreement : RowAgreement 8344 8346 := RowAgreement.join atomic_8345_agreement atomic_8346_agreement
private def within_065_023 : CountSegment 1 10 13120 8346 8348 3522 3522 := CountSegment.join atomic_8347 atomic_8348
private theorem within_065_023_agreement : RowAgreement 8346 8348 := RowAgreement.join atomic_8347_agreement atomic_8348_agreement
private def within_065_024 : CountSegment 1 10 13120 8344 8348 3521 3522 := CountSegment.join within_065_022 within_065_023
private theorem within_065_024_agreement : RowAgreement 8344 8348 := RowAgreement.join within_065_022_agreement within_065_023_agreement
private def within_065_025 : CountSegment 1 10 13120 8348 8350 3522 3523 := CountSegment.join atomic_8349 atomic_8350
private theorem within_065_025_agreement : RowAgreement 8348 8350 := RowAgreement.join atomic_8349_agreement atomic_8350_agreement
private def within_065_026 : CountSegment 1 10 13120 8350 8352 3523 3524 := CountSegment.join atomic_8351 atomic_8352
private theorem within_065_026_agreement : RowAgreement 8350 8352 := RowAgreement.join atomic_8351_agreement atomic_8352_agreement
private def within_065_027 : CountSegment 1 10 13120 8348 8352 3522 3524 := CountSegment.join within_065_025 within_065_026
private theorem within_065_027_agreement : RowAgreement 8348 8352 := RowAgreement.join within_065_025_agreement within_065_026_agreement
private def within_065_028 : CountSegment 1 10 13120 8344 8352 3521 3524 := CountSegment.join within_065_024 within_065_027
private theorem within_065_028_agreement : RowAgreement 8344 8352 := RowAgreement.join within_065_024_agreement within_065_027_agreement
private def within_065_029 : CountSegment 1 10 13120 8336 8352 3518 3524 := CountSegment.join within_065_021 within_065_028
private theorem within_065_029_agreement : RowAgreement 8336 8352 := RowAgreement.join within_065_021_agreement within_065_028_agreement
private def within_065_030 : CountSegment 1 10 13120 8320 8352 3513 3524 := CountSegment.join within_065_014 within_065_029
private theorem within_065_030_agreement : RowAgreement 8320 8352 := RowAgreement.join within_065_014_agreement within_065_029_agreement
private def within_065_031 : CountSegment 1 10 13120 8352 8354 3524 3524 := CountSegment.join atomic_8353 atomic_8354
private theorem within_065_031_agreement : RowAgreement 8352 8354 := RowAgreement.join atomic_8353_agreement atomic_8354_agreement
private def within_065_032 : CountSegment 1 10 13120 8354 8356 3524 3524 := CountSegment.join atomic_8355 atomic_8356
private theorem within_065_032_agreement : RowAgreement 8354 8356 := RowAgreement.join atomic_8355_agreement atomic_8356_agreement
private def within_065_033 : CountSegment 1 10 13120 8352 8356 3524 3524 := CountSegment.join within_065_031 within_065_032
private theorem within_065_033_agreement : RowAgreement 8352 8356 := RowAgreement.join within_065_031_agreement within_065_032_agreement
private def within_065_034 : CountSegment 1 10 13120 8356 8358 3524 3525 := CountSegment.join atomic_8357 atomic_8358
private theorem within_065_034_agreement : RowAgreement 8356 8358 := RowAgreement.join atomic_8357_agreement atomic_8358_agreement
private def within_065_035 : CountSegment 1 10 13120 8358 8360 3525 3526 := CountSegment.join atomic_8359 atomic_8360
private theorem within_065_035_agreement : RowAgreement 8358 8360 := RowAgreement.join atomic_8359_agreement atomic_8360_agreement
private def within_065_036 : CountSegment 1 10 13120 8356 8360 3524 3526 := CountSegment.join within_065_034 within_065_035
private theorem within_065_036_agreement : RowAgreement 8356 8360 := RowAgreement.join within_065_034_agreement within_065_035_agreement
private def within_065_037 : CountSegment 1 10 13120 8352 8360 3524 3526 := CountSegment.join within_065_033 within_065_036
private theorem within_065_037_agreement : RowAgreement 8352 8360 := RowAgreement.join within_065_033_agreement within_065_036_agreement
private def within_065_038 : CountSegment 1 10 13120 8360 8362 3526 3526 := CountSegment.join atomic_8361 atomic_8362
private theorem within_065_038_agreement : RowAgreement 8360 8362 := RowAgreement.join atomic_8361_agreement atomic_8362_agreement
private def within_065_039 : CountSegment 1 10 13120 8362 8364 3526 3526 := CountSegment.join atomic_8363 atomic_8364
private theorem within_065_039_agreement : RowAgreement 8362 8364 := RowAgreement.join atomic_8363_agreement atomic_8364_agreement
private def within_065_040 : CountSegment 1 10 13120 8360 8364 3526 3526 := CountSegment.join within_065_038 within_065_039
private theorem within_065_040_agreement : RowAgreement 8360 8364 := RowAgreement.join within_065_038_agreement within_065_039_agreement
private def within_065_041 : CountSegment 1 10 13120 8364 8366 3526 3526 := CountSegment.join atomic_8365 atomic_8366
private theorem within_065_041_agreement : RowAgreement 8364 8366 := RowAgreement.join atomic_8365_agreement atomic_8366_agreement
private def within_065_042 : CountSegment 1 10 13120 8366 8368 3526 3527 := CountSegment.join atomic_8367 atomic_8368
private theorem within_065_042_agreement : RowAgreement 8366 8368 := RowAgreement.join atomic_8367_agreement atomic_8368_agreement
private def within_065_043 : CountSegment 1 10 13120 8364 8368 3526 3527 := CountSegment.join within_065_041 within_065_042
private theorem within_065_043_agreement : RowAgreement 8364 8368 := RowAgreement.join within_065_041_agreement within_065_042_agreement
private def within_065_044 : CountSegment 1 10 13120 8360 8368 3526 3527 := CountSegment.join within_065_040 within_065_043
private theorem within_065_044_agreement : RowAgreement 8360 8368 := RowAgreement.join within_065_040_agreement within_065_043_agreement
private def within_065_045 : CountSegment 1 10 13120 8352 8368 3524 3527 := CountSegment.join within_065_037 within_065_044
private theorem within_065_045_agreement : RowAgreement 8352 8368 := RowAgreement.join within_065_037_agreement within_065_044_agreement
private def within_065_046 : CountSegment 1 10 13120 8368 8370 3527 3528 := CountSegment.join atomic_8369 atomic_8370
private theorem within_065_046_agreement : RowAgreement 8368 8370 := RowAgreement.join atomic_8369_agreement atomic_8370_agreement
private def within_065_047 : CountSegment 1 10 13120 8370 8372 3528 3528 := CountSegment.join atomic_8371 atomic_8372
private theorem within_065_047_agreement : RowAgreement 8370 8372 := RowAgreement.join atomic_8371_agreement atomic_8372_agreement
private def within_065_048 : CountSegment 1 10 13120 8368 8372 3527 3528 := CountSegment.join within_065_046 within_065_047
private theorem within_065_048_agreement : RowAgreement 8368 8372 := RowAgreement.join within_065_046_agreement within_065_047_agreement
private def within_065_049 : CountSegment 1 10 13120 8372 8374 3528 3529 := CountSegment.join atomic_8373 atomic_8374
private theorem within_065_049_agreement : RowAgreement 8372 8374 := RowAgreement.join atomic_8373_agreement atomic_8374_agreement
private def within_065_050 : CountSegment 1 10 13120 8374 8376 3529 3532 := CountSegment.join atomic_8375 atomic_8376
private theorem within_065_050_agreement : RowAgreement 8374 8376 := RowAgreement.join atomic_8375_agreement atomic_8376_agreement
private def within_065_051 : CountSegment 1 10 13120 8372 8376 3528 3532 := CountSegment.join within_065_049 within_065_050
private theorem within_065_051_agreement : RowAgreement 8372 8376 := RowAgreement.join within_065_049_agreement within_065_050_agreement
private def within_065_052 : CountSegment 1 10 13120 8368 8376 3527 3532 := CountSegment.join within_065_048 within_065_051
private theorem within_065_052_agreement : RowAgreement 8368 8376 := RowAgreement.join within_065_048_agreement within_065_051_agreement
private def within_065_053 : CountSegment 1 10 13120 8376 8378 3532 3534 := CountSegment.join atomic_8377 atomic_8378
private theorem within_065_053_agreement : RowAgreement 8376 8378 := RowAgreement.join atomic_8377_agreement atomic_8378_agreement
private def within_065_054 : CountSegment 1 10 13120 8378 8380 3534 3535 := CountSegment.join atomic_8379 atomic_8380
private theorem within_065_054_agreement : RowAgreement 8378 8380 := RowAgreement.join atomic_8379_agreement atomic_8380_agreement
private def within_065_055 : CountSegment 1 10 13120 8376 8380 3532 3535 := CountSegment.join within_065_053 within_065_054
private theorem within_065_055_agreement : RowAgreement 8376 8380 := RowAgreement.join within_065_053_agreement within_065_054_agreement
private def within_065_056 : CountSegment 1 10 13120 8380 8382 3535 3535 := CountSegment.join atomic_8381 atomic_8382
private theorem within_065_056_agreement : RowAgreement 8380 8382 := RowAgreement.join atomic_8381_agreement atomic_8382_agreement
private def within_065_057 : CountSegment 1 10 13120 8382 8384 3535 3535 := CountSegment.join atomic_8383 atomic_8384
private theorem within_065_057_agreement : RowAgreement 8382 8384 := RowAgreement.join atomic_8383_agreement atomic_8384_agreement
private def within_065_058 : CountSegment 1 10 13120 8380 8384 3535 3535 := CountSegment.join within_065_056 within_065_057
private theorem within_065_058_agreement : RowAgreement 8380 8384 := RowAgreement.join within_065_056_agreement within_065_057_agreement
private def within_065_059 : CountSegment 1 10 13120 8376 8384 3532 3535 := CountSegment.join within_065_055 within_065_058
private theorem within_065_059_agreement : RowAgreement 8376 8384 := RowAgreement.join within_065_055_agreement within_065_058_agreement
private def within_065_060 : CountSegment 1 10 13120 8368 8384 3527 3535 := CountSegment.join within_065_052 within_065_059
private theorem within_065_060_agreement : RowAgreement 8368 8384 := RowAgreement.join within_065_052_agreement within_065_059_agreement
private def within_065_061 : CountSegment 1 10 13120 8352 8384 3524 3535 := CountSegment.join within_065_045 within_065_060
private theorem within_065_061_agreement : RowAgreement 8352 8384 := RowAgreement.join within_065_045_agreement within_065_060_agreement
private def within_065_062 : CountSegment 1 10 13120 8320 8384 3513 3535 := CountSegment.join within_065_030 within_065_061
private theorem within_065_062_agreement : RowAgreement 8320 8384 := RowAgreement.join within_065_030_agreement within_065_061_agreement
private def within_065_063 : CountSegment 1 10 13120 8384 8386 3535 3538 := CountSegment.join atomic_8385 atomic_8386
private theorem within_065_063_agreement : RowAgreement 8384 8386 := RowAgreement.join atomic_8385_agreement atomic_8386_agreement
private def within_065_064 : CountSegment 1 10 13120 8386 8388 3538 3538 := CountSegment.join atomic_8387 atomic_8388
private theorem within_065_064_agreement : RowAgreement 8386 8388 := RowAgreement.join atomic_8387_agreement atomic_8388_agreement
private def within_065_065 : CountSegment 1 10 13120 8384 8388 3535 3538 := CountSegment.join within_065_063 within_065_064
private theorem within_065_065_agreement : RowAgreement 8384 8388 := RowAgreement.join within_065_063_agreement within_065_064_agreement
private def within_065_066 : CountSegment 1 10 13120 8388 8390 3538 3539 := CountSegment.join atomic_8389 atomic_8390
private theorem within_065_066_agreement : RowAgreement 8388 8390 := RowAgreement.join atomic_8389_agreement atomic_8390_agreement
private def within_065_067 : CountSegment 1 10 13120 8390 8392 3539 3540 := CountSegment.join atomic_8391 atomic_8392
private theorem within_065_067_agreement : RowAgreement 8390 8392 := RowAgreement.join atomic_8391_agreement atomic_8392_agreement
private def within_065_068 : CountSegment 1 10 13120 8388 8392 3538 3540 := CountSegment.join within_065_066 within_065_067
private theorem within_065_068_agreement : RowAgreement 8388 8392 := RowAgreement.join within_065_066_agreement within_065_067_agreement
private def within_065_069 : CountSegment 1 10 13120 8384 8392 3535 3540 := CountSegment.join within_065_065 within_065_068
private theorem within_065_069_agreement : RowAgreement 8384 8392 := RowAgreement.join within_065_065_agreement within_065_068_agreement
private def within_065_070 : CountSegment 1 10 13120 8392 8394 3540 3540 := CountSegment.join atomic_8393 atomic_8394
private theorem within_065_070_agreement : RowAgreement 8392 8394 := RowAgreement.join atomic_8393_agreement atomic_8394_agreement
private def within_065_071 : CountSegment 1 10 13120 8394 8396 3540 3541 := CountSegment.join atomic_8395 atomic_8396
private theorem within_065_071_agreement : RowAgreement 8394 8396 := RowAgreement.join atomic_8395_agreement atomic_8396_agreement
private def within_065_072 : CountSegment 1 10 13120 8392 8396 3540 3541 := CountSegment.join within_065_070 within_065_071
private theorem within_065_072_agreement : RowAgreement 8392 8396 := RowAgreement.join within_065_070_agreement within_065_071_agreement
private def within_065_073 : CountSegment 1 10 13120 8396 8398 3541 3542 := CountSegment.join atomic_8397 atomic_8398
private theorem within_065_073_agreement : RowAgreement 8396 8398 := RowAgreement.join atomic_8397_agreement atomic_8398_agreement
private def within_065_074 : CountSegment 1 10 13120 8398 8400 3542 3542 := CountSegment.join atomic_8399 atomic_8400
private theorem within_065_074_agreement : RowAgreement 8398 8400 := RowAgreement.join atomic_8399_agreement atomic_8400_agreement
private def within_065_075 : CountSegment 1 10 13120 8396 8400 3541 3542 := CountSegment.join within_065_073 within_065_074
private theorem within_065_075_agreement : RowAgreement 8396 8400 := RowAgreement.join within_065_073_agreement within_065_074_agreement
private def within_065_076 : CountSegment 1 10 13120 8392 8400 3540 3542 := CountSegment.join within_065_072 within_065_075
private theorem within_065_076_agreement : RowAgreement 8392 8400 := RowAgreement.join within_065_072_agreement within_065_075_agreement
private def within_065_077 : CountSegment 1 10 13120 8384 8400 3535 3542 := CountSegment.join within_065_069 within_065_076
private theorem within_065_077_agreement : RowAgreement 8384 8400 := RowAgreement.join within_065_069_agreement within_065_076_agreement
private def within_065_078 : CountSegment 1 10 13120 8400 8402 3542 3542 := CountSegment.join atomic_8401 atomic_8402
private theorem within_065_078_agreement : RowAgreement 8400 8402 := RowAgreement.join atomic_8401_agreement atomic_8402_agreement
private def within_065_079 : CountSegment 1 10 13120 8402 8404 3542 3542 := CountSegment.join atomic_8403 atomic_8404
private theorem within_065_079_agreement : RowAgreement 8402 8404 := RowAgreement.join atomic_8403_agreement atomic_8404_agreement
private def within_065_080 : CountSegment 1 10 13120 8400 8404 3542 3542 := CountSegment.join within_065_078 within_065_079
private theorem within_065_080_agreement : RowAgreement 8400 8404 := RowAgreement.join within_065_078_agreement within_065_079_agreement
private def within_065_081 : CountSegment 1 10 13120 8404 8406 3542 3542 := CountSegment.join atomic_8405 atomic_8406
private theorem within_065_081_agreement : RowAgreement 8404 8406 := RowAgreement.join atomic_8405_agreement atomic_8406_agreement
private def within_065_082 : CountSegment 1 10 13120 8406 8408 3542 3544 := CountSegment.join atomic_8407 atomic_8408
private theorem within_065_082_agreement : RowAgreement 8406 8408 := RowAgreement.join atomic_8407_agreement atomic_8408_agreement
private def within_065_083 : CountSegment 1 10 13120 8404 8408 3542 3544 := CountSegment.join within_065_081 within_065_082
private theorem within_065_083_agreement : RowAgreement 8404 8408 := RowAgreement.join within_065_081_agreement within_065_082_agreement
private def within_065_084 : CountSegment 1 10 13120 8400 8408 3542 3544 := CountSegment.join within_065_080 within_065_083
private theorem within_065_084_agreement : RowAgreement 8400 8408 := RowAgreement.join within_065_080_agreement within_065_083_agreement
private def within_065_085 : CountSegment 1 10 13120 8408 8410 3544 3545 := CountSegment.join atomic_8409 atomic_8410
private theorem within_065_085_agreement : RowAgreement 8408 8410 := RowAgreement.join atomic_8409_agreement atomic_8410_agreement
private def within_065_086 : CountSegment 1 10 13120 8410 8412 3545 3545 := CountSegment.join atomic_8411 atomic_8412
private theorem within_065_086_agreement : RowAgreement 8410 8412 := RowAgreement.join atomic_8411_agreement atomic_8412_agreement
private def within_065_087 : CountSegment 1 10 13120 8408 8412 3544 3545 := CountSegment.join within_065_085 within_065_086
private theorem within_065_087_agreement : RowAgreement 8408 8412 := RowAgreement.join within_065_085_agreement within_065_086_agreement
private def within_065_088 : CountSegment 1 10 13120 8412 8414 3545 3545 := CountSegment.join atomic_8413 atomic_8414
private theorem within_065_088_agreement : RowAgreement 8412 8414 := RowAgreement.join atomic_8413_agreement atomic_8414_agreement
private def within_065_089 : CountSegment 1 10 13120 8414 8416 3545 3545 := CountSegment.join atomic_8415 atomic_8416
private theorem within_065_089_agreement : RowAgreement 8414 8416 := RowAgreement.join atomic_8415_agreement atomic_8416_agreement
private def within_065_090 : CountSegment 1 10 13120 8412 8416 3545 3545 := CountSegment.join within_065_088 within_065_089
private theorem within_065_090_agreement : RowAgreement 8412 8416 := RowAgreement.join within_065_088_agreement within_065_089_agreement
private def within_065_091 : CountSegment 1 10 13120 8408 8416 3544 3545 := CountSegment.join within_065_087 within_065_090
private theorem within_065_091_agreement : RowAgreement 8408 8416 := RowAgreement.join within_065_087_agreement within_065_090_agreement
private def within_065_092 : CountSegment 1 10 13120 8400 8416 3542 3545 := CountSegment.join within_065_084 within_065_091
private theorem within_065_092_agreement : RowAgreement 8400 8416 := RowAgreement.join within_065_084_agreement within_065_091_agreement
private def within_065_093 : CountSegment 1 10 13120 8384 8416 3535 3545 := CountSegment.join within_065_077 within_065_092
private theorem within_065_093_agreement : RowAgreement 8384 8416 := RowAgreement.join within_065_077_agreement within_065_092_agreement
private def within_065_094 : CountSegment 1 10 13120 8416 8418 3545 3547 := CountSegment.join atomic_8417 atomic_8418
private theorem within_065_094_agreement : RowAgreement 8416 8418 := RowAgreement.join atomic_8417_agreement atomic_8418_agreement
private def within_065_095 : CountSegment 1 10 13120 8418 8420 3547 3547 := CountSegment.join atomic_8419 atomic_8420
private theorem within_065_095_agreement : RowAgreement 8418 8420 := RowAgreement.join atomic_8419_agreement atomic_8420_agreement
private def within_065_096 : CountSegment 1 10 13120 8416 8420 3545 3547 := CountSegment.join within_065_094 within_065_095
private theorem within_065_096_agreement : RowAgreement 8416 8420 := RowAgreement.join within_065_094_agreement within_065_095_agreement
private def within_065_097 : CountSegment 1 10 13120 8420 8422 3547 3547 := CountSegment.join atomic_8421 atomic_8422
private theorem within_065_097_agreement : RowAgreement 8420 8422 := RowAgreement.join atomic_8421_agreement atomic_8422_agreement
private def within_065_098 : CountSegment 1 10 13120 8422 8424 3547 3548 := CountSegment.join atomic_8423 atomic_8424
private theorem within_065_098_agreement : RowAgreement 8422 8424 := RowAgreement.join atomic_8423_agreement atomic_8424_agreement
private def within_065_099 : CountSegment 1 10 13120 8420 8424 3547 3548 := CountSegment.join within_065_097 within_065_098
private theorem within_065_099_agreement : RowAgreement 8420 8424 := RowAgreement.join within_065_097_agreement within_065_098_agreement
private def within_065_100 : CountSegment 1 10 13120 8416 8424 3545 3548 := CountSegment.join within_065_096 within_065_099
private theorem within_065_100_agreement : RowAgreement 8416 8424 := RowAgreement.join within_065_096_agreement within_065_099_agreement
private def within_065_101 : CountSegment 1 10 13120 8424 8426 3548 3548 := CountSegment.join atomic_8425 atomic_8426
private theorem within_065_101_agreement : RowAgreement 8424 8426 := RowAgreement.join atomic_8425_agreement atomic_8426_agreement
private def within_065_102 : CountSegment 1 10 13120 8426 8428 3548 3550 := CountSegment.join atomic_8427 atomic_8428
private theorem within_065_102_agreement : RowAgreement 8426 8428 := RowAgreement.join atomic_8427_agreement atomic_8428_agreement
private def within_065_103 : CountSegment 1 10 13120 8424 8428 3548 3550 := CountSegment.join within_065_101 within_065_102
private theorem within_065_103_agreement : RowAgreement 8424 8428 := RowAgreement.join within_065_101_agreement within_065_102_agreement
private def within_065_104 : CountSegment 1 10 13120 8428 8430 3550 3550 := CountSegment.join atomic_8429 atomic_8430
private theorem within_065_104_agreement : RowAgreement 8428 8430 := RowAgreement.join atomic_8429_agreement atomic_8430_agreement
private def within_065_105 : CountSegment 1 10 13120 8430 8432 3550 3551 := CountSegment.join atomic_8431 atomic_8432
private theorem within_065_105_agreement : RowAgreement 8430 8432 := RowAgreement.join atomic_8431_agreement atomic_8432_agreement
private def within_065_106 : CountSegment 1 10 13120 8428 8432 3550 3551 := CountSegment.join within_065_104 within_065_105
private theorem within_065_106_agreement : RowAgreement 8428 8432 := RowAgreement.join within_065_104_agreement within_065_105_agreement
private def within_065_107 : CountSegment 1 10 13120 8424 8432 3548 3551 := CountSegment.join within_065_103 within_065_106
private theorem within_065_107_agreement : RowAgreement 8424 8432 := RowAgreement.join within_065_103_agreement within_065_106_agreement
private def within_065_108 : CountSegment 1 10 13120 8416 8432 3545 3551 := CountSegment.join within_065_100 within_065_107
private theorem within_065_108_agreement : RowAgreement 8416 8432 := RowAgreement.join within_065_100_agreement within_065_107_agreement
private def within_065_109 : CountSegment 1 10 13120 8432 8434 3551 3553 := CountSegment.join atomic_8433 atomic_8434
private theorem within_065_109_agreement : RowAgreement 8432 8434 := RowAgreement.join atomic_8433_agreement atomic_8434_agreement
private def within_065_110 : CountSegment 1 10 13120 8434 8436 3553 3554 := CountSegment.join atomic_8435 atomic_8436
private theorem within_065_110_agreement : RowAgreement 8434 8436 := RowAgreement.join atomic_8435_agreement atomic_8436_agreement
private def within_065_111 : CountSegment 1 10 13120 8432 8436 3551 3554 := CountSegment.join within_065_109 within_065_110
private theorem within_065_111_agreement : RowAgreement 8432 8436 := RowAgreement.join within_065_109_agreement within_065_110_agreement
private def within_065_112 : CountSegment 1 10 13120 8436 8438 3554 3554 := CountSegment.join atomic_8437 atomic_8438
private theorem within_065_112_agreement : RowAgreement 8436 8438 := RowAgreement.join atomic_8437_agreement atomic_8438_agreement
private def within_065_113 : CountSegment 1 10 13120 8438 8440 3554 3556 := CountSegment.join atomic_8439 atomic_8440
private theorem within_065_113_agreement : RowAgreement 8438 8440 := RowAgreement.join atomic_8439_agreement atomic_8440_agreement
private def within_065_114 : CountSegment 1 10 13120 8436 8440 3554 3556 := CountSegment.join within_065_112 within_065_113
private theorem within_065_114_agreement : RowAgreement 8436 8440 := RowAgreement.join within_065_112_agreement within_065_113_agreement
private def within_065_115 : CountSegment 1 10 13120 8432 8440 3551 3556 := CountSegment.join within_065_111 within_065_114
private theorem within_065_115_agreement : RowAgreement 8432 8440 := RowAgreement.join within_065_111_agreement within_065_114_agreement
private def within_065_116 : CountSegment 1 10 13120 8440 8442 3556 3557 := CountSegment.join atomic_8441 atomic_8442
private theorem within_065_116_agreement : RowAgreement 8440 8442 := RowAgreement.join atomic_8441_agreement atomic_8442_agreement
private def within_065_117 : CountSegment 1 10 13120 8442 8444 3557 3558 := CountSegment.join atomic_8443 atomic_8444
private theorem within_065_117_agreement : RowAgreement 8442 8444 := RowAgreement.join atomic_8443_agreement atomic_8444_agreement
private def within_065_118 : CountSegment 1 10 13120 8440 8444 3556 3558 := CountSegment.join within_065_116 within_065_117
private theorem within_065_118_agreement : RowAgreement 8440 8444 := RowAgreement.join within_065_116_agreement within_065_117_agreement
private def within_065_119 : CountSegment 1 10 13120 8444 8446 3558 3558 := CountSegment.join atomic_8445 atomic_8446
private theorem within_065_119_agreement : RowAgreement 8444 8446 := RowAgreement.join atomic_8445_agreement atomic_8446_agreement
private def within_065_120 : CountSegment 1 10 13120 8446 8448 3558 3559 := CountSegment.join atomic_8447 atomic_8448
private theorem within_065_120_agreement : RowAgreement 8446 8448 := RowAgreement.join atomic_8447_agreement atomic_8448_agreement
private def within_065_121 : CountSegment 1 10 13120 8444 8448 3558 3559 := CountSegment.join within_065_119 within_065_120
private theorem within_065_121_agreement : RowAgreement 8444 8448 := RowAgreement.join within_065_119_agreement within_065_120_agreement
private def within_065_122 : CountSegment 1 10 13120 8440 8448 3556 3559 := CountSegment.join within_065_118 within_065_121
private theorem within_065_122_agreement : RowAgreement 8440 8448 := RowAgreement.join within_065_118_agreement within_065_121_agreement
private def within_065_123 : CountSegment 1 10 13120 8432 8448 3551 3559 := CountSegment.join within_065_115 within_065_122
private theorem within_065_123_agreement : RowAgreement 8432 8448 := RowAgreement.join within_065_115_agreement within_065_122_agreement
private def within_065_124 : CountSegment 1 10 13120 8416 8448 3545 3559 := CountSegment.join within_065_108 within_065_123
private theorem within_065_124_agreement : RowAgreement 8416 8448 := RowAgreement.join within_065_108_agreement within_065_123_agreement
private def within_065_125 : CountSegment 1 10 13120 8384 8448 3535 3559 := CountSegment.join within_065_093 within_065_124
private theorem within_065_125_agreement : RowAgreement 8384 8448 := RowAgreement.join within_065_093_agreement within_065_124_agreement
private def within_065_126 : CountSegment 1 10 13120 8320 8448 3513 3559 := CountSegment.join within_065_062 within_065_125
private theorem within_065_126_agreement : RowAgreement 8320 8448 := RowAgreement.join within_065_062_agreement within_065_125_agreement
def segment_065 : CountSegment 1 10 13120 8320 8448 3513 3559 := within_065_126
theorem agreement_065 : RowAgreement 8320 8448 := within_065_126_agreement
end ProjectEulerSolutions.P58Numerical
