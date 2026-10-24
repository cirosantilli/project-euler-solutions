import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9345 : RowCertificate 9345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9345 1) := by change ¬ Nat.Prime 349334791; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26871907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9345 2) := by change ¬ Nat.Prime 349316101; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 287977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9345 3) := by change ¬ Nat.Prime 349297411; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3391237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9345 : row 9345 = 0 := (row_9345.checked).trans rfl
theorem model_count_9345 : modelRow 9345 = 0 := by decide
theorem modelrow_agrees_9345 : modelRow 9345 = row 9345 := model_count_9345.trans row_count_9345.symm
def row_9346 : RowCertificate 9346 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9346 1) := by change ¬ Nat.Prime 349409557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49915651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9346 2) := by change ¬ Nat.Prime 349390865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69878173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9346 3) := by change ¬ Nat.Prime 349372173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116457391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9346 : row 9346 = 0 := (row_9346.checked).trans rfl
theorem model_count_9346 : modelRow 9346 = 0 := by decide
theorem modelrow_agrees_9346 : modelRow 9346 = row 9346 := model_count_9346.trans row_count_9346.symm
def row_9347 : RowCertificate 9347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9347 1) := by change ¬ Nat.Prime 349484331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116494777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9347 2) := by change Nat.Prime 349465637; exact ProjectEulerSolutions.P58PrimeNodes.p_349465637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9347 3) := by change ¬ Nat.Prime 349446943; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 880219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9347 : row 9347 = 1 := (row_9347.checked).trans rfl
theorem model_count_9347 : modelRow 9347 = 1 := by decide
theorem modelrow_agrees_9347 : modelRow 9347 = row 9347 := model_count_9347.trans row_count_9347.symm
def row_9348 : RowCertificate 9348 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9348 1) := by change ¬ Nat.Prime 349559113; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4788481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9348 2) := by change ¬ Nat.Prime 349540417; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20561201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9348 3) := by change ¬ Nat.Prime 349521721; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9446533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9348 : row 9348 = 0 := (row_9348.checked).trans rfl
theorem model_count_9348 : modelRow 9348 = 0 := by decide
theorem modelrow_agrees_9348 : modelRow 9348 = row 9348 := model_count_9348.trans row_count_9348.symm
def row_9349 : RowCertificate 9349 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9349 1) := by change ¬ Nat.Prime 349633903; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11278513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9349 2) := by change ¬ Nat.Prime 349615205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69923041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9349 3) := by change ¬ Nat.Prime 349596507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116532169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9349 : row 9349 = 0 := (row_9349.checked).trans rfl
theorem model_count_9349 : modelRow 9349 = 0 := by decide
theorem modelrow_agrees_9349 : modelRow 9349 = row 9349 := model_count_9349.trans row_count_9349.symm
def row_9350 : RowCertificate 9350 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9350 1) := by change ¬ Nat.Prime 349708701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116569567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9350 2) := by change Nat.Prime 349690001; exact ProjectEulerSolutions.P58PrimeNodes.p_349690001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9350 3) := by change ¬ Nat.Prime 349671301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49953043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9350 : row 9350 = 1 := (row_9350.checked).trans rfl
theorem model_count_9350 : modelRow 9350 = 1 := by decide
theorem modelrow_agrees_9350 : modelRow 9350 = row 9350 := model_count_9350.trans row_count_9350.symm
def row_9351 : RowCertificate 9351 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9351 1) := by change ¬ Nat.Prime 349783507; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3209023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9351 2) := by change ¬ Nat.Prime 349764805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69952961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9351 3) := by change ¬ Nat.Prime 349746103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49963729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9351 : row 9351 = 0 := (row_9351.checked).trans rfl
theorem model_count_9351 : modelRow 9351 = 0 := by decide
theorem modelrow_agrees_9351 : modelRow 9351 = row 9351 := model_count_9351.trans row_count_9351.symm
def row_9352 : RowCertificate 9352 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9352 1) := by change Nat.Prime 349858321; exact ProjectEulerSolutions.P58PrimeNodes.p_349858321.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9352 2) := by change ¬ Nat.Prime 349839617; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20578801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9352 3) := by change ¬ Nat.Prime 349820913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116606971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9352 : row 9352 = 1 := (row_9352.checked).trans rfl
theorem model_count_9352 : modelRow 9352 = 1 := by decide
theorem modelrow_agrees_9352 : modelRow 9352 = row 9352 := model_count_9352.trans row_count_9352.symm
def row_9353 : RowCertificate 9353 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9353 1) := by change ¬ Nat.Prime 349933143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116644381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9353 2) := by change Nat.Prime 349914437; exact ProjectEulerSolutions.P58PrimeNodes.p_349914437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9353 3) := by change ¬ Nat.Prime 349895731; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 797029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9353 : row 9353 = 1 := (row_9353.checked).trans rfl
theorem model_count_9353 : modelRow 9353 = 1 := by decide
theorem modelrow_agrees_9353 : modelRow 9353 = row 9353 := model_count_9353.trans row_count_9353.symm
def row_9354 : RowCertificate 9354 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9354 1) := by change ¬ Nat.Prime 350007973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50001139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9354 2) := by change ¬ Nat.Prime 349989265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69997853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9354 3) := by change ¬ Nat.Prime 349970557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18419503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9354 : row 9354 = 0 := (row_9354.checked).trans rfl
theorem model_count_9354 : modelRow 9354 = 0 := by decide
theorem modelrow_agrees_9354 : modelRow 9354 = row 9354 := model_count_9354.trans row_count_9354.symm
def row_9355 : RowCertificate 9355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9355 1) := by change ¬ Nat.Prime 350082811; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26929447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9355 2) := by change ¬ Nat.Prime 350064101; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1003049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9355 3) := by change ¬ Nat.Prime 350045391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116681797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9355 : row 9355 = 0 := (row_9355.checked).trans rfl
theorem model_count_9355 : modelRow 9355 = 0 := by decide
theorem modelrow_agrees_9355 : modelRow 9355 = row 9355 := model_count_9355.trans row_count_9355.symm
def row_9356 : RowCertificate 9356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9356 1) := by change ¬ Nat.Prime 350157657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116719219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9356 2) := by change ¬ Nat.Prime 350138945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70027789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9356 3) := by change ¬ Nat.Prime 350120233; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9462709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9356 : row 9356 = 0 := (row_9356.checked).trans rfl
theorem model_count_9356 : modelRow 9356 = 0 := by decide
theorem modelrow_agrees_9356 : modelRow 9356 = row 9356 := model_count_9356.trans row_count_9356.symm
def row_9357 : RowCertificate 9357 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9357 1) := by change ¬ Nat.Prime 350232511; exact PrattCertificates.composite_of_product (factor := 2647) (cofactor := 132313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9357 2) := by change ¬ Nat.Prime 350213797; exact PrattCertificates.composite_of_product (factor := 6481) (cofactor := 54037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9357 3) := by change ¬ Nat.Prime 350195083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50027869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9357 : row 9357 = 0 := (row_9357.checked).trans rfl
theorem model_count_9357 : modelRow 9357 = 0 := by decide
theorem modelrow_agrees_9357 : modelRow 9357 = row 9357 := model_count_9357.trans row_count_9357.symm
def row_9358 : RowCertificate 9358 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9358 1) := by change ¬ Nat.Prime 350307373; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26946721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9358 2) := by change ¬ Nat.Prime 350288657; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9467261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9358 3) := by change ¬ Nat.Prime 350269941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116756647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9358 : row 9358 = 0 := (row_9358.checked).trans rfl
theorem model_count_9358 : modelRow 9358 = 0 := by decide
theorem modelrow_agrees_9358 : modelRow 9358 = row 9358 := model_count_9358.trans row_count_9358.symm
def row_9359 : RowCertificate 9359 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9359 1) := by change ¬ Nat.Prime 350382243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116794081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9359 2) := by change ¬ Nat.Prime 350363525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70072705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9359 3) := by change Nat.Prime 350344807; exact ProjectEulerSolutions.P58PrimeNodes.p_350344807.prime
    exact decide_eq_true prime
theorem row_count_9359 : row 9359 = 1 := (row_9359.checked).trans rfl
theorem model_count_9359 : modelRow 9359 = 1 := by decide
theorem modelrow_agrees_9359 : modelRow 9359 = row 9359 := model_count_9359.trans row_count_9359.symm
def row_9360 : RowCertificate 9360 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9360 1) := by change ¬ Nat.Prime 350457121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50065303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9360 2) := by change Nat.Prime 350438401; exact ProjectEulerSolutions.P58PrimeNodes.p_350438401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9360 3) := by change ¬ Nat.Prime 350419681; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1265053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9360 : row 9360 = 1 := (row_9360.checked).trans rfl
theorem model_count_9360 : modelRow 9360 = 1 := by decide
theorem modelrow_agrees_9360 : modelRow 9360 = row 9360 := model_count_9360.trans row_count_9360.symm
def row_9361 : RowCertificate 9361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9361 1) := by change ¬ Nat.Prime 350532007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50076001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9361 2) := by change ¬ Nat.Prime 350513285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70102657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9361 3) := by change ¬ Nat.Prime 350494563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116831521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9361 : row 9361 = 0 := (row_9361.checked).trans rfl
theorem model_count_9361 : modelRow 9361 = 0 := by decide
theorem modelrow_agrees_9361 : modelRow 9361 = row 9361 := model_count_9361.trans row_count_9361.symm
def row_9362 : RowCertificate 9362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9362 1) := by change ¬ Nat.Prime 350606901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116868967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9362 2) := by change ¬ Nat.Prime 350588177; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3939193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9362 3) := by change ¬ Nat.Prime 350569453; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26966881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9362 : row 9362 = 0 := (row_9362.checked).trans rfl
theorem model_count_9362 : modelRow 9362 = 0 := by decide
theorem modelrow_agrees_9362 : modelRow 9362 = row 9362 := model_count_9362.trans row_count_9362.symm
def row_9363 : RowCertificate 9363 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9363 1) := by change ¬ Nat.Prime 350681803; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18456937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9363 2) := by change ¬ Nat.Prime 350663077; exact PrattCertificates.composite_of_product (factor := 12401) (cofactor := 28277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9363 3) := by change ¬ Nat.Prime 350644351; exact PrattCertificates.composite_of_product (factor := 12157) (cofactor := 28843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9363 : row 9363 = 0 := (row_9363.checked).trans rfl
theorem model_count_9363 : modelRow 9363 = 0 := by decide
theorem modelrow_agrees_9363 : modelRow 9363 = row 9363 := model_count_9363.trans row_count_9363.symm
def row_9364 : RowCertificate 9364 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9364 1) := by change ¬ Nat.Prime 350756713; exact PrattCertificates.composite_of_product (factor := 5953) (cofactor := 58921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9364 2) := by change ¬ Nat.Prime 350737985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70147597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9364 3) := by change ¬ Nat.Prime 350719257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116906419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9364 : row 9364 = 0 := (row_9364.checked).trans rfl
theorem model_count_9364 : modelRow 9364 = 0 := by decide
theorem modelrow_agrees_9364 : modelRow 9364 = row 9364 := model_count_9364.trans row_count_9364.symm
def row_9365 : RowCertificate 9365 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9365 1) := by change ¬ Nat.Prime 350831631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116943877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9365 2) := by change ¬ Nat.Prime 350812901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20636053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9365 3) := by change ¬ Nat.Prime 350794171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50113453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9365 : row 9365 = 0 := (row_9365.checked).trans rfl
theorem model_count_9365 : modelRow 9365 = 0 := by decide
theorem modelrow_agrees_9365 : modelRow 9365 = row 9365 := model_count_9365.trans row_count_9365.symm
def row_9366 : RowCertificate 9366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9366 1) := by change ¬ Nat.Prime 350906557; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9483961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9366 2) := by change ¬ Nat.Prime 350887825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70177565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9366 3) := by change Nat.Prime 350869093; exact ProjectEulerSolutions.P58PrimeNodes.p_350869093.prime
    exact decide_eq_true prime
theorem row_count_9366 : row 9366 = 1 := (row_9366.checked).trans rfl
theorem model_count_9366 : modelRow 9366 = 1 := by decide
theorem modelrow_agrees_9366 : modelRow 9366 = row 9366 := model_count_9366.trans row_count_9366.symm
def row_9367 : RowCertificate 9367 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9367 1) := by change ¬ Nat.Prime 350981491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50140213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9367 2) := by change ¬ Nat.Prime 350962757; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4807709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9367 3) := by change ¬ Nat.Prime 350944023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 116981341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9367 : row 9367 = 0 := (row_9367.checked).trans rfl
theorem model_count_9367 : modelRow 9367 = 0 := by decide
theorem modelrow_agrees_9367 : modelRow 9367 = row 9367 := model_count_9367.trans row_count_9367.symm
def row_9368 : RowCertificate 9368 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9368 1) := by change ¬ Nat.Prime 351056433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117018811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9368 2) := by change ¬ Nat.Prime 351037697; exact PrattCertificates.composite_of_product (factor := 5573) (cofactor := 62989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9368 3) := by change Nat.Prime 351018961; exact ProjectEulerSolutions.P58PrimeNodes.p_351018961.prime
    exact decide_eq_true prime
theorem row_count_9368 : row 9368 = 1 := (row_9368.checked).trans rfl
theorem model_count_9368 : modelRow 9368 = 1 := by decide
theorem modelrow_agrees_9368 : modelRow 9368 = row 9368 := model_count_9368.trans row_count_9368.symm
def row_9369 : RowCertificate 9369 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9369 1) := by change ¬ Nat.Prime 351131383; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 941371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9369 2) := by change ¬ Nat.Prime 351112645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70222529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9369 3) := by change ¬ Nat.Prime 351093907; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 334057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9369 : row 9369 = 0 := (row_9369.checked).trans rfl
theorem model_count_9369 : modelRow 9369 = 0 := by decide
theorem modelrow_agrees_9369 : modelRow 9369 = row 9369 := model_count_9369.trans row_count_9369.symm
def row_9370 : RowCertificate 9370 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9370 1) := by change ¬ Nat.Prime 351206341; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5757481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9370 2) := by change Nat.Prime 351187601; exact ProjectEulerSolutions.P58PrimeNodes.p_351187601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9370 3) := by change ¬ Nat.Prime 351168861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117056287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9370 : row 9370 = 1 := (row_9370.checked).trans rfl
theorem model_count_9370 : modelRow 9370 = 1 := by decide
theorem modelrow_agrees_9370 : modelRow 9370 = row 9370 := model_count_9370.trans row_count_9370.symm
def row_9371 : RowCertificate 9371 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9371 1) := by change ¬ Nat.Prime 351281307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117093769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9371 2) := by change ¬ Nat.Prime 351262565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70252513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9371 3) := by change ¬ Nat.Prime 351243823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50177689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9371 : row 9371 = 0 := (row_9371.checked).trans rfl
theorem model_count_9371 : modelRow 9371 = 0 := by decide
theorem modelrow_agrees_9371 : modelRow 9371 = row 9371 := model_count_9371.trans row_count_9371.symm
def row_9372 : RowCertificate 9372 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9372 1) := by change ¬ Nat.Prime 351356281; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 1144483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9372 2) := by change ¬ Nat.Prime 351337537; exact PrattCertificates.composite_of_product (factor := 2417) (cofactor := 145361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9372 3) := by change ¬ Nat.Prime 351318793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50188399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9372 : row 9372 = 0 := (row_9372.checked).trans rfl
theorem model_count_9372 : modelRow 9372 = 0 := by decide
theorem modelrow_agrees_9372 : modelRow 9372 = row 9372 := model_count_9372.trans row_count_9372.symm
def row_9373 : RowCertificate 9373 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9373 1) := by change ¬ Nat.Prime 351431263; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4448497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9373 2) := by change ¬ Nat.Prime 351412517; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12117673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9373 3) := by change ¬ Nat.Prime 351393771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117131257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9373 : row 9373 = 0 := (row_9373.checked).trans rfl
theorem model_count_9373 : modelRow 9373 = 0 := by decide
theorem modelrow_agrees_9373 : modelRow 9373 = row 9373 := model_count_9373.trans row_count_9373.symm
def row_9374 : RowCertificate 9374 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9374 1) := by change ¬ Nat.Prime 351506253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117168751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9374 2) := by change ¬ Nat.Prime 351487505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70297501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9374 3) := by change ¬ Nat.Prime 351468757; exact PrattCertificates.composite_of_product (factor := 2503) (cofactor := 140419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9374 : row 9374 = 0 := (row_9374.checked).trans rfl
theorem model_count_9374 : modelRow 9374 = 0 := by decide
theorem modelrow_agrees_9374 : modelRow 9374 = row 9374 := model_count_9374.trans row_count_9374.symm
def row_9375 : RowCertificate 9375 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9375 1) := by change ¬ Nat.Prime 351581251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50225893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9375 2) := by change ¬ Nat.Prime 351562501; exact PrattCertificates.composite_of_product (factor := 3917) (cofactor := 89753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9375 3) := by change ¬ Nat.Prime 351543751; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27041827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9375 : row 9375 = 0 := (row_9375.checked).trans rfl
theorem model_count_9375 : modelRow 9375 = 0 := by decide
theorem modelrow_agrees_9375 : modelRow 9375 = row 9375 := model_count_9375.trans row_count_9375.symm
def row_9376 : RowCertificate 9376 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9376 1) := by change ¬ Nat.Prime 351656257; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4817209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9376 2) := by change ¬ Nat.Prime 351637505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70327501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9376 3) := by change ¬ Nat.Prime 351618753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117206251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9376 : row 9376 = 0 := (row_9376.checked).trans rfl
theorem model_count_9376 : modelRow 9376 = 0 := by decide
theorem modelrow_agrees_9376 : modelRow 9376 = row 9376 := model_count_9376.trans row_count_9376.symm
def row_9377 : RowCertificate 9377 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9377 1) := by change ¬ Nat.Prime 351731271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117243757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9377 2) := by change ¬ Nat.Prime 351712517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27054809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9377 3) := by change Nat.Prime 351693763; exact ProjectEulerSolutions.P58PrimeNodes.p_351693763.prime
    exact decide_eq_true prime
theorem row_count_9377 : row 9377 = 1 := (row_9377.checked).trans rfl
theorem model_count_9377 : modelRow 9377 = 1 := by decide
theorem modelrow_agrees_9377 : modelRow 9377 = row 9377 := model_count_9377.trans row_count_9377.symm
def row_9378 : RowCertificate 9378 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9378 1) := by change ¬ Nat.Prime 351806293; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3626869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9378 2) := by change ¬ Nat.Prime 351787537; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 585337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9378 3) := by change ¬ Nat.Prime 351768781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50252683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9378 : row 9378 = 0 := (row_9378.checked).trans rfl
theorem model_count_9378 : modelRow 9378 = 0 := by decide
theorem modelrow_agrees_9378 : modelRow 9378 = row 9378 := model_count_9378.trans row_count_9378.symm
def row_9379 : RowCertificate 9379 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9379 1) := by change Nat.Prime 351881323; exact ProjectEulerSolutions.P58PrimeNodes.p_351881323.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9379 2) := by change ¬ Nat.Prime 351862565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70372513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9379 3) := by change ¬ Nat.Prime 351843807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117281269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9379 : row 9379 = 1 := (row_9379.checked).trans rfl
theorem model_count_9379 : modelRow 9379 = 1 := by decide
theorem modelrow_agrees_9379 : modelRow 9379 = row 9379 := model_count_9379.trans row_count_9379.symm
def row_9380 : RowCertificate 9380 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9380 1) := by change ¬ Nat.Prime 351956361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117318787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9380 2) := by change Nat.Prime 351937601; exact ProjectEulerSolutions.P58PrimeNodes.p_351937601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9380 3) := by change Nat.Prime 351918841; exact ProjectEulerSolutions.P58PrimeNodes.p_351918841.prime
    exact decide_eq_true prime
theorem row_count_9380 : row 9380 = 2 := (row_9380.checked).trans rfl
theorem model_count_9380 : modelRow 9380 = 2 := by decide
theorem modelrow_agrees_9380 : modelRow 9380 = row 9380 := model_count_9380.trans row_count_9380.symm
def row_9381 : RowCertificate 9381 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9381 1) := by change ¬ Nat.Prime 352031407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50290201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9381 2) := by change ¬ Nat.Prime 352012645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70402529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9381 3) := by change Nat.Prime 351993883; exact ProjectEulerSolutions.P58PrimeNodes.p_351993883.prime
    exact decide_eq_true prime
theorem row_count_9381 : row 9381 = 1 := (row_9381.checked).trans rfl
theorem model_count_9381 : modelRow 9381 = 1 := by decide
theorem modelrow_agrees_9381 : modelRow 9381 = row 9381 := model_count_9381.trans row_count_9381.symm
def row_9382 : RowCertificate 9382 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9382 1) := by change ¬ Nat.Prime 352106461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50300923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9382 2) := by change ¬ Nat.Prime 352087697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27083669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9382 3) := by change ¬ Nat.Prime 352068933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117356311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9382 : row 9382 = 0 := (row_9382.checked).trans rfl
theorem model_count_9382 : modelRow 9382 = 0 := by decide
theorem modelrow_agrees_9382 : modelRow 9382 = row 9382 := model_count_9382.trans row_count_9382.symm
def row_9383 : RowCertificate 9383 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9383 1) := by change ¬ Nat.Prime 352181523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117393841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9383 2) := by change ¬ Nat.Prime 352162757; exact PrattCertificates.composite_of_product (factor := 3137) (cofactor := 112261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9383 3) := by change ¬ Nat.Prime 352143991; exact PrattCertificates.composite_of_product (factor := 4597) (cofactor := 76603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9383 : row 9383 = 0 := (row_9383.checked).trans rfl
theorem model_count_9383 : modelRow 9383 = 0 := by decide
theorem modelrow_agrees_9383 : modelRow 9383 = row 9383 := model_count_9383.trans row_count_9383.symm
def row_9384 : RowCertificate 9384 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9384 1) := by change ¬ Nat.Prime 352256593; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27096661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9384 2) := by change ¬ Nat.Prime 352237825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70447565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9384 3) := by change ¬ Nat.Prime 352219057; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 673459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9384 : row 9384 = 0 := (row_9384.checked).trans rfl
theorem model_count_9384 : modelRow 9384 = 0 := by decide
theorem modelrow_agrees_9384 : modelRow 9384 = row 9384 := model_count_9384.trans row_count_9384.symm
def row_9385 : RowCertificate 9385 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9385 1) := by change Nat.Prime 352331671; exact ProjectEulerSolutions.P58PrimeNodes.p_352331671.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9385 2) := by change ¬ Nat.Prime 352312901; exact PrattCertificates.composite_of_product (factor := 3209) (cofactor := 109789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9385 3) := by change ¬ Nat.Prime 352294131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117431377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9385 : row 9385 = 1 := (row_9385.checked).trans rfl
theorem model_count_9385 : modelRow 9385 = 1 := by decide
theorem modelrow_agrees_9385 : modelRow 9385 = row 9385 := model_count_9385.trans row_count_9385.symm
def row_9386 : RowCertificate 9386 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9386 1) := by change ¬ Nat.Prime 352406757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117468919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9386 2) := by change ¬ Nat.Prime 352387985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70477597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9386 3) := by change ¬ Nat.Prime 352369213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50338459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9386 : row 9386 = 0 := (row_9386.checked).trans rfl
theorem model_count_9386 : modelRow 9386 = 0 := by decide
theorem modelrow_agrees_9386 : modelRow 9386 = row 9386 := model_count_9386.trans row_count_9386.symm
def row_9387 : RowCertificate 9387 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9387 1) := by change ¬ Nat.Prime 352481851; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5778391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9387 2) := by change ¬ Nat.Prime 352463077; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 1254317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9387 3) := by change Nat.Prime 352444303; exact ProjectEulerSolutions.P58PrimeNodes.p_352444303.prime
    exact decide_eq_true prime
theorem row_count_9387 : row 9387 = 1 := (row_9387.checked).trans rfl
theorem model_count_9387 : modelRow 9387 = 1 := by decide
theorem modelrow_agrees_9387 : modelRow 9387 = row 9387 := model_count_9387.trans row_count_9387.symm
def row_9388 : RowCertificate 9388 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9388 1) := by change ¬ Nat.Prime 352556953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50365279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9388 2) := by change ¬ Nat.Prime 352538177; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3490477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9388 3) := by change ¬ Nat.Prime 352519401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117506467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9388 : row 9388 = 0 := (row_9388.checked).trans rfl
theorem model_count_9388 : modelRow 9388 = 0 := by decide
theorem modelrow_agrees_9388 : modelRow 9388 = row 9388 := model_count_9388.trans row_count_9388.symm
def row_9389 : RowCertificate 9389 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9389 1) := by change ¬ Nat.Prime 352632063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117544021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9389 2) := by change ¬ Nat.Prime 352613285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70522657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9389 3) := by change Nat.Prime 352594507; exact ProjectEulerSolutions.P58PrimeNodes.p_352594507.prime
    exact decide_eq_true prime
theorem row_count_9389 : row 9389 = 1 := (row_9389.checked).trans rfl
theorem model_count_9389 : modelRow 9389 = 1 := by decide
theorem modelrow_agrees_9389 : modelRow 9389 = row 9389 := model_count_9389.trans row_count_9389.symm
def row_9390 : RowCertificate 9390 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9390 1) := by change ¬ Nat.Prime 352707181; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11377651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9390 2) := by change ¬ Nat.Prime 352688401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27129877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9390 3) := by change ¬ Nat.Prime 352669621; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18561559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9390 : row 9390 = 0 := (row_9390.checked).trans rfl
theorem model_count_9390 : modelRow 9390 = 0 := by decide
theorem modelrow_agrees_9390 : modelRow 9390 = row 9390 := model_count_9390.trans row_count_9390.symm
def row_9391 : RowCertificate 9391 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9391 1) := by change ¬ Nat.Prime 352782307; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3636931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9391 2) := by change ¬ Nat.Prime 352763525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70552705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9391 3) := by change ¬ Nat.Prime 352744743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117581581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9391 : row 9391 = 0 := (row_9391.checked).trans rfl
theorem model_count_9391 : modelRow 9391 = 0 := by decide
theorem modelrow_agrees_9391 : modelRow 9391 = row 9391 := model_count_9391.trans row_count_9391.symm
def row_9392 : RowCertificate 9392 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9392 1) := by change ¬ Nat.Prime 352857441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117619147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9392 2) := by change ¬ Nat.Prime 352838657; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 1514329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9392 3) := by change ¬ Nat.Prime 352819873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50402839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9392 : row 9392 = 0 := (row_9392.checked).trans rfl
theorem model_count_9392 : modelRow 9392 = 0 := by decide
theorem modelrow_agrees_9392 : modelRow 9392 = row 9392 := model_count_9392.trans row_count_9392.symm
def row_9393 : RowCertificate 9393 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9393 1) := by change Nat.Prime 352932583; exact ProjectEulerSolutions.P58PrimeNodes.p_352932583.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9393 2) := by change ¬ Nat.Prime 352913797; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 524389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9393 3) := by change ¬ Nat.Prime 352895011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50413573) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9393 : row 9393 = 1 := (row_9393.checked).trans rfl
theorem model_count_9393 : modelRow 9393 = 1 := by decide
theorem modelrow_agrees_9393 : modelRow 9393 = row 9393 := model_count_9393.trans row_count_9393.symm
def row_9394 : RowCertificate 9394 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9394 1) := by change ¬ Nat.Prime 353007733; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27154441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9394 2) := by change ¬ Nat.Prime 352988945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70597789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9394 3) := by change ¬ Nat.Prime 352970157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117656719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9394 : row 9394 = 0 := (row_9394.checked).trans rfl
theorem model_count_9394 : modelRow 9394 = 0 := by decide
theorem modelrow_agrees_9394 : modelRow 9394 = row 9394 := model_count_9394.trans row_count_9394.symm
def row_9395 : RowCertificate 9395 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9395 1) := by change ¬ Nat.Prime 353082891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117694297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9395 2) := by change ¬ Nat.Prime 353064101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27158777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9395 3) := by change ¬ Nat.Prime 353045311; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5269333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9395 : row 9395 = 0 := (row_9395.checked).trans rfl
theorem model_count_9395 : modelRow 9395 = 0 := by decide
theorem modelrow_agrees_9395 : modelRow 9395 = row 9395 := model_count_9395.trans row_count_9395.symm
def row_9396 : RowCertificate 9396 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9396 1) := by change ¬ Nat.Prime 353158057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50451151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9396 2) := by change ¬ Nat.Prime 353139265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70627853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9396 3) := by change ¬ Nat.Prime 353120473; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11390983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9396 : row 9396 = 0 := (row_9396.checked).trans rfl
theorem model_count_9396 : modelRow 9396 = 0 := by decide
theorem modelrow_agrees_9396 : modelRow 9396 = row 9396 := model_count_9396.trans row_count_9396.symm
def row_9397 : RowCertificate 9397 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9397 1) := by change ¬ Nat.Prime 353233231; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27171787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9397 2) := by change Nat.Prime 353214437; exact ProjectEulerSolutions.P58PrimeNodes.p_353214437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9397 3) := by change ¬ Nat.Prime 353195643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117731881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9397 : row 9397 = 1 := (row_9397.checked).trans rfl
theorem model_count_9397 : modelRow 9397 = 1 := by decide
theorem modelrow_agrees_9397 : modelRow 9397 = row 9397 := model_count_9397.trans row_count_9397.symm
def row_9398 : RowCertificate 9398 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9398 1) := by change ¬ Nat.Prime 353308413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117769471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9398 2) := by change ¬ Nat.Prime 353289617; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3642161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9398 3) := by change ¬ Nat.Prime 353270821; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 478039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9398 : row 9398 = 0 := (row_9398.checked).trans rfl
theorem model_count_9398 : modelRow 9398 = 0 := by decide
theorem modelrow_agrees_9398 : modelRow 9398 = row 9398 := model_count_9398.trans row_count_9398.symm
def row_9399 : RowCertificate 9399 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9399 1) := by change ¬ Nat.Prime 353383603; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18599137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9399 2) := by change ¬ Nat.Prime 353364805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70672961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9399 3) := by change ¬ Nat.Prime 353346007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50478001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9399 : row 9399 = 0 := (row_9399.checked).trans rfl
theorem model_count_9399 : modelRow 9399 = 0 := by decide
theorem modelrow_agrees_9399 : modelRow 9399 = row 9399 := model_count_9399.trans row_count_9399.symm
def row_9400 : RowCertificate 9400 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9400 1) := by change ¬ Nat.Prime 353458801; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 963103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9400 2) := by change ¬ Nat.Prime 353440001; exact PrattCertificates.composite_of_product (factor := 4969) (cofactor := 71129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9400 3) := by change ¬ Nat.Prime 353421201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117807067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9400 : row 9400 = 0 := (row_9400.checked).trans rfl
theorem model_count_9400 : modelRow 9400 = 0 := by decide
theorem modelrow_agrees_9400 : modelRow 9400 = row 9400 := model_count_9400.trans row_count_9400.symm
def row_9401 : RowCertificate 9401 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9401 1) := by change ¬ Nat.Prime 353534007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117844669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9401 2) := by change ¬ Nat.Prime 353515205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70703041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9401 3) := by change ¬ Nat.Prime 353496403; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27192031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9401 : row 9401 = 0 := (row_9401.checked).trans rfl
theorem model_count_9401 : modelRow 9401 = 0 := by decide
theorem modelrow_agrees_9401 : modelRow 9401 = row 9401 := model_count_9401.trans row_count_9401.symm
def row_9402 : RowCertificate 9402 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9402 1) := by change ¬ Nat.Prime 353609221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50515603) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9402 2) := by change ¬ Nat.Prime 353590417; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12192773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9402 3) := by change ¬ Nat.Prime 353571613; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2169151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9402 : row 9402 = 0 := (row_9402.checked).trans rfl
theorem model_count_9402 : modelRow 9402 = 0 := by decide
theorem modelrow_agrees_9402 : modelRow 9402 = row 9402 := model_count_9402.trans row_count_9402.symm
def row_9403 : RowCertificate 9403 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9403 1) := by change ¬ Nat.Prime 353684443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50526349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9403 2) := by change ¬ Nat.Prime 353665637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27205049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9403 3) := by change ¬ Nat.Prime 353646831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117882277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9403 : row 9403 = 0 := (row_9403.checked).trans rfl
theorem model_count_9403 : modelRow 9403 = 0 := by decide
theorem modelrow_agrees_9403 : modelRow 9403 = row 9403 := model_count_9403.trans row_count_9403.symm
def row_9404 : RowCertificate 9404 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9404 1) := by change ¬ Nat.Prime 353759673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117919891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9404 2) := by change ¬ Nat.Prime 353740865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70748173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9404 3) := by change ¬ Nat.Prime 353722057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27209389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9404 : row 9404 = 0 := (row_9404.checked).trans rfl
theorem model_count_9404 : modelRow 9404 = 0 := by decide
theorem modelrow_agrees_9404 : modelRow 9404 = row 9404 := model_count_9404.trans row_count_9404.symm
def row_9405 : RowCertificate 9405 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9405 1) := by change Nat.Prime 353834911; exact ProjectEulerSolutions.P58PrimeNodes.p_353834911.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9405 2) := by change ¬ Nat.Prime 353816101; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 8629661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9405 3) := by change ¬ Nat.Prime 353797291; exact PrattCertificates.composite_of_product (factor := 16963) (cofactor := 20857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9405 : row 9405 = 1 := (row_9405.checked).trans rfl
theorem model_count_9405 : modelRow 9405 = 1 := by decide
theorem modelrow_agrees_9405 : modelRow 9405 = row 9405 := model_count_9405.trans row_count_9405.symm
def row_9406 : RowCertificate 9406 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9406 1) := by change ¬ Nat.Prime 353910157; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1778443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9406 2) := by change ¬ Nat.Prime 353891345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70778269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9406 3) := by change ¬ Nat.Prime 353872533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117957511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9406 : row 9406 = 0 := (row_9406.checked).trans rfl
theorem model_count_9406 : modelRow 9406 = 0 := by decide
theorem modelrow_agrees_9406 : modelRow 9406 = row 9406 := model_count_9406.trans row_count_9406.symm
def row_9407 : RowCertificate 9407 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9407 1) := by change ¬ Nat.Prime 353985411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117995137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9407 2) := by change Nat.Prime 353966597; exact ProjectEulerSolutions.P58PrimeNodes.p_353966597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9407 3) := by change ¬ Nat.Prime 353947783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50563969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9407 : row 9407 = 1 := (row_9407.checked).trans rfl
theorem model_count_9407 : modelRow 9407 = 1 := by decide
theorem modelrow_agrees_9407 : modelRow 9407 = row 9407 := model_count_9407.trans row_count_9407.symm
def row_9408 : RowCertificate 9408 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9408 1) := by change ¬ Nat.Prime 354060673; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 460417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9408 2) := by change ¬ Nat.Prime 354041857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27233989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9408 3) := by change ¬ Nat.Prime 354023041; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2787583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9408 : row 9408 = 0 := (row_9408.checked).trans rfl
theorem model_count_9408 : modelRow 9408 = 0 := by decide
theorem modelrow_agrees_9408 : modelRow 9408 = row 9408 := model_count_9408.trans row_count_9408.symm
def row_9409 : RowCertificate 9409 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9409 1) := by change ¬ Nat.Prime 354135943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50590849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9409 2) := by change ¬ Nat.Prime 354117125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70823425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9409 3) := by change ¬ Nat.Prime 354098307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118032769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9409 : row 9409 = 0 := (row_9409.checked).trans rfl
theorem model_count_9409 : modelRow 9409 = 0 := by decide
theorem modelrow_agrees_9409 : modelRow 9409 = row 9409 := model_count_9409.trans row_count_9409.symm
def row_9410 : RowCertificate 9410 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9410 1) := by change ¬ Nat.Prime 354211221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118070407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9410 2) := by change ¬ Nat.Prime 354192401; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 635893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9410 3) := by change ¬ Nat.Prime 354173581; exact PrattCertificates.composite_of_product (factor := 4507) (cofactor := 78583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9410 : row 9410 = 0 := (row_9410.checked).trans rfl
theorem model_count_9410 : modelRow 9410 = 0 := by decide
theorem modelrow_agrees_9410 : modelRow 9410 = row 9410 := model_count_9410.trans row_count_9410.symm
def row_9411 : RowCertificate 9411 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9411 1) := by change ¬ Nat.Prime 354286507; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11428597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9411 2) := by change ¬ Nat.Prime 354267685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70853537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9411 3) := by change ¬ Nat.Prime 354248863; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18644677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9411 : row 9411 = 0 := (row_9411.checked).trans rfl
theorem model_count_9411 : modelRow 9411 = 0 := by decide
theorem modelrow_agrees_9411 : modelRow 9411 = row 9411 := model_count_9411.trans row_count_9411.symm
def row_9412 : RowCertificate 9412 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9412 1) := by change ¬ Nat.Prime 354361801; exact PrattCertificates.composite_of_product (factor := 643) (cofactor := 551107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9412 2) := by change ¬ Nat.Prime 354342977; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3250853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9412 3) := by change ¬ Nat.Prime 354324153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118108051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9412 : row 9412 = 0 := (row_9412.checked).trans rfl
theorem model_count_9412 : modelRow 9412 = 0 := by decide
theorem modelrow_agrees_9412 : modelRow 9412 = row 9412 := model_count_9412.trans row_count_9412.symm
def row_9413 : RowCertificate 9413 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9413 1) := by change ¬ Nat.Prime 354437103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118145701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9413 2) := by change ¬ Nat.Prime 354418277; exact PrattCertificates.composite_of_product (factor := 4889) (cofactor := 72493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9413 3) := by change ¬ Nat.Prime 354399451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50628493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9413 : row 9413 = 0 := (row_9413.checked).trans rfl
theorem model_count_9413 : modelRow 9413 = 0 := by decide
theorem modelrow_agrees_9413 : modelRow 9413 = row 9413 := model_count_9413.trans row_count_9413.symm
def row_9414 : RowCertificate 9414 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9414 1) := by change ¬ Nat.Prime 354512413; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 283837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9414 2) := by change ¬ Nat.Prime 354493585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70898717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9414 3) := by change ¬ Nat.Prime 354474757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50639251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9414 : row 9414 = 0 := (row_9414.checked).trans rfl
theorem model_count_9414 : modelRow 9414 = 0 := by decide
theorem modelrow_agrees_9414 : modelRow 9414 = row 9414 := model_count_9414.trans row_count_9414.symm
def row_9415 : RowCertificate 9415 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9415 1) := by change Nat.Prime 354587731; exact ProjectEulerSolutions.P58PrimeNodes.p_354587731.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9415 2) := by change Nat.Prime 354568901; exact ProjectEulerSolutions.P58PrimeNodes.p_354568901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9415 3) := by change ¬ Nat.Prime 354550071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118183357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9415 : row 9415 = 2 := (row_9415.checked).trans rfl
theorem model_count_9415 : modelRow 9415 = 2 := by decide
theorem modelrow_agrees_9415 : modelRow 9415 = row 9415 := model_count_9415.trans row_count_9415.symm
def row_9416 : RowCertificate 9416 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9416 1) := by change ¬ Nat.Prime 354663057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118221019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9416 2) := by change ¬ Nat.Prime 354644225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70928845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9416 3) := by change ¬ Nat.Prime 354625393; exact PrattCertificates.composite_of_product (factor := 5827) (cofactor := 60859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9416 : row 9416 = 0 := (row_9416.checked).trans rfl
theorem model_count_9416 : modelRow 9416 = 0 := by decide
theorem modelrow_agrees_9416 : modelRow 9416 = row 9416 := model_count_9416.trans row_count_9416.symm
def row_9417 : RowCertificate 9417 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9417 1) := by change ¬ Nat.Prime 354738391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50676913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9417 2) := by change ¬ Nat.Prime 354719557; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 3985613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9417 3) := by change ¬ Nat.Prime 354700723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27284671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9417 : row 9417 = 0 := (row_9417.checked).trans rfl
theorem model_count_9417 : modelRow 9417 = 0 := by decide
theorem modelrow_agrees_9417 : modelRow 9417 = row 9417 := model_count_9417.trans row_count_9417.symm
def row_9418 : RowCertificate 9418 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9418 1) := by change ¬ Nat.Prime 354813733; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18674407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9418 2) := by change ¬ Nat.Prime 354794897; exact PrattCertificates.composite_of_product (factor := 14753) (cofactor := 24049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9418 3) := by change ¬ Nat.Prime 354776061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118258687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9418 : row 9418 = 0 := (row_9418.checked).trans rfl
theorem model_count_9418 : modelRow 9418 = 0 := by decide
theorem modelrow_agrees_9418 : modelRow 9418 = row 9418 := model_count_9418.trans row_count_9418.symm
def row_9419 : RowCertificate 9419 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9419 1) := by change ¬ Nat.Prime 354889083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118296361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9419 2) := by change ¬ Nat.Prime 354870245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70974049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9419 3) := by change Nat.Prime 354851407; exact ProjectEulerSolutions.P58PrimeNodes.p_354851407.prime
    exact decide_eq_true prime
theorem row_count_9419 : row 9419 = 1 := (row_9419.checked).trans rfl
theorem model_count_9419 : modelRow 9419 = 1 := by decide
theorem modelrow_agrees_9419 : modelRow 9419 = row 9419 := model_count_9419.trans row_count_9419.symm
def row_9420 : RowCertificate 9420 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9420 1) := by change ¬ Nat.Prime 354964441; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27304957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9420 2) := by change ¬ Nat.Prime 354945601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20879153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9420 3) := by change ¬ Nat.Prime 354926761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50703823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9420 : row 9420 = 0 := (row_9420.checked).trans rfl
theorem model_count_9420 : modelRow 9420 = 0 := by decide
theorem modelrow_agrees_9420 : modelRow 9420 = row 9420 := model_count_9420.trans row_count_9420.symm
def row_9421 : RowCertificate 9421 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9421 1) := by change ¬ Nat.Prime 355039807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11452897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9421 2) := by change ¬ Nat.Prime 355020965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71004193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9421 3) := by change ¬ Nat.Prime 355002123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118334041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9421 : row 9421 = 0 := (row_9421.checked).trans rfl
theorem model_count_9421 : modelRow 9421 = 0 := by decide
theorem modelrow_agrees_9421 : modelRow 9421 = row 9421 := model_count_9421.trans row_count_9421.symm
def row_9422 : RowCertificate 9422 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9422 1) := by change ¬ Nat.Prime 355115181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118371727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9422 2) := by change Nat.Prime 355096337; exact ProjectEulerSolutions.P58PrimeNodes.p_355096337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9422 3) := by change ¬ Nat.Prime 355077493; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9596689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9422 : row 9422 = 1 := (row_9422.checked).trans rfl
theorem model_count_9422 : modelRow 9422 = 1 := by decide
theorem modelrow_agrees_9422 : modelRow 9422 = row 9422 := model_count_9422.trans row_count_9422.symm
def row_9423 : RowCertificate 9423 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9423 1) := by change ¬ Nat.Prime 355190563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50741509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9423 2) := by change ¬ Nat.Prime 355171717; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 777181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9423 3) := by change ¬ Nat.Prime 355152871; exact PrattCertificates.composite_of_product (factor := 1747) (cofactor := 203293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9423 : row 9423 = 0 := (row_9423.checked).trans rfl
theorem model_count_9423 : modelRow 9423 = 0 := by decide
theorem modelrow_agrees_9423 : modelRow 9423 = row 9423 := model_count_9423.trans row_count_9423.symm
def row_9424 : RowCertificate 9424 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9424 1) := by change ¬ Nat.Prime 355265953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50752279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9424 2) := by change ¬ Nat.Prime 355247105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71049421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9424 3) := by change ¬ Nat.Prime 355228257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118409419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9424 : row 9424 = 0 := (row_9424.checked).trans rfl
theorem model_count_9424 : modelRow 9424 = 0 := by decide
theorem modelrow_agrees_9424 : modelRow 9424 = row 9424 := model_count_9424.trans row_count_9424.symm
def row_9425 : RowCertificate 9425 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9425 1) := by change ¬ Nat.Prime 355341351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118447117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9425 2) := by change ¬ Nat.Prime 355322501; exact PrattCertificates.composite_of_product (factor := 2141) (cofactor := 165961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9425 3) := by change ¬ Nat.Prime 355303651; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2179777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9425 : row 9425 = 0 := (row_9425.checked).trans rfl
theorem model_count_9425 : modelRow 9425 = 0 := by decide
theorem modelrow_agrees_9425 : modelRow 9425 = row 9425 := model_count_9425.trans row_count_9425.symm
def row_9426 : RowCertificate 9426 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9426 1) := by change ¬ Nat.Prime 355416757; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2263801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9426 2) := by change ¬ Nat.Prime 355397905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71079581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9426 3) := by change Nat.Prime 355379053; exact ProjectEulerSolutions.P58PrimeNodes.p_355379053.prime
    exact decide_eq_true prime
theorem row_count_9426 : row 9426 = 1 := (row_9426.checked).trans rfl
theorem model_count_9426 : modelRow 9426 = 1 := by decide
theorem modelrow_agrees_9426 : modelRow 9426 = row 9426 := model_count_9426.trans row_count_9426.symm
def row_9427 : RowCertificate 9427 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9427 1) := by change ¬ Nat.Prime 355492171; exact PrattCertificates.composite_of_product (factor := 10867) (cofactor := 32713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9427 2) := by change Nat.Prime 355473317; exact ProjectEulerSolutions.P58PrimeNodes.p_355473317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9427 3) := by change ¬ Nat.Prime 355454463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118484821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9427 : row 9427 = 1 := (row_9427.checked).trans rfl
theorem model_count_9427 : modelRow 9427 = 1 := by decide
theorem modelrow_agrees_9427 : modelRow 9427 = row 9427 := model_count_9427.trans row_count_9427.symm
def row_9428 : RowCertificate 9428 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9428 1) := by change ¬ Nat.Prime 355567593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118522531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9428 2) := by change ¬ Nat.Prime 355548737; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3146449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9428 3) := by change ¬ Nat.Prime 355529881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50789983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9428 : row 9428 = 0 := (row_9428.checked).trans rfl
theorem model_count_9428 : modelRow 9428 = 0 := by decide
theorem modelrow_agrees_9428 : modelRow 9428 = row 9428 := model_count_9428.trans row_count_9428.symm
def row_9429 : RowCertificate 9429 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9429 1) := by change ¬ Nat.Prime 355643023; exact PrattCertificates.composite_of_product (factor := 14563) (cofactor := 24421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9429 2) := by change ¬ Nat.Prime 355624165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71124833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9429 3) := by change ¬ Nat.Prime 355605307; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4501333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9429 : row 9429 = 0 := (row_9429.checked).trans rfl
theorem model_count_9429 : modelRow 9429 = 0 := by decide
theorem modelrow_agrees_9429 : modelRow 9429 = row 9429 := model_count_9429.trans row_count_9429.symm
def row_9430 : RowCertificate 9430 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9430 1) := by change ¬ Nat.Prime 355718461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50816923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9430 2) := by change ¬ Nat.Prime 355699601; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 5831141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9430 3) := by change ¬ Nat.Prime 355680741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118560247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9430 : row 9430 = 0 := (row_9430.checked).trans rfl
theorem model_count_9430 : modelRow 9430 = 0 := by decide
theorem modelrow_agrees_9430 : modelRow 9430 = row 9430 := model_count_9430.trans row_count_9430.symm
def row_9431 : RowCertificate 9431 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9431 1) := by change ¬ Nat.Prime 355793907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118597969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9431 2) := by change ¬ Nat.Prime 355775045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71155009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9431 3) := by change ¬ Nat.Prime 355756183; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2559397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9431 : row 9431 = 0 := (row_9431.checked).trans rfl
theorem model_count_9431 : modelRow 9431 = 0 := by decide
theorem modelrow_agrees_9431 : modelRow 9431 = row 9431 := model_count_9431.trans row_count_9431.symm
def row_9432 : RowCertificate 9432 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9432 1) := by change ¬ Nat.Prime 355869361; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5311483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9432 2) := by change ¬ Nat.Prime 355850497; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9617581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9432 3) := by change ¬ Nat.Prime 355831633; exact PrattCertificates.composite_of_product (factor := 1549) (cofactor := 229717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9432 : row 9432 = 0 := (row_9432.checked).trans rfl
theorem model_count_9432 : modelRow 9432 = 0 := by decide
theorem modelrow_agrees_9432 : modelRow 9432 = row 9432 := model_count_9432.trans row_count_9432.symm
def row_9433 : RowCertificate 9433 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9433 1) := by change ¬ Nat.Prime 355944823; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27380371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9433 2) := by change ¬ Nat.Prime 355925957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20936821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9433 3) := by change ¬ Nat.Prime 355907091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118635697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9433 : row 9433 = 0 := (row_9433.checked).trans rfl
theorem model_count_9433 : modelRow 9433 = 0 := by decide
theorem modelrow_agrees_9433 : modelRow 9433 = row 9433 := model_count_9433.trans row_count_9433.symm
def row_9434 : RowCertificate 9434 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9434 1) := by change ¬ Nat.Prime 356020293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118673431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9434 2) := by change ¬ Nat.Prime 356001425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71200285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9434 3) := by change ¬ Nat.Prime 355982557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50854651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9434 : row 9434 = 0 := (row_9434.checked).trans rfl
theorem model_count_9434 : modelRow 9434 = 0 := by decide
theorem modelrow_agrees_9434 : modelRow 9434 = row 9434 := model_count_9434.trans row_count_9434.symm
def row_9435 : RowCertificate 9435 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9435 1) := by change ¬ Nat.Prime 356095771; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8281297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9435 2) := by change ¬ Nat.Prime 356076901; exact PrattCertificates.composite_of_product (factor := 8069) (cofactor := 44129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9435 3) := by change ¬ Nat.Prime 356058031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50865433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9435 : row 9435 = 0 := (row_9435.checked).trans rfl
theorem model_count_9435 : modelRow 9435 = 0 := by decide
theorem modelrow_agrees_9435 : modelRow 9435 = row 9435 := model_count_9435.trans row_count_9435.symm
def row_9436 : RowCertificate 9436 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9436 1) := by change ¬ Nat.Prime 356171257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27397789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9436 2) := by change ¬ Nat.Prime 356152385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71230477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9436 3) := by change ¬ Nat.Prime 356133513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118711171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9436 : row 9436 = 0 := (row_9436.checked).trans rfl
theorem model_count_9436 : modelRow 9436 = 0 := by decide
theorem modelrow_agrees_9436 : modelRow 9436 = row 9436 := model_count_9436.trans row_count_9436.symm
def row_9437 : RowCertificate 9437 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9437 1) := by change ¬ Nat.Prime 356246751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118748917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9437 2) := by change ¬ Nat.Prime 356227877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 20954581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9437 3) := by change ¬ Nat.Prime 356209003; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11490613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9437 : row 9437 = 0 := (row_9437.checked).trans rfl
theorem model_count_9437 : modelRow 9437 = 0 := by decide
theorem modelrow_agrees_9437 : modelRow 9437 = row 9437 := model_count_9437.trans row_count_9437.symm
def row_9438 : RowCertificate 9438 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9438 1) := by change ¬ Nat.Prime 356322253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50903179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9438 2) := by change ¬ Nat.Prime 356303377; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9629821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9438 3) := by change Nat.Prime 356284501; exact ProjectEulerSolutions.P58PrimeNodes.p_356284501.prime
    exact decide_eq_true prime
theorem row_count_9438 : row 9438 = 1 := (row_9438.checked).trans rfl
theorem model_count_9438 : modelRow 9438 = 1 := by decide
theorem modelrow_agrees_9438 : modelRow 9438 = row 9438 := model_count_9438.trans row_count_9438.symm
def row_9439 : RowCertificate 9439 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9439 1) := by change ¬ Nat.Prime 356397763; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18757777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9439 2) := by change ¬ Nat.Prime 356378885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71275777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9439 3) := by change ¬ Nat.Prime 356360007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118786669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9439 : row 9439 = 0 := (row_9439.checked).trans rfl
theorem model_count_9439 : modelRow 9439 = 0 := by decide
theorem modelrow_agrees_9439 : modelRow 9439 = row 9439 := model_count_9439.trans row_count_9439.symm
def row_9440 : RowCertificate 9440 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9440 1) := by change ¬ Nat.Prime 356473281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118824427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9440 2) := by change ¬ Nat.Prime 356454401; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 4882937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9440 3) := by change ¬ Nat.Prime 356435521; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27418117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9440 : row 9440 = 0 := (row_9440.checked).trans rfl
theorem model_count_9440 : modelRow 9440 = 0 := by decide
theorem modelrow_agrees_9440 : modelRow 9440 = row 9440 := model_count_9440.trans row_count_9440.symm
def row_9441 : RowCertificate 9441 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9441 1) := by change ¬ Nat.Prime 356548807; exact PrattCertificates.composite_of_product (factor := 5623) (cofactor := 63409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9441 2) := by change ¬ Nat.Prime 356529925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71305985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9441 3) := by change ¬ Nat.Prime 356511043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50930149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9441 : row 9441 = 0 := (row_9441.checked).trans rfl
theorem model_count_9441 : modelRow 9441 = 0 := by decide
theorem modelrow_agrees_9441 : modelRow 9441 = row 9441 := model_count_9441.trans row_count_9441.symm
def row_9442 : RowCertificate 9442 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9442 1) := by change ¬ Nat.Prime 356624341; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11504011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9442 2) := by change ¬ Nat.Prime 356605457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27431189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9442 3) := by change ¬ Nat.Prime 356586573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118862191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9442 : row 9442 = 0 := (row_9442.checked).trans rfl
theorem model_count_9442 : modelRow 9442 = 0 := by decide
theorem modelrow_agrees_9442 : modelRow 9442 = row 9442 := model_count_9442.trans row_count_9442.symm
def row_9443 : RowCertificate 9443 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9443 1) := by change ¬ Nat.Prime 356699883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118899961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9443 2) := by change ¬ Nat.Prime 356680997; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3156469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9443 3) := by change ¬ Nat.Prime 356662111; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27435547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9443 : row 9443 = 0 := (row_9443.checked).trans rfl
theorem model_count_9443 : modelRow 9443 = 0 := by decide
theorem modelrow_agrees_9443 : modelRow 9443 = row 9443 := model_count_9443.trans row_count_9443.symm
def row_9444 : RowCertificate 9444 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9444 1) := by change ¬ Nat.Prime 356775433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50967919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9444 2) := by change ¬ Nat.Prime 356756545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71351309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9444 3) := by change ¬ Nat.Prime 356737657; exact PrattCertificates.composite_of_product (factor := 6709) (cofactor := 53173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9444 : row 9444 = 0 := (row_9444.checked).trans rfl
theorem model_count_9444 : modelRow 9444 = 0 := by decide
theorem modelrow_agrees_9444 : modelRow 9444 = row 9444 := model_count_9444.trans row_count_9444.symm
def row_9445 : RowCertificate 9445 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9445 1) := by change ¬ Nat.Prime 356850991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 50978713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9445 2) := by change ¬ Nat.Prime 356832101; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3273689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9445 3) := by change ¬ Nat.Prime 356813211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118937737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9445 : row 9445 = 0 := (row_9445.checked).trans rfl
theorem model_count_9445 : modelRow 9445 = 0 := by decide
theorem modelrow_agrees_9445 : modelRow 9445 = row 9445 := model_count_9445.trans row_count_9445.symm
def row_9446 : RowCertificate 9446 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9446 1) := by change ¬ Nat.Prime 356926557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118975519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9446 2) := by change ¬ Nat.Prime 356907665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71381533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9446 3) := by change Nat.Prime 356888773; exact ProjectEulerSolutions.P58PrimeNodes.p_356888773.prime
    exact decide_eq_true prime
theorem row_count_9446 : row 9446 = 1 := (row_9446.checked).trans rfl
theorem model_count_9446 : modelRow 9446 = 1 := by decide
theorem modelrow_agrees_9446 : modelRow 9446 = row 9446 := model_count_9446.trans row_count_9446.symm
def row_9447 : RowCertificate 9447 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9447 1) := by change Nat.Prime 357002131; exact ProjectEulerSolutions.P58PrimeNodes.p_357002131.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9447 2) := by change ¬ Nat.Prime 356983237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27460249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9447 3) := by change ¬ Nat.Prime 356964343; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18787597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9447 : row 9447 = 1 := (row_9447.checked).trans rfl
theorem model_count_9447 : modelRow 9447 = 1 := by decide
theorem modelrow_agrees_9447 : modelRow 9447 = row 9447 := model_count_9447.trans row_count_9447.symm
def row_9448 : RowCertificate 9448 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9448 1) := by change ¬ Nat.Prime 357077713; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9650749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9448 2) := by change ¬ Nat.Prime 357058817; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 12312373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9448 3) := by change ¬ Nat.Prime 357039921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119013307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9448 : row 9448 = 0 := (row_9448.checked).trans rfl
theorem model_count_9448 : modelRow 9448 = 0 := by decide
theorem modelrow_agrees_9448 : modelRow 9448 = row 9448 := model_count_9448.trans row_count_9448.symm
def row_9449 : RowCertificate 9449 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9449 1) := by change ¬ Nat.Prime 357153303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119051101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9449 2) := by change ¬ Nat.Prime 357134405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71426881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9449 3) := by change ¬ Nat.Prime 357115507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51016501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9449 : row 9449 = 0 := (row_9449.checked).trans rfl
theorem model_count_9449 : modelRow 9449 = 0 := by decide
theorem modelrow_agrees_9449 : modelRow 9449 = row 9449 := model_count_9449.trans row_count_9449.symm
def row_9450 : RowCertificate 9450 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9450 1) := by change Nat.Prime 357228901; exact ProjectEulerSolutions.P58PrimeNodes.p_357228901.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9450 2) := by change ¬ Nat.Prime 357210001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21012353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9450 3) := by change Nat.Prime 357191101; exact ProjectEulerSolutions.P58PrimeNodes.p_357191101.prime
    exact decide_eq_true prime
theorem row_count_9450 : row 9450 = 2 := (row_9450.checked).trans rfl
theorem model_count_9450 : modelRow 9450 = 2 := by decide
theorem modelrow_agrees_9450 : modelRow 9450 = row 9450 := model_count_9450.trans row_count_9450.symm
def row_9451 : RowCertificate 9451 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9451 1) := by change ¬ Nat.Prime 357304507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51043501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9451 2) := by change ¬ Nat.Prime 357285605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71457121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9451 3) := by change ¬ Nat.Prime 357266703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119088901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9451 : row 9451 = 0 := (row_9451.checked).trans rfl
theorem model_count_9451 : modelRow 9451 = 0 := by decide
theorem modelrow_agrees_9451 : modelRow 9451 = row 9451 := model_count_9451.trans row_count_9451.symm
def row_9452 : RowCertificate 9452 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9452 1) := by change ¬ Nat.Prime 357380121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119126707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9452 2) := by change ¬ Nat.Prime 357361217; exact PrattCertificates.composite_of_product (factor := 929) (cofactor := 384673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9452 3) := by change ¬ Nat.Prime 357342313; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2813719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9452 : row 9452 = 0 := (row_9452.checked).trans rfl
theorem model_count_9452 : modelRow 9452 = 0 := by decide
theorem modelrow_agrees_9452 : modelRow 9452 = row 9452 := model_count_9452.trans row_count_9452.symm
def row_9453 : RowCertificate 9453 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9453 1) := by change Nat.Prime 357455743; exact ProjectEulerSolutions.P58PrimeNodes.p_357455743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9453 2) := by change ¬ Nat.Prime 357436837; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 1127561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9453 3) := by change ¬ Nat.Prime 357417931; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27493687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9453 : row 9453 = 1 := (row_9453.checked).trans rfl
theorem model_count_9453 : modelRow 9453 = 1 := by decide
theorem modelrow_agrees_9453 : modelRow 9453 = row 9453 := model_count_9453.trans row_count_9453.symm
def row_9454 : RowCertificate 9454 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9454 1) := by change Nat.Prime 357531373; exact ProjectEulerSolutions.P58PrimeNodes.p_357531373.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9454 2) := by change ¬ Nat.Prime 357512465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71502493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9454 3) := by change ¬ Nat.Prime 357493557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119164519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9454 : row 9454 = 1 := (row_9454.checked).trans rfl
theorem model_count_9454 : modelRow 9454 = 1 := by decide
theorem modelrow_agrees_9454 : modelRow 9454 = row 9454 := model_count_9454.trans row_count_9454.symm
def row_9455 : RowCertificate 9455 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9455 1) := by change ¬ Nat.Prime 357607011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119202337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9455 2) := by change ¬ Nat.Prime 357588101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27506777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9455 3) := by change ¬ Nat.Prime 357569191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51081313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9455 : row 9455 = 0 := (row_9455.checked).trans rfl
theorem model_count_9455 : modelRow 9455 = 0 := by decide
theorem modelrow_agrees_9455 : modelRow 9455 = row 9455 := model_count_9455.trans row_count_9455.symm
def row_9456 : RowCertificate 9456 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9456 1) := by change ¬ Nat.Prime 357682657; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18825403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9456 2) := by change ¬ Nat.Prime 357663745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71532749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9456 3) := by change ¬ Nat.Prime 357644833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51092119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9456 : row 9456 = 0 := (row_9456.checked).trans rfl
theorem model_count_9456 : modelRow 9456 = 0 := by decide
theorem modelrow_agrees_9456 : modelRow 9456 = row 9456 := model_count_9456.trans row_count_9456.symm
def row_9457 : RowCertificate 9457 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9457 1) := by change Nat.Prime 357758311; exact ProjectEulerSolutions.P58PrimeNodes.p_357758311.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9457 2) := by change ¬ Nat.Prime 357739397; exact PrattCertificates.composite_of_product (factor := 17093) (cofactor := 20929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9457 3) := by change ¬ Nat.Prime 357720483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119240161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9457 : row 9457 = 1 := (row_9457.checked).trans rfl
theorem model_count_9457 : modelRow 9457 = 1 := by decide
theorem modelrow_agrees_9457 : modelRow 9457 = row 9457 := model_count_9457.trans row_count_9457.symm
def row_9458 : RowCertificate 9458 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9458 1) := by change ¬ Nat.Prime 357833973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119277991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9458 2) := by change ¬ Nat.Prime 357815057; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 849917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9458 3) := by change ¬ Nat.Prime 357796141; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 11541811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9458 : row 9458 = 0 := (row_9458.checked).trans rfl
theorem model_count_9458 : modelRow 9458 = 0 := by decide
theorem modelrow_agrees_9458 : modelRow 9458 = row 9458 := model_count_9458.trans row_count_9458.symm
def row_9459 : RowCertificate 9459 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9459 1) := by change ¬ Nat.Prime 357909643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51129949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9459 2) := by change ¬ Nat.Prime 357890725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71578145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9459 3) := by change ¬ Nat.Prime 357871807; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9672211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9459 : row 9459 = 0 := (row_9459.checked).trans rfl
theorem model_count_9459 : modelRow 9459 = 0 := by decide
theorem modelrow_agrees_9459 : modelRow 9459 = row 9459 := model_count_9459.trans row_count_9459.symm
def row_9460 : RowCertificate 9460 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9460 1) := by change ¬ Nat.Prime 357985321; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3284269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9460 2) := by change ¬ Nat.Prime 357966401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27535877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9460 3) := by change ¬ Nat.Prime 357947481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119315827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9460 : row 9460 = 0 := (row_9460.checked).trans rfl
theorem model_count_9460 : modelRow 9460 = 0 := by decide
theorem modelrow_agrees_9460 : modelRow 9460 = row 9460 := model_count_9460.trans row_count_9460.symm
def row_9461 : RowCertificate 9461 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9461 1) := by change ¬ Nat.Prime 358061007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119353669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9461 2) := by change ¬ Nat.Prime 358042085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71608417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9461 3) := by change ¬ Nat.Prime 358023163; exact PrattCertificates.composite_of_product (factor := 14437) (cofactor := 24799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9461 : row 9461 = 0 := (row_9461.checked).trans rfl
theorem model_count_9461 : modelRow 9461 = 0 := by decide
theorem modelrow_agrees_9461 : modelRow 9461 = row 9461 := model_count_9461.trans row_count_9461.symm
def row_9462 : RowCertificate 9462 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9462 1) := by change ¬ Nat.Prime 358136701; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27548977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9462 2) := by change ¬ Nat.Prime 358117777; exact PrattCertificates.composite_of_product (factor := 15241) (cofactor := 23497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9462 3) := by change ¬ Nat.Prime 358098853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51156979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9462 : row 9462 = 0 := (row_9462.checked).trans rfl
theorem model_count_9462 : modelRow 9462 = 0 := by decide
theorem modelrow_agrees_9462 : modelRow 9462 = row 9462 := model_count_9462.trans row_count_9462.symm
def row_9463 : RowCertificate 9463 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9463 1) := by change ¬ Nat.Prime 358212403; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8330521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9463 2) := by change ¬ Nat.Prime 358193477; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 1818241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9463 3) := by change ¬ Nat.Prime 358174551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119391517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9463 : row 9463 = 0 := (row_9463.checked).trans rfl
theorem model_count_9463 : modelRow 9463 = 0 := by decide
theorem modelrow_agrees_9463 : modelRow 9463 = row 9463 := model_count_9463.trans row_count_9463.symm
def row_9464 : RowCertificate 9464 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9464 1) := by change ¬ Nat.Prime 358288113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119429371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9464 2) := by change ¬ Nat.Prime 358269185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71653837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9464 3) := by change ¬ Nat.Prime 358250257; exact PrattCertificates.composite_of_product (factor := 3061) (cofactor := 117037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9464 : row 9464 = 0 := (row_9464.checked).trans rfl
theorem model_count_9464 : modelRow 9464 = 0 := by decide
theorem modelrow_agrees_9464 : modelRow 9464 = row 9464 := model_count_9464.trans row_count_9464.symm
def row_9465 : RowCertificate 9465 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9465 1) := by change ¬ Nat.Prime 358363831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51194833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9465 2) := by change ¬ Nat.Prime 358344901; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 529313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9465 3) := by change ¬ Nat.Prime 358325971; exact PrattCertificates.composite_of_product (factor := 3691) (cofactor := 97081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9465 : row 9465 = 0 := (row_9465.checked).trans rfl
theorem model_count_9465 : modelRow 9465 = 0 := by decide
theorem modelrow_agrees_9465 : modelRow 9465 = row 9465 := model_count_9465.trans row_count_9465.symm
def row_9466 : RowCertificate 9466 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9466 1) := by change ¬ Nat.Prime 358439557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51205651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9466 2) := by change ¬ Nat.Prime 358420625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71684125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9466 3) := by change ¬ Nat.Prime 358401693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119467231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9466 : row 9466 = 0 := (row_9466.checked).trans rfl
theorem model_count_9466 : modelRow 9466 = 0 := by decide
theorem modelrow_agrees_9466 : modelRow 9466 = row 9466 := model_count_9466.trans row_count_9466.symm
def row_9467 : RowCertificate 9467 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9467 1) := by change ¬ Nat.Prime 358515291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119505097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9467 2) := by change ¬ Nat.Prime 358496357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 21088021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9467 3) := by change ¬ Nat.Prime 358477423; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9688579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9467 : row 9467 = 0 := (row_9467.checked).trans rfl
theorem model_count_9467 : modelRow 9467 = 0 := by decide
theorem modelrow_agrees_9467 : modelRow 9467 = row 9467 := model_count_9467.trans row_count_9467.symm
def row_9468 : RowCertificate 9468 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9468 1) := by change ¬ Nat.Prime 358591033; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4539127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9468 2) := by change ¬ Nat.Prime 358572097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27582469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9468 3) := by change ¬ Nat.Prime 358553161; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18871219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9468 : row 9468 = 0 := (row_9468.checked).trans rfl
theorem model_count_9468 : modelRow 9468 = 0 := by decide
theorem modelrow_agrees_9468 : modelRow 9468 = row 9468 := model_count_9468.trans row_count_9468.symm
def row_9469 : RowCertificate 9469 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9469 1) := by change Nat.Prime 358666783; exact ProjectEulerSolutions.P58PrimeNodes.p_358666783.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9469 2) := by change ¬ Nat.Prime 358647845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71729569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9469 3) := by change ¬ Nat.Prime 358628907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119542969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9469 : row 9469 = 1 := (row_9469.checked).trans rfl
theorem model_count_9469 : modelRow 9469 = 1 := by decide
theorem modelrow_agrees_9469 : modelRow 9469 = row 9469 := model_count_9469.trans row_count_9469.symm
def row_9470 : RowCertificate 9470 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9470 1) := by change ¬ Nat.Prime 358742541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119580847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9470 2) := by change Nat.Prime 358723601; exact ProjectEulerSolutions.P58PrimeNodes.p_358723601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9470 3) := by change ¬ Nat.Prime 358704661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51243523) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9470 : row 9470 = 1 := (row_9470.checked).trans rfl
theorem model_count_9470 : modelRow 9470 = 1 := by decide
theorem modelrow_agrees_9470 : modelRow 9470 = row 9470 := model_count_9470.trans row_count_9470.symm
def row_9471 : RowCertificate 9471 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9471 1) := by change ¬ Nat.Prime 358818307; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 2825341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9471 2) := by change ¬ Nat.Prime 358799365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71759873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9471 3) := by change ¬ Nat.Prime 358780423; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 1323913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9471 : row 9471 = 0 := (row_9471.checked).trans rfl
theorem model_count_9471 : modelRow 9471 = 0 := by decide
theorem modelrow_agrees_9471 : modelRow 9471 = row 9471 := model_count_9471.trans row_count_9471.symm
def row_9472 : RowCertificate 9472 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9472 1) := by change ¬ Nat.Prime 358894081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51270583) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9472 2) := by change ¬ Nat.Prime 358875137; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 6771229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9472 3) := by change ¬ Nat.Prime 358856193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119618731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9472 : row 9472 = 0 := (row_9472.checked).trans rfl
theorem model_count_9472 : modelRow 9472 = 0 := by decide
theorem modelrow_agrees_9472 : modelRow 9472 = row 9472 := model_count_9472.trans row_count_9472.symm
private def atomic_9345 : CountSegment 1 10 13120 9344 9345 3897 3897 := CountSegment.single (first := 9344) (initial := 3897) row_count_9345 (by decide)
private theorem atomic_9345_agreement : RowAgreement 9344 9345 := RowAgreement.single (first := 9344) modelrow_agrees_9345
private def atomic_9346 : CountSegment 1 10 13120 9345 9346 3897 3897 := CountSegment.single (first := 9345) (initial := 3897) row_count_9346 (by decide)
private theorem atomic_9346_agreement : RowAgreement 9345 9346 := RowAgreement.single (first := 9345) modelrow_agrees_9346
private def atomic_9347 : CountSegment 1 10 13120 9346 9347 3897 3898 := CountSegment.single (first := 9346) (initial := 3897) row_count_9347 (by decide)
private theorem atomic_9347_agreement : RowAgreement 9346 9347 := RowAgreement.single (first := 9346) modelrow_agrees_9347
private def atomic_9348 : CountSegment 1 10 13120 9347 9348 3898 3898 := CountSegment.single (first := 9347) (initial := 3898) row_count_9348 (by decide)
private theorem atomic_9348_agreement : RowAgreement 9347 9348 := RowAgreement.single (first := 9347) modelrow_agrees_9348
private def atomic_9349 : CountSegment 1 10 13120 9348 9349 3898 3898 := CountSegment.single (first := 9348) (initial := 3898) row_count_9349 (by decide)
private theorem atomic_9349_agreement : RowAgreement 9348 9349 := RowAgreement.single (first := 9348) modelrow_agrees_9349
private def atomic_9350 : CountSegment 1 10 13120 9349 9350 3898 3899 := CountSegment.single (first := 9349) (initial := 3898) row_count_9350 (by decide)
private theorem atomic_9350_agreement : RowAgreement 9349 9350 := RowAgreement.single (first := 9349) modelrow_agrees_9350
private def atomic_9351 : CountSegment 1 10 13120 9350 9351 3899 3899 := CountSegment.single (first := 9350) (initial := 3899) row_count_9351 (by decide)
private theorem atomic_9351_agreement : RowAgreement 9350 9351 := RowAgreement.single (first := 9350) modelrow_agrees_9351
private def atomic_9352 : CountSegment 1 10 13120 9351 9352 3899 3900 := CountSegment.single (first := 9351) (initial := 3899) row_count_9352 (by decide)
private theorem atomic_9352_agreement : RowAgreement 9351 9352 := RowAgreement.single (first := 9351) modelrow_agrees_9352
private def atomic_9353 : CountSegment 1 10 13120 9352 9353 3900 3901 := CountSegment.single (first := 9352) (initial := 3900) row_count_9353 (by decide)
private theorem atomic_9353_agreement : RowAgreement 9352 9353 := RowAgreement.single (first := 9352) modelrow_agrees_9353
private def atomic_9354 : CountSegment 1 10 13120 9353 9354 3901 3901 := CountSegment.single (first := 9353) (initial := 3901) row_count_9354 (by decide)
private theorem atomic_9354_agreement : RowAgreement 9353 9354 := RowAgreement.single (first := 9353) modelrow_agrees_9354
private def atomic_9355 : CountSegment 1 10 13120 9354 9355 3901 3901 := CountSegment.single (first := 9354) (initial := 3901) row_count_9355 (by decide)
private theorem atomic_9355_agreement : RowAgreement 9354 9355 := RowAgreement.single (first := 9354) modelrow_agrees_9355
private def atomic_9356 : CountSegment 1 10 13120 9355 9356 3901 3901 := CountSegment.single (first := 9355) (initial := 3901) row_count_9356 (by decide)
private theorem atomic_9356_agreement : RowAgreement 9355 9356 := RowAgreement.single (first := 9355) modelrow_agrees_9356
private def atomic_9357 : CountSegment 1 10 13120 9356 9357 3901 3901 := CountSegment.single (first := 9356) (initial := 3901) row_count_9357 (by decide)
private theorem atomic_9357_agreement : RowAgreement 9356 9357 := RowAgreement.single (first := 9356) modelrow_agrees_9357
private def atomic_9358 : CountSegment 1 10 13120 9357 9358 3901 3901 := CountSegment.single (first := 9357) (initial := 3901) row_count_9358 (by decide)
private theorem atomic_9358_agreement : RowAgreement 9357 9358 := RowAgreement.single (first := 9357) modelrow_agrees_9358
private def atomic_9359 : CountSegment 1 10 13120 9358 9359 3901 3902 := CountSegment.single (first := 9358) (initial := 3901) row_count_9359 (by decide)
private theorem atomic_9359_agreement : RowAgreement 9358 9359 := RowAgreement.single (first := 9358) modelrow_agrees_9359
private def atomic_9360 : CountSegment 1 10 13120 9359 9360 3902 3903 := CountSegment.single (first := 9359) (initial := 3902) row_count_9360 (by decide)
private theorem atomic_9360_agreement : RowAgreement 9359 9360 := RowAgreement.single (first := 9359) modelrow_agrees_9360
private def atomic_9361 : CountSegment 1 10 13120 9360 9361 3903 3903 := CountSegment.single (first := 9360) (initial := 3903) row_count_9361 (by decide)
private theorem atomic_9361_agreement : RowAgreement 9360 9361 := RowAgreement.single (first := 9360) modelrow_agrees_9361
private def atomic_9362 : CountSegment 1 10 13120 9361 9362 3903 3903 := CountSegment.single (first := 9361) (initial := 3903) row_count_9362 (by decide)
private theorem atomic_9362_agreement : RowAgreement 9361 9362 := RowAgreement.single (first := 9361) modelrow_agrees_9362
private def atomic_9363 : CountSegment 1 10 13120 9362 9363 3903 3903 := CountSegment.single (first := 9362) (initial := 3903) row_count_9363 (by decide)
private theorem atomic_9363_agreement : RowAgreement 9362 9363 := RowAgreement.single (first := 9362) modelrow_agrees_9363
private def atomic_9364 : CountSegment 1 10 13120 9363 9364 3903 3903 := CountSegment.single (first := 9363) (initial := 3903) row_count_9364 (by decide)
private theorem atomic_9364_agreement : RowAgreement 9363 9364 := RowAgreement.single (first := 9363) modelrow_agrees_9364
private def atomic_9365 : CountSegment 1 10 13120 9364 9365 3903 3903 := CountSegment.single (first := 9364) (initial := 3903) row_count_9365 (by decide)
private theorem atomic_9365_agreement : RowAgreement 9364 9365 := RowAgreement.single (first := 9364) modelrow_agrees_9365
private def atomic_9366 : CountSegment 1 10 13120 9365 9366 3903 3904 := CountSegment.single (first := 9365) (initial := 3903) row_count_9366 (by decide)
private theorem atomic_9366_agreement : RowAgreement 9365 9366 := RowAgreement.single (first := 9365) modelrow_agrees_9366
private def atomic_9367 : CountSegment 1 10 13120 9366 9367 3904 3904 := CountSegment.single (first := 9366) (initial := 3904) row_count_9367 (by decide)
private theorem atomic_9367_agreement : RowAgreement 9366 9367 := RowAgreement.single (first := 9366) modelrow_agrees_9367
private def atomic_9368 : CountSegment 1 10 13120 9367 9368 3904 3905 := CountSegment.single (first := 9367) (initial := 3904) row_count_9368 (by decide)
private theorem atomic_9368_agreement : RowAgreement 9367 9368 := RowAgreement.single (first := 9367) modelrow_agrees_9368
private def atomic_9369 : CountSegment 1 10 13120 9368 9369 3905 3905 := CountSegment.single (first := 9368) (initial := 3905) row_count_9369 (by decide)
private theorem atomic_9369_agreement : RowAgreement 9368 9369 := RowAgreement.single (first := 9368) modelrow_agrees_9369
private def atomic_9370 : CountSegment 1 10 13120 9369 9370 3905 3906 := CountSegment.single (first := 9369) (initial := 3905) row_count_9370 (by decide)
private theorem atomic_9370_agreement : RowAgreement 9369 9370 := RowAgreement.single (first := 9369) modelrow_agrees_9370
private def atomic_9371 : CountSegment 1 10 13120 9370 9371 3906 3906 := CountSegment.single (first := 9370) (initial := 3906) row_count_9371 (by decide)
private theorem atomic_9371_agreement : RowAgreement 9370 9371 := RowAgreement.single (first := 9370) modelrow_agrees_9371
private def atomic_9372 : CountSegment 1 10 13120 9371 9372 3906 3906 := CountSegment.single (first := 9371) (initial := 3906) row_count_9372 (by decide)
private theorem atomic_9372_agreement : RowAgreement 9371 9372 := RowAgreement.single (first := 9371) modelrow_agrees_9372
private def atomic_9373 : CountSegment 1 10 13120 9372 9373 3906 3906 := CountSegment.single (first := 9372) (initial := 3906) row_count_9373 (by decide)
private theorem atomic_9373_agreement : RowAgreement 9372 9373 := RowAgreement.single (first := 9372) modelrow_agrees_9373
private def atomic_9374 : CountSegment 1 10 13120 9373 9374 3906 3906 := CountSegment.single (first := 9373) (initial := 3906) row_count_9374 (by decide)
private theorem atomic_9374_agreement : RowAgreement 9373 9374 := RowAgreement.single (first := 9373) modelrow_agrees_9374
private def atomic_9375 : CountSegment 1 10 13120 9374 9375 3906 3906 := CountSegment.single (first := 9374) (initial := 3906) row_count_9375 (by decide)
private theorem atomic_9375_agreement : RowAgreement 9374 9375 := RowAgreement.single (first := 9374) modelrow_agrees_9375
private def atomic_9376 : CountSegment 1 10 13120 9375 9376 3906 3906 := CountSegment.single (first := 9375) (initial := 3906) row_count_9376 (by decide)
private theorem atomic_9376_agreement : RowAgreement 9375 9376 := RowAgreement.single (first := 9375) modelrow_agrees_9376
private def atomic_9377 : CountSegment 1 10 13120 9376 9377 3906 3907 := CountSegment.single (first := 9376) (initial := 3906) row_count_9377 (by decide)
private theorem atomic_9377_agreement : RowAgreement 9376 9377 := RowAgreement.single (first := 9376) modelrow_agrees_9377
private def atomic_9378 : CountSegment 1 10 13120 9377 9378 3907 3907 := CountSegment.single (first := 9377) (initial := 3907) row_count_9378 (by decide)
private theorem atomic_9378_agreement : RowAgreement 9377 9378 := RowAgreement.single (first := 9377) modelrow_agrees_9378
private def atomic_9379 : CountSegment 1 10 13120 9378 9379 3907 3908 := CountSegment.single (first := 9378) (initial := 3907) row_count_9379 (by decide)
private theorem atomic_9379_agreement : RowAgreement 9378 9379 := RowAgreement.single (first := 9378) modelrow_agrees_9379
private def atomic_9380 : CountSegment 1 10 13120 9379 9380 3908 3910 := CountSegment.single (first := 9379) (initial := 3908) row_count_9380 (by decide)
private theorem atomic_9380_agreement : RowAgreement 9379 9380 := RowAgreement.single (first := 9379) modelrow_agrees_9380
private def atomic_9381 : CountSegment 1 10 13120 9380 9381 3910 3911 := CountSegment.single (first := 9380) (initial := 3910) row_count_9381 (by decide)
private theorem atomic_9381_agreement : RowAgreement 9380 9381 := RowAgreement.single (first := 9380) modelrow_agrees_9381
private def atomic_9382 : CountSegment 1 10 13120 9381 9382 3911 3911 := CountSegment.single (first := 9381) (initial := 3911) row_count_9382 (by decide)
private theorem atomic_9382_agreement : RowAgreement 9381 9382 := RowAgreement.single (first := 9381) modelrow_agrees_9382
private def atomic_9383 : CountSegment 1 10 13120 9382 9383 3911 3911 := CountSegment.single (first := 9382) (initial := 3911) row_count_9383 (by decide)
private theorem atomic_9383_agreement : RowAgreement 9382 9383 := RowAgreement.single (first := 9382) modelrow_agrees_9383
private def atomic_9384 : CountSegment 1 10 13120 9383 9384 3911 3911 := CountSegment.single (first := 9383) (initial := 3911) row_count_9384 (by decide)
private theorem atomic_9384_agreement : RowAgreement 9383 9384 := RowAgreement.single (first := 9383) modelrow_agrees_9384
private def atomic_9385 : CountSegment 1 10 13120 9384 9385 3911 3912 := CountSegment.single (first := 9384) (initial := 3911) row_count_9385 (by decide)
private theorem atomic_9385_agreement : RowAgreement 9384 9385 := RowAgreement.single (first := 9384) modelrow_agrees_9385
private def atomic_9386 : CountSegment 1 10 13120 9385 9386 3912 3912 := CountSegment.single (first := 9385) (initial := 3912) row_count_9386 (by decide)
private theorem atomic_9386_agreement : RowAgreement 9385 9386 := RowAgreement.single (first := 9385) modelrow_agrees_9386
private def atomic_9387 : CountSegment 1 10 13120 9386 9387 3912 3913 := CountSegment.single (first := 9386) (initial := 3912) row_count_9387 (by decide)
private theorem atomic_9387_agreement : RowAgreement 9386 9387 := RowAgreement.single (first := 9386) modelrow_agrees_9387
private def atomic_9388 : CountSegment 1 10 13120 9387 9388 3913 3913 := CountSegment.single (first := 9387) (initial := 3913) row_count_9388 (by decide)
private theorem atomic_9388_agreement : RowAgreement 9387 9388 := RowAgreement.single (first := 9387) modelrow_agrees_9388
private def atomic_9389 : CountSegment 1 10 13120 9388 9389 3913 3914 := CountSegment.single (first := 9388) (initial := 3913) row_count_9389 (by decide)
private theorem atomic_9389_agreement : RowAgreement 9388 9389 := RowAgreement.single (first := 9388) modelrow_agrees_9389
private def atomic_9390 : CountSegment 1 10 13120 9389 9390 3914 3914 := CountSegment.single (first := 9389) (initial := 3914) row_count_9390 (by decide)
private theorem atomic_9390_agreement : RowAgreement 9389 9390 := RowAgreement.single (first := 9389) modelrow_agrees_9390
private def atomic_9391 : CountSegment 1 10 13120 9390 9391 3914 3914 := CountSegment.single (first := 9390) (initial := 3914) row_count_9391 (by decide)
private theorem atomic_9391_agreement : RowAgreement 9390 9391 := RowAgreement.single (first := 9390) modelrow_agrees_9391
private def atomic_9392 : CountSegment 1 10 13120 9391 9392 3914 3914 := CountSegment.single (first := 9391) (initial := 3914) row_count_9392 (by decide)
private theorem atomic_9392_agreement : RowAgreement 9391 9392 := RowAgreement.single (first := 9391) modelrow_agrees_9392
private def atomic_9393 : CountSegment 1 10 13120 9392 9393 3914 3915 := CountSegment.single (first := 9392) (initial := 3914) row_count_9393 (by decide)
private theorem atomic_9393_agreement : RowAgreement 9392 9393 := RowAgreement.single (first := 9392) modelrow_agrees_9393
private def atomic_9394 : CountSegment 1 10 13120 9393 9394 3915 3915 := CountSegment.single (first := 9393) (initial := 3915) row_count_9394 (by decide)
private theorem atomic_9394_agreement : RowAgreement 9393 9394 := RowAgreement.single (first := 9393) modelrow_agrees_9394
private def atomic_9395 : CountSegment 1 10 13120 9394 9395 3915 3915 := CountSegment.single (first := 9394) (initial := 3915) row_count_9395 (by decide)
private theorem atomic_9395_agreement : RowAgreement 9394 9395 := RowAgreement.single (first := 9394) modelrow_agrees_9395
private def atomic_9396 : CountSegment 1 10 13120 9395 9396 3915 3915 := CountSegment.single (first := 9395) (initial := 3915) row_count_9396 (by decide)
private theorem atomic_9396_agreement : RowAgreement 9395 9396 := RowAgreement.single (first := 9395) modelrow_agrees_9396
private def atomic_9397 : CountSegment 1 10 13120 9396 9397 3915 3916 := CountSegment.single (first := 9396) (initial := 3915) row_count_9397 (by decide)
private theorem atomic_9397_agreement : RowAgreement 9396 9397 := RowAgreement.single (first := 9396) modelrow_agrees_9397
private def atomic_9398 : CountSegment 1 10 13120 9397 9398 3916 3916 := CountSegment.single (first := 9397) (initial := 3916) row_count_9398 (by decide)
private theorem atomic_9398_agreement : RowAgreement 9397 9398 := RowAgreement.single (first := 9397) modelrow_agrees_9398
private def atomic_9399 : CountSegment 1 10 13120 9398 9399 3916 3916 := CountSegment.single (first := 9398) (initial := 3916) row_count_9399 (by decide)
private theorem atomic_9399_agreement : RowAgreement 9398 9399 := RowAgreement.single (first := 9398) modelrow_agrees_9399
private def atomic_9400 : CountSegment 1 10 13120 9399 9400 3916 3916 := CountSegment.single (first := 9399) (initial := 3916) row_count_9400 (by decide)
private theorem atomic_9400_agreement : RowAgreement 9399 9400 := RowAgreement.single (first := 9399) modelrow_agrees_9400
private def atomic_9401 : CountSegment 1 10 13120 9400 9401 3916 3916 := CountSegment.single (first := 9400) (initial := 3916) row_count_9401 (by decide)
private theorem atomic_9401_agreement : RowAgreement 9400 9401 := RowAgreement.single (first := 9400) modelrow_agrees_9401
private def atomic_9402 : CountSegment 1 10 13120 9401 9402 3916 3916 := CountSegment.single (first := 9401) (initial := 3916) row_count_9402 (by decide)
private theorem atomic_9402_agreement : RowAgreement 9401 9402 := RowAgreement.single (first := 9401) modelrow_agrees_9402
private def atomic_9403 : CountSegment 1 10 13120 9402 9403 3916 3916 := CountSegment.single (first := 9402) (initial := 3916) row_count_9403 (by decide)
private theorem atomic_9403_agreement : RowAgreement 9402 9403 := RowAgreement.single (first := 9402) modelrow_agrees_9403
private def atomic_9404 : CountSegment 1 10 13120 9403 9404 3916 3916 := CountSegment.single (first := 9403) (initial := 3916) row_count_9404 (by decide)
private theorem atomic_9404_agreement : RowAgreement 9403 9404 := RowAgreement.single (first := 9403) modelrow_agrees_9404
private def atomic_9405 : CountSegment 1 10 13120 9404 9405 3916 3917 := CountSegment.single (first := 9404) (initial := 3916) row_count_9405 (by decide)
private theorem atomic_9405_agreement : RowAgreement 9404 9405 := RowAgreement.single (first := 9404) modelrow_agrees_9405
private def atomic_9406 : CountSegment 1 10 13120 9405 9406 3917 3917 := CountSegment.single (first := 9405) (initial := 3917) row_count_9406 (by decide)
private theorem atomic_9406_agreement : RowAgreement 9405 9406 := RowAgreement.single (first := 9405) modelrow_agrees_9406
private def atomic_9407 : CountSegment 1 10 13120 9406 9407 3917 3918 := CountSegment.single (first := 9406) (initial := 3917) row_count_9407 (by decide)
private theorem atomic_9407_agreement : RowAgreement 9406 9407 := RowAgreement.single (first := 9406) modelrow_agrees_9407
private def atomic_9408 : CountSegment 1 10 13120 9407 9408 3918 3918 := CountSegment.single (first := 9407) (initial := 3918) row_count_9408 (by decide)
private theorem atomic_9408_agreement : RowAgreement 9407 9408 := RowAgreement.single (first := 9407) modelrow_agrees_9408
private def atomic_9409 : CountSegment 1 10 13120 9408 9409 3918 3918 := CountSegment.single (first := 9408) (initial := 3918) row_count_9409 (by decide)
private theorem atomic_9409_agreement : RowAgreement 9408 9409 := RowAgreement.single (first := 9408) modelrow_agrees_9409
private def atomic_9410 : CountSegment 1 10 13120 9409 9410 3918 3918 := CountSegment.single (first := 9409) (initial := 3918) row_count_9410 (by decide)
private theorem atomic_9410_agreement : RowAgreement 9409 9410 := RowAgreement.single (first := 9409) modelrow_agrees_9410
private def atomic_9411 : CountSegment 1 10 13120 9410 9411 3918 3918 := CountSegment.single (first := 9410) (initial := 3918) row_count_9411 (by decide)
private theorem atomic_9411_agreement : RowAgreement 9410 9411 := RowAgreement.single (first := 9410) modelrow_agrees_9411
private def atomic_9412 : CountSegment 1 10 13120 9411 9412 3918 3918 := CountSegment.single (first := 9411) (initial := 3918) row_count_9412 (by decide)
private theorem atomic_9412_agreement : RowAgreement 9411 9412 := RowAgreement.single (first := 9411) modelrow_agrees_9412
private def atomic_9413 : CountSegment 1 10 13120 9412 9413 3918 3918 := CountSegment.single (first := 9412) (initial := 3918) row_count_9413 (by decide)
private theorem atomic_9413_agreement : RowAgreement 9412 9413 := RowAgreement.single (first := 9412) modelrow_agrees_9413
private def atomic_9414 : CountSegment 1 10 13120 9413 9414 3918 3918 := CountSegment.single (first := 9413) (initial := 3918) row_count_9414 (by decide)
private theorem atomic_9414_agreement : RowAgreement 9413 9414 := RowAgreement.single (first := 9413) modelrow_agrees_9414
private def atomic_9415 : CountSegment 1 10 13120 9414 9415 3918 3920 := CountSegment.single (first := 9414) (initial := 3918) row_count_9415 (by decide)
private theorem atomic_9415_agreement : RowAgreement 9414 9415 := RowAgreement.single (first := 9414) modelrow_agrees_9415
private def atomic_9416 : CountSegment 1 10 13120 9415 9416 3920 3920 := CountSegment.single (first := 9415) (initial := 3920) row_count_9416 (by decide)
private theorem atomic_9416_agreement : RowAgreement 9415 9416 := RowAgreement.single (first := 9415) modelrow_agrees_9416
private def atomic_9417 : CountSegment 1 10 13120 9416 9417 3920 3920 := CountSegment.single (first := 9416) (initial := 3920) row_count_9417 (by decide)
private theorem atomic_9417_agreement : RowAgreement 9416 9417 := RowAgreement.single (first := 9416) modelrow_agrees_9417
private def atomic_9418 : CountSegment 1 10 13120 9417 9418 3920 3920 := CountSegment.single (first := 9417) (initial := 3920) row_count_9418 (by decide)
private theorem atomic_9418_agreement : RowAgreement 9417 9418 := RowAgreement.single (first := 9417) modelrow_agrees_9418
private def atomic_9419 : CountSegment 1 10 13120 9418 9419 3920 3921 := CountSegment.single (first := 9418) (initial := 3920) row_count_9419 (by decide)
private theorem atomic_9419_agreement : RowAgreement 9418 9419 := RowAgreement.single (first := 9418) modelrow_agrees_9419
private def atomic_9420 : CountSegment 1 10 13120 9419 9420 3921 3921 := CountSegment.single (first := 9419) (initial := 3921) row_count_9420 (by decide)
private theorem atomic_9420_agreement : RowAgreement 9419 9420 := RowAgreement.single (first := 9419) modelrow_agrees_9420
private def atomic_9421 : CountSegment 1 10 13120 9420 9421 3921 3921 := CountSegment.single (first := 9420) (initial := 3921) row_count_9421 (by decide)
private theorem atomic_9421_agreement : RowAgreement 9420 9421 := RowAgreement.single (first := 9420) modelrow_agrees_9421
private def atomic_9422 : CountSegment 1 10 13120 9421 9422 3921 3922 := CountSegment.single (first := 9421) (initial := 3921) row_count_9422 (by decide)
private theorem atomic_9422_agreement : RowAgreement 9421 9422 := RowAgreement.single (first := 9421) modelrow_agrees_9422
private def atomic_9423 : CountSegment 1 10 13120 9422 9423 3922 3922 := CountSegment.single (first := 9422) (initial := 3922) row_count_9423 (by decide)
private theorem atomic_9423_agreement : RowAgreement 9422 9423 := RowAgreement.single (first := 9422) modelrow_agrees_9423
private def atomic_9424 : CountSegment 1 10 13120 9423 9424 3922 3922 := CountSegment.single (first := 9423) (initial := 3922) row_count_9424 (by decide)
private theorem atomic_9424_agreement : RowAgreement 9423 9424 := RowAgreement.single (first := 9423) modelrow_agrees_9424
private def atomic_9425 : CountSegment 1 10 13120 9424 9425 3922 3922 := CountSegment.single (first := 9424) (initial := 3922) row_count_9425 (by decide)
private theorem atomic_9425_agreement : RowAgreement 9424 9425 := RowAgreement.single (first := 9424) modelrow_agrees_9425
private def atomic_9426 : CountSegment 1 10 13120 9425 9426 3922 3923 := CountSegment.single (first := 9425) (initial := 3922) row_count_9426 (by decide)
private theorem atomic_9426_agreement : RowAgreement 9425 9426 := RowAgreement.single (first := 9425) modelrow_agrees_9426
private def atomic_9427 : CountSegment 1 10 13120 9426 9427 3923 3924 := CountSegment.single (first := 9426) (initial := 3923) row_count_9427 (by decide)
private theorem atomic_9427_agreement : RowAgreement 9426 9427 := RowAgreement.single (first := 9426) modelrow_agrees_9427
private def atomic_9428 : CountSegment 1 10 13120 9427 9428 3924 3924 := CountSegment.single (first := 9427) (initial := 3924) row_count_9428 (by decide)
private theorem atomic_9428_agreement : RowAgreement 9427 9428 := RowAgreement.single (first := 9427) modelrow_agrees_9428
private def atomic_9429 : CountSegment 1 10 13120 9428 9429 3924 3924 := CountSegment.single (first := 9428) (initial := 3924) row_count_9429 (by decide)
private theorem atomic_9429_agreement : RowAgreement 9428 9429 := RowAgreement.single (first := 9428) modelrow_agrees_9429
private def atomic_9430 : CountSegment 1 10 13120 9429 9430 3924 3924 := CountSegment.single (first := 9429) (initial := 3924) row_count_9430 (by decide)
private theorem atomic_9430_agreement : RowAgreement 9429 9430 := RowAgreement.single (first := 9429) modelrow_agrees_9430
private def atomic_9431 : CountSegment 1 10 13120 9430 9431 3924 3924 := CountSegment.single (first := 9430) (initial := 3924) row_count_9431 (by decide)
private theorem atomic_9431_agreement : RowAgreement 9430 9431 := RowAgreement.single (first := 9430) modelrow_agrees_9431
private def atomic_9432 : CountSegment 1 10 13120 9431 9432 3924 3924 := CountSegment.single (first := 9431) (initial := 3924) row_count_9432 (by decide)
private theorem atomic_9432_agreement : RowAgreement 9431 9432 := RowAgreement.single (first := 9431) modelrow_agrees_9432
private def atomic_9433 : CountSegment 1 10 13120 9432 9433 3924 3924 := CountSegment.single (first := 9432) (initial := 3924) row_count_9433 (by decide)
private theorem atomic_9433_agreement : RowAgreement 9432 9433 := RowAgreement.single (first := 9432) modelrow_agrees_9433
private def atomic_9434 : CountSegment 1 10 13120 9433 9434 3924 3924 := CountSegment.single (first := 9433) (initial := 3924) row_count_9434 (by decide)
private theorem atomic_9434_agreement : RowAgreement 9433 9434 := RowAgreement.single (first := 9433) modelrow_agrees_9434
private def atomic_9435 : CountSegment 1 10 13120 9434 9435 3924 3924 := CountSegment.single (first := 9434) (initial := 3924) row_count_9435 (by decide)
private theorem atomic_9435_agreement : RowAgreement 9434 9435 := RowAgreement.single (first := 9434) modelrow_agrees_9435
private def atomic_9436 : CountSegment 1 10 13120 9435 9436 3924 3924 := CountSegment.single (first := 9435) (initial := 3924) row_count_9436 (by decide)
private theorem atomic_9436_agreement : RowAgreement 9435 9436 := RowAgreement.single (first := 9435) modelrow_agrees_9436
private def atomic_9437 : CountSegment 1 10 13120 9436 9437 3924 3924 := CountSegment.single (first := 9436) (initial := 3924) row_count_9437 (by decide)
private theorem atomic_9437_agreement : RowAgreement 9436 9437 := RowAgreement.single (first := 9436) modelrow_agrees_9437
private def atomic_9438 : CountSegment 1 10 13120 9437 9438 3924 3925 := CountSegment.single (first := 9437) (initial := 3924) row_count_9438 (by decide)
private theorem atomic_9438_agreement : RowAgreement 9437 9438 := RowAgreement.single (first := 9437) modelrow_agrees_9438
private def atomic_9439 : CountSegment 1 10 13120 9438 9439 3925 3925 := CountSegment.single (first := 9438) (initial := 3925) row_count_9439 (by decide)
private theorem atomic_9439_agreement : RowAgreement 9438 9439 := RowAgreement.single (first := 9438) modelrow_agrees_9439
private def atomic_9440 : CountSegment 1 10 13120 9439 9440 3925 3925 := CountSegment.single (first := 9439) (initial := 3925) row_count_9440 (by decide)
private theorem atomic_9440_agreement : RowAgreement 9439 9440 := RowAgreement.single (first := 9439) modelrow_agrees_9440
private def atomic_9441 : CountSegment 1 10 13120 9440 9441 3925 3925 := CountSegment.single (first := 9440) (initial := 3925) row_count_9441 (by decide)
private theorem atomic_9441_agreement : RowAgreement 9440 9441 := RowAgreement.single (first := 9440) modelrow_agrees_9441
private def atomic_9442 : CountSegment 1 10 13120 9441 9442 3925 3925 := CountSegment.single (first := 9441) (initial := 3925) row_count_9442 (by decide)
private theorem atomic_9442_agreement : RowAgreement 9441 9442 := RowAgreement.single (first := 9441) modelrow_agrees_9442
private def atomic_9443 : CountSegment 1 10 13120 9442 9443 3925 3925 := CountSegment.single (first := 9442) (initial := 3925) row_count_9443 (by decide)
private theorem atomic_9443_agreement : RowAgreement 9442 9443 := RowAgreement.single (first := 9442) modelrow_agrees_9443
private def atomic_9444 : CountSegment 1 10 13120 9443 9444 3925 3925 := CountSegment.single (first := 9443) (initial := 3925) row_count_9444 (by decide)
private theorem atomic_9444_agreement : RowAgreement 9443 9444 := RowAgreement.single (first := 9443) modelrow_agrees_9444
private def atomic_9445 : CountSegment 1 10 13120 9444 9445 3925 3925 := CountSegment.single (first := 9444) (initial := 3925) row_count_9445 (by decide)
private theorem atomic_9445_agreement : RowAgreement 9444 9445 := RowAgreement.single (first := 9444) modelrow_agrees_9445
private def atomic_9446 : CountSegment 1 10 13120 9445 9446 3925 3926 := CountSegment.single (first := 9445) (initial := 3925) row_count_9446 (by decide)
private theorem atomic_9446_agreement : RowAgreement 9445 9446 := RowAgreement.single (first := 9445) modelrow_agrees_9446
private def atomic_9447 : CountSegment 1 10 13120 9446 9447 3926 3927 := CountSegment.single (first := 9446) (initial := 3926) row_count_9447 (by decide)
private theorem atomic_9447_agreement : RowAgreement 9446 9447 := RowAgreement.single (first := 9446) modelrow_agrees_9447
private def atomic_9448 : CountSegment 1 10 13120 9447 9448 3927 3927 := CountSegment.single (first := 9447) (initial := 3927) row_count_9448 (by decide)
private theorem atomic_9448_agreement : RowAgreement 9447 9448 := RowAgreement.single (first := 9447) modelrow_agrees_9448
private def atomic_9449 : CountSegment 1 10 13120 9448 9449 3927 3927 := CountSegment.single (first := 9448) (initial := 3927) row_count_9449 (by decide)
private theorem atomic_9449_agreement : RowAgreement 9448 9449 := RowAgreement.single (first := 9448) modelrow_agrees_9449
private def atomic_9450 : CountSegment 1 10 13120 9449 9450 3927 3929 := CountSegment.single (first := 9449) (initial := 3927) row_count_9450 (by decide)
private theorem atomic_9450_agreement : RowAgreement 9449 9450 := RowAgreement.single (first := 9449) modelrow_agrees_9450
private def atomic_9451 : CountSegment 1 10 13120 9450 9451 3929 3929 := CountSegment.single (first := 9450) (initial := 3929) row_count_9451 (by decide)
private theorem atomic_9451_agreement : RowAgreement 9450 9451 := RowAgreement.single (first := 9450) modelrow_agrees_9451
private def atomic_9452 : CountSegment 1 10 13120 9451 9452 3929 3929 := CountSegment.single (first := 9451) (initial := 3929) row_count_9452 (by decide)
private theorem atomic_9452_agreement : RowAgreement 9451 9452 := RowAgreement.single (first := 9451) modelrow_agrees_9452
private def atomic_9453 : CountSegment 1 10 13120 9452 9453 3929 3930 := CountSegment.single (first := 9452) (initial := 3929) row_count_9453 (by decide)
private theorem atomic_9453_agreement : RowAgreement 9452 9453 := RowAgreement.single (first := 9452) modelrow_agrees_9453
private def atomic_9454 : CountSegment 1 10 13120 9453 9454 3930 3931 := CountSegment.single (first := 9453) (initial := 3930) row_count_9454 (by decide)
private theorem atomic_9454_agreement : RowAgreement 9453 9454 := RowAgreement.single (first := 9453) modelrow_agrees_9454
private def atomic_9455 : CountSegment 1 10 13120 9454 9455 3931 3931 := CountSegment.single (first := 9454) (initial := 3931) row_count_9455 (by decide)
private theorem atomic_9455_agreement : RowAgreement 9454 9455 := RowAgreement.single (first := 9454) modelrow_agrees_9455
private def atomic_9456 : CountSegment 1 10 13120 9455 9456 3931 3931 := CountSegment.single (first := 9455) (initial := 3931) row_count_9456 (by decide)
private theorem atomic_9456_agreement : RowAgreement 9455 9456 := RowAgreement.single (first := 9455) modelrow_agrees_9456
private def atomic_9457 : CountSegment 1 10 13120 9456 9457 3931 3932 := CountSegment.single (first := 9456) (initial := 3931) row_count_9457 (by decide)
private theorem atomic_9457_agreement : RowAgreement 9456 9457 := RowAgreement.single (first := 9456) modelrow_agrees_9457
private def atomic_9458 : CountSegment 1 10 13120 9457 9458 3932 3932 := CountSegment.single (first := 9457) (initial := 3932) row_count_9458 (by decide)
private theorem atomic_9458_agreement : RowAgreement 9457 9458 := RowAgreement.single (first := 9457) modelrow_agrees_9458
private def atomic_9459 : CountSegment 1 10 13120 9458 9459 3932 3932 := CountSegment.single (first := 9458) (initial := 3932) row_count_9459 (by decide)
private theorem atomic_9459_agreement : RowAgreement 9458 9459 := RowAgreement.single (first := 9458) modelrow_agrees_9459
private def atomic_9460 : CountSegment 1 10 13120 9459 9460 3932 3932 := CountSegment.single (first := 9459) (initial := 3932) row_count_9460 (by decide)
private theorem atomic_9460_agreement : RowAgreement 9459 9460 := RowAgreement.single (first := 9459) modelrow_agrees_9460
private def atomic_9461 : CountSegment 1 10 13120 9460 9461 3932 3932 := CountSegment.single (first := 9460) (initial := 3932) row_count_9461 (by decide)
private theorem atomic_9461_agreement : RowAgreement 9460 9461 := RowAgreement.single (first := 9460) modelrow_agrees_9461
private def atomic_9462 : CountSegment 1 10 13120 9461 9462 3932 3932 := CountSegment.single (first := 9461) (initial := 3932) row_count_9462 (by decide)
private theorem atomic_9462_agreement : RowAgreement 9461 9462 := RowAgreement.single (first := 9461) modelrow_agrees_9462
private def atomic_9463 : CountSegment 1 10 13120 9462 9463 3932 3932 := CountSegment.single (first := 9462) (initial := 3932) row_count_9463 (by decide)
private theorem atomic_9463_agreement : RowAgreement 9462 9463 := RowAgreement.single (first := 9462) modelrow_agrees_9463
private def atomic_9464 : CountSegment 1 10 13120 9463 9464 3932 3932 := CountSegment.single (first := 9463) (initial := 3932) row_count_9464 (by decide)
private theorem atomic_9464_agreement : RowAgreement 9463 9464 := RowAgreement.single (first := 9463) modelrow_agrees_9464
private def atomic_9465 : CountSegment 1 10 13120 9464 9465 3932 3932 := CountSegment.single (first := 9464) (initial := 3932) row_count_9465 (by decide)
private theorem atomic_9465_agreement : RowAgreement 9464 9465 := RowAgreement.single (first := 9464) modelrow_agrees_9465
private def atomic_9466 : CountSegment 1 10 13120 9465 9466 3932 3932 := CountSegment.single (first := 9465) (initial := 3932) row_count_9466 (by decide)
private theorem atomic_9466_agreement : RowAgreement 9465 9466 := RowAgreement.single (first := 9465) modelrow_agrees_9466
private def atomic_9467 : CountSegment 1 10 13120 9466 9467 3932 3932 := CountSegment.single (first := 9466) (initial := 3932) row_count_9467 (by decide)
private theorem atomic_9467_agreement : RowAgreement 9466 9467 := RowAgreement.single (first := 9466) modelrow_agrees_9467
private def atomic_9468 : CountSegment 1 10 13120 9467 9468 3932 3932 := CountSegment.single (first := 9467) (initial := 3932) row_count_9468 (by decide)
private theorem atomic_9468_agreement : RowAgreement 9467 9468 := RowAgreement.single (first := 9467) modelrow_agrees_9468
private def atomic_9469 : CountSegment 1 10 13120 9468 9469 3932 3933 := CountSegment.single (first := 9468) (initial := 3932) row_count_9469 (by decide)
private theorem atomic_9469_agreement : RowAgreement 9468 9469 := RowAgreement.single (first := 9468) modelrow_agrees_9469
private def atomic_9470 : CountSegment 1 10 13120 9469 9470 3933 3934 := CountSegment.single (first := 9469) (initial := 3933) row_count_9470 (by decide)
private theorem atomic_9470_agreement : RowAgreement 9469 9470 := RowAgreement.single (first := 9469) modelrow_agrees_9470
private def atomic_9471 : CountSegment 1 10 13120 9470 9471 3934 3934 := CountSegment.single (first := 9470) (initial := 3934) row_count_9471 (by decide)
private theorem atomic_9471_agreement : RowAgreement 9470 9471 := RowAgreement.single (first := 9470) modelrow_agrees_9471
private def atomic_9472 : CountSegment 1 10 13120 9471 9472 3934 3934 := CountSegment.single (first := 9471) (initial := 3934) row_count_9472 (by decide)
private theorem atomic_9472_agreement : RowAgreement 9471 9472 := RowAgreement.single (first := 9471) modelrow_agrees_9472
private def within_073_000 : CountSegment 1 10 13120 9344 9346 3897 3897 := CountSegment.join atomic_9345 atomic_9346
private theorem within_073_000_agreement : RowAgreement 9344 9346 := RowAgreement.join atomic_9345_agreement atomic_9346_agreement
private def within_073_001 : CountSegment 1 10 13120 9346 9348 3897 3898 := CountSegment.join atomic_9347 atomic_9348
private theorem within_073_001_agreement : RowAgreement 9346 9348 := RowAgreement.join atomic_9347_agreement atomic_9348_agreement
private def within_073_002 : CountSegment 1 10 13120 9344 9348 3897 3898 := CountSegment.join within_073_000 within_073_001
private theorem within_073_002_agreement : RowAgreement 9344 9348 := RowAgreement.join within_073_000_agreement within_073_001_agreement
private def within_073_003 : CountSegment 1 10 13120 9348 9350 3898 3899 := CountSegment.join atomic_9349 atomic_9350
private theorem within_073_003_agreement : RowAgreement 9348 9350 := RowAgreement.join atomic_9349_agreement atomic_9350_agreement
private def within_073_004 : CountSegment 1 10 13120 9350 9352 3899 3900 := CountSegment.join atomic_9351 atomic_9352
private theorem within_073_004_agreement : RowAgreement 9350 9352 := RowAgreement.join atomic_9351_agreement atomic_9352_agreement
private def within_073_005 : CountSegment 1 10 13120 9348 9352 3898 3900 := CountSegment.join within_073_003 within_073_004
private theorem within_073_005_agreement : RowAgreement 9348 9352 := RowAgreement.join within_073_003_agreement within_073_004_agreement
private def within_073_006 : CountSegment 1 10 13120 9344 9352 3897 3900 := CountSegment.join within_073_002 within_073_005
private theorem within_073_006_agreement : RowAgreement 9344 9352 := RowAgreement.join within_073_002_agreement within_073_005_agreement
private def within_073_007 : CountSegment 1 10 13120 9352 9354 3900 3901 := CountSegment.join atomic_9353 atomic_9354
private theorem within_073_007_agreement : RowAgreement 9352 9354 := RowAgreement.join atomic_9353_agreement atomic_9354_agreement
private def within_073_008 : CountSegment 1 10 13120 9354 9356 3901 3901 := CountSegment.join atomic_9355 atomic_9356
private theorem within_073_008_agreement : RowAgreement 9354 9356 := RowAgreement.join atomic_9355_agreement atomic_9356_agreement
private def within_073_009 : CountSegment 1 10 13120 9352 9356 3900 3901 := CountSegment.join within_073_007 within_073_008
private theorem within_073_009_agreement : RowAgreement 9352 9356 := RowAgreement.join within_073_007_agreement within_073_008_agreement
private def within_073_010 : CountSegment 1 10 13120 9356 9358 3901 3901 := CountSegment.join atomic_9357 atomic_9358
private theorem within_073_010_agreement : RowAgreement 9356 9358 := RowAgreement.join atomic_9357_agreement atomic_9358_agreement
private def within_073_011 : CountSegment 1 10 13120 9358 9360 3901 3903 := CountSegment.join atomic_9359 atomic_9360
private theorem within_073_011_agreement : RowAgreement 9358 9360 := RowAgreement.join atomic_9359_agreement atomic_9360_agreement
private def within_073_012 : CountSegment 1 10 13120 9356 9360 3901 3903 := CountSegment.join within_073_010 within_073_011
private theorem within_073_012_agreement : RowAgreement 9356 9360 := RowAgreement.join within_073_010_agreement within_073_011_agreement
private def within_073_013 : CountSegment 1 10 13120 9352 9360 3900 3903 := CountSegment.join within_073_009 within_073_012
private theorem within_073_013_agreement : RowAgreement 9352 9360 := RowAgreement.join within_073_009_agreement within_073_012_agreement
private def within_073_014 : CountSegment 1 10 13120 9344 9360 3897 3903 := CountSegment.join within_073_006 within_073_013
private theorem within_073_014_agreement : RowAgreement 9344 9360 := RowAgreement.join within_073_006_agreement within_073_013_agreement
private def within_073_015 : CountSegment 1 10 13120 9360 9362 3903 3903 := CountSegment.join atomic_9361 atomic_9362
private theorem within_073_015_agreement : RowAgreement 9360 9362 := RowAgreement.join atomic_9361_agreement atomic_9362_agreement
private def within_073_016 : CountSegment 1 10 13120 9362 9364 3903 3903 := CountSegment.join atomic_9363 atomic_9364
private theorem within_073_016_agreement : RowAgreement 9362 9364 := RowAgreement.join atomic_9363_agreement atomic_9364_agreement
private def within_073_017 : CountSegment 1 10 13120 9360 9364 3903 3903 := CountSegment.join within_073_015 within_073_016
private theorem within_073_017_agreement : RowAgreement 9360 9364 := RowAgreement.join within_073_015_agreement within_073_016_agreement
private def within_073_018 : CountSegment 1 10 13120 9364 9366 3903 3904 := CountSegment.join atomic_9365 atomic_9366
private theorem within_073_018_agreement : RowAgreement 9364 9366 := RowAgreement.join atomic_9365_agreement atomic_9366_agreement
private def within_073_019 : CountSegment 1 10 13120 9366 9368 3904 3905 := CountSegment.join atomic_9367 atomic_9368
private theorem within_073_019_agreement : RowAgreement 9366 9368 := RowAgreement.join atomic_9367_agreement atomic_9368_agreement
private def within_073_020 : CountSegment 1 10 13120 9364 9368 3903 3905 := CountSegment.join within_073_018 within_073_019
private theorem within_073_020_agreement : RowAgreement 9364 9368 := RowAgreement.join within_073_018_agreement within_073_019_agreement
private def within_073_021 : CountSegment 1 10 13120 9360 9368 3903 3905 := CountSegment.join within_073_017 within_073_020
private theorem within_073_021_agreement : RowAgreement 9360 9368 := RowAgreement.join within_073_017_agreement within_073_020_agreement
private def within_073_022 : CountSegment 1 10 13120 9368 9370 3905 3906 := CountSegment.join atomic_9369 atomic_9370
private theorem within_073_022_agreement : RowAgreement 9368 9370 := RowAgreement.join atomic_9369_agreement atomic_9370_agreement
private def within_073_023 : CountSegment 1 10 13120 9370 9372 3906 3906 := CountSegment.join atomic_9371 atomic_9372
private theorem within_073_023_agreement : RowAgreement 9370 9372 := RowAgreement.join atomic_9371_agreement atomic_9372_agreement
private def within_073_024 : CountSegment 1 10 13120 9368 9372 3905 3906 := CountSegment.join within_073_022 within_073_023
private theorem within_073_024_agreement : RowAgreement 9368 9372 := RowAgreement.join within_073_022_agreement within_073_023_agreement
private def within_073_025 : CountSegment 1 10 13120 9372 9374 3906 3906 := CountSegment.join atomic_9373 atomic_9374
private theorem within_073_025_agreement : RowAgreement 9372 9374 := RowAgreement.join atomic_9373_agreement atomic_9374_agreement
private def within_073_026 : CountSegment 1 10 13120 9374 9376 3906 3906 := CountSegment.join atomic_9375 atomic_9376
private theorem within_073_026_agreement : RowAgreement 9374 9376 := RowAgreement.join atomic_9375_agreement atomic_9376_agreement
private def within_073_027 : CountSegment 1 10 13120 9372 9376 3906 3906 := CountSegment.join within_073_025 within_073_026
private theorem within_073_027_agreement : RowAgreement 9372 9376 := RowAgreement.join within_073_025_agreement within_073_026_agreement
private def within_073_028 : CountSegment 1 10 13120 9368 9376 3905 3906 := CountSegment.join within_073_024 within_073_027
private theorem within_073_028_agreement : RowAgreement 9368 9376 := RowAgreement.join within_073_024_agreement within_073_027_agreement
private def within_073_029 : CountSegment 1 10 13120 9360 9376 3903 3906 := CountSegment.join within_073_021 within_073_028
private theorem within_073_029_agreement : RowAgreement 9360 9376 := RowAgreement.join within_073_021_agreement within_073_028_agreement
private def within_073_030 : CountSegment 1 10 13120 9344 9376 3897 3906 := CountSegment.join within_073_014 within_073_029
private theorem within_073_030_agreement : RowAgreement 9344 9376 := RowAgreement.join within_073_014_agreement within_073_029_agreement
private def within_073_031 : CountSegment 1 10 13120 9376 9378 3906 3907 := CountSegment.join atomic_9377 atomic_9378
private theorem within_073_031_agreement : RowAgreement 9376 9378 := RowAgreement.join atomic_9377_agreement atomic_9378_agreement
private def within_073_032 : CountSegment 1 10 13120 9378 9380 3907 3910 := CountSegment.join atomic_9379 atomic_9380
private theorem within_073_032_agreement : RowAgreement 9378 9380 := RowAgreement.join atomic_9379_agreement atomic_9380_agreement
private def within_073_033 : CountSegment 1 10 13120 9376 9380 3906 3910 := CountSegment.join within_073_031 within_073_032
private theorem within_073_033_agreement : RowAgreement 9376 9380 := RowAgreement.join within_073_031_agreement within_073_032_agreement
private def within_073_034 : CountSegment 1 10 13120 9380 9382 3910 3911 := CountSegment.join atomic_9381 atomic_9382
private theorem within_073_034_agreement : RowAgreement 9380 9382 := RowAgreement.join atomic_9381_agreement atomic_9382_agreement
private def within_073_035 : CountSegment 1 10 13120 9382 9384 3911 3911 := CountSegment.join atomic_9383 atomic_9384
private theorem within_073_035_agreement : RowAgreement 9382 9384 := RowAgreement.join atomic_9383_agreement atomic_9384_agreement
private def within_073_036 : CountSegment 1 10 13120 9380 9384 3910 3911 := CountSegment.join within_073_034 within_073_035
private theorem within_073_036_agreement : RowAgreement 9380 9384 := RowAgreement.join within_073_034_agreement within_073_035_agreement
private def within_073_037 : CountSegment 1 10 13120 9376 9384 3906 3911 := CountSegment.join within_073_033 within_073_036
private theorem within_073_037_agreement : RowAgreement 9376 9384 := RowAgreement.join within_073_033_agreement within_073_036_agreement
private def within_073_038 : CountSegment 1 10 13120 9384 9386 3911 3912 := CountSegment.join atomic_9385 atomic_9386
private theorem within_073_038_agreement : RowAgreement 9384 9386 := RowAgreement.join atomic_9385_agreement atomic_9386_agreement
private def within_073_039 : CountSegment 1 10 13120 9386 9388 3912 3913 := CountSegment.join atomic_9387 atomic_9388
private theorem within_073_039_agreement : RowAgreement 9386 9388 := RowAgreement.join atomic_9387_agreement atomic_9388_agreement
private def within_073_040 : CountSegment 1 10 13120 9384 9388 3911 3913 := CountSegment.join within_073_038 within_073_039
private theorem within_073_040_agreement : RowAgreement 9384 9388 := RowAgreement.join within_073_038_agreement within_073_039_agreement
private def within_073_041 : CountSegment 1 10 13120 9388 9390 3913 3914 := CountSegment.join atomic_9389 atomic_9390
private theorem within_073_041_agreement : RowAgreement 9388 9390 := RowAgreement.join atomic_9389_agreement atomic_9390_agreement
private def within_073_042 : CountSegment 1 10 13120 9390 9392 3914 3914 := CountSegment.join atomic_9391 atomic_9392
private theorem within_073_042_agreement : RowAgreement 9390 9392 := RowAgreement.join atomic_9391_agreement atomic_9392_agreement
private def within_073_043 : CountSegment 1 10 13120 9388 9392 3913 3914 := CountSegment.join within_073_041 within_073_042
private theorem within_073_043_agreement : RowAgreement 9388 9392 := RowAgreement.join within_073_041_agreement within_073_042_agreement
private def within_073_044 : CountSegment 1 10 13120 9384 9392 3911 3914 := CountSegment.join within_073_040 within_073_043
private theorem within_073_044_agreement : RowAgreement 9384 9392 := RowAgreement.join within_073_040_agreement within_073_043_agreement
private def within_073_045 : CountSegment 1 10 13120 9376 9392 3906 3914 := CountSegment.join within_073_037 within_073_044
private theorem within_073_045_agreement : RowAgreement 9376 9392 := RowAgreement.join within_073_037_agreement within_073_044_agreement
private def within_073_046 : CountSegment 1 10 13120 9392 9394 3914 3915 := CountSegment.join atomic_9393 atomic_9394
private theorem within_073_046_agreement : RowAgreement 9392 9394 := RowAgreement.join atomic_9393_agreement atomic_9394_agreement
private def within_073_047 : CountSegment 1 10 13120 9394 9396 3915 3915 := CountSegment.join atomic_9395 atomic_9396
private theorem within_073_047_agreement : RowAgreement 9394 9396 := RowAgreement.join atomic_9395_agreement atomic_9396_agreement
private def within_073_048 : CountSegment 1 10 13120 9392 9396 3914 3915 := CountSegment.join within_073_046 within_073_047
private theorem within_073_048_agreement : RowAgreement 9392 9396 := RowAgreement.join within_073_046_agreement within_073_047_agreement
private def within_073_049 : CountSegment 1 10 13120 9396 9398 3915 3916 := CountSegment.join atomic_9397 atomic_9398
private theorem within_073_049_agreement : RowAgreement 9396 9398 := RowAgreement.join atomic_9397_agreement atomic_9398_agreement
private def within_073_050 : CountSegment 1 10 13120 9398 9400 3916 3916 := CountSegment.join atomic_9399 atomic_9400
private theorem within_073_050_agreement : RowAgreement 9398 9400 := RowAgreement.join atomic_9399_agreement atomic_9400_agreement
private def within_073_051 : CountSegment 1 10 13120 9396 9400 3915 3916 := CountSegment.join within_073_049 within_073_050
private theorem within_073_051_agreement : RowAgreement 9396 9400 := RowAgreement.join within_073_049_agreement within_073_050_agreement
private def within_073_052 : CountSegment 1 10 13120 9392 9400 3914 3916 := CountSegment.join within_073_048 within_073_051
private theorem within_073_052_agreement : RowAgreement 9392 9400 := RowAgreement.join within_073_048_agreement within_073_051_agreement
private def within_073_053 : CountSegment 1 10 13120 9400 9402 3916 3916 := CountSegment.join atomic_9401 atomic_9402
private theorem within_073_053_agreement : RowAgreement 9400 9402 := RowAgreement.join atomic_9401_agreement atomic_9402_agreement
private def within_073_054 : CountSegment 1 10 13120 9402 9404 3916 3916 := CountSegment.join atomic_9403 atomic_9404
private theorem within_073_054_agreement : RowAgreement 9402 9404 := RowAgreement.join atomic_9403_agreement atomic_9404_agreement
private def within_073_055 : CountSegment 1 10 13120 9400 9404 3916 3916 := CountSegment.join within_073_053 within_073_054
private theorem within_073_055_agreement : RowAgreement 9400 9404 := RowAgreement.join within_073_053_agreement within_073_054_agreement
private def within_073_056 : CountSegment 1 10 13120 9404 9406 3916 3917 := CountSegment.join atomic_9405 atomic_9406
private theorem within_073_056_agreement : RowAgreement 9404 9406 := RowAgreement.join atomic_9405_agreement atomic_9406_agreement
private def within_073_057 : CountSegment 1 10 13120 9406 9408 3917 3918 := CountSegment.join atomic_9407 atomic_9408
private theorem within_073_057_agreement : RowAgreement 9406 9408 := RowAgreement.join atomic_9407_agreement atomic_9408_agreement
private def within_073_058 : CountSegment 1 10 13120 9404 9408 3916 3918 := CountSegment.join within_073_056 within_073_057
private theorem within_073_058_agreement : RowAgreement 9404 9408 := RowAgreement.join within_073_056_agreement within_073_057_agreement
private def within_073_059 : CountSegment 1 10 13120 9400 9408 3916 3918 := CountSegment.join within_073_055 within_073_058
private theorem within_073_059_agreement : RowAgreement 9400 9408 := RowAgreement.join within_073_055_agreement within_073_058_agreement
private def within_073_060 : CountSegment 1 10 13120 9392 9408 3914 3918 := CountSegment.join within_073_052 within_073_059
private theorem within_073_060_agreement : RowAgreement 9392 9408 := RowAgreement.join within_073_052_agreement within_073_059_agreement
private def within_073_061 : CountSegment 1 10 13120 9376 9408 3906 3918 := CountSegment.join within_073_045 within_073_060
private theorem within_073_061_agreement : RowAgreement 9376 9408 := RowAgreement.join within_073_045_agreement within_073_060_agreement
private def within_073_062 : CountSegment 1 10 13120 9344 9408 3897 3918 := CountSegment.join within_073_030 within_073_061
private theorem within_073_062_agreement : RowAgreement 9344 9408 := RowAgreement.join within_073_030_agreement within_073_061_agreement
private def within_073_063 : CountSegment 1 10 13120 9408 9410 3918 3918 := CountSegment.join atomic_9409 atomic_9410
private theorem within_073_063_agreement : RowAgreement 9408 9410 := RowAgreement.join atomic_9409_agreement atomic_9410_agreement
private def within_073_064 : CountSegment 1 10 13120 9410 9412 3918 3918 := CountSegment.join atomic_9411 atomic_9412
private theorem within_073_064_agreement : RowAgreement 9410 9412 := RowAgreement.join atomic_9411_agreement atomic_9412_agreement
private def within_073_065 : CountSegment 1 10 13120 9408 9412 3918 3918 := CountSegment.join within_073_063 within_073_064
private theorem within_073_065_agreement : RowAgreement 9408 9412 := RowAgreement.join within_073_063_agreement within_073_064_agreement
private def within_073_066 : CountSegment 1 10 13120 9412 9414 3918 3918 := CountSegment.join atomic_9413 atomic_9414
private theorem within_073_066_agreement : RowAgreement 9412 9414 := RowAgreement.join atomic_9413_agreement atomic_9414_agreement
private def within_073_067 : CountSegment 1 10 13120 9414 9416 3918 3920 := CountSegment.join atomic_9415 atomic_9416
private theorem within_073_067_agreement : RowAgreement 9414 9416 := RowAgreement.join atomic_9415_agreement atomic_9416_agreement
private def within_073_068 : CountSegment 1 10 13120 9412 9416 3918 3920 := CountSegment.join within_073_066 within_073_067
private theorem within_073_068_agreement : RowAgreement 9412 9416 := RowAgreement.join within_073_066_agreement within_073_067_agreement
private def within_073_069 : CountSegment 1 10 13120 9408 9416 3918 3920 := CountSegment.join within_073_065 within_073_068
private theorem within_073_069_agreement : RowAgreement 9408 9416 := RowAgreement.join within_073_065_agreement within_073_068_agreement
private def within_073_070 : CountSegment 1 10 13120 9416 9418 3920 3920 := CountSegment.join atomic_9417 atomic_9418
private theorem within_073_070_agreement : RowAgreement 9416 9418 := RowAgreement.join atomic_9417_agreement atomic_9418_agreement
private def within_073_071 : CountSegment 1 10 13120 9418 9420 3920 3921 := CountSegment.join atomic_9419 atomic_9420
private theorem within_073_071_agreement : RowAgreement 9418 9420 := RowAgreement.join atomic_9419_agreement atomic_9420_agreement
private def within_073_072 : CountSegment 1 10 13120 9416 9420 3920 3921 := CountSegment.join within_073_070 within_073_071
private theorem within_073_072_agreement : RowAgreement 9416 9420 := RowAgreement.join within_073_070_agreement within_073_071_agreement
private def within_073_073 : CountSegment 1 10 13120 9420 9422 3921 3922 := CountSegment.join atomic_9421 atomic_9422
private theorem within_073_073_agreement : RowAgreement 9420 9422 := RowAgreement.join atomic_9421_agreement atomic_9422_agreement
private def within_073_074 : CountSegment 1 10 13120 9422 9424 3922 3922 := CountSegment.join atomic_9423 atomic_9424
private theorem within_073_074_agreement : RowAgreement 9422 9424 := RowAgreement.join atomic_9423_agreement atomic_9424_agreement
private def within_073_075 : CountSegment 1 10 13120 9420 9424 3921 3922 := CountSegment.join within_073_073 within_073_074
private theorem within_073_075_agreement : RowAgreement 9420 9424 := RowAgreement.join within_073_073_agreement within_073_074_agreement
private def within_073_076 : CountSegment 1 10 13120 9416 9424 3920 3922 := CountSegment.join within_073_072 within_073_075
private theorem within_073_076_agreement : RowAgreement 9416 9424 := RowAgreement.join within_073_072_agreement within_073_075_agreement
private def within_073_077 : CountSegment 1 10 13120 9408 9424 3918 3922 := CountSegment.join within_073_069 within_073_076
private theorem within_073_077_agreement : RowAgreement 9408 9424 := RowAgreement.join within_073_069_agreement within_073_076_agreement
private def within_073_078 : CountSegment 1 10 13120 9424 9426 3922 3923 := CountSegment.join atomic_9425 atomic_9426
private theorem within_073_078_agreement : RowAgreement 9424 9426 := RowAgreement.join atomic_9425_agreement atomic_9426_agreement
private def within_073_079 : CountSegment 1 10 13120 9426 9428 3923 3924 := CountSegment.join atomic_9427 atomic_9428
private theorem within_073_079_agreement : RowAgreement 9426 9428 := RowAgreement.join atomic_9427_agreement atomic_9428_agreement
private def within_073_080 : CountSegment 1 10 13120 9424 9428 3922 3924 := CountSegment.join within_073_078 within_073_079
private theorem within_073_080_agreement : RowAgreement 9424 9428 := RowAgreement.join within_073_078_agreement within_073_079_agreement
private def within_073_081 : CountSegment 1 10 13120 9428 9430 3924 3924 := CountSegment.join atomic_9429 atomic_9430
private theorem within_073_081_agreement : RowAgreement 9428 9430 := RowAgreement.join atomic_9429_agreement atomic_9430_agreement
private def within_073_082 : CountSegment 1 10 13120 9430 9432 3924 3924 := CountSegment.join atomic_9431 atomic_9432
private theorem within_073_082_agreement : RowAgreement 9430 9432 := RowAgreement.join atomic_9431_agreement atomic_9432_agreement
private def within_073_083 : CountSegment 1 10 13120 9428 9432 3924 3924 := CountSegment.join within_073_081 within_073_082
private theorem within_073_083_agreement : RowAgreement 9428 9432 := RowAgreement.join within_073_081_agreement within_073_082_agreement
private def within_073_084 : CountSegment 1 10 13120 9424 9432 3922 3924 := CountSegment.join within_073_080 within_073_083
private theorem within_073_084_agreement : RowAgreement 9424 9432 := RowAgreement.join within_073_080_agreement within_073_083_agreement
private def within_073_085 : CountSegment 1 10 13120 9432 9434 3924 3924 := CountSegment.join atomic_9433 atomic_9434
private theorem within_073_085_agreement : RowAgreement 9432 9434 := RowAgreement.join atomic_9433_agreement atomic_9434_agreement
private def within_073_086 : CountSegment 1 10 13120 9434 9436 3924 3924 := CountSegment.join atomic_9435 atomic_9436
private theorem within_073_086_agreement : RowAgreement 9434 9436 := RowAgreement.join atomic_9435_agreement atomic_9436_agreement
private def within_073_087 : CountSegment 1 10 13120 9432 9436 3924 3924 := CountSegment.join within_073_085 within_073_086
private theorem within_073_087_agreement : RowAgreement 9432 9436 := RowAgreement.join within_073_085_agreement within_073_086_agreement
private def within_073_088 : CountSegment 1 10 13120 9436 9438 3924 3925 := CountSegment.join atomic_9437 atomic_9438
private theorem within_073_088_agreement : RowAgreement 9436 9438 := RowAgreement.join atomic_9437_agreement atomic_9438_agreement
private def within_073_089 : CountSegment 1 10 13120 9438 9440 3925 3925 := CountSegment.join atomic_9439 atomic_9440
private theorem within_073_089_agreement : RowAgreement 9438 9440 := RowAgreement.join atomic_9439_agreement atomic_9440_agreement
private def within_073_090 : CountSegment 1 10 13120 9436 9440 3924 3925 := CountSegment.join within_073_088 within_073_089
private theorem within_073_090_agreement : RowAgreement 9436 9440 := RowAgreement.join within_073_088_agreement within_073_089_agreement
private def within_073_091 : CountSegment 1 10 13120 9432 9440 3924 3925 := CountSegment.join within_073_087 within_073_090
private theorem within_073_091_agreement : RowAgreement 9432 9440 := RowAgreement.join within_073_087_agreement within_073_090_agreement
private def within_073_092 : CountSegment 1 10 13120 9424 9440 3922 3925 := CountSegment.join within_073_084 within_073_091
private theorem within_073_092_agreement : RowAgreement 9424 9440 := RowAgreement.join within_073_084_agreement within_073_091_agreement
private def within_073_093 : CountSegment 1 10 13120 9408 9440 3918 3925 := CountSegment.join within_073_077 within_073_092
private theorem within_073_093_agreement : RowAgreement 9408 9440 := RowAgreement.join within_073_077_agreement within_073_092_agreement
private def within_073_094 : CountSegment 1 10 13120 9440 9442 3925 3925 := CountSegment.join atomic_9441 atomic_9442
private theorem within_073_094_agreement : RowAgreement 9440 9442 := RowAgreement.join atomic_9441_agreement atomic_9442_agreement
private def within_073_095 : CountSegment 1 10 13120 9442 9444 3925 3925 := CountSegment.join atomic_9443 atomic_9444
private theorem within_073_095_agreement : RowAgreement 9442 9444 := RowAgreement.join atomic_9443_agreement atomic_9444_agreement
private def within_073_096 : CountSegment 1 10 13120 9440 9444 3925 3925 := CountSegment.join within_073_094 within_073_095
private theorem within_073_096_agreement : RowAgreement 9440 9444 := RowAgreement.join within_073_094_agreement within_073_095_agreement
private def within_073_097 : CountSegment 1 10 13120 9444 9446 3925 3926 := CountSegment.join atomic_9445 atomic_9446
private theorem within_073_097_agreement : RowAgreement 9444 9446 := RowAgreement.join atomic_9445_agreement atomic_9446_agreement
private def within_073_098 : CountSegment 1 10 13120 9446 9448 3926 3927 := CountSegment.join atomic_9447 atomic_9448
private theorem within_073_098_agreement : RowAgreement 9446 9448 := RowAgreement.join atomic_9447_agreement atomic_9448_agreement
private def within_073_099 : CountSegment 1 10 13120 9444 9448 3925 3927 := CountSegment.join within_073_097 within_073_098
private theorem within_073_099_agreement : RowAgreement 9444 9448 := RowAgreement.join within_073_097_agreement within_073_098_agreement
private def within_073_100 : CountSegment 1 10 13120 9440 9448 3925 3927 := CountSegment.join within_073_096 within_073_099
private theorem within_073_100_agreement : RowAgreement 9440 9448 := RowAgreement.join within_073_096_agreement within_073_099_agreement
private def within_073_101 : CountSegment 1 10 13120 9448 9450 3927 3929 := CountSegment.join atomic_9449 atomic_9450
private theorem within_073_101_agreement : RowAgreement 9448 9450 := RowAgreement.join atomic_9449_agreement atomic_9450_agreement
private def within_073_102 : CountSegment 1 10 13120 9450 9452 3929 3929 := CountSegment.join atomic_9451 atomic_9452
private theorem within_073_102_agreement : RowAgreement 9450 9452 := RowAgreement.join atomic_9451_agreement atomic_9452_agreement
private def within_073_103 : CountSegment 1 10 13120 9448 9452 3927 3929 := CountSegment.join within_073_101 within_073_102
private theorem within_073_103_agreement : RowAgreement 9448 9452 := RowAgreement.join within_073_101_agreement within_073_102_agreement
private def within_073_104 : CountSegment 1 10 13120 9452 9454 3929 3931 := CountSegment.join atomic_9453 atomic_9454
private theorem within_073_104_agreement : RowAgreement 9452 9454 := RowAgreement.join atomic_9453_agreement atomic_9454_agreement
private def within_073_105 : CountSegment 1 10 13120 9454 9456 3931 3931 := CountSegment.join atomic_9455 atomic_9456
private theorem within_073_105_agreement : RowAgreement 9454 9456 := RowAgreement.join atomic_9455_agreement atomic_9456_agreement
private def within_073_106 : CountSegment 1 10 13120 9452 9456 3929 3931 := CountSegment.join within_073_104 within_073_105
private theorem within_073_106_agreement : RowAgreement 9452 9456 := RowAgreement.join within_073_104_agreement within_073_105_agreement
private def within_073_107 : CountSegment 1 10 13120 9448 9456 3927 3931 := CountSegment.join within_073_103 within_073_106
private theorem within_073_107_agreement : RowAgreement 9448 9456 := RowAgreement.join within_073_103_agreement within_073_106_agreement
private def within_073_108 : CountSegment 1 10 13120 9440 9456 3925 3931 := CountSegment.join within_073_100 within_073_107
private theorem within_073_108_agreement : RowAgreement 9440 9456 := RowAgreement.join within_073_100_agreement within_073_107_agreement
private def within_073_109 : CountSegment 1 10 13120 9456 9458 3931 3932 := CountSegment.join atomic_9457 atomic_9458
private theorem within_073_109_agreement : RowAgreement 9456 9458 := RowAgreement.join atomic_9457_agreement atomic_9458_agreement
private def within_073_110 : CountSegment 1 10 13120 9458 9460 3932 3932 := CountSegment.join atomic_9459 atomic_9460
private theorem within_073_110_agreement : RowAgreement 9458 9460 := RowAgreement.join atomic_9459_agreement atomic_9460_agreement
private def within_073_111 : CountSegment 1 10 13120 9456 9460 3931 3932 := CountSegment.join within_073_109 within_073_110
private theorem within_073_111_agreement : RowAgreement 9456 9460 := RowAgreement.join within_073_109_agreement within_073_110_agreement
private def within_073_112 : CountSegment 1 10 13120 9460 9462 3932 3932 := CountSegment.join atomic_9461 atomic_9462
private theorem within_073_112_agreement : RowAgreement 9460 9462 := RowAgreement.join atomic_9461_agreement atomic_9462_agreement
private def within_073_113 : CountSegment 1 10 13120 9462 9464 3932 3932 := CountSegment.join atomic_9463 atomic_9464
private theorem within_073_113_agreement : RowAgreement 9462 9464 := RowAgreement.join atomic_9463_agreement atomic_9464_agreement
private def within_073_114 : CountSegment 1 10 13120 9460 9464 3932 3932 := CountSegment.join within_073_112 within_073_113
private theorem within_073_114_agreement : RowAgreement 9460 9464 := RowAgreement.join within_073_112_agreement within_073_113_agreement
private def within_073_115 : CountSegment 1 10 13120 9456 9464 3931 3932 := CountSegment.join within_073_111 within_073_114
private theorem within_073_115_agreement : RowAgreement 9456 9464 := RowAgreement.join within_073_111_agreement within_073_114_agreement
private def within_073_116 : CountSegment 1 10 13120 9464 9466 3932 3932 := CountSegment.join atomic_9465 atomic_9466
private theorem within_073_116_agreement : RowAgreement 9464 9466 := RowAgreement.join atomic_9465_agreement atomic_9466_agreement
private def within_073_117 : CountSegment 1 10 13120 9466 9468 3932 3932 := CountSegment.join atomic_9467 atomic_9468
private theorem within_073_117_agreement : RowAgreement 9466 9468 := RowAgreement.join atomic_9467_agreement atomic_9468_agreement
private def within_073_118 : CountSegment 1 10 13120 9464 9468 3932 3932 := CountSegment.join within_073_116 within_073_117
private theorem within_073_118_agreement : RowAgreement 9464 9468 := RowAgreement.join within_073_116_agreement within_073_117_agreement
private def within_073_119 : CountSegment 1 10 13120 9468 9470 3932 3934 := CountSegment.join atomic_9469 atomic_9470
private theorem within_073_119_agreement : RowAgreement 9468 9470 := RowAgreement.join atomic_9469_agreement atomic_9470_agreement
private def within_073_120 : CountSegment 1 10 13120 9470 9472 3934 3934 := CountSegment.join atomic_9471 atomic_9472
private theorem within_073_120_agreement : RowAgreement 9470 9472 := RowAgreement.join atomic_9471_agreement atomic_9472_agreement
private def within_073_121 : CountSegment 1 10 13120 9468 9472 3932 3934 := CountSegment.join within_073_119 within_073_120
private theorem within_073_121_agreement : RowAgreement 9468 9472 := RowAgreement.join within_073_119_agreement within_073_120_agreement
private def within_073_122 : CountSegment 1 10 13120 9464 9472 3932 3934 := CountSegment.join within_073_118 within_073_121
private theorem within_073_122_agreement : RowAgreement 9464 9472 := RowAgreement.join within_073_118_agreement within_073_121_agreement
private def within_073_123 : CountSegment 1 10 13120 9456 9472 3931 3934 := CountSegment.join within_073_115 within_073_122
private theorem within_073_123_agreement : RowAgreement 9456 9472 := RowAgreement.join within_073_115_agreement within_073_122_agreement
private def within_073_124 : CountSegment 1 10 13120 9440 9472 3925 3934 := CountSegment.join within_073_108 within_073_123
private theorem within_073_124_agreement : RowAgreement 9440 9472 := RowAgreement.join within_073_108_agreement within_073_123_agreement
private def within_073_125 : CountSegment 1 10 13120 9408 9472 3918 3934 := CountSegment.join within_073_093 within_073_124
private theorem within_073_125_agreement : RowAgreement 9408 9472 := RowAgreement.join within_073_093_agreement within_073_124_agreement
private def within_073_126 : CountSegment 1 10 13120 9344 9472 3897 3934 := CountSegment.join within_073_062 within_073_125
private theorem within_073_126_agreement : RowAgreement 9344 9472 := RowAgreement.join within_073_062_agreement within_073_125_agreement
def segment_073 : CountSegment 1 10 13120 9344 9472 3897 3934 := within_073_126
theorem agreement_073 : RowAgreement 9344 9472 := within_073_126_agreement
end ProjectEulerSolutions.P58Numerical
