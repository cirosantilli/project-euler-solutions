import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_257 : RowCertificate 257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 257 1) := by change ¬ Nat.Prime 264711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 88237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 257 2) := by change ¬ Nat.Prime 264197; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 257 3) := by change ¬ Nat.Prime 263683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_257 : row 257 = 0 := (row_257.checked).trans rfl
theorem model_count_257 : modelRow 257 = 0 := by decide
theorem modelrow_agrees_257 : modelRow 257 = row 257 := model_count_257.trans row_count_257.symm
def row_258 : RowCertificate 258 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 258 1) := by change ¬ Nat.Prime 266773; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 20521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 258 2) := by change ¬ Nat.Prime 266257; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 258 3) := by change ¬ Nat.Prime 265741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_258 : row 258 = 0 := (row_258.checked).trans rfl
theorem model_count_258 : modelRow 258 = 0 := by decide
theorem modelrow_agrees_258 : modelRow 258 = row 258 := model_count_258.trans row_count_258.symm
def row_259 : RowCertificate 259 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 259 1) := by change Nat.Prime 268843; exact ProjectEulerSolutions.P58PrimeNodes.p_268843.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 259 2) := by change ¬ Nat.Prime 268325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 53665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 259 3) := by change ¬ Nat.Prime 267807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 89269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_259 : row 259 = 1 := (row_259.checked).trans rfl
theorem model_count_259 : modelRow 259 = 1 := by decide
theorem modelrow_agrees_259 : modelRow 259 = row 259 := model_count_259.trans row_count_259.symm
def row_260 : RowCertificate 260 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 260 1) := by change ¬ Nat.Prime 270921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 90307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 260 2) := by change ¬ Nat.Prime 270401; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 260 3) := by change ¬ Nat.Prime 269881; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_260 : row 260 = 0 := (row_260.checked).trans rfl
theorem model_count_260 : modelRow 260 = 0 := by decide
theorem modelrow_agrees_260 : modelRow 260 = row 260 := model_count_260.trans row_count_260.symm
def row_261 : RowCertificate 261 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 261 1) := by change ¬ Nat.Prime 273007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 261 2) := by change ¬ Nat.Prime 272485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 54497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 261 3) := by change ¬ Nat.Prime 271963; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_261 : row 261 = 0 := (row_261.checked).trans rfl
theorem model_count_261 : modelRow 261 = 0 := by decide
theorem modelrow_agrees_261 : modelRow 261 = row 261 := model_count_261.trans row_count_261.symm
def row_262 : RowCertificate 262 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 262 1) := by change ¬ Nat.Prime 275101; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 14479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 262 2) := by change ¬ Nat.Prime 274577; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 262 3) := by change ¬ Nat.Prime 274053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 91351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_262 : row 262 = 0 := (row_262.checked).trans rfl
theorem model_count_262 : modelRow 262 = 0 := by decide
theorem modelrow_agrees_262 : modelRow 262 = row 262 := model_count_262.trans row_count_262.symm
def row_263 : RowCertificate 263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 263 1) := by change ¬ Nat.Prime 277203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 92401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 263 2) := by change ¬ Nat.Prime 276677; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 263 3) := by change Nat.Prime 276151; exact ProjectEulerSolutions.P58PrimeNodes.p_276151.prime
    exact decide_eq_true prime
theorem row_count_263 : row 263 = 1 := (row_263.checked).trans rfl
theorem model_count_263 : modelRow 263 = 1 := by decide
theorem modelrow_agrees_263 : modelRow 263 = row 263 := model_count_263.trans row_count_263.symm
def row_264 : RowCertificate 264 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 264 1) := by change ¬ Nat.Prime 279313; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 264 2) := by change ¬ Nat.Prime 278785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 55757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 264 3) := by change ¬ Nat.Prime 278257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 39751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_264 : row 264 = 0 := (row_264.checked).trans rfl
theorem model_count_264 : modelRow 264 = 0 := by decide
theorem modelrow_agrees_264 : modelRow 264 = row 264 := model_count_264.trans row_count_264.symm
def row_265 : RowCertificate 265 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 265 1) := by change Nat.Prime 281431; exact ProjectEulerSolutions.P58PrimeNodes.p_281431.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 265 2) := by change ¬ Nat.Prime 280901; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 1093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 265 3) := by change ¬ Nat.Prime 280371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 93457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_265 : row 265 = 1 := (row_265.checked).trans rfl
theorem model_count_265 : modelRow 265 = 1 := by decide
theorem modelrow_agrees_265 : modelRow 265 = row 265 := model_count_265.trans row_count_265.symm
def row_266 : RowCertificate 266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 266 1) := by change ¬ Nat.Prime 283557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 94519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 266 2) := by change ¬ Nat.Prime 283025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 56605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 266 3) := by change Nat.Prime 282493; exact ProjectEulerSolutions.P58PrimeNodes.p_282493.prime
    exact decide_eq_true prime
theorem row_count_266 : row 266 = 1 := (row_266.checked).trans rfl
theorem model_count_266 : modelRow 266 = 1 := by decide
theorem modelrow_agrees_266 : modelRow 266 = row 266 := model_count_266.trans row_count_266.symm
def row_267 : RowCertificate 267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 267 1) := by change ¬ Nat.Prime 285691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 40813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 267 2) := by change ¬ Nat.Prime 285157; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 9833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 267 3) := by change Nat.Prime 284623; exact ProjectEulerSolutions.P58PrimeNodes.p_284623.prime
    exact decide_eq_true prime
theorem row_count_267 : row 267 = 1 := (row_267.checked).trans rfl
theorem model_count_267 : modelRow 267 = 1 := by decide
theorem modelrow_agrees_267 : modelRow 267 = row 267 := model_count_267.trans row_count_267.symm
def row_268 : RowCertificate 268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 268 1) := by change ¬ Nat.Prime 287833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 41119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 268 2) := by change Nat.Prime 287297; exact ProjectEulerSolutions.P58PrimeNodes.p_287297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 268 3) := by change ¬ Nat.Prime 286761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 95587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_268 : row 268 = 1 := (row_268.checked).trans rfl
theorem model_count_268 : modelRow 268 = 1 := by decide
theorem modelrow_agrees_268 : modelRow 268 = row 268 := model_count_268.trans row_count_268.symm
def row_269 : RowCertificate 269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 269 1) := by change ¬ Nat.Prime 289983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 96661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 269 2) := by change ¬ Nat.Prime 289445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 57889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 269 3) := by change Nat.Prime 288907; exact ProjectEulerSolutions.P58PrimeNodes.p_288907.prime
    exact decide_eq_true prime
theorem row_count_269 : row 269 = 1 := (row_269.checked).trans rfl
theorem model_count_269 : modelRow 269 = 1 := by decide
theorem modelrow_agrees_269 : modelRow 269 = row 269 := model_count_269.trans row_count_269.symm
def row_270 : RowCertificate 270 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 270 1) := by change Nat.Prime 292141; exact ProjectEulerSolutions.P58PrimeNodes.p_292141.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 270 2) := by change ¬ Nat.Prime 291601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 17153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 270 3) := by change ¬ Nat.Prime 291061; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 15319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_270 : row 270 = 1 := (row_270.checked).trans rfl
theorem model_count_270 : modelRow 270 = 1 := by decide
theorem modelrow_agrees_270 : modelRow 270 = row 270 := model_count_270.trans row_count_270.symm
def row_271 : RowCertificate 271 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 271 1) := by change ¬ Nat.Prime 294307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 22639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 271 2) := by change ¬ Nat.Prime 293765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 58753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 271 3) := by change ¬ Nat.Prime 293223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 97741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_271 : row 271 = 0 := (row_271.checked).trans rfl
theorem model_count_271 : modelRow 271 = 0 := by decide
theorem modelrow_agrees_271 : modelRow 271 = row 271 := model_count_271.trans row_count_271.symm
def row_272 : RowCertificate 272 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 272 1) := by change ¬ Nat.Prime 296481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 98827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 272 2) := by change Nat.Prime 295937; exact ProjectEulerSolutions.P58PrimeNodes.p_295937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 272 3) := by change ¬ Nat.Prime 295393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 42199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_272 : row 272 = 1 := (row_272.checked).trans rfl
theorem model_count_272 : modelRow 272 = 1 := by decide
theorem modelrow_agrees_272 : modelRow 272 = row 272 := model_count_272.trans row_count_272.symm
def row_273 : RowCertificate 273 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 273 1) := by change ¬ Nat.Prime 298663; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 273 2) := by change ¬ Nat.Prime 298117; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 273 3) := by change ¬ Nat.Prime 297571; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_273 : row 273 = 0 := (row_273.checked).trans rfl
theorem model_count_273 : modelRow 273 = 0 := by decide
theorem modelrow_agrees_273 : modelRow 273 = row 273 := model_count_273.trans row_count_273.symm
def row_274 : RowCertificate 274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 274 1) := by change ¬ Nat.Prime 300853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 42979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 274 2) := by change ¬ Nat.Prime 300305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 274 3) := by change ¬ Nat.Prime 299757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 99919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_274 : row 274 = 0 := (row_274.checked).trans rfl
theorem model_count_274 : modelRow 274 = 0 := by decide
theorem modelrow_agrees_274 : modelRow 274 = row 274 := model_count_274.trans row_count_274.symm
def row_275 : RowCertificate 275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 275 1) := by change ¬ Nat.Prime 303051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 101017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 275 2) := by change ¬ Nat.Prime 302501; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 2677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 275 3) := by change ¬ Nat.Prime 301951; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_275 : row 275 = 0 := (row_275.checked).trans rfl
theorem model_count_275 : modelRow 275 = 0 := by decide
theorem modelrow_agrees_275 : modelRow 275 = row 275 := model_count_275.trans row_count_275.symm
def row_276 : RowCertificate 276 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 276 1) := by change ¬ Nat.Prime 305257; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 9847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 276 2) := by change ¬ Nat.Prime 304705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 60941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 276 3) := by change Nat.Prime 304153; exact ProjectEulerSolutions.P58PrimeNodes.p_304153.prime
    exact decide_eq_true prime
theorem row_count_276 : row 276 = 1 := (row_276.checked).trans rfl
theorem model_count_276 : modelRow 276 = 1 := by decide
theorem modelrow_agrees_276 : modelRow 276 = row 276 := model_count_276.trans row_count_276.symm
def row_277 : RowCertificate 277 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 277 1) := by change Nat.Prime 307471; exact ProjectEulerSolutions.P58PrimeNodes.p_307471.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 277 2) := by change ¬ Nat.Prime 306917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 23609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 277 3) := by change ¬ Nat.Prime 306363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 102121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_277 : row 277 = 1 := (row_277.checked).trans rfl
theorem model_count_277 : modelRow 277 = 1 := by decide
theorem modelrow_agrees_277 : modelRow 277 = row 277 := model_count_277.trans row_count_277.symm
def row_278 : RowCertificate 278 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 278 1) := by change ¬ Nat.Prime 309693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 103231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 278 2) := by change Nat.Prime 309137; exact ProjectEulerSolutions.P58PrimeNodes.p_309137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 278 3) := by change ¬ Nat.Prime 308581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_278 : row 278 = 1 := (row_278.checked).trans rfl
theorem model_count_278 : modelRow 278 = 1 := by decide
theorem modelrow_agrees_278 : modelRow 278 = row 278 := model_count_278.trans row_count_278.symm
def row_279 : RowCertificate 279 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 279 1) := by change ¬ Nat.Prime 311923; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 16417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 279 2) := by change ¬ Nat.Prime 311365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 62273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 279 3) := by change ¬ Nat.Prime 310807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 44401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_279 : row 279 = 0 := (row_279.checked).trans rfl
theorem model_count_279 : modelRow 279 = 0 := by decide
theorem modelrow_agrees_279 : modelRow 279 = row 279 := model_count_279.trans row_count_279.symm
def row_280 : RowCertificate 280 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 280 1) := by change Nat.Prime 314161; exact ProjectEulerSolutions.P58PrimeNodes.p_314161.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 280 2) := by change ¬ Nat.Prime 313601; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 5917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 280 3) := by change ¬ Nat.Prime 313041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 104347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_280 : row 280 = 1 := (row_280.checked).trans rfl
theorem model_count_280 : modelRow 280 = 1 := by decide
theorem modelrow_agrees_280 : modelRow 280 = row 280 := model_count_280.trans row_count_280.symm
def row_281 : RowCertificate 281 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 281 1) := by change ¬ Nat.Prime 316407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 105469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 281 2) := by change ¬ Nat.Prime 315845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 63169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 281 3) := by change ¬ Nat.Prime 315283; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_281 : row 281 = 0 := (row_281.checked).trans rfl
theorem model_count_281 : modelRow 281 = 0 := by decide
theorem modelrow_agrees_281 : modelRow 281 = row 281 := model_count_281.trans row_count_281.symm
def row_282 : RowCertificate 282 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 282 1) := by change ¬ Nat.Prime 318661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 45523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 282 2) := by change ¬ Nat.Prime 318097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 24469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 282 3) := by change ¬ Nat.Prime 317533; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 10243) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_282 : row 282 = 0 := (row_282.checked).trans rfl
theorem model_count_282 : modelRow 282 = 0 := by decide
theorem modelrow_agrees_282 : modelRow 282 = row 282 := model_count_282.trans row_count_282.symm
def row_283 : RowCertificate 283 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 283 1) := by change Nat.Prime 320923; exact ProjectEulerSolutions.P58PrimeNodes.p_320923.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 283 2) := by change ¬ Nat.Prime 320357; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 283 3) := by change ¬ Nat.Prime 319791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 106597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_283 : row 283 = 1 := (row_283.checked).trans rfl
theorem model_count_283 : modelRow 283 = 1 := by decide
theorem modelrow_agrees_283 : modelRow 283 = row 283 := model_count_283.trans row_count_283.symm
def row_284 : RowCertificate 284 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 284 1) := by change ¬ Nat.Prime 323193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 107731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 284 2) := by change ¬ Nat.Prime 322625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 64525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 284 3) := by change Nat.Prime 322057; exact ProjectEulerSolutions.P58PrimeNodes.p_322057.prime
    exact decide_eq_true prime
theorem row_count_284 : row 284 = 1 := (row_284.checked).trans rfl
theorem model_count_284 : modelRow 284 = 1 := by decide
theorem modelrow_agrees_284 : modelRow 284 = row 284 := model_count_284.trans row_count_284.symm
def row_285 : RowCertificate 285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 285 1) := by change ¬ Nat.Prime 325471; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 1201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 285 2) := by change Nat.Prime 324901; exact ProjectEulerSolutions.P58PrimeNodes.p_324901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 285 3) := by change ¬ Nat.Prime 324331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 46333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_285 : row 285 = 1 := (row_285.checked).trans rfl
theorem model_count_285 : modelRow 285 = 1 := by decide
theorem modelrow_agrees_285 : modelRow 285 = row 285 := model_count_285.trans row_count_285.symm
def row_286 : RowCertificate 286 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 286 1) := by change Nat.Prime 327757; exact ProjectEulerSolutions.P58PrimeNodes.p_327757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 286 2) := by change ¬ Nat.Prime 327185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 65437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 286 3) := by change ¬ Nat.Prime 326613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 108871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_286 : row 286 = 1 := (row_286.checked).trans rfl
theorem model_count_286 : modelRow 286 = 1 := by decide
theorem modelrow_agrees_286 : modelRow 286 = row 286 := model_count_286.trans row_count_286.symm
def row_287 : RowCertificate 287 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 287 1) := by change ¬ Nat.Prime 330051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 110017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 287 2) := by change ¬ Nat.Prime 329477; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 19381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 287 3) := by change ¬ Nat.Prime 328903; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 4909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_287 : row 287 = 0 := (row_287.checked).trans rfl
theorem model_count_287 : modelRow 287 = 0 := by decide
theorem modelrow_agrees_287 : modelRow 287 = row 287 := model_count_287.trans row_count_287.symm
def row_288 : RowCertificate 288 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 288 1) := by change ¬ Nat.Prime 332353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 288 2) := by change Nat.Prime 331777; exact ProjectEulerSolutions.P58PrimeNodes.p_331777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 288 3) := by change ¬ Nat.Prime 331201; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_288 : row 288 = 1 := (row_288.checked).trans rfl
theorem model_count_288 : modelRow 288 = 1 := by decide
theorem modelrow_agrees_288 : modelRow 288 = row 288 := model_count_288.trans row_count_288.symm
def row_289 : RowCertificate 289 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 289 1) := by change ¬ Nat.Prime 334663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 47809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 289 2) := by change ¬ Nat.Prime 334085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 66817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 289 3) := by change ¬ Nat.Prime 333507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 111169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_289 : row 289 = 0 := (row_289.checked).trans rfl
theorem model_count_289 : modelRow 289 = 0 := by decide
theorem modelrow_agrees_289 : modelRow 289 = row 289 := model_count_289.trans row_count_289.symm
def row_290 : RowCertificate 290 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 290 1) := by change ¬ Nat.Prime 336981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 112327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 290 2) := by change ¬ Nat.Prime 336401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 25877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 290 3) := by change Nat.Prime 335821; exact ProjectEulerSolutions.P58PrimeNodes.p_335821.prime
    exact decide_eq_true prime
theorem row_count_290 : row 290 = 1 := (row_290.checked).trans rfl
theorem model_count_290 : modelRow 290 = 1 := by decide
theorem modelrow_agrees_290 : modelRow 290 = row 290 := model_count_290.trans row_count_290.symm
def row_291 : RowCertificate 291 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 291 1) := by change Nat.Prime 339307; exact ProjectEulerSolutions.P58PrimeNodes.p_339307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 291 2) := by change ¬ Nat.Prime 338725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 67745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 291 3) := by change ¬ Nat.Prime 338143; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_291 : row 291 = 1 := (row_291.checked).trans rfl
theorem model_count_291 : modelRow 291 = 1 := by decide
theorem modelrow_agrees_291 : modelRow 291 = row 291 := model_count_291.trans row_count_291.symm
def row_292 : RowCertificate 292 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 292 1) := by change Nat.Prime 341641; exact ProjectEulerSolutions.P58PrimeNodes.p_341641.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 292 2) := by change Nat.Prime 341057; exact ProjectEulerSolutions.P58PrimeNodes.p_341057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 292 3) := by change ¬ Nat.Prime 340473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 113491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_292 : row 292 = 2 := (row_292.checked).trans rfl
theorem model_count_292 : modelRow 292 = 2 := by decide
theorem modelrow_agrees_292 : modelRow 292 = row 292 := model_count_292.trans row_count_292.symm
def row_293 : RowCertificate 293 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 293 1) := by change ¬ Nat.Prime 343983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 114661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 293 2) := by change ¬ Nat.Prime 343397; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 293 3) := by change ¬ Nat.Prime 342811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 48973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_293 : row 293 = 0 := (row_293.checked).trans rfl
theorem model_count_293 : modelRow 293 = 0 := by decide
theorem modelrow_agrees_293 : modelRow 293 = row 293 := model_count_293.trans row_count_293.symm
def row_294 : RowCertificate 294 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 294 1) := by change ¬ Nat.Prime 346333; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 294 2) := by change ¬ Nat.Prime 345745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 69149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 294 3) := by change ¬ Nat.Prime 345157; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_294 : row 294 = 0 := (row_294.checked).trans rfl
theorem model_count_294 : modelRow 294 = 0 := by decide
theorem modelrow_agrees_294 : modelRow 294 = row 294 := model_count_294.trans row_count_294.symm
def row_295 : RowCertificate 295 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 295 1) := by change ¬ Nat.Prime 348691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 49813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 295 2) := by change ¬ Nat.Prime 348101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 26777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 295 3) := by change ¬ Nat.Prime 347511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 115837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_295 : row 295 = 0 := (row_295.checked).trans rfl
theorem model_count_295 : modelRow 295 = 0 := by decide
theorem modelrow_agrees_295 : modelRow 295 = row 295 := model_count_295.trans row_count_295.symm
def row_296 : RowCertificate 296 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 296 1) := by change ¬ Nat.Prime 351057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 117019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 296 2) := by change ¬ Nat.Prime 350465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 70093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 296 3) := by change ¬ Nat.Prime 349873; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_296 : row 296 = 0 := (row_296.checked).trans rfl
theorem model_count_296 : modelRow 296 = 0 := by decide
theorem modelrow_agrees_296 : modelRow 296 = row 296 := model_count_296.trans row_count_296.symm
def row_297 : RowCertificate 297 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 297 1) := by change ¬ Nat.Prime 353431; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 27187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 297 2) := by change Nat.Prime 352837; exact ProjectEulerSolutions.P58PrimeNodes.p_352837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 297 3) := by change ¬ Nat.Prime 352243; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_297 : row 297 = 1 := (row_297.checked).trans rfl
theorem model_count_297 : modelRow 297 = 1 := by decide
theorem modelrow_agrees_297 : modelRow 297 = row 297 := model_count_297.trans row_count_297.symm
def row_298 : RowCertificate 298 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 298 1) := by change ¬ Nat.Prime 355813; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 298 2) := by change ¬ Nat.Prime 355217; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 298 3) := by change ¬ Nat.Prime 354621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 118207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_298 : row 298 = 0 := (row_298.checked).trans rfl
theorem model_count_298 : modelRow 298 = 0 := by decide
theorem modelrow_agrees_298 : modelRow 298 = row 298 := model_count_298.trans row_count_298.symm
def row_299 : RowCertificate 299 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 299 1) := by change ¬ Nat.Prime 358203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 119401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 299 2) := by change ¬ Nat.Prime 357605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 71521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 299 3) := by change ¬ Nat.Prime 357007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_299 : row 299 = 0 := (row_299.checked).trans rfl
theorem model_count_299 : modelRow 299 = 0 := by decide
theorem modelrow_agrees_299 : modelRow 299 = row 299 := model_count_299.trans row_count_299.symm
def row_300 : RowCertificate 300 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 300 1) := by change ¬ Nat.Prime 360601; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 18979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 300 2) := by change ¬ Nat.Prime 360001; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 300 3) := by change ¬ Nat.Prime 359401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 51343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_300 : row 300 = 0 := (row_300.checked).trans rfl
theorem model_count_300 : modelRow 300 = 0 := by decide
theorem modelrow_agrees_300 : modelRow 300 = row 300 := model_count_300.trans row_count_300.symm
def row_301 : RowCertificate 301 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 301 1) := by change ¬ Nat.Prime 363007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 9811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 301 2) := by change ¬ Nat.Prime 362405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 72481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 301 3) := by change ¬ Nat.Prime 361803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 120601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_301 : row 301 = 0 := (row_301.checked).trans rfl
theorem model_count_301 : modelRow 301 = 0 := by decide
theorem modelrow_agrees_301 : modelRow 301 = row 301 := model_count_301.trans row_count_301.symm
def row_302 : RowCertificate 302 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 302 1) := by change ¬ Nat.Prime 365421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 121807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 302 2) := by change ¬ Nat.Prime 364817; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 302 3) := by change Nat.Prime 364213; exact ProjectEulerSolutions.P58PrimeNodes.p_364213.prime
    exact decide_eq_true prime
theorem row_count_302 : row 302 = 1 := (row_302.checked).trans rfl
theorem model_count_302 : modelRow 302 = 1 := by decide
theorem modelrow_agrees_302 : modelRow 302 = row 302 := model_count_302.trans row_count_302.symm
def row_303 : RowCertificate 303 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 303 1) := by change ¬ Nat.Prime 367843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 52549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 303 2) := by change ¬ Nat.Prime 367237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 28249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 303 3) := by change Nat.Prime 366631; exact ProjectEulerSolutions.P58PrimeNodes.p_366631.prime
    exact decide_eq_true prime
theorem row_count_303 : row 303 = 1 := (row_303.checked).trans rfl
theorem model_count_303 : modelRow 303 = 1 := by decide
theorem modelrow_agrees_303 : modelRow 303 = row 303 := model_count_303.trans row_count_303.symm
def row_304 : RowCertificate 304 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 304 1) := by change ¬ Nat.Prime 370273; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 304 2) := by change ¬ Nat.Prime 369665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 73933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 304 3) := by change ¬ Nat.Prime 369057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 123019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_304 : row 304 = 0 := (row_304.checked).trans rfl
theorem model_count_304 : modelRow 304 = 0 := by decide
theorem modelrow_agrees_304 : modelRow 304 = row 304 := model_count_304.trans row_count_304.symm
def row_305 : RowCertificate 305 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 305 1) := by change ¬ Nat.Prime 372711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 124237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 305 2) := by change ¬ Nat.Prime 372101; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 1597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 305 3) := by change Nat.Prime 371491; exact ProjectEulerSolutions.P58PrimeNodes.p_371491.prime
    exact decide_eq_true prime
theorem row_count_305 : row 305 = 1 := (row_305.checked).trans rfl
theorem model_count_305 : modelRow 305 = 1 := by decide
theorem modelrow_agrees_305 : modelRow 305 = row 305 := model_count_305.trans row_count_305.symm
def row_306 : RowCertificate 306 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 306 1) := by change Nat.Prime 375157; exact ProjectEulerSolutions.P58PrimeNodes.p_375157.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 306 2) := by change ¬ Nat.Prime 374545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 74909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 306 3) := by change ¬ Nat.Prime 373933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 53419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_306 : row 306 = 1 := (row_306.checked).trans rfl
theorem model_count_306 : modelRow 306 = 1 := by decide
theorem modelrow_agrees_306 : modelRow 306 = row 306 := model_count_306.trans row_count_306.symm
def row_307 : RowCertificate 307 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 307 1) := by change ¬ Nat.Prime 377611; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 307 2) := by change ¬ Nat.Prime 376997; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 307 3) := by change ¬ Nat.Prime 376383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 125461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_307 : row 307 = 0 := (row_307.checked).trans rfl
theorem model_count_307 : modelRow 307 = 0 := by decide
theorem modelrow_agrees_307 : modelRow 307 = row 307 := model_count_307.trans row_count_307.symm
def row_308 : RowCertificate 308 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 308 1) := by change ¬ Nat.Prime 380073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 308 2) := by change ¬ Nat.Prime 379457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 308 3) := by change ¬ Nat.Prime 378841; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_308 : row 308 = 0 := (row_308.checked).trans rfl
theorem model_count_308 : modelRow 308 = 0 := by decide
theorem modelrow_agrees_308 : modelRow 308 = row 308 := model_count_308.trans row_count_308.symm
def row_309 : RowCertificate 309 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 309 1) := by change ¬ Nat.Prime 382543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 309 2) := by change ¬ Nat.Prime 381925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 309 3) := by change ¬ Nat.Prime 381307; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_309 : row 309 = 0 := (row_309.checked).trans rfl
theorem model_count_309 : modelRow 309 = 0 := by decide
theorem modelrow_agrees_309 : modelRow 309 = row 309 := model_count_309.trans row_count_309.symm
def row_310 : RowCertificate 310 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 310 1) := by change ¬ Nat.Prime 385021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 310 2) := by change ¬ Nat.Prime 384401; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 1429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 310 3) := by change ¬ Nat.Prime 383781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_310 : row 310 = 0 := (row_310.checked).trans rfl
theorem model_count_310 : modelRow 310 = 0 := by decide
theorem modelrow_agrees_310 : modelRow 310 = row 310 := model_count_310.trans row_count_310.symm
def row_311 : RowCertificate 311 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 311 1) := by change ¬ Nat.Prime 387507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 311 2) := by change ¬ Nat.Prime 386885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 311 3) := by change Nat.Prime 386263; exact ProjectEulerSolutions.P58PrimeNodes.p_386263.prime
    exact decide_eq_true prime
theorem row_count_311 : row 311 = 1 := (row_311.checked).trans rfl
theorem model_count_311 : modelRow 311 = 1 := by decide
theorem modelrow_agrees_311 : modelRow 311 = row 311 := model_count_311.trans row_count_311.symm
def row_312 : RowCertificate 312 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 312 1) := by change Nat.Prime 390001; exact ProjectEulerSolutions.P58PrimeNodes.p_390001.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 312 2) := by change ¬ Nat.Prime 389377; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 312 3) := by change ¬ Nat.Prime 388753; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_312 : row 312 = 1 := (row_312.checked).trans rfl
theorem model_count_312 : modelRow 312 = 1 := by decide
theorem modelrow_agrees_312 : modelRow 312 = row 312 := model_count_312.trans row_count_312.symm
def row_313 : RowCertificate 313 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 313 1) := by change Nat.Prime 392503; exact ProjectEulerSolutions.P58PrimeNodes.p_392503.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 313 2) := by change ¬ Nat.Prime 391877; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 313 3) := by change ¬ Nat.Prime 391251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_313 : row 313 = 1 := (row_313.checked).trans rfl
theorem model_count_313 : modelRow 313 = 1 := by decide
theorem modelrow_agrees_313 : modelRow 313 = row 313 := model_count_313.trans row_count_313.symm
def row_314 : RowCertificate 314 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 314 1) := by change ¬ Nat.Prime 395013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 314 2) := by change ¬ Nat.Prime 394385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 314 3) := by change ¬ Nat.Prime 393757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_314 : row 314 = 0 := (row_314.checked).trans rfl
theorem model_count_314 : modelRow 314 = 0 := by decide
theorem modelrow_agrees_314 : modelRow 314 = row 314 := model_count_314.trans row_count_314.symm
def row_315 : RowCertificate 315 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 315 1) := by change ¬ Nat.Prime 397531; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 1201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 315 2) := by change ¬ Nat.Prime 396901; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 315 3) := by change ¬ Nat.Prime 396271; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_315 : row 315 = 0 := (row_315.checked).trans rfl
theorem model_count_315 : modelRow 315 = 0 := by decide
theorem modelrow_agrees_315 : modelRow 315 = row 315 := model_count_315.trans row_count_315.symm
def row_316 : RowCertificate 316 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 316 1) := by change ¬ Nat.Prime 400057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 316 2) := by change ¬ Nat.Prime 399425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 316 3) := by change ¬ Nat.Prime 398793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_316 : row 316 = 0 := (row_316.checked).trans rfl
theorem model_count_316 : modelRow 316 = 0 := by decide
theorem modelrow_agrees_316 : modelRow 316 = row 316 := model_count_316.trans row_count_316.symm
def row_317 : RowCertificate 317 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 317 1) := by change ¬ Nat.Prime 402591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 317 2) := by change Nat.Prime 401957; exact ProjectEulerSolutions.P58PrimeNodes.p_401957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 317 3) := by change ¬ Nat.Prime 401323; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_317 : row 317 = 1 := (row_317.checked).trans rfl
theorem model_count_317 : modelRow 317 = 1 := by decide
theorem modelrow_agrees_317 : modelRow 317 = row 317 := model_count_317.trans row_count_317.symm
def row_318 : RowCertificate 318 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 318 1) := by change ¬ Nat.Prime 405133; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 318 2) := by change Nat.Prime 404497; exact ProjectEulerSolutions.P58PrimeNodes.p_404497.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 318 3) := by change Nat.Prime 403861; exact ProjectEulerSolutions.P58PrimeNodes.p_403861.prime
    exact decide_eq_true prime
theorem row_count_318 : row 318 = 2 := (row_318.checked).trans rfl
theorem model_count_318 : modelRow 318 = 2 := by decide
theorem modelrow_agrees_318 : modelRow 318 = row 318 := model_count_318.trans row_count_318.symm
def row_319 : RowCertificate 319 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 319 1) := by change ¬ Nat.Prime 407683; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 319 2) := by change ¬ Nat.Prime 407045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 319 3) := by change ¬ Nat.Prime 406407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_319 : row 319 = 0 := (row_319.checked).trans rfl
theorem model_count_319 : modelRow 319 = 0 := by decide
theorem modelrow_agrees_319 : modelRow 319 = row 319 := model_count_319.trans row_count_319.symm
def row_320 : RowCertificate 320 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 320 1) := by change ¬ Nat.Prime 410241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 320 2) := by change ¬ Nat.Prime 409601; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 320 3) := by change ¬ Nat.Prime 408961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58423) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_320 : row 320 = 0 := (row_320.checked).trans rfl
theorem model_count_320 : modelRow 320 = 0 := by decide
theorem modelrow_agrees_320 : modelRow 320 = row 320 := model_count_320.trans row_count_320.symm
def row_321 : RowCertificate 321 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 321 1) := by change Nat.Prime 412807; exact ProjectEulerSolutions.P58PrimeNodes.p_412807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 321 2) := by change ¬ Nat.Prime 412165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 82433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 321 3) := by change ¬ Nat.Prime 411523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_321 : row 321 = 1 := (row_321.checked).trans rfl
theorem model_count_321 : modelRow 321 = 1 := by decide
theorem modelrow_agrees_321 : modelRow 321 = row 321 := model_count_321.trans row_count_321.symm
def row_322 : RowCertificate 322 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 322 1) := by change Nat.Prime 415381; exact ProjectEulerSolutions.P58PrimeNodes.p_415381.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 322 2) := by change Nat.Prime 414737; exact ProjectEulerSolutions.P58PrimeNodes.p_414737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 322 3) := by change ¬ Nat.Prime 414093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 138031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_322 : row 322 = 2 := (row_322.checked).trans rfl
theorem model_count_322 : modelRow 322 = 2 := by decide
theorem modelrow_agrees_322 : modelRow 322 = row 322 := model_count_322.trans row_count_322.symm
def row_323 : RowCertificate 323 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 323 1) := by change ¬ Nat.Prime 417963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 323 2) := by change Nat.Prime 417317; exact ProjectEulerSolutions.P58PrimeNodes.p_417317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 323 3) := by change ¬ Nat.Prime 416671; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_323 : row 323 = 1 := (row_323.checked).trans rfl
theorem model_count_323 : modelRow 323 = 1 := by decide
theorem modelrow_agrees_323 : modelRow 323 = row 323 := model_count_323.trans row_count_323.symm
def row_324 : RowCertificate 324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 324 1) := by change ¬ Nat.Prime 420553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 324 2) := by change ¬ Nat.Prime 419905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 324 3) := by change ¬ Nat.Prime 419257; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_324 : row 324 = 0 := (row_324.checked).trans rfl
theorem model_count_324 : modelRow 324 = 0 := by decide
theorem modelrow_agrees_324 : modelRow 324 = row 324 := model_count_324.trans row_count_324.symm
def row_325 : RowCertificate 325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 325 1) := by change ¬ Nat.Prime 423151; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 1153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 325 2) := by change ¬ Nat.Prime 422501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 325 3) := by change ¬ Nat.Prime 421851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_325 : row 325 = 0 := (row_325.checked).trans rfl
theorem model_count_325 : modelRow 325 = 0 := by decide
theorem modelrow_agrees_325 : modelRow 325 = row 325 := model_count_325.trans row_count_325.symm
def row_326 : RowCertificate 326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 326 1) := by change ¬ Nat.Prime 425757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 326 2) := by change ¬ Nat.Prime 425105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 326 3) := by change ¬ Nat.Prime 424453; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_326 : row 326 = 0 := (row_326.checked).trans rfl
theorem model_count_326 : modelRow 326 = 0 := by decide
theorem modelrow_agrees_326 : modelRow 326 = row 326 := model_count_326.trans row_count_326.symm
def row_327 : RowCertificate 327 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 327 1) := by change ¬ Nat.Prime 428371; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 327 2) := by change Nat.Prime 427717; exact ProjectEulerSolutions.P58PrimeNodes.p_427717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 327 3) := by change ¬ Nat.Prime 427063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_327 : row 327 = 1 := (row_327.checked).trans rfl
theorem model_count_327 : modelRow 327 = 1 := by decide
theorem modelrow_agrees_327 : modelRow 327 = row 327 := model_count_327.trans row_count_327.symm
def row_328 : RowCertificate 328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 328 1) := by change ¬ Nat.Prime 430993; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 328 2) := by change ¬ Nat.Prime 430337; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 328 3) := by change ¬ Nat.Prime 429681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_328 : row 328 = 0 := (row_328.checked).trans rfl
theorem model_count_328 : modelRow 328 = 0 := by decide
theorem modelrow_agrees_328 : modelRow 328 = row 328 := model_count_328.trans row_count_328.symm
def row_329 : RowCertificate 329 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 329 1) := by change ¬ Nat.Prime 433623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 144541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 329 2) := by change ¬ Nat.Prime 432965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 86593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 329 3) := by change ¬ Nat.Prime 432307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_329 : row 329 = 0 := (row_329.checked).trans rfl
theorem model_count_329 : modelRow 329 = 0 := by decide
theorem modelrow_agrees_329 : modelRow 329 = row 329 := model_count_329.trans row_count_329.symm
def row_330 : RowCertificate 330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 330 1) := by change ¬ Nat.Prime 436261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 62323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 330 2) := by change ¬ Nat.Prime 435601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 330 3) := by change ¬ Nat.Prime 434941; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 33457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_330 : row 330 = 0 := (row_330.checked).trans rfl
theorem model_count_330 : modelRow 330 = 0 := by decide
theorem modelrow_agrees_330 : modelRow 330 = row 330 := model_count_330.trans row_count_330.symm
def row_331 : RowCertificate 331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 331 1) := by change ¬ Nat.Prime 438907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 62701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 331 2) := by change ¬ Nat.Prime 438245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 87649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 331 3) := by change ¬ Nat.Prime 437583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 145861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_331 : row 331 = 0 := (row_331.checked).trans rfl
theorem model_count_331 : modelRow 331 = 0 := by decide
theorem modelrow_agrees_331 : modelRow 331 = row 331 := model_count_331.trans row_count_331.symm
def row_332 : RowCertificate 332 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 332 1) := by change ¬ Nat.Prime 441561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 147187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 332 2) := by change ¬ Nat.Prime 440897; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 1249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 332 3) := by change ¬ Nat.Prime 440233; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_332 : row 332 = 0 := (row_332.checked).trans rfl
theorem model_count_332 : modelRow 332 = 0 := by decide
theorem modelrow_agrees_332 : modelRow 332 = row 332 := model_count_332.trans row_count_332.symm
def row_333 : RowCertificate 333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 333 1) := by change ¬ Nat.Prime 444223; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 34171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 333 2) := by change ¬ Nat.Prime 443557; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 8369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 333 3) := by change ¬ Nat.Prime 442891; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 6067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_333 : row 333 = 0 := (row_333.checked).trans rfl
theorem model_count_333 : modelRow 333 = 0 := by decide
theorem modelrow_agrees_333 : modelRow 333 = row 333 := model_count_333.trans row_count_333.symm
def row_334 : RowCertificate 334 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 334 1) := by change Nat.Prime 446893; exact ProjectEulerSolutions.P58PrimeNodes.p_446893.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 334 2) := by change ¬ Nat.Prime 446225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 89245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 334 3) := by change ¬ Nat.Prime 445557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 148519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_334 : row 334 = 1 := (row_334.checked).trans rfl
theorem model_count_334 : modelRow 334 = 1 := by decide
theorem modelrow_agrees_334 : modelRow 334 = row 334 := model_count_334.trans row_count_334.symm
def row_335 : RowCertificate 335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 335 1) := by change ¬ Nat.Prime 449571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 149857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 335 2) := by change ¬ Nat.Prime 448901; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 335 3) := by change ¬ Nat.Prime 448231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 64033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_335 : row 335 = 0 := (row_335.checked).trans rfl
theorem model_count_335 : modelRow 335 = 0 := by decide
theorem modelrow_agrees_335 : modelRow 335 = row 335 := model_count_335.trans row_count_335.symm
def row_336 : RowCertificate 336 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 336 1) := by change ¬ Nat.Prime 452257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 34789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 336 2) := by change ¬ Nat.Prime 451585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 90317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 336 3) := by change Nat.Prime 450913; exact ProjectEulerSolutions.P58PrimeNodes.p_450913.prime
    exact decide_eq_true prime
theorem row_count_336 : row 336 = 1 := (row_336.checked).trans rfl
theorem model_count_336 : modelRow 336 = 1 := by decide
theorem modelrow_agrees_336 : modelRow 336 = row 336 := model_count_336.trans row_count_336.symm
def row_337 : RowCertificate 337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 337 1) := by change ¬ Nat.Prime 454951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 64993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 337 2) := by change Nat.Prime 454277; exact ProjectEulerSolutions.P58PrimeNodes.p_454277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 337 3) := by change ¬ Nat.Prime 453603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 151201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_337 : row 337 = 1 := (row_337.checked).trans rfl
theorem model_count_337 : modelRow 337 = 1 := by decide
theorem modelrow_agrees_337 : modelRow 337 = row 337 := model_count_337.trans row_count_337.symm
def row_338 : RowCertificate 338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 338 1) := by change ¬ Nat.Prime 457653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 152551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 338 2) := by change ¬ Nat.Prime 456977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 26881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 338 3) := by change ¬ Nat.Prime 456301; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_338 : row 338 = 0 := (row_338.checked).trans rfl
theorem model_count_338 : modelRow 338 = 0 := by decide
theorem modelrow_agrees_338 : modelRow 338 = row 338 := model_count_338.trans row_count_338.symm
def row_339 : RowCertificate 339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 339 1) := by change ¬ Nat.Prime 460363; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 339 2) := by change ¬ Nat.Prime 459685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 91937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 339 3) := by change Nat.Prime 459007; exact ProjectEulerSolutions.P58PrimeNodes.p_459007.prime
    exact decide_eq_true prime
theorem row_count_339 : row 339 = 1 := (row_339.checked).trans rfl
theorem model_count_339 : modelRow 339 = 1 := by decide
theorem modelrow_agrees_339 : modelRow 339 = row 339 := model_count_339.trans row_count_339.symm
def row_340 : RowCertificate 340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 340 1) := by change ¬ Nat.Prime 463081; exact PrattCertificates.composite_of_product (factor := 571) (cofactor := 811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 340 2) := by change Nat.Prime 462401; exact ProjectEulerSolutions.P58PrimeNodes.p_462401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 340 3) := by change ¬ Nat.Prime 461721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 153907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_340 : row 340 = 1 := (row_340.checked).trans rfl
theorem model_count_340 : modelRow 340 = 1 := by decide
theorem modelrow_agrees_340 : modelRow 340 = row 340 := model_count_340.trans row_count_340.symm
def row_341 : RowCertificate 341 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 341 1) := by change ¬ Nat.Prime 465807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 155269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 341 2) := by change ¬ Nat.Prime 465125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 93025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 341 3) := by change ¬ Nat.Prime 464443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 66349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_341 : row 341 = 0 := (row_341.checked).trans rfl
theorem model_count_341 : modelRow 341 = 0 := by decide
theorem modelrow_agrees_341 : modelRow 341 = row 341 := model_count_341.trans row_count_341.symm
def row_342 : RowCertificate 342 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 342 1) := by change ¬ Nat.Prime 468541; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 342 2) := by change ¬ Nat.Prime 467857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 35989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 342 3) := by change ¬ Nat.Prime 467173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 66739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_342 : row 342 = 0 := (row_342.checked).trans rfl
theorem model_count_342 : modelRow 342 = 0 := by decide
theorem modelrow_agrees_342 : modelRow 342 = row 342 := model_count_342.trans row_count_342.symm
def row_343 : RowCertificate 343 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 343 1) := by change Nat.Prime 471283; exact ProjectEulerSolutions.P58PrimeNodes.p_471283.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 343 2) := by change Nat.Prime 470597; exact ProjectEulerSolutions.P58PrimeNodes.p_470597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 343 3) := by change ¬ Nat.Prime 469911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 156637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_343 : row 343 = 2 := (row_343.checked).trans rfl
theorem model_count_343 : modelRow 343 = 2 := by decide
theorem modelrow_agrees_343 : modelRow 343 = row 343 := model_count_343.trans row_count_343.symm
def row_344 : RowCertificate 344 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 344 1) := by change ¬ Nat.Prime 474033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 344 2) := by change ¬ Nat.Prime 473345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 344 3) := by change ¬ Nat.Prime 472657; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 15247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_344 : row 344 = 0 := (row_344.checked).trans rfl
theorem model_count_344 : modelRow 344 = 0 := by decide
theorem modelrow_agrees_344 : modelRow 344 = row 344 := model_count_344.trans row_count_344.symm
def row_345 : RowCertificate 345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 345 1) := by change ¬ Nat.Prime 476791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 345 2) := by change Nat.Prime 476101; exact ProjectEulerSolutions.P58PrimeNodes.p_476101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 345 3) := by change ¬ Nat.Prime 475411; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 2389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_345 : row 345 = 1 := (row_345.checked).trans rfl
theorem model_count_345 : modelRow 345 = 1 := by decide
theorem modelrow_agrees_345 : modelRow 345 = row 345 := model_count_345.trans row_count_345.symm
def row_346 : RowCertificate 346 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 346 1) := by change ¬ Nat.Prime 479557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 346 2) := by change ¬ Nat.Prime 478865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 346 3) := by change ¬ Nat.Prime 478173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_346 : row 346 = 0 := (row_346.checked).trans rfl
theorem model_count_346 : modelRow 346 = 0 := by decide
theorem modelrow_agrees_346 : modelRow 346 = row 346 := model_count_346.trans row_count_346.symm
def row_347 : RowCertificate 347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 347 1) := by change ¬ Nat.Prime 482331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 347 2) := by change ¬ Nat.Prime 481637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 347 3) := by change ¬ Nat.Prime 480943; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 1453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_347 : row 347 = 0 := (row_347.checked).trans rfl
theorem model_count_347 : modelRow 347 = 0 := by decide
theorem modelrow_agrees_347 : modelRow 347 = row 347 := model_count_347.trans row_count_347.symm
def row_348 : RowCertificate 348 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 348 1) := by change Nat.Prime 485113; exact ProjectEulerSolutions.P58PrimeNodes.p_485113.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 348 2) := by change Nat.Prime 484417; exact ProjectEulerSolutions.P58PrimeNodes.p_484417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 348 3) := by change ¬ Nat.Prime 483721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69103) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_348 : row 348 = 2 := (row_348.checked).trans rfl
theorem model_count_348 : modelRow 348 = 2 := by decide
theorem modelrow_agrees_348 : modelRow 348 = row 348 := model_count_348.trans row_count_348.symm
def row_349 : RowCertificate 349 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 349 1) := by change ¬ Nat.Prime 487903; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 349 2) := by change ¬ Nat.Prime 487205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 97441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 349 3) := by change ¬ Nat.Prime 486507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 162169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_349 : row 349 = 0 := (row_349.checked).trans rfl
theorem model_count_349 : modelRow 349 = 0 := by decide
theorem modelrow_agrees_349 : modelRow 349 = row 349 := model_count_349.trans row_count_349.symm
def row_350 : RowCertificate 350 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 350 1) := by change ¬ Nat.Prime 490701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 163567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 350 2) := by change Nat.Prime 490001; exact ProjectEulerSolutions.P58PrimeNodes.p_490001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 350 3) := by change ¬ Nat.Prime 489301; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 7303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_350 : row 350 = 1 := (row_350.checked).trans rfl
theorem model_count_350 : modelRow 350 = 1 := by decide
theorem modelrow_agrees_350 : modelRow 350 = row 350 := model_count_350.trans row_count_350.symm
def row_351 : RowCertificate 351 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 351 1) := by change ¬ Nat.Prime 493507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 70501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 351 2) := by change ¬ Nat.Prime 492805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 98561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 351 3) := by change Nat.Prime 492103; exact ProjectEulerSolutions.P58PrimeNodes.p_492103.prime
    exact decide_eq_true prime
theorem row_count_351 : row 351 = 1 := (row_351.checked).trans rfl
theorem model_count_351 : modelRow 351 = 1 := by decide
theorem modelrow_agrees_351 : modelRow 351 = row 351 := model_count_351.trans row_count_351.symm
def row_352 : RowCertificate 352 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 352 1) := by change ¬ Nat.Prime 496321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 70903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 352 2) := by change Nat.Prime 495617; exact ProjectEulerSolutions.P58PrimeNodes.p_495617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 352 3) := by change ¬ Nat.Prime 494913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 164971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_352 : row 352 = 1 := (row_352.checked).trans rfl
theorem model_count_352 : modelRow 352 = 1 := by decide
theorem modelrow_agrees_352 : modelRow 352 = row 352 := model_count_352.trans row_count_352.symm
def row_353 : RowCertificate 353 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 353 1) := by change ¬ Nat.Prime 499143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 166381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 353 2) := by change ¬ Nat.Prime 498437; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 12157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 353 3) := by change ¬ Nat.Prime 497731; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 38287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_353 : row 353 = 0 := (row_353.checked).trans rfl
theorem model_count_353 : modelRow 353 = 0 := by decide
theorem modelrow_agrees_353 : modelRow 353 = row 353 := model_count_353.trans row_count_353.symm
def row_354 : RowCertificate 354 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 354 1) := by change ¬ Nat.Prime 501973; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 2251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 354 2) := by change ¬ Nat.Prime 501265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 100253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 354 3) := by change ¬ Nat.Prime 500557; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 16147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_354 : row 354 = 0 := (row_354.checked).trans rfl
theorem model_count_354 : modelRow 354 = 0 := by decide
theorem modelrow_agrees_354 : modelRow 354 = row 354 := model_count_354.trans row_count_354.symm
def row_355 : RowCertificate 355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 355 1) := by change ¬ Nat.Prime 504811; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 26569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 355 2) := by change ¬ Nat.Prime 504101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 38777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 355 3) := by change ¬ Nat.Prime 503391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 167797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_355 : row 355 = 0 := (row_355.checked).trans rfl
theorem model_count_355 : modelRow 355 = 0 := by decide
theorem modelrow_agrees_355 : modelRow 355 = row 355 := model_count_355.trans row_count_355.symm
def row_356 : RowCertificate 356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 356 1) := by change ¬ Nat.Prime 507657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 169219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 356 2) := by change ¬ Nat.Prime 506945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 101389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 356 3) := by change ¬ Nat.Prime 506233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 72319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_356 : row 356 = 0 := (row_356.checked).trans rfl
theorem model_count_356 : modelRow 356 = 0 := by decide
theorem modelrow_agrees_356 : modelRow 356 = row 356 := model_count_356.trans row_count_356.symm
def row_357 : RowCertificate 357 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 357 1) := by change ¬ Nat.Prime 510511; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 26869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 357 2) := by change Nat.Prime 509797; exact ProjectEulerSolutions.P58PrimeNodes.p_509797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 357 3) := by change ¬ Nat.Prime 509083; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 13759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_357 : row 357 = 1 := (row_357.checked).trans rfl
theorem model_count_357 : modelRow 357 = 1 := by decide
theorem modelrow_agrees_357 : modelRow 357 = row 357 := model_count_357.trans row_count_357.symm
def row_358 : RowCertificate 358 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 358 1) := by change ¬ Nat.Prime 513373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 73339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 358 2) := by change Nat.Prime 512657; exact ProjectEulerSolutions.P58PrimeNodes.p_512657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 358 3) := by change ¬ Nat.Prime 511941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 170647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_358 : row 358 = 1 := (row_358.checked).trans rfl
theorem model_count_358 : modelRow 358 = 1 := by decide
theorem modelrow_agrees_358 : modelRow 358 = row 358 := model_count_358.trans row_count_358.symm
def row_359 : RowCertificate 359 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 359 1) := by change ¬ Nat.Prime 516243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 172081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 359 2) := by change ¬ Nat.Prime 515525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 103105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 359 3) := by change ¬ Nat.Prime 514807; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 4723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_359 : row 359 = 0 := (row_359.checked).trans rfl
theorem model_count_359 : modelRow 359 = 0 := by decide
theorem modelrow_agrees_359 : modelRow 359 = row 359 := model_count_359.trans row_count_359.symm
def row_360 : RowCertificate 360 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 360 1) := by change Nat.Prime 519121; exact ProjectEulerSolutions.P58PrimeNodes.p_519121.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 360 2) := by change ¬ Nat.Prime 518401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 39877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 360 3) := by change ¬ Nat.Prime 517681; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 1063) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_360 : row 360 = 1 := (row_360.checked).trans rfl
theorem model_count_360 : modelRow 360 = 1 := by decide
theorem modelrow_agrees_360 : modelRow 360 = row 360 := model_count_360.trans row_count_360.symm
def row_361 : RowCertificate 361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 361 1) := by change ¬ Nat.Prime 522007; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 361 2) := by change ¬ Nat.Prime 521285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 104257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 361 3) := by change ¬ Nat.Prime 520563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 173521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_361 : row 361 = 0 := (row_361.checked).trans rfl
theorem model_count_361 : modelRow 361 = 0 := by decide
theorem modelrow_agrees_361 : modelRow 361 = row 361 := model_count_361.trans row_count_361.symm
def row_362 : RowCertificate 362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 362 1) := by change ¬ Nat.Prime 524901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 174967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 362 2) := by change ¬ Nat.Prime 524177; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 1789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 362 3) := by change ¬ Nat.Prime 523453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 74779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_362 : row 362 = 0 := (row_362.checked).trans rfl
theorem model_count_362 : modelRow 362 = 0 := by decide
theorem modelrow_agrees_362 : modelRow 362 = row 362 := model_count_362.trans row_count_362.symm
def row_363 : RowCertificate 363 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 363 1) := by change Nat.Prime 527803; exact ProjectEulerSolutions.P58PrimeNodes.p_527803.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 363 2) := by change ¬ Nat.Prime 527077; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 363 3) := by change ¬ Nat.Prime 526351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 75193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_363 : row 363 = 1 := (row_363.checked).trans rfl
theorem model_count_363 : modelRow 363 = 1 := by decide
theorem modelrow_agrees_363 : modelRow 363 = row 363 := model_count_363.trans row_count_363.symm
def row_364 : RowCertificate 364 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 364 1) := by change Nat.Prime 530713; exact ProjectEulerSolutions.P58PrimeNodes.p_530713.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 364 2) := by change ¬ Nat.Prime 529985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 105997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 364 3) := by change ¬ Nat.Prime 529257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 176419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_364 : row 364 = 1 := (row_364.checked).trans rfl
theorem model_count_364 : modelRow 364 = 1 := by decide
theorem modelrow_agrees_364 : modelRow 364 = row 364 := model_count_364.trans row_count_364.symm
def row_365 : RowCertificate 365 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 365 1) := by change ¬ Nat.Prime 533631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 177877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 365 2) := by change ¬ Nat.Prime 532901; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 4889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 365 3) := by change ¬ Nat.Prime 532171; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 28009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_365 : row 365 = 0 := (row_365.checked).trans rfl
theorem model_count_365 : modelRow 365 = 0 := by decide
theorem modelrow_agrees_365 : modelRow 365 = row 365 := model_count_365.trans row_count_365.symm
def row_366 : RowCertificate 366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 366 1) := by change ¬ Nat.Prime 536557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 76651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 366 2) := by change ¬ Nat.Prime 535825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 107165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 366 3) := by change ¬ Nat.Prime 535093; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 41161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_366 : row 366 = 0 := (row_366.checked).trans rfl
theorem model_count_366 : modelRow 366 = 0 := by decide
theorem modelrow_agrees_366 : modelRow 366 = row 366 := model_count_366.trans row_count_366.symm
def row_367 : RowCertificate 367 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 367 1) := by change ¬ Nat.Prime 539491; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 6829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 367 2) := by change ¬ Nat.Prime 538757; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 14561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 367 3) := by change ¬ Nat.Prime 538023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 179341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_367 : row 367 = 0 := (row_367.checked).trans rfl
theorem model_count_367 : modelRow 367 = 0 := by decide
theorem modelrow_agrees_367 : modelRow 367 = row 367 := model_count_367.trans row_count_367.symm
def row_368 : RowCertificate 368 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 368 1) := by change ¬ Nat.Prime 542433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 180811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 368 2) := by change ¬ Nat.Prime 541697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 41669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 368 3) := by change Nat.Prime 540961; exact ProjectEulerSolutions.P58PrimeNodes.p_540961.prime
    exact decide_eq_true prime
theorem row_count_368 : row 368 = 1 := (row_368.checked).trans rfl
theorem model_count_368 : modelRow 368 = 1 := by decide
theorem modelrow_agrees_368 : modelRow 368 = row 368 := model_count_368.trans row_count_368.symm
def row_369 : RowCertificate 369 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 369 1) := by change ¬ Nat.Prime 545383; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 17593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 369 2) := by change ¬ Nat.Prime 544645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 108929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 369 3) := by change ¬ Nat.Prime 543907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 77701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_369 : row 369 = 0 := (row_369.checked).trans rfl
theorem model_count_369 : modelRow 369 = 0 := by decide
theorem modelrow_agrees_369 : modelRow 369 = row 369 := model_count_369.trans row_count_369.symm
def row_370 : RowCertificate 370 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 370 1) := by change ¬ Nat.Prime 548341; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 5653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 370 2) := by change Nat.Prime 547601; exact ProjectEulerSolutions.P58PrimeNodes.p_547601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 370 3) := by change ¬ Nat.Prime 546861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 182287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_370 : row 370 = 1 := (row_370.checked).trans rfl
theorem model_count_370 : modelRow 370 = 1 := by decide
theorem modelrow_agrees_370 : modelRow 370 = row 370 := model_count_370.trans row_count_370.symm
def row_371 : RowCertificate 371 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 371 1) := by change ¬ Nat.Prime 551307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 183769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 371 2) := by change ¬ Nat.Prime 550565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 110113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 371 3) := by change ¬ Nat.Prime 549823; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 1129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_371 : row 371 = 0 := (row_371.checked).trans rfl
theorem model_count_371 : modelRow 371 = 0 := by decide
theorem modelrow_agrees_371 : modelRow 371 = row 371 := model_count_371.trans row_count_371.symm
def row_372 : RowCertificate 372 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 372 1) := by change ¬ Nat.Prime 554281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 79183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 372 2) := by change ¬ Nat.Prime 553537; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 32561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 372 3) := by change Nat.Prime 552793; exact ProjectEulerSolutions.P58PrimeNodes.p_552793.prime
    exact decide_eq_true prime
theorem row_count_372 : row 372 = 1 := (row_372.checked).trans rfl
theorem model_count_372 : modelRow 372 = 1 := by decide
theorem modelrow_agrees_372 : modelRow 372 = row 372 := model_count_372.trans row_count_372.symm
def row_373 : RowCertificate 373 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 373 1) := by change ¬ Nat.Prime 557263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 79609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 373 2) := by change ¬ Nat.Prime 556517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 42809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 373 3) := by change ¬ Nat.Prime 555771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 185257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_373 : row 373 = 0 := (row_373.checked).trans rfl
theorem model_count_373 : modelRow 373 = 0 := by decide
theorem modelrow_agrees_373 : modelRow 373 = row 373 := model_count_373.trans row_count_373.symm
def row_374 : RowCertificate 374 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 374 1) := by change ¬ Nat.Prime 560253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 186751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 374 2) := by change ¬ Nat.Prime 559505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 111901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 374 3) := by change Nat.Prime 558757; exact ProjectEulerSolutions.P58PrimeNodes.p_558757.prime
    exact decide_eq_true prime
theorem row_count_374 : row 374 = 1 := (row_374.checked).trans rfl
theorem model_count_374 : modelRow 374 = 1 := by decide
theorem modelrow_agrees_374 : modelRow 374 = row 374 := model_count_374.trans row_count_374.symm
def row_375 : RowCertificate 375 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 375 1) := by change ¬ Nat.Prime 563251; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 375 2) := by change Nat.Prime 562501; exact ProjectEulerSolutions.P58PrimeNodes.p_562501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 375 3) := by change ¬ Nat.Prime 561751; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_375 : row 375 = 1 := (row_375.checked).trans rfl
theorem model_count_375 : modelRow 375 = 1 := by decide
theorem modelrow_agrees_375 : modelRow 375 = row 375 := model_count_375.trans row_count_375.symm
def row_376 : RowCertificate 376 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 376 1) := by change ¬ Nat.Prime 566257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 29803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 376 2) := by change ¬ Nat.Prime 565505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 376 3) := by change ¬ Nat.Prime 564753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 188251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_376 : row 376 = 0 := (row_376.checked).trans rfl
theorem model_count_376 : modelRow 376 = 0 := by decide
theorem modelrow_agrees_376 : modelRow 376 = row 376 := model_count_376.trans row_count_376.symm
def row_377 : RowCertificate 377 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 377 1) := by change ¬ Nat.Prime 569271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 377 2) := by change ¬ Nat.Prime 568517; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 5861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 377 3) := by change ¬ Nat.Prime 567763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_377 : row 377 = 0 := (row_377.checked).trans rfl
theorem model_count_377 : modelRow 377 = 0 := by decide
theorem modelrow_agrees_377 : modelRow 377 = row 377 := model_count_377.trans row_count_377.symm
def row_378 : RowCertificate 378 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 378 1) := by change ¬ Nat.Prime 572293; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 3511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 378 2) := by change ¬ Nat.Prime 571537; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 1097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 378 3) := by change Nat.Prime 570781; exact ProjectEulerSolutions.P58PrimeNodes.p_570781.prime
    exact decide_eq_true prime
theorem row_count_378 : row 378 = 1 := (row_378.checked).trans rfl
theorem model_count_378 : modelRow 378 = 1 := by decide
theorem modelrow_agrees_378 : modelRow 378 = row 378 := model_count_378.trans row_count_378.symm
def row_379 : RowCertificate 379 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 379 1) := by change ¬ Nat.Prime 575323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 379 2) := by change ¬ Nat.Prime 574565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 379 3) := by change ¬ Nat.Prime 573807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_379 : row 379 = 0 := (row_379.checked).trans rfl
theorem model_count_379 : modelRow 379 = 0 := by decide
theorem modelrow_agrees_379 : modelRow 379 = row 379 := model_count_379.trans row_count_379.symm
def row_380 : RowCertificate 380 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 380 1) := by change ¬ Nat.Prime 578361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 380 2) := by change Nat.Prime 577601; exact ProjectEulerSolutions.P58PrimeNodes.p_577601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 380 3) := by change ¬ Nat.Prime 576841; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 1453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_380 : row 380 = 1 := (row_380.checked).trans rfl
theorem model_count_380 : modelRow 380 = 1 := by decide
theorem modelrow_agrees_380 : modelRow 380 = row 380 := model_count_380.trans row_count_380.symm
def row_381 : RowCertificate 381 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 381 1) := by change Nat.Prime 581407; exact ProjectEulerSolutions.P58PrimeNodes.p_581407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 381 2) := by change ¬ Nat.Prime 580645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 116129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 381 3) := by change Nat.Prime 579883; exact ProjectEulerSolutions.P58PrimeNodes.p_579883.prime
    exact decide_eq_true prime
theorem row_count_381 : row 381 = 2 := (row_381.checked).trans rfl
theorem model_count_381 : modelRow 381 = 2 := by decide
theorem modelrow_agrees_381 : modelRow 381 = row 381 := model_count_381.trans row_count_381.symm
def row_382 : RowCertificate 382 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 382 1) := by change ¬ Nat.Prime 584461; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 1429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 382 2) := by change Nat.Prime 583697; exact ProjectEulerSolutions.P58PrimeNodes.p_583697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 382 3) := by change ¬ Nat.Prime 582933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 194311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_382 : row 382 = 1 := (row_382.checked).trans rfl
theorem model_count_382 : modelRow 382 = 1 := by decide
theorem modelrow_agrees_382 : modelRow 382 = row 382 := model_count_382.trans row_count_382.symm
def row_383 : RowCertificate 383 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 383 1) := by change ¬ Nat.Prime 587523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 195841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 383 2) := by change ¬ Nat.Prime 586757; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 383 3) := by change ¬ Nat.Prime 585991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 83713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_383 : row 383 = 0 := (row_383.checked).trans rfl
theorem model_count_383 : modelRow 383 = 0 := by decide
theorem modelrow_agrees_383 : modelRow 383 = row 383 := model_count_383.trans row_count_383.symm
def row_384 : RowCertificate 384 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 384 1) := by change Nat.Prime 590593; exact ProjectEulerSolutions.P58PrimeNodes.p_590593.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 384 2) := by change ¬ Nat.Prime 589825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 117965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 384 3) := by change ¬ Nat.Prime 589057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_384 : row 384 = 1 := (row_384.checked).trans rfl
theorem model_count_384 : modelRow 384 = 1 := by decide
theorem modelrow_agrees_384 : modelRow 384 = row 384 := model_count_384.trans row_count_384.symm
private def atomic_257 : CountSegment 1 10 13120 256 257 177 177 := CountSegment.single (first := 256) (initial := 177) row_count_257 (by decide)
private theorem atomic_257_agreement : RowAgreement 256 257 := RowAgreement.single (first := 256) modelrow_agrees_257
private def atomic_258 : CountSegment 1 10 13120 257 258 177 177 := CountSegment.single (first := 257) (initial := 177) row_count_258 (by decide)
private theorem atomic_258_agreement : RowAgreement 257 258 := RowAgreement.single (first := 257) modelrow_agrees_258
private def atomic_259 : CountSegment 1 10 13120 258 259 177 178 := CountSegment.single (first := 258) (initial := 177) row_count_259 (by decide)
private theorem atomic_259_agreement : RowAgreement 258 259 := RowAgreement.single (first := 258) modelrow_agrees_259
private def atomic_260 : CountSegment 1 10 13120 259 260 178 178 := CountSegment.single (first := 259) (initial := 178) row_count_260 (by decide)
private theorem atomic_260_agreement : RowAgreement 259 260 := RowAgreement.single (first := 259) modelrow_agrees_260
private def atomic_261 : CountSegment 1 10 13120 260 261 178 178 := CountSegment.single (first := 260) (initial := 178) row_count_261 (by decide)
private theorem atomic_261_agreement : RowAgreement 260 261 := RowAgreement.single (first := 260) modelrow_agrees_261
private def atomic_262 : CountSegment 1 10 13120 261 262 178 178 := CountSegment.single (first := 261) (initial := 178) row_count_262 (by decide)
private theorem atomic_262_agreement : RowAgreement 261 262 := RowAgreement.single (first := 261) modelrow_agrees_262
private def atomic_263 : CountSegment 1 10 13120 262 263 178 179 := CountSegment.single (first := 262) (initial := 178) row_count_263 (by decide)
private theorem atomic_263_agreement : RowAgreement 262 263 := RowAgreement.single (first := 262) modelrow_agrees_263
private def atomic_264 : CountSegment 1 10 13120 263 264 179 179 := CountSegment.single (first := 263) (initial := 179) row_count_264 (by decide)
private theorem atomic_264_agreement : RowAgreement 263 264 := RowAgreement.single (first := 263) modelrow_agrees_264
private def atomic_265 : CountSegment 1 10 13120 264 265 179 180 := CountSegment.single (first := 264) (initial := 179) row_count_265 (by decide)
private theorem atomic_265_agreement : RowAgreement 264 265 := RowAgreement.single (first := 264) modelrow_agrees_265
private def atomic_266 : CountSegment 1 10 13120 265 266 180 181 := CountSegment.single (first := 265) (initial := 180) row_count_266 (by decide)
private theorem atomic_266_agreement : RowAgreement 265 266 := RowAgreement.single (first := 265) modelrow_agrees_266
private def atomic_267 : CountSegment 1 10 13120 266 267 181 182 := CountSegment.single (first := 266) (initial := 181) row_count_267 (by decide)
private theorem atomic_267_agreement : RowAgreement 266 267 := RowAgreement.single (first := 266) modelrow_agrees_267
private def atomic_268 : CountSegment 1 10 13120 267 268 182 183 := CountSegment.single (first := 267) (initial := 182) row_count_268 (by decide)
private theorem atomic_268_agreement : RowAgreement 267 268 := RowAgreement.single (first := 267) modelrow_agrees_268
private def atomic_269 : CountSegment 1 10 13120 268 269 183 184 := CountSegment.single (first := 268) (initial := 183) row_count_269 (by decide)
private theorem atomic_269_agreement : RowAgreement 268 269 := RowAgreement.single (first := 268) modelrow_agrees_269
private def atomic_270 : CountSegment 1 10 13120 269 270 184 185 := CountSegment.single (first := 269) (initial := 184) row_count_270 (by decide)
private theorem atomic_270_agreement : RowAgreement 269 270 := RowAgreement.single (first := 269) modelrow_agrees_270
private def atomic_271 : CountSegment 1 10 13120 270 271 185 185 := CountSegment.single (first := 270) (initial := 185) row_count_271 (by decide)
private theorem atomic_271_agreement : RowAgreement 270 271 := RowAgreement.single (first := 270) modelrow_agrees_271
private def atomic_272 : CountSegment 1 10 13120 271 272 185 186 := CountSegment.single (first := 271) (initial := 185) row_count_272 (by decide)
private theorem atomic_272_agreement : RowAgreement 271 272 := RowAgreement.single (first := 271) modelrow_agrees_272
private def atomic_273 : CountSegment 1 10 13120 272 273 186 186 := CountSegment.single (first := 272) (initial := 186) row_count_273 (by decide)
private theorem atomic_273_agreement : RowAgreement 272 273 := RowAgreement.single (first := 272) modelrow_agrees_273
private def atomic_274 : CountSegment 1 10 13120 273 274 186 186 := CountSegment.single (first := 273) (initial := 186) row_count_274 (by decide)
private theorem atomic_274_agreement : RowAgreement 273 274 := RowAgreement.single (first := 273) modelrow_agrees_274
private def atomic_275 : CountSegment 1 10 13120 274 275 186 186 := CountSegment.single (first := 274) (initial := 186) row_count_275 (by decide)
private theorem atomic_275_agreement : RowAgreement 274 275 := RowAgreement.single (first := 274) modelrow_agrees_275
private def atomic_276 : CountSegment 1 10 13120 275 276 186 187 := CountSegment.single (first := 275) (initial := 186) row_count_276 (by decide)
private theorem atomic_276_agreement : RowAgreement 275 276 := RowAgreement.single (first := 275) modelrow_agrees_276
private def atomic_277 : CountSegment 1 10 13120 276 277 187 188 := CountSegment.single (first := 276) (initial := 187) row_count_277 (by decide)
private theorem atomic_277_agreement : RowAgreement 276 277 := RowAgreement.single (first := 276) modelrow_agrees_277
private def atomic_278 : CountSegment 1 10 13120 277 278 188 189 := CountSegment.single (first := 277) (initial := 188) row_count_278 (by decide)
private theorem atomic_278_agreement : RowAgreement 277 278 := RowAgreement.single (first := 277) modelrow_agrees_278
private def atomic_279 : CountSegment 1 10 13120 278 279 189 189 := CountSegment.single (first := 278) (initial := 189) row_count_279 (by decide)
private theorem atomic_279_agreement : RowAgreement 278 279 := RowAgreement.single (first := 278) modelrow_agrees_279
private def atomic_280 : CountSegment 1 10 13120 279 280 189 190 := CountSegment.single (first := 279) (initial := 189) row_count_280 (by decide)
private theorem atomic_280_agreement : RowAgreement 279 280 := RowAgreement.single (first := 279) modelrow_agrees_280
private def atomic_281 : CountSegment 1 10 13120 280 281 190 190 := CountSegment.single (first := 280) (initial := 190) row_count_281 (by decide)
private theorem atomic_281_agreement : RowAgreement 280 281 := RowAgreement.single (first := 280) modelrow_agrees_281
private def atomic_282 : CountSegment 1 10 13120 281 282 190 190 := CountSegment.single (first := 281) (initial := 190) row_count_282 (by decide)
private theorem atomic_282_agreement : RowAgreement 281 282 := RowAgreement.single (first := 281) modelrow_agrees_282
private def atomic_283 : CountSegment 1 10 13120 282 283 190 191 := CountSegment.single (first := 282) (initial := 190) row_count_283 (by decide)
private theorem atomic_283_agreement : RowAgreement 282 283 := RowAgreement.single (first := 282) modelrow_agrees_283
private def atomic_284 : CountSegment 1 10 13120 283 284 191 192 := CountSegment.single (first := 283) (initial := 191) row_count_284 (by decide)
private theorem atomic_284_agreement : RowAgreement 283 284 := RowAgreement.single (first := 283) modelrow_agrees_284
private def atomic_285 : CountSegment 1 10 13120 284 285 192 193 := CountSegment.single (first := 284) (initial := 192) row_count_285 (by decide)
private theorem atomic_285_agreement : RowAgreement 284 285 := RowAgreement.single (first := 284) modelrow_agrees_285
private def atomic_286 : CountSegment 1 10 13120 285 286 193 194 := CountSegment.single (first := 285) (initial := 193) row_count_286 (by decide)
private theorem atomic_286_agreement : RowAgreement 285 286 := RowAgreement.single (first := 285) modelrow_agrees_286
private def atomic_287 : CountSegment 1 10 13120 286 287 194 194 := CountSegment.single (first := 286) (initial := 194) row_count_287 (by decide)
private theorem atomic_287_agreement : RowAgreement 286 287 := RowAgreement.single (first := 286) modelrow_agrees_287
private def atomic_288 : CountSegment 1 10 13120 287 288 194 195 := CountSegment.single (first := 287) (initial := 194) row_count_288 (by decide)
private theorem atomic_288_agreement : RowAgreement 287 288 := RowAgreement.single (first := 287) modelrow_agrees_288
private def atomic_289 : CountSegment 1 10 13120 288 289 195 195 := CountSegment.single (first := 288) (initial := 195) row_count_289 (by decide)
private theorem atomic_289_agreement : RowAgreement 288 289 := RowAgreement.single (first := 288) modelrow_agrees_289
private def atomic_290 : CountSegment 1 10 13120 289 290 195 196 := CountSegment.single (first := 289) (initial := 195) row_count_290 (by decide)
private theorem atomic_290_agreement : RowAgreement 289 290 := RowAgreement.single (first := 289) modelrow_agrees_290
private def atomic_291 : CountSegment 1 10 13120 290 291 196 197 := CountSegment.single (first := 290) (initial := 196) row_count_291 (by decide)
private theorem atomic_291_agreement : RowAgreement 290 291 := RowAgreement.single (first := 290) modelrow_agrees_291
private def atomic_292 : CountSegment 1 10 13120 291 292 197 199 := CountSegment.single (first := 291) (initial := 197) row_count_292 (by decide)
private theorem atomic_292_agreement : RowAgreement 291 292 := RowAgreement.single (first := 291) modelrow_agrees_292
private def atomic_293 : CountSegment 1 10 13120 292 293 199 199 := CountSegment.single (first := 292) (initial := 199) row_count_293 (by decide)
private theorem atomic_293_agreement : RowAgreement 292 293 := RowAgreement.single (first := 292) modelrow_agrees_293
private def atomic_294 : CountSegment 1 10 13120 293 294 199 199 := CountSegment.single (first := 293) (initial := 199) row_count_294 (by decide)
private theorem atomic_294_agreement : RowAgreement 293 294 := RowAgreement.single (first := 293) modelrow_agrees_294
private def atomic_295 : CountSegment 1 10 13120 294 295 199 199 := CountSegment.single (first := 294) (initial := 199) row_count_295 (by decide)
private theorem atomic_295_agreement : RowAgreement 294 295 := RowAgreement.single (first := 294) modelrow_agrees_295
private def atomic_296 : CountSegment 1 10 13120 295 296 199 199 := CountSegment.single (first := 295) (initial := 199) row_count_296 (by decide)
private theorem atomic_296_agreement : RowAgreement 295 296 := RowAgreement.single (first := 295) modelrow_agrees_296
private def atomic_297 : CountSegment 1 10 13120 296 297 199 200 := CountSegment.single (first := 296) (initial := 199) row_count_297 (by decide)
private theorem atomic_297_agreement : RowAgreement 296 297 := RowAgreement.single (first := 296) modelrow_agrees_297
private def atomic_298 : CountSegment 1 10 13120 297 298 200 200 := CountSegment.single (first := 297) (initial := 200) row_count_298 (by decide)
private theorem atomic_298_agreement : RowAgreement 297 298 := RowAgreement.single (first := 297) modelrow_agrees_298
private def atomic_299 : CountSegment 1 10 13120 298 299 200 200 := CountSegment.single (first := 298) (initial := 200) row_count_299 (by decide)
private theorem atomic_299_agreement : RowAgreement 298 299 := RowAgreement.single (first := 298) modelrow_agrees_299
private def atomic_300 : CountSegment 1 10 13120 299 300 200 200 := CountSegment.single (first := 299) (initial := 200) row_count_300 (by decide)
private theorem atomic_300_agreement : RowAgreement 299 300 := RowAgreement.single (first := 299) modelrow_agrees_300
private def atomic_301 : CountSegment 1 10 13120 300 301 200 200 := CountSegment.single (first := 300) (initial := 200) row_count_301 (by decide)
private theorem atomic_301_agreement : RowAgreement 300 301 := RowAgreement.single (first := 300) modelrow_agrees_301
private def atomic_302 : CountSegment 1 10 13120 301 302 200 201 := CountSegment.single (first := 301) (initial := 200) row_count_302 (by decide)
private theorem atomic_302_agreement : RowAgreement 301 302 := RowAgreement.single (first := 301) modelrow_agrees_302
private def atomic_303 : CountSegment 1 10 13120 302 303 201 202 := CountSegment.single (first := 302) (initial := 201) row_count_303 (by decide)
private theorem atomic_303_agreement : RowAgreement 302 303 := RowAgreement.single (first := 302) modelrow_agrees_303
private def atomic_304 : CountSegment 1 10 13120 303 304 202 202 := CountSegment.single (first := 303) (initial := 202) row_count_304 (by decide)
private theorem atomic_304_agreement : RowAgreement 303 304 := RowAgreement.single (first := 303) modelrow_agrees_304
private def atomic_305 : CountSegment 1 10 13120 304 305 202 203 := CountSegment.single (first := 304) (initial := 202) row_count_305 (by decide)
private theorem atomic_305_agreement : RowAgreement 304 305 := RowAgreement.single (first := 304) modelrow_agrees_305
private def atomic_306 : CountSegment 1 10 13120 305 306 203 204 := CountSegment.single (first := 305) (initial := 203) row_count_306 (by decide)
private theorem atomic_306_agreement : RowAgreement 305 306 := RowAgreement.single (first := 305) modelrow_agrees_306
private def atomic_307 : CountSegment 1 10 13120 306 307 204 204 := CountSegment.single (first := 306) (initial := 204) row_count_307 (by decide)
private theorem atomic_307_agreement : RowAgreement 306 307 := RowAgreement.single (first := 306) modelrow_agrees_307
private def atomic_308 : CountSegment 1 10 13120 307 308 204 204 := CountSegment.single (first := 307) (initial := 204) row_count_308 (by decide)
private theorem atomic_308_agreement : RowAgreement 307 308 := RowAgreement.single (first := 307) modelrow_agrees_308
private def atomic_309 : CountSegment 1 10 13120 308 309 204 204 := CountSegment.single (first := 308) (initial := 204) row_count_309 (by decide)
private theorem atomic_309_agreement : RowAgreement 308 309 := RowAgreement.single (first := 308) modelrow_agrees_309
private def atomic_310 : CountSegment 1 10 13120 309 310 204 204 := CountSegment.single (first := 309) (initial := 204) row_count_310 (by decide)
private theorem atomic_310_agreement : RowAgreement 309 310 := RowAgreement.single (first := 309) modelrow_agrees_310
private def atomic_311 : CountSegment 1 10 13120 310 311 204 205 := CountSegment.single (first := 310) (initial := 204) row_count_311 (by decide)
private theorem atomic_311_agreement : RowAgreement 310 311 := RowAgreement.single (first := 310) modelrow_agrees_311
private def atomic_312 : CountSegment 1 10 13120 311 312 205 206 := CountSegment.single (first := 311) (initial := 205) row_count_312 (by decide)
private theorem atomic_312_agreement : RowAgreement 311 312 := RowAgreement.single (first := 311) modelrow_agrees_312
private def atomic_313 : CountSegment 1 10 13120 312 313 206 207 := CountSegment.single (first := 312) (initial := 206) row_count_313 (by decide)
private theorem atomic_313_agreement : RowAgreement 312 313 := RowAgreement.single (first := 312) modelrow_agrees_313
private def atomic_314 : CountSegment 1 10 13120 313 314 207 207 := CountSegment.single (first := 313) (initial := 207) row_count_314 (by decide)
private theorem atomic_314_agreement : RowAgreement 313 314 := RowAgreement.single (first := 313) modelrow_agrees_314
private def atomic_315 : CountSegment 1 10 13120 314 315 207 207 := CountSegment.single (first := 314) (initial := 207) row_count_315 (by decide)
private theorem atomic_315_agreement : RowAgreement 314 315 := RowAgreement.single (first := 314) modelrow_agrees_315
private def atomic_316 : CountSegment 1 10 13120 315 316 207 207 := CountSegment.single (first := 315) (initial := 207) row_count_316 (by decide)
private theorem atomic_316_agreement : RowAgreement 315 316 := RowAgreement.single (first := 315) modelrow_agrees_316
private def atomic_317 : CountSegment 1 10 13120 316 317 207 208 := CountSegment.single (first := 316) (initial := 207) row_count_317 (by decide)
private theorem atomic_317_agreement : RowAgreement 316 317 := RowAgreement.single (first := 316) modelrow_agrees_317
private def atomic_318 : CountSegment 1 10 13120 317 318 208 210 := CountSegment.single (first := 317) (initial := 208) row_count_318 (by decide)
private theorem atomic_318_agreement : RowAgreement 317 318 := RowAgreement.single (first := 317) modelrow_agrees_318
private def atomic_319 : CountSegment 1 10 13120 318 319 210 210 := CountSegment.single (first := 318) (initial := 210) row_count_319 (by decide)
private theorem atomic_319_agreement : RowAgreement 318 319 := RowAgreement.single (first := 318) modelrow_agrees_319
private def atomic_320 : CountSegment 1 10 13120 319 320 210 210 := CountSegment.single (first := 319) (initial := 210) row_count_320 (by decide)
private theorem atomic_320_agreement : RowAgreement 319 320 := RowAgreement.single (first := 319) modelrow_agrees_320
private def atomic_321 : CountSegment 1 10 13120 320 321 210 211 := CountSegment.single (first := 320) (initial := 210) row_count_321 (by decide)
private theorem atomic_321_agreement : RowAgreement 320 321 := RowAgreement.single (first := 320) modelrow_agrees_321
private def atomic_322 : CountSegment 1 10 13120 321 322 211 213 := CountSegment.single (first := 321) (initial := 211) row_count_322 (by decide)
private theorem atomic_322_agreement : RowAgreement 321 322 := RowAgreement.single (first := 321) modelrow_agrees_322
private def atomic_323 : CountSegment 1 10 13120 322 323 213 214 := CountSegment.single (first := 322) (initial := 213) row_count_323 (by decide)
private theorem atomic_323_agreement : RowAgreement 322 323 := RowAgreement.single (first := 322) modelrow_agrees_323
private def atomic_324 : CountSegment 1 10 13120 323 324 214 214 := CountSegment.single (first := 323) (initial := 214) row_count_324 (by decide)
private theorem atomic_324_agreement : RowAgreement 323 324 := RowAgreement.single (first := 323) modelrow_agrees_324
private def atomic_325 : CountSegment 1 10 13120 324 325 214 214 := CountSegment.single (first := 324) (initial := 214) row_count_325 (by decide)
private theorem atomic_325_agreement : RowAgreement 324 325 := RowAgreement.single (first := 324) modelrow_agrees_325
private def atomic_326 : CountSegment 1 10 13120 325 326 214 214 := CountSegment.single (first := 325) (initial := 214) row_count_326 (by decide)
private theorem atomic_326_agreement : RowAgreement 325 326 := RowAgreement.single (first := 325) modelrow_agrees_326
private def atomic_327 : CountSegment 1 10 13120 326 327 214 215 := CountSegment.single (first := 326) (initial := 214) row_count_327 (by decide)
private theorem atomic_327_agreement : RowAgreement 326 327 := RowAgreement.single (first := 326) modelrow_agrees_327
private def atomic_328 : CountSegment 1 10 13120 327 328 215 215 := CountSegment.single (first := 327) (initial := 215) row_count_328 (by decide)
private theorem atomic_328_agreement : RowAgreement 327 328 := RowAgreement.single (first := 327) modelrow_agrees_328
private def atomic_329 : CountSegment 1 10 13120 328 329 215 215 := CountSegment.single (first := 328) (initial := 215) row_count_329 (by decide)
private theorem atomic_329_agreement : RowAgreement 328 329 := RowAgreement.single (first := 328) modelrow_agrees_329
private def atomic_330 : CountSegment 1 10 13120 329 330 215 215 := CountSegment.single (first := 329) (initial := 215) row_count_330 (by decide)
private theorem atomic_330_agreement : RowAgreement 329 330 := RowAgreement.single (first := 329) modelrow_agrees_330
private def atomic_331 : CountSegment 1 10 13120 330 331 215 215 := CountSegment.single (first := 330) (initial := 215) row_count_331 (by decide)
private theorem atomic_331_agreement : RowAgreement 330 331 := RowAgreement.single (first := 330) modelrow_agrees_331
private def atomic_332 : CountSegment 1 10 13120 331 332 215 215 := CountSegment.single (first := 331) (initial := 215) row_count_332 (by decide)
private theorem atomic_332_agreement : RowAgreement 331 332 := RowAgreement.single (first := 331) modelrow_agrees_332
private def atomic_333 : CountSegment 1 10 13120 332 333 215 215 := CountSegment.single (first := 332) (initial := 215) row_count_333 (by decide)
private theorem atomic_333_agreement : RowAgreement 332 333 := RowAgreement.single (first := 332) modelrow_agrees_333
private def atomic_334 : CountSegment 1 10 13120 333 334 215 216 := CountSegment.single (first := 333) (initial := 215) row_count_334 (by decide)
private theorem atomic_334_agreement : RowAgreement 333 334 := RowAgreement.single (first := 333) modelrow_agrees_334
private def atomic_335 : CountSegment 1 10 13120 334 335 216 216 := CountSegment.single (first := 334) (initial := 216) row_count_335 (by decide)
private theorem atomic_335_agreement : RowAgreement 334 335 := RowAgreement.single (first := 334) modelrow_agrees_335
private def atomic_336 : CountSegment 1 10 13120 335 336 216 217 := CountSegment.single (first := 335) (initial := 216) row_count_336 (by decide)
private theorem atomic_336_agreement : RowAgreement 335 336 := RowAgreement.single (first := 335) modelrow_agrees_336
private def atomic_337 : CountSegment 1 10 13120 336 337 217 218 := CountSegment.single (first := 336) (initial := 217) row_count_337 (by decide)
private theorem atomic_337_agreement : RowAgreement 336 337 := RowAgreement.single (first := 336) modelrow_agrees_337
private def atomic_338 : CountSegment 1 10 13120 337 338 218 218 := CountSegment.single (first := 337) (initial := 218) row_count_338 (by decide)
private theorem atomic_338_agreement : RowAgreement 337 338 := RowAgreement.single (first := 337) modelrow_agrees_338
private def atomic_339 : CountSegment 1 10 13120 338 339 218 219 := CountSegment.single (first := 338) (initial := 218) row_count_339 (by decide)
private theorem atomic_339_agreement : RowAgreement 338 339 := RowAgreement.single (first := 338) modelrow_agrees_339
private def atomic_340 : CountSegment 1 10 13120 339 340 219 220 := CountSegment.single (first := 339) (initial := 219) row_count_340 (by decide)
private theorem atomic_340_agreement : RowAgreement 339 340 := RowAgreement.single (first := 339) modelrow_agrees_340
private def atomic_341 : CountSegment 1 10 13120 340 341 220 220 := CountSegment.single (first := 340) (initial := 220) row_count_341 (by decide)
private theorem atomic_341_agreement : RowAgreement 340 341 := RowAgreement.single (first := 340) modelrow_agrees_341
private def atomic_342 : CountSegment 1 10 13120 341 342 220 220 := CountSegment.single (first := 341) (initial := 220) row_count_342 (by decide)
private theorem atomic_342_agreement : RowAgreement 341 342 := RowAgreement.single (first := 341) modelrow_agrees_342
private def atomic_343 : CountSegment 1 10 13120 342 343 220 222 := CountSegment.single (first := 342) (initial := 220) row_count_343 (by decide)
private theorem atomic_343_agreement : RowAgreement 342 343 := RowAgreement.single (first := 342) modelrow_agrees_343
private def atomic_344 : CountSegment 1 10 13120 343 344 222 222 := CountSegment.single (first := 343) (initial := 222) row_count_344 (by decide)
private theorem atomic_344_agreement : RowAgreement 343 344 := RowAgreement.single (first := 343) modelrow_agrees_344
private def atomic_345 : CountSegment 1 10 13120 344 345 222 223 := CountSegment.single (first := 344) (initial := 222) row_count_345 (by decide)
private theorem atomic_345_agreement : RowAgreement 344 345 := RowAgreement.single (first := 344) modelrow_agrees_345
private def atomic_346 : CountSegment 1 10 13120 345 346 223 223 := CountSegment.single (first := 345) (initial := 223) row_count_346 (by decide)
private theorem atomic_346_agreement : RowAgreement 345 346 := RowAgreement.single (first := 345) modelrow_agrees_346
private def atomic_347 : CountSegment 1 10 13120 346 347 223 223 := CountSegment.single (first := 346) (initial := 223) row_count_347 (by decide)
private theorem atomic_347_agreement : RowAgreement 346 347 := RowAgreement.single (first := 346) modelrow_agrees_347
private def atomic_348 : CountSegment 1 10 13120 347 348 223 225 := CountSegment.single (first := 347) (initial := 223) row_count_348 (by decide)
private theorem atomic_348_agreement : RowAgreement 347 348 := RowAgreement.single (first := 347) modelrow_agrees_348
private def atomic_349 : CountSegment 1 10 13120 348 349 225 225 := CountSegment.single (first := 348) (initial := 225) row_count_349 (by decide)
private theorem atomic_349_agreement : RowAgreement 348 349 := RowAgreement.single (first := 348) modelrow_agrees_349
private def atomic_350 : CountSegment 1 10 13120 349 350 225 226 := CountSegment.single (first := 349) (initial := 225) row_count_350 (by decide)
private theorem atomic_350_agreement : RowAgreement 349 350 := RowAgreement.single (first := 349) modelrow_agrees_350
private def atomic_351 : CountSegment 1 10 13120 350 351 226 227 := CountSegment.single (first := 350) (initial := 226) row_count_351 (by decide)
private theorem atomic_351_agreement : RowAgreement 350 351 := RowAgreement.single (first := 350) modelrow_agrees_351
private def atomic_352 : CountSegment 1 10 13120 351 352 227 228 := CountSegment.single (first := 351) (initial := 227) row_count_352 (by decide)
private theorem atomic_352_agreement : RowAgreement 351 352 := RowAgreement.single (first := 351) modelrow_agrees_352
private def atomic_353 : CountSegment 1 10 13120 352 353 228 228 := CountSegment.single (first := 352) (initial := 228) row_count_353 (by decide)
private theorem atomic_353_agreement : RowAgreement 352 353 := RowAgreement.single (first := 352) modelrow_agrees_353
private def atomic_354 : CountSegment 1 10 13120 353 354 228 228 := CountSegment.single (first := 353) (initial := 228) row_count_354 (by decide)
private theorem atomic_354_agreement : RowAgreement 353 354 := RowAgreement.single (first := 353) modelrow_agrees_354
private def atomic_355 : CountSegment 1 10 13120 354 355 228 228 := CountSegment.single (first := 354) (initial := 228) row_count_355 (by decide)
private theorem atomic_355_agreement : RowAgreement 354 355 := RowAgreement.single (first := 354) modelrow_agrees_355
private def atomic_356 : CountSegment 1 10 13120 355 356 228 228 := CountSegment.single (first := 355) (initial := 228) row_count_356 (by decide)
private theorem atomic_356_agreement : RowAgreement 355 356 := RowAgreement.single (first := 355) modelrow_agrees_356
private def atomic_357 : CountSegment 1 10 13120 356 357 228 229 := CountSegment.single (first := 356) (initial := 228) row_count_357 (by decide)
private theorem atomic_357_agreement : RowAgreement 356 357 := RowAgreement.single (first := 356) modelrow_agrees_357
private def atomic_358 : CountSegment 1 10 13120 357 358 229 230 := CountSegment.single (first := 357) (initial := 229) row_count_358 (by decide)
private theorem atomic_358_agreement : RowAgreement 357 358 := RowAgreement.single (first := 357) modelrow_agrees_358
private def atomic_359 : CountSegment 1 10 13120 358 359 230 230 := CountSegment.single (first := 358) (initial := 230) row_count_359 (by decide)
private theorem atomic_359_agreement : RowAgreement 358 359 := RowAgreement.single (first := 358) modelrow_agrees_359
private def atomic_360 : CountSegment 1 10 13120 359 360 230 231 := CountSegment.single (first := 359) (initial := 230) row_count_360 (by decide)
private theorem atomic_360_agreement : RowAgreement 359 360 := RowAgreement.single (first := 359) modelrow_agrees_360
private def atomic_361 : CountSegment 1 10 13120 360 361 231 231 := CountSegment.single (first := 360) (initial := 231) row_count_361 (by decide)
private theorem atomic_361_agreement : RowAgreement 360 361 := RowAgreement.single (first := 360) modelrow_agrees_361
private def atomic_362 : CountSegment 1 10 13120 361 362 231 231 := CountSegment.single (first := 361) (initial := 231) row_count_362 (by decide)
private theorem atomic_362_agreement : RowAgreement 361 362 := RowAgreement.single (first := 361) modelrow_agrees_362
private def atomic_363 : CountSegment 1 10 13120 362 363 231 232 := CountSegment.single (first := 362) (initial := 231) row_count_363 (by decide)
private theorem atomic_363_agreement : RowAgreement 362 363 := RowAgreement.single (first := 362) modelrow_agrees_363
private def atomic_364 : CountSegment 1 10 13120 363 364 232 233 := CountSegment.single (first := 363) (initial := 232) row_count_364 (by decide)
private theorem atomic_364_agreement : RowAgreement 363 364 := RowAgreement.single (first := 363) modelrow_agrees_364
private def atomic_365 : CountSegment 1 10 13120 364 365 233 233 := CountSegment.single (first := 364) (initial := 233) row_count_365 (by decide)
private theorem atomic_365_agreement : RowAgreement 364 365 := RowAgreement.single (first := 364) modelrow_agrees_365
private def atomic_366 : CountSegment 1 10 13120 365 366 233 233 := CountSegment.single (first := 365) (initial := 233) row_count_366 (by decide)
private theorem atomic_366_agreement : RowAgreement 365 366 := RowAgreement.single (first := 365) modelrow_agrees_366
private def atomic_367 : CountSegment 1 10 13120 366 367 233 233 := CountSegment.single (first := 366) (initial := 233) row_count_367 (by decide)
private theorem atomic_367_agreement : RowAgreement 366 367 := RowAgreement.single (first := 366) modelrow_agrees_367
private def atomic_368 : CountSegment 1 10 13120 367 368 233 234 := CountSegment.single (first := 367) (initial := 233) row_count_368 (by decide)
private theorem atomic_368_agreement : RowAgreement 367 368 := RowAgreement.single (first := 367) modelrow_agrees_368
private def atomic_369 : CountSegment 1 10 13120 368 369 234 234 := CountSegment.single (first := 368) (initial := 234) row_count_369 (by decide)
private theorem atomic_369_agreement : RowAgreement 368 369 := RowAgreement.single (first := 368) modelrow_agrees_369
private def atomic_370 : CountSegment 1 10 13120 369 370 234 235 := CountSegment.single (first := 369) (initial := 234) row_count_370 (by decide)
private theorem atomic_370_agreement : RowAgreement 369 370 := RowAgreement.single (first := 369) modelrow_agrees_370
private def atomic_371 : CountSegment 1 10 13120 370 371 235 235 := CountSegment.single (first := 370) (initial := 235) row_count_371 (by decide)
private theorem atomic_371_agreement : RowAgreement 370 371 := RowAgreement.single (first := 370) modelrow_agrees_371
private def atomic_372 : CountSegment 1 10 13120 371 372 235 236 := CountSegment.single (first := 371) (initial := 235) row_count_372 (by decide)
private theorem atomic_372_agreement : RowAgreement 371 372 := RowAgreement.single (first := 371) modelrow_agrees_372
private def atomic_373 : CountSegment 1 10 13120 372 373 236 236 := CountSegment.single (first := 372) (initial := 236) row_count_373 (by decide)
private theorem atomic_373_agreement : RowAgreement 372 373 := RowAgreement.single (first := 372) modelrow_agrees_373
private def atomic_374 : CountSegment 1 10 13120 373 374 236 237 := CountSegment.single (first := 373) (initial := 236) row_count_374 (by decide)
private theorem atomic_374_agreement : RowAgreement 373 374 := RowAgreement.single (first := 373) modelrow_agrees_374
private def atomic_375 : CountSegment 1 10 13120 374 375 237 238 := CountSegment.single (first := 374) (initial := 237) row_count_375 (by decide)
private theorem atomic_375_agreement : RowAgreement 374 375 := RowAgreement.single (first := 374) modelrow_agrees_375
private def atomic_376 : CountSegment 1 10 13120 375 376 238 238 := CountSegment.single (first := 375) (initial := 238) row_count_376 (by decide)
private theorem atomic_376_agreement : RowAgreement 375 376 := RowAgreement.single (first := 375) modelrow_agrees_376
private def atomic_377 : CountSegment 1 10 13120 376 377 238 238 := CountSegment.single (first := 376) (initial := 238) row_count_377 (by decide)
private theorem atomic_377_agreement : RowAgreement 376 377 := RowAgreement.single (first := 376) modelrow_agrees_377
private def atomic_378 : CountSegment 1 10 13120 377 378 238 239 := CountSegment.single (first := 377) (initial := 238) row_count_378 (by decide)
private theorem atomic_378_agreement : RowAgreement 377 378 := RowAgreement.single (first := 377) modelrow_agrees_378
private def atomic_379 : CountSegment 1 10 13120 378 379 239 239 := CountSegment.single (first := 378) (initial := 239) row_count_379 (by decide)
private theorem atomic_379_agreement : RowAgreement 378 379 := RowAgreement.single (first := 378) modelrow_agrees_379
private def atomic_380 : CountSegment 1 10 13120 379 380 239 240 := CountSegment.single (first := 379) (initial := 239) row_count_380 (by decide)
private theorem atomic_380_agreement : RowAgreement 379 380 := RowAgreement.single (first := 379) modelrow_agrees_380
private def atomic_381 : CountSegment 1 10 13120 380 381 240 242 := CountSegment.single (first := 380) (initial := 240) row_count_381 (by decide)
private theorem atomic_381_agreement : RowAgreement 380 381 := RowAgreement.single (first := 380) modelrow_agrees_381
private def atomic_382 : CountSegment 1 10 13120 381 382 242 243 := CountSegment.single (first := 381) (initial := 242) row_count_382 (by decide)
private theorem atomic_382_agreement : RowAgreement 381 382 := RowAgreement.single (first := 381) modelrow_agrees_382
private def atomic_383 : CountSegment 1 10 13120 382 383 243 243 := CountSegment.single (first := 382) (initial := 243) row_count_383 (by decide)
private theorem atomic_383_agreement : RowAgreement 382 383 := RowAgreement.single (first := 382) modelrow_agrees_383
private def atomic_384 : CountSegment 1 10 13120 383 384 243 244 := CountSegment.single (first := 383) (initial := 243) row_count_384 (by decide)
private theorem atomic_384_agreement : RowAgreement 383 384 := RowAgreement.single (first := 383) modelrow_agrees_384
private def within_002_000 : CountSegment 1 10 13120 256 258 177 177 := CountSegment.join atomic_257 atomic_258
private theorem within_002_000_agreement : RowAgreement 256 258 := RowAgreement.join atomic_257_agreement atomic_258_agreement
private def within_002_001 : CountSegment 1 10 13120 258 260 177 178 := CountSegment.join atomic_259 atomic_260
private theorem within_002_001_agreement : RowAgreement 258 260 := RowAgreement.join atomic_259_agreement atomic_260_agreement
private def within_002_002 : CountSegment 1 10 13120 256 260 177 178 := CountSegment.join within_002_000 within_002_001
private theorem within_002_002_agreement : RowAgreement 256 260 := RowAgreement.join within_002_000_agreement within_002_001_agreement
private def within_002_003 : CountSegment 1 10 13120 260 262 178 178 := CountSegment.join atomic_261 atomic_262
private theorem within_002_003_agreement : RowAgreement 260 262 := RowAgreement.join atomic_261_agreement atomic_262_agreement
private def within_002_004 : CountSegment 1 10 13120 262 264 178 179 := CountSegment.join atomic_263 atomic_264
private theorem within_002_004_agreement : RowAgreement 262 264 := RowAgreement.join atomic_263_agreement atomic_264_agreement
private def within_002_005 : CountSegment 1 10 13120 260 264 178 179 := CountSegment.join within_002_003 within_002_004
private theorem within_002_005_agreement : RowAgreement 260 264 := RowAgreement.join within_002_003_agreement within_002_004_agreement
private def within_002_006 : CountSegment 1 10 13120 256 264 177 179 := CountSegment.join within_002_002 within_002_005
private theorem within_002_006_agreement : RowAgreement 256 264 := RowAgreement.join within_002_002_agreement within_002_005_agreement
private def within_002_007 : CountSegment 1 10 13120 264 266 179 181 := CountSegment.join atomic_265 atomic_266
private theorem within_002_007_agreement : RowAgreement 264 266 := RowAgreement.join atomic_265_agreement atomic_266_agreement
private def within_002_008 : CountSegment 1 10 13120 266 268 181 183 := CountSegment.join atomic_267 atomic_268
private theorem within_002_008_agreement : RowAgreement 266 268 := RowAgreement.join atomic_267_agreement atomic_268_agreement
private def within_002_009 : CountSegment 1 10 13120 264 268 179 183 := CountSegment.join within_002_007 within_002_008
private theorem within_002_009_agreement : RowAgreement 264 268 := RowAgreement.join within_002_007_agreement within_002_008_agreement
private def within_002_010 : CountSegment 1 10 13120 268 270 183 185 := CountSegment.join atomic_269 atomic_270
private theorem within_002_010_agreement : RowAgreement 268 270 := RowAgreement.join atomic_269_agreement atomic_270_agreement
private def within_002_011 : CountSegment 1 10 13120 270 272 185 186 := CountSegment.join atomic_271 atomic_272
private theorem within_002_011_agreement : RowAgreement 270 272 := RowAgreement.join atomic_271_agreement atomic_272_agreement
private def within_002_012 : CountSegment 1 10 13120 268 272 183 186 := CountSegment.join within_002_010 within_002_011
private theorem within_002_012_agreement : RowAgreement 268 272 := RowAgreement.join within_002_010_agreement within_002_011_agreement
private def within_002_013 : CountSegment 1 10 13120 264 272 179 186 := CountSegment.join within_002_009 within_002_012
private theorem within_002_013_agreement : RowAgreement 264 272 := RowAgreement.join within_002_009_agreement within_002_012_agreement
private def within_002_014 : CountSegment 1 10 13120 256 272 177 186 := CountSegment.join within_002_006 within_002_013
private theorem within_002_014_agreement : RowAgreement 256 272 := RowAgreement.join within_002_006_agreement within_002_013_agreement
private def within_002_015 : CountSegment 1 10 13120 272 274 186 186 := CountSegment.join atomic_273 atomic_274
private theorem within_002_015_agreement : RowAgreement 272 274 := RowAgreement.join atomic_273_agreement atomic_274_agreement
private def within_002_016 : CountSegment 1 10 13120 274 276 186 187 := CountSegment.join atomic_275 atomic_276
private theorem within_002_016_agreement : RowAgreement 274 276 := RowAgreement.join atomic_275_agreement atomic_276_agreement
private def within_002_017 : CountSegment 1 10 13120 272 276 186 187 := CountSegment.join within_002_015 within_002_016
private theorem within_002_017_agreement : RowAgreement 272 276 := RowAgreement.join within_002_015_agreement within_002_016_agreement
private def within_002_018 : CountSegment 1 10 13120 276 278 187 189 := CountSegment.join atomic_277 atomic_278
private theorem within_002_018_agreement : RowAgreement 276 278 := RowAgreement.join atomic_277_agreement atomic_278_agreement
private def within_002_019 : CountSegment 1 10 13120 278 280 189 190 := CountSegment.join atomic_279 atomic_280
private theorem within_002_019_agreement : RowAgreement 278 280 := RowAgreement.join atomic_279_agreement atomic_280_agreement
private def within_002_020 : CountSegment 1 10 13120 276 280 187 190 := CountSegment.join within_002_018 within_002_019
private theorem within_002_020_agreement : RowAgreement 276 280 := RowAgreement.join within_002_018_agreement within_002_019_agreement
private def within_002_021 : CountSegment 1 10 13120 272 280 186 190 := CountSegment.join within_002_017 within_002_020
private theorem within_002_021_agreement : RowAgreement 272 280 := RowAgreement.join within_002_017_agreement within_002_020_agreement
private def within_002_022 : CountSegment 1 10 13120 280 282 190 190 := CountSegment.join atomic_281 atomic_282
private theorem within_002_022_agreement : RowAgreement 280 282 := RowAgreement.join atomic_281_agreement atomic_282_agreement
private def within_002_023 : CountSegment 1 10 13120 282 284 190 192 := CountSegment.join atomic_283 atomic_284
private theorem within_002_023_agreement : RowAgreement 282 284 := RowAgreement.join atomic_283_agreement atomic_284_agreement
private def within_002_024 : CountSegment 1 10 13120 280 284 190 192 := CountSegment.join within_002_022 within_002_023
private theorem within_002_024_agreement : RowAgreement 280 284 := RowAgreement.join within_002_022_agreement within_002_023_agreement
private def within_002_025 : CountSegment 1 10 13120 284 286 192 194 := CountSegment.join atomic_285 atomic_286
private theorem within_002_025_agreement : RowAgreement 284 286 := RowAgreement.join atomic_285_agreement atomic_286_agreement
private def within_002_026 : CountSegment 1 10 13120 286 288 194 195 := CountSegment.join atomic_287 atomic_288
private theorem within_002_026_agreement : RowAgreement 286 288 := RowAgreement.join atomic_287_agreement atomic_288_agreement
private def within_002_027 : CountSegment 1 10 13120 284 288 192 195 := CountSegment.join within_002_025 within_002_026
private theorem within_002_027_agreement : RowAgreement 284 288 := RowAgreement.join within_002_025_agreement within_002_026_agreement
private def within_002_028 : CountSegment 1 10 13120 280 288 190 195 := CountSegment.join within_002_024 within_002_027
private theorem within_002_028_agreement : RowAgreement 280 288 := RowAgreement.join within_002_024_agreement within_002_027_agreement
private def within_002_029 : CountSegment 1 10 13120 272 288 186 195 := CountSegment.join within_002_021 within_002_028
private theorem within_002_029_agreement : RowAgreement 272 288 := RowAgreement.join within_002_021_agreement within_002_028_agreement
private def within_002_030 : CountSegment 1 10 13120 256 288 177 195 := CountSegment.join within_002_014 within_002_029
private theorem within_002_030_agreement : RowAgreement 256 288 := RowAgreement.join within_002_014_agreement within_002_029_agreement
private def within_002_031 : CountSegment 1 10 13120 288 290 195 196 := CountSegment.join atomic_289 atomic_290
private theorem within_002_031_agreement : RowAgreement 288 290 := RowAgreement.join atomic_289_agreement atomic_290_agreement
private def within_002_032 : CountSegment 1 10 13120 290 292 196 199 := CountSegment.join atomic_291 atomic_292
private theorem within_002_032_agreement : RowAgreement 290 292 := RowAgreement.join atomic_291_agreement atomic_292_agreement
private def within_002_033 : CountSegment 1 10 13120 288 292 195 199 := CountSegment.join within_002_031 within_002_032
private theorem within_002_033_agreement : RowAgreement 288 292 := RowAgreement.join within_002_031_agreement within_002_032_agreement
private def within_002_034 : CountSegment 1 10 13120 292 294 199 199 := CountSegment.join atomic_293 atomic_294
private theorem within_002_034_agreement : RowAgreement 292 294 := RowAgreement.join atomic_293_agreement atomic_294_agreement
private def within_002_035 : CountSegment 1 10 13120 294 296 199 199 := CountSegment.join atomic_295 atomic_296
private theorem within_002_035_agreement : RowAgreement 294 296 := RowAgreement.join atomic_295_agreement atomic_296_agreement
private def within_002_036 : CountSegment 1 10 13120 292 296 199 199 := CountSegment.join within_002_034 within_002_035
private theorem within_002_036_agreement : RowAgreement 292 296 := RowAgreement.join within_002_034_agreement within_002_035_agreement
private def within_002_037 : CountSegment 1 10 13120 288 296 195 199 := CountSegment.join within_002_033 within_002_036
private theorem within_002_037_agreement : RowAgreement 288 296 := RowAgreement.join within_002_033_agreement within_002_036_agreement
private def within_002_038 : CountSegment 1 10 13120 296 298 199 200 := CountSegment.join atomic_297 atomic_298
private theorem within_002_038_agreement : RowAgreement 296 298 := RowAgreement.join atomic_297_agreement atomic_298_agreement
private def within_002_039 : CountSegment 1 10 13120 298 300 200 200 := CountSegment.join atomic_299 atomic_300
private theorem within_002_039_agreement : RowAgreement 298 300 := RowAgreement.join atomic_299_agreement atomic_300_agreement
private def within_002_040 : CountSegment 1 10 13120 296 300 199 200 := CountSegment.join within_002_038 within_002_039
private theorem within_002_040_agreement : RowAgreement 296 300 := RowAgreement.join within_002_038_agreement within_002_039_agreement
private def within_002_041 : CountSegment 1 10 13120 300 302 200 201 := CountSegment.join atomic_301 atomic_302
private theorem within_002_041_agreement : RowAgreement 300 302 := RowAgreement.join atomic_301_agreement atomic_302_agreement
private def within_002_042 : CountSegment 1 10 13120 302 304 201 202 := CountSegment.join atomic_303 atomic_304
private theorem within_002_042_agreement : RowAgreement 302 304 := RowAgreement.join atomic_303_agreement atomic_304_agreement
private def within_002_043 : CountSegment 1 10 13120 300 304 200 202 := CountSegment.join within_002_041 within_002_042
private theorem within_002_043_agreement : RowAgreement 300 304 := RowAgreement.join within_002_041_agreement within_002_042_agreement
private def within_002_044 : CountSegment 1 10 13120 296 304 199 202 := CountSegment.join within_002_040 within_002_043
private theorem within_002_044_agreement : RowAgreement 296 304 := RowAgreement.join within_002_040_agreement within_002_043_agreement
private def within_002_045 : CountSegment 1 10 13120 288 304 195 202 := CountSegment.join within_002_037 within_002_044
private theorem within_002_045_agreement : RowAgreement 288 304 := RowAgreement.join within_002_037_agreement within_002_044_agreement
private def within_002_046 : CountSegment 1 10 13120 304 306 202 204 := CountSegment.join atomic_305 atomic_306
private theorem within_002_046_agreement : RowAgreement 304 306 := RowAgreement.join atomic_305_agreement atomic_306_agreement
private def within_002_047 : CountSegment 1 10 13120 306 308 204 204 := CountSegment.join atomic_307 atomic_308
private theorem within_002_047_agreement : RowAgreement 306 308 := RowAgreement.join atomic_307_agreement atomic_308_agreement
private def within_002_048 : CountSegment 1 10 13120 304 308 202 204 := CountSegment.join within_002_046 within_002_047
private theorem within_002_048_agreement : RowAgreement 304 308 := RowAgreement.join within_002_046_agreement within_002_047_agreement
private def within_002_049 : CountSegment 1 10 13120 308 310 204 204 := CountSegment.join atomic_309 atomic_310
private theorem within_002_049_agreement : RowAgreement 308 310 := RowAgreement.join atomic_309_agreement atomic_310_agreement
private def within_002_050 : CountSegment 1 10 13120 310 312 204 206 := CountSegment.join atomic_311 atomic_312
private theorem within_002_050_agreement : RowAgreement 310 312 := RowAgreement.join atomic_311_agreement atomic_312_agreement
private def within_002_051 : CountSegment 1 10 13120 308 312 204 206 := CountSegment.join within_002_049 within_002_050
private theorem within_002_051_agreement : RowAgreement 308 312 := RowAgreement.join within_002_049_agreement within_002_050_agreement
private def within_002_052 : CountSegment 1 10 13120 304 312 202 206 := CountSegment.join within_002_048 within_002_051
private theorem within_002_052_agreement : RowAgreement 304 312 := RowAgreement.join within_002_048_agreement within_002_051_agreement
private def within_002_053 : CountSegment 1 10 13120 312 314 206 207 := CountSegment.join atomic_313 atomic_314
private theorem within_002_053_agreement : RowAgreement 312 314 := RowAgreement.join atomic_313_agreement atomic_314_agreement
private def within_002_054 : CountSegment 1 10 13120 314 316 207 207 := CountSegment.join atomic_315 atomic_316
private theorem within_002_054_agreement : RowAgreement 314 316 := RowAgreement.join atomic_315_agreement atomic_316_agreement
private def within_002_055 : CountSegment 1 10 13120 312 316 206 207 := CountSegment.join within_002_053 within_002_054
private theorem within_002_055_agreement : RowAgreement 312 316 := RowAgreement.join within_002_053_agreement within_002_054_agreement
private def within_002_056 : CountSegment 1 10 13120 316 318 207 210 := CountSegment.join atomic_317 atomic_318
private theorem within_002_056_agreement : RowAgreement 316 318 := RowAgreement.join atomic_317_agreement atomic_318_agreement
private def within_002_057 : CountSegment 1 10 13120 318 320 210 210 := CountSegment.join atomic_319 atomic_320
private theorem within_002_057_agreement : RowAgreement 318 320 := RowAgreement.join atomic_319_agreement atomic_320_agreement
private def within_002_058 : CountSegment 1 10 13120 316 320 207 210 := CountSegment.join within_002_056 within_002_057
private theorem within_002_058_agreement : RowAgreement 316 320 := RowAgreement.join within_002_056_agreement within_002_057_agreement
private def within_002_059 : CountSegment 1 10 13120 312 320 206 210 := CountSegment.join within_002_055 within_002_058
private theorem within_002_059_agreement : RowAgreement 312 320 := RowAgreement.join within_002_055_agreement within_002_058_agreement
private def within_002_060 : CountSegment 1 10 13120 304 320 202 210 := CountSegment.join within_002_052 within_002_059
private theorem within_002_060_agreement : RowAgreement 304 320 := RowAgreement.join within_002_052_agreement within_002_059_agreement
private def within_002_061 : CountSegment 1 10 13120 288 320 195 210 := CountSegment.join within_002_045 within_002_060
private theorem within_002_061_agreement : RowAgreement 288 320 := RowAgreement.join within_002_045_agreement within_002_060_agreement
private def within_002_062 : CountSegment 1 10 13120 256 320 177 210 := CountSegment.join within_002_030 within_002_061
private theorem within_002_062_agreement : RowAgreement 256 320 := RowAgreement.join within_002_030_agreement within_002_061_agreement
private def within_002_063 : CountSegment 1 10 13120 320 322 210 213 := CountSegment.join atomic_321 atomic_322
private theorem within_002_063_agreement : RowAgreement 320 322 := RowAgreement.join atomic_321_agreement atomic_322_agreement
private def within_002_064 : CountSegment 1 10 13120 322 324 213 214 := CountSegment.join atomic_323 atomic_324
private theorem within_002_064_agreement : RowAgreement 322 324 := RowAgreement.join atomic_323_agreement atomic_324_agreement
private def within_002_065 : CountSegment 1 10 13120 320 324 210 214 := CountSegment.join within_002_063 within_002_064
private theorem within_002_065_agreement : RowAgreement 320 324 := RowAgreement.join within_002_063_agreement within_002_064_agreement
private def within_002_066 : CountSegment 1 10 13120 324 326 214 214 := CountSegment.join atomic_325 atomic_326
private theorem within_002_066_agreement : RowAgreement 324 326 := RowAgreement.join atomic_325_agreement atomic_326_agreement
private def within_002_067 : CountSegment 1 10 13120 326 328 214 215 := CountSegment.join atomic_327 atomic_328
private theorem within_002_067_agreement : RowAgreement 326 328 := RowAgreement.join atomic_327_agreement atomic_328_agreement
private def within_002_068 : CountSegment 1 10 13120 324 328 214 215 := CountSegment.join within_002_066 within_002_067
private theorem within_002_068_agreement : RowAgreement 324 328 := RowAgreement.join within_002_066_agreement within_002_067_agreement
private def within_002_069 : CountSegment 1 10 13120 320 328 210 215 := CountSegment.join within_002_065 within_002_068
private theorem within_002_069_agreement : RowAgreement 320 328 := RowAgreement.join within_002_065_agreement within_002_068_agreement
private def within_002_070 : CountSegment 1 10 13120 328 330 215 215 := CountSegment.join atomic_329 atomic_330
private theorem within_002_070_agreement : RowAgreement 328 330 := RowAgreement.join atomic_329_agreement atomic_330_agreement
private def within_002_071 : CountSegment 1 10 13120 330 332 215 215 := CountSegment.join atomic_331 atomic_332
private theorem within_002_071_agreement : RowAgreement 330 332 := RowAgreement.join atomic_331_agreement atomic_332_agreement
private def within_002_072 : CountSegment 1 10 13120 328 332 215 215 := CountSegment.join within_002_070 within_002_071
private theorem within_002_072_agreement : RowAgreement 328 332 := RowAgreement.join within_002_070_agreement within_002_071_agreement
private def within_002_073 : CountSegment 1 10 13120 332 334 215 216 := CountSegment.join atomic_333 atomic_334
private theorem within_002_073_agreement : RowAgreement 332 334 := RowAgreement.join atomic_333_agreement atomic_334_agreement
private def within_002_074 : CountSegment 1 10 13120 334 336 216 217 := CountSegment.join atomic_335 atomic_336
private theorem within_002_074_agreement : RowAgreement 334 336 := RowAgreement.join atomic_335_agreement atomic_336_agreement
private def within_002_075 : CountSegment 1 10 13120 332 336 215 217 := CountSegment.join within_002_073 within_002_074
private theorem within_002_075_agreement : RowAgreement 332 336 := RowAgreement.join within_002_073_agreement within_002_074_agreement
private def within_002_076 : CountSegment 1 10 13120 328 336 215 217 := CountSegment.join within_002_072 within_002_075
private theorem within_002_076_agreement : RowAgreement 328 336 := RowAgreement.join within_002_072_agreement within_002_075_agreement
private def within_002_077 : CountSegment 1 10 13120 320 336 210 217 := CountSegment.join within_002_069 within_002_076
private theorem within_002_077_agreement : RowAgreement 320 336 := RowAgreement.join within_002_069_agreement within_002_076_agreement
private def within_002_078 : CountSegment 1 10 13120 336 338 217 218 := CountSegment.join atomic_337 atomic_338
private theorem within_002_078_agreement : RowAgreement 336 338 := RowAgreement.join atomic_337_agreement atomic_338_agreement
private def within_002_079 : CountSegment 1 10 13120 338 340 218 220 := CountSegment.join atomic_339 atomic_340
private theorem within_002_079_agreement : RowAgreement 338 340 := RowAgreement.join atomic_339_agreement atomic_340_agreement
private def within_002_080 : CountSegment 1 10 13120 336 340 217 220 := CountSegment.join within_002_078 within_002_079
private theorem within_002_080_agreement : RowAgreement 336 340 := RowAgreement.join within_002_078_agreement within_002_079_agreement
private def within_002_081 : CountSegment 1 10 13120 340 342 220 220 := CountSegment.join atomic_341 atomic_342
private theorem within_002_081_agreement : RowAgreement 340 342 := RowAgreement.join atomic_341_agreement atomic_342_agreement
private def within_002_082 : CountSegment 1 10 13120 342 344 220 222 := CountSegment.join atomic_343 atomic_344
private theorem within_002_082_agreement : RowAgreement 342 344 := RowAgreement.join atomic_343_agreement atomic_344_agreement
private def within_002_083 : CountSegment 1 10 13120 340 344 220 222 := CountSegment.join within_002_081 within_002_082
private theorem within_002_083_agreement : RowAgreement 340 344 := RowAgreement.join within_002_081_agreement within_002_082_agreement
private def within_002_084 : CountSegment 1 10 13120 336 344 217 222 := CountSegment.join within_002_080 within_002_083
private theorem within_002_084_agreement : RowAgreement 336 344 := RowAgreement.join within_002_080_agreement within_002_083_agreement
private def within_002_085 : CountSegment 1 10 13120 344 346 222 223 := CountSegment.join atomic_345 atomic_346
private theorem within_002_085_agreement : RowAgreement 344 346 := RowAgreement.join atomic_345_agreement atomic_346_agreement
private def within_002_086 : CountSegment 1 10 13120 346 348 223 225 := CountSegment.join atomic_347 atomic_348
private theorem within_002_086_agreement : RowAgreement 346 348 := RowAgreement.join atomic_347_agreement atomic_348_agreement
private def within_002_087 : CountSegment 1 10 13120 344 348 222 225 := CountSegment.join within_002_085 within_002_086
private theorem within_002_087_agreement : RowAgreement 344 348 := RowAgreement.join within_002_085_agreement within_002_086_agreement
private def within_002_088 : CountSegment 1 10 13120 348 350 225 226 := CountSegment.join atomic_349 atomic_350
private theorem within_002_088_agreement : RowAgreement 348 350 := RowAgreement.join atomic_349_agreement atomic_350_agreement
private def within_002_089 : CountSegment 1 10 13120 350 352 226 228 := CountSegment.join atomic_351 atomic_352
private theorem within_002_089_agreement : RowAgreement 350 352 := RowAgreement.join atomic_351_agreement atomic_352_agreement
private def within_002_090 : CountSegment 1 10 13120 348 352 225 228 := CountSegment.join within_002_088 within_002_089
private theorem within_002_090_agreement : RowAgreement 348 352 := RowAgreement.join within_002_088_agreement within_002_089_agreement
private def within_002_091 : CountSegment 1 10 13120 344 352 222 228 := CountSegment.join within_002_087 within_002_090
private theorem within_002_091_agreement : RowAgreement 344 352 := RowAgreement.join within_002_087_agreement within_002_090_agreement
private def within_002_092 : CountSegment 1 10 13120 336 352 217 228 := CountSegment.join within_002_084 within_002_091
private theorem within_002_092_agreement : RowAgreement 336 352 := RowAgreement.join within_002_084_agreement within_002_091_agreement
private def within_002_093 : CountSegment 1 10 13120 320 352 210 228 := CountSegment.join within_002_077 within_002_092
private theorem within_002_093_agreement : RowAgreement 320 352 := RowAgreement.join within_002_077_agreement within_002_092_agreement
private def within_002_094 : CountSegment 1 10 13120 352 354 228 228 := CountSegment.join atomic_353 atomic_354
private theorem within_002_094_agreement : RowAgreement 352 354 := RowAgreement.join atomic_353_agreement atomic_354_agreement
private def within_002_095 : CountSegment 1 10 13120 354 356 228 228 := CountSegment.join atomic_355 atomic_356
private theorem within_002_095_agreement : RowAgreement 354 356 := RowAgreement.join atomic_355_agreement atomic_356_agreement
private def within_002_096 : CountSegment 1 10 13120 352 356 228 228 := CountSegment.join within_002_094 within_002_095
private theorem within_002_096_agreement : RowAgreement 352 356 := RowAgreement.join within_002_094_agreement within_002_095_agreement
private def within_002_097 : CountSegment 1 10 13120 356 358 228 230 := CountSegment.join atomic_357 atomic_358
private theorem within_002_097_agreement : RowAgreement 356 358 := RowAgreement.join atomic_357_agreement atomic_358_agreement
private def within_002_098 : CountSegment 1 10 13120 358 360 230 231 := CountSegment.join atomic_359 atomic_360
private theorem within_002_098_agreement : RowAgreement 358 360 := RowAgreement.join atomic_359_agreement atomic_360_agreement
private def within_002_099 : CountSegment 1 10 13120 356 360 228 231 := CountSegment.join within_002_097 within_002_098
private theorem within_002_099_agreement : RowAgreement 356 360 := RowAgreement.join within_002_097_agreement within_002_098_agreement
private def within_002_100 : CountSegment 1 10 13120 352 360 228 231 := CountSegment.join within_002_096 within_002_099
private theorem within_002_100_agreement : RowAgreement 352 360 := RowAgreement.join within_002_096_agreement within_002_099_agreement
private def within_002_101 : CountSegment 1 10 13120 360 362 231 231 := CountSegment.join atomic_361 atomic_362
private theorem within_002_101_agreement : RowAgreement 360 362 := RowAgreement.join atomic_361_agreement atomic_362_agreement
private def within_002_102 : CountSegment 1 10 13120 362 364 231 233 := CountSegment.join atomic_363 atomic_364
private theorem within_002_102_agreement : RowAgreement 362 364 := RowAgreement.join atomic_363_agreement atomic_364_agreement
private def within_002_103 : CountSegment 1 10 13120 360 364 231 233 := CountSegment.join within_002_101 within_002_102
private theorem within_002_103_agreement : RowAgreement 360 364 := RowAgreement.join within_002_101_agreement within_002_102_agreement
private def within_002_104 : CountSegment 1 10 13120 364 366 233 233 := CountSegment.join atomic_365 atomic_366
private theorem within_002_104_agreement : RowAgreement 364 366 := RowAgreement.join atomic_365_agreement atomic_366_agreement
private def within_002_105 : CountSegment 1 10 13120 366 368 233 234 := CountSegment.join atomic_367 atomic_368
private theorem within_002_105_agreement : RowAgreement 366 368 := RowAgreement.join atomic_367_agreement atomic_368_agreement
private def within_002_106 : CountSegment 1 10 13120 364 368 233 234 := CountSegment.join within_002_104 within_002_105
private theorem within_002_106_agreement : RowAgreement 364 368 := RowAgreement.join within_002_104_agreement within_002_105_agreement
private def within_002_107 : CountSegment 1 10 13120 360 368 231 234 := CountSegment.join within_002_103 within_002_106
private theorem within_002_107_agreement : RowAgreement 360 368 := RowAgreement.join within_002_103_agreement within_002_106_agreement
private def within_002_108 : CountSegment 1 10 13120 352 368 228 234 := CountSegment.join within_002_100 within_002_107
private theorem within_002_108_agreement : RowAgreement 352 368 := RowAgreement.join within_002_100_agreement within_002_107_agreement
private def within_002_109 : CountSegment 1 10 13120 368 370 234 235 := CountSegment.join atomic_369 atomic_370
private theorem within_002_109_agreement : RowAgreement 368 370 := RowAgreement.join atomic_369_agreement atomic_370_agreement
private def within_002_110 : CountSegment 1 10 13120 370 372 235 236 := CountSegment.join atomic_371 atomic_372
private theorem within_002_110_agreement : RowAgreement 370 372 := RowAgreement.join atomic_371_agreement atomic_372_agreement
private def within_002_111 : CountSegment 1 10 13120 368 372 234 236 := CountSegment.join within_002_109 within_002_110
private theorem within_002_111_agreement : RowAgreement 368 372 := RowAgreement.join within_002_109_agreement within_002_110_agreement
private def within_002_112 : CountSegment 1 10 13120 372 374 236 237 := CountSegment.join atomic_373 atomic_374
private theorem within_002_112_agreement : RowAgreement 372 374 := RowAgreement.join atomic_373_agreement atomic_374_agreement
private def within_002_113 : CountSegment 1 10 13120 374 376 237 238 := CountSegment.join atomic_375 atomic_376
private theorem within_002_113_agreement : RowAgreement 374 376 := RowAgreement.join atomic_375_agreement atomic_376_agreement
private def within_002_114 : CountSegment 1 10 13120 372 376 236 238 := CountSegment.join within_002_112 within_002_113
private theorem within_002_114_agreement : RowAgreement 372 376 := RowAgreement.join within_002_112_agreement within_002_113_agreement
private def within_002_115 : CountSegment 1 10 13120 368 376 234 238 := CountSegment.join within_002_111 within_002_114
private theorem within_002_115_agreement : RowAgreement 368 376 := RowAgreement.join within_002_111_agreement within_002_114_agreement
private def within_002_116 : CountSegment 1 10 13120 376 378 238 239 := CountSegment.join atomic_377 atomic_378
private theorem within_002_116_agreement : RowAgreement 376 378 := RowAgreement.join atomic_377_agreement atomic_378_agreement
private def within_002_117 : CountSegment 1 10 13120 378 380 239 240 := CountSegment.join atomic_379 atomic_380
private theorem within_002_117_agreement : RowAgreement 378 380 := RowAgreement.join atomic_379_agreement atomic_380_agreement
private def within_002_118 : CountSegment 1 10 13120 376 380 238 240 := CountSegment.join within_002_116 within_002_117
private theorem within_002_118_agreement : RowAgreement 376 380 := RowAgreement.join within_002_116_agreement within_002_117_agreement
private def within_002_119 : CountSegment 1 10 13120 380 382 240 243 := CountSegment.join atomic_381 atomic_382
private theorem within_002_119_agreement : RowAgreement 380 382 := RowAgreement.join atomic_381_agreement atomic_382_agreement
private def within_002_120 : CountSegment 1 10 13120 382 384 243 244 := CountSegment.join atomic_383 atomic_384
private theorem within_002_120_agreement : RowAgreement 382 384 := RowAgreement.join atomic_383_agreement atomic_384_agreement
private def within_002_121 : CountSegment 1 10 13120 380 384 240 244 := CountSegment.join within_002_119 within_002_120
private theorem within_002_121_agreement : RowAgreement 380 384 := RowAgreement.join within_002_119_agreement within_002_120_agreement
private def within_002_122 : CountSegment 1 10 13120 376 384 238 244 := CountSegment.join within_002_118 within_002_121
private theorem within_002_122_agreement : RowAgreement 376 384 := RowAgreement.join within_002_118_agreement within_002_121_agreement
private def within_002_123 : CountSegment 1 10 13120 368 384 234 244 := CountSegment.join within_002_115 within_002_122
private theorem within_002_123_agreement : RowAgreement 368 384 := RowAgreement.join within_002_115_agreement within_002_122_agreement
private def within_002_124 : CountSegment 1 10 13120 352 384 228 244 := CountSegment.join within_002_108 within_002_123
private theorem within_002_124_agreement : RowAgreement 352 384 := RowAgreement.join within_002_108_agreement within_002_123_agreement
private def within_002_125 : CountSegment 1 10 13120 320 384 210 244 := CountSegment.join within_002_093 within_002_124
private theorem within_002_125_agreement : RowAgreement 320 384 := RowAgreement.join within_002_093_agreement within_002_124_agreement
private def within_002_126 : CountSegment 1 10 13120 256 384 177 244 := CountSegment.join within_002_062 within_002_125
private theorem within_002_126_agreement : RowAgreement 256 384 := RowAgreement.join within_002_062_agreement within_002_125_agreement
def segment_002 : CountSegment 1 10 13120 256 384 177 244 := within_002_126
theorem agreement_002 : RowAgreement 256 384 := within_002_126_agreement
end ProjectEulerSolutions.P58Numerical
