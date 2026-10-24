import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_10241 : RowCertificate 10241 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10241 1) := by change ¬ Nat.Prime 419532807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139844269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10241 2) := by change ¬ Nat.Prime 419512325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83902465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10241 3) := by change ¬ Nat.Prime 419491843; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1988113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10241 : row 10241 = 0 := (row_10241.checked).trans rfl
theorem model_count_10241 : modelRow 10241 = 0 := by decide
theorem modelrow_agrees_10241 : modelRow 10241 = row 10241 := model_count_10241.trans row_count_10241.symm
def row_10242 : RowCertificate 10242 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10242 1) := by change ¬ Nat.Prime 419614741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59944963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10242 2) := by change Nat.Prime 419594257; exact ProjectEulerSolutions.P58PrimeNodes.p_419594257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10242 3) := by change ¬ Nat.Prime 419573773; exact PrattCertificates.composite_of_product (factor := 19609) (cofactor := 21397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10242 : row 10242 = 1 := (row_10242.checked).trans rfl
theorem model_count_10242 : modelRow 10242 = 1 := by decide
theorem modelrow_agrees_10242 : modelRow 10242 = row 10242 := model_count_10242.trans row_count_10242.symm
def row_10243 : RowCertificate 10243 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10243 1) := by change ¬ Nat.Prime 419696683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59956669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10243 2) := by change ¬ Nat.Prime 419676197; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14471593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10243 3) := by change ¬ Nat.Prime 419655711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139885237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10243 : row 10243 = 0 := (row_10243.checked).trans rfl
theorem model_count_10243 : modelRow 10243 = 0 := by decide
theorem modelrow_agrees_10243 : modelRow 10243 = row 10243 := model_count_10243.trans row_count_10243.symm
def row_10244 : RowCertificate 10244 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10244 1) := by change ¬ Nat.Prime 419778633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139926211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10244 2) := by change ¬ Nat.Prime 419758145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83951629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10244 3) := by change ¬ Nat.Prime 419737657; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11344261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10244 : row 10244 = 0 := (row_10244.checked).trans rfl
theorem model_count_10244 : modelRow 10244 = 0 := by decide
theorem modelrow_agrees_10244 : modelRow 10244 = row 10244 := model_count_10244.trans row_count_10244.symm
def row_10245 : RowCertificate 10245 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10245 1) := by change ¬ Nat.Prime 419860591; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 1268461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10245 2) := by change ¬ Nat.Prime 419840101; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 371869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10245 3) := by change ¬ Nat.Prime 419819611; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22095769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10245 : row 10245 = 0 := (row_10245.checked).trans rfl
theorem model_count_10245 : modelRow 10245 = 0 := by decide
theorem modelrow_agrees_10245 : modelRow 10245 = row 10245 := model_count_10245.trans row_count_10245.symm
def row_10246 : RowCertificate 10246 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10246 1) := by change ¬ Nat.Prime 419942557; exact PrattCertificates.composite_of_product (factor := 2713) (cofactor := 154789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10246 2) := by change ¬ Nat.Prime 419922065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 83984413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10246 3) := by change ¬ Nat.Prime 419901573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 139967191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10246 : row 10246 = 0 := (row_10246.checked).trans rfl
theorem model_count_10246 : modelRow 10246 = 0 := by decide
theorem modelrow_agrees_10246 : modelRow 10246 = row 10246 := model_count_10246.trans row_count_10246.symm
def row_10247 : RowCertificate 10247 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10247 1) := by change ¬ Nat.Prime 420024531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140008177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10247 2) := by change ¬ Nat.Prime 420004037; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 2427769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10247 3) := by change ¬ Nat.Prime 419983543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 59997649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10247 : row 10247 = 0 := (row_10247.checked).trans rfl
theorem model_count_10247 : modelRow 10247 = 0 := by decide
theorem modelrow_agrees_10247 : modelRow 10247 = row 10247 := model_count_10247.trans row_count_10247.symm
def row_10248 : RowCertificate 10248 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10248 1) := by change ¬ Nat.Prime 420106513; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13551823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10248 2) := by change ¬ Nat.Prime 420086017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32314309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10248 3) := by change ¬ Nat.Prime 420065521; exact PrattCertificates.composite_of_product (factor := 3067) (cofactor := 136963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10248 : row 10248 = 0 := (row_10248.checked).trans rfl
theorem model_count_10248 : modelRow 10248 = 0 := by decide
theorem modelrow_agrees_10248 : modelRow 10248 = row 10248 := model_count_10248.trans row_count_10248.symm
def row_10249 : RowCertificate 10249 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10249 1) := by change ¬ Nat.Prime 420188503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60026929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10249 2) := by change ¬ Nat.Prime 420168005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84033601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10249 3) := by change ¬ Nat.Prime 420147507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140049169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10249 : row 10249 = 0 := (row_10249.checked).trans rfl
theorem model_count_10249 : modelRow 10249 = 0 := by decide
theorem modelrow_agrees_10249 : modelRow 10249 = row 10249 := model_count_10249.trans row_count_10249.symm
def row_10250 : RowCertificate 10250 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10250 1) := by change ¬ Nat.Prime 420270501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140090167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10250 2) := by change Nat.Prime 420250001; exact ProjectEulerSolutions.P58PrimeNodes.p_420250001.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10250 3) := by change Nat.Prime 420229501; exact ProjectEulerSolutions.P58PrimeNodes.p_420229501.prime
    exact decide_eq_true prime
theorem row_count_10250 : row 10250 = 2 := (row_10250.checked).trans rfl
theorem model_count_10250 : modelRow 10250 = 2 := by decide
theorem modelrow_agrees_10250 : modelRow 10250 = row 10250 := model_count_10250.trans row_count_10250.symm
def row_10251 : RowCertificate 10251 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10251 1) := by change ¬ Nat.Prime 420352507; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4333531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10251 2) := by change ¬ Nat.Prime 420332005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84066401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10251 3) := by change ¬ Nat.Prime 420311503; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2322163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10251 : row 10251 = 0 := (row_10251.checked).trans rfl
theorem model_count_10251 : modelRow 10251 = 0 := by decide
theorem modelrow_agrees_10251 : modelRow 10251 = row 10251 := model_count_10251.trans row_count_10251.symm
def row_10252 : RowCertificate 10252 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10252 1) := by change ¬ Nat.Prime 420434521; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32341117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10252 2) := by change ¬ Nat.Prime 420414017; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11362541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10252 3) := by change ¬ Nat.Prime 420393513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140131171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10252 : row 10252 = 0 := (row_10252.checked).trans rfl
theorem model_count_10252 : modelRow 10252 = 0 := by decide
theorem modelrow_agrees_10252 : modelRow 10252 = row 10252 := model_count_10252.trans row_count_10252.symm
def row_10253 : RowCertificate 10253 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10253 1) := by change ¬ Nat.Prime 420516543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140172181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10253 2) := by change ¬ Nat.Prime 420496037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32345849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10253 3) := by change ¬ Nat.Prime 420475531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60067933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10253 : row 10253 = 0 := (row_10253.checked).trans rfl
theorem model_count_10253 : modelRow 10253 = 0 := by decide
theorem modelrow_agrees_10253 : modelRow 10253 = row 10253 := model_count_10253.trans row_count_10253.symm
def row_10254 : RowCertificate 10254 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10254 1) := by change ¬ Nat.Prime 420598573; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22136767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10254 2) := by change ¬ Nat.Prime 420578065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84115613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10254 3) := by change ¬ Nat.Prime 420557557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60079651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10254 : row 10254 = 0 := (row_10254.checked).trans rfl
theorem model_count_10254 : modelRow 10254 = 0 := by decide
theorem modelrow_agrees_10254 : modelRow 10254 = row 10254 := model_count_10254.trans row_count_10254.symm
def row_10255 : RowCertificate 10255 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10255 1) := by change ¬ Nat.Prime 420680611; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32360047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10255 2) := by change ¬ Nat.Prime 420660101; exact PrattCertificates.composite_of_product (factor := 1877) (cofactor := 224113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10255 3) := by change ¬ Nat.Prime 420639591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140213197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10255 : row 10255 = 0 := (row_10255.checked).trans rfl
theorem model_count_10255 : modelRow 10255 = 0 := by decide
theorem modelrow_agrees_10255 : modelRow 10255 = row 10255 := model_count_10255.trans row_count_10255.symm
def row_10256 : RowCertificate 10256 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10256 1) := by change ¬ Nat.Prime 420762657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140254219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10256 2) := by change ¬ Nat.Prime 420742145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84148429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10256 3) := by change ¬ Nat.Prime 420721633; exact PrattCertificates.composite_of_product (factor := 12721) (cofactor := 33073) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10256 : row 10256 = 0 := (row_10256.checked).trans rfl
theorem model_count_10256 : modelRow 10256 = 0 := by decide
theorem modelrow_agrees_10256 : modelRow 10256 = row 10256 := model_count_10256.trans row_count_10256.symm
def row_10257 : RowCertificate 10257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10257 1) := by change ¬ Nat.Prime 420844711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60120673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10257 2) := by change Nat.Prime 420824197; exact ProjectEulerSolutions.P58PrimeNodes.p_420824197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10257 3) := by change ¬ Nat.Prime 420803683; exact PrattCertificates.composite_of_product (factor := 3037) (cofactor := 138559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10257 : row 10257 = 1 := (row_10257.checked).trans rfl
theorem model_count_10257 : modelRow 10257 = 1 := by decide
theorem modelrow_agrees_10257 : modelRow 10257 = row 10257 := model_count_10257.trans row_count_10257.symm
def row_10258 : RowCertificate 10258 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10258 1) := by change ¬ Nat.Prime 420926773; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13578283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10258 2) := by change ¬ Nat.Prime 420906257; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 1192369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10258 3) := by change ¬ Nat.Prime 420885741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140295247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10258 : row 10258 = 0 := (row_10258.checked).trans rfl
theorem model_count_10258 : modelRow 10258 = 0 := by decide
theorem modelrow_agrees_10258 : modelRow 10258 = row 10258 := model_count_10258.trans row_count_10258.symm
def row_10259 : RowCertificate 10259 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10259 1) := by change ¬ Nat.Prime 421008843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140336281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10259 2) := by change ¬ Nat.Prime 420988325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84197665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10259 3) := by change ¬ Nat.Prime 420967807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32382139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10259 : row 10259 = 0 := (row_10259.checked).trans rfl
theorem model_count_10259 : modelRow 10259 = 0 := by decide
theorem modelrow_agrees_10259 : modelRow 10259 = row 10259 := model_count_10259.trans row_count_10259.symm
def row_10260 : RowCertificate 10260 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10260 1) := by change Nat.Prime 421090921; exact ProjectEulerSolutions.P58PrimeNodes.p_421090921.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10260 2) := by change ¬ Nat.Prime 421070401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14519669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10260 3) := by change ¬ Nat.Prime 421049881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60149983) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10260 : row 10260 = 1 := (row_10260.checked).trans rfl
theorem model_count_10260 : modelRow 10260 = 1 := by decide
theorem modelrow_agrees_10260 : modelRow 10260 = row 10260 := model_count_10260.trans row_count_10260.symm
def row_10261 : RowCertificate 10261 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10261 1) := by change Nat.Prime 421173007; exact ProjectEulerSolutions.P58PrimeNodes.p_421173007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10261 2) := by change ¬ Nat.Prime 421152485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84230497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10261 3) := by change ¬ Nat.Prime 421131963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140377321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10261 : row 10261 = 1 := (row_10261.checked).trans rfl
theorem model_count_10261 : modelRow 10261 = 1 := by decide
theorem modelrow_agrees_10261 : modelRow 10261 = row 10261 := model_count_10261.trans row_count_10261.symm
def row_10262 : RowCertificate 10262 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10262 1) := by change ¬ Nat.Prime 421255101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140418367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10262 2) := by change ¬ Nat.Prime 421234577; exact PrattCertificates.composite_of_product (factor := 953) (cofactor := 442009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10262 3) := by change ¬ Nat.Prime 421214053; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32401081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10262 : row 10262 = 0 := (row_10262.checked).trans rfl
theorem model_count_10262 : modelRow 10262 = 0 := by decide
theorem modelrow_agrees_10262 : modelRow 10262 = row 10262 := model_count_10262.trans row_count_10262.symm
def row_10263 : RowCertificate 10263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10263 1) := by change ¬ Nat.Prime 421337203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60191029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10263 2) := by change ¬ Nat.Prime 421316677; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 1566233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10263 3) := by change Nat.Prime 421296151; exact ProjectEulerSolutions.P58PrimeNodes.p_421296151.prime
    exact decide_eq_true prime
theorem row_count_10263 : row 10263 = 1 := (row_10263.checked).trans rfl
theorem model_count_10263 : modelRow 10263 = 1 := by decide
theorem modelrow_agrees_10263 : modelRow 10263 = row 10263 := model_count_10263.trans row_count_10263.symm
def row_10264 : RowCertificate 10264 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10264 1) := by change ¬ Nat.Prime 421419313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60202759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10264 2) := by change ¬ Nat.Prime 421398785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84279757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10264 3) := by change ¬ Nat.Prime 421378257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140459419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10264 : row 10264 = 0 := (row_10264.checked).trans rfl
theorem model_count_10264 : modelRow 10264 = 0 := by decide
theorem modelrow_agrees_10264 : modelRow 10264 = row 10264 := model_count_10264.trans row_count_10264.symm
def row_10265 : RowCertificate 10265 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10265 1) := by change ¬ Nat.Prime 421501431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140500477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10265 2) := by change ¬ Nat.Prime 421480901; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 973397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10265 3) := by change ¬ Nat.Prime 421460371; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 3032089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10265 : row 10265 = 0 := (row_10265.checked).trans rfl
theorem model_count_10265 : modelRow 10265 = 0 := by decide
theorem modelrow_agrees_10265 : modelRow 10265 = row 10265 := model_count_10265.trans row_count_10265.symm
def row_10266 : RowCertificate 10266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10266 1) := by change ¬ Nat.Prime 421583557; exact PrattCertificates.composite_of_product (factor := 2221) (cofactor := 189817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10266 2) := by change ¬ Nat.Prime 421563025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84312605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10266 3) := by change ¬ Nat.Prime 421542493; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22186447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10266 : row 10266 = 0 := (row_10266.checked).trans rfl
theorem model_count_10266 : modelRow 10266 = 0 := by decide
theorem modelrow_agrees_10266 : modelRow 10266 = row 10266 := model_count_10266.trans row_count_10266.symm
def row_10267 : RowCertificate 10267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10267 1) := by change ¬ Nat.Prime 421665691; exact PrattCertificates.composite_of_product (factor := 4231) (cofactor := 99661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10267 2) := by change ¬ Nat.Prime 421645157; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7955569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10267 3) := by change ¬ Nat.Prime 421624623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140541541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10267 : row 10267 = 0 := (row_10267.checked).trans rfl
theorem model_count_10267 : modelRow 10267 = 0 := by decide
theorem modelrow_agrees_10267 : modelRow 10267 = row 10267 := model_count_10267.trans row_count_10267.symm
def row_10268 : RowCertificate 10268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10268 1) := by change ¬ Nat.Prime 421747833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140582611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10268 2) := by change Nat.Prime 421727297; exact ProjectEulerSolutions.P58PrimeNodes.p_421727297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10268 3) := by change ¬ Nat.Prime 421706761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60243823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10268 : row 10268 = 1 := (row_10268.checked).trans rfl
theorem model_count_10268 : modelRow 10268 = 1 := by decide
theorem modelrow_agrees_10268 : modelRow 10268 = row 10268 := model_count_10268.trans row_count_10268.symm
def row_10269 : RowCertificate 10269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10269 1) := by change ¬ Nat.Prime 421829983; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 264139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10269 2) := by change ¬ Nat.Prime 421809445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84361889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10269 3) := by change ¬ Nat.Prime 421788907; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3869623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10269 : row 10269 = 0 := (row_10269.checked).trans rfl
theorem model_count_10269 : modelRow 10269 = 0 := by decide
theorem modelrow_agrees_10269 : modelRow 10269 = row 10269 := model_count_10269.trans row_count_10269.symm
def row_10270 : RowCertificate 10270 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10270 1) := by change ¬ Nat.Prime 421912141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60273163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10270 2) := by change ¬ Nat.Prime 421891601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24817153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10270 3) := by change ¬ Nat.Prime 421871061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140623687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10270 : row 10270 = 0 := (row_10270.checked).trans rfl
theorem model_count_10270 : modelRow 10270 = 0 := by decide
theorem modelrow_agrees_10270 : modelRow 10270 = row 10270 := model_count_10270.trans row_count_10270.symm
def row_10271 : RowCertificate 10271 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10271 1) := by change ¬ Nat.Prime 421994307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140664769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10271 2) := by change ¬ Nat.Prime 421973765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84394753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10271 3) := by change Nat.Prime 421953223; exact ProjectEulerSolutions.P58PrimeNodes.p_421953223.prime
    exact decide_eq_true prime
theorem row_count_10271 : row 10271 = 1 := (row_10271.checked).trans rfl
theorem model_count_10271 : modelRow 10271 = 1 := by decide
theorem modelrow_agrees_10271 : modelRow 10271 = row 10271 := model_count_10271.trans row_count_10271.symm
def row_10272 : RowCertificate 10272 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10272 1) := by change Nat.Prime 422076481; exact ProjectEulerSolutions.P58PrimeNodes.p_422076481.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10272 2) := by change ¬ Nat.Prime 422055937; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14553653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10272 3) := by change ¬ Nat.Prime 422035393; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32464261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10272 : row 10272 = 1 := (row_10272.checked).trans rfl
theorem model_count_10272 : modelRow 10272 = 1 := by decide
theorem modelrow_agrees_10272 : modelRow 10272 = row 10272 := model_count_10272.trans row_count_10272.symm
def row_10273 : RowCertificate 10273 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10273 1) := by change ¬ Nat.Prime 422158663; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22218877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10273 2) := by change ¬ Nat.Prime 422138117; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6920297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10273 3) := by change ¬ Nat.Prime 422117571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140705857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10273 : row 10273 = 0 := (row_10273.checked).trans rfl
theorem model_count_10273 : modelRow 10273 = 0 := by decide
theorem modelrow_agrees_10273 : modelRow 10273 = row 10273 := model_count_10273.trans row_count_10273.symm
def row_10274 : RowCertificate 10274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10274 1) := by change ¬ Nat.Prime 422240853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140746951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10274 2) := by change ¬ Nat.Prime 422220305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84444061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10274 3) := by change ¬ Nat.Prime 422199757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60314251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10274 : row 10274 = 0 := (row_10274.checked).trans rfl
theorem model_count_10274 : modelRow 10274 = 0 := by decide
theorem modelrow_agrees_10274 : modelRow 10274 = row 10274 := model_count_10274.trans row_count_10274.symm
def row_10275 : RowCertificate 10275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10275 1) := by change ¬ Nat.Prime 422323051; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22227529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10275 2) := by change ¬ Nat.Prime 422302501; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10300061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10275 3) := by change ¬ Nat.Prime 422281951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60325993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10275 : row 10275 = 0 := (row_10275.checked).trans rfl
theorem model_count_10275 : modelRow 10275 = 0 := by decide
theorem modelrow_agrees_10275 : modelRow 10275 = row 10275 := model_count_10275.trans row_count_10275.symm
def row_10276 : RowCertificate 10276 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10276 1) := by change Nat.Prime 422405257; exact ProjectEulerSolutions.P58PrimeNodes.p_422405257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10276 2) := by change ¬ Nat.Prime 422384705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84476941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10276 3) := by change ¬ Nat.Prime 422364153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140788051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10276 : row 10276 = 1 := (row_10276.checked).trans rfl
theorem model_count_10276 : modelRow 10276 = 1 := by decide
theorem modelrow_agrees_10276 : modelRow 10276 = row 10276 := model_count_10276.trans row_count_10276.symm
def row_10277 : RowCertificate 10277 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10277 1) := by change ¬ Nat.Prime 422487471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140829157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10277 2) := by change ¬ Nat.Prime 422466917; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 810877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10277 3) := by change ¬ Nat.Prime 422446363; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 4101421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10277 : row 10277 = 0 := (row_10277.checked).trans rfl
theorem model_count_10277 : modelRow 10277 = 0 := by decide
theorem modelrow_agrees_10277 : modelRow 10277 = row 10277 := model_count_10277.trans row_count_10277.symm
def row_10278 : RowCertificate 10278 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10278 1) := by change ¬ Nat.Prime 422569693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60367099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10278 2) := by change Nat.Prime 422549137; exact ProjectEulerSolutions.P58PrimeNodes.p_422549137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10278 3) := by change ¬ Nat.Prime 422528581; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3876409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10278 : row 10278 = 1 := (row_10278.checked).trans rfl
theorem model_count_10278 : modelRow 10278 = 1 := by decide
theorem modelrow_agrees_10278 : modelRow 10278 = row 10278 := model_count_10278.trans row_count_10278.symm
def row_10279 : RowCertificate 10279 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10279 1) := by change ¬ Nat.Prime 422651923; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13633933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10279 2) := by change ¬ Nat.Prime 422631365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84526273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10279 3) := by change ¬ Nat.Prime 422610807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140870269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10279 : row 10279 = 0 := (row_10279.checked).trans rfl
theorem model_count_10279 : modelRow 10279 = 0 := by decide
theorem modelrow_agrees_10279 : modelRow 10279 = row 10279 := model_count_10279.trans row_count_10279.symm
def row_10280 : RowCertificate 10280 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10280 1) := by change ¬ Nat.Prime 422734161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140911387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10280 2) := by change ¬ Nat.Prime 422713601; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 395429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10280 3) := by change ¬ Nat.Prime 422693041; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1350457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10280 : row 10280 = 0 := (row_10280.checked).trans rfl
theorem model_count_10280 : modelRow 10280 = 0 := by decide
theorem modelrow_agrees_10280 : modelRow 10280 = row 10280 := model_count_10280.trans row_count_10280.symm
def row_10281 : RowCertificate 10281 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10281 1) := by change ¬ Nat.Prime 422816407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32524339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10281 2) := by change ¬ Nat.Prime 422795845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84559169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10281 3) := by change ¬ Nat.Prime 422775283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60396469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10281 : row 10281 = 0 := (row_10281.checked).trans rfl
theorem model_count_10281 : modelRow 10281 = 0 := by decide
theorem modelrow_agrees_10281 : modelRow 10281 = row 10281 := model_count_10281.trans row_count_10281.symm
def row_10282 : RowCertificate 10282 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10282 1) := by change ¬ Nat.Prime 422898661; exact PrattCertificates.composite_of_product (factor := 8629) (cofactor := 49009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10282 2) := by change Nat.Prime 422878097; exact ProjectEulerSolutions.P58PrimeNodes.p_422878097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10282 3) := by change ¬ Nat.Prime 422857533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140952511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10282 : row 10282 = 1 := (row_10282.checked).trans rfl
theorem model_count_10282 : modelRow 10282 = 1 := by decide
theorem modelrow_agrees_10282 : modelRow 10282 = row 10282 := model_count_10282.trans row_count_10282.symm
def row_10283 : RowCertificate 10283 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10283 1) := by change ¬ Nat.Prime 422980923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 140993641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10283 2) := by change ¬ Nat.Prime 422960357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24880021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10283 3) := by change ¬ Nat.Prime 422939791; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22259989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10283 : row 10283 = 0 := (row_10283.checked).trans rfl
theorem model_count_10283 : modelRow 10283 = 0 := by decide
theorem modelrow_agrees_10283 : modelRow 10283 = row 10283 := model_count_10283.trans row_count_10283.symm
def row_10284 : RowCertificate 10284 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10284 1) := by change ¬ Nat.Prime 423063193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60437599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10284 2) := by change ¬ Nat.Prime 423042625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84608525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10284 3) := by change ¬ Nat.Prime 423022057; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 1560967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10284 : row 10284 = 0 := (row_10284.checked).trans rfl
theorem model_count_10284 : modelRow 10284 = 0 := by decide
theorem modelrow_agrees_10284 : modelRow 10284 = row 10284 := model_count_10284.trans row_count_10284.symm
def row_10285 : RowCertificate 10285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10285 1) := by change ¬ Nat.Prime 423145471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60449353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10285 2) := by change ¬ Nat.Prime 423124901; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2192357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10285 3) := by change ¬ Nat.Prime 423104331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141034777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10285 : row 10285 = 0 := (row_10285.checked).trans rfl
theorem model_count_10285 : modelRow 10285 = 0 := by decide
theorem modelrow_agrees_10285 : modelRow 10285 = row 10285 := model_count_10285.trans row_count_10285.symm
def row_10286 : RowCertificate 10286 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10286 1) := by change ¬ Nat.Prime 423227757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141075919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10286 2) := by change ¬ Nat.Prime 423207185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84641437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10286 3) := by change ¬ Nat.Prime 423186613; exact PrattCertificates.composite_of_product (factor := 15511) (cofactor := 27283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10286 : row 10286 = 0 := (row_10286.checked).trans rfl
theorem model_count_10286 : modelRow 10286 = 0 := by decide
theorem modelrow_agrees_10286 : modelRow 10286 = row 10286 := model_count_10286.trans row_count_10286.symm
def row_10287 : RowCertificate 10287 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10287 1) := by change Nat.Prime 423310051; exact ProjectEulerSolutions.P58PrimeNodes.p_423310051.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10287 2) := by change ¬ Nat.Prime 423289477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32560729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10287 3) := by change Nat.Prime 423268903; exact ProjectEulerSolutions.P58PrimeNodes.p_423268903.prime
    exact decide_eq_true prime
theorem row_count_10287 : row 10287 = 2 := (row_10287.checked).trans rfl
theorem model_count_10287 : modelRow 10287 = 2 := by decide
theorem modelrow_agrees_10287 : modelRow 10287 = row 10287 := model_count_10287.trans row_count_10287.symm
def row_10288 : RowCertificate 10288 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10288 1) := by change Nat.Prime 423392353; exact ProjectEulerSolutions.P58PrimeNodes.p_423392353.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10288 2) := by change ¬ Nat.Prime 423371777; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 1647361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10288 3) := by change ¬ Nat.Prime 423351201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141117067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10288 : row 10288 = 1 := (row_10288.checked).trans rfl
theorem model_count_10288 : modelRow 10288 = 1 := by decide
theorem modelrow_agrees_10288 : modelRow 10288 = row 10288 := model_count_10288.trans row_count_10288.symm
def row_10289 : RowCertificate 10289 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10289 1) := by change ¬ Nat.Prime 423474663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141158221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10289 2) := by change ¬ Nat.Prime 423454085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84690817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10289 3) := by change ¬ Nat.Prime 423433507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60490501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10289 : row 10289 = 0 := (row_10289.checked).trans rfl
theorem model_count_10289 : modelRow 10289 = 0 := by decide
theorem modelrow_agrees_10289 : modelRow 10289 = row 10289 := model_count_10289.trans row_count_10289.symm
def row_10290 : RowCertificate 10290 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10290 1) := by change Nat.Prime 423556981; exact ProjectEulerSolutions.P58PrimeNodes.p_423556981.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10290 2) := by change ¬ Nat.Prime 423536401; exact PrattCertificates.composite_of_product (factor := 2357) (cofactor := 179693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10290 3) := by change ¬ Nat.Prime 423515821; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2697553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10290 : row 10290 = 1 := (row_10290.checked).trans rfl
theorem model_count_10290 : modelRow 10290 = 1 := by decide
theorem modelrow_agrees_10290 : modelRow 10290 = row 10290 := model_count_10290.trans row_count_10290.symm
def row_10291 : RowCertificate 10291 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10291 1) := by change ¬ Nat.Prime 423639307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60519901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10291 2) := by change ¬ Nat.Prime 423618725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84723745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10291 3) := by change ¬ Nat.Prime 423598143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141199381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10291 : row 10291 = 0 := (row_10291.checked).trans rfl
theorem model_count_10291 : modelRow 10291 = 0 := by decide
theorem modelrow_agrees_10291 : modelRow 10291 = row 10291 := model_count_10291.trans row_count_10291.symm
def row_10292 : RowCertificate 10292 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10292 1) := by change ¬ Nat.Prime 423721641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141240547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10292 2) := by change ¬ Nat.Prime 423701057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32592389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10292 3) := by change Nat.Prime 423680473; exact ProjectEulerSolutions.P58PrimeNodes.p_423680473.prime
    exact decide_eq_true prime
theorem row_count_10292 : row 10292 = 1 := (row_10292.checked).trans rfl
theorem model_count_10292 : modelRow 10292 = 1 := by decide
theorem modelrow_agrees_10292 : modelRow 10292 = row 10292 := model_count_10292.trans row_count_10292.symm
def row_10293 : RowCertificate 10293 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10293 1) := by change ¬ Nat.Prime 423803983; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 1380469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10293 2) := by change ¬ Nat.Prime 423783397; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4368901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10293 3) := by change ¬ Nat.Prime 423762811; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2341231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10293 : row 10293 = 0 := (row_10293.checked).trans rfl
theorem model_count_10293 : modelRow 10293 = 0 := by decide
theorem modelrow_agrees_10293 : modelRow 10293 = row 10293 := model_count_10293.trans row_count_10293.symm
def row_10294 : RowCertificate 10294 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10294 1) := by change ¬ Nat.Prime 423886333; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32606641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10294 2) := by change ¬ Nat.Prime 423865745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84773149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10294 3) := by change ¬ Nat.Prime 423845157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141281719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10294 : row 10294 = 0 := (row_10294.checked).trans rfl
theorem model_count_10294 : modelRow 10294 = 0 := by decide
theorem modelrow_agrees_10294 : modelRow 10294 = row 10294 := model_count_10294.trans row_count_10294.symm
def row_10295 : RowCertificate 10295 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10295 1) := by change ¬ Nat.Prime 423968691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141322897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10295 2) := by change Nat.Prime 423948101; exact ProjectEulerSolutions.P58PrimeNodes.p_423948101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10295 3) := by change ¬ Nat.Prime 423927511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60561073) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10295 : row 10295 = 1 := (row_10295.checked).trans rfl
theorem model_count_10295 : modelRow 10295 = 1 := by decide
theorem modelrow_agrees_10295 : modelRow 10295 = row 10295 := model_count_10295.trans row_count_10295.symm
def row_10296 : RowCertificate 10296 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10296 1) := by change ¬ Nat.Prime 424051057; exact PrattCertificates.composite_of_product (factor := 1747) (cofactor := 242731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10296 2) := by change ¬ Nat.Prime 424030465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84806093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10296 3) := by change ¬ Nat.Prime 424009873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60572839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10296 : row 10296 = 0 := (row_10296.checked).trans rfl
theorem model_count_10296 : modelRow 10296 = 0 := by decide
theorem modelrow_agrees_10296 : modelRow 10296 = row 10296 := model_count_10296.trans row_count_10296.symm
def row_10297 : RowCertificate 10297 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10297 1) := by change ¬ Nat.Prime 424133431; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5810047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10297 2) := by change ¬ Nat.Prime 424112837; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 8002129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10297 3) := by change ¬ Nat.Prime 424092243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141364081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10297 : row 10297 = 0 := (row_10297.checked).trans rfl
theorem model_count_10297 : modelRow 10297 = 0 := by decide
theorem modelrow_agrees_10297 : modelRow 10297 = row 10297 := model_count_10297.trans row_count_10297.symm
def row_10298 : RowCertificate 10298 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10298 1) := by change ¬ Nat.Prime 424215813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141405271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10298 2) := by change Nat.Prime 424195217; exact ProjectEulerSolutions.P58PrimeNodes.p_424195217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10298 3) := by change ¬ Nat.Prime 424174621; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32628817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10298 : row 10298 = 1 := (row_10298.checked).trans rfl
theorem model_count_10298 : modelRow 10298 = 1 := by decide
theorem modelrow_agrees_10298 : modelRow 10298 = row 10298 := model_count_10298.trans row_count_10298.symm
def row_10299 : RowCertificate 10299 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10299 1) := by change ¬ Nat.Prime 424298203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60614029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10299 2) := by change ¬ Nat.Prime 424277605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84855521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10299 3) := by change ¬ Nat.Prime 424257007; exact PrattCertificates.composite_of_product (factor := 17761) (cofactor := 23887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10299 : row 10299 = 0 := (row_10299.checked).trans rfl
theorem model_count_10299 : modelRow 10299 = 0 := by decide
theorem modelrow_agrees_10299 : modelRow 10299 = row 10299 := model_count_10299.trans row_count_10299.symm
def row_10300 : RowCertificate 10300 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10300 1) := by change ¬ Nat.Prime 424380601; exact PrattCertificates.composite_of_product (factor := 5233) (cofactor := 81097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10300 2) := by change ¬ Nat.Prime 424360001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32643077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10300 3) := by change ¬ Nat.Prime 424339401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141446467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10300 : row 10300 = 0 := (row_10300.checked).trans rfl
theorem model_count_10300 : modelRow 10300 = 0 := by decide
theorem modelrow_agrees_10300 : modelRow 10300 = row 10300 := model_count_10300.trans row_count_10300.symm
def row_10301 : RowCertificate 10301 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10301 1) := by change ¬ Nat.Prime 424463007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141487669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10301 2) := by change ¬ Nat.Prime 424442405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84888481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10301 3) := by change ¬ Nat.Prime 424421803; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32647831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10301 : row 10301 = 0 := (row_10301.checked).trans rfl
theorem model_count_10301 : modelRow 10301 = 0 := by decide
theorem modelrow_agrees_10301 : modelRow 10301 = row 10301 := model_count_10301.trans row_count_10301.symm
def row_10302 : RowCertificate 10302 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10302 1) := by change ¬ Nat.Prime 424545421; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6959761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10302 2) := by change ¬ Nat.Prime 424524817; exact PrattCertificates.composite_of_product (factor := 1433) (cofactor := 296249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10302 3) := by change ¬ Nat.Prime 424504213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60643459) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10302 : row 10302 = 0 := (row_10302.checked).trans rfl
theorem model_count_10302 : modelRow 10302 = 0 := by decide
theorem modelrow_agrees_10302 : modelRow 10302 = row 10302 := model_count_10302.trans row_count_10302.symm
def row_10303 : RowCertificate 10303 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10303 1) := by change ¬ Nat.Prime 424627843; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 6337729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10303 2) := by change ¬ Nat.Prime 424607237; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2849713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10303 3) := by change ¬ Nat.Prime 424586631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141528877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10303 : row 10303 = 0 := (row_10303.checked).trans rfl
theorem model_count_10303 : modelRow 10303 = 0 := by decide
theorem modelrow_agrees_10303 : modelRow 10303 = row 10303 := model_count_10303.trans row_count_10303.symm
def row_10304 : RowCertificate 10304 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10304 1) := by change ¬ Nat.Prime 424710273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141570091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10304 2) := by change ¬ Nat.Prime 424689665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84937933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10304 3) := by change ¬ Nat.Prime 424669057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22351003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10304 : row 10304 = 0 := (row_10304.checked).trans rfl
theorem model_count_10304 : modelRow 10304 = 0 := by decide
theorem modelrow_agrees_10304 : modelRow 10304 = row 10304 := model_count_10304.trans row_count_10304.symm
def row_10305 : RowCertificate 10305 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10305 1) := by change ¬ Nat.Prime 424792711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60684673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10305 2) := by change ¬ Nat.Prime 424772101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32674777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10305 3) := by change ¬ Nat.Prime 424751491; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13701661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10305 : row 10305 = 0 := (row_10305.checked).trans rfl
theorem model_count_10305 : modelRow 10305 = 0 := by decide
theorem modelrow_agrees_10305 : modelRow 10305 = row 10305 := model_count_10305.trans row_count_10305.symm
def row_10306 : RowCertificate 10306 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10306 1) := by change ¬ Nat.Prime 424875157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60696451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10306 2) := by change ¬ Nat.Prime 424854545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 84970909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10306 3) := by change ¬ Nat.Prime 424833933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141611311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10306 : row 10306 = 0 := (row_10306.checked).trans rfl
theorem model_count_10306 : modelRow 10306 = 0 := by decide
theorem modelrow_agrees_10306 : modelRow 10306 = row 10306 := model_count_10306.trans row_count_10306.symm
def row_10307 : RowCertificate 10307 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10307 1) := by change ¬ Nat.Prime 424957611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141652537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10307 2) := by change ¬ Nat.Prime 424936997; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10364317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10307 3) := by change Nat.Prime 424916383; exact ProjectEulerSolutions.P58PrimeNodes.p_424916383.prime
    exact decide_eq_true prime
theorem row_count_10307 : row 10307 = 1 := (row_10307.checked).trans rfl
theorem model_count_10307 : modelRow 10307 = 1 := by decide
theorem modelrow_agrees_10307 : modelRow 10307 = row 10307 := model_count_10307.trans row_count_10307.symm
def row_10308 : RowCertificate 10308 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10308 1) := by change Nat.Prime 425040073; exact ProjectEulerSolutions.P58PrimeNodes.p_425040073.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10308 2) := by change Nat.Prime 425019457; exact ProjectEulerSolutions.P58PrimeNodes.p_425019457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10308 3) := by change ¬ Nat.Prime 424998841; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 388837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10308 : row 10308 = 2 := (row_10308.checked).trans rfl
theorem model_count_10308 : modelRow 10308 = 2 := by decide
theorem modelrow_agrees_10308 : modelRow 10308 = row 10308 := model_count_10308.trans row_count_10308.symm
def row_10309 : RowCertificate 10309 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10309 1) := by change ¬ Nat.Prime 425122543; exact PrattCertificates.composite_of_product (factor := 2539) (cofactor := 167437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10309 2) := by change ¬ Nat.Prime 425101925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85020385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10309 3) := by change ¬ Nat.Prime 425081307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141693769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10309 : row 10309 = 0 := (row_10309.checked).trans rfl
theorem model_count_10309 : modelRow 10309 = 0 := by decide
theorem modelrow_agrees_10309 : modelRow 10309 = row 10309 := model_count_10309.trans row_count_10309.symm
def row_10310 : RowCertificate 10310 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10310 1) := by change ¬ Nat.Prime 425205021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141735007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10310 2) := by change Nat.Prime 425184401; exact ProjectEulerSolutions.P58PrimeNodes.p_425184401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10310 3) := by change ¬ Nat.Prime 425163781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60737683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10310 : row 10310 = 1 := (row_10310.checked).trans rfl
theorem model_count_10310 : modelRow 10310 = 1 := by decide
theorem modelrow_agrees_10310 : modelRow 10310 = row 10310 := model_count_10310.trans row_count_10310.symm
def row_10311 : RowCertificate 10311 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10311 1) := by change ¬ Nat.Prime 425287507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22383553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10311 2) := by change ¬ Nat.Prime 425266885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85053377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10311 3) := by change ¬ Nat.Prime 425246263; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32711251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10311 : row 10311 = 0 := (row_10311.checked).trans rfl
theorem model_count_10311 : modelRow 10311 = 0 := by decide
theorem modelrow_agrees_10311 : modelRow 10311 = row 10311 := model_count_10311.trans row_count_10311.symm
def row_10312 : RowCertificate 10312 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10312 1) := by change ¬ Nat.Prime 425370001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60767143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10312 2) := by change Nat.Prime 425349377; exact ProjectEulerSolutions.P58PrimeNodes.p_425349377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10312 3) := by change ¬ Nat.Prime 425328753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141776251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10312 : row 10312 = 1 := (row_10312.checked).trans rfl
theorem model_count_10312 : modelRow 10312 = 1 := by decide
theorem modelrow_agrees_10312 : modelRow 10312 = row 10312 := model_count_10312.trans row_count_10312.symm
def row_10313 : RowCertificate 10313 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10313 1) := by change ¬ Nat.Prime 425452503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141817501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10313 2) := by change ¬ Nat.Prime 425431877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32725529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10313 3) := by change ¬ Nat.Prime 425411251; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4385683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10313 : row 10313 = 0 := (row_10313.checked).trans rfl
theorem model_count_10313 : modelRow 10313 = 0 := by decide
theorem modelrow_agrees_10313 : modelRow 10313 = row 10313 := model_count_10313.trans row_count_10313.symm
def row_10314 : RowCertificate 10314 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10314 1) := by change Nat.Prime 425535013; exact ProjectEulerSolutions.P58PrimeNodes.p_425535013.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10314 2) := by change ¬ Nat.Prime 425514385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85102877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10314 3) := by change ¬ Nat.Prime 425493757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32730289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10314 : row 10314 = 1 := (row_10314.checked).trans rfl
theorem model_count_10314 : modelRow 10314 = 1 := by decide
theorem modelrow_agrees_10314 : modelRow 10314 = row 10314 := model_count_10314.trans row_count_10314.symm
def row_10315 : RowCertificate 10315 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10315 1) := by change Nat.Prime 425617531; exact ProjectEulerSolutions.P58PrimeNodes.p_425617531.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10315 2) := by change Nat.Prime 425596901; exact ProjectEulerSolutions.P58PrimeNodes.p_425596901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10315 3) := by change ¬ Nat.Prime 425576271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141858757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10315 : row 10315 = 2 := (row_10315.checked).trans rfl
theorem model_count_10315 : modelRow 10315 = 2 := by decide
theorem modelrow_agrees_10315 : modelRow 10315 = row 10315 := model_count_10315.trans row_count_10315.symm
def row_10316 : RowCertificate 10316 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10316 1) := by change ¬ Nat.Prime 425700057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141900019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10316 2) := by change ¬ Nat.Prime 425679425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85135885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10316 3) := by change ¬ Nat.Prime 425658793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60808399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10316 : row 10316 = 0 := (row_10316.checked).trans rfl
theorem model_count_10316 : modelRow 10316 = 0 := by decide
theorem modelrow_agrees_10316 : modelRow 10316 = row 10316 := model_count_10316.trans row_count_10316.symm
def row_10317 : RowCertificate 10317 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10317 1) := by change ¬ Nat.Prime 425782591; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32752507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10317 2) := by change ¬ Nat.Prime 425761957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 25044821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10317 3) := by change ¬ Nat.Prime 425741323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60820189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10317 : row 10317 = 0 := (row_10317.checked).trans rfl
theorem model_count_10317 : modelRow 10317 = 0 := by decide
theorem modelrow_agrees_10317 : modelRow 10317 = row 10317 := model_count_10317.trans row_count_10317.symm
def row_10318 : RowCertificate 10318 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10318 1) := by change Nat.Prime 425865133; exact ProjectEulerSolutions.P58PrimeNodes.p_425865133.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10318 2) := by change ¬ Nat.Prime 425844497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32757269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10318 3) := by change ¬ Nat.Prime 425823861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141941287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10318 : row 10318 = 1 := (row_10318.checked).trans rfl
theorem model_count_10318 : modelRow 10318 = 1 := by decide
theorem modelrow_agrees_10318 : modelRow 10318 = row 10318 := model_count_10318.trans row_count_10318.symm
def row_10319 : RowCertificate 10319 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10319 1) := by change ¬ Nat.Prime 425947683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 141982561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10319 2) := by change ¬ Nat.Prime 425927045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85185409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10319 3) := by change ¬ Nat.Prime 425906407; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 585841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10319 : row 10319 = 0 := (row_10319.checked).trans rfl
theorem model_count_10319 : modelRow 10319 = 0 := by decide
theorem modelrow_agrees_10319 : modelRow 10319 = row 10319 := model_count_10319.trans row_count_10319.symm
def row_10320 : RowCertificate 10320 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10320 1) := by change ¬ Nat.Prime 426030241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60861463) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10320 2) := by change ¬ Nat.Prime 426009601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11513773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10320 3) := by change ¬ Nat.Prime 425988961; exact PrattCertificates.composite_of_product (factor := 1039) (cofactor := 409999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10320 : row 10320 = 0 := (row_10320.checked).trans rfl
theorem model_count_10320 : modelRow 10320 = 0 := by decide
theorem modelrow_agrees_10320 : modelRow 10320 = row 10320 := model_count_10320.trans row_count_10320.symm
def row_10321 : RowCertificate 10321 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10321 1) := by change ¬ Nat.Prime 426112807; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 5393833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10321 2) := by change ¬ Nat.Prime 426092165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85218433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10321 3) := by change ¬ Nat.Prime 426071523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142023841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10321 : row 10321 = 0 := (row_10321.checked).trans rfl
theorem model_count_10321 : modelRow 10321 = 0 := by decide
theorem modelrow_agrees_10321 : modelRow 10321 = row 10321 := model_count_10321.trans row_count_10321.symm
def row_10322 : RowCertificate 10322 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10322 1) := by change ¬ Nat.Prime 426195381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142065127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10322 2) := by change Nat.Prime 426174737; exact ProjectEulerSolutions.P58PrimeNodes.p_426174737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10322 3) := by change ¬ Nat.Prime 426154093; exact PrattCertificates.composite_of_product (factor := 16633) (cofactor := 25621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10322 : row 10322 = 1 := (row_10322.checked).trans rfl
theorem model_count_10322 : modelRow 10322 = 1 := by decide
theorem modelrow_agrees_10322 : modelRow 10322 = row 10322 := model_count_10322.trans row_count_10322.symm
def row_10323 : RowCertificate 10323 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10323 1) := by change ¬ Nat.Prime 426277963; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9913441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10323 2) := by change ¬ Nat.Prime 426257317; exact PrattCertificates.composite_of_product (factor := 9397) (cofactor := 45361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10323 3) := by change ¬ Nat.Prime 426236671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60890953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10323 : row 10323 = 0 := (row_10323.checked).trans rfl
theorem model_count_10323 : modelRow 10323 = 0 := by decide
theorem modelrow_agrees_10323 : modelRow 10323 = row 10323 := model_count_10323.trans row_count_10323.symm
def row_10324 : RowCertificate 10324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10324 1) := by change ¬ Nat.Prime 426360553; exact PrattCertificates.composite_of_product (factor := 7537) (cofactor := 56569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10324 2) := by change ¬ Nat.Prime 426339905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85267981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10324 3) := by change ¬ Nat.Prime 426319257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142106419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10324 : row 10324 = 0 := (row_10324.checked).trans rfl
theorem model_count_10324 : modelRow 10324 = 0 := by decide
theorem modelrow_agrees_10324 : modelRow 10324 = row 10324 := model_count_10324.trans row_count_10324.symm
def row_10325 : RowCertificate 10325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10325 1) := by change ¬ Nat.Prime 426443151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142147717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10325 2) := by change ¬ Nat.Prime 426422501; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 3112573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10325 3) := by change ¬ Nat.Prime 426401851; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 1012831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10325 : row 10325 = 0 := (row_10325.checked).trans rfl
theorem model_count_10325 : modelRow 10325 = 0 := by decide
theorem modelrow_agrees_10325 : modelRow 10325 = row 10325 := model_count_10325.trans row_count_10325.symm
def row_10326 : RowCertificate 10326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10326 1) := by change ¬ Nat.Prime 426525757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60932251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10326 2) := by change ¬ Nat.Prime 426505105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85301021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10326 3) := by change ¬ Nat.Prime 426484453; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13757563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10326 : row 10326 = 0 := (row_10326.checked).trans rfl
theorem model_count_10326 : modelRow 10326 = 0 := by decide
theorem modelrow_agrees_10326 : modelRow 10326 = row 10326 := model_count_10326.trans row_count_10326.symm
def row_10327 : RowCertificate 10327 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10327 1) := by change ¬ Nat.Prime 426608371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60944053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10327 2) := by change Nat.Prime 426587717; exact ProjectEulerSolutions.P58PrimeNodes.p_426587717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10327 3) := by change ¬ Nat.Prime 426567063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142189021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10327 : row 10327 = 1 := (row_10327.checked).trans rfl
theorem model_count_10327 : modelRow 10327 = 1 := by decide
theorem modelrow_agrees_10327 : modelRow 10327 = row 10327 := model_count_10327.trans row_count_10327.symm
def row_10328 : RowCertificate 10328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10328 1) := by change ¬ Nat.Prime 426690993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142230331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10328 2) := by change Nat.Prime 426670337; exact ProjectEulerSolutions.P58PrimeNodes.p_426670337.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10328 3) := by change Nat.Prime 426649681; exact ProjectEulerSolutions.P58PrimeNodes.p_426649681.prime
    exact decide_eq_true prime
theorem row_count_10328 : row 10328 = 2 := (row_10328.checked).trans rfl
theorem model_count_10328 : modelRow 10328 = 2 := by decide
theorem modelrow_agrees_10328 : modelRow 10328 = row 10328 := model_count_10328.trans row_count_10328.symm
def row_10329 : RowCertificate 10329 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10329 1) := by change ¬ Nat.Prime 426773623; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1540699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10329 2) := by change ¬ Nat.Prime 426752965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85350593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10329 3) := by change ¬ Nat.Prime 426732307; exact PrattCertificates.composite_of_product (factor := 1171) (cofactor := 364417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10329 : row 10329 = 0 := (row_10329.checked).trans rfl
theorem model_count_10329 : modelRow 10329 = 0 := by decide
theorem modelrow_agrees_10329 : modelRow 10329 = row 10329 := model_count_10329.trans row_count_10329.symm
def row_10330 : RowCertificate 10330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10330 1) := by change ¬ Nat.Prime 426856261; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32835097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10330 2) := by change ¬ Nat.Prime 426835601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14718469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10330 3) := by change ¬ Nat.Prime 426814941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142271647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10330 : row 10330 = 0 := (row_10330.checked).trans rfl
theorem model_count_10330 : modelRow 10330 = 0 := by decide
theorem modelrow_agrees_10330 : modelRow 10330 = row 10330 := model_count_10330.trans row_count_10330.symm
def row_10331 : RowCertificate 10331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10331 1) := by change ¬ Nat.Prime 426938907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142312969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10331 2) := by change ¬ Nat.Prime 426918245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85383649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10331 3) := by change ¬ Nat.Prime 426897583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 60985369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10331 : row 10331 = 0 := (row_10331.checked).trans rfl
theorem model_count_10331 : modelRow 10331 = 0 := by decide
theorem modelrow_agrees_10331 : modelRow 10331 = row 10331 := model_count_10331.trans row_count_10331.symm
def row_10332 : RowCertificate 10332 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10332 1) := by change ¬ Nat.Prime 427021561; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22474819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10332 2) := by change ¬ Nat.Prime 427000897; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3778769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10332 3) := by change ¬ Nat.Prime 426980233; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 2023603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10332 : row 10332 = 0 := (row_10332.checked).trans rfl
theorem model_count_10332 : modelRow 10332 = 0 := by decide
theorem modelrow_agrees_10332 : modelRow 10332 = row 10332 := model_count_10332.trans row_count_10332.symm
def row_10333 : RowCertificate 10333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10333 1) := by change ¬ Nat.Prime 427104223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61014889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10333 2) := by change ¬ Nat.Prime 427083557; exact PrattCertificates.composite_of_product (factor := 641) (cofactor := 666277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10333 3) := by change ¬ Nat.Prime 427062891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142354297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10333 : row 10333 = 0 := (row_10333.checked).trans rfl
theorem model_count_10333 : modelRow 10333 = 0 := by decide
theorem modelrow_agrees_10333 : modelRow 10333 = row 10333 := model_count_10333.trans row_count_10333.symm
def row_10334 : RowCertificate 10334 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10334 1) := by change ¬ Nat.Prime 427186893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142395631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10334 2) := by change ¬ Nat.Prime 427166225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85433245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10334 3) := by change ¬ Nat.Prime 427145557; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5851309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10334 : row 10334 = 0 := (row_10334.checked).trans rfl
theorem model_count_10334 : modelRow 10334 = 0 := by decide
theorem modelrow_agrees_10334 : modelRow 10334 = row 10334 := model_count_10334.trans row_count_10334.symm
def row_10335 : RowCertificate 10335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10335 1) := by change ¬ Nat.Prime 427269571; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 298999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10335 2) := by change Nat.Prime 427248901; exact ProjectEulerSolutions.P58PrimeNodes.p_427248901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10335 3) := by change Nat.Prime 427228231; exact ProjectEulerSolutions.P58PrimeNodes.p_427228231.prime
    exact decide_eq_true prime
theorem row_count_10335 : row 10335 = 2 := (row_10335.checked).trans rfl
theorem model_count_10335 : modelRow 10335 = 2 := by decide
theorem modelrow_agrees_10335 : modelRow 10335 = row 10335 := model_count_10335.trans row_count_10335.symm
def row_10336 : RowCertificate 10336 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10336 1) := by change ¬ Nat.Prime 427352257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11550061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10336 2) := by change ¬ Nat.Prime 427331585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85466317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10336 3) := by change ¬ Nat.Prime 427310913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142436971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10336 : row 10336 = 0 := (row_10336.checked).trans rfl
theorem model_count_10336 : modelRow 10336 = 0 := by decide
theorem modelrow_agrees_10336 : modelRow 10336 = row 10336 := model_count_10336.trans row_count_10336.symm
def row_10337 : RowCertificate 10337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10337 1) := by change ¬ Nat.Prime 427434951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142478317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10337 2) := by change ¬ Nat.Prime 427414277; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 935261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10337 3) := by change ¬ Nat.Prime 427393603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61056229) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10337 : row 10337 = 0 := (row_10337.checked).trans rfl
theorem model_count_10337 : modelRow 10337 = 0 := by decide
theorem modelrow_agrees_10337 : modelRow 10337 = row 10337 := model_count_10337.trans row_count_10337.symm
def row_10338 : RowCertificate 10338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10338 1) := by change ¬ Nat.Prime 427517653; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9942271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10338 2) := by change ¬ Nat.Prime 427496977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 25146881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10338 3) := by change ¬ Nat.Prime 427476301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61068043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10338 : row 10338 = 0 := (row_10338.checked).trans rfl
theorem model_count_10338 : modelRow 10338 = 0 := by decide
theorem modelrow_agrees_10338 : modelRow 10338 = row 10338 := model_count_10338.trans row_count_10338.symm
def row_10339 : RowCertificate 10339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10339 1) := by change ¬ Nat.Prime 427600363; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 1077079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10339 2) := by change ¬ Nat.Prime 427579685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85515937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10339 3) := by change ¬ Nat.Prime 427559007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142519669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10339 : row 10339 = 0 := (row_10339.checked).trans rfl
theorem model_count_10339 : modelRow 10339 = 0 := by decide
theorem modelrow_agrees_10339 : modelRow 10339 = row 10339 := model_count_10339.trans row_count_10339.symm
def row_10340 : RowCertificate 10340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10340 1) := by change ¬ Nat.Prime 427683081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142561027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10340 2) := by change ¬ Nat.Prime 427662401; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 2472037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10340 3) := by change ¬ Nat.Prime 427641721; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32895517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10340 : row 10340 = 0 := (row_10340.checked).trans rfl
theorem model_count_10340 : modelRow 10340 = 0 := by decide
theorem modelrow_agrees_10340 : modelRow 10340 = row 10340 := model_count_10340.trans row_count_10340.symm
def row_10341 : RowCertificate 10341 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10341 1) := by change ¬ Nat.Prime 427765807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61109401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10341 2) := by change ¬ Nat.Prime 427745125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85549025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10341 3) := by change ¬ Nat.Prime 427724443; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3367909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10341 : row 10341 = 0 := (row_10341.checked).trans rfl
theorem model_count_10341 : modelRow 10341 = 0 := by decide
theorem modelrow_agrees_10341 : modelRow 10341 = row 10341 := model_count_10341.trans row_count_10341.symm
def row_10342 : RowCertificate 10342 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10342 1) := by change Nat.Prime 427848541; exact ProjectEulerSolutions.P58PrimeNodes.p_427848541.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10342 2) := by change Nat.Prime 427827857; exact ProjectEulerSolutions.P58PrimeNodes.p_427827857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10342 3) := by change ¬ Nat.Prime 427807173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142602391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10342 : row 10342 = 2 := (row_10342.checked).trans rfl
theorem model_count_10342 : modelRow 10342 = 2 := by decide
theorem modelrow_agrees_10342 : modelRow 10342 = row 10342 := model_count_10342.trans row_count_10342.symm
def row_10343 : RowCertificate 10343 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10343 1) := by change ¬ Nat.Prime 427931283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142643761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10343 2) := by change ¬ Nat.Prime 427910597; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5861789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10343 3) := by change ¬ Nat.Prime 427889911; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 1269703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10343 : row 10343 = 0 := (row_10343.checked).trans rfl
theorem model_count_10343 : modelRow 10343 = 0 := by decide
theorem modelrow_agrees_10343 : modelRow 10343 = row 10343 := model_count_10343.trans row_count_10343.symm
def row_10344 : RowCertificate 10344 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10344 1) := by change Nat.Prime 428014033; exact ProjectEulerSolutions.P58PrimeNodes.p_428014033.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10344 2) := by change ¬ Nat.Prime 427993345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85598669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10344 3) := by change ¬ Nat.Prime 427972657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61138951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10344 : row 10344 = 1 := (row_10344.checked).trans rfl
theorem model_count_10344 : modelRow 10344 = 1 := by decide
theorem modelrow_agrees_10344 : modelRow 10344 = row 10344 := model_count_10344.trans row_count_10344.symm
def row_10345 : RowCertificate 10345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10345 1) := by change ¬ Nat.Prime 428096791; exact PrattCertificates.composite_of_product (factor := 11941) (cofactor := 35851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10345 2) := by change ¬ Nat.Prime 428076101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7017641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10345 3) := by change ¬ Nat.Prime 428055411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142685137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10345 : row 10345 = 0 := (row_10345.checked).trans rfl
theorem model_count_10345 : modelRow 10345 = 0 := by decide
theorem modelrow_agrees_10345 : modelRow 10345 = row 10345 := model_count_10345.trans row_count_10345.symm
def row_10346 : RowCertificate 10346 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10346 1) := by change ¬ Nat.Prime 428179557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142726519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10346 2) := by change ¬ Nat.Prime 428158865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85631773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10346 3) := by change ¬ Nat.Prime 428138173; exact PrattCertificates.composite_of_product (factor := 2113) (cofactor := 202621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10346 : row 10346 = 0 := (row_10346.checked).trans rfl
theorem model_count_10346 : modelRow 10346 = 0 := by decide
theorem modelrow_agrees_10346 : modelRow 10346 = row 10346 := model_count_10346.trans row_count_10346.symm
def row_10347 : RowCertificate 10347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10347 1) := by change ¬ Nat.Prime 428262331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61180333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10347 2) := by change ¬ Nat.Prime 428241637; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14766953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10347 3) := by change ¬ Nat.Prime 428220943; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11573539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10347 : row 10347 = 0 := (row_10347.checked).trans rfl
theorem model_count_10347 : modelRow 10347 = 0 := by decide
theorem modelrow_agrees_10347 : modelRow 10347 = row 10347 := model_count_10347.trans row_count_10347.symm
def row_10348 : RowCertificate 10348 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10348 1) := by change ¬ Nat.Prime 428345113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61192159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10348 2) := by change ¬ Nat.Prime 428324417; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 10446937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10348 3) := by change ¬ Nat.Prime 428303721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142767907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10348 : row 10348 = 0 := (row_10348.checked).trans rfl
theorem model_count_10348 : modelRow 10348 = 0 := by decide
theorem modelrow_agrees_10348 : modelRow 10348 = row 10348 := model_count_10348.trans row_count_10348.symm
def row_10349 : RowCertificate 10349 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10349 1) := by change ¬ Nat.Prime 428427903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142809301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10349 2) := by change ¬ Nat.Prime 428407205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85681441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10349 3) := by change ¬ Nat.Prime 428386507; exact PrattCertificates.composite_of_product (factor := 991) (cofactor := 432277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10349 : row 10349 = 0 := (row_10349.checked).trans rfl
theorem model_count_10349 : modelRow 10349 = 0 := by decide
theorem modelrow_agrees_10349 : modelRow 10349 = row 10349 := model_count_10349.trans row_count_10349.symm
def row_10350 : RowCertificate 10350 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10350 1) := by change Nat.Prime 428510701; exact ProjectEulerSolutions.P58PrimeNodes.p_428510701.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10350 2) := by change ¬ Nat.Prime 428490001; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 8084717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10350 3) := by change ¬ Nat.Prime 428469301; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32959177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10350 : row 10350 = 1 := (row_10350.checked).trans rfl
theorem model_count_10350 : modelRow 10350 = 1 := by decide
theorem modelrow_agrees_10350 : modelRow 10350 = row 10350 := model_count_10350.trans row_count_10350.symm
def row_10351 : RowCertificate 10351 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10351 1) := by change ¬ Nat.Prime 428593507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22557553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10351 2) := by change ¬ Nat.Prime 428572805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85714561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10351 3) := by change ¬ Nat.Prime 428552103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142850701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10351 : row 10351 = 0 := (row_10351.checked).trans rfl
theorem model_count_10351 : modelRow 10351 = 0 := by decide
theorem modelrow_agrees_10351 : modelRow 10351 = row 10351 := model_count_10351.trans row_count_10351.symm
def row_10352 : RowCertificate 10352 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10352 1) := by change ¬ Nat.Prime 428676321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142892107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10352 2) := by change ¬ Nat.Prime 428655617; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32973509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10352 3) := by change ¬ Nat.Prime 428634913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61233559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10352 : row 10352 = 0 := (row_10352.checked).trans rfl
theorem model_count_10352 : modelRow 10352 = 0 := by decide
theorem modelrow_agrees_10352 : modelRow 10352 = row 10352 := model_count_10352.trans row_count_10352.symm
def row_10353 : RowCertificate 10353 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10353 1) := by change Nat.Prime 428759143; exact ProjectEulerSolutions.P58PrimeNodes.p_428759143.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10353 2) := by change Nat.Prime 428738437; exact ProjectEulerSolutions.P58PrimeNodes.p_428738437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10353 3) := by change ¬ Nat.Prime 428717731; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 32978287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10353 : row 10353 = 2 := (row_10353.checked).trans rfl
theorem model_count_10353 : modelRow 10353 = 2 := by decide
theorem modelrow_agrees_10353 : modelRow 10353 = row 10353 := model_count_10353.trans row_count_10353.symm
def row_10354 : RowCertificate 10354 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10354 1) := by change ¬ Nat.Prime 428841973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61263139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10354 2) := by change ¬ Nat.Prime 428821265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85764253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10354 3) := by change ¬ Nat.Prime 428800557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142933519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10354 : row 10354 = 0 := (row_10354.checked).trans rfl
theorem model_count_10354 : modelRow 10354 = 0 := by decide
theorem modelrow_agrees_10354 : modelRow 10354 = row 10354 := model_count_10354.trans row_count_10354.symm
def row_10355 : RowCertificate 10355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10355 1) := by change ¬ Nat.Prime 428924811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 142974937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10355 2) := by change ¬ Nat.Prime 428904101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 25229653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10355 3) := by change ¬ Nat.Prime 428883391; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11591443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10355 : row 10355 = 0 := (row_10355.checked).trans rfl
theorem model_count_10355 : modelRow 10355 = 0 := by decide
theorem modelrow_agrees_10355 : modelRow 10355 = row 10355 := model_count_10355.trans row_count_10355.symm
def row_10356 : RowCertificate 10356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10356 1) := by change ¬ Nat.Prime 429007657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 33000589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10356 2) := by change ¬ Nat.Prime 428986945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85797389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10356 3) := by change ¬ Nat.Prime 428966233; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2732269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10356 : row 10356 = 0 := (row_10356.checked).trans rfl
theorem model_count_10356 : modelRow 10356 = 0 := by decide
theorem modelrow_agrees_10356 : modelRow 10356 = row 10356 := model_count_10356.trans row_count_10356.symm
def row_10357 : RowCertificate 10357 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10357 1) := by change ¬ Nat.Prime 429090511; exact PrattCertificates.composite_of_product (factor := 7603) (cofactor := 56437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10357 2) := by change ¬ Nat.Prime 429069797; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 33005369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10357 3) := by change ¬ Nat.Prime 429049083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143016361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10357 : row 10357 = 0 := (row_10357.checked).trans rfl
theorem model_count_10357 : modelRow 10357 = 0 := by decide
theorem modelrow_agrees_10357 : modelRow 10357 = row 10357 := model_count_10357.trans row_count_10357.symm
def row_10358 : RowCertificate 10358 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10358 1) := by change ¬ Nat.Prime 429173373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143057791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10358 2) := by change ¬ Nat.Prime 429152657; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 1049273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10358 3) := by change ¬ Nat.Prime 429131941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61304563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10358 : row 10358 = 0 := (row_10358.checked).trans rfl
theorem model_count_10358 : modelRow 10358 = 0 := by decide
theorem modelrow_agrees_10358 : modelRow 10358 = row 10358 := model_count_10358.trans row_count_10358.symm
def row_10359 : RowCertificate 10359 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10359 1) := by change ¬ Nat.Prime 429256243; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 33019711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10359 2) := by change ¬ Nat.Prime 429235525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85847105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10359 3) := by change ¬ Nat.Prime 429214807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61316401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10359 : row 10359 = 0 := (row_10359.checked).trans rfl
theorem model_count_10359 : modelRow 10359 = 0 := by decide
theorem modelrow_agrees_10359 : modelRow 10359 = row 10359 := model_count_10359.trans row_count_10359.symm
def row_10360 : RowCertificate 10360 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10360 1) := by change ¬ Nat.Prime 429339121; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3380623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10360 2) := by change ¬ Nat.Prime 429318401; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 425489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10360 3) := by change ¬ Nat.Prime 429297681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143099227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10360 : row 10360 = 0 := (row_10360.checked).trans rfl
theorem model_count_10360 : modelRow 10360 = 0 := by decide
theorem modelrow_agrees_10360 : modelRow 10360 = row 10360 := model_count_10360.trans row_count_10360.symm
def row_10361 : RowCertificate 10361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10361 1) := by change ¬ Nat.Prime 429422007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143140669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10361 2) := by change ¬ Nat.Prime 429401285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85880257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10361 3) := by change ¬ Nat.Prime 429380563; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 22598977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10361 : row 10361 = 0 := (row_10361.checked).trans rfl
theorem model_count_10361 : modelRow 10361 = 0 := by decide
theorem modelrow_agrees_10361 : modelRow 10361 = row 10361 := model_count_10361.trans row_count_10361.symm
def row_10362 : RowCertificate 10362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10362 1) := by change ¬ Nat.Prime 429504901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61357843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10362 2) := by change ¬ Nat.Prime 429484177; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 3134921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10362 3) := by change ¬ Nat.Prime 429463453; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5883061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10362 : row 10362 = 0 := (row_10362.checked).trans rfl
theorem model_count_10362 : modelRow 10362 = 0 := by decide
theorem modelrow_agrees_10362 : modelRow 10362 = row 10362 := model_count_10362.trans row_count_10362.symm
def row_10363 : RowCertificate 10363 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10363 1) := by change ¬ Nat.Prime 429587803; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7042423) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10363 2) := by change ¬ Nat.Prime 429567077; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11609921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10363 3) := by change ¬ Nat.Prime 429546351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143182117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10363 : row 10363 = 0 := (row_10363.checked).trans rfl
theorem model_count_10363 : modelRow 10363 = 0 := by decide
theorem modelrow_agrees_10363 : modelRow 10363 = row 10363 := model_count_10363.trans row_count_10363.symm
def row_10364 : RowCertificate 10364 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10364 1) := by change ¬ Nat.Prime 429670713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143223571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10364 2) := by change ¬ Nat.Prime 429649985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85929997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10364 3) := by change ¬ Nat.Prime 429629257; exact PrattCertificates.composite_of_product (factor := 2053) (cofactor := 209269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10364 : row 10364 = 0 := (row_10364.checked).trans rfl
theorem model_count_10364 : modelRow 10364 = 0 := by decide
theorem modelrow_agrees_10364 : modelRow 10364 = row 10364 := model_count_10364.trans row_count_10364.symm
def row_10365 : RowCertificate 10365 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10365 1) := by change ¬ Nat.Prime 429753631; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 11614963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10365 2) := by change ¬ Nat.Prime 429732901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 33056377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10365 3) := by change ¬ Nat.Prime 429712171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61387453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10365 : row 10365 = 0 := (row_10365.checked).trans rfl
theorem model_count_10365 : modelRow 10365 = 0 := by decide
theorem modelrow_agrees_10365 : modelRow 10365 = row 10365 := model_count_10365.trans row_count_10365.symm
def row_10366 : RowCertificate 10366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10366 1) := by change ¬ Nat.Prime 429836557; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9996199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10366 2) := by change ¬ Nat.Prime 429815825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 85963165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10366 3) := by change ¬ Nat.Prime 429795093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143265031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10366 : row 10366 = 0 := (row_10366.checked).trans rfl
theorem model_count_10366 : modelRow 10366 = 0 := by decide
theorem modelrow_agrees_10366 : modelRow 10366 = row 10366 := model_count_10366.trans row_count_10366.symm
def row_10367 : RowCertificate 10367 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10367 1) := by change ¬ Nat.Prime 429919491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 143306497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10367 2) := by change Nat.Prime 429898757; exact ProjectEulerSolutions.P58PrimeNodes.p_429898757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10367 3) := by change ¬ Nat.Prime 429878023; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13867033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10367 : row 10367 = 1 := (row_10367.checked).trans rfl
theorem model_count_10367 : modelRow 10367 = 1 := by decide
theorem modelrow_agrees_10367 : modelRow 10367 = row 10367 := model_count_10367.trans row_count_10367.symm
def row_10368 : RowCertificate 10368 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10368 1) := by change ¬ Nat.Prime 430002433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 61428919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10368 2) := by change ¬ Nat.Prime 429981697; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 25293041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10368 3) := by change ¬ Nat.Prime 429960961; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2227777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10368 : row 10368 = 0 := (row_10368.checked).trans rfl
theorem model_count_10368 : modelRow 10368 = 0 := by decide
theorem modelrow_agrees_10368 : modelRow 10368 = row 10368 := model_count_10368.trans row_count_10368.symm
private def atomic_10241 : CountSegment 1 10 13120 10240 10241 4224 4224 := CountSegment.single (first := 10240) (initial := 4224) row_count_10241 (by decide)
private theorem atomic_10241_agreement : RowAgreement 10240 10241 := RowAgreement.single (first := 10240) modelrow_agrees_10241
private def atomic_10242 : CountSegment 1 10 13120 10241 10242 4224 4225 := CountSegment.single (first := 10241) (initial := 4224) row_count_10242 (by decide)
private theorem atomic_10242_agreement : RowAgreement 10241 10242 := RowAgreement.single (first := 10241) modelrow_agrees_10242
private def atomic_10243 : CountSegment 1 10 13120 10242 10243 4225 4225 := CountSegment.single (first := 10242) (initial := 4225) row_count_10243 (by decide)
private theorem atomic_10243_agreement : RowAgreement 10242 10243 := RowAgreement.single (first := 10242) modelrow_agrees_10243
private def atomic_10244 : CountSegment 1 10 13120 10243 10244 4225 4225 := CountSegment.single (first := 10243) (initial := 4225) row_count_10244 (by decide)
private theorem atomic_10244_agreement : RowAgreement 10243 10244 := RowAgreement.single (first := 10243) modelrow_agrees_10244
private def atomic_10245 : CountSegment 1 10 13120 10244 10245 4225 4225 := CountSegment.single (first := 10244) (initial := 4225) row_count_10245 (by decide)
private theorem atomic_10245_agreement : RowAgreement 10244 10245 := RowAgreement.single (first := 10244) modelrow_agrees_10245
private def atomic_10246 : CountSegment 1 10 13120 10245 10246 4225 4225 := CountSegment.single (first := 10245) (initial := 4225) row_count_10246 (by decide)
private theorem atomic_10246_agreement : RowAgreement 10245 10246 := RowAgreement.single (first := 10245) modelrow_agrees_10246
private def atomic_10247 : CountSegment 1 10 13120 10246 10247 4225 4225 := CountSegment.single (first := 10246) (initial := 4225) row_count_10247 (by decide)
private theorem atomic_10247_agreement : RowAgreement 10246 10247 := RowAgreement.single (first := 10246) modelrow_agrees_10247
private def atomic_10248 : CountSegment 1 10 13120 10247 10248 4225 4225 := CountSegment.single (first := 10247) (initial := 4225) row_count_10248 (by decide)
private theorem atomic_10248_agreement : RowAgreement 10247 10248 := RowAgreement.single (first := 10247) modelrow_agrees_10248
private def atomic_10249 : CountSegment 1 10 13120 10248 10249 4225 4225 := CountSegment.single (first := 10248) (initial := 4225) row_count_10249 (by decide)
private theorem atomic_10249_agreement : RowAgreement 10248 10249 := RowAgreement.single (first := 10248) modelrow_agrees_10249
private def atomic_10250 : CountSegment 1 10 13120 10249 10250 4225 4227 := CountSegment.single (first := 10249) (initial := 4225) row_count_10250 (by decide)
private theorem atomic_10250_agreement : RowAgreement 10249 10250 := RowAgreement.single (first := 10249) modelrow_agrees_10250
private def atomic_10251 : CountSegment 1 10 13120 10250 10251 4227 4227 := CountSegment.single (first := 10250) (initial := 4227) row_count_10251 (by decide)
private theorem atomic_10251_agreement : RowAgreement 10250 10251 := RowAgreement.single (first := 10250) modelrow_agrees_10251
private def atomic_10252 : CountSegment 1 10 13120 10251 10252 4227 4227 := CountSegment.single (first := 10251) (initial := 4227) row_count_10252 (by decide)
private theorem atomic_10252_agreement : RowAgreement 10251 10252 := RowAgreement.single (first := 10251) modelrow_agrees_10252
private def atomic_10253 : CountSegment 1 10 13120 10252 10253 4227 4227 := CountSegment.single (first := 10252) (initial := 4227) row_count_10253 (by decide)
private theorem atomic_10253_agreement : RowAgreement 10252 10253 := RowAgreement.single (first := 10252) modelrow_agrees_10253
private def atomic_10254 : CountSegment 1 10 13120 10253 10254 4227 4227 := CountSegment.single (first := 10253) (initial := 4227) row_count_10254 (by decide)
private theorem atomic_10254_agreement : RowAgreement 10253 10254 := RowAgreement.single (first := 10253) modelrow_agrees_10254
private def atomic_10255 : CountSegment 1 10 13120 10254 10255 4227 4227 := CountSegment.single (first := 10254) (initial := 4227) row_count_10255 (by decide)
private theorem atomic_10255_agreement : RowAgreement 10254 10255 := RowAgreement.single (first := 10254) modelrow_agrees_10255
private def atomic_10256 : CountSegment 1 10 13120 10255 10256 4227 4227 := CountSegment.single (first := 10255) (initial := 4227) row_count_10256 (by decide)
private theorem atomic_10256_agreement : RowAgreement 10255 10256 := RowAgreement.single (first := 10255) modelrow_agrees_10256
private def atomic_10257 : CountSegment 1 10 13120 10256 10257 4227 4228 := CountSegment.single (first := 10256) (initial := 4227) row_count_10257 (by decide)
private theorem atomic_10257_agreement : RowAgreement 10256 10257 := RowAgreement.single (first := 10256) modelrow_agrees_10257
private def atomic_10258 : CountSegment 1 10 13120 10257 10258 4228 4228 := CountSegment.single (first := 10257) (initial := 4228) row_count_10258 (by decide)
private theorem atomic_10258_agreement : RowAgreement 10257 10258 := RowAgreement.single (first := 10257) modelrow_agrees_10258
private def atomic_10259 : CountSegment 1 10 13120 10258 10259 4228 4228 := CountSegment.single (first := 10258) (initial := 4228) row_count_10259 (by decide)
private theorem atomic_10259_agreement : RowAgreement 10258 10259 := RowAgreement.single (first := 10258) modelrow_agrees_10259
private def atomic_10260 : CountSegment 1 10 13120 10259 10260 4228 4229 := CountSegment.single (first := 10259) (initial := 4228) row_count_10260 (by decide)
private theorem atomic_10260_agreement : RowAgreement 10259 10260 := RowAgreement.single (first := 10259) modelrow_agrees_10260
private def atomic_10261 : CountSegment 1 10 13120 10260 10261 4229 4230 := CountSegment.single (first := 10260) (initial := 4229) row_count_10261 (by decide)
private theorem atomic_10261_agreement : RowAgreement 10260 10261 := RowAgreement.single (first := 10260) modelrow_agrees_10261
private def atomic_10262 : CountSegment 1 10 13120 10261 10262 4230 4230 := CountSegment.single (first := 10261) (initial := 4230) row_count_10262 (by decide)
private theorem atomic_10262_agreement : RowAgreement 10261 10262 := RowAgreement.single (first := 10261) modelrow_agrees_10262
private def atomic_10263 : CountSegment 1 10 13120 10262 10263 4230 4231 := CountSegment.single (first := 10262) (initial := 4230) row_count_10263 (by decide)
private theorem atomic_10263_agreement : RowAgreement 10262 10263 := RowAgreement.single (first := 10262) modelrow_agrees_10263
private def atomic_10264 : CountSegment 1 10 13120 10263 10264 4231 4231 := CountSegment.single (first := 10263) (initial := 4231) row_count_10264 (by decide)
private theorem atomic_10264_agreement : RowAgreement 10263 10264 := RowAgreement.single (first := 10263) modelrow_agrees_10264
private def atomic_10265 : CountSegment 1 10 13120 10264 10265 4231 4231 := CountSegment.single (first := 10264) (initial := 4231) row_count_10265 (by decide)
private theorem atomic_10265_agreement : RowAgreement 10264 10265 := RowAgreement.single (first := 10264) modelrow_agrees_10265
private def atomic_10266 : CountSegment 1 10 13120 10265 10266 4231 4231 := CountSegment.single (first := 10265) (initial := 4231) row_count_10266 (by decide)
private theorem atomic_10266_agreement : RowAgreement 10265 10266 := RowAgreement.single (first := 10265) modelrow_agrees_10266
private def atomic_10267 : CountSegment 1 10 13120 10266 10267 4231 4231 := CountSegment.single (first := 10266) (initial := 4231) row_count_10267 (by decide)
private theorem atomic_10267_agreement : RowAgreement 10266 10267 := RowAgreement.single (first := 10266) modelrow_agrees_10267
private def atomic_10268 : CountSegment 1 10 13120 10267 10268 4231 4232 := CountSegment.single (first := 10267) (initial := 4231) row_count_10268 (by decide)
private theorem atomic_10268_agreement : RowAgreement 10267 10268 := RowAgreement.single (first := 10267) modelrow_agrees_10268
private def atomic_10269 : CountSegment 1 10 13120 10268 10269 4232 4232 := CountSegment.single (first := 10268) (initial := 4232) row_count_10269 (by decide)
private theorem atomic_10269_agreement : RowAgreement 10268 10269 := RowAgreement.single (first := 10268) modelrow_agrees_10269
private def atomic_10270 : CountSegment 1 10 13120 10269 10270 4232 4232 := CountSegment.single (first := 10269) (initial := 4232) row_count_10270 (by decide)
private theorem atomic_10270_agreement : RowAgreement 10269 10270 := RowAgreement.single (first := 10269) modelrow_agrees_10270
private def atomic_10271 : CountSegment 1 10 13120 10270 10271 4232 4233 := CountSegment.single (first := 10270) (initial := 4232) row_count_10271 (by decide)
private theorem atomic_10271_agreement : RowAgreement 10270 10271 := RowAgreement.single (first := 10270) modelrow_agrees_10271
private def atomic_10272 : CountSegment 1 10 13120 10271 10272 4233 4234 := CountSegment.single (first := 10271) (initial := 4233) row_count_10272 (by decide)
private theorem atomic_10272_agreement : RowAgreement 10271 10272 := RowAgreement.single (first := 10271) modelrow_agrees_10272
private def atomic_10273 : CountSegment 1 10 13120 10272 10273 4234 4234 := CountSegment.single (first := 10272) (initial := 4234) row_count_10273 (by decide)
private theorem atomic_10273_agreement : RowAgreement 10272 10273 := RowAgreement.single (first := 10272) modelrow_agrees_10273
private def atomic_10274 : CountSegment 1 10 13120 10273 10274 4234 4234 := CountSegment.single (first := 10273) (initial := 4234) row_count_10274 (by decide)
private theorem atomic_10274_agreement : RowAgreement 10273 10274 := RowAgreement.single (first := 10273) modelrow_agrees_10274
private def atomic_10275 : CountSegment 1 10 13120 10274 10275 4234 4234 := CountSegment.single (first := 10274) (initial := 4234) row_count_10275 (by decide)
private theorem atomic_10275_agreement : RowAgreement 10274 10275 := RowAgreement.single (first := 10274) modelrow_agrees_10275
private def atomic_10276 : CountSegment 1 10 13120 10275 10276 4234 4235 := CountSegment.single (first := 10275) (initial := 4234) row_count_10276 (by decide)
private theorem atomic_10276_agreement : RowAgreement 10275 10276 := RowAgreement.single (first := 10275) modelrow_agrees_10276
private def atomic_10277 : CountSegment 1 10 13120 10276 10277 4235 4235 := CountSegment.single (first := 10276) (initial := 4235) row_count_10277 (by decide)
private theorem atomic_10277_agreement : RowAgreement 10276 10277 := RowAgreement.single (first := 10276) modelrow_agrees_10277
private def atomic_10278 : CountSegment 1 10 13120 10277 10278 4235 4236 := CountSegment.single (first := 10277) (initial := 4235) row_count_10278 (by decide)
private theorem atomic_10278_agreement : RowAgreement 10277 10278 := RowAgreement.single (first := 10277) modelrow_agrees_10278
private def atomic_10279 : CountSegment 1 10 13120 10278 10279 4236 4236 := CountSegment.single (first := 10278) (initial := 4236) row_count_10279 (by decide)
private theorem atomic_10279_agreement : RowAgreement 10278 10279 := RowAgreement.single (first := 10278) modelrow_agrees_10279
private def atomic_10280 : CountSegment 1 10 13120 10279 10280 4236 4236 := CountSegment.single (first := 10279) (initial := 4236) row_count_10280 (by decide)
private theorem atomic_10280_agreement : RowAgreement 10279 10280 := RowAgreement.single (first := 10279) modelrow_agrees_10280
private def atomic_10281 : CountSegment 1 10 13120 10280 10281 4236 4236 := CountSegment.single (first := 10280) (initial := 4236) row_count_10281 (by decide)
private theorem atomic_10281_agreement : RowAgreement 10280 10281 := RowAgreement.single (first := 10280) modelrow_agrees_10281
private def atomic_10282 : CountSegment 1 10 13120 10281 10282 4236 4237 := CountSegment.single (first := 10281) (initial := 4236) row_count_10282 (by decide)
private theorem atomic_10282_agreement : RowAgreement 10281 10282 := RowAgreement.single (first := 10281) modelrow_agrees_10282
private def atomic_10283 : CountSegment 1 10 13120 10282 10283 4237 4237 := CountSegment.single (first := 10282) (initial := 4237) row_count_10283 (by decide)
private theorem atomic_10283_agreement : RowAgreement 10282 10283 := RowAgreement.single (first := 10282) modelrow_agrees_10283
private def atomic_10284 : CountSegment 1 10 13120 10283 10284 4237 4237 := CountSegment.single (first := 10283) (initial := 4237) row_count_10284 (by decide)
private theorem atomic_10284_agreement : RowAgreement 10283 10284 := RowAgreement.single (first := 10283) modelrow_agrees_10284
private def atomic_10285 : CountSegment 1 10 13120 10284 10285 4237 4237 := CountSegment.single (first := 10284) (initial := 4237) row_count_10285 (by decide)
private theorem atomic_10285_agreement : RowAgreement 10284 10285 := RowAgreement.single (first := 10284) modelrow_agrees_10285
private def atomic_10286 : CountSegment 1 10 13120 10285 10286 4237 4237 := CountSegment.single (first := 10285) (initial := 4237) row_count_10286 (by decide)
private theorem atomic_10286_agreement : RowAgreement 10285 10286 := RowAgreement.single (first := 10285) modelrow_agrees_10286
private def atomic_10287 : CountSegment 1 10 13120 10286 10287 4237 4239 := CountSegment.single (first := 10286) (initial := 4237) row_count_10287 (by decide)
private theorem atomic_10287_agreement : RowAgreement 10286 10287 := RowAgreement.single (first := 10286) modelrow_agrees_10287
private def atomic_10288 : CountSegment 1 10 13120 10287 10288 4239 4240 := CountSegment.single (first := 10287) (initial := 4239) row_count_10288 (by decide)
private theorem atomic_10288_agreement : RowAgreement 10287 10288 := RowAgreement.single (first := 10287) modelrow_agrees_10288
private def atomic_10289 : CountSegment 1 10 13120 10288 10289 4240 4240 := CountSegment.single (first := 10288) (initial := 4240) row_count_10289 (by decide)
private theorem atomic_10289_agreement : RowAgreement 10288 10289 := RowAgreement.single (first := 10288) modelrow_agrees_10289
private def atomic_10290 : CountSegment 1 10 13120 10289 10290 4240 4241 := CountSegment.single (first := 10289) (initial := 4240) row_count_10290 (by decide)
private theorem atomic_10290_agreement : RowAgreement 10289 10290 := RowAgreement.single (first := 10289) modelrow_agrees_10290
private def atomic_10291 : CountSegment 1 10 13120 10290 10291 4241 4241 := CountSegment.single (first := 10290) (initial := 4241) row_count_10291 (by decide)
private theorem atomic_10291_agreement : RowAgreement 10290 10291 := RowAgreement.single (first := 10290) modelrow_agrees_10291
private def atomic_10292 : CountSegment 1 10 13120 10291 10292 4241 4242 := CountSegment.single (first := 10291) (initial := 4241) row_count_10292 (by decide)
private theorem atomic_10292_agreement : RowAgreement 10291 10292 := RowAgreement.single (first := 10291) modelrow_agrees_10292
private def atomic_10293 : CountSegment 1 10 13120 10292 10293 4242 4242 := CountSegment.single (first := 10292) (initial := 4242) row_count_10293 (by decide)
private theorem atomic_10293_agreement : RowAgreement 10292 10293 := RowAgreement.single (first := 10292) modelrow_agrees_10293
private def atomic_10294 : CountSegment 1 10 13120 10293 10294 4242 4242 := CountSegment.single (first := 10293) (initial := 4242) row_count_10294 (by decide)
private theorem atomic_10294_agreement : RowAgreement 10293 10294 := RowAgreement.single (first := 10293) modelrow_agrees_10294
private def atomic_10295 : CountSegment 1 10 13120 10294 10295 4242 4243 := CountSegment.single (first := 10294) (initial := 4242) row_count_10295 (by decide)
private theorem atomic_10295_agreement : RowAgreement 10294 10295 := RowAgreement.single (first := 10294) modelrow_agrees_10295
private def atomic_10296 : CountSegment 1 10 13120 10295 10296 4243 4243 := CountSegment.single (first := 10295) (initial := 4243) row_count_10296 (by decide)
private theorem atomic_10296_agreement : RowAgreement 10295 10296 := RowAgreement.single (first := 10295) modelrow_agrees_10296
private def atomic_10297 : CountSegment 1 10 13120 10296 10297 4243 4243 := CountSegment.single (first := 10296) (initial := 4243) row_count_10297 (by decide)
private theorem atomic_10297_agreement : RowAgreement 10296 10297 := RowAgreement.single (first := 10296) modelrow_agrees_10297
private def atomic_10298 : CountSegment 1 10 13120 10297 10298 4243 4244 := CountSegment.single (first := 10297) (initial := 4243) row_count_10298 (by decide)
private theorem atomic_10298_agreement : RowAgreement 10297 10298 := RowAgreement.single (first := 10297) modelrow_agrees_10298
private def atomic_10299 : CountSegment 1 10 13120 10298 10299 4244 4244 := CountSegment.single (first := 10298) (initial := 4244) row_count_10299 (by decide)
private theorem atomic_10299_agreement : RowAgreement 10298 10299 := RowAgreement.single (first := 10298) modelrow_agrees_10299
private def atomic_10300 : CountSegment 1 10 13120 10299 10300 4244 4244 := CountSegment.single (first := 10299) (initial := 4244) row_count_10300 (by decide)
private theorem atomic_10300_agreement : RowAgreement 10299 10300 := RowAgreement.single (first := 10299) modelrow_agrees_10300
private def atomic_10301 : CountSegment 1 10 13120 10300 10301 4244 4244 := CountSegment.single (first := 10300) (initial := 4244) row_count_10301 (by decide)
private theorem atomic_10301_agreement : RowAgreement 10300 10301 := RowAgreement.single (first := 10300) modelrow_agrees_10301
private def atomic_10302 : CountSegment 1 10 13120 10301 10302 4244 4244 := CountSegment.single (first := 10301) (initial := 4244) row_count_10302 (by decide)
private theorem atomic_10302_agreement : RowAgreement 10301 10302 := RowAgreement.single (first := 10301) modelrow_agrees_10302
private def atomic_10303 : CountSegment 1 10 13120 10302 10303 4244 4244 := CountSegment.single (first := 10302) (initial := 4244) row_count_10303 (by decide)
private theorem atomic_10303_agreement : RowAgreement 10302 10303 := RowAgreement.single (first := 10302) modelrow_agrees_10303
private def atomic_10304 : CountSegment 1 10 13120 10303 10304 4244 4244 := CountSegment.single (first := 10303) (initial := 4244) row_count_10304 (by decide)
private theorem atomic_10304_agreement : RowAgreement 10303 10304 := RowAgreement.single (first := 10303) modelrow_agrees_10304
private def atomic_10305 : CountSegment 1 10 13120 10304 10305 4244 4244 := CountSegment.single (first := 10304) (initial := 4244) row_count_10305 (by decide)
private theorem atomic_10305_agreement : RowAgreement 10304 10305 := RowAgreement.single (first := 10304) modelrow_agrees_10305
private def atomic_10306 : CountSegment 1 10 13120 10305 10306 4244 4244 := CountSegment.single (first := 10305) (initial := 4244) row_count_10306 (by decide)
private theorem atomic_10306_agreement : RowAgreement 10305 10306 := RowAgreement.single (first := 10305) modelrow_agrees_10306
private def atomic_10307 : CountSegment 1 10 13120 10306 10307 4244 4245 := CountSegment.single (first := 10306) (initial := 4244) row_count_10307 (by decide)
private theorem atomic_10307_agreement : RowAgreement 10306 10307 := RowAgreement.single (first := 10306) modelrow_agrees_10307
private def atomic_10308 : CountSegment 1 10 13120 10307 10308 4245 4247 := CountSegment.single (first := 10307) (initial := 4245) row_count_10308 (by decide)
private theorem atomic_10308_agreement : RowAgreement 10307 10308 := RowAgreement.single (first := 10307) modelrow_agrees_10308
private def atomic_10309 : CountSegment 1 10 13120 10308 10309 4247 4247 := CountSegment.single (first := 10308) (initial := 4247) row_count_10309 (by decide)
private theorem atomic_10309_agreement : RowAgreement 10308 10309 := RowAgreement.single (first := 10308) modelrow_agrees_10309
private def atomic_10310 : CountSegment 1 10 13120 10309 10310 4247 4248 := CountSegment.single (first := 10309) (initial := 4247) row_count_10310 (by decide)
private theorem atomic_10310_agreement : RowAgreement 10309 10310 := RowAgreement.single (first := 10309) modelrow_agrees_10310
private def atomic_10311 : CountSegment 1 10 13120 10310 10311 4248 4248 := CountSegment.single (first := 10310) (initial := 4248) row_count_10311 (by decide)
private theorem atomic_10311_agreement : RowAgreement 10310 10311 := RowAgreement.single (first := 10310) modelrow_agrees_10311
private def atomic_10312 : CountSegment 1 10 13120 10311 10312 4248 4249 := CountSegment.single (first := 10311) (initial := 4248) row_count_10312 (by decide)
private theorem atomic_10312_agreement : RowAgreement 10311 10312 := RowAgreement.single (first := 10311) modelrow_agrees_10312
private def atomic_10313 : CountSegment 1 10 13120 10312 10313 4249 4249 := CountSegment.single (first := 10312) (initial := 4249) row_count_10313 (by decide)
private theorem atomic_10313_agreement : RowAgreement 10312 10313 := RowAgreement.single (first := 10312) modelrow_agrees_10313
private def atomic_10314 : CountSegment 1 10 13120 10313 10314 4249 4250 := CountSegment.single (first := 10313) (initial := 4249) row_count_10314 (by decide)
private theorem atomic_10314_agreement : RowAgreement 10313 10314 := RowAgreement.single (first := 10313) modelrow_agrees_10314
private def atomic_10315 : CountSegment 1 10 13120 10314 10315 4250 4252 := CountSegment.single (first := 10314) (initial := 4250) row_count_10315 (by decide)
private theorem atomic_10315_agreement : RowAgreement 10314 10315 := RowAgreement.single (first := 10314) modelrow_agrees_10315
private def atomic_10316 : CountSegment 1 10 13120 10315 10316 4252 4252 := CountSegment.single (first := 10315) (initial := 4252) row_count_10316 (by decide)
private theorem atomic_10316_agreement : RowAgreement 10315 10316 := RowAgreement.single (first := 10315) modelrow_agrees_10316
private def atomic_10317 : CountSegment 1 10 13120 10316 10317 4252 4252 := CountSegment.single (first := 10316) (initial := 4252) row_count_10317 (by decide)
private theorem atomic_10317_agreement : RowAgreement 10316 10317 := RowAgreement.single (first := 10316) modelrow_agrees_10317
private def atomic_10318 : CountSegment 1 10 13120 10317 10318 4252 4253 := CountSegment.single (first := 10317) (initial := 4252) row_count_10318 (by decide)
private theorem atomic_10318_agreement : RowAgreement 10317 10318 := RowAgreement.single (first := 10317) modelrow_agrees_10318
private def atomic_10319 : CountSegment 1 10 13120 10318 10319 4253 4253 := CountSegment.single (first := 10318) (initial := 4253) row_count_10319 (by decide)
private theorem atomic_10319_agreement : RowAgreement 10318 10319 := RowAgreement.single (first := 10318) modelrow_agrees_10319
private def atomic_10320 : CountSegment 1 10 13120 10319 10320 4253 4253 := CountSegment.single (first := 10319) (initial := 4253) row_count_10320 (by decide)
private theorem atomic_10320_agreement : RowAgreement 10319 10320 := RowAgreement.single (first := 10319) modelrow_agrees_10320
private def atomic_10321 : CountSegment 1 10 13120 10320 10321 4253 4253 := CountSegment.single (first := 10320) (initial := 4253) row_count_10321 (by decide)
private theorem atomic_10321_agreement : RowAgreement 10320 10321 := RowAgreement.single (first := 10320) modelrow_agrees_10321
private def atomic_10322 : CountSegment 1 10 13120 10321 10322 4253 4254 := CountSegment.single (first := 10321) (initial := 4253) row_count_10322 (by decide)
private theorem atomic_10322_agreement : RowAgreement 10321 10322 := RowAgreement.single (first := 10321) modelrow_agrees_10322
private def atomic_10323 : CountSegment 1 10 13120 10322 10323 4254 4254 := CountSegment.single (first := 10322) (initial := 4254) row_count_10323 (by decide)
private theorem atomic_10323_agreement : RowAgreement 10322 10323 := RowAgreement.single (first := 10322) modelrow_agrees_10323
private def atomic_10324 : CountSegment 1 10 13120 10323 10324 4254 4254 := CountSegment.single (first := 10323) (initial := 4254) row_count_10324 (by decide)
private theorem atomic_10324_agreement : RowAgreement 10323 10324 := RowAgreement.single (first := 10323) modelrow_agrees_10324
private def atomic_10325 : CountSegment 1 10 13120 10324 10325 4254 4254 := CountSegment.single (first := 10324) (initial := 4254) row_count_10325 (by decide)
private theorem atomic_10325_agreement : RowAgreement 10324 10325 := RowAgreement.single (first := 10324) modelrow_agrees_10325
private def atomic_10326 : CountSegment 1 10 13120 10325 10326 4254 4254 := CountSegment.single (first := 10325) (initial := 4254) row_count_10326 (by decide)
private theorem atomic_10326_agreement : RowAgreement 10325 10326 := RowAgreement.single (first := 10325) modelrow_agrees_10326
private def atomic_10327 : CountSegment 1 10 13120 10326 10327 4254 4255 := CountSegment.single (first := 10326) (initial := 4254) row_count_10327 (by decide)
private theorem atomic_10327_agreement : RowAgreement 10326 10327 := RowAgreement.single (first := 10326) modelrow_agrees_10327
private def atomic_10328 : CountSegment 1 10 13120 10327 10328 4255 4257 := CountSegment.single (first := 10327) (initial := 4255) row_count_10328 (by decide)
private theorem atomic_10328_agreement : RowAgreement 10327 10328 := RowAgreement.single (first := 10327) modelrow_agrees_10328
private def atomic_10329 : CountSegment 1 10 13120 10328 10329 4257 4257 := CountSegment.single (first := 10328) (initial := 4257) row_count_10329 (by decide)
private theorem atomic_10329_agreement : RowAgreement 10328 10329 := RowAgreement.single (first := 10328) modelrow_agrees_10329
private def atomic_10330 : CountSegment 1 10 13120 10329 10330 4257 4257 := CountSegment.single (first := 10329) (initial := 4257) row_count_10330 (by decide)
private theorem atomic_10330_agreement : RowAgreement 10329 10330 := RowAgreement.single (first := 10329) modelrow_agrees_10330
private def atomic_10331 : CountSegment 1 10 13120 10330 10331 4257 4257 := CountSegment.single (first := 10330) (initial := 4257) row_count_10331 (by decide)
private theorem atomic_10331_agreement : RowAgreement 10330 10331 := RowAgreement.single (first := 10330) modelrow_agrees_10331
private def atomic_10332 : CountSegment 1 10 13120 10331 10332 4257 4257 := CountSegment.single (first := 10331) (initial := 4257) row_count_10332 (by decide)
private theorem atomic_10332_agreement : RowAgreement 10331 10332 := RowAgreement.single (first := 10331) modelrow_agrees_10332
private def atomic_10333 : CountSegment 1 10 13120 10332 10333 4257 4257 := CountSegment.single (first := 10332) (initial := 4257) row_count_10333 (by decide)
private theorem atomic_10333_agreement : RowAgreement 10332 10333 := RowAgreement.single (first := 10332) modelrow_agrees_10333
private def atomic_10334 : CountSegment 1 10 13120 10333 10334 4257 4257 := CountSegment.single (first := 10333) (initial := 4257) row_count_10334 (by decide)
private theorem atomic_10334_agreement : RowAgreement 10333 10334 := RowAgreement.single (first := 10333) modelrow_agrees_10334
private def atomic_10335 : CountSegment 1 10 13120 10334 10335 4257 4259 := CountSegment.single (first := 10334) (initial := 4257) row_count_10335 (by decide)
private theorem atomic_10335_agreement : RowAgreement 10334 10335 := RowAgreement.single (first := 10334) modelrow_agrees_10335
private def atomic_10336 : CountSegment 1 10 13120 10335 10336 4259 4259 := CountSegment.single (first := 10335) (initial := 4259) row_count_10336 (by decide)
private theorem atomic_10336_agreement : RowAgreement 10335 10336 := RowAgreement.single (first := 10335) modelrow_agrees_10336
private def atomic_10337 : CountSegment 1 10 13120 10336 10337 4259 4259 := CountSegment.single (first := 10336) (initial := 4259) row_count_10337 (by decide)
private theorem atomic_10337_agreement : RowAgreement 10336 10337 := RowAgreement.single (first := 10336) modelrow_agrees_10337
private def atomic_10338 : CountSegment 1 10 13120 10337 10338 4259 4259 := CountSegment.single (first := 10337) (initial := 4259) row_count_10338 (by decide)
private theorem atomic_10338_agreement : RowAgreement 10337 10338 := RowAgreement.single (first := 10337) modelrow_agrees_10338
private def atomic_10339 : CountSegment 1 10 13120 10338 10339 4259 4259 := CountSegment.single (first := 10338) (initial := 4259) row_count_10339 (by decide)
private theorem atomic_10339_agreement : RowAgreement 10338 10339 := RowAgreement.single (first := 10338) modelrow_agrees_10339
private def atomic_10340 : CountSegment 1 10 13120 10339 10340 4259 4259 := CountSegment.single (first := 10339) (initial := 4259) row_count_10340 (by decide)
private theorem atomic_10340_agreement : RowAgreement 10339 10340 := RowAgreement.single (first := 10339) modelrow_agrees_10340
private def atomic_10341 : CountSegment 1 10 13120 10340 10341 4259 4259 := CountSegment.single (first := 10340) (initial := 4259) row_count_10341 (by decide)
private theorem atomic_10341_agreement : RowAgreement 10340 10341 := RowAgreement.single (first := 10340) modelrow_agrees_10341
private def atomic_10342 : CountSegment 1 10 13120 10341 10342 4259 4261 := CountSegment.single (first := 10341) (initial := 4259) row_count_10342 (by decide)
private theorem atomic_10342_agreement : RowAgreement 10341 10342 := RowAgreement.single (first := 10341) modelrow_agrees_10342
private def atomic_10343 : CountSegment 1 10 13120 10342 10343 4261 4261 := CountSegment.single (first := 10342) (initial := 4261) row_count_10343 (by decide)
private theorem atomic_10343_agreement : RowAgreement 10342 10343 := RowAgreement.single (first := 10342) modelrow_agrees_10343
private def atomic_10344 : CountSegment 1 10 13120 10343 10344 4261 4262 := CountSegment.single (first := 10343) (initial := 4261) row_count_10344 (by decide)
private theorem atomic_10344_agreement : RowAgreement 10343 10344 := RowAgreement.single (first := 10343) modelrow_agrees_10344
private def atomic_10345 : CountSegment 1 10 13120 10344 10345 4262 4262 := CountSegment.single (first := 10344) (initial := 4262) row_count_10345 (by decide)
private theorem atomic_10345_agreement : RowAgreement 10344 10345 := RowAgreement.single (first := 10344) modelrow_agrees_10345
private def atomic_10346 : CountSegment 1 10 13120 10345 10346 4262 4262 := CountSegment.single (first := 10345) (initial := 4262) row_count_10346 (by decide)
private theorem atomic_10346_agreement : RowAgreement 10345 10346 := RowAgreement.single (first := 10345) modelrow_agrees_10346
private def atomic_10347 : CountSegment 1 10 13120 10346 10347 4262 4262 := CountSegment.single (first := 10346) (initial := 4262) row_count_10347 (by decide)
private theorem atomic_10347_agreement : RowAgreement 10346 10347 := RowAgreement.single (first := 10346) modelrow_agrees_10347
private def atomic_10348 : CountSegment 1 10 13120 10347 10348 4262 4262 := CountSegment.single (first := 10347) (initial := 4262) row_count_10348 (by decide)
private theorem atomic_10348_agreement : RowAgreement 10347 10348 := RowAgreement.single (first := 10347) modelrow_agrees_10348
private def atomic_10349 : CountSegment 1 10 13120 10348 10349 4262 4262 := CountSegment.single (first := 10348) (initial := 4262) row_count_10349 (by decide)
private theorem atomic_10349_agreement : RowAgreement 10348 10349 := RowAgreement.single (first := 10348) modelrow_agrees_10349
private def atomic_10350 : CountSegment 1 10 13120 10349 10350 4262 4263 := CountSegment.single (first := 10349) (initial := 4262) row_count_10350 (by decide)
private theorem atomic_10350_agreement : RowAgreement 10349 10350 := RowAgreement.single (first := 10349) modelrow_agrees_10350
private def atomic_10351 : CountSegment 1 10 13120 10350 10351 4263 4263 := CountSegment.single (first := 10350) (initial := 4263) row_count_10351 (by decide)
private theorem atomic_10351_agreement : RowAgreement 10350 10351 := RowAgreement.single (first := 10350) modelrow_agrees_10351
private def atomic_10352 : CountSegment 1 10 13120 10351 10352 4263 4263 := CountSegment.single (first := 10351) (initial := 4263) row_count_10352 (by decide)
private theorem atomic_10352_agreement : RowAgreement 10351 10352 := RowAgreement.single (first := 10351) modelrow_agrees_10352
private def atomic_10353 : CountSegment 1 10 13120 10352 10353 4263 4265 := CountSegment.single (first := 10352) (initial := 4263) row_count_10353 (by decide)
private theorem atomic_10353_agreement : RowAgreement 10352 10353 := RowAgreement.single (first := 10352) modelrow_agrees_10353
private def atomic_10354 : CountSegment 1 10 13120 10353 10354 4265 4265 := CountSegment.single (first := 10353) (initial := 4265) row_count_10354 (by decide)
private theorem atomic_10354_agreement : RowAgreement 10353 10354 := RowAgreement.single (first := 10353) modelrow_agrees_10354
private def atomic_10355 : CountSegment 1 10 13120 10354 10355 4265 4265 := CountSegment.single (first := 10354) (initial := 4265) row_count_10355 (by decide)
private theorem atomic_10355_agreement : RowAgreement 10354 10355 := RowAgreement.single (first := 10354) modelrow_agrees_10355
private def atomic_10356 : CountSegment 1 10 13120 10355 10356 4265 4265 := CountSegment.single (first := 10355) (initial := 4265) row_count_10356 (by decide)
private theorem atomic_10356_agreement : RowAgreement 10355 10356 := RowAgreement.single (first := 10355) modelrow_agrees_10356
private def atomic_10357 : CountSegment 1 10 13120 10356 10357 4265 4265 := CountSegment.single (first := 10356) (initial := 4265) row_count_10357 (by decide)
private theorem atomic_10357_agreement : RowAgreement 10356 10357 := RowAgreement.single (first := 10356) modelrow_agrees_10357
private def atomic_10358 : CountSegment 1 10 13120 10357 10358 4265 4265 := CountSegment.single (first := 10357) (initial := 4265) row_count_10358 (by decide)
private theorem atomic_10358_agreement : RowAgreement 10357 10358 := RowAgreement.single (first := 10357) modelrow_agrees_10358
private def atomic_10359 : CountSegment 1 10 13120 10358 10359 4265 4265 := CountSegment.single (first := 10358) (initial := 4265) row_count_10359 (by decide)
private theorem atomic_10359_agreement : RowAgreement 10358 10359 := RowAgreement.single (first := 10358) modelrow_agrees_10359
private def atomic_10360 : CountSegment 1 10 13120 10359 10360 4265 4265 := CountSegment.single (first := 10359) (initial := 4265) row_count_10360 (by decide)
private theorem atomic_10360_agreement : RowAgreement 10359 10360 := RowAgreement.single (first := 10359) modelrow_agrees_10360
private def atomic_10361 : CountSegment 1 10 13120 10360 10361 4265 4265 := CountSegment.single (first := 10360) (initial := 4265) row_count_10361 (by decide)
private theorem atomic_10361_agreement : RowAgreement 10360 10361 := RowAgreement.single (first := 10360) modelrow_agrees_10361
private def atomic_10362 : CountSegment 1 10 13120 10361 10362 4265 4265 := CountSegment.single (first := 10361) (initial := 4265) row_count_10362 (by decide)
private theorem atomic_10362_agreement : RowAgreement 10361 10362 := RowAgreement.single (first := 10361) modelrow_agrees_10362
private def atomic_10363 : CountSegment 1 10 13120 10362 10363 4265 4265 := CountSegment.single (first := 10362) (initial := 4265) row_count_10363 (by decide)
private theorem atomic_10363_agreement : RowAgreement 10362 10363 := RowAgreement.single (first := 10362) modelrow_agrees_10363
private def atomic_10364 : CountSegment 1 10 13120 10363 10364 4265 4265 := CountSegment.single (first := 10363) (initial := 4265) row_count_10364 (by decide)
private theorem atomic_10364_agreement : RowAgreement 10363 10364 := RowAgreement.single (first := 10363) modelrow_agrees_10364
private def atomic_10365 : CountSegment 1 10 13120 10364 10365 4265 4265 := CountSegment.single (first := 10364) (initial := 4265) row_count_10365 (by decide)
private theorem atomic_10365_agreement : RowAgreement 10364 10365 := RowAgreement.single (first := 10364) modelrow_agrees_10365
private def atomic_10366 : CountSegment 1 10 13120 10365 10366 4265 4265 := CountSegment.single (first := 10365) (initial := 4265) row_count_10366 (by decide)
private theorem atomic_10366_agreement : RowAgreement 10365 10366 := RowAgreement.single (first := 10365) modelrow_agrees_10366
private def atomic_10367 : CountSegment 1 10 13120 10366 10367 4265 4266 := CountSegment.single (first := 10366) (initial := 4265) row_count_10367 (by decide)
private theorem atomic_10367_agreement : RowAgreement 10366 10367 := RowAgreement.single (first := 10366) modelrow_agrees_10367
private def atomic_10368 : CountSegment 1 10 13120 10367 10368 4266 4266 := CountSegment.single (first := 10367) (initial := 4266) row_count_10368 (by decide)
private theorem atomic_10368_agreement : RowAgreement 10367 10368 := RowAgreement.single (first := 10367) modelrow_agrees_10368
private def within_080_000 : CountSegment 1 10 13120 10240 10242 4224 4225 := CountSegment.join atomic_10241 atomic_10242
private theorem within_080_000_agreement : RowAgreement 10240 10242 := RowAgreement.join atomic_10241_agreement atomic_10242_agreement
private def within_080_001 : CountSegment 1 10 13120 10242 10244 4225 4225 := CountSegment.join atomic_10243 atomic_10244
private theorem within_080_001_agreement : RowAgreement 10242 10244 := RowAgreement.join atomic_10243_agreement atomic_10244_agreement
private def within_080_002 : CountSegment 1 10 13120 10240 10244 4224 4225 := CountSegment.join within_080_000 within_080_001
private theorem within_080_002_agreement : RowAgreement 10240 10244 := RowAgreement.join within_080_000_agreement within_080_001_agreement
private def within_080_003 : CountSegment 1 10 13120 10244 10246 4225 4225 := CountSegment.join atomic_10245 atomic_10246
private theorem within_080_003_agreement : RowAgreement 10244 10246 := RowAgreement.join atomic_10245_agreement atomic_10246_agreement
private def within_080_004 : CountSegment 1 10 13120 10246 10248 4225 4225 := CountSegment.join atomic_10247 atomic_10248
private theorem within_080_004_agreement : RowAgreement 10246 10248 := RowAgreement.join atomic_10247_agreement atomic_10248_agreement
private def within_080_005 : CountSegment 1 10 13120 10244 10248 4225 4225 := CountSegment.join within_080_003 within_080_004
private theorem within_080_005_agreement : RowAgreement 10244 10248 := RowAgreement.join within_080_003_agreement within_080_004_agreement
private def within_080_006 : CountSegment 1 10 13120 10240 10248 4224 4225 := CountSegment.join within_080_002 within_080_005
private theorem within_080_006_agreement : RowAgreement 10240 10248 := RowAgreement.join within_080_002_agreement within_080_005_agreement
private def within_080_007 : CountSegment 1 10 13120 10248 10250 4225 4227 := CountSegment.join atomic_10249 atomic_10250
private theorem within_080_007_agreement : RowAgreement 10248 10250 := RowAgreement.join atomic_10249_agreement atomic_10250_agreement
private def within_080_008 : CountSegment 1 10 13120 10250 10252 4227 4227 := CountSegment.join atomic_10251 atomic_10252
private theorem within_080_008_agreement : RowAgreement 10250 10252 := RowAgreement.join atomic_10251_agreement atomic_10252_agreement
private def within_080_009 : CountSegment 1 10 13120 10248 10252 4225 4227 := CountSegment.join within_080_007 within_080_008
private theorem within_080_009_agreement : RowAgreement 10248 10252 := RowAgreement.join within_080_007_agreement within_080_008_agreement
private def within_080_010 : CountSegment 1 10 13120 10252 10254 4227 4227 := CountSegment.join atomic_10253 atomic_10254
private theorem within_080_010_agreement : RowAgreement 10252 10254 := RowAgreement.join atomic_10253_agreement atomic_10254_agreement
private def within_080_011 : CountSegment 1 10 13120 10254 10256 4227 4227 := CountSegment.join atomic_10255 atomic_10256
private theorem within_080_011_agreement : RowAgreement 10254 10256 := RowAgreement.join atomic_10255_agreement atomic_10256_agreement
private def within_080_012 : CountSegment 1 10 13120 10252 10256 4227 4227 := CountSegment.join within_080_010 within_080_011
private theorem within_080_012_agreement : RowAgreement 10252 10256 := RowAgreement.join within_080_010_agreement within_080_011_agreement
private def within_080_013 : CountSegment 1 10 13120 10248 10256 4225 4227 := CountSegment.join within_080_009 within_080_012
private theorem within_080_013_agreement : RowAgreement 10248 10256 := RowAgreement.join within_080_009_agreement within_080_012_agreement
private def within_080_014 : CountSegment 1 10 13120 10240 10256 4224 4227 := CountSegment.join within_080_006 within_080_013
private theorem within_080_014_agreement : RowAgreement 10240 10256 := RowAgreement.join within_080_006_agreement within_080_013_agreement
private def within_080_015 : CountSegment 1 10 13120 10256 10258 4227 4228 := CountSegment.join atomic_10257 atomic_10258
private theorem within_080_015_agreement : RowAgreement 10256 10258 := RowAgreement.join atomic_10257_agreement atomic_10258_agreement
private def within_080_016 : CountSegment 1 10 13120 10258 10260 4228 4229 := CountSegment.join atomic_10259 atomic_10260
private theorem within_080_016_agreement : RowAgreement 10258 10260 := RowAgreement.join atomic_10259_agreement atomic_10260_agreement
private def within_080_017 : CountSegment 1 10 13120 10256 10260 4227 4229 := CountSegment.join within_080_015 within_080_016
private theorem within_080_017_agreement : RowAgreement 10256 10260 := RowAgreement.join within_080_015_agreement within_080_016_agreement
private def within_080_018 : CountSegment 1 10 13120 10260 10262 4229 4230 := CountSegment.join atomic_10261 atomic_10262
private theorem within_080_018_agreement : RowAgreement 10260 10262 := RowAgreement.join atomic_10261_agreement atomic_10262_agreement
private def within_080_019 : CountSegment 1 10 13120 10262 10264 4230 4231 := CountSegment.join atomic_10263 atomic_10264
private theorem within_080_019_agreement : RowAgreement 10262 10264 := RowAgreement.join atomic_10263_agreement atomic_10264_agreement
private def within_080_020 : CountSegment 1 10 13120 10260 10264 4229 4231 := CountSegment.join within_080_018 within_080_019
private theorem within_080_020_agreement : RowAgreement 10260 10264 := RowAgreement.join within_080_018_agreement within_080_019_agreement
private def within_080_021 : CountSegment 1 10 13120 10256 10264 4227 4231 := CountSegment.join within_080_017 within_080_020
private theorem within_080_021_agreement : RowAgreement 10256 10264 := RowAgreement.join within_080_017_agreement within_080_020_agreement
private def within_080_022 : CountSegment 1 10 13120 10264 10266 4231 4231 := CountSegment.join atomic_10265 atomic_10266
private theorem within_080_022_agreement : RowAgreement 10264 10266 := RowAgreement.join atomic_10265_agreement atomic_10266_agreement
private def within_080_023 : CountSegment 1 10 13120 10266 10268 4231 4232 := CountSegment.join atomic_10267 atomic_10268
private theorem within_080_023_agreement : RowAgreement 10266 10268 := RowAgreement.join atomic_10267_agreement atomic_10268_agreement
private def within_080_024 : CountSegment 1 10 13120 10264 10268 4231 4232 := CountSegment.join within_080_022 within_080_023
private theorem within_080_024_agreement : RowAgreement 10264 10268 := RowAgreement.join within_080_022_agreement within_080_023_agreement
private def within_080_025 : CountSegment 1 10 13120 10268 10270 4232 4232 := CountSegment.join atomic_10269 atomic_10270
private theorem within_080_025_agreement : RowAgreement 10268 10270 := RowAgreement.join atomic_10269_agreement atomic_10270_agreement
private def within_080_026 : CountSegment 1 10 13120 10270 10272 4232 4234 := CountSegment.join atomic_10271 atomic_10272
private theorem within_080_026_agreement : RowAgreement 10270 10272 := RowAgreement.join atomic_10271_agreement atomic_10272_agreement
private def within_080_027 : CountSegment 1 10 13120 10268 10272 4232 4234 := CountSegment.join within_080_025 within_080_026
private theorem within_080_027_agreement : RowAgreement 10268 10272 := RowAgreement.join within_080_025_agreement within_080_026_agreement
private def within_080_028 : CountSegment 1 10 13120 10264 10272 4231 4234 := CountSegment.join within_080_024 within_080_027
private theorem within_080_028_agreement : RowAgreement 10264 10272 := RowAgreement.join within_080_024_agreement within_080_027_agreement
private def within_080_029 : CountSegment 1 10 13120 10256 10272 4227 4234 := CountSegment.join within_080_021 within_080_028
private theorem within_080_029_agreement : RowAgreement 10256 10272 := RowAgreement.join within_080_021_agreement within_080_028_agreement
private def within_080_030 : CountSegment 1 10 13120 10240 10272 4224 4234 := CountSegment.join within_080_014 within_080_029
private theorem within_080_030_agreement : RowAgreement 10240 10272 := RowAgreement.join within_080_014_agreement within_080_029_agreement
private def within_080_031 : CountSegment 1 10 13120 10272 10274 4234 4234 := CountSegment.join atomic_10273 atomic_10274
private theorem within_080_031_agreement : RowAgreement 10272 10274 := RowAgreement.join atomic_10273_agreement atomic_10274_agreement
private def within_080_032 : CountSegment 1 10 13120 10274 10276 4234 4235 := CountSegment.join atomic_10275 atomic_10276
private theorem within_080_032_agreement : RowAgreement 10274 10276 := RowAgreement.join atomic_10275_agreement atomic_10276_agreement
private def within_080_033 : CountSegment 1 10 13120 10272 10276 4234 4235 := CountSegment.join within_080_031 within_080_032
private theorem within_080_033_agreement : RowAgreement 10272 10276 := RowAgreement.join within_080_031_agreement within_080_032_agreement
private def within_080_034 : CountSegment 1 10 13120 10276 10278 4235 4236 := CountSegment.join atomic_10277 atomic_10278
private theorem within_080_034_agreement : RowAgreement 10276 10278 := RowAgreement.join atomic_10277_agreement atomic_10278_agreement
private def within_080_035 : CountSegment 1 10 13120 10278 10280 4236 4236 := CountSegment.join atomic_10279 atomic_10280
private theorem within_080_035_agreement : RowAgreement 10278 10280 := RowAgreement.join atomic_10279_agreement atomic_10280_agreement
private def within_080_036 : CountSegment 1 10 13120 10276 10280 4235 4236 := CountSegment.join within_080_034 within_080_035
private theorem within_080_036_agreement : RowAgreement 10276 10280 := RowAgreement.join within_080_034_agreement within_080_035_agreement
private def within_080_037 : CountSegment 1 10 13120 10272 10280 4234 4236 := CountSegment.join within_080_033 within_080_036
private theorem within_080_037_agreement : RowAgreement 10272 10280 := RowAgreement.join within_080_033_agreement within_080_036_agreement
private def within_080_038 : CountSegment 1 10 13120 10280 10282 4236 4237 := CountSegment.join atomic_10281 atomic_10282
private theorem within_080_038_agreement : RowAgreement 10280 10282 := RowAgreement.join atomic_10281_agreement atomic_10282_agreement
private def within_080_039 : CountSegment 1 10 13120 10282 10284 4237 4237 := CountSegment.join atomic_10283 atomic_10284
private theorem within_080_039_agreement : RowAgreement 10282 10284 := RowAgreement.join atomic_10283_agreement atomic_10284_agreement
private def within_080_040 : CountSegment 1 10 13120 10280 10284 4236 4237 := CountSegment.join within_080_038 within_080_039
private theorem within_080_040_agreement : RowAgreement 10280 10284 := RowAgreement.join within_080_038_agreement within_080_039_agreement
private def within_080_041 : CountSegment 1 10 13120 10284 10286 4237 4237 := CountSegment.join atomic_10285 atomic_10286
private theorem within_080_041_agreement : RowAgreement 10284 10286 := RowAgreement.join atomic_10285_agreement atomic_10286_agreement
private def within_080_042 : CountSegment 1 10 13120 10286 10288 4237 4240 := CountSegment.join atomic_10287 atomic_10288
private theorem within_080_042_agreement : RowAgreement 10286 10288 := RowAgreement.join atomic_10287_agreement atomic_10288_agreement
private def within_080_043 : CountSegment 1 10 13120 10284 10288 4237 4240 := CountSegment.join within_080_041 within_080_042
private theorem within_080_043_agreement : RowAgreement 10284 10288 := RowAgreement.join within_080_041_agreement within_080_042_agreement
private def within_080_044 : CountSegment 1 10 13120 10280 10288 4236 4240 := CountSegment.join within_080_040 within_080_043
private theorem within_080_044_agreement : RowAgreement 10280 10288 := RowAgreement.join within_080_040_agreement within_080_043_agreement
private def within_080_045 : CountSegment 1 10 13120 10272 10288 4234 4240 := CountSegment.join within_080_037 within_080_044
private theorem within_080_045_agreement : RowAgreement 10272 10288 := RowAgreement.join within_080_037_agreement within_080_044_agreement
private def within_080_046 : CountSegment 1 10 13120 10288 10290 4240 4241 := CountSegment.join atomic_10289 atomic_10290
private theorem within_080_046_agreement : RowAgreement 10288 10290 := RowAgreement.join atomic_10289_agreement atomic_10290_agreement
private def within_080_047 : CountSegment 1 10 13120 10290 10292 4241 4242 := CountSegment.join atomic_10291 atomic_10292
private theorem within_080_047_agreement : RowAgreement 10290 10292 := RowAgreement.join atomic_10291_agreement atomic_10292_agreement
private def within_080_048 : CountSegment 1 10 13120 10288 10292 4240 4242 := CountSegment.join within_080_046 within_080_047
private theorem within_080_048_agreement : RowAgreement 10288 10292 := RowAgreement.join within_080_046_agreement within_080_047_agreement
private def within_080_049 : CountSegment 1 10 13120 10292 10294 4242 4242 := CountSegment.join atomic_10293 atomic_10294
private theorem within_080_049_agreement : RowAgreement 10292 10294 := RowAgreement.join atomic_10293_agreement atomic_10294_agreement
private def within_080_050 : CountSegment 1 10 13120 10294 10296 4242 4243 := CountSegment.join atomic_10295 atomic_10296
private theorem within_080_050_agreement : RowAgreement 10294 10296 := RowAgreement.join atomic_10295_agreement atomic_10296_agreement
private def within_080_051 : CountSegment 1 10 13120 10292 10296 4242 4243 := CountSegment.join within_080_049 within_080_050
private theorem within_080_051_agreement : RowAgreement 10292 10296 := RowAgreement.join within_080_049_agreement within_080_050_agreement
private def within_080_052 : CountSegment 1 10 13120 10288 10296 4240 4243 := CountSegment.join within_080_048 within_080_051
private theorem within_080_052_agreement : RowAgreement 10288 10296 := RowAgreement.join within_080_048_agreement within_080_051_agreement
private def within_080_053 : CountSegment 1 10 13120 10296 10298 4243 4244 := CountSegment.join atomic_10297 atomic_10298
private theorem within_080_053_agreement : RowAgreement 10296 10298 := RowAgreement.join atomic_10297_agreement atomic_10298_agreement
private def within_080_054 : CountSegment 1 10 13120 10298 10300 4244 4244 := CountSegment.join atomic_10299 atomic_10300
private theorem within_080_054_agreement : RowAgreement 10298 10300 := RowAgreement.join atomic_10299_agreement atomic_10300_agreement
private def within_080_055 : CountSegment 1 10 13120 10296 10300 4243 4244 := CountSegment.join within_080_053 within_080_054
private theorem within_080_055_agreement : RowAgreement 10296 10300 := RowAgreement.join within_080_053_agreement within_080_054_agreement
private def within_080_056 : CountSegment 1 10 13120 10300 10302 4244 4244 := CountSegment.join atomic_10301 atomic_10302
private theorem within_080_056_agreement : RowAgreement 10300 10302 := RowAgreement.join atomic_10301_agreement atomic_10302_agreement
private def within_080_057 : CountSegment 1 10 13120 10302 10304 4244 4244 := CountSegment.join atomic_10303 atomic_10304
private theorem within_080_057_agreement : RowAgreement 10302 10304 := RowAgreement.join atomic_10303_agreement atomic_10304_agreement
private def within_080_058 : CountSegment 1 10 13120 10300 10304 4244 4244 := CountSegment.join within_080_056 within_080_057
private theorem within_080_058_agreement : RowAgreement 10300 10304 := RowAgreement.join within_080_056_agreement within_080_057_agreement
private def within_080_059 : CountSegment 1 10 13120 10296 10304 4243 4244 := CountSegment.join within_080_055 within_080_058
private theorem within_080_059_agreement : RowAgreement 10296 10304 := RowAgreement.join within_080_055_agreement within_080_058_agreement
private def within_080_060 : CountSegment 1 10 13120 10288 10304 4240 4244 := CountSegment.join within_080_052 within_080_059
private theorem within_080_060_agreement : RowAgreement 10288 10304 := RowAgreement.join within_080_052_agreement within_080_059_agreement
private def within_080_061 : CountSegment 1 10 13120 10272 10304 4234 4244 := CountSegment.join within_080_045 within_080_060
private theorem within_080_061_agreement : RowAgreement 10272 10304 := RowAgreement.join within_080_045_agreement within_080_060_agreement
private def within_080_062 : CountSegment 1 10 13120 10240 10304 4224 4244 := CountSegment.join within_080_030 within_080_061
private theorem within_080_062_agreement : RowAgreement 10240 10304 := RowAgreement.join within_080_030_agreement within_080_061_agreement
private def within_080_063 : CountSegment 1 10 13120 10304 10306 4244 4244 := CountSegment.join atomic_10305 atomic_10306
private theorem within_080_063_agreement : RowAgreement 10304 10306 := RowAgreement.join atomic_10305_agreement atomic_10306_agreement
private def within_080_064 : CountSegment 1 10 13120 10306 10308 4244 4247 := CountSegment.join atomic_10307 atomic_10308
private theorem within_080_064_agreement : RowAgreement 10306 10308 := RowAgreement.join atomic_10307_agreement atomic_10308_agreement
private def within_080_065 : CountSegment 1 10 13120 10304 10308 4244 4247 := CountSegment.join within_080_063 within_080_064
private theorem within_080_065_agreement : RowAgreement 10304 10308 := RowAgreement.join within_080_063_agreement within_080_064_agreement
private def within_080_066 : CountSegment 1 10 13120 10308 10310 4247 4248 := CountSegment.join atomic_10309 atomic_10310
private theorem within_080_066_agreement : RowAgreement 10308 10310 := RowAgreement.join atomic_10309_agreement atomic_10310_agreement
private def within_080_067 : CountSegment 1 10 13120 10310 10312 4248 4249 := CountSegment.join atomic_10311 atomic_10312
private theorem within_080_067_agreement : RowAgreement 10310 10312 := RowAgreement.join atomic_10311_agreement atomic_10312_agreement
private def within_080_068 : CountSegment 1 10 13120 10308 10312 4247 4249 := CountSegment.join within_080_066 within_080_067
private theorem within_080_068_agreement : RowAgreement 10308 10312 := RowAgreement.join within_080_066_agreement within_080_067_agreement
private def within_080_069 : CountSegment 1 10 13120 10304 10312 4244 4249 := CountSegment.join within_080_065 within_080_068
private theorem within_080_069_agreement : RowAgreement 10304 10312 := RowAgreement.join within_080_065_agreement within_080_068_agreement
private def within_080_070 : CountSegment 1 10 13120 10312 10314 4249 4250 := CountSegment.join atomic_10313 atomic_10314
private theorem within_080_070_agreement : RowAgreement 10312 10314 := RowAgreement.join atomic_10313_agreement atomic_10314_agreement
private def within_080_071 : CountSegment 1 10 13120 10314 10316 4250 4252 := CountSegment.join atomic_10315 atomic_10316
private theorem within_080_071_agreement : RowAgreement 10314 10316 := RowAgreement.join atomic_10315_agreement atomic_10316_agreement
private def within_080_072 : CountSegment 1 10 13120 10312 10316 4249 4252 := CountSegment.join within_080_070 within_080_071
private theorem within_080_072_agreement : RowAgreement 10312 10316 := RowAgreement.join within_080_070_agreement within_080_071_agreement
private def within_080_073 : CountSegment 1 10 13120 10316 10318 4252 4253 := CountSegment.join atomic_10317 atomic_10318
private theorem within_080_073_agreement : RowAgreement 10316 10318 := RowAgreement.join atomic_10317_agreement atomic_10318_agreement
private def within_080_074 : CountSegment 1 10 13120 10318 10320 4253 4253 := CountSegment.join atomic_10319 atomic_10320
private theorem within_080_074_agreement : RowAgreement 10318 10320 := RowAgreement.join atomic_10319_agreement atomic_10320_agreement
private def within_080_075 : CountSegment 1 10 13120 10316 10320 4252 4253 := CountSegment.join within_080_073 within_080_074
private theorem within_080_075_agreement : RowAgreement 10316 10320 := RowAgreement.join within_080_073_agreement within_080_074_agreement
private def within_080_076 : CountSegment 1 10 13120 10312 10320 4249 4253 := CountSegment.join within_080_072 within_080_075
private theorem within_080_076_agreement : RowAgreement 10312 10320 := RowAgreement.join within_080_072_agreement within_080_075_agreement
private def within_080_077 : CountSegment 1 10 13120 10304 10320 4244 4253 := CountSegment.join within_080_069 within_080_076
private theorem within_080_077_agreement : RowAgreement 10304 10320 := RowAgreement.join within_080_069_agreement within_080_076_agreement
private def within_080_078 : CountSegment 1 10 13120 10320 10322 4253 4254 := CountSegment.join atomic_10321 atomic_10322
private theorem within_080_078_agreement : RowAgreement 10320 10322 := RowAgreement.join atomic_10321_agreement atomic_10322_agreement
private def within_080_079 : CountSegment 1 10 13120 10322 10324 4254 4254 := CountSegment.join atomic_10323 atomic_10324
private theorem within_080_079_agreement : RowAgreement 10322 10324 := RowAgreement.join atomic_10323_agreement atomic_10324_agreement
private def within_080_080 : CountSegment 1 10 13120 10320 10324 4253 4254 := CountSegment.join within_080_078 within_080_079
private theorem within_080_080_agreement : RowAgreement 10320 10324 := RowAgreement.join within_080_078_agreement within_080_079_agreement
private def within_080_081 : CountSegment 1 10 13120 10324 10326 4254 4254 := CountSegment.join atomic_10325 atomic_10326
private theorem within_080_081_agreement : RowAgreement 10324 10326 := RowAgreement.join atomic_10325_agreement atomic_10326_agreement
private def within_080_082 : CountSegment 1 10 13120 10326 10328 4254 4257 := CountSegment.join atomic_10327 atomic_10328
private theorem within_080_082_agreement : RowAgreement 10326 10328 := RowAgreement.join atomic_10327_agreement atomic_10328_agreement
private def within_080_083 : CountSegment 1 10 13120 10324 10328 4254 4257 := CountSegment.join within_080_081 within_080_082
private theorem within_080_083_agreement : RowAgreement 10324 10328 := RowAgreement.join within_080_081_agreement within_080_082_agreement
private def within_080_084 : CountSegment 1 10 13120 10320 10328 4253 4257 := CountSegment.join within_080_080 within_080_083
private theorem within_080_084_agreement : RowAgreement 10320 10328 := RowAgreement.join within_080_080_agreement within_080_083_agreement
private def within_080_085 : CountSegment 1 10 13120 10328 10330 4257 4257 := CountSegment.join atomic_10329 atomic_10330
private theorem within_080_085_agreement : RowAgreement 10328 10330 := RowAgreement.join atomic_10329_agreement atomic_10330_agreement
private def within_080_086 : CountSegment 1 10 13120 10330 10332 4257 4257 := CountSegment.join atomic_10331 atomic_10332
private theorem within_080_086_agreement : RowAgreement 10330 10332 := RowAgreement.join atomic_10331_agreement atomic_10332_agreement
private def within_080_087 : CountSegment 1 10 13120 10328 10332 4257 4257 := CountSegment.join within_080_085 within_080_086
private theorem within_080_087_agreement : RowAgreement 10328 10332 := RowAgreement.join within_080_085_agreement within_080_086_agreement
private def within_080_088 : CountSegment 1 10 13120 10332 10334 4257 4257 := CountSegment.join atomic_10333 atomic_10334
private theorem within_080_088_agreement : RowAgreement 10332 10334 := RowAgreement.join atomic_10333_agreement atomic_10334_agreement
private def within_080_089 : CountSegment 1 10 13120 10334 10336 4257 4259 := CountSegment.join atomic_10335 atomic_10336
private theorem within_080_089_agreement : RowAgreement 10334 10336 := RowAgreement.join atomic_10335_agreement atomic_10336_agreement
private def within_080_090 : CountSegment 1 10 13120 10332 10336 4257 4259 := CountSegment.join within_080_088 within_080_089
private theorem within_080_090_agreement : RowAgreement 10332 10336 := RowAgreement.join within_080_088_agreement within_080_089_agreement
private def within_080_091 : CountSegment 1 10 13120 10328 10336 4257 4259 := CountSegment.join within_080_087 within_080_090
private theorem within_080_091_agreement : RowAgreement 10328 10336 := RowAgreement.join within_080_087_agreement within_080_090_agreement
private def within_080_092 : CountSegment 1 10 13120 10320 10336 4253 4259 := CountSegment.join within_080_084 within_080_091
private theorem within_080_092_agreement : RowAgreement 10320 10336 := RowAgreement.join within_080_084_agreement within_080_091_agreement
private def within_080_093 : CountSegment 1 10 13120 10304 10336 4244 4259 := CountSegment.join within_080_077 within_080_092
private theorem within_080_093_agreement : RowAgreement 10304 10336 := RowAgreement.join within_080_077_agreement within_080_092_agreement
private def within_080_094 : CountSegment 1 10 13120 10336 10338 4259 4259 := CountSegment.join atomic_10337 atomic_10338
private theorem within_080_094_agreement : RowAgreement 10336 10338 := RowAgreement.join atomic_10337_agreement atomic_10338_agreement
private def within_080_095 : CountSegment 1 10 13120 10338 10340 4259 4259 := CountSegment.join atomic_10339 atomic_10340
private theorem within_080_095_agreement : RowAgreement 10338 10340 := RowAgreement.join atomic_10339_agreement atomic_10340_agreement
private def within_080_096 : CountSegment 1 10 13120 10336 10340 4259 4259 := CountSegment.join within_080_094 within_080_095
private theorem within_080_096_agreement : RowAgreement 10336 10340 := RowAgreement.join within_080_094_agreement within_080_095_agreement
private def within_080_097 : CountSegment 1 10 13120 10340 10342 4259 4261 := CountSegment.join atomic_10341 atomic_10342
private theorem within_080_097_agreement : RowAgreement 10340 10342 := RowAgreement.join atomic_10341_agreement atomic_10342_agreement
private def within_080_098 : CountSegment 1 10 13120 10342 10344 4261 4262 := CountSegment.join atomic_10343 atomic_10344
private theorem within_080_098_agreement : RowAgreement 10342 10344 := RowAgreement.join atomic_10343_agreement atomic_10344_agreement
private def within_080_099 : CountSegment 1 10 13120 10340 10344 4259 4262 := CountSegment.join within_080_097 within_080_098
private theorem within_080_099_agreement : RowAgreement 10340 10344 := RowAgreement.join within_080_097_agreement within_080_098_agreement
private def within_080_100 : CountSegment 1 10 13120 10336 10344 4259 4262 := CountSegment.join within_080_096 within_080_099
private theorem within_080_100_agreement : RowAgreement 10336 10344 := RowAgreement.join within_080_096_agreement within_080_099_agreement
private def within_080_101 : CountSegment 1 10 13120 10344 10346 4262 4262 := CountSegment.join atomic_10345 atomic_10346
private theorem within_080_101_agreement : RowAgreement 10344 10346 := RowAgreement.join atomic_10345_agreement atomic_10346_agreement
private def within_080_102 : CountSegment 1 10 13120 10346 10348 4262 4262 := CountSegment.join atomic_10347 atomic_10348
private theorem within_080_102_agreement : RowAgreement 10346 10348 := RowAgreement.join atomic_10347_agreement atomic_10348_agreement
private def within_080_103 : CountSegment 1 10 13120 10344 10348 4262 4262 := CountSegment.join within_080_101 within_080_102
private theorem within_080_103_agreement : RowAgreement 10344 10348 := RowAgreement.join within_080_101_agreement within_080_102_agreement
private def within_080_104 : CountSegment 1 10 13120 10348 10350 4262 4263 := CountSegment.join atomic_10349 atomic_10350
private theorem within_080_104_agreement : RowAgreement 10348 10350 := RowAgreement.join atomic_10349_agreement atomic_10350_agreement
private def within_080_105 : CountSegment 1 10 13120 10350 10352 4263 4263 := CountSegment.join atomic_10351 atomic_10352
private theorem within_080_105_agreement : RowAgreement 10350 10352 := RowAgreement.join atomic_10351_agreement atomic_10352_agreement
private def within_080_106 : CountSegment 1 10 13120 10348 10352 4262 4263 := CountSegment.join within_080_104 within_080_105
private theorem within_080_106_agreement : RowAgreement 10348 10352 := RowAgreement.join within_080_104_agreement within_080_105_agreement
private def within_080_107 : CountSegment 1 10 13120 10344 10352 4262 4263 := CountSegment.join within_080_103 within_080_106
private theorem within_080_107_agreement : RowAgreement 10344 10352 := RowAgreement.join within_080_103_agreement within_080_106_agreement
private def within_080_108 : CountSegment 1 10 13120 10336 10352 4259 4263 := CountSegment.join within_080_100 within_080_107
private theorem within_080_108_agreement : RowAgreement 10336 10352 := RowAgreement.join within_080_100_agreement within_080_107_agreement
private def within_080_109 : CountSegment 1 10 13120 10352 10354 4263 4265 := CountSegment.join atomic_10353 atomic_10354
private theorem within_080_109_agreement : RowAgreement 10352 10354 := RowAgreement.join atomic_10353_agreement atomic_10354_agreement
private def within_080_110 : CountSegment 1 10 13120 10354 10356 4265 4265 := CountSegment.join atomic_10355 atomic_10356
private theorem within_080_110_agreement : RowAgreement 10354 10356 := RowAgreement.join atomic_10355_agreement atomic_10356_agreement
private def within_080_111 : CountSegment 1 10 13120 10352 10356 4263 4265 := CountSegment.join within_080_109 within_080_110
private theorem within_080_111_agreement : RowAgreement 10352 10356 := RowAgreement.join within_080_109_agreement within_080_110_agreement
private def within_080_112 : CountSegment 1 10 13120 10356 10358 4265 4265 := CountSegment.join atomic_10357 atomic_10358
private theorem within_080_112_agreement : RowAgreement 10356 10358 := RowAgreement.join atomic_10357_agreement atomic_10358_agreement
private def within_080_113 : CountSegment 1 10 13120 10358 10360 4265 4265 := CountSegment.join atomic_10359 atomic_10360
private theorem within_080_113_agreement : RowAgreement 10358 10360 := RowAgreement.join atomic_10359_agreement atomic_10360_agreement
private def within_080_114 : CountSegment 1 10 13120 10356 10360 4265 4265 := CountSegment.join within_080_112 within_080_113
private theorem within_080_114_agreement : RowAgreement 10356 10360 := RowAgreement.join within_080_112_agreement within_080_113_agreement
private def within_080_115 : CountSegment 1 10 13120 10352 10360 4263 4265 := CountSegment.join within_080_111 within_080_114
private theorem within_080_115_agreement : RowAgreement 10352 10360 := RowAgreement.join within_080_111_agreement within_080_114_agreement
private def within_080_116 : CountSegment 1 10 13120 10360 10362 4265 4265 := CountSegment.join atomic_10361 atomic_10362
private theorem within_080_116_agreement : RowAgreement 10360 10362 := RowAgreement.join atomic_10361_agreement atomic_10362_agreement
private def within_080_117 : CountSegment 1 10 13120 10362 10364 4265 4265 := CountSegment.join atomic_10363 atomic_10364
private theorem within_080_117_agreement : RowAgreement 10362 10364 := RowAgreement.join atomic_10363_agreement atomic_10364_agreement
private def within_080_118 : CountSegment 1 10 13120 10360 10364 4265 4265 := CountSegment.join within_080_116 within_080_117
private theorem within_080_118_agreement : RowAgreement 10360 10364 := RowAgreement.join within_080_116_agreement within_080_117_agreement
private def within_080_119 : CountSegment 1 10 13120 10364 10366 4265 4265 := CountSegment.join atomic_10365 atomic_10366
private theorem within_080_119_agreement : RowAgreement 10364 10366 := RowAgreement.join atomic_10365_agreement atomic_10366_agreement
private def within_080_120 : CountSegment 1 10 13120 10366 10368 4265 4266 := CountSegment.join atomic_10367 atomic_10368
private theorem within_080_120_agreement : RowAgreement 10366 10368 := RowAgreement.join atomic_10367_agreement atomic_10368_agreement
private def within_080_121 : CountSegment 1 10 13120 10364 10368 4265 4266 := CountSegment.join within_080_119 within_080_120
private theorem within_080_121_agreement : RowAgreement 10364 10368 := RowAgreement.join within_080_119_agreement within_080_120_agreement
private def within_080_122 : CountSegment 1 10 13120 10360 10368 4265 4266 := CountSegment.join within_080_118 within_080_121
private theorem within_080_122_agreement : RowAgreement 10360 10368 := RowAgreement.join within_080_118_agreement within_080_121_agreement
private def within_080_123 : CountSegment 1 10 13120 10352 10368 4263 4266 := CountSegment.join within_080_115 within_080_122
private theorem within_080_123_agreement : RowAgreement 10352 10368 := RowAgreement.join within_080_115_agreement within_080_122_agreement
private def within_080_124 : CountSegment 1 10 13120 10336 10368 4259 4266 := CountSegment.join within_080_108 within_080_123
private theorem within_080_124_agreement : RowAgreement 10336 10368 := RowAgreement.join within_080_108_agreement within_080_123_agreement
private def within_080_125 : CountSegment 1 10 13120 10304 10368 4244 4266 := CountSegment.join within_080_093 within_080_124
private theorem within_080_125_agreement : RowAgreement 10304 10368 := RowAgreement.join within_080_093_agreement within_080_124_agreement
private def within_080_126 : CountSegment 1 10 13120 10240 10368 4224 4266 := CountSegment.join within_080_062 within_080_125
private theorem within_080_126_agreement : RowAgreement 10240 10368 := RowAgreement.join within_080_062_agreement within_080_125_agreement
def segment_080 : CountSegment 1 10 13120 10240 10368 4224 4266 := within_080_126
theorem agreement_080 : RowAgreement 10240 10368 := within_080_126_agreement
end ProjectEulerSolutions.P58Numerical
