import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_7041 : RowCertificate 7041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7041 1) := by change ¬ Nat.Prime 198316807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15255139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7041 2) := by change ¬ Nat.Prime 198302725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39660545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7041 3) := by change ¬ Nat.Prime 198288643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28326949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7041 : row 7041 = 0 := (row_7041.checked).trans rfl
theorem model_count_7041 : modelRow 7041 = 0 := by decide
theorem modelrow_agrees_7041 : modelRow 7041 = row 7041 := model_count_7041.trans row_count_7041.symm
def row_7042 : RowCertificate 7042 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7042 1) := by change Nat.Prime 198373141; exact ProjectEulerSolutions.P58PrimeNodes.p_198373141.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7042 2) := by change ¬ Nat.Prime 198359057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15258389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7042 3) := by change ¬ Nat.Prime 198344973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66114991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7042 : row 7042 = 1 := (row_7042.checked).trans rfl
theorem model_count_7042 : modelRow 7042 = 1 := by decide
theorem modelrow_agrees_7042 : modelRow 7042 = row 7042 := model_count_7042.trans row_count_7042.symm
def row_7043 : RowCertificate 7043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7043 1) := by change ¬ Nat.Prime 198429483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66143161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7043 2) := by change Nat.Prime 198415397; exact ProjectEulerSolutions.P58PrimeNodes.p_198415397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7043 3) := by change ¬ Nat.Prime 198401311; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 2511409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7043 : row 7043 = 1 := (row_7043.checked).trans rfl
theorem model_count_7043 : modelRow 7043 = 1 := by decide
theorem modelrow_agrees_7043 : modelRow 7043 = row 7043 := model_count_7043.trans row_count_7043.symm
def row_7044 : RowCertificate 7044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7044 1) := by change ¬ Nat.Prime 198485833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28355119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7044 2) := by change ¬ Nat.Prime 198471745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39694349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7044 3) := by change Nat.Prime 198457657; exact ProjectEulerSolutions.P58PrimeNodes.p_198457657.prime
    exact decide_eq_true prime
theorem row_count_7044 : row 7044 = 1 := (row_7044.checked).trans rfl
theorem model_count_7044 : modelRow 7044 = 1 := by decide
theorem modelrow_agrees_7044 : modelRow 7044 = row 7044 := model_count_7044.trans row_count_7044.symm
def row_7045 : RowCertificate 7045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7045 1) := by change ¬ Nat.Prime 198542191; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10449589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7045 2) := by change ¬ Nat.Prime 198528101; exact PrattCertificates.composite_of_product (factor := 4861) (cofactor := 40841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7045 3) := by change ¬ Nat.Prime 198514011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66171337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7045 : row 7045 = 0 := (row_7045.checked).trans rfl
theorem model_count_7045 : modelRow 7045 = 0 := by decide
theorem modelrow_agrees_7045 : modelRow 7045 = row 7045 := model_count_7045.trans row_count_7045.symm
def row_7046 : RowCertificate 7046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7046 1) := by change ¬ Nat.Prime 198598557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66199519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7046 2) := by change ¬ Nat.Prime 198584465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39716893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7046 3) := by change Nat.Prime 198570373; exact ProjectEulerSolutions.P58PrimeNodes.p_198570373.prime
    exact decide_eq_true prime
theorem row_count_7046 : row 7046 = 1 := (row_7046.checked).trans rfl
theorem model_count_7046 : modelRow 7046 = 1 := by decide
theorem modelrow_agrees_7046 : modelRow 7046 = row 7046 := model_count_7046.trans row_count_7046.symm
def row_7047 : RowCertificate 7047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7047 1) := by change ¬ Nat.Prime 198654931; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 273253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7047 2) := by change ¬ Nat.Prime 198640837; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1822393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7047 3) := by change ¬ Nat.Prime 198626743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28375249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7047 : row 7047 = 0 := (row_7047.checked).trans rfl
theorem model_count_7047 : modelRow 7047 = 0 := by decide
theorem modelrow_agrees_7047 : modelRow 7047 = row 7047 := model_count_7047.trans row_count_7047.symm
def row_7048 : RowCertificate 7048 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7048 1) := by change Nat.Prime 198711313; exact ProjectEulerSolutions.P58PrimeNodes.p_198711313.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7048 2) := by change ¬ Nat.Prime 198697217; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 2232553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7048 3) := by change ¬ Nat.Prime 198683121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66227707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7048 : row 7048 = 1 := (row_7048.checked).trans rfl
theorem model_count_7048 : modelRow 7048 = 1 := by decide
theorem modelrow_agrees_7048 : modelRow 7048 = row 7048 := model_count_7048.trans row_count_7048.symm
def row_7049 : RowCertificate 7049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7049 1) := by change ¬ Nat.Prime 198767703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66255901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7049 2) := by change ¬ Nat.Prime 198753605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39750721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7049 3) := by change ¬ Nat.Prime 198739507; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4621849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7049 : row 7049 = 0 := (row_7049.checked).trans rfl
theorem model_count_7049 : modelRow 7049 = 0 := by decide
theorem modelrow_agrees_7049 : modelRow 7049 = row 7049 := model_count_7049.trans row_count_7049.symm
def row_7050 : RowCertificate 7050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7050 1) := by change ¬ Nat.Prime 198824101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28403443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7050 2) := by change ¬ Nat.Prime 198810001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15293077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7050 3) := by change ¬ Nat.Prime 198795901; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6412771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7050 : row 7050 = 0 := (row_7050.checked).trans rfl
theorem model_count_7050 : modelRow 7050 = 0 := by decide
theorem modelrow_agrees_7050 : modelRow 7050 = row 7050 := model_count_7050.trans row_count_7050.symm
def row_7051 : RowCertificate 7051 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7051 1) := by change ¬ Nat.Prime 198880507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28411501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7051 2) := by change ¬ Nat.Prime 198866405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39773281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7051 3) := by change ¬ Nat.Prime 198852303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66284101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7051 : row 7051 = 0 := (row_7051.checked).trans rfl
theorem model_count_7051 : modelRow 7051 = 0 := by decide
theorem modelrow_agrees_7051 : modelRow 7051 = row 7051 := model_count_7051.trans row_count_7051.symm
def row_7052 : RowCertificate 7052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7052 1) := by change ¬ Nat.Prime 198936921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66312307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7052 2) := by change Nat.Prime 198922817; exact ProjectEulerSolutions.P58PrimeNodes.p_198922817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7052 3) := by change ¬ Nat.Prime 198908713; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 868597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7052 : row 7052 = 1 := (row_7052.checked).trans rfl
theorem model_count_7052 : modelRow 7052 = 1 := by decide
theorem modelrow_agrees_7052 : modelRow 7052 = row 7052 := model_count_7052.trans row_count_7052.symm
def row_7053 : RowCertificate 7053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7053 1) := by change ¬ Nat.Prime 198993343; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1825627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7053 2) := by change ¬ Nat.Prime 198979237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11704661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7053 3) := by change ¬ Nat.Prime 198965131; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10471849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7053 : row 7053 = 0 := (row_7053.checked).trans rfl
theorem model_count_7053 : modelRow 7053 = 0 := by decide
theorem modelrow_agrees_7053 : modelRow 7053 = row 7053 := model_count_7053.trans row_count_7053.symm
def row_7054 : RowCertificate 7054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7054 1) := by change ¬ Nat.Prime 199049773; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15311521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7054 2) := by change ¬ Nat.Prime 199035665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39807133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7054 3) := by change ¬ Nat.Prime 199021557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66340519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7054 : row 7054 = 0 := (row_7054.checked).trans rfl
theorem model_count_7054 : modelRow 7054 = 0 := by decide
theorem modelrow_agrees_7054 : modelRow 7054 = row 7054 := model_count_7054.trans row_count_7054.symm
def row_7055 : RowCertificate 7055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7055 1) := by change ¬ Nat.Prime 199106211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66368737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7055 2) := by change ¬ Nat.Prime 199092101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15314777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7055 3) := by change ¬ Nat.Prime 199077991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28439713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7055 : row 7055 = 0 := (row_7055.checked).trans rfl
theorem model_count_7055 : modelRow 7055 = 0 := by decide
theorem modelrow_agrees_7055 : modelRow 7055 = row 7055 := model_count_7055.trans row_count_7055.symm
def row_7056 : RowCertificate 7056 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7056 1) := by change Nat.Prime 199162657; exact ProjectEulerSolutions.P58PrimeNodes.p_199162657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7056 2) := by change ¬ Nat.Prime 199148545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39829709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7056 3) := by change ¬ Nat.Prime 199134433; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 399067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7056 : row 7056 = 1 := (row_7056.checked).trans rfl
theorem model_count_7056 : modelRow 7056 = 1 := by decide
theorem modelrow_agrees_7056 : modelRow 7056 = row 7056 := model_count_7056.trans row_count_7056.symm
def row_7057 : RowCertificate 7057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7057 1) := by change ¬ Nat.Prime 199219111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28459873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7057 2) := by change ¬ Nat.Prime 199204997; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11717941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7057 3) := by change ¬ Nat.Prime 199190883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66396961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7057 : row 7057 = 0 := (row_7057.checked).trans rfl
theorem model_count_7057 : modelRow 7057 = 0 := by decide
theorem modelrow_agrees_7057 : modelRow 7057 = row 7057 := model_count_7057.trans row_count_7057.symm
def row_7058 : RowCertificate 7058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7058 1) := by change ¬ Nat.Prime 199275573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66425191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7058 2) := by change ¬ Nat.Prime 199261457; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 2729609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7058 3) := by change ¬ Nat.Prime 199247341; exact PrattCertificates.composite_of_product (factor := 5023) (cofactor := 39667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7058 : row 7058 = 0 := (row_7058.checked).trans rfl
theorem model_count_7058 : modelRow 7058 = 0 := by decide
theorem modelrow_agrees_7058 : modelRow 7058 = row 7058 := model_count_7058.trans row_count_7058.symm
def row_7059 : RowCertificate 7059 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7059 1) := by change Nat.Prime 199332043; exact ProjectEulerSolutions.P58PrimeNodes.p_199332043.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7059 2) := by change ¬ Nat.Prime 199317925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39863585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7059 3) := by change ¬ Nat.Prime 199303807; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 2522833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7059 : row 7059 = 1 := (row_7059.checked).trans rfl
theorem model_count_7059 : modelRow 7059 = 1 := by decide
theorem modelrow_agrees_7059 : modelRow 7059 = row 7059 := model_count_7059.trans row_count_7059.symm
def row_7060 : RowCertificate 7060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7060 1) := by change ¬ Nat.Prime 199388521; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 735751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7060 2) := by change Nat.Prime 199374401; exact ProjectEulerSolutions.P58PrimeNodes.p_199374401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7060 3) := by change ¬ Nat.Prime 199360281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66453427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7060 : row 7060 = 1 := (row_7060.checked).trans rfl
theorem model_count_7060 : modelRow 7060 = 1 := by decide
theorem modelrow_agrees_7060 : modelRow 7060 = row 7060 := model_count_7060.trans row_count_7060.symm
def row_7061 : RowCertificate 7061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7061 1) := by change ¬ Nat.Prime 199445007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66481669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7061 2) := by change ¬ Nat.Prime 199430885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39886177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7061 3) := by change ¬ Nat.Prime 199416763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28488109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7061 : row 7061 = 0 := (row_7061.checked).trans rfl
theorem model_count_7061 : modelRow 7061 = 0 := by decide
theorem modelrow_agrees_7061 : modelRow 7061 = row 7061 := model_count_7061.trans row_count_7061.symm
def row_7062 : RowCertificate 7062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7062 1) := by change ¬ Nat.Prime 199501501; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10500079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7062 2) := by change ¬ Nat.Prime 199487377; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 856169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7062 3) := by change ¬ Nat.Prime 199473253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28496179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7062 : row 7062 = 0 := (row_7062.checked).trans rfl
theorem model_count_7062 : modelRow 7062 = 0 := by decide
theorem modelrow_agrees_7062 : modelRow 7062 = row 7062 := model_count_7062.trans row_count_7062.symm
def row_7063 : RowCertificate 7063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7063 1) := by change ¬ Nat.Prime 199558003; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2057299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7063 2) := by change ¬ Nat.Prime 199543877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15349529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7063 3) := by change ¬ Nat.Prime 199529751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66509917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7063 : row 7063 = 0 := (row_7063.checked).trans rfl
theorem model_count_7063 : modelRow 7063 = 0 := by decide
theorem modelrow_agrees_7063 : modelRow 7063 = row 7063 := model_count_7063.trans row_count_7063.symm
def row_7064 : RowCertificate 7064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7064 1) := by change ¬ Nat.Prime 199614513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66538171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7064 2) := by change ¬ Nat.Prime 199600385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39920077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7064 3) := by change ¬ Nat.Prime 199586257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15352789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7064 : row 7064 = 0 := (row_7064.checked).trans rfl
theorem model_count_7064 : modelRow 7064 = 0 := by decide
theorem modelrow_agrees_7064 : modelRow 7064 = row 7064 := model_count_7064.trans row_count_7064.symm
def row_7065 : RowCertificate 7065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7065 1) := by change ¬ Nat.Prime 199671031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28524433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7065 2) := by change ¬ Nat.Prime 199656901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 1976801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7065 3) := by change Nat.Prime 199642771; exact ProjectEulerSolutions.P58PrimeNodes.p_199642771.prime
    exact decide_eq_true prime
theorem row_count_7065 : row 7065 = 1 := (row_7065.checked).trans rfl
theorem model_count_7065 : modelRow 7065 = 1 := by decide
theorem modelrow_agrees_7065 : modelRow 7065 = row 7065 := model_count_7065.trans row_count_7065.symm
def row_7066 : RowCertificate 7066 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7066 1) := by change Nat.Prime 199727557; exact ProjectEulerSolutions.P58PrimeNodes.p_199727557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7066 2) := by change ¬ Nat.Prime 199713425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39942685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7066 3) := by change ¬ Nat.Prime 199699293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66566431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7066 : row 7066 = 1 := (row_7066.checked).trans rfl
theorem model_count_7066 : modelRow 7066 = 1 := by decide
theorem modelrow_agrees_7066 : modelRow 7066 = row 7066 := model_count_7066.trans row_count_7066.symm
def row_7067 : RowCertificate 7067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7067 1) := by change ¬ Nat.Prime 199784091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66594697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7067 2) := by change ¬ Nat.Prime 199769957; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 325889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7067 3) := by change Nat.Prime 199755823; exact ProjectEulerSolutions.P58PrimeNodes.p_199755823.prime
    exact decide_eq_true prime
theorem row_count_7067 : row 7067 = 1 := (row_7067.checked).trans rfl
theorem model_count_7067 : modelRow 7067 = 1 := by decide
theorem modelrow_agrees_7067 : modelRow 7067 = row 7067 := model_count_7067.trans row_count_7067.symm
def row_7068 : RowCertificate 7068 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7068 1) := by change Nat.Prime 199840633; exact ProjectEulerSolutions.P58PrimeNodes.p_199840633.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7068 2) := by change ¬ Nat.Prime 199826497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15371269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7068 3) := by change ¬ Nat.Prime 199812361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28544623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7068 : row 7068 = 1 := (row_7068.checked).trans rfl
theorem model_count_7068 : modelRow 7068 = 1 := by decide
theorem modelrow_agrees_7068 : modelRow 7068 = row 7068 := model_count_7068.trans row_count_7068.symm
def row_7069 : RowCertificate 7069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7069 1) := by change ¬ Nat.Prime 199897183; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3277003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7069 2) := by change ¬ Nat.Prime 199883045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39976609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7069 3) := by change ¬ Nat.Prime 199868907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66622969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7069 : row 7069 = 0 := (row_7069.checked).trans rfl
theorem model_count_7069 : modelRow 7069 = 0 := by decide
theorem modelrow_agrees_7069 : modelRow 7069 = row 7069 := model_count_7069.trans row_count_7069.symm
def row_7070 : RowCertificate 7070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7070 1) := by change ¬ Nat.Prime 199953741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66651247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7070 2) := by change ¬ Nat.Prime 199939601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11761153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7070 3) := by change ¬ Nat.Prime 199925461; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 651223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7070 : row 7070 = 0 := (row_7070.checked).trans rfl
theorem model_count_7070 : modelRow 7070 = 0 := by decide
theorem modelrow_agrees_7070 : modelRow 7070 = row 7070 := model_count_7070.trans row_count_7070.symm
def row_7071 : RowCertificate 7071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7071 1) := by change ¬ Nat.Prime 200010307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28572901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7071 2) := by change ¬ Nat.Prime 199996165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39999233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7071 3) := by change ¬ Nat.Prime 199982023; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6451033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7071 : row 7071 = 0 := (row_7071.checked).trans rfl
theorem model_count_7071 : modelRow 7071 = 0 := by decide
theorem modelrow_agrees_7071 : modelRow 7071 = row 7071 := model_count_7071.trans row_count_7071.symm
def row_7072 : RowCertificate 7072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7072 1) := by change ¬ Nat.Prime 200066881; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28580983) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7072 2) := by change Nat.Prime 200052737; exact ProjectEulerSolutions.P58PrimeNodes.p_200052737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7072 3) := by change ¬ Nat.Prime 200038593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66679531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7072 : row 7072 = 1 := (row_7072.checked).trans rfl
theorem model_count_7072 : modelRow 7072 = 1 := by decide
theorem modelrow_agrees_7072 : modelRow 7072 = row 7072 := model_count_7072.trans row_count_7072.symm
def row_7073 : RowCertificate 7073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7073 1) := by change ¬ Nat.Prime 200123463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66707821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7073 2) := by change ¬ Nat.Prime 200109317; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 187193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7073 3) := by change Nat.Prime 200095171; exact ProjectEulerSolutions.P58PrimeNodes.p_200095171.prime
    exact decide_eq_true prime
theorem row_count_7073 : row 7073 = 1 := (row_7073.checked).trans rfl
theorem model_count_7073 : modelRow 7073 = 1 := by decide
theorem modelrow_agrees_7073 : modelRow 7073 = row 7073 := model_count_7073.trans row_count_7073.symm
def row_7074 : RowCertificate 7074 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7074 1) := by change Nat.Prime 200180053; exact ProjectEulerSolutions.P58PrimeNodes.p_200180053.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7074 2) := by change ¬ Nat.Prime 200165905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40033181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7074 3) := by change ¬ Nat.Prime 200151757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15396289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7074 : row 7074 = 1 := (row_7074.checked).trans rfl
theorem model_count_7074 : modelRow 7074 = 1 := by decide
theorem modelrow_agrees_7074 : modelRow 7074 = row 7074 := model_count_7074.trans row_count_7074.symm
def row_7075 : RowCertificate 7075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7075 1) := by change ¬ Nat.Prime 200236651; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 125383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7075 2) := by change ¬ Nat.Prime 200222501; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 1982401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7075 3) := by change ¬ Nat.Prime 200208351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66736117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7075 : row 7075 = 0 := (row_7075.checked).trans rfl
theorem model_count_7075 : modelRow 7075 = 0 := by decide
theorem modelrow_agrees_7075 : modelRow 7075 = row 7075 := model_count_7075.trans row_count_7075.symm
def row_7076 : RowCertificate 7076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7076 1) := by change ¬ Nat.Prime 200293257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66764419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7076 2) := by change ¬ Nat.Prime 200279105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40055821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7076 3) := by change ¬ Nat.Prime 200264953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28609279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7076 : row 7076 = 0 := (row_7076.checked).trans rfl
theorem model_count_7076 : modelRow 7076 = 0 := by decide
theorem modelrow_agrees_7076 : modelRow 7076 = row 7076 := model_count_7076.trans row_count_7076.symm
def row_7077 : RowCertificate 7077 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7077 1) := by change Nat.Prime 200349871; exact ProjectEulerSolutions.P58PrimeNodes.p_200349871.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7077 2) := by change ¬ Nat.Prime 200335717; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4886237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7077 3) := by change ¬ Nat.Prime 200321563; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15409351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7077 : row 7077 = 1 := (row_7077.checked).trans rfl
theorem model_count_7077 : modelRow 7077 = 1 := by decide
theorem modelrow_agrees_7077 : modelRow 7077 = row 7077 := model_count_7077.trans row_count_7077.symm
def row_7078 : RowCertificate 7078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7078 1) := by change ¬ Nat.Prime 200406493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28629499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7078 2) := by change Nat.Prime 200392337; exact ProjectEulerSolutions.P58PrimeNodes.p_200392337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7078 3) := by change ¬ Nat.Prime 200378181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66792727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7078 : row 7078 = 1 := (row_7078.checked).trans rfl
theorem model_count_7078 : modelRow 7078 = 1 := by decide
theorem modelrow_agrees_7078 : modelRow 7078 = row 7078 := model_count_7078.trans row_count_7078.symm
def row_7079 : RowCertificate 7079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7079 1) := by change ¬ Nat.Prime 200463123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66821041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7079 2) := by change ¬ Nat.Prime 200448965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40089793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7079 3) := by change ¬ Nat.Prime 200434807; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1945969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7079 : row 7079 = 0 := (row_7079.checked).trans rfl
theorem model_count_7079 : modelRow 7079 = 0 := by decide
theorem modelrow_agrees_7079 : modelRow 7079 = row 7079 := model_count_7079.trans row_count_7079.symm
def row_7080 : RowCertificate 7080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7080 1) := by change ¬ Nat.Prime 200519761; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15424597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7080 2) := by change ¬ Nat.Prime 200505601; exact PrattCertificates.composite_of_product (factor := 1277) (cofactor := 157013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7080 3) := by change ¬ Nat.Prime 200491441; exact PrattCertificates.composite_of_product (factor := 1789) (cofactor := 112069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7080 : row 7080 = 0 := (row_7080.checked).trans rfl
theorem model_count_7080 : modelRow 7080 = 0 := by decide
theorem modelrow_agrees_7080 : modelRow 7080 = row 7080 := model_count_7080.trans row_count_7080.symm
def row_7081 : RowCertificate 7081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7081 1) := by change ¬ Nat.Prime 200576407; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10556653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7081 2) := by change ¬ Nat.Prime 200562245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40112449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7081 3) := by change ¬ Nat.Prime 200548083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66849361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7081 : row 7081 = 0 := (row_7081.checked).trans rfl
theorem model_count_7081 : modelRow 7081 = 0 := by decide
theorem modelrow_agrees_7081 : modelRow 7081 = row 7081 := model_count_7081.trans row_count_7081.symm
def row_7082 : RowCertificate 7082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7082 1) := by change ¬ Nat.Prime 200633061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66877687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7082 2) := by change ¬ Nat.Prime 200618897; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6917893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7082 3) := by change ¬ Nat.Prime 200604733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28657819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7082 : row 7082 = 0 := (row_7082.checked).trans rfl
theorem model_count_7082 : modelRow 7082 = 0 := by decide
theorem modelrow_agrees_7082 : modelRow 7082 = row 7082 := model_count_7082.trans row_count_7082.symm
def row_7083 : RowCertificate 7083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7083 1) := by change ¬ Nat.Prime 200689723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15437671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7083 2) := by change Nat.Prime 200675557; exact ProjectEulerSolutions.P58PrimeNodes.p_200675557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7083 3) := by change ¬ Nat.Prime 200661391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28665913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7083 : row 7083 = 1 := (row_7083.checked).trans rfl
theorem model_count_7083 : modelRow 7083 = 1 := by decide
theorem modelrow_agrees_7083 : modelRow 7083 = row 7083 := model_count_7083.trans row_count_7083.symm
def row_7084 : RowCertificate 7084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7084 1) := by change ¬ Nat.Prime 200746393; exact PrattCertificates.composite_of_product (factor := 4597) (cofactor := 43669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7084 2) := by change ¬ Nat.Prime 200732225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40146445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7084 3) := by change ¬ Nat.Prime 200718057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66906019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7084 : row 7084 = 0 := (row_7084.checked).trans rfl
theorem model_count_7084 : modelRow 7084 = 0 := by decide
theorem modelrow_agrees_7084 : modelRow 7084 = row 7084 := model_count_7084.trans row_count_7084.symm
def row_7085 : RowCertificate 7085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7085 1) := by change ¬ Nat.Prime 200803071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66934357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7085 2) := by change ¬ Nat.Prime 200788901; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1040357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7085 3) := by change Nat.Prime 200774731; exact ProjectEulerSolutions.P58PrimeNodes.p_200774731.prime
    exact decide_eq_true prime
theorem row_count_7085 : row 7085 = 1 := (row_7085.checked).trans rfl
theorem model_count_7085 : modelRow 7085 = 1 := by decide
theorem modelrow_agrees_7085 : modelRow 7085 = row 7085 := model_count_7085.trans row_count_7085.symm
def row_7086 : RowCertificate 7086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7086 1) := by change ¬ Nat.Prime 200859757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28694251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7086 2) := by change ¬ Nat.Prime 200845585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40169117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7086 3) := by change ¬ Nat.Prime 200831413; exact PrattCertificates.composite_of_product (factor := 607) (cofactor := 330859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7086 : row 7086 = 0 := (row_7086.checked).trans rfl
theorem model_count_7086 : modelRow 7086 = 0 := by decide
theorem modelrow_agrees_7086 : modelRow 7086 = row 7086 := model_count_7086.trans row_count_7086.symm
def row_7087 : RowCertificate 7087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7087 1) := by change ¬ Nat.Prime 200916451; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2998753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7087 2) := by change ¬ Nat.Prime 200902277; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11817781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7087 3) := by change ¬ Nat.Prime 200888103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66962701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7087 : row 7087 = 0 := (row_7087.checked).trans rfl
theorem model_count_7087 : modelRow 7087 = 0 := by decide
theorem modelrow_agrees_7087 : modelRow 7087 = row 7087 := model_count_7087.trans row_count_7087.symm
def row_7088 : RowCertificate 7088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7088 1) := by change ¬ Nat.Prime 200973153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66991051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7088 2) := by change ¬ Nat.Prime 200958977; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 145517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7088 3) := by change Nat.Prime 200944801; exact ProjectEulerSolutions.P58PrimeNodes.p_200944801.prime
    exact decide_eq_true prime
theorem row_count_7088 : row 7088 = 1 := (row_7088.checked).trans rfl
theorem model_count_7088 : modelRow 7088 = 1 := by decide
theorem modelrow_agrees_7088 : modelRow 7088 = row 7088 := model_count_7088.trans row_count_7088.symm
def row_7089 : RowCertificate 7089 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7089 1) := by change Nat.Prime 201029863; exact ProjectEulerSolutions.P58PrimeNodes.p_201029863.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7089 2) := by change ¬ Nat.Prime 201015685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40203137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7089 3) := by change ¬ Nat.Prime 201001507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28714501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7089 : row 7089 = 1 := (row_7089.checked).trans rfl
theorem model_count_7089 : modelRow 7089 = 1 := by decide
theorem modelrow_agrees_7089 : modelRow 7089 = row 7089 := model_count_7089.trans row_count_7089.symm
def row_7090 : RowCertificate 7090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7090 1) := by change ¬ Nat.Prime 201086581; exact PrattCertificates.composite_of_product (factor := 3571) (cofactor := 56311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7090 2) := by change Nat.Prime 201072401; exact ProjectEulerSolutions.P58PrimeNodes.p_201072401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7090 3) := by change ¬ Nat.Prime 201058221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67019407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7090 : row 7090 = 1 := (row_7090.checked).trans rfl
theorem model_count_7090 : modelRow 7090 = 1 := by decide
theorem modelrow_agrees_7090 : modelRow 7090 = row 7090 := model_count_7090.trans row_count_7090.symm
def row_7091 : RowCertificate 7091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7091 1) := by change ¬ Nat.Prime 201143307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67047769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7091 2) := by change ¬ Nat.Prime 201129125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40225825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7091 3) := by change ¬ Nat.Prime 201114943; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10584997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7091 : row 7091 = 0 := (row_7091.checked).trans rfl
theorem model_count_7091 : modelRow 7091 = 0 := by decide
theorem modelrow_agrees_7091 : modelRow 7091 = row 7091 := model_count_7091.trans row_count_7091.symm
def row_7092 : RowCertificate 7092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7092 1) := by change ¬ Nat.Prime 201200041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28742863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7092 2) := by change ¬ Nat.Prime 201185857; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2074081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7092 3) := by change ¬ Nat.Prime 201171673; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4678411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7092 : row 7092 = 0 := (row_7092.checked).trans rfl
theorem model_count_7092 : modelRow 7092 = 0 := by decide
theorem modelrow_agrees_7092 : modelRow 7092 = row 7092 := model_count_7092.trans row_count_7092.symm
def row_7093 : RowCertificate 7093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7093 1) := by change ¬ Nat.Prime 201256783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28750969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7093 2) := by change Nat.Prime 201242597; exact ProjectEulerSolutions.P58PrimeNodes.p_201242597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7093 3) := by change ¬ Nat.Prime 201228411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67076137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7093 : row 7093 = 1 := (row_7093.checked).trans rfl
theorem model_count_7093 : modelRow 7093 = 1 := by decide
theorem modelrow_agrees_7093 : modelRow 7093 = row 7093 := model_count_7093.trans row_count_7093.symm
def row_7094 : RowCertificate 7094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7094 1) := by change ¬ Nat.Prime 201313533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67104511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7094 2) := by change ¬ Nat.Prime 201299345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40259869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7094 3) := by change Nat.Prime 201285157; exact ProjectEulerSolutions.P58PrimeNodes.p_201285157.prime
    exact decide_eq_true prime
theorem row_count_7094 : row 7094 = 1 := (row_7094.checked).trans rfl
theorem model_count_7094 : modelRow 7094 = 1 := by decide
theorem modelrow_agrees_7094 : modelRow 7094 = row 7094 := model_count_7094.trans row_count_7094.symm
def row_7095 : RowCertificate 7095 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7095 1) := by change Nat.Prime 201370291; exact ProjectEulerSolutions.P58PrimeNodes.p_201370291.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7095 2) := by change Nat.Prime 201356101; exact ProjectEulerSolutions.P58PrimeNodes.p_201356101.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7095 3) := by change Nat.Prime 201341911; exact ProjectEulerSolutions.P58PrimeNodes.p_201341911.prime
    exact decide_eq_true prime
theorem row_count_7095 : row 7095 = 3 := (row_7095.checked).trans rfl
theorem model_count_7095 : modelRow 7095 = 3 := by decide
theorem modelrow_agrees_7095 : modelRow 7095 = row 7095 := model_count_7095.trans row_count_7095.symm
def row_7096 : RowCertificate 7096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7096 1) := by change ¬ Nat.Prime 201427057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15494389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7096 2) := by change ¬ Nat.Prime 201412865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40282573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7096 3) := by change ¬ Nat.Prime 201398673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67132891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7096 : row 7096 = 0 := (row_7096.checked).trans rfl
theorem model_count_7096 : modelRow 7096 = 0 := by decide
theorem modelrow_agrees_7096 : modelRow 7096 = row 7096 := model_count_7096.trans row_count_7096.symm
def row_7097 : RowCertificate 7097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7097 1) := by change ¬ Nat.Prime 201483831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67161277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7097 2) := by change Nat.Prime 201469637; exact ProjectEulerSolutions.P58PrimeNodes.p_201469637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7097 3) := by change ¬ Nat.Prime 201455443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28779349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7097 : row 7097 = 1 := (row_7097.checked).trans rfl
theorem model_count_7097 : modelRow 7097 = 1 := by decide
theorem modelrow_agrees_7097 : modelRow 7097 = row 7097 := model_count_7097.trans row_count_7097.symm
def row_7098 : RowCertificate 7098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7098 1) := by change ¬ Nat.Prime 201540613; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4686991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7098 2) := by change Nat.Prime 201526417; exact ProjectEulerSolutions.P58PrimeNodes.p_201526417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7098 3) := by change ¬ Nat.Prime 201512221; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 372481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7098 : row 7098 = 1 := (row_7098.checked).trans rfl
theorem model_count_7098 : modelRow 7098 = 1 := by decide
theorem modelrow_agrees_7098 : modelRow 7098 = row 7098 := model_count_7098.trans row_count_7098.symm
def row_7099 : RowCertificate 7099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7099 1) := by change ¬ Nat.Prime 201597403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28799629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7099 2) := by change ¬ Nat.Prime 201583205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40316641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7099 3) := by change ¬ Nat.Prime 201569007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67189669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7099 : row 7099 = 0 := (row_7099.checked).trans rfl
theorem model_count_7099 : modelRow 7099 = 0 := by decide
theorem modelrow_agrees_7099 : modelRow 7099 = row 7099 := model_count_7099.trans row_count_7099.symm
def row_7100 : RowCertificate 7100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7100 1) := by change ¬ Nat.Prime 201654201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67218067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7100 2) := by change Nat.Prime 201640001; exact ProjectEulerSolutions.P58PrimeNodes.p_201640001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7100 3) := by change ¬ Nat.Prime 201625801; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15509677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7100 : row 7100 = 1 := (row_7100.checked).trans rfl
theorem model_count_7100 : modelRow 7100 = 1 := by decide
theorem modelrow_agrees_7100 : modelRow 7100 = row 7100 := model_count_7100.trans row_count_7100.symm
def row_7101 : RowCertificate 7101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7101 1) := by change ¬ Nat.Prime 201711007; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 262303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7101 2) := by change ¬ Nat.Prime 201696805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40339361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7101 3) := by change ¬ Nat.Prime 201682603; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 609313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7101 : row 7101 = 0 := (row_7101.checked).trans rfl
theorem model_count_7101 : modelRow 7101 = 0 := by decide
theorem modelrow_agrees_7101 : modelRow 7101 = row 7101 := model_count_7101.trans row_count_7101.symm
def row_7102 : RowCertificate 7102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7102 1) := by change ¬ Nat.Prime 201767821; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10619359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7102 2) := by change ¬ Nat.Prime 201753617; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15519509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7102 3) := by change ¬ Nat.Prime 201739413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67246471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7102 : row 7102 = 0 := (row_7102.checked).trans rfl
theorem model_count_7102 : modelRow 7102 = 0 := by decide
theorem modelrow_agrees_7102 : modelRow 7102 = row 7102 := model_count_7102.trans row_count_7102.symm
def row_7103 : RowCertificate 7103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7103 1) := by change ¬ Nat.Prime 201824643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67274881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7103 2) := by change ¬ Nat.Prime 201810437; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 2267533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7103 3) := by change ¬ Nat.Prime 201796231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28828033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7103 : row 7103 = 0 := (row_7103.checked).trans rfl
theorem model_count_7103 : modelRow 7103 = 0 := by decide
theorem modelrow_agrees_7103 : modelRow 7103 = row 7103 := model_count_7103.trans row_count_7103.symm
def row_7104 : RowCertificate 7104 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7104 1) := by change Nat.Prime 201881473; exact ProjectEulerSolutions.P58PrimeNodes.p_201881473.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7104 2) := by change ¬ Nat.Prime 201867265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40373453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7104 3) := by change ¬ Nat.Prime 201853057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28836151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7104 : row 7104 = 1 := (row_7104.checked).trans rfl
theorem model_count_7104 : modelRow 7104 = 1 := by decide
theorem modelrow_agrees_7104 : modelRow 7104 = row 7104 := model_count_7104.trans row_count_7104.symm
def row_7105 : RowCertificate 7105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7105 1) := by change ¬ Nat.Prime 201938311; exact PrattCertificates.composite_of_product (factor := 11617) (cofactor := 17383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7105 2) := by change Nat.Prime 201924101; exact ProjectEulerSolutions.P58PrimeNodes.p_201924101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7105 3) := by change ¬ Nat.Prime 201909891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67303297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7105 : row 7105 = 1 := (row_7105.checked).trans rfl
theorem model_count_7105 : modelRow 7105 = 1 := by decide
theorem modelrow_agrees_7105 : modelRow 7105 = row 7105 := model_count_7105.trans row_count_7105.symm
def row_7106 : RowCertificate 7106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7106 1) := by change ¬ Nat.Prime 201995157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67331719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7106 2) := by change ¬ Nat.Prime 201980945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40396189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7106 3) := by change ¬ Nat.Prime 201966733; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 350029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7106 : row 7106 = 0 := (row_7106.checked).trans rfl
theorem model_count_7106 : modelRow 7106 = 0 := by decide
theorem modelrow_agrees_7106 : modelRow 7106 = row 7106 := model_count_7106.trans row_count_7106.symm
def row_7107 : RowCertificate 7107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7107 1) := by change ¬ Nat.Prime 202052011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28864573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7107 2) := by change ¬ Nat.Prime 202037797; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15541369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7107 3) := by change Nat.Prime 202023583; exact ProjectEulerSolutions.P58PrimeNodes.p_202023583.prime
    exact decide_eq_true prime
theorem row_count_7107 : row 7107 = 1 := (row_7107.checked).trans rfl
theorem model_count_7107 : modelRow 7107 = 1 := by decide
theorem modelrow_agrees_7107 : modelRow 7107 = row 7107 := model_count_7107.trans row_count_7107.symm
def row_7108 : RowCertificate 7108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7108 1) := by change ¬ Nat.Prime 202108873; exact PrattCertificates.composite_of_product (factor := 1549) (cofactor := 130477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7108 2) := by change ¬ Nat.Prime 202094657; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11887921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7108 3) := by change ¬ Nat.Prime 202080441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67360147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7108 : row 7108 = 0 := (row_7108.checked).trans rfl
theorem model_count_7108 : modelRow 7108 = 0 := by decide
theorem modelrow_agrees_7108 : modelRow 7108 = row 7108 := model_count_7108.trans row_count_7108.symm
def row_7109 : RowCertificate 7109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7109 1) := by change ¬ Nat.Prime 202165743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67388581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7109 2) := by change ¬ Nat.Prime 202151525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40430305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7109 3) := by change Nat.Prime 202137307; exact ProjectEulerSolutions.P58PrimeNodes.p_202137307.prime
    exact decide_eq_true prime
theorem row_count_7109 : row 7109 = 1 := (row_7109.checked).trans rfl
theorem model_count_7109 : modelRow 7109 = 1 := by decide
theorem modelrow_agrees_7109 : modelRow 7109 = row 7109 := model_count_7109.trans row_count_7109.symm
def row_7110 : RowCertificate 7110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7110 1) := by change ¬ Nat.Prime 202222621; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1454839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7110 2) := by change Nat.Prime 202208401; exact ProjectEulerSolutions.P58PrimeNodes.p_202208401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7110 3) := by change ¬ Nat.Prime 202194181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28884883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7110 : row 7110 = 1 := (row_7110.checked).trans rfl
theorem model_count_7110 : modelRow 7110 = 1 := by decide
theorem modelrow_agrees_7110 : modelRow 7110 = row 7110 := model_count_7110.trans row_count_7110.symm
def row_7111 : RowCertificate 7111 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7111 1) := by change Nat.Prime 202279507; exact ProjectEulerSolutions.P58PrimeNodes.p_202279507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7111 2) := by change ¬ Nat.Prime 202265285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40453057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7111 3) := by change ¬ Nat.Prime 202251063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67417021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7111 : row 7111 = 1 := (row_7111.checked).trans rfl
theorem model_count_7111 : modelRow 7111 = 1 := by decide
theorem modelrow_agrees_7111 : modelRow 7111 = row 7111 := model_count_7111.trans row_count_7111.symm
def row_7112 : RowCertificate 7112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7112 1) := by change ¬ Nat.Prime 202336401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67445467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7112 2) := by change ¬ Nat.Prime 202322177; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3316757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7112 3) := by change ¬ Nat.Prime 202307953; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10647787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7112 : row 7112 = 0 := (row_7112.checked).trans rfl
theorem model_count_7112 : modelRow 7112 = 0 := by decide
theorem modelrow_agrees_7112 : modelRow 7112 = row 7112 := model_count_7112.trans row_count_7112.symm
def row_7113 : RowCertificate 7113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7113 1) := by change ¬ Nat.Prime 202393303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28913329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7113 2) := by change Nat.Prime 202379077; exact ProjectEulerSolutions.P58PrimeNodes.p_202379077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7113 3) := by change ¬ Nat.Prime 202364851; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15566527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7113 : row 7113 = 1 := (row_7113.checked).trans rfl
theorem model_count_7113 : modelRow 7113 = 1 := by decide
theorem modelrow_agrees_7113 : modelRow 7113 = row 7113 := model_count_7113.trans row_count_7113.symm
def row_7114 : RowCertificate 7114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7114 1) := by change ¬ Nat.Prime 202450213; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28921459) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7114 2) := by change ¬ Nat.Prime 202435985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40487197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7114 3) := by change ¬ Nat.Prime 202421757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67473919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7114 : row 7114 = 0 := (row_7114.checked).trans rfl
theorem model_count_7114 : modelRow 7114 = 0 := by decide
theorem modelrow_agrees_7114 : modelRow 7114 = row 7114 := model_count_7114.trans row_count_7114.symm
def row_7115 : RowCertificate 7115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7115 1) := by change ¬ Nat.Prime 202507131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67502377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7115 2) := by change ¬ Nat.Prime 202492901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15576377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7115 3) := by change ¬ Nat.Prime 202478671; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 551713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7115 : row 7115 = 0 := (row_7115.checked).trans rfl
theorem model_count_7115 : modelRow 7115 = 0 := by decide
theorem modelrow_agrees_7115 : modelRow 7115 = row 7115 := model_count_7115.trans row_count_7115.symm
def row_7116 : RowCertificate 7116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7116 1) := by change ¬ Nat.Prime 202564057; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 908359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7116 2) := by change ¬ Nat.Prime 202549825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40509965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7116 3) := by change ¬ Nat.Prime 202535593; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15579661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7116 : row 7116 = 0 := (row_7116.checked).trans rfl
theorem model_count_7116 : modelRow 7116 = 0 := by decide
theorem modelrow_agrees_7116 : modelRow 7116 = row 7116 := model_count_7116.trans row_count_7116.symm
def row_7117 : RowCertificate 7117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7117 1) := by change ¬ Nat.Prime 202620991; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6536161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7117 2) := by change ¬ Nat.Prime 202606757; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 3822769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7117 3) := by change ¬ Nat.Prime 202592523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67530841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7117 : row 7117 = 0 := (row_7117.checked).trans rfl
theorem model_count_7117 : modelRow 7117 = 0 := by decide
theorem modelrow_agrees_7117 : modelRow 7117 = row 7117 := model_count_7117.trans row_count_7117.symm
def row_7118 : RowCertificate 7118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7118 1) := by change ¬ Nat.Prime 202677933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67559311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7118 2) := by change ¬ Nat.Prime 202663697; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4943017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7118 3) := by change ¬ Nat.Prime 202649461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28949923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7118 : row 7118 = 0 := (row_7118.checked).trans rfl
theorem model_count_7118 : modelRow 7118 = 0 := by decide
theorem modelrow_agrees_7118 : modelRow 7118 = row 7118 := model_count_7118.trans row_count_7118.symm
def row_7119 : RowCertificate 7119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7119 1) := by change ¬ Nat.Prime 202734883; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15594991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7119 2) := by change ¬ Nat.Prime 202720645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40544129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7119 3) := by change ¬ Nat.Prime 202706407; exact PrattCertificates.composite_of_product (factor := 1303) (cofactor := 155569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7119 : row 7119 = 0 := (row_7119.checked).trans rfl
theorem model_count_7119 : modelRow 7119 = 0 := by decide
theorem modelrow_agrees_7119 : modelRow 7119 = row 7119 := model_count_7119.trans row_count_7119.symm
def row_7120 : RowCertificate 7120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7120 1) := by change ¬ Nat.Prime 202791841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28970263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7120 2) := by change ¬ Nat.Prime 202777601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15598277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7120 3) := by change ¬ Nat.Prime 202763361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67587787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7120 : row 7120 = 0 := (row_7120.checked).trans rfl
theorem model_count_7120 : modelRow 7120 = 0 := by decide
theorem modelrow_agrees_7120 : modelRow 7120 = row 7120 := model_count_7120.trans row_count_7120.symm
def row_7121 : RowCertificate 7121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7121 1) := by change ¬ Nat.Prime 202848807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67616269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7121 2) := by change ¬ Nat.Prime 202834565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40566913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7121 3) := by change ¬ Nat.Prime 202820323; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3027169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7121 : row 7121 = 0 := (row_7121.checked).trans rfl
theorem model_count_7121 : modelRow 7121 = 0 := by decide
theorem modelrow_agrees_7121 : modelRow 7121 = row 7121 := model_count_7121.trans row_count_7121.symm
def row_7122 : RowCertificate 7122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7122 1) := by change ¬ Nat.Prime 202905781; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15608137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7122 2) := by change ¬ Nat.Prime 202891537; exact PrattCertificates.composite_of_product (factor := 809) (cofactor := 250793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7122 3) := by change ¬ Nat.Prime 202877293; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 2779141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7122 : row 7122 = 0 := (row_7122.checked).trans rfl
theorem model_count_7122 : modelRow 7122 = 0 := by decide
theorem modelrow_agrees_7122 : modelRow 7122 = row 7122 := model_count_7122.trans row_count_7122.symm
def row_7123 : RowCertificate 7123 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7123 1) := by change Nat.Prime 202962763; exact ProjectEulerSolutions.P58PrimeNodes.p_202962763.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7123 2) := by change ¬ Nat.Prime 202948517; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1861913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7123 3) := by change ¬ Nat.Prime 202934271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67644757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7123 : row 7123 = 1 := (row_7123.checked).trans rfl
theorem model_count_7123 : modelRow 7123 = 1 := by decide
theorem modelrow_agrees_7123 : modelRow 7123 = row 7123 := model_count_7123.trans row_count_7123.symm
def row_7124 : RowCertificate 7124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7124 1) := by change ¬ Nat.Prime 203019753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67673251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7124 2) := by change ¬ Nat.Prime 203005505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40601101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7124 3) := by change ¬ Nat.Prime 202991257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28998751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7124 : row 7124 = 0 := (row_7124.checked).trans rfl
theorem model_count_7124 : modelRow 7124 = 0 := by decide
theorem modelrow_agrees_7124 : modelRow 7124 = row 7124 := model_count_7124.trans row_count_7124.symm
def row_7125 : RowCertificate 7125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7125 1) := by change ¬ Nat.Prime 203076751; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1121971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7125 2) := by change ¬ Nat.Prime 203062501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11944853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7125 3) := by change ¬ Nat.Prime 203048251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29006893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7125 : row 7125 = 0 := (row_7125.checked).trans rfl
theorem model_count_7125 : modelRow 7125 = 0 := by decide
theorem modelrow_agrees_7125 : modelRow 7125 = row 7125 := model_count_7125.trans row_count_7125.symm
def row_7126 : RowCertificate 7126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7126 1) := by change ¬ Nat.Prime 203133757; exact PrattCertificates.composite_of_product (factor := 9187) (cofactor := 22111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7126 2) := by change ¬ Nat.Prime 203119505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40623901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7126 3) := by change ¬ Nat.Prime 203105253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67701751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7126 : row 7126 = 0 := (row_7126.checked).trans rfl
theorem model_count_7126 : modelRow 7126 = 0 := by decide
theorem modelrow_agrees_7126 : modelRow 7126 = row 7126 := model_count_7126.trans row_count_7126.symm
def row_7127 : RowCertificate 7127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7127 1) := by change ¬ Nat.Prime 203190771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67730257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7127 2) := by change ¬ Nat.Prime 203176517; exact PrattCertificates.composite_of_product (factor := 12073) (cofactor := 16829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7127 3) := by change Nat.Prime 203162263; exact ProjectEulerSolutions.P58PrimeNodes.p_203162263.prime
    exact decide_eq_true prime
theorem row_count_7127 : row 7127 = 1 := (row_7127.checked).trans rfl
theorem model_count_7127 : modelRow 7127 = 1 := by decide
theorem modelrow_agrees_7127 : modelRow 7127 = row 7127 := model_count_7127.trans row_count_7127.symm
def row_7128 : RowCertificate 7128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7128 1) := by change ¬ Nat.Prime 203247793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29035399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7128 2) := by change ¬ Nat.Prime 203233537; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15633349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7128 3) := by change ¬ Nat.Prime 203219281; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5492413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7128 : row 7128 = 0 := (row_7128.checked).trans rfl
theorem model_count_7128 : modelRow 7128 = 0 := by decide
theorem modelrow_agrees_7128 : modelRow 7128 = row 7128 := model_count_7128.trans row_count_7128.symm
def row_7129 : RowCertificate 7129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7129 1) := by change ¬ Nat.Prime 203304823; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 279649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7129 2) := by change ¬ Nat.Prime 203290565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40658113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7129 3) := by change ¬ Nat.Prime 203276307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67758769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7129 : row 7129 = 0 := (row_7129.checked).trans rfl
theorem model_count_7129 : modelRow 7129 = 0 := by decide
theorem modelrow_agrees_7129 : modelRow 7129 = row 7129 := model_count_7129.trans row_count_7129.symm
def row_7130 : RowCertificate 7130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7130 1) := by change ¬ Nat.Prime 203361861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67787287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7130 2) := by change ¬ Nat.Prime 203347601; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 1364749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7130 3) := by change ¬ Nat.Prime 203333341; exact PrattCertificates.composite_of_product (factor := 7057) (cofactor := 28813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7130 : row 7130 = 0 := (row_7130.checked).trans rfl
theorem model_count_7130 : modelRow 7130 = 0 := by decide
theorem modelrow_agrees_7130 : modelRow 7130 = row 7130 := model_count_7130.trans row_count_7130.symm
def row_7131 : RowCertificate 7131 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7131 1) := by change ¬ Nat.Prime 203418907; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 545359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7131 2) := by change ¬ Nat.Prime 203404645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40680929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7131 3) := by change ¬ Nat.Prime 203390383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29055769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7131 : row 7131 = 0 := (row_7131.checked).trans rfl
theorem model_count_7131 : modelRow 7131 = 0 := by decide
theorem modelrow_agrees_7131 : modelRow 7131 = row 7131 := model_count_7131.trans row_count_7131.symm
def row_7132 : RowCertificate 7132 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7132 1) := by change ¬ Nat.Prime 203475961; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15651997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7132 2) := by change Nat.Prime 203461697; exact ProjectEulerSolutions.P58PrimeNodes.p_203461697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7132 3) := by change ¬ Nat.Prime 203447433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67815811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7132 : row 7132 = 1 := (row_7132.checked).trans rfl
theorem model_count_7132 : modelRow 7132 = 1 := by decide
theorem modelrow_agrees_7132 : modelRow 7132 = row 7132 := model_count_7132.trans row_count_7132.symm
def row_7133 : RowCertificate 7133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7133 1) := by change ¬ Nat.Prime 203533023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67844341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7133 2) := by change ¬ Nat.Prime 203518757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15655289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7133 3) := by change ¬ Nat.Prime 203504491; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6564661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7133 : row 7133 = 0 := (row_7133.checked).trans rfl
theorem model_count_7133 : modelRow 7133 = 0 := by decide
theorem modelrow_agrees_7133 : modelRow 7133 = row 7133 := model_count_7133.trans row_count_7133.symm
def row_7134 : RowCertificate 7134 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7134 1) := by change ¬ Nat.Prime 203590093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29084299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7134 2) := by change ¬ Nat.Prime 203575825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40715165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7134 3) := by change Nat.Prime 203561557; exact ProjectEulerSolutions.P58PrimeNodes.p_203561557.prime
    exact decide_eq_true prime
theorem row_count_7134 : row 7134 = 1 := (row_7134.checked).trans rfl
theorem model_count_7134 : modelRow 7134 = 1 := by decide
theorem modelrow_agrees_7134 : modelRow 7134 = row 7134 := model_count_7134.trans row_count_7134.symm
def row_7135 : RowCertificate 7135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7135 1) := by change ¬ Nat.Prime 203647171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29092453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7135 2) := by change Nat.Prime 203632901; exact ProjectEulerSolutions.P58PrimeNodes.p_203632901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7135 3) := by change ¬ Nat.Prime 203618631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67872877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7135 : row 7135 = 1 := (row_7135.checked).trans rfl
theorem model_count_7135 : modelRow 7135 = 1 := by decide
theorem modelrow_agrees_7135 : modelRow 7135 = row 7135 := model_count_7135.trans row_count_7135.symm
def row_7136 : RowCertificate 7136 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7136 1) := by change ¬ Nat.Prime 203704257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67901419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7136 2) := by change ¬ Nat.Prime 203689985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40737997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7136 3) := by change ¬ Nat.Prime 203675713; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5504749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7136 : row 7136 = 0 := (row_7136.checked).trans rfl
theorem model_count_7136 : modelRow 7136 = 0 := by decide
theorem modelrow_agrees_7136 : modelRow 7136 = row 7136 := model_count_7136.trans row_count_7136.symm
def row_7137 : RowCertificate 7137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7137 1) := by change ¬ Nat.Prime 203761351; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1465909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7137 2) := by change ¬ Nat.Prime 203747077; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 2289293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7137 3) := by change Nat.Prime 203732803; exact ProjectEulerSolutions.P58PrimeNodes.p_203732803.prime
    exact decide_eq_true prime
theorem row_count_7137 : row 7137 = 1 := (row_7137.checked).trans rfl
theorem model_count_7137 : modelRow 7137 = 1 := by decide
theorem modelrow_agrees_7137 : modelRow 7137 = row 7137 := model_count_7137.trans row_count_7137.symm
def row_7138 : RowCertificate 7138 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7138 1) := by change ¬ Nat.Prime 203818453; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10727287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7138 2) := by change ¬ Nat.Prime 203804177; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11988481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7138 3) := by change ¬ Nat.Prime 203789901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67929967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7138 : row 7138 = 0 := (row_7138.checked).trans rfl
theorem model_count_7138 : modelRow 7138 = 0 := by decide
theorem modelrow_agrees_7138 : modelRow 7138 = row 7138 := model_count_7138.trans row_count_7138.symm
def row_7139 : RowCertificate 7139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7139 1) := by change ¬ Nat.Prime 203875563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67958521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7139 2) := by change ¬ Nat.Prime 203861285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40772257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7139 3) := by change ¬ Nat.Prime 203847007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29121001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7139 : row 7139 = 0 := (row_7139.checked).trans rfl
theorem model_count_7139 : modelRow 7139 = 0 := by decide
theorem modelrow_agrees_7139 : modelRow 7139 = row 7139 := model_count_7139.trans row_count_7139.symm
def row_7140 : RowCertificate 7140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7140 1) := by change ¬ Nat.Prime 203932681; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10733299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7140 2) := by change ¬ Nat.Prime 203918401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 7031669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7140 3) := by change Nat.Prime 203904121; exact ProjectEulerSolutions.P58PrimeNodes.p_203904121.prime
    exact decide_eq_true prime
theorem row_count_7140 : row 7140 = 1 := (row_7140.checked).trans rfl
theorem model_count_7140 : modelRow 7140 = 1 := by decide
theorem modelrow_agrees_7140 : modelRow 7140 = row 7140 := model_count_7140.trans row_count_7140.symm
def row_7141 : RowCertificate 7141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7141 1) := by change ¬ Nat.Prime 203989807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29141401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7141 2) := by change ¬ Nat.Prime 203975525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40795105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7141 3) := by change ¬ Nat.Prime 203961243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 67987081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7141 : row 7141 = 0 := (row_7141.checked).trans rfl
theorem model_count_7141 : modelRow 7141 = 0 := by decide
theorem modelrow_agrees_7141 : modelRow 7141 = row 7141 := model_count_7141.trans row_count_7141.symm
def row_7142 : RowCertificate 7142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7142 1) := by change ¬ Nat.Prime 204046941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68015647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7142 2) := by change ¬ Nat.Prime 204032657; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 12001921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7142 3) := by change ¬ Nat.Prime 204018373; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15693721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7142 : row 7142 = 0 := (row_7142.checked).trans rfl
theorem model_count_7142 : modelRow 7142 = 0 := by decide
theorem modelrow_agrees_7142 : modelRow 7142 = row 7142 := model_count_7142.trans row_count_7142.symm
def row_7143 : RowCertificate 7143 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7143 1) := by change Nat.Prime 204104083; exact ProjectEulerSolutions.P58PrimeNodes.p_204104083.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7143 2) := by change ¬ Nat.Prime 204089797; exact PrattCertificates.composite_of_product (factor := 13913) (cofactor := 14669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7143 3) := by change ¬ Nat.Prime 204075511; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6583081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7143 : row 7143 = 1 := (row_7143.checked).trans rfl
theorem model_count_7143 : modelRow 7143 = 1 := by decide
theorem modelrow_agrees_7143 : modelRow 7143 = row 7143 := model_count_7143.trans row_count_7143.symm
def row_7144 : RowCertificate 7144 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7144 1) := by change Nat.Prime 204161233; exact ProjectEulerSolutions.P58PrimeNodes.p_204161233.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7144 2) := by change ¬ Nat.Prime 204146945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40829389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7144 3) := by change ¬ Nat.Prime 204132657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68044219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7144 : row 7144 = 1 := (row_7144.checked).trans rfl
theorem model_count_7144 : modelRow 7144 = 1 := by decide
theorem modelrow_agrees_7144 : modelRow 7144 = row 7144 := model_count_7144.trans row_count_7144.symm
def row_7145 : RowCertificate 7145 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7145 1) := by change ¬ Nat.Prime 204218391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68072797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7145 2) := by change Nat.Prime 204204101; exact ProjectEulerSolutions.P58PrimeNodes.p_204204101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7145 3) := by change ¬ Nat.Prime 204189811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29169973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7145 : row 7145 = 1 := (row_7145.checked).trans rfl
theorem model_count_7145 : modelRow 7145 = 1 := by decide
theorem modelrow_agrees_7145 : modelRow 7145 = row 7145 := model_count_7145.trans row_count_7145.symm
def row_7146 : RowCertificate 7146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7146 1) := by change ¬ Nat.Prime 204275557; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5520961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7146 2) := by change ¬ Nat.Prime 204261265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40852253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7146 3) := by change ¬ Nat.Prime 204246973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29178139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7146 : row 7146 = 0 := (row_7146.checked).trans rfl
theorem model_count_7146 : modelRow 7146 = 0 := by decide
theorem modelrow_agrees_7146 : modelRow 7146 = row 7146 := model_count_7146.trans row_count_7146.symm
def row_7147 : RowCertificate 7147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7147 1) := by change ¬ Nat.Prime 204332731; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2106523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7147 2) := by change Nat.Prime 204318437; exact ProjectEulerSolutions.P58PrimeNodes.p_204318437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7147 3) := by change ¬ Nat.Prime 204304143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68101381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7147 : row 7147 = 1 := (row_7147.checked).trans rfl
theorem model_count_7147 : modelRow 7147 = 1 := by decide
theorem modelrow_agrees_7147 : modelRow 7147 = row 7147 := model_count_7147.trans row_count_7147.symm
def row_7148 : RowCertificate 7148 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7148 1) := by change ¬ Nat.Prime 204389913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68129971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7148 2) := by change ¬ Nat.Prime 204375617; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 269981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7148 3) := by change ¬ Nat.Prime 204361321; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10755859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7148 : row 7148 = 0 := (row_7148.checked).trans rfl
theorem model_count_7148 : modelRow 7148 = 0 := by decide
theorem modelrow_agrees_7148 : modelRow 7148 = row 7148 := model_count_7148.trans row_count_7148.symm
def row_7149 : RowCertificate 7149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7149 1) := by change ¬ Nat.Prime 204447103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29206729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7149 2) := by change ¬ Nat.Prime 204432805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40886561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7149 3) := by change Nat.Prime 204418507; exact ProjectEulerSolutions.P58PrimeNodes.p_204418507.prime
    exact decide_eq_true prime
theorem row_count_7149 : row 7149 = 1 := (row_7149.checked).trans rfl
theorem model_count_7149 : modelRow 7149 = 1 := by decide
theorem modelrow_agrees_7149 : modelRow 7149 = row 7149 := model_count_7149.trans row_count_7149.symm
def row_7150 : RowCertificate 7150 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7150 1) := by change ¬ Nat.Prime 204504301; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3052303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7150 2) := by change ¬ Nat.Prime 204490001; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4987561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7150 3) := by change ¬ Nat.Prime 204475701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68158567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7150 : row 7150 = 0 := (row_7150.checked).trans rfl
theorem model_count_7150 : modelRow 7150 = 0 := by decide
theorem modelrow_agrees_7150 : modelRow 7150 = row 7150 := model_count_7150.trans row_count_7150.symm
def row_7151 : RowCertificate 7151 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7151 1) := by change ¬ Nat.Prime 204561507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68187169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7151 2) := by change ¬ Nat.Prime 204547205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40909441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7151 3) := by change ¬ Nat.Prime 204532903; exact PrattCertificates.composite_of_product (factor := 4297) (cofactor := 47599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7151 : row 7151 = 0 := (row_7151.checked).trans rfl
theorem model_count_7151 : modelRow 7151 = 0 := by decide
theorem modelrow_agrees_7151 : modelRow 7151 = row 7151 := model_count_7151.trans row_count_7151.symm
def row_7152 : RowCertificate 7152 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7152 1) := by change Nat.Prime 204618721; exact ProjectEulerSolutions.P58PrimeNodes.p_204618721.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7152 2) := by change ¬ Nat.Prime 204604417; exact PrattCertificates.composite_of_product (factor := 3217) (cofactor := 63601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7152 3) := by change ¬ Nat.Prime 204590113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29227159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7152 : row 7152 = 1 := (row_7152.checked).trans rfl
theorem model_count_7152 : modelRow 7152 = 1 := by decide
theorem modelrow_agrees_7152 : modelRow 7152 = row 7152 := model_count_7152.trans row_count_7152.symm
def row_7153 : RowCertificate 7153 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7153 1) := by change Nat.Prime 204675943; exact ProjectEulerSolutions.P58PrimeNodes.p_204675943.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7153 2) := by change ¬ Nat.Prime 204661637; exact PrattCertificates.composite_of_product (factor := 12409) (cofactor := 16493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7153 3) := by change ¬ Nat.Prime 204647331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68215777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7153 : row 7153 = 1 := (row_7153.checked).trans rfl
theorem model_count_7153 : modelRow 7153 = 1 := by decide
theorem modelrow_agrees_7153 : modelRow 7153 = row 7153 := model_count_7153.trans row_count_7153.symm
def row_7154 : RowCertificate 7154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7154 1) := by change ¬ Nat.Prime 204733173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68244391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7154 2) := by change ¬ Nat.Prime 204718865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40943773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7154 3) := by change Nat.Prime 204704557; exact ProjectEulerSolutions.P58PrimeNodes.p_204704557.prime
    exact decide_eq_true prime
theorem row_count_7154 : row 7154 = 1 := (row_7154.checked).trans rfl
theorem model_count_7154 : modelRow 7154 = 1 := by decide
theorem modelrow_agrees_7154 : modelRow 7154 = row 7154 := model_count_7154.trans row_count_7154.symm
def row_7155 : RowCertificate 7155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7155 1) := by change ¬ Nat.Prime 204790411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29255773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7155 2) := by change ¬ Nat.Prime 204776101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 12045653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7155 3) := by change ¬ Nat.Prime 204761791; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15750907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7155 : row 7155 = 0 := (row_7155.checked).trans rfl
theorem model_count_7155 : modelRow 7155 = 0 := by decide
theorem modelrow_agrees_7155 : modelRow 7155 = row 7155 := model_count_7155.trans row_count_7155.symm
def row_7156 : RowCertificate 7156 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7156 1) := by change ¬ Nat.Prime 204847657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29263951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7156 2) := by change ¬ Nat.Prime 204833345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 40966669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7156 3) := by change ¬ Nat.Prime 204819033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68273011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7156 : row 7156 = 0 := (row_7156.checked).trans rfl
theorem model_count_7156 : modelRow 7156 = 0 := by decide
theorem modelrow_agrees_7156 : modelRow 7156 = row 7156 := model_count_7156.trans row_count_7156.symm
def row_7157 : RowCertificate 7157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7157 1) := by change ¬ Nat.Prime 204904911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68301637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7157 2) := by change ¬ Nat.Prime 204890597; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 7065193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7157 3) := by change Nat.Prime 204876283; exact ProjectEulerSolutions.P58PrimeNodes.p_204876283.prime
    exact decide_eq_true prime
theorem row_count_7157 : row 7157 = 1 := (row_7157.checked).trans rfl
theorem model_count_7157 : modelRow 7157 = 1 := by decide
theorem modelrow_agrees_7157 : modelRow 7157 = row 7157 := model_count_7157.trans row_count_7157.symm
def row_7158 : RowCertificate 7158 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7158 1) := by change ¬ Nat.Prime 204962173; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15766321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7158 2) := by change ¬ Nat.Prime 204947857; exact PrattCertificates.composite_of_product (factor := 1553) (cofactor := 131969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7158 3) := by change ¬ Nat.Prime 204933541; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 466819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7158 : row 7158 = 0 := (row_7158.checked).trans rfl
theorem model_count_7158 : modelRow 7158 = 0 := by decide
theorem modelrow_agrees_7158 : modelRow 7158 = row 7158 := model_count_7158.trans row_count_7158.symm
def row_7159 : RowCertificate 7159 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7159 1) := by change ¬ Nat.Prime 205019443; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10790497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7159 2) := by change ¬ Nat.Prime 205005125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 41001025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7159 3) := by change ¬ Nat.Prime 204990807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68330269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7159 : row 7159 = 0 := (row_7159.checked).trans rfl
theorem model_count_7159 : modelRow 7159 = 0 := by decide
theorem modelrow_agrees_7159 : modelRow 7159 = row 7159 := model_count_7159.trans row_count_7159.symm
def row_7160 : RowCertificate 7160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7160 1) := by change ¬ Nat.Prime 205076721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68358907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7160 2) := by change ¬ Nat.Prime 205062401; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 895469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7160 3) := by change ¬ Nat.Prime 205048081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29292583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7160 : row 7160 = 0 := (row_7160.checked).trans rfl
theorem model_count_7160 : modelRow 7160 = 0 := by decide
theorem modelrow_agrees_7160 : modelRow 7160 = row 7160 := model_count_7160.trans row_count_7160.symm
def row_7161 : RowCertificate 7161 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7161 1) := by change ¬ Nat.Prime 205134007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15779539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7161 2) := by change ¬ Nat.Prime 205119685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 41023937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7161 3) := by change ¬ Nat.Prime 205105363; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3362383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7161 : row 7161 = 0 := (row_7161.checked).trans rfl
theorem model_count_7161 : modelRow 7161 = 0 := by decide
theorem modelrow_agrees_7161 : modelRow 7161 = row 7161 := model_count_7161.trans row_count_7161.symm
def row_7162 : RowCertificate 7162 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7162 1) := by change ¬ Nat.Prime 205191301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29313043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7162 2) := by change ¬ Nat.Prime 205176977; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3363557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7162 3) := by change ¬ Nat.Prime 205162653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68387551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7162 : row 7162 = 0 := (row_7162.checked).trans rfl
theorem model_count_7162 : modelRow 7162 = 0 := by decide
theorem modelrow_agrees_7162 : modelRow 7162 = row 7162 := model_count_7162.trans row_count_7162.symm
def row_7163 : RowCertificate 7163 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7163 1) := by change ¬ Nat.Prime 205248603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68416201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7163 2) := by change ¬ Nat.Prime 205234277; exact PrattCertificates.composite_of_product (factor := 7069) (cofactor := 29033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7163 3) := by change ¬ Nat.Prime 205219951; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4772557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7163 : row 7163 = 0 := (row_7163.checked).trans rfl
theorem model_count_7163 : modelRow 7163 = 0 := by decide
theorem modelrow_agrees_7163 : modelRow 7163 = row 7163 := model_count_7163.trans row_count_7163.symm
def row_7164 : RowCertificate 7164 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7164 1) := by change Nat.Prime 205305913; exact ProjectEulerSolutions.P58PrimeNodes.p_205305913.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7164 2) := by change ¬ Nat.Prime 205291585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 41058317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7164 3) := by change ¬ Nat.Prime 205277257; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6621847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7164 : row 7164 = 1 := (row_7164.checked).trans rfl
theorem model_count_7164 : modelRow 7164 = 1 := by decide
theorem modelrow_agrees_7164 : modelRow 7164 = row 7164 := model_count_7164.trans row_count_7164.symm
def row_7165 : RowCertificate 7165 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7165 1) := by change Nat.Prime 205363231; exact ProjectEulerSolutions.P58PrimeNodes.p_205363231.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7165 2) := by change Nat.Prime 205348901; exact ProjectEulerSolutions.P58PrimeNodes.p_205348901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7165 3) := by change ¬ Nat.Prime 205334571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68444857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7165 : row 7165 = 2 := (row_7165.checked).trans rfl
theorem model_count_7165 : modelRow 7165 = 2 := by decide
theorem modelrow_agrees_7165 : modelRow 7165 = row 7165 := model_count_7165.trans row_count_7165.symm
def row_7166 : RowCertificate 7166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7166 1) := by change ¬ Nat.Prime 205420557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68473519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7166 2) := by change ¬ Nat.Prime 205406225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 41081245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7166 3) := by change ¬ Nat.Prime 205391893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29341699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7166 : row 7166 = 0 := (row_7166.checked).trans rfl
theorem model_count_7166 : modelRow 7166 = 0 := by decide
theorem modelrow_agrees_7166 : modelRow 7166 = row 7166 := model_count_7166.trans row_count_7166.symm
def row_7167 : RowCertificate 7167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7167 1) := by change ¬ Nat.Prime 205477891; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 758221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7167 2) := by change ¬ Nat.Prime 205463557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15804889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7167 3) := by change ¬ Nat.Prime 205449223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 29349889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7167 : row 7167 = 0 := (row_7167.checked).trans rfl
theorem model_count_7167 : modelRow 7167 = 0 := by decide
theorem modelrow_agrees_7167 : modelRow 7167 = row 7167 := model_count_7167.trans row_count_7167.symm
def row_7168 : RowCertificate 7168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7168 1) := by change ¬ Nat.Prime 205535233; exact PrattCertificates.composite_of_product (factor := 3361) (cofactor := 61153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7168 2) := by change ¬ Nat.Prime 205520897; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 1818769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7168 3) := by change ¬ Nat.Prime 205506561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 68502187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7168 : row 7168 = 0 := (row_7168.checked).trans rfl
theorem model_count_7168 : modelRow 7168 = 0 := by decide
theorem modelrow_agrees_7168 : modelRow 7168 = row 7168 := model_count_7168.trans row_count_7168.symm
private def atomic_7041 : CountSegment 1 10 13120 7040 7041 3015 3015 := CountSegment.single (first := 7040) (initial := 3015) row_count_7041 (by decide)
private theorem atomic_7041_agreement : RowAgreement 7040 7041 := RowAgreement.single (first := 7040) modelrow_agrees_7041
private def atomic_7042 : CountSegment 1 10 13120 7041 7042 3015 3016 := CountSegment.single (first := 7041) (initial := 3015) row_count_7042 (by decide)
private theorem atomic_7042_agreement : RowAgreement 7041 7042 := RowAgreement.single (first := 7041) modelrow_agrees_7042
private def atomic_7043 : CountSegment 1 10 13120 7042 7043 3016 3017 := CountSegment.single (first := 7042) (initial := 3016) row_count_7043 (by decide)
private theorem atomic_7043_agreement : RowAgreement 7042 7043 := RowAgreement.single (first := 7042) modelrow_agrees_7043
private def atomic_7044 : CountSegment 1 10 13120 7043 7044 3017 3018 := CountSegment.single (first := 7043) (initial := 3017) row_count_7044 (by decide)
private theorem atomic_7044_agreement : RowAgreement 7043 7044 := RowAgreement.single (first := 7043) modelrow_agrees_7044
private def atomic_7045 : CountSegment 1 10 13120 7044 7045 3018 3018 := CountSegment.single (first := 7044) (initial := 3018) row_count_7045 (by decide)
private theorem atomic_7045_agreement : RowAgreement 7044 7045 := RowAgreement.single (first := 7044) modelrow_agrees_7045
private def atomic_7046 : CountSegment 1 10 13120 7045 7046 3018 3019 := CountSegment.single (first := 7045) (initial := 3018) row_count_7046 (by decide)
private theorem atomic_7046_agreement : RowAgreement 7045 7046 := RowAgreement.single (first := 7045) modelrow_agrees_7046
private def atomic_7047 : CountSegment 1 10 13120 7046 7047 3019 3019 := CountSegment.single (first := 7046) (initial := 3019) row_count_7047 (by decide)
private theorem atomic_7047_agreement : RowAgreement 7046 7047 := RowAgreement.single (first := 7046) modelrow_agrees_7047
private def atomic_7048 : CountSegment 1 10 13120 7047 7048 3019 3020 := CountSegment.single (first := 7047) (initial := 3019) row_count_7048 (by decide)
private theorem atomic_7048_agreement : RowAgreement 7047 7048 := RowAgreement.single (first := 7047) modelrow_agrees_7048
private def atomic_7049 : CountSegment 1 10 13120 7048 7049 3020 3020 := CountSegment.single (first := 7048) (initial := 3020) row_count_7049 (by decide)
private theorem atomic_7049_agreement : RowAgreement 7048 7049 := RowAgreement.single (first := 7048) modelrow_agrees_7049
private def atomic_7050 : CountSegment 1 10 13120 7049 7050 3020 3020 := CountSegment.single (first := 7049) (initial := 3020) row_count_7050 (by decide)
private theorem atomic_7050_agreement : RowAgreement 7049 7050 := RowAgreement.single (first := 7049) modelrow_agrees_7050
private def atomic_7051 : CountSegment 1 10 13120 7050 7051 3020 3020 := CountSegment.single (first := 7050) (initial := 3020) row_count_7051 (by decide)
private theorem atomic_7051_agreement : RowAgreement 7050 7051 := RowAgreement.single (first := 7050) modelrow_agrees_7051
private def atomic_7052 : CountSegment 1 10 13120 7051 7052 3020 3021 := CountSegment.single (first := 7051) (initial := 3020) row_count_7052 (by decide)
private theorem atomic_7052_agreement : RowAgreement 7051 7052 := RowAgreement.single (first := 7051) modelrow_agrees_7052
private def atomic_7053 : CountSegment 1 10 13120 7052 7053 3021 3021 := CountSegment.single (first := 7052) (initial := 3021) row_count_7053 (by decide)
private theorem atomic_7053_agreement : RowAgreement 7052 7053 := RowAgreement.single (first := 7052) modelrow_agrees_7053
private def atomic_7054 : CountSegment 1 10 13120 7053 7054 3021 3021 := CountSegment.single (first := 7053) (initial := 3021) row_count_7054 (by decide)
private theorem atomic_7054_agreement : RowAgreement 7053 7054 := RowAgreement.single (first := 7053) modelrow_agrees_7054
private def atomic_7055 : CountSegment 1 10 13120 7054 7055 3021 3021 := CountSegment.single (first := 7054) (initial := 3021) row_count_7055 (by decide)
private theorem atomic_7055_agreement : RowAgreement 7054 7055 := RowAgreement.single (first := 7054) modelrow_agrees_7055
private def atomic_7056 : CountSegment 1 10 13120 7055 7056 3021 3022 := CountSegment.single (first := 7055) (initial := 3021) row_count_7056 (by decide)
private theorem atomic_7056_agreement : RowAgreement 7055 7056 := RowAgreement.single (first := 7055) modelrow_agrees_7056
private def atomic_7057 : CountSegment 1 10 13120 7056 7057 3022 3022 := CountSegment.single (first := 7056) (initial := 3022) row_count_7057 (by decide)
private theorem atomic_7057_agreement : RowAgreement 7056 7057 := RowAgreement.single (first := 7056) modelrow_agrees_7057
private def atomic_7058 : CountSegment 1 10 13120 7057 7058 3022 3022 := CountSegment.single (first := 7057) (initial := 3022) row_count_7058 (by decide)
private theorem atomic_7058_agreement : RowAgreement 7057 7058 := RowAgreement.single (first := 7057) modelrow_agrees_7058
private def atomic_7059 : CountSegment 1 10 13120 7058 7059 3022 3023 := CountSegment.single (first := 7058) (initial := 3022) row_count_7059 (by decide)
private theorem atomic_7059_agreement : RowAgreement 7058 7059 := RowAgreement.single (first := 7058) modelrow_agrees_7059
private def atomic_7060 : CountSegment 1 10 13120 7059 7060 3023 3024 := CountSegment.single (first := 7059) (initial := 3023) row_count_7060 (by decide)
private theorem atomic_7060_agreement : RowAgreement 7059 7060 := RowAgreement.single (first := 7059) modelrow_agrees_7060
private def atomic_7061 : CountSegment 1 10 13120 7060 7061 3024 3024 := CountSegment.single (first := 7060) (initial := 3024) row_count_7061 (by decide)
private theorem atomic_7061_agreement : RowAgreement 7060 7061 := RowAgreement.single (first := 7060) modelrow_agrees_7061
private def atomic_7062 : CountSegment 1 10 13120 7061 7062 3024 3024 := CountSegment.single (first := 7061) (initial := 3024) row_count_7062 (by decide)
private theorem atomic_7062_agreement : RowAgreement 7061 7062 := RowAgreement.single (first := 7061) modelrow_agrees_7062
private def atomic_7063 : CountSegment 1 10 13120 7062 7063 3024 3024 := CountSegment.single (first := 7062) (initial := 3024) row_count_7063 (by decide)
private theorem atomic_7063_agreement : RowAgreement 7062 7063 := RowAgreement.single (first := 7062) modelrow_agrees_7063
private def atomic_7064 : CountSegment 1 10 13120 7063 7064 3024 3024 := CountSegment.single (first := 7063) (initial := 3024) row_count_7064 (by decide)
private theorem atomic_7064_agreement : RowAgreement 7063 7064 := RowAgreement.single (first := 7063) modelrow_agrees_7064
private def atomic_7065 : CountSegment 1 10 13120 7064 7065 3024 3025 := CountSegment.single (first := 7064) (initial := 3024) row_count_7065 (by decide)
private theorem atomic_7065_agreement : RowAgreement 7064 7065 := RowAgreement.single (first := 7064) modelrow_agrees_7065
private def atomic_7066 : CountSegment 1 10 13120 7065 7066 3025 3026 := CountSegment.single (first := 7065) (initial := 3025) row_count_7066 (by decide)
private theorem atomic_7066_agreement : RowAgreement 7065 7066 := RowAgreement.single (first := 7065) modelrow_agrees_7066
private def atomic_7067 : CountSegment 1 10 13120 7066 7067 3026 3027 := CountSegment.single (first := 7066) (initial := 3026) row_count_7067 (by decide)
private theorem atomic_7067_agreement : RowAgreement 7066 7067 := RowAgreement.single (first := 7066) modelrow_agrees_7067
private def atomic_7068 : CountSegment 1 10 13120 7067 7068 3027 3028 := CountSegment.single (first := 7067) (initial := 3027) row_count_7068 (by decide)
private theorem atomic_7068_agreement : RowAgreement 7067 7068 := RowAgreement.single (first := 7067) modelrow_agrees_7068
private def atomic_7069 : CountSegment 1 10 13120 7068 7069 3028 3028 := CountSegment.single (first := 7068) (initial := 3028) row_count_7069 (by decide)
private theorem atomic_7069_agreement : RowAgreement 7068 7069 := RowAgreement.single (first := 7068) modelrow_agrees_7069
private def atomic_7070 : CountSegment 1 10 13120 7069 7070 3028 3028 := CountSegment.single (first := 7069) (initial := 3028) row_count_7070 (by decide)
private theorem atomic_7070_agreement : RowAgreement 7069 7070 := RowAgreement.single (first := 7069) modelrow_agrees_7070
private def atomic_7071 : CountSegment 1 10 13120 7070 7071 3028 3028 := CountSegment.single (first := 7070) (initial := 3028) row_count_7071 (by decide)
private theorem atomic_7071_agreement : RowAgreement 7070 7071 := RowAgreement.single (first := 7070) modelrow_agrees_7071
private def atomic_7072 : CountSegment 1 10 13120 7071 7072 3028 3029 := CountSegment.single (first := 7071) (initial := 3028) row_count_7072 (by decide)
private theorem atomic_7072_agreement : RowAgreement 7071 7072 := RowAgreement.single (first := 7071) modelrow_agrees_7072
private def atomic_7073 : CountSegment 1 10 13120 7072 7073 3029 3030 := CountSegment.single (first := 7072) (initial := 3029) row_count_7073 (by decide)
private theorem atomic_7073_agreement : RowAgreement 7072 7073 := RowAgreement.single (first := 7072) modelrow_agrees_7073
private def atomic_7074 : CountSegment 1 10 13120 7073 7074 3030 3031 := CountSegment.single (first := 7073) (initial := 3030) row_count_7074 (by decide)
private theorem atomic_7074_agreement : RowAgreement 7073 7074 := RowAgreement.single (first := 7073) modelrow_agrees_7074
private def atomic_7075 : CountSegment 1 10 13120 7074 7075 3031 3031 := CountSegment.single (first := 7074) (initial := 3031) row_count_7075 (by decide)
private theorem atomic_7075_agreement : RowAgreement 7074 7075 := RowAgreement.single (first := 7074) modelrow_agrees_7075
private def atomic_7076 : CountSegment 1 10 13120 7075 7076 3031 3031 := CountSegment.single (first := 7075) (initial := 3031) row_count_7076 (by decide)
private theorem atomic_7076_agreement : RowAgreement 7075 7076 := RowAgreement.single (first := 7075) modelrow_agrees_7076
private def atomic_7077 : CountSegment 1 10 13120 7076 7077 3031 3032 := CountSegment.single (first := 7076) (initial := 3031) row_count_7077 (by decide)
private theorem atomic_7077_agreement : RowAgreement 7076 7077 := RowAgreement.single (first := 7076) modelrow_agrees_7077
private def atomic_7078 : CountSegment 1 10 13120 7077 7078 3032 3033 := CountSegment.single (first := 7077) (initial := 3032) row_count_7078 (by decide)
private theorem atomic_7078_agreement : RowAgreement 7077 7078 := RowAgreement.single (first := 7077) modelrow_agrees_7078
private def atomic_7079 : CountSegment 1 10 13120 7078 7079 3033 3033 := CountSegment.single (first := 7078) (initial := 3033) row_count_7079 (by decide)
private theorem atomic_7079_agreement : RowAgreement 7078 7079 := RowAgreement.single (first := 7078) modelrow_agrees_7079
private def atomic_7080 : CountSegment 1 10 13120 7079 7080 3033 3033 := CountSegment.single (first := 7079) (initial := 3033) row_count_7080 (by decide)
private theorem atomic_7080_agreement : RowAgreement 7079 7080 := RowAgreement.single (first := 7079) modelrow_agrees_7080
private def atomic_7081 : CountSegment 1 10 13120 7080 7081 3033 3033 := CountSegment.single (first := 7080) (initial := 3033) row_count_7081 (by decide)
private theorem atomic_7081_agreement : RowAgreement 7080 7081 := RowAgreement.single (first := 7080) modelrow_agrees_7081
private def atomic_7082 : CountSegment 1 10 13120 7081 7082 3033 3033 := CountSegment.single (first := 7081) (initial := 3033) row_count_7082 (by decide)
private theorem atomic_7082_agreement : RowAgreement 7081 7082 := RowAgreement.single (first := 7081) modelrow_agrees_7082
private def atomic_7083 : CountSegment 1 10 13120 7082 7083 3033 3034 := CountSegment.single (first := 7082) (initial := 3033) row_count_7083 (by decide)
private theorem atomic_7083_agreement : RowAgreement 7082 7083 := RowAgreement.single (first := 7082) modelrow_agrees_7083
private def atomic_7084 : CountSegment 1 10 13120 7083 7084 3034 3034 := CountSegment.single (first := 7083) (initial := 3034) row_count_7084 (by decide)
private theorem atomic_7084_agreement : RowAgreement 7083 7084 := RowAgreement.single (first := 7083) modelrow_agrees_7084
private def atomic_7085 : CountSegment 1 10 13120 7084 7085 3034 3035 := CountSegment.single (first := 7084) (initial := 3034) row_count_7085 (by decide)
private theorem atomic_7085_agreement : RowAgreement 7084 7085 := RowAgreement.single (first := 7084) modelrow_agrees_7085
private def atomic_7086 : CountSegment 1 10 13120 7085 7086 3035 3035 := CountSegment.single (first := 7085) (initial := 3035) row_count_7086 (by decide)
private theorem atomic_7086_agreement : RowAgreement 7085 7086 := RowAgreement.single (first := 7085) modelrow_agrees_7086
private def atomic_7087 : CountSegment 1 10 13120 7086 7087 3035 3035 := CountSegment.single (first := 7086) (initial := 3035) row_count_7087 (by decide)
private theorem atomic_7087_agreement : RowAgreement 7086 7087 := RowAgreement.single (first := 7086) modelrow_agrees_7087
private def atomic_7088 : CountSegment 1 10 13120 7087 7088 3035 3036 := CountSegment.single (first := 7087) (initial := 3035) row_count_7088 (by decide)
private theorem atomic_7088_agreement : RowAgreement 7087 7088 := RowAgreement.single (first := 7087) modelrow_agrees_7088
private def atomic_7089 : CountSegment 1 10 13120 7088 7089 3036 3037 := CountSegment.single (first := 7088) (initial := 3036) row_count_7089 (by decide)
private theorem atomic_7089_agreement : RowAgreement 7088 7089 := RowAgreement.single (first := 7088) modelrow_agrees_7089
private def atomic_7090 : CountSegment 1 10 13120 7089 7090 3037 3038 := CountSegment.single (first := 7089) (initial := 3037) row_count_7090 (by decide)
private theorem atomic_7090_agreement : RowAgreement 7089 7090 := RowAgreement.single (first := 7089) modelrow_agrees_7090
private def atomic_7091 : CountSegment 1 10 13120 7090 7091 3038 3038 := CountSegment.single (first := 7090) (initial := 3038) row_count_7091 (by decide)
private theorem atomic_7091_agreement : RowAgreement 7090 7091 := RowAgreement.single (first := 7090) modelrow_agrees_7091
private def atomic_7092 : CountSegment 1 10 13120 7091 7092 3038 3038 := CountSegment.single (first := 7091) (initial := 3038) row_count_7092 (by decide)
private theorem atomic_7092_agreement : RowAgreement 7091 7092 := RowAgreement.single (first := 7091) modelrow_agrees_7092
private def atomic_7093 : CountSegment 1 10 13120 7092 7093 3038 3039 := CountSegment.single (first := 7092) (initial := 3038) row_count_7093 (by decide)
private theorem atomic_7093_agreement : RowAgreement 7092 7093 := RowAgreement.single (first := 7092) modelrow_agrees_7093
private def atomic_7094 : CountSegment 1 10 13120 7093 7094 3039 3040 := CountSegment.single (first := 7093) (initial := 3039) row_count_7094 (by decide)
private theorem atomic_7094_agreement : RowAgreement 7093 7094 := RowAgreement.single (first := 7093) modelrow_agrees_7094
private def atomic_7095 : CountSegment 1 10 13120 7094 7095 3040 3043 := CountSegment.single (first := 7094) (initial := 3040) row_count_7095 (by decide)
private theorem atomic_7095_agreement : RowAgreement 7094 7095 := RowAgreement.single (first := 7094) modelrow_agrees_7095
private def atomic_7096 : CountSegment 1 10 13120 7095 7096 3043 3043 := CountSegment.single (first := 7095) (initial := 3043) row_count_7096 (by decide)
private theorem atomic_7096_agreement : RowAgreement 7095 7096 := RowAgreement.single (first := 7095) modelrow_agrees_7096
private def atomic_7097 : CountSegment 1 10 13120 7096 7097 3043 3044 := CountSegment.single (first := 7096) (initial := 3043) row_count_7097 (by decide)
private theorem atomic_7097_agreement : RowAgreement 7096 7097 := RowAgreement.single (first := 7096) modelrow_agrees_7097
private def atomic_7098 : CountSegment 1 10 13120 7097 7098 3044 3045 := CountSegment.single (first := 7097) (initial := 3044) row_count_7098 (by decide)
private theorem atomic_7098_agreement : RowAgreement 7097 7098 := RowAgreement.single (first := 7097) modelrow_agrees_7098
private def atomic_7099 : CountSegment 1 10 13120 7098 7099 3045 3045 := CountSegment.single (first := 7098) (initial := 3045) row_count_7099 (by decide)
private theorem atomic_7099_agreement : RowAgreement 7098 7099 := RowAgreement.single (first := 7098) modelrow_agrees_7099
private def atomic_7100 : CountSegment 1 10 13120 7099 7100 3045 3046 := CountSegment.single (first := 7099) (initial := 3045) row_count_7100 (by decide)
private theorem atomic_7100_agreement : RowAgreement 7099 7100 := RowAgreement.single (first := 7099) modelrow_agrees_7100
private def atomic_7101 : CountSegment 1 10 13120 7100 7101 3046 3046 := CountSegment.single (first := 7100) (initial := 3046) row_count_7101 (by decide)
private theorem atomic_7101_agreement : RowAgreement 7100 7101 := RowAgreement.single (first := 7100) modelrow_agrees_7101
private def atomic_7102 : CountSegment 1 10 13120 7101 7102 3046 3046 := CountSegment.single (first := 7101) (initial := 3046) row_count_7102 (by decide)
private theorem atomic_7102_agreement : RowAgreement 7101 7102 := RowAgreement.single (first := 7101) modelrow_agrees_7102
private def atomic_7103 : CountSegment 1 10 13120 7102 7103 3046 3046 := CountSegment.single (first := 7102) (initial := 3046) row_count_7103 (by decide)
private theorem atomic_7103_agreement : RowAgreement 7102 7103 := RowAgreement.single (first := 7102) modelrow_agrees_7103
private def atomic_7104 : CountSegment 1 10 13120 7103 7104 3046 3047 := CountSegment.single (first := 7103) (initial := 3046) row_count_7104 (by decide)
private theorem atomic_7104_agreement : RowAgreement 7103 7104 := RowAgreement.single (first := 7103) modelrow_agrees_7104
private def atomic_7105 : CountSegment 1 10 13120 7104 7105 3047 3048 := CountSegment.single (first := 7104) (initial := 3047) row_count_7105 (by decide)
private theorem atomic_7105_agreement : RowAgreement 7104 7105 := RowAgreement.single (first := 7104) modelrow_agrees_7105
private def atomic_7106 : CountSegment 1 10 13120 7105 7106 3048 3048 := CountSegment.single (first := 7105) (initial := 3048) row_count_7106 (by decide)
private theorem atomic_7106_agreement : RowAgreement 7105 7106 := RowAgreement.single (first := 7105) modelrow_agrees_7106
private def atomic_7107 : CountSegment 1 10 13120 7106 7107 3048 3049 := CountSegment.single (first := 7106) (initial := 3048) row_count_7107 (by decide)
private theorem atomic_7107_agreement : RowAgreement 7106 7107 := RowAgreement.single (first := 7106) modelrow_agrees_7107
private def atomic_7108 : CountSegment 1 10 13120 7107 7108 3049 3049 := CountSegment.single (first := 7107) (initial := 3049) row_count_7108 (by decide)
private theorem atomic_7108_agreement : RowAgreement 7107 7108 := RowAgreement.single (first := 7107) modelrow_agrees_7108
private def atomic_7109 : CountSegment 1 10 13120 7108 7109 3049 3050 := CountSegment.single (first := 7108) (initial := 3049) row_count_7109 (by decide)
private theorem atomic_7109_agreement : RowAgreement 7108 7109 := RowAgreement.single (first := 7108) modelrow_agrees_7109
private def atomic_7110 : CountSegment 1 10 13120 7109 7110 3050 3051 := CountSegment.single (first := 7109) (initial := 3050) row_count_7110 (by decide)
private theorem atomic_7110_agreement : RowAgreement 7109 7110 := RowAgreement.single (first := 7109) modelrow_agrees_7110
private def atomic_7111 : CountSegment 1 10 13120 7110 7111 3051 3052 := CountSegment.single (first := 7110) (initial := 3051) row_count_7111 (by decide)
private theorem atomic_7111_agreement : RowAgreement 7110 7111 := RowAgreement.single (first := 7110) modelrow_agrees_7111
private def atomic_7112 : CountSegment 1 10 13120 7111 7112 3052 3052 := CountSegment.single (first := 7111) (initial := 3052) row_count_7112 (by decide)
private theorem atomic_7112_agreement : RowAgreement 7111 7112 := RowAgreement.single (first := 7111) modelrow_agrees_7112
private def atomic_7113 : CountSegment 1 10 13120 7112 7113 3052 3053 := CountSegment.single (first := 7112) (initial := 3052) row_count_7113 (by decide)
private theorem atomic_7113_agreement : RowAgreement 7112 7113 := RowAgreement.single (first := 7112) modelrow_agrees_7113
private def atomic_7114 : CountSegment 1 10 13120 7113 7114 3053 3053 := CountSegment.single (first := 7113) (initial := 3053) row_count_7114 (by decide)
private theorem atomic_7114_agreement : RowAgreement 7113 7114 := RowAgreement.single (first := 7113) modelrow_agrees_7114
private def atomic_7115 : CountSegment 1 10 13120 7114 7115 3053 3053 := CountSegment.single (first := 7114) (initial := 3053) row_count_7115 (by decide)
private theorem atomic_7115_agreement : RowAgreement 7114 7115 := RowAgreement.single (first := 7114) modelrow_agrees_7115
private def atomic_7116 : CountSegment 1 10 13120 7115 7116 3053 3053 := CountSegment.single (first := 7115) (initial := 3053) row_count_7116 (by decide)
private theorem atomic_7116_agreement : RowAgreement 7115 7116 := RowAgreement.single (first := 7115) modelrow_agrees_7116
private def atomic_7117 : CountSegment 1 10 13120 7116 7117 3053 3053 := CountSegment.single (first := 7116) (initial := 3053) row_count_7117 (by decide)
private theorem atomic_7117_agreement : RowAgreement 7116 7117 := RowAgreement.single (first := 7116) modelrow_agrees_7117
private def atomic_7118 : CountSegment 1 10 13120 7117 7118 3053 3053 := CountSegment.single (first := 7117) (initial := 3053) row_count_7118 (by decide)
private theorem atomic_7118_agreement : RowAgreement 7117 7118 := RowAgreement.single (first := 7117) modelrow_agrees_7118
private def atomic_7119 : CountSegment 1 10 13120 7118 7119 3053 3053 := CountSegment.single (first := 7118) (initial := 3053) row_count_7119 (by decide)
private theorem atomic_7119_agreement : RowAgreement 7118 7119 := RowAgreement.single (first := 7118) modelrow_agrees_7119
private def atomic_7120 : CountSegment 1 10 13120 7119 7120 3053 3053 := CountSegment.single (first := 7119) (initial := 3053) row_count_7120 (by decide)
private theorem atomic_7120_agreement : RowAgreement 7119 7120 := RowAgreement.single (first := 7119) modelrow_agrees_7120
private def atomic_7121 : CountSegment 1 10 13120 7120 7121 3053 3053 := CountSegment.single (first := 7120) (initial := 3053) row_count_7121 (by decide)
private theorem atomic_7121_agreement : RowAgreement 7120 7121 := RowAgreement.single (first := 7120) modelrow_agrees_7121
private def atomic_7122 : CountSegment 1 10 13120 7121 7122 3053 3053 := CountSegment.single (first := 7121) (initial := 3053) row_count_7122 (by decide)
private theorem atomic_7122_agreement : RowAgreement 7121 7122 := RowAgreement.single (first := 7121) modelrow_agrees_7122
private def atomic_7123 : CountSegment 1 10 13120 7122 7123 3053 3054 := CountSegment.single (first := 7122) (initial := 3053) row_count_7123 (by decide)
private theorem atomic_7123_agreement : RowAgreement 7122 7123 := RowAgreement.single (first := 7122) modelrow_agrees_7123
private def atomic_7124 : CountSegment 1 10 13120 7123 7124 3054 3054 := CountSegment.single (first := 7123) (initial := 3054) row_count_7124 (by decide)
private theorem atomic_7124_agreement : RowAgreement 7123 7124 := RowAgreement.single (first := 7123) modelrow_agrees_7124
private def atomic_7125 : CountSegment 1 10 13120 7124 7125 3054 3054 := CountSegment.single (first := 7124) (initial := 3054) row_count_7125 (by decide)
private theorem atomic_7125_agreement : RowAgreement 7124 7125 := RowAgreement.single (first := 7124) modelrow_agrees_7125
private def atomic_7126 : CountSegment 1 10 13120 7125 7126 3054 3054 := CountSegment.single (first := 7125) (initial := 3054) row_count_7126 (by decide)
private theorem atomic_7126_agreement : RowAgreement 7125 7126 := RowAgreement.single (first := 7125) modelrow_agrees_7126
private def atomic_7127 : CountSegment 1 10 13120 7126 7127 3054 3055 := CountSegment.single (first := 7126) (initial := 3054) row_count_7127 (by decide)
private theorem atomic_7127_agreement : RowAgreement 7126 7127 := RowAgreement.single (first := 7126) modelrow_agrees_7127
private def atomic_7128 : CountSegment 1 10 13120 7127 7128 3055 3055 := CountSegment.single (first := 7127) (initial := 3055) row_count_7128 (by decide)
private theorem atomic_7128_agreement : RowAgreement 7127 7128 := RowAgreement.single (first := 7127) modelrow_agrees_7128
private def atomic_7129 : CountSegment 1 10 13120 7128 7129 3055 3055 := CountSegment.single (first := 7128) (initial := 3055) row_count_7129 (by decide)
private theorem atomic_7129_agreement : RowAgreement 7128 7129 := RowAgreement.single (first := 7128) modelrow_agrees_7129
private def atomic_7130 : CountSegment 1 10 13120 7129 7130 3055 3055 := CountSegment.single (first := 7129) (initial := 3055) row_count_7130 (by decide)
private theorem atomic_7130_agreement : RowAgreement 7129 7130 := RowAgreement.single (first := 7129) modelrow_agrees_7130
private def atomic_7131 : CountSegment 1 10 13120 7130 7131 3055 3055 := CountSegment.single (first := 7130) (initial := 3055) row_count_7131 (by decide)
private theorem atomic_7131_agreement : RowAgreement 7130 7131 := RowAgreement.single (first := 7130) modelrow_agrees_7131
private def atomic_7132 : CountSegment 1 10 13120 7131 7132 3055 3056 := CountSegment.single (first := 7131) (initial := 3055) row_count_7132 (by decide)
private theorem atomic_7132_agreement : RowAgreement 7131 7132 := RowAgreement.single (first := 7131) modelrow_agrees_7132
private def atomic_7133 : CountSegment 1 10 13120 7132 7133 3056 3056 := CountSegment.single (first := 7132) (initial := 3056) row_count_7133 (by decide)
private theorem atomic_7133_agreement : RowAgreement 7132 7133 := RowAgreement.single (first := 7132) modelrow_agrees_7133
private def atomic_7134 : CountSegment 1 10 13120 7133 7134 3056 3057 := CountSegment.single (first := 7133) (initial := 3056) row_count_7134 (by decide)
private theorem atomic_7134_agreement : RowAgreement 7133 7134 := RowAgreement.single (first := 7133) modelrow_agrees_7134
private def atomic_7135 : CountSegment 1 10 13120 7134 7135 3057 3058 := CountSegment.single (first := 7134) (initial := 3057) row_count_7135 (by decide)
private theorem atomic_7135_agreement : RowAgreement 7134 7135 := RowAgreement.single (first := 7134) modelrow_agrees_7135
private def atomic_7136 : CountSegment 1 10 13120 7135 7136 3058 3058 := CountSegment.single (first := 7135) (initial := 3058) row_count_7136 (by decide)
private theorem atomic_7136_agreement : RowAgreement 7135 7136 := RowAgreement.single (first := 7135) modelrow_agrees_7136
private def atomic_7137 : CountSegment 1 10 13120 7136 7137 3058 3059 := CountSegment.single (first := 7136) (initial := 3058) row_count_7137 (by decide)
private theorem atomic_7137_agreement : RowAgreement 7136 7137 := RowAgreement.single (first := 7136) modelrow_agrees_7137
private def atomic_7138 : CountSegment 1 10 13120 7137 7138 3059 3059 := CountSegment.single (first := 7137) (initial := 3059) row_count_7138 (by decide)
private theorem atomic_7138_agreement : RowAgreement 7137 7138 := RowAgreement.single (first := 7137) modelrow_agrees_7138
private def atomic_7139 : CountSegment 1 10 13120 7138 7139 3059 3059 := CountSegment.single (first := 7138) (initial := 3059) row_count_7139 (by decide)
private theorem atomic_7139_agreement : RowAgreement 7138 7139 := RowAgreement.single (first := 7138) modelrow_agrees_7139
private def atomic_7140 : CountSegment 1 10 13120 7139 7140 3059 3060 := CountSegment.single (first := 7139) (initial := 3059) row_count_7140 (by decide)
private theorem atomic_7140_agreement : RowAgreement 7139 7140 := RowAgreement.single (first := 7139) modelrow_agrees_7140
private def atomic_7141 : CountSegment 1 10 13120 7140 7141 3060 3060 := CountSegment.single (first := 7140) (initial := 3060) row_count_7141 (by decide)
private theorem atomic_7141_agreement : RowAgreement 7140 7141 := RowAgreement.single (first := 7140) modelrow_agrees_7141
private def atomic_7142 : CountSegment 1 10 13120 7141 7142 3060 3060 := CountSegment.single (first := 7141) (initial := 3060) row_count_7142 (by decide)
private theorem atomic_7142_agreement : RowAgreement 7141 7142 := RowAgreement.single (first := 7141) modelrow_agrees_7142
private def atomic_7143 : CountSegment 1 10 13120 7142 7143 3060 3061 := CountSegment.single (first := 7142) (initial := 3060) row_count_7143 (by decide)
private theorem atomic_7143_agreement : RowAgreement 7142 7143 := RowAgreement.single (first := 7142) modelrow_agrees_7143
private def atomic_7144 : CountSegment 1 10 13120 7143 7144 3061 3062 := CountSegment.single (first := 7143) (initial := 3061) row_count_7144 (by decide)
private theorem atomic_7144_agreement : RowAgreement 7143 7144 := RowAgreement.single (first := 7143) modelrow_agrees_7144
private def atomic_7145 : CountSegment 1 10 13120 7144 7145 3062 3063 := CountSegment.single (first := 7144) (initial := 3062) row_count_7145 (by decide)
private theorem atomic_7145_agreement : RowAgreement 7144 7145 := RowAgreement.single (first := 7144) modelrow_agrees_7145
private def atomic_7146 : CountSegment 1 10 13120 7145 7146 3063 3063 := CountSegment.single (first := 7145) (initial := 3063) row_count_7146 (by decide)
private theorem atomic_7146_agreement : RowAgreement 7145 7146 := RowAgreement.single (first := 7145) modelrow_agrees_7146
private def atomic_7147 : CountSegment 1 10 13120 7146 7147 3063 3064 := CountSegment.single (first := 7146) (initial := 3063) row_count_7147 (by decide)
private theorem atomic_7147_agreement : RowAgreement 7146 7147 := RowAgreement.single (first := 7146) modelrow_agrees_7147
private def atomic_7148 : CountSegment 1 10 13120 7147 7148 3064 3064 := CountSegment.single (first := 7147) (initial := 3064) row_count_7148 (by decide)
private theorem atomic_7148_agreement : RowAgreement 7147 7148 := RowAgreement.single (first := 7147) modelrow_agrees_7148
private def atomic_7149 : CountSegment 1 10 13120 7148 7149 3064 3065 := CountSegment.single (first := 7148) (initial := 3064) row_count_7149 (by decide)
private theorem atomic_7149_agreement : RowAgreement 7148 7149 := RowAgreement.single (first := 7148) modelrow_agrees_7149
private def atomic_7150 : CountSegment 1 10 13120 7149 7150 3065 3065 := CountSegment.single (first := 7149) (initial := 3065) row_count_7150 (by decide)
private theorem atomic_7150_agreement : RowAgreement 7149 7150 := RowAgreement.single (first := 7149) modelrow_agrees_7150
private def atomic_7151 : CountSegment 1 10 13120 7150 7151 3065 3065 := CountSegment.single (first := 7150) (initial := 3065) row_count_7151 (by decide)
private theorem atomic_7151_agreement : RowAgreement 7150 7151 := RowAgreement.single (first := 7150) modelrow_agrees_7151
private def atomic_7152 : CountSegment 1 10 13120 7151 7152 3065 3066 := CountSegment.single (first := 7151) (initial := 3065) row_count_7152 (by decide)
private theorem atomic_7152_agreement : RowAgreement 7151 7152 := RowAgreement.single (first := 7151) modelrow_agrees_7152
private def atomic_7153 : CountSegment 1 10 13120 7152 7153 3066 3067 := CountSegment.single (first := 7152) (initial := 3066) row_count_7153 (by decide)
private theorem atomic_7153_agreement : RowAgreement 7152 7153 := RowAgreement.single (first := 7152) modelrow_agrees_7153
private def atomic_7154 : CountSegment 1 10 13120 7153 7154 3067 3068 := CountSegment.single (first := 7153) (initial := 3067) row_count_7154 (by decide)
private theorem atomic_7154_agreement : RowAgreement 7153 7154 := RowAgreement.single (first := 7153) modelrow_agrees_7154
private def atomic_7155 : CountSegment 1 10 13120 7154 7155 3068 3068 := CountSegment.single (first := 7154) (initial := 3068) row_count_7155 (by decide)
private theorem atomic_7155_agreement : RowAgreement 7154 7155 := RowAgreement.single (first := 7154) modelrow_agrees_7155
private def atomic_7156 : CountSegment 1 10 13120 7155 7156 3068 3068 := CountSegment.single (first := 7155) (initial := 3068) row_count_7156 (by decide)
private theorem atomic_7156_agreement : RowAgreement 7155 7156 := RowAgreement.single (first := 7155) modelrow_agrees_7156
private def atomic_7157 : CountSegment 1 10 13120 7156 7157 3068 3069 := CountSegment.single (first := 7156) (initial := 3068) row_count_7157 (by decide)
private theorem atomic_7157_agreement : RowAgreement 7156 7157 := RowAgreement.single (first := 7156) modelrow_agrees_7157
private def atomic_7158 : CountSegment 1 10 13120 7157 7158 3069 3069 := CountSegment.single (first := 7157) (initial := 3069) row_count_7158 (by decide)
private theorem atomic_7158_agreement : RowAgreement 7157 7158 := RowAgreement.single (first := 7157) modelrow_agrees_7158
private def atomic_7159 : CountSegment 1 10 13120 7158 7159 3069 3069 := CountSegment.single (first := 7158) (initial := 3069) row_count_7159 (by decide)
private theorem atomic_7159_agreement : RowAgreement 7158 7159 := RowAgreement.single (first := 7158) modelrow_agrees_7159
private def atomic_7160 : CountSegment 1 10 13120 7159 7160 3069 3069 := CountSegment.single (first := 7159) (initial := 3069) row_count_7160 (by decide)
private theorem atomic_7160_agreement : RowAgreement 7159 7160 := RowAgreement.single (first := 7159) modelrow_agrees_7160
private def atomic_7161 : CountSegment 1 10 13120 7160 7161 3069 3069 := CountSegment.single (first := 7160) (initial := 3069) row_count_7161 (by decide)
private theorem atomic_7161_agreement : RowAgreement 7160 7161 := RowAgreement.single (first := 7160) modelrow_agrees_7161
private def atomic_7162 : CountSegment 1 10 13120 7161 7162 3069 3069 := CountSegment.single (first := 7161) (initial := 3069) row_count_7162 (by decide)
private theorem atomic_7162_agreement : RowAgreement 7161 7162 := RowAgreement.single (first := 7161) modelrow_agrees_7162
private def atomic_7163 : CountSegment 1 10 13120 7162 7163 3069 3069 := CountSegment.single (first := 7162) (initial := 3069) row_count_7163 (by decide)
private theorem atomic_7163_agreement : RowAgreement 7162 7163 := RowAgreement.single (first := 7162) modelrow_agrees_7163
private def atomic_7164 : CountSegment 1 10 13120 7163 7164 3069 3070 := CountSegment.single (first := 7163) (initial := 3069) row_count_7164 (by decide)
private theorem atomic_7164_agreement : RowAgreement 7163 7164 := RowAgreement.single (first := 7163) modelrow_agrees_7164
private def atomic_7165 : CountSegment 1 10 13120 7164 7165 3070 3072 := CountSegment.single (first := 7164) (initial := 3070) row_count_7165 (by decide)
private theorem atomic_7165_agreement : RowAgreement 7164 7165 := RowAgreement.single (first := 7164) modelrow_agrees_7165
private def atomic_7166 : CountSegment 1 10 13120 7165 7166 3072 3072 := CountSegment.single (first := 7165) (initial := 3072) row_count_7166 (by decide)
private theorem atomic_7166_agreement : RowAgreement 7165 7166 := RowAgreement.single (first := 7165) modelrow_agrees_7166
private def atomic_7167 : CountSegment 1 10 13120 7166 7167 3072 3072 := CountSegment.single (first := 7166) (initial := 3072) row_count_7167 (by decide)
private theorem atomic_7167_agreement : RowAgreement 7166 7167 := RowAgreement.single (first := 7166) modelrow_agrees_7167
private def atomic_7168 : CountSegment 1 10 13120 7167 7168 3072 3072 := CountSegment.single (first := 7167) (initial := 3072) row_count_7168 (by decide)
private theorem atomic_7168_agreement : RowAgreement 7167 7168 := RowAgreement.single (first := 7167) modelrow_agrees_7168
private def within_055_000 : CountSegment 1 10 13120 7040 7042 3015 3016 := CountSegment.join atomic_7041 atomic_7042
private theorem within_055_000_agreement : RowAgreement 7040 7042 := RowAgreement.join atomic_7041_agreement atomic_7042_agreement
private def within_055_001 : CountSegment 1 10 13120 7042 7044 3016 3018 := CountSegment.join atomic_7043 atomic_7044
private theorem within_055_001_agreement : RowAgreement 7042 7044 := RowAgreement.join atomic_7043_agreement atomic_7044_agreement
private def within_055_002 : CountSegment 1 10 13120 7040 7044 3015 3018 := CountSegment.join within_055_000 within_055_001
private theorem within_055_002_agreement : RowAgreement 7040 7044 := RowAgreement.join within_055_000_agreement within_055_001_agreement
private def within_055_003 : CountSegment 1 10 13120 7044 7046 3018 3019 := CountSegment.join atomic_7045 atomic_7046
private theorem within_055_003_agreement : RowAgreement 7044 7046 := RowAgreement.join atomic_7045_agreement atomic_7046_agreement
private def within_055_004 : CountSegment 1 10 13120 7046 7048 3019 3020 := CountSegment.join atomic_7047 atomic_7048
private theorem within_055_004_agreement : RowAgreement 7046 7048 := RowAgreement.join atomic_7047_agreement atomic_7048_agreement
private def within_055_005 : CountSegment 1 10 13120 7044 7048 3018 3020 := CountSegment.join within_055_003 within_055_004
private theorem within_055_005_agreement : RowAgreement 7044 7048 := RowAgreement.join within_055_003_agreement within_055_004_agreement
private def within_055_006 : CountSegment 1 10 13120 7040 7048 3015 3020 := CountSegment.join within_055_002 within_055_005
private theorem within_055_006_agreement : RowAgreement 7040 7048 := RowAgreement.join within_055_002_agreement within_055_005_agreement
private def within_055_007 : CountSegment 1 10 13120 7048 7050 3020 3020 := CountSegment.join atomic_7049 atomic_7050
private theorem within_055_007_agreement : RowAgreement 7048 7050 := RowAgreement.join atomic_7049_agreement atomic_7050_agreement
private def within_055_008 : CountSegment 1 10 13120 7050 7052 3020 3021 := CountSegment.join atomic_7051 atomic_7052
private theorem within_055_008_agreement : RowAgreement 7050 7052 := RowAgreement.join atomic_7051_agreement atomic_7052_agreement
private def within_055_009 : CountSegment 1 10 13120 7048 7052 3020 3021 := CountSegment.join within_055_007 within_055_008
private theorem within_055_009_agreement : RowAgreement 7048 7052 := RowAgreement.join within_055_007_agreement within_055_008_agreement
private def within_055_010 : CountSegment 1 10 13120 7052 7054 3021 3021 := CountSegment.join atomic_7053 atomic_7054
private theorem within_055_010_agreement : RowAgreement 7052 7054 := RowAgreement.join atomic_7053_agreement atomic_7054_agreement
private def within_055_011 : CountSegment 1 10 13120 7054 7056 3021 3022 := CountSegment.join atomic_7055 atomic_7056
private theorem within_055_011_agreement : RowAgreement 7054 7056 := RowAgreement.join atomic_7055_agreement atomic_7056_agreement
private def within_055_012 : CountSegment 1 10 13120 7052 7056 3021 3022 := CountSegment.join within_055_010 within_055_011
private theorem within_055_012_agreement : RowAgreement 7052 7056 := RowAgreement.join within_055_010_agreement within_055_011_agreement
private def within_055_013 : CountSegment 1 10 13120 7048 7056 3020 3022 := CountSegment.join within_055_009 within_055_012
private theorem within_055_013_agreement : RowAgreement 7048 7056 := RowAgreement.join within_055_009_agreement within_055_012_agreement
private def within_055_014 : CountSegment 1 10 13120 7040 7056 3015 3022 := CountSegment.join within_055_006 within_055_013
private theorem within_055_014_agreement : RowAgreement 7040 7056 := RowAgreement.join within_055_006_agreement within_055_013_agreement
private def within_055_015 : CountSegment 1 10 13120 7056 7058 3022 3022 := CountSegment.join atomic_7057 atomic_7058
private theorem within_055_015_agreement : RowAgreement 7056 7058 := RowAgreement.join atomic_7057_agreement atomic_7058_agreement
private def within_055_016 : CountSegment 1 10 13120 7058 7060 3022 3024 := CountSegment.join atomic_7059 atomic_7060
private theorem within_055_016_agreement : RowAgreement 7058 7060 := RowAgreement.join atomic_7059_agreement atomic_7060_agreement
private def within_055_017 : CountSegment 1 10 13120 7056 7060 3022 3024 := CountSegment.join within_055_015 within_055_016
private theorem within_055_017_agreement : RowAgreement 7056 7060 := RowAgreement.join within_055_015_agreement within_055_016_agreement
private def within_055_018 : CountSegment 1 10 13120 7060 7062 3024 3024 := CountSegment.join atomic_7061 atomic_7062
private theorem within_055_018_agreement : RowAgreement 7060 7062 := RowAgreement.join atomic_7061_agreement atomic_7062_agreement
private def within_055_019 : CountSegment 1 10 13120 7062 7064 3024 3024 := CountSegment.join atomic_7063 atomic_7064
private theorem within_055_019_agreement : RowAgreement 7062 7064 := RowAgreement.join atomic_7063_agreement atomic_7064_agreement
private def within_055_020 : CountSegment 1 10 13120 7060 7064 3024 3024 := CountSegment.join within_055_018 within_055_019
private theorem within_055_020_agreement : RowAgreement 7060 7064 := RowAgreement.join within_055_018_agreement within_055_019_agreement
private def within_055_021 : CountSegment 1 10 13120 7056 7064 3022 3024 := CountSegment.join within_055_017 within_055_020
private theorem within_055_021_agreement : RowAgreement 7056 7064 := RowAgreement.join within_055_017_agreement within_055_020_agreement
private def within_055_022 : CountSegment 1 10 13120 7064 7066 3024 3026 := CountSegment.join atomic_7065 atomic_7066
private theorem within_055_022_agreement : RowAgreement 7064 7066 := RowAgreement.join atomic_7065_agreement atomic_7066_agreement
private def within_055_023 : CountSegment 1 10 13120 7066 7068 3026 3028 := CountSegment.join atomic_7067 atomic_7068
private theorem within_055_023_agreement : RowAgreement 7066 7068 := RowAgreement.join atomic_7067_agreement atomic_7068_agreement
private def within_055_024 : CountSegment 1 10 13120 7064 7068 3024 3028 := CountSegment.join within_055_022 within_055_023
private theorem within_055_024_agreement : RowAgreement 7064 7068 := RowAgreement.join within_055_022_agreement within_055_023_agreement
private def within_055_025 : CountSegment 1 10 13120 7068 7070 3028 3028 := CountSegment.join atomic_7069 atomic_7070
private theorem within_055_025_agreement : RowAgreement 7068 7070 := RowAgreement.join atomic_7069_agreement atomic_7070_agreement
private def within_055_026 : CountSegment 1 10 13120 7070 7072 3028 3029 := CountSegment.join atomic_7071 atomic_7072
private theorem within_055_026_agreement : RowAgreement 7070 7072 := RowAgreement.join atomic_7071_agreement atomic_7072_agreement
private def within_055_027 : CountSegment 1 10 13120 7068 7072 3028 3029 := CountSegment.join within_055_025 within_055_026
private theorem within_055_027_agreement : RowAgreement 7068 7072 := RowAgreement.join within_055_025_agreement within_055_026_agreement
private def within_055_028 : CountSegment 1 10 13120 7064 7072 3024 3029 := CountSegment.join within_055_024 within_055_027
private theorem within_055_028_agreement : RowAgreement 7064 7072 := RowAgreement.join within_055_024_agreement within_055_027_agreement
private def within_055_029 : CountSegment 1 10 13120 7056 7072 3022 3029 := CountSegment.join within_055_021 within_055_028
private theorem within_055_029_agreement : RowAgreement 7056 7072 := RowAgreement.join within_055_021_agreement within_055_028_agreement
private def within_055_030 : CountSegment 1 10 13120 7040 7072 3015 3029 := CountSegment.join within_055_014 within_055_029
private theorem within_055_030_agreement : RowAgreement 7040 7072 := RowAgreement.join within_055_014_agreement within_055_029_agreement
private def within_055_031 : CountSegment 1 10 13120 7072 7074 3029 3031 := CountSegment.join atomic_7073 atomic_7074
private theorem within_055_031_agreement : RowAgreement 7072 7074 := RowAgreement.join atomic_7073_agreement atomic_7074_agreement
private def within_055_032 : CountSegment 1 10 13120 7074 7076 3031 3031 := CountSegment.join atomic_7075 atomic_7076
private theorem within_055_032_agreement : RowAgreement 7074 7076 := RowAgreement.join atomic_7075_agreement atomic_7076_agreement
private def within_055_033 : CountSegment 1 10 13120 7072 7076 3029 3031 := CountSegment.join within_055_031 within_055_032
private theorem within_055_033_agreement : RowAgreement 7072 7076 := RowAgreement.join within_055_031_agreement within_055_032_agreement
private def within_055_034 : CountSegment 1 10 13120 7076 7078 3031 3033 := CountSegment.join atomic_7077 atomic_7078
private theorem within_055_034_agreement : RowAgreement 7076 7078 := RowAgreement.join atomic_7077_agreement atomic_7078_agreement
private def within_055_035 : CountSegment 1 10 13120 7078 7080 3033 3033 := CountSegment.join atomic_7079 atomic_7080
private theorem within_055_035_agreement : RowAgreement 7078 7080 := RowAgreement.join atomic_7079_agreement atomic_7080_agreement
private def within_055_036 : CountSegment 1 10 13120 7076 7080 3031 3033 := CountSegment.join within_055_034 within_055_035
private theorem within_055_036_agreement : RowAgreement 7076 7080 := RowAgreement.join within_055_034_agreement within_055_035_agreement
private def within_055_037 : CountSegment 1 10 13120 7072 7080 3029 3033 := CountSegment.join within_055_033 within_055_036
private theorem within_055_037_agreement : RowAgreement 7072 7080 := RowAgreement.join within_055_033_agreement within_055_036_agreement
private def within_055_038 : CountSegment 1 10 13120 7080 7082 3033 3033 := CountSegment.join atomic_7081 atomic_7082
private theorem within_055_038_agreement : RowAgreement 7080 7082 := RowAgreement.join atomic_7081_agreement atomic_7082_agreement
private def within_055_039 : CountSegment 1 10 13120 7082 7084 3033 3034 := CountSegment.join atomic_7083 atomic_7084
private theorem within_055_039_agreement : RowAgreement 7082 7084 := RowAgreement.join atomic_7083_agreement atomic_7084_agreement
private def within_055_040 : CountSegment 1 10 13120 7080 7084 3033 3034 := CountSegment.join within_055_038 within_055_039
private theorem within_055_040_agreement : RowAgreement 7080 7084 := RowAgreement.join within_055_038_agreement within_055_039_agreement
private def within_055_041 : CountSegment 1 10 13120 7084 7086 3034 3035 := CountSegment.join atomic_7085 atomic_7086
private theorem within_055_041_agreement : RowAgreement 7084 7086 := RowAgreement.join atomic_7085_agreement atomic_7086_agreement
private def within_055_042 : CountSegment 1 10 13120 7086 7088 3035 3036 := CountSegment.join atomic_7087 atomic_7088
private theorem within_055_042_agreement : RowAgreement 7086 7088 := RowAgreement.join atomic_7087_agreement atomic_7088_agreement
private def within_055_043 : CountSegment 1 10 13120 7084 7088 3034 3036 := CountSegment.join within_055_041 within_055_042
private theorem within_055_043_agreement : RowAgreement 7084 7088 := RowAgreement.join within_055_041_agreement within_055_042_agreement
private def within_055_044 : CountSegment 1 10 13120 7080 7088 3033 3036 := CountSegment.join within_055_040 within_055_043
private theorem within_055_044_agreement : RowAgreement 7080 7088 := RowAgreement.join within_055_040_agreement within_055_043_agreement
private def within_055_045 : CountSegment 1 10 13120 7072 7088 3029 3036 := CountSegment.join within_055_037 within_055_044
private theorem within_055_045_agreement : RowAgreement 7072 7088 := RowAgreement.join within_055_037_agreement within_055_044_agreement
private def within_055_046 : CountSegment 1 10 13120 7088 7090 3036 3038 := CountSegment.join atomic_7089 atomic_7090
private theorem within_055_046_agreement : RowAgreement 7088 7090 := RowAgreement.join atomic_7089_agreement atomic_7090_agreement
private def within_055_047 : CountSegment 1 10 13120 7090 7092 3038 3038 := CountSegment.join atomic_7091 atomic_7092
private theorem within_055_047_agreement : RowAgreement 7090 7092 := RowAgreement.join atomic_7091_agreement atomic_7092_agreement
private def within_055_048 : CountSegment 1 10 13120 7088 7092 3036 3038 := CountSegment.join within_055_046 within_055_047
private theorem within_055_048_agreement : RowAgreement 7088 7092 := RowAgreement.join within_055_046_agreement within_055_047_agreement
private def within_055_049 : CountSegment 1 10 13120 7092 7094 3038 3040 := CountSegment.join atomic_7093 atomic_7094
private theorem within_055_049_agreement : RowAgreement 7092 7094 := RowAgreement.join atomic_7093_agreement atomic_7094_agreement
private def within_055_050 : CountSegment 1 10 13120 7094 7096 3040 3043 := CountSegment.join atomic_7095 atomic_7096
private theorem within_055_050_agreement : RowAgreement 7094 7096 := RowAgreement.join atomic_7095_agreement atomic_7096_agreement
private def within_055_051 : CountSegment 1 10 13120 7092 7096 3038 3043 := CountSegment.join within_055_049 within_055_050
private theorem within_055_051_agreement : RowAgreement 7092 7096 := RowAgreement.join within_055_049_agreement within_055_050_agreement
private def within_055_052 : CountSegment 1 10 13120 7088 7096 3036 3043 := CountSegment.join within_055_048 within_055_051
private theorem within_055_052_agreement : RowAgreement 7088 7096 := RowAgreement.join within_055_048_agreement within_055_051_agreement
private def within_055_053 : CountSegment 1 10 13120 7096 7098 3043 3045 := CountSegment.join atomic_7097 atomic_7098
private theorem within_055_053_agreement : RowAgreement 7096 7098 := RowAgreement.join atomic_7097_agreement atomic_7098_agreement
private def within_055_054 : CountSegment 1 10 13120 7098 7100 3045 3046 := CountSegment.join atomic_7099 atomic_7100
private theorem within_055_054_agreement : RowAgreement 7098 7100 := RowAgreement.join atomic_7099_agreement atomic_7100_agreement
private def within_055_055 : CountSegment 1 10 13120 7096 7100 3043 3046 := CountSegment.join within_055_053 within_055_054
private theorem within_055_055_agreement : RowAgreement 7096 7100 := RowAgreement.join within_055_053_agreement within_055_054_agreement
private def within_055_056 : CountSegment 1 10 13120 7100 7102 3046 3046 := CountSegment.join atomic_7101 atomic_7102
private theorem within_055_056_agreement : RowAgreement 7100 7102 := RowAgreement.join atomic_7101_agreement atomic_7102_agreement
private def within_055_057 : CountSegment 1 10 13120 7102 7104 3046 3047 := CountSegment.join atomic_7103 atomic_7104
private theorem within_055_057_agreement : RowAgreement 7102 7104 := RowAgreement.join atomic_7103_agreement atomic_7104_agreement
private def within_055_058 : CountSegment 1 10 13120 7100 7104 3046 3047 := CountSegment.join within_055_056 within_055_057
private theorem within_055_058_agreement : RowAgreement 7100 7104 := RowAgreement.join within_055_056_agreement within_055_057_agreement
private def within_055_059 : CountSegment 1 10 13120 7096 7104 3043 3047 := CountSegment.join within_055_055 within_055_058
private theorem within_055_059_agreement : RowAgreement 7096 7104 := RowAgreement.join within_055_055_agreement within_055_058_agreement
private def within_055_060 : CountSegment 1 10 13120 7088 7104 3036 3047 := CountSegment.join within_055_052 within_055_059
private theorem within_055_060_agreement : RowAgreement 7088 7104 := RowAgreement.join within_055_052_agreement within_055_059_agreement
private def within_055_061 : CountSegment 1 10 13120 7072 7104 3029 3047 := CountSegment.join within_055_045 within_055_060
private theorem within_055_061_agreement : RowAgreement 7072 7104 := RowAgreement.join within_055_045_agreement within_055_060_agreement
private def within_055_062 : CountSegment 1 10 13120 7040 7104 3015 3047 := CountSegment.join within_055_030 within_055_061
private theorem within_055_062_agreement : RowAgreement 7040 7104 := RowAgreement.join within_055_030_agreement within_055_061_agreement
private def within_055_063 : CountSegment 1 10 13120 7104 7106 3047 3048 := CountSegment.join atomic_7105 atomic_7106
private theorem within_055_063_agreement : RowAgreement 7104 7106 := RowAgreement.join atomic_7105_agreement atomic_7106_agreement
private def within_055_064 : CountSegment 1 10 13120 7106 7108 3048 3049 := CountSegment.join atomic_7107 atomic_7108
private theorem within_055_064_agreement : RowAgreement 7106 7108 := RowAgreement.join atomic_7107_agreement atomic_7108_agreement
private def within_055_065 : CountSegment 1 10 13120 7104 7108 3047 3049 := CountSegment.join within_055_063 within_055_064
private theorem within_055_065_agreement : RowAgreement 7104 7108 := RowAgreement.join within_055_063_agreement within_055_064_agreement
private def within_055_066 : CountSegment 1 10 13120 7108 7110 3049 3051 := CountSegment.join atomic_7109 atomic_7110
private theorem within_055_066_agreement : RowAgreement 7108 7110 := RowAgreement.join atomic_7109_agreement atomic_7110_agreement
private def within_055_067 : CountSegment 1 10 13120 7110 7112 3051 3052 := CountSegment.join atomic_7111 atomic_7112
private theorem within_055_067_agreement : RowAgreement 7110 7112 := RowAgreement.join atomic_7111_agreement atomic_7112_agreement
private def within_055_068 : CountSegment 1 10 13120 7108 7112 3049 3052 := CountSegment.join within_055_066 within_055_067
private theorem within_055_068_agreement : RowAgreement 7108 7112 := RowAgreement.join within_055_066_agreement within_055_067_agreement
private def within_055_069 : CountSegment 1 10 13120 7104 7112 3047 3052 := CountSegment.join within_055_065 within_055_068
private theorem within_055_069_agreement : RowAgreement 7104 7112 := RowAgreement.join within_055_065_agreement within_055_068_agreement
private def within_055_070 : CountSegment 1 10 13120 7112 7114 3052 3053 := CountSegment.join atomic_7113 atomic_7114
private theorem within_055_070_agreement : RowAgreement 7112 7114 := RowAgreement.join atomic_7113_agreement atomic_7114_agreement
private def within_055_071 : CountSegment 1 10 13120 7114 7116 3053 3053 := CountSegment.join atomic_7115 atomic_7116
private theorem within_055_071_agreement : RowAgreement 7114 7116 := RowAgreement.join atomic_7115_agreement atomic_7116_agreement
private def within_055_072 : CountSegment 1 10 13120 7112 7116 3052 3053 := CountSegment.join within_055_070 within_055_071
private theorem within_055_072_agreement : RowAgreement 7112 7116 := RowAgreement.join within_055_070_agreement within_055_071_agreement
private def within_055_073 : CountSegment 1 10 13120 7116 7118 3053 3053 := CountSegment.join atomic_7117 atomic_7118
private theorem within_055_073_agreement : RowAgreement 7116 7118 := RowAgreement.join atomic_7117_agreement atomic_7118_agreement
private def within_055_074 : CountSegment 1 10 13120 7118 7120 3053 3053 := CountSegment.join atomic_7119 atomic_7120
private theorem within_055_074_agreement : RowAgreement 7118 7120 := RowAgreement.join atomic_7119_agreement atomic_7120_agreement
private def within_055_075 : CountSegment 1 10 13120 7116 7120 3053 3053 := CountSegment.join within_055_073 within_055_074
private theorem within_055_075_agreement : RowAgreement 7116 7120 := RowAgreement.join within_055_073_agreement within_055_074_agreement
private def within_055_076 : CountSegment 1 10 13120 7112 7120 3052 3053 := CountSegment.join within_055_072 within_055_075
private theorem within_055_076_agreement : RowAgreement 7112 7120 := RowAgreement.join within_055_072_agreement within_055_075_agreement
private def within_055_077 : CountSegment 1 10 13120 7104 7120 3047 3053 := CountSegment.join within_055_069 within_055_076
private theorem within_055_077_agreement : RowAgreement 7104 7120 := RowAgreement.join within_055_069_agreement within_055_076_agreement
private def within_055_078 : CountSegment 1 10 13120 7120 7122 3053 3053 := CountSegment.join atomic_7121 atomic_7122
private theorem within_055_078_agreement : RowAgreement 7120 7122 := RowAgreement.join atomic_7121_agreement atomic_7122_agreement
private def within_055_079 : CountSegment 1 10 13120 7122 7124 3053 3054 := CountSegment.join atomic_7123 atomic_7124
private theorem within_055_079_agreement : RowAgreement 7122 7124 := RowAgreement.join atomic_7123_agreement atomic_7124_agreement
private def within_055_080 : CountSegment 1 10 13120 7120 7124 3053 3054 := CountSegment.join within_055_078 within_055_079
private theorem within_055_080_agreement : RowAgreement 7120 7124 := RowAgreement.join within_055_078_agreement within_055_079_agreement
private def within_055_081 : CountSegment 1 10 13120 7124 7126 3054 3054 := CountSegment.join atomic_7125 atomic_7126
private theorem within_055_081_agreement : RowAgreement 7124 7126 := RowAgreement.join atomic_7125_agreement atomic_7126_agreement
private def within_055_082 : CountSegment 1 10 13120 7126 7128 3054 3055 := CountSegment.join atomic_7127 atomic_7128
private theorem within_055_082_agreement : RowAgreement 7126 7128 := RowAgreement.join atomic_7127_agreement atomic_7128_agreement
private def within_055_083 : CountSegment 1 10 13120 7124 7128 3054 3055 := CountSegment.join within_055_081 within_055_082
private theorem within_055_083_agreement : RowAgreement 7124 7128 := RowAgreement.join within_055_081_agreement within_055_082_agreement
private def within_055_084 : CountSegment 1 10 13120 7120 7128 3053 3055 := CountSegment.join within_055_080 within_055_083
private theorem within_055_084_agreement : RowAgreement 7120 7128 := RowAgreement.join within_055_080_agreement within_055_083_agreement
private def within_055_085 : CountSegment 1 10 13120 7128 7130 3055 3055 := CountSegment.join atomic_7129 atomic_7130
private theorem within_055_085_agreement : RowAgreement 7128 7130 := RowAgreement.join atomic_7129_agreement atomic_7130_agreement
private def within_055_086 : CountSegment 1 10 13120 7130 7132 3055 3056 := CountSegment.join atomic_7131 atomic_7132
private theorem within_055_086_agreement : RowAgreement 7130 7132 := RowAgreement.join atomic_7131_agreement atomic_7132_agreement
private def within_055_087 : CountSegment 1 10 13120 7128 7132 3055 3056 := CountSegment.join within_055_085 within_055_086
private theorem within_055_087_agreement : RowAgreement 7128 7132 := RowAgreement.join within_055_085_agreement within_055_086_agreement
private def within_055_088 : CountSegment 1 10 13120 7132 7134 3056 3057 := CountSegment.join atomic_7133 atomic_7134
private theorem within_055_088_agreement : RowAgreement 7132 7134 := RowAgreement.join atomic_7133_agreement atomic_7134_agreement
private def within_055_089 : CountSegment 1 10 13120 7134 7136 3057 3058 := CountSegment.join atomic_7135 atomic_7136
private theorem within_055_089_agreement : RowAgreement 7134 7136 := RowAgreement.join atomic_7135_agreement atomic_7136_agreement
private def within_055_090 : CountSegment 1 10 13120 7132 7136 3056 3058 := CountSegment.join within_055_088 within_055_089
private theorem within_055_090_agreement : RowAgreement 7132 7136 := RowAgreement.join within_055_088_agreement within_055_089_agreement
private def within_055_091 : CountSegment 1 10 13120 7128 7136 3055 3058 := CountSegment.join within_055_087 within_055_090
private theorem within_055_091_agreement : RowAgreement 7128 7136 := RowAgreement.join within_055_087_agreement within_055_090_agreement
private def within_055_092 : CountSegment 1 10 13120 7120 7136 3053 3058 := CountSegment.join within_055_084 within_055_091
private theorem within_055_092_agreement : RowAgreement 7120 7136 := RowAgreement.join within_055_084_agreement within_055_091_agreement
private def within_055_093 : CountSegment 1 10 13120 7104 7136 3047 3058 := CountSegment.join within_055_077 within_055_092
private theorem within_055_093_agreement : RowAgreement 7104 7136 := RowAgreement.join within_055_077_agreement within_055_092_agreement
private def within_055_094 : CountSegment 1 10 13120 7136 7138 3058 3059 := CountSegment.join atomic_7137 atomic_7138
private theorem within_055_094_agreement : RowAgreement 7136 7138 := RowAgreement.join atomic_7137_agreement atomic_7138_agreement
private def within_055_095 : CountSegment 1 10 13120 7138 7140 3059 3060 := CountSegment.join atomic_7139 atomic_7140
private theorem within_055_095_agreement : RowAgreement 7138 7140 := RowAgreement.join atomic_7139_agreement atomic_7140_agreement
private def within_055_096 : CountSegment 1 10 13120 7136 7140 3058 3060 := CountSegment.join within_055_094 within_055_095
private theorem within_055_096_agreement : RowAgreement 7136 7140 := RowAgreement.join within_055_094_agreement within_055_095_agreement
private def within_055_097 : CountSegment 1 10 13120 7140 7142 3060 3060 := CountSegment.join atomic_7141 atomic_7142
private theorem within_055_097_agreement : RowAgreement 7140 7142 := RowAgreement.join atomic_7141_agreement atomic_7142_agreement
private def within_055_098 : CountSegment 1 10 13120 7142 7144 3060 3062 := CountSegment.join atomic_7143 atomic_7144
private theorem within_055_098_agreement : RowAgreement 7142 7144 := RowAgreement.join atomic_7143_agreement atomic_7144_agreement
private def within_055_099 : CountSegment 1 10 13120 7140 7144 3060 3062 := CountSegment.join within_055_097 within_055_098
private theorem within_055_099_agreement : RowAgreement 7140 7144 := RowAgreement.join within_055_097_agreement within_055_098_agreement
private def within_055_100 : CountSegment 1 10 13120 7136 7144 3058 3062 := CountSegment.join within_055_096 within_055_099
private theorem within_055_100_agreement : RowAgreement 7136 7144 := RowAgreement.join within_055_096_agreement within_055_099_agreement
private def within_055_101 : CountSegment 1 10 13120 7144 7146 3062 3063 := CountSegment.join atomic_7145 atomic_7146
private theorem within_055_101_agreement : RowAgreement 7144 7146 := RowAgreement.join atomic_7145_agreement atomic_7146_agreement
private def within_055_102 : CountSegment 1 10 13120 7146 7148 3063 3064 := CountSegment.join atomic_7147 atomic_7148
private theorem within_055_102_agreement : RowAgreement 7146 7148 := RowAgreement.join atomic_7147_agreement atomic_7148_agreement
private def within_055_103 : CountSegment 1 10 13120 7144 7148 3062 3064 := CountSegment.join within_055_101 within_055_102
private theorem within_055_103_agreement : RowAgreement 7144 7148 := RowAgreement.join within_055_101_agreement within_055_102_agreement
private def within_055_104 : CountSegment 1 10 13120 7148 7150 3064 3065 := CountSegment.join atomic_7149 atomic_7150
private theorem within_055_104_agreement : RowAgreement 7148 7150 := RowAgreement.join atomic_7149_agreement atomic_7150_agreement
private def within_055_105 : CountSegment 1 10 13120 7150 7152 3065 3066 := CountSegment.join atomic_7151 atomic_7152
private theorem within_055_105_agreement : RowAgreement 7150 7152 := RowAgreement.join atomic_7151_agreement atomic_7152_agreement
private def within_055_106 : CountSegment 1 10 13120 7148 7152 3064 3066 := CountSegment.join within_055_104 within_055_105
private theorem within_055_106_agreement : RowAgreement 7148 7152 := RowAgreement.join within_055_104_agreement within_055_105_agreement
private def within_055_107 : CountSegment 1 10 13120 7144 7152 3062 3066 := CountSegment.join within_055_103 within_055_106
private theorem within_055_107_agreement : RowAgreement 7144 7152 := RowAgreement.join within_055_103_agreement within_055_106_agreement
private def within_055_108 : CountSegment 1 10 13120 7136 7152 3058 3066 := CountSegment.join within_055_100 within_055_107
private theorem within_055_108_agreement : RowAgreement 7136 7152 := RowAgreement.join within_055_100_agreement within_055_107_agreement
private def within_055_109 : CountSegment 1 10 13120 7152 7154 3066 3068 := CountSegment.join atomic_7153 atomic_7154
private theorem within_055_109_agreement : RowAgreement 7152 7154 := RowAgreement.join atomic_7153_agreement atomic_7154_agreement
private def within_055_110 : CountSegment 1 10 13120 7154 7156 3068 3068 := CountSegment.join atomic_7155 atomic_7156
private theorem within_055_110_agreement : RowAgreement 7154 7156 := RowAgreement.join atomic_7155_agreement atomic_7156_agreement
private def within_055_111 : CountSegment 1 10 13120 7152 7156 3066 3068 := CountSegment.join within_055_109 within_055_110
private theorem within_055_111_agreement : RowAgreement 7152 7156 := RowAgreement.join within_055_109_agreement within_055_110_agreement
private def within_055_112 : CountSegment 1 10 13120 7156 7158 3068 3069 := CountSegment.join atomic_7157 atomic_7158
private theorem within_055_112_agreement : RowAgreement 7156 7158 := RowAgreement.join atomic_7157_agreement atomic_7158_agreement
private def within_055_113 : CountSegment 1 10 13120 7158 7160 3069 3069 := CountSegment.join atomic_7159 atomic_7160
private theorem within_055_113_agreement : RowAgreement 7158 7160 := RowAgreement.join atomic_7159_agreement atomic_7160_agreement
private def within_055_114 : CountSegment 1 10 13120 7156 7160 3068 3069 := CountSegment.join within_055_112 within_055_113
private theorem within_055_114_agreement : RowAgreement 7156 7160 := RowAgreement.join within_055_112_agreement within_055_113_agreement
private def within_055_115 : CountSegment 1 10 13120 7152 7160 3066 3069 := CountSegment.join within_055_111 within_055_114
private theorem within_055_115_agreement : RowAgreement 7152 7160 := RowAgreement.join within_055_111_agreement within_055_114_agreement
private def within_055_116 : CountSegment 1 10 13120 7160 7162 3069 3069 := CountSegment.join atomic_7161 atomic_7162
private theorem within_055_116_agreement : RowAgreement 7160 7162 := RowAgreement.join atomic_7161_agreement atomic_7162_agreement
private def within_055_117 : CountSegment 1 10 13120 7162 7164 3069 3070 := CountSegment.join atomic_7163 atomic_7164
private theorem within_055_117_agreement : RowAgreement 7162 7164 := RowAgreement.join atomic_7163_agreement atomic_7164_agreement
private def within_055_118 : CountSegment 1 10 13120 7160 7164 3069 3070 := CountSegment.join within_055_116 within_055_117
private theorem within_055_118_agreement : RowAgreement 7160 7164 := RowAgreement.join within_055_116_agreement within_055_117_agreement
private def within_055_119 : CountSegment 1 10 13120 7164 7166 3070 3072 := CountSegment.join atomic_7165 atomic_7166
private theorem within_055_119_agreement : RowAgreement 7164 7166 := RowAgreement.join atomic_7165_agreement atomic_7166_agreement
private def within_055_120 : CountSegment 1 10 13120 7166 7168 3072 3072 := CountSegment.join atomic_7167 atomic_7168
private theorem within_055_120_agreement : RowAgreement 7166 7168 := RowAgreement.join atomic_7167_agreement atomic_7168_agreement
private def within_055_121 : CountSegment 1 10 13120 7164 7168 3070 3072 := CountSegment.join within_055_119 within_055_120
private theorem within_055_121_agreement : RowAgreement 7164 7168 := RowAgreement.join within_055_119_agreement within_055_120_agreement
private def within_055_122 : CountSegment 1 10 13120 7160 7168 3069 3072 := CountSegment.join within_055_118 within_055_121
private theorem within_055_122_agreement : RowAgreement 7160 7168 := RowAgreement.join within_055_118_agreement within_055_121_agreement
private def within_055_123 : CountSegment 1 10 13120 7152 7168 3066 3072 := CountSegment.join within_055_115 within_055_122
private theorem within_055_123_agreement : RowAgreement 7152 7168 := RowAgreement.join within_055_115_agreement within_055_122_agreement
private def within_055_124 : CountSegment 1 10 13120 7136 7168 3058 3072 := CountSegment.join within_055_108 within_055_123
private theorem within_055_124_agreement : RowAgreement 7136 7168 := RowAgreement.join within_055_108_agreement within_055_123_agreement
private def within_055_125 : CountSegment 1 10 13120 7104 7168 3047 3072 := CountSegment.join within_055_093 within_055_124
private theorem within_055_125_agreement : RowAgreement 7104 7168 := RowAgreement.join within_055_093_agreement within_055_124_agreement
private def within_055_126 : CountSegment 1 10 13120 7040 7168 3015 3072 := CountSegment.join within_055_062 within_055_125
private theorem within_055_126_agreement : RowAgreement 7040 7168 := RowAgreement.join within_055_062_agreement within_055_125_agreement
def segment_055 : CountSegment 1 10 13120 7040 7168 3015 3072 := within_055_126
theorem agreement_055 : RowAgreement 7040 7168 := within_055_126_agreement
end ProjectEulerSolutions.P58Numerical
