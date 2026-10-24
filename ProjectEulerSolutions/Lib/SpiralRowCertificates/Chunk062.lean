import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_7937 : RowCertificate 7937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7937 1) := by change ¬ Nat.Prime 251999751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83999917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7937 2) := by change ¬ Nat.Prime 251983877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14822581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7937 3) := by change ¬ Nat.Prime 251968003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35995429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7937 : row 7937 = 0 := (row_7937.checked).trans rfl
theorem model_count_7937 : modelRow 7937 = 0 := by decide
theorem modelrow_agrees_7937 : modelRow 7937 = row 7937 := model_count_7937.trans row_count_7937.symm
def row_7938 : RowCertificate 7938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7938 1) := by change ¬ Nat.Prime 252063253; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19389481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7938 2) := by change ¬ Nat.Prime 252047377; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6147497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7938 3) := by change Nat.Prime 252031501; exact ProjectEulerSolutions.P58PrimeNodes.p_252031501.prime
    exact decide_eq_true prime
theorem row_count_7938 : row 7938 = 1 := (row_7938.checked).trans rfl
theorem model_count_7938 : modelRow 7938 = 1 := by decide
theorem modelrow_agrees_7938 : modelRow 7938 = row 7938 := model_count_7938.trans row_count_7938.symm
def row_7939 : RowCertificate 7939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7939 1) := by change ¬ Nat.Prime 252126763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36018109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7939 2) := by change ¬ Nat.Prime 252110885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50422177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7939 3) := by change ¬ Nat.Prime 252095007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84031669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7939 : row 7939 = 0 := (row_7939.checked).trans rfl
theorem model_count_7939 : modelRow 7939 = 0 := by decide
theorem modelrow_agrees_7939 : modelRow 7939 = row 7939 := model_count_7939.trans row_count_7939.symm
def row_7940 : RowCertificate 7940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7940 1) := by change ¬ Nat.Prime 252190281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84063427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7940 2) := by change ¬ Nat.Prime 252174401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8695669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7940 3) := by change ¬ Nat.Prime 252158521; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1393141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7940 : row 7940 = 0 := (row_7940.checked).trans rfl
theorem model_count_7940 : modelRow 7940 = 0 := by decide
theorem modelrow_agrees_7940 : modelRow 7940 = row 7940 := model_count_7940.trans row_count_7940.symm
def row_7941 : RowCertificate 7941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7941 1) := by change ¬ Nat.Prime 252253807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19404139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7941 2) := by change ¬ Nat.Prime 252237925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50447585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7941 3) := by change Nat.Prime 252222043; exact ProjectEulerSolutions.P58PrimeNodes.p_252222043.prime
    exact decide_eq_true prime
theorem row_count_7941 : row 7941 = 1 := (row_7941.checked).trans rfl
theorem model_count_7941 : modelRow 7941 = 1 := by decide
theorem modelrow_agrees_7941 : modelRow 7941 = row 7941 := model_count_7941.trans row_count_7941.symm
def row_7942 : RowCertificate 7942 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7942 1) := by change Nat.Prime 252317341; exact ProjectEulerSolutions.P58PrimeNodes.p_252317341.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7942 2) := by change ¬ Nat.Prime 252301457; exact PrattCertificates.composite_of_product (factor := 977) (cofactor := 258241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7942 3) := by change ¬ Nat.Prime 252285573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84095191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7942 : row 7942 = 1 := (row_7942.checked).trans rfl
theorem model_count_7942 : modelRow 7942 = 1 := by decide
theorem modelrow_agrees_7942 : modelRow 7942 = row 7942 := model_count_7942.trans row_count_7942.symm
def row_7943 : RowCertificate 7943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7943 1) := by change ¬ Nat.Prime 252380883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84126961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7943 2) := by change ¬ Nat.Prime 252364997; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2601701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7943 3) := by change ¬ Nat.Prime 252349111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36049873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7943 : row 7943 = 0 := (row_7943.checked).trans rfl
theorem model_count_7943 : modelRow 7943 = 0 := by decide
theorem modelrow_agrees_7943 : modelRow 7943 = row 7943 := model_count_7943.trans row_count_7943.symm
def row_7944 : RowCertificate 7944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7944 1) := by change ¬ Nat.Prime 252444433; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1816147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7944 2) := by change ¬ Nat.Prime 252428545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50485709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7944 3) := by change ¬ Nat.Prime 252412657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36058951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7944 : row 7944 = 0 := (row_7944.checked).trans rfl
theorem model_count_7944 : modelRow 7944 = 0 := by decide
theorem modelrow_agrees_7944 : modelRow 7944 = row 7944 := model_count_7944.trans row_count_7944.symm
def row_7945 : RowCertificate 7945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7945 1) := by change ¬ Nat.Prime 252507991; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1047751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7945 2) := by change Nat.Prime 252492101; exact ProjectEulerSolutions.P58PrimeNodes.p_252492101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7945 3) := by change ¬ Nat.Prime 252476211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84158737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7945 : row 7945 = 1 := (row_7945.checked).trans rfl
theorem model_count_7945 : modelRow 7945 = 1 := by decide
theorem modelrow_agrees_7945 : modelRow 7945 = row 7945 := model_count_7945.trans row_count_7945.symm
def row_7946 : RowCertificate 7946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7946 1) := by change ¬ Nat.Prime 252571557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84190519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7946 2) := by change ¬ Nat.Prime 252555665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50511133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7946 3) := by change ¬ Nat.Prime 252539773; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13291567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7946 : row 7946 = 0 := (row_7946.checked).trans rfl
theorem model_count_7946 : modelRow 7946 = 0 := by decide
theorem modelrow_agrees_7946 : modelRow 7946 = row 7946 := model_count_7946.trans row_count_7946.symm
def row_7947 : RowCertificate 7947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7947 1) := by change ¬ Nat.Prime 252635131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36090733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7947 2) := by change ¬ Nat.Prime 252619237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19432249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7947 3) := by change Nat.Prime 252603343; exact ProjectEulerSolutions.P58PrimeNodes.p_252603343.prime
    exact decide_eq_true prime
theorem row_count_7947 : row 7947 = 1 := (row_7947.checked).trans rfl
theorem model_count_7947 : modelRow 7947 = 1 := by decide
theorem modelrow_agrees_7947 : modelRow 7947 = row 7947 := model_count_7947.trans row_count_7947.symm
def row_7948 : RowCertificate 7948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7948 1) := by change ¬ Nat.Prime 252698713; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 375481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7948 2) := by change Nat.Prime 252682817; exact ProjectEulerSolutions.P58PrimeNodes.p_252682817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7948 3) := by change ¬ Nat.Prime 252666921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84222307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7948 : row 7948 = 1 := (row_7948.checked).trans rfl
theorem model_count_7948 : modelRow 7948 = 1 := by decide
theorem modelrow_agrees_7948 : modelRow 7948 = row 7948 := model_count_7948.trans row_count_7948.symm
def row_7949 : RowCertificate 7949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7949 1) := by change ¬ Nat.Prime 252762303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84254101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7949 2) := by change ¬ Nat.Prime 252746405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50549281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7949 3) := by change ¬ Nat.Prime 252730507; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8152597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7949 : row 7949 = 0 := (row_7949.checked).trans rfl
theorem model_count_7949 : modelRow 7949 = 0 := by decide
theorem modelrow_agrees_7949 : modelRow 7949 = row 7949 := model_count_7949.trans row_count_7949.symm
def row_7950 : RowCertificate 7950 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7950 1) := by change Nat.Prime 252825901; exact ProjectEulerSolutions.P58PrimeNodes.p_252825901.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7950 2) := by change ¬ Nat.Prime 252810001; exact PrattCertificates.composite_of_product (factor := 941) (cofactor := 268661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7950 3) := by change ¬ Nat.Prime 252794101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36113443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7950 : row 7950 = 1 := (row_7950.checked).trans rfl
theorem model_count_7950 : modelRow 7950 = 1 := by decide
theorem modelrow_agrees_7950 : modelRow 7950 = row 7950 := model_count_7950.trans row_count_7950.symm
def row_7951 : RowCertificate 7951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7951 1) := by change ¬ Nat.Prime 252889507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19453039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7951 2) := by change ¬ Nat.Prime 252873605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50574721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7951 3) := by change ¬ Nat.Prime 252857703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84285901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7951 : row 7951 = 0 := (row_7951.checked).trans rfl
theorem model_count_7951 : modelRow 7951 = 0 := by decide
theorem modelrow_agrees_7951 : modelRow 7951 = row 7951 := model_count_7951.trans row_count_7951.symm
def row_7952 : RowCertificate 7952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7952 1) := by change ¬ Nat.Prime 252953121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84317707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7952 2) := by change ¬ Nat.Prime 252937217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19456709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7952 3) := by change ¬ Nat.Prime 252921313; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5881891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7952 : row 7952 = 0 := (row_7952.checked).trans rfl
theorem model_count_7952 : modelRow 7952 = 0 := by decide
theorem modelrow_agrees_7952 : modelRow 7952 = row 7952 := model_count_7952.trans row_count_7952.symm
def row_7953 : RowCertificate 7953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7953 1) := by change ¬ Nat.Prime 253016743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36145249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7953 2) := by change ¬ Nat.Prime 253000837; exact PrattCertificates.composite_of_product (factor := 9377) (cofactor := 26981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7953 3) := by change ¬ Nat.Prime 252984931; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3465547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7953 : row 7953 = 0 := (row_7953.checked).trans rfl
theorem model_count_7953 : modelRow 7953 = 0 := by decide
theorem modelrow_agrees_7953 : modelRow 7953 = row 7953 := model_count_7953.trans row_count_7953.symm
def row_7954 : RowCertificate 7954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7954 1) := by change ¬ Nat.Prime 253080373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36154339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7954 2) := by change ¬ Nat.Prime 253064465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50612893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7954 3) := by change ¬ Nat.Prime 253048557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84349519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7954 : row 7954 = 0 := (row_7954.checked).trans rfl
theorem model_count_7954 : modelRow 7954 = 0 := by decide
theorem modelrow_agrees_7954 : modelRow 7954 = row 7954 := model_count_7954.trans row_count_7954.symm
def row_7955 : RowCertificate 7955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7955 1) := by change ¬ Nat.Prime 253144011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84381337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7955 2) := by change ¬ Nat.Prime 253128101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4149641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7955 3) := by change ¬ Nat.Prime 253112191; exact PrattCertificates.composite_of_product (factor := 7753) (cofactor := 32647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7955 : row 7955 = 0 := (row_7955.checked).trans rfl
theorem model_count_7955 : modelRow 7955 = 0 := by decide
theorem modelrow_agrees_7955 : modelRow 7955 = row 7955 := model_count_7955.trans row_count_7955.symm
def row_7956 : RowCertificate 7956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7956 1) := by change ¬ Nat.Prime 253207657; exact PrattCertificates.composite_of_product (factor := 3361) (cofactor := 75337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7956 2) := by change ¬ Nat.Prime 253191745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50638349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7956 3) := by change Nat.Prime 253175833; exact ProjectEulerSolutions.P58PrimeNodes.p_253175833.prime
    exact decide_eq_true prime
theorem row_count_7956 : row 7956 = 1 := (row_7956.checked).trans rfl
theorem model_count_7956 : modelRow 7956 = 1 := by decide
theorem modelrow_agrees_7956 : modelRow 7956 = row 7956 := model_count_7956.trans row_count_7956.symm
def row_7957 : RowCertificate 7957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7957 1) := by change ¬ Nat.Prime 253271311; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13330069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7957 2) := by change Nat.Prime 253255397; exact ProjectEulerSolutions.P58PrimeNodes.p_253255397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7957 3) := by change ¬ Nat.Prime 253239483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84413161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7957 : row 7957 = 1 := (row_7957.checked).trans rfl
theorem model_count_7957 : modelRow 7957 = 1 := by decide
theorem modelrow_agrees_7957 : modelRow 7957 = row 7957 := model_count_7957.trans row_count_7957.symm
def row_7958 : RowCertificate 7958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7958 1) := by change ¬ Nat.Prime 253334973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84444991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7958 2) := by change ¬ Nat.Prime 253319057; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14901121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7958 3) := by change ¬ Nat.Prime 253303141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36186163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7958 : row 7958 = 0 := (row_7958.checked).trans rfl
theorem model_count_7958 : modelRow 7958 = 0 := by decide
theorem modelrow_agrees_7958 : modelRow 7958 = row 7958 := model_count_7958.trans row_count_7958.symm
def row_7959 : RowCertificate 7959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7959 1) := by change ¬ Nat.Prime 253398643; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2460181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7959 2) := by change ¬ Nat.Prime 253382725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50676545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7959 3) := by change ¬ Nat.Prime 253366807; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 106591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7959 : row 7959 = 0 := (row_7959.checked).trans rfl
theorem model_count_7959 : modelRow 7959 = 0 := by decide
theorem modelrow_agrees_7959 : modelRow 7959 = row 7959 := model_count_7959.trans row_count_7959.symm
def row_7960 : RowCertificate 7960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7960 1) := by change ¬ Nat.Prime 253462321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36208903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7960 2) := by change ¬ Nat.Prime 253446401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19495877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7960 3) := by change ¬ Nat.Prime 253430481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84476827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7960 : row 7960 = 0 := (row_7960.checked).trans rfl
theorem model_count_7960 : modelRow 7960 = 0 := by decide
theorem modelrow_agrees_7960 : modelRow 7960 = row 7960 := model_count_7960.trans row_count_7960.symm
def row_7961 : RowCertificate 7961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7961 1) := by change ¬ Nat.Prime 253526007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84508669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7961 2) := by change ¬ Nat.Prime 253510085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50702017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7961 3) := by change ¬ Nat.Prime 253494163; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19499551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7961 : row 7961 = 0 := (row_7961.checked).trans rfl
theorem model_count_7961 : modelRow 7961 = 0 := by decide
theorem modelrow_agrees_7961 : modelRow 7961 = row 7961 := model_count_7961.trans row_count_7961.symm
def row_7962 : RowCertificate 7962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7962 1) := by change ¬ Nat.Prime 253589701; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 334993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7962 2) := by change Nat.Prime 253573777; exact ProjectEulerSolutions.P58PrimeNodes.p_253573777.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7962 3) := by change Nat.Prime 253557853; exact ProjectEulerSolutions.P58PrimeNodes.p_253557853.prime
    exact decide_eq_true prime
theorem row_count_7962 : row 7962 = 2 := (row_7962.checked).trans rfl
theorem model_count_7962 : modelRow 7962 = 2 := by decide
theorem modelrow_agrees_7962 : modelRow 7962 = row 7962 := model_count_7962.trans row_count_7962.symm
def row_7963 : RowCertificate 7963 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7963 1) := by change Nat.Prime 253653403; exact ProjectEulerSolutions.P58PrimeNodes.p_253653403.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7963 2) := by change ¬ Nat.Prime 253637477; exact PrattCertificates.composite_of_product (factor := 10889) (cofactor := 23293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7963 3) := by change ¬ Nat.Prime 253621551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84540517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7963 : row 7963 = 1 := (row_7963.checked).trans rfl
theorem model_count_7963 : modelRow 7963 = 1 := by decide
theorem modelrow_agrees_7963 : modelRow 7963 = row 7963 := model_count_7963.trans row_count_7963.symm
def row_7964 : RowCertificate 7964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7964 1) := by change ¬ Nat.Prime 253717113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84572371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7964 2) := by change ¬ Nat.Prime 253701185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50740237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7964 3) := by change ¬ Nat.Prime 253685257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36240751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7964 : row 7964 = 0 := (row_7964.checked).trans rfl
theorem model_count_7964 : modelRow 7964 = 0 := by decide
theorem modelrow_agrees_7964 : modelRow 7964 = row 7964 := model_count_7964.trans row_count_7964.symm
def row_7965 : RowCertificate 7965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7965 1) := by change ¬ Nat.Prime 253780831; exact PrattCertificates.composite_of_product (factor := 3037) (cofactor := 83563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7965 2) := by change ¬ Nat.Prime 253764901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19520377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7965 3) := by change ¬ Nat.Prime 253748971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36249853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7965 : row 7965 = 0 := (row_7965.checked).trans rfl
theorem model_count_7965 : modelRow 7965 = 0 := by decide
theorem modelrow_agrees_7965 : modelRow 7965 = row 7965 := model_count_7965.trans row_count_7965.symm
def row_7966 : RowCertificate 7966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7966 1) := by change ¬ Nat.Prime 253844557; exact PrattCertificates.composite_of_product (factor := 8011) (cofactor := 31687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7966 2) := by change ¬ Nat.Prime 253828625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50765725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7966 3) := by change ¬ Nat.Prime 253812693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84604231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7966 : row 7966 = 0 := (row_7966.checked).trans rfl
theorem model_count_7966 : modelRow 7966 = 0 := by decide
theorem modelrow_agrees_7966 : modelRow 7966 = row 7966 := model_count_7966.trans row_count_7966.symm
def row_7967 : RowCertificate 7967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7967 1) := by change ¬ Nat.Prime 253908291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84636097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7967 2) := by change ¬ Nat.Prime 253892357; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 428149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7967 3) := by change ¬ Nat.Prime 253876423; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13361917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7967 : row 7967 = 0 := (row_7967.checked).trans rfl
theorem model_count_7967 : modelRow 7967 = 0 := by decide
theorem modelrow_agrees_7967 : modelRow 7967 = row 7967 := model_count_7967.trans row_count_7967.symm
def row_7968 : RowCertificate 7968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7968 1) := by change ¬ Nat.Prime 253972033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36281719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7968 2) := by change Nat.Prime 253956097; exact ProjectEulerSolutions.P58PrimeNodes.p_253956097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7968 3) := by change ¬ Nat.Prime 253940161; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1108909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7968 : row 7968 = 1 := (row_7968.checked).trans rfl
theorem model_count_7968 : modelRow 7968 = 1 := by decide
theorem modelrow_agrees_7968 : modelRow 7968 = row 7968 := model_count_7968.trans row_count_7968.symm
def row_7969 : RowCertificate 7969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7969 1) := by change ¬ Nat.Prime 254035783; exact PrattCertificates.composite_of_product (factor := 5683) (cofactor := 44701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7969 2) := by change ¬ Nat.Prime 254019845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50803969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7969 3) := by change ¬ Nat.Prime 254003907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84667969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7969 : row 7969 = 0 := (row_7969.checked).trans rfl
theorem model_count_7969 : modelRow 7969 = 0 := by decide
theorem modelrow_agrees_7969 : modelRow 7969 = row 7969 := model_count_7969.trans row_count_7969.symm
def row_7970 : RowCertificate 7970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7970 1) := by change ¬ Nat.Prime 254099541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84699847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7970 2) := by change ¬ Nat.Prime 254083601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6197161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7970 3) := by change ¬ Nat.Prime 254067661; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8195731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7970 : row 7970 = 0 := (row_7970.checked).trans rfl
theorem model_count_7970 : modelRow 7970 = 0 := by decide
theorem modelrow_agrees_7970 : modelRow 7970 = row 7970 := model_count_7970.trans row_count_7970.symm
def row_7971 : RowCertificate 7971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7971 1) := by change ¬ Nat.Prime 254163307; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 1828513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7971 2) := by change ¬ Nat.Prime 254147365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50829473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7971 3) := by change ¬ Nat.Prime 254131423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36304489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7971 : row 7971 = 0 := (row_7971.checked).trans rfl
theorem model_count_7971 : modelRow 7971 = 0 := by decide
theorem modelrow_agrees_7971 : modelRow 7971 = row 7971 := model_count_7971.trans row_count_7971.symm
def row_7972 : RowCertificate 7972 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7972 1) := by change Nat.Prime 254227081; exact ProjectEulerSolutions.P58PrimeNodes.p_254227081.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7972 2) := by change Nat.Prime 254211137; exact ProjectEulerSolutions.P58PrimeNodes.p_254211137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7972 3) := by change ¬ Nat.Prime 254195193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84731731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7972 : row 7972 = 2 := (row_7972.checked).trans rfl
theorem model_count_7972 : modelRow 7972 = 2 := by decide
theorem modelrow_agrees_7972 : modelRow 7972 = row 7972 := model_count_7972.trans row_count_7972.symm
def row_7973 : RowCertificate 7973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7973 1) := by change ¬ Nat.Prime 254290863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84763621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7973 2) := by change ¬ Nat.Prime 254274917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19559609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7973 3) := by change ¬ Nat.Prime 254258971; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 241921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7973 : row 7973 = 0 := (row_7973.checked).trans rfl
theorem model_count_7973 : modelRow 7973 = 0 := by decide
theorem modelrow_agrees_7973 : modelRow 7973 = row 7973 := model_count_7973.trans row_count_7973.symm
def row_7974 : RowCertificate 7974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7974 1) := by change ¬ Nat.Prime 254354653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36336379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7974 2) := by change ¬ Nat.Prime 254338705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50867741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7974 3) := by change ¬ Nat.Prime 254322757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19563289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7974 : row 7974 = 0 := (row_7974.checked).trans rfl
theorem model_count_7974 : modelRow 7974 = 0 := by decide
theorem modelrow_agrees_7974 : modelRow 7974 = row 7974 := model_count_7974.trans row_count_7974.symm
def row_7975 : RowCertificate 7975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7975 1) := by change ¬ Nat.Prime 254418451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36345493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7975 2) := by change ¬ Nat.Prime 254402501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14964853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7975 3) := by change ¬ Nat.Prime 254386551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84795517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7975 : row 7975 = 0 := (row_7975.checked).trans rfl
theorem model_count_7975 : modelRow 7975 = 0 := by decide
theorem modelrow_agrees_7975 : modelRow 7975 = row 7975 := model_count_7975.trans row_count_7975.symm
def row_7976 : RowCertificate 7976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7976 1) := by change ¬ Nat.Prime 254482257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84827419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7976 2) := by change ¬ Nat.Prime 254466305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50893261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7976 3) := by change ¬ Nat.Prime 254450353; exact PrattCertificates.composite_of_product (factor := 6199) (cofactor := 41047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7976 : row 7976 = 0 := (row_7976.checked).trans rfl
theorem model_count_7976 : modelRow 7976 = 0 := by decide
theorem modelrow_agrees_7976 : modelRow 7976 = row 7976 := model_count_7976.trans row_count_7976.symm
def row_7977 : RowCertificate 7977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7977 1) := by change ¬ Nat.Prime 254546071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19580467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7977 2) := by change Nat.Prime 254530117; exact ProjectEulerSolutions.P58PrimeNodes.p_254530117.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7977 3) := by change Nat.Prime 254514163; exact ProjectEulerSolutions.P58PrimeNodes.p_254514163.prime
    exact decide_eq_true prime
theorem row_count_7977 : row 7977 = 2 := (row_7977.checked).trans rfl
theorem model_count_7977 : modelRow 7977 = 2 := by decide
theorem modelrow_agrees_7977 : modelRow 7977 = row 7977 := model_count_7977.trans row_count_7977.symm
def row_7978 : RowCertificate 7978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7978 1) := by change ¬ Nat.Prime 254609893; exact PrattCertificates.composite_of_product (factor := 1531) (cofactor := 166303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7978 2) := by change ¬ Nat.Prime 254593937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19584149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7978 3) := by change ¬ Nat.Prime 254577981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84859327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7978 : row 7978 = 0 := (row_7978.checked).trans rfl
theorem model_count_7978 : modelRow 7978 = 0 := by decide
theorem modelrow_agrees_7978 : modelRow 7978 = row 7978 := model_count_7978.trans row_count_7978.symm
def row_7979 : RowCertificate 7979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7979 1) := by change ¬ Nat.Prime 254673723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84891241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7979 2) := by change ¬ Nat.Prime 254657765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50931553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7979 3) := by change ¬ Nat.Prime 254641807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36377401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7979 : row 7979 = 0 := (row_7979.checked).trans rfl
theorem model_count_7979 : modelRow 7979 = 0 := by decide
theorem modelrow_agrees_7979 : modelRow 7979 = row 7979 := model_count_7979.trans row_count_7979.symm
def row_7980 : RowCertificate 7980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7980 1) := by change ¬ Nat.Prime 254737561; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19595197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7980 2) := by change ¬ Nat.Prime 254721601; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3489337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7980 3) := by change ¬ Nat.Prime 254705641; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8216311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7980 : row 7980 = 0 := (row_7980.checked).trans rfl
theorem model_count_7980 : modelRow 7980 = 0 := by decide
theorem modelrow_agrees_7980 : modelRow 7980 = row 7980 := model_count_7980.trans row_count_7980.symm
def row_7981 : RowCertificate 7981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7981 1) := by change ¬ Nat.Prime 254801407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36400201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7981 2) := by change ¬ Nat.Prime 254785445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50957089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7981 3) := by change ¬ Nat.Prime 254769483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84923161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7981 : row 7981 = 0 := (row_7981.checked).trans rfl
theorem model_count_7981 : modelRow 7981 = 0 := by decide
theorem modelrow_agrees_7981 : modelRow 7981 = row 7981 := model_count_7981.trans row_count_7981.symm
def row_7982 : RowCertificate 7982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7982 1) := by change ¬ Nat.Prime 254865261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84955087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7982 2) := by change ¬ Nat.Prime 254849297; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 906937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7982 3) := by change Nat.Prime 254833333; exact ProjectEulerSolutions.P58PrimeNodes.p_254833333.prime
    exact decide_eq_true prime
theorem row_count_7982 : row 7982 = 1 := (row_7982.checked).trans rfl
theorem model_count_7982 : modelRow 7982 = 1 := by decide
theorem modelrow_agrees_7982 : modelRow 7982 = row 7982 := model_count_7982.trans row_count_7982.symm
def row_7983 : RowCertificate 7983 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7983 1) := by change Nat.Prime 254929123; exact ProjectEulerSolutions.P58PrimeNodes.p_254929123.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7983 2) := by change ¬ Nat.Prime 254913157; exact PrattCertificates.composite_of_product (factor := 14717) (cofactor := 17321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7983 3) := by change ¬ Nat.Prime 254897191; exact PrattCertificates.composite_of_product (factor := 2269) (cofactor := 112339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7983 : row 7983 = 1 := (row_7983.checked).trans rfl
theorem model_count_7983 : modelRow 7983 = 1 := by decide
theorem modelrow_agrees_7983 : modelRow 7983 = row 7983 := model_count_7983.trans row_count_7983.symm
def row_7984 : RowCertificate 7984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7984 1) := by change ¬ Nat.Prime 254992993; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4180213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7984 2) := by change ¬ Nat.Prime 254977025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50995405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7984 3) := by change ¬ Nat.Prime 254961057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 84987019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7984 : row 7984 = 0 := (row_7984.checked).trans rfl
theorem model_count_7984 : modelRow 7984 = 0 := by decide
theorem modelrow_agrees_7984 : modelRow 7984 = row 7984 := model_count_7984.trans row_count_7984.symm
def row_7985 : RowCertificate 7985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7985 1) := by change ¬ Nat.Prime 255056871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85018957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7985 2) := by change ¬ Nat.Prime 255040901; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1058261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7985 3) := by change ¬ Nat.Prime 255024931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36432133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7985 : row 7985 = 0 := (row_7985.checked).trans rfl
theorem model_count_7985 : modelRow 7985 = 0 := by decide
theorem modelrow_agrees_7985 : modelRow 7985 = row 7985 := model_count_7985.trans row_count_7985.symm
def row_7986 : RowCertificate 7986 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7986 1) := by change Nat.Prime 255120757; exact ProjectEulerSolutions.P58PrimeNodes.p_255120757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7986 2) := by change ¬ Nat.Prime 255104785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51020957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7986 3) := by change ¬ Nat.Prime 255088813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36441259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7986 : row 7986 = 1 := (row_7986.checked).trans rfl
theorem model_count_7986 : modelRow 7986 = 1 := by decide
theorem modelrow_agrees_7986 : modelRow 7986 = row 7986 := model_count_7986.trans row_count_7986.symm
def row_7987 : RowCertificate 7987 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7987 1) := by change Nat.Prime 255184651; exact ProjectEulerSolutions.P58PrimeNodes.p_255184651.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7987 2) := by change ¬ Nat.Prime 255168677; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 226013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7987 3) := by change ¬ Nat.Prime 255152703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85050901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7987 : row 7987 = 1 := (row_7987.checked).trans rfl
theorem model_count_7987 : modelRow 7987 = 1 := by decide
theorem modelrow_agrees_7987 : modelRow 7987 = row 7987 := model_count_7987.trans row_count_7987.symm
def row_7988 : RowCertificate 7988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7988 1) := by change ¬ Nat.Prime 255248553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85082851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7988 2) := by change ¬ Nat.Prime 255232577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15013681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7988 3) := by change ¬ Nat.Prime 255216601; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3809203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7988 : row 7988 = 0 := (row_7988.checked).trans rfl
theorem model_count_7988 : modelRow 7988 = 0 := by decide
theorem modelrow_agrees_7988 : modelRow 7988 = row 7988 := model_count_7988.trans row_count_7988.symm
def row_7989 : RowCertificate 7989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7989 1) := by change ¬ Nat.Prime 255312463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36473209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7989 2) := by change ¬ Nat.Prime 255296485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51059297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7989 3) := by change ¬ Nat.Prime 255280507; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2342023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7989 : row 7989 = 0 := (row_7989.checked).trans rfl
theorem model_count_7989 : modelRow 7989 = 0 := by decide
theorem modelrow_agrees_7989 : modelRow 7989 = row 7989 := model_count_7989.trans row_count_7989.symm
def row_7990 : RowCertificate 7990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7990 1) := by change ¬ Nat.Prime 255376381; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19644337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7990 2) := by change Nat.Prime 255360401; exact ProjectEulerSolutions.P58PrimeNodes.p_255360401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7990 3) := by change ¬ Nat.Prime 255344421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85114807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7990 : row 7990 = 1 := (row_7990.checked).trans rfl
theorem model_count_7990 : modelRow 7990 = 1 := by decide
theorem modelrow_agrees_7990 : modelRow 7990 = row 7990 := model_count_7990.trans row_count_7990.symm
def row_7991 : RowCertificate 7991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7991 1) := by change ¬ Nat.Prime 255440307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85146769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7991 2) := by change ¬ Nat.Prime 255424325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51084865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7991 3) := by change ¬ Nat.Prime 255408343; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3233017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7991 : row 7991 = 0 := (row_7991.checked).trans rfl
theorem model_count_7991 : modelRow 7991 = 0 := by decide
theorem modelrow_agrees_7991 : modelRow 7991 = row 7991 := model_count_7991.trans row_count_7991.symm
def row_7992 : RowCertificate 7992 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7992 1) := by change Nat.Prime 255504241; exact ProjectEulerSolutions.P58PrimeNodes.p_255504241.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7992 2) := by change ¬ Nat.Prime 255488257; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15028721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7992 3) := by change ¬ Nat.Prime 255472273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36496039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7992 : row 7992 = 1 := (row_7992.checked).trans rfl
theorem model_count_7992 : modelRow 7992 = 1 := by decide
theorem modelrow_agrees_7992 : modelRow 7992 = row 7992 := model_count_7992.trans row_count_7992.symm
def row_7993 : RowCertificate 7993 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7993 1) := by change ¬ Nat.Prime 255568183; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19659091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7993 2) := by change ¬ Nat.Prime 255552197; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 2871373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7993 3) := by change ¬ Nat.Prime 255536211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85178737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7993 : row 7993 = 0 := (row_7993.checked).trans rfl
theorem model_count_7993 : modelRow 7993 = 0 := by decide
theorem modelrow_agrees_7993 : modelRow 7993 = row 7993 := model_count_7993.trans row_count_7993.symm
def row_7994 : RowCertificate 7994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7994 1) := by change ¬ Nat.Prime 255632133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85210711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7994 2) := by change ¬ Nat.Prime 255616145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51123229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7994 3) := by change Nat.Prime 255600157; exact ProjectEulerSolutions.P58PrimeNodes.p_255600157.prime
    exact decide_eq_true prime
theorem row_count_7994 : row 7994 = 1 := (row_7994.checked).trans rfl
theorem model_count_7994 : modelRow 7994 = 1 := by decide
theorem modelrow_agrees_7994 : modelRow 7994 = row 7994 := model_count_7994.trans row_count_7994.symm
def row_7995 : RowCertificate 7995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7995 1) := by change ¬ Nat.Prime 255696091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36528013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7995 2) := by change ¬ Nat.Prime 255680101; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6910273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7995 3) := by change ¬ Nat.Prime 255664111; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5945677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7995 : row 7995 = 0 := (row_7995.checked).trans rfl
theorem model_count_7995 : modelRow 7995 = 0 := by decide
theorem modelrow_agrees_7995 : modelRow 7995 = row 7995 := model_count_7995.trans row_count_7995.symm
def row_7996 : RowCertificate 7996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7996 1) := by change ¬ Nat.Prime 255760057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36537151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7996 2) := by change ¬ Nat.Prime 255744065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51148813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7996 3) := by change ¬ Nat.Prime 255728073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85242691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7996 : row 7996 = 0 := (row_7996.checked).trans rfl
theorem model_count_7996 : modelRow 7996 = 0 := by decide
theorem modelrow_agrees_7996 : modelRow 7996 = row 7996 := model_count_7996.trans row_count_7996.symm
def row_7997 : RowCertificate 7997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7997 1) := by change ¬ Nat.Prime 255824031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85274677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7997 2) := by change ¬ Nat.Prime 255808037; exact PrattCertificates.composite_of_product (factor := 3257) (cofactor := 78541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7997 3) := by change ¬ Nat.Prime 255792043; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19676311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7997 : row 7997 = 0 := (row_7997.checked).trans rfl
theorem model_count_7997 : modelRow 7997 = 0 := by decide
theorem modelrow_agrees_7997 : modelRow 7997 = row 7997 := model_count_7997.trans row_count_7997.symm
def row_7998 : RowCertificate 7998 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7998 1) := by change Nat.Prime 255888013; exact ProjectEulerSolutions.P58PrimeNodes.p_255888013.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7998 2) := by change ¬ Nat.Prime 255872017; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8823173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7998 3) := by change ¬ Nat.Prime 255856021; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4194361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7998 : row 7998 = 1 := (row_7998.checked).trans rfl
theorem model_count_7998 : modelRow 7998 = 1 := by decide
theorem modelrow_agrees_7998 : modelRow 7998 = row 7998 := model_count_7998.trans row_count_7998.symm
def row_7999 : RowCertificate 7999 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7999 1) := by change Nat.Prime 255952003; exact ProjectEulerSolutions.P58PrimeNodes.p_255952003.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7999 2) := by change ¬ Nat.Prime 255936005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51187201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7999 3) := by change ¬ Nat.Prime 255920007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85306669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7999 : row 7999 = 1 := (row_7999.checked).trans rfl
theorem model_count_7999 : modelRow 7999 = 1 := by decide
theorem modelrow_agrees_7999 : modelRow 7999 = row 7999 := model_count_7999.trans row_count_7999.symm
def row_8000 : RowCertificate 8000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8000 1) := by change ¬ Nat.Prime 256016001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85338667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8000 2) := by change Nat.Prime 256000001; exact ProjectEulerSolutions.P58PrimeNodes.p_256000001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8000 3) := by change ¬ Nat.Prime 255984001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36569143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8000 : row 8000 = 1 := (row_8000.checked).trans rfl
theorem model_count_8000 : modelRow 8000 = 1 := by decide
theorem modelrow_agrees_8000 : modelRow 8000 = row 8000 := model_count_8000.trans row_count_8000.symm
def row_8001 : RowCertificate 8001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8001 1) := by change ¬ Nat.Prime 256080007; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5955349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8001 2) := by change ¬ Nat.Prime 256064005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51212801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8001 3) := by change ¬ Nat.Prime 256048003; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8259613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8001 : row 8001 = 0 := (row_8001.checked).trans rfl
theorem model_count_8001 : modelRow 8001 = 0 := by decide
theorem modelrow_agrees_8001 : modelRow 8001 = row 8001 := model_count_8001.trans row_count_8001.symm
def row_8002 : RowCertificate 8002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8002 1) := by change ¬ Nat.Prime 256144021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36592003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8002 2) := by change Nat.Prime 256128017; exact ProjectEulerSolutions.P58PrimeNodes.p_256128017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8002 3) := by change ¬ Nat.Prime 256112013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85370671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8002 : row 8002 = 1 := (row_8002.checked).trans rfl
theorem model_count_8002 : modelRow 8002 = 1 := by decide
theorem modelrow_agrees_8002 : modelRow 8002 = row 8002 := model_count_8002.trans row_count_8002.symm
def row_8003 : RowCertificate 8003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8003 1) := by change ¬ Nat.Prime 256208043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85402681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8003 2) := by change ¬ Nat.Prime 256192037; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 924881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8003 3) := by change ¬ Nat.Prime 256176031; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13482949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8003 : row 8003 = 0 := (row_8003.checked).trans rfl
theorem model_count_8003 : modelRow 8003 = 0 := by decide
theorem modelrow_agrees_8003 : modelRow 8003 = row 8003 := model_count_8003.trans row_count_8003.symm
def row_8004 : RowCertificate 8004 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8004 1) := by change Nat.Prime 256272073; exact ProjectEulerSolutions.P58PrimeNodes.p_256272073.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8004 2) := by change ¬ Nat.Prime 256256065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51251213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8004 3) := by change Nat.Prime 256240057; exact ProjectEulerSolutions.P58PrimeNodes.p_256240057.prime
    exact decide_eq_true prime
theorem row_count_8004 : row 8004 = 2 := (row_8004.checked).trans rfl
theorem model_count_8004 : modelRow 8004 = 2 := by decide
theorem modelrow_agrees_8004 : modelRow 8004 = row 8004 := model_count_8004.trans row_count_8004.symm
def row_8005 : RowCertificate 8005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8005 1) := by change ¬ Nat.Prime 256336111; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6928003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8005 2) := by change ¬ Nat.Prime 256320101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15077653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8005 3) := by change ¬ Nat.Prime 256304091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85434697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8005 : row 8005 = 0 := (row_8005.checked).trans rfl
theorem model_count_8005 : modelRow 8005 = 0 := by decide
theorem modelrow_agrees_8005 : modelRow 8005 = row 8005 := model_count_8005.trans row_count_8005.symm
def row_8006 : RowCertificate 8006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8006 1) := by change ¬ Nat.Prime 256400157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85466719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8006 2) := by change ¬ Nat.Prime 256384145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51276829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8006 3) := by change ¬ Nat.Prime 256368133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36624019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8006 : row 8006 = 0 := (row_8006.checked).trans rfl
theorem model_count_8006 : modelRow 8006 = 0 := by decide
theorem modelrow_agrees_8006 : modelRow 8006 = row 8006 := model_count_8006.trans row_count_8006.symm
def row_8007 : RowCertificate 8007 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8007 1) := by change Nat.Prime 256464211; exact ProjectEulerSolutions.P58PrimeNodes.p_256464211.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8007 2) := by change ¬ Nat.Prime 256448197; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3512989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8007 3) := by change ¬ Nat.Prime 256432183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36633169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8007 : row 8007 = 1 := (row_8007.checked).trans rfl
theorem model_count_8007 : modelRow 8007 = 1 := by decide
theorem modelrow_agrees_8007 : modelRow 8007 = row 8007 := model_count_8007.trans row_count_8007.symm
def row_8008 : RowCertificate 8008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8008 1) := by change ¬ Nat.Prime 256528273; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1150351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8008 2) := by change ¬ Nat.Prime 256512257; exact PrattCertificates.composite_of_product (factor := 1913) (cofactor := 134089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8008 3) := by change ¬ Nat.Prime 256496241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85498747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8008 : row 8008 = 0 := (row_8008.checked).trans rfl
theorem model_count_8008 : modelRow 8008 = 0 := by decide
theorem modelrow_agrees_8008 : modelRow 8008 = row 8008 := model_count_8008.trans row_count_8008.symm
def row_8009 : RowCertificate 8009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8009 1) := by change ¬ Nat.Prime 256592343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85530781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8009 2) := by change ¬ Nat.Prime 256576325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51315265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8009 3) := by change ¬ Nat.Prime 256560307; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 946717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8009 : row 8009 = 0 := (row_8009.checked).trans rfl
theorem model_count_8009 : modelRow 8009 = 0 := by decide
theorem modelrow_agrees_8009 : modelRow 8009 = row 8009 := model_count_8009.trans row_count_8009.symm
def row_8010 : RowCertificate 8010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8010 1) := by change ¬ Nat.Prime 256656421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36665203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8010 2) := by change ¬ Nat.Prime 256640401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8849669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8010 3) := by change ¬ Nat.Prime 256624381; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19740337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8010 : row 8010 = 0 := (row_8010.checked).trans rfl
theorem model_count_8010 : modelRow 8010 = 0 := by decide
theorem modelrow_agrees_8010 : modelRow 8010 = row 8010 := model_count_8010.trans row_count_8010.symm
def row_8011 : RowCertificate 8011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8011 1) := by change ¬ Nat.Prime 256720507; exact PrattCertificates.composite_of_product (factor := 5023) (cofactor := 51109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8011 2) := by change ¬ Nat.Prime 256704485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51340897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8011 3) := by change ¬ Nat.Prime 256688463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85562821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8011 : row 8011 = 0 := (row_8011.checked).trans rfl
theorem model_count_8011 : modelRow 8011 = 0 := by decide
theorem modelrow_agrees_8011 : modelRow 8011 = row 8011 := model_count_8011.trans row_count_8011.symm
def row_8012 : RowCertificate 8012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8012 1) := by change ¬ Nat.Prime 256784601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85594867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8012 2) := by change ¬ Nat.Prime 256768577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19751429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8012 3) := by change Nat.Prime 256752553; exact ProjectEulerSolutions.P58PrimeNodes.p_256752553.prime
    exact decide_eq_true prime
theorem row_count_8012 : row 8012 = 1 := (row_8012.checked).trans rfl
theorem model_count_8012 : modelRow 8012 = 1 := by decide
theorem modelrow_agrees_8012 : modelRow 8012 = row 8012 := model_count_8012.trans row_count_8012.symm
def row_8013 : RowCertificate 8013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8013 1) := by change ¬ Nat.Prime 256848703; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1635979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8013 2) := by change Nat.Prime 256832677; exact ProjectEulerSolutions.P58PrimeNodes.p_256832677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8013 3) := by change ¬ Nat.Prime 256816651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36688093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8013 : row 8013 = 1 := (row_8013.checked).trans rfl
theorem model_count_8013 : modelRow 8013 = 1 := by decide
theorem modelrow_agrees_8013 : modelRow 8013 = row 8013 := model_count_8013.trans row_count_8013.symm
def row_8014 : RowCertificate 8014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8014 1) := by change ¬ Nat.Prime 256912813; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13521727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8014 2) := by change ¬ Nat.Prime 256896785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51379357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8014 3) := by change ¬ Nat.Prime 256880757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85626919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8014 : row 8014 = 0 := (row_8014.checked).trans rfl
theorem model_count_8014 : modelRow 8014 = 0 := by decide
theorem modelrow_agrees_8014 : modelRow 8014 = row 8014 := model_count_8014.trans row_count_8014.symm
def row_8015 : RowCertificate 8015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8015 1) := by change ¬ Nat.Prime 256976931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85658977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8015 2) := by change Nat.Prime 256960901; exact ProjectEulerSolutions.P58PrimeNodes.p_256960901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8015 3) := by change ¬ Nat.Prime 256944871; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4212211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8015 : row 8015 = 1 := (row_8015.checked).trans rfl
theorem model_count_8015 : modelRow 8015 = 1 := by decide
theorem modelrow_agrees_8015 : modelRow 8015 = row 8015 := model_count_8015.trans row_count_8015.symm
def row_8016 : RowCertificate 8016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8016 1) := by change ¬ Nat.Prime 257041057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36720151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8016 2) := by change ¬ Nat.Prime 257025025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51405005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8016 3) := by change ¬ Nat.Prime 257008993; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6946189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8016 : row 8016 = 0 := (row_8016.checked).trans rfl
theorem model_count_8016 : modelRow 8016 = 0 := by decide
theorem modelrow_agrees_8016 : modelRow 8016 = row 8016 := model_count_8016.trans row_count_8016.symm
def row_8017 : RowCertificate 8017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8017 1) := by change ¬ Nat.Prime 257105191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36729313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8017 2) := by change ¬ Nat.Prime 257089157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19776089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8017 3) := by change ¬ Nat.Prime 257073123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85691041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8017 : row 8017 = 0 := (row_8017.checked).trans rfl
theorem model_count_8017 : modelRow 8017 = 0 := by decide
theorem modelrow_agrees_8017 : modelRow 8017 = row 8017 := model_count_8017.trans row_count_8017.symm
def row_8018 : RowCertificate 8018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8018 1) := by change ¬ Nat.Prime 257169333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85723111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8018 2) := by change ¬ Nat.Prime 257153297; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 4851949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8018 3) := by change Nat.Prime 257137261; exact ProjectEulerSolutions.P58PrimeNodes.p_257137261.prime
    exact decide_eq_true prime
theorem row_count_8018 : row 8018 = 1 := (row_8018.checked).trans rfl
theorem model_count_8018 : modelRow 8018 = 1 := by decide
theorem modelrow_agrees_8018 : modelRow 8018 = row 8018 := model_count_8018.trans row_count_8018.symm
def row_8019 : RowCertificate 8019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8019 1) := by change ¬ Nat.Prime 257233483; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19787191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8019 2) := by change ¬ Nat.Prime 257217445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51443489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8019 3) := by change Nat.Prime 257201407; exact ProjectEulerSolutions.P58PrimeNodes.p_257201407.prime
    exact decide_eq_true prime
theorem row_count_8019 : row 8019 = 1 := (row_8019.checked).trans rfl
theorem model_count_8019 : modelRow 8019 = 1 := by decide
theorem modelrow_agrees_8019 : modelRow 8019 = row 8019 := model_count_8019.trans row_count_8019.symm
def row_8020 : RowCertificate 8020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8020 1) := by change ¬ Nat.Prime 257297641; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2652553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8020 2) := by change ¬ Nat.Prime 257281601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6275161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8020 3) := by change ¬ Nat.Prime 257265561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85755187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8020 : row 8020 = 0 := (row_8020.checked).trans rfl
theorem model_count_8020 : modelRow 8020 = 0 := by decide
theorem modelrow_agrees_8020 : modelRow 8020 = row 8020 := model_count_8020.trans row_count_8020.symm
def row_8021 : RowCertificate 8021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8021 1) := by change ¬ Nat.Prime 257361807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85787269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8021 2) := by change ¬ Nat.Prime 257345765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51469153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8021 3) := by change ¬ Nat.Prime 257329723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36761389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8021 : row 8021 = 0 := (row_8021.checked).trans rfl
theorem model_count_8021 : modelRow 8021 = 0 := by decide
theorem modelrow_agrees_8021 : modelRow 8021 = row 8021 := model_count_8021.trans row_count_8021.symm
def row_8022 : RowCertificate 8022 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8022 1) := by change Nat.Prime 257425981; exact ProjectEulerSolutions.P58PrimeNodes.p_257425981.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8022 2) := by change ¬ Nat.Prime 257409937; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15141761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8022 3) := by change ¬ Nat.Prime 257393893; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13547047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8022 : row 8022 = 1 := (row_8022.checked).trans rfl
theorem model_count_8022 : modelRow 8022 = 1 := by decide
theorem modelrow_agrees_8022 : modelRow 8022 = row 8022 := model_count_8022.trans row_count_8022.symm
def row_8023 : RowCertificate 8023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8023 1) := by change ¬ Nat.Prime 257490163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36784309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8023 2) := by change Nat.Prime 257474117; exact ProjectEulerSolutions.P58PrimeNodes.p_257474117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8023 3) := by change ¬ Nat.Prime 257458071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85819357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8023 : row 8023 = 1 := (row_8023.checked).trans rfl
theorem model_count_8023 : modelRow 8023 = 1 := by decide
theorem modelrow_agrees_8023 : modelRow 8023 = row 8023 := model_count_8023.trans row_count_8023.symm
def row_8024 : RowCertificate 8024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8024 1) := by change ¬ Nat.Prime 257554353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85851451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8024 2) := by change ¬ Nat.Prime 257538305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51507661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8024 3) := by change ¬ Nat.Prime 257522257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13553803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8024 : row 8024 = 0 := (row_8024.checked).trans rfl
theorem model_count_8024 : modelRow 8024 = 0 := by decide
theorem modelrow_agrees_8024 : modelRow 8024 = row 8024 := model_count_8024.trans row_count_8024.symm
def row_8025 : RowCertificate 8025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8025 1) := by change ¬ Nat.Prime 257618551; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3845053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8025 2) := by change ¬ Nat.Prime 257602501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19815577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8025 3) := by change Nat.Prime 257586451; exact ProjectEulerSolutions.P58PrimeNodes.p_257586451.prime
    exact decide_eq_true prime
theorem row_count_8025 : row 8025 = 1 := (row_8025.checked).trans rfl
theorem model_count_8025 : modelRow 8025 = 1 := by decide
theorem modelrow_agrees_8025 : modelRow 8025 = row 8025 := model_count_8025.trans row_count_8025.symm
def row_8026 : RowCertificate 8026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8026 1) := by change ¬ Nat.Prime 257682757; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8312347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8026 2) := by change ¬ Nat.Prime 257666705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51533341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8026 3) := by change ¬ Nat.Prime 257650653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85883551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8026 : row 8026 = 0 := (row_8026.checked).trans rfl
theorem model_count_8026 : modelRow 8026 = 0 := by decide
theorem modelrow_agrees_8026 : modelRow 8026 = row 8026 := model_count_8026.trans row_count_8026.symm
def row_8027 : RowCertificate 8027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8027 1) := by change ¬ Nat.Prime 257746971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85915657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8027 2) := by change ¬ Nat.Prime 257730917; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8887273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8027 3) := by change ¬ Nat.Prime 257714863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36816409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8027 : row 8027 = 0 := (row_8027.checked).trans rfl
theorem model_count_8027 : modelRow 8027 = 0 := by decide
theorem modelrow_agrees_8027 : modelRow 8027 = row 8027 := model_count_8027.trans row_count_8027.symm
def row_8028 : RowCertificate 8028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8028 1) := by change ¬ Nat.Prime 257811193; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4226413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8028 2) := by change ¬ Nat.Prime 257795137; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2365093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8028 3) := by change ¬ Nat.Prime 257779081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36825583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8028 : row 8028 = 0 := (row_8028.checked).trans rfl
theorem model_count_8028 : modelRow 8028 = 0 := by decide
theorem modelrow_agrees_8028 : modelRow 8028 = row 8028 := model_count_8028.trans row_count_8028.symm
def row_8029 : RowCertificate 8029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8029 1) := by change ¬ Nat.Prime 257875423; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19836571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8029 2) := by change ¬ Nat.Prime 257859365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51571873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8029 3) := by change ¬ Nat.Prime 257843307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85947769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8029 : row 8029 = 0 := (row_8029.checked).trans rfl
theorem model_count_8029 : modelRow 8029 = 0 := by decide
theorem modelrow_agrees_8029 : modelRow 8029 = row 8029 := model_count_8029.trans row_count_8029.symm
def row_8030 : RowCertificate 8030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8030 1) := by change ¬ Nat.Prime 257939661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 85979887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8030 2) := by change ¬ Nat.Prime 257923601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19840277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8030 3) := by change Nat.Prime 257907541; exact ProjectEulerSolutions.P58PrimeNodes.p_257907541.prime
    exact decide_eq_true prime
theorem row_count_8030 : row 8030 = 1 := (row_8030.checked).trans rfl
theorem model_count_8030 : modelRow 8030 = 1 := by decide
theorem modelrow_agrees_8030 : modelRow 8030 = row 8030 := model_count_8030.trans row_count_8030.symm
def row_8031 : RowCertificate 8031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8031 1) := by change ¬ Nat.Prime 258003907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36857701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8031 2) := by change ¬ Nat.Prime 257987845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51597569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8031 3) := by change ¬ Nat.Prime 257971783; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 824191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8031 : row 8031 = 0 := (row_8031.checked).trans rfl
theorem model_count_8031 : modelRow 8031 = 0 := by decide
theorem modelrow_agrees_8031 : modelRow 8031 = row 8031 := model_count_8031.trans row_count_8031.symm
def row_8032 : RowCertificate 8032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8032 1) := by change ¬ Nat.Prime 258068161; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19851397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8032 2) := by change ¬ Nat.Prime 258052097; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6974381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8032 3) := by change ¬ Nat.Prime 258036033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86012011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8032 : row 8032 = 0 := (row_8032.checked).trans rfl
theorem model_count_8032 : modelRow 8032 = 0 := by decide
theorem modelrow_agrees_8032 : modelRow 8032 = row 8032 := model_count_8032.trans row_count_8032.symm
def row_8033 : RowCertificate 8033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8033 1) := by change ¬ Nat.Prime 258132423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86044141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8033 2) := by change ¬ Nat.Prime 258116357; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 1884061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8033 3) := by change ¬ Nat.Prime 258100291; exact PrattCertificates.composite_of_product (factor := 13831) (cofactor := 18661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8033 : row 8033 = 0 := (row_8033.checked).trans rfl
theorem model_count_8033 : modelRow 8033 = 0 := by decide
theorem modelrow_agrees_8033 : modelRow 8033 = row 8033 := model_count_8033.trans row_count_8033.symm
def row_8034 : RowCertificate 8034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8034 1) := by change ¬ Nat.Prime 258196693; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6978289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8034 2) := by change ¬ Nat.Prime 258180625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51636125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8034 3) := by change ¬ Nat.Prime 258164557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36880651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8034 : row 8034 = 0 := (row_8034.checked).trans rfl
theorem model_count_8034 : modelRow 8034 = 0 := by decide
theorem modelrow_agrees_8034 : modelRow 8034 = row 8034 := model_count_8034.trans row_count_8034.symm
def row_8035 : RowCertificate 8035 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8035 1) := by change Nat.Prime 258260971; exact ProjectEulerSolutions.P58PrimeNodes.p_258260971.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8035 2) := by change ¬ Nat.Prime 258244901; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 919021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8035 3) := by change ¬ Nat.Prime 258228831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86076277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8035 : row 8035 = 1 := (row_8035.checked).trans rfl
theorem model_count_8035 : modelRow 8035 = 1 := by decide
theorem modelrow_agrees_8035 : modelRow 8035 = row 8035 := model_count_8035.trans row_count_8035.symm
def row_8036 : RowCertificate 8036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8036 1) := by change ¬ Nat.Prime 258325257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86108419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8036 2) := by change ¬ Nat.Prime 258309185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51661837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8036 3) := by change ¬ Nat.Prime 258293113; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19868701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8036 : row 8036 = 0 := (row_8036.checked).trans rfl
theorem model_count_8036 : modelRow 8036 = 0 := by decide
theorem modelrow_agrees_8036 : modelRow 8036 = row 8036 := model_count_8036.trans row_count_8036.symm
def row_8037 : RowCertificate 8037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8037 1) := by change ¬ Nat.Prime 258389551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36912793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8037 2) := by change ¬ Nat.Prime 258373477; exact PrattCertificates.composite_of_product (factor := 7877) (cofactor := 32801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8037 3) := by change Nat.Prime 258357403; exact ProjectEulerSolutions.P58PrimeNodes.p_258357403.prime
    exact decide_eq_true prime
theorem row_count_8037 : row 8037 = 1 := (row_8037.checked).trans rfl
theorem model_count_8037 : modelRow 8037 = 1 := by decide
theorem modelrow_agrees_8037 : modelRow 8037 = row 8037 := model_count_8037.trans row_count_8037.symm
def row_8038 : RowCertificate 8038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8038 1) := by change ¬ Nat.Prime 258453853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36921979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8038 2) := by change ¬ Nat.Prime 258437777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19879829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8038 3) := by change ¬ Nat.Prime 258421701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86140567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8038 : row 8038 = 0 := (row_8038.checked).trans rfl
theorem model_count_8038 : modelRow 8038 = 0 := by decide
theorem modelrow_agrees_8038 : modelRow 8038 = row 8038 := model_count_8038.trans row_count_8038.symm
def row_8039 : RowCertificate 8039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8039 1) := by change ¬ Nat.Prime 258518163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86172721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8039 2) := by change ¬ Nat.Prime 258502085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51700417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8039 3) := by change ¬ Nat.Prime 258486007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19883539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8039 : row 8039 = 0 := (row_8039.checked).trans rfl
theorem model_count_8039 : modelRow 8039 = 0 := by decide
theorem modelrow_agrees_8039 : modelRow 8039 = row 8039 := model_count_8039.trans row_count_8039.symm
def row_8040 : RowCertificate 8040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8040 1) := by change ¬ Nat.Prime 258582481; exact PrattCertificates.composite_of_product (factor := 15217) (cofactor := 16993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8040 2) := by change ¬ Nat.Prime 258566401; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2665633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8040 3) := by change ¬ Nat.Prime 258550321; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 364669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8040 : row 8040 = 0 := (row_8040.checked).trans rfl
theorem model_count_8040 : modelRow 8040 = 0 := by decide
theorem modelrow_agrees_8040 : modelRow 8040 = row 8040 := model_count_8040.trans row_count_8040.symm
def row_8041 : RowCertificate 8041 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8041 1) := by change Nat.Prime 258646807; exact ProjectEulerSolutions.P58PrimeNodes.p_258646807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8041 2) := by change ¬ Nat.Prime 258630725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51726145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8041 3) := by change ¬ Nat.Prime 258614643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86204881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8041 : row 8041 = 1 := (row_8041.checked).trans rfl
theorem model_count_8041 : modelRow 8041 = 1 := by decide
theorem modelrow_agrees_8041 : modelRow 8041 = row 8041 := model_count_8041.trans row_count_8041.symm
def row_8042 : RowCertificate 8042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8042 1) := by change ¬ Nat.Prime 258711141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86237047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8042 2) := by change Nat.Prime 258695057; exact ProjectEulerSolutions.P58PrimeNodes.p_258695057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8042 3) := by change ¬ Nat.Prime 258678973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36954139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8042 : row 8042 = 1 := (row_8042.checked).trans rfl
theorem model_count_8042 : modelRow 8042 = 1 := by decide
theorem modelrow_agrees_8042 : modelRow 8042 = row 8042 := model_count_8042.trans row_count_8042.symm
def row_8043 : RowCertificate 8043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8043 1) := by change ¬ Nat.Prime 258775483; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2374087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8043 2) := by change ¬ Nat.Prime 258759397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19904569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8043 3) := by change ¬ Nat.Prime 258743311; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13618069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8043 : row 8043 = 0 := (row_8043.checked).trans rfl
theorem model_count_8043 : modelRow 8043 = 0 := by decide
theorem modelrow_agrees_8043 : modelRow 8043 = row 8043 := model_count_8043.trans row_count_8043.symm
def row_8044 : RowCertificate 8044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8044 1) := by change ¬ Nat.Prime 258839833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 36977119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8044 2) := by change ¬ Nat.Prime 258823745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51764749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8044 3) := by change ¬ Nat.Prime 258807657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86269219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8044 : row 8044 = 0 := (row_8044.checked).trans rfl
theorem model_count_8044 : modelRow 8044 = 0 := by decide
theorem modelrow_agrees_8044 : modelRow 8044 = row 8044 := model_count_8044.trans row_count_8044.symm
def row_8045 : RowCertificate 8045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8045 1) := by change ¬ Nat.Prime 258904191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86301397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8045 2) := by change ¬ Nat.Prime 258888101; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1430321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8045 3) := by change ¬ Nat.Prime 258872011; exact PrattCertificates.composite_of_product (factor := 811) (cofactor := 319201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8045 : row 8045 = 0 := (row_8045.checked).trans rfl
theorem model_count_8045 : modelRow 8045 = 0 := by decide
theorem modelrow_agrees_8045 : modelRow 8045 = row 8045 := model_count_8045.trans row_count_8045.symm
def row_8046 : RowCertificate 8046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8046 1) := by change ¬ Nat.Prime 258968557; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3278083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8046 2) := by change ¬ Nat.Prime 258952465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51790493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8046 3) := by change ¬ Nat.Prime 258936373; exact PrattCertificates.composite_of_product (factor := 2851) (cofactor := 90823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8046 : row 8046 = 0 := (row_8046.checked).trans rfl
theorem model_count_8046 : modelRow 8046 = 0 := by decide
theorem modelrow_agrees_8046 : modelRow 8046 = row 8046 := model_count_8046.trans row_count_8046.symm
def row_8047 : RowCertificate 8047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8047 1) := by change ¬ Nat.Prime 259032931; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8355901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8047 2) := by change ¬ Nat.Prime 259016837; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 633293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8047 3) := by change ¬ Nat.Prime 259000743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86333581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8047 : row 8047 = 0 := (row_8047.checked).trans rfl
theorem model_count_8047 : modelRow 8047 = 0 := by decide
theorem modelrow_agrees_8047 : modelRow 8047 = row 8047 := model_count_8047.trans row_count_8047.symm
def row_8048 : RowCertificate 8048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8048 1) := by change ¬ Nat.Prime 259097313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86365771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8048 2) := by change ¬ Nat.Prime 259081217; exact PrattCertificates.composite_of_product (factor := 14713) (cofactor := 17609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8048 3) := by change ¬ Nat.Prime 259065121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37009303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8048 : row 8048 = 0 := (row_8048.checked).trans rfl
theorem model_count_8048 : modelRow 8048 = 0 := by decide
theorem modelrow_agrees_8048 : modelRow 8048 = row 8048 := model_count_8048.trans row_count_8048.symm
def row_8049 : RowCertificate 8049 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8049 1) := by change Nat.Prime 259161703; exact ProjectEulerSolutions.P58PrimeNodes.p_259161703.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8049 2) := by change ¬ Nat.Prime 259145605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51829121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8049 3) := by change ¬ Nat.Prime 259129507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37018501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8049 : row 8049 = 1 := (row_8049.checked).trans rfl
theorem model_count_8049 : modelRow 8049 = 1 := by decide
theorem modelrow_agrees_8049 : modelRow 8049 = row 8049 := model_count_8049.trans row_count_8049.symm
def row_8050 : RowCertificate 8050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8050 1) := by change ¬ Nat.Prime 259226101; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13643479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8050 2) := by change ¬ Nat.Prime 259210001; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1343057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8050 3) := by change ¬ Nat.Prime 259193901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86397967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8050 : row 8050 = 0 := (row_8050.checked).trans rfl
theorem model_count_8050 : modelRow 8050 = 0 := by decide
theorem modelrow_agrees_8050 : modelRow 8050 = row 8050 := model_count_8050.trans row_count_8050.symm
def row_8051 : RowCertificate 8051 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8051 1) := by change ¬ Nat.Prime 259290507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86430169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8051 2) := by change ¬ Nat.Prime 259274405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51854881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8051 3) := by change Nat.Prime 259258303; exact ProjectEulerSolutions.P58PrimeNodes.p_259258303.prime
    exact decide_eq_true prime
theorem row_count_8051 : row 8051 = 1 := (row_8051.checked).trans rfl
theorem model_count_8051 : modelRow 8051 = 1 := by decide
theorem modelrow_agrees_8051 : modelRow 8051 = row 8051 := model_count_8051.trans row_count_8051.symm
def row_8052 : RowCertificate 8052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8052 1) := by change ¬ Nat.Prime 259354921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37050703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8052 2) := by change ¬ Nat.Prime 259338817; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6325337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8052 3) := by change ¬ Nat.Prime 259322713; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19947901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8052 : row 8052 = 0 := (row_8052.checked).trans rfl
theorem model_count_8052 : modelRow 8052 = 0 := by decide
theorem modelrow_agrees_8052 : modelRow 8052 = row 8052 := model_count_8052.trans row_count_8052.symm
def row_8053 : RowCertificate 8053 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8053 1) := by change Nat.Prime 259419343; exact ProjectEulerSolutions.P58PrimeNodes.p_259419343.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8053 2) := by change ¬ Nat.Prime 259403237; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3553469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8053 3) := by change ¬ Nat.Prime 259387131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86462377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8053 : row 8053 = 1 := (row_8053.checked).trans rfl
theorem model_count_8053 : modelRow 8053 = 1 := by decide
theorem modelrow_agrees_8053 : modelRow 8053 = row 8053 := model_count_8053.trans row_count_8053.symm
def row_8054 : RowCertificate 8054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8054 1) := by change ¬ Nat.Prime 259483773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86494591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8054 2) := by change ¬ Nat.Prime 259467665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51893533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8054 3) := by change ¬ Nat.Prime 259451557; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 519943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8054 : row 8054 = 0 := (row_8054.checked).trans rfl
theorem model_count_8054 : modelRow 8054 = 0 := by decide
theorem modelrow_agrees_8054 : modelRow 8054 = row 8054 := model_count_8054.trans row_count_8054.symm
def row_8055 : RowCertificate 8055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8055 1) := by change ¬ Nat.Prime 259548211; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19965247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8055 2) := by change Nat.Prime 259532101; exact ProjectEulerSolutions.P58PrimeNodes.p_259532101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8055 3) := by change ¬ Nat.Prime 259515991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37073713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8055 : row 8055 = 1 := (row_8055.checked).trans rfl
theorem model_count_8055 : modelRow 8055 = 1 := by decide
theorem modelrow_agrees_8055 : modelRow 8055 = row 8055 := model_count_8055.trans row_count_8055.symm
def row_8056 : RowCertificate 8056 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 8056 1) := by change Nat.Prime 259612657; exact ProjectEulerSolutions.P58PrimeNodes.p_259612657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8056 2) := by change ¬ Nat.Prime 259596545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51919309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8056 3) := by change ¬ Nat.Prime 259580433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86526811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8056 : row 8056 = 1 := (row_8056.checked).trans rfl
theorem model_count_8056 : modelRow 8056 = 1 := by decide
theorem modelrow_agrees_8056 : modelRow 8056 = row 8056 := model_count_8056.trans row_count_8056.symm
def row_8057 : RowCertificate 8057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8057 1) := by change ¬ Nat.Prime 259677111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86559037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8057 2) := by change ¬ Nat.Prime 259660997; exact PrattCertificates.composite_of_product (factor := 1097) (cofactor := 236701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8057 3) := by change Nat.Prime 259644883; exact ProjectEulerSolutions.P58PrimeNodes.p_259644883.prime
    exact decide_eq_true prime
theorem row_count_8057 : row 8057 = 1 := (row_8057.checked).trans rfl
theorem model_count_8057 : modelRow 8057 = 1 := by decide
theorem modelrow_agrees_8057 : modelRow 8057 = row 8057 := model_count_8057.trans row_count_8057.symm
def row_8058 : RowCertificate 8058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8058 1) := by change ¬ Nat.Prime 259741573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37105939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 8058 2) := by change Nat.Prime 259725457; exact ProjectEulerSolutions.P58PrimeNodes.p_259725457.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 8058 3) := by change Nat.Prime 259709341; exact ProjectEulerSolutions.P58PrimeNodes.p_259709341.prime
    exact decide_eq_true prime
theorem row_count_8058 : row 8058 = 2 := (row_8058.checked).trans rfl
theorem model_count_8058 : modelRow 8058 = 2 := by decide
theorem modelrow_agrees_8058 : modelRow 8058 = row 8058 := model_count_8058.trans row_count_8058.symm
def row_8059 : RowCertificate 8059 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8059 1) := by change ¬ Nat.Prime 259806043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37115149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8059 2) := by change ¬ Nat.Prime 259789925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51957985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8059 3) := by change ¬ Nat.Prime 259773807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86591269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8059 : row 8059 = 0 := (row_8059.checked).trans rfl
theorem model_count_8059 : modelRow 8059 = 0 := by decide
theorem modelrow_agrees_8059 : modelRow 8059 = row 8059 := model_count_8059.trans row_count_8059.symm
def row_8060 : RowCertificate 8060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8060 1) := by change ¬ Nat.Prime 259870521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86623507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8060 2) := by change ¬ Nat.Prime 259854401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 15285553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8060 3) := by change ¬ Nat.Prime 259838281; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13675699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8060 : row 8060 = 0 := (row_8060.checked).trans rfl
theorem model_count_8060 : modelRow 8060 = 0 := by decide
theorem modelrow_agrees_8060 : modelRow 8060 = row 8060 := model_count_8060.trans row_count_8060.symm
def row_8061 : RowCertificate 8061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8061 1) := by change ¬ Nat.Prime 259935007; exact PrattCertificates.composite_of_product (factor := 1279) (cofactor := 203233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8061 2) := by change ¬ Nat.Prime 259918885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 51983777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8061 3) := by change ¬ Nat.Prime 259902763; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7024399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8061 : row 8061 = 0 := (row_8061.checked).trans rfl
theorem model_count_8061 : modelRow 8061 = 0 := by decide
theorem modelrow_agrees_8061 : modelRow 8061 = row 8061 := model_count_8061.trans row_count_8061.symm
def row_8062 : RowCertificate 8062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8062 1) := by change ¬ Nat.Prime 259999501; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3561637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8062 2) := by change ¬ Nat.Prime 259983377; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2680241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8062 3) := by change ¬ Nat.Prime 259967253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86655751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8062 : row 8062 = 0 := (row_8062.checked).trans rfl
theorem model_count_8062 : modelRow 8062 = 0 := by decide
theorem modelrow_agrees_8062 : modelRow 8062 = row 8062 := model_count_8062.trans row_count_8062.symm
def row_8063 : RowCertificate 8063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8063 1) := by change ¬ Nat.Prime 260064003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 86688001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8063 2) := by change ¬ Nat.Prime 260047877; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 7028321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8063 3) := by change ¬ Nat.Prime 260031751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 37147393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8063 : row 8063 = 0 := (row_8063.checked).trans rfl
theorem model_count_8063 : modelRow 8063 = 0 := by decide
theorem modelrow_agrees_8063 : modelRow 8063 = row 8063 := model_count_8063.trans row_count_8063.symm
def row_8064 : RowCertificate 8064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 8064 1) := by change ¬ Nat.Prime 260128513; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 569209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 8064 2) := by change ¬ Nat.Prime 260112385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 52022477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 8064 3) := by change ¬ Nat.Prime 260096257; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1347649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_8064 : row 8064 = 0 := (row_8064.checked).trans rfl
theorem model_count_8064 : modelRow 8064 = 0 := by decide
theorem modelrow_agrees_8064 : modelRow 8064 = row 8064 := model_count_8064.trans row_count_8064.symm
private def atomic_7937 : CountSegment 1 10 13120 7936 7937 3363 3363 := CountSegment.single (first := 7936) (initial := 3363) row_count_7937 (by decide)
private theorem atomic_7937_agreement : RowAgreement 7936 7937 := RowAgreement.single (first := 7936) modelrow_agrees_7937
private def atomic_7938 : CountSegment 1 10 13120 7937 7938 3363 3364 := CountSegment.single (first := 7937) (initial := 3363) row_count_7938 (by decide)
private theorem atomic_7938_agreement : RowAgreement 7937 7938 := RowAgreement.single (first := 7937) modelrow_agrees_7938
private def atomic_7939 : CountSegment 1 10 13120 7938 7939 3364 3364 := CountSegment.single (first := 7938) (initial := 3364) row_count_7939 (by decide)
private theorem atomic_7939_agreement : RowAgreement 7938 7939 := RowAgreement.single (first := 7938) modelrow_agrees_7939
private def atomic_7940 : CountSegment 1 10 13120 7939 7940 3364 3364 := CountSegment.single (first := 7939) (initial := 3364) row_count_7940 (by decide)
private theorem atomic_7940_agreement : RowAgreement 7939 7940 := RowAgreement.single (first := 7939) modelrow_agrees_7940
private def atomic_7941 : CountSegment 1 10 13120 7940 7941 3364 3365 := CountSegment.single (first := 7940) (initial := 3364) row_count_7941 (by decide)
private theorem atomic_7941_agreement : RowAgreement 7940 7941 := RowAgreement.single (first := 7940) modelrow_agrees_7941
private def atomic_7942 : CountSegment 1 10 13120 7941 7942 3365 3366 := CountSegment.single (first := 7941) (initial := 3365) row_count_7942 (by decide)
private theorem atomic_7942_agreement : RowAgreement 7941 7942 := RowAgreement.single (first := 7941) modelrow_agrees_7942
private def atomic_7943 : CountSegment 1 10 13120 7942 7943 3366 3366 := CountSegment.single (first := 7942) (initial := 3366) row_count_7943 (by decide)
private theorem atomic_7943_agreement : RowAgreement 7942 7943 := RowAgreement.single (first := 7942) modelrow_agrees_7943
private def atomic_7944 : CountSegment 1 10 13120 7943 7944 3366 3366 := CountSegment.single (first := 7943) (initial := 3366) row_count_7944 (by decide)
private theorem atomic_7944_agreement : RowAgreement 7943 7944 := RowAgreement.single (first := 7943) modelrow_agrees_7944
private def atomic_7945 : CountSegment 1 10 13120 7944 7945 3366 3367 := CountSegment.single (first := 7944) (initial := 3366) row_count_7945 (by decide)
private theorem atomic_7945_agreement : RowAgreement 7944 7945 := RowAgreement.single (first := 7944) modelrow_agrees_7945
private def atomic_7946 : CountSegment 1 10 13120 7945 7946 3367 3367 := CountSegment.single (first := 7945) (initial := 3367) row_count_7946 (by decide)
private theorem atomic_7946_agreement : RowAgreement 7945 7946 := RowAgreement.single (first := 7945) modelrow_agrees_7946
private def atomic_7947 : CountSegment 1 10 13120 7946 7947 3367 3368 := CountSegment.single (first := 7946) (initial := 3367) row_count_7947 (by decide)
private theorem atomic_7947_agreement : RowAgreement 7946 7947 := RowAgreement.single (first := 7946) modelrow_agrees_7947
private def atomic_7948 : CountSegment 1 10 13120 7947 7948 3368 3369 := CountSegment.single (first := 7947) (initial := 3368) row_count_7948 (by decide)
private theorem atomic_7948_agreement : RowAgreement 7947 7948 := RowAgreement.single (first := 7947) modelrow_agrees_7948
private def atomic_7949 : CountSegment 1 10 13120 7948 7949 3369 3369 := CountSegment.single (first := 7948) (initial := 3369) row_count_7949 (by decide)
private theorem atomic_7949_agreement : RowAgreement 7948 7949 := RowAgreement.single (first := 7948) modelrow_agrees_7949
private def atomic_7950 : CountSegment 1 10 13120 7949 7950 3369 3370 := CountSegment.single (first := 7949) (initial := 3369) row_count_7950 (by decide)
private theorem atomic_7950_agreement : RowAgreement 7949 7950 := RowAgreement.single (first := 7949) modelrow_agrees_7950
private def atomic_7951 : CountSegment 1 10 13120 7950 7951 3370 3370 := CountSegment.single (first := 7950) (initial := 3370) row_count_7951 (by decide)
private theorem atomic_7951_agreement : RowAgreement 7950 7951 := RowAgreement.single (first := 7950) modelrow_agrees_7951
private def atomic_7952 : CountSegment 1 10 13120 7951 7952 3370 3370 := CountSegment.single (first := 7951) (initial := 3370) row_count_7952 (by decide)
private theorem atomic_7952_agreement : RowAgreement 7951 7952 := RowAgreement.single (first := 7951) modelrow_agrees_7952
private def atomic_7953 : CountSegment 1 10 13120 7952 7953 3370 3370 := CountSegment.single (first := 7952) (initial := 3370) row_count_7953 (by decide)
private theorem atomic_7953_agreement : RowAgreement 7952 7953 := RowAgreement.single (first := 7952) modelrow_agrees_7953
private def atomic_7954 : CountSegment 1 10 13120 7953 7954 3370 3370 := CountSegment.single (first := 7953) (initial := 3370) row_count_7954 (by decide)
private theorem atomic_7954_agreement : RowAgreement 7953 7954 := RowAgreement.single (first := 7953) modelrow_agrees_7954
private def atomic_7955 : CountSegment 1 10 13120 7954 7955 3370 3370 := CountSegment.single (first := 7954) (initial := 3370) row_count_7955 (by decide)
private theorem atomic_7955_agreement : RowAgreement 7954 7955 := RowAgreement.single (first := 7954) modelrow_agrees_7955
private def atomic_7956 : CountSegment 1 10 13120 7955 7956 3370 3371 := CountSegment.single (first := 7955) (initial := 3370) row_count_7956 (by decide)
private theorem atomic_7956_agreement : RowAgreement 7955 7956 := RowAgreement.single (first := 7955) modelrow_agrees_7956
private def atomic_7957 : CountSegment 1 10 13120 7956 7957 3371 3372 := CountSegment.single (first := 7956) (initial := 3371) row_count_7957 (by decide)
private theorem atomic_7957_agreement : RowAgreement 7956 7957 := RowAgreement.single (first := 7956) modelrow_agrees_7957
private def atomic_7958 : CountSegment 1 10 13120 7957 7958 3372 3372 := CountSegment.single (first := 7957) (initial := 3372) row_count_7958 (by decide)
private theorem atomic_7958_agreement : RowAgreement 7957 7958 := RowAgreement.single (first := 7957) modelrow_agrees_7958
private def atomic_7959 : CountSegment 1 10 13120 7958 7959 3372 3372 := CountSegment.single (first := 7958) (initial := 3372) row_count_7959 (by decide)
private theorem atomic_7959_agreement : RowAgreement 7958 7959 := RowAgreement.single (first := 7958) modelrow_agrees_7959
private def atomic_7960 : CountSegment 1 10 13120 7959 7960 3372 3372 := CountSegment.single (first := 7959) (initial := 3372) row_count_7960 (by decide)
private theorem atomic_7960_agreement : RowAgreement 7959 7960 := RowAgreement.single (first := 7959) modelrow_agrees_7960
private def atomic_7961 : CountSegment 1 10 13120 7960 7961 3372 3372 := CountSegment.single (first := 7960) (initial := 3372) row_count_7961 (by decide)
private theorem atomic_7961_agreement : RowAgreement 7960 7961 := RowAgreement.single (first := 7960) modelrow_agrees_7961
private def atomic_7962 : CountSegment 1 10 13120 7961 7962 3372 3374 := CountSegment.single (first := 7961) (initial := 3372) row_count_7962 (by decide)
private theorem atomic_7962_agreement : RowAgreement 7961 7962 := RowAgreement.single (first := 7961) modelrow_agrees_7962
private def atomic_7963 : CountSegment 1 10 13120 7962 7963 3374 3375 := CountSegment.single (first := 7962) (initial := 3374) row_count_7963 (by decide)
private theorem atomic_7963_agreement : RowAgreement 7962 7963 := RowAgreement.single (first := 7962) modelrow_agrees_7963
private def atomic_7964 : CountSegment 1 10 13120 7963 7964 3375 3375 := CountSegment.single (first := 7963) (initial := 3375) row_count_7964 (by decide)
private theorem atomic_7964_agreement : RowAgreement 7963 7964 := RowAgreement.single (first := 7963) modelrow_agrees_7964
private def atomic_7965 : CountSegment 1 10 13120 7964 7965 3375 3375 := CountSegment.single (first := 7964) (initial := 3375) row_count_7965 (by decide)
private theorem atomic_7965_agreement : RowAgreement 7964 7965 := RowAgreement.single (first := 7964) modelrow_agrees_7965
private def atomic_7966 : CountSegment 1 10 13120 7965 7966 3375 3375 := CountSegment.single (first := 7965) (initial := 3375) row_count_7966 (by decide)
private theorem atomic_7966_agreement : RowAgreement 7965 7966 := RowAgreement.single (first := 7965) modelrow_agrees_7966
private def atomic_7967 : CountSegment 1 10 13120 7966 7967 3375 3375 := CountSegment.single (first := 7966) (initial := 3375) row_count_7967 (by decide)
private theorem atomic_7967_agreement : RowAgreement 7966 7967 := RowAgreement.single (first := 7966) modelrow_agrees_7967
private def atomic_7968 : CountSegment 1 10 13120 7967 7968 3375 3376 := CountSegment.single (first := 7967) (initial := 3375) row_count_7968 (by decide)
private theorem atomic_7968_agreement : RowAgreement 7967 7968 := RowAgreement.single (first := 7967) modelrow_agrees_7968
private def atomic_7969 : CountSegment 1 10 13120 7968 7969 3376 3376 := CountSegment.single (first := 7968) (initial := 3376) row_count_7969 (by decide)
private theorem atomic_7969_agreement : RowAgreement 7968 7969 := RowAgreement.single (first := 7968) modelrow_agrees_7969
private def atomic_7970 : CountSegment 1 10 13120 7969 7970 3376 3376 := CountSegment.single (first := 7969) (initial := 3376) row_count_7970 (by decide)
private theorem atomic_7970_agreement : RowAgreement 7969 7970 := RowAgreement.single (first := 7969) modelrow_agrees_7970
private def atomic_7971 : CountSegment 1 10 13120 7970 7971 3376 3376 := CountSegment.single (first := 7970) (initial := 3376) row_count_7971 (by decide)
private theorem atomic_7971_agreement : RowAgreement 7970 7971 := RowAgreement.single (first := 7970) modelrow_agrees_7971
private def atomic_7972 : CountSegment 1 10 13120 7971 7972 3376 3378 := CountSegment.single (first := 7971) (initial := 3376) row_count_7972 (by decide)
private theorem atomic_7972_agreement : RowAgreement 7971 7972 := RowAgreement.single (first := 7971) modelrow_agrees_7972
private def atomic_7973 : CountSegment 1 10 13120 7972 7973 3378 3378 := CountSegment.single (first := 7972) (initial := 3378) row_count_7973 (by decide)
private theorem atomic_7973_agreement : RowAgreement 7972 7973 := RowAgreement.single (first := 7972) modelrow_agrees_7973
private def atomic_7974 : CountSegment 1 10 13120 7973 7974 3378 3378 := CountSegment.single (first := 7973) (initial := 3378) row_count_7974 (by decide)
private theorem atomic_7974_agreement : RowAgreement 7973 7974 := RowAgreement.single (first := 7973) modelrow_agrees_7974
private def atomic_7975 : CountSegment 1 10 13120 7974 7975 3378 3378 := CountSegment.single (first := 7974) (initial := 3378) row_count_7975 (by decide)
private theorem atomic_7975_agreement : RowAgreement 7974 7975 := RowAgreement.single (first := 7974) modelrow_agrees_7975
private def atomic_7976 : CountSegment 1 10 13120 7975 7976 3378 3378 := CountSegment.single (first := 7975) (initial := 3378) row_count_7976 (by decide)
private theorem atomic_7976_agreement : RowAgreement 7975 7976 := RowAgreement.single (first := 7975) modelrow_agrees_7976
private def atomic_7977 : CountSegment 1 10 13120 7976 7977 3378 3380 := CountSegment.single (first := 7976) (initial := 3378) row_count_7977 (by decide)
private theorem atomic_7977_agreement : RowAgreement 7976 7977 := RowAgreement.single (first := 7976) modelrow_agrees_7977
private def atomic_7978 : CountSegment 1 10 13120 7977 7978 3380 3380 := CountSegment.single (first := 7977) (initial := 3380) row_count_7978 (by decide)
private theorem atomic_7978_agreement : RowAgreement 7977 7978 := RowAgreement.single (first := 7977) modelrow_agrees_7978
private def atomic_7979 : CountSegment 1 10 13120 7978 7979 3380 3380 := CountSegment.single (first := 7978) (initial := 3380) row_count_7979 (by decide)
private theorem atomic_7979_agreement : RowAgreement 7978 7979 := RowAgreement.single (first := 7978) modelrow_agrees_7979
private def atomic_7980 : CountSegment 1 10 13120 7979 7980 3380 3380 := CountSegment.single (first := 7979) (initial := 3380) row_count_7980 (by decide)
private theorem atomic_7980_agreement : RowAgreement 7979 7980 := RowAgreement.single (first := 7979) modelrow_agrees_7980
private def atomic_7981 : CountSegment 1 10 13120 7980 7981 3380 3380 := CountSegment.single (first := 7980) (initial := 3380) row_count_7981 (by decide)
private theorem atomic_7981_agreement : RowAgreement 7980 7981 := RowAgreement.single (first := 7980) modelrow_agrees_7981
private def atomic_7982 : CountSegment 1 10 13120 7981 7982 3380 3381 := CountSegment.single (first := 7981) (initial := 3380) row_count_7982 (by decide)
private theorem atomic_7982_agreement : RowAgreement 7981 7982 := RowAgreement.single (first := 7981) modelrow_agrees_7982
private def atomic_7983 : CountSegment 1 10 13120 7982 7983 3381 3382 := CountSegment.single (first := 7982) (initial := 3381) row_count_7983 (by decide)
private theorem atomic_7983_agreement : RowAgreement 7982 7983 := RowAgreement.single (first := 7982) modelrow_agrees_7983
private def atomic_7984 : CountSegment 1 10 13120 7983 7984 3382 3382 := CountSegment.single (first := 7983) (initial := 3382) row_count_7984 (by decide)
private theorem atomic_7984_agreement : RowAgreement 7983 7984 := RowAgreement.single (first := 7983) modelrow_agrees_7984
private def atomic_7985 : CountSegment 1 10 13120 7984 7985 3382 3382 := CountSegment.single (first := 7984) (initial := 3382) row_count_7985 (by decide)
private theorem atomic_7985_agreement : RowAgreement 7984 7985 := RowAgreement.single (first := 7984) modelrow_agrees_7985
private def atomic_7986 : CountSegment 1 10 13120 7985 7986 3382 3383 := CountSegment.single (first := 7985) (initial := 3382) row_count_7986 (by decide)
private theorem atomic_7986_agreement : RowAgreement 7985 7986 := RowAgreement.single (first := 7985) modelrow_agrees_7986
private def atomic_7987 : CountSegment 1 10 13120 7986 7987 3383 3384 := CountSegment.single (first := 7986) (initial := 3383) row_count_7987 (by decide)
private theorem atomic_7987_agreement : RowAgreement 7986 7987 := RowAgreement.single (first := 7986) modelrow_agrees_7987
private def atomic_7988 : CountSegment 1 10 13120 7987 7988 3384 3384 := CountSegment.single (first := 7987) (initial := 3384) row_count_7988 (by decide)
private theorem atomic_7988_agreement : RowAgreement 7987 7988 := RowAgreement.single (first := 7987) modelrow_agrees_7988
private def atomic_7989 : CountSegment 1 10 13120 7988 7989 3384 3384 := CountSegment.single (first := 7988) (initial := 3384) row_count_7989 (by decide)
private theorem atomic_7989_agreement : RowAgreement 7988 7989 := RowAgreement.single (first := 7988) modelrow_agrees_7989
private def atomic_7990 : CountSegment 1 10 13120 7989 7990 3384 3385 := CountSegment.single (first := 7989) (initial := 3384) row_count_7990 (by decide)
private theorem atomic_7990_agreement : RowAgreement 7989 7990 := RowAgreement.single (first := 7989) modelrow_agrees_7990
private def atomic_7991 : CountSegment 1 10 13120 7990 7991 3385 3385 := CountSegment.single (first := 7990) (initial := 3385) row_count_7991 (by decide)
private theorem atomic_7991_agreement : RowAgreement 7990 7991 := RowAgreement.single (first := 7990) modelrow_agrees_7991
private def atomic_7992 : CountSegment 1 10 13120 7991 7992 3385 3386 := CountSegment.single (first := 7991) (initial := 3385) row_count_7992 (by decide)
private theorem atomic_7992_agreement : RowAgreement 7991 7992 := RowAgreement.single (first := 7991) modelrow_agrees_7992
private def atomic_7993 : CountSegment 1 10 13120 7992 7993 3386 3386 := CountSegment.single (first := 7992) (initial := 3386) row_count_7993 (by decide)
private theorem atomic_7993_agreement : RowAgreement 7992 7993 := RowAgreement.single (first := 7992) modelrow_agrees_7993
private def atomic_7994 : CountSegment 1 10 13120 7993 7994 3386 3387 := CountSegment.single (first := 7993) (initial := 3386) row_count_7994 (by decide)
private theorem atomic_7994_agreement : RowAgreement 7993 7994 := RowAgreement.single (first := 7993) modelrow_agrees_7994
private def atomic_7995 : CountSegment 1 10 13120 7994 7995 3387 3387 := CountSegment.single (first := 7994) (initial := 3387) row_count_7995 (by decide)
private theorem atomic_7995_agreement : RowAgreement 7994 7995 := RowAgreement.single (first := 7994) modelrow_agrees_7995
private def atomic_7996 : CountSegment 1 10 13120 7995 7996 3387 3387 := CountSegment.single (first := 7995) (initial := 3387) row_count_7996 (by decide)
private theorem atomic_7996_agreement : RowAgreement 7995 7996 := RowAgreement.single (first := 7995) modelrow_agrees_7996
private def atomic_7997 : CountSegment 1 10 13120 7996 7997 3387 3387 := CountSegment.single (first := 7996) (initial := 3387) row_count_7997 (by decide)
private theorem atomic_7997_agreement : RowAgreement 7996 7997 := RowAgreement.single (first := 7996) modelrow_agrees_7997
private def atomic_7998 : CountSegment 1 10 13120 7997 7998 3387 3388 := CountSegment.single (first := 7997) (initial := 3387) row_count_7998 (by decide)
private theorem atomic_7998_agreement : RowAgreement 7997 7998 := RowAgreement.single (first := 7997) modelrow_agrees_7998
private def atomic_7999 : CountSegment 1 10 13120 7998 7999 3388 3389 := CountSegment.single (first := 7998) (initial := 3388) row_count_7999 (by decide)
private theorem atomic_7999_agreement : RowAgreement 7998 7999 := RowAgreement.single (first := 7998) modelrow_agrees_7999
private def atomic_8000 : CountSegment 1 10 13120 7999 8000 3389 3390 := CountSegment.single (first := 7999) (initial := 3389) row_count_8000 (by decide)
private theorem atomic_8000_agreement : RowAgreement 7999 8000 := RowAgreement.single (first := 7999) modelrow_agrees_8000
private def atomic_8001 : CountSegment 1 10 13120 8000 8001 3390 3390 := CountSegment.single (first := 8000) (initial := 3390) row_count_8001 (by decide)
private theorem atomic_8001_agreement : RowAgreement 8000 8001 := RowAgreement.single (first := 8000) modelrow_agrees_8001
private def atomic_8002 : CountSegment 1 10 13120 8001 8002 3390 3391 := CountSegment.single (first := 8001) (initial := 3390) row_count_8002 (by decide)
private theorem atomic_8002_agreement : RowAgreement 8001 8002 := RowAgreement.single (first := 8001) modelrow_agrees_8002
private def atomic_8003 : CountSegment 1 10 13120 8002 8003 3391 3391 := CountSegment.single (first := 8002) (initial := 3391) row_count_8003 (by decide)
private theorem atomic_8003_agreement : RowAgreement 8002 8003 := RowAgreement.single (first := 8002) modelrow_agrees_8003
private def atomic_8004 : CountSegment 1 10 13120 8003 8004 3391 3393 := CountSegment.single (first := 8003) (initial := 3391) row_count_8004 (by decide)
private theorem atomic_8004_agreement : RowAgreement 8003 8004 := RowAgreement.single (first := 8003) modelrow_agrees_8004
private def atomic_8005 : CountSegment 1 10 13120 8004 8005 3393 3393 := CountSegment.single (first := 8004) (initial := 3393) row_count_8005 (by decide)
private theorem atomic_8005_agreement : RowAgreement 8004 8005 := RowAgreement.single (first := 8004) modelrow_agrees_8005
private def atomic_8006 : CountSegment 1 10 13120 8005 8006 3393 3393 := CountSegment.single (first := 8005) (initial := 3393) row_count_8006 (by decide)
private theorem atomic_8006_agreement : RowAgreement 8005 8006 := RowAgreement.single (first := 8005) modelrow_agrees_8006
private def atomic_8007 : CountSegment 1 10 13120 8006 8007 3393 3394 := CountSegment.single (first := 8006) (initial := 3393) row_count_8007 (by decide)
private theorem atomic_8007_agreement : RowAgreement 8006 8007 := RowAgreement.single (first := 8006) modelrow_agrees_8007
private def atomic_8008 : CountSegment 1 10 13120 8007 8008 3394 3394 := CountSegment.single (first := 8007) (initial := 3394) row_count_8008 (by decide)
private theorem atomic_8008_agreement : RowAgreement 8007 8008 := RowAgreement.single (first := 8007) modelrow_agrees_8008
private def atomic_8009 : CountSegment 1 10 13120 8008 8009 3394 3394 := CountSegment.single (first := 8008) (initial := 3394) row_count_8009 (by decide)
private theorem atomic_8009_agreement : RowAgreement 8008 8009 := RowAgreement.single (first := 8008) modelrow_agrees_8009
private def atomic_8010 : CountSegment 1 10 13120 8009 8010 3394 3394 := CountSegment.single (first := 8009) (initial := 3394) row_count_8010 (by decide)
private theorem atomic_8010_agreement : RowAgreement 8009 8010 := RowAgreement.single (first := 8009) modelrow_agrees_8010
private def atomic_8011 : CountSegment 1 10 13120 8010 8011 3394 3394 := CountSegment.single (first := 8010) (initial := 3394) row_count_8011 (by decide)
private theorem atomic_8011_agreement : RowAgreement 8010 8011 := RowAgreement.single (first := 8010) modelrow_agrees_8011
private def atomic_8012 : CountSegment 1 10 13120 8011 8012 3394 3395 := CountSegment.single (first := 8011) (initial := 3394) row_count_8012 (by decide)
private theorem atomic_8012_agreement : RowAgreement 8011 8012 := RowAgreement.single (first := 8011) modelrow_agrees_8012
private def atomic_8013 : CountSegment 1 10 13120 8012 8013 3395 3396 := CountSegment.single (first := 8012) (initial := 3395) row_count_8013 (by decide)
private theorem atomic_8013_agreement : RowAgreement 8012 8013 := RowAgreement.single (first := 8012) modelrow_agrees_8013
private def atomic_8014 : CountSegment 1 10 13120 8013 8014 3396 3396 := CountSegment.single (first := 8013) (initial := 3396) row_count_8014 (by decide)
private theorem atomic_8014_agreement : RowAgreement 8013 8014 := RowAgreement.single (first := 8013) modelrow_agrees_8014
private def atomic_8015 : CountSegment 1 10 13120 8014 8015 3396 3397 := CountSegment.single (first := 8014) (initial := 3396) row_count_8015 (by decide)
private theorem atomic_8015_agreement : RowAgreement 8014 8015 := RowAgreement.single (first := 8014) modelrow_agrees_8015
private def atomic_8016 : CountSegment 1 10 13120 8015 8016 3397 3397 := CountSegment.single (first := 8015) (initial := 3397) row_count_8016 (by decide)
private theorem atomic_8016_agreement : RowAgreement 8015 8016 := RowAgreement.single (first := 8015) modelrow_agrees_8016
private def atomic_8017 : CountSegment 1 10 13120 8016 8017 3397 3397 := CountSegment.single (first := 8016) (initial := 3397) row_count_8017 (by decide)
private theorem atomic_8017_agreement : RowAgreement 8016 8017 := RowAgreement.single (first := 8016) modelrow_agrees_8017
private def atomic_8018 : CountSegment 1 10 13120 8017 8018 3397 3398 := CountSegment.single (first := 8017) (initial := 3397) row_count_8018 (by decide)
private theorem atomic_8018_agreement : RowAgreement 8017 8018 := RowAgreement.single (first := 8017) modelrow_agrees_8018
private def atomic_8019 : CountSegment 1 10 13120 8018 8019 3398 3399 := CountSegment.single (first := 8018) (initial := 3398) row_count_8019 (by decide)
private theorem atomic_8019_agreement : RowAgreement 8018 8019 := RowAgreement.single (first := 8018) modelrow_agrees_8019
private def atomic_8020 : CountSegment 1 10 13120 8019 8020 3399 3399 := CountSegment.single (first := 8019) (initial := 3399) row_count_8020 (by decide)
private theorem atomic_8020_agreement : RowAgreement 8019 8020 := RowAgreement.single (first := 8019) modelrow_agrees_8020
private def atomic_8021 : CountSegment 1 10 13120 8020 8021 3399 3399 := CountSegment.single (first := 8020) (initial := 3399) row_count_8021 (by decide)
private theorem atomic_8021_agreement : RowAgreement 8020 8021 := RowAgreement.single (first := 8020) modelrow_agrees_8021
private def atomic_8022 : CountSegment 1 10 13120 8021 8022 3399 3400 := CountSegment.single (first := 8021) (initial := 3399) row_count_8022 (by decide)
private theorem atomic_8022_agreement : RowAgreement 8021 8022 := RowAgreement.single (first := 8021) modelrow_agrees_8022
private def atomic_8023 : CountSegment 1 10 13120 8022 8023 3400 3401 := CountSegment.single (first := 8022) (initial := 3400) row_count_8023 (by decide)
private theorem atomic_8023_agreement : RowAgreement 8022 8023 := RowAgreement.single (first := 8022) modelrow_agrees_8023
private def atomic_8024 : CountSegment 1 10 13120 8023 8024 3401 3401 := CountSegment.single (first := 8023) (initial := 3401) row_count_8024 (by decide)
private theorem atomic_8024_agreement : RowAgreement 8023 8024 := RowAgreement.single (first := 8023) modelrow_agrees_8024
private def atomic_8025 : CountSegment 1 10 13120 8024 8025 3401 3402 := CountSegment.single (first := 8024) (initial := 3401) row_count_8025 (by decide)
private theorem atomic_8025_agreement : RowAgreement 8024 8025 := RowAgreement.single (first := 8024) modelrow_agrees_8025
private def atomic_8026 : CountSegment 1 10 13120 8025 8026 3402 3402 := CountSegment.single (first := 8025) (initial := 3402) row_count_8026 (by decide)
private theorem atomic_8026_agreement : RowAgreement 8025 8026 := RowAgreement.single (first := 8025) modelrow_agrees_8026
private def atomic_8027 : CountSegment 1 10 13120 8026 8027 3402 3402 := CountSegment.single (first := 8026) (initial := 3402) row_count_8027 (by decide)
private theorem atomic_8027_agreement : RowAgreement 8026 8027 := RowAgreement.single (first := 8026) modelrow_agrees_8027
private def atomic_8028 : CountSegment 1 10 13120 8027 8028 3402 3402 := CountSegment.single (first := 8027) (initial := 3402) row_count_8028 (by decide)
private theorem atomic_8028_agreement : RowAgreement 8027 8028 := RowAgreement.single (first := 8027) modelrow_agrees_8028
private def atomic_8029 : CountSegment 1 10 13120 8028 8029 3402 3402 := CountSegment.single (first := 8028) (initial := 3402) row_count_8029 (by decide)
private theorem atomic_8029_agreement : RowAgreement 8028 8029 := RowAgreement.single (first := 8028) modelrow_agrees_8029
private def atomic_8030 : CountSegment 1 10 13120 8029 8030 3402 3403 := CountSegment.single (first := 8029) (initial := 3402) row_count_8030 (by decide)
private theorem atomic_8030_agreement : RowAgreement 8029 8030 := RowAgreement.single (first := 8029) modelrow_agrees_8030
private def atomic_8031 : CountSegment 1 10 13120 8030 8031 3403 3403 := CountSegment.single (first := 8030) (initial := 3403) row_count_8031 (by decide)
private theorem atomic_8031_agreement : RowAgreement 8030 8031 := RowAgreement.single (first := 8030) modelrow_agrees_8031
private def atomic_8032 : CountSegment 1 10 13120 8031 8032 3403 3403 := CountSegment.single (first := 8031) (initial := 3403) row_count_8032 (by decide)
private theorem atomic_8032_agreement : RowAgreement 8031 8032 := RowAgreement.single (first := 8031) modelrow_agrees_8032
private def atomic_8033 : CountSegment 1 10 13120 8032 8033 3403 3403 := CountSegment.single (first := 8032) (initial := 3403) row_count_8033 (by decide)
private theorem atomic_8033_agreement : RowAgreement 8032 8033 := RowAgreement.single (first := 8032) modelrow_agrees_8033
private def atomic_8034 : CountSegment 1 10 13120 8033 8034 3403 3403 := CountSegment.single (first := 8033) (initial := 3403) row_count_8034 (by decide)
private theorem atomic_8034_agreement : RowAgreement 8033 8034 := RowAgreement.single (first := 8033) modelrow_agrees_8034
private def atomic_8035 : CountSegment 1 10 13120 8034 8035 3403 3404 := CountSegment.single (first := 8034) (initial := 3403) row_count_8035 (by decide)
private theorem atomic_8035_agreement : RowAgreement 8034 8035 := RowAgreement.single (first := 8034) modelrow_agrees_8035
private def atomic_8036 : CountSegment 1 10 13120 8035 8036 3404 3404 := CountSegment.single (first := 8035) (initial := 3404) row_count_8036 (by decide)
private theorem atomic_8036_agreement : RowAgreement 8035 8036 := RowAgreement.single (first := 8035) modelrow_agrees_8036
private def atomic_8037 : CountSegment 1 10 13120 8036 8037 3404 3405 := CountSegment.single (first := 8036) (initial := 3404) row_count_8037 (by decide)
private theorem atomic_8037_agreement : RowAgreement 8036 8037 := RowAgreement.single (first := 8036) modelrow_agrees_8037
private def atomic_8038 : CountSegment 1 10 13120 8037 8038 3405 3405 := CountSegment.single (first := 8037) (initial := 3405) row_count_8038 (by decide)
private theorem atomic_8038_agreement : RowAgreement 8037 8038 := RowAgreement.single (first := 8037) modelrow_agrees_8038
private def atomic_8039 : CountSegment 1 10 13120 8038 8039 3405 3405 := CountSegment.single (first := 8038) (initial := 3405) row_count_8039 (by decide)
private theorem atomic_8039_agreement : RowAgreement 8038 8039 := RowAgreement.single (first := 8038) modelrow_agrees_8039
private def atomic_8040 : CountSegment 1 10 13120 8039 8040 3405 3405 := CountSegment.single (first := 8039) (initial := 3405) row_count_8040 (by decide)
private theorem atomic_8040_agreement : RowAgreement 8039 8040 := RowAgreement.single (first := 8039) modelrow_agrees_8040
private def atomic_8041 : CountSegment 1 10 13120 8040 8041 3405 3406 := CountSegment.single (first := 8040) (initial := 3405) row_count_8041 (by decide)
private theorem atomic_8041_agreement : RowAgreement 8040 8041 := RowAgreement.single (first := 8040) modelrow_agrees_8041
private def atomic_8042 : CountSegment 1 10 13120 8041 8042 3406 3407 := CountSegment.single (first := 8041) (initial := 3406) row_count_8042 (by decide)
private theorem atomic_8042_agreement : RowAgreement 8041 8042 := RowAgreement.single (first := 8041) modelrow_agrees_8042
private def atomic_8043 : CountSegment 1 10 13120 8042 8043 3407 3407 := CountSegment.single (first := 8042) (initial := 3407) row_count_8043 (by decide)
private theorem atomic_8043_agreement : RowAgreement 8042 8043 := RowAgreement.single (first := 8042) modelrow_agrees_8043
private def atomic_8044 : CountSegment 1 10 13120 8043 8044 3407 3407 := CountSegment.single (first := 8043) (initial := 3407) row_count_8044 (by decide)
private theorem atomic_8044_agreement : RowAgreement 8043 8044 := RowAgreement.single (first := 8043) modelrow_agrees_8044
private def atomic_8045 : CountSegment 1 10 13120 8044 8045 3407 3407 := CountSegment.single (first := 8044) (initial := 3407) row_count_8045 (by decide)
private theorem atomic_8045_agreement : RowAgreement 8044 8045 := RowAgreement.single (first := 8044) modelrow_agrees_8045
private def atomic_8046 : CountSegment 1 10 13120 8045 8046 3407 3407 := CountSegment.single (first := 8045) (initial := 3407) row_count_8046 (by decide)
private theorem atomic_8046_agreement : RowAgreement 8045 8046 := RowAgreement.single (first := 8045) modelrow_agrees_8046
private def atomic_8047 : CountSegment 1 10 13120 8046 8047 3407 3407 := CountSegment.single (first := 8046) (initial := 3407) row_count_8047 (by decide)
private theorem atomic_8047_agreement : RowAgreement 8046 8047 := RowAgreement.single (first := 8046) modelrow_agrees_8047
private def atomic_8048 : CountSegment 1 10 13120 8047 8048 3407 3407 := CountSegment.single (first := 8047) (initial := 3407) row_count_8048 (by decide)
private theorem atomic_8048_agreement : RowAgreement 8047 8048 := RowAgreement.single (first := 8047) modelrow_agrees_8048
private def atomic_8049 : CountSegment 1 10 13120 8048 8049 3407 3408 := CountSegment.single (first := 8048) (initial := 3407) row_count_8049 (by decide)
private theorem atomic_8049_agreement : RowAgreement 8048 8049 := RowAgreement.single (first := 8048) modelrow_agrees_8049
private def atomic_8050 : CountSegment 1 10 13120 8049 8050 3408 3408 := CountSegment.single (first := 8049) (initial := 3408) row_count_8050 (by decide)
private theorem atomic_8050_agreement : RowAgreement 8049 8050 := RowAgreement.single (first := 8049) modelrow_agrees_8050
private def atomic_8051 : CountSegment 1 10 13120 8050 8051 3408 3409 := CountSegment.single (first := 8050) (initial := 3408) row_count_8051 (by decide)
private theorem atomic_8051_agreement : RowAgreement 8050 8051 := RowAgreement.single (first := 8050) modelrow_agrees_8051
private def atomic_8052 : CountSegment 1 10 13120 8051 8052 3409 3409 := CountSegment.single (first := 8051) (initial := 3409) row_count_8052 (by decide)
private theorem atomic_8052_agreement : RowAgreement 8051 8052 := RowAgreement.single (first := 8051) modelrow_agrees_8052
private def atomic_8053 : CountSegment 1 10 13120 8052 8053 3409 3410 := CountSegment.single (first := 8052) (initial := 3409) row_count_8053 (by decide)
private theorem atomic_8053_agreement : RowAgreement 8052 8053 := RowAgreement.single (first := 8052) modelrow_agrees_8053
private def atomic_8054 : CountSegment 1 10 13120 8053 8054 3410 3410 := CountSegment.single (first := 8053) (initial := 3410) row_count_8054 (by decide)
private theorem atomic_8054_agreement : RowAgreement 8053 8054 := RowAgreement.single (first := 8053) modelrow_agrees_8054
private def atomic_8055 : CountSegment 1 10 13120 8054 8055 3410 3411 := CountSegment.single (first := 8054) (initial := 3410) row_count_8055 (by decide)
private theorem atomic_8055_agreement : RowAgreement 8054 8055 := RowAgreement.single (first := 8054) modelrow_agrees_8055
private def atomic_8056 : CountSegment 1 10 13120 8055 8056 3411 3412 := CountSegment.single (first := 8055) (initial := 3411) row_count_8056 (by decide)
private theorem atomic_8056_agreement : RowAgreement 8055 8056 := RowAgreement.single (first := 8055) modelrow_agrees_8056
private def atomic_8057 : CountSegment 1 10 13120 8056 8057 3412 3413 := CountSegment.single (first := 8056) (initial := 3412) row_count_8057 (by decide)
private theorem atomic_8057_agreement : RowAgreement 8056 8057 := RowAgreement.single (first := 8056) modelrow_agrees_8057
private def atomic_8058 : CountSegment 1 10 13120 8057 8058 3413 3415 := CountSegment.single (first := 8057) (initial := 3413) row_count_8058 (by decide)
private theorem atomic_8058_agreement : RowAgreement 8057 8058 := RowAgreement.single (first := 8057) modelrow_agrees_8058
private def atomic_8059 : CountSegment 1 10 13120 8058 8059 3415 3415 := CountSegment.single (first := 8058) (initial := 3415) row_count_8059 (by decide)
private theorem atomic_8059_agreement : RowAgreement 8058 8059 := RowAgreement.single (first := 8058) modelrow_agrees_8059
private def atomic_8060 : CountSegment 1 10 13120 8059 8060 3415 3415 := CountSegment.single (first := 8059) (initial := 3415) row_count_8060 (by decide)
private theorem atomic_8060_agreement : RowAgreement 8059 8060 := RowAgreement.single (first := 8059) modelrow_agrees_8060
private def atomic_8061 : CountSegment 1 10 13120 8060 8061 3415 3415 := CountSegment.single (first := 8060) (initial := 3415) row_count_8061 (by decide)
private theorem atomic_8061_agreement : RowAgreement 8060 8061 := RowAgreement.single (first := 8060) modelrow_agrees_8061
private def atomic_8062 : CountSegment 1 10 13120 8061 8062 3415 3415 := CountSegment.single (first := 8061) (initial := 3415) row_count_8062 (by decide)
private theorem atomic_8062_agreement : RowAgreement 8061 8062 := RowAgreement.single (first := 8061) modelrow_agrees_8062
private def atomic_8063 : CountSegment 1 10 13120 8062 8063 3415 3415 := CountSegment.single (first := 8062) (initial := 3415) row_count_8063 (by decide)
private theorem atomic_8063_agreement : RowAgreement 8062 8063 := RowAgreement.single (first := 8062) modelrow_agrees_8063
private def atomic_8064 : CountSegment 1 10 13120 8063 8064 3415 3415 := CountSegment.single (first := 8063) (initial := 3415) row_count_8064 (by decide)
private theorem atomic_8064_agreement : RowAgreement 8063 8064 := RowAgreement.single (first := 8063) modelrow_agrees_8064
private def within_062_000 : CountSegment 1 10 13120 7936 7938 3363 3364 := CountSegment.join atomic_7937 atomic_7938
private theorem within_062_000_agreement : RowAgreement 7936 7938 := RowAgreement.join atomic_7937_agreement atomic_7938_agreement
private def within_062_001 : CountSegment 1 10 13120 7938 7940 3364 3364 := CountSegment.join atomic_7939 atomic_7940
private theorem within_062_001_agreement : RowAgreement 7938 7940 := RowAgreement.join atomic_7939_agreement atomic_7940_agreement
private def within_062_002 : CountSegment 1 10 13120 7936 7940 3363 3364 := CountSegment.join within_062_000 within_062_001
private theorem within_062_002_agreement : RowAgreement 7936 7940 := RowAgreement.join within_062_000_agreement within_062_001_agreement
private def within_062_003 : CountSegment 1 10 13120 7940 7942 3364 3366 := CountSegment.join atomic_7941 atomic_7942
private theorem within_062_003_agreement : RowAgreement 7940 7942 := RowAgreement.join atomic_7941_agreement atomic_7942_agreement
private def within_062_004 : CountSegment 1 10 13120 7942 7944 3366 3366 := CountSegment.join atomic_7943 atomic_7944
private theorem within_062_004_agreement : RowAgreement 7942 7944 := RowAgreement.join atomic_7943_agreement atomic_7944_agreement
private def within_062_005 : CountSegment 1 10 13120 7940 7944 3364 3366 := CountSegment.join within_062_003 within_062_004
private theorem within_062_005_agreement : RowAgreement 7940 7944 := RowAgreement.join within_062_003_agreement within_062_004_agreement
private def within_062_006 : CountSegment 1 10 13120 7936 7944 3363 3366 := CountSegment.join within_062_002 within_062_005
private theorem within_062_006_agreement : RowAgreement 7936 7944 := RowAgreement.join within_062_002_agreement within_062_005_agreement
private def within_062_007 : CountSegment 1 10 13120 7944 7946 3366 3367 := CountSegment.join atomic_7945 atomic_7946
private theorem within_062_007_agreement : RowAgreement 7944 7946 := RowAgreement.join atomic_7945_agreement atomic_7946_agreement
private def within_062_008 : CountSegment 1 10 13120 7946 7948 3367 3369 := CountSegment.join atomic_7947 atomic_7948
private theorem within_062_008_agreement : RowAgreement 7946 7948 := RowAgreement.join atomic_7947_agreement atomic_7948_agreement
private def within_062_009 : CountSegment 1 10 13120 7944 7948 3366 3369 := CountSegment.join within_062_007 within_062_008
private theorem within_062_009_agreement : RowAgreement 7944 7948 := RowAgreement.join within_062_007_agreement within_062_008_agreement
private def within_062_010 : CountSegment 1 10 13120 7948 7950 3369 3370 := CountSegment.join atomic_7949 atomic_7950
private theorem within_062_010_agreement : RowAgreement 7948 7950 := RowAgreement.join atomic_7949_agreement atomic_7950_agreement
private def within_062_011 : CountSegment 1 10 13120 7950 7952 3370 3370 := CountSegment.join atomic_7951 atomic_7952
private theorem within_062_011_agreement : RowAgreement 7950 7952 := RowAgreement.join atomic_7951_agreement atomic_7952_agreement
private def within_062_012 : CountSegment 1 10 13120 7948 7952 3369 3370 := CountSegment.join within_062_010 within_062_011
private theorem within_062_012_agreement : RowAgreement 7948 7952 := RowAgreement.join within_062_010_agreement within_062_011_agreement
private def within_062_013 : CountSegment 1 10 13120 7944 7952 3366 3370 := CountSegment.join within_062_009 within_062_012
private theorem within_062_013_agreement : RowAgreement 7944 7952 := RowAgreement.join within_062_009_agreement within_062_012_agreement
private def within_062_014 : CountSegment 1 10 13120 7936 7952 3363 3370 := CountSegment.join within_062_006 within_062_013
private theorem within_062_014_agreement : RowAgreement 7936 7952 := RowAgreement.join within_062_006_agreement within_062_013_agreement
private def within_062_015 : CountSegment 1 10 13120 7952 7954 3370 3370 := CountSegment.join atomic_7953 atomic_7954
private theorem within_062_015_agreement : RowAgreement 7952 7954 := RowAgreement.join atomic_7953_agreement atomic_7954_agreement
private def within_062_016 : CountSegment 1 10 13120 7954 7956 3370 3371 := CountSegment.join atomic_7955 atomic_7956
private theorem within_062_016_agreement : RowAgreement 7954 7956 := RowAgreement.join atomic_7955_agreement atomic_7956_agreement
private def within_062_017 : CountSegment 1 10 13120 7952 7956 3370 3371 := CountSegment.join within_062_015 within_062_016
private theorem within_062_017_agreement : RowAgreement 7952 7956 := RowAgreement.join within_062_015_agreement within_062_016_agreement
private def within_062_018 : CountSegment 1 10 13120 7956 7958 3371 3372 := CountSegment.join atomic_7957 atomic_7958
private theorem within_062_018_agreement : RowAgreement 7956 7958 := RowAgreement.join atomic_7957_agreement atomic_7958_agreement
private def within_062_019 : CountSegment 1 10 13120 7958 7960 3372 3372 := CountSegment.join atomic_7959 atomic_7960
private theorem within_062_019_agreement : RowAgreement 7958 7960 := RowAgreement.join atomic_7959_agreement atomic_7960_agreement
private def within_062_020 : CountSegment 1 10 13120 7956 7960 3371 3372 := CountSegment.join within_062_018 within_062_019
private theorem within_062_020_agreement : RowAgreement 7956 7960 := RowAgreement.join within_062_018_agreement within_062_019_agreement
private def within_062_021 : CountSegment 1 10 13120 7952 7960 3370 3372 := CountSegment.join within_062_017 within_062_020
private theorem within_062_021_agreement : RowAgreement 7952 7960 := RowAgreement.join within_062_017_agreement within_062_020_agreement
private def within_062_022 : CountSegment 1 10 13120 7960 7962 3372 3374 := CountSegment.join atomic_7961 atomic_7962
private theorem within_062_022_agreement : RowAgreement 7960 7962 := RowAgreement.join atomic_7961_agreement atomic_7962_agreement
private def within_062_023 : CountSegment 1 10 13120 7962 7964 3374 3375 := CountSegment.join atomic_7963 atomic_7964
private theorem within_062_023_agreement : RowAgreement 7962 7964 := RowAgreement.join atomic_7963_agreement atomic_7964_agreement
private def within_062_024 : CountSegment 1 10 13120 7960 7964 3372 3375 := CountSegment.join within_062_022 within_062_023
private theorem within_062_024_agreement : RowAgreement 7960 7964 := RowAgreement.join within_062_022_agreement within_062_023_agreement
private def within_062_025 : CountSegment 1 10 13120 7964 7966 3375 3375 := CountSegment.join atomic_7965 atomic_7966
private theorem within_062_025_agreement : RowAgreement 7964 7966 := RowAgreement.join atomic_7965_agreement atomic_7966_agreement
private def within_062_026 : CountSegment 1 10 13120 7966 7968 3375 3376 := CountSegment.join atomic_7967 atomic_7968
private theorem within_062_026_agreement : RowAgreement 7966 7968 := RowAgreement.join atomic_7967_agreement atomic_7968_agreement
private def within_062_027 : CountSegment 1 10 13120 7964 7968 3375 3376 := CountSegment.join within_062_025 within_062_026
private theorem within_062_027_agreement : RowAgreement 7964 7968 := RowAgreement.join within_062_025_agreement within_062_026_agreement
private def within_062_028 : CountSegment 1 10 13120 7960 7968 3372 3376 := CountSegment.join within_062_024 within_062_027
private theorem within_062_028_agreement : RowAgreement 7960 7968 := RowAgreement.join within_062_024_agreement within_062_027_agreement
private def within_062_029 : CountSegment 1 10 13120 7952 7968 3370 3376 := CountSegment.join within_062_021 within_062_028
private theorem within_062_029_agreement : RowAgreement 7952 7968 := RowAgreement.join within_062_021_agreement within_062_028_agreement
private def within_062_030 : CountSegment 1 10 13120 7936 7968 3363 3376 := CountSegment.join within_062_014 within_062_029
private theorem within_062_030_agreement : RowAgreement 7936 7968 := RowAgreement.join within_062_014_agreement within_062_029_agreement
private def within_062_031 : CountSegment 1 10 13120 7968 7970 3376 3376 := CountSegment.join atomic_7969 atomic_7970
private theorem within_062_031_agreement : RowAgreement 7968 7970 := RowAgreement.join atomic_7969_agreement atomic_7970_agreement
private def within_062_032 : CountSegment 1 10 13120 7970 7972 3376 3378 := CountSegment.join atomic_7971 atomic_7972
private theorem within_062_032_agreement : RowAgreement 7970 7972 := RowAgreement.join atomic_7971_agreement atomic_7972_agreement
private def within_062_033 : CountSegment 1 10 13120 7968 7972 3376 3378 := CountSegment.join within_062_031 within_062_032
private theorem within_062_033_agreement : RowAgreement 7968 7972 := RowAgreement.join within_062_031_agreement within_062_032_agreement
private def within_062_034 : CountSegment 1 10 13120 7972 7974 3378 3378 := CountSegment.join atomic_7973 atomic_7974
private theorem within_062_034_agreement : RowAgreement 7972 7974 := RowAgreement.join atomic_7973_agreement atomic_7974_agreement
private def within_062_035 : CountSegment 1 10 13120 7974 7976 3378 3378 := CountSegment.join atomic_7975 atomic_7976
private theorem within_062_035_agreement : RowAgreement 7974 7976 := RowAgreement.join atomic_7975_agreement atomic_7976_agreement
private def within_062_036 : CountSegment 1 10 13120 7972 7976 3378 3378 := CountSegment.join within_062_034 within_062_035
private theorem within_062_036_agreement : RowAgreement 7972 7976 := RowAgreement.join within_062_034_agreement within_062_035_agreement
private def within_062_037 : CountSegment 1 10 13120 7968 7976 3376 3378 := CountSegment.join within_062_033 within_062_036
private theorem within_062_037_agreement : RowAgreement 7968 7976 := RowAgreement.join within_062_033_agreement within_062_036_agreement
private def within_062_038 : CountSegment 1 10 13120 7976 7978 3378 3380 := CountSegment.join atomic_7977 atomic_7978
private theorem within_062_038_agreement : RowAgreement 7976 7978 := RowAgreement.join atomic_7977_agreement atomic_7978_agreement
private def within_062_039 : CountSegment 1 10 13120 7978 7980 3380 3380 := CountSegment.join atomic_7979 atomic_7980
private theorem within_062_039_agreement : RowAgreement 7978 7980 := RowAgreement.join atomic_7979_agreement atomic_7980_agreement
private def within_062_040 : CountSegment 1 10 13120 7976 7980 3378 3380 := CountSegment.join within_062_038 within_062_039
private theorem within_062_040_agreement : RowAgreement 7976 7980 := RowAgreement.join within_062_038_agreement within_062_039_agreement
private def within_062_041 : CountSegment 1 10 13120 7980 7982 3380 3381 := CountSegment.join atomic_7981 atomic_7982
private theorem within_062_041_agreement : RowAgreement 7980 7982 := RowAgreement.join atomic_7981_agreement atomic_7982_agreement
private def within_062_042 : CountSegment 1 10 13120 7982 7984 3381 3382 := CountSegment.join atomic_7983 atomic_7984
private theorem within_062_042_agreement : RowAgreement 7982 7984 := RowAgreement.join atomic_7983_agreement atomic_7984_agreement
private def within_062_043 : CountSegment 1 10 13120 7980 7984 3380 3382 := CountSegment.join within_062_041 within_062_042
private theorem within_062_043_agreement : RowAgreement 7980 7984 := RowAgreement.join within_062_041_agreement within_062_042_agreement
private def within_062_044 : CountSegment 1 10 13120 7976 7984 3378 3382 := CountSegment.join within_062_040 within_062_043
private theorem within_062_044_agreement : RowAgreement 7976 7984 := RowAgreement.join within_062_040_agreement within_062_043_agreement
private def within_062_045 : CountSegment 1 10 13120 7968 7984 3376 3382 := CountSegment.join within_062_037 within_062_044
private theorem within_062_045_agreement : RowAgreement 7968 7984 := RowAgreement.join within_062_037_agreement within_062_044_agreement
private def within_062_046 : CountSegment 1 10 13120 7984 7986 3382 3383 := CountSegment.join atomic_7985 atomic_7986
private theorem within_062_046_agreement : RowAgreement 7984 7986 := RowAgreement.join atomic_7985_agreement atomic_7986_agreement
private def within_062_047 : CountSegment 1 10 13120 7986 7988 3383 3384 := CountSegment.join atomic_7987 atomic_7988
private theorem within_062_047_agreement : RowAgreement 7986 7988 := RowAgreement.join atomic_7987_agreement atomic_7988_agreement
private def within_062_048 : CountSegment 1 10 13120 7984 7988 3382 3384 := CountSegment.join within_062_046 within_062_047
private theorem within_062_048_agreement : RowAgreement 7984 7988 := RowAgreement.join within_062_046_agreement within_062_047_agreement
private def within_062_049 : CountSegment 1 10 13120 7988 7990 3384 3385 := CountSegment.join atomic_7989 atomic_7990
private theorem within_062_049_agreement : RowAgreement 7988 7990 := RowAgreement.join atomic_7989_agreement atomic_7990_agreement
private def within_062_050 : CountSegment 1 10 13120 7990 7992 3385 3386 := CountSegment.join atomic_7991 atomic_7992
private theorem within_062_050_agreement : RowAgreement 7990 7992 := RowAgreement.join atomic_7991_agreement atomic_7992_agreement
private def within_062_051 : CountSegment 1 10 13120 7988 7992 3384 3386 := CountSegment.join within_062_049 within_062_050
private theorem within_062_051_agreement : RowAgreement 7988 7992 := RowAgreement.join within_062_049_agreement within_062_050_agreement
private def within_062_052 : CountSegment 1 10 13120 7984 7992 3382 3386 := CountSegment.join within_062_048 within_062_051
private theorem within_062_052_agreement : RowAgreement 7984 7992 := RowAgreement.join within_062_048_agreement within_062_051_agreement
private def within_062_053 : CountSegment 1 10 13120 7992 7994 3386 3387 := CountSegment.join atomic_7993 atomic_7994
private theorem within_062_053_agreement : RowAgreement 7992 7994 := RowAgreement.join atomic_7993_agreement atomic_7994_agreement
private def within_062_054 : CountSegment 1 10 13120 7994 7996 3387 3387 := CountSegment.join atomic_7995 atomic_7996
private theorem within_062_054_agreement : RowAgreement 7994 7996 := RowAgreement.join atomic_7995_agreement atomic_7996_agreement
private def within_062_055 : CountSegment 1 10 13120 7992 7996 3386 3387 := CountSegment.join within_062_053 within_062_054
private theorem within_062_055_agreement : RowAgreement 7992 7996 := RowAgreement.join within_062_053_agreement within_062_054_agreement
private def within_062_056 : CountSegment 1 10 13120 7996 7998 3387 3388 := CountSegment.join atomic_7997 atomic_7998
private theorem within_062_056_agreement : RowAgreement 7996 7998 := RowAgreement.join atomic_7997_agreement atomic_7998_agreement
private def within_062_057 : CountSegment 1 10 13120 7998 8000 3388 3390 := CountSegment.join atomic_7999 atomic_8000
private theorem within_062_057_agreement : RowAgreement 7998 8000 := RowAgreement.join atomic_7999_agreement atomic_8000_agreement
private def within_062_058 : CountSegment 1 10 13120 7996 8000 3387 3390 := CountSegment.join within_062_056 within_062_057
private theorem within_062_058_agreement : RowAgreement 7996 8000 := RowAgreement.join within_062_056_agreement within_062_057_agreement
private def within_062_059 : CountSegment 1 10 13120 7992 8000 3386 3390 := CountSegment.join within_062_055 within_062_058
private theorem within_062_059_agreement : RowAgreement 7992 8000 := RowAgreement.join within_062_055_agreement within_062_058_agreement
private def within_062_060 : CountSegment 1 10 13120 7984 8000 3382 3390 := CountSegment.join within_062_052 within_062_059
private theorem within_062_060_agreement : RowAgreement 7984 8000 := RowAgreement.join within_062_052_agreement within_062_059_agreement
private def within_062_061 : CountSegment 1 10 13120 7968 8000 3376 3390 := CountSegment.join within_062_045 within_062_060
private theorem within_062_061_agreement : RowAgreement 7968 8000 := RowAgreement.join within_062_045_agreement within_062_060_agreement
private def within_062_062 : CountSegment 1 10 13120 7936 8000 3363 3390 := CountSegment.join within_062_030 within_062_061
private theorem within_062_062_agreement : RowAgreement 7936 8000 := RowAgreement.join within_062_030_agreement within_062_061_agreement
private def within_062_063 : CountSegment 1 10 13120 8000 8002 3390 3391 := CountSegment.join atomic_8001 atomic_8002
private theorem within_062_063_agreement : RowAgreement 8000 8002 := RowAgreement.join atomic_8001_agreement atomic_8002_agreement
private def within_062_064 : CountSegment 1 10 13120 8002 8004 3391 3393 := CountSegment.join atomic_8003 atomic_8004
private theorem within_062_064_agreement : RowAgreement 8002 8004 := RowAgreement.join atomic_8003_agreement atomic_8004_agreement
private def within_062_065 : CountSegment 1 10 13120 8000 8004 3390 3393 := CountSegment.join within_062_063 within_062_064
private theorem within_062_065_agreement : RowAgreement 8000 8004 := RowAgreement.join within_062_063_agreement within_062_064_agreement
private def within_062_066 : CountSegment 1 10 13120 8004 8006 3393 3393 := CountSegment.join atomic_8005 atomic_8006
private theorem within_062_066_agreement : RowAgreement 8004 8006 := RowAgreement.join atomic_8005_agreement atomic_8006_agreement
private def within_062_067 : CountSegment 1 10 13120 8006 8008 3393 3394 := CountSegment.join atomic_8007 atomic_8008
private theorem within_062_067_agreement : RowAgreement 8006 8008 := RowAgreement.join atomic_8007_agreement atomic_8008_agreement
private def within_062_068 : CountSegment 1 10 13120 8004 8008 3393 3394 := CountSegment.join within_062_066 within_062_067
private theorem within_062_068_agreement : RowAgreement 8004 8008 := RowAgreement.join within_062_066_agreement within_062_067_agreement
private def within_062_069 : CountSegment 1 10 13120 8000 8008 3390 3394 := CountSegment.join within_062_065 within_062_068
private theorem within_062_069_agreement : RowAgreement 8000 8008 := RowAgreement.join within_062_065_agreement within_062_068_agreement
private def within_062_070 : CountSegment 1 10 13120 8008 8010 3394 3394 := CountSegment.join atomic_8009 atomic_8010
private theorem within_062_070_agreement : RowAgreement 8008 8010 := RowAgreement.join atomic_8009_agreement atomic_8010_agreement
private def within_062_071 : CountSegment 1 10 13120 8010 8012 3394 3395 := CountSegment.join atomic_8011 atomic_8012
private theorem within_062_071_agreement : RowAgreement 8010 8012 := RowAgreement.join atomic_8011_agreement atomic_8012_agreement
private def within_062_072 : CountSegment 1 10 13120 8008 8012 3394 3395 := CountSegment.join within_062_070 within_062_071
private theorem within_062_072_agreement : RowAgreement 8008 8012 := RowAgreement.join within_062_070_agreement within_062_071_agreement
private def within_062_073 : CountSegment 1 10 13120 8012 8014 3395 3396 := CountSegment.join atomic_8013 atomic_8014
private theorem within_062_073_agreement : RowAgreement 8012 8014 := RowAgreement.join atomic_8013_agreement atomic_8014_agreement
private def within_062_074 : CountSegment 1 10 13120 8014 8016 3396 3397 := CountSegment.join atomic_8015 atomic_8016
private theorem within_062_074_agreement : RowAgreement 8014 8016 := RowAgreement.join atomic_8015_agreement atomic_8016_agreement
private def within_062_075 : CountSegment 1 10 13120 8012 8016 3395 3397 := CountSegment.join within_062_073 within_062_074
private theorem within_062_075_agreement : RowAgreement 8012 8016 := RowAgreement.join within_062_073_agreement within_062_074_agreement
private def within_062_076 : CountSegment 1 10 13120 8008 8016 3394 3397 := CountSegment.join within_062_072 within_062_075
private theorem within_062_076_agreement : RowAgreement 8008 8016 := RowAgreement.join within_062_072_agreement within_062_075_agreement
private def within_062_077 : CountSegment 1 10 13120 8000 8016 3390 3397 := CountSegment.join within_062_069 within_062_076
private theorem within_062_077_agreement : RowAgreement 8000 8016 := RowAgreement.join within_062_069_agreement within_062_076_agreement
private def within_062_078 : CountSegment 1 10 13120 8016 8018 3397 3398 := CountSegment.join atomic_8017 atomic_8018
private theorem within_062_078_agreement : RowAgreement 8016 8018 := RowAgreement.join atomic_8017_agreement atomic_8018_agreement
private def within_062_079 : CountSegment 1 10 13120 8018 8020 3398 3399 := CountSegment.join atomic_8019 atomic_8020
private theorem within_062_079_agreement : RowAgreement 8018 8020 := RowAgreement.join atomic_8019_agreement atomic_8020_agreement
private def within_062_080 : CountSegment 1 10 13120 8016 8020 3397 3399 := CountSegment.join within_062_078 within_062_079
private theorem within_062_080_agreement : RowAgreement 8016 8020 := RowAgreement.join within_062_078_agreement within_062_079_agreement
private def within_062_081 : CountSegment 1 10 13120 8020 8022 3399 3400 := CountSegment.join atomic_8021 atomic_8022
private theorem within_062_081_agreement : RowAgreement 8020 8022 := RowAgreement.join atomic_8021_agreement atomic_8022_agreement
private def within_062_082 : CountSegment 1 10 13120 8022 8024 3400 3401 := CountSegment.join atomic_8023 atomic_8024
private theorem within_062_082_agreement : RowAgreement 8022 8024 := RowAgreement.join atomic_8023_agreement atomic_8024_agreement
private def within_062_083 : CountSegment 1 10 13120 8020 8024 3399 3401 := CountSegment.join within_062_081 within_062_082
private theorem within_062_083_agreement : RowAgreement 8020 8024 := RowAgreement.join within_062_081_agreement within_062_082_agreement
private def within_062_084 : CountSegment 1 10 13120 8016 8024 3397 3401 := CountSegment.join within_062_080 within_062_083
private theorem within_062_084_agreement : RowAgreement 8016 8024 := RowAgreement.join within_062_080_agreement within_062_083_agreement
private def within_062_085 : CountSegment 1 10 13120 8024 8026 3401 3402 := CountSegment.join atomic_8025 atomic_8026
private theorem within_062_085_agreement : RowAgreement 8024 8026 := RowAgreement.join atomic_8025_agreement atomic_8026_agreement
private def within_062_086 : CountSegment 1 10 13120 8026 8028 3402 3402 := CountSegment.join atomic_8027 atomic_8028
private theorem within_062_086_agreement : RowAgreement 8026 8028 := RowAgreement.join atomic_8027_agreement atomic_8028_agreement
private def within_062_087 : CountSegment 1 10 13120 8024 8028 3401 3402 := CountSegment.join within_062_085 within_062_086
private theorem within_062_087_agreement : RowAgreement 8024 8028 := RowAgreement.join within_062_085_agreement within_062_086_agreement
private def within_062_088 : CountSegment 1 10 13120 8028 8030 3402 3403 := CountSegment.join atomic_8029 atomic_8030
private theorem within_062_088_agreement : RowAgreement 8028 8030 := RowAgreement.join atomic_8029_agreement atomic_8030_agreement
private def within_062_089 : CountSegment 1 10 13120 8030 8032 3403 3403 := CountSegment.join atomic_8031 atomic_8032
private theorem within_062_089_agreement : RowAgreement 8030 8032 := RowAgreement.join atomic_8031_agreement atomic_8032_agreement
private def within_062_090 : CountSegment 1 10 13120 8028 8032 3402 3403 := CountSegment.join within_062_088 within_062_089
private theorem within_062_090_agreement : RowAgreement 8028 8032 := RowAgreement.join within_062_088_agreement within_062_089_agreement
private def within_062_091 : CountSegment 1 10 13120 8024 8032 3401 3403 := CountSegment.join within_062_087 within_062_090
private theorem within_062_091_agreement : RowAgreement 8024 8032 := RowAgreement.join within_062_087_agreement within_062_090_agreement
private def within_062_092 : CountSegment 1 10 13120 8016 8032 3397 3403 := CountSegment.join within_062_084 within_062_091
private theorem within_062_092_agreement : RowAgreement 8016 8032 := RowAgreement.join within_062_084_agreement within_062_091_agreement
private def within_062_093 : CountSegment 1 10 13120 8000 8032 3390 3403 := CountSegment.join within_062_077 within_062_092
private theorem within_062_093_agreement : RowAgreement 8000 8032 := RowAgreement.join within_062_077_agreement within_062_092_agreement
private def within_062_094 : CountSegment 1 10 13120 8032 8034 3403 3403 := CountSegment.join atomic_8033 atomic_8034
private theorem within_062_094_agreement : RowAgreement 8032 8034 := RowAgreement.join atomic_8033_agreement atomic_8034_agreement
private def within_062_095 : CountSegment 1 10 13120 8034 8036 3403 3404 := CountSegment.join atomic_8035 atomic_8036
private theorem within_062_095_agreement : RowAgreement 8034 8036 := RowAgreement.join atomic_8035_agreement atomic_8036_agreement
private def within_062_096 : CountSegment 1 10 13120 8032 8036 3403 3404 := CountSegment.join within_062_094 within_062_095
private theorem within_062_096_agreement : RowAgreement 8032 8036 := RowAgreement.join within_062_094_agreement within_062_095_agreement
private def within_062_097 : CountSegment 1 10 13120 8036 8038 3404 3405 := CountSegment.join atomic_8037 atomic_8038
private theorem within_062_097_agreement : RowAgreement 8036 8038 := RowAgreement.join atomic_8037_agreement atomic_8038_agreement
private def within_062_098 : CountSegment 1 10 13120 8038 8040 3405 3405 := CountSegment.join atomic_8039 atomic_8040
private theorem within_062_098_agreement : RowAgreement 8038 8040 := RowAgreement.join atomic_8039_agreement atomic_8040_agreement
private def within_062_099 : CountSegment 1 10 13120 8036 8040 3404 3405 := CountSegment.join within_062_097 within_062_098
private theorem within_062_099_agreement : RowAgreement 8036 8040 := RowAgreement.join within_062_097_agreement within_062_098_agreement
private def within_062_100 : CountSegment 1 10 13120 8032 8040 3403 3405 := CountSegment.join within_062_096 within_062_099
private theorem within_062_100_agreement : RowAgreement 8032 8040 := RowAgreement.join within_062_096_agreement within_062_099_agreement
private def within_062_101 : CountSegment 1 10 13120 8040 8042 3405 3407 := CountSegment.join atomic_8041 atomic_8042
private theorem within_062_101_agreement : RowAgreement 8040 8042 := RowAgreement.join atomic_8041_agreement atomic_8042_agreement
private def within_062_102 : CountSegment 1 10 13120 8042 8044 3407 3407 := CountSegment.join atomic_8043 atomic_8044
private theorem within_062_102_agreement : RowAgreement 8042 8044 := RowAgreement.join atomic_8043_agreement atomic_8044_agreement
private def within_062_103 : CountSegment 1 10 13120 8040 8044 3405 3407 := CountSegment.join within_062_101 within_062_102
private theorem within_062_103_agreement : RowAgreement 8040 8044 := RowAgreement.join within_062_101_agreement within_062_102_agreement
private def within_062_104 : CountSegment 1 10 13120 8044 8046 3407 3407 := CountSegment.join atomic_8045 atomic_8046
private theorem within_062_104_agreement : RowAgreement 8044 8046 := RowAgreement.join atomic_8045_agreement atomic_8046_agreement
private def within_062_105 : CountSegment 1 10 13120 8046 8048 3407 3407 := CountSegment.join atomic_8047 atomic_8048
private theorem within_062_105_agreement : RowAgreement 8046 8048 := RowAgreement.join atomic_8047_agreement atomic_8048_agreement
private def within_062_106 : CountSegment 1 10 13120 8044 8048 3407 3407 := CountSegment.join within_062_104 within_062_105
private theorem within_062_106_agreement : RowAgreement 8044 8048 := RowAgreement.join within_062_104_agreement within_062_105_agreement
private def within_062_107 : CountSegment 1 10 13120 8040 8048 3405 3407 := CountSegment.join within_062_103 within_062_106
private theorem within_062_107_agreement : RowAgreement 8040 8048 := RowAgreement.join within_062_103_agreement within_062_106_agreement
private def within_062_108 : CountSegment 1 10 13120 8032 8048 3403 3407 := CountSegment.join within_062_100 within_062_107
private theorem within_062_108_agreement : RowAgreement 8032 8048 := RowAgreement.join within_062_100_agreement within_062_107_agreement
private def within_062_109 : CountSegment 1 10 13120 8048 8050 3407 3408 := CountSegment.join atomic_8049 atomic_8050
private theorem within_062_109_agreement : RowAgreement 8048 8050 := RowAgreement.join atomic_8049_agreement atomic_8050_agreement
private def within_062_110 : CountSegment 1 10 13120 8050 8052 3408 3409 := CountSegment.join atomic_8051 atomic_8052
private theorem within_062_110_agreement : RowAgreement 8050 8052 := RowAgreement.join atomic_8051_agreement atomic_8052_agreement
private def within_062_111 : CountSegment 1 10 13120 8048 8052 3407 3409 := CountSegment.join within_062_109 within_062_110
private theorem within_062_111_agreement : RowAgreement 8048 8052 := RowAgreement.join within_062_109_agreement within_062_110_agreement
private def within_062_112 : CountSegment 1 10 13120 8052 8054 3409 3410 := CountSegment.join atomic_8053 atomic_8054
private theorem within_062_112_agreement : RowAgreement 8052 8054 := RowAgreement.join atomic_8053_agreement atomic_8054_agreement
private def within_062_113 : CountSegment 1 10 13120 8054 8056 3410 3412 := CountSegment.join atomic_8055 atomic_8056
private theorem within_062_113_agreement : RowAgreement 8054 8056 := RowAgreement.join atomic_8055_agreement atomic_8056_agreement
private def within_062_114 : CountSegment 1 10 13120 8052 8056 3409 3412 := CountSegment.join within_062_112 within_062_113
private theorem within_062_114_agreement : RowAgreement 8052 8056 := RowAgreement.join within_062_112_agreement within_062_113_agreement
private def within_062_115 : CountSegment 1 10 13120 8048 8056 3407 3412 := CountSegment.join within_062_111 within_062_114
private theorem within_062_115_agreement : RowAgreement 8048 8056 := RowAgreement.join within_062_111_agreement within_062_114_agreement
private def within_062_116 : CountSegment 1 10 13120 8056 8058 3412 3415 := CountSegment.join atomic_8057 atomic_8058
private theorem within_062_116_agreement : RowAgreement 8056 8058 := RowAgreement.join atomic_8057_agreement atomic_8058_agreement
private def within_062_117 : CountSegment 1 10 13120 8058 8060 3415 3415 := CountSegment.join atomic_8059 atomic_8060
private theorem within_062_117_agreement : RowAgreement 8058 8060 := RowAgreement.join atomic_8059_agreement atomic_8060_agreement
private def within_062_118 : CountSegment 1 10 13120 8056 8060 3412 3415 := CountSegment.join within_062_116 within_062_117
private theorem within_062_118_agreement : RowAgreement 8056 8060 := RowAgreement.join within_062_116_agreement within_062_117_agreement
private def within_062_119 : CountSegment 1 10 13120 8060 8062 3415 3415 := CountSegment.join atomic_8061 atomic_8062
private theorem within_062_119_agreement : RowAgreement 8060 8062 := RowAgreement.join atomic_8061_agreement atomic_8062_agreement
private def within_062_120 : CountSegment 1 10 13120 8062 8064 3415 3415 := CountSegment.join atomic_8063 atomic_8064
private theorem within_062_120_agreement : RowAgreement 8062 8064 := RowAgreement.join atomic_8063_agreement atomic_8064_agreement
private def within_062_121 : CountSegment 1 10 13120 8060 8064 3415 3415 := CountSegment.join within_062_119 within_062_120
private theorem within_062_121_agreement : RowAgreement 8060 8064 := RowAgreement.join within_062_119_agreement within_062_120_agreement
private def within_062_122 : CountSegment 1 10 13120 8056 8064 3412 3415 := CountSegment.join within_062_118 within_062_121
private theorem within_062_122_agreement : RowAgreement 8056 8064 := RowAgreement.join within_062_118_agreement within_062_121_agreement
private def within_062_123 : CountSegment 1 10 13120 8048 8064 3407 3415 := CountSegment.join within_062_115 within_062_122
private theorem within_062_123_agreement : RowAgreement 8048 8064 := RowAgreement.join within_062_115_agreement within_062_122_agreement
private def within_062_124 : CountSegment 1 10 13120 8032 8064 3403 3415 := CountSegment.join within_062_108 within_062_123
private theorem within_062_124_agreement : RowAgreement 8032 8064 := RowAgreement.join within_062_108_agreement within_062_123_agreement
private def within_062_125 : CountSegment 1 10 13120 8000 8064 3390 3415 := CountSegment.join within_062_093 within_062_124
private theorem within_062_125_agreement : RowAgreement 8000 8064 := RowAgreement.join within_062_093_agreement within_062_124_agreement
private def within_062_126 : CountSegment 1 10 13120 7936 8064 3363 3415 := CountSegment.join within_062_062 within_062_125
private theorem within_062_126_agreement : RowAgreement 7936 8064 := RowAgreement.join within_062_062_agreement within_062_125_agreement
def segment_062 : CountSegment 1 10 13120 7936 8064 3363 3415 := within_062_126
theorem agreement_062 : RowAgreement 7936 8064 := within_062_126_agreement
end ProjectEulerSolutions.P58Numerical
