import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_7809 : RowCertificate 7809 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7809 1) := by change ¬ Nat.Prime 243937543; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7868953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7809 2) := by change ¬ Nat.Prime 243921925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48784385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7809 3) := by change Nat.Prime 243906307; exact ProjectEulerSolutions.P58PrimeNodes.p_243906307.prime
    exact decide_eq_true prime
theorem row_count_7809 : row 7809 = 1 := (row_7809.checked).trans rfl
theorem model_count_7809 : modelRow 7809 = 1 := by decide
theorem modelrow_agrees_7809 : modelRow 7809 = row 7809 := model_count_7809.trans row_count_7809.symm
def row_7810 : RowCertificate 7810 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7810 1) := by change ¬ Nat.Prime 244000021; exact PrattCertificates.composite_of_product (factor := 7507) (cofactor := 32503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7810 2) := by change ¬ Nat.Prime 243984401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6594173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7810 3) := by change ¬ Nat.Prime 243968781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81322927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7810 : row 7810 = 0 := (row_7810.checked).trans rfl
theorem model_count_7810 : modelRow 7810 = 0 := by decide
theorem modelrow_agrees_7810 : modelRow 7810 = row 7810 := model_count_7810.trans row_count_7810.symm
def row_7811 : RowCertificate 7811 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7811 1) := by change ¬ Nat.Prime 244062507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81354169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7811 2) := by change ¬ Nat.Prime 244046885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48809377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7811 3) := by change ¬ Nat.Prime 244031263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34861609) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7811 : row 7811 = 0 := (row_7811.checked).trans rfl
theorem model_count_7811 : modelRow 7811 = 0 := by decide
theorem modelrow_agrees_7811 : modelRow 7811 = row 7811 := model_count_7811.trans row_count_7811.symm
def row_7812 : RowCertificate 7812 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7812 1) := by change ¬ Nat.Prime 244125001; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6597973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7812 2) := by change Nat.Prime 244109377; exact ProjectEulerSolutions.P58PrimeNodes.p_244109377.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7812 3) := by change Nat.Prime 244093753; exact ProjectEulerSolutions.P58PrimeNodes.p_244093753.prime
    exact decide_eq_true prime
theorem row_count_7812 : row 7812 = 2 := (row_7812.checked).trans rfl
theorem model_count_7812 : modelRow 7812 = 2 := by decide
theorem modelrow_agrees_7812 : modelRow 7812 = row 7812 := model_count_7812.trans row_count_7812.symm
def row_7813 : RowCertificate 7813 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7813 1) := by change ¬ Nat.Prime 244187503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34883929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7813 2) := by change ¬ Nat.Prime 244171877; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 529657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7813 3) := by change ¬ Nat.Prime 244156251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81385417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7813 : row 7813 = 0 := (row_7813.checked).trans rfl
theorem model_count_7813 : modelRow 7813 = 0 := by decide
theorem modelrow_agrees_7813 : modelRow 7813 = row 7813 := model_count_7813.trans row_count_7813.symm
def row_7814 : RowCertificate 7814 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7814 1) := by change ¬ Nat.Prime 244250013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81416671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7814 2) := by change ¬ Nat.Prime 244234385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48846877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7814 3) := by change Nat.Prime 244218757; exact ProjectEulerSolutions.P58PrimeNodes.p_244218757.prime
    exact decide_eq_true prime
theorem row_count_7814 : row 7814 = 1 := (row_7814.checked).trans rfl
theorem model_count_7814 : modelRow 7814 = 1 := by decide
theorem modelrow_agrees_7814 : modelRow 7814 = row 7814 := model_count_7814.trans row_count_7814.symm
def row_7815 : RowCertificate 7815 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7815 1) := by change ¬ Nat.Prime 244312531; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3346747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7815 2) := by change ¬ Nat.Prime 244296901; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 5958461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7815 3) := by change ¬ Nat.Prime 244281271; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18790867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7815 : row 7815 = 0 := (row_7815.checked).trans rfl
theorem model_count_7815 : modelRow 7815 = 0 := by decide
theorem modelrow_agrees_7815 : modelRow 7815 = row 7815 := model_count_7815.trans row_count_7815.symm
def row_7816 : RowCertificate 7816 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7816 1) := by change ¬ Nat.Prime 244375057; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2241973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7816 2) := by change ¬ Nat.Prime 244359425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48871885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7816 3) := by change ¬ Nat.Prime 244343793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81447931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7816 : row 7816 = 0 := (row_7816.checked).trans rfl
theorem model_count_7816 : modelRow 7816 = 0 := by decide
theorem modelrow_agrees_7816 : modelRow 7816 = row 7816 := model_count_7816.trans row_count_7816.symm
def row_7817 : RowCertificate 7817 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7817 1) := by change ¬ Nat.Prime 244437591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81479197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7817 2) := by change ¬ Nat.Prime 244421957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18801689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7817 3) := by change ¬ Nat.Prime 244406323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34915189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7817 : row 7817 = 0 := (row_7817.checked).trans rfl
theorem model_count_7817 : modelRow 7817 = 0 := by decide
theorem modelrow_agrees_7817 : modelRow 7817 = row 7817 := model_count_7817.trans row_count_7817.symm
def row_7818 : RowCertificate 7818 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7818 1) := by change Nat.Prime 244500133; exact ProjectEulerSolutions.P58PrimeNodes.p_244500133.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7818 2) := by change ¬ Nat.Prime 244484497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14381441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7818 3) := by change ¬ Nat.Prime 244468861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34924123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7818 : row 7818 = 1 := (row_7818.checked).trans rfl
theorem model_count_7818 : modelRow 7818 = 1 := by decide
theorem modelrow_agrees_7818 : modelRow 7818 = row 7818 := model_count_7818.trans row_count_7818.symm
def row_7819 : RowCertificate 7819 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7819 1) := by change Nat.Prime 244562683; exact ProjectEulerSolutions.P58PrimeNodes.p_244562683.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7819 2) := by change ¬ Nat.Prime 244547045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48909409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7819 3) := by change ¬ Nat.Prime 244531407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81510469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7819 : row 7819 = 1 := (row_7819.checked).trans rfl
theorem model_count_7819 : modelRow 7819 = 1 := by decide
theorem modelrow_agrees_7819 : modelRow 7819 = row 7819 := model_count_7819.trans row_count_7819.symm
def row_7820 : RowCertificate 7820 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7820 1) := by change ¬ Nat.Prime 244625241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81541747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7820 2) := by change Nat.Prime 244609601; exact ProjectEulerSolutions.P58PrimeNodes.p_244609601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7820 3) := by change ¬ Nat.Prime 244593961; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 796723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7820 : row 7820 = 1 := (row_7820.checked).trans rfl
theorem model_count_7820 : modelRow 7820 = 1 := by decide
theorem modelrow_agrees_7820 : modelRow 7820 = row 7820 := model_count_7820.trans row_count_7820.symm
def row_7821 : RowCertificate 7821 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7821 1) := by change ¬ Nat.Prime 244687807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34955401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7821 2) := by change ¬ Nat.Prime 244672165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48934433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7821 3) := by change Nat.Prime 244656523; exact ProjectEulerSolutions.P58PrimeNodes.p_244656523.prime
    exact decide_eq_true prime
theorem row_count_7821 : row 7821 = 1 := (row_7821.checked).trans rfl
theorem model_count_7821 : modelRow 7821 = 1 := by decide
theorem modelrow_agrees_7821 : modelRow 7821 = row 7821 := model_count_7821.trans row_count_7821.symm
def row_7822 : RowCertificate 7822 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7822 1) := by change ¬ Nat.Prime 244750381; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12881599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7822 2) := by change ¬ Nat.Prime 244734737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18825749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7822 3) := by change ¬ Nat.Prime 244719093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81573031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7822 : row 7822 = 0 := (row_7822.checked).trans rfl
theorem model_count_7822 : modelRow 7822 = 0 := by decide
theorem modelrow_agrees_7822 : modelRow 7822 = row 7822 := model_count_7822.trans row_count_7822.symm
def row_7823 : RowCertificate 7823 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7823 1) := by change ¬ Nat.Prime 244812963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81604321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7823 2) := by change Nat.Prime 244797317; exact ProjectEulerSolutions.P58PrimeNodes.p_244797317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7823 3) := by change ¬ Nat.Prime 244781671; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5692597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7823 : row 7823 = 1 := (row_7823.checked).trans rfl
theorem model_count_7823 : modelRow 7823 = 1 := by decide
theorem modelrow_agrees_7823 : modelRow 7823 = row 7823 := model_count_7823.trans row_count_7823.symm
def row_7824 : RowCertificate 7824 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7824 1) := by change ¬ Nat.Prime 244875553; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18836581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7824 2) := by change ¬ Nat.Prime 244859905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48971981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7824 3) := by change ¬ Nat.Prime 244844257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 34977751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7824 : row 7824 = 0 := (row_7824.checked).trans rfl
theorem model_count_7824 : modelRow 7824 = 0 := by decide
theorem modelrow_agrees_7824 : modelRow 7824 = row 7824 := model_count_7824.trans row_count_7824.symm
def row_7825 : RowCertificate 7825 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7825 1) := by change ¬ Nat.Prime 244938151; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2247139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7825 2) := by change ¬ Nat.Prime 244922501; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 1415737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7825 3) := by change ¬ Nat.Prime 244906851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81635617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7825 : row 7825 = 0 := (row_7825.checked).trans rfl
theorem model_count_7825 : modelRow 7825 = 0 := by decide
theorem modelrow_agrees_7825 : modelRow 7825 = row 7825 := model_count_7825.trans row_count_7825.symm
def row_7826 : RowCertificate 7826 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7826 1) := by change ¬ Nat.Prime 245000757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81666919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7826 2) := by change ¬ Nat.Prime 244985105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 48997021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7826 3) := by change Nat.Prime 244969453; exact ProjectEulerSolutions.P58PrimeNodes.p_244969453.prime
    exact decide_eq_true prime
theorem row_count_7826 : row 7826 = 1 := (row_7826.checked).trans rfl
theorem model_count_7826 : modelRow 7826 = 1 := by decide
theorem modelrow_agrees_7826 : modelRow 7826 = row 7826 := model_count_7826.trans row_count_7826.symm
def row_7827 : RowCertificate 7827 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7827 1) := by change ¬ Nat.Prime 245063371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35009053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7827 2) := by change Nat.Prime 245047717; exact ProjectEulerSolutions.P58PrimeNodes.p_245047717.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7827 3) := by change Nat.Prime 245032063; exact ProjectEulerSolutions.P58PrimeNodes.p_245032063.prime
    exact decide_eq_true prime
theorem row_count_7827 : row 7827 = 2 := (row_7827.checked).trans rfl
theorem model_count_7827 : modelRow 7827 = 2 := by decide
theorem modelrow_agrees_7827 : modelRow 7827 = row 7827 := model_count_7827.trans row_count_7827.symm
def row_7828 : RowCertificate 7828 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7828 1) := by change ¬ Nat.Prime 245125993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35017999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7828 2) := by change Nat.Prime 245110337; exact ProjectEulerSolutions.P58PrimeNodes.p_245110337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7828 3) := by change ¬ Nat.Prime 245094681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81698227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7828 : row 7828 = 1 := (row_7828.checked).trans rfl
theorem model_count_7828 : modelRow 7828 = 1 := by decide
theorem modelrow_agrees_7828 : modelRow 7828 = row 7828 := model_count_7828.trans row_count_7828.symm
def row_7829 : RowCertificate 7829 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7829 1) := by change ¬ Nat.Prime 245188623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81729541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7829 2) := by change ¬ Nat.Prime 245172965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49034593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7829 3) := by change Nat.Prime 245157307; exact ProjectEulerSolutions.P58PrimeNodes.p_245157307.prime
    exact decide_eq_true prime
theorem row_count_7829 : row 7829 = 1 := (row_7829.checked).trans rfl
theorem model_count_7829 : modelRow 7829 = 1 := by decide
theorem modelrow_agrees_7829 : modelRow 7829 = row 7829 := model_count_7829.trans row_count_7829.symm
def row_7830 : RowCertificate 7830 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7830 1) := by change ¬ Nat.Prime 245251261; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7911331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7830 2) := by change ¬ Nat.Prime 245235601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18864277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7830 3) := by change Nat.Prime 245219941; exact ProjectEulerSolutions.P58PrimeNodes.p_245219941.prime
    exact decide_eq_true prime
theorem row_count_7830 : row 7830 = 1 := (row_7830.checked).trans rfl
theorem model_count_7830 : modelRow 7830 = 1 := by decide
theorem modelrow_agrees_7830 : modelRow 7830 = row 7830 := model_count_7830.trans row_count_7830.symm
def row_7831 : RowCertificate 7831 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7831 1) := by change ¬ Nat.Prime 245313907; exact PrattCertificates.composite_of_product (factor := 5839) (cofactor := 42013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7831 2) := by change ¬ Nat.Prime 245298245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49059649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7831 3) := by change ¬ Nat.Prime 245282583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81760861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7831 : row 7831 = 0 := (row_7831.checked).trans rfl
theorem model_count_7831 : modelRow 7831 = 0 := by decide
theorem modelrow_agrees_7831 : modelRow 7831 = row 7831 := model_count_7831.trans row_count_7831.symm
def row_7832 : RowCertificate 7832 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7832 1) := by change ¬ Nat.Prime 245376561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81792187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7832 2) := by change Nat.Prime 245360897; exact ProjectEulerSolutions.P58PrimeNodes.p_245360897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7832 3) := by change ¬ Nat.Prime 245345233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35049319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7832 : row 7832 = 1 := (row_7832.checked).trans rfl
theorem model_count_7832 : modelRow 7832 = 1 := by decide
theorem modelrow_agrees_7832 : modelRow 7832 = row 7832 := model_count_7832.trans row_count_7832.symm
def row_7833 : RowCertificate 7833 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7833 1) := by change ¬ Nat.Prime 245439223; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1071787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7833 2) := by change ¬ Nat.Prime 245423557; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4023337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7833 3) := by change ¬ Nat.Prime 245407891; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3106429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7833 : row 7833 = 0 := (row_7833.checked).trans rfl
theorem model_count_7833 : modelRow 7833 = 0 := by decide
theorem modelrow_agrees_7833 : modelRow 7833 = row 7833 := model_count_7833.trans row_count_7833.symm
def row_7834 : RowCertificate 7834 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7834 1) := by change ¬ Nat.Prime 245501893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35071699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7834 2) := by change ¬ Nat.Prime 245486225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49097245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7834 3) := by change ¬ Nat.Prime 245470557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81823519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7834 : row 7834 = 0 := (row_7834.checked).trans rfl
theorem model_count_7834 : modelRow 7834 = 0 := by decide
theorem modelrow_agrees_7834 : modelRow 7834 = row 7834 := model_count_7834.trans row_count_7834.symm
def row_7835 : RowCertificate 7835 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7835 1) := by change ¬ Nat.Prime 245564571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81854857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7835 2) := by change ¬ Nat.Prime 245548901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18888377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7835 3) := by change Nat.Prime 245533231; exact ProjectEulerSolutions.P58PrimeNodes.p_245533231.prime
    exact decide_eq_true prime
theorem row_count_7835 : row 7835 = 1 := (row_7835.checked).trans rfl
theorem model_count_7835 : modelRow 7835 = 1 := by decide
theorem modelrow_agrees_7835 : modelRow 7835 = row 7835 := model_count_7835.trans row_count_7835.symm
def row_7836 : RowCertificate 7836 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7836 1) := by change ¬ Nat.Prime 245627257; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 886741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7836 2) := by change ¬ Nat.Prime 245611585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49122317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7836 3) := by change ¬ Nat.Prime 245595913; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1626463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7836 : row 7836 = 0 := (row_7836.checked).trans rfl
theorem model_count_7836 : modelRow 7836 = 0 := by decide
theorem modelrow_agrees_7836 : modelRow 7836 = row 7836 := model_count_7836.trans row_count_7836.symm
def row_7837 : RowCertificate 7837 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7837 1) := by change ¬ Nat.Prime 245689951; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18899227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7837 2) := by change Nat.Prime 245674277; exact ProjectEulerSolutions.P58PrimeNodes.p_245674277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7837 3) := by change ¬ Nat.Prime 245658603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81886201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7837 : row 7837 = 1 := (row_7837.checked).trans rfl
theorem model_count_7837 : modelRow 7837 = 1 := by decide
theorem modelrow_agrees_7837 : modelRow 7837 = row 7837 := model_count_7837.trans row_count_7837.symm
def row_7838 : RowCertificate 7838 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7838 1) := by change ¬ Nat.Prime 245752653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81917551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7838 2) := by change Nat.Prime 245736977; exact ProjectEulerSolutions.P58PrimeNodes.p_245736977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7838 3) := by change ¬ Nat.Prime 245721301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35103043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7838 : row 7838 = 1 := (row_7838.checked).trans rfl
theorem model_count_7838 : modelRow 7838 = 1 := by decide
theorem modelrow_agrees_7838 : modelRow 7838 = row 7838 := model_count_7838.trans row_count_7838.symm
def row_7839 : RowCertificate 7839 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7839 1) := by change ¬ Nat.Prime 245815363; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2534179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7839 2) := by change ¬ Nat.Prime 245799685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49159937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7839 3) := by change ¬ Nat.Prime 245784007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35112001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7839 : row 7839 = 0 := (row_7839.checked).trans rfl
theorem model_count_7839 : modelRow 7839 = 0 := by decide
theorem modelrow_agrees_7839 : modelRow 7839 = row 7839 := model_count_7839.trans row_count_7839.symm
def row_7840 : RowCertificate 7840 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7840 1) := by change ¬ Nat.Prime 245878081; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7931551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7840 2) := by change Nat.Prime 245862401; exact ProjectEulerSolutions.P58PrimeNodes.p_245862401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7840 3) := by change ¬ Nat.Prime 245846721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81948907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7840 : row 7840 = 1 := (row_7840.checked).trans rfl
theorem model_count_7840 : modelRow 7840 = 1 := by decide
theorem modelrow_agrees_7840 : modelRow 7840 = row 7840 := model_count_7840.trans row_count_7840.symm
def row_7841 : RowCertificate 7841 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7841 1) := by change ¬ Nat.Prime 245940807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 81980269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7841 2) := by change ¬ Nat.Prime 245925125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49185025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7841 3) := by change ¬ Nat.Prime 245909443; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18916111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7841 : row 7841 = 0 := (row_7841.checked).trans rfl
theorem model_count_7841 : modelRow 7841 = 0 := by decide
theorem modelrow_agrees_7841 : modelRow 7841 = row 7841 := model_count_7841.trans row_count_7841.symm
def row_7842 : RowCertificate 7842 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7842 1) := by change ¬ Nat.Prime 246003541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35143363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7842 2) := by change ¬ Nat.Prime 245987857; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 914453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7842 3) := by change ¬ Nat.Prime 245972173; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 240913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7842 : row 7842 = 0 := (row_7842.checked).trans rfl
theorem model_count_7842 : modelRow 7842 = 0 := by decide
theorem modelrow_agrees_7842 : modelRow 7842 = row 7842 := model_count_7842.trans row_count_7842.symm
def row_7843 : RowCertificate 7843 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7843 1) := by change ¬ Nat.Prime 246066283; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12950857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7843 2) := by change ¬ Nat.Prime 246050597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18926969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7843 3) := by change ¬ Nat.Prime 246034911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82011637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7843 : row 7843 = 0 := (row_7843.checked).trans rfl
theorem model_count_7843 : modelRow 7843 = 0 := by decide
theorem modelrow_agrees_7843 : modelRow 7843 = row 7843 := model_count_7843.trans row_count_7843.symm
def row_7844 : RowCertificate 7844 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7844 1) := by change ¬ Nat.Prime 246129033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82043011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7844 2) := by change ¬ Nat.Prime 246113345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49222669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7844 3) := by change ¬ Nat.Prime 246097657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18930589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7844 : row 7844 = 0 := (row_7844.checked).trans rfl
theorem model_count_7844 : modelRow 7844 = 0 := by decide
theorem modelrow_agrees_7844 : modelRow 7844 = row 7844 := model_count_7844.trans row_count_7844.symm
def row_7845 : RowCertificate 7845 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7845 1) := by change ¬ Nat.Prime 246191791; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4035931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7845 2) := by change ¬ Nat.Prime 246176101; exact PrattCertificates.composite_of_product (factor := 953) (cofactor := 258317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7845 3) := by change ¬ Nat.Prime 246160411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35165773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7845 : row 7845 = 0 := (row_7845.checked).trans rfl
theorem model_count_7845 : modelRow 7845 = 0 := by decide
theorem modelrow_agrees_7845 : modelRow 7845 = row 7845 := model_count_7845.trans row_count_7845.symm
def row_7846 : RowCertificate 7846 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7846 1) := by change Nat.Prime 246254557; exact ProjectEulerSolutions.P58PrimeNodes.p_246254557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7846 2) := by change ¬ Nat.Prime 246238865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49247773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7846 3) := by change ¬ Nat.Prime 246223173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82074391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7846 : row 7846 = 1 := (row_7846.checked).trans rfl
theorem model_count_7846 : modelRow 7846 = 1 := by decide
theorem modelrow_agrees_7846 : modelRow 7846 = row 7846 := model_count_7846.trans row_count_7846.symm
def row_7847 : RowCertificate 7847 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7847 1) := by change ¬ Nat.Prime 246317331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82105777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7847 2) := by change ¬ Nat.Prime 246301637; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6656801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7847 3) := by change Nat.Prime 246285943; exact ProjectEulerSolutions.P58PrimeNodes.p_246285943.prime
    exact decide_eq_true prime
theorem row_count_7847 : row 7847 = 1 := (row_7847.checked).trans rfl
theorem model_count_7847 : modelRow 7847 = 1 := by decide
theorem modelrow_agrees_7847 : modelRow 7847 = row 7847 := model_count_7847.trans row_count_7847.symm
def row_7848 : RowCertificate 7848 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7848 1) := by change ¬ Nat.Prime 246380113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35197159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7848 2) := by change ¬ Nat.Prime 246364417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18951109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7848 3) := by change Nat.Prime 246348721; exact ProjectEulerSolutions.P58PrimeNodes.p_246348721.prime
    exact decide_eq_true prime
theorem row_count_7848 : row 7848 = 1 := (row_7848.checked).trans rfl
theorem model_count_7848 : modelRow 7848 = 1 := by decide
theorem modelrow_agrees_7848 : modelRow 7848 = row 7848 := model_count_7848.trans row_count_7848.symm
def row_7849 : RowCertificate 7849 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7849 1) := by change ¬ Nat.Prime 246442903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35206129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7849 2) := by change ¬ Nat.Prime 246427205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49285441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7849 3) := by change ¬ Nat.Prime 246411507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82137169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7849 : row 7849 = 0 := (row_7849.checked).trans rfl
theorem model_count_7849 : modelRow 7849 = 0 := by decide
theorem modelrow_agrees_7849 : modelRow 7849 = row 7849 := model_count_7849.trans row_count_7849.symm
def row_7850 : RowCertificate 7850 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7850 1) := by change ¬ Nat.Prime 246505701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82168567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7850 2) := by change ¬ Nat.Prime 246490001; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 1057897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7850 3) := by change Nat.Prime 246474301; exact ProjectEulerSolutions.P58PrimeNodes.p_246474301.prime
    exact decide_eq_true prime
theorem row_count_7850 : row 7850 = 1 := (row_7850.checked).trans rfl
theorem model_count_7850 : modelRow 7850 = 1 := by decide
theorem modelrow_agrees_7850 : modelRow 7850 = row 7850 := model_count_7850.trans row_count_7850.symm
def row_7851 : RowCertificate 7851 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7851 1) := by change ¬ Nat.Prime 246568507; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 2393869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7851 2) := by change ¬ Nat.Prime 246552805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49310561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7851 3) := by change ¬ Nat.Prime 246537103; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 12975637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7851 : row 7851 = 0 := (row_7851.checked).trans rfl
theorem model_count_7851 : modelRow 7851 = 0 := by decide
theorem modelrow_agrees_7851 : modelRow 7851 = row 7851 := model_count_7851.trans row_count_7851.symm
def row_7852 : RowCertificate 7852 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7852 1) := by change ¬ Nat.Prime 246631321; exact PrattCertificates.composite_of_product (factor := 5527) (cofactor := 44623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7852 2) := by change ¬ Nat.Prime 246615617; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14506801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7852 3) := by change ¬ Nat.Prime 246599913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82199971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7852 : row 7852 = 0 := (row_7852.checked).trans rfl
theorem model_count_7852 : modelRow 7852 = 0 := by decide
theorem modelrow_agrees_7852 : modelRow 7852 = row 7852 := model_count_7852.trans row_count_7852.symm
def row_7853 : RowCertificate 7853 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7853 1) := by change ¬ Nat.Prime 246694143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82231381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7853 2) := by change ¬ Nat.Prime 246678437; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8506153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7853 3) := by change ¬ Nat.Prime 246662731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35237533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7853 : row 7853 = 0 := (row_7853.checked).trans rfl
theorem model_count_7853 : modelRow 7853 = 0 := by decide
theorem modelrow_agrees_7853 : modelRow 7853 = row 7853 := model_count_7853.trans row_count_7853.symm
def row_7854 : RowCertificate 7854 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7854 1) := by change Nat.Prime 246756973; exact ProjectEulerSolutions.P58PrimeNodes.p_246756973.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7854 2) := by change ¬ Nat.Prime 246741265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49348253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7854 3) := by change ¬ Nat.Prime 246725557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18978889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7854 : row 7854 = 1 := (row_7854.checked).trans rfl
theorem model_count_7854 : modelRow 7854 = 1 := by decide
theorem modelrow_agrees_7854 : modelRow 7854 = row 7854 := model_count_7854.trans row_count_7854.symm
def row_7855 : RowCertificate 7855 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7855 1) := by change ¬ Nat.Prime 246819811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35259973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7855 2) := by change Nat.Prime 246804101; exact ProjectEulerSolutions.P58PrimeNodes.p_246804101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7855 3) := by change ¬ Nat.Prime 246788391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82262797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7855 : row 7855 = 1 := (row_7855.checked).trans rfl
theorem model_count_7855 : modelRow 7855 = 1 := by decide
theorem modelrow_agrees_7855 : modelRow 7855 = row 7855 := model_count_7855.trans row_count_7855.symm
def row_7856 : RowCertificate 7856 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7856 1) := by change ¬ Nat.Prime 246882657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82294219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7856 2) := by change ¬ Nat.Prime 246866945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49373389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7856 3) := by change ¬ Nat.Prime 246851233; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7962943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7856 : row 7856 = 0 := (row_7856.checked).trans rfl
theorem model_count_7856 : modelRow 7856 = 0 := by decide
theorem modelrow_agrees_7856 : modelRow 7856 = row 7856 := model_count_7856.trans row_count_7856.symm
def row_7857 : RowCertificate 7857 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7857 1) := by change ¬ Nat.Prime 246945511; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6674203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7857 2) := by change ¬ Nat.Prime 246929797; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1279429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7857 3) := by change ¬ Nat.Prime 246914083; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 18993391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7857 : row 7857 = 0 := (row_7857.checked).trans rfl
theorem model_count_7857 : modelRow 7857 = 0 := by decide
theorem modelrow_agrees_7857 : modelRow 7857 = row 7857 := model_count_7857.trans row_count_7857.symm
def row_7858 : RowCertificate 7858 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7858 1) := by change Nat.Prime 247008373; exact ProjectEulerSolutions.P58PrimeNodes.p_247008373.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7858 2) := by change Nat.Prime 246992657; exact ProjectEulerSolutions.P58PrimeNodes.p_246992657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7858 3) := by change ¬ Nat.Prime 246976941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82325647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7858 : row 7858 = 2 := (row_7858.checked).trans rfl
theorem model_count_7858 : modelRow 7858 = 2 := by decide
theorem modelrow_agrees_7858 : modelRow 7858 = row 7858 := model_count_7858.trans row_count_7858.symm
def row_7859 : RowCertificate 7859 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7859 1) := by change ¬ Nat.Prime 247071243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82357081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7859 2) := by change ¬ Nat.Prime 247055525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49411105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7859 3) := by change ¬ Nat.Prime 247039807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35291401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7859 : row 7859 = 0 := (row_7859.checked).trans rfl
theorem model_count_7859 : modelRow 7859 = 0 := by decide
theorem modelrow_agrees_7859 : modelRow 7859 = row 7859 := model_count_7859.trans row_count_7859.symm
def row_7860 : RowCertificate 7860 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7860 1) := by change ¬ Nat.Prime 247134121; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19010317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7860 2) := by change ¬ Nat.Prime 247118401; exact PrattCertificates.composite_of_product (factor := 3049) (cofactor := 81049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7860 3) := by change ¬ Nat.Prime 247102681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35300383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7860 : row 7860 = 0 := (row_7860.checked).trans rfl
theorem model_count_7860 : modelRow 7860 = 0 := by decide
theorem modelrow_agrees_7860 : modelRow 7860 = row 7860 := model_count_7860.trans row_count_7860.symm
def row_7861 : RowCertificate 7861 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7861 1) := by change ¬ Nat.Prime 247197007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 7974097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7861 2) := by change ¬ Nat.Prime 247181285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49436257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7861 3) := by change ¬ Nat.Prime 247165563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82388521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7861 : row 7861 = 0 := (row_7861.checked).trans rfl
theorem model_count_7861 : modelRow 7861 = 0 := by decide
theorem modelrow_agrees_7861 : modelRow 7861 = row 7861 := model_count_7861.trans row_count_7861.symm
def row_7862 : RowCertificate 7862 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7862 1) := by change ¬ Nat.Prime 247259901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82419967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7862 2) := by change ¬ Nat.Prime 247244177; exact PrattCertificates.composite_of_product (factor := 2753) (cofactor := 89809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7862 3) := by change Nat.Prime 247228453; exact ProjectEulerSolutions.P58PrimeNodes.p_247228453.prime
    exact decide_eq_true prime
theorem row_count_7862 : row 7862 = 1 := (row_7862.checked).trans rfl
theorem model_count_7862 : modelRow 7862 = 1 := by decide
theorem modelrow_agrees_7862 : modelRow 7862 = row 7862 := model_count_7862.trans row_count_7862.symm
def row_7863 : RowCertificate 7863 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7863 1) := by change ¬ Nat.Prime 247322803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35331829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7863 2) := by change Nat.Prime 247307077; exact ProjectEulerSolutions.P58PrimeNodes.p_247307077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7863 3) := by change ¬ Nat.Prime 247291351; exact PrattCertificates.composite_of_product (factor := 1039) (cofactor := 238009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7863 : row 7863 = 1 := (row_7863.checked).trans rfl
theorem model_count_7863 : modelRow 7863 = 1 := by decide
theorem modelrow_agrees_7863 : modelRow 7863 = row 7863 := model_count_7863.trans row_count_7863.symm
def row_7864 : RowCertificate 7864 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7864 1) := by change ¬ Nat.Prime 247385713; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1947919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7864 2) := by change ¬ Nat.Prime 247369985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49473997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7864 3) := by change ¬ Nat.Prime 247354257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82451419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7864 : row 7864 = 0 := (row_7864.checked).trans rfl
theorem model_count_7864 : modelRow 7864 = 0 := by decide
theorem modelrow_agrees_7864 : modelRow 7864 = row 7864 := model_count_7864.trans row_count_7864.symm
def row_7865 : RowCertificate 7865 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7865 1) := by change ¬ Nat.Prime 247448631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82482877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7865 2) := by change ¬ Nat.Prime 247432901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8532169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7865 3) := by change Nat.Prime 247417171; exact ProjectEulerSolutions.P58PrimeNodes.p_247417171.prime
    exact decide_eq_true prime
theorem row_count_7865 : row 7865 = 1 := (row_7865.checked).trans rfl
theorem model_count_7865 : modelRow 7865 = 1 := by decide
theorem modelrow_agrees_7865 : modelRow 7865 = row 7865 := model_count_7865.trans row_count_7865.symm
def row_7866 : RowCertificate 7866 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7866 1) := by change ¬ Nat.Prime 247511557; exact PrattCertificates.composite_of_product (factor := 3877) (cofactor := 63841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7866 2) := by change ¬ Nat.Prime 247495825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49499165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7866 3) := by change ¬ Nat.Prime 247480093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35354299) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7866 : row 7866 = 0 := (row_7866.checked).trans rfl
theorem model_count_7866 : modelRow 7866 = 0 := by decide
theorem modelrow_agrees_7866 : modelRow 7866 = row 7866 := model_count_7866.trans row_count_7866.symm
def row_7867 : RowCertificate 7867 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7867 1) := by change ¬ Nat.Prime 247574491; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 367867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7867 2) := by change ¬ Nat.Prime 247558757; exact PrattCertificates.composite_of_product (factor := 5569) (cofactor := 44453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7867 3) := by change ¬ Nat.Prime 247543023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82514341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7867 : row 7867 = 0 := (row_7867.checked).trans rfl
theorem model_count_7867 : modelRow 7867 = 0 := by decide
theorem modelrow_agrees_7867 : modelRow 7867 = row 7867 := model_count_7867.trans row_count_7867.symm
def row_7868 : RowCertificate 7868 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7868 1) := by change ¬ Nat.Prime 247637433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82545811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7868 2) := by change ¬ Nat.Prime 247621697; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2552801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7868 3) := by change ¬ Nat.Prime 247605961; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6692053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7868 : row 7868 = 0 := (row_7868.checked).trans rfl
theorem model_count_7868 : modelRow 7868 = 0 := by decide
theorem modelrow_agrees_7868 : modelRow 7868 = row 7868 := model_count_7868.trans row_count_7868.symm
def row_7869 : RowCertificate 7869 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7869 1) := by change ¬ Nat.Prime 247700383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35385769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7869 2) := by change ¬ Nat.Prime 247684645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49536929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7869 3) := by change Nat.Prime 247668907; exact ProjectEulerSolutions.P58PrimeNodes.p_247668907.prime
    exact decide_eq_true prime
theorem row_count_7869 : row 7869 = 1 := (row_7869.checked).trans rfl
theorem model_count_7869 : modelRow 7869 = 1 := by decide
theorem modelrow_agrees_7869 : modelRow 7869 = row 7869 := model_count_7869.trans row_count_7869.symm
def row_7870 : RowCertificate 7870 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7870 1) := by change ¬ Nat.Prime 247763341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35394763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7870 2) := by change Nat.Prime 247747601; exact ProjectEulerSolutions.P58PrimeNodes.p_247747601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7870 3) := by change ¬ Nat.Prime 247731861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82577287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7870 : row 7870 = 1 := (row_7870.checked).trans rfl
theorem model_count_7870 : modelRow 7870 = 1 := by decide
theorem modelrow_agrees_7870 : modelRow 7870 = row 7870 := model_count_7870.trans row_count_7870.symm
def row_7871 : RowCertificate 7871 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7871 1) := by change ¬ Nat.Prime 247826307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82608769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7871 2) := by change ¬ Nat.Prime 247810565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49562113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7871 3) := by change ¬ Nat.Prime 247794823; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 2273347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7871 : row 7871 = 0 := (row_7871.checked).trans rfl
theorem model_count_7871 : modelRow 7871 = 0 := by decide
theorem modelrow_agrees_7871 : modelRow 7871 = row 7871 := model_count_7871.trans row_count_7871.symm
def row_7872 : RowCertificate 7872 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7872 1) := by change ¬ Nat.Prime 247889281; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5764867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7872 2) := by change Nat.Prime 247873537; exact ProjectEulerSolutions.P58PrimeNodes.p_247873537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7872 3) := by change ¬ Nat.Prime 247857793; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13045147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7872 : row 7872 = 1 := (row_7872.checked).trans rfl
theorem model_count_7872 : modelRow 7872 = 1 := by decide
theorem modelrow_agrees_7872 : modelRow 7872 = row 7872 := model_count_7872.trans row_count_7872.symm
def row_7873 : RowCertificate 7873 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7873 1) := by change ¬ Nat.Prime 247952263; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19073251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7873 2) := by change ¬ Nat.Prime 247936517; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14584501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7873 3) := by change ¬ Nat.Prime 247920771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82640257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7873 : row 7873 = 0 := (row_7873.checked).trans rfl
theorem model_count_7873 : modelRow 7873 = 0 := by decide
theorem modelrow_agrees_7873 : modelRow 7873 = row 7873 := model_count_7873.trans row_count_7873.symm
def row_7874 : RowCertificate 7874 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7874 1) := by change ¬ Nat.Prime 248015253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82671751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7874 2) := by change ¬ Nat.Prime 247999505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49599901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7874 3) := by change ¬ Nat.Prime 247983757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35426251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7874 : row 7874 = 0 := (row_7874.checked).trans rfl
theorem model_count_7874 : modelRow 7874 = 0 := by decide
theorem modelrow_agrees_7874 : modelRow 7874 = row 7874 := model_count_7874.trans row_count_7874.symm
def row_7875 : RowCertificate 7875 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7875 1) := by change ¬ Nat.Prime 248078251; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 474337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7875 2) := by change ¬ Nat.Prime 248062501; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 1664849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7875 3) := by change ¬ Nat.Prime 248046751; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2557183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7875 : row 7875 = 0 := (row_7875.checked).trans rfl
theorem model_count_7875 : modelRow 7875 = 0 := by decide
theorem modelrow_agrees_7875 : modelRow 7875 = row 7875 := model_count_7875.trans row_count_7875.symm
def row_7876 : RowCertificate 7876 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7876 1) := by change ¬ Nat.Prime 248141257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35448751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7876 2) := by change ¬ Nat.Prime 248125505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49625101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7876 3) := by change ¬ Nat.Prime 248109753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82703251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7876 : row 7876 = 0 := (row_7876.checked).trans rfl
theorem model_count_7876 : modelRow 7876 = 0 := by decide
theorem modelrow_agrees_7876 : modelRow 7876 = row 7876 := model_count_7876.trans row_count_7876.symm
def row_7877 : RowCertificate 7877 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7877 1) := by change ¬ Nat.Prime 248204271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82734757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7877 2) := by change Nat.Prime 248188517; exact ProjectEulerSolutions.P58PrimeNodes.p_248188517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7877 3) := by change ¬ Nat.Prime 248172763; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8005573) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7877 : row 7877 = 1 := (row_7877.checked).trans rfl
theorem model_count_7877 : modelRow 7877 = 1 := by decide
theorem modelrow_agrees_7877 : modelRow 7877 = row 7877 := model_count_7877.trans row_count_7877.symm
def row_7878 : RowCertificate 7878 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7878 1) := by change Nat.Prime 248267293; exact ProjectEulerSolutions.P58PrimeNodes.p_248267293.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7878 2) := by change ¬ Nat.Prime 248251537; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6709501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7878 3) := by change ¬ Nat.Prime 248235781; exact PrattCertificates.composite_of_product (factor := 1123) (cofactor := 221047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7878 : row 7878 = 1 := (row_7878.checked).trans rfl
theorem model_count_7878 : modelRow 7878 = 1 := by decide
theorem modelrow_agrees_7878 : modelRow 7878 = row 7878 := model_count_7878.trans row_count_7878.symm
def row_7879 : RowCertificate 7879 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7879 1) := by change ¬ Nat.Prime 248330323; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13070017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7879 2) := by change ¬ Nat.Prime 248314565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49662913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7879 3) := by change ¬ Nat.Prime 248298807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82766269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7879 : row 7879 = 0 := (row_7879.checked).trans rfl
theorem model_count_7879 : modelRow 7879 = 0 := by decide
theorem modelrow_agrees_7879 : modelRow 7879 = row 7879 := model_count_7879.trans row_count_7879.symm
def row_7880 : RowCertificate 7880 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7880 1) := by change ¬ Nat.Prime 248393361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82797787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7880 2) := by change ¬ Nat.Prime 248377601; exact PrattCertificates.composite_of_product (factor := 3373) (cofactor := 73637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7880 3) := by change ¬ Nat.Prime 248361841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35480263) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7880 : row 7880 = 0 := (row_7880.checked).trans rfl
theorem model_count_7880 : modelRow 7880 = 0 := by decide
theorem modelrow_agrees_7880 : modelRow 7880 = row 7880 := model_count_7880.trans row_count_7880.symm
def row_7881 : RowCertificate 7881 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7881 1) := by change ¬ Nat.Prime 248456407; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13076653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7881 2) := by change ¬ Nat.Prime 248440645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49688129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7881 3) := by change ¬ Nat.Prime 248424883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35489269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7881 : row 7881 = 0 := (row_7881.checked).trans rfl
theorem model_count_7881 : modelRow 7881 = 0 := by decide
theorem modelrow_agrees_7881 : modelRow 7881 = row 7881 := model_count_7881.trans row_count_7881.symm
def row_7882 : RowCertificate 7882 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7882 1) := by change ¬ Nat.Prime 248519461; exact PrattCertificates.composite_of_product (factor := 6379) (cofactor := 38959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7882 2) := by change ¬ Nat.Prime 248503697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19115669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7882 3) := by change ¬ Nat.Prime 248487933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82829311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7882 : row 7882 = 0 := (row_7882.checked).trans rfl
theorem model_count_7882 : modelRow 7882 = 0 := by decide
theorem modelrow_agrees_7882 : modelRow 7882 = row 7882 := model_count_7882.trans row_count_7882.symm
def row_7883 : RowCertificate 7883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7883 1) := by change ¬ Nat.Prime 248582523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82860841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7883 2) := by change ¬ Nat.Prime 248566757; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 2461057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7883 3) := by change ¬ Nat.Prime 248550991; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19119307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7883 : row 7883 = 0 := (row_7883.checked).trans rfl
theorem model_count_7883 : modelRow 7883 = 0 := by decide
theorem modelrow_agrees_7883 : modelRow 7883 = row 7883 := model_count_7883.trans row_count_7883.symm
def row_7884 : RowCertificate 7884 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7884 1) := by change ¬ Nat.Prime 248645593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35520799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7884 2) := by change ¬ Nat.Prime 248629825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49725965) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7884 3) := by change ¬ Nat.Prime 248614057; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1957591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7884 : row 7884 = 0 := (row_7884.checked).trans rfl
theorem model_count_7884 : modelRow 7884 = 0 := by decide
theorem modelrow_agrees_7884 : modelRow 7884 = row 7884 := model_count_7884.trans row_count_7884.symm
def row_7885 : RowCertificate 7885 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7885 1) := by change ¬ Nat.Prime 248708671; exact PrattCertificates.composite_of_product (factor := 1867) (cofactor := 133213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7885 2) := by change ¬ Nat.Prime 248692901; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 666737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7885 3) := by change ¬ Nat.Prime 248677131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82892377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7885 : row 7885 = 0 := (row_7885.checked).trans rfl
theorem model_count_7885 : modelRow 7885 = 0 := by decide
theorem modelrow_agrees_7885 : modelRow 7885 = row 7885 := model_count_7885.trans row_count_7885.symm
def row_7886 : RowCertificate 7886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7886 1) := by change ¬ Nat.Prime 248771757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82923919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7886 2) := by change ¬ Nat.Prime 248755985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49751197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7886 3) := by change Nat.Prime 248740213; exact ProjectEulerSolutions.P58PrimeNodes.p_248740213.prime
    exact decide_eq_true prime
theorem row_count_7886 : row 7886 = 1 := (row_7886.checked).trans rfl
theorem model_count_7886 : modelRow 7886 = 1 := by decide
theorem modelrow_agrees_7886 : modelRow 7886 = row 7886 := model_count_7886.trans row_count_7886.symm
def row_7887 : RowCertificate 7887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7887 1) := by change ¬ Nat.Prime 248834851; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5786857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7887 2) := by change ¬ Nat.Prime 248819077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19139929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7887 3) := by change ¬ Nat.Prime 248803303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35543329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7887 : row 7887 = 0 := (row_7887.checked).trans rfl
theorem model_count_7887 : modelRow 7887 = 0 := by decide
theorem modelrow_agrees_7887 : modelRow 7887 = row 7887 := model_count_7887.trans row_count_7887.symm
def row_7888 : RowCertificate 7888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7888 1) := by change ¬ Nat.Prime 248897953; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 3409561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7888 2) := by change ¬ Nat.Prime 248882177; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6070297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7888 3) := by change ¬ Nat.Prime 248866401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82955467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7888 : row 7888 = 0 := (row_7888.checked).trans rfl
theorem model_count_7888 : modelRow 7888 = 0 := by decide
theorem modelrow_agrees_7888 : modelRow 7888 = row 7888 := model_count_7888.trans row_count_7888.symm
def row_7889 : RowCertificate 7889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7889 1) := by change ¬ Nat.Prime 248961063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 82987021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7889 2) := by change ¬ Nat.Prime 248945285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49789057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7889 3) := by change ¬ Nat.Prime 248929507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13101553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7889 : row 7889 = 0 := (row_7889.checked).trans rfl
theorem model_count_7889 : modelRow 7889 = 0 := by decide
theorem modelrow_agrees_7889 : modelRow 7889 = row 7889 := model_count_7889.trans row_count_7889.symm
def row_7890 : RowCertificate 7890 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7890 1) := by change ¬ Nat.Prime 249024181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35574883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7890 2) := by change ¬ Nat.Prime 249008401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14647553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7890 3) := by change ¬ Nat.Prime 248992621; exact PrattCertificates.composite_of_product (factor := 1399) (cofactor := 177979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7890 : row 7890 = 0 := (row_7890.checked).trans rfl
theorem model_count_7890 : modelRow 7890 = 0 := by decide
theorem modelrow_agrees_7890 : modelRow 7890 = row 7890 := model_count_7890.trans row_count_7890.symm
def row_7891 : RowCertificate 7891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7891 1) := by change ¬ Nat.Prime 249087307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35583901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7891 2) := by change ¬ Nat.Prime 249071525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49814305) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7891 3) := by change ¬ Nat.Prime 249055743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83018581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7891 : row 7891 = 0 := (row_7891.checked).trans rfl
theorem model_count_7891 : modelRow 7891 = 0 := by decide
theorem modelrow_agrees_7891 : modelRow 7891 = row 7891 := model_count_7891.trans row_count_7891.symm
def row_7892 : RowCertificate 7892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7892 1) := by change ¬ Nat.Prime 249150441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83050147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7892 2) := by change Nat.Prime 249134657; exact ProjectEulerSolutions.P58PrimeNodes.p_249134657.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7892 3) := by change Nat.Prime 249118873; exact ProjectEulerSolutions.P58PrimeNodes.p_249118873.prime
    exact decide_eq_true prime
theorem row_count_7892 : row 7892 = 2 := (row_7892.checked).trans rfl
theorem model_count_7892 : modelRow 7892 = 2 := by decide
theorem modelrow_agrees_7892 : modelRow 7892 = row 7892 := model_count_7892.trans row_count_7892.symm
def row_7893 : RowCertificate 7893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7893 1) := by change ¬ Nat.Prime 249213583; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 575551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7893 2) := by change ¬ Nat.Prime 249197797; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 233113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7893 3) := by change ¬ Nat.Prime 249182011; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19167847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7893 : row 7893 = 0 := (row_7893.checked).trans rfl
theorem model_count_7893 : modelRow 7893 = 0 := by decide
theorem modelrow_agrees_7893 : modelRow 7893 = row 7893 := model_count_7893.trans row_count_7893.symm
def row_7894 : RowCertificate 7894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7894 1) := by change ¬ Nat.Prime 249276733; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6737209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7894 2) := by change ¬ Nat.Prime 249260945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49852189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7894 3) := by change ¬ Nat.Prime 249245157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83081719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7894 : row 7894 = 0 := (row_7894.checked).trans rfl
theorem model_count_7894 : modelRow 7894 = 0 := by decide
theorem modelrow_agrees_7894 : modelRow 7894 = row 7894 := model_count_7894.trans row_count_7894.symm
def row_7895 : RowCertificate 7895 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7895 1) := by change ¬ Nat.Prime 249339891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83113297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7895 2) := by change ¬ Nat.Prime 249324101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19178777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7895 3) := by change ¬ Nat.Prime 249308311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35615473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7895 : row 7895 = 0 := (row_7895.checked).trans rfl
theorem model_count_7895 : modelRow 7895 = 0 := by decide
theorem modelrow_agrees_7895 : modelRow 7895 = row 7895 := model_count_7895.trans row_count_7895.symm
def row_7896 : RowCertificate 7896 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7896 1) := by change ¬ Nat.Prime 249403057; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 432241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7896 2) := by change ¬ Nat.Prime 249387265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49877453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7896 3) := by change ¬ Nat.Prime 249371473; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19182421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7896 : row 7896 = 0 := (row_7896.checked).trans rfl
theorem model_count_7896 : modelRow 7896 = 0 := by decide
theorem modelrow_agrees_7896 : modelRow 7896 = row 7896 := model_count_7896.trans row_count_7896.symm
def row_7897 : RowCertificate 7897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7897 1) := by change ¬ Nat.Prime 249466231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35638033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7897 2) := by change ¬ Nat.Prime 249450437; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 6084157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7897 3) := by change ¬ Nat.Prime 249434643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83144881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7897 : row 7897 = 0 := (row_7897.checked).trans rfl
theorem model_count_7897 : modelRow 7897 = 0 := by decide
theorem modelrow_agrees_7897 : modelRow 7897 = row 7897 := model_count_7897.trans row_count_7897.symm
def row_7898 : RowCertificate 7898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7898 1) := by change ¬ Nat.Prime 249529413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83176471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7898 2) := by change Nat.Prime 249513617; exact ProjectEulerSolutions.P58PrimeNodes.p_249513617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7898 3) := by change ¬ Nat.Prime 249497821; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1378441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7898 : row 7898 = 1 := (row_7898.checked).trans rfl
theorem model_count_7898 : modelRow 7898 = 1 := by decide
theorem modelrow_agrees_7898 : modelRow 7898 = row 7898 := model_count_7898.trans row_count_7898.symm
def row_7899 : RowCertificate 7899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7899 1) := by change ¬ Nat.Prime 249592603; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19199431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7899 2) := by change ¬ Nat.Prime 249576805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49915361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7899 3) := by change Nat.Prime 249561007; exact ProjectEulerSolutions.P58PrimeNodes.p_249561007.prime
    exact decide_eq_true prime
theorem row_count_7899 : row 7899 = 1 := (row_7899.checked).trans rfl
theorem model_count_7899 : modelRow 7899 = 1 := by decide
theorem modelrow_agrees_7899 : modelRow 7899 = row 7899 := model_count_7899.trans row_count_7899.symm
def row_7900 : RowCertificate 7900 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7900 1) := by change ¬ Nat.Prime 249655801; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13139779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7900 2) := by change ¬ Nat.Prime 249640001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19203077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7900 3) := by change ¬ Nat.Prime 249624201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83208067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7900 : row 7900 = 0 := (row_7900.checked).trans rfl
theorem model_count_7900 : modelRow 7900 = 0 := by decide
theorem modelrow_agrees_7900 : modelRow 7900 = row 7900 := model_count_7900.trans row_count_7900.symm
def row_7901 : RowCertificate 7901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7901 1) := by change ¬ Nat.Prime 249719007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83239669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7901 2) := by change ¬ Nat.Prime 249703205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49940641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7901 3) := by change ¬ Nat.Prime 249687403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35669629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7901 : row 7901 = 0 := (row_7901.checked).trans rfl
theorem model_count_7901 : modelRow 7901 = 0 := by decide
theorem modelrow_agrees_7901 : modelRow 7901 = row 7901 := model_count_7901.trans row_count_7901.symm
def row_7902 : RowCertificate 7902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7902 1) := by change ¬ Nat.Prime 249782221; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19214017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7902 2) := by change Nat.Prime 249766417; exact ProjectEulerSolutions.P58PrimeNodes.p_249766417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7902 3) := by change ¬ Nat.Prime 249750613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35678659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7902 : row 7902 = 1 := (row_7902.checked).trans rfl
theorem model_count_7902 : modelRow 7902 = 1 := by decide
theorem modelrow_agrees_7902 : modelRow 7902 = row 7902 := model_count_7902.trans row_count_7902.symm
def row_7903 : RowCertificate 7903 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7903 1) := by change ¬ Nat.Prime 249845443; exact PrattCertificates.composite_of_product (factor := 1831) (cofactor := 136453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7903 2) := by change ¬ Nat.Prime 249829637; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14695861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7903 3) := by change ¬ Nat.Prime 249813831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83271277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7903 : row 7903 = 0 := (row_7903.checked).trans rfl
theorem model_count_7903 : modelRow 7903 = 0 := by decide
theorem modelrow_agrees_7903 : modelRow 7903 = row 7903 := model_count_7903.trans row_count_7903.symm
def row_7904 : RowCertificate 7904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7904 1) := by change ¬ Nat.Prime 249908673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83302891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7904 2) := by change ¬ Nat.Prime 249892865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 49978573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7904 3) := by change ¬ Nat.Prime 249877057; exact PrattCertificates.composite_of_product (factor := 13477) (cofactor := 18541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7904 : row 7904 = 0 := (row_7904.checked).trans rfl
theorem model_count_7904 : modelRow 7904 = 0 := by decide
theorem modelrow_agrees_7904 : modelRow 7904 = row 7904 := model_count_7904.trans row_count_7904.symm
def row_7905 : RowCertificate 7905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7905 1) := by change ¬ Nat.Prime 249971911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35710273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7905 2) := by change ¬ Nat.Prime 249956101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4097641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7905 3) := by change ¬ Nat.Prime 249940291; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6755143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7905 : row 7905 = 0 := (row_7905.checked).trans rfl
theorem model_count_7905 : modelRow 7905 = 0 := by decide
theorem modelrow_agrees_7905 : modelRow 7905 = row 7905 := model_count_7905.trans row_count_7905.symm
def row_7906 : RowCertificate 7906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7906 1) := by change ¬ Nat.Prime 250035157; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 4098937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7906 2) := by change ¬ Nat.Prime 250019345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50003869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7906 3) := by change ¬ Nat.Prime 250003533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83334511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7906 : row 7906 = 0 := (row_7906.checked).trans rfl
theorem model_count_7906 : modelRow 7906 = 0 := by decide
theorem modelrow_agrees_7906 : modelRow 7906 = row 7906 := model_count_7906.trans row_count_7906.symm
def row_7907 : RowCertificate 7907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7907 1) := by change ¬ Nat.Prime 250098411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83366137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7907 2) := by change ¬ Nat.Prime 250082597; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14710741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7907 3) := by change ¬ Nat.Prime 250066783; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1256617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7907 : row 7907 = 0 := (row_7907.checked).trans rfl
theorem model_count_7907 : modelRow 7907 = 0 := by decide
theorem modelrow_agrees_7907 : modelRow 7907 = row 7907 := model_count_7907.trans row_count_7907.symm
def row_7908 : RowCertificate 7908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7908 1) := by change ¬ Nat.Prime 250161673; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 513679) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7908 2) := by change ¬ Nat.Prime 250145857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19241989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7908 3) := by change ¬ Nat.Prime 250130041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35732863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7908 : row 7908 = 0 := (row_7908.checked).trans rfl
theorem model_count_7908 : modelRow 7908 = 0 := by decide
theorem modelrow_agrees_7908 : modelRow 7908 = row 7908 := model_count_7908.trans row_count_7908.symm
def row_7909 : RowCertificate 7909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7909 1) := by change ¬ Nat.Prime 250224943; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 462523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7909 2) := by change ¬ Nat.Prime 250209125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50041825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7909 3) := by change ¬ Nat.Prime 250193307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83397769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7909 : row 7909 = 0 := (row_7909.checked).trans rfl
theorem model_count_7909 : modelRow 7909 = 0 := by decide
theorem modelrow_agrees_7909 : modelRow 7909 = row 7909 := model_count_7909.trans row_count_7909.symm
def row_7910 : RowCertificate 7910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7910 1) := by change ¬ Nat.Prime 250288221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83429407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7910 2) := by change Nat.Prime 250272401; exact ProjectEulerSolutions.P58PrimeNodes.p_250272401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7910 3) := by change ¬ Nat.Prime 250256581; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13171399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7910 : row 7910 = 1 := (row_7910.checked).trans rfl
theorem model_count_7910 : modelRow 7910 = 1 := by decide
theorem modelrow_agrees_7910 : modelRow 7910 = row 7910 := model_count_7910.trans row_count_7910.symm
def row_7911 : RowCertificate 7911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7911 1) := by change ¬ Nat.Prime 250351507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35764501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7911 2) := by change ¬ Nat.Prime 250335685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50067137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7911 3) := by change Nat.Prime 250319863; exact ProjectEulerSolutions.P58PrimeNodes.p_250319863.prime
    exact decide_eq_true prime
theorem row_count_7911 : row 7911 = 1 := (row_7911.checked).trans rfl
theorem model_count_7911 : modelRow 7911 = 1 := by decide
theorem modelrow_agrees_7911 : modelRow 7911 = row 7911 := model_count_7911.trans row_count_7911.symm
def row_7912 : RowCertificate 7912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7912 1) := by change ¬ Nat.Prime 250414801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35773543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7912 2) := by change ¬ Nat.Prime 250398977; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 4724509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7912 3) := by change ¬ Nat.Prime 250383153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83461051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7912 : row 7912 = 0 := (row_7912.checked).trans rfl
theorem model_count_7912 : modelRow 7912 = 0 := by decide
theorem modelrow_agrees_7912 : modelRow 7912 = row 7912 := model_count_7912.trans row_count_7912.symm
def row_7913 : RowCertificate 7913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7913 1) := by change ¬ Nat.Prime 250478103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83492701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7913 2) := by change ¬ Nat.Prime 250462277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19266329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7913 3) := by change ¬ Nat.Prime 250446451; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6768823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7913 : row 7913 = 0 := (row_7913.checked).trans rfl
theorem model_count_7913 : modelRow 7913 = 0 := by decide
theorem modelrow_agrees_7913 : modelRow 7913 = row 7913 := model_count_7913.trans row_count_7913.symm
def row_7914 : RowCertificate 7914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7914 1) := by change ¬ Nat.Prime 250541413; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 502087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7914 2) := by change ¬ Nat.Prime 250525585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50105117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7914 3) := by change ¬ Nat.Prime 250509757; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1258843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7914 : row 7914 = 0 := (row_7914.checked).trans rfl
theorem model_count_7914 : modelRow 7914 = 0 := by decide
theorem modelrow_agrees_7914 : modelRow 7914 = row 7914 := model_count_7914.trans row_count_7914.symm
def row_7915 : RowCertificate 7915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7915 1) := by change ¬ Nat.Prime 250604731; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19277287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7915 2) := by change ¬ Nat.Prime 250588901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 6772673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7915 3) := by change ¬ Nat.Prime 250573071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83524357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7915 : row 7915 = 0 := (row_7915.checked).trans rfl
theorem model_count_7915 : modelRow 7915 = 0 := by decide
theorem modelrow_agrees_7915 : modelRow 7915 = row 7915 := model_count_7915.trans row_count_7915.symm
def row_7916 : RowCertificate 7916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7916 1) := by change ¬ Nat.Prime 250668057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83556019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7916 2) := by change ¬ Nat.Prime 250652225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50130445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7916 3) := by change ¬ Nat.Prime 250636393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35805199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7916 : row 7916 = 0 := (row_7916.checked).trans rfl
theorem model_count_7916 : modelRow 7916 = 0 := by decide
theorem modelrow_agrees_7916 : modelRow 7916 = row 7916 := model_count_7916.trans row_count_7916.symm
def row_7917 : RowCertificate 7917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7917 1) := by change ¬ Nat.Prime 250731391; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13196389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7917 2) := by change ¬ Nat.Prime 250715557; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 644513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7917 3) := by change Nat.Prime 250699723; exact ProjectEulerSolutions.P58PrimeNodes.p_250699723.prime
    exact decide_eq_true prime
theorem row_count_7917 : row 7917 = 1 := (row_7917.checked).trans rfl
theorem model_count_7917 : modelRow 7917 = 1 := by decide
theorem modelrow_agrees_7917 : modelRow 7917 = row 7917 := model_count_7917.trans row_count_7917.symm
def row_7918 : RowCertificate 7918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7918 1) := by change ¬ Nat.Prime 250794733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35827819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7918 2) := by change ¬ Nat.Prime 250778897; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 1449589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7918 3) := by change ¬ Nat.Prime 250763061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83587687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7918 : row 7918 = 0 := (row_7918.checked).trans rfl
theorem model_count_7918 : modelRow 7918 = 0 := by decide
theorem modelrow_agrees_7918 : modelRow 7918 = row 7918 := model_count_7918.trans row_count_7918.symm
def row_7919 : RowCertificate 7919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7919 1) := by change ¬ Nat.Prime 250858083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83619361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7919 2) := by change ¬ Nat.Prime 250842245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50168449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7919 3) := by change ¬ Nat.Prime 250826407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19294339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7919 : row 7919 = 0 := (row_7919.checked).trans rfl
theorem model_count_7919 : modelRow 7919 = 0 := by decide
theorem modelrow_agrees_7919 : modelRow 7919 = row 7919 := model_count_7919.trans row_count_7919.symm
def row_7920 : RowCertificate 7920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7920 1) := by change ¬ Nat.Prime 250921441; exact PrattCertificates.composite_of_product (factor := 6949) (cofactor := 36109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7920 2) := by change ¬ Nat.Prime 250905601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 14759153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7920 3) := by change ¬ Nat.Prime 250889761; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1189051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7920 : row 7920 = 0 := (row_7920.checked).trans rfl
theorem model_count_7920 : modelRow 7920 = 0 := by decide
theorem modelrow_agrees_7920 : modelRow 7920 = row 7920 := model_count_7920.trans row_count_7920.symm
def row_7921 : RowCertificate 7921 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7921 1) := by change Nat.Prime 250984807; exact ProjectEulerSolutions.P58PrimeNodes.p_250984807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7921 2) := by change ¬ Nat.Prime 250968965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50193793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7921 3) := by change ¬ Nat.Prime 250953123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83651041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7921 : row 7921 = 1 := (row_7921.checked).trans rfl
theorem model_count_7921 : modelRow 7921 = 1 := by decide
theorem modelrow_agrees_7921 : modelRow 7921 = row 7921 := model_count_7921.trans row_count_7921.symm
def row_7922 : RowCertificate 7922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7922 1) := by change ¬ Nat.Prime 251048181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83682727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7922 2) := by change Nat.Prime 251032337; exact ProjectEulerSolutions.P58PrimeNodes.p_251032337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7922 3) := by change ¬ Nat.Prime 251016493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35859499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7922 : row 7922 = 1 := (row_7922.checked).trans rfl
theorem model_count_7922 : modelRow 7922 = 1 := by decide
theorem modelrow_agrees_7922 : modelRow 7922 = row 7922 := model_count_7922.trans row_count_7922.symm
def row_7923 : RowCertificate 7923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7923 1) := by change ¬ Nat.Prime 251111563; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 8100373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7923 2) := by change ¬ Nat.Prime 251095717; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 8658473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7923 3) := by change ¬ Nat.Prime 251079871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35868553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7923 : row 7923 = 0 := (row_7923.checked).trans rfl
theorem model_count_7923 : modelRow 7923 = 0 := by decide
theorem modelrow_agrees_7923 : modelRow 7923 = row 7923 := model_count_7923.trans row_count_7923.symm
def row_7924 : RowCertificate 7924 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7924 1) := by change Nat.Prime 251174953; exact ProjectEulerSolutions.P58PrimeNodes.p_251174953.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7924 2) := by change ¬ Nat.Prime 251159105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50231821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7924 3) := by change ¬ Nat.Prime 251143257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83714419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7924 : row 7924 = 1 := (row_7924.checked).trans rfl
theorem model_count_7924 : modelRow 7924 = 1 := by decide
theorem modelrow_agrees_7924 : modelRow 7924 = row 7924 := model_count_7924.trans row_count_7924.symm
def row_7925 : RowCertificate 7925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7925 1) := by change ¬ Nat.Prime 251238351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83746117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7925 2) := by change ¬ Nat.Prime 251222501; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 243197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7925 3) := by change ¬ Nat.Prime 251206651; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 3749353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7925 : row 7925 = 0 := (row_7925.checked).trans rfl
theorem model_count_7925 : modelRow 7925 = 0 := by decide
theorem modelrow_agrees_7925 : modelRow 7925 = row 7925 := model_count_7925.trans row_count_7925.symm
def row_7926 : RowCertificate 7926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7926 1) := by change ¬ Nat.Prime 251301757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35900251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7926 2) := by change ¬ Nat.Prime 251285905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50257181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7926 3) := by change Nat.Prime 251270053; exact ProjectEulerSolutions.P58PrimeNodes.p_251270053.prime
    exact decide_eq_true prime
theorem row_count_7926 : row 7926 = 1 := (row_7926.checked).trans rfl
theorem model_count_7926 : modelRow 7926 = 1 := by decide
theorem modelrow_agrees_7926 : modelRow 7926 = row 7926 := model_count_7926.trans row_count_7926.symm
def row_7927 : RowCertificate 7927 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7927 1) := by change Nat.Prime 251365171; exact ProjectEulerSolutions.P58PrimeNodes.p_251365171.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7927 2) := by change ¬ Nat.Prime 251349317; exact PrattCertificates.composite_of_product (factor := 2749) (cofactor := 91433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7927 3) := by change ¬ Nat.Prime 251333463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83777821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7927 : row 7927 = 1 := (row_7927.checked).trans rfl
theorem model_count_7927 : modelRow 7927 = 1 := by decide
theorem modelrow_agrees_7927 : modelRow 7927 = row 7927 := model_count_7927.trans row_count_7927.symm
def row_7928 : RowCertificate 7928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7928 1) := by change ¬ Nat.Prime 251428593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83809531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7928 2) := by change Nat.Prime 251412737; exact ProjectEulerSolutions.P58PrimeNodes.p_251412737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7928 3) := by change ¬ Nat.Prime 251396881; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 3182239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7928 : row 7928 = 1 := (row_7928.checked).trans rfl
theorem model_count_7928 : modelRow 7928 = 1 := by decide
theorem modelrow_agrees_7928 : modelRow 7928 = row 7928 := model_count_7928.trans row_count_7928.symm
def row_7929 : RowCertificate 7929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7929 1) := by change ¬ Nat.Prime 251492023; exact PrattCertificates.composite_of_product (factor := 15319) (cofactor := 16417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7929 2) := by change ¬ Nat.Prime 251476165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50295233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7929 3) := by change ¬ Nat.Prime 251460307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35922901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7929 : row 7929 = 0 := (row_7929.checked).trans rfl
theorem model_count_7929 : modelRow 7929 = 0 := by decide
theorem modelrow_agrees_7929 : modelRow 7929 = row 7929 := model_count_7929.trans row_count_7929.symm
def row_7930 : RowCertificate 7930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7930 1) := by change ¬ Nat.Prime 251555461; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 5850127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7930 2) := by change ¬ Nat.Prime 251539601; exact PrattCertificates.composite_of_product (factor := 4721) (cofactor := 53281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7930 3) := by change ¬ Nat.Prime 251523741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83841247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7930 : row 7930 = 0 := (row_7930.checked).trans rfl
theorem model_count_7930 : modelRow 7930 = 0 := by decide
theorem modelrow_agrees_7930 : modelRow 7930 = row 7930 := model_count_7930.trans row_count_7930.symm
def row_7931 : RowCertificate 7931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7931 1) := by change ¬ Nat.Prime 251618907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83872969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7931 2) := by change ¬ Nat.Prime 251603045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50320609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7931 3) := by change ¬ Nat.Prime 251587183; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 550519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7931 : row 7931 = 0 := (row_7931.checked).trans rfl
theorem model_count_7931 : modelRow 7931 = 0 := by decide
theorem modelrow_agrees_7931 : modelRow 7931 = row 7931 := model_count_7931.trans row_count_7931.symm
def row_7932 : RowCertificate 7932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7932 1) := by change ¬ Nat.Prime 251682361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35954623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7932 2) := by change ¬ Nat.Prime 251666497; exact PrattCertificates.composite_of_product (factor := 11597) (cofactor := 21701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7932 3) := by change ¬ Nat.Prime 251650633; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19357741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7932 : row 7932 = 0 := (row_7932.checked).trans rfl
theorem model_count_7932 : modelRow 7932 = 0 := by decide
theorem modelrow_agrees_7932 : modelRow 7932 = row 7932 := model_count_7932.trans row_count_7932.symm
def row_7933 : RowCertificate 7933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7933 1) := by change ¬ Nat.Prime 251745823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 35963689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7933 2) := by change ¬ Nat.Prime 251729957; exact PrattCertificates.composite_of_product (factor := 2357) (cofactor := 106801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7933 3) := by change ¬ Nat.Prime 251714091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83904697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7933 : row 7933 = 0 := (row_7933.checked).trans rfl
theorem model_count_7933 : modelRow 7933 = 0 := by decide
theorem modelrow_agrees_7933 : modelRow 7933 = row 7933 := model_count_7933.trans row_count_7933.symm
def row_7934 : RowCertificate 7934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7934 1) := by change ¬ Nat.Prime 251809293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83936431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7934 2) := by change ¬ Nat.Prime 251793425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50358685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7934 3) := by change Nat.Prime 251777557; exact ProjectEulerSolutions.P58PrimeNodes.p_251777557.prime
    exact decide_eq_true prime
theorem row_count_7934 : row 7934 = 1 := (row_7934.checked).trans rfl
theorem model_count_7934 : modelRow 7934 = 1 := by decide
theorem modelrow_agrees_7934 : modelRow 7934 = row 7934 := model_count_7934.trans row_count_7934.symm
def row_7935 : RowCertificate 7935 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7935 1) := by change Nat.Prime 251872771; exact ProjectEulerSolutions.P58PrimeNodes.p_251872771.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7935 2) := by change ¬ Nat.Prime 251856901; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 4752017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7935 3) := by change ¬ Nat.Prime 251841031; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 19372387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7935 : row 7935 = 1 := (row_7935.checked).trans rfl
theorem model_count_7935 : modelRow 7935 = 1 := by decide
theorem modelrow_agrees_7935 : modelRow 7935 = row 7935 := model_count_7935.trans row_count_7935.symm
def row_7936 : RowCertificate 7936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7936 1) := by change ¬ Nat.Prime 251936257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 13259803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7936 2) := by change ¬ Nat.Prime 251920385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 50384077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7936 3) := by change ¬ Nat.Prime 251904513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 83968171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7936 : row 7936 = 0 := (row_7936.checked).trans rfl
theorem model_count_7936 : modelRow 7936 = 0 := by decide
theorem modelrow_agrees_7936 : modelRow 7936 = row 7936 := model_count_7936.trans row_count_7936.symm
private def atomic_7809 : CountSegment 1 10 13120 7808 7809 3310 3311 := CountSegment.single (first := 7808) (initial := 3310) row_count_7809 (by decide)
private theorem atomic_7809_agreement : RowAgreement 7808 7809 := RowAgreement.single (first := 7808) modelrow_agrees_7809
private def atomic_7810 : CountSegment 1 10 13120 7809 7810 3311 3311 := CountSegment.single (first := 7809) (initial := 3311) row_count_7810 (by decide)
private theorem atomic_7810_agreement : RowAgreement 7809 7810 := RowAgreement.single (first := 7809) modelrow_agrees_7810
private def atomic_7811 : CountSegment 1 10 13120 7810 7811 3311 3311 := CountSegment.single (first := 7810) (initial := 3311) row_count_7811 (by decide)
private theorem atomic_7811_agreement : RowAgreement 7810 7811 := RowAgreement.single (first := 7810) modelrow_agrees_7811
private def atomic_7812 : CountSegment 1 10 13120 7811 7812 3311 3313 := CountSegment.single (first := 7811) (initial := 3311) row_count_7812 (by decide)
private theorem atomic_7812_agreement : RowAgreement 7811 7812 := RowAgreement.single (first := 7811) modelrow_agrees_7812
private def atomic_7813 : CountSegment 1 10 13120 7812 7813 3313 3313 := CountSegment.single (first := 7812) (initial := 3313) row_count_7813 (by decide)
private theorem atomic_7813_agreement : RowAgreement 7812 7813 := RowAgreement.single (first := 7812) modelrow_agrees_7813
private def atomic_7814 : CountSegment 1 10 13120 7813 7814 3313 3314 := CountSegment.single (first := 7813) (initial := 3313) row_count_7814 (by decide)
private theorem atomic_7814_agreement : RowAgreement 7813 7814 := RowAgreement.single (first := 7813) modelrow_agrees_7814
private def atomic_7815 : CountSegment 1 10 13120 7814 7815 3314 3314 := CountSegment.single (first := 7814) (initial := 3314) row_count_7815 (by decide)
private theorem atomic_7815_agreement : RowAgreement 7814 7815 := RowAgreement.single (first := 7814) modelrow_agrees_7815
private def atomic_7816 : CountSegment 1 10 13120 7815 7816 3314 3314 := CountSegment.single (first := 7815) (initial := 3314) row_count_7816 (by decide)
private theorem atomic_7816_agreement : RowAgreement 7815 7816 := RowAgreement.single (first := 7815) modelrow_agrees_7816
private def atomic_7817 : CountSegment 1 10 13120 7816 7817 3314 3314 := CountSegment.single (first := 7816) (initial := 3314) row_count_7817 (by decide)
private theorem atomic_7817_agreement : RowAgreement 7816 7817 := RowAgreement.single (first := 7816) modelrow_agrees_7817
private def atomic_7818 : CountSegment 1 10 13120 7817 7818 3314 3315 := CountSegment.single (first := 7817) (initial := 3314) row_count_7818 (by decide)
private theorem atomic_7818_agreement : RowAgreement 7817 7818 := RowAgreement.single (first := 7817) modelrow_agrees_7818
private def atomic_7819 : CountSegment 1 10 13120 7818 7819 3315 3316 := CountSegment.single (first := 7818) (initial := 3315) row_count_7819 (by decide)
private theorem atomic_7819_agreement : RowAgreement 7818 7819 := RowAgreement.single (first := 7818) modelrow_agrees_7819
private def atomic_7820 : CountSegment 1 10 13120 7819 7820 3316 3317 := CountSegment.single (first := 7819) (initial := 3316) row_count_7820 (by decide)
private theorem atomic_7820_agreement : RowAgreement 7819 7820 := RowAgreement.single (first := 7819) modelrow_agrees_7820
private def atomic_7821 : CountSegment 1 10 13120 7820 7821 3317 3318 := CountSegment.single (first := 7820) (initial := 3317) row_count_7821 (by decide)
private theorem atomic_7821_agreement : RowAgreement 7820 7821 := RowAgreement.single (first := 7820) modelrow_agrees_7821
private def atomic_7822 : CountSegment 1 10 13120 7821 7822 3318 3318 := CountSegment.single (first := 7821) (initial := 3318) row_count_7822 (by decide)
private theorem atomic_7822_agreement : RowAgreement 7821 7822 := RowAgreement.single (first := 7821) modelrow_agrees_7822
private def atomic_7823 : CountSegment 1 10 13120 7822 7823 3318 3319 := CountSegment.single (first := 7822) (initial := 3318) row_count_7823 (by decide)
private theorem atomic_7823_agreement : RowAgreement 7822 7823 := RowAgreement.single (first := 7822) modelrow_agrees_7823
private def atomic_7824 : CountSegment 1 10 13120 7823 7824 3319 3319 := CountSegment.single (first := 7823) (initial := 3319) row_count_7824 (by decide)
private theorem atomic_7824_agreement : RowAgreement 7823 7824 := RowAgreement.single (first := 7823) modelrow_agrees_7824
private def atomic_7825 : CountSegment 1 10 13120 7824 7825 3319 3319 := CountSegment.single (first := 7824) (initial := 3319) row_count_7825 (by decide)
private theorem atomic_7825_agreement : RowAgreement 7824 7825 := RowAgreement.single (first := 7824) modelrow_agrees_7825
private def atomic_7826 : CountSegment 1 10 13120 7825 7826 3319 3320 := CountSegment.single (first := 7825) (initial := 3319) row_count_7826 (by decide)
private theorem atomic_7826_agreement : RowAgreement 7825 7826 := RowAgreement.single (first := 7825) modelrow_agrees_7826
private def atomic_7827 : CountSegment 1 10 13120 7826 7827 3320 3322 := CountSegment.single (first := 7826) (initial := 3320) row_count_7827 (by decide)
private theorem atomic_7827_agreement : RowAgreement 7826 7827 := RowAgreement.single (first := 7826) modelrow_agrees_7827
private def atomic_7828 : CountSegment 1 10 13120 7827 7828 3322 3323 := CountSegment.single (first := 7827) (initial := 3322) row_count_7828 (by decide)
private theorem atomic_7828_agreement : RowAgreement 7827 7828 := RowAgreement.single (first := 7827) modelrow_agrees_7828
private def atomic_7829 : CountSegment 1 10 13120 7828 7829 3323 3324 := CountSegment.single (first := 7828) (initial := 3323) row_count_7829 (by decide)
private theorem atomic_7829_agreement : RowAgreement 7828 7829 := RowAgreement.single (first := 7828) modelrow_agrees_7829
private def atomic_7830 : CountSegment 1 10 13120 7829 7830 3324 3325 := CountSegment.single (first := 7829) (initial := 3324) row_count_7830 (by decide)
private theorem atomic_7830_agreement : RowAgreement 7829 7830 := RowAgreement.single (first := 7829) modelrow_agrees_7830
private def atomic_7831 : CountSegment 1 10 13120 7830 7831 3325 3325 := CountSegment.single (first := 7830) (initial := 3325) row_count_7831 (by decide)
private theorem atomic_7831_agreement : RowAgreement 7830 7831 := RowAgreement.single (first := 7830) modelrow_agrees_7831
private def atomic_7832 : CountSegment 1 10 13120 7831 7832 3325 3326 := CountSegment.single (first := 7831) (initial := 3325) row_count_7832 (by decide)
private theorem atomic_7832_agreement : RowAgreement 7831 7832 := RowAgreement.single (first := 7831) modelrow_agrees_7832
private def atomic_7833 : CountSegment 1 10 13120 7832 7833 3326 3326 := CountSegment.single (first := 7832) (initial := 3326) row_count_7833 (by decide)
private theorem atomic_7833_agreement : RowAgreement 7832 7833 := RowAgreement.single (first := 7832) modelrow_agrees_7833
private def atomic_7834 : CountSegment 1 10 13120 7833 7834 3326 3326 := CountSegment.single (first := 7833) (initial := 3326) row_count_7834 (by decide)
private theorem atomic_7834_agreement : RowAgreement 7833 7834 := RowAgreement.single (first := 7833) modelrow_agrees_7834
private def atomic_7835 : CountSegment 1 10 13120 7834 7835 3326 3327 := CountSegment.single (first := 7834) (initial := 3326) row_count_7835 (by decide)
private theorem atomic_7835_agreement : RowAgreement 7834 7835 := RowAgreement.single (first := 7834) modelrow_agrees_7835
private def atomic_7836 : CountSegment 1 10 13120 7835 7836 3327 3327 := CountSegment.single (first := 7835) (initial := 3327) row_count_7836 (by decide)
private theorem atomic_7836_agreement : RowAgreement 7835 7836 := RowAgreement.single (first := 7835) modelrow_agrees_7836
private def atomic_7837 : CountSegment 1 10 13120 7836 7837 3327 3328 := CountSegment.single (first := 7836) (initial := 3327) row_count_7837 (by decide)
private theorem atomic_7837_agreement : RowAgreement 7836 7837 := RowAgreement.single (first := 7836) modelrow_agrees_7837
private def atomic_7838 : CountSegment 1 10 13120 7837 7838 3328 3329 := CountSegment.single (first := 7837) (initial := 3328) row_count_7838 (by decide)
private theorem atomic_7838_agreement : RowAgreement 7837 7838 := RowAgreement.single (first := 7837) modelrow_agrees_7838
private def atomic_7839 : CountSegment 1 10 13120 7838 7839 3329 3329 := CountSegment.single (first := 7838) (initial := 3329) row_count_7839 (by decide)
private theorem atomic_7839_agreement : RowAgreement 7838 7839 := RowAgreement.single (first := 7838) modelrow_agrees_7839
private def atomic_7840 : CountSegment 1 10 13120 7839 7840 3329 3330 := CountSegment.single (first := 7839) (initial := 3329) row_count_7840 (by decide)
private theorem atomic_7840_agreement : RowAgreement 7839 7840 := RowAgreement.single (first := 7839) modelrow_agrees_7840
private def atomic_7841 : CountSegment 1 10 13120 7840 7841 3330 3330 := CountSegment.single (first := 7840) (initial := 3330) row_count_7841 (by decide)
private theorem atomic_7841_agreement : RowAgreement 7840 7841 := RowAgreement.single (first := 7840) modelrow_agrees_7841
private def atomic_7842 : CountSegment 1 10 13120 7841 7842 3330 3330 := CountSegment.single (first := 7841) (initial := 3330) row_count_7842 (by decide)
private theorem atomic_7842_agreement : RowAgreement 7841 7842 := RowAgreement.single (first := 7841) modelrow_agrees_7842
private def atomic_7843 : CountSegment 1 10 13120 7842 7843 3330 3330 := CountSegment.single (first := 7842) (initial := 3330) row_count_7843 (by decide)
private theorem atomic_7843_agreement : RowAgreement 7842 7843 := RowAgreement.single (first := 7842) modelrow_agrees_7843
private def atomic_7844 : CountSegment 1 10 13120 7843 7844 3330 3330 := CountSegment.single (first := 7843) (initial := 3330) row_count_7844 (by decide)
private theorem atomic_7844_agreement : RowAgreement 7843 7844 := RowAgreement.single (first := 7843) modelrow_agrees_7844
private def atomic_7845 : CountSegment 1 10 13120 7844 7845 3330 3330 := CountSegment.single (first := 7844) (initial := 3330) row_count_7845 (by decide)
private theorem atomic_7845_agreement : RowAgreement 7844 7845 := RowAgreement.single (first := 7844) modelrow_agrees_7845
private def atomic_7846 : CountSegment 1 10 13120 7845 7846 3330 3331 := CountSegment.single (first := 7845) (initial := 3330) row_count_7846 (by decide)
private theorem atomic_7846_agreement : RowAgreement 7845 7846 := RowAgreement.single (first := 7845) modelrow_agrees_7846
private def atomic_7847 : CountSegment 1 10 13120 7846 7847 3331 3332 := CountSegment.single (first := 7846) (initial := 3331) row_count_7847 (by decide)
private theorem atomic_7847_agreement : RowAgreement 7846 7847 := RowAgreement.single (first := 7846) modelrow_agrees_7847
private def atomic_7848 : CountSegment 1 10 13120 7847 7848 3332 3333 := CountSegment.single (first := 7847) (initial := 3332) row_count_7848 (by decide)
private theorem atomic_7848_agreement : RowAgreement 7847 7848 := RowAgreement.single (first := 7847) modelrow_agrees_7848
private def atomic_7849 : CountSegment 1 10 13120 7848 7849 3333 3333 := CountSegment.single (first := 7848) (initial := 3333) row_count_7849 (by decide)
private theorem atomic_7849_agreement : RowAgreement 7848 7849 := RowAgreement.single (first := 7848) modelrow_agrees_7849
private def atomic_7850 : CountSegment 1 10 13120 7849 7850 3333 3334 := CountSegment.single (first := 7849) (initial := 3333) row_count_7850 (by decide)
private theorem atomic_7850_agreement : RowAgreement 7849 7850 := RowAgreement.single (first := 7849) modelrow_agrees_7850
private def atomic_7851 : CountSegment 1 10 13120 7850 7851 3334 3334 := CountSegment.single (first := 7850) (initial := 3334) row_count_7851 (by decide)
private theorem atomic_7851_agreement : RowAgreement 7850 7851 := RowAgreement.single (first := 7850) modelrow_agrees_7851
private def atomic_7852 : CountSegment 1 10 13120 7851 7852 3334 3334 := CountSegment.single (first := 7851) (initial := 3334) row_count_7852 (by decide)
private theorem atomic_7852_agreement : RowAgreement 7851 7852 := RowAgreement.single (first := 7851) modelrow_agrees_7852
private def atomic_7853 : CountSegment 1 10 13120 7852 7853 3334 3334 := CountSegment.single (first := 7852) (initial := 3334) row_count_7853 (by decide)
private theorem atomic_7853_agreement : RowAgreement 7852 7853 := RowAgreement.single (first := 7852) modelrow_agrees_7853
private def atomic_7854 : CountSegment 1 10 13120 7853 7854 3334 3335 := CountSegment.single (first := 7853) (initial := 3334) row_count_7854 (by decide)
private theorem atomic_7854_agreement : RowAgreement 7853 7854 := RowAgreement.single (first := 7853) modelrow_agrees_7854
private def atomic_7855 : CountSegment 1 10 13120 7854 7855 3335 3336 := CountSegment.single (first := 7854) (initial := 3335) row_count_7855 (by decide)
private theorem atomic_7855_agreement : RowAgreement 7854 7855 := RowAgreement.single (first := 7854) modelrow_agrees_7855
private def atomic_7856 : CountSegment 1 10 13120 7855 7856 3336 3336 := CountSegment.single (first := 7855) (initial := 3336) row_count_7856 (by decide)
private theorem atomic_7856_agreement : RowAgreement 7855 7856 := RowAgreement.single (first := 7855) modelrow_agrees_7856
private def atomic_7857 : CountSegment 1 10 13120 7856 7857 3336 3336 := CountSegment.single (first := 7856) (initial := 3336) row_count_7857 (by decide)
private theorem atomic_7857_agreement : RowAgreement 7856 7857 := RowAgreement.single (first := 7856) modelrow_agrees_7857
private def atomic_7858 : CountSegment 1 10 13120 7857 7858 3336 3338 := CountSegment.single (first := 7857) (initial := 3336) row_count_7858 (by decide)
private theorem atomic_7858_agreement : RowAgreement 7857 7858 := RowAgreement.single (first := 7857) modelrow_agrees_7858
private def atomic_7859 : CountSegment 1 10 13120 7858 7859 3338 3338 := CountSegment.single (first := 7858) (initial := 3338) row_count_7859 (by decide)
private theorem atomic_7859_agreement : RowAgreement 7858 7859 := RowAgreement.single (first := 7858) modelrow_agrees_7859
private def atomic_7860 : CountSegment 1 10 13120 7859 7860 3338 3338 := CountSegment.single (first := 7859) (initial := 3338) row_count_7860 (by decide)
private theorem atomic_7860_agreement : RowAgreement 7859 7860 := RowAgreement.single (first := 7859) modelrow_agrees_7860
private def atomic_7861 : CountSegment 1 10 13120 7860 7861 3338 3338 := CountSegment.single (first := 7860) (initial := 3338) row_count_7861 (by decide)
private theorem atomic_7861_agreement : RowAgreement 7860 7861 := RowAgreement.single (first := 7860) modelrow_agrees_7861
private def atomic_7862 : CountSegment 1 10 13120 7861 7862 3338 3339 := CountSegment.single (first := 7861) (initial := 3338) row_count_7862 (by decide)
private theorem atomic_7862_agreement : RowAgreement 7861 7862 := RowAgreement.single (first := 7861) modelrow_agrees_7862
private def atomic_7863 : CountSegment 1 10 13120 7862 7863 3339 3340 := CountSegment.single (first := 7862) (initial := 3339) row_count_7863 (by decide)
private theorem atomic_7863_agreement : RowAgreement 7862 7863 := RowAgreement.single (first := 7862) modelrow_agrees_7863
private def atomic_7864 : CountSegment 1 10 13120 7863 7864 3340 3340 := CountSegment.single (first := 7863) (initial := 3340) row_count_7864 (by decide)
private theorem atomic_7864_agreement : RowAgreement 7863 7864 := RowAgreement.single (first := 7863) modelrow_agrees_7864
private def atomic_7865 : CountSegment 1 10 13120 7864 7865 3340 3341 := CountSegment.single (first := 7864) (initial := 3340) row_count_7865 (by decide)
private theorem atomic_7865_agreement : RowAgreement 7864 7865 := RowAgreement.single (first := 7864) modelrow_agrees_7865
private def atomic_7866 : CountSegment 1 10 13120 7865 7866 3341 3341 := CountSegment.single (first := 7865) (initial := 3341) row_count_7866 (by decide)
private theorem atomic_7866_agreement : RowAgreement 7865 7866 := RowAgreement.single (first := 7865) modelrow_agrees_7866
private def atomic_7867 : CountSegment 1 10 13120 7866 7867 3341 3341 := CountSegment.single (first := 7866) (initial := 3341) row_count_7867 (by decide)
private theorem atomic_7867_agreement : RowAgreement 7866 7867 := RowAgreement.single (first := 7866) modelrow_agrees_7867
private def atomic_7868 : CountSegment 1 10 13120 7867 7868 3341 3341 := CountSegment.single (first := 7867) (initial := 3341) row_count_7868 (by decide)
private theorem atomic_7868_agreement : RowAgreement 7867 7868 := RowAgreement.single (first := 7867) modelrow_agrees_7868
private def atomic_7869 : CountSegment 1 10 13120 7868 7869 3341 3342 := CountSegment.single (first := 7868) (initial := 3341) row_count_7869 (by decide)
private theorem atomic_7869_agreement : RowAgreement 7868 7869 := RowAgreement.single (first := 7868) modelrow_agrees_7869
private def atomic_7870 : CountSegment 1 10 13120 7869 7870 3342 3343 := CountSegment.single (first := 7869) (initial := 3342) row_count_7870 (by decide)
private theorem atomic_7870_agreement : RowAgreement 7869 7870 := RowAgreement.single (first := 7869) modelrow_agrees_7870
private def atomic_7871 : CountSegment 1 10 13120 7870 7871 3343 3343 := CountSegment.single (first := 7870) (initial := 3343) row_count_7871 (by decide)
private theorem atomic_7871_agreement : RowAgreement 7870 7871 := RowAgreement.single (first := 7870) modelrow_agrees_7871
private def atomic_7872 : CountSegment 1 10 13120 7871 7872 3343 3344 := CountSegment.single (first := 7871) (initial := 3343) row_count_7872 (by decide)
private theorem atomic_7872_agreement : RowAgreement 7871 7872 := RowAgreement.single (first := 7871) modelrow_agrees_7872
private def atomic_7873 : CountSegment 1 10 13120 7872 7873 3344 3344 := CountSegment.single (first := 7872) (initial := 3344) row_count_7873 (by decide)
private theorem atomic_7873_agreement : RowAgreement 7872 7873 := RowAgreement.single (first := 7872) modelrow_agrees_7873
private def atomic_7874 : CountSegment 1 10 13120 7873 7874 3344 3344 := CountSegment.single (first := 7873) (initial := 3344) row_count_7874 (by decide)
private theorem atomic_7874_agreement : RowAgreement 7873 7874 := RowAgreement.single (first := 7873) modelrow_agrees_7874
private def atomic_7875 : CountSegment 1 10 13120 7874 7875 3344 3344 := CountSegment.single (first := 7874) (initial := 3344) row_count_7875 (by decide)
private theorem atomic_7875_agreement : RowAgreement 7874 7875 := RowAgreement.single (first := 7874) modelrow_agrees_7875
private def atomic_7876 : CountSegment 1 10 13120 7875 7876 3344 3344 := CountSegment.single (first := 7875) (initial := 3344) row_count_7876 (by decide)
private theorem atomic_7876_agreement : RowAgreement 7875 7876 := RowAgreement.single (first := 7875) modelrow_agrees_7876
private def atomic_7877 : CountSegment 1 10 13120 7876 7877 3344 3345 := CountSegment.single (first := 7876) (initial := 3344) row_count_7877 (by decide)
private theorem atomic_7877_agreement : RowAgreement 7876 7877 := RowAgreement.single (first := 7876) modelrow_agrees_7877
private def atomic_7878 : CountSegment 1 10 13120 7877 7878 3345 3346 := CountSegment.single (first := 7877) (initial := 3345) row_count_7878 (by decide)
private theorem atomic_7878_agreement : RowAgreement 7877 7878 := RowAgreement.single (first := 7877) modelrow_agrees_7878
private def atomic_7879 : CountSegment 1 10 13120 7878 7879 3346 3346 := CountSegment.single (first := 7878) (initial := 3346) row_count_7879 (by decide)
private theorem atomic_7879_agreement : RowAgreement 7878 7879 := RowAgreement.single (first := 7878) modelrow_agrees_7879
private def atomic_7880 : CountSegment 1 10 13120 7879 7880 3346 3346 := CountSegment.single (first := 7879) (initial := 3346) row_count_7880 (by decide)
private theorem atomic_7880_agreement : RowAgreement 7879 7880 := RowAgreement.single (first := 7879) modelrow_agrees_7880
private def atomic_7881 : CountSegment 1 10 13120 7880 7881 3346 3346 := CountSegment.single (first := 7880) (initial := 3346) row_count_7881 (by decide)
private theorem atomic_7881_agreement : RowAgreement 7880 7881 := RowAgreement.single (first := 7880) modelrow_agrees_7881
private def atomic_7882 : CountSegment 1 10 13120 7881 7882 3346 3346 := CountSegment.single (first := 7881) (initial := 3346) row_count_7882 (by decide)
private theorem atomic_7882_agreement : RowAgreement 7881 7882 := RowAgreement.single (first := 7881) modelrow_agrees_7882
private def atomic_7883 : CountSegment 1 10 13120 7882 7883 3346 3346 := CountSegment.single (first := 7882) (initial := 3346) row_count_7883 (by decide)
private theorem atomic_7883_agreement : RowAgreement 7882 7883 := RowAgreement.single (first := 7882) modelrow_agrees_7883
private def atomic_7884 : CountSegment 1 10 13120 7883 7884 3346 3346 := CountSegment.single (first := 7883) (initial := 3346) row_count_7884 (by decide)
private theorem atomic_7884_agreement : RowAgreement 7883 7884 := RowAgreement.single (first := 7883) modelrow_agrees_7884
private def atomic_7885 : CountSegment 1 10 13120 7884 7885 3346 3346 := CountSegment.single (first := 7884) (initial := 3346) row_count_7885 (by decide)
private theorem atomic_7885_agreement : RowAgreement 7884 7885 := RowAgreement.single (first := 7884) modelrow_agrees_7885
private def atomic_7886 : CountSegment 1 10 13120 7885 7886 3346 3347 := CountSegment.single (first := 7885) (initial := 3346) row_count_7886 (by decide)
private theorem atomic_7886_agreement : RowAgreement 7885 7886 := RowAgreement.single (first := 7885) modelrow_agrees_7886
private def atomic_7887 : CountSegment 1 10 13120 7886 7887 3347 3347 := CountSegment.single (first := 7886) (initial := 3347) row_count_7887 (by decide)
private theorem atomic_7887_agreement : RowAgreement 7886 7887 := RowAgreement.single (first := 7886) modelrow_agrees_7887
private def atomic_7888 : CountSegment 1 10 13120 7887 7888 3347 3347 := CountSegment.single (first := 7887) (initial := 3347) row_count_7888 (by decide)
private theorem atomic_7888_agreement : RowAgreement 7887 7888 := RowAgreement.single (first := 7887) modelrow_agrees_7888
private def atomic_7889 : CountSegment 1 10 13120 7888 7889 3347 3347 := CountSegment.single (first := 7888) (initial := 3347) row_count_7889 (by decide)
private theorem atomic_7889_agreement : RowAgreement 7888 7889 := RowAgreement.single (first := 7888) modelrow_agrees_7889
private def atomic_7890 : CountSegment 1 10 13120 7889 7890 3347 3347 := CountSegment.single (first := 7889) (initial := 3347) row_count_7890 (by decide)
private theorem atomic_7890_agreement : RowAgreement 7889 7890 := RowAgreement.single (first := 7889) modelrow_agrees_7890
private def atomic_7891 : CountSegment 1 10 13120 7890 7891 3347 3347 := CountSegment.single (first := 7890) (initial := 3347) row_count_7891 (by decide)
private theorem atomic_7891_agreement : RowAgreement 7890 7891 := RowAgreement.single (first := 7890) modelrow_agrees_7891
private def atomic_7892 : CountSegment 1 10 13120 7891 7892 3347 3349 := CountSegment.single (first := 7891) (initial := 3347) row_count_7892 (by decide)
private theorem atomic_7892_agreement : RowAgreement 7891 7892 := RowAgreement.single (first := 7891) modelrow_agrees_7892
private def atomic_7893 : CountSegment 1 10 13120 7892 7893 3349 3349 := CountSegment.single (first := 7892) (initial := 3349) row_count_7893 (by decide)
private theorem atomic_7893_agreement : RowAgreement 7892 7893 := RowAgreement.single (first := 7892) modelrow_agrees_7893
private def atomic_7894 : CountSegment 1 10 13120 7893 7894 3349 3349 := CountSegment.single (first := 7893) (initial := 3349) row_count_7894 (by decide)
private theorem atomic_7894_agreement : RowAgreement 7893 7894 := RowAgreement.single (first := 7893) modelrow_agrees_7894
private def atomic_7895 : CountSegment 1 10 13120 7894 7895 3349 3349 := CountSegment.single (first := 7894) (initial := 3349) row_count_7895 (by decide)
private theorem atomic_7895_agreement : RowAgreement 7894 7895 := RowAgreement.single (first := 7894) modelrow_agrees_7895
private def atomic_7896 : CountSegment 1 10 13120 7895 7896 3349 3349 := CountSegment.single (first := 7895) (initial := 3349) row_count_7896 (by decide)
private theorem atomic_7896_agreement : RowAgreement 7895 7896 := RowAgreement.single (first := 7895) modelrow_agrees_7896
private def atomic_7897 : CountSegment 1 10 13120 7896 7897 3349 3349 := CountSegment.single (first := 7896) (initial := 3349) row_count_7897 (by decide)
private theorem atomic_7897_agreement : RowAgreement 7896 7897 := RowAgreement.single (first := 7896) modelrow_agrees_7897
private def atomic_7898 : CountSegment 1 10 13120 7897 7898 3349 3350 := CountSegment.single (first := 7897) (initial := 3349) row_count_7898 (by decide)
private theorem atomic_7898_agreement : RowAgreement 7897 7898 := RowAgreement.single (first := 7897) modelrow_agrees_7898
private def atomic_7899 : CountSegment 1 10 13120 7898 7899 3350 3351 := CountSegment.single (first := 7898) (initial := 3350) row_count_7899 (by decide)
private theorem atomic_7899_agreement : RowAgreement 7898 7899 := RowAgreement.single (first := 7898) modelrow_agrees_7899
private def atomic_7900 : CountSegment 1 10 13120 7899 7900 3351 3351 := CountSegment.single (first := 7899) (initial := 3351) row_count_7900 (by decide)
private theorem atomic_7900_agreement : RowAgreement 7899 7900 := RowAgreement.single (first := 7899) modelrow_agrees_7900
private def atomic_7901 : CountSegment 1 10 13120 7900 7901 3351 3351 := CountSegment.single (first := 7900) (initial := 3351) row_count_7901 (by decide)
private theorem atomic_7901_agreement : RowAgreement 7900 7901 := RowAgreement.single (first := 7900) modelrow_agrees_7901
private def atomic_7902 : CountSegment 1 10 13120 7901 7902 3351 3352 := CountSegment.single (first := 7901) (initial := 3351) row_count_7902 (by decide)
private theorem atomic_7902_agreement : RowAgreement 7901 7902 := RowAgreement.single (first := 7901) modelrow_agrees_7902
private def atomic_7903 : CountSegment 1 10 13120 7902 7903 3352 3352 := CountSegment.single (first := 7902) (initial := 3352) row_count_7903 (by decide)
private theorem atomic_7903_agreement : RowAgreement 7902 7903 := RowAgreement.single (first := 7902) modelrow_agrees_7903
private def atomic_7904 : CountSegment 1 10 13120 7903 7904 3352 3352 := CountSegment.single (first := 7903) (initial := 3352) row_count_7904 (by decide)
private theorem atomic_7904_agreement : RowAgreement 7903 7904 := RowAgreement.single (first := 7903) modelrow_agrees_7904
private def atomic_7905 : CountSegment 1 10 13120 7904 7905 3352 3352 := CountSegment.single (first := 7904) (initial := 3352) row_count_7905 (by decide)
private theorem atomic_7905_agreement : RowAgreement 7904 7905 := RowAgreement.single (first := 7904) modelrow_agrees_7905
private def atomic_7906 : CountSegment 1 10 13120 7905 7906 3352 3352 := CountSegment.single (first := 7905) (initial := 3352) row_count_7906 (by decide)
private theorem atomic_7906_agreement : RowAgreement 7905 7906 := RowAgreement.single (first := 7905) modelrow_agrees_7906
private def atomic_7907 : CountSegment 1 10 13120 7906 7907 3352 3352 := CountSegment.single (first := 7906) (initial := 3352) row_count_7907 (by decide)
private theorem atomic_7907_agreement : RowAgreement 7906 7907 := RowAgreement.single (first := 7906) modelrow_agrees_7907
private def atomic_7908 : CountSegment 1 10 13120 7907 7908 3352 3352 := CountSegment.single (first := 7907) (initial := 3352) row_count_7908 (by decide)
private theorem atomic_7908_agreement : RowAgreement 7907 7908 := RowAgreement.single (first := 7907) modelrow_agrees_7908
private def atomic_7909 : CountSegment 1 10 13120 7908 7909 3352 3352 := CountSegment.single (first := 7908) (initial := 3352) row_count_7909 (by decide)
private theorem atomic_7909_agreement : RowAgreement 7908 7909 := RowAgreement.single (first := 7908) modelrow_agrees_7909
private def atomic_7910 : CountSegment 1 10 13120 7909 7910 3352 3353 := CountSegment.single (first := 7909) (initial := 3352) row_count_7910 (by decide)
private theorem atomic_7910_agreement : RowAgreement 7909 7910 := RowAgreement.single (first := 7909) modelrow_agrees_7910
private def atomic_7911 : CountSegment 1 10 13120 7910 7911 3353 3354 := CountSegment.single (first := 7910) (initial := 3353) row_count_7911 (by decide)
private theorem atomic_7911_agreement : RowAgreement 7910 7911 := RowAgreement.single (first := 7910) modelrow_agrees_7911
private def atomic_7912 : CountSegment 1 10 13120 7911 7912 3354 3354 := CountSegment.single (first := 7911) (initial := 3354) row_count_7912 (by decide)
private theorem atomic_7912_agreement : RowAgreement 7911 7912 := RowAgreement.single (first := 7911) modelrow_agrees_7912
private def atomic_7913 : CountSegment 1 10 13120 7912 7913 3354 3354 := CountSegment.single (first := 7912) (initial := 3354) row_count_7913 (by decide)
private theorem atomic_7913_agreement : RowAgreement 7912 7913 := RowAgreement.single (first := 7912) modelrow_agrees_7913
private def atomic_7914 : CountSegment 1 10 13120 7913 7914 3354 3354 := CountSegment.single (first := 7913) (initial := 3354) row_count_7914 (by decide)
private theorem atomic_7914_agreement : RowAgreement 7913 7914 := RowAgreement.single (first := 7913) modelrow_agrees_7914
private def atomic_7915 : CountSegment 1 10 13120 7914 7915 3354 3354 := CountSegment.single (first := 7914) (initial := 3354) row_count_7915 (by decide)
private theorem atomic_7915_agreement : RowAgreement 7914 7915 := RowAgreement.single (first := 7914) modelrow_agrees_7915
private def atomic_7916 : CountSegment 1 10 13120 7915 7916 3354 3354 := CountSegment.single (first := 7915) (initial := 3354) row_count_7916 (by decide)
private theorem atomic_7916_agreement : RowAgreement 7915 7916 := RowAgreement.single (first := 7915) modelrow_agrees_7916
private def atomic_7917 : CountSegment 1 10 13120 7916 7917 3354 3355 := CountSegment.single (first := 7916) (initial := 3354) row_count_7917 (by decide)
private theorem atomic_7917_agreement : RowAgreement 7916 7917 := RowAgreement.single (first := 7916) modelrow_agrees_7917
private def atomic_7918 : CountSegment 1 10 13120 7917 7918 3355 3355 := CountSegment.single (first := 7917) (initial := 3355) row_count_7918 (by decide)
private theorem atomic_7918_agreement : RowAgreement 7917 7918 := RowAgreement.single (first := 7917) modelrow_agrees_7918
private def atomic_7919 : CountSegment 1 10 13120 7918 7919 3355 3355 := CountSegment.single (first := 7918) (initial := 3355) row_count_7919 (by decide)
private theorem atomic_7919_agreement : RowAgreement 7918 7919 := RowAgreement.single (first := 7918) modelrow_agrees_7919
private def atomic_7920 : CountSegment 1 10 13120 7919 7920 3355 3355 := CountSegment.single (first := 7919) (initial := 3355) row_count_7920 (by decide)
private theorem atomic_7920_agreement : RowAgreement 7919 7920 := RowAgreement.single (first := 7919) modelrow_agrees_7920
private def atomic_7921 : CountSegment 1 10 13120 7920 7921 3355 3356 := CountSegment.single (first := 7920) (initial := 3355) row_count_7921 (by decide)
private theorem atomic_7921_agreement : RowAgreement 7920 7921 := RowAgreement.single (first := 7920) modelrow_agrees_7921
private def atomic_7922 : CountSegment 1 10 13120 7921 7922 3356 3357 := CountSegment.single (first := 7921) (initial := 3356) row_count_7922 (by decide)
private theorem atomic_7922_agreement : RowAgreement 7921 7922 := RowAgreement.single (first := 7921) modelrow_agrees_7922
private def atomic_7923 : CountSegment 1 10 13120 7922 7923 3357 3357 := CountSegment.single (first := 7922) (initial := 3357) row_count_7923 (by decide)
private theorem atomic_7923_agreement : RowAgreement 7922 7923 := RowAgreement.single (first := 7922) modelrow_agrees_7923
private def atomic_7924 : CountSegment 1 10 13120 7923 7924 3357 3358 := CountSegment.single (first := 7923) (initial := 3357) row_count_7924 (by decide)
private theorem atomic_7924_agreement : RowAgreement 7923 7924 := RowAgreement.single (first := 7923) modelrow_agrees_7924
private def atomic_7925 : CountSegment 1 10 13120 7924 7925 3358 3358 := CountSegment.single (first := 7924) (initial := 3358) row_count_7925 (by decide)
private theorem atomic_7925_agreement : RowAgreement 7924 7925 := RowAgreement.single (first := 7924) modelrow_agrees_7925
private def atomic_7926 : CountSegment 1 10 13120 7925 7926 3358 3359 := CountSegment.single (first := 7925) (initial := 3358) row_count_7926 (by decide)
private theorem atomic_7926_agreement : RowAgreement 7925 7926 := RowAgreement.single (first := 7925) modelrow_agrees_7926
private def atomic_7927 : CountSegment 1 10 13120 7926 7927 3359 3360 := CountSegment.single (first := 7926) (initial := 3359) row_count_7927 (by decide)
private theorem atomic_7927_agreement : RowAgreement 7926 7927 := RowAgreement.single (first := 7926) modelrow_agrees_7927
private def atomic_7928 : CountSegment 1 10 13120 7927 7928 3360 3361 := CountSegment.single (first := 7927) (initial := 3360) row_count_7928 (by decide)
private theorem atomic_7928_agreement : RowAgreement 7927 7928 := RowAgreement.single (first := 7927) modelrow_agrees_7928
private def atomic_7929 : CountSegment 1 10 13120 7928 7929 3361 3361 := CountSegment.single (first := 7928) (initial := 3361) row_count_7929 (by decide)
private theorem atomic_7929_agreement : RowAgreement 7928 7929 := RowAgreement.single (first := 7928) modelrow_agrees_7929
private def atomic_7930 : CountSegment 1 10 13120 7929 7930 3361 3361 := CountSegment.single (first := 7929) (initial := 3361) row_count_7930 (by decide)
private theorem atomic_7930_agreement : RowAgreement 7929 7930 := RowAgreement.single (first := 7929) modelrow_agrees_7930
private def atomic_7931 : CountSegment 1 10 13120 7930 7931 3361 3361 := CountSegment.single (first := 7930) (initial := 3361) row_count_7931 (by decide)
private theorem atomic_7931_agreement : RowAgreement 7930 7931 := RowAgreement.single (first := 7930) modelrow_agrees_7931
private def atomic_7932 : CountSegment 1 10 13120 7931 7932 3361 3361 := CountSegment.single (first := 7931) (initial := 3361) row_count_7932 (by decide)
private theorem atomic_7932_agreement : RowAgreement 7931 7932 := RowAgreement.single (first := 7931) modelrow_agrees_7932
private def atomic_7933 : CountSegment 1 10 13120 7932 7933 3361 3361 := CountSegment.single (first := 7932) (initial := 3361) row_count_7933 (by decide)
private theorem atomic_7933_agreement : RowAgreement 7932 7933 := RowAgreement.single (first := 7932) modelrow_agrees_7933
private def atomic_7934 : CountSegment 1 10 13120 7933 7934 3361 3362 := CountSegment.single (first := 7933) (initial := 3361) row_count_7934 (by decide)
private theorem atomic_7934_agreement : RowAgreement 7933 7934 := RowAgreement.single (first := 7933) modelrow_agrees_7934
private def atomic_7935 : CountSegment 1 10 13120 7934 7935 3362 3363 := CountSegment.single (first := 7934) (initial := 3362) row_count_7935 (by decide)
private theorem atomic_7935_agreement : RowAgreement 7934 7935 := RowAgreement.single (first := 7934) modelrow_agrees_7935
private def atomic_7936 : CountSegment 1 10 13120 7935 7936 3363 3363 := CountSegment.single (first := 7935) (initial := 3363) row_count_7936 (by decide)
private theorem atomic_7936_agreement : RowAgreement 7935 7936 := RowAgreement.single (first := 7935) modelrow_agrees_7936
private def within_061_000 : CountSegment 1 10 13120 7808 7810 3310 3311 := CountSegment.join atomic_7809 atomic_7810
private theorem within_061_000_agreement : RowAgreement 7808 7810 := RowAgreement.join atomic_7809_agreement atomic_7810_agreement
private def within_061_001 : CountSegment 1 10 13120 7810 7812 3311 3313 := CountSegment.join atomic_7811 atomic_7812
private theorem within_061_001_agreement : RowAgreement 7810 7812 := RowAgreement.join atomic_7811_agreement atomic_7812_agreement
private def within_061_002 : CountSegment 1 10 13120 7808 7812 3310 3313 := CountSegment.join within_061_000 within_061_001
private theorem within_061_002_agreement : RowAgreement 7808 7812 := RowAgreement.join within_061_000_agreement within_061_001_agreement
private def within_061_003 : CountSegment 1 10 13120 7812 7814 3313 3314 := CountSegment.join atomic_7813 atomic_7814
private theorem within_061_003_agreement : RowAgreement 7812 7814 := RowAgreement.join atomic_7813_agreement atomic_7814_agreement
private def within_061_004 : CountSegment 1 10 13120 7814 7816 3314 3314 := CountSegment.join atomic_7815 atomic_7816
private theorem within_061_004_agreement : RowAgreement 7814 7816 := RowAgreement.join atomic_7815_agreement atomic_7816_agreement
private def within_061_005 : CountSegment 1 10 13120 7812 7816 3313 3314 := CountSegment.join within_061_003 within_061_004
private theorem within_061_005_agreement : RowAgreement 7812 7816 := RowAgreement.join within_061_003_agreement within_061_004_agreement
private def within_061_006 : CountSegment 1 10 13120 7808 7816 3310 3314 := CountSegment.join within_061_002 within_061_005
private theorem within_061_006_agreement : RowAgreement 7808 7816 := RowAgreement.join within_061_002_agreement within_061_005_agreement
private def within_061_007 : CountSegment 1 10 13120 7816 7818 3314 3315 := CountSegment.join atomic_7817 atomic_7818
private theorem within_061_007_agreement : RowAgreement 7816 7818 := RowAgreement.join atomic_7817_agreement atomic_7818_agreement
private def within_061_008 : CountSegment 1 10 13120 7818 7820 3315 3317 := CountSegment.join atomic_7819 atomic_7820
private theorem within_061_008_agreement : RowAgreement 7818 7820 := RowAgreement.join atomic_7819_agreement atomic_7820_agreement
private def within_061_009 : CountSegment 1 10 13120 7816 7820 3314 3317 := CountSegment.join within_061_007 within_061_008
private theorem within_061_009_agreement : RowAgreement 7816 7820 := RowAgreement.join within_061_007_agreement within_061_008_agreement
private def within_061_010 : CountSegment 1 10 13120 7820 7822 3317 3318 := CountSegment.join atomic_7821 atomic_7822
private theorem within_061_010_agreement : RowAgreement 7820 7822 := RowAgreement.join atomic_7821_agreement atomic_7822_agreement
private def within_061_011 : CountSegment 1 10 13120 7822 7824 3318 3319 := CountSegment.join atomic_7823 atomic_7824
private theorem within_061_011_agreement : RowAgreement 7822 7824 := RowAgreement.join atomic_7823_agreement atomic_7824_agreement
private def within_061_012 : CountSegment 1 10 13120 7820 7824 3317 3319 := CountSegment.join within_061_010 within_061_011
private theorem within_061_012_agreement : RowAgreement 7820 7824 := RowAgreement.join within_061_010_agreement within_061_011_agreement
private def within_061_013 : CountSegment 1 10 13120 7816 7824 3314 3319 := CountSegment.join within_061_009 within_061_012
private theorem within_061_013_agreement : RowAgreement 7816 7824 := RowAgreement.join within_061_009_agreement within_061_012_agreement
private def within_061_014 : CountSegment 1 10 13120 7808 7824 3310 3319 := CountSegment.join within_061_006 within_061_013
private theorem within_061_014_agreement : RowAgreement 7808 7824 := RowAgreement.join within_061_006_agreement within_061_013_agreement
private def within_061_015 : CountSegment 1 10 13120 7824 7826 3319 3320 := CountSegment.join atomic_7825 atomic_7826
private theorem within_061_015_agreement : RowAgreement 7824 7826 := RowAgreement.join atomic_7825_agreement atomic_7826_agreement
private def within_061_016 : CountSegment 1 10 13120 7826 7828 3320 3323 := CountSegment.join atomic_7827 atomic_7828
private theorem within_061_016_agreement : RowAgreement 7826 7828 := RowAgreement.join atomic_7827_agreement atomic_7828_agreement
private def within_061_017 : CountSegment 1 10 13120 7824 7828 3319 3323 := CountSegment.join within_061_015 within_061_016
private theorem within_061_017_agreement : RowAgreement 7824 7828 := RowAgreement.join within_061_015_agreement within_061_016_agreement
private def within_061_018 : CountSegment 1 10 13120 7828 7830 3323 3325 := CountSegment.join atomic_7829 atomic_7830
private theorem within_061_018_agreement : RowAgreement 7828 7830 := RowAgreement.join atomic_7829_agreement atomic_7830_agreement
private def within_061_019 : CountSegment 1 10 13120 7830 7832 3325 3326 := CountSegment.join atomic_7831 atomic_7832
private theorem within_061_019_agreement : RowAgreement 7830 7832 := RowAgreement.join atomic_7831_agreement atomic_7832_agreement
private def within_061_020 : CountSegment 1 10 13120 7828 7832 3323 3326 := CountSegment.join within_061_018 within_061_019
private theorem within_061_020_agreement : RowAgreement 7828 7832 := RowAgreement.join within_061_018_agreement within_061_019_agreement
private def within_061_021 : CountSegment 1 10 13120 7824 7832 3319 3326 := CountSegment.join within_061_017 within_061_020
private theorem within_061_021_agreement : RowAgreement 7824 7832 := RowAgreement.join within_061_017_agreement within_061_020_agreement
private def within_061_022 : CountSegment 1 10 13120 7832 7834 3326 3326 := CountSegment.join atomic_7833 atomic_7834
private theorem within_061_022_agreement : RowAgreement 7832 7834 := RowAgreement.join atomic_7833_agreement atomic_7834_agreement
private def within_061_023 : CountSegment 1 10 13120 7834 7836 3326 3327 := CountSegment.join atomic_7835 atomic_7836
private theorem within_061_023_agreement : RowAgreement 7834 7836 := RowAgreement.join atomic_7835_agreement atomic_7836_agreement
private def within_061_024 : CountSegment 1 10 13120 7832 7836 3326 3327 := CountSegment.join within_061_022 within_061_023
private theorem within_061_024_agreement : RowAgreement 7832 7836 := RowAgreement.join within_061_022_agreement within_061_023_agreement
private def within_061_025 : CountSegment 1 10 13120 7836 7838 3327 3329 := CountSegment.join atomic_7837 atomic_7838
private theorem within_061_025_agreement : RowAgreement 7836 7838 := RowAgreement.join atomic_7837_agreement atomic_7838_agreement
private def within_061_026 : CountSegment 1 10 13120 7838 7840 3329 3330 := CountSegment.join atomic_7839 atomic_7840
private theorem within_061_026_agreement : RowAgreement 7838 7840 := RowAgreement.join atomic_7839_agreement atomic_7840_agreement
private def within_061_027 : CountSegment 1 10 13120 7836 7840 3327 3330 := CountSegment.join within_061_025 within_061_026
private theorem within_061_027_agreement : RowAgreement 7836 7840 := RowAgreement.join within_061_025_agreement within_061_026_agreement
private def within_061_028 : CountSegment 1 10 13120 7832 7840 3326 3330 := CountSegment.join within_061_024 within_061_027
private theorem within_061_028_agreement : RowAgreement 7832 7840 := RowAgreement.join within_061_024_agreement within_061_027_agreement
private def within_061_029 : CountSegment 1 10 13120 7824 7840 3319 3330 := CountSegment.join within_061_021 within_061_028
private theorem within_061_029_agreement : RowAgreement 7824 7840 := RowAgreement.join within_061_021_agreement within_061_028_agreement
private def within_061_030 : CountSegment 1 10 13120 7808 7840 3310 3330 := CountSegment.join within_061_014 within_061_029
private theorem within_061_030_agreement : RowAgreement 7808 7840 := RowAgreement.join within_061_014_agreement within_061_029_agreement
private def within_061_031 : CountSegment 1 10 13120 7840 7842 3330 3330 := CountSegment.join atomic_7841 atomic_7842
private theorem within_061_031_agreement : RowAgreement 7840 7842 := RowAgreement.join atomic_7841_agreement atomic_7842_agreement
private def within_061_032 : CountSegment 1 10 13120 7842 7844 3330 3330 := CountSegment.join atomic_7843 atomic_7844
private theorem within_061_032_agreement : RowAgreement 7842 7844 := RowAgreement.join atomic_7843_agreement atomic_7844_agreement
private def within_061_033 : CountSegment 1 10 13120 7840 7844 3330 3330 := CountSegment.join within_061_031 within_061_032
private theorem within_061_033_agreement : RowAgreement 7840 7844 := RowAgreement.join within_061_031_agreement within_061_032_agreement
private def within_061_034 : CountSegment 1 10 13120 7844 7846 3330 3331 := CountSegment.join atomic_7845 atomic_7846
private theorem within_061_034_agreement : RowAgreement 7844 7846 := RowAgreement.join atomic_7845_agreement atomic_7846_agreement
private def within_061_035 : CountSegment 1 10 13120 7846 7848 3331 3333 := CountSegment.join atomic_7847 atomic_7848
private theorem within_061_035_agreement : RowAgreement 7846 7848 := RowAgreement.join atomic_7847_agreement atomic_7848_agreement
private def within_061_036 : CountSegment 1 10 13120 7844 7848 3330 3333 := CountSegment.join within_061_034 within_061_035
private theorem within_061_036_agreement : RowAgreement 7844 7848 := RowAgreement.join within_061_034_agreement within_061_035_agreement
private def within_061_037 : CountSegment 1 10 13120 7840 7848 3330 3333 := CountSegment.join within_061_033 within_061_036
private theorem within_061_037_agreement : RowAgreement 7840 7848 := RowAgreement.join within_061_033_agreement within_061_036_agreement
private def within_061_038 : CountSegment 1 10 13120 7848 7850 3333 3334 := CountSegment.join atomic_7849 atomic_7850
private theorem within_061_038_agreement : RowAgreement 7848 7850 := RowAgreement.join atomic_7849_agreement atomic_7850_agreement
private def within_061_039 : CountSegment 1 10 13120 7850 7852 3334 3334 := CountSegment.join atomic_7851 atomic_7852
private theorem within_061_039_agreement : RowAgreement 7850 7852 := RowAgreement.join atomic_7851_agreement atomic_7852_agreement
private def within_061_040 : CountSegment 1 10 13120 7848 7852 3333 3334 := CountSegment.join within_061_038 within_061_039
private theorem within_061_040_agreement : RowAgreement 7848 7852 := RowAgreement.join within_061_038_agreement within_061_039_agreement
private def within_061_041 : CountSegment 1 10 13120 7852 7854 3334 3335 := CountSegment.join atomic_7853 atomic_7854
private theorem within_061_041_agreement : RowAgreement 7852 7854 := RowAgreement.join atomic_7853_agreement atomic_7854_agreement
private def within_061_042 : CountSegment 1 10 13120 7854 7856 3335 3336 := CountSegment.join atomic_7855 atomic_7856
private theorem within_061_042_agreement : RowAgreement 7854 7856 := RowAgreement.join atomic_7855_agreement atomic_7856_agreement
private def within_061_043 : CountSegment 1 10 13120 7852 7856 3334 3336 := CountSegment.join within_061_041 within_061_042
private theorem within_061_043_agreement : RowAgreement 7852 7856 := RowAgreement.join within_061_041_agreement within_061_042_agreement
private def within_061_044 : CountSegment 1 10 13120 7848 7856 3333 3336 := CountSegment.join within_061_040 within_061_043
private theorem within_061_044_agreement : RowAgreement 7848 7856 := RowAgreement.join within_061_040_agreement within_061_043_agreement
private def within_061_045 : CountSegment 1 10 13120 7840 7856 3330 3336 := CountSegment.join within_061_037 within_061_044
private theorem within_061_045_agreement : RowAgreement 7840 7856 := RowAgreement.join within_061_037_agreement within_061_044_agreement
private def within_061_046 : CountSegment 1 10 13120 7856 7858 3336 3338 := CountSegment.join atomic_7857 atomic_7858
private theorem within_061_046_agreement : RowAgreement 7856 7858 := RowAgreement.join atomic_7857_agreement atomic_7858_agreement
private def within_061_047 : CountSegment 1 10 13120 7858 7860 3338 3338 := CountSegment.join atomic_7859 atomic_7860
private theorem within_061_047_agreement : RowAgreement 7858 7860 := RowAgreement.join atomic_7859_agreement atomic_7860_agreement
private def within_061_048 : CountSegment 1 10 13120 7856 7860 3336 3338 := CountSegment.join within_061_046 within_061_047
private theorem within_061_048_agreement : RowAgreement 7856 7860 := RowAgreement.join within_061_046_agreement within_061_047_agreement
private def within_061_049 : CountSegment 1 10 13120 7860 7862 3338 3339 := CountSegment.join atomic_7861 atomic_7862
private theorem within_061_049_agreement : RowAgreement 7860 7862 := RowAgreement.join atomic_7861_agreement atomic_7862_agreement
private def within_061_050 : CountSegment 1 10 13120 7862 7864 3339 3340 := CountSegment.join atomic_7863 atomic_7864
private theorem within_061_050_agreement : RowAgreement 7862 7864 := RowAgreement.join atomic_7863_agreement atomic_7864_agreement
private def within_061_051 : CountSegment 1 10 13120 7860 7864 3338 3340 := CountSegment.join within_061_049 within_061_050
private theorem within_061_051_agreement : RowAgreement 7860 7864 := RowAgreement.join within_061_049_agreement within_061_050_agreement
private def within_061_052 : CountSegment 1 10 13120 7856 7864 3336 3340 := CountSegment.join within_061_048 within_061_051
private theorem within_061_052_agreement : RowAgreement 7856 7864 := RowAgreement.join within_061_048_agreement within_061_051_agreement
private def within_061_053 : CountSegment 1 10 13120 7864 7866 3340 3341 := CountSegment.join atomic_7865 atomic_7866
private theorem within_061_053_agreement : RowAgreement 7864 7866 := RowAgreement.join atomic_7865_agreement atomic_7866_agreement
private def within_061_054 : CountSegment 1 10 13120 7866 7868 3341 3341 := CountSegment.join atomic_7867 atomic_7868
private theorem within_061_054_agreement : RowAgreement 7866 7868 := RowAgreement.join atomic_7867_agreement atomic_7868_agreement
private def within_061_055 : CountSegment 1 10 13120 7864 7868 3340 3341 := CountSegment.join within_061_053 within_061_054
private theorem within_061_055_agreement : RowAgreement 7864 7868 := RowAgreement.join within_061_053_agreement within_061_054_agreement
private def within_061_056 : CountSegment 1 10 13120 7868 7870 3341 3343 := CountSegment.join atomic_7869 atomic_7870
private theorem within_061_056_agreement : RowAgreement 7868 7870 := RowAgreement.join atomic_7869_agreement atomic_7870_agreement
private def within_061_057 : CountSegment 1 10 13120 7870 7872 3343 3344 := CountSegment.join atomic_7871 atomic_7872
private theorem within_061_057_agreement : RowAgreement 7870 7872 := RowAgreement.join atomic_7871_agreement atomic_7872_agreement
private def within_061_058 : CountSegment 1 10 13120 7868 7872 3341 3344 := CountSegment.join within_061_056 within_061_057
private theorem within_061_058_agreement : RowAgreement 7868 7872 := RowAgreement.join within_061_056_agreement within_061_057_agreement
private def within_061_059 : CountSegment 1 10 13120 7864 7872 3340 3344 := CountSegment.join within_061_055 within_061_058
private theorem within_061_059_agreement : RowAgreement 7864 7872 := RowAgreement.join within_061_055_agreement within_061_058_agreement
private def within_061_060 : CountSegment 1 10 13120 7856 7872 3336 3344 := CountSegment.join within_061_052 within_061_059
private theorem within_061_060_agreement : RowAgreement 7856 7872 := RowAgreement.join within_061_052_agreement within_061_059_agreement
private def within_061_061 : CountSegment 1 10 13120 7840 7872 3330 3344 := CountSegment.join within_061_045 within_061_060
private theorem within_061_061_agreement : RowAgreement 7840 7872 := RowAgreement.join within_061_045_agreement within_061_060_agreement
private def within_061_062 : CountSegment 1 10 13120 7808 7872 3310 3344 := CountSegment.join within_061_030 within_061_061
private theorem within_061_062_agreement : RowAgreement 7808 7872 := RowAgreement.join within_061_030_agreement within_061_061_agreement
private def within_061_063 : CountSegment 1 10 13120 7872 7874 3344 3344 := CountSegment.join atomic_7873 atomic_7874
private theorem within_061_063_agreement : RowAgreement 7872 7874 := RowAgreement.join atomic_7873_agreement atomic_7874_agreement
private def within_061_064 : CountSegment 1 10 13120 7874 7876 3344 3344 := CountSegment.join atomic_7875 atomic_7876
private theorem within_061_064_agreement : RowAgreement 7874 7876 := RowAgreement.join atomic_7875_agreement atomic_7876_agreement
private def within_061_065 : CountSegment 1 10 13120 7872 7876 3344 3344 := CountSegment.join within_061_063 within_061_064
private theorem within_061_065_agreement : RowAgreement 7872 7876 := RowAgreement.join within_061_063_agreement within_061_064_agreement
private def within_061_066 : CountSegment 1 10 13120 7876 7878 3344 3346 := CountSegment.join atomic_7877 atomic_7878
private theorem within_061_066_agreement : RowAgreement 7876 7878 := RowAgreement.join atomic_7877_agreement atomic_7878_agreement
private def within_061_067 : CountSegment 1 10 13120 7878 7880 3346 3346 := CountSegment.join atomic_7879 atomic_7880
private theorem within_061_067_agreement : RowAgreement 7878 7880 := RowAgreement.join atomic_7879_agreement atomic_7880_agreement
private def within_061_068 : CountSegment 1 10 13120 7876 7880 3344 3346 := CountSegment.join within_061_066 within_061_067
private theorem within_061_068_agreement : RowAgreement 7876 7880 := RowAgreement.join within_061_066_agreement within_061_067_agreement
private def within_061_069 : CountSegment 1 10 13120 7872 7880 3344 3346 := CountSegment.join within_061_065 within_061_068
private theorem within_061_069_agreement : RowAgreement 7872 7880 := RowAgreement.join within_061_065_agreement within_061_068_agreement
private def within_061_070 : CountSegment 1 10 13120 7880 7882 3346 3346 := CountSegment.join atomic_7881 atomic_7882
private theorem within_061_070_agreement : RowAgreement 7880 7882 := RowAgreement.join atomic_7881_agreement atomic_7882_agreement
private def within_061_071 : CountSegment 1 10 13120 7882 7884 3346 3346 := CountSegment.join atomic_7883 atomic_7884
private theorem within_061_071_agreement : RowAgreement 7882 7884 := RowAgreement.join atomic_7883_agreement atomic_7884_agreement
private def within_061_072 : CountSegment 1 10 13120 7880 7884 3346 3346 := CountSegment.join within_061_070 within_061_071
private theorem within_061_072_agreement : RowAgreement 7880 7884 := RowAgreement.join within_061_070_agreement within_061_071_agreement
private def within_061_073 : CountSegment 1 10 13120 7884 7886 3346 3347 := CountSegment.join atomic_7885 atomic_7886
private theorem within_061_073_agreement : RowAgreement 7884 7886 := RowAgreement.join atomic_7885_agreement atomic_7886_agreement
private def within_061_074 : CountSegment 1 10 13120 7886 7888 3347 3347 := CountSegment.join atomic_7887 atomic_7888
private theorem within_061_074_agreement : RowAgreement 7886 7888 := RowAgreement.join atomic_7887_agreement atomic_7888_agreement
private def within_061_075 : CountSegment 1 10 13120 7884 7888 3346 3347 := CountSegment.join within_061_073 within_061_074
private theorem within_061_075_agreement : RowAgreement 7884 7888 := RowAgreement.join within_061_073_agreement within_061_074_agreement
private def within_061_076 : CountSegment 1 10 13120 7880 7888 3346 3347 := CountSegment.join within_061_072 within_061_075
private theorem within_061_076_agreement : RowAgreement 7880 7888 := RowAgreement.join within_061_072_agreement within_061_075_agreement
private def within_061_077 : CountSegment 1 10 13120 7872 7888 3344 3347 := CountSegment.join within_061_069 within_061_076
private theorem within_061_077_agreement : RowAgreement 7872 7888 := RowAgreement.join within_061_069_agreement within_061_076_agreement
private def within_061_078 : CountSegment 1 10 13120 7888 7890 3347 3347 := CountSegment.join atomic_7889 atomic_7890
private theorem within_061_078_agreement : RowAgreement 7888 7890 := RowAgreement.join atomic_7889_agreement atomic_7890_agreement
private def within_061_079 : CountSegment 1 10 13120 7890 7892 3347 3349 := CountSegment.join atomic_7891 atomic_7892
private theorem within_061_079_agreement : RowAgreement 7890 7892 := RowAgreement.join atomic_7891_agreement atomic_7892_agreement
private def within_061_080 : CountSegment 1 10 13120 7888 7892 3347 3349 := CountSegment.join within_061_078 within_061_079
private theorem within_061_080_agreement : RowAgreement 7888 7892 := RowAgreement.join within_061_078_agreement within_061_079_agreement
private def within_061_081 : CountSegment 1 10 13120 7892 7894 3349 3349 := CountSegment.join atomic_7893 atomic_7894
private theorem within_061_081_agreement : RowAgreement 7892 7894 := RowAgreement.join atomic_7893_agreement atomic_7894_agreement
private def within_061_082 : CountSegment 1 10 13120 7894 7896 3349 3349 := CountSegment.join atomic_7895 atomic_7896
private theorem within_061_082_agreement : RowAgreement 7894 7896 := RowAgreement.join atomic_7895_agreement atomic_7896_agreement
private def within_061_083 : CountSegment 1 10 13120 7892 7896 3349 3349 := CountSegment.join within_061_081 within_061_082
private theorem within_061_083_agreement : RowAgreement 7892 7896 := RowAgreement.join within_061_081_agreement within_061_082_agreement
private def within_061_084 : CountSegment 1 10 13120 7888 7896 3347 3349 := CountSegment.join within_061_080 within_061_083
private theorem within_061_084_agreement : RowAgreement 7888 7896 := RowAgreement.join within_061_080_agreement within_061_083_agreement
private def within_061_085 : CountSegment 1 10 13120 7896 7898 3349 3350 := CountSegment.join atomic_7897 atomic_7898
private theorem within_061_085_agreement : RowAgreement 7896 7898 := RowAgreement.join atomic_7897_agreement atomic_7898_agreement
private def within_061_086 : CountSegment 1 10 13120 7898 7900 3350 3351 := CountSegment.join atomic_7899 atomic_7900
private theorem within_061_086_agreement : RowAgreement 7898 7900 := RowAgreement.join atomic_7899_agreement atomic_7900_agreement
private def within_061_087 : CountSegment 1 10 13120 7896 7900 3349 3351 := CountSegment.join within_061_085 within_061_086
private theorem within_061_087_agreement : RowAgreement 7896 7900 := RowAgreement.join within_061_085_agreement within_061_086_agreement
private def within_061_088 : CountSegment 1 10 13120 7900 7902 3351 3352 := CountSegment.join atomic_7901 atomic_7902
private theorem within_061_088_agreement : RowAgreement 7900 7902 := RowAgreement.join atomic_7901_agreement atomic_7902_agreement
private def within_061_089 : CountSegment 1 10 13120 7902 7904 3352 3352 := CountSegment.join atomic_7903 atomic_7904
private theorem within_061_089_agreement : RowAgreement 7902 7904 := RowAgreement.join atomic_7903_agreement atomic_7904_agreement
private def within_061_090 : CountSegment 1 10 13120 7900 7904 3351 3352 := CountSegment.join within_061_088 within_061_089
private theorem within_061_090_agreement : RowAgreement 7900 7904 := RowAgreement.join within_061_088_agreement within_061_089_agreement
private def within_061_091 : CountSegment 1 10 13120 7896 7904 3349 3352 := CountSegment.join within_061_087 within_061_090
private theorem within_061_091_agreement : RowAgreement 7896 7904 := RowAgreement.join within_061_087_agreement within_061_090_agreement
private def within_061_092 : CountSegment 1 10 13120 7888 7904 3347 3352 := CountSegment.join within_061_084 within_061_091
private theorem within_061_092_agreement : RowAgreement 7888 7904 := RowAgreement.join within_061_084_agreement within_061_091_agreement
private def within_061_093 : CountSegment 1 10 13120 7872 7904 3344 3352 := CountSegment.join within_061_077 within_061_092
private theorem within_061_093_agreement : RowAgreement 7872 7904 := RowAgreement.join within_061_077_agreement within_061_092_agreement
private def within_061_094 : CountSegment 1 10 13120 7904 7906 3352 3352 := CountSegment.join atomic_7905 atomic_7906
private theorem within_061_094_agreement : RowAgreement 7904 7906 := RowAgreement.join atomic_7905_agreement atomic_7906_agreement
private def within_061_095 : CountSegment 1 10 13120 7906 7908 3352 3352 := CountSegment.join atomic_7907 atomic_7908
private theorem within_061_095_agreement : RowAgreement 7906 7908 := RowAgreement.join atomic_7907_agreement atomic_7908_agreement
private def within_061_096 : CountSegment 1 10 13120 7904 7908 3352 3352 := CountSegment.join within_061_094 within_061_095
private theorem within_061_096_agreement : RowAgreement 7904 7908 := RowAgreement.join within_061_094_agreement within_061_095_agreement
private def within_061_097 : CountSegment 1 10 13120 7908 7910 3352 3353 := CountSegment.join atomic_7909 atomic_7910
private theorem within_061_097_agreement : RowAgreement 7908 7910 := RowAgreement.join atomic_7909_agreement atomic_7910_agreement
private def within_061_098 : CountSegment 1 10 13120 7910 7912 3353 3354 := CountSegment.join atomic_7911 atomic_7912
private theorem within_061_098_agreement : RowAgreement 7910 7912 := RowAgreement.join atomic_7911_agreement atomic_7912_agreement
private def within_061_099 : CountSegment 1 10 13120 7908 7912 3352 3354 := CountSegment.join within_061_097 within_061_098
private theorem within_061_099_agreement : RowAgreement 7908 7912 := RowAgreement.join within_061_097_agreement within_061_098_agreement
private def within_061_100 : CountSegment 1 10 13120 7904 7912 3352 3354 := CountSegment.join within_061_096 within_061_099
private theorem within_061_100_agreement : RowAgreement 7904 7912 := RowAgreement.join within_061_096_agreement within_061_099_agreement
private def within_061_101 : CountSegment 1 10 13120 7912 7914 3354 3354 := CountSegment.join atomic_7913 atomic_7914
private theorem within_061_101_agreement : RowAgreement 7912 7914 := RowAgreement.join atomic_7913_agreement atomic_7914_agreement
private def within_061_102 : CountSegment 1 10 13120 7914 7916 3354 3354 := CountSegment.join atomic_7915 atomic_7916
private theorem within_061_102_agreement : RowAgreement 7914 7916 := RowAgreement.join atomic_7915_agreement atomic_7916_agreement
private def within_061_103 : CountSegment 1 10 13120 7912 7916 3354 3354 := CountSegment.join within_061_101 within_061_102
private theorem within_061_103_agreement : RowAgreement 7912 7916 := RowAgreement.join within_061_101_agreement within_061_102_agreement
private def within_061_104 : CountSegment 1 10 13120 7916 7918 3354 3355 := CountSegment.join atomic_7917 atomic_7918
private theorem within_061_104_agreement : RowAgreement 7916 7918 := RowAgreement.join atomic_7917_agreement atomic_7918_agreement
private def within_061_105 : CountSegment 1 10 13120 7918 7920 3355 3355 := CountSegment.join atomic_7919 atomic_7920
private theorem within_061_105_agreement : RowAgreement 7918 7920 := RowAgreement.join atomic_7919_agreement atomic_7920_agreement
private def within_061_106 : CountSegment 1 10 13120 7916 7920 3354 3355 := CountSegment.join within_061_104 within_061_105
private theorem within_061_106_agreement : RowAgreement 7916 7920 := RowAgreement.join within_061_104_agreement within_061_105_agreement
private def within_061_107 : CountSegment 1 10 13120 7912 7920 3354 3355 := CountSegment.join within_061_103 within_061_106
private theorem within_061_107_agreement : RowAgreement 7912 7920 := RowAgreement.join within_061_103_agreement within_061_106_agreement
private def within_061_108 : CountSegment 1 10 13120 7904 7920 3352 3355 := CountSegment.join within_061_100 within_061_107
private theorem within_061_108_agreement : RowAgreement 7904 7920 := RowAgreement.join within_061_100_agreement within_061_107_agreement
private def within_061_109 : CountSegment 1 10 13120 7920 7922 3355 3357 := CountSegment.join atomic_7921 atomic_7922
private theorem within_061_109_agreement : RowAgreement 7920 7922 := RowAgreement.join atomic_7921_agreement atomic_7922_agreement
private def within_061_110 : CountSegment 1 10 13120 7922 7924 3357 3358 := CountSegment.join atomic_7923 atomic_7924
private theorem within_061_110_agreement : RowAgreement 7922 7924 := RowAgreement.join atomic_7923_agreement atomic_7924_agreement
private def within_061_111 : CountSegment 1 10 13120 7920 7924 3355 3358 := CountSegment.join within_061_109 within_061_110
private theorem within_061_111_agreement : RowAgreement 7920 7924 := RowAgreement.join within_061_109_agreement within_061_110_agreement
private def within_061_112 : CountSegment 1 10 13120 7924 7926 3358 3359 := CountSegment.join atomic_7925 atomic_7926
private theorem within_061_112_agreement : RowAgreement 7924 7926 := RowAgreement.join atomic_7925_agreement atomic_7926_agreement
private def within_061_113 : CountSegment 1 10 13120 7926 7928 3359 3361 := CountSegment.join atomic_7927 atomic_7928
private theorem within_061_113_agreement : RowAgreement 7926 7928 := RowAgreement.join atomic_7927_agreement atomic_7928_agreement
private def within_061_114 : CountSegment 1 10 13120 7924 7928 3358 3361 := CountSegment.join within_061_112 within_061_113
private theorem within_061_114_agreement : RowAgreement 7924 7928 := RowAgreement.join within_061_112_agreement within_061_113_agreement
private def within_061_115 : CountSegment 1 10 13120 7920 7928 3355 3361 := CountSegment.join within_061_111 within_061_114
private theorem within_061_115_agreement : RowAgreement 7920 7928 := RowAgreement.join within_061_111_agreement within_061_114_agreement
private def within_061_116 : CountSegment 1 10 13120 7928 7930 3361 3361 := CountSegment.join atomic_7929 atomic_7930
private theorem within_061_116_agreement : RowAgreement 7928 7930 := RowAgreement.join atomic_7929_agreement atomic_7930_agreement
private def within_061_117 : CountSegment 1 10 13120 7930 7932 3361 3361 := CountSegment.join atomic_7931 atomic_7932
private theorem within_061_117_agreement : RowAgreement 7930 7932 := RowAgreement.join atomic_7931_agreement atomic_7932_agreement
private def within_061_118 : CountSegment 1 10 13120 7928 7932 3361 3361 := CountSegment.join within_061_116 within_061_117
private theorem within_061_118_agreement : RowAgreement 7928 7932 := RowAgreement.join within_061_116_agreement within_061_117_agreement
private def within_061_119 : CountSegment 1 10 13120 7932 7934 3361 3362 := CountSegment.join atomic_7933 atomic_7934
private theorem within_061_119_agreement : RowAgreement 7932 7934 := RowAgreement.join atomic_7933_agreement atomic_7934_agreement
private def within_061_120 : CountSegment 1 10 13120 7934 7936 3362 3363 := CountSegment.join atomic_7935 atomic_7936
private theorem within_061_120_agreement : RowAgreement 7934 7936 := RowAgreement.join atomic_7935_agreement atomic_7936_agreement
private def within_061_121 : CountSegment 1 10 13120 7932 7936 3361 3363 := CountSegment.join within_061_119 within_061_120
private theorem within_061_121_agreement : RowAgreement 7932 7936 := RowAgreement.join within_061_119_agreement within_061_120_agreement
private def within_061_122 : CountSegment 1 10 13120 7928 7936 3361 3363 := CountSegment.join within_061_118 within_061_121
private theorem within_061_122_agreement : RowAgreement 7928 7936 := RowAgreement.join within_061_118_agreement within_061_121_agreement
private def within_061_123 : CountSegment 1 10 13120 7920 7936 3355 3363 := CountSegment.join within_061_115 within_061_122
private theorem within_061_123_agreement : RowAgreement 7920 7936 := RowAgreement.join within_061_115_agreement within_061_122_agreement
private def within_061_124 : CountSegment 1 10 13120 7904 7936 3352 3363 := CountSegment.join within_061_108 within_061_123
private theorem within_061_124_agreement : RowAgreement 7904 7936 := RowAgreement.join within_061_108_agreement within_061_123_agreement
private def within_061_125 : CountSegment 1 10 13120 7872 7936 3344 3363 := CountSegment.join within_061_093 within_061_124
private theorem within_061_125_agreement : RowAgreement 7872 7936 := RowAgreement.join within_061_093_agreement within_061_124_agreement
private def within_061_126 : CountSegment 1 10 13120 7808 7936 3310 3363 := CountSegment.join within_061_062 within_061_125
private theorem within_061_126_agreement : RowAgreement 7808 7936 := RowAgreement.join within_061_062_agreement within_061_125_agreement
def segment_061 : CountSegment 1 10 13120 7808 7936 3310 3363 := within_061_126
theorem agreement_061 : RowAgreement 7808 7936 := within_061_126_agreement
end ProjectEulerSolutions.P58Numerical
