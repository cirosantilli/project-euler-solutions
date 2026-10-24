import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_11905 : RowCertificate 11905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11905 1) := by change ¬ Nat.Prime 566939911; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 842407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11905 2) := by change ¬ Nat.Prime 566916101; exact PrattCertificates.composite_of_product (factor := 4297) (cofactor := 131933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11905 3) := by change ¬ Nat.Prime 566892291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 188964097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11905 : row 11905 = 0 := (row_11905.checked).trans rfl
theorem model_count_11905 : modelRow 11905 = 0 := by decide
theorem modelrow_agrees_11905 : modelRow 11905 = row 11905 := model_count_11905.trans row_count_11905.symm
def row_11906 : RowCertificate 11906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11906 1) := by change ¬ Nat.Prime 567035157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189011719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11906 2) := by change ¬ Nat.Prime 567011345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113402269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11906 3) := by change ¬ Nat.Prime 566987533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 80998219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11906 : row 11906 = 0 := (row_11906.checked).trans rfl
theorem model_count_11906 : modelRow 11906 = 0 := by decide
theorem modelrow_agrees_11906 : modelRow 11906 = row 11906 := model_count_11906.trans row_count_11906.symm
def row_11907 : RowCertificate 11907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11907 1) := by change ¬ Nat.Prime 567130411; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 29848969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11907 2) := by change Nat.Prime 567106597; exact ProjectEulerSolutions.P58PrimeNodes.p_567106597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11907 3) := by change ¬ Nat.Prime 567082783; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18292993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11907 : row 11907 = 1 := (row_11907.checked).trans rfl
theorem model_count_11907 : modelRow 11907 = 1 := by decide
theorem modelrow_agrees_11907 : modelRow 11907 = row 11907 := model_count_11907.trans row_count_11907.symm
def row_11908 : RowCertificate 11908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11908 1) := by change ¬ Nat.Prime 567225673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81032239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11908 2) := by change Nat.Prime 567201857; exact ProjectEulerSolutions.P58PrimeNodes.p_567201857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11908 3) := by change ¬ Nat.Prime 567178041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189059347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11908 : row 11908 = 1 := (row_11908.checked).trans rfl
theorem model_count_11908 : modelRow 11908 = 1 := by decide
theorem modelrow_agrees_11908 : modelRow 11908 = row 11908 := model_count_11908.trans row_count_11908.symm
def row_11909 : RowCertificate 11909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11909 1) := by change ¬ Nat.Prime 567320943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189106981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11909 2) := by change ¬ Nat.Prime 567297125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113459425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11909 3) := by change ¬ Nat.Prime 567273307; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15331711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11909 : row 11909 = 0 := (row_11909.checked).trans rfl
theorem model_count_11909 : modelRow 11909 = 0 := by decide
theorem modelrow_agrees_11909 : modelRow 11909 = row 11909 := model_count_11909.trans row_count_11909.symm
def row_11910 : RowCertificate 11910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11910 1) := by change ¬ Nat.Prime 567416221; exact PrattCertificates.composite_of_product (factor := 3511) (cofactor := 161611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11910 2) := by change ¬ Nat.Prime 567392401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 10705517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11910 3) := by change ¬ Nat.Prime 567368581; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43643737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11910 : row 11910 = 0 := (row_11910.checked).trans rfl
theorem model_count_11910 : modelRow 11910 = 0 := by decide
theorem modelrow_agrees_11910 : modelRow 11910 = row 11910 := model_count_11910.trans row_count_11910.symm
def row_11911 : RowCertificate 11911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11911 1) := by change ¬ Nat.Prime 567511507; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8470321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11911 2) := by change ¬ Nat.Prime 567487685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113497537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11911 3) := by change ¬ Nat.Prime 567463863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189154621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11911 : row 11911 = 0 := (row_11911.checked).trans rfl
theorem model_count_11911 : modelRow 11911 = 0 := by decide
theorem modelrow_agrees_11911 : modelRow 11911 = row 11911 := model_count_11911.trans row_count_11911.symm
def row_11912 : RowCertificate 11912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11912 1) := by change ¬ Nat.Prime 567606801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189202267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11912 2) := by change ¬ Nat.Prime 567582977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43660229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11912 3) := by change ¬ Nat.Prime 567559153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81079879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11912 : row 11912 = 0 := (row_11912.checked).trans rfl
theorem model_count_11912 : modelRow 11912 = 0 := by decide
theorem modelrow_agrees_11912 : modelRow 11912 = row 11912 := model_count_11912.trans row_count_11912.symm
def row_11913 : RowCertificate 11913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11913 1) := by change ¬ Nat.Prime 567702103; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 5852599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11913 2) := by change ¬ Nat.Prime 567678277; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 19575113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11913 3) := by change ¬ Nat.Prime 567654451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81093493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11913 : row 11913 = 0 := (row_11913.checked).trans rfl
theorem model_count_11913 : modelRow 11913 = 0 := by decide
theorem modelrow_agrees_11913 : modelRow 11913 = row 11913 := model_count_11913.trans row_count_11913.symm
def row_11914 : RowCertificate 11914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11914 1) := by change ¬ Nat.Prime 567797413; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13204591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11914 2) := by change ¬ Nat.Prime 567773585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113554717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11914 3) := by change ¬ Nat.Prime 567749757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189249919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11914 : row 11914 = 0 := (row_11914.checked).trans rfl
theorem model_count_11914 : modelRow 11914 = 0 := by decide
theorem modelrow_agrees_11914 : modelRow 11914 = row 11914 := model_count_11914.trans row_count_11914.symm
def row_11915 : RowCertificate 11915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11915 1) := by change ¬ Nat.Prime 567892731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189297577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11915 2) := by change ¬ Nat.Prime 567868901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33404053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11915 3) := by change ¬ Nat.Prime 567845071; exact PrattCertificates.composite_of_product (factor := 9199) (cofactor := 61729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11915 : row 11915 = 0 := (row_11915.checked).trans rfl
theorem model_count_11915 : modelRow 11915 = 0 := by decide
theorem modelrow_agrees_11915 : modelRow 11915 = row 11915 := model_count_11915.trans row_count_11915.symm
def row_11916 : RowCertificate 11916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11916 1) := by change ¬ Nat.Prime 567988057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81141151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11916 2) := by change ¬ Nat.Prime 567964225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113592845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11916 3) := by change ¬ Nat.Prime 567940393; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 411253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11916 : row 11916 = 0 := (row_11916.checked).trans rfl
theorem model_count_11916 : modelRow 11916 = 0 := by decide
theorem modelrow_agrees_11916 : modelRow 11916 = row 11916 := model_count_11916.trans row_count_11916.symm
def row_11917 : RowCertificate 11917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11917 1) := by change ¬ Nat.Prime 568083391; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7190929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11917 2) := by change ¬ Nat.Prime 568059557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43696889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11917 3) := by change ¬ Nat.Prime 568035723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189345241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11917 : row 11917 = 0 := (row_11917.checked).trans rfl
theorem model_count_11917 : modelRow 11917 = 0 := by decide
theorem modelrow_agrees_11917 : modelRow 11917 = row 11917 := model_count_11917.trans row_count_11917.symm
def row_11918 : RowCertificate 11918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11918 1) := by change ¬ Nat.Prime 568178733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189392911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11918 2) := by change ¬ Nat.Prime 568154897; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 3618821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11918 3) := by change ¬ Nat.Prime 568131061; exact PrattCertificates.composite_of_product (factor := 6337) (cofactor := 89653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11918 : row 11918 = 0 := (row_11918.checked).trans rfl
theorem model_count_11918 : modelRow 11918 = 0 := by decide
theorem modelrow_agrees_11918 : modelRow 11918 = row 11918 := model_count_11918.trans row_count_11918.symm
def row_11919 : RowCertificate 11919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11919 1) := by change ¬ Nat.Prime 568274083; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43713391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11919 2) := by change ¬ Nat.Prime 568250245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113650049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11919 3) := by change ¬ Nat.Prime 568226407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81175201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11919 : row 11919 = 0 := (row_11919.checked).trans rfl
theorem model_count_11919 : modelRow 11919 = 0 := by decide
theorem modelrow_agrees_11919 : modelRow 11919 = row 11919 := model_count_11919.trans row_count_11919.symm
def row_11920 : RowCertificate 11920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11920 1) := by change ¬ Nat.Prime 568369441; exact PrattCertificates.composite_of_product (factor := 1171) (cofactor := 485371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11920 2) := by change ¬ Nat.Prime 568345601; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9317141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11920 3) := by change ¬ Nat.Prime 568321761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189440587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11920 : row 11920 = 0 := (row_11920.checked).trans rfl
theorem model_count_11920 : modelRow 11920 = 0 := by decide
theorem modelrow_agrees_11920 : modelRow 11920 = row 11920 := model_count_11920.trans row_count_11920.symm
def row_11921 : RowCertificate 11921 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11921 1) := by change ¬ Nat.Prime 568464807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189488269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11921 2) := by change ¬ Nat.Prime 568440965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113688193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11921 3) := by change ¬ Nat.Prime 568417123; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 927271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11921 : row 11921 = 0 := (row_11921.checked).trans rfl
theorem model_count_11921 : modelRow 11921 = 0 := by decide
theorem modelrow_agrees_11921 : modelRow 11921 = row 11921 := model_count_11921.trans row_count_11921.symm
def row_11922 : RowCertificate 11922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11922 1) := by change ¬ Nat.Prime 568560181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81222883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11922 2) := by change ¬ Nat.Prime 568536337; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 7788169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11922 3) := by change ¬ Nat.Prime 568512493; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 3140953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11922 : row 11922 = 0 := (row_11922.checked).trans rfl
theorem model_count_11922 : modelRow 11922 = 0 := by decide
theorem modelrow_agrees_11922 : modelRow 11922 = row 11922 := model_count_11922.trans row_count_11922.symm
def row_11923 : RowCertificate 11923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11923 1) := by change ¬ Nat.Prime 568655563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81236509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11923 2) := by change ¬ Nat.Prime 568631717; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 5630017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11923 3) := by change ¬ Nat.Prime 568607871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189535957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11923 : row 11923 = 0 := (row_11923.checked).trans rfl
theorem model_count_11923 : modelRow 11923 = 0 := by decide
theorem modelrow_agrees_11923 : modelRow 11923 = row 11923 := model_count_11923.trans row_count_11923.symm
def row_11924 : RowCertificate 11924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11924 1) := by change ¬ Nat.Prime 568750953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189583651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11924 2) := by change ¬ Nat.Prime 568727105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113745421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11924 3) := by change ¬ Nat.Prime 568703257; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2946649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11924 : row 11924 = 0 := (row_11924.checked).trans rfl
theorem model_count_11924 : modelRow 11924 = 0 := by decide
theorem modelrow_agrees_11924 : modelRow 11924 = row 11924 := model_count_11924.trans row_count_11924.symm
def row_11925 : RowCertificate 11925 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11925 1) := by change Nat.Prime 568846351; exact ProjectEulerSolutions.P58PrimeNodes.p_568846351.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11925 2) := by change ¬ Nat.Prime 568822501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43755577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11925 3) := by change ¬ Nat.Prime 568798651; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 5522317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11925 : row 11925 = 1 := (row_11925.checked).trans rfl
theorem model_count_11925 : modelRow 11925 = 1 := by decide
theorem modelrow_agrees_11925 : modelRow 11925 = row 11925 := model_count_11925.trans row_count_11925.symm
def row_11926 : RowCertificate 11926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11926 1) := by change ¬ Nat.Prime 568941757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 29944303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11926 2) := by change ¬ Nat.Prime 568917905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113783581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11926 3) := by change ¬ Nat.Prime 568894053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189631351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11926 : row 11926 = 0 := (row_11926.checked).trans rfl
theorem model_count_11926 : modelRow 11926 = 0 := by decide
theorem modelrow_agrees_11926 : modelRow 11926 = row 11926 := model_count_11926.trans row_count_11926.symm
def row_11927 : RowCertificate 11927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11927 1) := by change ¬ Nat.Prime 569037171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189679057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11927 2) := by change Nat.Prime 569013317; exact ProjectEulerSolutions.P58PrimeNodes.p_569013317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11927 3) := by change ¬ Nat.Prime 568989463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81284209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11927 : row 11927 = 1 := (row_11927.checked).trans rfl
theorem model_count_11927 : modelRow 11927 = 1 := by decide
theorem modelrow_agrees_11927 : modelRow 11927 = row 11927 := model_count_11927.trans row_count_11927.symm
def row_11928 : RowCertificate 11928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11928 1) := by change ¬ Nat.Prime 569132593; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 29954347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11928 2) := by change ¬ Nat.Prime 569108737; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 1051957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11928 3) := by change ¬ Nat.Prime 569084881; exact PrattCertificates.composite_of_product (factor := 6547) (cofactor := 86923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11928 : row 11928 = 0 := (row_11928.checked).trans rfl
theorem model_count_11928 : modelRow 11928 = 0 := by decide
theorem modelrow_agrees_11928 : modelRow 11928 = row 11928 := model_count_11928.trans row_count_11928.symm
def row_11929 : RowCertificate 11929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11929 1) := by change ¬ Nat.Prime 569228023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81318289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11929 2) := by change ¬ Nat.Prime 569204165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113840833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11929 3) := by change ¬ Nat.Prime 569180307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189726769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11929 : row 11929 = 0 := (row_11929.checked).trans rfl
theorem model_count_11929 : modelRow 11929 = 0 := by decide
theorem modelrow_agrees_11929 : modelRow 11929 = row 11929 := model_count_11929.trans row_count_11929.symm
def row_11930 : RowCertificate 11930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11930 1) := by change ¬ Nat.Prime 569323461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189774487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11930 2) := by change ¬ Nat.Prime 569299601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43792277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11930 3) := by change ¬ Nat.Prime 569275741; exact PrattCertificates.composite_of_product (factor := 3049) (cofactor := 186709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11930 : row 11930 = 0 := (row_11930.checked).trans rfl
theorem model_count_11930 : modelRow 11930 = 0 := by decide
theorem modelrow_agrees_11930 : modelRow 11930 = row 11930 := model_count_11930.trans row_count_11930.symm
def row_11931 : RowCertificate 11931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11931 1) := by change ¬ Nat.Prime 569418907; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 7800259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11931 2) := by change ¬ Nat.Prime 569395045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113879009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 11931 3) := by change Nat.Prime 569371183; exact ProjectEulerSolutions.P58PrimeNodes.p_569371183.prime
    exact decide_eq_true prime
theorem row_count_11931 : row 11931 = 1 := (row_11931.checked).trans rfl
theorem model_count_11931 : modelRow 11931 = 1 := by decide
theorem modelrow_agrees_11931 : modelRow 11931 = row 11931 := model_count_11931.trans row_count_11931.symm
def row_11932 : RowCertificate 11932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11932 1) := by change ¬ Nat.Prime 569514361; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43808797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11932 2) := by change ¬ Nat.Prime 569490497; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33499441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11932 3) := by change ¬ Nat.Prime 569466633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189822211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11932 : row 11932 = 0 := (row_11932.checked).trans rfl
theorem model_count_11932 : modelRow 11932 = 0 := by decide
theorem modelrow_agrees_11932 : modelRow 11932 = row 11932 := model_count_11932.trans row_count_11932.symm
def row_11933 : RowCertificate 11933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11933 1) := by change ¬ Nat.Prime 569609823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189869941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11933 2) := by change Nat.Prime 569585957; exact ProjectEulerSolutions.P58PrimeNodes.p_569585957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11933 3) := by change ¬ Nat.Prime 569562091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81366013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11933 : row 11933 = 1 := (row_11933.checked).trans rfl
theorem model_count_11933 : modelRow 11933 = 1 := by decide
theorem modelrow_agrees_11933 : modelRow 11933 = row 11933 := model_count_11933.trans row_count_11933.symm
def row_11934 : RowCertificate 11934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11934 1) := by change ¬ Nat.Prime 569705293; exact PrattCertificates.composite_of_product (factor := 20749) (cofactor := 27457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11934 2) := by change ¬ Nat.Prime 569681425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113936285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11934 3) := by change ¬ Nat.Prime 569657557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81379651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11934 : row 11934 = 0 := (row_11934.checked).trans rfl
theorem model_count_11934 : modelRow 11934 = 0 := by decide
theorem modelrow_agrees_11934 : modelRow 11934 = row 11934 := model_count_11934.trans row_count_11934.symm
def row_11935 : RowCertificate 11935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11935 1) := by change ¬ Nat.Prime 569800771; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 987523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11935 2) := by change ¬ Nat.Prime 569776901; exact PrattCertificates.composite_of_product (factor := 2609) (cofactor := 218389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11935 3) := by change ¬ Nat.Prime 569753031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189917677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11935 : row 11935 = 0 := (row_11935.checked).trans rfl
theorem model_count_11935 : modelRow 11935 = 0 := by decide
theorem modelrow_agrees_11935 : modelRow 11935 = row 11935 := model_count_11935.trans row_count_11935.symm
def row_11936 : RowCertificate 11936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11936 1) := by change ¬ Nat.Prime 569896257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 189965419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11936 2) := by change ¬ Nat.Prime 569872385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 113974477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11936 3) := by change ¬ Nat.Prime 569848513; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43834501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11936 : row 11936 = 0 := (row_11936.checked).trans rfl
theorem model_count_11936 : modelRow 11936 = 0 := by decide
theorem modelrow_agrees_11936 : modelRow 11936 = row 11936 := model_count_11936.trans row_count_11936.symm
def row_11937 : RowCertificate 11937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11937 1) := by change ¬ Nat.Prime 569991751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81427393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11937 2) := by change Nat.Prime 569967877; exact ProjectEulerSolutions.P58PrimeNodes.p_569967877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11937 3) := by change ¬ Nat.Prime 569944003; exact PrattCertificates.composite_of_product (factor := 3463) (cofactor := 164581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11937 : row 11937 = 1 := (row_11937.checked).trans rfl
theorem model_count_11937 : modelRow 11937 = 1 := by decide
theorem modelrow_agrees_11937 : modelRow 11937 = row 11937 := model_count_11937.trans row_count_11937.symm
def row_11938 : RowCertificate 11938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11938 1) := by change ¬ Nat.Prime 570087253; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 3631129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11938 2) := by change ¬ Nat.Prime 570063377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43851029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11938 3) := by change ¬ Nat.Prime 570039501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190013167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11938 : row 11938 = 0 := (row_11938.checked).trans rfl
theorem model_count_11938 : modelRow 11938 = 0 := by decide
theorem modelrow_agrees_11938 : modelRow 11938 = row 11938 := model_count_11938.trans row_count_11938.symm
def row_11939 : RowCertificate 11939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11939 1) := by change ¬ Nat.Prime 570182763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190060921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11939 2) := by change ¬ Nat.Prime 570158885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114031777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11939 3) := by change ¬ Nat.Prime 570135007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43856539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11939 : row 11939 = 0 := (row_11939.checked).trans rfl
theorem model_count_11939 : modelRow 11939 = 0 := by decide
theorem modelrow_agrees_11939 : modelRow 11939 = row 11939 := model_count_11939.trans row_count_11939.symm
def row_11940 : RowCertificate 11940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11940 1) := by change ¬ Nat.Prime 570278281; exact PrattCertificates.composite_of_product (factor := 11071) (cofactor := 51511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11940 2) := by change ¬ Nat.Prime 570254401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 10759517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11940 3) := by change ¬ Nat.Prime 570230521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81461503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11940 : row 11940 = 0 := (row_11940.checked).trans rfl
theorem model_count_11940 : modelRow 11940 = 0 := by decide
theorem modelrow_agrees_11940 : modelRow 11940 = row 11940 := model_count_11940.trans row_count_11940.symm
def row_11941 : RowCertificate 11941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11941 1) := by change ¬ Nat.Prime 570373807; exact PrattCertificates.composite_of_product (factor := 9643) (cofactor := 59149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11941 2) := by change ¬ Nat.Prime 570349925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114069985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11941 3) := by change ¬ Nat.Prime 570326043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190108681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11941 : row 11941 = 0 := (row_11941.checked).trans rfl
theorem model_count_11941 : modelRow 11941 = 0 := by decide
theorem modelrow_agrees_11941 : modelRow 11941 = row 11941 := model_count_11941.trans row_count_11941.symm
def row_11942 : RowCertificate 11942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11942 1) := by change ¬ Nat.Prime 570469341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190156447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11942 2) := by change ¬ Nat.Prime 570445457; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 19670533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11942 3) := by change ¬ Nat.Prime 570421573; exact PrattCertificates.composite_of_product (factor := 607) (cofactor := 939739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11942 : row 11942 = 0 := (row_11942.checked).trans rfl
theorem model_count_11942 : modelRow 11942 = 0 := by decide
theorem modelrow_agrees_11942 : modelRow 11942 = row 11942 := model_count_11942.trans row_count_11942.symm
def row_11943 : RowCertificate 11943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11943 1) := by change ¬ Nat.Prime 570564883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81509269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11943 2) := by change ¬ Nat.Prime 570540997; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43887769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11943 3) := by change ¬ Nat.Prime 570517111; exact PrattCertificates.composite_of_product (factor := 11329) (cofactor := 50359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11943 : row 11943 = 0 := (row_11943.checked).trans rfl
theorem model_count_11943 : modelRow 11943 = 0 := by decide
theorem modelrow_agrees_11943 : modelRow 11943 = row 11943 := model_count_11943.trans row_count_11943.symm
def row_11944 : RowCertificate 11944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11944 1) := by change ¬ Nat.Prime 570660433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81522919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11944 2) := by change ¬ Nat.Prime 570636545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114127309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11944 3) := by change ¬ Nat.Prime 570612657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190204219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11944 : row 11944 = 0 := (row_11944.checked).trans rfl
theorem model_count_11944 : modelRow 11944 = 0 := by decide
theorem modelrow_agrees_11944 : modelRow 11944 = row 11944 := model_count_11944.trans row_count_11944.symm
def row_11945 : RowCertificate 11945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11945 1) := by change ¬ Nat.Prime 570755991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190251997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11945 2) := by change Nat.Prime 570732101; exact ProjectEulerSolutions.P58PrimeNodes.p_570732101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11945 3) := by change ¬ Nat.Prime 570708211; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8518033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11945 : row 11945 = 1 := (row_11945.checked).trans rfl
theorem model_count_11945 : modelRow 11945 = 1 := by decide
theorem modelrow_agrees_11945 : modelRow 11945 = row 11945 := model_count_11945.trans row_count_11945.symm
def row_11946 : RowCertificate 11946 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11946 1) := by change Nat.Prime 570851557; exact ProjectEulerSolutions.P58PrimeNodes.p_570851557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11946 2) := by change ¬ Nat.Prime 570827665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114165533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11946 3) := by change ¬ Nat.Prime 570803773; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15427129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11946 : row 11946 = 1 := (row_11946.checked).trans rfl
theorem model_count_11946 : modelRow 11946 = 1 := by decide
theorem modelrow_agrees_11946 : modelRow 11946 = row 11946 := model_count_11946.trans row_count_11946.symm
def row_11947 : RowCertificate 11947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11947 1) := by change ¬ Nat.Prime 570947131; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30049849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11947 2) := by change ¬ Nat.Prime 570923237; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 13924957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11947 3) := by change ¬ Nat.Prime 570899343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190299781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11947 : row 11947 = 0 := (row_11947.checked).trans rfl
theorem model_count_11947 : modelRow 11947 = 0 := by decide
theorem modelrow_agrees_11947 : modelRow 11947 = row 11947 := model_count_11947.trans row_count_11947.symm
def row_11948 : RowCertificate 11948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11948 1) := by change ¬ Nat.Prime 571042713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190347571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11948 2) := by change ¬ Nat.Prime 571018817; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15432941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11948 3) := by change ¬ Nat.Prime 570994921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81570703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11948 : row 11948 = 0 := (row_11948.checked).trans rfl
theorem model_count_11948 : modelRow 11948 = 0 := by decide
theorem modelrow_agrees_11948 : modelRow 11948 = row 11948 := model_count_11948.trans row_count_11948.symm
def row_11949 : RowCertificate 11949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11949 1) := by change ¬ Nat.Prime 571138303; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9362923) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11949 2) := by change ¬ Nat.Prime 571114405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114222881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11949 3) := by change ¬ Nat.Prime 571090507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43930039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11949 : row 11949 = 0 := (row_11949.checked).trans rfl
theorem model_count_11949 : modelRow 11949 = 0 := by decide
theorem modelrow_agrees_11949 : modelRow 11949 = row 11949 := model_count_11949.trans row_count_11949.symm
def row_11950 : RowCertificate 11950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11950 1) := by change ¬ Nat.Prime 571233901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81604843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11950 2) := by change Nat.Prime 571210001; exact ProjectEulerSolutions.P58PrimeNodes.p_571210001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11950 3) := by change ¬ Nat.Prime 571186101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190395367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11950 : row 11950 = 1 := (row_11950.checked).trans rfl
theorem model_count_11950 : modelRow 11950 = 1 := by decide
theorem modelrow_agrees_11950 : modelRow 11950 = row 11950 := model_count_11950.trans row_count_11950.symm
def row_11951 : RowCertificate 11951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11951 1) := by change ¬ Nat.Prime 571329507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190443169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11951 2) := by change ¬ Nat.Prime 571305605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114261121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11951 3) := by change ¬ Nat.Prime 571281703; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13285621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11951 : row 11951 = 0 := (row_11951.checked).trans rfl
theorem model_count_11951 : modelRow 11951 = 0 := by decide
theorem modelrow_agrees_11951 : modelRow 11951 = row 11951 := model_count_11951.trans row_count_11951.symm
def row_11952 : RowCertificate 11952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11952 1) := by change ¬ Nat.Prime 571425121; exact PrattCertificates.composite_of_product (factor := 17107) (cofactor := 33403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11952 2) := by change ¬ Nat.Prime 571401217; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 5242213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11952 3) := by change ¬ Nat.Prime 571377313; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43952101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11952 : row 11952 = 0 := (row_11952.checked).trans rfl
theorem model_count_11952 : modelRow 11952 = 0 := by decide
theorem modelrow_agrees_11952 : modelRow 11952 = row 11952 := model_count_11952.trans row_count_11952.symm
def row_11953 : RowCertificate 11953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11953 1) := by change ¬ Nat.Prime 571520743; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18436153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11953 2) := by change ¬ Nat.Prime 571496837; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33617461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11953 3) := by change ¬ Nat.Prime 571472931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190490977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11953 : row 11953 = 0 := (row_11953.checked).trans rfl
theorem model_count_11953 : modelRow 11953 = 0 := by decide
theorem modelrow_agrees_11953 : modelRow 11953 = row 11953 := model_count_11953.trans row_count_11953.symm
def row_11954 : RowCertificate 11954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11954 1) := by change ¬ Nat.Prime 571616373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190538791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11954 2) := by change ¬ Nat.Prime 571592465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114318493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11954 3) := by change ¬ Nat.Prime 571568557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81652651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11954 : row 11954 = 0 := (row_11954.checked).trans rfl
theorem model_count_11954 : modelRow 11954 = 0 := by decide
theorem modelrow_agrees_11954 : modelRow 11954 = row 11954 := model_count_11954.trans row_count_11954.symm
def row_11955 : RowCertificate 11955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11955 1) := by change ¬ Nat.Prime 571712011; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 43977847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11955 2) := by change ¬ Nat.Prime 571688101; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 2453597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11955 3) := by change ¬ Nat.Prime 571664191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81666313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11955 : row 11955 = 0 := (row_11955.checked).trans rfl
theorem model_count_11955 : modelRow 11955 = 0 := by decide
theorem modelrow_agrees_11955 : modelRow 11955 = row 11955 := model_count_11955.trans row_count_11955.symm
def row_11956 : RowCertificate 11956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11956 1) := by change ¬ Nat.Prime 571807657; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15454261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11956 2) := by change ¬ Nat.Prime 571783745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114356749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11956 3) := by change ¬ Nat.Prime 571759833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190586611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11956 : row 11956 = 0 := (row_11956.checked).trans rfl
theorem model_count_11956 : modelRow 11956 = 0 := by decide
theorem modelrow_agrees_11956 : modelRow 11956 = row 11956 := model_count_11956.trans row_count_11956.symm
def row_11957 : RowCertificate 11957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11957 1) := by change ¬ Nat.Prime 571903311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190634437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11957 2) := by change Nat.Prime 571879397; exact ProjectEulerSolutions.P58PrimeNodes.p_571879397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11957 3) := by change ¬ Nat.Prime 571855483; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30097657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11957 : row 11957 = 1 := (row_11957.checked).trans rfl
theorem model_count_11957 : modelRow 11957 = 1 := by decide
theorem modelrow_agrees_11957 : modelRow 11957 = row 11957 := model_count_11957.trans row_count_11957.symm
def row_11958 : RowCertificate 11958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11958 1) := by change ¬ Nat.Prime 571998973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81714139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11958 2) := by change Nat.Prime 571975057; exact ProjectEulerSolutions.P58PrimeNodes.p_571975057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11958 3) := by change ¬ Nat.Prime 571951141; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 689929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11958 : row 11958 = 1 := (row_11958.checked).trans rfl
theorem model_count_11958 : modelRow 11958 = 1 := by decide
theorem modelrow_agrees_11958 : modelRow 11958 = row 11958 := model_count_11958.trans row_count_11958.symm
def row_11959 : RowCertificate 11959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11959 1) := by change ¬ Nat.Prime 572094643; exact PrattCertificates.composite_of_product (factor := 4261) (cofactor := 134263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11959 2) := by change ¬ Nat.Prime 572070725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114414145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11959 3) := by change ¬ Nat.Prime 572046807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190682269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11959 : row 11959 = 0 := (row_11959.checked).trans rfl
theorem model_count_11959 : modelRow 11959 = 0 := by decide
theorem modelrow_agrees_11959 : modelRow 11959 = row 11959 := model_count_11959.trans row_count_11959.symm
def row_11960 : RowCertificate 11960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11960 1) := by change ¬ Nat.Prime 572190321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190730107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11960 2) := by change ¬ Nat.Prime 572166401; exact PrattCertificates.composite_of_product (factor := 20393) (cofactor := 28057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11960 3) := by change ¬ Nat.Prime 572142481; exact PrattCertificates.composite_of_product (factor := 2293) (cofactor := 249517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11960 : row 11960 = 0 := (row_11960.checked).trans rfl
theorem model_count_11960 : modelRow 11960 = 0 := by decide
theorem modelrow_agrees_11960 : modelRow 11960 = row 11960 := model_count_11960.trans row_count_11960.symm
def row_11961 : RowCertificate 11961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11961 1) := by change ¬ Nat.Prime 572286007; exact PrattCertificates.composite_of_product (factor := 2143) (cofactor := 267049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11961 2) := by change ¬ Nat.Prime 572262085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114452417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11961 3) := by change ¬ Nat.Prime 572238163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81748309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11961 : row 11961 = 0 := (row_11961.checked).trans rfl
theorem model_count_11961 : modelRow 11961 = 0 := by decide
theorem modelrow_agrees_11961 : modelRow 11961 = row 11961 := model_count_11961.trans row_count_11961.symm
def row_11962 : RowCertificate 11962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11962 1) := by change ¬ Nat.Prime 572381701; exact PrattCertificates.composite_of_product (factor := 16573) (cofactor := 34537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11962 2) := by change ¬ Nat.Prime 572357777; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 1241557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11962 3) := by change ¬ Nat.Prime 572333853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190777951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11962 : row 11962 = 0 := (row_11962.checked).trans rfl
theorem model_count_11962 : modelRow 11962 = 0 := by decide
theorem modelrow_agrees_11962 : modelRow 11962 = row 11962 := model_count_11962.trans row_count_11962.symm
def row_11963 : RowCertificate 11963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11963 1) := by change ¬ Nat.Prime 572477403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190825801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11963 2) := by change ¬ Nat.Prime 572453477; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 10801009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11963 3) := by change ¬ Nat.Prime 572429551; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9384091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11963 : row 11963 = 0 := (row_11963.checked).trans rfl
theorem model_count_11963 : modelRow 11963 = 0 := by decide
theorem modelrow_agrees_11963 : modelRow 11963 = row 11963 := model_count_11963.trans row_count_11963.symm
def row_11964 : RowCertificate 11964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11964 1) := by change ¬ Nat.Prime 572573113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81796159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11964 2) := by change ¬ Nat.Prime 572549185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114509837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11964 3) := by change ¬ Nat.Prime 572525257; exact PrattCertificates.composite_of_product (factor := 11587) (cofactor := 49411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11964 : row 11964 = 0 := (row_11964.checked).trans rfl
theorem model_count_11964 : modelRow 11964 = 0 := by decide
theorem modelrow_agrees_11964 : modelRow 11964 = row 11964 := model_count_11964.trans row_count_11964.symm
def row_11965 : RowCertificate 11965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11965 1) := by change ¬ Nat.Prime 572668831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81809833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11965 2) := by change Nat.Prime 572644901; exact ProjectEulerSolutions.P58PrimeNodes.p_572644901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11965 3) := by change ¬ Nat.Prime 572620971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190873657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11965 : row 11965 = 1 := (row_11965.checked).trans rfl
theorem model_count_11965 : modelRow 11965 = 1 := by decide
theorem modelrow_agrees_11965 : modelRow 11965 = row 11965 := model_count_11965.trans row_count_11965.symm
def row_11966 : RowCertificate 11966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11966 1) := by change ¬ Nat.Prime 572764557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190921519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11966 2) := by change ¬ Nat.Prime 572740625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114548125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11966 3) := by change ¬ Nat.Prime 572716693; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 828823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11966 : row 11966 = 0 := (row_11966.checked).trans rfl
theorem model_count_11966 : modelRow 11966 = 0 := by decide
theorem modelrow_agrees_11966 : modelRow 11966 = row 11966 := model_count_11966.trans row_count_11966.symm
def row_11967 : RowCertificate 11967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11967 1) := by change ¬ Nat.Prime 572860291; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 5255599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11967 2) := by change Nat.Prime 572836357; exact ProjectEulerSolutions.P58PrimeNodes.p_572836357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11967 3) := by change ¬ Nat.Prime 572812423; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 1058803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11967 : row 11967 = 1 := (row_11967.checked).trans rfl
theorem model_count_11967 : modelRow 11967 = 1 := by decide
theorem modelrow_agrees_11967 : modelRow 11967 = row 11967 := model_count_11967.trans row_count_11967.symm
def row_11968 : RowCertificate 11968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11968 1) := by change ¬ Nat.Prime 572956033; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44073541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11968 2) := by change Nat.Prime 572932097; exact ProjectEulerSolutions.P58PrimeNodes.p_572932097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11968 3) := by change ¬ Nat.Prime 572908161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 190969387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11968 : row 11968 = 1 := (row_11968.checked).trans rfl
theorem model_count_11968 : modelRow 11968 = 1 := by decide
theorem modelrow_agrees_11968 : modelRow 11968 = row 11968 := model_count_11968.trans row_count_11968.symm
def row_11969 : RowCertificate 11969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11969 1) := by change ¬ Nat.Prime 573051783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191017261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11969 2) := by change ¬ Nat.Prime 573027845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114605569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11969 3) := by change ¬ Nat.Prime 573003907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81857701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11969 : row 11969 = 0 := (row_11969.checked).trans rfl
theorem model_count_11969 : modelRow 11969 = 0 := by decide
theorem modelrow_agrees_11969 : modelRow 11969 = row 11969 := model_count_11969.trans row_count_11969.symm
def row_11970 : RowCertificate 11970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11970 1) := by change ¬ Nat.Prime 573147541; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 3166561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11970 2) := by change ¬ Nat.Prime 573123601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33713153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 11970 3) := by change Nat.Prime 573099661; exact ProjectEulerSolutions.P58PrimeNodes.p_573099661.prime
    exact decide_eq_true prime
theorem row_count_11970 : row 11970 = 1 := (row_11970.checked).trans rfl
theorem model_count_11970 : modelRow 11970 = 1 := by decide
theorem modelrow_agrees_11970 : modelRow 11970 = row 11970 := model_count_11970.trans row_count_11970.symm
def row_11971 : RowCertificate 11971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11971 1) := by change ¬ Nat.Prime 573243307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81891901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11971 2) := by change ¬ Nat.Prime 573219365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114643873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11971 3) := by change ¬ Nat.Prime 573195423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191065141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11971 : row 11971 = 0 := (row_11971.checked).trans rfl
theorem model_count_11971 : modelRow 11971 = 0 := by decide
theorem modelrow_agrees_11971 : modelRow 11971 = row 11971 := model_count_11971.trans row_count_11971.symm
def row_11972 : RowCertificate 11972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11972 1) := by change ¬ Nat.Prime 573339081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191113027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11972 2) := by change Nat.Prime 573315137; exact ProjectEulerSolutions.P58PrimeNodes.p_573315137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11972 3) := by change ¬ Nat.Prime 573291193; exact PrattCertificates.composite_of_product (factor := 8011) (cofactor := 71563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11972 : row 11972 = 1 := (row_11972.checked).trans rfl
theorem model_count_11972 : modelRow 11972 = 1 := by decide
theorem modelrow_agrees_11972 : modelRow 11972 = row 11972 := model_count_11972.trans row_count_11972.symm
def row_11973 : RowCertificate 11973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11973 1) := by change ¬ Nat.Prime 573434863; exact PrattCertificates.composite_of_product (factor := 919) (cofactor := 623977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11973 2) := by change Nat.Prime 573410917; exact ProjectEulerSolutions.P58PrimeNodes.p_573410917.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 11973 3) := by change Nat.Prime 573386971; exact ProjectEulerSolutions.P58PrimeNodes.p_573386971.prime
    exact decide_eq_true prime
theorem row_count_11973 : row 11973 = 2 := (row_11973.checked).trans rfl
theorem model_count_11973 : modelRow 11973 = 2 := by decide
theorem modelrow_agrees_11973 : modelRow 11973 = row 11973 := model_count_11973.trans row_count_11973.symm
def row_11974 : RowCertificate 11974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11974 1) := by change ¬ Nat.Prime 573530653; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8560159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11974 2) := by change ¬ Nat.Prime 573506705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114701341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11974 3) := by change ¬ Nat.Prime 573482757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191160919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11974 : row 11974 = 0 := (row_11974.checked).trans rfl
theorem model_count_11974 : modelRow 11974 = 0 := by decide
theorem modelrow_agrees_11974 : modelRow 11974 = row 11974 := model_count_11974.trans row_count_11974.symm
def row_11975 : RowCertificate 11975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11975 1) := by change ¬ Nat.Prime 573626451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191208817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11975 2) := by change Nat.Prime 573602501; exact ProjectEulerSolutions.P58PrimeNodes.p_573602501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11975 3) := by change ¬ Nat.Prime 573578551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81939793) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11975 : row 11975 = 1 := (row_11975.checked).trans rfl
theorem model_count_11975 : modelRow 11975 = 1 := by decide
theorem modelrow_agrees_11975 : modelRow 11975 = row 11975 := model_count_11975.trans row_count_11975.symm
def row_11976 : RowCertificate 11976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11976 1) := by change ¬ Nat.Prime 573722257; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 7859209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11976 2) := by change ¬ Nat.Prime 573698305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114739661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11976 3) := by change ¬ Nat.Prime 573674353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 81953479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11976 : row 11976 = 0 := (row_11976.checked).trans rfl
theorem model_count_11976 : modelRow 11976 = 0 := by decide
theorem modelrow_agrees_11976 : modelRow 11976 = row 11976 := model_count_11976.trans row_count_11976.symm
def row_11977 : RowCertificate 11977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11977 1) := by change ¬ Nat.Prime 573818071; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 555487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11977 2) := by change ¬ Nat.Prime 573794117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44138009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11977 3) := by change ¬ Nat.Prime 573770163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191256721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11977 : row 11977 = 0 := (row_11977.checked).trans rfl
theorem model_count_11977 : modelRow 11977 = 0 := by decide
theorem modelrow_agrees_11977 : modelRow 11977 = row 11977 := model_count_11977.trans row_count_11977.symm
def row_11978 : RowCertificate 11978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11978 1) := by change ¬ Nat.Prime 573913893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191304631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11978 2) := by change ¬ Nat.Prime 573889937; exact PrattCertificates.composite_of_product (factor := 3557) (cofactor := 161341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11978 3) := by change ¬ Nat.Prime 573865981; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44143537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11978 : row 11978 = 0 := (row_11978.checked).trans rfl
theorem model_count_11978 : modelRow 11978 = 0 := by decide
theorem modelrow_agrees_11978 : modelRow 11978 = row 11978 := model_count_11978.trans row_count_11978.symm
def row_11979 : RowCertificate 11979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11979 1) := by change ¬ Nat.Prime 574009723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82001389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11979 2) := by change ¬ Nat.Prime 573985765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114797153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11979 3) := by change ¬ Nat.Prime 573961807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18514897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11979 : row 11979 = 0 := (row_11979.checked).trans rfl
theorem model_count_11979 : modelRow 11979 = 0 := by decide
theorem modelrow_agrees_11979 : modelRow 11979 = row 11979 := model_count_11979.trans row_count_11979.symm
def row_11980 : RowCertificate 11980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11980 1) := by change ¬ Nat.Prime 574105561; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7267159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 11980 2) := by change Nat.Prime 574081601; exact ProjectEulerSolutions.P58PrimeNodes.p_574081601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11980 3) := by change ¬ Nat.Prime 574057641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191352547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11980 : row 11980 = 1 := (row_11980.checked).trans rfl
theorem model_count_11980 : modelRow 11980 = 1 := by decide
theorem modelrow_agrees_11980 : modelRow 11980 = row 11980 := model_count_11980.trans row_count_11980.symm
def row_11981 : RowCertificate 11981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11981 1) := by change ¬ Nat.Prime 574201407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191400469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11981 2) := by change ¬ Nat.Prime 574177445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114835489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11981 3) := by change ¬ Nat.Prime 574153483; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 654679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11981 : row 11981 = 0 := (row_11981.checked).trans rfl
theorem model_count_11981 : modelRow 11981 = 0 := by decide
theorem modelrow_agrees_11981 : modelRow 11981 = row 11981 := model_count_11981.trans row_count_11981.symm
def row_11982 : RowCertificate 11982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11982 1) := by change ¬ Nat.Prime 574297261; exact PrattCertificates.composite_of_product (factor := 2833) (cofactor := 202717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11982 2) := by change ¬ Nat.Prime 574273297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44174869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11982 3) := by change ¬ Nat.Prime 574249333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82035619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11982 : row 11982 = 0 := (row_11982.checked).trans rfl
theorem model_count_11982 : modelRow 11982 = 0 := by decide
theorem modelrow_agrees_11982 : modelRow 11982 = row 11982 := model_count_11982.trans row_count_11982.symm
def row_11983 : RowCertificate 11983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11983 1) := by change ¬ Nat.Prime 574393123; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30231217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11983 2) := by change ¬ Nat.Prime 574369157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33786421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11983 3) := by change ¬ Nat.Prime 574345191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191448397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11983 : row 11983 = 0 := (row_11983.checked).trans rfl
theorem model_count_11983 : modelRow 11983 = 0 := by decide
theorem modelrow_agrees_11983 : modelRow 11983 = row 11983 := model_count_11983.trans row_count_11983.symm
def row_11984 : RowCertificate 11984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11984 1) := by change ¬ Nat.Prime 574488993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191496331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11984 2) := by change ¬ Nat.Prime 574465025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114893005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 11984 3) := by change Nat.Prime 574441057; exact ProjectEulerSolutions.P58PrimeNodes.p_574441057.prime
    exact decide_eq_true prime
theorem row_count_11984 : row 11984 = 1 := (row_11984.checked).trans rfl
theorem model_count_11984 : modelRow 11984 = 1 := by decide
theorem modelrow_agrees_11984 : modelRow 11984 = row 11984 := model_count_11984.trans row_count_11984.symm
def row_11985 : RowCertificate 11985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11985 1) := by change ¬ Nat.Prime 574584871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82083553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11985 2) := by change ¬ Nat.Prime 574560901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15528673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 11985 3) := by change Nat.Prime 574536931; exact ProjectEulerSolutions.P58PrimeNodes.p_574536931.prime
    exact decide_eq_true prime
theorem row_count_11985 : row 11985 = 1 := (row_11985.checked).trans rfl
theorem model_count_11985 : modelRow 11985 = 1 := by decide
theorem modelrow_agrees_11985 : modelRow 11985 = row 11985 := model_count_11985.trans row_count_11985.symm
def row_11986 : RowCertificate 11986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11986 1) := by change ¬ Nat.Prime 574680757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82097251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11986 2) := by change ¬ Nat.Prime 574656785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114931357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11986 3) := by change ¬ Nat.Prime 574632813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191544271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11986 : row 11986 = 0 := (row_11986.checked).trans rfl
theorem model_count_11986 : modelRow 11986 = 0 := by decide
theorem modelrow_agrees_11986 : modelRow 11986 = row 11986 := model_count_11986.trans row_count_11986.symm
def row_11987 : RowCertificate 11987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11987 1) := by change ¬ Nat.Prime 574776651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191592217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11987 2) := by change ¬ Nat.Prime 574752677; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33808981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11987 3) := by change ¬ Nat.Prime 574728703; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 1309177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11987 : row 11987 = 0 := (row_11987.checked).trans rfl
theorem model_count_11987 : modelRow 11987 = 0 := by decide
theorem modelrow_agrees_11987 : modelRow 11987 = row 11987 := model_count_11987.trans row_count_11987.symm
def row_11988 : RowCertificate 11988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11988 1) := by change ¬ Nat.Prime 574872553; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3807103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11988 2) := by change ¬ Nat.Prime 574848577; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14020697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11988 3) := by change ¬ Nat.Prime 574824601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44217277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11988 : row 11988 = 0 := (row_11988.checked).trans rfl
theorem model_count_11988 : modelRow 11988 = 0 := by decide
theorem modelrow_agrees_11988 : modelRow 11988 = row 11988 := model_count_11988.trans row_count_11988.symm
def row_11989 : RowCertificate 11989 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11989 1) := by change Nat.Prime 574968463; exact ProjectEulerSolutions.P58PrimeNodes.p_574968463.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11989 2) := by change ¬ Nat.Prime 574944485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 114988897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11989 3) := by change ¬ Nat.Prime 574920507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191640169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11989 : row 11989 = 1 := (row_11989.checked).trans rfl
theorem model_count_11989 : modelRow 11989 = 1 := by decide
theorem modelrow_agrees_11989 : modelRow 11989 = row 11989 := model_count_11989.trans row_count_11989.symm
def row_11990 : RowCertificate 11990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11990 1) := by change ¬ Nat.Prime 575064381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191688127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11990 2) := by change ¬ Nat.Prime 575040401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44233877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11990 3) := by change ¬ Nat.Prime 575016421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82145203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11990 : row 11990 = 0 := (row_11990.checked).trans rfl
theorem model_count_11990 : modelRow 11990 = 0 := by decide
theorem modelrow_agrees_11990 : modelRow 11990 = row 11990 := model_count_11990.trans row_count_11990.symm
def row_11991 : RowCertificate 11991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11991 1) := by change ¬ Nat.Prime 575160307; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 2076391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11991 2) := by change ¬ Nat.Prime 575136325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115027265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11991 3) := by change ¬ Nat.Prime 575112343; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44239411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11991 : row 11991 = 0 := (row_11991.checked).trans rfl
theorem model_count_11991 : modelRow 11991 = 0 := by decide
theorem modelrow_agrees_11991 : modelRow 11991 = row 11991 := model_count_11991.trans row_count_11991.symm
def row_11992 : RowCertificate 11992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11992 1) := by change ¬ Nat.Prime 575256241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82179463) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11992 2) := by change ¬ Nat.Prime 575232257; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9430037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11992 3) := by change ¬ Nat.Prime 575208273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191736091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11992 : row 11992 = 0 := (row_11992.checked).trans rfl
theorem model_count_11992 : modelRow 11992 = 0 := by decide
theorem modelrow_agrees_11992 : modelRow 11992 = row 11992 := model_count_11992.trans row_count_11992.symm
def row_11993 : RowCertificate 11993 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11993 1) := by change ¬ Nat.Prime 575352183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191784061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11993 2) := by change ¬ Nat.Prime 575328197; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 10855249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11993 3) := by change ¬ Nat.Prime 575304211; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30279169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11993 : row 11993 = 0 := (row_11993.checked).trans rfl
theorem model_count_11993 : modelRow 11993 = 0 := by decide
theorem modelrow_agrees_11993 : modelRow 11993 = row 11993 := model_count_11993.trans row_count_11993.symm
def row_11994 : RowCertificate 11994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11994 1) := by change ¬ Nat.Prime 575448133; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44265241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11994 2) := by change ¬ Nat.Prime 575424145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115084829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11994 3) := by change ¬ Nat.Prime 575400157; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13381399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11994 : row 11994 = 0 := (row_11994.checked).trans rfl
theorem model_count_11994 : modelRow 11994 = 0 := by decide
theorem modelrow_agrees_11994 : modelRow 11994 = row 11994 := model_count_11994.trans row_count_11994.symm
def row_11995 : RowCertificate 11995 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11995 1) := by change Nat.Prime 575544091; exact ProjectEulerSolutions.P58PrimeNodes.p_575544091.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11995 2) := by change ¬ Nat.Prime 575520101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44270777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11995 3) := by change ¬ Nat.Prime 575496111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191832037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11995 : row 11995 = 1 := (row_11995.checked).trans rfl
theorem model_count_11995 : modelRow 11995 = 1 := by decide
theorem modelrow_agrees_11995 : modelRow 11995 = row 11995 := model_count_11995.trans row_count_11995.symm
def row_11996 : RowCertificate 11996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11996 1) := by change ¬ Nat.Prime 575640057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191880019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11996 2) := by change ¬ Nat.Prime 575616065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115123213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11996 3) := by change ¬ Nat.Prime 575592073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82227439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11996 : row 11996 = 0 := (row_11996.checked).trans rfl
theorem model_count_11996 : modelRow 11996 = 0 := by decide
theorem modelrow_agrees_11996 : modelRow 11996 = row 11996 := model_count_11996.trans row_count_11996.symm
def row_11997 : RowCertificate 11997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11997 1) := by change ¬ Nat.Prime 575736031; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44287387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11997 2) := by change ¬ Nat.Prime 575712037; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14041757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11997 3) := by change ¬ Nat.Prime 575688043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82241149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11997 : row 11997 = 0 := (row_11997.checked).trans rfl
theorem model_count_11997 : modelRow 11997 = 0 := by decide
theorem modelrow_agrees_11997 : modelRow 11997 = row 11997 := model_count_11997.trans row_count_11997.symm
def row_11998 : RowCertificate 11998 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11998 1) := by change Nat.Prime 575832013; exact ProjectEulerSolutions.P58PrimeNodes.p_575832013.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11998 2) := by change ¬ Nat.Prime 575808017; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 6469753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11998 3) := by change ¬ Nat.Prime 575784021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191928007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11998 : row 11998 = 1 := (row_11998.checked).trans rfl
theorem model_count_11998 : modelRow 11998 = 1 := by decide
theorem modelrow_agrees_11998 : modelRow 11998 = row 11998 := model_count_11998.trans row_count_11998.symm
def row_11999 : RowCertificate 11999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11999 1) := by change ¬ Nat.Prime 575928003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 191976001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11999 2) := by change ¬ Nat.Prime 575904005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115180801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11999 3) := by change ¬ Nat.Prime 575880007; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 1101109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11999 : row 11999 = 0 := (row_11999.checked).trans rfl
theorem model_count_11999 : modelRow 11999 = 0 := by decide
theorem modelrow_agrees_11999 : modelRow 11999 = row 11999 := model_count_11999.trans row_count_11999.symm
def row_12000 : RowCertificate 12000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12000 1) := by change ¬ Nat.Prime 576024001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82289143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12000 2) := by change ¬ Nat.Prime 576000001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33882353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12000 3) := by change ¬ Nat.Prime 575976001; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18579871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12000 : row 12000 = 0 := (row_12000.checked).trans rfl
theorem model_count_12000 : modelRow 12000 = 0 := by decide
theorem modelrow_agrees_12000 : modelRow 12000 = row 12000 := model_count_12000.trans row_count_12000.symm
def row_12001 : RowCertificate 12001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12001 1) := by change ¬ Nat.Prime 576120007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15570811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12001 2) := by change ¬ Nat.Prime 576096005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115219201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12001 3) := by change ¬ Nat.Prime 576072003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192024001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12001 : row 12001 = 0 := (row_12001.checked).trans rfl
theorem model_count_12001 : modelRow 12001 = 0 := by decide
theorem modelrow_agrees_12001 : modelRow 12001 = row 12001 := model_count_12001.trans row_count_12001.symm
def row_12002 : RowCertificate 12002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12002 1) := by change ¬ Nat.Prime 576216021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192072007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12002 2) := by change Nat.Prime 576192017; exact ProjectEulerSolutions.P58PrimeNodes.p_576192017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12002 3) := by change ¬ Nat.Prime 576168013; exact PrattCertificates.composite_of_product (factor := 12097) (cofactor := 47629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12002 : row 12002 = 1 := (row_12002.checked).trans rfl
theorem model_count_12002 : modelRow 12002 = 1 := by decide
theorem modelrow_agrees_12002 : modelRow 12002 = row 12002 := model_count_12002.trans row_count_12002.symm
def row_12003 : RowCertificate 12003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12003 1) := by change ¬ Nat.Prime 576312043; exact PrattCertificates.composite_of_product (factor := 1543) (cofactor := 373501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12003 2) := by change ¬ Nat.Prime 576288037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44329849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12003 3) := by change ¬ Nat.Prime 576264031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82323433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12003 : row 12003 = 0 := (row_12003.checked).trans rfl
theorem model_count_12003 : modelRow 12003 = 0 := by decide
theorem modelrow_agrees_12003 : modelRow 12003 = row 12003 := model_count_12003.trans row_count_12003.symm
def row_12004 : RowCertificate 12004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12004 1) := by change ¬ Nat.Prime 576408073; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30337267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12004 2) := by change ¬ Nat.Prime 576384065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115276813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12004 3) := by change ¬ Nat.Prime 576360057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192120019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12004 : row 12004 = 0 := (row_12004.checked).trans rfl
theorem model_count_12004 : modelRow 12004 = 0 := by decide
theorem modelrow_agrees_12004 : modelRow 12004 = row 12004 := model_count_12004.trans row_count_12004.symm
def row_12005 : RowCertificate 12005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12005 1) := by change ¬ Nat.Prime 576504111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192168037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12005 2) := by change ¬ Nat.Prime 576480101; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 813089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12005 3) := by change ¬ Nat.Prime 576456091; exact PrattCertificates.composite_of_product (factor := 19081) (cofactor := 30211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12005 : row 12005 = 0 := (row_12005.checked).trans rfl
theorem model_count_12005 : modelRow 12005 = 0 := by decide
theorem modelrow_agrees_12005 : modelRow 12005 = row 12005 := model_count_12005.trans row_count_12005.symm
def row_12006 : RowCertificate 12006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12006 1) := by change ¬ Nat.Prime 576600157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82371451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12006 2) := by change ¬ Nat.Prime 576576145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115315229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12006 3) := by change Nat.Prime 576552133; exact ProjectEulerSolutions.P58PrimeNodes.p_576552133.prime
    exact decide_eq_true prime
theorem row_count_12006 : row 12006 = 1 := (row_12006.checked).trans rfl
theorem model_count_12006 : modelRow 12006 = 1 := by decide
theorem modelrow_agrees_12006 : modelRow 12006 = row 12006 := model_count_12006.trans row_count_12006.symm
def row_12007 : RowCertificate 12007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12007 1) := by change ¬ Nat.Prime 576696211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82385173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12007 2) := by change Nat.Prime 576672197; exact ProjectEulerSolutions.P58PrimeNodes.p_576672197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12007 3) := by change ¬ Nat.Prime 576648183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192216061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12007 : row 12007 = 1 := (row_12007.checked).trans rfl
theorem model_count_12007 : modelRow 12007 = 1 := by decide
theorem modelrow_agrees_12007 : modelRow 12007 = row 12007 := model_count_12007.trans row_count_12007.symm
def row_12008 : RowCertificate 12008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12008 1) := by change ¬ Nat.Prime 576792273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192264091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12008 2) := by change ¬ Nat.Prime 576768257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44366789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12008 3) := by change ¬ Nat.Prime 576744241; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8608123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12008 : row 12008 = 0 := (row_12008.checked).trans rfl
theorem model_count_12008 : modelRow 12008 = 0 := by decide
theorem modelrow_agrees_12008 : modelRow 12008 = row 12008 := model_count_12008.trans row_count_12008.symm
def row_12009 : RowCertificate 12009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12009 1) := by change ¬ Nat.Prime 576888343; exact PrattCertificates.composite_of_product (factor := 22783) (cofactor := 25321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12009 2) := by change ¬ Nat.Prime 576864325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115372865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12009 3) := by change ¬ Nat.Prime 576840307; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 1370167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12009 : row 12009 = 0 := (row_12009.checked).trans rfl
theorem model_count_12009 : modelRow 12009 = 0 := by decide
theorem modelrow_agrees_12009 : modelRow 12009 = row 12009 := model_count_12009.trans row_count_12009.symm
def row_12010 : RowCertificate 12010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12010 1) := by change ¬ Nat.Prime 576984421; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44383417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12010 2) := by change ¬ Nat.Prime 576960401; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 2928733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12010 3) := by change ¬ Nat.Prime 576936381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192312127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12010 : row 12010 = 0 := (row_12010.checked).trans rfl
theorem model_count_12010 : modelRow 12010 = 0 := by decide
theorem modelrow_agrees_12010 : modelRow 12010 = row 12010 := model_count_12010.trans row_count_12010.symm
def row_12011 : RowCertificate 12011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12011 1) := by change ¬ Nat.Prime 577080507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192360169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12011 2) := by change ¬ Nat.Prime 577056485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115411297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12011 3) := by change ¬ Nat.Prime 577032463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82433209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12011 : row 12011 = 0 := (row_12011.checked).trans rfl
theorem model_count_12011 : modelRow 12011 = 0 := by decide
theorem modelrow_agrees_12011 : modelRow 12011 = row 12011 := model_count_12011.trans row_count_12011.symm
def row_12012 : RowCertificate 12012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12012 1) := by change ¬ Nat.Prime 577176601; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 3188821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12012 2) := by change ¬ Nat.Prime 577152577; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 19901813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12012 3) := by change ¬ Nat.Prime 577128553; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 30375187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12012 : row 12012 = 0 := (row_12012.checked).trans rfl
theorem model_count_12012 : modelRow 12012 = 0 := by decide
theorem modelrow_agrees_12012 : modelRow 12012 = row 12012 := model_count_12012.trans row_count_12012.symm
def row_12013 : RowCertificate 12013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12013 1) := by change ¬ Nat.Prime 577272703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82467529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12013 2) := by change ¬ Nat.Prime 577248677; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 1371137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12013 3) := by change ¬ Nat.Prime 577224651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192408217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12013 : row 12013 = 0 := (row_12013.checked).trans rfl
theorem model_count_12013 : modelRow 12013 = 0 := by decide
theorem modelrow_agrees_12013 : modelRow 12013 = row 12013 := model_count_12013.trans row_count_12013.symm
def row_12014 : RowCertificate 12014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12014 1) := by change ¬ Nat.Prime 577368813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192456271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12014 2) := by change ¬ Nat.Prime 577344785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115468957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12014 3) := by change ¬ Nat.Prime 577320757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44409289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12014 : row 12014 = 0 := (row_12014.checked).trans rfl
theorem model_count_12014 : modelRow 12014 = 0 := by decide
theorem modelrow_agrees_12014 : modelRow 12014 = row 12014 := model_count_12014.trans row_count_12014.symm
def row_12015 : RowCertificate 12015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12015 1) := by change ¬ Nat.Prime 577464931; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18627901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12015 2) := by change Nat.Prime 577440901; exact ProjectEulerSolutions.P58PrimeNodes.p_577440901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12015 3) := by change ¬ Nat.Prime 577416871; exact PrattCertificates.composite_of_product (factor := 13627) (cofactor := 42373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12015 : row 12015 = 1 := (row_12015.checked).trans rfl
theorem model_count_12015 : modelRow 12015 = 1 := by decide
theorem modelrow_agrees_12015 : modelRow 12015 = row 12015 := model_count_12015.trans row_count_12015.symm
def row_12016 : RowCertificate 12016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12016 1) := by change ¬ Nat.Prime 577561057; exact PrattCertificates.composite_of_product (factor := 967) (cofactor := 597271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12016 2) := by change ¬ Nat.Prime 577537025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115507405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12016 3) := by change ¬ Nat.Prime 577512993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192504331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12016 : row 12016 = 0 := (row_12016.checked).trans rfl
theorem model_count_12016 : modelRow 12016 = 0 := by decide
theorem modelrow_agrees_12016 : modelRow 12016 = row 12016 := model_count_12016.trans row_count_12016.symm
def row_12017 : RowCertificate 12017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12017 1) := by change ¬ Nat.Prime 577657191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192552397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12017 2) := by change ¬ Nat.Prime 577633157; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 33978421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12017 3) := by change ¬ Nat.Prime 577609123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82515589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12017 : row 12017 = 0 := (row_12017.checked).trans rfl
theorem model_count_12017 : modelRow 12017 = 0 := by decide
theorem modelrow_agrees_12017 : modelRow 12017 = row 12017 := model_count_12017.trans row_count_12017.symm
def row_12018 : RowCertificate 12018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12018 1) := by change ¬ Nat.Prime 577753333; exact PrattCertificates.composite_of_product (factor := 21493) (cofactor := 26881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12018 2) := by change ¬ Nat.Prime 577729297; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 2397217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12018 3) := by change ¬ Nat.Prime 577705261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82529323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12018 : row 12018 = 0 := (row_12018.checked).trans rfl
theorem model_count_12018 : modelRow 12018 = 0 := by decide
theorem modelrow_agrees_12018 : modelRow 12018 = row 12018 := model_count_12018.trans row_count_12018.symm
def row_12019 : RowCertificate 12019 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12019 1) := by change Nat.Prime 577849483; exact ProjectEulerSolutions.P58PrimeNodes.p_577849483.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12019 2) := by change ¬ Nat.Prime 577825445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115565089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12019 3) := by change ¬ Nat.Prime 577801407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192600469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12019 : row 12019 = 1 := (row_12019.checked).trans rfl
theorem model_count_12019 : modelRow 12019 = 1 := by decide
theorem modelrow_agrees_12019 : modelRow 12019 = row 12019 := model_count_12019.trans row_count_12019.symm
def row_12020 : RowCertificate 12020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12020 1) := by change ¬ Nat.Prime 577945641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192648547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12020 2) := by change ¬ Nat.Prime 577921601; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 1441201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12020 3) := by change ¬ Nat.Prime 577897561; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15618853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12020 : row 12020 = 0 := (row_12020.checked).trans rfl
theorem model_count_12020 : modelRow 12020 = 0 := by decide
theorem modelrow_agrees_12020 : modelRow 12020 = row 12020 := model_count_12020.trans row_count_12020.symm
def row_12021 : RowCertificate 12021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12021 1) := by change ¬ Nat.Prime 578041807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82577401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12021 2) := by change ¬ Nat.Prime 578017765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115603553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12021 3) := by change ¬ Nat.Prime 577993723; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3827773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12021 : row 12021 = 0 := (row_12021.checked).trans rfl
theorem model_count_12021 : modelRow 12021 = 0 := by decide
theorem modelrow_agrees_12021 : modelRow 12021 = row 12021 := model_count_12021.trans row_count_12021.symm
def row_12022 : RowCertificate 12022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12022 1) := by change ¬ Nat.Prime 578137981; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 2592547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12022 2) := by change ¬ Nat.Prime 578113937; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15624701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12022 3) := by change ¬ Nat.Prime 578089893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192696631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12022 : row 12022 = 0 := (row_12022.checked).trans rfl
theorem model_count_12022 : modelRow 12022 = 0 := by decide
theorem modelrow_agrees_12022 : modelRow 12022 = row 12022 := model_count_12022.trans row_count_12022.symm
def row_12023 : RowCertificate 12023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12023 1) := by change ¬ Nat.Prime 578234163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192744721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12023 2) := by change ¬ Nat.Prime 578210117; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 1637989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12023 3) := by change ¬ Nat.Prime 578186071; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 5613457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12023 : row 12023 = 0 := (row_12023.checked).trans rfl
theorem model_count_12023 : modelRow 12023 = 0 := by decide
theorem modelrow_agrees_12023 : modelRow 12023 = row 12023 := model_count_12023.trans row_count_12023.symm
def row_12024 : RowCertificate 12024 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12024 1) := by change Nat.Prime 578330353; exact ProjectEulerSolutions.P58PrimeNodes.p_578330353.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12024 2) := by change ¬ Nat.Prime 578306305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115661261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12024 3) := by change ¬ Nat.Prime 578282257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82611751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12024 : row 12024 = 1 := (row_12024.checked).trans rfl
theorem model_count_12024 : modelRow 12024 = 1 := by decide
theorem modelrow_agrees_12024 : modelRow 12024 = row 12024 := model_count_12024.trans row_count_12024.symm
def row_12025 : RowCertificate 12025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12025 1) := by change ¬ Nat.Prime 578426551; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 18658921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12025 2) := by change Nat.Prime 578402501; exact ProjectEulerSolutions.P58PrimeNodes.p_578402501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12025 3) := by change ¬ Nat.Prime 578378451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192792817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12025 : row 12025 = 1 := (row_12025.checked).trans rfl
theorem model_count_12025 : modelRow 12025 = 1 := by decide
theorem modelrow_agrees_12025 : modelRow 12025 = row 12025 := model_count_12025.trans row_count_12025.symm
def row_12026 : RowCertificate 12026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12026 1) := by change ¬ Nat.Prime 578522757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192840919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12026 2) := by change ¬ Nat.Prime 578498705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115699741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12026 3) := by change Nat.Prime 578474653; exact ProjectEulerSolutions.P58PrimeNodes.p_578474653.prime
    exact decide_eq_true prime
theorem row_count_12026 : row 12026 = 1 := (row_12026.checked).trans rfl
theorem model_count_12026 : modelRow 12026 = 1 := by decide
theorem modelrow_agrees_12026 : modelRow 12026 = row 12026 := model_count_12026.trans row_count_12026.symm
def row_12027 : RowCertificate 12027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12027 1) := by change ¬ Nat.Prime 578618971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82659853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12027 2) := by change ¬ Nat.Prime 578594917; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 5120309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12027 3) := by change ¬ Nat.Prime 578570863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44505451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12027 : row 12027 = 0 := (row_12027.checked).trans rfl
theorem model_count_12027 : modelRow 12027 = 0 := by decide
theorem modelrow_agrees_12027 : modelRow 12027 = row 12027 := model_count_12027.trans row_count_12027.symm
def row_12028 : RowCertificate 12028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12028 1) := by change ¬ Nat.Prime 578715193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82673599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12028 2) := by change ¬ Nat.Prime 578691137; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15640301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12028 3) := by change ¬ Nat.Prime 578667081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192889027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12028 : row 12028 = 0 := (row_12028.checked).trans rfl
theorem model_count_12028 : modelRow 12028 = 0 := by decide
theorem modelrow_agrees_12028 : modelRow 12028 = row 12028 := model_count_12028.trans row_count_12028.symm
def row_12029 : RowCertificate 12029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12029 1) := by change ¬ Nat.Prime 578811423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192937141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12029 2) := by change ¬ Nat.Prime 578787365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115757473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12029 3) := by change Nat.Prime 578763307; exact ProjectEulerSolutions.P58PrimeNodes.p_578763307.prime
    exact decide_eq_true prime
theorem row_count_12029 : row 12029 = 1 := (row_12029.checked).trans rfl
theorem model_count_12029 : modelRow 12029 = 1 := by decide
theorem modelrow_agrees_12029 : modelRow 12029 = row 12029 := model_count_12029.trans row_count_12029.symm
def row_12030 : RowCertificate 12030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12030 1) := by change ¬ Nat.Prime 578907661; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 15646153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12030 2) := by change Nat.Prime 578883601; exact ProjectEulerSolutions.P58PrimeNodes.p_578883601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12030 3) := by change ¬ Nat.Prime 578859541; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 44527657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12030 : row 12030 = 1 := (row_12030.checked).trans rfl
theorem model_count_12030 : modelRow 12030 = 1 := by decide
theorem modelrow_agrees_12030 : modelRow 12030 = row 12030 := model_count_12030.trans row_count_12030.symm
def row_12031 : RowCertificate 12031 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12031 1) := by change Nat.Prime 579003907; exact ProjectEulerSolutions.P58PrimeNodes.p_579003907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12031 2) := by change ¬ Nat.Prime 578979845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 115795969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12031 3) := by change ¬ Nat.Prime 578955783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 192985261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12031 : row 12031 = 1 := (row_12031.checked).trans rfl
theorem model_count_12031 : modelRow 12031 = 1 := by decide
theorem modelrow_agrees_12031 : modelRow 12031 = row 12031 := model_count_12031.trans row_count_12031.symm
def row_12032 : RowCertificate 12032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12032 1) := by change ¬ Nat.Prime 579100161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 193033387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12032 2) := by change ¬ Nat.Prime 579076097; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 6506473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12032 3) := by change ¬ Nat.Prime 579052033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 82721719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12032 : row 12032 = 0 := (row_12032.checked).trans rfl
theorem model_count_12032 : modelRow 12032 = 0 := by decide
theorem modelrow_agrees_12032 : modelRow 12032 = row 12032 := model_count_12032.trans row_count_12032.symm
private def atomic_11905 : CountSegment 1 10 13120 11904 11905 4829 4829 := CountSegment.single (first := 11904) (initial := 4829) row_count_11905 (by decide)
private theorem atomic_11905_agreement : RowAgreement 11904 11905 := RowAgreement.single (first := 11904) modelrow_agrees_11905
private def atomic_11906 : CountSegment 1 10 13120 11905 11906 4829 4829 := CountSegment.single (first := 11905) (initial := 4829) row_count_11906 (by decide)
private theorem atomic_11906_agreement : RowAgreement 11905 11906 := RowAgreement.single (first := 11905) modelrow_agrees_11906
private def atomic_11907 : CountSegment 1 10 13120 11906 11907 4829 4830 := CountSegment.single (first := 11906) (initial := 4829) row_count_11907 (by decide)
private theorem atomic_11907_agreement : RowAgreement 11906 11907 := RowAgreement.single (first := 11906) modelrow_agrees_11907
private def atomic_11908 : CountSegment 1 10 13120 11907 11908 4830 4831 := CountSegment.single (first := 11907) (initial := 4830) row_count_11908 (by decide)
private theorem atomic_11908_agreement : RowAgreement 11907 11908 := RowAgreement.single (first := 11907) modelrow_agrees_11908
private def atomic_11909 : CountSegment 1 10 13120 11908 11909 4831 4831 := CountSegment.single (first := 11908) (initial := 4831) row_count_11909 (by decide)
private theorem atomic_11909_agreement : RowAgreement 11908 11909 := RowAgreement.single (first := 11908) modelrow_agrees_11909
private def atomic_11910 : CountSegment 1 10 13120 11909 11910 4831 4831 := CountSegment.single (first := 11909) (initial := 4831) row_count_11910 (by decide)
private theorem atomic_11910_agreement : RowAgreement 11909 11910 := RowAgreement.single (first := 11909) modelrow_agrees_11910
private def atomic_11911 : CountSegment 1 10 13120 11910 11911 4831 4831 := CountSegment.single (first := 11910) (initial := 4831) row_count_11911 (by decide)
private theorem atomic_11911_agreement : RowAgreement 11910 11911 := RowAgreement.single (first := 11910) modelrow_agrees_11911
private def atomic_11912 : CountSegment 1 10 13120 11911 11912 4831 4831 := CountSegment.single (first := 11911) (initial := 4831) row_count_11912 (by decide)
private theorem atomic_11912_agreement : RowAgreement 11911 11912 := RowAgreement.single (first := 11911) modelrow_agrees_11912
private def atomic_11913 : CountSegment 1 10 13120 11912 11913 4831 4831 := CountSegment.single (first := 11912) (initial := 4831) row_count_11913 (by decide)
private theorem atomic_11913_agreement : RowAgreement 11912 11913 := RowAgreement.single (first := 11912) modelrow_agrees_11913
private def atomic_11914 : CountSegment 1 10 13120 11913 11914 4831 4831 := CountSegment.single (first := 11913) (initial := 4831) row_count_11914 (by decide)
private theorem atomic_11914_agreement : RowAgreement 11913 11914 := RowAgreement.single (first := 11913) modelrow_agrees_11914
private def atomic_11915 : CountSegment 1 10 13120 11914 11915 4831 4831 := CountSegment.single (first := 11914) (initial := 4831) row_count_11915 (by decide)
private theorem atomic_11915_agreement : RowAgreement 11914 11915 := RowAgreement.single (first := 11914) modelrow_agrees_11915
private def atomic_11916 : CountSegment 1 10 13120 11915 11916 4831 4831 := CountSegment.single (first := 11915) (initial := 4831) row_count_11916 (by decide)
private theorem atomic_11916_agreement : RowAgreement 11915 11916 := RowAgreement.single (first := 11915) modelrow_agrees_11916
private def atomic_11917 : CountSegment 1 10 13120 11916 11917 4831 4831 := CountSegment.single (first := 11916) (initial := 4831) row_count_11917 (by decide)
private theorem atomic_11917_agreement : RowAgreement 11916 11917 := RowAgreement.single (first := 11916) modelrow_agrees_11917
private def atomic_11918 : CountSegment 1 10 13120 11917 11918 4831 4831 := CountSegment.single (first := 11917) (initial := 4831) row_count_11918 (by decide)
private theorem atomic_11918_agreement : RowAgreement 11917 11918 := RowAgreement.single (first := 11917) modelrow_agrees_11918
private def atomic_11919 : CountSegment 1 10 13120 11918 11919 4831 4831 := CountSegment.single (first := 11918) (initial := 4831) row_count_11919 (by decide)
private theorem atomic_11919_agreement : RowAgreement 11918 11919 := RowAgreement.single (first := 11918) modelrow_agrees_11919
private def atomic_11920 : CountSegment 1 10 13120 11919 11920 4831 4831 := CountSegment.single (first := 11919) (initial := 4831) row_count_11920 (by decide)
private theorem atomic_11920_agreement : RowAgreement 11919 11920 := RowAgreement.single (first := 11919) modelrow_agrees_11920
private def atomic_11921 : CountSegment 1 10 13120 11920 11921 4831 4831 := CountSegment.single (first := 11920) (initial := 4831) row_count_11921 (by decide)
private theorem atomic_11921_agreement : RowAgreement 11920 11921 := RowAgreement.single (first := 11920) modelrow_agrees_11921
private def atomic_11922 : CountSegment 1 10 13120 11921 11922 4831 4831 := CountSegment.single (first := 11921) (initial := 4831) row_count_11922 (by decide)
private theorem atomic_11922_agreement : RowAgreement 11921 11922 := RowAgreement.single (first := 11921) modelrow_agrees_11922
private def atomic_11923 : CountSegment 1 10 13120 11922 11923 4831 4831 := CountSegment.single (first := 11922) (initial := 4831) row_count_11923 (by decide)
private theorem atomic_11923_agreement : RowAgreement 11922 11923 := RowAgreement.single (first := 11922) modelrow_agrees_11923
private def atomic_11924 : CountSegment 1 10 13120 11923 11924 4831 4831 := CountSegment.single (first := 11923) (initial := 4831) row_count_11924 (by decide)
private theorem atomic_11924_agreement : RowAgreement 11923 11924 := RowAgreement.single (first := 11923) modelrow_agrees_11924
private def atomic_11925 : CountSegment 1 10 13120 11924 11925 4831 4832 := CountSegment.single (first := 11924) (initial := 4831) row_count_11925 (by decide)
private theorem atomic_11925_agreement : RowAgreement 11924 11925 := RowAgreement.single (first := 11924) modelrow_agrees_11925
private def atomic_11926 : CountSegment 1 10 13120 11925 11926 4832 4832 := CountSegment.single (first := 11925) (initial := 4832) row_count_11926 (by decide)
private theorem atomic_11926_agreement : RowAgreement 11925 11926 := RowAgreement.single (first := 11925) modelrow_agrees_11926
private def atomic_11927 : CountSegment 1 10 13120 11926 11927 4832 4833 := CountSegment.single (first := 11926) (initial := 4832) row_count_11927 (by decide)
private theorem atomic_11927_agreement : RowAgreement 11926 11927 := RowAgreement.single (first := 11926) modelrow_agrees_11927
private def atomic_11928 : CountSegment 1 10 13120 11927 11928 4833 4833 := CountSegment.single (first := 11927) (initial := 4833) row_count_11928 (by decide)
private theorem atomic_11928_agreement : RowAgreement 11927 11928 := RowAgreement.single (first := 11927) modelrow_agrees_11928
private def atomic_11929 : CountSegment 1 10 13120 11928 11929 4833 4833 := CountSegment.single (first := 11928) (initial := 4833) row_count_11929 (by decide)
private theorem atomic_11929_agreement : RowAgreement 11928 11929 := RowAgreement.single (first := 11928) modelrow_agrees_11929
private def atomic_11930 : CountSegment 1 10 13120 11929 11930 4833 4833 := CountSegment.single (first := 11929) (initial := 4833) row_count_11930 (by decide)
private theorem atomic_11930_agreement : RowAgreement 11929 11930 := RowAgreement.single (first := 11929) modelrow_agrees_11930
private def atomic_11931 : CountSegment 1 10 13120 11930 11931 4833 4834 := CountSegment.single (first := 11930) (initial := 4833) row_count_11931 (by decide)
private theorem atomic_11931_agreement : RowAgreement 11930 11931 := RowAgreement.single (first := 11930) modelrow_agrees_11931
private def atomic_11932 : CountSegment 1 10 13120 11931 11932 4834 4834 := CountSegment.single (first := 11931) (initial := 4834) row_count_11932 (by decide)
private theorem atomic_11932_agreement : RowAgreement 11931 11932 := RowAgreement.single (first := 11931) modelrow_agrees_11932
private def atomic_11933 : CountSegment 1 10 13120 11932 11933 4834 4835 := CountSegment.single (first := 11932) (initial := 4834) row_count_11933 (by decide)
private theorem atomic_11933_agreement : RowAgreement 11932 11933 := RowAgreement.single (first := 11932) modelrow_agrees_11933
private def atomic_11934 : CountSegment 1 10 13120 11933 11934 4835 4835 := CountSegment.single (first := 11933) (initial := 4835) row_count_11934 (by decide)
private theorem atomic_11934_agreement : RowAgreement 11933 11934 := RowAgreement.single (first := 11933) modelrow_agrees_11934
private def atomic_11935 : CountSegment 1 10 13120 11934 11935 4835 4835 := CountSegment.single (first := 11934) (initial := 4835) row_count_11935 (by decide)
private theorem atomic_11935_agreement : RowAgreement 11934 11935 := RowAgreement.single (first := 11934) modelrow_agrees_11935
private def atomic_11936 : CountSegment 1 10 13120 11935 11936 4835 4835 := CountSegment.single (first := 11935) (initial := 4835) row_count_11936 (by decide)
private theorem atomic_11936_agreement : RowAgreement 11935 11936 := RowAgreement.single (first := 11935) modelrow_agrees_11936
private def atomic_11937 : CountSegment 1 10 13120 11936 11937 4835 4836 := CountSegment.single (first := 11936) (initial := 4835) row_count_11937 (by decide)
private theorem atomic_11937_agreement : RowAgreement 11936 11937 := RowAgreement.single (first := 11936) modelrow_agrees_11937
private def atomic_11938 : CountSegment 1 10 13120 11937 11938 4836 4836 := CountSegment.single (first := 11937) (initial := 4836) row_count_11938 (by decide)
private theorem atomic_11938_agreement : RowAgreement 11937 11938 := RowAgreement.single (first := 11937) modelrow_agrees_11938
private def atomic_11939 : CountSegment 1 10 13120 11938 11939 4836 4836 := CountSegment.single (first := 11938) (initial := 4836) row_count_11939 (by decide)
private theorem atomic_11939_agreement : RowAgreement 11938 11939 := RowAgreement.single (first := 11938) modelrow_agrees_11939
private def atomic_11940 : CountSegment 1 10 13120 11939 11940 4836 4836 := CountSegment.single (first := 11939) (initial := 4836) row_count_11940 (by decide)
private theorem atomic_11940_agreement : RowAgreement 11939 11940 := RowAgreement.single (first := 11939) modelrow_agrees_11940
private def atomic_11941 : CountSegment 1 10 13120 11940 11941 4836 4836 := CountSegment.single (first := 11940) (initial := 4836) row_count_11941 (by decide)
private theorem atomic_11941_agreement : RowAgreement 11940 11941 := RowAgreement.single (first := 11940) modelrow_agrees_11941
private def atomic_11942 : CountSegment 1 10 13120 11941 11942 4836 4836 := CountSegment.single (first := 11941) (initial := 4836) row_count_11942 (by decide)
private theorem atomic_11942_agreement : RowAgreement 11941 11942 := RowAgreement.single (first := 11941) modelrow_agrees_11942
private def atomic_11943 : CountSegment 1 10 13120 11942 11943 4836 4836 := CountSegment.single (first := 11942) (initial := 4836) row_count_11943 (by decide)
private theorem atomic_11943_agreement : RowAgreement 11942 11943 := RowAgreement.single (first := 11942) modelrow_agrees_11943
private def atomic_11944 : CountSegment 1 10 13120 11943 11944 4836 4836 := CountSegment.single (first := 11943) (initial := 4836) row_count_11944 (by decide)
private theorem atomic_11944_agreement : RowAgreement 11943 11944 := RowAgreement.single (first := 11943) modelrow_agrees_11944
private def atomic_11945 : CountSegment 1 10 13120 11944 11945 4836 4837 := CountSegment.single (first := 11944) (initial := 4836) row_count_11945 (by decide)
private theorem atomic_11945_agreement : RowAgreement 11944 11945 := RowAgreement.single (first := 11944) modelrow_agrees_11945
private def atomic_11946 : CountSegment 1 10 13120 11945 11946 4837 4838 := CountSegment.single (first := 11945) (initial := 4837) row_count_11946 (by decide)
private theorem atomic_11946_agreement : RowAgreement 11945 11946 := RowAgreement.single (first := 11945) modelrow_agrees_11946
private def atomic_11947 : CountSegment 1 10 13120 11946 11947 4838 4838 := CountSegment.single (first := 11946) (initial := 4838) row_count_11947 (by decide)
private theorem atomic_11947_agreement : RowAgreement 11946 11947 := RowAgreement.single (first := 11946) modelrow_agrees_11947
private def atomic_11948 : CountSegment 1 10 13120 11947 11948 4838 4838 := CountSegment.single (first := 11947) (initial := 4838) row_count_11948 (by decide)
private theorem atomic_11948_agreement : RowAgreement 11947 11948 := RowAgreement.single (first := 11947) modelrow_agrees_11948
private def atomic_11949 : CountSegment 1 10 13120 11948 11949 4838 4838 := CountSegment.single (first := 11948) (initial := 4838) row_count_11949 (by decide)
private theorem atomic_11949_agreement : RowAgreement 11948 11949 := RowAgreement.single (first := 11948) modelrow_agrees_11949
private def atomic_11950 : CountSegment 1 10 13120 11949 11950 4838 4839 := CountSegment.single (first := 11949) (initial := 4838) row_count_11950 (by decide)
private theorem atomic_11950_agreement : RowAgreement 11949 11950 := RowAgreement.single (first := 11949) modelrow_agrees_11950
private def atomic_11951 : CountSegment 1 10 13120 11950 11951 4839 4839 := CountSegment.single (first := 11950) (initial := 4839) row_count_11951 (by decide)
private theorem atomic_11951_agreement : RowAgreement 11950 11951 := RowAgreement.single (first := 11950) modelrow_agrees_11951
private def atomic_11952 : CountSegment 1 10 13120 11951 11952 4839 4839 := CountSegment.single (first := 11951) (initial := 4839) row_count_11952 (by decide)
private theorem atomic_11952_agreement : RowAgreement 11951 11952 := RowAgreement.single (first := 11951) modelrow_agrees_11952
private def atomic_11953 : CountSegment 1 10 13120 11952 11953 4839 4839 := CountSegment.single (first := 11952) (initial := 4839) row_count_11953 (by decide)
private theorem atomic_11953_agreement : RowAgreement 11952 11953 := RowAgreement.single (first := 11952) modelrow_agrees_11953
private def atomic_11954 : CountSegment 1 10 13120 11953 11954 4839 4839 := CountSegment.single (first := 11953) (initial := 4839) row_count_11954 (by decide)
private theorem atomic_11954_agreement : RowAgreement 11953 11954 := RowAgreement.single (first := 11953) modelrow_agrees_11954
private def atomic_11955 : CountSegment 1 10 13120 11954 11955 4839 4839 := CountSegment.single (first := 11954) (initial := 4839) row_count_11955 (by decide)
private theorem atomic_11955_agreement : RowAgreement 11954 11955 := RowAgreement.single (first := 11954) modelrow_agrees_11955
private def atomic_11956 : CountSegment 1 10 13120 11955 11956 4839 4839 := CountSegment.single (first := 11955) (initial := 4839) row_count_11956 (by decide)
private theorem atomic_11956_agreement : RowAgreement 11955 11956 := RowAgreement.single (first := 11955) modelrow_agrees_11956
private def atomic_11957 : CountSegment 1 10 13120 11956 11957 4839 4840 := CountSegment.single (first := 11956) (initial := 4839) row_count_11957 (by decide)
private theorem atomic_11957_agreement : RowAgreement 11956 11957 := RowAgreement.single (first := 11956) modelrow_agrees_11957
private def atomic_11958 : CountSegment 1 10 13120 11957 11958 4840 4841 := CountSegment.single (first := 11957) (initial := 4840) row_count_11958 (by decide)
private theorem atomic_11958_agreement : RowAgreement 11957 11958 := RowAgreement.single (first := 11957) modelrow_agrees_11958
private def atomic_11959 : CountSegment 1 10 13120 11958 11959 4841 4841 := CountSegment.single (first := 11958) (initial := 4841) row_count_11959 (by decide)
private theorem atomic_11959_agreement : RowAgreement 11958 11959 := RowAgreement.single (first := 11958) modelrow_agrees_11959
private def atomic_11960 : CountSegment 1 10 13120 11959 11960 4841 4841 := CountSegment.single (first := 11959) (initial := 4841) row_count_11960 (by decide)
private theorem atomic_11960_agreement : RowAgreement 11959 11960 := RowAgreement.single (first := 11959) modelrow_agrees_11960
private def atomic_11961 : CountSegment 1 10 13120 11960 11961 4841 4841 := CountSegment.single (first := 11960) (initial := 4841) row_count_11961 (by decide)
private theorem atomic_11961_agreement : RowAgreement 11960 11961 := RowAgreement.single (first := 11960) modelrow_agrees_11961
private def atomic_11962 : CountSegment 1 10 13120 11961 11962 4841 4841 := CountSegment.single (first := 11961) (initial := 4841) row_count_11962 (by decide)
private theorem atomic_11962_agreement : RowAgreement 11961 11962 := RowAgreement.single (first := 11961) modelrow_agrees_11962
private def atomic_11963 : CountSegment 1 10 13120 11962 11963 4841 4841 := CountSegment.single (first := 11962) (initial := 4841) row_count_11963 (by decide)
private theorem atomic_11963_agreement : RowAgreement 11962 11963 := RowAgreement.single (first := 11962) modelrow_agrees_11963
private def atomic_11964 : CountSegment 1 10 13120 11963 11964 4841 4841 := CountSegment.single (first := 11963) (initial := 4841) row_count_11964 (by decide)
private theorem atomic_11964_agreement : RowAgreement 11963 11964 := RowAgreement.single (first := 11963) modelrow_agrees_11964
private def atomic_11965 : CountSegment 1 10 13120 11964 11965 4841 4842 := CountSegment.single (first := 11964) (initial := 4841) row_count_11965 (by decide)
private theorem atomic_11965_agreement : RowAgreement 11964 11965 := RowAgreement.single (first := 11964) modelrow_agrees_11965
private def atomic_11966 : CountSegment 1 10 13120 11965 11966 4842 4842 := CountSegment.single (first := 11965) (initial := 4842) row_count_11966 (by decide)
private theorem atomic_11966_agreement : RowAgreement 11965 11966 := RowAgreement.single (first := 11965) modelrow_agrees_11966
private def atomic_11967 : CountSegment 1 10 13120 11966 11967 4842 4843 := CountSegment.single (first := 11966) (initial := 4842) row_count_11967 (by decide)
private theorem atomic_11967_agreement : RowAgreement 11966 11967 := RowAgreement.single (first := 11966) modelrow_agrees_11967
private def atomic_11968 : CountSegment 1 10 13120 11967 11968 4843 4844 := CountSegment.single (first := 11967) (initial := 4843) row_count_11968 (by decide)
private theorem atomic_11968_agreement : RowAgreement 11967 11968 := RowAgreement.single (first := 11967) modelrow_agrees_11968
private def atomic_11969 : CountSegment 1 10 13120 11968 11969 4844 4844 := CountSegment.single (first := 11968) (initial := 4844) row_count_11969 (by decide)
private theorem atomic_11969_agreement : RowAgreement 11968 11969 := RowAgreement.single (first := 11968) modelrow_agrees_11969
private def atomic_11970 : CountSegment 1 10 13120 11969 11970 4844 4845 := CountSegment.single (first := 11969) (initial := 4844) row_count_11970 (by decide)
private theorem atomic_11970_agreement : RowAgreement 11969 11970 := RowAgreement.single (first := 11969) modelrow_agrees_11970
private def atomic_11971 : CountSegment 1 10 13120 11970 11971 4845 4845 := CountSegment.single (first := 11970) (initial := 4845) row_count_11971 (by decide)
private theorem atomic_11971_agreement : RowAgreement 11970 11971 := RowAgreement.single (first := 11970) modelrow_agrees_11971
private def atomic_11972 : CountSegment 1 10 13120 11971 11972 4845 4846 := CountSegment.single (first := 11971) (initial := 4845) row_count_11972 (by decide)
private theorem atomic_11972_agreement : RowAgreement 11971 11972 := RowAgreement.single (first := 11971) modelrow_agrees_11972
private def atomic_11973 : CountSegment 1 10 13120 11972 11973 4846 4848 := CountSegment.single (first := 11972) (initial := 4846) row_count_11973 (by decide)
private theorem atomic_11973_agreement : RowAgreement 11972 11973 := RowAgreement.single (first := 11972) modelrow_agrees_11973
private def atomic_11974 : CountSegment 1 10 13120 11973 11974 4848 4848 := CountSegment.single (first := 11973) (initial := 4848) row_count_11974 (by decide)
private theorem atomic_11974_agreement : RowAgreement 11973 11974 := RowAgreement.single (first := 11973) modelrow_agrees_11974
private def atomic_11975 : CountSegment 1 10 13120 11974 11975 4848 4849 := CountSegment.single (first := 11974) (initial := 4848) row_count_11975 (by decide)
private theorem atomic_11975_agreement : RowAgreement 11974 11975 := RowAgreement.single (first := 11974) modelrow_agrees_11975
private def atomic_11976 : CountSegment 1 10 13120 11975 11976 4849 4849 := CountSegment.single (first := 11975) (initial := 4849) row_count_11976 (by decide)
private theorem atomic_11976_agreement : RowAgreement 11975 11976 := RowAgreement.single (first := 11975) modelrow_agrees_11976
private def atomic_11977 : CountSegment 1 10 13120 11976 11977 4849 4849 := CountSegment.single (first := 11976) (initial := 4849) row_count_11977 (by decide)
private theorem atomic_11977_agreement : RowAgreement 11976 11977 := RowAgreement.single (first := 11976) modelrow_agrees_11977
private def atomic_11978 : CountSegment 1 10 13120 11977 11978 4849 4849 := CountSegment.single (first := 11977) (initial := 4849) row_count_11978 (by decide)
private theorem atomic_11978_agreement : RowAgreement 11977 11978 := RowAgreement.single (first := 11977) modelrow_agrees_11978
private def atomic_11979 : CountSegment 1 10 13120 11978 11979 4849 4849 := CountSegment.single (first := 11978) (initial := 4849) row_count_11979 (by decide)
private theorem atomic_11979_agreement : RowAgreement 11978 11979 := RowAgreement.single (first := 11978) modelrow_agrees_11979
private def atomic_11980 : CountSegment 1 10 13120 11979 11980 4849 4850 := CountSegment.single (first := 11979) (initial := 4849) row_count_11980 (by decide)
private theorem atomic_11980_agreement : RowAgreement 11979 11980 := RowAgreement.single (first := 11979) modelrow_agrees_11980
private def atomic_11981 : CountSegment 1 10 13120 11980 11981 4850 4850 := CountSegment.single (first := 11980) (initial := 4850) row_count_11981 (by decide)
private theorem atomic_11981_agreement : RowAgreement 11980 11981 := RowAgreement.single (first := 11980) modelrow_agrees_11981
private def atomic_11982 : CountSegment 1 10 13120 11981 11982 4850 4850 := CountSegment.single (first := 11981) (initial := 4850) row_count_11982 (by decide)
private theorem atomic_11982_agreement : RowAgreement 11981 11982 := RowAgreement.single (first := 11981) modelrow_agrees_11982
private def atomic_11983 : CountSegment 1 10 13120 11982 11983 4850 4850 := CountSegment.single (first := 11982) (initial := 4850) row_count_11983 (by decide)
private theorem atomic_11983_agreement : RowAgreement 11982 11983 := RowAgreement.single (first := 11982) modelrow_agrees_11983
private def atomic_11984 : CountSegment 1 10 13120 11983 11984 4850 4851 := CountSegment.single (first := 11983) (initial := 4850) row_count_11984 (by decide)
private theorem atomic_11984_agreement : RowAgreement 11983 11984 := RowAgreement.single (first := 11983) modelrow_agrees_11984
private def atomic_11985 : CountSegment 1 10 13120 11984 11985 4851 4852 := CountSegment.single (first := 11984) (initial := 4851) row_count_11985 (by decide)
private theorem atomic_11985_agreement : RowAgreement 11984 11985 := RowAgreement.single (first := 11984) modelrow_agrees_11985
private def atomic_11986 : CountSegment 1 10 13120 11985 11986 4852 4852 := CountSegment.single (first := 11985) (initial := 4852) row_count_11986 (by decide)
private theorem atomic_11986_agreement : RowAgreement 11985 11986 := RowAgreement.single (first := 11985) modelrow_agrees_11986
private def atomic_11987 : CountSegment 1 10 13120 11986 11987 4852 4852 := CountSegment.single (first := 11986) (initial := 4852) row_count_11987 (by decide)
private theorem atomic_11987_agreement : RowAgreement 11986 11987 := RowAgreement.single (first := 11986) modelrow_agrees_11987
private def atomic_11988 : CountSegment 1 10 13120 11987 11988 4852 4852 := CountSegment.single (first := 11987) (initial := 4852) row_count_11988 (by decide)
private theorem atomic_11988_agreement : RowAgreement 11987 11988 := RowAgreement.single (first := 11987) modelrow_agrees_11988
private def atomic_11989 : CountSegment 1 10 13120 11988 11989 4852 4853 := CountSegment.single (first := 11988) (initial := 4852) row_count_11989 (by decide)
private theorem atomic_11989_agreement : RowAgreement 11988 11989 := RowAgreement.single (first := 11988) modelrow_agrees_11989
private def atomic_11990 : CountSegment 1 10 13120 11989 11990 4853 4853 := CountSegment.single (first := 11989) (initial := 4853) row_count_11990 (by decide)
private theorem atomic_11990_agreement : RowAgreement 11989 11990 := RowAgreement.single (first := 11989) modelrow_agrees_11990
private def atomic_11991 : CountSegment 1 10 13120 11990 11991 4853 4853 := CountSegment.single (first := 11990) (initial := 4853) row_count_11991 (by decide)
private theorem atomic_11991_agreement : RowAgreement 11990 11991 := RowAgreement.single (first := 11990) modelrow_agrees_11991
private def atomic_11992 : CountSegment 1 10 13120 11991 11992 4853 4853 := CountSegment.single (first := 11991) (initial := 4853) row_count_11992 (by decide)
private theorem atomic_11992_agreement : RowAgreement 11991 11992 := RowAgreement.single (first := 11991) modelrow_agrees_11992
private def atomic_11993 : CountSegment 1 10 13120 11992 11993 4853 4853 := CountSegment.single (first := 11992) (initial := 4853) row_count_11993 (by decide)
private theorem atomic_11993_agreement : RowAgreement 11992 11993 := RowAgreement.single (first := 11992) modelrow_agrees_11993
private def atomic_11994 : CountSegment 1 10 13120 11993 11994 4853 4853 := CountSegment.single (first := 11993) (initial := 4853) row_count_11994 (by decide)
private theorem atomic_11994_agreement : RowAgreement 11993 11994 := RowAgreement.single (first := 11993) modelrow_agrees_11994
private def atomic_11995 : CountSegment 1 10 13120 11994 11995 4853 4854 := CountSegment.single (first := 11994) (initial := 4853) row_count_11995 (by decide)
private theorem atomic_11995_agreement : RowAgreement 11994 11995 := RowAgreement.single (first := 11994) modelrow_agrees_11995
private def atomic_11996 : CountSegment 1 10 13120 11995 11996 4854 4854 := CountSegment.single (first := 11995) (initial := 4854) row_count_11996 (by decide)
private theorem atomic_11996_agreement : RowAgreement 11995 11996 := RowAgreement.single (first := 11995) modelrow_agrees_11996
private def atomic_11997 : CountSegment 1 10 13120 11996 11997 4854 4854 := CountSegment.single (first := 11996) (initial := 4854) row_count_11997 (by decide)
private theorem atomic_11997_agreement : RowAgreement 11996 11997 := RowAgreement.single (first := 11996) modelrow_agrees_11997
private def atomic_11998 : CountSegment 1 10 13120 11997 11998 4854 4855 := CountSegment.single (first := 11997) (initial := 4854) row_count_11998 (by decide)
private theorem atomic_11998_agreement : RowAgreement 11997 11998 := RowAgreement.single (first := 11997) modelrow_agrees_11998
private def atomic_11999 : CountSegment 1 10 13120 11998 11999 4855 4855 := CountSegment.single (first := 11998) (initial := 4855) row_count_11999 (by decide)
private theorem atomic_11999_agreement : RowAgreement 11998 11999 := RowAgreement.single (first := 11998) modelrow_agrees_11999
private def atomic_12000 : CountSegment 1 10 13120 11999 12000 4855 4855 := CountSegment.single (first := 11999) (initial := 4855) row_count_12000 (by decide)
private theorem atomic_12000_agreement : RowAgreement 11999 12000 := RowAgreement.single (first := 11999) modelrow_agrees_12000
private def atomic_12001 : CountSegment 1 10 13120 12000 12001 4855 4855 := CountSegment.single (first := 12000) (initial := 4855) row_count_12001 (by decide)
private theorem atomic_12001_agreement : RowAgreement 12000 12001 := RowAgreement.single (first := 12000) modelrow_agrees_12001
private def atomic_12002 : CountSegment 1 10 13120 12001 12002 4855 4856 := CountSegment.single (first := 12001) (initial := 4855) row_count_12002 (by decide)
private theorem atomic_12002_agreement : RowAgreement 12001 12002 := RowAgreement.single (first := 12001) modelrow_agrees_12002
private def atomic_12003 : CountSegment 1 10 13120 12002 12003 4856 4856 := CountSegment.single (first := 12002) (initial := 4856) row_count_12003 (by decide)
private theorem atomic_12003_agreement : RowAgreement 12002 12003 := RowAgreement.single (first := 12002) modelrow_agrees_12003
private def atomic_12004 : CountSegment 1 10 13120 12003 12004 4856 4856 := CountSegment.single (first := 12003) (initial := 4856) row_count_12004 (by decide)
private theorem atomic_12004_agreement : RowAgreement 12003 12004 := RowAgreement.single (first := 12003) modelrow_agrees_12004
private def atomic_12005 : CountSegment 1 10 13120 12004 12005 4856 4856 := CountSegment.single (first := 12004) (initial := 4856) row_count_12005 (by decide)
private theorem atomic_12005_agreement : RowAgreement 12004 12005 := RowAgreement.single (first := 12004) modelrow_agrees_12005
private def atomic_12006 : CountSegment 1 10 13120 12005 12006 4856 4857 := CountSegment.single (first := 12005) (initial := 4856) row_count_12006 (by decide)
private theorem atomic_12006_agreement : RowAgreement 12005 12006 := RowAgreement.single (first := 12005) modelrow_agrees_12006
private def atomic_12007 : CountSegment 1 10 13120 12006 12007 4857 4858 := CountSegment.single (first := 12006) (initial := 4857) row_count_12007 (by decide)
private theorem atomic_12007_agreement : RowAgreement 12006 12007 := RowAgreement.single (first := 12006) modelrow_agrees_12007
private def atomic_12008 : CountSegment 1 10 13120 12007 12008 4858 4858 := CountSegment.single (first := 12007) (initial := 4858) row_count_12008 (by decide)
private theorem atomic_12008_agreement : RowAgreement 12007 12008 := RowAgreement.single (first := 12007) modelrow_agrees_12008
private def atomic_12009 : CountSegment 1 10 13120 12008 12009 4858 4858 := CountSegment.single (first := 12008) (initial := 4858) row_count_12009 (by decide)
private theorem atomic_12009_agreement : RowAgreement 12008 12009 := RowAgreement.single (first := 12008) modelrow_agrees_12009
private def atomic_12010 : CountSegment 1 10 13120 12009 12010 4858 4858 := CountSegment.single (first := 12009) (initial := 4858) row_count_12010 (by decide)
private theorem atomic_12010_agreement : RowAgreement 12009 12010 := RowAgreement.single (first := 12009) modelrow_agrees_12010
private def atomic_12011 : CountSegment 1 10 13120 12010 12011 4858 4858 := CountSegment.single (first := 12010) (initial := 4858) row_count_12011 (by decide)
private theorem atomic_12011_agreement : RowAgreement 12010 12011 := RowAgreement.single (first := 12010) modelrow_agrees_12011
private def atomic_12012 : CountSegment 1 10 13120 12011 12012 4858 4858 := CountSegment.single (first := 12011) (initial := 4858) row_count_12012 (by decide)
private theorem atomic_12012_agreement : RowAgreement 12011 12012 := RowAgreement.single (first := 12011) modelrow_agrees_12012
private def atomic_12013 : CountSegment 1 10 13120 12012 12013 4858 4858 := CountSegment.single (first := 12012) (initial := 4858) row_count_12013 (by decide)
private theorem atomic_12013_agreement : RowAgreement 12012 12013 := RowAgreement.single (first := 12012) modelrow_agrees_12013
private def atomic_12014 : CountSegment 1 10 13120 12013 12014 4858 4858 := CountSegment.single (first := 12013) (initial := 4858) row_count_12014 (by decide)
private theorem atomic_12014_agreement : RowAgreement 12013 12014 := RowAgreement.single (first := 12013) modelrow_agrees_12014
private def atomic_12015 : CountSegment 1 10 13120 12014 12015 4858 4859 := CountSegment.single (first := 12014) (initial := 4858) row_count_12015 (by decide)
private theorem atomic_12015_agreement : RowAgreement 12014 12015 := RowAgreement.single (first := 12014) modelrow_agrees_12015
private def atomic_12016 : CountSegment 1 10 13120 12015 12016 4859 4859 := CountSegment.single (first := 12015) (initial := 4859) row_count_12016 (by decide)
private theorem atomic_12016_agreement : RowAgreement 12015 12016 := RowAgreement.single (first := 12015) modelrow_agrees_12016
private def atomic_12017 : CountSegment 1 10 13120 12016 12017 4859 4859 := CountSegment.single (first := 12016) (initial := 4859) row_count_12017 (by decide)
private theorem atomic_12017_agreement : RowAgreement 12016 12017 := RowAgreement.single (first := 12016) modelrow_agrees_12017
private def atomic_12018 : CountSegment 1 10 13120 12017 12018 4859 4859 := CountSegment.single (first := 12017) (initial := 4859) row_count_12018 (by decide)
private theorem atomic_12018_agreement : RowAgreement 12017 12018 := RowAgreement.single (first := 12017) modelrow_agrees_12018
private def atomic_12019 : CountSegment 1 10 13120 12018 12019 4859 4860 := CountSegment.single (first := 12018) (initial := 4859) row_count_12019 (by decide)
private theorem atomic_12019_agreement : RowAgreement 12018 12019 := RowAgreement.single (first := 12018) modelrow_agrees_12019
private def atomic_12020 : CountSegment 1 10 13120 12019 12020 4860 4860 := CountSegment.single (first := 12019) (initial := 4860) row_count_12020 (by decide)
private theorem atomic_12020_agreement : RowAgreement 12019 12020 := RowAgreement.single (first := 12019) modelrow_agrees_12020
private def atomic_12021 : CountSegment 1 10 13120 12020 12021 4860 4860 := CountSegment.single (first := 12020) (initial := 4860) row_count_12021 (by decide)
private theorem atomic_12021_agreement : RowAgreement 12020 12021 := RowAgreement.single (first := 12020) modelrow_agrees_12021
private def atomic_12022 : CountSegment 1 10 13120 12021 12022 4860 4860 := CountSegment.single (first := 12021) (initial := 4860) row_count_12022 (by decide)
private theorem atomic_12022_agreement : RowAgreement 12021 12022 := RowAgreement.single (first := 12021) modelrow_agrees_12022
private def atomic_12023 : CountSegment 1 10 13120 12022 12023 4860 4860 := CountSegment.single (first := 12022) (initial := 4860) row_count_12023 (by decide)
private theorem atomic_12023_agreement : RowAgreement 12022 12023 := RowAgreement.single (first := 12022) modelrow_agrees_12023
private def atomic_12024 : CountSegment 1 10 13120 12023 12024 4860 4861 := CountSegment.single (first := 12023) (initial := 4860) row_count_12024 (by decide)
private theorem atomic_12024_agreement : RowAgreement 12023 12024 := RowAgreement.single (first := 12023) modelrow_agrees_12024
private def atomic_12025 : CountSegment 1 10 13120 12024 12025 4861 4862 := CountSegment.single (first := 12024) (initial := 4861) row_count_12025 (by decide)
private theorem atomic_12025_agreement : RowAgreement 12024 12025 := RowAgreement.single (first := 12024) modelrow_agrees_12025
private def atomic_12026 : CountSegment 1 10 13120 12025 12026 4862 4863 := CountSegment.single (first := 12025) (initial := 4862) row_count_12026 (by decide)
private theorem atomic_12026_agreement : RowAgreement 12025 12026 := RowAgreement.single (first := 12025) modelrow_agrees_12026
private def atomic_12027 : CountSegment 1 10 13120 12026 12027 4863 4863 := CountSegment.single (first := 12026) (initial := 4863) row_count_12027 (by decide)
private theorem atomic_12027_agreement : RowAgreement 12026 12027 := RowAgreement.single (first := 12026) modelrow_agrees_12027
private def atomic_12028 : CountSegment 1 10 13120 12027 12028 4863 4863 := CountSegment.single (first := 12027) (initial := 4863) row_count_12028 (by decide)
private theorem atomic_12028_agreement : RowAgreement 12027 12028 := RowAgreement.single (first := 12027) modelrow_agrees_12028
private def atomic_12029 : CountSegment 1 10 13120 12028 12029 4863 4864 := CountSegment.single (first := 12028) (initial := 4863) row_count_12029 (by decide)
private theorem atomic_12029_agreement : RowAgreement 12028 12029 := RowAgreement.single (first := 12028) modelrow_agrees_12029
private def atomic_12030 : CountSegment 1 10 13120 12029 12030 4864 4865 := CountSegment.single (first := 12029) (initial := 4864) row_count_12030 (by decide)
private theorem atomic_12030_agreement : RowAgreement 12029 12030 := RowAgreement.single (first := 12029) modelrow_agrees_12030
private def atomic_12031 : CountSegment 1 10 13120 12030 12031 4865 4866 := CountSegment.single (first := 12030) (initial := 4865) row_count_12031 (by decide)
private theorem atomic_12031_agreement : RowAgreement 12030 12031 := RowAgreement.single (first := 12030) modelrow_agrees_12031
private def atomic_12032 : CountSegment 1 10 13120 12031 12032 4866 4866 := CountSegment.single (first := 12031) (initial := 4866) row_count_12032 (by decide)
private theorem atomic_12032_agreement : RowAgreement 12031 12032 := RowAgreement.single (first := 12031) modelrow_agrees_12032
private def within_093_000 : CountSegment 1 10 13120 11904 11906 4829 4829 := CountSegment.join atomic_11905 atomic_11906
private theorem within_093_000_agreement : RowAgreement 11904 11906 := RowAgreement.join atomic_11905_agreement atomic_11906_agreement
private def within_093_001 : CountSegment 1 10 13120 11906 11908 4829 4831 := CountSegment.join atomic_11907 atomic_11908
private theorem within_093_001_agreement : RowAgreement 11906 11908 := RowAgreement.join atomic_11907_agreement atomic_11908_agreement
private def within_093_002 : CountSegment 1 10 13120 11904 11908 4829 4831 := CountSegment.join within_093_000 within_093_001
private theorem within_093_002_agreement : RowAgreement 11904 11908 := RowAgreement.join within_093_000_agreement within_093_001_agreement
private def within_093_003 : CountSegment 1 10 13120 11908 11910 4831 4831 := CountSegment.join atomic_11909 atomic_11910
private theorem within_093_003_agreement : RowAgreement 11908 11910 := RowAgreement.join atomic_11909_agreement atomic_11910_agreement
private def within_093_004 : CountSegment 1 10 13120 11910 11912 4831 4831 := CountSegment.join atomic_11911 atomic_11912
private theorem within_093_004_agreement : RowAgreement 11910 11912 := RowAgreement.join atomic_11911_agreement atomic_11912_agreement
private def within_093_005 : CountSegment 1 10 13120 11908 11912 4831 4831 := CountSegment.join within_093_003 within_093_004
private theorem within_093_005_agreement : RowAgreement 11908 11912 := RowAgreement.join within_093_003_agreement within_093_004_agreement
private def within_093_006 : CountSegment 1 10 13120 11904 11912 4829 4831 := CountSegment.join within_093_002 within_093_005
private theorem within_093_006_agreement : RowAgreement 11904 11912 := RowAgreement.join within_093_002_agreement within_093_005_agreement
private def within_093_007 : CountSegment 1 10 13120 11912 11914 4831 4831 := CountSegment.join atomic_11913 atomic_11914
private theorem within_093_007_agreement : RowAgreement 11912 11914 := RowAgreement.join atomic_11913_agreement atomic_11914_agreement
private def within_093_008 : CountSegment 1 10 13120 11914 11916 4831 4831 := CountSegment.join atomic_11915 atomic_11916
private theorem within_093_008_agreement : RowAgreement 11914 11916 := RowAgreement.join atomic_11915_agreement atomic_11916_agreement
private def within_093_009 : CountSegment 1 10 13120 11912 11916 4831 4831 := CountSegment.join within_093_007 within_093_008
private theorem within_093_009_agreement : RowAgreement 11912 11916 := RowAgreement.join within_093_007_agreement within_093_008_agreement
private def within_093_010 : CountSegment 1 10 13120 11916 11918 4831 4831 := CountSegment.join atomic_11917 atomic_11918
private theorem within_093_010_agreement : RowAgreement 11916 11918 := RowAgreement.join atomic_11917_agreement atomic_11918_agreement
private def within_093_011 : CountSegment 1 10 13120 11918 11920 4831 4831 := CountSegment.join atomic_11919 atomic_11920
private theorem within_093_011_agreement : RowAgreement 11918 11920 := RowAgreement.join atomic_11919_agreement atomic_11920_agreement
private def within_093_012 : CountSegment 1 10 13120 11916 11920 4831 4831 := CountSegment.join within_093_010 within_093_011
private theorem within_093_012_agreement : RowAgreement 11916 11920 := RowAgreement.join within_093_010_agreement within_093_011_agreement
private def within_093_013 : CountSegment 1 10 13120 11912 11920 4831 4831 := CountSegment.join within_093_009 within_093_012
private theorem within_093_013_agreement : RowAgreement 11912 11920 := RowAgreement.join within_093_009_agreement within_093_012_agreement
private def within_093_014 : CountSegment 1 10 13120 11904 11920 4829 4831 := CountSegment.join within_093_006 within_093_013
private theorem within_093_014_agreement : RowAgreement 11904 11920 := RowAgreement.join within_093_006_agreement within_093_013_agreement
private def within_093_015 : CountSegment 1 10 13120 11920 11922 4831 4831 := CountSegment.join atomic_11921 atomic_11922
private theorem within_093_015_agreement : RowAgreement 11920 11922 := RowAgreement.join atomic_11921_agreement atomic_11922_agreement
private def within_093_016 : CountSegment 1 10 13120 11922 11924 4831 4831 := CountSegment.join atomic_11923 atomic_11924
private theorem within_093_016_agreement : RowAgreement 11922 11924 := RowAgreement.join atomic_11923_agreement atomic_11924_agreement
private def within_093_017 : CountSegment 1 10 13120 11920 11924 4831 4831 := CountSegment.join within_093_015 within_093_016
private theorem within_093_017_agreement : RowAgreement 11920 11924 := RowAgreement.join within_093_015_agreement within_093_016_agreement
private def within_093_018 : CountSegment 1 10 13120 11924 11926 4831 4832 := CountSegment.join atomic_11925 atomic_11926
private theorem within_093_018_agreement : RowAgreement 11924 11926 := RowAgreement.join atomic_11925_agreement atomic_11926_agreement
private def within_093_019 : CountSegment 1 10 13120 11926 11928 4832 4833 := CountSegment.join atomic_11927 atomic_11928
private theorem within_093_019_agreement : RowAgreement 11926 11928 := RowAgreement.join atomic_11927_agreement atomic_11928_agreement
private def within_093_020 : CountSegment 1 10 13120 11924 11928 4831 4833 := CountSegment.join within_093_018 within_093_019
private theorem within_093_020_agreement : RowAgreement 11924 11928 := RowAgreement.join within_093_018_agreement within_093_019_agreement
private def within_093_021 : CountSegment 1 10 13120 11920 11928 4831 4833 := CountSegment.join within_093_017 within_093_020
private theorem within_093_021_agreement : RowAgreement 11920 11928 := RowAgreement.join within_093_017_agreement within_093_020_agreement
private def within_093_022 : CountSegment 1 10 13120 11928 11930 4833 4833 := CountSegment.join atomic_11929 atomic_11930
private theorem within_093_022_agreement : RowAgreement 11928 11930 := RowAgreement.join atomic_11929_agreement atomic_11930_agreement
private def within_093_023 : CountSegment 1 10 13120 11930 11932 4833 4834 := CountSegment.join atomic_11931 atomic_11932
private theorem within_093_023_agreement : RowAgreement 11930 11932 := RowAgreement.join atomic_11931_agreement atomic_11932_agreement
private def within_093_024 : CountSegment 1 10 13120 11928 11932 4833 4834 := CountSegment.join within_093_022 within_093_023
private theorem within_093_024_agreement : RowAgreement 11928 11932 := RowAgreement.join within_093_022_agreement within_093_023_agreement
private def within_093_025 : CountSegment 1 10 13120 11932 11934 4834 4835 := CountSegment.join atomic_11933 atomic_11934
private theorem within_093_025_agreement : RowAgreement 11932 11934 := RowAgreement.join atomic_11933_agreement atomic_11934_agreement
private def within_093_026 : CountSegment 1 10 13120 11934 11936 4835 4835 := CountSegment.join atomic_11935 atomic_11936
private theorem within_093_026_agreement : RowAgreement 11934 11936 := RowAgreement.join atomic_11935_agreement atomic_11936_agreement
private def within_093_027 : CountSegment 1 10 13120 11932 11936 4834 4835 := CountSegment.join within_093_025 within_093_026
private theorem within_093_027_agreement : RowAgreement 11932 11936 := RowAgreement.join within_093_025_agreement within_093_026_agreement
private def within_093_028 : CountSegment 1 10 13120 11928 11936 4833 4835 := CountSegment.join within_093_024 within_093_027
private theorem within_093_028_agreement : RowAgreement 11928 11936 := RowAgreement.join within_093_024_agreement within_093_027_agreement
private def within_093_029 : CountSegment 1 10 13120 11920 11936 4831 4835 := CountSegment.join within_093_021 within_093_028
private theorem within_093_029_agreement : RowAgreement 11920 11936 := RowAgreement.join within_093_021_agreement within_093_028_agreement
private def within_093_030 : CountSegment 1 10 13120 11904 11936 4829 4835 := CountSegment.join within_093_014 within_093_029
private theorem within_093_030_agreement : RowAgreement 11904 11936 := RowAgreement.join within_093_014_agreement within_093_029_agreement
private def within_093_031 : CountSegment 1 10 13120 11936 11938 4835 4836 := CountSegment.join atomic_11937 atomic_11938
private theorem within_093_031_agreement : RowAgreement 11936 11938 := RowAgreement.join atomic_11937_agreement atomic_11938_agreement
private def within_093_032 : CountSegment 1 10 13120 11938 11940 4836 4836 := CountSegment.join atomic_11939 atomic_11940
private theorem within_093_032_agreement : RowAgreement 11938 11940 := RowAgreement.join atomic_11939_agreement atomic_11940_agreement
private def within_093_033 : CountSegment 1 10 13120 11936 11940 4835 4836 := CountSegment.join within_093_031 within_093_032
private theorem within_093_033_agreement : RowAgreement 11936 11940 := RowAgreement.join within_093_031_agreement within_093_032_agreement
private def within_093_034 : CountSegment 1 10 13120 11940 11942 4836 4836 := CountSegment.join atomic_11941 atomic_11942
private theorem within_093_034_agreement : RowAgreement 11940 11942 := RowAgreement.join atomic_11941_agreement atomic_11942_agreement
private def within_093_035 : CountSegment 1 10 13120 11942 11944 4836 4836 := CountSegment.join atomic_11943 atomic_11944
private theorem within_093_035_agreement : RowAgreement 11942 11944 := RowAgreement.join atomic_11943_agreement atomic_11944_agreement
private def within_093_036 : CountSegment 1 10 13120 11940 11944 4836 4836 := CountSegment.join within_093_034 within_093_035
private theorem within_093_036_agreement : RowAgreement 11940 11944 := RowAgreement.join within_093_034_agreement within_093_035_agreement
private def within_093_037 : CountSegment 1 10 13120 11936 11944 4835 4836 := CountSegment.join within_093_033 within_093_036
private theorem within_093_037_agreement : RowAgreement 11936 11944 := RowAgreement.join within_093_033_agreement within_093_036_agreement
private def within_093_038 : CountSegment 1 10 13120 11944 11946 4836 4838 := CountSegment.join atomic_11945 atomic_11946
private theorem within_093_038_agreement : RowAgreement 11944 11946 := RowAgreement.join atomic_11945_agreement atomic_11946_agreement
private def within_093_039 : CountSegment 1 10 13120 11946 11948 4838 4838 := CountSegment.join atomic_11947 atomic_11948
private theorem within_093_039_agreement : RowAgreement 11946 11948 := RowAgreement.join atomic_11947_agreement atomic_11948_agreement
private def within_093_040 : CountSegment 1 10 13120 11944 11948 4836 4838 := CountSegment.join within_093_038 within_093_039
private theorem within_093_040_agreement : RowAgreement 11944 11948 := RowAgreement.join within_093_038_agreement within_093_039_agreement
private def within_093_041 : CountSegment 1 10 13120 11948 11950 4838 4839 := CountSegment.join atomic_11949 atomic_11950
private theorem within_093_041_agreement : RowAgreement 11948 11950 := RowAgreement.join atomic_11949_agreement atomic_11950_agreement
private def within_093_042 : CountSegment 1 10 13120 11950 11952 4839 4839 := CountSegment.join atomic_11951 atomic_11952
private theorem within_093_042_agreement : RowAgreement 11950 11952 := RowAgreement.join atomic_11951_agreement atomic_11952_agreement
private def within_093_043 : CountSegment 1 10 13120 11948 11952 4838 4839 := CountSegment.join within_093_041 within_093_042
private theorem within_093_043_agreement : RowAgreement 11948 11952 := RowAgreement.join within_093_041_agreement within_093_042_agreement
private def within_093_044 : CountSegment 1 10 13120 11944 11952 4836 4839 := CountSegment.join within_093_040 within_093_043
private theorem within_093_044_agreement : RowAgreement 11944 11952 := RowAgreement.join within_093_040_agreement within_093_043_agreement
private def within_093_045 : CountSegment 1 10 13120 11936 11952 4835 4839 := CountSegment.join within_093_037 within_093_044
private theorem within_093_045_agreement : RowAgreement 11936 11952 := RowAgreement.join within_093_037_agreement within_093_044_agreement
private def within_093_046 : CountSegment 1 10 13120 11952 11954 4839 4839 := CountSegment.join atomic_11953 atomic_11954
private theorem within_093_046_agreement : RowAgreement 11952 11954 := RowAgreement.join atomic_11953_agreement atomic_11954_agreement
private def within_093_047 : CountSegment 1 10 13120 11954 11956 4839 4839 := CountSegment.join atomic_11955 atomic_11956
private theorem within_093_047_agreement : RowAgreement 11954 11956 := RowAgreement.join atomic_11955_agreement atomic_11956_agreement
private def within_093_048 : CountSegment 1 10 13120 11952 11956 4839 4839 := CountSegment.join within_093_046 within_093_047
private theorem within_093_048_agreement : RowAgreement 11952 11956 := RowAgreement.join within_093_046_agreement within_093_047_agreement
private def within_093_049 : CountSegment 1 10 13120 11956 11958 4839 4841 := CountSegment.join atomic_11957 atomic_11958
private theorem within_093_049_agreement : RowAgreement 11956 11958 := RowAgreement.join atomic_11957_agreement atomic_11958_agreement
private def within_093_050 : CountSegment 1 10 13120 11958 11960 4841 4841 := CountSegment.join atomic_11959 atomic_11960
private theorem within_093_050_agreement : RowAgreement 11958 11960 := RowAgreement.join atomic_11959_agreement atomic_11960_agreement
private def within_093_051 : CountSegment 1 10 13120 11956 11960 4839 4841 := CountSegment.join within_093_049 within_093_050
private theorem within_093_051_agreement : RowAgreement 11956 11960 := RowAgreement.join within_093_049_agreement within_093_050_agreement
private def within_093_052 : CountSegment 1 10 13120 11952 11960 4839 4841 := CountSegment.join within_093_048 within_093_051
private theorem within_093_052_agreement : RowAgreement 11952 11960 := RowAgreement.join within_093_048_agreement within_093_051_agreement
private def within_093_053 : CountSegment 1 10 13120 11960 11962 4841 4841 := CountSegment.join atomic_11961 atomic_11962
private theorem within_093_053_agreement : RowAgreement 11960 11962 := RowAgreement.join atomic_11961_agreement atomic_11962_agreement
private def within_093_054 : CountSegment 1 10 13120 11962 11964 4841 4841 := CountSegment.join atomic_11963 atomic_11964
private theorem within_093_054_agreement : RowAgreement 11962 11964 := RowAgreement.join atomic_11963_agreement atomic_11964_agreement
private def within_093_055 : CountSegment 1 10 13120 11960 11964 4841 4841 := CountSegment.join within_093_053 within_093_054
private theorem within_093_055_agreement : RowAgreement 11960 11964 := RowAgreement.join within_093_053_agreement within_093_054_agreement
private def within_093_056 : CountSegment 1 10 13120 11964 11966 4841 4842 := CountSegment.join atomic_11965 atomic_11966
private theorem within_093_056_agreement : RowAgreement 11964 11966 := RowAgreement.join atomic_11965_agreement atomic_11966_agreement
private def within_093_057 : CountSegment 1 10 13120 11966 11968 4842 4844 := CountSegment.join atomic_11967 atomic_11968
private theorem within_093_057_agreement : RowAgreement 11966 11968 := RowAgreement.join atomic_11967_agreement atomic_11968_agreement
private def within_093_058 : CountSegment 1 10 13120 11964 11968 4841 4844 := CountSegment.join within_093_056 within_093_057
private theorem within_093_058_agreement : RowAgreement 11964 11968 := RowAgreement.join within_093_056_agreement within_093_057_agreement
private def within_093_059 : CountSegment 1 10 13120 11960 11968 4841 4844 := CountSegment.join within_093_055 within_093_058
private theorem within_093_059_agreement : RowAgreement 11960 11968 := RowAgreement.join within_093_055_agreement within_093_058_agreement
private def within_093_060 : CountSegment 1 10 13120 11952 11968 4839 4844 := CountSegment.join within_093_052 within_093_059
private theorem within_093_060_agreement : RowAgreement 11952 11968 := RowAgreement.join within_093_052_agreement within_093_059_agreement
private def within_093_061 : CountSegment 1 10 13120 11936 11968 4835 4844 := CountSegment.join within_093_045 within_093_060
private theorem within_093_061_agreement : RowAgreement 11936 11968 := RowAgreement.join within_093_045_agreement within_093_060_agreement
private def within_093_062 : CountSegment 1 10 13120 11904 11968 4829 4844 := CountSegment.join within_093_030 within_093_061
private theorem within_093_062_agreement : RowAgreement 11904 11968 := RowAgreement.join within_093_030_agreement within_093_061_agreement
private def within_093_063 : CountSegment 1 10 13120 11968 11970 4844 4845 := CountSegment.join atomic_11969 atomic_11970
private theorem within_093_063_agreement : RowAgreement 11968 11970 := RowAgreement.join atomic_11969_agreement atomic_11970_agreement
private def within_093_064 : CountSegment 1 10 13120 11970 11972 4845 4846 := CountSegment.join atomic_11971 atomic_11972
private theorem within_093_064_agreement : RowAgreement 11970 11972 := RowAgreement.join atomic_11971_agreement atomic_11972_agreement
private def within_093_065 : CountSegment 1 10 13120 11968 11972 4844 4846 := CountSegment.join within_093_063 within_093_064
private theorem within_093_065_agreement : RowAgreement 11968 11972 := RowAgreement.join within_093_063_agreement within_093_064_agreement
private def within_093_066 : CountSegment 1 10 13120 11972 11974 4846 4848 := CountSegment.join atomic_11973 atomic_11974
private theorem within_093_066_agreement : RowAgreement 11972 11974 := RowAgreement.join atomic_11973_agreement atomic_11974_agreement
private def within_093_067 : CountSegment 1 10 13120 11974 11976 4848 4849 := CountSegment.join atomic_11975 atomic_11976
private theorem within_093_067_agreement : RowAgreement 11974 11976 := RowAgreement.join atomic_11975_agreement atomic_11976_agreement
private def within_093_068 : CountSegment 1 10 13120 11972 11976 4846 4849 := CountSegment.join within_093_066 within_093_067
private theorem within_093_068_agreement : RowAgreement 11972 11976 := RowAgreement.join within_093_066_agreement within_093_067_agreement
private def within_093_069 : CountSegment 1 10 13120 11968 11976 4844 4849 := CountSegment.join within_093_065 within_093_068
private theorem within_093_069_agreement : RowAgreement 11968 11976 := RowAgreement.join within_093_065_agreement within_093_068_agreement
private def within_093_070 : CountSegment 1 10 13120 11976 11978 4849 4849 := CountSegment.join atomic_11977 atomic_11978
private theorem within_093_070_agreement : RowAgreement 11976 11978 := RowAgreement.join atomic_11977_agreement atomic_11978_agreement
private def within_093_071 : CountSegment 1 10 13120 11978 11980 4849 4850 := CountSegment.join atomic_11979 atomic_11980
private theorem within_093_071_agreement : RowAgreement 11978 11980 := RowAgreement.join atomic_11979_agreement atomic_11980_agreement
private def within_093_072 : CountSegment 1 10 13120 11976 11980 4849 4850 := CountSegment.join within_093_070 within_093_071
private theorem within_093_072_agreement : RowAgreement 11976 11980 := RowAgreement.join within_093_070_agreement within_093_071_agreement
private def within_093_073 : CountSegment 1 10 13120 11980 11982 4850 4850 := CountSegment.join atomic_11981 atomic_11982
private theorem within_093_073_agreement : RowAgreement 11980 11982 := RowAgreement.join atomic_11981_agreement atomic_11982_agreement
private def within_093_074 : CountSegment 1 10 13120 11982 11984 4850 4851 := CountSegment.join atomic_11983 atomic_11984
private theorem within_093_074_agreement : RowAgreement 11982 11984 := RowAgreement.join atomic_11983_agreement atomic_11984_agreement
private def within_093_075 : CountSegment 1 10 13120 11980 11984 4850 4851 := CountSegment.join within_093_073 within_093_074
private theorem within_093_075_agreement : RowAgreement 11980 11984 := RowAgreement.join within_093_073_agreement within_093_074_agreement
private def within_093_076 : CountSegment 1 10 13120 11976 11984 4849 4851 := CountSegment.join within_093_072 within_093_075
private theorem within_093_076_agreement : RowAgreement 11976 11984 := RowAgreement.join within_093_072_agreement within_093_075_agreement
private def within_093_077 : CountSegment 1 10 13120 11968 11984 4844 4851 := CountSegment.join within_093_069 within_093_076
private theorem within_093_077_agreement : RowAgreement 11968 11984 := RowAgreement.join within_093_069_agreement within_093_076_agreement
private def within_093_078 : CountSegment 1 10 13120 11984 11986 4851 4852 := CountSegment.join atomic_11985 atomic_11986
private theorem within_093_078_agreement : RowAgreement 11984 11986 := RowAgreement.join atomic_11985_agreement atomic_11986_agreement
private def within_093_079 : CountSegment 1 10 13120 11986 11988 4852 4852 := CountSegment.join atomic_11987 atomic_11988
private theorem within_093_079_agreement : RowAgreement 11986 11988 := RowAgreement.join atomic_11987_agreement atomic_11988_agreement
private def within_093_080 : CountSegment 1 10 13120 11984 11988 4851 4852 := CountSegment.join within_093_078 within_093_079
private theorem within_093_080_agreement : RowAgreement 11984 11988 := RowAgreement.join within_093_078_agreement within_093_079_agreement
private def within_093_081 : CountSegment 1 10 13120 11988 11990 4852 4853 := CountSegment.join atomic_11989 atomic_11990
private theorem within_093_081_agreement : RowAgreement 11988 11990 := RowAgreement.join atomic_11989_agreement atomic_11990_agreement
private def within_093_082 : CountSegment 1 10 13120 11990 11992 4853 4853 := CountSegment.join atomic_11991 atomic_11992
private theorem within_093_082_agreement : RowAgreement 11990 11992 := RowAgreement.join atomic_11991_agreement atomic_11992_agreement
private def within_093_083 : CountSegment 1 10 13120 11988 11992 4852 4853 := CountSegment.join within_093_081 within_093_082
private theorem within_093_083_agreement : RowAgreement 11988 11992 := RowAgreement.join within_093_081_agreement within_093_082_agreement
private def within_093_084 : CountSegment 1 10 13120 11984 11992 4851 4853 := CountSegment.join within_093_080 within_093_083
private theorem within_093_084_agreement : RowAgreement 11984 11992 := RowAgreement.join within_093_080_agreement within_093_083_agreement
private def within_093_085 : CountSegment 1 10 13120 11992 11994 4853 4853 := CountSegment.join atomic_11993 atomic_11994
private theorem within_093_085_agreement : RowAgreement 11992 11994 := RowAgreement.join atomic_11993_agreement atomic_11994_agreement
private def within_093_086 : CountSegment 1 10 13120 11994 11996 4853 4854 := CountSegment.join atomic_11995 atomic_11996
private theorem within_093_086_agreement : RowAgreement 11994 11996 := RowAgreement.join atomic_11995_agreement atomic_11996_agreement
private def within_093_087 : CountSegment 1 10 13120 11992 11996 4853 4854 := CountSegment.join within_093_085 within_093_086
private theorem within_093_087_agreement : RowAgreement 11992 11996 := RowAgreement.join within_093_085_agreement within_093_086_agreement
private def within_093_088 : CountSegment 1 10 13120 11996 11998 4854 4855 := CountSegment.join atomic_11997 atomic_11998
private theorem within_093_088_agreement : RowAgreement 11996 11998 := RowAgreement.join atomic_11997_agreement atomic_11998_agreement
private def within_093_089 : CountSegment 1 10 13120 11998 12000 4855 4855 := CountSegment.join atomic_11999 atomic_12000
private theorem within_093_089_agreement : RowAgreement 11998 12000 := RowAgreement.join atomic_11999_agreement atomic_12000_agreement
private def within_093_090 : CountSegment 1 10 13120 11996 12000 4854 4855 := CountSegment.join within_093_088 within_093_089
private theorem within_093_090_agreement : RowAgreement 11996 12000 := RowAgreement.join within_093_088_agreement within_093_089_agreement
private def within_093_091 : CountSegment 1 10 13120 11992 12000 4853 4855 := CountSegment.join within_093_087 within_093_090
private theorem within_093_091_agreement : RowAgreement 11992 12000 := RowAgreement.join within_093_087_agreement within_093_090_agreement
private def within_093_092 : CountSegment 1 10 13120 11984 12000 4851 4855 := CountSegment.join within_093_084 within_093_091
private theorem within_093_092_agreement : RowAgreement 11984 12000 := RowAgreement.join within_093_084_agreement within_093_091_agreement
private def within_093_093 : CountSegment 1 10 13120 11968 12000 4844 4855 := CountSegment.join within_093_077 within_093_092
private theorem within_093_093_agreement : RowAgreement 11968 12000 := RowAgreement.join within_093_077_agreement within_093_092_agreement
private def within_093_094 : CountSegment 1 10 13120 12000 12002 4855 4856 := CountSegment.join atomic_12001 atomic_12002
private theorem within_093_094_agreement : RowAgreement 12000 12002 := RowAgreement.join atomic_12001_agreement atomic_12002_agreement
private def within_093_095 : CountSegment 1 10 13120 12002 12004 4856 4856 := CountSegment.join atomic_12003 atomic_12004
private theorem within_093_095_agreement : RowAgreement 12002 12004 := RowAgreement.join atomic_12003_agreement atomic_12004_agreement
private def within_093_096 : CountSegment 1 10 13120 12000 12004 4855 4856 := CountSegment.join within_093_094 within_093_095
private theorem within_093_096_agreement : RowAgreement 12000 12004 := RowAgreement.join within_093_094_agreement within_093_095_agreement
private def within_093_097 : CountSegment 1 10 13120 12004 12006 4856 4857 := CountSegment.join atomic_12005 atomic_12006
private theorem within_093_097_agreement : RowAgreement 12004 12006 := RowAgreement.join atomic_12005_agreement atomic_12006_agreement
private def within_093_098 : CountSegment 1 10 13120 12006 12008 4857 4858 := CountSegment.join atomic_12007 atomic_12008
private theorem within_093_098_agreement : RowAgreement 12006 12008 := RowAgreement.join atomic_12007_agreement atomic_12008_agreement
private def within_093_099 : CountSegment 1 10 13120 12004 12008 4856 4858 := CountSegment.join within_093_097 within_093_098
private theorem within_093_099_agreement : RowAgreement 12004 12008 := RowAgreement.join within_093_097_agreement within_093_098_agreement
private def within_093_100 : CountSegment 1 10 13120 12000 12008 4855 4858 := CountSegment.join within_093_096 within_093_099
private theorem within_093_100_agreement : RowAgreement 12000 12008 := RowAgreement.join within_093_096_agreement within_093_099_agreement
private def within_093_101 : CountSegment 1 10 13120 12008 12010 4858 4858 := CountSegment.join atomic_12009 atomic_12010
private theorem within_093_101_agreement : RowAgreement 12008 12010 := RowAgreement.join atomic_12009_agreement atomic_12010_agreement
private def within_093_102 : CountSegment 1 10 13120 12010 12012 4858 4858 := CountSegment.join atomic_12011 atomic_12012
private theorem within_093_102_agreement : RowAgreement 12010 12012 := RowAgreement.join atomic_12011_agreement atomic_12012_agreement
private def within_093_103 : CountSegment 1 10 13120 12008 12012 4858 4858 := CountSegment.join within_093_101 within_093_102
private theorem within_093_103_agreement : RowAgreement 12008 12012 := RowAgreement.join within_093_101_agreement within_093_102_agreement
private def within_093_104 : CountSegment 1 10 13120 12012 12014 4858 4858 := CountSegment.join atomic_12013 atomic_12014
private theorem within_093_104_agreement : RowAgreement 12012 12014 := RowAgreement.join atomic_12013_agreement atomic_12014_agreement
private def within_093_105 : CountSegment 1 10 13120 12014 12016 4858 4859 := CountSegment.join atomic_12015 atomic_12016
private theorem within_093_105_agreement : RowAgreement 12014 12016 := RowAgreement.join atomic_12015_agreement atomic_12016_agreement
private def within_093_106 : CountSegment 1 10 13120 12012 12016 4858 4859 := CountSegment.join within_093_104 within_093_105
private theorem within_093_106_agreement : RowAgreement 12012 12016 := RowAgreement.join within_093_104_agreement within_093_105_agreement
private def within_093_107 : CountSegment 1 10 13120 12008 12016 4858 4859 := CountSegment.join within_093_103 within_093_106
private theorem within_093_107_agreement : RowAgreement 12008 12016 := RowAgreement.join within_093_103_agreement within_093_106_agreement
private def within_093_108 : CountSegment 1 10 13120 12000 12016 4855 4859 := CountSegment.join within_093_100 within_093_107
private theorem within_093_108_agreement : RowAgreement 12000 12016 := RowAgreement.join within_093_100_agreement within_093_107_agreement
private def within_093_109 : CountSegment 1 10 13120 12016 12018 4859 4859 := CountSegment.join atomic_12017 atomic_12018
private theorem within_093_109_agreement : RowAgreement 12016 12018 := RowAgreement.join atomic_12017_agreement atomic_12018_agreement
private def within_093_110 : CountSegment 1 10 13120 12018 12020 4859 4860 := CountSegment.join atomic_12019 atomic_12020
private theorem within_093_110_agreement : RowAgreement 12018 12020 := RowAgreement.join atomic_12019_agreement atomic_12020_agreement
private def within_093_111 : CountSegment 1 10 13120 12016 12020 4859 4860 := CountSegment.join within_093_109 within_093_110
private theorem within_093_111_agreement : RowAgreement 12016 12020 := RowAgreement.join within_093_109_agreement within_093_110_agreement
private def within_093_112 : CountSegment 1 10 13120 12020 12022 4860 4860 := CountSegment.join atomic_12021 atomic_12022
private theorem within_093_112_agreement : RowAgreement 12020 12022 := RowAgreement.join atomic_12021_agreement atomic_12022_agreement
private def within_093_113 : CountSegment 1 10 13120 12022 12024 4860 4861 := CountSegment.join atomic_12023 atomic_12024
private theorem within_093_113_agreement : RowAgreement 12022 12024 := RowAgreement.join atomic_12023_agreement atomic_12024_agreement
private def within_093_114 : CountSegment 1 10 13120 12020 12024 4860 4861 := CountSegment.join within_093_112 within_093_113
private theorem within_093_114_agreement : RowAgreement 12020 12024 := RowAgreement.join within_093_112_agreement within_093_113_agreement
private def within_093_115 : CountSegment 1 10 13120 12016 12024 4859 4861 := CountSegment.join within_093_111 within_093_114
private theorem within_093_115_agreement : RowAgreement 12016 12024 := RowAgreement.join within_093_111_agreement within_093_114_agreement
private def within_093_116 : CountSegment 1 10 13120 12024 12026 4861 4863 := CountSegment.join atomic_12025 atomic_12026
private theorem within_093_116_agreement : RowAgreement 12024 12026 := RowAgreement.join atomic_12025_agreement atomic_12026_agreement
private def within_093_117 : CountSegment 1 10 13120 12026 12028 4863 4863 := CountSegment.join atomic_12027 atomic_12028
private theorem within_093_117_agreement : RowAgreement 12026 12028 := RowAgreement.join atomic_12027_agreement atomic_12028_agreement
private def within_093_118 : CountSegment 1 10 13120 12024 12028 4861 4863 := CountSegment.join within_093_116 within_093_117
private theorem within_093_118_agreement : RowAgreement 12024 12028 := RowAgreement.join within_093_116_agreement within_093_117_agreement
private def within_093_119 : CountSegment 1 10 13120 12028 12030 4863 4865 := CountSegment.join atomic_12029 atomic_12030
private theorem within_093_119_agreement : RowAgreement 12028 12030 := RowAgreement.join atomic_12029_agreement atomic_12030_agreement
private def within_093_120 : CountSegment 1 10 13120 12030 12032 4865 4866 := CountSegment.join atomic_12031 atomic_12032
private theorem within_093_120_agreement : RowAgreement 12030 12032 := RowAgreement.join atomic_12031_agreement atomic_12032_agreement
private def within_093_121 : CountSegment 1 10 13120 12028 12032 4863 4866 := CountSegment.join within_093_119 within_093_120
private theorem within_093_121_agreement : RowAgreement 12028 12032 := RowAgreement.join within_093_119_agreement within_093_120_agreement
private def within_093_122 : CountSegment 1 10 13120 12024 12032 4861 4866 := CountSegment.join within_093_118 within_093_121
private theorem within_093_122_agreement : RowAgreement 12024 12032 := RowAgreement.join within_093_118_agreement within_093_121_agreement
private def within_093_123 : CountSegment 1 10 13120 12016 12032 4859 4866 := CountSegment.join within_093_115 within_093_122
private theorem within_093_123_agreement : RowAgreement 12016 12032 := RowAgreement.join within_093_115_agreement within_093_122_agreement
private def within_093_124 : CountSegment 1 10 13120 12000 12032 4855 4866 := CountSegment.join within_093_108 within_093_123
private theorem within_093_124_agreement : RowAgreement 12000 12032 := RowAgreement.join within_093_108_agreement within_093_123_agreement
private def within_093_125 : CountSegment 1 10 13120 11968 12032 4844 4866 := CountSegment.join within_093_093 within_093_124
private theorem within_093_125_agreement : RowAgreement 11968 12032 := RowAgreement.join within_093_093_agreement within_093_124_agreement
private def within_093_126 : CountSegment 1 10 13120 11904 12032 4829 4866 := CountSegment.join within_093_062 within_093_125
private theorem within_093_126_agreement : RowAgreement 11904 12032 := RowAgreement.join within_093_062_agreement within_093_125_agreement
def segment_093 : CountSegment 1 10 13120 11904 12032 4829 4866 := within_093_126
theorem agreement_093 : RowAgreement 11904 12032 := within_093_126_agreement
end ProjectEulerSolutions.P58Numerical
