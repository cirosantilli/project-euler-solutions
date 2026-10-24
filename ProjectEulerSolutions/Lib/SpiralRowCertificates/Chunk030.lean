import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3841 : RowCertificate 3841 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3841 1) := by change ¬ Nat.Prime 59020807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1903897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3841 2) := by change ¬ Nat.Prime 59013125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11802625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3841 3) := by change ¬ Nat.Prime 59005443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19668481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3841 : row 3841 = 0 := (row_3841.checked).trans rfl
theorem model_count_3841 : modelRow 3841 = 0 := by decide
theorem modelrow_agrees_3841 : modelRow 3841 = row 3841 := model_count_3841.trans row_count_3841.symm
def row_3842 : RowCertificate 3842 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3842 1) := by change ¬ Nat.Prime 59051541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19683847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3842 2) := by change ¬ Nat.Prime 59043857; exact PrattCertificates.composite_of_product (factor := 821) (cofactor := 71917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3842 3) := by change ¬ Nat.Prime 59036173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8433739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3842 : row 3842 = 0 := (row_3842.checked).trans rfl
theorem model_count_3842 : modelRow 3842 = 0 := by decide
theorem modelrow_agrees_3842 : modelRow 3842 = row 3842 := model_count_3842.trans row_count_3842.symm
def row_3843 : RowCertificate 3843 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3843 1) := by change ¬ Nat.Prime 59082283; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4544791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3843 2) := by change ¬ Nat.Prime 59074597; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 584897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3843 3) := by change ¬ Nat.Prime 59066911; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1596403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3843 : row 3843 = 0 := (row_3843.checked).trans rfl
theorem model_count_3843 : modelRow 3843 = 0 := by decide
theorem modelrow_agrees_3843 : modelRow 3843 = row 3843 := model_count_3843.trans row_count_3843.symm
def row_3844 : RowCertificate 3844 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3844 1) := by change ¬ Nat.Prime 59113033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8444719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3844 2) := by change ¬ Nat.Prime 59105345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11821069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3844 3) := by change ¬ Nat.Prime 59097657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19699219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3844 : row 3844 = 0 := (row_3844.checked).trans rfl
theorem model_count_3844 : modelRow 3844 = 0 := by decide
theorem modelrow_agrees_3844 : modelRow 3844 = row 3844 := model_count_3844.trans row_count_3844.symm
def row_3845 : RowCertificate 3845 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3845 1) := by change ¬ Nat.Prime 59143791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19714597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3845 2) := by change ¬ Nat.Prime 59136101; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1598273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3845 3) := by change ¬ Nat.Prime 59128411; exact PrattCertificates.composite_of_product (factor := 3187) (cofactor := 18553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3845 : row 3845 = 0 := (row_3845.checked).trans rfl
theorem model_count_3845 : modelRow 3845 = 0 := by decide
theorem modelrow_agrees_3845 : modelRow 3845 = row 3845 := model_count_3845.trans row_count_3845.symm
def row_3846 : RowCertificate 3846 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3846 1) := by change ¬ Nat.Prime 59174557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4551889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3846 2) := by change ¬ Nat.Prime 59166865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11833373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3846 3) := by change ¬ Nat.Prime 59159173; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 258337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3846 : row 3846 = 0 := (row_3846.checked).trans rfl
theorem model_count_3846 : modelRow 3846 = 0 := by decide
theorem modelrow_agrees_3846 : modelRow 3846 = row 3846 := model_count_3846.trans row_count_3846.symm
def row_3847 : RowCertificate 3847 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3847 1) := by change Nat.Prime 59205331; exact ProjectEulerSolutions.P58PrimeNodes.p_59205331.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3847 2) := by change ¬ Nat.Prime 59197637; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 230341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3847 3) := by change ¬ Nat.Prime 59189943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19729981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3847 : row 3847 = 1 := (row_3847.checked).trans rfl
theorem model_count_3847 : modelRow 3847 = 1 := by decide
theorem modelrow_agrees_3847 : modelRow 3847 = row 3847 := model_count_3847.trans row_count_3847.symm
def row_3848 : RowCertificate 3848 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3848 1) := by change ¬ Nat.Prime 59236113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19745371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3848 2) := by change ¬ Nat.Prime 59228417; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 213821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3848 3) := by change ¬ Nat.Prime 59220721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8460103) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3848 : row 3848 = 0 := (row_3848.checked).trans rfl
theorem model_count_3848 : modelRow 3848 = 0 := by decide
theorem modelrow_agrees_3848 : modelRow 3848 = row 3848 := model_count_3848.trans row_count_3848.symm
def row_3849 : RowCertificate 3849 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3849 1) := by change ¬ Nat.Prime 59266903; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 610999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3849 2) := by change ¬ Nat.Prime 59259205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11851841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3849 3) := by change ¬ Nat.Prime 59251507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8464501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3849 : row 3849 = 0 := (row_3849.checked).trans rfl
theorem model_count_3849 : modelRow 3849 = 0 := by decide
theorem modelrow_agrees_3849 : modelRow 3849 = row 3849 := model_count_3849.trans row_count_3849.symm
def row_3850 : RowCertificate 3850 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3850 1) := by change Nat.Prime 59297701; exact ProjectEulerSolutions.P58PrimeNodes.p_59297701.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3850 2) := by change ¬ Nat.Prime 59290001; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 132049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3850 3) := by change ¬ Nat.Prime 59282301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19760767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3850 : row 3850 = 1 := (row_3850.checked).trans rfl
theorem model_count_3850 : modelRow 3850 = 1 := by decide
theorem modelrow_agrees_3850 : modelRow 3850 = row 3850 := model_count_3850.trans row_count_3850.symm
def row_3851 : RowCertificate 3851 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3851 1) := by change ¬ Nat.Prime 59328507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19776169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3851 2) := by change ¬ Nat.Prime 59320805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11864161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3851 3) := by change Nat.Prime 59313103; exact ProjectEulerSolutions.P58PrimeNodes.p_59313103.prime
    exact decide_eq_true prime
theorem row_count_3851 : row 3851 = 1 := (row_3851.checked).trans rfl
theorem model_count_3851 : modelRow 3851 = 1 := by decide
theorem modelrow_agrees_3851 : modelRow 3851 = row 3851 := model_count_3851.trans row_count_3851.symm
def row_3852 : RowCertificate 3852 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3852 1) := by change ¬ Nat.Prime 59359321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8479903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3852 2) := by change ¬ Nat.Prime 59351617; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4565509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3852 3) := by change ¬ Nat.Prime 59343913; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1380091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3852 : row 3852 = 0 := (row_3852.checked).trans rfl
theorem model_count_3852 : modelRow 3852 = 0 := by decide
theorem modelrow_agrees_3852 : modelRow 3852 = row 3852 := model_count_3852.trans row_count_3852.symm
def row_3853 : RowCertificate 3853 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3853 1) := by change ¬ Nat.Prime 59390143; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3125797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3853 2) := by change ¬ Nat.Prime 59382437; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 544793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3853 3) := by change ¬ Nat.Prime 59374731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19791577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3853 : row 3853 = 0 := (row_3853.checked).trans rfl
theorem model_count_3853 : modelRow 3853 = 0 := by decide
theorem modelrow_agrees_3853 : modelRow 3853 = row 3853 := model_count_3853.trans row_count_3853.symm
def row_3854 : RowCertificate 3854 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3854 1) := by change ¬ Nat.Prime 59420973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19806991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3854 2) := by change ¬ Nat.Prime 59413265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11882653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3854 3) := by change Nat.Prime 59405557; exact ProjectEulerSolutions.P58PrimeNodes.p_59405557.prime
    exact decide_eq_true prime
theorem row_count_3854 : row 3854 = 1 := (row_3854.checked).trans rfl
theorem model_count_3854 : modelRow 3854 = 1 := by decide
theorem modelrow_agrees_3854 : modelRow 3854 = row 3854 := model_count_3854.trans row_count_3854.symm
def row_3855 : RowCertificate 3855 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3855 1) := by change ¬ Nat.Prime 59451811; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 80449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3855 2) := by change ¬ Nat.Prime 59444101; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 81097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3855 3) := by change ¬ Nat.Prime 59436391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8490913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3855 : row 3855 = 0 := (row_3855.checked).trans rfl
theorem model_count_3855 : modelRow 3855 = 0 := by decide
theorem modelrow_agrees_3855 : modelRow 3855 = row 3855 := model_count_3855.trans row_count_3855.symm
def row_3856 : RowCertificate 3856 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3856 1) := by change ¬ Nat.Prime 59482657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4575589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3856 2) := by change ¬ Nat.Prime 59474945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11894989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3856 3) := by change ¬ Nat.Prime 59467233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19822411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3856 : row 3856 = 0 := (row_3856.checked).trans rfl
theorem model_count_3856 : modelRow 3856 = 0 := by decide
theorem modelrow_agrees_3856 : modelRow 3856 = row 3856 := model_count_3856.trans row_count_3856.symm
def row_3857 : RowCertificate 3857 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3857 1) := by change ¬ Nat.Prime 59513511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19837837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3857 2) := by change ¬ Nat.Prime 59505797; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4577369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3857 3) := by change ¬ Nat.Prime 59498083; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1919293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3857 : row 3857 = 0 := (row_3857.checked).trans rfl
theorem model_count_3857 : modelRow 3857 = 0 := by decide
theorem modelrow_agrees_3857 : modelRow 3857 = row 3857 := model_count_3857.trans row_count_3857.symm
def row_3858 : RowCertificate 3858 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3858 1) := by change ¬ Nat.Prime 59544373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8506339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3858 2) := by change Nat.Prime 59536657; exact ProjectEulerSolutions.P58PrimeNodes.p_59536657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3858 3) := by change ¬ Nat.Prime 59528941; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 81883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3858 : row 3858 = 1 := (row_3858.checked).trans rfl
theorem model_count_3858 : modelRow 3858 = 1 := by decide
theorem modelrow_agrees_3858 : modelRow 3858 = row 3858 := model_count_3858.trans row_count_3858.symm
def row_3859 : RowCertificate 3859 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3859 1) := by change ¬ Nat.Prime 59575243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8510749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3859 2) := by change ¬ Nat.Prime 59567525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11913505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3859 3) := by change ¬ Nat.Prime 59559807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19853269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3859 : row 3859 = 0 := (row_3859.checked).trans rfl
theorem model_count_3859 : modelRow 3859 = 0 := by decide
theorem modelrow_agrees_3859 : modelRow 3859 = row 3859 := model_count_3859.trans row_count_3859.symm
def row_3860 : RowCertificate 3860 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3860 1) := by change ¬ Nat.Prime 59606121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19868707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3860 2) := by change Nat.Prime 59598401; exact ProjectEulerSolutions.P58PrimeNodes.p_59598401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3860 3) := by change ¬ Nat.Prime 59590681; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 365587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3860 : row 3860 = 1 := (row_3860.checked).trans rfl
theorem model_count_3860 : modelRow 3860 = 1 := by decide
theorem modelrow_agrees_3860 : modelRow 3860 = row 3860 := model_count_3860.trans row_count_3860.symm
def row_3861 : RowCertificate 3861 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3861 1) := by change ¬ Nat.Prime 59637007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1611811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3861 2) := by change ¬ Nat.Prime 59629285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11925857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3861 3) := by change ¬ Nat.Prime 59621563; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3137977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3861 : row 3861 = 0 := (row_3861.checked).trans rfl
theorem model_count_3861 : modelRow 3861 = 0 := by decide
theorem modelrow_agrees_3861 : modelRow 3861 = row 3861 := model_count_3861.trans row_count_3861.symm
def row_3862 : RowCertificate 3862 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3862 1) := by change ¬ Nat.Prime 59667901; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1924771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3862 2) := by change ¬ Nat.Prime 59660177; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 169009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3862 3) := by change ¬ Nat.Prime 59652453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19884151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3862 : row 3862 = 0 := (row_3862.checked).trans rfl
theorem model_count_3862 : modelRow 3862 = 0 := by decide
theorem modelrow_agrees_3862 : modelRow 3862 = row 3862 := model_count_3862.trans row_count_3862.symm
def row_3863 : RowCertificate 3863 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3863 1) := by change ¬ Nat.Prime 59698803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19899601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3863 2) := by change ¬ Nat.Prime 59691077; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2058313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3863 3) := by change ¬ Nat.Prime 59683351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8526193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3863 : row 3863 = 0 := (row_3863.checked).trans rfl
theorem model_count_3863 : modelRow 3863 = 0 := by decide
theorem modelrow_agrees_3863 : modelRow 3863 = row 3863 := model_count_3863.trans row_count_3863.symm
def row_3864 : RowCertificate 3864 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3864 1) := by change Nat.Prime 59729713; exact ProjectEulerSolutions.P58PrimeNodes.p_59729713.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3864 2) := by change ¬ Nat.Prime 59721985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11944397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3864 3) := by change ¬ Nat.Prime 59714257; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 470191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3864 : row 3864 = 1 := (row_3864.checked).trans rfl
theorem model_count_3864 : modelRow 3864 = 1 := by decide
theorem modelrow_agrees_3864 : modelRow 3864 = row 3864 := model_count_3864.trans row_count_3864.symm
def row_3865 : RowCertificate 3865 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3865 1) := by change ¬ Nat.Prime 59760631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8537233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3865 2) := by change ¬ Nat.Prime 59752901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4596377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3865 3) := by change ¬ Nat.Prime 59745171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19915057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3865 : row 3865 = 0 := (row_3865.checked).trans rfl
theorem model_count_3865 : modelRow 3865 = 0 := by decide
theorem modelrow_agrees_3865 : modelRow 3865 = row 3865 := model_count_3865.trans row_count_3865.symm
def row_3866 : RowCertificate 3866 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3866 1) := by change ¬ Nat.Prime 59791557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19930519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3866 2) := by change ¬ Nat.Prime 59783825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11956765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3866 3) := by change ¬ Nat.Prime 59776093; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4598161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3866 : row 3866 = 0 := (row_3866.checked).trans rfl
theorem model_count_3866 : modelRow 3866 = 0 := by decide
theorem modelrow_agrees_3866 : modelRow 3866 = row 3866 := model_count_3866.trans row_count_3866.symm
def row_3867 : RowCertificate 3867 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3867 1) := by change ¬ Nat.Prime 59822491; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 892873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3867 2) := by change Nat.Prime 59814757; exact ProjectEulerSolutions.P58PrimeNodes.p_59814757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3867 3) := by change ¬ Nat.Prime 59807023; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1390861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3867 : row 3867 = 1 := (row_3867.checked).trans rfl
theorem model_count_3867 : modelRow 3867 = 1 := by decide
theorem modelrow_agrees_3867 : modelRow 3867 = row 3867 := model_count_3867.trans row_count_3867.symm
def row_3868 : RowCertificate 3868 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3868 1) := by change Nat.Prime 59853433; exact ProjectEulerSolutions.P58PrimeNodes.p_59853433.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3868 2) := by change ¬ Nat.Prime 59845697; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 981077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3868 3) := by change ¬ Nat.Prime 59837961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19945987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3868 : row 3868 = 1 := (row_3868.checked).trans rfl
theorem model_count_3868 : modelRow 3868 = 1 := by decide
theorem modelrow_agrees_3868 : modelRow 3868 = row 3868 := model_count_3868.trans row_count_3868.symm
def row_3869 : RowCertificate 3869 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3869 1) := by change ¬ Nat.Prime 59884383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19961461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3869 2) := by change ¬ Nat.Prime 59876645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11975329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3869 3) := by change ¬ Nat.Prime 59868907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8552701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3869 : row 3869 = 0 := (row_3869.checked).trans rfl
theorem model_count_3869 : modelRow 3869 = 0 := by decide
theorem modelrow_agrees_3869 : modelRow 3869 = row 3869 := model_count_3869.trans row_count_3869.symm
def row_3870 : RowCertificate 3870 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3870 1) := by change ¬ Nat.Prime 59915341; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3153439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3870 2) := by change ¬ Nat.Prime 59907601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4608277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3870 3) := by change ¬ Nat.Prime 59899861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8557123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3870 : row 3870 = 0 := (row_3870.checked).trans rfl
theorem model_count_3870 : modelRow 3870 = 0 := by decide
theorem modelrow_agrees_3870 : modelRow 3870 = row 3870 := model_count_3870.trans row_count_3870.symm
def row_3871 : RowCertificate 3871 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3871 1) := by change ¬ Nat.Prime 59946307; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 894721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3871 2) := by change ¬ Nat.Prime 59938565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11987713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3871 3) := by change ¬ Nat.Prime 59930823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19976941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3871 : row 3871 = 0 := (row_3871.checked).trans rfl
theorem model_count_3871 : modelRow 3871 = 0 := by decide
theorem modelrow_agrees_3871 : modelRow 3871 = row 3871 := model_count_3871.trans row_count_3871.symm
def row_3872 : RowCertificate 3872 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3872 1) := by change ¬ Nat.Prime 59977281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19992427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3872 2) := by change Nat.Prime 59969537; exact ProjectEulerSolutions.P58PrimeNodes.p_59969537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3872 3) := by change ¬ Nat.Prime 59961793; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1620589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3872 : row 3872 = 1 := (row_3872.checked).trans rfl
theorem model_count_3872 : modelRow 3872 = 1 := by decide
theorem modelrow_agrees_3872 : modelRow 3872 = row 3872 := model_count_3872.trans row_count_3872.symm
def row_3873 : RowCertificate 3873 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3873 1) := by change ¬ Nat.Prime 60008263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8572609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3873 2) := by change ¬ Nat.Prime 60000517; exact PrattCertificates.composite_of_product (factor := 1297) (cofactor := 46261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3873 3) := by change Nat.Prime 59992771; exact ProjectEulerSolutions.P58PrimeNodes.p_59992771.prime
    exact decide_eq_true prime
theorem row_count_3873 : row 3873 = 1 := (row_3873.checked).trans rfl
theorem model_count_3873 : modelRow 3873 = 1 := by decide
theorem modelrow_agrees_3873 : modelRow 3873 = row 3873 := model_count_3873.trans row_count_3873.symm
def row_3874 : RowCertificate 3874 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3874 1) := by change Nat.Prime 60039253; exact ProjectEulerSolutions.P58PrimeNodes.p_60039253.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3874 2) := by change ¬ Nat.Prime 60031505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12006301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3874 3) := by change ¬ Nat.Prime 60023757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20007919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3874 : row 3874 = 1 := (row_3874.checked).trans rfl
theorem model_count_3874 : modelRow 3874 = 1 := by decide
theorem modelrow_agrees_3874 : modelRow 3874 = row 3874 := model_count_3874.trans row_count_3874.symm
def row_3875 : RowCertificate 3875 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3875 1) := by change ¬ Nat.Prime 60070251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20023417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3875 2) := by change ¬ Nat.Prime 60062501; exact PrattCertificates.composite_of_product (factor := 5081) (cofactor := 11821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3875 3) := by change ¬ Nat.Prime 60054751; exact PrattCertificates.composite_of_product (factor := 3313) (cofactor := 18127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3875 : row 3875 = 0 := (row_3875.checked).trans rfl
theorem model_count_3875 : modelRow 3875 = 0 := by decide
theorem modelrow_agrees_3875 : modelRow 3875 = row 3875 := model_count_3875.trans row_count_3875.symm
def row_3876 : RowCertificate 3876 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3876 1) := by change Nat.Prime 60101257; exact ProjectEulerSolutions.P58PrimeNodes.p_60101257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3876 2) := by change ¬ Nat.Prime 60093505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12018701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3876 3) := by change ¬ Nat.Prime 60085753; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8583679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3876 : row 3876 = 1 := (row_3876.checked).trans rfl
theorem model_count_3876 : modelRow 3876 = 1 := by decide
theorem modelrow_agrees_3876 : modelRow 3876 = row 3876 := model_count_3876.trans row_count_3876.symm
def row_3877 : RowCertificate 3877 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3877 1) := by change Nat.Prime 60132271; exact ProjectEulerSolutions.P58PrimeNodes.p_60132271.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3877 2) := by change Nat.Prime 60124517; exact ProjectEulerSolutions.P58PrimeNodes.p_60124517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3877 3) := by change ¬ Nat.Prime 60116763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20038921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3877 : row 3877 = 2 := (row_3877.checked).trans rfl
theorem model_count_3877 : modelRow 3877 = 2 := by decide
theorem modelrow_agrees_3877 : modelRow 3877 = row 3877 := model_count_3877.trans row_count_3877.symm
def row_3878 : RowCertificate 3878 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3878 1) := by change ¬ Nat.Prime 60163293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20054431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3878 2) := by change ¬ Nat.Prime 60155537; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4627349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3878 3) := by change ¬ Nat.Prime 60147781; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1940251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3878 : row 3878 = 0 := (row_3878.checked).trans rfl
theorem model_count_3878 : modelRow 3878 = 0 := by decide
theorem modelrow_agrees_3878 : modelRow 3878 = row 3878 := model_count_3878.trans row_count_3878.symm
def row_3879 : RowCertificate 3879 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3879 1) := by change ¬ Nat.Prime 60194323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8599189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3879 2) := by change ¬ Nat.Prime 60186565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12037313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3879 3) := by change ¬ Nat.Prime 60178807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4629139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3879 : row 3879 = 0 := (row_3879.checked).trans rfl
theorem model_count_3879 : modelRow 3879 = 0 := by decide
theorem modelrow_agrees_3879 : modelRow 3879 = row 3879 := model_count_3879.trans row_count_3879.symm
def row_3880 : RowCertificate 3880 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3880 1) := by change ¬ Nat.Prime 60225361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8603623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3880 2) := by change ¬ Nat.Prime 60217601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2076469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3880 3) := by change ¬ Nat.Prime 60209841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20069947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3880 : row 3880 = 0 := (row_3880.checked).trans rfl
theorem model_count_3880 : modelRow 3880 = 0 := by decide
theorem modelrow_agrees_3880 : modelRow 3880 = row 3880 := model_count_3880.trans row_count_3880.symm
def row_3881 : RowCertificate 3881 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3881 1) := by change ¬ Nat.Prime 60256407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20085469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3881 2) := by change ¬ Nat.Prime 60248645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12049729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3881 3) := by change Nat.Prime 60240883; exact ProjectEulerSolutions.P58PrimeNodes.p_60240883.prime
    exact decide_eq_true prime
theorem row_count_3881 : row 3881 = 1 := (row_3881.checked).trans rfl
theorem model_count_3881 : modelRow 3881 = 1 := by decide
theorem modelrow_agrees_3881 : modelRow 3881 = row 3881 := model_count_3881.trans row_count_3881.symm
def row_3882 : RowCertificate 3882 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3882 1) := by change ¬ Nat.Prime 60287461; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4637497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3882 2) := by change ¬ Nat.Prime 60279697; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1629181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3882 3) := by change ¬ Nat.Prime 60271933; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3172207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3882 : row 3882 = 0 := (row_3882.checked).trans rfl
theorem model_count_3882 : modelRow 3882 = 0 := by decide
theorem modelrow_agrees_3882 : modelRow 3882 = row 3882 := model_count_3882.trans row_count_3882.symm
def row_3883 : RowCertificate 3883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3883 1) := by change ¬ Nat.Prime 60318523; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 474949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3883 2) := by change ¬ Nat.Prime 60310757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4639289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3883 3) := by change ¬ Nat.Prime 60302991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20100997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3883 : row 3883 = 0 := (row_3883.checked).trans rfl
theorem model_count_3883 : modelRow 3883 = 0 := by decide
theorem modelrow_agrees_3883 : modelRow 3883 = row 3883 := model_count_3883.trans row_count_3883.symm
def row_3884 : RowCertificate 3884 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3884 1) := by change ¬ Nat.Prime 60349593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20116531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3884 2) := by change ¬ Nat.Prime 60341825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12068365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3884 3) := by change ¬ Nat.Prime 60334057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8619151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3884 : row 3884 = 0 := (row_3884.checked).trans rfl
theorem model_count_3884 : modelRow 3884 = 0 := by decide
theorem modelrow_agrees_3884 : modelRow 3884 = row 3884 := model_count_3884.trans row_count_3884.symm
def row_3885 : RowCertificate 3885 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3885 1) := by change ¬ Nat.Prime 60380671; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4644667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3885 2) := by change Nat.Prime 60372901; exact ProjectEulerSolutions.P58PrimeNodes.p_60372901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3885 3) := by change Nat.Prime 60365131; exact ProjectEulerSolutions.P58PrimeNodes.p_60365131.prime
    exact decide_eq_true prime
theorem row_count_3885 : row 3885 = 2 := (row_3885.checked).trans rfl
theorem model_count_3885 : modelRow 3885 = 2 := by decide
theorem modelrow_agrees_3885 : modelRow 3885 = row 3885 := model_count_3885.trans row_count_3885.symm
def row_3886 : RowCertificate 3886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3886 1) := by change ¬ Nat.Prime 60411757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8630251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3886 2) := by change ¬ Nat.Prime 60403985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12080797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3886 3) := by change ¬ Nat.Prime 60396213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20132071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3886 : row 3886 = 0 := (row_3886.checked).trans rfl
theorem model_count_3886 : modelRow 3886 = 0 := by decide
theorem modelrow_agrees_3886 : modelRow 3886 = row 3886 := model_count_3886.trans row_count_3886.symm
def row_3887 : RowCertificate 3887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3887 1) := by change ¬ Nat.Prime 60442851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20147617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3887 2) := by change Nat.Prime 60435077; exact ProjectEulerSolutions.P58PrimeNodes.p_60435077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3887 3) := by change ¬ Nat.Prime 60427303; exact PrattCertificates.composite_of_product (factor := 2017) (cofactor := 29959) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3887 : row 3887 = 1 := (row_3887.checked).trans rfl
theorem model_count_3887 : modelRow 3887 = 1 := by decide
theorem modelrow_agrees_3887 : modelRow 3887 = row 3887 := model_count_3887.trans row_count_3887.symm
def row_3888 : RowCertificate 3888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3888 1) := by change ¬ Nat.Prime 60473953; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1406371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3888 2) := by change ¬ Nat.Prime 60466177; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1634221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3888 3) := by change ¬ Nat.Prime 60458401; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1950271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3888 : row 3888 = 0 := (row_3888.checked).trans rfl
theorem model_count_3888 : modelRow 3888 = 0 := by decide
theorem modelrow_agrees_3888 : modelRow 3888 = row 3888 := model_count_3888.trans row_count_3888.symm
def row_3889 : RowCertificate 3889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3889 1) := by change ¬ Nat.Prime 60505063; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3184477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3889 2) := by change ¬ Nat.Prime 60497285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12099457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3889 3) := by change ¬ Nat.Prime 60489507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20163169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3889 : row 3889 = 0 := (row_3889.checked).trans rfl
theorem model_count_3889 : modelRow 3889 = 0 := by decide
theorem modelrow_agrees_3889 : modelRow 3889 = row 3889 := model_count_3889.trans row_count_3889.symm
def row_3890 : RowCertificate 3890 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3890 1) := by change ¬ Nat.Prime 60536181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20178727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3890 2) := by change Nat.Prime 60528401; exact ProjectEulerSolutions.P58PrimeNodes.p_60528401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3890 3) := by change ¬ Nat.Prime 60520621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8645803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3890 : row 3890 = 1 := (row_3890.checked).trans rfl
theorem model_count_3890 : modelRow 3890 = 1 := by decide
theorem modelrow_agrees_3890 : modelRow 3890 = row 3890 := model_count_3890.trans row_count_3890.symm
def row_3891 : RowCertificate 3891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3891 1) := by change ¬ Nat.Prime 60567307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3187753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3891 2) := by change ¬ Nat.Prime 60559525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12111905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3891 3) := by change ¬ Nat.Prime 60551743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8650249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3891 : row 3891 = 0 := (row_3891.checked).trans rfl
theorem model_count_3891 : modelRow 3891 = 0 := by decide
theorem modelrow_agrees_3891 : modelRow 3891 = row 3891 := model_count_3891.trans row_count_3891.symm
def row_3892 : RowCertificate 3892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3892 1) := by change ¬ Nat.Prime 60598441; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 555949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3892 2) := by change ¬ Nat.Prime 60590657; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2089333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3892 3) := by change ¬ Nat.Prime 60582873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20194291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3892 : row 3892 = 0 := (row_3892.checked).trans rfl
theorem model_count_3892 : modelRow 3892 = 0 := by decide
theorem modelrow_agrees_3892 : modelRow 3892 = row 3892 := model_count_3892.trans row_count_3892.symm
def row_3893 : RowCertificate 3893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3893 1) := by change ¬ Nat.Prime 60629583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20209861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3893 2) := by change ¬ Nat.Prime 60621797; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 102229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3893 3) := by change Nat.Prime 60614011; exact ProjectEulerSolutions.P58PrimeNodes.p_60614011.prime
    exact decide_eq_true prime
theorem row_count_3893 : row 3893 = 1 := (row_3893.checked).trans rfl
theorem model_count_3893 : modelRow 3893 = 1 := by decide
theorem modelrow_agrees_3893 : modelRow 3893 = row 3893 := model_count_3893.trans row_count_3893.symm
def row_3894 : RowCertificate 3894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3894 1) := by change ¬ Nat.Prime 60660733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8665819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3894 2) := by change ¬ Nat.Prime 60652945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12130589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3894 3) := by change Nat.Prime 60645157; exact ProjectEulerSolutions.P58PrimeNodes.p_60645157.prime
    exact decide_eq_true prime
theorem row_count_3894 : row 3894 = 1 := (row_3894.checked).trans rfl
theorem model_count_3894 : modelRow 3894 = 1 := by decide
theorem modelrow_agrees_3894 : modelRow 3894 = row 3894 := model_count_3894.trans row_count_3894.symm
def row_3895 : RowCertificate 3895 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3895 1) := by change ¬ Nat.Prime 60691891; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4668607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3895 2) := by change ¬ Nat.Prime 60684101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3569653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3895 3) := by change ¬ Nat.Prime 60676311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20225437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3895 : row 3895 = 0 := (row_3895.checked).trans rfl
theorem model_count_3895 : modelRow 3895 = 0 := by decide
theorem modelrow_agrees_3895 : modelRow 3895 = row 3895 := model_count_3895.trans row_count_3895.symm
def row_3896 : RowCertificate 3896 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3896 1) := by change ¬ Nat.Prime 60723057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20241019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3896 2) := by change ¬ Nat.Prime 60715265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12143053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3896 3) := by change Nat.Prime 60707473; exact ProjectEulerSolutions.P58PrimeNodes.p_60707473.prime
    exact decide_eq_true prime
theorem row_count_3896 : row 3896 = 1 := (row_3896.checked).trans rfl
theorem model_count_3896 : modelRow 3896 = 1 := by decide
theorem modelrow_agrees_3896 : modelRow 3896 = row 3896 := model_count_3896.trans row_count_3896.symm
def row_3897 : RowCertificate 3897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3897 1) := by change ¬ Nat.Prime 60754231; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 995971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3897 2) := by change ¬ Nat.Prime 60746437; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 59497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3897 3) := by change ¬ Nat.Prime 60738643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8676949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3897 : row 3897 = 0 := (row_3897.checked).trans rfl
theorem model_count_3897 : modelRow 3897 = 0 := by decide
theorem modelrow_agrees_3897 : modelRow 3897 = row 3897 := model_count_3897.trans row_count_3897.symm
def row_3898 : RowCertificate 3898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3898 1) := by change ¬ Nat.Prime 60785413; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4675801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3898 2) := by change Nat.Prime 60777617; exact ProjectEulerSolutions.P58PrimeNodes.p_60777617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3898 3) := by change ¬ Nat.Prime 60769821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20256607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3898 : row 3898 = 1 := (row_3898.checked).trans rfl
theorem model_count_3898 : modelRow 3898 = 1 := by decide
theorem modelrow_agrees_3898 : modelRow 3898 = row 3898 := model_count_3898.trans row_count_3898.symm
def row_3899 : RowCertificate 3899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3899 1) := by change ¬ Nat.Prime 60816603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20272201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3899 2) := by change ¬ Nat.Prime 60808805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12161761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3899 3) := by change ¬ Nat.Prime 60801007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3200053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3899 : row 3899 = 0 := (row_3899.checked).trans rfl
theorem model_count_3899 : modelRow 3899 = 0 := by decide
theorem modelrow_agrees_3899 : modelRow 3899 = row 3899 := model_count_3899.trans row_count_3899.symm
def row_3900 : RowCertificate 3900 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3900 1) := by change ¬ Nat.Prime 60847801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8692543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3900 2) := by change ¬ Nat.Prime 60840001; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 194377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3900 3) := by change ¬ Nat.Prime 60832201; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 59581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3900 : row 3900 = 0 := (row_3900.checked).trans rfl
theorem model_count_3900 : modelRow 3900 = 0 := by decide
theorem modelrow_agrees_3900 : modelRow 3900 = row 3900 := model_count_3900.trans row_count_3900.symm
def row_3901 : RowCertificate 3901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3901 1) := by change ¬ Nat.Prime 60879007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8697001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3901 2) := by change ¬ Nat.Prime 60871205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12174241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3901 3) := by change ¬ Nat.Prime 60863403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20287801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3901 : row 3901 = 0 := (row_3901.checked).trans rfl
theorem model_count_3901 : modelRow 3901 = 0 := by decide
theorem modelrow_agrees_3901 : modelRow 3901 = row 3901 := model_count_3901.trans row_count_3901.symm
def row_3902 : RowCertificate 3902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3902 1) := by change ¬ Nat.Prime 60910221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20303407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3902 2) := by change Nat.Prime 60902417; exact ProjectEulerSolutions.P58PrimeNodes.p_60902417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3902 3) := by change ¬ Nat.Prime 60894613; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4684201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3902 : row 3902 = 1 := (row_3902.checked).trans rfl
theorem model_count_3902 : modelRow 3902 = 1 := by decide
theorem modelrow_agrees_3902 : modelRow 3902 = row 3902 := model_count_3902.trans row_count_3902.symm
def row_3903 : RowCertificate 3903 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3903 1) := by change ¬ Nat.Prime 60941443; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1965853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3903 2) := by change Nat.Prime 60933637; exact ProjectEulerSolutions.P58PrimeNodes.p_60933637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3903 3) := by change ¬ Nat.Prime 60925831; exact PrattCertificates.composite_of_product (factor := 7411) (cofactor := 8221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3903 : row 3903 = 1 := (row_3903.checked).trans rfl
theorem model_count_3903 : modelRow 3903 = 1 := by decide
theorem modelrow_agrees_3903 : modelRow 3903 = row 3903 := model_count_3903.trans row_count_3903.symm
def row_3904 : RowCertificate 3904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3904 1) := by change ¬ Nat.Prime 60972673; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 82507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3904 2) := by change ¬ Nat.Prime 60964865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12192973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3904 3) := by change ¬ Nat.Prime 60957057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20319019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3904 : row 3904 = 0 := (row_3904.checked).trans rfl
theorem model_count_3904 : modelRow 3904 = 0 := by decide
theorem modelrow_agrees_3904 : modelRow 3904 = row 3904 := model_count_3904.trans row_count_3904.symm
def row_3905 : RowCertificate 3905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3905 1) := by change ¬ Nat.Prime 61003911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20334637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3905 2) := by change Nat.Prime 60996101; exact ProjectEulerSolutions.P58PrimeNodes.p_60996101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3905 3) := by change ¬ Nat.Prime 60988291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8712613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3905 : row 3905 = 1 := (row_3905.checked).trans rfl
theorem model_count_3905 : modelRow 3905 = 1 := by decide
theorem modelrow_agrees_3905 : modelRow 3905 = row 3905 := model_count_3905.trans row_count_3905.symm
def row_3906 : RowCertificate 3906 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3906 1) := by change Nat.Prime 61035157; exact ProjectEulerSolutions.P58PrimeNodes.p_61035157.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3906 2) := by change ¬ Nat.Prime 61027345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12205469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3906 3) := by change ¬ Nat.Prime 61019533; exact PrattCertificates.composite_of_product (factor := 5569) (cofactor := 10957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3906 : row 3906 = 1 := (row_3906.checked).trans rfl
theorem model_count_3906 : modelRow 3906 = 1 := by decide
theorem modelrow_agrees_3906 : modelRow 3906 = row 3906 := model_count_3906.trans row_count_3906.symm
def row_3907 : RowCertificate 3907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3907 1) := by change ¬ Nat.Prime 61066411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8723773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3907 2) := by change ¬ Nat.Prime 61058597; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1152049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3907 3) := by change ¬ Nat.Prime 61050783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20350261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3907 : row 3907 = 0 := (row_3907.checked).trans rfl
theorem model_count_3907 : modelRow 3907 = 0 := by decide
theorem modelrow_agrees_3907 : modelRow 3907 = row 3907 := model_count_3907.trans row_count_3907.symm
def row_3908 : RowCertificate 3908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3908 1) := by change ¬ Nat.Prime 61097673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20365891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3908 2) := by change ¬ Nat.Prime 61089857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3593521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3908 3) := by change ¬ Nat.Prime 61082041; exact PrattCertificates.composite_of_product (factor := 2311) (cofactor := 26431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3908 : row 3908 = 0 := (row_3908.checked).trans rfl
theorem model_count_3908 : modelRow 3908 = 0 := by decide
theorem modelrow_agrees_3908 : modelRow 3908 = row 3908 := model_count_3908.trans row_count_3908.symm
def row_3909 : RowCertificate 3909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3909 1) := by change ¬ Nat.Prime 61128943; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 65239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3909 2) := by change ¬ Nat.Prime 61121125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12224225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3909 3) := by change ¬ Nat.Prime 61113307; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1971397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3909 : row 3909 = 0 := (row_3909.checked).trans rfl
theorem model_count_3909 : modelRow 3909 = 0 := by decide
theorem modelrow_agrees_3909 : modelRow 3909 = row 3909 := model_count_3909.trans row_count_3909.symm
def row_3910 : RowCertificate 3910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3910 1) := by change ¬ Nat.Prime 61160221; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3218959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3910 2) := by change Nat.Prime 61152401; exact ProjectEulerSolutions.P58PrimeNodes.p_61152401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3910 3) := by change ¬ Nat.Prime 61144581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20381527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3910 : row 3910 = 1 := (row_3910.checked).trans rfl
theorem model_count_3910 : modelRow 3910 = 1 := by decide
theorem modelrow_agrees_3910 : modelRow 3910 = row 3910 := model_count_3910.trans row_count_3910.symm
def row_3911 : RowCertificate 3911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3911 1) := by change ¬ Nat.Prime 61191507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20397169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3911 2) := by change ¬ Nat.Prime 61183685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12236737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3911 3) := by change ¬ Nat.Prime 61175863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8739409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3911 : row 3911 = 0 := (row_3911.checked).trans rfl
theorem model_count_3911 : modelRow 3911 = 0 := by decide
theorem modelrow_agrees_3911 : modelRow 3911 = row 3911 := model_count_3911.trans row_count_3911.symm
def row_3912 : RowCertificate 3912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3912 1) := by change ¬ Nat.Prime 61222801; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 149689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3912 2) := by change ¬ Nat.Prime 61214977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3600881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3912 3) := by change ¬ Nat.Prime 61207153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8743879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3912 : row 3912 = 0 := (row_3912.checked).trans rfl
theorem model_count_3912 : modelRow 3912 = 0 := by decide
theorem modelrow_agrees_3912 : modelRow 3912 = row 3912 := model_count_3912.trans row_count_3912.symm
def row_3913 : RowCertificate 3913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3913 1) := by change ¬ Nat.Prime 61254103; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 440677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3913 2) := by change ¬ Nat.Prime 61246277; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 92657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3913 3) := by change ¬ Nat.Prime 61238451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20412817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3913 : row 3913 = 0 := (row_3913.checked).trans rfl
theorem model_count_3913 : modelRow 3913 = 0 := by decide
theorem modelrow_agrees_3913 : modelRow 3913 = row 3913 := model_count_3913.trans row_count_3913.symm
def row_3914 : RowCertificate 3914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3914 1) := by change ¬ Nat.Prime 61285413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20428471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3914 2) := by change ¬ Nat.Prime 61277585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12255517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3914 3) := by change ¬ Nat.Prime 61269757; exact PrattCertificates.composite_of_product (factor := 1123) (cofactor := 54559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3914 : row 3914 = 0 := (row_3914.checked).trans rfl
theorem model_count_3914 : modelRow 3914 = 0 := by decide
theorem modelrow_agrees_3914 : modelRow 3914 = row 3914 := model_count_3914.trans row_count_3914.symm
def row_3915 : RowCertificate 3915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3915 1) := by change ¬ Nat.Prime 61316731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8759533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3915 2) := by change ¬ Nat.Prime 61308901; exact PrattCertificates.composite_of_product (factor := 6029) (cofactor := 10169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3915 3) := by change ¬ Nat.Prime 61301071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4715467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3915 : row 3915 = 0 := (row_3915.checked).trans rfl
theorem model_count_3915 : modelRow 3915 = 0 := by decide
theorem modelrow_agrees_3915 : modelRow 3915 = row 3915 := model_count_3915.trans row_count_3915.symm
def row_3916 : RowCertificate 3916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3916 1) := by change ¬ Nat.Prime 61348057; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1426699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3916 2) := by change ¬ Nat.Prime 61340225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12268045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3916 3) := by change ¬ Nat.Prime 61332393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20444131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3916 : row 3916 = 0 := (row_3916.checked).trans rfl
theorem model_count_3916 : modelRow 3916 = 0 := by decide
theorem modelrow_agrees_3916 : modelRow 3916 = row 3916 := model_count_3916.trans row_count_3916.symm
def row_3917 : RowCertificate 3917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3917 1) := by change ¬ Nat.Prime 61379391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20459797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3917 2) := by change ¬ Nat.Prime 61371557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4720889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3917 3) := by change ¬ Nat.Prime 61363723; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1658479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3917 : row 3917 = 0 := (row_3917.checked).trans rfl
theorem model_count_3917 : modelRow 3917 = 0 := by decide
theorem modelrow_agrees_3917 : modelRow 3917 = row 3917 := model_count_3917.trans row_count_3917.symm
def row_3918 : RowCertificate 3918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3918 1) := by change ¬ Nat.Prime 61410733; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 51133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3918 2) := by change Nat.Prime 61402897; exact ProjectEulerSolutions.P58PrimeNodes.p_61402897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3918 3) := by change ¬ Nat.Prime 61395061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8770723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3918 : row 3918 = 1 := (row_3918.checked).trans rfl
theorem model_count_3918 : modelRow 3918 = 1 := by decide
theorem modelrow_agrees_3918 : modelRow 3918 = row 3918 := model_count_3918.trans row_count_3918.symm
def row_3919 : RowCertificate 3919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3919 1) := by change ¬ Nat.Prime 61442083; exact PrattCertificates.composite_of_product (factor := 2011) (cofactor := 30553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3919 2) := by change ¬ Nat.Prime 61434245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12286849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3919 3) := by change ¬ Nat.Prime 61426407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20475469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3919 : row 3919 = 0 := (row_3919.checked).trans rfl
theorem model_count_3919 : modelRow 3919 = 0 := by decide
theorem modelrow_agrees_3919 : modelRow 3919 = row 3919 := model_count_3919.trans row_count_3919.symm
def row_3920 : RowCertificate 3920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3920 1) := by change ¬ Nat.Prime 61473441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20491147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3920 2) := by change ¬ Nat.Prime 61465601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1499161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3920 3) := by change ¬ Nat.Prime 61457761; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3234619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3920 : row 3920 = 0 := (row_3920.checked).trans rfl
theorem model_count_3920 : modelRow 3920 = 0 := by decide
theorem modelrow_agrees_3920 : modelRow 3920 = row 3920 := model_count_3920.trans row_count_3920.symm
def row_3921 : RowCertificate 3921 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3921 1) := by change ¬ Nat.Prime 61504807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8786401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3921 2) := by change ¬ Nat.Prime 61496965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12299393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3921 3) := by change ¬ Nat.Prime 61489123; exact PrattCertificates.composite_of_product (factor := 4231) (cofactor := 14533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3921 : row 3921 = 0 := (row_3921.checked).trans rfl
theorem model_count_3921 : modelRow 3921 = 0 := by decide
theorem modelrow_agrees_3921 : modelRow 3921 = row 3921 := model_count_3921.trans row_count_3921.symm
def row_3922 : RowCertificate 3922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3922 1) := by change ¬ Nat.Prime 61536181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8790883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3922 2) := by change ¬ Nat.Prime 61528337; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4732949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3922 3) := by change ¬ Nat.Prime 61520493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20506831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3922 : row 3922 = 0 := (row_3922.checked).trans rfl
theorem model_count_3922 : modelRow 3922 = 0 := by decide
theorem modelrow_agrees_3922 : modelRow 3922 = row 3922 := model_count_3922.trans row_count_3922.symm
def row_3923 : RowCertificate 3923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3923 1) := by change ¬ Nat.Prime 61567563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20522521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3923 2) := by change ¬ Nat.Prime 61559717; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 449341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3923 3) := by change Nat.Prime 61551871; exact ProjectEulerSolutions.P58PrimeNodes.p_61551871.prime
    exact decide_eq_true prime
theorem row_count_3923 : row 3923 = 1 := (row_3923.checked).trans rfl
theorem model_count_3923 : modelRow 3923 = 1 := by decide
theorem modelrow_agrees_3923 : modelRow 3923 = row 3923 := model_count_3923.trans row_count_3923.symm
def row_3924 : RowCertificate 3924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3924 1) := by change ¬ Nat.Prime 61598953; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4738381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3924 2) := by change ¬ Nat.Prime 61591105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12318221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3924 3) := by change Nat.Prime 61583257; exact ProjectEulerSolutions.P58PrimeNodes.p_61583257.prime
    exact decide_eq_true prime
theorem row_count_3924 : row 3924 = 1 := (row_3924.checked).trans rfl
theorem model_count_3924 : modelRow 3924 = 1 := by decide
theorem modelrow_agrees_3924 : modelRow 3924 = row 3924 := model_count_3924.trans row_count_3924.symm
def row_3925 : RowCertificate 3925 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3925 1) := by change Nat.Prime 61630351; exact ProjectEulerSolutions.P58PrimeNodes.p_61630351.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3925 2) := by change ¬ Nat.Prime 61622501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3624853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3925 3) := by change ¬ Nat.Prime 61614651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20538217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3925 : row 3925 = 1 := (row_3925.checked).trans rfl
theorem model_count_3925 : modelRow 3925 = 1 := by decide
theorem modelrow_agrees_3925 : modelRow 3925 = row 3925 := model_count_3925.trans row_count_3925.symm
def row_3926 : RowCertificate 3926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3926 1) := by change ¬ Nat.Prime 61661757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20553919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3926 2) := by change ¬ Nat.Prime 61653905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12330781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3926 3) := by change ¬ Nat.Prime 61646053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8806579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3926 : row 3926 = 0 := (row_3926.checked).trans rfl
theorem model_count_3926 : modelRow 3926 = 0 := by decide
theorem modelrow_agrees_3926 : modelRow 3926 = row 3926 := model_count_3926.trans row_count_3926.symm
def row_3927 : RowCertificate 3927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3927 1) := by change ¬ Nat.Prime 61693171; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3247009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3927 2) := by change Nat.Prime 61685317; exact ProjectEulerSolutions.P58PrimeNodes.p_61685317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3927 3) := by change ¬ Nat.Prime 61677463; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 309937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3927 : row 3927 = 1 := (row_3927.checked).trans rfl
theorem model_count_3927 : modelRow 3927 = 1 := by decide
theorem modelrow_agrees_3927 : modelRow 3927 = row 3927 := model_count_3927.trans row_count_3927.symm
def row_3928 : RowCertificate 3928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3928 1) := by change ¬ Nat.Prime 61724593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8817799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3928 2) := by change Nat.Prime 61716737; exact ProjectEulerSolutions.P58PrimeNodes.p_61716737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3928 3) := by change ¬ Nat.Prime 61708881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20569627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3928 : row 3928 = 1 := (row_3928.checked).trans rfl
theorem model_count_3928 : modelRow 3928 = 1 := by decide
theorem modelrow_agrees_3928 : modelRow 3928 = row 3928 := model_count_3928.trans row_count_3928.symm
def row_3929 : RowCertificate 3929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3929 1) := by change ¬ Nat.Prime 61756023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20585341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3929 2) := by change ¬ Nat.Prime 61748165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12349633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3929 3) := by change Nat.Prime 61740307; exact ProjectEulerSolutions.P58PrimeNodes.p_61740307.prime
    exact decide_eq_true prime
theorem row_count_3929 : row 3929 = 1 := (row_3929.checked).trans rfl
theorem model_count_3929 : modelRow 3929 = 1 := by decide
theorem modelrow_agrees_3929 : modelRow 3929 = row 3929 := model_count_3929.trans row_count_3929.symm
def row_3930 : RowCertificate 3930 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3930 1) := by change Nat.Prime 61787461; exact ProjectEulerSolutions.P58PrimeNodes.p_61787461.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3930 2) := by change ¬ Nat.Prime 61779601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4752277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3930 3) := by change Nat.Prime 61771741; exact ProjectEulerSolutions.P58PrimeNodes.p_61771741.prime
    exact decide_eq_true prime
theorem row_count_3930 : row 3930 = 2 := (row_3930.checked).trans rfl
theorem model_count_3930 : modelRow 3930 = 2 := by decide
theorem modelrow_agrees_3930 : modelRow 3930 = row 3930 := model_count_3930.trans row_count_3930.symm
def row_3931 : RowCertificate 3931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3931 1) := by change ¬ Nat.Prime 61818907; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1437649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3931 2) := by change ¬ Nat.Prime 61811045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12362209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3931 3) := by change ¬ Nat.Prime 61803183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20601061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3931 : row 3931 = 0 := (row_3931.checked).trans rfl
theorem model_count_3931 : modelRow 3931 = 0 := by decide
theorem modelrow_agrees_3931 : modelRow 3931 = row 3931 := model_count_3931.trans row_count_3931.symm
def row_3932 : RowCertificate 3932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3932 1) := by change ¬ Nat.Prime 61850361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20616787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3932 2) := by change Nat.Prime 61842497; exact ProjectEulerSolutions.P58PrimeNodes.p_61842497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3932 3) := by change ¬ Nat.Prime 61834633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8833519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3932 : row 3932 = 1 := (row_3932.checked).trans rfl
theorem model_count_3932 : modelRow 3932 = 1 := by decide
theorem modelrow_agrees_3932 : modelRow 3932 = row 3932 := model_count_3932.trans row_count_3932.symm
def row_3933 : RowCertificate 3933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3933 1) := by change ¬ Nat.Prime 61881823; exact PrattCertificates.composite_of_product (factor := 1087) (cofactor := 56929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3933 2) := by change ¬ Nat.Prime 61873957; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 695213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3933 3) := by change ¬ Nat.Prime 61866091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8838013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3933 : row 3933 = 0 := (row_3933.checked).trans rfl
theorem model_count_3933 : modelRow 3933 = 0 := by decide
theorem modelrow_agrees_3933 : modelRow 3933 = row 3933 := model_count_3933.trans row_count_3933.symm
def row_3934 : RowCertificate 3934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3934 1) := by change ¬ Nat.Prime 61913293; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4762561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3934 2) := by change ¬ Nat.Prime 61905425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12381085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3934 3) := by change ¬ Nat.Prime 61897557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20632519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3934 : row 3934 = 0 := (row_3934.checked).trans rfl
theorem model_count_3934 : modelRow 3934 = 0 := by decide
theorem modelrow_agrees_3934 : modelRow 3934 = row 3934 := model_count_3934.trans row_count_3934.symm
def row_3935 : RowCertificate 3935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3935 1) := by change ¬ Nat.Prime 61944771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20648257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3935 2) := by change ¬ Nat.Prime 61936901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4764377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3935 3) := by change Nat.Prime 61929031; exact ProjectEulerSolutions.P58PrimeNodes.p_61929031.prime
    exact decide_eq_true prime
theorem row_count_3935 : row 3935 = 1 := (row_3935.checked).trans rfl
theorem model_count_3935 : modelRow 3935 = 1 := by decide
theorem modelrow_agrees_3935 : modelRow 3935 = row 3935 := model_count_3935.trans row_count_3935.symm
def row_3936 : RowCertificate 3936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3936 1) := by change ¬ Nat.Prime 61976257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8853751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3936 2) := by change ¬ Nat.Prime 61968385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12393677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3936 3) := by change Nat.Prime 61960513; exact ProjectEulerSolutions.P58PrimeNodes.p_61960513.prime
    exact decide_eq_true prime
theorem row_count_3936 : row 3936 = 1 := (row_3936.checked).trans rfl
theorem model_count_3936 : modelRow 3936 = 1 := by decide
theorem modelrow_agrees_3936 : modelRow 3936 = row 3936 := model_count_3936.trans row_count_3936.symm
def row_3937 : RowCertificate 3937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3937 1) := by change ¬ Nat.Prime 62007751; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4769827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3937 2) := by change ¬ Nat.Prime 61999877; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1169809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3937 3) := by change ¬ Nat.Prime 61992003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20664001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3937 : row 3937 = 0 := (row_3937.checked).trans rfl
theorem model_count_3937 : modelRow 3937 = 0 := by decide
theorem modelrow_agrees_3937 : modelRow 3937 = row 3937 := model_count_3937.trans row_count_3937.symm
def row_3938 : RowCertificate 3938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3938 1) := by change ¬ Nat.Prime 62039253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20679751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3938 2) := by change ¬ Nat.Prime 62031377; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2139013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3938 3) := by change ¬ Nat.Prime 62023501; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1442407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3938 : row 3938 = 0 := (row_3938.checked).trans rfl
theorem model_count_3938 : modelRow 3938 = 0 := by decide
theorem modelrow_agrees_3938 : modelRow 3938 = row 3938 := model_count_3938.trans row_count_3938.symm
def row_3939 : RowCertificate 3939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3939 1) := by change ¬ Nat.Prime 62070763; exact PrattCertificates.composite_of_product (factor := 967) (cofactor := 64189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3939 2) := by change ¬ Nat.Prime 62062885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12412577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3939 3) := by change ¬ Nat.Prime 62055007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8865001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3939 : row 3939 = 0 := (row_3939.checked).trans rfl
theorem model_count_3939 : modelRow 3939 = 0 := by decide
theorem modelrow_agrees_3939 : modelRow 3939 = row 3939 := model_count_3939.trans row_count_3939.symm
def row_3940 : RowCertificate 3940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3940 1) := by change ¬ Nat.Prime 62102281; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 446779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3940 2) := by change ¬ Nat.Prime 62094401; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1017941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3940 3) := by change ¬ Nat.Prime 62086521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20695507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3940 : row 3940 = 0 := (row_3940.checked).trans rfl
theorem model_count_3940 : modelRow 3940 = 0 := by decide
theorem modelrow_agrees_3940 : modelRow 3940 = row 3940 := model_count_3940.trans row_count_3940.symm
def row_3941 : RowCertificate 3941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3941 1) := by change ¬ Nat.Prime 62133807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20711269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3941 2) := by change ¬ Nat.Prime 62125925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12425185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3941 3) := by change ¬ Nat.Prime 62118043; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4778311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3941 : row 3941 = 0 := (row_3941.checked).trans rfl
theorem model_count_3941 : modelRow 3941 = 0 := by decide
theorem modelrow_agrees_3941 : modelRow 3941 = row 3941 := model_count_3941.trans row_count_3941.symm
def row_3942 : RowCertificate 3942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3942 1) := by change ¬ Nat.Prime 62165341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8880763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3942 2) := by change ¬ Nat.Prime 62157457; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3656321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3942 3) := by change Nat.Prime 62149573; exact ProjectEulerSolutions.P58PrimeNodes.p_62149573.prime
    exact decide_eq_true prime
theorem row_count_3942 : row 3942 = 1 := (row_3942.checked).trans rfl
theorem model_count_3942 : modelRow 3942 = 1 := by decide
theorem modelrow_agrees_3942 : modelRow 3942 = row 3942 := model_count_3942.trans row_count_3942.symm
def row_3943 : RowCertificate 3943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3943 1) := by change ¬ Nat.Prime 62196883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8885269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3943 2) := by change ¬ Nat.Prime 62188997; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4783769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3943 3) := by change ¬ Nat.Prime 62181111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20727037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3943 : row 3943 = 0 := (row_3943.checked).trans rfl
theorem model_count_3943 : modelRow 3943 = 0 := by decide
theorem modelrow_agrees_3943 : modelRow 3943 = row 3943 := model_count_3943.trans row_count_3943.symm
def row_3944 : RowCertificate 3944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3944 1) := by change ¬ Nat.Prime 62228433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20742811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3944 2) := by change ¬ Nat.Prime 62220545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12444109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3944 3) := by change ¬ Nat.Prime 62212657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4785589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3944 : row 3944 = 0 := (row_3944.checked).trans rfl
theorem model_count_3944 : modelRow 3944 = 0 := by decide
theorem modelrow_agrees_3944 : modelRow 3944 = row 3944 := model_count_3944.trans row_count_3944.symm
def row_3945 : RowCertificate 3945 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3945 1) := by change Nat.Prime 62259991; exact ProjectEulerSolutions.P58PrimeNodes.p_62259991.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3945 2) := by change ¬ Nat.Prime 62252101; exact PrattCertificates.composite_of_product (factor := 4421) (cofactor := 14081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3945 3) := by change Nat.Prime 62244211; exact ProjectEulerSolutions.P58PrimeNodes.p_62244211.prime
    exact decide_eq_true prime
theorem row_count_3945 : row 3945 = 2 := (row_3945.checked).trans rfl
theorem model_count_3945 : modelRow 3945 = 2 := by decide
theorem modelrow_agrees_3945 : modelRow 3945 = row 3945 := model_count_3945.trans row_count_3945.symm
def row_3946 : RowCertificate 3946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3946 1) := by change ¬ Nat.Prime 62291557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3278503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3946 2) := by change ¬ Nat.Prime 62283665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12456733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3946 3) := by change ¬ Nat.Prime 62275773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20758591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3946 : row 3946 = 0 := (row_3946.checked).trans rfl
theorem model_count_3946 : modelRow 3946 = 0 := by decide
theorem modelrow_agrees_3946 : modelRow 3946 = row 3946 := model_count_3946.trans row_count_3946.symm
def row_3947 : RowCertificate 3947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3947 1) := by change ¬ Nat.Prime 62323131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20774377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3947 2) := by change ¬ Nat.Prime 62315237; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 316321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3947 3) := by change ¬ Nat.Prime 62307343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8901049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3947 : row 3947 = 0 := (row_3947.checked).trans rfl
theorem model_count_3947 : modelRow 3947 = 0 := by decide
theorem modelrow_agrees_3947 : modelRow 3947 = row 3947 := model_count_3947.trans row_count_3947.symm
def row_3948 : RowCertificate 3948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3948 1) := by change ¬ Nat.Prime 62354713; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3281827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3948 2) := by change ¬ Nat.Prime 62346817; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4795909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3948 3) := by change Nat.Prime 62338921; exact ProjectEulerSolutions.P58PrimeNodes.p_62338921.prime
    exact decide_eq_true prime
theorem row_count_3948 : row 3948 = 1 := (row_3948.checked).trans rfl
theorem model_count_3948 : modelRow 3948 = 1 := by decide
theorem modelrow_agrees_3948 : modelRow 3948 = row 3948 := model_count_3948.trans row_count_3948.symm
def row_3949 : RowCertificate 3949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3949 1) := by change ¬ Nat.Prime 62386303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8912329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3949 2) := by change ¬ Nat.Prime 62378405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12475681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3949 3) := by change ¬ Nat.Prime 62370507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20790169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3949 : row 3949 = 0 := (row_3949.checked).trans rfl
theorem model_count_3949 : modelRow 3949 = 0 := by decide
theorem modelrow_agrees_3949 : modelRow 3949 = row 3949 := model_count_3949.trans row_count_3949.symm
def row_3950 : RowCertificate 3950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3950 1) := by change ¬ Nat.Prime 62417901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20805967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3950 2) := by change ¬ Nat.Prime 62410001; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2152069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3950 3) := by change ¬ Nat.Prime 62402101; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2012971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3950 : row 3950 = 0 := (row_3950.checked).trans rfl
theorem model_count_3950 : modelRow 3950 = 0 := by decide
theorem modelrow_agrees_3950 : modelRow 3950 = row 3950 := model_count_3950.trans row_count_3950.symm
def row_3951 : RowCertificate 3951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3951 1) := by change ¬ Nat.Prime 62449507; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 136651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3951 2) := by change ¬ Nat.Prime 62441605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12488321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3951 3) := by change Nat.Prime 62433703; exact ProjectEulerSolutions.P58PrimeNodes.p_62433703.prime
    exact decide_eq_true prime
theorem row_count_3951 : row 3951 = 1 := (row_3951.checked).trans rfl
theorem model_count_3951 : modelRow 3951 = 1 := by decide
theorem modelrow_agrees_3951 : modelRow 3951 = row 3951 := model_count_3951.trans row_count_3951.symm
def row_3952 : RowCertificate 3952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3952 1) := by change ¬ Nat.Prime 62481121; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 179029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3952 2) := by change ¬ Nat.Prime 62473217; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1523737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3952 3) := by change ¬ Nat.Prime 62465313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20821771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3952 : row 3952 = 0 := (row_3952.checked).trans rfl
theorem model_count_3952 : modelRow 3952 = 0 := by decide
theorem modelrow_agrees_3952 : modelRow 3952 = row 3952 := model_count_3952.trans row_count_3952.symm
def row_3953 : RowCertificate 3953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3953 1) := by change ¬ Nat.Prime 62512743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20837581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3953 2) := by change Nat.Prime 62504837; exact ProjectEulerSolutions.P58PrimeNodes.p_62504837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3953 3) := by change ¬ Nat.Prime 62496931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8928133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3953 : row 3953 = 1 := (row_3953.checked).trans rfl
theorem model_count_3953 : modelRow 3953 = 1 := by decide
theorem modelrow_agrees_3953 : modelRow 3953 = row 3953 := model_count_3953.trans row_count_3953.symm
def row_3954 : RowCertificate 3954 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3954 1) := by change Nat.Prime 62544373; exact ProjectEulerSolutions.P58PrimeNodes.p_62544373.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3954 2) := by change ¬ Nat.Prime 62536465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12507293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3954 3) := by change ¬ Nat.Prime 62528557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8932651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3954 : row 3954 = 1 := (row_3954.checked).trans rfl
theorem model_count_3954 : modelRow 3954 = 1 := by decide
theorem modelrow_agrees_3954 : modelRow 3954 = row 3954 := model_count_3954.trans row_count_3954.symm
def row_3955 : RowCertificate 3955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3955 1) := by change ¬ Nat.Prime 62576011; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2018581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3955 2) := by change Nat.Prime 62568101; exact ProjectEulerSolutions.P58PrimeNodes.p_62568101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3955 3) := by change ¬ Nat.Prime 62560191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20853397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3955 : row 3955 = 1 := (row_3955.checked).trans rfl
theorem model_count_3955 : modelRow 3955 = 1 := by decide
theorem modelrow_agrees_3955 : modelRow 3955 = row 3955 := model_count_3955.trans row_count_3955.symm
def row_3956 : RowCertificate 3956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3956 1) := by change ¬ Nat.Prime 62607657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20869219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3956 2) := by change ¬ Nat.Prime 62599745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12519949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3956 3) := by change ¬ Nat.Prime 62591833; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3294307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3956 : row 3956 = 0 := (row_3956.checked).trans rfl
theorem model_count_3956 : modelRow 3956 = 0 := by decide
theorem modelrow_agrees_3956 : modelRow 3956 = row 3956 := model_count_3956.trans row_count_3956.symm
def row_3957 : RowCertificate 3957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3957 1) := by change ¬ Nat.Prime 62639311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8948473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3957 2) := by change ¬ Nat.Prime 62631397; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 153133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3957 3) := by change ¬ Nat.Prime 62623483; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4817191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3957 : row 3957 = 0 := (row_3957.checked).trans rfl
theorem model_count_3957 : modelRow 3957 = 0 := by decide
theorem modelrow_agrees_3957 : modelRow 3957 = row 3957 := model_count_3957.trans row_count_3957.symm
def row_3958 : RowCertificate 3958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3958 1) := by change ¬ Nat.Prime 62670973; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1027393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3958 2) := by change ¬ Nat.Prime 62663057; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 112501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3958 3) := by change ¬ Nat.Prime 62655141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20885047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3958 : row 3958 = 0 := (row_3958.checked).trans rfl
theorem model_count_3958 : modelRow 3958 = 0 := by decide
theorem modelrow_agrees_3958 : modelRow 3958 = row 3958 := model_count_3958.trans row_count_3958.symm
def row_3959 : RowCertificate 3959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3959 1) := by change ¬ Nat.Prime 62702643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20900881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3959 2) := by change ¬ Nat.Prime 62694725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12538945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3959 3) := by change ¬ Nat.Prime 62686807; exact PrattCertificates.composite_of_product (factor := 4639) (cofactor := 13513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3959 : row 3959 = 0 := (row_3959.checked).trans rfl
theorem model_count_3959 : modelRow 3959 = 0 := by decide
theorem modelrow_agrees_3959 : modelRow 3959 = row 3959 := model_count_3959.trans row_count_3959.symm
def row_3960 : RowCertificate 3960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3960 1) := by change ¬ Nat.Prime 62734321; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4825717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3960 2) := by change ¬ Nat.Prime 62726401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1183517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3960 3) := by change ¬ Nat.Prime 62718481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8959783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3960 : row 3960 = 0 := (row_3960.checked).trans rfl
theorem model_count_3960 : modelRow 3960 = 0 := by decide
theorem modelrow_agrees_3960 : modelRow 3960 = row 3960 := model_count_3960.trans row_count_3960.symm
def row_3961 : RowCertificate 3961 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3961 1) := by change Nat.Prime 62766007; exact ProjectEulerSolutions.P58PrimeNodes.p_62766007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3961 2) := by change ¬ Nat.Prime 62758085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12551617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3961 3) := by change ¬ Nat.Prime 62750163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20916721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3961 : row 3961 = 1 := (row_3961.checked).trans rfl
theorem model_count_3961 : modelRow 3961 = 1 := by decide
theorem modelrow_agrees_3961 : modelRow 3961 = row 3961 := model_count_3961.trans row_count_3961.symm
def row_3962 : RowCertificate 3962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3962 1) := by change ¬ Nat.Prime 62797701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20932567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3962 2) := by change ¬ Nat.Prime 62789777; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1697021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3962 3) := by change ¬ Nat.Prime 62781853; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 794707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3962 : row 3962 = 0 := (row_3962.checked).trans rfl
theorem model_count_3962 : modelRow 3962 = 0 := by decide
theorem modelrow_agrees_3962 : modelRow 3962 = row 3962 := model_count_3962.trans row_count_3962.symm
def row_3963 : RowCertificate 3963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3963 1) := by change ¬ Nat.Prime 62829403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8975629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3963 2) := by change ¬ Nat.Prime 62821477; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3695381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3963 3) := by change Nat.Prime 62813551; exact ProjectEulerSolutions.P58PrimeNodes.p_62813551.prime
    exact decide_eq_true prime
theorem row_count_3963 : row 3963 = 1 := (row_3963.checked).trans rfl
theorem model_count_3963 : modelRow 3963 = 1 := by decide
theorem modelrow_agrees_3963 : modelRow 3963 = row 3963 := model_count_3963.trans row_count_3963.symm
def row_3964 : RowCertificate 3964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3964 1) := by change ¬ Nat.Prime 62861113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8980159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3964 2) := by change ¬ Nat.Prime 62853185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12570637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3964 3) := by change ¬ Nat.Prime 62845257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20948419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3964 : row 3964 = 0 := (row_3964.checked).trans rfl
theorem model_count_3964 : modelRow 3964 = 0 := by decide
theorem modelrow_agrees_3964 : modelRow 3964 = row 3964 := model_count_3964.trans row_count_3964.symm
def row_3965 : RowCertificate 3965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3965 1) := by change ¬ Nat.Prime 62892831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20964277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3965 2) := by change ¬ Nat.Prime 62884901; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 861437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3965 3) := by change Nat.Prime 62876971; exact ProjectEulerSolutions.P58PrimeNodes.p_62876971.prime
    exact decide_eq_true prime
theorem row_count_3965 : row 3965 = 1 := (row_3965.checked).trans rfl
theorem model_count_3965 : modelRow 3965 = 1 := by decide
theorem modelrow_agrees_3965 : modelRow 3965 = row 3965 := model_count_3965.trans row_count_3965.symm
def row_3966 : RowCertificate 3966 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3966 1) := by change Nat.Prime 62924557; exact ProjectEulerSolutions.P58PrimeNodes.p_62924557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3966 2) := by change ¬ Nat.Prime 62916625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12583325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3966 3) := by change ¬ Nat.Prime 62908693; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 45553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3966 : row 3966 = 1 := (row_3966.checked).trans rfl
theorem model_count_3966 : modelRow 3966 = 1 := by decide
theorem modelrow_agrees_3966 : modelRow 3966 = row 3966 := model_count_3966.trans row_count_3966.symm
def row_3967 : RowCertificate 3967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3967 1) := by change ¬ Nat.Prime 62956291; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3313489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3967 2) := by change ¬ Nat.Prime 62948357; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2170633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3967 3) := by change ¬ Nat.Prime 62940423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20980141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3967 : row 3967 = 0 := (row_3967.checked).trans rfl
theorem model_count_3967 : modelRow 3967 = 0 := by decide
theorem modelrow_agrees_3967 : modelRow 3967 = row 3967 := model_count_3967.trans row_count_3967.symm
def row_3968 : RowCertificate 3968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3968 1) := by change ¬ Nat.Prime 62988033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 20996011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3968 2) := by change ¬ Nat.Prime 62980097; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 123733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3968 3) := by change ¬ Nat.Prime 62972161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8996023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3968 : row 3968 = 0 := (row_3968.checked).trans rfl
theorem model_count_3968 : modelRow 3968 = 0 := by decide
theorem modelrow_agrees_3968 : modelRow 3968 = row 3968 := model_count_3968.trans row_count_3968.symm
private def atomic_3841 : CountSegment 1 10 13120 3840 3841 1783 1783 := CountSegment.single (first := 3840) (initial := 1783) row_count_3841 (by decide)
private theorem atomic_3841_agreement : RowAgreement 3840 3841 := RowAgreement.single (first := 3840) modelrow_agrees_3841
private def atomic_3842 : CountSegment 1 10 13120 3841 3842 1783 1783 := CountSegment.single (first := 3841) (initial := 1783) row_count_3842 (by decide)
private theorem atomic_3842_agreement : RowAgreement 3841 3842 := RowAgreement.single (first := 3841) modelrow_agrees_3842
private def atomic_3843 : CountSegment 1 10 13120 3842 3843 1783 1783 := CountSegment.single (first := 3842) (initial := 1783) row_count_3843 (by decide)
private theorem atomic_3843_agreement : RowAgreement 3842 3843 := RowAgreement.single (first := 3842) modelrow_agrees_3843
private def atomic_3844 : CountSegment 1 10 13120 3843 3844 1783 1783 := CountSegment.single (first := 3843) (initial := 1783) row_count_3844 (by decide)
private theorem atomic_3844_agreement : RowAgreement 3843 3844 := RowAgreement.single (first := 3843) modelrow_agrees_3844
private def atomic_3845 : CountSegment 1 10 13120 3844 3845 1783 1783 := CountSegment.single (first := 3844) (initial := 1783) row_count_3845 (by decide)
private theorem atomic_3845_agreement : RowAgreement 3844 3845 := RowAgreement.single (first := 3844) modelrow_agrees_3845
private def atomic_3846 : CountSegment 1 10 13120 3845 3846 1783 1783 := CountSegment.single (first := 3845) (initial := 1783) row_count_3846 (by decide)
private theorem atomic_3846_agreement : RowAgreement 3845 3846 := RowAgreement.single (first := 3845) modelrow_agrees_3846
private def atomic_3847 : CountSegment 1 10 13120 3846 3847 1783 1784 := CountSegment.single (first := 3846) (initial := 1783) row_count_3847 (by decide)
private theorem atomic_3847_agreement : RowAgreement 3846 3847 := RowAgreement.single (first := 3846) modelrow_agrees_3847
private def atomic_3848 : CountSegment 1 10 13120 3847 3848 1784 1784 := CountSegment.single (first := 3847) (initial := 1784) row_count_3848 (by decide)
private theorem atomic_3848_agreement : RowAgreement 3847 3848 := RowAgreement.single (first := 3847) modelrow_agrees_3848
private def atomic_3849 : CountSegment 1 10 13120 3848 3849 1784 1784 := CountSegment.single (first := 3848) (initial := 1784) row_count_3849 (by decide)
private theorem atomic_3849_agreement : RowAgreement 3848 3849 := RowAgreement.single (first := 3848) modelrow_agrees_3849
private def atomic_3850 : CountSegment 1 10 13120 3849 3850 1784 1785 := CountSegment.single (first := 3849) (initial := 1784) row_count_3850 (by decide)
private theorem atomic_3850_agreement : RowAgreement 3849 3850 := RowAgreement.single (first := 3849) modelrow_agrees_3850
private def atomic_3851 : CountSegment 1 10 13120 3850 3851 1785 1786 := CountSegment.single (first := 3850) (initial := 1785) row_count_3851 (by decide)
private theorem atomic_3851_agreement : RowAgreement 3850 3851 := RowAgreement.single (first := 3850) modelrow_agrees_3851
private def atomic_3852 : CountSegment 1 10 13120 3851 3852 1786 1786 := CountSegment.single (first := 3851) (initial := 1786) row_count_3852 (by decide)
private theorem atomic_3852_agreement : RowAgreement 3851 3852 := RowAgreement.single (first := 3851) modelrow_agrees_3852
private def atomic_3853 : CountSegment 1 10 13120 3852 3853 1786 1786 := CountSegment.single (first := 3852) (initial := 1786) row_count_3853 (by decide)
private theorem atomic_3853_agreement : RowAgreement 3852 3853 := RowAgreement.single (first := 3852) modelrow_agrees_3853
private def atomic_3854 : CountSegment 1 10 13120 3853 3854 1786 1787 := CountSegment.single (first := 3853) (initial := 1786) row_count_3854 (by decide)
private theorem atomic_3854_agreement : RowAgreement 3853 3854 := RowAgreement.single (first := 3853) modelrow_agrees_3854
private def atomic_3855 : CountSegment 1 10 13120 3854 3855 1787 1787 := CountSegment.single (first := 3854) (initial := 1787) row_count_3855 (by decide)
private theorem atomic_3855_agreement : RowAgreement 3854 3855 := RowAgreement.single (first := 3854) modelrow_agrees_3855
private def atomic_3856 : CountSegment 1 10 13120 3855 3856 1787 1787 := CountSegment.single (first := 3855) (initial := 1787) row_count_3856 (by decide)
private theorem atomic_3856_agreement : RowAgreement 3855 3856 := RowAgreement.single (first := 3855) modelrow_agrees_3856
private def atomic_3857 : CountSegment 1 10 13120 3856 3857 1787 1787 := CountSegment.single (first := 3856) (initial := 1787) row_count_3857 (by decide)
private theorem atomic_3857_agreement : RowAgreement 3856 3857 := RowAgreement.single (first := 3856) modelrow_agrees_3857
private def atomic_3858 : CountSegment 1 10 13120 3857 3858 1787 1788 := CountSegment.single (first := 3857) (initial := 1787) row_count_3858 (by decide)
private theorem atomic_3858_agreement : RowAgreement 3857 3858 := RowAgreement.single (first := 3857) modelrow_agrees_3858
private def atomic_3859 : CountSegment 1 10 13120 3858 3859 1788 1788 := CountSegment.single (first := 3858) (initial := 1788) row_count_3859 (by decide)
private theorem atomic_3859_agreement : RowAgreement 3858 3859 := RowAgreement.single (first := 3858) modelrow_agrees_3859
private def atomic_3860 : CountSegment 1 10 13120 3859 3860 1788 1789 := CountSegment.single (first := 3859) (initial := 1788) row_count_3860 (by decide)
private theorem atomic_3860_agreement : RowAgreement 3859 3860 := RowAgreement.single (first := 3859) modelrow_agrees_3860
private def atomic_3861 : CountSegment 1 10 13120 3860 3861 1789 1789 := CountSegment.single (first := 3860) (initial := 1789) row_count_3861 (by decide)
private theorem atomic_3861_agreement : RowAgreement 3860 3861 := RowAgreement.single (first := 3860) modelrow_agrees_3861
private def atomic_3862 : CountSegment 1 10 13120 3861 3862 1789 1789 := CountSegment.single (first := 3861) (initial := 1789) row_count_3862 (by decide)
private theorem atomic_3862_agreement : RowAgreement 3861 3862 := RowAgreement.single (first := 3861) modelrow_agrees_3862
private def atomic_3863 : CountSegment 1 10 13120 3862 3863 1789 1789 := CountSegment.single (first := 3862) (initial := 1789) row_count_3863 (by decide)
private theorem atomic_3863_agreement : RowAgreement 3862 3863 := RowAgreement.single (first := 3862) modelrow_agrees_3863
private def atomic_3864 : CountSegment 1 10 13120 3863 3864 1789 1790 := CountSegment.single (first := 3863) (initial := 1789) row_count_3864 (by decide)
private theorem atomic_3864_agreement : RowAgreement 3863 3864 := RowAgreement.single (first := 3863) modelrow_agrees_3864
private def atomic_3865 : CountSegment 1 10 13120 3864 3865 1790 1790 := CountSegment.single (first := 3864) (initial := 1790) row_count_3865 (by decide)
private theorem atomic_3865_agreement : RowAgreement 3864 3865 := RowAgreement.single (first := 3864) modelrow_agrees_3865
private def atomic_3866 : CountSegment 1 10 13120 3865 3866 1790 1790 := CountSegment.single (first := 3865) (initial := 1790) row_count_3866 (by decide)
private theorem atomic_3866_agreement : RowAgreement 3865 3866 := RowAgreement.single (first := 3865) modelrow_agrees_3866
private def atomic_3867 : CountSegment 1 10 13120 3866 3867 1790 1791 := CountSegment.single (first := 3866) (initial := 1790) row_count_3867 (by decide)
private theorem atomic_3867_agreement : RowAgreement 3866 3867 := RowAgreement.single (first := 3866) modelrow_agrees_3867
private def atomic_3868 : CountSegment 1 10 13120 3867 3868 1791 1792 := CountSegment.single (first := 3867) (initial := 1791) row_count_3868 (by decide)
private theorem atomic_3868_agreement : RowAgreement 3867 3868 := RowAgreement.single (first := 3867) modelrow_agrees_3868
private def atomic_3869 : CountSegment 1 10 13120 3868 3869 1792 1792 := CountSegment.single (first := 3868) (initial := 1792) row_count_3869 (by decide)
private theorem atomic_3869_agreement : RowAgreement 3868 3869 := RowAgreement.single (first := 3868) modelrow_agrees_3869
private def atomic_3870 : CountSegment 1 10 13120 3869 3870 1792 1792 := CountSegment.single (first := 3869) (initial := 1792) row_count_3870 (by decide)
private theorem atomic_3870_agreement : RowAgreement 3869 3870 := RowAgreement.single (first := 3869) modelrow_agrees_3870
private def atomic_3871 : CountSegment 1 10 13120 3870 3871 1792 1792 := CountSegment.single (first := 3870) (initial := 1792) row_count_3871 (by decide)
private theorem atomic_3871_agreement : RowAgreement 3870 3871 := RowAgreement.single (first := 3870) modelrow_agrees_3871
private def atomic_3872 : CountSegment 1 10 13120 3871 3872 1792 1793 := CountSegment.single (first := 3871) (initial := 1792) row_count_3872 (by decide)
private theorem atomic_3872_agreement : RowAgreement 3871 3872 := RowAgreement.single (first := 3871) modelrow_agrees_3872
private def atomic_3873 : CountSegment 1 10 13120 3872 3873 1793 1794 := CountSegment.single (first := 3872) (initial := 1793) row_count_3873 (by decide)
private theorem atomic_3873_agreement : RowAgreement 3872 3873 := RowAgreement.single (first := 3872) modelrow_agrees_3873
private def atomic_3874 : CountSegment 1 10 13120 3873 3874 1794 1795 := CountSegment.single (first := 3873) (initial := 1794) row_count_3874 (by decide)
private theorem atomic_3874_agreement : RowAgreement 3873 3874 := RowAgreement.single (first := 3873) modelrow_agrees_3874
private def atomic_3875 : CountSegment 1 10 13120 3874 3875 1795 1795 := CountSegment.single (first := 3874) (initial := 1795) row_count_3875 (by decide)
private theorem atomic_3875_agreement : RowAgreement 3874 3875 := RowAgreement.single (first := 3874) modelrow_agrees_3875
private def atomic_3876 : CountSegment 1 10 13120 3875 3876 1795 1796 := CountSegment.single (first := 3875) (initial := 1795) row_count_3876 (by decide)
private theorem atomic_3876_agreement : RowAgreement 3875 3876 := RowAgreement.single (first := 3875) modelrow_agrees_3876
private def atomic_3877 : CountSegment 1 10 13120 3876 3877 1796 1798 := CountSegment.single (first := 3876) (initial := 1796) row_count_3877 (by decide)
private theorem atomic_3877_agreement : RowAgreement 3876 3877 := RowAgreement.single (first := 3876) modelrow_agrees_3877
private def atomic_3878 : CountSegment 1 10 13120 3877 3878 1798 1798 := CountSegment.single (first := 3877) (initial := 1798) row_count_3878 (by decide)
private theorem atomic_3878_agreement : RowAgreement 3877 3878 := RowAgreement.single (first := 3877) modelrow_agrees_3878
private def atomic_3879 : CountSegment 1 10 13120 3878 3879 1798 1798 := CountSegment.single (first := 3878) (initial := 1798) row_count_3879 (by decide)
private theorem atomic_3879_agreement : RowAgreement 3878 3879 := RowAgreement.single (first := 3878) modelrow_agrees_3879
private def atomic_3880 : CountSegment 1 10 13120 3879 3880 1798 1798 := CountSegment.single (first := 3879) (initial := 1798) row_count_3880 (by decide)
private theorem atomic_3880_agreement : RowAgreement 3879 3880 := RowAgreement.single (first := 3879) modelrow_agrees_3880
private def atomic_3881 : CountSegment 1 10 13120 3880 3881 1798 1799 := CountSegment.single (first := 3880) (initial := 1798) row_count_3881 (by decide)
private theorem atomic_3881_agreement : RowAgreement 3880 3881 := RowAgreement.single (first := 3880) modelrow_agrees_3881
private def atomic_3882 : CountSegment 1 10 13120 3881 3882 1799 1799 := CountSegment.single (first := 3881) (initial := 1799) row_count_3882 (by decide)
private theorem atomic_3882_agreement : RowAgreement 3881 3882 := RowAgreement.single (first := 3881) modelrow_agrees_3882
private def atomic_3883 : CountSegment 1 10 13120 3882 3883 1799 1799 := CountSegment.single (first := 3882) (initial := 1799) row_count_3883 (by decide)
private theorem atomic_3883_agreement : RowAgreement 3882 3883 := RowAgreement.single (first := 3882) modelrow_agrees_3883
private def atomic_3884 : CountSegment 1 10 13120 3883 3884 1799 1799 := CountSegment.single (first := 3883) (initial := 1799) row_count_3884 (by decide)
private theorem atomic_3884_agreement : RowAgreement 3883 3884 := RowAgreement.single (first := 3883) modelrow_agrees_3884
private def atomic_3885 : CountSegment 1 10 13120 3884 3885 1799 1801 := CountSegment.single (first := 3884) (initial := 1799) row_count_3885 (by decide)
private theorem atomic_3885_agreement : RowAgreement 3884 3885 := RowAgreement.single (first := 3884) modelrow_agrees_3885
private def atomic_3886 : CountSegment 1 10 13120 3885 3886 1801 1801 := CountSegment.single (first := 3885) (initial := 1801) row_count_3886 (by decide)
private theorem atomic_3886_agreement : RowAgreement 3885 3886 := RowAgreement.single (first := 3885) modelrow_agrees_3886
private def atomic_3887 : CountSegment 1 10 13120 3886 3887 1801 1802 := CountSegment.single (first := 3886) (initial := 1801) row_count_3887 (by decide)
private theorem atomic_3887_agreement : RowAgreement 3886 3887 := RowAgreement.single (first := 3886) modelrow_agrees_3887
private def atomic_3888 : CountSegment 1 10 13120 3887 3888 1802 1802 := CountSegment.single (first := 3887) (initial := 1802) row_count_3888 (by decide)
private theorem atomic_3888_agreement : RowAgreement 3887 3888 := RowAgreement.single (first := 3887) modelrow_agrees_3888
private def atomic_3889 : CountSegment 1 10 13120 3888 3889 1802 1802 := CountSegment.single (first := 3888) (initial := 1802) row_count_3889 (by decide)
private theorem atomic_3889_agreement : RowAgreement 3888 3889 := RowAgreement.single (first := 3888) modelrow_agrees_3889
private def atomic_3890 : CountSegment 1 10 13120 3889 3890 1802 1803 := CountSegment.single (first := 3889) (initial := 1802) row_count_3890 (by decide)
private theorem atomic_3890_agreement : RowAgreement 3889 3890 := RowAgreement.single (first := 3889) modelrow_agrees_3890
private def atomic_3891 : CountSegment 1 10 13120 3890 3891 1803 1803 := CountSegment.single (first := 3890) (initial := 1803) row_count_3891 (by decide)
private theorem atomic_3891_agreement : RowAgreement 3890 3891 := RowAgreement.single (first := 3890) modelrow_agrees_3891
private def atomic_3892 : CountSegment 1 10 13120 3891 3892 1803 1803 := CountSegment.single (first := 3891) (initial := 1803) row_count_3892 (by decide)
private theorem atomic_3892_agreement : RowAgreement 3891 3892 := RowAgreement.single (first := 3891) modelrow_agrees_3892
private def atomic_3893 : CountSegment 1 10 13120 3892 3893 1803 1804 := CountSegment.single (first := 3892) (initial := 1803) row_count_3893 (by decide)
private theorem atomic_3893_agreement : RowAgreement 3892 3893 := RowAgreement.single (first := 3892) modelrow_agrees_3893
private def atomic_3894 : CountSegment 1 10 13120 3893 3894 1804 1805 := CountSegment.single (first := 3893) (initial := 1804) row_count_3894 (by decide)
private theorem atomic_3894_agreement : RowAgreement 3893 3894 := RowAgreement.single (first := 3893) modelrow_agrees_3894
private def atomic_3895 : CountSegment 1 10 13120 3894 3895 1805 1805 := CountSegment.single (first := 3894) (initial := 1805) row_count_3895 (by decide)
private theorem atomic_3895_agreement : RowAgreement 3894 3895 := RowAgreement.single (first := 3894) modelrow_agrees_3895
private def atomic_3896 : CountSegment 1 10 13120 3895 3896 1805 1806 := CountSegment.single (first := 3895) (initial := 1805) row_count_3896 (by decide)
private theorem atomic_3896_agreement : RowAgreement 3895 3896 := RowAgreement.single (first := 3895) modelrow_agrees_3896
private def atomic_3897 : CountSegment 1 10 13120 3896 3897 1806 1806 := CountSegment.single (first := 3896) (initial := 1806) row_count_3897 (by decide)
private theorem atomic_3897_agreement : RowAgreement 3896 3897 := RowAgreement.single (first := 3896) modelrow_agrees_3897
private def atomic_3898 : CountSegment 1 10 13120 3897 3898 1806 1807 := CountSegment.single (first := 3897) (initial := 1806) row_count_3898 (by decide)
private theorem atomic_3898_agreement : RowAgreement 3897 3898 := RowAgreement.single (first := 3897) modelrow_agrees_3898
private def atomic_3899 : CountSegment 1 10 13120 3898 3899 1807 1807 := CountSegment.single (first := 3898) (initial := 1807) row_count_3899 (by decide)
private theorem atomic_3899_agreement : RowAgreement 3898 3899 := RowAgreement.single (first := 3898) modelrow_agrees_3899
private def atomic_3900 : CountSegment 1 10 13120 3899 3900 1807 1807 := CountSegment.single (first := 3899) (initial := 1807) row_count_3900 (by decide)
private theorem atomic_3900_agreement : RowAgreement 3899 3900 := RowAgreement.single (first := 3899) modelrow_agrees_3900
private def atomic_3901 : CountSegment 1 10 13120 3900 3901 1807 1807 := CountSegment.single (first := 3900) (initial := 1807) row_count_3901 (by decide)
private theorem atomic_3901_agreement : RowAgreement 3900 3901 := RowAgreement.single (first := 3900) modelrow_agrees_3901
private def atomic_3902 : CountSegment 1 10 13120 3901 3902 1807 1808 := CountSegment.single (first := 3901) (initial := 1807) row_count_3902 (by decide)
private theorem atomic_3902_agreement : RowAgreement 3901 3902 := RowAgreement.single (first := 3901) modelrow_agrees_3902
private def atomic_3903 : CountSegment 1 10 13120 3902 3903 1808 1809 := CountSegment.single (first := 3902) (initial := 1808) row_count_3903 (by decide)
private theorem atomic_3903_agreement : RowAgreement 3902 3903 := RowAgreement.single (first := 3902) modelrow_agrees_3903
private def atomic_3904 : CountSegment 1 10 13120 3903 3904 1809 1809 := CountSegment.single (first := 3903) (initial := 1809) row_count_3904 (by decide)
private theorem atomic_3904_agreement : RowAgreement 3903 3904 := RowAgreement.single (first := 3903) modelrow_agrees_3904
private def atomic_3905 : CountSegment 1 10 13120 3904 3905 1809 1810 := CountSegment.single (first := 3904) (initial := 1809) row_count_3905 (by decide)
private theorem atomic_3905_agreement : RowAgreement 3904 3905 := RowAgreement.single (first := 3904) modelrow_agrees_3905
private def atomic_3906 : CountSegment 1 10 13120 3905 3906 1810 1811 := CountSegment.single (first := 3905) (initial := 1810) row_count_3906 (by decide)
private theorem atomic_3906_agreement : RowAgreement 3905 3906 := RowAgreement.single (first := 3905) modelrow_agrees_3906
private def atomic_3907 : CountSegment 1 10 13120 3906 3907 1811 1811 := CountSegment.single (first := 3906) (initial := 1811) row_count_3907 (by decide)
private theorem atomic_3907_agreement : RowAgreement 3906 3907 := RowAgreement.single (first := 3906) modelrow_agrees_3907
private def atomic_3908 : CountSegment 1 10 13120 3907 3908 1811 1811 := CountSegment.single (first := 3907) (initial := 1811) row_count_3908 (by decide)
private theorem atomic_3908_agreement : RowAgreement 3907 3908 := RowAgreement.single (first := 3907) modelrow_agrees_3908
private def atomic_3909 : CountSegment 1 10 13120 3908 3909 1811 1811 := CountSegment.single (first := 3908) (initial := 1811) row_count_3909 (by decide)
private theorem atomic_3909_agreement : RowAgreement 3908 3909 := RowAgreement.single (first := 3908) modelrow_agrees_3909
private def atomic_3910 : CountSegment 1 10 13120 3909 3910 1811 1812 := CountSegment.single (first := 3909) (initial := 1811) row_count_3910 (by decide)
private theorem atomic_3910_agreement : RowAgreement 3909 3910 := RowAgreement.single (first := 3909) modelrow_agrees_3910
private def atomic_3911 : CountSegment 1 10 13120 3910 3911 1812 1812 := CountSegment.single (first := 3910) (initial := 1812) row_count_3911 (by decide)
private theorem atomic_3911_agreement : RowAgreement 3910 3911 := RowAgreement.single (first := 3910) modelrow_agrees_3911
private def atomic_3912 : CountSegment 1 10 13120 3911 3912 1812 1812 := CountSegment.single (first := 3911) (initial := 1812) row_count_3912 (by decide)
private theorem atomic_3912_agreement : RowAgreement 3911 3912 := RowAgreement.single (first := 3911) modelrow_agrees_3912
private def atomic_3913 : CountSegment 1 10 13120 3912 3913 1812 1812 := CountSegment.single (first := 3912) (initial := 1812) row_count_3913 (by decide)
private theorem atomic_3913_agreement : RowAgreement 3912 3913 := RowAgreement.single (first := 3912) modelrow_agrees_3913
private def atomic_3914 : CountSegment 1 10 13120 3913 3914 1812 1812 := CountSegment.single (first := 3913) (initial := 1812) row_count_3914 (by decide)
private theorem atomic_3914_agreement : RowAgreement 3913 3914 := RowAgreement.single (first := 3913) modelrow_agrees_3914
private def atomic_3915 : CountSegment 1 10 13120 3914 3915 1812 1812 := CountSegment.single (first := 3914) (initial := 1812) row_count_3915 (by decide)
private theorem atomic_3915_agreement : RowAgreement 3914 3915 := RowAgreement.single (first := 3914) modelrow_agrees_3915
private def atomic_3916 : CountSegment 1 10 13120 3915 3916 1812 1812 := CountSegment.single (first := 3915) (initial := 1812) row_count_3916 (by decide)
private theorem atomic_3916_agreement : RowAgreement 3915 3916 := RowAgreement.single (first := 3915) modelrow_agrees_3916
private def atomic_3917 : CountSegment 1 10 13120 3916 3917 1812 1812 := CountSegment.single (first := 3916) (initial := 1812) row_count_3917 (by decide)
private theorem atomic_3917_agreement : RowAgreement 3916 3917 := RowAgreement.single (first := 3916) modelrow_agrees_3917
private def atomic_3918 : CountSegment 1 10 13120 3917 3918 1812 1813 := CountSegment.single (first := 3917) (initial := 1812) row_count_3918 (by decide)
private theorem atomic_3918_agreement : RowAgreement 3917 3918 := RowAgreement.single (first := 3917) modelrow_agrees_3918
private def atomic_3919 : CountSegment 1 10 13120 3918 3919 1813 1813 := CountSegment.single (first := 3918) (initial := 1813) row_count_3919 (by decide)
private theorem atomic_3919_agreement : RowAgreement 3918 3919 := RowAgreement.single (first := 3918) modelrow_agrees_3919
private def atomic_3920 : CountSegment 1 10 13120 3919 3920 1813 1813 := CountSegment.single (first := 3919) (initial := 1813) row_count_3920 (by decide)
private theorem atomic_3920_agreement : RowAgreement 3919 3920 := RowAgreement.single (first := 3919) modelrow_agrees_3920
private def atomic_3921 : CountSegment 1 10 13120 3920 3921 1813 1813 := CountSegment.single (first := 3920) (initial := 1813) row_count_3921 (by decide)
private theorem atomic_3921_agreement : RowAgreement 3920 3921 := RowAgreement.single (first := 3920) modelrow_agrees_3921
private def atomic_3922 : CountSegment 1 10 13120 3921 3922 1813 1813 := CountSegment.single (first := 3921) (initial := 1813) row_count_3922 (by decide)
private theorem atomic_3922_agreement : RowAgreement 3921 3922 := RowAgreement.single (first := 3921) modelrow_agrees_3922
private def atomic_3923 : CountSegment 1 10 13120 3922 3923 1813 1814 := CountSegment.single (first := 3922) (initial := 1813) row_count_3923 (by decide)
private theorem atomic_3923_agreement : RowAgreement 3922 3923 := RowAgreement.single (first := 3922) modelrow_agrees_3923
private def atomic_3924 : CountSegment 1 10 13120 3923 3924 1814 1815 := CountSegment.single (first := 3923) (initial := 1814) row_count_3924 (by decide)
private theorem atomic_3924_agreement : RowAgreement 3923 3924 := RowAgreement.single (first := 3923) modelrow_agrees_3924
private def atomic_3925 : CountSegment 1 10 13120 3924 3925 1815 1816 := CountSegment.single (first := 3924) (initial := 1815) row_count_3925 (by decide)
private theorem atomic_3925_agreement : RowAgreement 3924 3925 := RowAgreement.single (first := 3924) modelrow_agrees_3925
private def atomic_3926 : CountSegment 1 10 13120 3925 3926 1816 1816 := CountSegment.single (first := 3925) (initial := 1816) row_count_3926 (by decide)
private theorem atomic_3926_agreement : RowAgreement 3925 3926 := RowAgreement.single (first := 3925) modelrow_agrees_3926
private def atomic_3927 : CountSegment 1 10 13120 3926 3927 1816 1817 := CountSegment.single (first := 3926) (initial := 1816) row_count_3927 (by decide)
private theorem atomic_3927_agreement : RowAgreement 3926 3927 := RowAgreement.single (first := 3926) modelrow_agrees_3927
private def atomic_3928 : CountSegment 1 10 13120 3927 3928 1817 1818 := CountSegment.single (first := 3927) (initial := 1817) row_count_3928 (by decide)
private theorem atomic_3928_agreement : RowAgreement 3927 3928 := RowAgreement.single (first := 3927) modelrow_agrees_3928
private def atomic_3929 : CountSegment 1 10 13120 3928 3929 1818 1819 := CountSegment.single (first := 3928) (initial := 1818) row_count_3929 (by decide)
private theorem atomic_3929_agreement : RowAgreement 3928 3929 := RowAgreement.single (first := 3928) modelrow_agrees_3929
private def atomic_3930 : CountSegment 1 10 13120 3929 3930 1819 1821 := CountSegment.single (first := 3929) (initial := 1819) row_count_3930 (by decide)
private theorem atomic_3930_agreement : RowAgreement 3929 3930 := RowAgreement.single (first := 3929) modelrow_agrees_3930
private def atomic_3931 : CountSegment 1 10 13120 3930 3931 1821 1821 := CountSegment.single (first := 3930) (initial := 1821) row_count_3931 (by decide)
private theorem atomic_3931_agreement : RowAgreement 3930 3931 := RowAgreement.single (first := 3930) modelrow_agrees_3931
private def atomic_3932 : CountSegment 1 10 13120 3931 3932 1821 1822 := CountSegment.single (first := 3931) (initial := 1821) row_count_3932 (by decide)
private theorem atomic_3932_agreement : RowAgreement 3931 3932 := RowAgreement.single (first := 3931) modelrow_agrees_3932
private def atomic_3933 : CountSegment 1 10 13120 3932 3933 1822 1822 := CountSegment.single (first := 3932) (initial := 1822) row_count_3933 (by decide)
private theorem atomic_3933_agreement : RowAgreement 3932 3933 := RowAgreement.single (first := 3932) modelrow_agrees_3933
private def atomic_3934 : CountSegment 1 10 13120 3933 3934 1822 1822 := CountSegment.single (first := 3933) (initial := 1822) row_count_3934 (by decide)
private theorem atomic_3934_agreement : RowAgreement 3933 3934 := RowAgreement.single (first := 3933) modelrow_agrees_3934
private def atomic_3935 : CountSegment 1 10 13120 3934 3935 1822 1823 := CountSegment.single (first := 3934) (initial := 1822) row_count_3935 (by decide)
private theorem atomic_3935_agreement : RowAgreement 3934 3935 := RowAgreement.single (first := 3934) modelrow_agrees_3935
private def atomic_3936 : CountSegment 1 10 13120 3935 3936 1823 1824 := CountSegment.single (first := 3935) (initial := 1823) row_count_3936 (by decide)
private theorem atomic_3936_agreement : RowAgreement 3935 3936 := RowAgreement.single (first := 3935) modelrow_agrees_3936
private def atomic_3937 : CountSegment 1 10 13120 3936 3937 1824 1824 := CountSegment.single (first := 3936) (initial := 1824) row_count_3937 (by decide)
private theorem atomic_3937_agreement : RowAgreement 3936 3937 := RowAgreement.single (first := 3936) modelrow_agrees_3937
private def atomic_3938 : CountSegment 1 10 13120 3937 3938 1824 1824 := CountSegment.single (first := 3937) (initial := 1824) row_count_3938 (by decide)
private theorem atomic_3938_agreement : RowAgreement 3937 3938 := RowAgreement.single (first := 3937) modelrow_agrees_3938
private def atomic_3939 : CountSegment 1 10 13120 3938 3939 1824 1824 := CountSegment.single (first := 3938) (initial := 1824) row_count_3939 (by decide)
private theorem atomic_3939_agreement : RowAgreement 3938 3939 := RowAgreement.single (first := 3938) modelrow_agrees_3939
private def atomic_3940 : CountSegment 1 10 13120 3939 3940 1824 1824 := CountSegment.single (first := 3939) (initial := 1824) row_count_3940 (by decide)
private theorem atomic_3940_agreement : RowAgreement 3939 3940 := RowAgreement.single (first := 3939) modelrow_agrees_3940
private def atomic_3941 : CountSegment 1 10 13120 3940 3941 1824 1824 := CountSegment.single (first := 3940) (initial := 1824) row_count_3941 (by decide)
private theorem atomic_3941_agreement : RowAgreement 3940 3941 := RowAgreement.single (first := 3940) modelrow_agrees_3941
private def atomic_3942 : CountSegment 1 10 13120 3941 3942 1824 1825 := CountSegment.single (first := 3941) (initial := 1824) row_count_3942 (by decide)
private theorem atomic_3942_agreement : RowAgreement 3941 3942 := RowAgreement.single (first := 3941) modelrow_agrees_3942
private def atomic_3943 : CountSegment 1 10 13120 3942 3943 1825 1825 := CountSegment.single (first := 3942) (initial := 1825) row_count_3943 (by decide)
private theorem atomic_3943_agreement : RowAgreement 3942 3943 := RowAgreement.single (first := 3942) modelrow_agrees_3943
private def atomic_3944 : CountSegment 1 10 13120 3943 3944 1825 1825 := CountSegment.single (first := 3943) (initial := 1825) row_count_3944 (by decide)
private theorem atomic_3944_agreement : RowAgreement 3943 3944 := RowAgreement.single (first := 3943) modelrow_agrees_3944
private def atomic_3945 : CountSegment 1 10 13120 3944 3945 1825 1827 := CountSegment.single (first := 3944) (initial := 1825) row_count_3945 (by decide)
private theorem atomic_3945_agreement : RowAgreement 3944 3945 := RowAgreement.single (first := 3944) modelrow_agrees_3945
private def atomic_3946 : CountSegment 1 10 13120 3945 3946 1827 1827 := CountSegment.single (first := 3945) (initial := 1827) row_count_3946 (by decide)
private theorem atomic_3946_agreement : RowAgreement 3945 3946 := RowAgreement.single (first := 3945) modelrow_agrees_3946
private def atomic_3947 : CountSegment 1 10 13120 3946 3947 1827 1827 := CountSegment.single (first := 3946) (initial := 1827) row_count_3947 (by decide)
private theorem atomic_3947_agreement : RowAgreement 3946 3947 := RowAgreement.single (first := 3946) modelrow_agrees_3947
private def atomic_3948 : CountSegment 1 10 13120 3947 3948 1827 1828 := CountSegment.single (first := 3947) (initial := 1827) row_count_3948 (by decide)
private theorem atomic_3948_agreement : RowAgreement 3947 3948 := RowAgreement.single (first := 3947) modelrow_agrees_3948
private def atomic_3949 : CountSegment 1 10 13120 3948 3949 1828 1828 := CountSegment.single (first := 3948) (initial := 1828) row_count_3949 (by decide)
private theorem atomic_3949_agreement : RowAgreement 3948 3949 := RowAgreement.single (first := 3948) modelrow_agrees_3949
private def atomic_3950 : CountSegment 1 10 13120 3949 3950 1828 1828 := CountSegment.single (first := 3949) (initial := 1828) row_count_3950 (by decide)
private theorem atomic_3950_agreement : RowAgreement 3949 3950 := RowAgreement.single (first := 3949) modelrow_agrees_3950
private def atomic_3951 : CountSegment 1 10 13120 3950 3951 1828 1829 := CountSegment.single (first := 3950) (initial := 1828) row_count_3951 (by decide)
private theorem atomic_3951_agreement : RowAgreement 3950 3951 := RowAgreement.single (first := 3950) modelrow_agrees_3951
private def atomic_3952 : CountSegment 1 10 13120 3951 3952 1829 1829 := CountSegment.single (first := 3951) (initial := 1829) row_count_3952 (by decide)
private theorem atomic_3952_agreement : RowAgreement 3951 3952 := RowAgreement.single (first := 3951) modelrow_agrees_3952
private def atomic_3953 : CountSegment 1 10 13120 3952 3953 1829 1830 := CountSegment.single (first := 3952) (initial := 1829) row_count_3953 (by decide)
private theorem atomic_3953_agreement : RowAgreement 3952 3953 := RowAgreement.single (first := 3952) modelrow_agrees_3953
private def atomic_3954 : CountSegment 1 10 13120 3953 3954 1830 1831 := CountSegment.single (first := 3953) (initial := 1830) row_count_3954 (by decide)
private theorem atomic_3954_agreement : RowAgreement 3953 3954 := RowAgreement.single (first := 3953) modelrow_agrees_3954
private def atomic_3955 : CountSegment 1 10 13120 3954 3955 1831 1832 := CountSegment.single (first := 3954) (initial := 1831) row_count_3955 (by decide)
private theorem atomic_3955_agreement : RowAgreement 3954 3955 := RowAgreement.single (first := 3954) modelrow_agrees_3955
private def atomic_3956 : CountSegment 1 10 13120 3955 3956 1832 1832 := CountSegment.single (first := 3955) (initial := 1832) row_count_3956 (by decide)
private theorem atomic_3956_agreement : RowAgreement 3955 3956 := RowAgreement.single (first := 3955) modelrow_agrees_3956
private def atomic_3957 : CountSegment 1 10 13120 3956 3957 1832 1832 := CountSegment.single (first := 3956) (initial := 1832) row_count_3957 (by decide)
private theorem atomic_3957_agreement : RowAgreement 3956 3957 := RowAgreement.single (first := 3956) modelrow_agrees_3957
private def atomic_3958 : CountSegment 1 10 13120 3957 3958 1832 1832 := CountSegment.single (first := 3957) (initial := 1832) row_count_3958 (by decide)
private theorem atomic_3958_agreement : RowAgreement 3957 3958 := RowAgreement.single (first := 3957) modelrow_agrees_3958
private def atomic_3959 : CountSegment 1 10 13120 3958 3959 1832 1832 := CountSegment.single (first := 3958) (initial := 1832) row_count_3959 (by decide)
private theorem atomic_3959_agreement : RowAgreement 3958 3959 := RowAgreement.single (first := 3958) modelrow_agrees_3959
private def atomic_3960 : CountSegment 1 10 13120 3959 3960 1832 1832 := CountSegment.single (first := 3959) (initial := 1832) row_count_3960 (by decide)
private theorem atomic_3960_agreement : RowAgreement 3959 3960 := RowAgreement.single (first := 3959) modelrow_agrees_3960
private def atomic_3961 : CountSegment 1 10 13120 3960 3961 1832 1833 := CountSegment.single (first := 3960) (initial := 1832) row_count_3961 (by decide)
private theorem atomic_3961_agreement : RowAgreement 3960 3961 := RowAgreement.single (first := 3960) modelrow_agrees_3961
private def atomic_3962 : CountSegment 1 10 13120 3961 3962 1833 1833 := CountSegment.single (first := 3961) (initial := 1833) row_count_3962 (by decide)
private theorem atomic_3962_agreement : RowAgreement 3961 3962 := RowAgreement.single (first := 3961) modelrow_agrees_3962
private def atomic_3963 : CountSegment 1 10 13120 3962 3963 1833 1834 := CountSegment.single (first := 3962) (initial := 1833) row_count_3963 (by decide)
private theorem atomic_3963_agreement : RowAgreement 3962 3963 := RowAgreement.single (first := 3962) modelrow_agrees_3963
private def atomic_3964 : CountSegment 1 10 13120 3963 3964 1834 1834 := CountSegment.single (first := 3963) (initial := 1834) row_count_3964 (by decide)
private theorem atomic_3964_agreement : RowAgreement 3963 3964 := RowAgreement.single (first := 3963) modelrow_agrees_3964
private def atomic_3965 : CountSegment 1 10 13120 3964 3965 1834 1835 := CountSegment.single (first := 3964) (initial := 1834) row_count_3965 (by decide)
private theorem atomic_3965_agreement : RowAgreement 3964 3965 := RowAgreement.single (first := 3964) modelrow_agrees_3965
private def atomic_3966 : CountSegment 1 10 13120 3965 3966 1835 1836 := CountSegment.single (first := 3965) (initial := 1835) row_count_3966 (by decide)
private theorem atomic_3966_agreement : RowAgreement 3965 3966 := RowAgreement.single (first := 3965) modelrow_agrees_3966
private def atomic_3967 : CountSegment 1 10 13120 3966 3967 1836 1836 := CountSegment.single (first := 3966) (initial := 1836) row_count_3967 (by decide)
private theorem atomic_3967_agreement : RowAgreement 3966 3967 := RowAgreement.single (first := 3966) modelrow_agrees_3967
private def atomic_3968 : CountSegment 1 10 13120 3967 3968 1836 1836 := CountSegment.single (first := 3967) (initial := 1836) row_count_3968 (by decide)
private theorem atomic_3968_agreement : RowAgreement 3967 3968 := RowAgreement.single (first := 3967) modelrow_agrees_3968
private def within_030_000 : CountSegment 1 10 13120 3840 3842 1783 1783 := CountSegment.join atomic_3841 atomic_3842
private theorem within_030_000_agreement : RowAgreement 3840 3842 := RowAgreement.join atomic_3841_agreement atomic_3842_agreement
private def within_030_001 : CountSegment 1 10 13120 3842 3844 1783 1783 := CountSegment.join atomic_3843 atomic_3844
private theorem within_030_001_agreement : RowAgreement 3842 3844 := RowAgreement.join atomic_3843_agreement atomic_3844_agreement
private def within_030_002 : CountSegment 1 10 13120 3840 3844 1783 1783 := CountSegment.join within_030_000 within_030_001
private theorem within_030_002_agreement : RowAgreement 3840 3844 := RowAgreement.join within_030_000_agreement within_030_001_agreement
private def within_030_003 : CountSegment 1 10 13120 3844 3846 1783 1783 := CountSegment.join atomic_3845 atomic_3846
private theorem within_030_003_agreement : RowAgreement 3844 3846 := RowAgreement.join atomic_3845_agreement atomic_3846_agreement
private def within_030_004 : CountSegment 1 10 13120 3846 3848 1783 1784 := CountSegment.join atomic_3847 atomic_3848
private theorem within_030_004_agreement : RowAgreement 3846 3848 := RowAgreement.join atomic_3847_agreement atomic_3848_agreement
private def within_030_005 : CountSegment 1 10 13120 3844 3848 1783 1784 := CountSegment.join within_030_003 within_030_004
private theorem within_030_005_agreement : RowAgreement 3844 3848 := RowAgreement.join within_030_003_agreement within_030_004_agreement
private def within_030_006 : CountSegment 1 10 13120 3840 3848 1783 1784 := CountSegment.join within_030_002 within_030_005
private theorem within_030_006_agreement : RowAgreement 3840 3848 := RowAgreement.join within_030_002_agreement within_030_005_agreement
private def within_030_007 : CountSegment 1 10 13120 3848 3850 1784 1785 := CountSegment.join atomic_3849 atomic_3850
private theorem within_030_007_agreement : RowAgreement 3848 3850 := RowAgreement.join atomic_3849_agreement atomic_3850_agreement
private def within_030_008 : CountSegment 1 10 13120 3850 3852 1785 1786 := CountSegment.join atomic_3851 atomic_3852
private theorem within_030_008_agreement : RowAgreement 3850 3852 := RowAgreement.join atomic_3851_agreement atomic_3852_agreement
private def within_030_009 : CountSegment 1 10 13120 3848 3852 1784 1786 := CountSegment.join within_030_007 within_030_008
private theorem within_030_009_agreement : RowAgreement 3848 3852 := RowAgreement.join within_030_007_agreement within_030_008_agreement
private def within_030_010 : CountSegment 1 10 13120 3852 3854 1786 1787 := CountSegment.join atomic_3853 atomic_3854
private theorem within_030_010_agreement : RowAgreement 3852 3854 := RowAgreement.join atomic_3853_agreement atomic_3854_agreement
private def within_030_011 : CountSegment 1 10 13120 3854 3856 1787 1787 := CountSegment.join atomic_3855 atomic_3856
private theorem within_030_011_agreement : RowAgreement 3854 3856 := RowAgreement.join atomic_3855_agreement atomic_3856_agreement
private def within_030_012 : CountSegment 1 10 13120 3852 3856 1786 1787 := CountSegment.join within_030_010 within_030_011
private theorem within_030_012_agreement : RowAgreement 3852 3856 := RowAgreement.join within_030_010_agreement within_030_011_agreement
private def within_030_013 : CountSegment 1 10 13120 3848 3856 1784 1787 := CountSegment.join within_030_009 within_030_012
private theorem within_030_013_agreement : RowAgreement 3848 3856 := RowAgreement.join within_030_009_agreement within_030_012_agreement
private def within_030_014 : CountSegment 1 10 13120 3840 3856 1783 1787 := CountSegment.join within_030_006 within_030_013
private theorem within_030_014_agreement : RowAgreement 3840 3856 := RowAgreement.join within_030_006_agreement within_030_013_agreement
private def within_030_015 : CountSegment 1 10 13120 3856 3858 1787 1788 := CountSegment.join atomic_3857 atomic_3858
private theorem within_030_015_agreement : RowAgreement 3856 3858 := RowAgreement.join atomic_3857_agreement atomic_3858_agreement
private def within_030_016 : CountSegment 1 10 13120 3858 3860 1788 1789 := CountSegment.join atomic_3859 atomic_3860
private theorem within_030_016_agreement : RowAgreement 3858 3860 := RowAgreement.join atomic_3859_agreement atomic_3860_agreement
private def within_030_017 : CountSegment 1 10 13120 3856 3860 1787 1789 := CountSegment.join within_030_015 within_030_016
private theorem within_030_017_agreement : RowAgreement 3856 3860 := RowAgreement.join within_030_015_agreement within_030_016_agreement
private def within_030_018 : CountSegment 1 10 13120 3860 3862 1789 1789 := CountSegment.join atomic_3861 atomic_3862
private theorem within_030_018_agreement : RowAgreement 3860 3862 := RowAgreement.join atomic_3861_agreement atomic_3862_agreement
private def within_030_019 : CountSegment 1 10 13120 3862 3864 1789 1790 := CountSegment.join atomic_3863 atomic_3864
private theorem within_030_019_agreement : RowAgreement 3862 3864 := RowAgreement.join atomic_3863_agreement atomic_3864_agreement
private def within_030_020 : CountSegment 1 10 13120 3860 3864 1789 1790 := CountSegment.join within_030_018 within_030_019
private theorem within_030_020_agreement : RowAgreement 3860 3864 := RowAgreement.join within_030_018_agreement within_030_019_agreement
private def within_030_021 : CountSegment 1 10 13120 3856 3864 1787 1790 := CountSegment.join within_030_017 within_030_020
private theorem within_030_021_agreement : RowAgreement 3856 3864 := RowAgreement.join within_030_017_agreement within_030_020_agreement
private def within_030_022 : CountSegment 1 10 13120 3864 3866 1790 1790 := CountSegment.join atomic_3865 atomic_3866
private theorem within_030_022_agreement : RowAgreement 3864 3866 := RowAgreement.join atomic_3865_agreement atomic_3866_agreement
private def within_030_023 : CountSegment 1 10 13120 3866 3868 1790 1792 := CountSegment.join atomic_3867 atomic_3868
private theorem within_030_023_agreement : RowAgreement 3866 3868 := RowAgreement.join atomic_3867_agreement atomic_3868_agreement
private def within_030_024 : CountSegment 1 10 13120 3864 3868 1790 1792 := CountSegment.join within_030_022 within_030_023
private theorem within_030_024_agreement : RowAgreement 3864 3868 := RowAgreement.join within_030_022_agreement within_030_023_agreement
private def within_030_025 : CountSegment 1 10 13120 3868 3870 1792 1792 := CountSegment.join atomic_3869 atomic_3870
private theorem within_030_025_agreement : RowAgreement 3868 3870 := RowAgreement.join atomic_3869_agreement atomic_3870_agreement
private def within_030_026 : CountSegment 1 10 13120 3870 3872 1792 1793 := CountSegment.join atomic_3871 atomic_3872
private theorem within_030_026_agreement : RowAgreement 3870 3872 := RowAgreement.join atomic_3871_agreement atomic_3872_agreement
private def within_030_027 : CountSegment 1 10 13120 3868 3872 1792 1793 := CountSegment.join within_030_025 within_030_026
private theorem within_030_027_agreement : RowAgreement 3868 3872 := RowAgreement.join within_030_025_agreement within_030_026_agreement
private def within_030_028 : CountSegment 1 10 13120 3864 3872 1790 1793 := CountSegment.join within_030_024 within_030_027
private theorem within_030_028_agreement : RowAgreement 3864 3872 := RowAgreement.join within_030_024_agreement within_030_027_agreement
private def within_030_029 : CountSegment 1 10 13120 3856 3872 1787 1793 := CountSegment.join within_030_021 within_030_028
private theorem within_030_029_agreement : RowAgreement 3856 3872 := RowAgreement.join within_030_021_agreement within_030_028_agreement
private def within_030_030 : CountSegment 1 10 13120 3840 3872 1783 1793 := CountSegment.join within_030_014 within_030_029
private theorem within_030_030_agreement : RowAgreement 3840 3872 := RowAgreement.join within_030_014_agreement within_030_029_agreement
private def within_030_031 : CountSegment 1 10 13120 3872 3874 1793 1795 := CountSegment.join atomic_3873 atomic_3874
private theorem within_030_031_agreement : RowAgreement 3872 3874 := RowAgreement.join atomic_3873_agreement atomic_3874_agreement
private def within_030_032 : CountSegment 1 10 13120 3874 3876 1795 1796 := CountSegment.join atomic_3875 atomic_3876
private theorem within_030_032_agreement : RowAgreement 3874 3876 := RowAgreement.join atomic_3875_agreement atomic_3876_agreement
private def within_030_033 : CountSegment 1 10 13120 3872 3876 1793 1796 := CountSegment.join within_030_031 within_030_032
private theorem within_030_033_agreement : RowAgreement 3872 3876 := RowAgreement.join within_030_031_agreement within_030_032_agreement
private def within_030_034 : CountSegment 1 10 13120 3876 3878 1796 1798 := CountSegment.join atomic_3877 atomic_3878
private theorem within_030_034_agreement : RowAgreement 3876 3878 := RowAgreement.join atomic_3877_agreement atomic_3878_agreement
private def within_030_035 : CountSegment 1 10 13120 3878 3880 1798 1798 := CountSegment.join atomic_3879 atomic_3880
private theorem within_030_035_agreement : RowAgreement 3878 3880 := RowAgreement.join atomic_3879_agreement atomic_3880_agreement
private def within_030_036 : CountSegment 1 10 13120 3876 3880 1796 1798 := CountSegment.join within_030_034 within_030_035
private theorem within_030_036_agreement : RowAgreement 3876 3880 := RowAgreement.join within_030_034_agreement within_030_035_agreement
private def within_030_037 : CountSegment 1 10 13120 3872 3880 1793 1798 := CountSegment.join within_030_033 within_030_036
private theorem within_030_037_agreement : RowAgreement 3872 3880 := RowAgreement.join within_030_033_agreement within_030_036_agreement
private def within_030_038 : CountSegment 1 10 13120 3880 3882 1798 1799 := CountSegment.join atomic_3881 atomic_3882
private theorem within_030_038_agreement : RowAgreement 3880 3882 := RowAgreement.join atomic_3881_agreement atomic_3882_agreement
private def within_030_039 : CountSegment 1 10 13120 3882 3884 1799 1799 := CountSegment.join atomic_3883 atomic_3884
private theorem within_030_039_agreement : RowAgreement 3882 3884 := RowAgreement.join atomic_3883_agreement atomic_3884_agreement
private def within_030_040 : CountSegment 1 10 13120 3880 3884 1798 1799 := CountSegment.join within_030_038 within_030_039
private theorem within_030_040_agreement : RowAgreement 3880 3884 := RowAgreement.join within_030_038_agreement within_030_039_agreement
private def within_030_041 : CountSegment 1 10 13120 3884 3886 1799 1801 := CountSegment.join atomic_3885 atomic_3886
private theorem within_030_041_agreement : RowAgreement 3884 3886 := RowAgreement.join atomic_3885_agreement atomic_3886_agreement
private def within_030_042 : CountSegment 1 10 13120 3886 3888 1801 1802 := CountSegment.join atomic_3887 atomic_3888
private theorem within_030_042_agreement : RowAgreement 3886 3888 := RowAgreement.join atomic_3887_agreement atomic_3888_agreement
private def within_030_043 : CountSegment 1 10 13120 3884 3888 1799 1802 := CountSegment.join within_030_041 within_030_042
private theorem within_030_043_agreement : RowAgreement 3884 3888 := RowAgreement.join within_030_041_agreement within_030_042_agreement
private def within_030_044 : CountSegment 1 10 13120 3880 3888 1798 1802 := CountSegment.join within_030_040 within_030_043
private theorem within_030_044_agreement : RowAgreement 3880 3888 := RowAgreement.join within_030_040_agreement within_030_043_agreement
private def within_030_045 : CountSegment 1 10 13120 3872 3888 1793 1802 := CountSegment.join within_030_037 within_030_044
private theorem within_030_045_agreement : RowAgreement 3872 3888 := RowAgreement.join within_030_037_agreement within_030_044_agreement
private def within_030_046 : CountSegment 1 10 13120 3888 3890 1802 1803 := CountSegment.join atomic_3889 atomic_3890
private theorem within_030_046_agreement : RowAgreement 3888 3890 := RowAgreement.join atomic_3889_agreement atomic_3890_agreement
private def within_030_047 : CountSegment 1 10 13120 3890 3892 1803 1803 := CountSegment.join atomic_3891 atomic_3892
private theorem within_030_047_agreement : RowAgreement 3890 3892 := RowAgreement.join atomic_3891_agreement atomic_3892_agreement
private def within_030_048 : CountSegment 1 10 13120 3888 3892 1802 1803 := CountSegment.join within_030_046 within_030_047
private theorem within_030_048_agreement : RowAgreement 3888 3892 := RowAgreement.join within_030_046_agreement within_030_047_agreement
private def within_030_049 : CountSegment 1 10 13120 3892 3894 1803 1805 := CountSegment.join atomic_3893 atomic_3894
private theorem within_030_049_agreement : RowAgreement 3892 3894 := RowAgreement.join atomic_3893_agreement atomic_3894_agreement
private def within_030_050 : CountSegment 1 10 13120 3894 3896 1805 1806 := CountSegment.join atomic_3895 atomic_3896
private theorem within_030_050_agreement : RowAgreement 3894 3896 := RowAgreement.join atomic_3895_agreement atomic_3896_agreement
private def within_030_051 : CountSegment 1 10 13120 3892 3896 1803 1806 := CountSegment.join within_030_049 within_030_050
private theorem within_030_051_agreement : RowAgreement 3892 3896 := RowAgreement.join within_030_049_agreement within_030_050_agreement
private def within_030_052 : CountSegment 1 10 13120 3888 3896 1802 1806 := CountSegment.join within_030_048 within_030_051
private theorem within_030_052_agreement : RowAgreement 3888 3896 := RowAgreement.join within_030_048_agreement within_030_051_agreement
private def within_030_053 : CountSegment 1 10 13120 3896 3898 1806 1807 := CountSegment.join atomic_3897 atomic_3898
private theorem within_030_053_agreement : RowAgreement 3896 3898 := RowAgreement.join atomic_3897_agreement atomic_3898_agreement
private def within_030_054 : CountSegment 1 10 13120 3898 3900 1807 1807 := CountSegment.join atomic_3899 atomic_3900
private theorem within_030_054_agreement : RowAgreement 3898 3900 := RowAgreement.join atomic_3899_agreement atomic_3900_agreement
private def within_030_055 : CountSegment 1 10 13120 3896 3900 1806 1807 := CountSegment.join within_030_053 within_030_054
private theorem within_030_055_agreement : RowAgreement 3896 3900 := RowAgreement.join within_030_053_agreement within_030_054_agreement
private def within_030_056 : CountSegment 1 10 13120 3900 3902 1807 1808 := CountSegment.join atomic_3901 atomic_3902
private theorem within_030_056_agreement : RowAgreement 3900 3902 := RowAgreement.join atomic_3901_agreement atomic_3902_agreement
private def within_030_057 : CountSegment 1 10 13120 3902 3904 1808 1809 := CountSegment.join atomic_3903 atomic_3904
private theorem within_030_057_agreement : RowAgreement 3902 3904 := RowAgreement.join atomic_3903_agreement atomic_3904_agreement
private def within_030_058 : CountSegment 1 10 13120 3900 3904 1807 1809 := CountSegment.join within_030_056 within_030_057
private theorem within_030_058_agreement : RowAgreement 3900 3904 := RowAgreement.join within_030_056_agreement within_030_057_agreement
private def within_030_059 : CountSegment 1 10 13120 3896 3904 1806 1809 := CountSegment.join within_030_055 within_030_058
private theorem within_030_059_agreement : RowAgreement 3896 3904 := RowAgreement.join within_030_055_agreement within_030_058_agreement
private def within_030_060 : CountSegment 1 10 13120 3888 3904 1802 1809 := CountSegment.join within_030_052 within_030_059
private theorem within_030_060_agreement : RowAgreement 3888 3904 := RowAgreement.join within_030_052_agreement within_030_059_agreement
private def within_030_061 : CountSegment 1 10 13120 3872 3904 1793 1809 := CountSegment.join within_030_045 within_030_060
private theorem within_030_061_agreement : RowAgreement 3872 3904 := RowAgreement.join within_030_045_agreement within_030_060_agreement
private def within_030_062 : CountSegment 1 10 13120 3840 3904 1783 1809 := CountSegment.join within_030_030 within_030_061
private theorem within_030_062_agreement : RowAgreement 3840 3904 := RowAgreement.join within_030_030_agreement within_030_061_agreement
private def within_030_063 : CountSegment 1 10 13120 3904 3906 1809 1811 := CountSegment.join atomic_3905 atomic_3906
private theorem within_030_063_agreement : RowAgreement 3904 3906 := RowAgreement.join atomic_3905_agreement atomic_3906_agreement
private def within_030_064 : CountSegment 1 10 13120 3906 3908 1811 1811 := CountSegment.join atomic_3907 atomic_3908
private theorem within_030_064_agreement : RowAgreement 3906 3908 := RowAgreement.join atomic_3907_agreement atomic_3908_agreement
private def within_030_065 : CountSegment 1 10 13120 3904 3908 1809 1811 := CountSegment.join within_030_063 within_030_064
private theorem within_030_065_agreement : RowAgreement 3904 3908 := RowAgreement.join within_030_063_agreement within_030_064_agreement
private def within_030_066 : CountSegment 1 10 13120 3908 3910 1811 1812 := CountSegment.join atomic_3909 atomic_3910
private theorem within_030_066_agreement : RowAgreement 3908 3910 := RowAgreement.join atomic_3909_agreement atomic_3910_agreement
private def within_030_067 : CountSegment 1 10 13120 3910 3912 1812 1812 := CountSegment.join atomic_3911 atomic_3912
private theorem within_030_067_agreement : RowAgreement 3910 3912 := RowAgreement.join atomic_3911_agreement atomic_3912_agreement
private def within_030_068 : CountSegment 1 10 13120 3908 3912 1811 1812 := CountSegment.join within_030_066 within_030_067
private theorem within_030_068_agreement : RowAgreement 3908 3912 := RowAgreement.join within_030_066_agreement within_030_067_agreement
private def within_030_069 : CountSegment 1 10 13120 3904 3912 1809 1812 := CountSegment.join within_030_065 within_030_068
private theorem within_030_069_agreement : RowAgreement 3904 3912 := RowAgreement.join within_030_065_agreement within_030_068_agreement
private def within_030_070 : CountSegment 1 10 13120 3912 3914 1812 1812 := CountSegment.join atomic_3913 atomic_3914
private theorem within_030_070_agreement : RowAgreement 3912 3914 := RowAgreement.join atomic_3913_agreement atomic_3914_agreement
private def within_030_071 : CountSegment 1 10 13120 3914 3916 1812 1812 := CountSegment.join atomic_3915 atomic_3916
private theorem within_030_071_agreement : RowAgreement 3914 3916 := RowAgreement.join atomic_3915_agreement atomic_3916_agreement
private def within_030_072 : CountSegment 1 10 13120 3912 3916 1812 1812 := CountSegment.join within_030_070 within_030_071
private theorem within_030_072_agreement : RowAgreement 3912 3916 := RowAgreement.join within_030_070_agreement within_030_071_agreement
private def within_030_073 : CountSegment 1 10 13120 3916 3918 1812 1813 := CountSegment.join atomic_3917 atomic_3918
private theorem within_030_073_agreement : RowAgreement 3916 3918 := RowAgreement.join atomic_3917_agreement atomic_3918_agreement
private def within_030_074 : CountSegment 1 10 13120 3918 3920 1813 1813 := CountSegment.join atomic_3919 atomic_3920
private theorem within_030_074_agreement : RowAgreement 3918 3920 := RowAgreement.join atomic_3919_agreement atomic_3920_agreement
private def within_030_075 : CountSegment 1 10 13120 3916 3920 1812 1813 := CountSegment.join within_030_073 within_030_074
private theorem within_030_075_agreement : RowAgreement 3916 3920 := RowAgreement.join within_030_073_agreement within_030_074_agreement
private def within_030_076 : CountSegment 1 10 13120 3912 3920 1812 1813 := CountSegment.join within_030_072 within_030_075
private theorem within_030_076_agreement : RowAgreement 3912 3920 := RowAgreement.join within_030_072_agreement within_030_075_agreement
private def within_030_077 : CountSegment 1 10 13120 3904 3920 1809 1813 := CountSegment.join within_030_069 within_030_076
private theorem within_030_077_agreement : RowAgreement 3904 3920 := RowAgreement.join within_030_069_agreement within_030_076_agreement
private def within_030_078 : CountSegment 1 10 13120 3920 3922 1813 1813 := CountSegment.join atomic_3921 atomic_3922
private theorem within_030_078_agreement : RowAgreement 3920 3922 := RowAgreement.join atomic_3921_agreement atomic_3922_agreement
private def within_030_079 : CountSegment 1 10 13120 3922 3924 1813 1815 := CountSegment.join atomic_3923 atomic_3924
private theorem within_030_079_agreement : RowAgreement 3922 3924 := RowAgreement.join atomic_3923_agreement atomic_3924_agreement
private def within_030_080 : CountSegment 1 10 13120 3920 3924 1813 1815 := CountSegment.join within_030_078 within_030_079
private theorem within_030_080_agreement : RowAgreement 3920 3924 := RowAgreement.join within_030_078_agreement within_030_079_agreement
private def within_030_081 : CountSegment 1 10 13120 3924 3926 1815 1816 := CountSegment.join atomic_3925 atomic_3926
private theorem within_030_081_agreement : RowAgreement 3924 3926 := RowAgreement.join atomic_3925_agreement atomic_3926_agreement
private def within_030_082 : CountSegment 1 10 13120 3926 3928 1816 1818 := CountSegment.join atomic_3927 atomic_3928
private theorem within_030_082_agreement : RowAgreement 3926 3928 := RowAgreement.join atomic_3927_agreement atomic_3928_agreement
private def within_030_083 : CountSegment 1 10 13120 3924 3928 1815 1818 := CountSegment.join within_030_081 within_030_082
private theorem within_030_083_agreement : RowAgreement 3924 3928 := RowAgreement.join within_030_081_agreement within_030_082_agreement
private def within_030_084 : CountSegment 1 10 13120 3920 3928 1813 1818 := CountSegment.join within_030_080 within_030_083
private theorem within_030_084_agreement : RowAgreement 3920 3928 := RowAgreement.join within_030_080_agreement within_030_083_agreement
private def within_030_085 : CountSegment 1 10 13120 3928 3930 1818 1821 := CountSegment.join atomic_3929 atomic_3930
private theorem within_030_085_agreement : RowAgreement 3928 3930 := RowAgreement.join atomic_3929_agreement atomic_3930_agreement
private def within_030_086 : CountSegment 1 10 13120 3930 3932 1821 1822 := CountSegment.join atomic_3931 atomic_3932
private theorem within_030_086_agreement : RowAgreement 3930 3932 := RowAgreement.join atomic_3931_agreement atomic_3932_agreement
private def within_030_087 : CountSegment 1 10 13120 3928 3932 1818 1822 := CountSegment.join within_030_085 within_030_086
private theorem within_030_087_agreement : RowAgreement 3928 3932 := RowAgreement.join within_030_085_agreement within_030_086_agreement
private def within_030_088 : CountSegment 1 10 13120 3932 3934 1822 1822 := CountSegment.join atomic_3933 atomic_3934
private theorem within_030_088_agreement : RowAgreement 3932 3934 := RowAgreement.join atomic_3933_agreement atomic_3934_agreement
private def within_030_089 : CountSegment 1 10 13120 3934 3936 1822 1824 := CountSegment.join atomic_3935 atomic_3936
private theorem within_030_089_agreement : RowAgreement 3934 3936 := RowAgreement.join atomic_3935_agreement atomic_3936_agreement
private def within_030_090 : CountSegment 1 10 13120 3932 3936 1822 1824 := CountSegment.join within_030_088 within_030_089
private theorem within_030_090_agreement : RowAgreement 3932 3936 := RowAgreement.join within_030_088_agreement within_030_089_agreement
private def within_030_091 : CountSegment 1 10 13120 3928 3936 1818 1824 := CountSegment.join within_030_087 within_030_090
private theorem within_030_091_agreement : RowAgreement 3928 3936 := RowAgreement.join within_030_087_agreement within_030_090_agreement
private def within_030_092 : CountSegment 1 10 13120 3920 3936 1813 1824 := CountSegment.join within_030_084 within_030_091
private theorem within_030_092_agreement : RowAgreement 3920 3936 := RowAgreement.join within_030_084_agreement within_030_091_agreement
private def within_030_093 : CountSegment 1 10 13120 3904 3936 1809 1824 := CountSegment.join within_030_077 within_030_092
private theorem within_030_093_agreement : RowAgreement 3904 3936 := RowAgreement.join within_030_077_agreement within_030_092_agreement
private def within_030_094 : CountSegment 1 10 13120 3936 3938 1824 1824 := CountSegment.join atomic_3937 atomic_3938
private theorem within_030_094_agreement : RowAgreement 3936 3938 := RowAgreement.join atomic_3937_agreement atomic_3938_agreement
private def within_030_095 : CountSegment 1 10 13120 3938 3940 1824 1824 := CountSegment.join atomic_3939 atomic_3940
private theorem within_030_095_agreement : RowAgreement 3938 3940 := RowAgreement.join atomic_3939_agreement atomic_3940_agreement
private def within_030_096 : CountSegment 1 10 13120 3936 3940 1824 1824 := CountSegment.join within_030_094 within_030_095
private theorem within_030_096_agreement : RowAgreement 3936 3940 := RowAgreement.join within_030_094_agreement within_030_095_agreement
private def within_030_097 : CountSegment 1 10 13120 3940 3942 1824 1825 := CountSegment.join atomic_3941 atomic_3942
private theorem within_030_097_agreement : RowAgreement 3940 3942 := RowAgreement.join atomic_3941_agreement atomic_3942_agreement
private def within_030_098 : CountSegment 1 10 13120 3942 3944 1825 1825 := CountSegment.join atomic_3943 atomic_3944
private theorem within_030_098_agreement : RowAgreement 3942 3944 := RowAgreement.join atomic_3943_agreement atomic_3944_agreement
private def within_030_099 : CountSegment 1 10 13120 3940 3944 1824 1825 := CountSegment.join within_030_097 within_030_098
private theorem within_030_099_agreement : RowAgreement 3940 3944 := RowAgreement.join within_030_097_agreement within_030_098_agreement
private def within_030_100 : CountSegment 1 10 13120 3936 3944 1824 1825 := CountSegment.join within_030_096 within_030_099
private theorem within_030_100_agreement : RowAgreement 3936 3944 := RowAgreement.join within_030_096_agreement within_030_099_agreement
private def within_030_101 : CountSegment 1 10 13120 3944 3946 1825 1827 := CountSegment.join atomic_3945 atomic_3946
private theorem within_030_101_agreement : RowAgreement 3944 3946 := RowAgreement.join atomic_3945_agreement atomic_3946_agreement
private def within_030_102 : CountSegment 1 10 13120 3946 3948 1827 1828 := CountSegment.join atomic_3947 atomic_3948
private theorem within_030_102_agreement : RowAgreement 3946 3948 := RowAgreement.join atomic_3947_agreement atomic_3948_agreement
private def within_030_103 : CountSegment 1 10 13120 3944 3948 1825 1828 := CountSegment.join within_030_101 within_030_102
private theorem within_030_103_agreement : RowAgreement 3944 3948 := RowAgreement.join within_030_101_agreement within_030_102_agreement
private def within_030_104 : CountSegment 1 10 13120 3948 3950 1828 1828 := CountSegment.join atomic_3949 atomic_3950
private theorem within_030_104_agreement : RowAgreement 3948 3950 := RowAgreement.join atomic_3949_agreement atomic_3950_agreement
private def within_030_105 : CountSegment 1 10 13120 3950 3952 1828 1829 := CountSegment.join atomic_3951 atomic_3952
private theorem within_030_105_agreement : RowAgreement 3950 3952 := RowAgreement.join atomic_3951_agreement atomic_3952_agreement
private def within_030_106 : CountSegment 1 10 13120 3948 3952 1828 1829 := CountSegment.join within_030_104 within_030_105
private theorem within_030_106_agreement : RowAgreement 3948 3952 := RowAgreement.join within_030_104_agreement within_030_105_agreement
private def within_030_107 : CountSegment 1 10 13120 3944 3952 1825 1829 := CountSegment.join within_030_103 within_030_106
private theorem within_030_107_agreement : RowAgreement 3944 3952 := RowAgreement.join within_030_103_agreement within_030_106_agreement
private def within_030_108 : CountSegment 1 10 13120 3936 3952 1824 1829 := CountSegment.join within_030_100 within_030_107
private theorem within_030_108_agreement : RowAgreement 3936 3952 := RowAgreement.join within_030_100_agreement within_030_107_agreement
private def within_030_109 : CountSegment 1 10 13120 3952 3954 1829 1831 := CountSegment.join atomic_3953 atomic_3954
private theorem within_030_109_agreement : RowAgreement 3952 3954 := RowAgreement.join atomic_3953_agreement atomic_3954_agreement
private def within_030_110 : CountSegment 1 10 13120 3954 3956 1831 1832 := CountSegment.join atomic_3955 atomic_3956
private theorem within_030_110_agreement : RowAgreement 3954 3956 := RowAgreement.join atomic_3955_agreement atomic_3956_agreement
private def within_030_111 : CountSegment 1 10 13120 3952 3956 1829 1832 := CountSegment.join within_030_109 within_030_110
private theorem within_030_111_agreement : RowAgreement 3952 3956 := RowAgreement.join within_030_109_agreement within_030_110_agreement
private def within_030_112 : CountSegment 1 10 13120 3956 3958 1832 1832 := CountSegment.join atomic_3957 atomic_3958
private theorem within_030_112_agreement : RowAgreement 3956 3958 := RowAgreement.join atomic_3957_agreement atomic_3958_agreement
private def within_030_113 : CountSegment 1 10 13120 3958 3960 1832 1832 := CountSegment.join atomic_3959 atomic_3960
private theorem within_030_113_agreement : RowAgreement 3958 3960 := RowAgreement.join atomic_3959_agreement atomic_3960_agreement
private def within_030_114 : CountSegment 1 10 13120 3956 3960 1832 1832 := CountSegment.join within_030_112 within_030_113
private theorem within_030_114_agreement : RowAgreement 3956 3960 := RowAgreement.join within_030_112_agreement within_030_113_agreement
private def within_030_115 : CountSegment 1 10 13120 3952 3960 1829 1832 := CountSegment.join within_030_111 within_030_114
private theorem within_030_115_agreement : RowAgreement 3952 3960 := RowAgreement.join within_030_111_agreement within_030_114_agreement
private def within_030_116 : CountSegment 1 10 13120 3960 3962 1832 1833 := CountSegment.join atomic_3961 atomic_3962
private theorem within_030_116_agreement : RowAgreement 3960 3962 := RowAgreement.join atomic_3961_agreement atomic_3962_agreement
private def within_030_117 : CountSegment 1 10 13120 3962 3964 1833 1834 := CountSegment.join atomic_3963 atomic_3964
private theorem within_030_117_agreement : RowAgreement 3962 3964 := RowAgreement.join atomic_3963_agreement atomic_3964_agreement
private def within_030_118 : CountSegment 1 10 13120 3960 3964 1832 1834 := CountSegment.join within_030_116 within_030_117
private theorem within_030_118_agreement : RowAgreement 3960 3964 := RowAgreement.join within_030_116_agreement within_030_117_agreement
private def within_030_119 : CountSegment 1 10 13120 3964 3966 1834 1836 := CountSegment.join atomic_3965 atomic_3966
private theorem within_030_119_agreement : RowAgreement 3964 3966 := RowAgreement.join atomic_3965_agreement atomic_3966_agreement
private def within_030_120 : CountSegment 1 10 13120 3966 3968 1836 1836 := CountSegment.join atomic_3967 atomic_3968
private theorem within_030_120_agreement : RowAgreement 3966 3968 := RowAgreement.join atomic_3967_agreement atomic_3968_agreement
private def within_030_121 : CountSegment 1 10 13120 3964 3968 1834 1836 := CountSegment.join within_030_119 within_030_120
private theorem within_030_121_agreement : RowAgreement 3964 3968 := RowAgreement.join within_030_119_agreement within_030_120_agreement
private def within_030_122 : CountSegment 1 10 13120 3960 3968 1832 1836 := CountSegment.join within_030_118 within_030_121
private theorem within_030_122_agreement : RowAgreement 3960 3968 := RowAgreement.join within_030_118_agreement within_030_121_agreement
private def within_030_123 : CountSegment 1 10 13120 3952 3968 1829 1836 := CountSegment.join within_030_115 within_030_122
private theorem within_030_123_agreement : RowAgreement 3952 3968 := RowAgreement.join within_030_115_agreement within_030_122_agreement
private def within_030_124 : CountSegment 1 10 13120 3936 3968 1824 1836 := CountSegment.join within_030_108 within_030_123
private theorem within_030_124_agreement : RowAgreement 3936 3968 := RowAgreement.join within_030_108_agreement within_030_123_agreement
private def within_030_125 : CountSegment 1 10 13120 3904 3968 1809 1836 := CountSegment.join within_030_093 within_030_124
private theorem within_030_125_agreement : RowAgreement 3904 3968 := RowAgreement.join within_030_093_agreement within_030_124_agreement
private def within_030_126 : CountSegment 1 10 13120 3840 3968 1783 1836 := CountSegment.join within_030_062 within_030_125
private theorem within_030_126_agreement : RowAgreement 3840 3968 := RowAgreement.join within_030_062_agreement within_030_125_agreement
def segment_030 : CountSegment 1 10 13120 3840 3968 1783 1836 := within_030_126
theorem agreement_030 : RowAgreement 3840 3968 := within_030_126_agreement
end ProjectEulerSolutions.P58Numerical
