import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3585 : RowCertificate 3585 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3585 1) := by change ¬ Nat.Prime 51416071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7345153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3585 2) := by change ¬ Nat.Prime 51408901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3024053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3585 3) := by change ¬ Nat.Prime 51401731; exact PrattCertificates.composite_of_product (factor := 1279) (cofactor := 40189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3585 : row 3585 = 0 := (row_3585.checked).trans rfl
theorem model_count_3585 : modelRow 3585 = 0 := by decide
theorem modelrow_agrees_3585 : modelRow 3585 = row 3585 := model_count_3585.trans row_count_3585.symm
def row_3586 : RowCertificate 3586 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3586 1) := by change ¬ Nat.Prime 51444757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7349251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3586 2) := by change ¬ Nat.Prime 51437585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10287517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3586 3) := by change ¬ Nat.Prime 51430413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17143471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3586 : row 3586 = 0 := (row_3586.checked).trans rfl
theorem model_count_3586 : modelRow 3586 = 0 := by decide
theorem modelrow_agrees_3586 : modelRow 3586 = row 3586 := model_count_3586.trans row_count_3586.symm
def row_3587 : RowCertificate 3587 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3587 1) := by change ¬ Nat.Prime 51473451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17157817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3587 2) := by change ¬ Nat.Prime 51466277; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 164429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3587 3) := by change ¬ Nat.Prime 51459103; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 54919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3587 : row 3587 = 0 := (row_3587.checked).trans rfl
theorem model_count_3587 : modelRow 3587 = 0 := by decide
theorem modelrow_agrees_3587 : modelRow 3587 = row 3587 := model_count_3587.trans row_count_3587.symm
def row_3588 : RowCertificate 3588 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3588 1) := by change ¬ Nat.Prime 51502153; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 49003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3588 2) := by change ¬ Nat.Prime 51494977; exact PrattCertificates.composite_of_product (factor := 2341) (cofactor := 21997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3588 3) := by change ¬ Nat.Prime 51487801; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 230887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3588 : row 3588 = 0 := (row_3588.checked).trans rfl
theorem model_count_3588 : modelRow 3588 = 0 := by decide
theorem modelrow_agrees_3588 : modelRow 3588 = row 3588 := model_count_3588.trans row_count_3588.symm
def row_3589 : RowCertificate 3589 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3589 1) := by change ¬ Nat.Prime 51530863; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 112759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3589 2) := by change ¬ Nat.Prime 51523685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10304737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3589 3) := by change ¬ Nat.Prime 51516507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17172169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3589 : row 3589 = 0 := (row_3589.checked).trans rfl
theorem model_count_3589 : modelRow 3589 = 0 := by decide
theorem modelrow_agrees_3589 : modelRow 3589 = row 3589 := model_count_3589.trans row_count_3589.symm
def row_3590 : RowCertificate 3590 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3590 1) := by change ¬ Nat.Prime 51559581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17186527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3590 2) := by change ¬ Nat.Prime 51552401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1777669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3590 3) := by change ¬ Nat.Prime 51545221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7363603) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3590 : row 3590 = 0 := (row_3590.checked).trans rfl
theorem model_count_3590 : modelRow 3590 = 0 := by decide
theorem modelrow_agrees_3590 : modelRow 3590 = row 3590 := model_count_3590.trans row_count_3590.symm
def row_3591 : RowCertificate 3591 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3591 1) := by change Nat.Prime 51588307; exact ProjectEulerSolutions.P58PrimeNodes.p_51588307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3591 2) := by change ¬ Nat.Prime 51581125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10316225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3591 3) := by change Nat.Prime 51573943; exact ProjectEulerSolutions.P58PrimeNodes.p_51573943.prime
    exact decide_eq_true prime
theorem row_count_3591 : row 3591 = 2 := (row_3591.checked).trans rfl
theorem model_count_3591 : modelRow 3591 = 2 := by decide
theorem modelrow_agrees_3591 : modelRow 3591 = row 3591 := model_count_3591.trans row_count_3591.symm
def row_3592 : RowCertificate 3592 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3592 1) := by change ¬ Nat.Prime 51617041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7373863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3592 2) := by change ¬ Nat.Prime 51609857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3969989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3592 3) := by change ¬ Nat.Prime 51602673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17200891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3592 : row 3592 = 0 := (row_3592.checked).trans rfl
theorem model_count_3592 : modelRow 3592 = 0 := by decide
theorem modelrow_agrees_3592 : modelRow 3592 = row 3592 := model_count_3592.trans row_count_3592.symm
def row_3593 : RowCertificate 3593 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3593 1) := by change ¬ Nat.Prime 51645783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17215261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3593 2) := by change ¬ Nat.Prime 51638597; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 298489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3593 3) := by change ¬ Nat.Prime 51631411; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3971647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3593 : row 3593 = 0 := (row_3593.checked).trans rfl
theorem model_count_3593 : modelRow 3593 = 0 := by decide
theorem modelrow_agrees_3593 : modelRow 3593 = row 3593 := model_count_3593.trans row_count_3593.symm
def row_3594 : RowCertificate 3594 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3594 1) := by change ¬ Nat.Prime 51674533; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1396609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3594 2) := by change ¬ Nat.Prime 51667345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10333469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3594 3) := by change ¬ Nat.Prime 51660157; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1201399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3594 : row 3594 = 0 := (row_3594.checked).trans rfl
theorem model_count_3594 : modelRow 3594 = 0 := by decide
theorem modelrow_agrees_3594 : modelRow 3594 = row 3594 := model_count_3594.trans row_count_3594.symm
def row_3595 : RowCertificate 3595 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3595 1) := by change Nat.Prime 51703291; exact ProjectEulerSolutions.P58PrimeNodes.p_51703291.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3595 2) := by change Nat.Prime 51696101; exact ProjectEulerSolutions.P58PrimeNodes.p_51696101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3595 3) := by change ¬ Nat.Prime 51688911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17229637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3595 : row 3595 = 2 := (row_3595.checked).trans rfl
theorem model_count_3595 : modelRow 3595 = 2 := by decide
theorem modelrow_agrees_3595 : modelRow 3595 = row 3595 := model_count_3595.trans row_count_3595.symm
def row_3596 : RowCertificate 3596 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3596 1) := by change ¬ Nat.Prime 51732057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17244019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3596 2) := by change ¬ Nat.Prime 51724865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10344973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3596 3) := by change ¬ Nat.Prime 51717673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7388239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3596 : row 3596 = 0 := (row_3596.checked).trans rfl
theorem model_count_3596 : modelRow 3596 = 0 := by decide
theorem modelrow_agrees_3596 : modelRow 3596 = row 3596 := model_count_3596.trans row_count_3596.symm
def row_3597 : RowCertificate 3597 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3597 1) := by change ¬ Nat.Prime 51760831; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 29527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3597 2) := by change ¬ Nat.Prime 51753637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3981049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3597 3) := by change ¬ Nat.Prime 51746443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7392349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3597 : row 3597 = 0 := (row_3597.checked).trans rfl
theorem model_count_3597 : modelRow 3597 = 0 := by decide
theorem modelrow_agrees_3597 : modelRow 3597 = row 3597 := model_count_3597.trans row_count_3597.symm
def row_3598 : RowCertificate 3598 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3598 1) := by change Nat.Prime 51789613; exact ProjectEulerSolutions.P58PrimeNodes.p_51789613.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3598 2) := by change ¬ Nat.Prime 51782417; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 347533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3598 3) := by change ¬ Nat.Prime 51775221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17258407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3598 : row 3598 = 1 := (row_3598.checked).trans rfl
theorem model_count_3598 : modelRow 3598 = 1 := by decide
theorem modelrow_agrees_3598 : modelRow 3598 = row 3598 := model_count_3598.trans row_count_3598.symm
def row_3599 : RowCertificate 3599 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3599 1) := by change ¬ Nat.Prime 51818403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17272801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3599 2) := by change ¬ Nat.Prime 51811205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10362241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3599 3) := by change ¬ Nat.Prime 51804007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1671097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3599 : row 3599 = 0 := (row_3599.checked).trans rfl
theorem model_count_3599 : modelRow 3599 = 0 := by decide
theorem modelrow_agrees_3599 : modelRow 3599 = row 3599 := model_count_3599.trans row_count_3599.symm
def row_3600 : RowCertificate 3600 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3600 1) := by change ¬ Nat.Prime 51847201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7406743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3600 2) := by change ¬ Nat.Prime 51840001; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 710137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3600 3) := by change Nat.Prime 51832801; exact ProjectEulerSolutions.P58PrimeNodes.p_51832801.prime
    exact decide_eq_true prime
theorem row_count_3600 : row 3600 = 1 := (row_3600.checked).trans rfl
theorem model_count_3600 : modelRow 3600 = 1 := by decide
theorem modelrow_agrees_3600 : modelRow 3600 = row 3600 := model_count_3600.trans row_count_3600.symm
def row_3601 : RowCertificate 3601 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3601 1) := by change ¬ Nat.Prime 51876007; exact PrattCertificates.composite_of_product (factor := 7057) (cofactor := 7351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3601 2) := by change ¬ Nat.Prime 51868805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10373761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3601 3) := by change ¬ Nat.Prime 51861603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17287201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3601 : row 3601 = 0 := (row_3601.checked).trans rfl
theorem model_count_3601 : modelRow 3601 = 0 := by decide
theorem modelrow_agrees_3601 : modelRow 3601 = row 3601 := model_count_3601.trans row_count_3601.symm
def row_3602 : RowCertificate 3602 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3602 1) := by change ¬ Nat.Prime 51904821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17301607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3602 2) := by change ¬ Nat.Prime 51897617; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3052801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3602 3) := by change Nat.Prime 51890413; exact ProjectEulerSolutions.P58PrimeNodes.p_51890413.prime
    exact decide_eq_true prime
theorem row_count_3602 : row 3602 = 1 := (row_3602.checked).trans rfl
theorem model_count_3602 : modelRow 3602 = 1 := by decide
theorem modelrow_agrees_3602 : modelRow 3602 = row 3602 := model_count_3602.trans row_count_3602.symm
def row_3603 : RowCertificate 3603 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3603 1) := by change ¬ Nat.Prime 51933643; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 775129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3603 2) := by change ¬ Nat.Prime 51926437; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 226753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3603 3) := by change ¬ Nat.Prime 51919231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7417033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3603 : row 3603 = 0 := (row_3603.checked).trans rfl
theorem model_count_3603 : modelRow 3603 = 0 := by decide
theorem modelrow_agrees_3603 : modelRow 3603 = row 3603 := model_count_3603.trans row_count_3603.symm
def row_3604 : RowCertificate 3604 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3604 1) := by change ¬ Nat.Prime 51962473; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2734867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3604 2) := by change ¬ Nat.Prime 51955265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10391053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3604 3) := by change ¬ Nat.Prime 51948057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17316019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3604 : row 3604 = 0 := (row_3604.checked).trans rfl
theorem model_count_3604 : modelRow 3604 = 0 := by decide
theorem modelrow_agrees_3604 : modelRow 3604 = row 3604 := model_count_3604.trans row_count_3604.symm
def row_3605 : RowCertificate 3605 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3605 1) := by change ¬ Nat.Prime 51991311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17330437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3605 2) := by change ¬ Nat.Prime 51984101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3998777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3605 3) := by change ¬ Nat.Prime 51976891; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 331063) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3605 : row 3605 = 0 := (row_3605.checked).trans rfl
theorem model_count_3605 : modelRow 3605 = 0 := by decide
theorem modelrow_agrees_3605 : modelRow 3605 = row 3605 := model_count_3605.trans row_count_3605.symm
def row_3606 : RowCertificate 3606 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3606 1) := by change ¬ Nat.Prime 52020157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7431451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3606 2) := by change ¬ Nat.Prime 52012945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10402589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3606 3) := by change ¬ Nat.Prime 52005733; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4000441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3606 : row 3606 = 0 := (row_3606.checked).trans rfl
theorem model_count_3606 : modelRow 3606 = 0 := by decide
theorem modelrow_agrees_3606 : modelRow 3606 = row 3606 := model_count_3606.trans row_count_3606.symm
def row_3607 : RowCertificate 3607 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3607 1) := by change ¬ Nat.Prime 52049011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7435573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3607 2) := by change ¬ Nat.Prime 52041797; exact PrattCertificates.composite_of_product (factor := 4049) (cofactor := 12853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3607 3) := by change ¬ Nat.Prime 52034583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17344861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3607 : row 3607 = 0 := (row_3607.checked).trans rfl
theorem model_count_3607 : modelRow 3607 = 0 := by decide
theorem modelrow_agrees_3607 : modelRow 3607 = row 3607 := model_count_3607.trans row_count_3607.symm
def row_3608 : RowCertificate 3608 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3608 1) := by change ¬ Nat.Prime 52077873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17359291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3608 2) := by change Nat.Prime 52070657; exact ProjectEulerSolutions.P58PrimeNodes.p_52070657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3608 3) := by change ¬ Nat.Prime 52063441; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 344791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3608 : row 3608 = 1 := (row_3608.checked).trans rfl
theorem model_count_3608 : modelRow 3608 = 1 := by decide
theorem modelrow_agrees_3608 : modelRow 3608 = row 3608 := model_count_3608.trans row_count_3608.symm
def row_3609 : RowCertificate 3609 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3609 1) := by change ¬ Nat.Prime 52106743; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4008211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3609 2) := by change ¬ Nat.Prime 52099525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10419905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3609 3) := by change ¬ Nat.Prime 52092307; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1680397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3609 : row 3609 = 0 := (row_3609.checked).trans rfl
theorem model_count_3609 : modelRow 3609 = 0 := by decide
theorem modelrow_agrees_3609 : modelRow 3609 = row 3609 := model_count_3609.trans row_count_3609.symm
def row_3610 : RowCertificate 3610 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3610 1) := by change Nat.Prime 52135621; exact ProjectEulerSolutions.P58PrimeNodes.p_52135621.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3610 2) := by change ¬ Nat.Prime 52128401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4009877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3610 3) := by change ¬ Nat.Prime 52121181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17373727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3610 : row 3610 = 1 := (row_3610.checked).trans rfl
theorem model_count_3610 : modelRow 3610 = 1 := by decide
theorem modelrow_agrees_3610 : modelRow 3610 = row 3610 := model_count_3610.trans row_count_3610.symm
def row_3611 : RowCertificate 3611 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3611 1) := by change ¬ Nat.Prime 52164507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17388169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3611 2) := by change ¬ Nat.Prime 52157285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10431457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3611 3) := by change ¬ Nat.Prime 52150063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7450009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3611 : row 3611 = 0 := (row_3611.checked).trans rfl
theorem model_count_3611 : modelRow 3611 = 0 := by decide
theorem modelrow_agrees_3611 : modelRow 3611 = row 3611 := model_count_3611.trans row_count_3611.symm
def row_3612 : RowCertificate 3612 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3612 1) := by change ¬ Nat.Prime 52193401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4014877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3612 2) := by change ¬ Nat.Prime 52186177; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 380921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3612 3) := by change Nat.Prime 52178953; exact ProjectEulerSolutions.P58PrimeNodes.p_52178953.prime
    exact decide_eq_true prime
theorem row_count_3612 : row 3612 = 1 := (row_3612.checked).trans rfl
theorem model_count_3612 : modelRow 3612 = 1 := by decide
theorem modelrow_agrees_3612 : modelRow 3612 = row 3612 := model_count_3612.trans row_count_3612.symm
def row_3613 : RowCertificate 3613 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3613 1) := by change ¬ Nat.Prime 52222303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7460329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3613 2) := by change ¬ Nat.Prime 52215077; exact PrattCertificates.composite_of_product (factor := 1709) (cofactor := 30553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3613 3) := by change ¬ Nat.Prime 52207851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17402617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3613 : row 3613 = 0 := (row_3613.checked).trans rfl
theorem model_count_3613 : modelRow 3613 = 0 := by decide
theorem modelrow_agrees_3613 : modelRow 3613 = row 3613 := model_count_3613.trans row_count_3613.symm
def row_3614 : RowCertificate 3614 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3614 1) := by change ¬ Nat.Prime 52251213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17417071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3614 2) := by change ¬ Nat.Prime 52243985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10448797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3614 3) := by change ¬ Nat.Prime 52236757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2749303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3614 : row 3614 = 0 := (row_3614.checked).trans rfl
theorem model_count_3614 : modelRow 3614 = 0 := by decide
theorem modelrow_agrees_3614 : modelRow 3614 = row 3614 := model_count_3614.trans row_count_3614.symm
def row_3615 : RowCertificate 3615 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3615 1) := by change ¬ Nat.Prime 52280131; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1215817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3615 2) := by change ¬ Nat.Prime 52272901; exact PrattCertificates.composite_of_product (factor := 4637) (cofactor := 11273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3615 3) := by change Nat.Prime 52265671; exact ProjectEulerSolutions.P58PrimeNodes.p_52265671.prime
    exact decide_eq_true prime
theorem row_count_3615 : row 3615 = 1 := (row_3615.checked).trans rfl
theorem model_count_3615 : modelRow 3615 = 1 := by decide
theorem modelrow_agrees_3615 : modelRow 3615 = row 3615 := model_count_3615.trans row_count_3615.symm
def row_3616 : RowCertificate 3616 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3616 1) := by change ¬ Nat.Prime 52309057; exact PrattCertificates.composite_of_product (factor := 823) (cofactor := 63559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3616 2) := by change ¬ Nat.Prime 52301825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10460365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3616 3) := by change ¬ Nat.Prime 52294593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17431531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3616 : row 3616 = 0 := (row_3616.checked).trans rfl
theorem model_count_3616 : modelRow 3616 = 0 := by decide
theorem modelrow_agrees_3616 : modelRow 3616 = row 3616 := model_count_3616.trans row_count_3616.symm
def row_3617 : RowCertificate 3617 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3617 1) := by change ¬ Nat.Prime 52337991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17445997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3617 2) := by change ¬ Nat.Prime 52330757; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 48953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3617 3) := by change ¬ Nat.Prime 52323523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7474789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3617 : row 3617 = 0 := (row_3617.checked).trans rfl
theorem model_count_3617 : modelRow 3617 = 0 := by decide
theorem modelrow_agrees_3617 : modelRow 3617 = row 3617 := model_count_3617.trans row_count_3617.symm
def row_3618 : RowCertificate 3618 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3618 1) := by change ¬ Nat.Prime 52366933; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 87133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3618 2) := by change ¬ Nat.Prime 52359697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4027669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3618 3) := by change ¬ Nat.Prime 52352461; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7478923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3618 : row 3618 = 0 := (row_3618.checked).trans rfl
theorem model_count_3618 : modelRow 3618 = 0 := by decide
theorem modelrow_agrees_3618 : modelRow 3618 = row 3618 := model_count_3618.trans row_count_3618.symm
def row_3619 : RowCertificate 3619 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3619 1) := by change Nat.Prime 52395883; exact ProjectEulerSolutions.P58PrimeNodes.p_52395883.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3619 2) := by change ¬ Nat.Prime 52388645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10477729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3619 3) := by change ¬ Nat.Prime 52381407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17460469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3619 : row 3619 = 1 := (row_3619.checked).trans rfl
theorem model_count_3619 : modelRow 3619 = 1 := by decide
theorem modelrow_agrees_3619 : modelRow 3619 = row 3619 := model_count_3619.trans row_count_3619.symm
def row_3620 : RowCertificate 3620 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3620 1) := by change ¬ Nat.Prime 52424841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17474947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3620 2) := by change Nat.Prime 52417601; exact ProjectEulerSolutions.P58PrimeNodes.p_52417601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3620 3) := by change ¬ Nat.Prime 52410361; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 540313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3620 : row 3620 = 1 := (row_3620.checked).trans rfl
theorem model_count_3620 : modelRow 3620 = 1 := by decide
theorem modelrow_agrees_3620 : modelRow 3620 = row 3620 := model_count_3620.trans row_count_3620.symm
def row_3621 : RowCertificate 3621 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3621 1) := by change ¬ Nat.Prime 52453807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7493401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3621 2) := by change ¬ Nat.Prime 52446565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10489313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3621 3) := by change ¬ Nat.Prime 52439323; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1417279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3621 : row 3621 = 0 := (row_3621.checked).trans rfl
theorem model_count_3621 : modelRow 3621 = 0 := by decide
theorem modelrow_agrees_3621 : modelRow 3621 = row 3621 := model_count_3621.trans row_count_3621.symm
def row_3622 : RowCertificate 3622 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3622 1) := by change ¬ Nat.Prime 52482781; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4037137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3622 2) := by change Nat.Prime 52475537; exact ProjectEulerSolutions.P58PrimeNodes.p_52475537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3622 3) := by change ¬ Nat.Prime 52468293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17489431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3622 : row 3622 = 1 := (row_3622.checked).trans rfl
theorem model_count_3622 : modelRow 3622 = 1 := by decide
theorem modelrow_agrees_3622 : modelRow 3622 = row 3622 := model_count_3622.trans row_count_3622.symm
def row_3623 : RowCertificate 3623 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3623 1) := by change ¬ Nat.Prime 52511763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17503921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3623 2) := by change ¬ Nat.Prime 52504517; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4038809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3623 3) := by change ¬ Nat.Prime 52497271; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 860611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3623 : row 3623 = 0 := (row_3623.checked).trans rfl
theorem model_count_3623 : modelRow 3623 = 0 := by decide
theorem modelrow_agrees_3623 : modelRow 3623 = row 3623 := model_count_3623.trans row_count_3623.symm
def row_3624 : RowCertificate 3624 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3624 1) := by change ¬ Nat.Prime 52540753; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1694863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3624 2) := by change ¬ Nat.Prime 52533505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10506701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3624 3) := by change ¬ Nat.Prime 52526257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7503751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3624 : row 3624 = 0 := (row_3624.checked).trans rfl
theorem model_count_3624 : modelRow 3624 = 0 := by decide
theorem modelrow_agrees_3624 : modelRow 3624 = row 3624 := model_count_3624.trans row_count_3624.symm
def row_3625 : RowCertificate 3625 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3625 1) := by change ¬ Nat.Prime 52569751; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4043827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3625 2) := by change ¬ Nat.Prime 52562501; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 334793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3625 3) := by change ¬ Nat.Prime 52555251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17518417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3625 : row 3625 = 0 := (row_3625.checked).trans rfl
theorem model_count_3625 : modelRow 3625 = 0 := by decide
theorem modelrow_agrees_3625 : modelRow 3625 = row 3625 := model_count_3625.trans row_count_3625.symm
def row_3626 : RowCertificate 3626 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3626 1) := by change ¬ Nat.Prime 52598757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17532919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3626 2) := by change ¬ Nat.Prime 52591505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10518301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3626 3) := by change ¬ Nat.Prime 52584253; exact PrattCertificates.composite_of_product (factor := 4177) (cofactor := 12589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3626 : row 3626 = 0 := (row_3626.checked).trans rfl
theorem model_count_3626 : modelRow 3626 = 0 := by decide
theorem modelrow_agrees_3626 : modelRow 3626 = row 3626 := model_count_3626.trans row_count_3626.symm
def row_3627 : RowCertificate 3627 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3627 1) := by change ¬ Nat.Prime 52627771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7518253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3627 2) := by change ¬ Nat.Prime 52620517; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 720829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3627 3) := by change Nat.Prime 52613263; exact ProjectEulerSolutions.P58PrimeNodes.p_52613263.prime
    exact decide_eq_true prime
theorem row_count_3627 : row 3627 = 1 := (row_3627.checked).trans rfl
theorem model_count_3627 : modelRow 3627 = 1 := by decide
theorem modelrow_agrees_3627 : modelRow 3627 = row 3627 := model_count_3627.trans row_count_3627.symm
def row_3628 : RowCertificate 3628 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3628 1) := by change ¬ Nat.Prime 52656793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7522399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3628 2) := by change ¬ Nat.Prime 52649537; exact PrattCertificates.composite_of_product (factor := 2297) (cofactor := 22921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3628 3) := by change ¬ Nat.Prime 52642281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17547427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3628 : row 3628 = 0 := (row_3628.checked).trans rfl
theorem model_count_3628 : modelRow 3628 = 0 := by decide
theorem modelrow_agrees_3628 : modelRow 3628 = row 3628 := model_count_3628.trans row_count_3628.symm
def row_3629 : RowCertificate 3629 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3629 1) := by change ¬ Nat.Prime 52685823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17561941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3629 2) := by change ¬ Nat.Prime 52678565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10535713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3629 3) := by change ¬ Nat.Prime 52671307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4051639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3629 : row 3629 = 0 := (row_3629.checked).trans rfl
theorem model_count_3629 : modelRow 3629 = 0 := by decide
theorem modelrow_agrees_3629 : modelRow 3629 = row 3629 := model_count_3629.trans row_count_3629.symm
def row_3630 : RowCertificate 3630 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3630 1) := by change ¬ Nat.Prime 52714861; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1225927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3630 2) := by change Nat.Prime 52707601; exact ProjectEulerSolutions.P58PrimeNodes.p_52707601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3630 3) := by change ¬ Nat.Prime 52700341; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1700011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3630 : row 3630 = 1 := (row_3630.checked).trans rfl
theorem model_count_3630 : modelRow 3630 = 1 := by decide
theorem modelrow_agrees_3630 : modelRow 3630 = row 3630 := model_count_3630.trans row_count_3630.symm
def row_3631 : RowCertificate 3631 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3631 1) := by change ¬ Nat.Prime 52743907; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1425511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3631 2) := by change ¬ Nat.Prime 52736645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10547329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3631 3) := by change ¬ Nat.Prime 52729383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17576461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3631 : row 3631 = 0 := (row_3631.checked).trans rfl
theorem model_count_3631 : modelRow 3631 = 0 := by decide
theorem modelrow_agrees_3631 : modelRow 3631 = row 3631 := model_count_3631.trans row_count_3631.symm
def row_3632 : RowCertificate 3632 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3632 1) := by change ¬ Nat.Prime 52772961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17590987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3632 2) := by change ¬ Nat.Prime 52765697; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 592873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3632 3) := by change ¬ Nat.Prime 52758433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7536919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3632 : row 3632 = 0 := (row_3632.checked).trans rfl
theorem model_count_3632 : modelRow 3632 = 0 := by decide
theorem modelrow_agrees_3632 : modelRow 3632 = row 3632 := model_count_3632.trans row_count_3632.symm
def row_3633 : RowCertificate 3633 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3633 1) := by change ¬ Nat.Prime 52802023; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 512641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3633 2) := by change ¬ Nat.Prime 52794757; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1287677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3633 3) := by change ¬ Nat.Prime 52787491; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2778289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3633 : row 3633 = 0 := (row_3633.checked).trans rfl
theorem model_count_3633 : modelRow 3633 = 0 := by decide
theorem modelrow_agrees_3633 : modelRow 3633 = row 3633 := model_count_3633.trans row_count_3633.symm
def row_3634 : RowCertificate 3634 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3634 1) := by change ¬ Nat.Prime 52831093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7547299) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3634 2) := by change ¬ Nat.Prime 52823825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10564765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3634 3) := by change ¬ Nat.Prime 52816557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17605519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3634 : row 3634 = 0 := (row_3634.checked).trans rfl
theorem model_count_3634 : modelRow 3634 = 0 := by decide
theorem modelrow_agrees_3634 : modelRow 3634 = row 3634 := model_count_3634.trans row_count_3634.symm
def row_3635 : RowCertificate 3635 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3635 1) := by change ¬ Nat.Prime 52860171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17620057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3635 2) := by change ¬ Nat.Prime 52852901; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 866441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3635 3) := by change ¬ Nat.Prime 52845631; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2781349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3635 : row 3635 = 0 := (row_3635.checked).trans rfl
theorem model_count_3635 : modelRow 3635 = 0 := by decide
theorem modelrow_agrees_3635 : modelRow 3635 = row 3635 := model_count_3635.trans row_count_3635.symm
def row_3636 : RowCertificate 3636 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3636 1) := by change ¬ Nat.Prime 52889257; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 867037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3636 2) := by change ¬ Nat.Prime 52881985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10576397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3636 3) := by change Nat.Prime 52874713; exact ProjectEulerSolutions.P58PrimeNodes.p_52874713.prime
    exact decide_eq_true prime
theorem row_count_3636 : row 3636 = 1 := (row_3636.checked).trans rfl
theorem model_count_3636 : modelRow 3636 = 1 := by decide
theorem modelrow_agrees_3636 : modelRow 3636 = row 3636 := model_count_3636.trans row_count_3636.symm
def row_3637 : RowCertificate 3637 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3637 1) := by change Nat.Prime 52918351; exact ProjectEulerSolutions.P58PrimeNodes.p_52918351.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3637 2) := by change ¬ Nat.Prime 52911077; exact PrattCertificates.composite_of_product (factor := 773) (cofactor := 68449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3637 3) := by change ¬ Nat.Prime 52903803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17634601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3637 : row 3637 = 1 := (row_3637.checked).trans rfl
theorem model_count_3637 : modelRow 3637 = 1 := by decide
theorem modelrow_agrees_3637 : modelRow 3637 = row 3637 := model_count_3637.trans row_count_3637.symm
def row_3638 : RowCertificate 3638 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3638 1) := by change ¬ Nat.Prime 52947453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17649151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3638 2) := by change ¬ Nat.Prime 52940177; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 136093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3638 3) := by change ¬ Nat.Prime 52932901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7561843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3638 : row 3638 = 0 := (row_3638.checked).trans rfl
theorem model_count_3638 : modelRow 3638 = 0 := by decide
theorem modelrow_agrees_3638 : modelRow 3638 = row 3638 := model_count_3638.trans row_count_3638.symm
def row_3639 : RowCertificate 3639 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3639 1) := by change ¬ Nat.Prime 52976563; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1431799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3639 2) := by change ¬ Nat.Prime 52969285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10593857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3639 3) := by change ¬ Nat.Prime 52962007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7566001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3639 : row 3639 = 0 := (row_3639.checked).trans rfl
theorem model_count_3639 : modelRow 3639 = 0 := by decide
theorem modelrow_agrees_3639 : modelRow 3639 = row 3639 := model_count_3639.trans row_count_3639.symm
def row_3640 : RowCertificate 3640 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3640 1) := by change ¬ Nat.Prime 53005681; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 351031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3640 2) := by change ¬ Nat.Prime 52998401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3117553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3640 3) := by change ¬ Nat.Prime 52991121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17663707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3640 : row 3640 = 0 := (row_3640.checked).trans rfl
theorem model_count_3640 : modelRow 3640 = 0 := by decide
theorem modelrow_agrees_3640 : modelRow 3640 = row 3640 := model_count_3640.trans row_count_3640.symm
def row_3641 : RowCertificate 3641 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3641 1) := by change ¬ Nat.Prime 53034807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17678269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3641 2) := by change ¬ Nat.Prime 53027525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10605505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3641 3) := by change ¬ Nat.Prime 53020243; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 68947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3641 : row 3641 = 0 := (row_3641.checked).trans rfl
theorem model_count_3641 : modelRow 3641 = 0 := by decide
theorem modelrow_agrees_3641 : modelRow 3641 = row 3641 := model_count_3641.trans row_count_3641.symm
def row_3642 : RowCertificate 3642 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3642 1) := by change ¬ Nat.Prime 53063941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7580563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3642 2) := by change ¬ Nat.Prime 53056657; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1001069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3642 3) := by change ¬ Nat.Prime 53049373; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4080721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3642 : row 3642 = 0 := (row_3642.checked).trans rfl
theorem model_count_3642 : modelRow 3642 = 0 := by decide
theorem modelrow_agrees_3642 : modelRow 3642 = row 3642 := model_count_3642.trans row_count_3642.symm
def row_3643 : RowCertificate 3643 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3643 1) := by change Nat.Prime 53093083; exact ProjectEulerSolutions.P58PrimeNodes.p_53093083.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3643 2) := by change Nat.Prime 53085797; exact ProjectEulerSolutions.P58PrimeNodes.p_53085797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3643 3) := by change ¬ Nat.Prime 53078511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17692837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3643 : row 3643 = 2 := (row_3643.checked).trans rfl
theorem model_count_3643 : modelRow 3643 = 2 := by decide
theorem modelrow_agrees_3643 : modelRow 3643 = row 3643 := model_count_3643.trans row_count_3643.symm
def row_3644 : RowCertificate 3644 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3644 1) := by change ¬ Nat.Prime 53122233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17707411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3644 2) := by change ¬ Nat.Prime 53114945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10622989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3644 3) := by change ¬ Nat.Prime 53107657; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 92041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3644 : row 3644 = 0 := (row_3644.checked).trans rfl
theorem model_count_3644 : modelRow 3644 = 0 := by decide
theorem modelrow_agrees_3644 : modelRow 3644 = row 3644 := model_count_3644.trans row_count_3644.symm
def row_3645 : RowCertificate 3645 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3645 1) := by change ¬ Nat.Prime 53151391; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1714561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3645 2) := by change ¬ Nat.Prime 53144101; exact PrattCertificates.composite_of_product (factor := 2549) (cofactor := 20849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3645 3) := by change ¬ Nat.Prime 53136811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7590973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3645 : row 3645 = 0 := (row_3645.checked).trans rfl
theorem model_count_3645 : modelRow 3645 = 0 := by decide
theorem modelrow_agrees_3645 : modelRow 3645 = row 3645 := model_count_3645.trans row_count_3645.symm
def row_3646 : RowCertificate 3646 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3646 1) := by change ¬ Nat.Prime 53180557; exact PrattCertificates.composite_of_product (factor := 3517) (cofactor := 15121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3646 2) := by change ¬ Nat.Prime 53173265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10634653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3646 3) := by change ¬ Nat.Prime 53165973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17721991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3646 : row 3646 = 0 := (row_3646.checked).trans rfl
theorem model_count_3646 : modelRow 3646 = 0 := by decide
theorem modelrow_agrees_3646 : modelRow 3646 = row 3646 := model_count_3646.trans row_count_3646.symm
def row_3647 : RowCertificate 3647 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3647 1) := by change ¬ Nat.Prime 53209731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17736577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3647 2) := by change ¬ Nat.Prime 53202437; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 220757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3647 3) := by change ¬ Nat.Prime 53195143; exact PrattCertificates.composite_of_product (factor := 3463) (cofactor := 15361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3647 : row 3647 = 0 := (row_3647.checked).trans rfl
theorem model_count_3647 : modelRow 3647 = 0 := by decide
theorem modelrow_agrees_3647 : modelRow 3647 = row 3647 := model_count_3647.trans row_count_3647.symm
def row_3648 : RowCertificate 3648 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3648 1) := by change ¬ Nat.Prime 53238913; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7605559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3648 2) := by change ¬ Nat.Prime 53231617; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1835573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3648 3) := by change Nat.Prime 53224321; exact ProjectEulerSolutions.P58PrimeNodes.p_53224321.prime
    exact decide_eq_true prime
theorem row_count_3648 : row 3648 = 1 := (row_3648.checked).trans rfl
theorem model_count_3648 : modelRow 3648 = 1 := by decide
theorem modelrow_agrees_3648 : modelRow 3648 = row 3648 := model_count_3648.trans row_count_3648.symm
def row_3649 : RowCertificate 3649 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3649 1) := by change ¬ Nat.Prime 53268103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7609729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3649 2) := by change ¬ Nat.Prime 53260805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10652161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3649 3) := by change ¬ Nat.Prime 53253507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17751169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3649 : row 3649 = 0 := (row_3649.checked).trans rfl
theorem model_count_3649 : modelRow 3649 = 0 := by decide
theorem modelrow_agrees_3649 : modelRow 3649 = row 3649 := model_count_3649.trans row_count_3649.symm
def row_3650 : RowCertificate 3650 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3650 1) := by change ¬ Nat.Prime 53297301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17765767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3650 2) := by change ¬ Nat.Prime 53290001; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 56873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3650 3) := by change ¬ Nat.Prime 53282701; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1440073) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3650 : row 3650 = 0 := (row_3650.checked).trans rfl
theorem model_count_3650 : modelRow 3650 = 0 := by decide
theorem modelrow_agrees_3650 : modelRow 3650 = row 3650 := model_count_3650.trans row_count_3650.symm
def row_3651 : RowCertificate 3651 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3651 1) := by change ¬ Nat.Prime 53326507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4102039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3651 2) := by change ¬ Nat.Prime 53319205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10663841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3651 3) := by change Nat.Prime 53311903; exact ProjectEulerSolutions.P58PrimeNodes.p_53311903.prime
    exact decide_eq_true prime
theorem row_count_3651 : row 3651 = 1 := (row_3651.checked).trans rfl
theorem model_count_3651 : modelRow 3651 = 1 := by decide
theorem modelrow_agrees_3651 : modelRow 3651 = row 3651 := model_count_3651.trans row_count_3651.symm
def row_3652 : RowCertificate 3652 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3652 1) := by change ¬ Nat.Prime 53355721; exact PrattCertificates.composite_of_product (factor := 4051) (cofactor := 13171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3652 2) := by change Nat.Prime 53348417; exact ProjectEulerSolutions.P58PrimeNodes.p_53348417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3652 3) := by change ¬ Nat.Prime 53341113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17780371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3652 : row 3652 = 1 := (row_3652.checked).trans rfl
theorem model_count_3652 : modelRow 3652 = 1 := by decide
theorem modelrow_agrees_3652 : modelRow 3652 = row 3652 := model_count_3652.trans row_count_3652.symm
def row_3653 : RowCertificate 3653 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3653 1) := by change ¬ Nat.Prime 53384943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17794981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3653 2) := by change ¬ Nat.Prime 53377637; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3139861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3653 3) := by change ¬ Nat.Prime 53370331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7624333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3653 : row 3653 = 0 := (row_3653.checked).trans rfl
theorem model_count_3653 : modelRow 3653 = 0 := by decide
theorem modelrow_agrees_3653 : modelRow 3653 = row 3653 := model_count_3653.trans row_count_3653.symm
def row_3654 : RowCertificate 3654 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3654 1) := by change ¬ Nat.Prime 53414173; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 731701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3654 2) := by change ¬ Nat.Prime 53406865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10681373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3654 3) := by change ¬ Nat.Prime 53399557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2810503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3654 : row 3654 = 0 := (row_3654.checked).trans rfl
theorem model_count_3654 : modelRow 3654 = 0 := by decide
theorem modelrow_agrees_3654 : modelRow 3654 = row 3654 := model_count_3654.trans row_count_3654.symm
def row_3655 : RowCertificate 3655 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3655 1) := by change ¬ Nat.Prime 53443411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7634773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3655 2) := by change ¬ Nat.Prime 53436101; exact PrattCertificates.composite_of_product (factor := 6277) (cofactor := 8513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3655 3) := by change ¬ Nat.Prime 53428791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17809597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3655 : row 3655 = 0 := (row_3655.checked).trans rfl
theorem model_count_3655 : modelRow 3655 = 0 := by decide
theorem modelrow_agrees_3655 : modelRow 3655 = row 3655 := model_count_3655.trans row_count_3655.symm
def row_3656 : RowCertificate 3656 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3656 1) := by change ¬ Nat.Prime 53472657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17824219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3656 2) := by change ¬ Nat.Prime 53465345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10693069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3656 3) := by change Nat.Prime 53458033; exact ProjectEulerSolutions.P58PrimeNodes.p_53458033.prime
    exact decide_eq_true prime
theorem row_count_3656 : row 3656 = 1 := (row_3656.checked).trans rfl
theorem model_count_3656 : modelRow 3656 = 1 := by decide
theorem modelrow_agrees_3656 : modelRow 3656 = row 3656 := model_count_3656.trans row_count_3656.symm
def row_3657 : RowCertificate 3657 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3657 1) := by change Nat.Prime 53501911; exact ProjectEulerSolutions.P58PrimeNodes.p_53501911.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3657 2) := by change ¬ Nat.Prime 53494597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4114969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3657 3) := by change ¬ Nat.Prime 53487283; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 189001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3657 : row 3657 = 1 := (row_3657.checked).trans rfl
theorem model_count_3657 : modelRow 3657 = 1 := by decide
theorem modelrow_agrees_3657 : modelRow 3657 = row 3657 := model_count_3657.trans row_count_3657.symm
def row_3658 : RowCertificate 3658 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3658 1) := by change ¬ Nat.Prime 53531173; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1244911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3658 2) := by change Nat.Prime 53523857; exact ProjectEulerSolutions.P58PrimeNodes.p_53523857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3658 3) := by change ¬ Nat.Prime 53516541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17838847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3658 : row 3658 = 1 := (row_3658.checked).trans rfl
theorem model_count_3658 : modelRow 3658 = 1 := by decide
theorem modelrow_agrees_3658 : modelRow 3658 = row 3658 := model_count_3658.trans row_count_3658.symm
def row_3659 : RowCertificate 3659 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3659 1) := by change ¬ Nat.Prime 53560443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17853481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3659 2) := by change ¬ Nat.Prime 53553125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10710625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3659 3) := by change ¬ Nat.Prime 53545807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7649401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3659 : row 3659 = 0 := (row_3659.checked).trans rfl
theorem model_count_3659 : modelRow 3659 = 0 := by decide
theorem modelrow_agrees_3659 : modelRow 3659 = row 3659 := model_count_3659.trans row_count_3659.symm
def row_3660 : RowCertificate 3660 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3660 1) := by change ¬ Nat.Prime 53589721; exact PrattCertificates.composite_of_product (factor := 1741) (cofactor := 30781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3660 2) := by change ¬ Nat.Prime 53582401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1847669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3660 3) := by change ¬ Nat.Prime 53575081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7653583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3660 : row 3660 = 0 := (row_3660.checked).trans rfl
theorem model_count_3660 : modelRow 3660 = 0 := by decide
theorem modelrow_agrees_3660 : modelRow 3660 = row 3660 := model_count_3660.trans row_count_3660.symm
def row_3661 : RowCertificate 3661 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3661 1) := by change ¬ Nat.Prime 53619007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4124539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3661 2) := by change ¬ Nat.Prime 53611685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10722337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3661 3) := by change ¬ Nat.Prime 53604363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17868121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3661 : row 3661 = 0 := (row_3661.checked).trans rfl
theorem model_count_3661 : modelRow 3661 = 0 := by decide
theorem modelrow_agrees_3661 : modelRow 3661 = row 3661 := model_count_3661.trans row_count_3661.symm
def row_3662 : RowCertificate 3662 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3662 1) := by change ¬ Nat.Prime 53648301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17882767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3662 2) := by change ¬ Nat.Prime 53640977; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4126229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3662 3) := by change ¬ Nat.Prime 53633653; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 678907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3662 : row 3662 = 0 := (row_3662.checked).trans rfl
theorem model_count_3662 : modelRow 3662 = 0 := by decide
theorem modelrow_agrees_3662 : modelRow 3662 = row 3662 := model_count_3662.trans row_count_3662.symm
def row_3663 : RowCertificate 3663 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3663 1) := by change ¬ Nat.Prime 53677603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7668229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3663 2) := by change Nat.Prime 53670277; exact ProjectEulerSolutions.P58PrimeNodes.p_53670277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3663 3) := by change ¬ Nat.Prime 53662951; exact PrattCertificates.composite_of_product (factor := 1783) (cofactor := 30097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3663 : row 3663 = 1 := (row_3663.checked).trans rfl
theorem model_count_3663 : modelRow 3663 = 1 := by decide
theorem modelrow_agrees_3663 : modelRow 3663 = row 3663 := model_count_3663.trans row_count_3663.symm
def row_3664 : RowCertificate 3664 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3664 1) := by change ¬ Nat.Prime 53706913; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4131301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3664 2) := by change ¬ Nat.Prime 53699585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10739917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3664 3) := by change ¬ Nat.Prime 53692257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17897419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3664 : row 3664 = 0 := (row_3664.checked).trans rfl
theorem model_count_3664 : modelRow 3664 = 0 := by decide
theorem modelrow_agrees_3664 : modelRow 3664 = row 3664 := model_count_3664.trans row_count_3664.symm
def row_3665 : RowCertificate 3665 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3665 1) := by change ¬ Nat.Prime 53736231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17912077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3665 2) := by change ¬ Nat.Prime 53728901; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1310461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3665 3) := by change ¬ Nat.Prime 53721571; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 69859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3665 : row 3665 = 0 := (row_3665.checked).trans rfl
theorem model_count_3665 : modelRow 3665 = 0 := by decide
theorem modelrow_agrees_3665 : modelRow 3665 = row 3665 := model_count_3665.trans row_count_3665.symm
def row_3666 : RowCertificate 3666 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3666 1) := by change ¬ Nat.Prime 53765557; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 802471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3666 2) := by change ¬ Nat.Prime 53758225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10751645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3666 3) := by change ¬ Nat.Prime 53750893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7678699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3666 : row 3666 = 0 := (row_3666.checked).trans rfl
theorem model_count_3666 : modelRow 3666 = 0 := by decide
theorem modelrow_agrees_3666 : modelRow 3666 = row 3666 := model_count_3666.trans row_count_3666.symm
def row_3667 : RowCertificate 3667 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3667 1) := by change Nat.Prime 53794891; exact ProjectEulerSolutions.P58PrimeNodes.p_53794891.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3667 2) := by change ¬ Nat.Prime 53787557; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 105673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3667 3) := by change ¬ Nat.Prime 53780223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17926741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3667 : row 3667 = 1 := (row_3667.checked).trans rfl
theorem model_count_3667 : modelRow 3667 = 1 := by decide
theorem modelrow_agrees_3667 : modelRow 3667 = row 3667 := model_count_3667.trans row_count_3667.symm
def row_3668 : RowCertificate 3668 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3668 1) := by change ¬ Nat.Prime 53824233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17941411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3668 2) := by change ¬ Nat.Prime 53816897; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 493733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3668 3) := by change ¬ Nat.Prime 53809561; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4139197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3668 : row 3668 = 0 := (row_3668.checked).trans rfl
theorem model_count_3668 : modelRow 3668 = 0 := by decide
theorem modelrow_agrees_3668 : modelRow 3668 = row 3668 := model_count_3668.trans row_count_3668.symm
def row_3669 : RowCertificate 3669 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3669 1) := by change ¬ Nat.Prime 53853583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7693369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3669 2) := by change ¬ Nat.Prime 53846245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10769249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3669 3) := by change ¬ Nat.Prime 53838907; exact PrattCertificates.composite_of_product (factor := 3391) (cofactor := 15877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3669 : row 3669 = 0 := (row_3669.checked).trans rfl
theorem model_count_3669 : modelRow 3669 = 0 := by decide
theorem modelrow_agrees_3669 : modelRow 3669 = row 3669 := model_count_3669.trans row_count_3669.symm
def row_3670 : RowCertificate 3670 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3670 1) := by change ¬ Nat.Prime 53882941; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7697563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3670 2) := by change ¬ Nat.Prime 53875601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4144277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3670 3) := by change ¬ Nat.Prime 53868261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17956087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3670 : row 3670 = 0 := (row_3670.checked).trans rfl
theorem model_count_3670 : modelRow 3670 = 0 := by decide
theorem modelrow_agrees_3670 : modelRow 3670 = row 3670 := model_count_3670.trans row_count_3670.symm
def row_3671 : RowCertificate 3671 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3671 1) := by change ¬ Nat.Prime 53912307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17970769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3671 2) := by change ¬ Nat.Prime 53904965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10780993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3671 3) := by change ¬ Nat.Prime 53897623; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4145971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3671 : row 3671 = 0 := (row_3671.checked).trans rfl
theorem model_count_3671 : modelRow 3671 = 0 := by decide
theorem modelrow_agrees_3671 : modelRow 3671 = row 3671 := model_count_3671.trans row_count_3671.symm
def row_3672 : RowCertificate 3672 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3672 1) := by change Nat.Prime 53941681; exact ProjectEulerSolutions.P58PrimeNodes.p_53941681.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3672 2) := by change ¬ Nat.Prime 53934337; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1017629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3672 3) := by change Nat.Prime 53926993; exact ProjectEulerSolutions.P58PrimeNodes.p_53926993.prime
    exact decide_eq_true prime
theorem row_count_3672 : row 3672 = 2 := (row_3672.checked).trans rfl
theorem model_count_3672 : modelRow 3672 = 2 := by decide
theorem modelrow_agrees_3672 : modelRow 3672 = row 3672 := model_count_3672.trans row_count_3672.symm
def row_3673 : RowCertificate 3673 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3673 1) := by change ¬ Nat.Prime 53971063; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1255141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3673 2) := by change ¬ Nat.Prime 53963717; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 739229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3673 3) := by change ¬ Nat.Prime 53956371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 17985457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3673 : row 3673 = 0 := (row_3673.checked).trans rfl
theorem model_count_3673 : modelRow 3673 = 0 := by decide
theorem modelrow_agrees_3673 : modelRow 3673 = row 3673 := model_count_3673.trans row_count_3673.symm
def row_3674 : RowCertificate 3674 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3674 1) := by change ¬ Nat.Prime 54000453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18000151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3674 2) := by change ¬ Nat.Prime 53993105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10798621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3674 3) := by change ¬ Nat.Prime 53985757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7712251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3674 : row 3674 = 0 := (row_3674.checked).trans rfl
theorem model_count_3674 : modelRow 3674 = 0 := by decide
theorem modelrow_agrees_3674 : modelRow 3674 = row 3674 := model_count_3674.trans row_count_3674.symm
def row_3675 : RowCertificate 3675 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3675 1) := by change ¬ Nat.Prime 54029851; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 175993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3675 2) := by change ¬ Nat.Prime 54022501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4155577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3675 3) := by change ¬ Nat.Prime 54015151; exact PrattCertificates.composite_of_product (factor := 6967) (cofactor := 7753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3675 : row 3675 = 0 := (row_3675.checked).trans rfl
theorem model_count_3675 : modelRow 3675 = 0 := by decide
theorem modelrow_agrees_3675 : modelRow 3675 = row 3675 := model_count_3675.trans row_count_3675.symm
def row_3676 : RowCertificate 3676 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3676 1) := by change ¬ Nat.Prime 54059257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7722751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3676 2) := by change ¬ Nat.Prime 54051905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10810381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3676 3) := by change ¬ Nat.Prime 54044553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18014851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3676 : row 3676 = 0 := (row_3676.checked).trans rfl
theorem model_count_3676 : modelRow 3676 = 0 := by decide
theorem modelrow_agrees_3676 : modelRow 3676 = row 3676 := model_count_3676.trans row_count_3676.symm
def row_3677 : RowCertificate 3677 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3677 1) := by change ¬ Nat.Prime 54088671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18029557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3677 2) := by change ¬ Nat.Prime 54081317; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1864873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3677 3) := by change ¬ Nat.Prime 54073963; exact PrattCertificates.composite_of_product (factor := 4549) (cofactor := 11887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3677 : row 3677 = 0 := (row_3677.checked).trans rfl
theorem model_count_3677 : modelRow 3677 = 0 := by decide
theorem modelrow_agrees_3677 : modelRow 3677 = row 3677 := model_count_3677.trans row_count_3677.symm
def row_3678 : RowCertificate 3678 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3678 1) := by change Nat.Prime 54118093; exact ProjectEulerSolutions.P58PrimeNodes.p_54118093.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3678 2) := by change ¬ Nat.Prime 54110737; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 145069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3678 3) := by change Nat.Prime 54103381; exact ProjectEulerSolutions.P58PrimeNodes.p_54103381.prime
    exact decide_eq_true prime
theorem row_count_3678 : row 3678 = 2 := (row_3678.checked).trans rfl
theorem model_count_3678 : modelRow 3678 = 2 := by decide
theorem modelrow_agrees_3678 : modelRow 3678 = row 3678 := model_count_3678.trans row_count_3678.symm
def row_3679 : RowCertificate 3679 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3679 1) := by change Nat.Prime 54147523; exact ProjectEulerSolutions.P58PrimeNodes.p_54147523.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3679 2) := by change ¬ Nat.Prime 54140165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10828033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3679 3) := by change ¬ Nat.Prime 54132807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18044269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3679 : row 3679 = 1 := (row_3679.checked).trans rfl
theorem model_count_3679 : modelRow 3679 = 1 := by decide
theorem modelrow_agrees_3679 : modelRow 3679 = row 3679 := model_count_3679.trans row_count_3679.symm
def row_3680 : RowCertificate 3680 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3680 1) := by change ¬ Nat.Prime 54176961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18058987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3680 2) := by change ¬ Nat.Prime 54169601; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 479377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3680 3) := by change ¬ Nat.Prime 54162241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7737463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3680 : row 3680 = 0 := (row_3680.checked).trans rfl
theorem model_count_3680 : modelRow 3680 = 0 := by decide
theorem modelrow_agrees_3680 : modelRow 3680 = row 3680 := model_count_3680.trans row_count_3680.symm
def row_3681 : RowCertificate 3681 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3681 1) := by change ¬ Nat.Prime 54206407; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 195691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3681 2) := by change ¬ Nat.Prime 54199045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10839809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3681 3) := by change ¬ Nat.Prime 54191683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7741669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3681 : row 3681 = 0 := (row_3681.checked).trans rfl
theorem model_count_3681 : modelRow 3681 = 0 := by decide
theorem modelrow_agrees_3681 : modelRow 3681 = row 3681 := model_count_3681.trans row_count_3681.symm
def row_3682 : RowCertificate 3682 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3682 1) := by change ¬ Nat.Prime 54235861; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2854519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3682 2) := by change Nat.Prime 54228497; exact ProjectEulerSolutions.P58PrimeNodes.p_54228497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3682 3) := by change ¬ Nat.Prime 54221133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18073711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3682 : row 3682 = 1 := (row_3682.checked).trans rfl
theorem model_count_3682 : modelRow 3682 = 1 := by decide
theorem modelrow_agrees_3682 : modelRow 3682 = row 3682 := model_count_3682.trans row_count_3682.symm
def row_3683 : RowCertificate 3683 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3683 1) := by change ¬ Nat.Prime 54265323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18088441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3683 2) := by change ¬ Nat.Prime 54257957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4173689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3683 3) := by change Nat.Prime 54250591; exact ProjectEulerSolutions.P58PrimeNodes.p_54250591.prime
    exact decide_eq_true prime
theorem row_count_3683 : row 3683 = 1 := (row_3683.checked).trans rfl
theorem model_count_3683 : modelRow 3683 = 1 := by decide
theorem modelrow_agrees_3683 : modelRow 3683 = row 3683 := model_count_3683.trans row_count_3683.symm
def row_3684 : RowCertificate 3684 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3684 1) := by change ¬ Nat.Prime 54294793; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7756399) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3684 2) := by change ¬ Nat.Prime 54287425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10857485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3684 3) := by change ¬ Nat.Prime 54280057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4175389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3684 : row 3684 = 0 := (row_3684.checked).trans rfl
theorem model_count_3684 : modelRow 3684 = 0 := by decide
theorem modelrow_agrees_3684 : modelRow 3684 = row 3684 := model_count_3684.trans row_count_3684.symm
def row_3685 : RowCertificate 3685 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3685 1) := by change ¬ Nat.Prime 54324271; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 687649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3685 2) := by change ¬ Nat.Prime 54316901; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 890441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3685 3) := by change ¬ Nat.Prime 54309531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18103177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3685 : row 3685 = 0 := (row_3685.checked).trans rfl
theorem model_count_3685 : modelRow 3685 = 0 := by decide
theorem modelrow_agrees_3685 : modelRow 3685 = row 3685 := model_count_3685.trans row_count_3685.symm
def row_3686 : RowCertificate 3686 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3686 1) := by change ¬ Nat.Prime 54353757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18117919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3686 2) := by change ¬ Nat.Prime 54346385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10869277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3686 3) := by change ¬ Nat.Prime 54339013; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 145681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3686 : row 3686 = 0 := (row_3686.checked).trans rfl
theorem model_count_3686 : modelRow 3686 = 0 := by decide
theorem modelrow_agrees_3686 : modelRow 3686 = row 3686 := model_count_3686.trans row_count_3686.symm
def row_3687 : RowCertificate 3687 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3687 1) := by change ¬ Nat.Prime 54383251; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4183327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3687 2) := by change ¬ Nat.Prime 54375877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3198581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3687 3) := by change ¬ Nat.Prime 54368503; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7766929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3687 : row 3687 = 0 := (row_3687.checked).trans rfl
theorem model_count_3687 : modelRow 3687 = 0 := by decide
theorem modelrow_agrees_3687 : modelRow 3687 = row 3687 := model_count_3687.trans row_count_3687.symm
def row_3688 : RowCertificate 3688 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3688 1) := by change ¬ Nat.Prime 54412753; exact PrattCertificates.composite_of_product (factor := 1483) (cofactor := 36691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3688 2) := by change ¬ Nat.Prime 54405377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4185029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3688 3) := by change ¬ Nat.Prime 54398001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18132667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3688 : row 3688 = 0 := (row_3688.checked).trans rfl
theorem model_count_3688 : modelRow 3688 = 0 := by decide
theorem modelrow_agrees_3688 : modelRow 3688 = row 3688 := model_count_3688.trans row_count_3688.symm
def row_3689 : RowCertificate 3689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3689 1) := by change ¬ Nat.Prime 54442263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18147421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3689 2) := by change ¬ Nat.Prime 54434885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10886977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3689 3) := by change ¬ Nat.Prime 54427507; exact PrattCertificates.composite_of_product (factor := 1873) (cofactor := 29059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3689 : row 3689 = 0 := (row_3689.checked).trans rfl
theorem model_count_3689 : modelRow 3689 = 0 := by decide
theorem modelrow_agrees_3689 : modelRow 3689 = row 3689 := model_count_3689.trans row_count_3689.symm
def row_3690 : RowCertificate 3690 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3690 1) := by change ¬ Nat.Prime 54471781; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7781683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3690 2) := by change ¬ Nat.Prime 54464401; exact PrattCertificates.composite_of_product (factor := 1217) (cofactor := 44753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3690 3) := by change ¬ Nat.Prime 54457021; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2866159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3690 : row 3690 = 0 := (row_3690.checked).trans rfl
theorem model_count_3690 : modelRow 3690 = 0 := by decide
theorem modelrow_agrees_3690 : modelRow 3690 = row 3690 := model_count_3690.trans row_count_3690.symm
def row_3691 : RowCertificate 3691 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3691 1) := by change ¬ Nat.Prime 54501307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7785901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3691 2) := by change ¬ Nat.Prime 54493925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10898785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3691 3) := by change ¬ Nat.Prime 54486543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18162181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3691 : row 3691 = 0 := (row_3691.checked).trans rfl
theorem model_count_3691 : modelRow 3691 = 0 := by decide
theorem modelrow_agrees_3691 : modelRow 3691 = row 3691 := model_count_3691.trans row_count_3691.symm
def row_3692 : RowCertificate 3692 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3692 1) := by change ¬ Nat.Prime 54530841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18176947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3692 2) := by change Nat.Prime 54523457; exact ProjectEulerSolutions.P58PrimeNodes.p_54523457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3692 3) := by change ¬ Nat.Prime 54516073; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2869267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3692 : row 3692 = 1 := (row_3692.checked).trans rfl
theorem model_count_3692 : modelRow 3692 = 1 := by decide
theorem modelrow_agrees_3692 : modelRow 3692 = row 3692 := model_count_3692.trans row_count_3692.symm
def row_3693 : RowCertificate 3693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3693 1) := by change ¬ Nat.Prime 54560383; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 117841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3693 2) := by change ¬ Nat.Prime 54552997; exact PrattCertificates.composite_of_product (factor := 1901) (cofactor := 28697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3693 3) := by change ¬ Nat.Prime 54545611; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 429493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3693 : row 3693 = 0 := (row_3693.checked).trans rfl
theorem model_count_3693 : modelRow 3693 = 0 := by decide
theorem modelrow_agrees_3693 : modelRow 3693 = row 3693 := model_count_3693.trans row_count_3693.symm
def row_3694 : RowCertificate 3694 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3694 1) := by change Nat.Prime 54589933; exact ProjectEulerSolutions.P58PrimeNodes.p_54589933.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3694 2) := by change ¬ Nat.Prime 54582545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10916509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3694 3) := by change ¬ Nat.Prime 54575157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18191719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3694 : row 3694 = 1 := (row_3694.checked).trans rfl
theorem model_count_3694 : modelRow 3694 = 1 := by decide
theorem modelrow_agrees_3694 : modelRow 3694 = row 3694 := model_count_3694.trans row_count_3694.symm
def row_3695 : RowCertificate 3695 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3695 1) := by change ¬ Nat.Prime 54619491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18206497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3695 2) := by change ¬ Nat.Prime 54612101; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1030417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3695 3) := by change ¬ Nat.Prime 54604711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7800673) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3695 : row 3695 = 0 := (row_3695.checked).trans rfl
theorem model_count_3695 : modelRow 3695 = 0 := by decide
theorem modelrow_agrees_3695 : modelRow 3695 = row 3695 := model_count_3695.trans row_count_3695.symm
def row_3696 : RowCertificate 3696 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3696 1) := by change Nat.Prime 54649057; exact ProjectEulerSolutions.P58PrimeNodes.p_54649057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3696 2) := by change ¬ Nat.Prime 54641665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10928333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3696 3) := by change ¬ Nat.Prime 54634273; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 347989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3696 : row 3696 = 1 := (row_3696.checked).trans rfl
theorem model_count_3696 : modelRow 3696 = 1 := by decide
theorem modelrow_agrees_3696 : modelRow 3696 = row 3696 := model_count_3696.trans row_count_3696.symm
def row_3697 : RowCertificate 3697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3697 1) := by change ¬ Nat.Prime 54678631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7811233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3697 2) := by change ¬ Nat.Prime 54671237; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1477601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3697 3) := by change ¬ Nat.Prime 54663843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18221281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3697 : row 3697 = 0 := (row_3697.checked).trans rfl
theorem model_count_3697 : modelRow 3697 = 0 := by decide
theorem modelrow_agrees_3697 : modelRow 3697 = row 3697 := model_count_3697.trans row_count_3697.symm
def row_3698 : RowCertificate 3698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3698 1) := by change ¬ Nat.Prime 54708213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18236071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3698 2) := by change ¬ Nat.Prime 54700817; exact PrattCertificates.composite_of_product (factor := 7129) (cofactor := 7673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3698 3) := by change ¬ Nat.Prime 54693421; exact PrattCertificates.composite_of_product (factor := 4357) (cofactor := 12553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3698 : row 3698 = 0 := (row_3698.checked).trans rfl
theorem model_count_3698 : modelRow 3698 = 0 := by decide
theorem modelrow_agrees_3698 : modelRow 3698 = row 3698 := model_count_3698.trans row_count_3698.symm
def row_3699 : RowCertificate 3699 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3699 1) := by change ¬ Nat.Prime 54737803; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2880937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3699 2) := by change ¬ Nat.Prime 54730405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10946081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3699 3) := by change Nat.Prime 54723007; exact ProjectEulerSolutions.P58PrimeNodes.p_54723007.prime
    exact decide_eq_true prime
theorem row_count_3699 : row 3699 = 1 := (row_3699.checked).trans rfl
theorem model_count_3699 : modelRow 3699 = 1 := by decide
theorem modelrow_agrees_3699 : modelRow 3699 = row 3699 := model_count_3699.trans row_count_3699.symm
def row_3700 : RowCertificate 3700 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3700 1) := by change ¬ Nat.Prime 54767401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4212877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3700 2) := by change ¬ Nat.Prime 54760001; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 750137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3700 3) := by change ¬ Nat.Prime 54752601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18250867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3700 : row 3700 = 0 := (row_3700.checked).trans rfl
theorem model_count_3700 : modelRow 3700 = 0 := by decide
theorem modelrow_agrees_3700 : modelRow 3700 = row 3700 := model_count_3700.trans row_count_3700.symm
def row_3701 : RowCertificate 3701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3701 1) := by change ¬ Nat.Prime 54797007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18265669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3701 2) := by change ¬ Nat.Prime 54789605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10957921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3701 3) := by change ¬ Nat.Prime 54782203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7826029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3701 : row 3701 = 0 := (row_3701.checked).trans rfl
theorem model_count_3701 : modelRow 3701 = 0 := by decide
theorem modelrow_agrees_3701 : modelRow 3701 = row 3701 := model_count_3701.trans row_count_3701.symm
def row_3702 : RowCertificate 3702 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3702 1) := by change ¬ Nat.Prime 54826621; exact PrattCertificates.composite_of_product (factor := 919) (cofactor := 59659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3702 2) := by change Nat.Prime 54819217; exact ProjectEulerSolutions.P58PrimeNodes.p_54819217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3702 3) := by change ¬ Nat.Prime 54811813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7830259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3702 : row 3702 = 1 := (row_3702.checked).trans rfl
theorem model_count_3702 : modelRow 3702 = 1 := by decide
theorem modelrow_agrees_3702 : modelRow 3702 = row 3702 := model_count_3702.trans row_count_3702.symm
def row_3703 : RowCertificate 3703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3703 1) := by change ¬ Nat.Prime 54856243; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4219711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3703 2) := by change ¬ Nat.Prime 54848837; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1482401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3703 3) := by change ¬ Nat.Prime 54841431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18280477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3703 : row 3703 = 0 := (row_3703.checked).trans rfl
theorem model_count_3703 : modelRow 3703 = 0 := by decide
theorem modelrow_agrees_3703 : modelRow 3703 = row 3703 := model_count_3703.trans row_count_3703.symm
def row_3704 : RowCertificate 3704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3704 1) := by change ¬ Nat.Prime 54885873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18295291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3704 2) := by change ¬ Nat.Prime 54878465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10975693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3704 3) := by change ¬ Nat.Prime 54871057; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 818971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3704 : row 3704 = 0 := (row_3704.checked).trans rfl
theorem model_count_3704 : modelRow 3704 = 0 := by decide
theorem modelrow_agrees_3704 : modelRow 3704 = row 3704 := model_count_3704.trans row_count_3704.symm
def row_3705 : RowCertificate 3705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3705 1) := by change ¬ Nat.Prime 54915511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7845073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3705 2) := by change Nat.Prime 54908101; exact ProjectEulerSolutions.P58PrimeNodes.p_54908101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3705 3) := by change ¬ Nat.Prime 54900691; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 394969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3705 : row 3705 = 1 := (row_3705.checked).trans rfl
theorem model_count_3705 : modelRow 3705 = 1 := by decide
theorem modelrow_agrees_3705 : modelRow 3705 = row 3705 := model_count_3705.trans row_count_3705.symm
def row_3706 : RowCertificate 3706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3706 1) := by change ¬ Nat.Prime 54945157; exact PrattCertificates.composite_of_product (factor := 2683) (cofactor := 20479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3706 2) := by change ¬ Nat.Prime 54937745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 10987549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3706 3) := by change ¬ Nat.Prime 54930333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18310111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3706 : row 3706 = 0 := (row_3706.checked).trans rfl
theorem model_count_3706 : modelRow 3706 = 0 := by decide
theorem modelrow_agrees_3706 : modelRow 3706 = row 3706 := model_count_3706.trans row_count_3706.symm
def row_3707 : RowCertificate 3707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3707 1) := by change ¬ Nat.Prime 54974811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18324937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3707 2) := by change Nat.Prime 54967397; exact ProjectEulerSolutions.P58PrimeNodes.p_54967397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3707 3) := by change ¬ Nat.Prime 54959983; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4227691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3707 : row 3707 = 1 := (row_3707.checked).trans rfl
theorem model_count_3707 : modelRow 3707 = 1 := by decide
theorem modelrow_agrees_3707 : modelRow 3707 = row 3707 := model_count_3707.trans row_count_3707.symm
def row_3708 : RowCertificate 3708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3708 1) := by change ¬ Nat.Prime 55004473; exact PrattCertificates.composite_of_product (factor := 1297) (cofactor := 42409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3708 2) := by change ¬ Nat.Prime 54997057; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3235121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3708 3) := by change ¬ Nat.Prime 54989641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7855663) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3708 : row 3708 = 0 := (row_3708.checked).trans rfl
theorem model_count_3708 : modelRow 3708 = 0 := by decide
theorem modelrow_agrees_3708 : modelRow 3708 = row 3708 := model_count_3708.trans row_count_3708.symm
def row_3709 : RowCertificate 3709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3709 1) := by change ¬ Nat.Prime 55034143; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 285151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3709 2) := by change ¬ Nat.Prime 55026725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11005345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3709 3) := by change ¬ Nat.Prime 55019307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18339769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3709 : row 3709 = 0 := (row_3709.checked).trans rfl
theorem model_count_3709 : modelRow 3709 = 0 := by decide
theorem modelrow_agrees_3709 : modelRow 3709 = row 3709 := model_count_3709.trans row_count_3709.symm
def row_3710 : RowCertificate 3710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3710 1) := by change ¬ Nat.Prime 55063821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18354607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3710 2) := by change Nat.Prime 55056401; exact ProjectEulerSolutions.P58PrimeNodes.p_55056401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3710 3) := by change ¬ Nat.Prime 55048981; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4234537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3710 : row 3710 = 1 := (row_3710.checked).trans rfl
theorem model_count_3710 : modelRow 3710 = 1 := by decide
theorem modelrow_agrees_3710 : modelRow 3710 = row 3710 := model_count_3710.trans row_count_3710.symm
def row_3711 : RowCertificate 3711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3711 1) := by change ¬ Nat.Prime 55093507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7870501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3711 2) := by change ¬ Nat.Prime 55086085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11017217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3711 3) := by change ¬ Nat.Prime 55078663; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2898877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3711 : row 3711 = 0 := (row_3711.checked).trans rfl
theorem model_count_3711 : modelRow 3711 = 0 := by decide
theorem modelrow_agrees_3711 : modelRow 3711 = row 3711 := model_count_3711.trans row_count_3711.symm
def row_3712 : RowCertificate 3712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3712 1) := by change ¬ Nat.Prime 55123201; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7874743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3712 2) := by change ¬ Nat.Prime 55115777; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 119557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3712 3) := by change ¬ Nat.Prime 55108353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18369451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3712 : row 3712 = 0 := (row_3712.checked).trans rfl
theorem model_count_3712 : modelRow 3712 = 0 := by decide
theorem modelrow_agrees_3712 : modelRow 3712 = row 3712 := model_count_3712.trans row_count_3712.symm
private def atomic_3585 : CountSegment 1 10 13120 3584 3585 1697 1697 := CountSegment.single (first := 3584) (initial := 1697) row_count_3585 (by decide)
private theorem atomic_3585_agreement : RowAgreement 3584 3585 := RowAgreement.single (first := 3584) modelrow_agrees_3585
private def atomic_3586 : CountSegment 1 10 13120 3585 3586 1697 1697 := CountSegment.single (first := 3585) (initial := 1697) row_count_3586 (by decide)
private theorem atomic_3586_agreement : RowAgreement 3585 3586 := RowAgreement.single (first := 3585) modelrow_agrees_3586
private def atomic_3587 : CountSegment 1 10 13120 3586 3587 1697 1697 := CountSegment.single (first := 3586) (initial := 1697) row_count_3587 (by decide)
private theorem atomic_3587_agreement : RowAgreement 3586 3587 := RowAgreement.single (first := 3586) modelrow_agrees_3587
private def atomic_3588 : CountSegment 1 10 13120 3587 3588 1697 1697 := CountSegment.single (first := 3587) (initial := 1697) row_count_3588 (by decide)
private theorem atomic_3588_agreement : RowAgreement 3587 3588 := RowAgreement.single (first := 3587) modelrow_agrees_3588
private def atomic_3589 : CountSegment 1 10 13120 3588 3589 1697 1697 := CountSegment.single (first := 3588) (initial := 1697) row_count_3589 (by decide)
private theorem atomic_3589_agreement : RowAgreement 3588 3589 := RowAgreement.single (first := 3588) modelrow_agrees_3589
private def atomic_3590 : CountSegment 1 10 13120 3589 3590 1697 1697 := CountSegment.single (first := 3589) (initial := 1697) row_count_3590 (by decide)
private theorem atomic_3590_agreement : RowAgreement 3589 3590 := RowAgreement.single (first := 3589) modelrow_agrees_3590
private def atomic_3591 : CountSegment 1 10 13120 3590 3591 1697 1699 := CountSegment.single (first := 3590) (initial := 1697) row_count_3591 (by decide)
private theorem atomic_3591_agreement : RowAgreement 3590 3591 := RowAgreement.single (first := 3590) modelrow_agrees_3591
private def atomic_3592 : CountSegment 1 10 13120 3591 3592 1699 1699 := CountSegment.single (first := 3591) (initial := 1699) row_count_3592 (by decide)
private theorem atomic_3592_agreement : RowAgreement 3591 3592 := RowAgreement.single (first := 3591) modelrow_agrees_3592
private def atomic_3593 : CountSegment 1 10 13120 3592 3593 1699 1699 := CountSegment.single (first := 3592) (initial := 1699) row_count_3593 (by decide)
private theorem atomic_3593_agreement : RowAgreement 3592 3593 := RowAgreement.single (first := 3592) modelrow_agrees_3593
private def atomic_3594 : CountSegment 1 10 13120 3593 3594 1699 1699 := CountSegment.single (first := 3593) (initial := 1699) row_count_3594 (by decide)
private theorem atomic_3594_agreement : RowAgreement 3593 3594 := RowAgreement.single (first := 3593) modelrow_agrees_3594
private def atomic_3595 : CountSegment 1 10 13120 3594 3595 1699 1701 := CountSegment.single (first := 3594) (initial := 1699) row_count_3595 (by decide)
private theorem atomic_3595_agreement : RowAgreement 3594 3595 := RowAgreement.single (first := 3594) modelrow_agrees_3595
private def atomic_3596 : CountSegment 1 10 13120 3595 3596 1701 1701 := CountSegment.single (first := 3595) (initial := 1701) row_count_3596 (by decide)
private theorem atomic_3596_agreement : RowAgreement 3595 3596 := RowAgreement.single (first := 3595) modelrow_agrees_3596
private def atomic_3597 : CountSegment 1 10 13120 3596 3597 1701 1701 := CountSegment.single (first := 3596) (initial := 1701) row_count_3597 (by decide)
private theorem atomic_3597_agreement : RowAgreement 3596 3597 := RowAgreement.single (first := 3596) modelrow_agrees_3597
private def atomic_3598 : CountSegment 1 10 13120 3597 3598 1701 1702 := CountSegment.single (first := 3597) (initial := 1701) row_count_3598 (by decide)
private theorem atomic_3598_agreement : RowAgreement 3597 3598 := RowAgreement.single (first := 3597) modelrow_agrees_3598
private def atomic_3599 : CountSegment 1 10 13120 3598 3599 1702 1702 := CountSegment.single (first := 3598) (initial := 1702) row_count_3599 (by decide)
private theorem atomic_3599_agreement : RowAgreement 3598 3599 := RowAgreement.single (first := 3598) modelrow_agrees_3599
private def atomic_3600 : CountSegment 1 10 13120 3599 3600 1702 1703 := CountSegment.single (first := 3599) (initial := 1702) row_count_3600 (by decide)
private theorem atomic_3600_agreement : RowAgreement 3599 3600 := RowAgreement.single (first := 3599) modelrow_agrees_3600
private def atomic_3601 : CountSegment 1 10 13120 3600 3601 1703 1703 := CountSegment.single (first := 3600) (initial := 1703) row_count_3601 (by decide)
private theorem atomic_3601_agreement : RowAgreement 3600 3601 := RowAgreement.single (first := 3600) modelrow_agrees_3601
private def atomic_3602 : CountSegment 1 10 13120 3601 3602 1703 1704 := CountSegment.single (first := 3601) (initial := 1703) row_count_3602 (by decide)
private theorem atomic_3602_agreement : RowAgreement 3601 3602 := RowAgreement.single (first := 3601) modelrow_agrees_3602
private def atomic_3603 : CountSegment 1 10 13120 3602 3603 1704 1704 := CountSegment.single (first := 3602) (initial := 1704) row_count_3603 (by decide)
private theorem atomic_3603_agreement : RowAgreement 3602 3603 := RowAgreement.single (first := 3602) modelrow_agrees_3603
private def atomic_3604 : CountSegment 1 10 13120 3603 3604 1704 1704 := CountSegment.single (first := 3603) (initial := 1704) row_count_3604 (by decide)
private theorem atomic_3604_agreement : RowAgreement 3603 3604 := RowAgreement.single (first := 3603) modelrow_agrees_3604
private def atomic_3605 : CountSegment 1 10 13120 3604 3605 1704 1704 := CountSegment.single (first := 3604) (initial := 1704) row_count_3605 (by decide)
private theorem atomic_3605_agreement : RowAgreement 3604 3605 := RowAgreement.single (first := 3604) modelrow_agrees_3605
private def atomic_3606 : CountSegment 1 10 13120 3605 3606 1704 1704 := CountSegment.single (first := 3605) (initial := 1704) row_count_3606 (by decide)
private theorem atomic_3606_agreement : RowAgreement 3605 3606 := RowAgreement.single (first := 3605) modelrow_agrees_3606
private def atomic_3607 : CountSegment 1 10 13120 3606 3607 1704 1704 := CountSegment.single (first := 3606) (initial := 1704) row_count_3607 (by decide)
private theorem atomic_3607_agreement : RowAgreement 3606 3607 := RowAgreement.single (first := 3606) modelrow_agrees_3607
private def atomic_3608 : CountSegment 1 10 13120 3607 3608 1704 1705 := CountSegment.single (first := 3607) (initial := 1704) row_count_3608 (by decide)
private theorem atomic_3608_agreement : RowAgreement 3607 3608 := RowAgreement.single (first := 3607) modelrow_agrees_3608
private def atomic_3609 : CountSegment 1 10 13120 3608 3609 1705 1705 := CountSegment.single (first := 3608) (initial := 1705) row_count_3609 (by decide)
private theorem atomic_3609_agreement : RowAgreement 3608 3609 := RowAgreement.single (first := 3608) modelrow_agrees_3609
private def atomic_3610 : CountSegment 1 10 13120 3609 3610 1705 1706 := CountSegment.single (first := 3609) (initial := 1705) row_count_3610 (by decide)
private theorem atomic_3610_agreement : RowAgreement 3609 3610 := RowAgreement.single (first := 3609) modelrow_agrees_3610
private def atomic_3611 : CountSegment 1 10 13120 3610 3611 1706 1706 := CountSegment.single (first := 3610) (initial := 1706) row_count_3611 (by decide)
private theorem atomic_3611_agreement : RowAgreement 3610 3611 := RowAgreement.single (first := 3610) modelrow_agrees_3611
private def atomic_3612 : CountSegment 1 10 13120 3611 3612 1706 1707 := CountSegment.single (first := 3611) (initial := 1706) row_count_3612 (by decide)
private theorem atomic_3612_agreement : RowAgreement 3611 3612 := RowAgreement.single (first := 3611) modelrow_agrees_3612
private def atomic_3613 : CountSegment 1 10 13120 3612 3613 1707 1707 := CountSegment.single (first := 3612) (initial := 1707) row_count_3613 (by decide)
private theorem atomic_3613_agreement : RowAgreement 3612 3613 := RowAgreement.single (first := 3612) modelrow_agrees_3613
private def atomic_3614 : CountSegment 1 10 13120 3613 3614 1707 1707 := CountSegment.single (first := 3613) (initial := 1707) row_count_3614 (by decide)
private theorem atomic_3614_agreement : RowAgreement 3613 3614 := RowAgreement.single (first := 3613) modelrow_agrees_3614
private def atomic_3615 : CountSegment 1 10 13120 3614 3615 1707 1708 := CountSegment.single (first := 3614) (initial := 1707) row_count_3615 (by decide)
private theorem atomic_3615_agreement : RowAgreement 3614 3615 := RowAgreement.single (first := 3614) modelrow_agrees_3615
private def atomic_3616 : CountSegment 1 10 13120 3615 3616 1708 1708 := CountSegment.single (first := 3615) (initial := 1708) row_count_3616 (by decide)
private theorem atomic_3616_agreement : RowAgreement 3615 3616 := RowAgreement.single (first := 3615) modelrow_agrees_3616
private def atomic_3617 : CountSegment 1 10 13120 3616 3617 1708 1708 := CountSegment.single (first := 3616) (initial := 1708) row_count_3617 (by decide)
private theorem atomic_3617_agreement : RowAgreement 3616 3617 := RowAgreement.single (first := 3616) modelrow_agrees_3617
private def atomic_3618 : CountSegment 1 10 13120 3617 3618 1708 1708 := CountSegment.single (first := 3617) (initial := 1708) row_count_3618 (by decide)
private theorem atomic_3618_agreement : RowAgreement 3617 3618 := RowAgreement.single (first := 3617) modelrow_agrees_3618
private def atomic_3619 : CountSegment 1 10 13120 3618 3619 1708 1709 := CountSegment.single (first := 3618) (initial := 1708) row_count_3619 (by decide)
private theorem atomic_3619_agreement : RowAgreement 3618 3619 := RowAgreement.single (first := 3618) modelrow_agrees_3619
private def atomic_3620 : CountSegment 1 10 13120 3619 3620 1709 1710 := CountSegment.single (first := 3619) (initial := 1709) row_count_3620 (by decide)
private theorem atomic_3620_agreement : RowAgreement 3619 3620 := RowAgreement.single (first := 3619) modelrow_agrees_3620
private def atomic_3621 : CountSegment 1 10 13120 3620 3621 1710 1710 := CountSegment.single (first := 3620) (initial := 1710) row_count_3621 (by decide)
private theorem atomic_3621_agreement : RowAgreement 3620 3621 := RowAgreement.single (first := 3620) modelrow_agrees_3621
private def atomic_3622 : CountSegment 1 10 13120 3621 3622 1710 1711 := CountSegment.single (first := 3621) (initial := 1710) row_count_3622 (by decide)
private theorem atomic_3622_agreement : RowAgreement 3621 3622 := RowAgreement.single (first := 3621) modelrow_agrees_3622
private def atomic_3623 : CountSegment 1 10 13120 3622 3623 1711 1711 := CountSegment.single (first := 3622) (initial := 1711) row_count_3623 (by decide)
private theorem atomic_3623_agreement : RowAgreement 3622 3623 := RowAgreement.single (first := 3622) modelrow_agrees_3623
private def atomic_3624 : CountSegment 1 10 13120 3623 3624 1711 1711 := CountSegment.single (first := 3623) (initial := 1711) row_count_3624 (by decide)
private theorem atomic_3624_agreement : RowAgreement 3623 3624 := RowAgreement.single (first := 3623) modelrow_agrees_3624
private def atomic_3625 : CountSegment 1 10 13120 3624 3625 1711 1711 := CountSegment.single (first := 3624) (initial := 1711) row_count_3625 (by decide)
private theorem atomic_3625_agreement : RowAgreement 3624 3625 := RowAgreement.single (first := 3624) modelrow_agrees_3625
private def atomic_3626 : CountSegment 1 10 13120 3625 3626 1711 1711 := CountSegment.single (first := 3625) (initial := 1711) row_count_3626 (by decide)
private theorem atomic_3626_agreement : RowAgreement 3625 3626 := RowAgreement.single (first := 3625) modelrow_agrees_3626
private def atomic_3627 : CountSegment 1 10 13120 3626 3627 1711 1712 := CountSegment.single (first := 3626) (initial := 1711) row_count_3627 (by decide)
private theorem atomic_3627_agreement : RowAgreement 3626 3627 := RowAgreement.single (first := 3626) modelrow_agrees_3627
private def atomic_3628 : CountSegment 1 10 13120 3627 3628 1712 1712 := CountSegment.single (first := 3627) (initial := 1712) row_count_3628 (by decide)
private theorem atomic_3628_agreement : RowAgreement 3627 3628 := RowAgreement.single (first := 3627) modelrow_agrees_3628
private def atomic_3629 : CountSegment 1 10 13120 3628 3629 1712 1712 := CountSegment.single (first := 3628) (initial := 1712) row_count_3629 (by decide)
private theorem atomic_3629_agreement : RowAgreement 3628 3629 := RowAgreement.single (first := 3628) modelrow_agrees_3629
private def atomic_3630 : CountSegment 1 10 13120 3629 3630 1712 1713 := CountSegment.single (first := 3629) (initial := 1712) row_count_3630 (by decide)
private theorem atomic_3630_agreement : RowAgreement 3629 3630 := RowAgreement.single (first := 3629) modelrow_agrees_3630
private def atomic_3631 : CountSegment 1 10 13120 3630 3631 1713 1713 := CountSegment.single (first := 3630) (initial := 1713) row_count_3631 (by decide)
private theorem atomic_3631_agreement : RowAgreement 3630 3631 := RowAgreement.single (first := 3630) modelrow_agrees_3631
private def atomic_3632 : CountSegment 1 10 13120 3631 3632 1713 1713 := CountSegment.single (first := 3631) (initial := 1713) row_count_3632 (by decide)
private theorem atomic_3632_agreement : RowAgreement 3631 3632 := RowAgreement.single (first := 3631) modelrow_agrees_3632
private def atomic_3633 : CountSegment 1 10 13120 3632 3633 1713 1713 := CountSegment.single (first := 3632) (initial := 1713) row_count_3633 (by decide)
private theorem atomic_3633_agreement : RowAgreement 3632 3633 := RowAgreement.single (first := 3632) modelrow_agrees_3633
private def atomic_3634 : CountSegment 1 10 13120 3633 3634 1713 1713 := CountSegment.single (first := 3633) (initial := 1713) row_count_3634 (by decide)
private theorem atomic_3634_agreement : RowAgreement 3633 3634 := RowAgreement.single (first := 3633) modelrow_agrees_3634
private def atomic_3635 : CountSegment 1 10 13120 3634 3635 1713 1713 := CountSegment.single (first := 3634) (initial := 1713) row_count_3635 (by decide)
private theorem atomic_3635_agreement : RowAgreement 3634 3635 := RowAgreement.single (first := 3634) modelrow_agrees_3635
private def atomic_3636 : CountSegment 1 10 13120 3635 3636 1713 1714 := CountSegment.single (first := 3635) (initial := 1713) row_count_3636 (by decide)
private theorem atomic_3636_agreement : RowAgreement 3635 3636 := RowAgreement.single (first := 3635) modelrow_agrees_3636
private def atomic_3637 : CountSegment 1 10 13120 3636 3637 1714 1715 := CountSegment.single (first := 3636) (initial := 1714) row_count_3637 (by decide)
private theorem atomic_3637_agreement : RowAgreement 3636 3637 := RowAgreement.single (first := 3636) modelrow_agrees_3637
private def atomic_3638 : CountSegment 1 10 13120 3637 3638 1715 1715 := CountSegment.single (first := 3637) (initial := 1715) row_count_3638 (by decide)
private theorem atomic_3638_agreement : RowAgreement 3637 3638 := RowAgreement.single (first := 3637) modelrow_agrees_3638
private def atomic_3639 : CountSegment 1 10 13120 3638 3639 1715 1715 := CountSegment.single (first := 3638) (initial := 1715) row_count_3639 (by decide)
private theorem atomic_3639_agreement : RowAgreement 3638 3639 := RowAgreement.single (first := 3638) modelrow_agrees_3639
private def atomic_3640 : CountSegment 1 10 13120 3639 3640 1715 1715 := CountSegment.single (first := 3639) (initial := 1715) row_count_3640 (by decide)
private theorem atomic_3640_agreement : RowAgreement 3639 3640 := RowAgreement.single (first := 3639) modelrow_agrees_3640
private def atomic_3641 : CountSegment 1 10 13120 3640 3641 1715 1715 := CountSegment.single (first := 3640) (initial := 1715) row_count_3641 (by decide)
private theorem atomic_3641_agreement : RowAgreement 3640 3641 := RowAgreement.single (first := 3640) modelrow_agrees_3641
private def atomic_3642 : CountSegment 1 10 13120 3641 3642 1715 1715 := CountSegment.single (first := 3641) (initial := 1715) row_count_3642 (by decide)
private theorem atomic_3642_agreement : RowAgreement 3641 3642 := RowAgreement.single (first := 3641) modelrow_agrees_3642
private def atomic_3643 : CountSegment 1 10 13120 3642 3643 1715 1717 := CountSegment.single (first := 3642) (initial := 1715) row_count_3643 (by decide)
private theorem atomic_3643_agreement : RowAgreement 3642 3643 := RowAgreement.single (first := 3642) modelrow_agrees_3643
private def atomic_3644 : CountSegment 1 10 13120 3643 3644 1717 1717 := CountSegment.single (first := 3643) (initial := 1717) row_count_3644 (by decide)
private theorem atomic_3644_agreement : RowAgreement 3643 3644 := RowAgreement.single (first := 3643) modelrow_agrees_3644
private def atomic_3645 : CountSegment 1 10 13120 3644 3645 1717 1717 := CountSegment.single (first := 3644) (initial := 1717) row_count_3645 (by decide)
private theorem atomic_3645_agreement : RowAgreement 3644 3645 := RowAgreement.single (first := 3644) modelrow_agrees_3645
private def atomic_3646 : CountSegment 1 10 13120 3645 3646 1717 1717 := CountSegment.single (first := 3645) (initial := 1717) row_count_3646 (by decide)
private theorem atomic_3646_agreement : RowAgreement 3645 3646 := RowAgreement.single (first := 3645) modelrow_agrees_3646
private def atomic_3647 : CountSegment 1 10 13120 3646 3647 1717 1717 := CountSegment.single (first := 3646) (initial := 1717) row_count_3647 (by decide)
private theorem atomic_3647_agreement : RowAgreement 3646 3647 := RowAgreement.single (first := 3646) modelrow_agrees_3647
private def atomic_3648 : CountSegment 1 10 13120 3647 3648 1717 1718 := CountSegment.single (first := 3647) (initial := 1717) row_count_3648 (by decide)
private theorem atomic_3648_agreement : RowAgreement 3647 3648 := RowAgreement.single (first := 3647) modelrow_agrees_3648
private def atomic_3649 : CountSegment 1 10 13120 3648 3649 1718 1718 := CountSegment.single (first := 3648) (initial := 1718) row_count_3649 (by decide)
private theorem atomic_3649_agreement : RowAgreement 3648 3649 := RowAgreement.single (first := 3648) modelrow_agrees_3649
private def atomic_3650 : CountSegment 1 10 13120 3649 3650 1718 1718 := CountSegment.single (first := 3649) (initial := 1718) row_count_3650 (by decide)
private theorem atomic_3650_agreement : RowAgreement 3649 3650 := RowAgreement.single (first := 3649) modelrow_agrees_3650
private def atomic_3651 : CountSegment 1 10 13120 3650 3651 1718 1719 := CountSegment.single (first := 3650) (initial := 1718) row_count_3651 (by decide)
private theorem atomic_3651_agreement : RowAgreement 3650 3651 := RowAgreement.single (first := 3650) modelrow_agrees_3651
private def atomic_3652 : CountSegment 1 10 13120 3651 3652 1719 1720 := CountSegment.single (first := 3651) (initial := 1719) row_count_3652 (by decide)
private theorem atomic_3652_agreement : RowAgreement 3651 3652 := RowAgreement.single (first := 3651) modelrow_agrees_3652
private def atomic_3653 : CountSegment 1 10 13120 3652 3653 1720 1720 := CountSegment.single (first := 3652) (initial := 1720) row_count_3653 (by decide)
private theorem atomic_3653_agreement : RowAgreement 3652 3653 := RowAgreement.single (first := 3652) modelrow_agrees_3653
private def atomic_3654 : CountSegment 1 10 13120 3653 3654 1720 1720 := CountSegment.single (first := 3653) (initial := 1720) row_count_3654 (by decide)
private theorem atomic_3654_agreement : RowAgreement 3653 3654 := RowAgreement.single (first := 3653) modelrow_agrees_3654
private def atomic_3655 : CountSegment 1 10 13120 3654 3655 1720 1720 := CountSegment.single (first := 3654) (initial := 1720) row_count_3655 (by decide)
private theorem atomic_3655_agreement : RowAgreement 3654 3655 := RowAgreement.single (first := 3654) modelrow_agrees_3655
private def atomic_3656 : CountSegment 1 10 13120 3655 3656 1720 1721 := CountSegment.single (first := 3655) (initial := 1720) row_count_3656 (by decide)
private theorem atomic_3656_agreement : RowAgreement 3655 3656 := RowAgreement.single (first := 3655) modelrow_agrees_3656
private def atomic_3657 : CountSegment 1 10 13120 3656 3657 1721 1722 := CountSegment.single (first := 3656) (initial := 1721) row_count_3657 (by decide)
private theorem atomic_3657_agreement : RowAgreement 3656 3657 := RowAgreement.single (first := 3656) modelrow_agrees_3657
private def atomic_3658 : CountSegment 1 10 13120 3657 3658 1722 1723 := CountSegment.single (first := 3657) (initial := 1722) row_count_3658 (by decide)
private theorem atomic_3658_agreement : RowAgreement 3657 3658 := RowAgreement.single (first := 3657) modelrow_agrees_3658
private def atomic_3659 : CountSegment 1 10 13120 3658 3659 1723 1723 := CountSegment.single (first := 3658) (initial := 1723) row_count_3659 (by decide)
private theorem atomic_3659_agreement : RowAgreement 3658 3659 := RowAgreement.single (first := 3658) modelrow_agrees_3659
private def atomic_3660 : CountSegment 1 10 13120 3659 3660 1723 1723 := CountSegment.single (first := 3659) (initial := 1723) row_count_3660 (by decide)
private theorem atomic_3660_agreement : RowAgreement 3659 3660 := RowAgreement.single (first := 3659) modelrow_agrees_3660
private def atomic_3661 : CountSegment 1 10 13120 3660 3661 1723 1723 := CountSegment.single (first := 3660) (initial := 1723) row_count_3661 (by decide)
private theorem atomic_3661_agreement : RowAgreement 3660 3661 := RowAgreement.single (first := 3660) modelrow_agrees_3661
private def atomic_3662 : CountSegment 1 10 13120 3661 3662 1723 1723 := CountSegment.single (first := 3661) (initial := 1723) row_count_3662 (by decide)
private theorem atomic_3662_agreement : RowAgreement 3661 3662 := RowAgreement.single (first := 3661) modelrow_agrees_3662
private def atomic_3663 : CountSegment 1 10 13120 3662 3663 1723 1724 := CountSegment.single (first := 3662) (initial := 1723) row_count_3663 (by decide)
private theorem atomic_3663_agreement : RowAgreement 3662 3663 := RowAgreement.single (first := 3662) modelrow_agrees_3663
private def atomic_3664 : CountSegment 1 10 13120 3663 3664 1724 1724 := CountSegment.single (first := 3663) (initial := 1724) row_count_3664 (by decide)
private theorem atomic_3664_agreement : RowAgreement 3663 3664 := RowAgreement.single (first := 3663) modelrow_agrees_3664
private def atomic_3665 : CountSegment 1 10 13120 3664 3665 1724 1724 := CountSegment.single (first := 3664) (initial := 1724) row_count_3665 (by decide)
private theorem atomic_3665_agreement : RowAgreement 3664 3665 := RowAgreement.single (first := 3664) modelrow_agrees_3665
private def atomic_3666 : CountSegment 1 10 13120 3665 3666 1724 1724 := CountSegment.single (first := 3665) (initial := 1724) row_count_3666 (by decide)
private theorem atomic_3666_agreement : RowAgreement 3665 3666 := RowAgreement.single (first := 3665) modelrow_agrees_3666
private def atomic_3667 : CountSegment 1 10 13120 3666 3667 1724 1725 := CountSegment.single (first := 3666) (initial := 1724) row_count_3667 (by decide)
private theorem atomic_3667_agreement : RowAgreement 3666 3667 := RowAgreement.single (first := 3666) modelrow_agrees_3667
private def atomic_3668 : CountSegment 1 10 13120 3667 3668 1725 1725 := CountSegment.single (first := 3667) (initial := 1725) row_count_3668 (by decide)
private theorem atomic_3668_agreement : RowAgreement 3667 3668 := RowAgreement.single (first := 3667) modelrow_agrees_3668
private def atomic_3669 : CountSegment 1 10 13120 3668 3669 1725 1725 := CountSegment.single (first := 3668) (initial := 1725) row_count_3669 (by decide)
private theorem atomic_3669_agreement : RowAgreement 3668 3669 := RowAgreement.single (first := 3668) modelrow_agrees_3669
private def atomic_3670 : CountSegment 1 10 13120 3669 3670 1725 1725 := CountSegment.single (first := 3669) (initial := 1725) row_count_3670 (by decide)
private theorem atomic_3670_agreement : RowAgreement 3669 3670 := RowAgreement.single (first := 3669) modelrow_agrees_3670
private def atomic_3671 : CountSegment 1 10 13120 3670 3671 1725 1725 := CountSegment.single (first := 3670) (initial := 1725) row_count_3671 (by decide)
private theorem atomic_3671_agreement : RowAgreement 3670 3671 := RowAgreement.single (first := 3670) modelrow_agrees_3671
private def atomic_3672 : CountSegment 1 10 13120 3671 3672 1725 1727 := CountSegment.single (first := 3671) (initial := 1725) row_count_3672 (by decide)
private theorem atomic_3672_agreement : RowAgreement 3671 3672 := RowAgreement.single (first := 3671) modelrow_agrees_3672
private def atomic_3673 : CountSegment 1 10 13120 3672 3673 1727 1727 := CountSegment.single (first := 3672) (initial := 1727) row_count_3673 (by decide)
private theorem atomic_3673_agreement : RowAgreement 3672 3673 := RowAgreement.single (first := 3672) modelrow_agrees_3673
private def atomic_3674 : CountSegment 1 10 13120 3673 3674 1727 1727 := CountSegment.single (first := 3673) (initial := 1727) row_count_3674 (by decide)
private theorem atomic_3674_agreement : RowAgreement 3673 3674 := RowAgreement.single (first := 3673) modelrow_agrees_3674
private def atomic_3675 : CountSegment 1 10 13120 3674 3675 1727 1727 := CountSegment.single (first := 3674) (initial := 1727) row_count_3675 (by decide)
private theorem atomic_3675_agreement : RowAgreement 3674 3675 := RowAgreement.single (first := 3674) modelrow_agrees_3675
private def atomic_3676 : CountSegment 1 10 13120 3675 3676 1727 1727 := CountSegment.single (first := 3675) (initial := 1727) row_count_3676 (by decide)
private theorem atomic_3676_agreement : RowAgreement 3675 3676 := RowAgreement.single (first := 3675) modelrow_agrees_3676
private def atomic_3677 : CountSegment 1 10 13120 3676 3677 1727 1727 := CountSegment.single (first := 3676) (initial := 1727) row_count_3677 (by decide)
private theorem atomic_3677_agreement : RowAgreement 3676 3677 := RowAgreement.single (first := 3676) modelrow_agrees_3677
private def atomic_3678 : CountSegment 1 10 13120 3677 3678 1727 1729 := CountSegment.single (first := 3677) (initial := 1727) row_count_3678 (by decide)
private theorem atomic_3678_agreement : RowAgreement 3677 3678 := RowAgreement.single (first := 3677) modelrow_agrees_3678
private def atomic_3679 : CountSegment 1 10 13120 3678 3679 1729 1730 := CountSegment.single (first := 3678) (initial := 1729) row_count_3679 (by decide)
private theorem atomic_3679_agreement : RowAgreement 3678 3679 := RowAgreement.single (first := 3678) modelrow_agrees_3679
private def atomic_3680 : CountSegment 1 10 13120 3679 3680 1730 1730 := CountSegment.single (first := 3679) (initial := 1730) row_count_3680 (by decide)
private theorem atomic_3680_agreement : RowAgreement 3679 3680 := RowAgreement.single (first := 3679) modelrow_agrees_3680
private def atomic_3681 : CountSegment 1 10 13120 3680 3681 1730 1730 := CountSegment.single (first := 3680) (initial := 1730) row_count_3681 (by decide)
private theorem atomic_3681_agreement : RowAgreement 3680 3681 := RowAgreement.single (first := 3680) modelrow_agrees_3681
private def atomic_3682 : CountSegment 1 10 13120 3681 3682 1730 1731 := CountSegment.single (first := 3681) (initial := 1730) row_count_3682 (by decide)
private theorem atomic_3682_agreement : RowAgreement 3681 3682 := RowAgreement.single (first := 3681) modelrow_agrees_3682
private def atomic_3683 : CountSegment 1 10 13120 3682 3683 1731 1732 := CountSegment.single (first := 3682) (initial := 1731) row_count_3683 (by decide)
private theorem atomic_3683_agreement : RowAgreement 3682 3683 := RowAgreement.single (first := 3682) modelrow_agrees_3683
private def atomic_3684 : CountSegment 1 10 13120 3683 3684 1732 1732 := CountSegment.single (first := 3683) (initial := 1732) row_count_3684 (by decide)
private theorem atomic_3684_agreement : RowAgreement 3683 3684 := RowAgreement.single (first := 3683) modelrow_agrees_3684
private def atomic_3685 : CountSegment 1 10 13120 3684 3685 1732 1732 := CountSegment.single (first := 3684) (initial := 1732) row_count_3685 (by decide)
private theorem atomic_3685_agreement : RowAgreement 3684 3685 := RowAgreement.single (first := 3684) modelrow_agrees_3685
private def atomic_3686 : CountSegment 1 10 13120 3685 3686 1732 1732 := CountSegment.single (first := 3685) (initial := 1732) row_count_3686 (by decide)
private theorem atomic_3686_agreement : RowAgreement 3685 3686 := RowAgreement.single (first := 3685) modelrow_agrees_3686
private def atomic_3687 : CountSegment 1 10 13120 3686 3687 1732 1732 := CountSegment.single (first := 3686) (initial := 1732) row_count_3687 (by decide)
private theorem atomic_3687_agreement : RowAgreement 3686 3687 := RowAgreement.single (first := 3686) modelrow_agrees_3687
private def atomic_3688 : CountSegment 1 10 13120 3687 3688 1732 1732 := CountSegment.single (first := 3687) (initial := 1732) row_count_3688 (by decide)
private theorem atomic_3688_agreement : RowAgreement 3687 3688 := RowAgreement.single (first := 3687) modelrow_agrees_3688
private def atomic_3689 : CountSegment 1 10 13120 3688 3689 1732 1732 := CountSegment.single (first := 3688) (initial := 1732) row_count_3689 (by decide)
private theorem atomic_3689_agreement : RowAgreement 3688 3689 := RowAgreement.single (first := 3688) modelrow_agrees_3689
private def atomic_3690 : CountSegment 1 10 13120 3689 3690 1732 1732 := CountSegment.single (first := 3689) (initial := 1732) row_count_3690 (by decide)
private theorem atomic_3690_agreement : RowAgreement 3689 3690 := RowAgreement.single (first := 3689) modelrow_agrees_3690
private def atomic_3691 : CountSegment 1 10 13120 3690 3691 1732 1732 := CountSegment.single (first := 3690) (initial := 1732) row_count_3691 (by decide)
private theorem atomic_3691_agreement : RowAgreement 3690 3691 := RowAgreement.single (first := 3690) modelrow_agrees_3691
private def atomic_3692 : CountSegment 1 10 13120 3691 3692 1732 1733 := CountSegment.single (first := 3691) (initial := 1732) row_count_3692 (by decide)
private theorem atomic_3692_agreement : RowAgreement 3691 3692 := RowAgreement.single (first := 3691) modelrow_agrees_3692
private def atomic_3693 : CountSegment 1 10 13120 3692 3693 1733 1733 := CountSegment.single (first := 3692) (initial := 1733) row_count_3693 (by decide)
private theorem atomic_3693_agreement : RowAgreement 3692 3693 := RowAgreement.single (first := 3692) modelrow_agrees_3693
private def atomic_3694 : CountSegment 1 10 13120 3693 3694 1733 1734 := CountSegment.single (first := 3693) (initial := 1733) row_count_3694 (by decide)
private theorem atomic_3694_agreement : RowAgreement 3693 3694 := RowAgreement.single (first := 3693) modelrow_agrees_3694
private def atomic_3695 : CountSegment 1 10 13120 3694 3695 1734 1734 := CountSegment.single (first := 3694) (initial := 1734) row_count_3695 (by decide)
private theorem atomic_3695_agreement : RowAgreement 3694 3695 := RowAgreement.single (first := 3694) modelrow_agrees_3695
private def atomic_3696 : CountSegment 1 10 13120 3695 3696 1734 1735 := CountSegment.single (first := 3695) (initial := 1734) row_count_3696 (by decide)
private theorem atomic_3696_agreement : RowAgreement 3695 3696 := RowAgreement.single (first := 3695) modelrow_agrees_3696
private def atomic_3697 : CountSegment 1 10 13120 3696 3697 1735 1735 := CountSegment.single (first := 3696) (initial := 1735) row_count_3697 (by decide)
private theorem atomic_3697_agreement : RowAgreement 3696 3697 := RowAgreement.single (first := 3696) modelrow_agrees_3697
private def atomic_3698 : CountSegment 1 10 13120 3697 3698 1735 1735 := CountSegment.single (first := 3697) (initial := 1735) row_count_3698 (by decide)
private theorem atomic_3698_agreement : RowAgreement 3697 3698 := RowAgreement.single (first := 3697) modelrow_agrees_3698
private def atomic_3699 : CountSegment 1 10 13120 3698 3699 1735 1736 := CountSegment.single (first := 3698) (initial := 1735) row_count_3699 (by decide)
private theorem atomic_3699_agreement : RowAgreement 3698 3699 := RowAgreement.single (first := 3698) modelrow_agrees_3699
private def atomic_3700 : CountSegment 1 10 13120 3699 3700 1736 1736 := CountSegment.single (first := 3699) (initial := 1736) row_count_3700 (by decide)
private theorem atomic_3700_agreement : RowAgreement 3699 3700 := RowAgreement.single (first := 3699) modelrow_agrees_3700
private def atomic_3701 : CountSegment 1 10 13120 3700 3701 1736 1736 := CountSegment.single (first := 3700) (initial := 1736) row_count_3701 (by decide)
private theorem atomic_3701_agreement : RowAgreement 3700 3701 := RowAgreement.single (first := 3700) modelrow_agrees_3701
private def atomic_3702 : CountSegment 1 10 13120 3701 3702 1736 1737 := CountSegment.single (first := 3701) (initial := 1736) row_count_3702 (by decide)
private theorem atomic_3702_agreement : RowAgreement 3701 3702 := RowAgreement.single (first := 3701) modelrow_agrees_3702
private def atomic_3703 : CountSegment 1 10 13120 3702 3703 1737 1737 := CountSegment.single (first := 3702) (initial := 1737) row_count_3703 (by decide)
private theorem atomic_3703_agreement : RowAgreement 3702 3703 := RowAgreement.single (first := 3702) modelrow_agrees_3703
private def atomic_3704 : CountSegment 1 10 13120 3703 3704 1737 1737 := CountSegment.single (first := 3703) (initial := 1737) row_count_3704 (by decide)
private theorem atomic_3704_agreement : RowAgreement 3703 3704 := RowAgreement.single (first := 3703) modelrow_agrees_3704
private def atomic_3705 : CountSegment 1 10 13120 3704 3705 1737 1738 := CountSegment.single (first := 3704) (initial := 1737) row_count_3705 (by decide)
private theorem atomic_3705_agreement : RowAgreement 3704 3705 := RowAgreement.single (first := 3704) modelrow_agrees_3705
private def atomic_3706 : CountSegment 1 10 13120 3705 3706 1738 1738 := CountSegment.single (first := 3705) (initial := 1738) row_count_3706 (by decide)
private theorem atomic_3706_agreement : RowAgreement 3705 3706 := RowAgreement.single (first := 3705) modelrow_agrees_3706
private def atomic_3707 : CountSegment 1 10 13120 3706 3707 1738 1739 := CountSegment.single (first := 3706) (initial := 1738) row_count_3707 (by decide)
private theorem atomic_3707_agreement : RowAgreement 3706 3707 := RowAgreement.single (first := 3706) modelrow_agrees_3707
private def atomic_3708 : CountSegment 1 10 13120 3707 3708 1739 1739 := CountSegment.single (first := 3707) (initial := 1739) row_count_3708 (by decide)
private theorem atomic_3708_agreement : RowAgreement 3707 3708 := RowAgreement.single (first := 3707) modelrow_agrees_3708
private def atomic_3709 : CountSegment 1 10 13120 3708 3709 1739 1739 := CountSegment.single (first := 3708) (initial := 1739) row_count_3709 (by decide)
private theorem atomic_3709_agreement : RowAgreement 3708 3709 := RowAgreement.single (first := 3708) modelrow_agrees_3709
private def atomic_3710 : CountSegment 1 10 13120 3709 3710 1739 1740 := CountSegment.single (first := 3709) (initial := 1739) row_count_3710 (by decide)
private theorem atomic_3710_agreement : RowAgreement 3709 3710 := RowAgreement.single (first := 3709) modelrow_agrees_3710
private def atomic_3711 : CountSegment 1 10 13120 3710 3711 1740 1740 := CountSegment.single (first := 3710) (initial := 1740) row_count_3711 (by decide)
private theorem atomic_3711_agreement : RowAgreement 3710 3711 := RowAgreement.single (first := 3710) modelrow_agrees_3711
private def atomic_3712 : CountSegment 1 10 13120 3711 3712 1740 1740 := CountSegment.single (first := 3711) (initial := 1740) row_count_3712 (by decide)
private theorem atomic_3712_agreement : RowAgreement 3711 3712 := RowAgreement.single (first := 3711) modelrow_agrees_3712
private def within_028_000 : CountSegment 1 10 13120 3584 3586 1697 1697 := CountSegment.join atomic_3585 atomic_3586
private theorem within_028_000_agreement : RowAgreement 3584 3586 := RowAgreement.join atomic_3585_agreement atomic_3586_agreement
private def within_028_001 : CountSegment 1 10 13120 3586 3588 1697 1697 := CountSegment.join atomic_3587 atomic_3588
private theorem within_028_001_agreement : RowAgreement 3586 3588 := RowAgreement.join atomic_3587_agreement atomic_3588_agreement
private def within_028_002 : CountSegment 1 10 13120 3584 3588 1697 1697 := CountSegment.join within_028_000 within_028_001
private theorem within_028_002_agreement : RowAgreement 3584 3588 := RowAgreement.join within_028_000_agreement within_028_001_agreement
private def within_028_003 : CountSegment 1 10 13120 3588 3590 1697 1697 := CountSegment.join atomic_3589 atomic_3590
private theorem within_028_003_agreement : RowAgreement 3588 3590 := RowAgreement.join atomic_3589_agreement atomic_3590_agreement
private def within_028_004 : CountSegment 1 10 13120 3590 3592 1697 1699 := CountSegment.join atomic_3591 atomic_3592
private theorem within_028_004_agreement : RowAgreement 3590 3592 := RowAgreement.join atomic_3591_agreement atomic_3592_agreement
private def within_028_005 : CountSegment 1 10 13120 3588 3592 1697 1699 := CountSegment.join within_028_003 within_028_004
private theorem within_028_005_agreement : RowAgreement 3588 3592 := RowAgreement.join within_028_003_agreement within_028_004_agreement
private def within_028_006 : CountSegment 1 10 13120 3584 3592 1697 1699 := CountSegment.join within_028_002 within_028_005
private theorem within_028_006_agreement : RowAgreement 3584 3592 := RowAgreement.join within_028_002_agreement within_028_005_agreement
private def within_028_007 : CountSegment 1 10 13120 3592 3594 1699 1699 := CountSegment.join atomic_3593 atomic_3594
private theorem within_028_007_agreement : RowAgreement 3592 3594 := RowAgreement.join atomic_3593_agreement atomic_3594_agreement
private def within_028_008 : CountSegment 1 10 13120 3594 3596 1699 1701 := CountSegment.join atomic_3595 atomic_3596
private theorem within_028_008_agreement : RowAgreement 3594 3596 := RowAgreement.join atomic_3595_agreement atomic_3596_agreement
private def within_028_009 : CountSegment 1 10 13120 3592 3596 1699 1701 := CountSegment.join within_028_007 within_028_008
private theorem within_028_009_agreement : RowAgreement 3592 3596 := RowAgreement.join within_028_007_agreement within_028_008_agreement
private def within_028_010 : CountSegment 1 10 13120 3596 3598 1701 1702 := CountSegment.join atomic_3597 atomic_3598
private theorem within_028_010_agreement : RowAgreement 3596 3598 := RowAgreement.join atomic_3597_agreement atomic_3598_agreement
private def within_028_011 : CountSegment 1 10 13120 3598 3600 1702 1703 := CountSegment.join atomic_3599 atomic_3600
private theorem within_028_011_agreement : RowAgreement 3598 3600 := RowAgreement.join atomic_3599_agreement atomic_3600_agreement
private def within_028_012 : CountSegment 1 10 13120 3596 3600 1701 1703 := CountSegment.join within_028_010 within_028_011
private theorem within_028_012_agreement : RowAgreement 3596 3600 := RowAgreement.join within_028_010_agreement within_028_011_agreement
private def within_028_013 : CountSegment 1 10 13120 3592 3600 1699 1703 := CountSegment.join within_028_009 within_028_012
private theorem within_028_013_agreement : RowAgreement 3592 3600 := RowAgreement.join within_028_009_agreement within_028_012_agreement
private def within_028_014 : CountSegment 1 10 13120 3584 3600 1697 1703 := CountSegment.join within_028_006 within_028_013
private theorem within_028_014_agreement : RowAgreement 3584 3600 := RowAgreement.join within_028_006_agreement within_028_013_agreement
private def within_028_015 : CountSegment 1 10 13120 3600 3602 1703 1704 := CountSegment.join atomic_3601 atomic_3602
private theorem within_028_015_agreement : RowAgreement 3600 3602 := RowAgreement.join atomic_3601_agreement atomic_3602_agreement
private def within_028_016 : CountSegment 1 10 13120 3602 3604 1704 1704 := CountSegment.join atomic_3603 atomic_3604
private theorem within_028_016_agreement : RowAgreement 3602 3604 := RowAgreement.join atomic_3603_agreement atomic_3604_agreement
private def within_028_017 : CountSegment 1 10 13120 3600 3604 1703 1704 := CountSegment.join within_028_015 within_028_016
private theorem within_028_017_agreement : RowAgreement 3600 3604 := RowAgreement.join within_028_015_agreement within_028_016_agreement
private def within_028_018 : CountSegment 1 10 13120 3604 3606 1704 1704 := CountSegment.join atomic_3605 atomic_3606
private theorem within_028_018_agreement : RowAgreement 3604 3606 := RowAgreement.join atomic_3605_agreement atomic_3606_agreement
private def within_028_019 : CountSegment 1 10 13120 3606 3608 1704 1705 := CountSegment.join atomic_3607 atomic_3608
private theorem within_028_019_agreement : RowAgreement 3606 3608 := RowAgreement.join atomic_3607_agreement atomic_3608_agreement
private def within_028_020 : CountSegment 1 10 13120 3604 3608 1704 1705 := CountSegment.join within_028_018 within_028_019
private theorem within_028_020_agreement : RowAgreement 3604 3608 := RowAgreement.join within_028_018_agreement within_028_019_agreement
private def within_028_021 : CountSegment 1 10 13120 3600 3608 1703 1705 := CountSegment.join within_028_017 within_028_020
private theorem within_028_021_agreement : RowAgreement 3600 3608 := RowAgreement.join within_028_017_agreement within_028_020_agreement
private def within_028_022 : CountSegment 1 10 13120 3608 3610 1705 1706 := CountSegment.join atomic_3609 atomic_3610
private theorem within_028_022_agreement : RowAgreement 3608 3610 := RowAgreement.join atomic_3609_agreement atomic_3610_agreement
private def within_028_023 : CountSegment 1 10 13120 3610 3612 1706 1707 := CountSegment.join atomic_3611 atomic_3612
private theorem within_028_023_agreement : RowAgreement 3610 3612 := RowAgreement.join atomic_3611_agreement atomic_3612_agreement
private def within_028_024 : CountSegment 1 10 13120 3608 3612 1705 1707 := CountSegment.join within_028_022 within_028_023
private theorem within_028_024_agreement : RowAgreement 3608 3612 := RowAgreement.join within_028_022_agreement within_028_023_agreement
private def within_028_025 : CountSegment 1 10 13120 3612 3614 1707 1707 := CountSegment.join atomic_3613 atomic_3614
private theorem within_028_025_agreement : RowAgreement 3612 3614 := RowAgreement.join atomic_3613_agreement atomic_3614_agreement
private def within_028_026 : CountSegment 1 10 13120 3614 3616 1707 1708 := CountSegment.join atomic_3615 atomic_3616
private theorem within_028_026_agreement : RowAgreement 3614 3616 := RowAgreement.join atomic_3615_agreement atomic_3616_agreement
private def within_028_027 : CountSegment 1 10 13120 3612 3616 1707 1708 := CountSegment.join within_028_025 within_028_026
private theorem within_028_027_agreement : RowAgreement 3612 3616 := RowAgreement.join within_028_025_agreement within_028_026_agreement
private def within_028_028 : CountSegment 1 10 13120 3608 3616 1705 1708 := CountSegment.join within_028_024 within_028_027
private theorem within_028_028_agreement : RowAgreement 3608 3616 := RowAgreement.join within_028_024_agreement within_028_027_agreement
private def within_028_029 : CountSegment 1 10 13120 3600 3616 1703 1708 := CountSegment.join within_028_021 within_028_028
private theorem within_028_029_agreement : RowAgreement 3600 3616 := RowAgreement.join within_028_021_agreement within_028_028_agreement
private def within_028_030 : CountSegment 1 10 13120 3584 3616 1697 1708 := CountSegment.join within_028_014 within_028_029
private theorem within_028_030_agreement : RowAgreement 3584 3616 := RowAgreement.join within_028_014_agreement within_028_029_agreement
private def within_028_031 : CountSegment 1 10 13120 3616 3618 1708 1708 := CountSegment.join atomic_3617 atomic_3618
private theorem within_028_031_agreement : RowAgreement 3616 3618 := RowAgreement.join atomic_3617_agreement atomic_3618_agreement
private def within_028_032 : CountSegment 1 10 13120 3618 3620 1708 1710 := CountSegment.join atomic_3619 atomic_3620
private theorem within_028_032_agreement : RowAgreement 3618 3620 := RowAgreement.join atomic_3619_agreement atomic_3620_agreement
private def within_028_033 : CountSegment 1 10 13120 3616 3620 1708 1710 := CountSegment.join within_028_031 within_028_032
private theorem within_028_033_agreement : RowAgreement 3616 3620 := RowAgreement.join within_028_031_agreement within_028_032_agreement
private def within_028_034 : CountSegment 1 10 13120 3620 3622 1710 1711 := CountSegment.join atomic_3621 atomic_3622
private theorem within_028_034_agreement : RowAgreement 3620 3622 := RowAgreement.join atomic_3621_agreement atomic_3622_agreement
private def within_028_035 : CountSegment 1 10 13120 3622 3624 1711 1711 := CountSegment.join atomic_3623 atomic_3624
private theorem within_028_035_agreement : RowAgreement 3622 3624 := RowAgreement.join atomic_3623_agreement atomic_3624_agreement
private def within_028_036 : CountSegment 1 10 13120 3620 3624 1710 1711 := CountSegment.join within_028_034 within_028_035
private theorem within_028_036_agreement : RowAgreement 3620 3624 := RowAgreement.join within_028_034_agreement within_028_035_agreement
private def within_028_037 : CountSegment 1 10 13120 3616 3624 1708 1711 := CountSegment.join within_028_033 within_028_036
private theorem within_028_037_agreement : RowAgreement 3616 3624 := RowAgreement.join within_028_033_agreement within_028_036_agreement
private def within_028_038 : CountSegment 1 10 13120 3624 3626 1711 1711 := CountSegment.join atomic_3625 atomic_3626
private theorem within_028_038_agreement : RowAgreement 3624 3626 := RowAgreement.join atomic_3625_agreement atomic_3626_agreement
private def within_028_039 : CountSegment 1 10 13120 3626 3628 1711 1712 := CountSegment.join atomic_3627 atomic_3628
private theorem within_028_039_agreement : RowAgreement 3626 3628 := RowAgreement.join atomic_3627_agreement atomic_3628_agreement
private def within_028_040 : CountSegment 1 10 13120 3624 3628 1711 1712 := CountSegment.join within_028_038 within_028_039
private theorem within_028_040_agreement : RowAgreement 3624 3628 := RowAgreement.join within_028_038_agreement within_028_039_agreement
private def within_028_041 : CountSegment 1 10 13120 3628 3630 1712 1713 := CountSegment.join atomic_3629 atomic_3630
private theorem within_028_041_agreement : RowAgreement 3628 3630 := RowAgreement.join atomic_3629_agreement atomic_3630_agreement
private def within_028_042 : CountSegment 1 10 13120 3630 3632 1713 1713 := CountSegment.join atomic_3631 atomic_3632
private theorem within_028_042_agreement : RowAgreement 3630 3632 := RowAgreement.join atomic_3631_agreement atomic_3632_agreement
private def within_028_043 : CountSegment 1 10 13120 3628 3632 1712 1713 := CountSegment.join within_028_041 within_028_042
private theorem within_028_043_agreement : RowAgreement 3628 3632 := RowAgreement.join within_028_041_agreement within_028_042_agreement
private def within_028_044 : CountSegment 1 10 13120 3624 3632 1711 1713 := CountSegment.join within_028_040 within_028_043
private theorem within_028_044_agreement : RowAgreement 3624 3632 := RowAgreement.join within_028_040_agreement within_028_043_agreement
private def within_028_045 : CountSegment 1 10 13120 3616 3632 1708 1713 := CountSegment.join within_028_037 within_028_044
private theorem within_028_045_agreement : RowAgreement 3616 3632 := RowAgreement.join within_028_037_agreement within_028_044_agreement
private def within_028_046 : CountSegment 1 10 13120 3632 3634 1713 1713 := CountSegment.join atomic_3633 atomic_3634
private theorem within_028_046_agreement : RowAgreement 3632 3634 := RowAgreement.join atomic_3633_agreement atomic_3634_agreement
private def within_028_047 : CountSegment 1 10 13120 3634 3636 1713 1714 := CountSegment.join atomic_3635 atomic_3636
private theorem within_028_047_agreement : RowAgreement 3634 3636 := RowAgreement.join atomic_3635_agreement atomic_3636_agreement
private def within_028_048 : CountSegment 1 10 13120 3632 3636 1713 1714 := CountSegment.join within_028_046 within_028_047
private theorem within_028_048_agreement : RowAgreement 3632 3636 := RowAgreement.join within_028_046_agreement within_028_047_agreement
private def within_028_049 : CountSegment 1 10 13120 3636 3638 1714 1715 := CountSegment.join atomic_3637 atomic_3638
private theorem within_028_049_agreement : RowAgreement 3636 3638 := RowAgreement.join atomic_3637_agreement atomic_3638_agreement
private def within_028_050 : CountSegment 1 10 13120 3638 3640 1715 1715 := CountSegment.join atomic_3639 atomic_3640
private theorem within_028_050_agreement : RowAgreement 3638 3640 := RowAgreement.join atomic_3639_agreement atomic_3640_agreement
private def within_028_051 : CountSegment 1 10 13120 3636 3640 1714 1715 := CountSegment.join within_028_049 within_028_050
private theorem within_028_051_agreement : RowAgreement 3636 3640 := RowAgreement.join within_028_049_agreement within_028_050_agreement
private def within_028_052 : CountSegment 1 10 13120 3632 3640 1713 1715 := CountSegment.join within_028_048 within_028_051
private theorem within_028_052_agreement : RowAgreement 3632 3640 := RowAgreement.join within_028_048_agreement within_028_051_agreement
private def within_028_053 : CountSegment 1 10 13120 3640 3642 1715 1715 := CountSegment.join atomic_3641 atomic_3642
private theorem within_028_053_agreement : RowAgreement 3640 3642 := RowAgreement.join atomic_3641_agreement atomic_3642_agreement
private def within_028_054 : CountSegment 1 10 13120 3642 3644 1715 1717 := CountSegment.join atomic_3643 atomic_3644
private theorem within_028_054_agreement : RowAgreement 3642 3644 := RowAgreement.join atomic_3643_agreement atomic_3644_agreement
private def within_028_055 : CountSegment 1 10 13120 3640 3644 1715 1717 := CountSegment.join within_028_053 within_028_054
private theorem within_028_055_agreement : RowAgreement 3640 3644 := RowAgreement.join within_028_053_agreement within_028_054_agreement
private def within_028_056 : CountSegment 1 10 13120 3644 3646 1717 1717 := CountSegment.join atomic_3645 atomic_3646
private theorem within_028_056_agreement : RowAgreement 3644 3646 := RowAgreement.join atomic_3645_agreement atomic_3646_agreement
private def within_028_057 : CountSegment 1 10 13120 3646 3648 1717 1718 := CountSegment.join atomic_3647 atomic_3648
private theorem within_028_057_agreement : RowAgreement 3646 3648 := RowAgreement.join atomic_3647_agreement atomic_3648_agreement
private def within_028_058 : CountSegment 1 10 13120 3644 3648 1717 1718 := CountSegment.join within_028_056 within_028_057
private theorem within_028_058_agreement : RowAgreement 3644 3648 := RowAgreement.join within_028_056_agreement within_028_057_agreement
private def within_028_059 : CountSegment 1 10 13120 3640 3648 1715 1718 := CountSegment.join within_028_055 within_028_058
private theorem within_028_059_agreement : RowAgreement 3640 3648 := RowAgreement.join within_028_055_agreement within_028_058_agreement
private def within_028_060 : CountSegment 1 10 13120 3632 3648 1713 1718 := CountSegment.join within_028_052 within_028_059
private theorem within_028_060_agreement : RowAgreement 3632 3648 := RowAgreement.join within_028_052_agreement within_028_059_agreement
private def within_028_061 : CountSegment 1 10 13120 3616 3648 1708 1718 := CountSegment.join within_028_045 within_028_060
private theorem within_028_061_agreement : RowAgreement 3616 3648 := RowAgreement.join within_028_045_agreement within_028_060_agreement
private def within_028_062 : CountSegment 1 10 13120 3584 3648 1697 1718 := CountSegment.join within_028_030 within_028_061
private theorem within_028_062_agreement : RowAgreement 3584 3648 := RowAgreement.join within_028_030_agreement within_028_061_agreement
private def within_028_063 : CountSegment 1 10 13120 3648 3650 1718 1718 := CountSegment.join atomic_3649 atomic_3650
private theorem within_028_063_agreement : RowAgreement 3648 3650 := RowAgreement.join atomic_3649_agreement atomic_3650_agreement
private def within_028_064 : CountSegment 1 10 13120 3650 3652 1718 1720 := CountSegment.join atomic_3651 atomic_3652
private theorem within_028_064_agreement : RowAgreement 3650 3652 := RowAgreement.join atomic_3651_agreement atomic_3652_agreement
private def within_028_065 : CountSegment 1 10 13120 3648 3652 1718 1720 := CountSegment.join within_028_063 within_028_064
private theorem within_028_065_agreement : RowAgreement 3648 3652 := RowAgreement.join within_028_063_agreement within_028_064_agreement
private def within_028_066 : CountSegment 1 10 13120 3652 3654 1720 1720 := CountSegment.join atomic_3653 atomic_3654
private theorem within_028_066_agreement : RowAgreement 3652 3654 := RowAgreement.join atomic_3653_agreement atomic_3654_agreement
private def within_028_067 : CountSegment 1 10 13120 3654 3656 1720 1721 := CountSegment.join atomic_3655 atomic_3656
private theorem within_028_067_agreement : RowAgreement 3654 3656 := RowAgreement.join atomic_3655_agreement atomic_3656_agreement
private def within_028_068 : CountSegment 1 10 13120 3652 3656 1720 1721 := CountSegment.join within_028_066 within_028_067
private theorem within_028_068_agreement : RowAgreement 3652 3656 := RowAgreement.join within_028_066_agreement within_028_067_agreement
private def within_028_069 : CountSegment 1 10 13120 3648 3656 1718 1721 := CountSegment.join within_028_065 within_028_068
private theorem within_028_069_agreement : RowAgreement 3648 3656 := RowAgreement.join within_028_065_agreement within_028_068_agreement
private def within_028_070 : CountSegment 1 10 13120 3656 3658 1721 1723 := CountSegment.join atomic_3657 atomic_3658
private theorem within_028_070_agreement : RowAgreement 3656 3658 := RowAgreement.join atomic_3657_agreement atomic_3658_agreement
private def within_028_071 : CountSegment 1 10 13120 3658 3660 1723 1723 := CountSegment.join atomic_3659 atomic_3660
private theorem within_028_071_agreement : RowAgreement 3658 3660 := RowAgreement.join atomic_3659_agreement atomic_3660_agreement
private def within_028_072 : CountSegment 1 10 13120 3656 3660 1721 1723 := CountSegment.join within_028_070 within_028_071
private theorem within_028_072_agreement : RowAgreement 3656 3660 := RowAgreement.join within_028_070_agreement within_028_071_agreement
private def within_028_073 : CountSegment 1 10 13120 3660 3662 1723 1723 := CountSegment.join atomic_3661 atomic_3662
private theorem within_028_073_agreement : RowAgreement 3660 3662 := RowAgreement.join atomic_3661_agreement atomic_3662_agreement
private def within_028_074 : CountSegment 1 10 13120 3662 3664 1723 1724 := CountSegment.join atomic_3663 atomic_3664
private theorem within_028_074_agreement : RowAgreement 3662 3664 := RowAgreement.join atomic_3663_agreement atomic_3664_agreement
private def within_028_075 : CountSegment 1 10 13120 3660 3664 1723 1724 := CountSegment.join within_028_073 within_028_074
private theorem within_028_075_agreement : RowAgreement 3660 3664 := RowAgreement.join within_028_073_agreement within_028_074_agreement
private def within_028_076 : CountSegment 1 10 13120 3656 3664 1721 1724 := CountSegment.join within_028_072 within_028_075
private theorem within_028_076_agreement : RowAgreement 3656 3664 := RowAgreement.join within_028_072_agreement within_028_075_agreement
private def within_028_077 : CountSegment 1 10 13120 3648 3664 1718 1724 := CountSegment.join within_028_069 within_028_076
private theorem within_028_077_agreement : RowAgreement 3648 3664 := RowAgreement.join within_028_069_agreement within_028_076_agreement
private def within_028_078 : CountSegment 1 10 13120 3664 3666 1724 1724 := CountSegment.join atomic_3665 atomic_3666
private theorem within_028_078_agreement : RowAgreement 3664 3666 := RowAgreement.join atomic_3665_agreement atomic_3666_agreement
private def within_028_079 : CountSegment 1 10 13120 3666 3668 1724 1725 := CountSegment.join atomic_3667 atomic_3668
private theorem within_028_079_agreement : RowAgreement 3666 3668 := RowAgreement.join atomic_3667_agreement atomic_3668_agreement
private def within_028_080 : CountSegment 1 10 13120 3664 3668 1724 1725 := CountSegment.join within_028_078 within_028_079
private theorem within_028_080_agreement : RowAgreement 3664 3668 := RowAgreement.join within_028_078_agreement within_028_079_agreement
private def within_028_081 : CountSegment 1 10 13120 3668 3670 1725 1725 := CountSegment.join atomic_3669 atomic_3670
private theorem within_028_081_agreement : RowAgreement 3668 3670 := RowAgreement.join atomic_3669_agreement atomic_3670_agreement
private def within_028_082 : CountSegment 1 10 13120 3670 3672 1725 1727 := CountSegment.join atomic_3671 atomic_3672
private theorem within_028_082_agreement : RowAgreement 3670 3672 := RowAgreement.join atomic_3671_agreement atomic_3672_agreement
private def within_028_083 : CountSegment 1 10 13120 3668 3672 1725 1727 := CountSegment.join within_028_081 within_028_082
private theorem within_028_083_agreement : RowAgreement 3668 3672 := RowAgreement.join within_028_081_agreement within_028_082_agreement
private def within_028_084 : CountSegment 1 10 13120 3664 3672 1724 1727 := CountSegment.join within_028_080 within_028_083
private theorem within_028_084_agreement : RowAgreement 3664 3672 := RowAgreement.join within_028_080_agreement within_028_083_agreement
private def within_028_085 : CountSegment 1 10 13120 3672 3674 1727 1727 := CountSegment.join atomic_3673 atomic_3674
private theorem within_028_085_agreement : RowAgreement 3672 3674 := RowAgreement.join atomic_3673_agreement atomic_3674_agreement
private def within_028_086 : CountSegment 1 10 13120 3674 3676 1727 1727 := CountSegment.join atomic_3675 atomic_3676
private theorem within_028_086_agreement : RowAgreement 3674 3676 := RowAgreement.join atomic_3675_agreement atomic_3676_agreement
private def within_028_087 : CountSegment 1 10 13120 3672 3676 1727 1727 := CountSegment.join within_028_085 within_028_086
private theorem within_028_087_agreement : RowAgreement 3672 3676 := RowAgreement.join within_028_085_agreement within_028_086_agreement
private def within_028_088 : CountSegment 1 10 13120 3676 3678 1727 1729 := CountSegment.join atomic_3677 atomic_3678
private theorem within_028_088_agreement : RowAgreement 3676 3678 := RowAgreement.join atomic_3677_agreement atomic_3678_agreement
private def within_028_089 : CountSegment 1 10 13120 3678 3680 1729 1730 := CountSegment.join atomic_3679 atomic_3680
private theorem within_028_089_agreement : RowAgreement 3678 3680 := RowAgreement.join atomic_3679_agreement atomic_3680_agreement
private def within_028_090 : CountSegment 1 10 13120 3676 3680 1727 1730 := CountSegment.join within_028_088 within_028_089
private theorem within_028_090_agreement : RowAgreement 3676 3680 := RowAgreement.join within_028_088_agreement within_028_089_agreement
private def within_028_091 : CountSegment 1 10 13120 3672 3680 1727 1730 := CountSegment.join within_028_087 within_028_090
private theorem within_028_091_agreement : RowAgreement 3672 3680 := RowAgreement.join within_028_087_agreement within_028_090_agreement
private def within_028_092 : CountSegment 1 10 13120 3664 3680 1724 1730 := CountSegment.join within_028_084 within_028_091
private theorem within_028_092_agreement : RowAgreement 3664 3680 := RowAgreement.join within_028_084_agreement within_028_091_agreement
private def within_028_093 : CountSegment 1 10 13120 3648 3680 1718 1730 := CountSegment.join within_028_077 within_028_092
private theorem within_028_093_agreement : RowAgreement 3648 3680 := RowAgreement.join within_028_077_agreement within_028_092_agreement
private def within_028_094 : CountSegment 1 10 13120 3680 3682 1730 1731 := CountSegment.join atomic_3681 atomic_3682
private theorem within_028_094_agreement : RowAgreement 3680 3682 := RowAgreement.join atomic_3681_agreement atomic_3682_agreement
private def within_028_095 : CountSegment 1 10 13120 3682 3684 1731 1732 := CountSegment.join atomic_3683 atomic_3684
private theorem within_028_095_agreement : RowAgreement 3682 3684 := RowAgreement.join atomic_3683_agreement atomic_3684_agreement
private def within_028_096 : CountSegment 1 10 13120 3680 3684 1730 1732 := CountSegment.join within_028_094 within_028_095
private theorem within_028_096_agreement : RowAgreement 3680 3684 := RowAgreement.join within_028_094_agreement within_028_095_agreement
private def within_028_097 : CountSegment 1 10 13120 3684 3686 1732 1732 := CountSegment.join atomic_3685 atomic_3686
private theorem within_028_097_agreement : RowAgreement 3684 3686 := RowAgreement.join atomic_3685_agreement atomic_3686_agreement
private def within_028_098 : CountSegment 1 10 13120 3686 3688 1732 1732 := CountSegment.join atomic_3687 atomic_3688
private theorem within_028_098_agreement : RowAgreement 3686 3688 := RowAgreement.join atomic_3687_agreement atomic_3688_agreement
private def within_028_099 : CountSegment 1 10 13120 3684 3688 1732 1732 := CountSegment.join within_028_097 within_028_098
private theorem within_028_099_agreement : RowAgreement 3684 3688 := RowAgreement.join within_028_097_agreement within_028_098_agreement
private def within_028_100 : CountSegment 1 10 13120 3680 3688 1730 1732 := CountSegment.join within_028_096 within_028_099
private theorem within_028_100_agreement : RowAgreement 3680 3688 := RowAgreement.join within_028_096_agreement within_028_099_agreement
private def within_028_101 : CountSegment 1 10 13120 3688 3690 1732 1732 := CountSegment.join atomic_3689 atomic_3690
private theorem within_028_101_agreement : RowAgreement 3688 3690 := RowAgreement.join atomic_3689_agreement atomic_3690_agreement
private def within_028_102 : CountSegment 1 10 13120 3690 3692 1732 1733 := CountSegment.join atomic_3691 atomic_3692
private theorem within_028_102_agreement : RowAgreement 3690 3692 := RowAgreement.join atomic_3691_agreement atomic_3692_agreement
private def within_028_103 : CountSegment 1 10 13120 3688 3692 1732 1733 := CountSegment.join within_028_101 within_028_102
private theorem within_028_103_agreement : RowAgreement 3688 3692 := RowAgreement.join within_028_101_agreement within_028_102_agreement
private def within_028_104 : CountSegment 1 10 13120 3692 3694 1733 1734 := CountSegment.join atomic_3693 atomic_3694
private theorem within_028_104_agreement : RowAgreement 3692 3694 := RowAgreement.join atomic_3693_agreement atomic_3694_agreement
private def within_028_105 : CountSegment 1 10 13120 3694 3696 1734 1735 := CountSegment.join atomic_3695 atomic_3696
private theorem within_028_105_agreement : RowAgreement 3694 3696 := RowAgreement.join atomic_3695_agreement atomic_3696_agreement
private def within_028_106 : CountSegment 1 10 13120 3692 3696 1733 1735 := CountSegment.join within_028_104 within_028_105
private theorem within_028_106_agreement : RowAgreement 3692 3696 := RowAgreement.join within_028_104_agreement within_028_105_agreement
private def within_028_107 : CountSegment 1 10 13120 3688 3696 1732 1735 := CountSegment.join within_028_103 within_028_106
private theorem within_028_107_agreement : RowAgreement 3688 3696 := RowAgreement.join within_028_103_agreement within_028_106_agreement
private def within_028_108 : CountSegment 1 10 13120 3680 3696 1730 1735 := CountSegment.join within_028_100 within_028_107
private theorem within_028_108_agreement : RowAgreement 3680 3696 := RowAgreement.join within_028_100_agreement within_028_107_agreement
private def within_028_109 : CountSegment 1 10 13120 3696 3698 1735 1735 := CountSegment.join atomic_3697 atomic_3698
private theorem within_028_109_agreement : RowAgreement 3696 3698 := RowAgreement.join atomic_3697_agreement atomic_3698_agreement
private def within_028_110 : CountSegment 1 10 13120 3698 3700 1735 1736 := CountSegment.join atomic_3699 atomic_3700
private theorem within_028_110_agreement : RowAgreement 3698 3700 := RowAgreement.join atomic_3699_agreement atomic_3700_agreement
private def within_028_111 : CountSegment 1 10 13120 3696 3700 1735 1736 := CountSegment.join within_028_109 within_028_110
private theorem within_028_111_agreement : RowAgreement 3696 3700 := RowAgreement.join within_028_109_agreement within_028_110_agreement
private def within_028_112 : CountSegment 1 10 13120 3700 3702 1736 1737 := CountSegment.join atomic_3701 atomic_3702
private theorem within_028_112_agreement : RowAgreement 3700 3702 := RowAgreement.join atomic_3701_agreement atomic_3702_agreement
private def within_028_113 : CountSegment 1 10 13120 3702 3704 1737 1737 := CountSegment.join atomic_3703 atomic_3704
private theorem within_028_113_agreement : RowAgreement 3702 3704 := RowAgreement.join atomic_3703_agreement atomic_3704_agreement
private def within_028_114 : CountSegment 1 10 13120 3700 3704 1736 1737 := CountSegment.join within_028_112 within_028_113
private theorem within_028_114_agreement : RowAgreement 3700 3704 := RowAgreement.join within_028_112_agreement within_028_113_agreement
private def within_028_115 : CountSegment 1 10 13120 3696 3704 1735 1737 := CountSegment.join within_028_111 within_028_114
private theorem within_028_115_agreement : RowAgreement 3696 3704 := RowAgreement.join within_028_111_agreement within_028_114_agreement
private def within_028_116 : CountSegment 1 10 13120 3704 3706 1737 1738 := CountSegment.join atomic_3705 atomic_3706
private theorem within_028_116_agreement : RowAgreement 3704 3706 := RowAgreement.join atomic_3705_agreement atomic_3706_agreement
private def within_028_117 : CountSegment 1 10 13120 3706 3708 1738 1739 := CountSegment.join atomic_3707 atomic_3708
private theorem within_028_117_agreement : RowAgreement 3706 3708 := RowAgreement.join atomic_3707_agreement atomic_3708_agreement
private def within_028_118 : CountSegment 1 10 13120 3704 3708 1737 1739 := CountSegment.join within_028_116 within_028_117
private theorem within_028_118_agreement : RowAgreement 3704 3708 := RowAgreement.join within_028_116_agreement within_028_117_agreement
private def within_028_119 : CountSegment 1 10 13120 3708 3710 1739 1740 := CountSegment.join atomic_3709 atomic_3710
private theorem within_028_119_agreement : RowAgreement 3708 3710 := RowAgreement.join atomic_3709_agreement atomic_3710_agreement
private def within_028_120 : CountSegment 1 10 13120 3710 3712 1740 1740 := CountSegment.join atomic_3711 atomic_3712
private theorem within_028_120_agreement : RowAgreement 3710 3712 := RowAgreement.join atomic_3711_agreement atomic_3712_agreement
private def within_028_121 : CountSegment 1 10 13120 3708 3712 1739 1740 := CountSegment.join within_028_119 within_028_120
private theorem within_028_121_agreement : RowAgreement 3708 3712 := RowAgreement.join within_028_119_agreement within_028_120_agreement
private def within_028_122 : CountSegment 1 10 13120 3704 3712 1737 1740 := CountSegment.join within_028_118 within_028_121
private theorem within_028_122_agreement : RowAgreement 3704 3712 := RowAgreement.join within_028_118_agreement within_028_121_agreement
private def within_028_123 : CountSegment 1 10 13120 3696 3712 1735 1740 := CountSegment.join within_028_115 within_028_122
private theorem within_028_123_agreement : RowAgreement 3696 3712 := RowAgreement.join within_028_115_agreement within_028_122_agreement
private def within_028_124 : CountSegment 1 10 13120 3680 3712 1730 1740 := CountSegment.join within_028_108 within_028_123
private theorem within_028_124_agreement : RowAgreement 3680 3712 := RowAgreement.join within_028_108_agreement within_028_123_agreement
private def within_028_125 : CountSegment 1 10 13120 3648 3712 1718 1740 := CountSegment.join within_028_093 within_028_124
private theorem within_028_125_agreement : RowAgreement 3648 3712 := RowAgreement.join within_028_093_agreement within_028_124_agreement
private def within_028_126 : CountSegment 1 10 13120 3584 3712 1697 1740 := CountSegment.join within_028_062 within_028_125
private theorem within_028_126_agreement : RowAgreement 3584 3712 := RowAgreement.join within_028_062_agreement within_028_125_agreement
def segment_028 : CountSegment 1 10 13120 3584 3712 1697 1740 := within_028_126
theorem agreement_028 : RowAgreement 3584 3712 := within_028_126_agreement
end ProjectEulerSolutions.P58Numerical
