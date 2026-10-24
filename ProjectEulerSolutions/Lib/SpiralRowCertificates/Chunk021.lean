import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_2689 : RowCertificate 2689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2689 1) := by change ¬ Nat.Prime 28928263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4132609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2689 2) := by change ¬ Nat.Prime 28922885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5784577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2689 3) := by change ¬ Nat.Prime 28917507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9639169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2689 : row 2689 = 0 := (row_2689.checked).trans rfl
theorem model_count_2689 : modelRow 2689 = 0 := by decide
theorem modelrow_agrees_2689 : modelRow 2689 = row 2689 := model_count_2689.trans row_count_2689.symm
def row_2690 : RowCertificate 2690 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2690 1) := by change ¬ Nat.Prime 28949781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9649927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2690 2) := by change ¬ Nat.Prime 28944401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 705961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2690 3) := by change ¬ Nat.Prime 28939021; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 104473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2690 : row 2690 = 0 := (row_2690.checked).trans rfl
theorem model_count_2690 : modelRow 2690 = 0 := by decide
theorem modelrow_agrees_2690 : modelRow 2690 = row 2690 := model_count_2690.trans row_count_2690.symm
def row_2691 : RowCertificate 2691 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2691 1) := by change Nat.Prime 28971307; exact ProjectEulerSolutions.P58PrimeNodes.p_28971307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2691 2) := by change ¬ Nat.Prime 28965925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5793185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2691 3) := by change ¬ Nat.Prime 28960543; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 673501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2691 : row 2691 = 1 := (row_2691.checked).trans rfl
theorem model_count_2691 : modelRow 2691 = 1 := by decide
theorem modelrow_agrees_2691 : modelRow 2691 = row 2691 := model_count_2691.trans row_count_2691.symm
def row_2692 : RowCertificate 2692 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2692 1) := by change ¬ Nat.Prime 28992841; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1525939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2692 2) := by change Nat.Prime 28987457; exact ProjectEulerSolutions.P58PrimeNodes.p_28987457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2692 3) := by change ¬ Nat.Prime 28982073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9660691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2692 : row 2692 = 1 := (row_2692.checked).trans rfl
theorem model_count_2692 : modelRow 2692 = 1 := by decide
theorem modelrow_agrees_2692 : modelRow 2692 = row 2692 := model_count_2692.trans row_count_2692.symm
def row_2693 : RowCertificate 2693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2693 1) := by change ¬ Nat.Prime 29014383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9671461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2693 2) := by change ¬ Nat.Prime 29008997; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 38321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2693 3) := by change ¬ Nat.Prime 29003611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4143373) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2693 : row 2693 = 0 := (row_2693.checked).trans rfl
theorem model_count_2693 : modelRow 2693 = 0 := by decide
theorem modelrow_agrees_2693 : modelRow 2693 = row 2693 := model_count_2693.trans row_count_2693.symm
def row_2694 : RowCertificate 2694 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2694 1) := by change ¬ Nat.Prime 29035933; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1528207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2694 2) := by change ¬ Nat.Prime 29030545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5806109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2694 3) := by change ¬ Nat.Prime 29025157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4146451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2694 : row 2694 = 0 := (row_2694.checked).trans rfl
theorem model_count_2694 : modelRow 2694 = 0 := by decide
theorem modelrow_agrees_2694 : modelRow 2694 = row 2694 := model_count_2694.trans row_count_2694.symm
def row_2695 : RowCertificate 2695 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2695 1) := by change Nat.Prime 29057491; exact ProjectEulerSolutions.P58PrimeNodes.p_29057491.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2695 2) := by change ¬ Nat.Prime 29052101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2234777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2695 3) := by change ¬ Nat.Prime 29046711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9682237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2695 : row 2695 = 1 := (row_2695.checked).trans rfl
theorem model_count_2695 : modelRow 2695 = 1 := by decide
theorem modelrow_agrees_2695 : modelRow 2695 = row 2695 := model_count_2695.trans row_count_2695.symm
def row_2696 : RowCertificate 2696 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2696 1) := by change ¬ Nat.Prime 29079057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9693019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2696 2) := by change ¬ Nat.Prime 29073665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5814733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2696 3) := by change ¬ Nat.Prime 29068273; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2236021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2696 : row 2696 = 0 := (row_2696.checked).trans rfl
theorem model_count_2696 : modelRow 2696 = 0 := by decide
theorem modelrow_agrees_2696 : modelRow 2696 = row 2696 := model_count_2696.trans row_count_2696.symm
def row_2697 : RowCertificate 2697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2697 1) := by change ¬ Nat.Prime 29100631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4157233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2697 2) := by change ¬ Nat.Prime 29095237; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 44017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2697 3) := by change ¬ Nat.Prime 29089843; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 398491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2697 : row 2697 = 0 := (row_2697.checked).trans rfl
theorem model_count_2697 : modelRow 2697 = 0 := by decide
theorem modelrow_agrees_2697 : modelRow 2697 = row 2697 := model_count_2697.trans row_count_2697.symm
def row_2698 : RowCertificate 2698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2698 1) := by change ¬ Nat.Prime 29122213; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 300229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2698 2) := by change ¬ Nat.Prime 29116817; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 786941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2698 3) := by change ¬ Nat.Prime 29111421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9703807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2698 : row 2698 = 0 := (row_2698.checked).trans rfl
theorem model_count_2698 : modelRow 2698 = 0 := by decide
theorem modelrow_agrees_2698 : modelRow 2698 = row 2698 := model_count_2698.trans row_count_2698.symm
def row_2699 : RowCertificate 2699 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2699 1) := by change ¬ Nat.Prime 29143803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9714601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2699 2) := by change ¬ Nat.Prime 29138405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5827681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2699 3) := by change ¬ Nat.Prime 29133007; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 434821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2699 : row 2699 = 0 := (row_2699.checked).trans rfl
theorem model_count_2699 : modelRow 2699 = 0 := by decide
theorem modelrow_agrees_2699 : modelRow 2699 = row 2699 := model_count_2699.trans row_count_2699.symm
def row_2700 : RowCertificate 2700 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2700 1) := by change Nat.Prime 29165401; exact ProjectEulerSolutions.P58PrimeNodes.p_29165401.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2700 2) := by change ¬ Nat.Prime 29160001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2243077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2700 3) := by change ¬ Nat.Prime 29154601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4164943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2700 : row 2700 = 1 := (row_2700.checked).trans rfl
theorem model_count_2700 : modelRow 2700 = 1 := by decide
theorem modelrow_agrees_2700 : modelRow 2700 = row 2700 := model_count_2700.trans row_count_2700.symm
def row_2701 : RowCertificate 2701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2701 1) := by change ¬ Nat.Prime 29187007; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 283369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2701 2) := by change ¬ Nat.Prime 29181605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5836321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2701 3) := by change ¬ Nat.Prime 29176203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9725401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2701 : row 2701 = 0 := (row_2701.checked).trans rfl
theorem model_count_2701 : modelRow 2701 = 0 := by decide
theorem modelrow_agrees_2701 : modelRow 2701 = row 2701 := model_count_2701.trans row_count_2701.symm
def row_2702 : RowCertificate 2702 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2702 1) := by change ¬ Nat.Prime 29208621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9736207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2702 2) := by change Nat.Prime 29203217; exact ProjectEulerSolutions.P58PrimeNodes.p_29203217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2702 3) := by change ¬ Nat.Prime 29197813; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1536727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2702 : row 2702 = 1 := (row_2702.checked).trans rfl
theorem model_count_2702 : modelRow 2702 = 1 := by decide
theorem modelrow_agrees_2702 : modelRow 2702 = row 2702 := model_count_2702.trans row_count_2702.symm
def row_2703 : RowCertificate 2703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2703 1) := by change ¬ Nat.Prime 29230243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4175749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2703 2) := by change ¬ Nat.Prime 29224837; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1007753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2703 3) := by change Nat.Prime 29219431; exact ProjectEulerSolutions.P58PrimeNodes.p_29219431.prime
    exact decide_eq_true prime
theorem row_count_2703 : row 2703 = 1 := (row_2703.checked).trans rfl
theorem model_count_2703 : modelRow 2703 = 1 := by decide
theorem modelrow_agrees_2703 : modelRow 2703 = row 2703 := model_count_2703.trans row_count_2703.symm
def row_2704 : RowCertificate 2704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2704 1) := by change ¬ Nat.Prime 29251873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4178839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2704 2) := by change ¬ Nat.Prime 29246465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5849293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2704 3) := by change ¬ Nat.Prime 29241057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9747019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2704 : row 2704 = 0 := (row_2704.checked).trans rfl
theorem model_count_2704 : modelRow 2704 = 0 := by decide
theorem modelrow_agrees_2704 : modelRow 2704 = row 2704 := model_count_2704.trans row_count_2704.symm
def row_2705 : RowCertificate 2705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2705 1) := by change ¬ Nat.Prime 29273511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9757837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2705 2) := by change ¬ Nat.Prime 29268101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1721653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2705 3) := by change Nat.Prime 29262691; exact ProjectEulerSolutions.P58PrimeNodes.p_29262691.prime
    exact decide_eq_true prime
theorem row_count_2705 : row 2705 = 1 := (row_2705.checked).trans rfl
theorem model_count_2705 : modelRow 2705 = 1 := by decide
theorem modelrow_agrees_2705 : modelRow 2705 = row 2705 := model_count_2705.trans row_count_2705.symm
def row_2706 : RowCertificate 2706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2706 1) := by change ¬ Nat.Prime 29295157; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 791761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2706 2) := by change ¬ Nat.Prime 29289745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5857949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2706 3) := by change ¬ Nat.Prime 29284333; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2252641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2706 : row 2706 = 0 := (row_2706.checked).trans rfl
theorem model_count_2706 : modelRow 2706 = 0 := by decide
theorem modelrow_agrees_2706 : modelRow 2706 = row 2706 := model_count_2706.trans row_count_2706.symm
def row_2707 : RowCertificate 2707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2707 1) := by change ¬ Nat.Prime 29316811; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 34129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2707 2) := by change ¬ Nat.Prime 29311397; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 49429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2707 3) := by change ¬ Nat.Prime 29305983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9768661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2707 : row 2707 = 0 := (row_2707.checked).trans rfl
theorem model_count_2707 : modelRow 2707 = 0 := by decide
theorem modelrow_agrees_2707 : modelRow 2707 = row 2707 := model_count_2707.trans row_count_2707.symm
def row_2708 : RowCertificate 2708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2708 1) := by change ¬ Nat.Prime 29338473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9779491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2708 2) := by change ¬ Nat.Prime 29333057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2256389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2708 3) := by change ¬ Nat.Prime 29327641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4189663) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2708 : row 2708 = 0 := (row_2708.checked).trans rfl
theorem model_count_2708 : modelRow 2708 = 0 := by decide
theorem modelrow_agrees_2708 : modelRow 2708 = row 2708 := model_count_2708.trans row_count_2708.symm
def row_2709 : RowCertificate 2709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2709 1) := by change ¬ Nat.Prime 29360143; exact PrattCertificates.composite_of_product (factor := 5107) (cofactor := 5749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2709 2) := by change ¬ Nat.Prime 29354725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5870945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2709 3) := by change ¬ Nat.Prime 29349307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2257639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2709 : row 2709 = 0 := (row_2709.checked).trans rfl
theorem model_count_2709 : modelRow 2709 = 0 := by decide
theorem modelrow_agrees_2709 : modelRow 2709 = row 2709 := model_count_2709.trans row_count_2709.symm
def row_2710 : RowCertificate 2710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2710 1) := by change ¬ Nat.Prime 29381821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4197403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2710 2) := by change Nat.Prime 29376401; exact ProjectEulerSolutions.P58PrimeNodes.p_29376401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2710 3) := by change ¬ Nat.Prime 29370981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9790327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2710 : row 2710 = 1 := (row_2710.checked).trans rfl
theorem model_count_2710 : modelRow 2710 = 1 := by decide
theorem modelrow_agrees_2710 : modelRow 2710 = row 2710 := model_count_2710.trans row_count_2710.symm
def row_2711 : RowCertificate 2711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2711 1) := by change ¬ Nat.Prime 29403507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9801169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2711 2) := by change ¬ Nat.Prime 29398085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5879617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2711 3) := by change ¬ Nat.Prime 29392663; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 103861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2711 : row 2711 = 0 := (row_2711.checked).trans rfl
theorem model_count_2711 : modelRow 2711 = 0 := by decide
theorem modelrow_agrees_2711 : modelRow 2711 = row 2711 := model_count_2711.trans row_count_2711.symm
def row_2712 : RowCertificate 2712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2712 1) := by change ¬ Nat.Prime 29425201; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2263477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2712 2) := by change Nat.Prime 29419777; exact ProjectEulerSolutions.P58PrimeNodes.p_29419777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2712 3) := by change ¬ Nat.Prime 29414353; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 106189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2712 : row 2712 = 1 := (row_2712.checked).trans rfl
theorem model_count_2712 : modelRow 2712 = 1 := by decide
theorem modelrow_agrees_2712 : modelRow 2712 = row 2712 := model_count_2712.trans row_count_2712.symm
def row_2713 : RowCertificate 2713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2713 1) := by change ¬ Nat.Prime 29446903; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1549837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2713 2) := by change ¬ Nat.Prime 29441477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2264729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2713 3) := by change ¬ Nat.Prime 29436051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9812017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2713 : row 2713 = 0 := (row_2713.checked).trans rfl
theorem model_count_2713 : modelRow 2713 = 0 := by decide
theorem modelrow_agrees_2713 : modelRow 2713 = row 2713 := model_count_2713.trans row_count_2713.symm
def row_2714 : RowCertificate 2714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2714 1) := by change ¬ Nat.Prime 29468613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9822871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2714 2) := by change ¬ Nat.Prime 29463185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5892637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2714 3) := by change ¬ Nat.Prime 29457757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4208251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2714 : row 2714 = 0 := (row_2714.checked).trans rfl
theorem model_count_2714 : modelRow 2714 = 0 := by decide
theorem modelrow_agrees_2714 : modelRow 2714 = row 2714 := model_count_2714.trans row_count_2714.symm
def row_2715 : RowCertificate 2715 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2715 1) := by change ¬ Nat.Prime 29490331; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2268487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2715 2) := by change Nat.Prime 29484901; exact ProjectEulerSolutions.P58PrimeNodes.p_29484901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2715 3) := by change ¬ Nat.Prime 29479471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4211353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2715 : row 2715 = 1 := (row_2715.checked).trans rfl
theorem model_count_2715 : modelRow 2715 = 1 := by decide
theorem modelrow_agrees_2715 : modelRow 2715 = row 2715 := model_count_2715.trans row_count_2715.symm
def row_2716 : RowCertificate 2716 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2716 1) := by change Nat.Prime 29512057; exact ProjectEulerSolutions.P58PrimeNodes.p_29512057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2716 2) := by change ¬ Nat.Prime 29506625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5901325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2716 3) := by change ¬ Nat.Prime 29501193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9833731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2716 : row 2716 = 1 := (row_2716.checked).trans rfl
theorem model_count_2716 : modelRow 2716 = 1 := by decide
theorem modelrow_agrees_2716 : modelRow 2716 = row 2716 := model_count_2716.trans row_count_2716.symm
def row_2717 : RowCertificate 2717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2717 1) := by change ¬ Nat.Prime 29533791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9844597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2717 2) := by change ¬ Nat.Prime 29528357; exact PrattCertificates.composite_of_product (factor := 3389) (cofactor := 8713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2717 3) := by change ¬ Nat.Prime 29522923; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 49123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2717 : row 2717 = 0 := (row_2717.checked).trans rfl
theorem model_count_2717 : modelRow 2717 = 0 := by decide
theorem modelrow_agrees_2717 : modelRow 2717 = row 2717 := model_count_2717.trans row_count_2717.symm
def row_2718 : RowCertificate 2718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2718 1) := by change ¬ Nat.Prime 29555533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4222219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2718 2) := by change ¬ Nat.Prime 29550097; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1738241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2718 3) := by change ¬ Nat.Prime 29544661; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 28111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2718 : row 2718 = 0 := (row_2718.checked).trans rfl
theorem model_count_2718 : modelRow 2718 = 0 := by decide
theorem modelrow_agrees_2718 : modelRow 2718 = row 2718 := model_count_2718.trans row_count_2718.symm
def row_2719 : RowCertificate 2719 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2719 1) := by change ¬ Nat.Prime 29577283; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 40351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2719 2) := by change ¬ Nat.Prime 29571845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5914369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2719 3) := by change ¬ Nat.Prime 29566407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9855469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2719 : row 2719 = 0 := (row_2719.checked).trans rfl
theorem model_count_2719 : modelRow 2719 = 0 := by decide
theorem modelrow_agrees_2719 : modelRow 2719 = row 2719 := model_count_2719.trans row_count_2719.symm
def row_2720 : RowCertificate 2720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2720 1) := by change ¬ Nat.Prime 29599041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9866347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2720 2) := by change ¬ Nat.Prime 29593601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1020469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2720 3) := by change ¬ Nat.Prime 29588161; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 67399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2720 : row 2720 = 0 := (row_2720.checked).trans rfl
theorem model_count_2720 : modelRow 2720 = 0 := by decide
theorem modelrow_agrees_2720 : modelRow 2720 = row 2720 := model_count_2720.trans row_count_2720.symm
def row_2721 : RowCertificate 2721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2721 1) := by change ¬ Nat.Prime 29620807; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 485587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2721 2) := by change ¬ Nat.Prime 29615365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5923073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2721 3) := by change ¬ Nat.Prime 29609923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4229989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2721 : row 2721 = 0 := (row_2721.checked).trans rfl
theorem model_count_2721 : modelRow 2721 = 0 := by decide
theorem modelrow_agrees_2721 : modelRow 2721 = row 2721 := model_count_2721.trans row_count_2721.symm
def row_2722 : RowCertificate 2722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2722 1) := by change ¬ Nat.Prime 29642581; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 41809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2722 2) := by change ¬ Nat.Prime 29637137; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1743361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2722 3) := by change ¬ Nat.Prime 29631693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9877231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2722 : row 2722 = 0 := (row_2722.checked).trans rfl
theorem model_count_2722 : modelRow 2722 = 0 := by decide
theorem modelrow_agrees_2722 : modelRow 2722 = row 2722 := model_count_2722.trans row_count_2722.symm
def row_2723 : RowCertificate 2723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2723 1) := by change ¬ Nat.Prime 29664363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9888121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2723 2) := by change Nat.Prime 29658917; exact ProjectEulerSolutions.P58PrimeNodes.p_29658917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2723 3) := by change ¬ Nat.Prime 29653471; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1560709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2723 : row 2723 = 1 := (row_2723.checked).trans rfl
theorem model_count_2723 : modelRow 2723 = 1 := by decide
theorem modelrow_agrees_2723 : modelRow 2723 = row 2723 := model_count_2723.trans row_count_2723.symm
def row_2724 : RowCertificate 2724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2724 1) := by change ¬ Nat.Prime 29686153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4240879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2724 2) := by change ¬ Nat.Prime 29680705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5936141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2724 3) := by change Nat.Prime 29675257; exact ProjectEulerSolutions.P58PrimeNodes.p_29675257.prime
    exact decide_eq_true prime
theorem row_count_2724 : row 2724 = 1 := (row_2724.checked).trans rfl
theorem model_count_2724 : modelRow 2724 = 1 := by decide
theorem modelrow_agrees_2724 : modelRow 2724 = row 2724 := model_count_2724.trans row_count_2724.symm
def row_2725 : RowCertificate 2725 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2725 1) := by change ¬ Nat.Prime 29707951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4243993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2725 2) := by change ¬ Nat.Prime 29702501; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 68597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2725 3) := by change ¬ Nat.Prime 29697051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9899017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2725 : row 2725 = 0 := (row_2725.checked).trans rfl
theorem model_count_2725 : modelRow 2725 = 0 := by decide
theorem modelrow_agrees_2725 : modelRow 2725 = row 2725 := model_count_2725.trans row_count_2725.symm
def row_2726 : RowCertificate 2726 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2726 1) := by change ¬ Nat.Prime 29729757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9909919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2726 2) := by change ¬ Nat.Prime 29724305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5944861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2726 3) := by change ¬ Nat.Prime 29718853; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 48481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2726 : row 2726 = 0 := (row_2726.checked).trans rfl
theorem model_count_2726 : modelRow 2726 = 0 := by decide
theorem modelrow_agrees_2726 : modelRow 2726 = row 2726 := model_count_2726.trans row_count_2726.symm
def row_2727 : RowCertificate 2727 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2727 1) := by change ¬ Nat.Prime 29751571; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 691897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2727 2) := by change ¬ Nat.Prime 29746117; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 306661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2727 3) := by change Nat.Prime 29740663; exact ProjectEulerSolutions.P58PrimeNodes.p_29740663.prime
    exact decide_eq_true prime
theorem row_count_2727 : row 2727 = 1 := (row_2727.checked).trans rfl
theorem model_count_2727 : modelRow 2727 = 1 := by decide
theorem modelrow_agrees_2727 : modelRow 2727 = row 2727 := model_count_2727.trans row_count_2727.symm
def row_2728 : RowCertificate 2728 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2728 1) := by change ¬ Nat.Prime 29773393; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2290261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2728 2) := by change ¬ Nat.Prime 29767937; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 172069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2728 3) := by change ¬ Nat.Prime 29762481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9920827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2728 : row 2728 = 0 := (row_2728.checked).trans rfl
theorem model_count_2728 : modelRow 2728 = 0 := by decide
theorem modelrow_agrees_2728 : modelRow 2728 = row 2728 := model_count_2728.trans row_count_2728.symm
def row_2729 : RowCertificate 2729 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2729 1) := by change ¬ Nat.Prime 29795223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9931741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2729 2) := by change ¬ Nat.Prime 29789765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5957953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2729 3) := by change ¬ Nat.Prime 29784307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4254901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2729 : row 2729 = 0 := (row_2729.checked).trans rfl
theorem model_count_2729 : modelRow 2729 = 0 := by decide
theorem modelrow_agrees_2729 : modelRow 2729 = row 2729 := model_count_2729.trans row_count_2729.symm
def row_2730 : RowCertificate 2730 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2730 1) := by change ¬ Nat.Prime 29817061; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1569319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2730 2) := by change ¬ Nat.Prime 29811601; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 72889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2730 3) := by change Nat.Prime 29806141; exact ProjectEulerSolutions.P58PrimeNodes.p_29806141.prime
    exact decide_eq_true prime
theorem row_count_2730 : row 2730 = 1 := (row_2730.checked).trans rfl
theorem model_count_2730 : modelRow 2730 = 1 := by decide
theorem modelrow_agrees_2730 : modelRow 2730 = row 2730 := model_count_2730.trans row_count_2730.symm
def row_2731 : RowCertificate 2731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2731 1) := by change ¬ Nat.Prime 29838907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4262701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2731 2) := by change ¬ Nat.Prime 29833445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5966689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2731 3) := by change ¬ Nat.Prime 29827983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9942661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2731 : row 2731 = 0 := (row_2731.checked).trans rfl
theorem model_count_2731 : modelRow 2731 = 0 := by decide
theorem modelrow_agrees_2731 : modelRow 2731 = row 2731 := model_count_2731.trans row_count_2731.symm
def row_2732 : RowCertificate 2732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2732 1) := by change ¬ Nat.Prime 29860761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9953587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2732 2) := by change ¬ Nat.Prime 29855297; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1029493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2732 3) := by change ¬ Nat.Prime 29849833; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2296141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2732 : row 2732 = 0 := (row_2732.checked).trans rfl
theorem model_count_2732 : modelRow 2732 = 0 := by decide
theorem modelrow_agrees_2732 : modelRow 2732 = row 2732 := model_count_2732.trans row_count_2732.symm
def row_2733 : RowCertificate 2733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2733 1) := by change ¬ Nat.Prime 29882623; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 409351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2733 2) := by change Nat.Prime 29877157; exact ProjectEulerSolutions.P58PrimeNodes.p_29877157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2733 3) := by change ¬ Nat.Prime 29871691; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 807343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2733 : row 2733 = 1 := (row_2733.checked).trans rfl
theorem model_count_2733 : modelRow 2733 = 1 := by decide
theorem modelrow_agrees_2733 : modelRow 2733 = row 2733 := model_count_2733.trans row_count_2733.symm
def row_2734 : RowCertificate 2734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2734 1) := by change ¬ Nat.Prime 29904493; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 198043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2734 2) := by change ¬ Nat.Prime 29899025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5979805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2734 3) := by change ¬ Nat.Prime 29893557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9964519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2734 : row 2734 = 0 := (row_2734.checked).trans rfl
theorem model_count_2734 : modelRow 2734 = 0 := by decide
theorem modelrow_agrees_2734 : modelRow 2734 = row 2734 := model_count_2734.trans row_count_2734.symm
def row_2735 : RowCertificate 2735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2735 1) := by change ¬ Nat.Prime 29926371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9975457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2735 2) := by change ¬ Nat.Prime 29920901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1760053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2735 3) := by change ¬ Nat.Prime 29915431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4273633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2735 : row 2735 = 0 := (row_2735.checked).trans rfl
theorem model_count_2735 : modelRow 2735 = 0 := by decide
theorem modelrow_agrees_2735 : modelRow 2735 = row 2735 := model_count_2735.trans row_count_2735.symm
def row_2736 : RowCertificate 2736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2736 1) := by change ¬ Nat.Prime 29948257; exact PrattCertificates.composite_of_product (factor := 1489) (cofactor := 20113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2736 2) := by change ¬ Nat.Prime 29942785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 5988557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2736 3) := by change ¬ Nat.Prime 29937313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4276759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2736 : row 2736 = 0 := (row_2736.checked).trans rfl
theorem model_count_2736 : modelRow 2736 = 0 := by decide
theorem modelrow_agrees_2736 : modelRow 2736 = row 2736 := model_count_2736.trans row_count_2736.symm
def row_2737 : RowCertificate 2737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2737 1) := by change ¬ Nat.Prime 29970151; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 379369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2737 2) := by change Nat.Prime 29964677; exact ProjectEulerSolutions.P58PrimeNodes.p_29964677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2737 3) := by change ¬ Nat.Prime 29959203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9986401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2737 : row 2737 = 1 := (row_2737.checked).trans rfl
theorem model_count_2737 : modelRow 2737 = 1 := by decide
theorem modelrow_agrees_2737 : modelRow 2737 = row 2737 := model_count_2737.trans row_count_2737.symm
def row_2738 : RowCertificate 2738 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2738 1) := by change ¬ Nat.Prime 29992053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 9997351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2738 2) := by change Nat.Prime 29986577; exact ProjectEulerSolutions.P58PrimeNodes.p_29986577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2738 3) := by change ¬ Nat.Prime 29981101; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 12613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2738 : row 2738 = 1 := (row_2738.checked).trans rfl
theorem model_count_2738 : modelRow 2738 = 1 := by decide
theorem modelrow_agrees_2738 : modelRow 2738 = row 2738 := model_count_2738.trans row_count_2738.symm
def row_2739 : RowCertificate 2739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2739 1) := by change ¬ Nat.Prime 30013963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4287709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2739 2) := by change ¬ Nat.Prime 30008485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6001697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2739 3) := by change ¬ Nat.Prime 30003007; exact PrattCertificates.composite_of_product (factor := 1531) (cofactor := 19597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2739 : row 2739 = 0 := (row_2739.checked).trans rfl
theorem model_count_2739 : modelRow 2739 = 0 := by decide
theorem modelrow_agrees_2739 : modelRow 2739 = row 2739 := model_count_2739.trans row_count_2739.symm
def row_2740 : RowCertificate 2740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2740 1) := by change ¬ Nat.Prime 30035881; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 236503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2740 2) := by change ¬ Nat.Prime 30030401; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 108413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2740 3) := by change ¬ Nat.Prime 30024921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10008307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2740 : row 2740 = 0 := (row_2740.checked).trans rfl
theorem model_count_2740 : modelRow 2740 = 0 := by decide
theorem modelrow_agrees_2740 : modelRow 2740 = row 2740 := model_count_2740.trans row_count_2740.symm
def row_2741 : RowCertificate 2741 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2741 1) := by change ¬ Nat.Prime 30057807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10019269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2741 2) := by change ¬ Nat.Prime 30052325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6010465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2741 3) := by change ¬ Nat.Prime 30046843; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 969253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2741 : row 2741 = 0 := (row_2741.checked).trans rfl
theorem model_count_2741 : modelRow 2741 = 0 := by decide
theorem modelrow_agrees_2741 : modelRow 2741 = row 2741 := model_count_2741.trans row_count_2741.symm
def row_2742 : RowCertificate 2742 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2742 1) := by change Nat.Prime 30079741; exact ProjectEulerSolutions.P58PrimeNodes.p_30079741.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2742 2) := by change ¬ Nat.Prime 30074257; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 337913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2742 3) := by change ¬ Nat.Prime 30068773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4295539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2742 : row 2742 = 1 := (row_2742.checked).trans rfl
theorem model_count_2742 : modelRow 2742 = 1 := by decide
theorem modelrow_agrees_2742 : modelRow 2742 = row 2742 := model_count_2742.trans row_count_2742.symm
def row_2743 : RowCertificate 2743 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2743 1) := by change ¬ Nat.Prime 30101683; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 813559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2743 2) := by change Nat.Prime 30096197; exact ProjectEulerSolutions.P58PrimeNodes.p_30096197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2743 3) := by change ¬ Nat.Prime 30090711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10030237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2743 : row 2743 = 1 := (row_2743.checked).trans rfl
theorem model_count_2743 : modelRow 2743 = 1 := by decide
theorem modelrow_agrees_2743 : modelRow 2743 = row 2743 := model_count_2743.trans row_count_2743.symm
def row_2744 : RowCertificate 2744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2744 1) := by change ¬ Nat.Prime 30123633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10041211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2744 2) := by change ¬ Nat.Prime 30118145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6023629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2744 3) := by change Nat.Prime 30112657; exact ProjectEulerSolutions.P58PrimeNodes.p_30112657.prime
    exact decide_eq_true prime
theorem row_count_2744 : row 2744 = 1 := (row_2744.checked).trans rfl
theorem model_count_2744 : modelRow 2744 = 1 := by decide
theorem modelrow_agrees_2744 : modelRow 2744 = row 2744 := model_count_2744.trans row_count_2744.symm
def row_2745 : RowCertificate 2745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2745 1) := by change ¬ Nat.Prime 30145591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4306513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2745 2) := by change Nat.Prime 30140101; exact ProjectEulerSolutions.P58PrimeNodes.p_30140101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2745 3) := by change ¬ Nat.Prime 30134611; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2318047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2745 : row 2745 = 1 := (row_2745.checked).trans rfl
theorem model_count_2745 : modelRow 2745 = 1 := by decide
theorem modelrow_agrees_2745 : modelRow 2745 = row 2745 := model_count_2745.trans row_count_2745.symm
def row_2746 : RowCertificate 2746 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2746 1) := by change ¬ Nat.Prime 30167557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4309651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2746 2) := by change ¬ Nat.Prime 30162065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6032413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2746 3) := by change ¬ Nat.Prime 30156573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10052191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2746 : row 2746 = 0 := (row_2746.checked).trans rfl
theorem model_count_2746 : modelRow 2746 = 0 := by decide
theorem modelrow_agrees_2746 : modelRow 2746 = row 2746 := model_count_2746.trans row_count_2746.symm
def row_2747 : RowCertificate 2747 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2747 1) := by change ¬ Nat.Prime 30189531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10063177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2747 2) := by change ¬ Nat.Prime 30184037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2321849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2747 3) := by change ¬ Nat.Prime 30178543; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 311119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2747 : row 2747 = 0 := (row_2747.checked).trans rfl
theorem model_count_2747 : modelRow 2747 = 0 := by decide
theorem modelrow_agrees_2747 : modelRow 2747 = row 2747 := model_count_2747.trans row_count_2747.symm
def row_2748 : RowCertificate 2748 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2748 1) := by change ¬ Nat.Prime 30211513; exact PrattCertificates.composite_of_product (factor := 1459) (cofactor := 20707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2748 2) := by change ¬ Nat.Prime 30206017; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 57977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2748 3) := by change ¬ Nat.Prime 30200521; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2323117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2748 : row 2748 = 0 := (row_2748.checked).trans rfl
theorem model_count_2748 : modelRow 2748 = 0 := by decide
theorem modelrow_agrees_2748 : modelRow 2748 = row 2748 := model_count_2748.trans row_count_2748.symm
def row_2749 : RowCertificate 2749 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2749 1) := by change ¬ Nat.Prime 30233503; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1591237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2749 2) := by change ¬ Nat.Prime 30228005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6045601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2749 3) := by change ¬ Nat.Prime 30222507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10074169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2749 : row 2749 = 0 := (row_2749.checked).trans rfl
theorem model_count_2749 : modelRow 2749 = 0 := by decide
theorem modelrow_agrees_2749 : modelRow 2749 = row 2749 := model_count_2749.trans row_count_2749.symm
def row_2750 : RowCertificate 2750 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2750 1) := by change ¬ Nat.Prime 30255501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10085167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2750 2) := by change Nat.Prime 30250001; exact ProjectEulerSolutions.P58PrimeNodes.p_30250001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2750 3) := by change ¬ Nat.Prime 30244501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4320643) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2750 : row 2750 = 1 := (row_2750.checked).trans rfl
theorem model_count_2750 : modelRow 2750 = 1 := by decide
theorem modelrow_agrees_2750 : modelRow 2750 = row 2750 := model_count_2750.trans row_count_2750.symm
def row_2751 : RowCertificate 2751 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2751 1) := by change ¬ Nat.Prime 30277507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2329039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2751 2) := by change ¬ Nat.Prime 30272005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6054401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2751 3) := by change Nat.Prime 30266503; exact ProjectEulerSolutions.P58PrimeNodes.p_30266503.prime
    exact decide_eq_true prime
theorem row_count_2751 : row 2751 = 1 := (row_2751.checked).trans rfl
theorem model_count_2751 : modelRow 2751 = 1 := by decide
theorem modelrow_agrees_2751 : modelRow 2751 = row 2751 := model_count_2751.trans row_count_2751.symm
def row_2752 : RowCertificate 2752 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2752 1) := by change ¬ Nat.Prime 30299521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4328503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2752 2) := by change ¬ Nat.Prime 30294017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2330309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2752 3) := by change ¬ Nat.Prime 30288513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10096171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2752 : row 2752 = 0 := (row_2752.checked).trans rfl
theorem model_count_2752 : modelRow 2752 = 0 := by decide
theorem modelrow_agrees_2752 : modelRow 2752 = row 2752 := model_count_2752.trans row_count_2752.symm
def row_2753 : RowCertificate 2753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2753 1) := by change ¬ Nat.Prime 30321543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10107181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2753 2) := by change Nat.Prime 30316037; exact ProjectEulerSolutions.P58PrimeNodes.p_30316037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2753 3) := by change ¬ Nat.Prime 30310531; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 294277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2753 : row 2753 = 1 := (row_2753.checked).trans rfl
theorem model_count_2753 : modelRow 2753 = 1 := by decide
theorem modelrow_agrees_2753 : modelRow 2753 = row 2753 := model_count_2753.trans row_count_2753.symm
def row_2754 : RowCertificate 2754 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2754 1) := by change ¬ Nat.Prime 30343573; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2334121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2754 2) := by change ¬ Nat.Prime 30338065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6067613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2754 3) := by change ¬ Nat.Prime 30332557; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 193201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2754 : row 2754 = 0 := (row_2754.checked).trans rfl
theorem model_count_2754 : modelRow 2754 = 0 := by decide
theorem modelrow_agrees_2754 : modelRow 2754 = row 2754 := model_count_2754.trans row_count_2754.symm
def row_2755 : RowCertificate 2755 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2755 1) := by change ¬ Nat.Prime 30365611; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 706177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2755 2) := by change Nat.Prime 30360101; exact ProjectEulerSolutions.P58PrimeNodes.p_30360101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2755 3) := by change ¬ Nat.Prime 30354591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10118197) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2755 : row 2755 = 1 := (row_2755.checked).trans rfl
theorem model_count_2755 : modelRow 2755 = 1 := by decide
theorem modelrow_agrees_2755 : modelRow 2755 = row 2755 := model_count_2755.trans row_count_2755.symm
def row_2756 : RowCertificate 2756 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2756 1) := by change ¬ Nat.Prime 30387657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10129219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2756 2) := by change ¬ Nat.Prime 30382145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6076429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2756 3) := by change ¬ Nat.Prime 30376633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4339519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2756 : row 2756 = 0 := (row_2756.checked).trans rfl
theorem model_count_2756 : modelRow 2756 = 0 := by decide
theorem modelrow_agrees_2756 : modelRow 2756 = row 2756 := model_count_2756.trans row_count_2756.symm
def row_2757 : RowCertificate 2757 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2757 1) := by change Nat.Prime 30409711; exact ProjectEulerSolutions.P58PrimeNodes.p_30409711.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2757 2) := by change ¬ Nat.Prime 30404197; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 30133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2757 3) := by change ¬ Nat.Prime 30398683; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4342669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2757 : row 2757 = 1 := (row_2757.checked).trans rfl
theorem model_count_2757 : modelRow 2757 = 1 := by decide
theorem modelrow_agrees_2757 : modelRow 2757 = row 2757 := model_count_2757.trans row_count_2757.symm
def row_2758 : RowCertificate 2758 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2758 1) := by change Nat.Prime 30431773; exact ProjectEulerSolutions.P58PrimeNodes.p_30431773.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2758 2) := by change ¬ Nat.Prime 30426257; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 157649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2758 3) := by change ¬ Nat.Prime 30420741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10140247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2758 : row 2758 = 1 := (row_2758.checked).trans rfl
theorem model_count_2758 : modelRow 2758 = 1 := by decide
theorem modelrow_agrees_2758 : modelRow 2758 = row 2758 := model_count_2758.trans row_count_2758.symm
def row_2759 : RowCertificate 2759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2759 1) := by change ¬ Nat.Prime 30453843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10151281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2759 2) := by change ¬ Nat.Prime 30448325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6089665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2759 3) := by change ¬ Nat.Prime 30442807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1602253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2759 : row 2759 = 0 := (row_2759.checked).trans rfl
theorem model_count_2759 : modelRow 2759 = 0 := by decide
theorem modelrow_agrees_2759 : modelRow 2759 = row 2759 := model_count_2759.trans row_count_2759.symm
def row_2760 : RowCertificate 2760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2760 1) := by change ¬ Nat.Prime 30475921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4353703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2760 2) := by change ¬ Nat.Prime 30470401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2343877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2760 3) := by change ¬ Nat.Prime 30464881; exact PrattCertificates.composite_of_product (factor := 1789) (cofactor := 17029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2760 : row 2760 = 0 := (row_2760.checked).trans rfl
theorem model_count_2760 : modelRow 2760 = 0 := by decide
theorem modelrow_agrees_2760 : modelRow 2760 = row 2760 := model_count_2760.trans row_count_2760.symm
def row_2761 : RowCertificate 2761 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2761 1) := by change Nat.Prime 30498007; exact ProjectEulerSolutions.P58PrimeNodes.p_30498007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2761 2) := by change ¬ Nat.Prime 30492485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6098497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2761 3) := by change ¬ Nat.Prime 30486963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10162321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2761 : row 2761 = 1 := (row_2761.checked).trans rfl
theorem model_count_2761 : modelRow 2761 = 1 := by decide
theorem modelrow_agrees_2761 : modelRow 2761 = row 2761 := model_count_2761.trans row_count_2761.symm
def row_2762 : RowCertificate 2762 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2762 1) := by change ¬ Nat.Prime 30520101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10173367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2762 2) := by change Nat.Prime 30514577; exact ProjectEulerSolutions.P58PrimeNodes.p_30514577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2762 3) := by change ¬ Nat.Prime 30509053; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 984163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2762 : row 2762 = 1 := (row_2762.checked).trans rfl
theorem model_count_2762 : modelRow 2762 = 1 := by decide
theorem modelrow_agrees_2762 : modelRow 2762 = row 2762 := model_count_2762.trans row_count_2762.symm
def row_2763 : RowCertificate 2763 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2763 1) := by change Nat.Prime 30542203; exact ProjectEulerSolutions.P58PrimeNodes.p_30542203.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2763 2) := by change ¬ Nat.Prime 30536677; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 744797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2763 3) := by change ¬ Nat.Prime 30531151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4361593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2763 : row 2763 = 1 := (row_2763.checked).trans rfl
theorem model_count_2763 : modelRow 2763 = 1 := by decide
theorem modelrow_agrees_2763 : modelRow 2763 = row 2763 := model_count_2763.trans row_count_2763.symm
def row_2764 : RowCertificate 2764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2764 1) := by change ¬ Nat.Prime 30564313; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2351101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2764 2) := by change ¬ Nat.Prime 30558785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6111757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2764 3) := by change ¬ Nat.Prime 30553257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10184419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2764 : row 2764 = 0 := (row_2764.checked).trans rfl
theorem model_count_2764 : modelRow 2764 = 0 := by decide
theorem modelrow_agrees_2764 : modelRow 2764 = row 2764 := model_count_2764.trans row_count_2764.symm
def row_2765 : RowCertificate 2765 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2765 1) := by change ¬ Nat.Prime 30586431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10195477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2765 2) := by change ¬ Nat.Prime 30580901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2352377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2765 3) := by change Nat.Prime 30575371; exact ProjectEulerSolutions.P58PrimeNodes.p_30575371.prime
    exact decide_eq_true prime
theorem row_count_2765 : row 2765 = 1 := (row_2765.checked).trans rfl
theorem model_count_2765 : modelRow 2765 = 1 := by decide
theorem modelrow_agrees_2765 : modelRow 2765 = row 2765 := model_count_2765.trans row_count_2765.symm
def row_2766 : RowCertificate 2766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2766 1) := by change ¬ Nat.Prime 30608557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4372651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2766 2) := by change ¬ Nat.Prime 30603025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6120605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2766 3) := by change ¬ Nat.Prime 30597493; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 456679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2766 : row 2766 = 0 := (row_2766.checked).trans rfl
theorem model_count_2766 : modelRow 2766 = 0 := by decide
theorem modelrow_agrees_2766 : modelRow 2766 = row 2766 := model_count_2766.trans row_count_2766.symm
def row_2767 : RowCertificate 2767 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2767 1) := by change ¬ Nat.Prime 30630691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4375813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2767 2) := by change ¬ Nat.Prime 30625157; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 50957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2767 3) := by change ¬ Nat.Prime 30619623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10206541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2767 : row 2767 = 0 := (row_2767.checked).trans rfl
theorem model_count_2767 : modelRow 2767 = 0 := by decide
theorem modelrow_agrees_2767 : modelRow 2767 = row 2767 := model_count_2767.trans row_count_2767.symm
def row_2768 : RowCertificate 2768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2768 1) := by change ¬ Nat.Prime 30652833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10217611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2768 2) := by change Nat.Prime 30647297; exact ProjectEulerSolutions.P58PrimeNodes.p_30647297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2768 3) := by change ¬ Nat.Prime 30641761; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 137407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2768 : row 2768 = 1 := (row_2768.checked).trans rfl
theorem model_count_2768 : modelRow 2768 = 1 := by decide
theorem modelrow_agrees_2768 : modelRow 2768 = row 2768 := model_count_2768.trans row_count_2768.symm
def row_2769 : RowCertificate 2769 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2769 1) := by change Nat.Prime 30674983; exact ProjectEulerSolutions.P58PrimeNodes.p_30674983.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2769 2) := by change ¬ Nat.Prime 30669445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6133889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2769 3) := by change ¬ Nat.Prime 30663907; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 502687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2769 : row 2769 = 1 := (row_2769.checked).trans rfl
theorem model_count_2769 : modelRow 2769 = 1 := by decide
theorem modelrow_agrees_2769 : modelRow 2769 = row 2769 := model_count_2769.trans row_count_2769.symm
def row_2770 : RowCertificate 2770 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2770 1) := by change ¬ Nat.Prime 30697141; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1615639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2770 2) := by change ¬ Nat.Prime 30691601; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 503141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2770 3) := by change ¬ Nat.Prime 30686061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10228687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2770 : row 2770 = 0 := (row_2770.checked).trans rfl
theorem model_count_2770 : modelRow 2770 = 0 := by decide
theorem modelrow_agrees_2770 : modelRow 2770 = row 2770 := model_count_2770.trans row_count_2770.symm
def row_2771 : RowCertificate 2771 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2771 1) := by change ¬ Nat.Prime 30719307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10239769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2771 2) := by change ¬ Nat.Prime 30713765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6142753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2771 3) := by change ¬ Nat.Prime 30708223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4386889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2771 : row 2771 = 0 := (row_2771.checked).trans rfl
theorem model_count_2771 : modelRow 2771 = 0 := by decide
theorem modelrow_agrees_2771 : modelRow 2771 = row 2771 := model_count_2771.trans row_count_2771.symm
def row_2772 : RowCertificate 2772 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2772 1) := by change ¬ Nat.Prime 30741481; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 35053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2772 2) := by change ¬ Nat.Prime 30735937; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 830701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2772 3) := by change ¬ Nat.Prime 30730393; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 991303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2772 : row 2772 = 0 := (row_2772.checked).trans rfl
theorem model_count_2772 : modelRow 2772 = 0 := by decide
theorem modelrow_agrees_2772 : modelRow 2772 = row 2772 := model_count_2772.trans row_count_2772.symm
def row_2773 : RowCertificate 2773 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2773 1) := by change ¬ Nat.Prime 30763663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4394809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2773 2) := by change ¬ Nat.Prime 30758117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2366009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2773 3) := by change ¬ Nat.Prime 30752571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10250857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2773 : row 2773 = 0 := (row_2773.checked).trans rfl
theorem model_count_2773 : modelRow 2773 = 0 := by decide
theorem modelrow_agrees_2773 : modelRow 2773 = row 2773 := model_count_2773.trans row_count_2773.symm
def row_2774 : RowCertificate 2774 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2774 1) := by change ¬ Nat.Prime 30785853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10261951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2774 2) := by change ¬ Nat.Prime 30780305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6156061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2774 3) := by change ¬ Nat.Prime 30774757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2367289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2774 : row 2774 = 0 := (row_2774.checked).trans rfl
theorem model_count_2774 : modelRow 2774 = 0 := by decide
theorem modelrow_agrees_2774 : modelRow 2774 = row 2774 := model_count_2774.trans row_count_2774.symm
def row_2775 : RowCertificate 2775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2775 1) := by change ¬ Nat.Prime 30808051; exact PrattCertificates.composite_of_product (factor := 3613) (cofactor := 8527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2775 2) := by change Nat.Prime 30802501; exact ProjectEulerSolutions.P58PrimeNodes.p_30802501.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2775 3) := by change Nat.Prime 30796951; exact ProjectEulerSolutions.P58PrimeNodes.p_30796951.prime
    exact decide_eq_true prime
theorem row_count_2775 : row 2775 = 2 := (row_2775.checked).trans rfl
theorem model_count_2775 : modelRow 2775 = 2 := by decide
theorem modelrow_agrees_2775 : modelRow 2775 = row 2775 := model_count_2775.trans row_count_2775.symm
def row_2776 : RowCertificate 2776 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2776 1) := by change Nat.Prime 30830257; exact ProjectEulerSolutions.P58PrimeNodes.p_30830257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2776 2) := by change ¬ Nat.Prime 30824705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6164941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2776 3) := by change ¬ Nat.Prime 30819153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10273051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2776 : row 2776 = 1 := (row_2776.checked).trans rfl
theorem model_count_2776 : modelRow 2776 = 1 := by decide
theorem modelrow_agrees_2776 : modelRow 2776 = row 2776 := model_count_2776.trans row_count_2776.symm
def row_2777 : RowCertificate 2777 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2777 1) := by change ¬ Nat.Prime 30852471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10284157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2777 2) := by change ¬ Nat.Prime 30846917; exact PrattCertificates.composite_of_product (factor := 2441) (cofactor := 12637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2777 3) := by change ¬ Nat.Prime 30841363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4405909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2777 : row 2777 = 0 := (row_2777.checked).trans rfl
theorem model_count_2777 : modelRow 2777 = 0 := by decide
theorem modelrow_agrees_2777 : modelRow 2777 = row 2777 := model_count_2777.trans row_count_2777.symm
def row_2778 : RowCertificate 2778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2778 1) := by change ¬ Nat.Prime 30874693; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 422941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2778 2) := by change ¬ Nat.Prime 30869137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2374549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2778 3) := by change ¬ Nat.Prime 30863581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4409083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2778 : row 2778 = 0 := (row_2778.checked).trans rfl
theorem model_count_2778 : modelRow 2778 = 0 := by decide
theorem modelrow_agrees_2778 : modelRow 2778 = row 2778 := model_count_2778.trans row_count_2778.symm
def row_2779 : RowCertificate 2779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2779 1) := by change ¬ Nat.Prime 30896923; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 128203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2779 2) := by change ¬ Nat.Prime 30891365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6178273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2779 3) := by change ¬ Nat.Prime 30885807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10295269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2779 : row 2779 = 0 := (row_2779.checked).trans rfl
theorem model_count_2779 : modelRow 2779 = 0 := by decide
theorem modelrow_agrees_2779 : modelRow 2779 = row 2779 := model_count_2779.trans row_count_2779.symm
def row_2780 : RowCertificate 2780 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2780 1) := by change ¬ Nat.Prime 30919161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10306387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2780 2) := by change Nat.Prime 30913601; exact ProjectEulerSolutions.P58PrimeNodes.p_30913601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2780 3) := by change ¬ Nat.Prime 30908041; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1626739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2780 : row 2780 = 1 := (row_2780.checked).trans rfl
theorem model_count_2780 : modelRow 2780 = 1 := by decide
theorem modelrow_agrees_2780 : modelRow 2780 = row 2780 := model_count_2780.trans row_count_2780.symm
def row_2781 : RowCertificate 2781 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2781 1) := by change ¬ Nat.Prime 30941407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4420201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2781 2) := by change ¬ Nat.Prime 30935845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6187169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2781 3) := by change Nat.Prime 30930283; exact ProjectEulerSolutions.P58PrimeNodes.p_30930283.prime
    exact decide_eq_true prime
theorem row_count_2781 : row 2781 = 1 := (row_2781.checked).trans rfl
theorem model_count_2781 : modelRow 2781 = 1 := by decide
theorem modelrow_agrees_2781 : modelRow 2781 = row 2781 := model_count_2781.trans row_count_2781.symm
def row_2782 : RowCertificate 2782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2782 1) := by change ¬ Nat.Prime 30963661; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 507601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2782 2) := by change Nat.Prime 30958097; exact ProjectEulerSolutions.P58PrimeNodes.p_30958097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2782 3) := by change ¬ Nat.Prime 30952533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10317511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2782 : row 2782 = 1 := (row_2782.checked).trans rfl
theorem model_count_2782 : modelRow 2782 = 1 := by decide
theorem modelrow_agrees_2782 : modelRow 2782 = row 2782 := model_count_2782.trans row_count_2782.symm
def row_2783 : RowCertificate 2783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2783 1) := by change ¬ Nat.Prime 30985923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10328641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2783 2) := by change Nat.Prime 30980357; exact ProjectEulerSolutions.P58PrimeNodes.p_30980357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2783 3) := by change ¬ Nat.Prime 30974791; exact PrattCertificates.composite_of_product (factor := 3049) (cofactor := 10159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2783 : row 2783 = 1 := (row_2783.checked).trans rfl
theorem model_count_2783 : modelRow 2783 = 1 := by decide
theorem modelrow_agrees_2783 : modelRow 2783 = row 2783 := model_count_2783.trans row_count_2783.symm
def row_2784 : RowCertificate 2784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2784 1) := by change ¬ Nat.Prime 31008193; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 244159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2784 2) := by change ¬ Nat.Prime 31002625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6200525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2784 3) := by change ¬ Nat.Prime 30997057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4428151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2784 : row 2784 = 0 := (row_2784.checked).trans rfl
theorem model_count_2784 : modelRow 2784 = 0 := by decide
theorem modelrow_agrees_2784 : modelRow 2784 = row 2784 := model_count_2784.trans row_count_2784.symm
def row_2785 : RowCertificate 2785 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2785 1) := by change Nat.Prime 31030471; exact ProjectEulerSolutions.P58PrimeNodes.p_31030471.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2785 2) := by change ¬ Nat.Prime 31024901; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 25577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2785 3) := by change ¬ Nat.Prime 31019331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10339777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2785 : row 2785 = 1 := (row_2785.checked).trans rfl
theorem model_count_2785 : modelRow 2785 = 1 := by decide
theorem modelrow_agrees_2785 : modelRow 2785 = row 2785 := model_count_2785.trans row_count_2785.symm
def row_2786 : RowCertificate 2786 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2786 1) := by change ¬ Nat.Prime 31052757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10350919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2786 2) := by change ¬ Nat.Prime 31047185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6209437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2786 3) := by change Nat.Prime 31041613; exact ProjectEulerSolutions.P58PrimeNodes.p_31041613.prime
    exact decide_eq_true prime
theorem row_count_2786 : row 2786 = 1 := (row_2786.checked).trans rfl
theorem model_count_2786 : modelRow 2786 = 1 := by decide
theorem modelrow_agrees_2786 : modelRow 2786 = row 2786 := model_count_2786.trans row_count_2786.symm
def row_2787 : RowCertificate 2787 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2787 1) := by change ¬ Nat.Prime 31075051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4439293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2787 2) := by change Nat.Prime 31069477; exact ProjectEulerSolutions.P58PrimeNodes.p_31069477.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2787 3) := by change ¬ Nat.Prime 31063903; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2389531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2787 : row 2787 = 1 := (row_2787.checked).trans rfl
theorem model_count_2787 : modelRow 2787 = 1 := by decide
theorem modelrow_agrees_2787 : modelRow 2787 = row 2787 := model_count_2787.trans row_count_2787.symm
def row_2788 : RowCertificate 2788 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2788 1) := by change ¬ Nat.Prime 31097353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4442479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2788 2) := by change ¬ Nat.Prime 31091777; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 43853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2788 3) := by change ¬ Nat.Prime 31086201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10362067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2788 : row 2788 = 0 := (row_2788.checked).trans rfl
theorem model_count_2788 : modelRow 2788 = 0 := by decide
theorem modelrow_agrees_2788 : modelRow 2788 = row 2788 := model_count_2788.trans row_count_2788.symm
def row_2789 : RowCertificate 2789 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2789 1) := by change ¬ Nat.Prime 31119663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10373221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2789 2) := by change ¬ Nat.Prime 31114085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6222817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2789 3) := by change ¬ Nat.Prime 31108507; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 67189) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2789 : row 2789 = 0 := (row_2789.checked).trans rfl
theorem model_count_2789 : modelRow 2789 = 0 := by decide
theorem modelrow_agrees_2789 : modelRow 2789 = row 2789 := model_count_2789.trans row_count_2789.symm
def row_2790 : RowCertificate 2790 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2790 1) := by change ¬ Nat.Prime 31141981; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2395537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2790 2) := by change ¬ Nat.Prime 31136401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1831553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2790 3) := by change ¬ Nat.Prime 31130821; exact PrattCertificates.composite_of_product (factor := 1741) (cofactor := 17881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2790 : row 2790 = 0 := (row_2790.checked).trans rfl
theorem model_count_2790 : modelRow 2790 = 0 := by decide
theorem modelrow_agrees_2790 : modelRow 2790 = row 2790 := model_count_2790.trans row_count_2790.symm
def row_2791 : RowCertificate 2791 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2791 1) := by change ¬ Nat.Prime 31164307; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 50839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2791 2) := by change ¬ Nat.Prime 31158725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6231745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2791 3) := by change ¬ Nat.Prime 31153143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10384381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2791 : row 2791 = 0 := (row_2791.checked).trans rfl
theorem model_count_2791 : modelRow 2791 = 0 := by decide
theorem modelrow_agrees_2791 : modelRow 2791 = row 2791 := model_count_2791.trans row_count_2791.symm
def row_2792 : RowCertificate 2792 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2792 1) := by change ¬ Nat.Prime 31186641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10395547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2792 2) := by change Nat.Prime 31181057; exact ProjectEulerSolutions.P58PrimeNodes.p_31181057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2792 3) := by change ¬ Nat.Prime 31175473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4453639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2792 : row 2792 = 1 := (row_2792.checked).trans rfl
theorem model_count_2792 : modelRow 2792 = 1 := by decide
theorem modelrow_agrees_2792 : modelRow 2792 = row 2792 := model_count_2792.trans row_count_2792.symm
def row_2793 : RowCertificate 2793 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2793 1) := by change ¬ Nat.Prime 31208983; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2400691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2793 2) := by change Nat.Prime 31203397; exact ProjectEulerSolutions.P58PrimeNodes.p_31203397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2793 3) := by change ¬ Nat.Prime 31197811; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1006381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2793 : row 2793 = 1 := (row_2793.checked).trans rfl
theorem model_count_2793 : modelRow 2793 = 1 := by decide
theorem modelrow_agrees_2793 : modelRow 2793 = row 2793 := model_count_2793.trans row_count_2793.symm
def row_2794 : RowCertificate 2794 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2794 1) := by change ¬ Nat.Prime 31231333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4461619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2794 2) := by change ¬ Nat.Prime 31225745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6245149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2794 3) := by change ¬ Nat.Prime 31220157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10406719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2794 : row 2794 = 0 := (row_2794.checked).trans rfl
theorem model_count_2794 : modelRow 2794 = 0 := by decide
theorem modelrow_agrees_2794 : modelRow 2794 = row 2794 := model_count_2794.trans row_count_2794.symm
def row_2795 : RowCertificate 2795 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2795 1) := by change ¬ Nat.Prime 31253691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10417897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2795 2) := by change Nat.Prime 31248101; exact ProjectEulerSolutions.P58PrimeNodes.p_31248101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2795 3) := by change ¬ Nat.Prime 31242511; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 64153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2795 : row 2795 = 1 := (row_2795.checked).trans rfl
theorem model_count_2795 : modelRow 2795 = 1 := by decide
theorem modelrow_agrees_2795 : modelRow 2795 = row 2795 := model_count_2795.trans row_count_2795.symm
def row_2796 : RowCertificate 2796 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2796 1) := by change ¬ Nat.Prime 31276057; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 78781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2796 2) := by change ¬ Nat.Prime 31270465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6254093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2796 3) := by change ¬ Nat.Prime 31264873; exact PrattCertificates.composite_of_product (factor := 739) (cofactor := 42307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2796 : row 2796 = 0 := (row_2796.checked).trans rfl
theorem model_count_2796 : modelRow 2796 = 0 := by decide
theorem modelrow_agrees_2796 : modelRow 2796 = row 2796 := model_count_2796.trans row_count_2796.symm
def row_2797 : RowCertificate 2797 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2797 1) := by change Nat.Prime 31298431; exact ProjectEulerSolutions.P58PrimeNodes.p_31298431.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2797 2) := by change ¬ Nat.Prime 31292837; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 428669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2797 3) := by change ¬ Nat.Prime 31287243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10429081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2797 : row 2797 = 1 := (row_2797.checked).trans rfl
theorem model_count_2797 : modelRow 2797 = 1 := by decide
theorem modelrow_agrees_2797 : modelRow 2797 = row 2797 := model_count_2797.trans row_count_2797.symm
def row_2798 : RowCertificate 2798 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2798 1) := by change ¬ Nat.Prime 31320813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10440271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2798 2) := by change ¬ Nat.Prime 31315217; exact PrattCertificates.composite_of_product (factor := 4877) (cofactor := 6421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2798 3) := by change ¬ Nat.Prime 31309621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4472803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2798 : row 2798 = 0 := (row_2798.checked).trans rfl
theorem model_count_2798 : modelRow 2798 = 0 := by decide
theorem modelrow_agrees_2798 : modelRow 2798 = row 2798 := model_count_2798.trans row_count_2798.symm
def row_2799 : RowCertificate 2799 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2799 1) := by change ¬ Nat.Prime 31343203; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 513823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2799 2) := by change ¬ Nat.Prime 31337605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6267521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2799 3) := by change ¬ Nat.Prime 31332007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4476001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2799 : row 2799 = 0 := (row_2799.checked).trans rfl
theorem model_count_2799 : modelRow 2799 = 0 := by decide
theorem modelrow_agrees_2799 : modelRow 2799 = row 2799 := model_count_2799.trans row_count_2799.symm
def row_2800 : RowCertificate 2800 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2800 1) := by change ¬ Nat.Prime 31365601; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 192427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2800 2) := by change ¬ Nat.Prime 31360001; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 113213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2800 3) := by change ¬ Nat.Prime 31354401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10451467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2800 : row 2800 = 0 := (row_2800.checked).trans rfl
theorem model_count_2800 : modelRow 2800 = 0 := by decide
theorem modelrow_agrees_2800 : modelRow 2800 = row 2800 := model_count_2800.trans row_count_2800.symm
def row_2801 : RowCertificate 2801 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2801 1) := by change ¬ Nat.Prime 31388007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10462669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2801 2) := by change ¬ Nat.Prime 31382405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6276481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2801 3) := by change Nat.Prime 31376803; exact ProjectEulerSolutions.P58PrimeNodes.p_31376803.prime
    exact decide_eq_true prime
theorem row_count_2801 : row 2801 = 1 := (row_2801.checked).trans rfl
theorem model_count_2801 : modelRow 2801 = 1 := by decide
theorem modelrow_agrees_2801 : modelRow 2801 = row 2801 := model_count_2801.trans row_count_2801.symm
def row_2802 : RowCertificate 2802 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2802 1) := by change ¬ Nat.Prime 31410421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4487203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2802 2) := by change ¬ Nat.Prime 31404817; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 323761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2802 3) := by change Nat.Prime 31399213; exact ProjectEulerSolutions.P58PrimeNodes.p_31399213.prime
    exact decide_eq_true prime
theorem row_count_2802 : row 2802 = 1 := (row_2802.checked).trans rfl
theorem model_count_2802 : modelRow 2802 = 1 := by decide
theorem modelrow_agrees_2802 : modelRow 2802 = row 2802 := model_count_2802.trans row_count_2802.symm
def row_2803 : RowCertificate 2803 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2803 1) := by change ¬ Nat.Prime 31432843; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2417911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2803 2) := by change ¬ Nat.Prime 31427237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1848661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2803 3) := by change ¬ Nat.Prime 31421631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10473877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2803 : row 2803 = 0 := (row_2803.checked).trans rfl
theorem model_count_2803 : modelRow 2803 = 0 := by decide
theorem modelrow_agrees_2803 : modelRow 2803 = row 2803 := model_count_2803.trans row_count_2803.symm
def row_2804 : RowCertificate 2804 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2804 1) := by change ¬ Nat.Prime 31455273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10485091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2804 2) := by change ¬ Nat.Prime 31449665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6289933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2804 3) := by change ¬ Nat.Prime 31444057; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 247591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2804 : row 2804 = 0 := (row_2804.checked).trans rfl
theorem model_count_2804 : modelRow 2804 = 0 := by decide
theorem modelrow_agrees_2804 : modelRow 2804 = row 2804 := model_count_2804.trans row_count_2804.symm
def row_2805 : RowCertificate 2805 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2805 1) := by change Nat.Prime 31477711; exact ProjectEulerSolutions.P58PrimeNodes.p_31477711.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2805 2) := by change ¬ Nat.Prime 31472101; exact PrattCertificates.composite_of_product (factor := 977) (cofactor := 32213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2805 3) := by change ¬ Nat.Prime 31466491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4495213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2805 : row 2805 = 1 := (row_2805.checked).trans rfl
theorem model_count_2805 : modelRow 2805 = 1 := by decide
theorem modelrow_agrees_2805 : modelRow 2805 = row 2805 := model_count_2805.trans row_count_2805.symm
def row_2806 : RowCertificate 2806 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2806 1) := by change ¬ Nat.Prime 31500157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2423089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2806 2) := by change ¬ Nat.Prime 31494545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6298909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2806 3) := by change ¬ Nat.Prime 31488933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10496311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2806 : row 2806 = 0 := (row_2806.checked).trans rfl
theorem model_count_2806 : modelRow 2806 = 0 := by decide
theorem modelrow_agrees_2806 : modelRow 2806 = row 2806 := model_count_2806.trans row_count_2806.symm
def row_2807 : RowCertificate 2807 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2807 1) := by change ¬ Nat.Prime 31522611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10507537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2807 2) := by change ¬ Nat.Prime 31516997; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 1853941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2807 3) := by change ¬ Nat.Prime 31511383; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 851659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2807 : row 2807 = 0 := (row_2807.checked).trans rfl
theorem model_count_2807 : modelRow 2807 = 0 := by decide
theorem modelrow_agrees_2807 : modelRow 2807 = row 2807 := model_count_2807.trans row_count_2807.symm
def row_2808 : RowCertificate 2808 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2808 1) := by change ¬ Nat.Prime 31545073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4506439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2808 2) := by change ¬ Nat.Prime 31539457; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 182309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2808 3) := by change ¬ Nat.Prime 31533841; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 111427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2808 : row 2808 = 0 := (row_2808.checked).trans rfl
theorem model_count_2808 : modelRow 2808 = 0 := by decide
theorem modelrow_agrees_2808 : modelRow 2808 = row 2808 := model_count_2808.trans row_count_2808.symm
def row_2809 : RowCertificate 2809 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2809 1) := by change ¬ Nat.Prime 31567543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4509649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2809 2) := by change ¬ Nat.Prime 31561925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6312385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2809 3) := by change ¬ Nat.Prime 31556307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10518769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2809 : row 2809 = 0 := (row_2809.checked).trans rfl
theorem model_count_2809 : modelRow 2809 = 0 := by decide
theorem modelrow_agrees_2809 : modelRow 2809 = row 2809 := model_count_2809.trans row_count_2809.symm
def row_2810 : RowCertificate 2810 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2810 1) := by change ¬ Nat.Prime 31590021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10530007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2810 2) := by change Nat.Prime 31584401; exact ProjectEulerSolutions.P58PrimeNodes.p_31584401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2810 3) := by change ¬ Nat.Prime 31578781; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2429137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2810 : row 2810 = 1 := (row_2810.checked).trans rfl
theorem model_count_2810 : modelRow 2810 = 1 := by decide
theorem modelrow_agrees_2810 : modelRow 2810 = row 2810 := model_count_2810.trans row_count_2810.symm
def row_2811 : RowCertificate 2811 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2811 1) := by change ¬ Nat.Prime 31612507; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 290023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2811 2) := by change ¬ Nat.Prime 31606885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6321377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2811 3) := by change Nat.Prime 31601263; exact ProjectEulerSolutions.P58PrimeNodes.p_31601263.prime
    exact decide_eq_true prime
theorem row_count_2811 : row 2811 = 1 := (row_2811.checked).trans rfl
theorem model_count_2811 : modelRow 2811 = 1 := by decide
theorem modelrow_agrees_2811 : modelRow 2811 = row 2811 := model_count_2811.trans row_count_2811.symm
def row_2812 : RowCertificate 2812 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2812 1) := by change ¬ Nat.Prime 31635001; exact PrattCertificates.composite_of_product (factor := 4657) (cofactor := 6793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2812 2) := by change ¬ Nat.Prime 31629377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2433029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2812 3) := by change ¬ Nat.Prime 31623753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10541251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2812 : row 2812 = 0 := (row_2812.checked).trans rfl
theorem model_count_2812 : modelRow 2812 = 0 := by decide
theorem modelrow_agrees_2812 : modelRow 2812 = row 2812 := model_count_2812.trans row_count_2812.symm
def row_2813 : RowCertificate 2813 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2813 1) := by change ¬ Nat.Prime 31657503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10552501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2813 2) := by change ¬ Nat.Prime 31651877; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 771997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2813 3) := by change ¬ Nat.Prime 31646251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4520893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2813 : row 2813 = 0 := (row_2813.checked).trans rfl
theorem model_count_2813 : modelRow 2813 = 0 := by decide
theorem modelrow_agrees_2813 : modelRow 2813 = row 2813 := model_count_2813.trans row_count_2813.symm
def row_2814 : RowCertificate 2814 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2814 1) := by change ¬ Nat.Prime 31680013; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 63487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2814 2) := by change ¬ Nat.Prime 31674385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6334877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2814 3) := by change Nat.Prime 31668757; exact ProjectEulerSolutions.P58PrimeNodes.p_31668757.prime
    exact decide_eq_true prime
theorem row_count_2814 : row 2814 = 1 := (row_2814.checked).trans rfl
theorem model_count_2814 : modelRow 2814 = 1 := by decide
theorem modelrow_agrees_2814 : modelRow 2814 = row 2814 := model_count_2814.trans row_count_2814.symm
def row_2815 : RowCertificate 2815 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2815 1) := by change ¬ Nat.Prime 31702531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4528933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2815 2) := by change ¬ Nat.Prime 31696901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 856673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2815 3) := by change ¬ Nat.Prime 31691271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10563757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2815 : row 2815 = 0 := (row_2815.checked).trans rfl
theorem model_count_2815 : modelRow 2815 = 0 := by decide
theorem modelrow_agrees_2815 : modelRow 2815 = row 2815 := model_count_2815.trans row_count_2815.symm
def row_2816 : RowCertificate 2816 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2816 1) := by change ¬ Nat.Prime 31725057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 10575019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2816 2) := by change ¬ Nat.Prime 31719425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6343885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2816 3) := by change ¬ Nat.Prime 31713793; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1669147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2816 : row 2816 = 0 := (row_2816.checked).trans rfl
theorem model_count_2816 : modelRow 2816 = 0 := by decide
theorem modelrow_agrees_2816 : modelRow 2816 = row 2816 := model_count_2816.trans row_count_2816.symm
private def atomic_2689 : CountSegment 1 10 13120 2688 2689 1332 1332 := CountSegment.single (first := 2688) (initial := 1332) row_count_2689 (by decide)
private theorem atomic_2689_agreement : RowAgreement 2688 2689 := RowAgreement.single (first := 2688) modelrow_agrees_2689
private def atomic_2690 : CountSegment 1 10 13120 2689 2690 1332 1332 := CountSegment.single (first := 2689) (initial := 1332) row_count_2690 (by decide)
private theorem atomic_2690_agreement : RowAgreement 2689 2690 := RowAgreement.single (first := 2689) modelrow_agrees_2690
private def atomic_2691 : CountSegment 1 10 13120 2690 2691 1332 1333 := CountSegment.single (first := 2690) (initial := 1332) row_count_2691 (by decide)
private theorem atomic_2691_agreement : RowAgreement 2690 2691 := RowAgreement.single (first := 2690) modelrow_agrees_2691
private def atomic_2692 : CountSegment 1 10 13120 2691 2692 1333 1334 := CountSegment.single (first := 2691) (initial := 1333) row_count_2692 (by decide)
private theorem atomic_2692_agreement : RowAgreement 2691 2692 := RowAgreement.single (first := 2691) modelrow_agrees_2692
private def atomic_2693 : CountSegment 1 10 13120 2692 2693 1334 1334 := CountSegment.single (first := 2692) (initial := 1334) row_count_2693 (by decide)
private theorem atomic_2693_agreement : RowAgreement 2692 2693 := RowAgreement.single (first := 2692) modelrow_agrees_2693
private def atomic_2694 : CountSegment 1 10 13120 2693 2694 1334 1334 := CountSegment.single (first := 2693) (initial := 1334) row_count_2694 (by decide)
private theorem atomic_2694_agreement : RowAgreement 2693 2694 := RowAgreement.single (first := 2693) modelrow_agrees_2694
private def atomic_2695 : CountSegment 1 10 13120 2694 2695 1334 1335 := CountSegment.single (first := 2694) (initial := 1334) row_count_2695 (by decide)
private theorem atomic_2695_agreement : RowAgreement 2694 2695 := RowAgreement.single (first := 2694) modelrow_agrees_2695
private def atomic_2696 : CountSegment 1 10 13120 2695 2696 1335 1335 := CountSegment.single (first := 2695) (initial := 1335) row_count_2696 (by decide)
private theorem atomic_2696_agreement : RowAgreement 2695 2696 := RowAgreement.single (first := 2695) modelrow_agrees_2696
private def atomic_2697 : CountSegment 1 10 13120 2696 2697 1335 1335 := CountSegment.single (first := 2696) (initial := 1335) row_count_2697 (by decide)
private theorem atomic_2697_agreement : RowAgreement 2696 2697 := RowAgreement.single (first := 2696) modelrow_agrees_2697
private def atomic_2698 : CountSegment 1 10 13120 2697 2698 1335 1335 := CountSegment.single (first := 2697) (initial := 1335) row_count_2698 (by decide)
private theorem atomic_2698_agreement : RowAgreement 2697 2698 := RowAgreement.single (first := 2697) modelrow_agrees_2698
private def atomic_2699 : CountSegment 1 10 13120 2698 2699 1335 1335 := CountSegment.single (first := 2698) (initial := 1335) row_count_2699 (by decide)
private theorem atomic_2699_agreement : RowAgreement 2698 2699 := RowAgreement.single (first := 2698) modelrow_agrees_2699
private def atomic_2700 : CountSegment 1 10 13120 2699 2700 1335 1336 := CountSegment.single (first := 2699) (initial := 1335) row_count_2700 (by decide)
private theorem atomic_2700_agreement : RowAgreement 2699 2700 := RowAgreement.single (first := 2699) modelrow_agrees_2700
private def atomic_2701 : CountSegment 1 10 13120 2700 2701 1336 1336 := CountSegment.single (first := 2700) (initial := 1336) row_count_2701 (by decide)
private theorem atomic_2701_agreement : RowAgreement 2700 2701 := RowAgreement.single (first := 2700) modelrow_agrees_2701
private def atomic_2702 : CountSegment 1 10 13120 2701 2702 1336 1337 := CountSegment.single (first := 2701) (initial := 1336) row_count_2702 (by decide)
private theorem atomic_2702_agreement : RowAgreement 2701 2702 := RowAgreement.single (first := 2701) modelrow_agrees_2702
private def atomic_2703 : CountSegment 1 10 13120 2702 2703 1337 1338 := CountSegment.single (first := 2702) (initial := 1337) row_count_2703 (by decide)
private theorem atomic_2703_agreement : RowAgreement 2702 2703 := RowAgreement.single (first := 2702) modelrow_agrees_2703
private def atomic_2704 : CountSegment 1 10 13120 2703 2704 1338 1338 := CountSegment.single (first := 2703) (initial := 1338) row_count_2704 (by decide)
private theorem atomic_2704_agreement : RowAgreement 2703 2704 := RowAgreement.single (first := 2703) modelrow_agrees_2704
private def atomic_2705 : CountSegment 1 10 13120 2704 2705 1338 1339 := CountSegment.single (first := 2704) (initial := 1338) row_count_2705 (by decide)
private theorem atomic_2705_agreement : RowAgreement 2704 2705 := RowAgreement.single (first := 2704) modelrow_agrees_2705
private def atomic_2706 : CountSegment 1 10 13120 2705 2706 1339 1339 := CountSegment.single (first := 2705) (initial := 1339) row_count_2706 (by decide)
private theorem atomic_2706_agreement : RowAgreement 2705 2706 := RowAgreement.single (first := 2705) modelrow_agrees_2706
private def atomic_2707 : CountSegment 1 10 13120 2706 2707 1339 1339 := CountSegment.single (first := 2706) (initial := 1339) row_count_2707 (by decide)
private theorem atomic_2707_agreement : RowAgreement 2706 2707 := RowAgreement.single (first := 2706) modelrow_agrees_2707
private def atomic_2708 : CountSegment 1 10 13120 2707 2708 1339 1339 := CountSegment.single (first := 2707) (initial := 1339) row_count_2708 (by decide)
private theorem atomic_2708_agreement : RowAgreement 2707 2708 := RowAgreement.single (first := 2707) modelrow_agrees_2708
private def atomic_2709 : CountSegment 1 10 13120 2708 2709 1339 1339 := CountSegment.single (first := 2708) (initial := 1339) row_count_2709 (by decide)
private theorem atomic_2709_agreement : RowAgreement 2708 2709 := RowAgreement.single (first := 2708) modelrow_agrees_2709
private def atomic_2710 : CountSegment 1 10 13120 2709 2710 1339 1340 := CountSegment.single (first := 2709) (initial := 1339) row_count_2710 (by decide)
private theorem atomic_2710_agreement : RowAgreement 2709 2710 := RowAgreement.single (first := 2709) modelrow_agrees_2710
private def atomic_2711 : CountSegment 1 10 13120 2710 2711 1340 1340 := CountSegment.single (first := 2710) (initial := 1340) row_count_2711 (by decide)
private theorem atomic_2711_agreement : RowAgreement 2710 2711 := RowAgreement.single (first := 2710) modelrow_agrees_2711
private def atomic_2712 : CountSegment 1 10 13120 2711 2712 1340 1341 := CountSegment.single (first := 2711) (initial := 1340) row_count_2712 (by decide)
private theorem atomic_2712_agreement : RowAgreement 2711 2712 := RowAgreement.single (first := 2711) modelrow_agrees_2712
private def atomic_2713 : CountSegment 1 10 13120 2712 2713 1341 1341 := CountSegment.single (first := 2712) (initial := 1341) row_count_2713 (by decide)
private theorem atomic_2713_agreement : RowAgreement 2712 2713 := RowAgreement.single (first := 2712) modelrow_agrees_2713
private def atomic_2714 : CountSegment 1 10 13120 2713 2714 1341 1341 := CountSegment.single (first := 2713) (initial := 1341) row_count_2714 (by decide)
private theorem atomic_2714_agreement : RowAgreement 2713 2714 := RowAgreement.single (first := 2713) modelrow_agrees_2714
private def atomic_2715 : CountSegment 1 10 13120 2714 2715 1341 1342 := CountSegment.single (first := 2714) (initial := 1341) row_count_2715 (by decide)
private theorem atomic_2715_agreement : RowAgreement 2714 2715 := RowAgreement.single (first := 2714) modelrow_agrees_2715
private def atomic_2716 : CountSegment 1 10 13120 2715 2716 1342 1343 := CountSegment.single (first := 2715) (initial := 1342) row_count_2716 (by decide)
private theorem atomic_2716_agreement : RowAgreement 2715 2716 := RowAgreement.single (first := 2715) modelrow_agrees_2716
private def atomic_2717 : CountSegment 1 10 13120 2716 2717 1343 1343 := CountSegment.single (first := 2716) (initial := 1343) row_count_2717 (by decide)
private theorem atomic_2717_agreement : RowAgreement 2716 2717 := RowAgreement.single (first := 2716) modelrow_agrees_2717
private def atomic_2718 : CountSegment 1 10 13120 2717 2718 1343 1343 := CountSegment.single (first := 2717) (initial := 1343) row_count_2718 (by decide)
private theorem atomic_2718_agreement : RowAgreement 2717 2718 := RowAgreement.single (first := 2717) modelrow_agrees_2718
private def atomic_2719 : CountSegment 1 10 13120 2718 2719 1343 1343 := CountSegment.single (first := 2718) (initial := 1343) row_count_2719 (by decide)
private theorem atomic_2719_agreement : RowAgreement 2718 2719 := RowAgreement.single (first := 2718) modelrow_agrees_2719
private def atomic_2720 : CountSegment 1 10 13120 2719 2720 1343 1343 := CountSegment.single (first := 2719) (initial := 1343) row_count_2720 (by decide)
private theorem atomic_2720_agreement : RowAgreement 2719 2720 := RowAgreement.single (first := 2719) modelrow_agrees_2720
private def atomic_2721 : CountSegment 1 10 13120 2720 2721 1343 1343 := CountSegment.single (first := 2720) (initial := 1343) row_count_2721 (by decide)
private theorem atomic_2721_agreement : RowAgreement 2720 2721 := RowAgreement.single (first := 2720) modelrow_agrees_2721
private def atomic_2722 : CountSegment 1 10 13120 2721 2722 1343 1343 := CountSegment.single (first := 2721) (initial := 1343) row_count_2722 (by decide)
private theorem atomic_2722_agreement : RowAgreement 2721 2722 := RowAgreement.single (first := 2721) modelrow_agrees_2722
private def atomic_2723 : CountSegment 1 10 13120 2722 2723 1343 1344 := CountSegment.single (first := 2722) (initial := 1343) row_count_2723 (by decide)
private theorem atomic_2723_agreement : RowAgreement 2722 2723 := RowAgreement.single (first := 2722) modelrow_agrees_2723
private def atomic_2724 : CountSegment 1 10 13120 2723 2724 1344 1345 := CountSegment.single (first := 2723) (initial := 1344) row_count_2724 (by decide)
private theorem atomic_2724_agreement : RowAgreement 2723 2724 := RowAgreement.single (first := 2723) modelrow_agrees_2724
private def atomic_2725 : CountSegment 1 10 13120 2724 2725 1345 1345 := CountSegment.single (first := 2724) (initial := 1345) row_count_2725 (by decide)
private theorem atomic_2725_agreement : RowAgreement 2724 2725 := RowAgreement.single (first := 2724) modelrow_agrees_2725
private def atomic_2726 : CountSegment 1 10 13120 2725 2726 1345 1345 := CountSegment.single (first := 2725) (initial := 1345) row_count_2726 (by decide)
private theorem atomic_2726_agreement : RowAgreement 2725 2726 := RowAgreement.single (first := 2725) modelrow_agrees_2726
private def atomic_2727 : CountSegment 1 10 13120 2726 2727 1345 1346 := CountSegment.single (first := 2726) (initial := 1345) row_count_2727 (by decide)
private theorem atomic_2727_agreement : RowAgreement 2726 2727 := RowAgreement.single (first := 2726) modelrow_agrees_2727
private def atomic_2728 : CountSegment 1 10 13120 2727 2728 1346 1346 := CountSegment.single (first := 2727) (initial := 1346) row_count_2728 (by decide)
private theorem atomic_2728_agreement : RowAgreement 2727 2728 := RowAgreement.single (first := 2727) modelrow_agrees_2728
private def atomic_2729 : CountSegment 1 10 13120 2728 2729 1346 1346 := CountSegment.single (first := 2728) (initial := 1346) row_count_2729 (by decide)
private theorem atomic_2729_agreement : RowAgreement 2728 2729 := RowAgreement.single (first := 2728) modelrow_agrees_2729
private def atomic_2730 : CountSegment 1 10 13120 2729 2730 1346 1347 := CountSegment.single (first := 2729) (initial := 1346) row_count_2730 (by decide)
private theorem atomic_2730_agreement : RowAgreement 2729 2730 := RowAgreement.single (first := 2729) modelrow_agrees_2730
private def atomic_2731 : CountSegment 1 10 13120 2730 2731 1347 1347 := CountSegment.single (first := 2730) (initial := 1347) row_count_2731 (by decide)
private theorem atomic_2731_agreement : RowAgreement 2730 2731 := RowAgreement.single (first := 2730) modelrow_agrees_2731
private def atomic_2732 : CountSegment 1 10 13120 2731 2732 1347 1347 := CountSegment.single (first := 2731) (initial := 1347) row_count_2732 (by decide)
private theorem atomic_2732_agreement : RowAgreement 2731 2732 := RowAgreement.single (first := 2731) modelrow_agrees_2732
private def atomic_2733 : CountSegment 1 10 13120 2732 2733 1347 1348 := CountSegment.single (first := 2732) (initial := 1347) row_count_2733 (by decide)
private theorem atomic_2733_agreement : RowAgreement 2732 2733 := RowAgreement.single (first := 2732) modelrow_agrees_2733
private def atomic_2734 : CountSegment 1 10 13120 2733 2734 1348 1348 := CountSegment.single (first := 2733) (initial := 1348) row_count_2734 (by decide)
private theorem atomic_2734_agreement : RowAgreement 2733 2734 := RowAgreement.single (first := 2733) modelrow_agrees_2734
private def atomic_2735 : CountSegment 1 10 13120 2734 2735 1348 1348 := CountSegment.single (first := 2734) (initial := 1348) row_count_2735 (by decide)
private theorem atomic_2735_agreement : RowAgreement 2734 2735 := RowAgreement.single (first := 2734) modelrow_agrees_2735
private def atomic_2736 : CountSegment 1 10 13120 2735 2736 1348 1348 := CountSegment.single (first := 2735) (initial := 1348) row_count_2736 (by decide)
private theorem atomic_2736_agreement : RowAgreement 2735 2736 := RowAgreement.single (first := 2735) modelrow_agrees_2736
private def atomic_2737 : CountSegment 1 10 13120 2736 2737 1348 1349 := CountSegment.single (first := 2736) (initial := 1348) row_count_2737 (by decide)
private theorem atomic_2737_agreement : RowAgreement 2736 2737 := RowAgreement.single (first := 2736) modelrow_agrees_2737
private def atomic_2738 : CountSegment 1 10 13120 2737 2738 1349 1350 := CountSegment.single (first := 2737) (initial := 1349) row_count_2738 (by decide)
private theorem atomic_2738_agreement : RowAgreement 2737 2738 := RowAgreement.single (first := 2737) modelrow_agrees_2738
private def atomic_2739 : CountSegment 1 10 13120 2738 2739 1350 1350 := CountSegment.single (first := 2738) (initial := 1350) row_count_2739 (by decide)
private theorem atomic_2739_agreement : RowAgreement 2738 2739 := RowAgreement.single (first := 2738) modelrow_agrees_2739
private def atomic_2740 : CountSegment 1 10 13120 2739 2740 1350 1350 := CountSegment.single (first := 2739) (initial := 1350) row_count_2740 (by decide)
private theorem atomic_2740_agreement : RowAgreement 2739 2740 := RowAgreement.single (first := 2739) modelrow_agrees_2740
private def atomic_2741 : CountSegment 1 10 13120 2740 2741 1350 1350 := CountSegment.single (first := 2740) (initial := 1350) row_count_2741 (by decide)
private theorem atomic_2741_agreement : RowAgreement 2740 2741 := RowAgreement.single (first := 2740) modelrow_agrees_2741
private def atomic_2742 : CountSegment 1 10 13120 2741 2742 1350 1351 := CountSegment.single (first := 2741) (initial := 1350) row_count_2742 (by decide)
private theorem atomic_2742_agreement : RowAgreement 2741 2742 := RowAgreement.single (first := 2741) modelrow_agrees_2742
private def atomic_2743 : CountSegment 1 10 13120 2742 2743 1351 1352 := CountSegment.single (first := 2742) (initial := 1351) row_count_2743 (by decide)
private theorem atomic_2743_agreement : RowAgreement 2742 2743 := RowAgreement.single (first := 2742) modelrow_agrees_2743
private def atomic_2744 : CountSegment 1 10 13120 2743 2744 1352 1353 := CountSegment.single (first := 2743) (initial := 1352) row_count_2744 (by decide)
private theorem atomic_2744_agreement : RowAgreement 2743 2744 := RowAgreement.single (first := 2743) modelrow_agrees_2744
private def atomic_2745 : CountSegment 1 10 13120 2744 2745 1353 1354 := CountSegment.single (first := 2744) (initial := 1353) row_count_2745 (by decide)
private theorem atomic_2745_agreement : RowAgreement 2744 2745 := RowAgreement.single (first := 2744) modelrow_agrees_2745
private def atomic_2746 : CountSegment 1 10 13120 2745 2746 1354 1354 := CountSegment.single (first := 2745) (initial := 1354) row_count_2746 (by decide)
private theorem atomic_2746_agreement : RowAgreement 2745 2746 := RowAgreement.single (first := 2745) modelrow_agrees_2746
private def atomic_2747 : CountSegment 1 10 13120 2746 2747 1354 1354 := CountSegment.single (first := 2746) (initial := 1354) row_count_2747 (by decide)
private theorem atomic_2747_agreement : RowAgreement 2746 2747 := RowAgreement.single (first := 2746) modelrow_agrees_2747
private def atomic_2748 : CountSegment 1 10 13120 2747 2748 1354 1354 := CountSegment.single (first := 2747) (initial := 1354) row_count_2748 (by decide)
private theorem atomic_2748_agreement : RowAgreement 2747 2748 := RowAgreement.single (first := 2747) modelrow_agrees_2748
private def atomic_2749 : CountSegment 1 10 13120 2748 2749 1354 1354 := CountSegment.single (first := 2748) (initial := 1354) row_count_2749 (by decide)
private theorem atomic_2749_agreement : RowAgreement 2748 2749 := RowAgreement.single (first := 2748) modelrow_agrees_2749
private def atomic_2750 : CountSegment 1 10 13120 2749 2750 1354 1355 := CountSegment.single (first := 2749) (initial := 1354) row_count_2750 (by decide)
private theorem atomic_2750_agreement : RowAgreement 2749 2750 := RowAgreement.single (first := 2749) modelrow_agrees_2750
private def atomic_2751 : CountSegment 1 10 13120 2750 2751 1355 1356 := CountSegment.single (first := 2750) (initial := 1355) row_count_2751 (by decide)
private theorem atomic_2751_agreement : RowAgreement 2750 2751 := RowAgreement.single (first := 2750) modelrow_agrees_2751
private def atomic_2752 : CountSegment 1 10 13120 2751 2752 1356 1356 := CountSegment.single (first := 2751) (initial := 1356) row_count_2752 (by decide)
private theorem atomic_2752_agreement : RowAgreement 2751 2752 := RowAgreement.single (first := 2751) modelrow_agrees_2752
private def atomic_2753 : CountSegment 1 10 13120 2752 2753 1356 1357 := CountSegment.single (first := 2752) (initial := 1356) row_count_2753 (by decide)
private theorem atomic_2753_agreement : RowAgreement 2752 2753 := RowAgreement.single (first := 2752) modelrow_agrees_2753
private def atomic_2754 : CountSegment 1 10 13120 2753 2754 1357 1357 := CountSegment.single (first := 2753) (initial := 1357) row_count_2754 (by decide)
private theorem atomic_2754_agreement : RowAgreement 2753 2754 := RowAgreement.single (first := 2753) modelrow_agrees_2754
private def atomic_2755 : CountSegment 1 10 13120 2754 2755 1357 1358 := CountSegment.single (first := 2754) (initial := 1357) row_count_2755 (by decide)
private theorem atomic_2755_agreement : RowAgreement 2754 2755 := RowAgreement.single (first := 2754) modelrow_agrees_2755
private def atomic_2756 : CountSegment 1 10 13120 2755 2756 1358 1358 := CountSegment.single (first := 2755) (initial := 1358) row_count_2756 (by decide)
private theorem atomic_2756_agreement : RowAgreement 2755 2756 := RowAgreement.single (first := 2755) modelrow_agrees_2756
private def atomic_2757 : CountSegment 1 10 13120 2756 2757 1358 1359 := CountSegment.single (first := 2756) (initial := 1358) row_count_2757 (by decide)
private theorem atomic_2757_agreement : RowAgreement 2756 2757 := RowAgreement.single (first := 2756) modelrow_agrees_2757
private def atomic_2758 : CountSegment 1 10 13120 2757 2758 1359 1360 := CountSegment.single (first := 2757) (initial := 1359) row_count_2758 (by decide)
private theorem atomic_2758_agreement : RowAgreement 2757 2758 := RowAgreement.single (first := 2757) modelrow_agrees_2758
private def atomic_2759 : CountSegment 1 10 13120 2758 2759 1360 1360 := CountSegment.single (first := 2758) (initial := 1360) row_count_2759 (by decide)
private theorem atomic_2759_agreement : RowAgreement 2758 2759 := RowAgreement.single (first := 2758) modelrow_agrees_2759
private def atomic_2760 : CountSegment 1 10 13120 2759 2760 1360 1360 := CountSegment.single (first := 2759) (initial := 1360) row_count_2760 (by decide)
private theorem atomic_2760_agreement : RowAgreement 2759 2760 := RowAgreement.single (first := 2759) modelrow_agrees_2760
private def atomic_2761 : CountSegment 1 10 13120 2760 2761 1360 1361 := CountSegment.single (first := 2760) (initial := 1360) row_count_2761 (by decide)
private theorem atomic_2761_agreement : RowAgreement 2760 2761 := RowAgreement.single (first := 2760) modelrow_agrees_2761
private def atomic_2762 : CountSegment 1 10 13120 2761 2762 1361 1362 := CountSegment.single (first := 2761) (initial := 1361) row_count_2762 (by decide)
private theorem atomic_2762_agreement : RowAgreement 2761 2762 := RowAgreement.single (first := 2761) modelrow_agrees_2762
private def atomic_2763 : CountSegment 1 10 13120 2762 2763 1362 1363 := CountSegment.single (first := 2762) (initial := 1362) row_count_2763 (by decide)
private theorem atomic_2763_agreement : RowAgreement 2762 2763 := RowAgreement.single (first := 2762) modelrow_agrees_2763
private def atomic_2764 : CountSegment 1 10 13120 2763 2764 1363 1363 := CountSegment.single (first := 2763) (initial := 1363) row_count_2764 (by decide)
private theorem atomic_2764_agreement : RowAgreement 2763 2764 := RowAgreement.single (first := 2763) modelrow_agrees_2764
private def atomic_2765 : CountSegment 1 10 13120 2764 2765 1363 1364 := CountSegment.single (first := 2764) (initial := 1363) row_count_2765 (by decide)
private theorem atomic_2765_agreement : RowAgreement 2764 2765 := RowAgreement.single (first := 2764) modelrow_agrees_2765
private def atomic_2766 : CountSegment 1 10 13120 2765 2766 1364 1364 := CountSegment.single (first := 2765) (initial := 1364) row_count_2766 (by decide)
private theorem atomic_2766_agreement : RowAgreement 2765 2766 := RowAgreement.single (first := 2765) modelrow_agrees_2766
private def atomic_2767 : CountSegment 1 10 13120 2766 2767 1364 1364 := CountSegment.single (first := 2766) (initial := 1364) row_count_2767 (by decide)
private theorem atomic_2767_agreement : RowAgreement 2766 2767 := RowAgreement.single (first := 2766) modelrow_agrees_2767
private def atomic_2768 : CountSegment 1 10 13120 2767 2768 1364 1365 := CountSegment.single (first := 2767) (initial := 1364) row_count_2768 (by decide)
private theorem atomic_2768_agreement : RowAgreement 2767 2768 := RowAgreement.single (first := 2767) modelrow_agrees_2768
private def atomic_2769 : CountSegment 1 10 13120 2768 2769 1365 1366 := CountSegment.single (first := 2768) (initial := 1365) row_count_2769 (by decide)
private theorem atomic_2769_agreement : RowAgreement 2768 2769 := RowAgreement.single (first := 2768) modelrow_agrees_2769
private def atomic_2770 : CountSegment 1 10 13120 2769 2770 1366 1366 := CountSegment.single (first := 2769) (initial := 1366) row_count_2770 (by decide)
private theorem atomic_2770_agreement : RowAgreement 2769 2770 := RowAgreement.single (first := 2769) modelrow_agrees_2770
private def atomic_2771 : CountSegment 1 10 13120 2770 2771 1366 1366 := CountSegment.single (first := 2770) (initial := 1366) row_count_2771 (by decide)
private theorem atomic_2771_agreement : RowAgreement 2770 2771 := RowAgreement.single (first := 2770) modelrow_agrees_2771
private def atomic_2772 : CountSegment 1 10 13120 2771 2772 1366 1366 := CountSegment.single (first := 2771) (initial := 1366) row_count_2772 (by decide)
private theorem atomic_2772_agreement : RowAgreement 2771 2772 := RowAgreement.single (first := 2771) modelrow_agrees_2772
private def atomic_2773 : CountSegment 1 10 13120 2772 2773 1366 1366 := CountSegment.single (first := 2772) (initial := 1366) row_count_2773 (by decide)
private theorem atomic_2773_agreement : RowAgreement 2772 2773 := RowAgreement.single (first := 2772) modelrow_agrees_2773
private def atomic_2774 : CountSegment 1 10 13120 2773 2774 1366 1366 := CountSegment.single (first := 2773) (initial := 1366) row_count_2774 (by decide)
private theorem atomic_2774_agreement : RowAgreement 2773 2774 := RowAgreement.single (first := 2773) modelrow_agrees_2774
private def atomic_2775 : CountSegment 1 10 13120 2774 2775 1366 1368 := CountSegment.single (first := 2774) (initial := 1366) row_count_2775 (by decide)
private theorem atomic_2775_agreement : RowAgreement 2774 2775 := RowAgreement.single (first := 2774) modelrow_agrees_2775
private def atomic_2776 : CountSegment 1 10 13120 2775 2776 1368 1369 := CountSegment.single (first := 2775) (initial := 1368) row_count_2776 (by decide)
private theorem atomic_2776_agreement : RowAgreement 2775 2776 := RowAgreement.single (first := 2775) modelrow_agrees_2776
private def atomic_2777 : CountSegment 1 10 13120 2776 2777 1369 1369 := CountSegment.single (first := 2776) (initial := 1369) row_count_2777 (by decide)
private theorem atomic_2777_agreement : RowAgreement 2776 2777 := RowAgreement.single (first := 2776) modelrow_agrees_2777
private def atomic_2778 : CountSegment 1 10 13120 2777 2778 1369 1369 := CountSegment.single (first := 2777) (initial := 1369) row_count_2778 (by decide)
private theorem atomic_2778_agreement : RowAgreement 2777 2778 := RowAgreement.single (first := 2777) modelrow_agrees_2778
private def atomic_2779 : CountSegment 1 10 13120 2778 2779 1369 1369 := CountSegment.single (first := 2778) (initial := 1369) row_count_2779 (by decide)
private theorem atomic_2779_agreement : RowAgreement 2778 2779 := RowAgreement.single (first := 2778) modelrow_agrees_2779
private def atomic_2780 : CountSegment 1 10 13120 2779 2780 1369 1370 := CountSegment.single (first := 2779) (initial := 1369) row_count_2780 (by decide)
private theorem atomic_2780_agreement : RowAgreement 2779 2780 := RowAgreement.single (first := 2779) modelrow_agrees_2780
private def atomic_2781 : CountSegment 1 10 13120 2780 2781 1370 1371 := CountSegment.single (first := 2780) (initial := 1370) row_count_2781 (by decide)
private theorem atomic_2781_agreement : RowAgreement 2780 2781 := RowAgreement.single (first := 2780) modelrow_agrees_2781
private def atomic_2782 : CountSegment 1 10 13120 2781 2782 1371 1372 := CountSegment.single (first := 2781) (initial := 1371) row_count_2782 (by decide)
private theorem atomic_2782_agreement : RowAgreement 2781 2782 := RowAgreement.single (first := 2781) modelrow_agrees_2782
private def atomic_2783 : CountSegment 1 10 13120 2782 2783 1372 1373 := CountSegment.single (first := 2782) (initial := 1372) row_count_2783 (by decide)
private theorem atomic_2783_agreement : RowAgreement 2782 2783 := RowAgreement.single (first := 2782) modelrow_agrees_2783
private def atomic_2784 : CountSegment 1 10 13120 2783 2784 1373 1373 := CountSegment.single (first := 2783) (initial := 1373) row_count_2784 (by decide)
private theorem atomic_2784_agreement : RowAgreement 2783 2784 := RowAgreement.single (first := 2783) modelrow_agrees_2784
private def atomic_2785 : CountSegment 1 10 13120 2784 2785 1373 1374 := CountSegment.single (first := 2784) (initial := 1373) row_count_2785 (by decide)
private theorem atomic_2785_agreement : RowAgreement 2784 2785 := RowAgreement.single (first := 2784) modelrow_agrees_2785
private def atomic_2786 : CountSegment 1 10 13120 2785 2786 1374 1375 := CountSegment.single (first := 2785) (initial := 1374) row_count_2786 (by decide)
private theorem atomic_2786_agreement : RowAgreement 2785 2786 := RowAgreement.single (first := 2785) modelrow_agrees_2786
private def atomic_2787 : CountSegment 1 10 13120 2786 2787 1375 1376 := CountSegment.single (first := 2786) (initial := 1375) row_count_2787 (by decide)
private theorem atomic_2787_agreement : RowAgreement 2786 2787 := RowAgreement.single (first := 2786) modelrow_agrees_2787
private def atomic_2788 : CountSegment 1 10 13120 2787 2788 1376 1376 := CountSegment.single (first := 2787) (initial := 1376) row_count_2788 (by decide)
private theorem atomic_2788_agreement : RowAgreement 2787 2788 := RowAgreement.single (first := 2787) modelrow_agrees_2788
private def atomic_2789 : CountSegment 1 10 13120 2788 2789 1376 1376 := CountSegment.single (first := 2788) (initial := 1376) row_count_2789 (by decide)
private theorem atomic_2789_agreement : RowAgreement 2788 2789 := RowAgreement.single (first := 2788) modelrow_agrees_2789
private def atomic_2790 : CountSegment 1 10 13120 2789 2790 1376 1376 := CountSegment.single (first := 2789) (initial := 1376) row_count_2790 (by decide)
private theorem atomic_2790_agreement : RowAgreement 2789 2790 := RowAgreement.single (first := 2789) modelrow_agrees_2790
private def atomic_2791 : CountSegment 1 10 13120 2790 2791 1376 1376 := CountSegment.single (first := 2790) (initial := 1376) row_count_2791 (by decide)
private theorem atomic_2791_agreement : RowAgreement 2790 2791 := RowAgreement.single (first := 2790) modelrow_agrees_2791
private def atomic_2792 : CountSegment 1 10 13120 2791 2792 1376 1377 := CountSegment.single (first := 2791) (initial := 1376) row_count_2792 (by decide)
private theorem atomic_2792_agreement : RowAgreement 2791 2792 := RowAgreement.single (first := 2791) modelrow_agrees_2792
private def atomic_2793 : CountSegment 1 10 13120 2792 2793 1377 1378 := CountSegment.single (first := 2792) (initial := 1377) row_count_2793 (by decide)
private theorem atomic_2793_agreement : RowAgreement 2792 2793 := RowAgreement.single (first := 2792) modelrow_agrees_2793
private def atomic_2794 : CountSegment 1 10 13120 2793 2794 1378 1378 := CountSegment.single (first := 2793) (initial := 1378) row_count_2794 (by decide)
private theorem atomic_2794_agreement : RowAgreement 2793 2794 := RowAgreement.single (first := 2793) modelrow_agrees_2794
private def atomic_2795 : CountSegment 1 10 13120 2794 2795 1378 1379 := CountSegment.single (first := 2794) (initial := 1378) row_count_2795 (by decide)
private theorem atomic_2795_agreement : RowAgreement 2794 2795 := RowAgreement.single (first := 2794) modelrow_agrees_2795
private def atomic_2796 : CountSegment 1 10 13120 2795 2796 1379 1379 := CountSegment.single (first := 2795) (initial := 1379) row_count_2796 (by decide)
private theorem atomic_2796_agreement : RowAgreement 2795 2796 := RowAgreement.single (first := 2795) modelrow_agrees_2796
private def atomic_2797 : CountSegment 1 10 13120 2796 2797 1379 1380 := CountSegment.single (first := 2796) (initial := 1379) row_count_2797 (by decide)
private theorem atomic_2797_agreement : RowAgreement 2796 2797 := RowAgreement.single (first := 2796) modelrow_agrees_2797
private def atomic_2798 : CountSegment 1 10 13120 2797 2798 1380 1380 := CountSegment.single (first := 2797) (initial := 1380) row_count_2798 (by decide)
private theorem atomic_2798_agreement : RowAgreement 2797 2798 := RowAgreement.single (first := 2797) modelrow_agrees_2798
private def atomic_2799 : CountSegment 1 10 13120 2798 2799 1380 1380 := CountSegment.single (first := 2798) (initial := 1380) row_count_2799 (by decide)
private theorem atomic_2799_agreement : RowAgreement 2798 2799 := RowAgreement.single (first := 2798) modelrow_agrees_2799
private def atomic_2800 : CountSegment 1 10 13120 2799 2800 1380 1380 := CountSegment.single (first := 2799) (initial := 1380) row_count_2800 (by decide)
private theorem atomic_2800_agreement : RowAgreement 2799 2800 := RowAgreement.single (first := 2799) modelrow_agrees_2800
private def atomic_2801 : CountSegment 1 10 13120 2800 2801 1380 1381 := CountSegment.single (first := 2800) (initial := 1380) row_count_2801 (by decide)
private theorem atomic_2801_agreement : RowAgreement 2800 2801 := RowAgreement.single (first := 2800) modelrow_agrees_2801
private def atomic_2802 : CountSegment 1 10 13120 2801 2802 1381 1382 := CountSegment.single (first := 2801) (initial := 1381) row_count_2802 (by decide)
private theorem atomic_2802_agreement : RowAgreement 2801 2802 := RowAgreement.single (first := 2801) modelrow_agrees_2802
private def atomic_2803 : CountSegment 1 10 13120 2802 2803 1382 1382 := CountSegment.single (first := 2802) (initial := 1382) row_count_2803 (by decide)
private theorem atomic_2803_agreement : RowAgreement 2802 2803 := RowAgreement.single (first := 2802) modelrow_agrees_2803
private def atomic_2804 : CountSegment 1 10 13120 2803 2804 1382 1382 := CountSegment.single (first := 2803) (initial := 1382) row_count_2804 (by decide)
private theorem atomic_2804_agreement : RowAgreement 2803 2804 := RowAgreement.single (first := 2803) modelrow_agrees_2804
private def atomic_2805 : CountSegment 1 10 13120 2804 2805 1382 1383 := CountSegment.single (first := 2804) (initial := 1382) row_count_2805 (by decide)
private theorem atomic_2805_agreement : RowAgreement 2804 2805 := RowAgreement.single (first := 2804) modelrow_agrees_2805
private def atomic_2806 : CountSegment 1 10 13120 2805 2806 1383 1383 := CountSegment.single (first := 2805) (initial := 1383) row_count_2806 (by decide)
private theorem atomic_2806_agreement : RowAgreement 2805 2806 := RowAgreement.single (first := 2805) modelrow_agrees_2806
private def atomic_2807 : CountSegment 1 10 13120 2806 2807 1383 1383 := CountSegment.single (first := 2806) (initial := 1383) row_count_2807 (by decide)
private theorem atomic_2807_agreement : RowAgreement 2806 2807 := RowAgreement.single (first := 2806) modelrow_agrees_2807
private def atomic_2808 : CountSegment 1 10 13120 2807 2808 1383 1383 := CountSegment.single (first := 2807) (initial := 1383) row_count_2808 (by decide)
private theorem atomic_2808_agreement : RowAgreement 2807 2808 := RowAgreement.single (first := 2807) modelrow_agrees_2808
private def atomic_2809 : CountSegment 1 10 13120 2808 2809 1383 1383 := CountSegment.single (first := 2808) (initial := 1383) row_count_2809 (by decide)
private theorem atomic_2809_agreement : RowAgreement 2808 2809 := RowAgreement.single (first := 2808) modelrow_agrees_2809
private def atomic_2810 : CountSegment 1 10 13120 2809 2810 1383 1384 := CountSegment.single (first := 2809) (initial := 1383) row_count_2810 (by decide)
private theorem atomic_2810_agreement : RowAgreement 2809 2810 := RowAgreement.single (first := 2809) modelrow_agrees_2810
private def atomic_2811 : CountSegment 1 10 13120 2810 2811 1384 1385 := CountSegment.single (first := 2810) (initial := 1384) row_count_2811 (by decide)
private theorem atomic_2811_agreement : RowAgreement 2810 2811 := RowAgreement.single (first := 2810) modelrow_agrees_2811
private def atomic_2812 : CountSegment 1 10 13120 2811 2812 1385 1385 := CountSegment.single (first := 2811) (initial := 1385) row_count_2812 (by decide)
private theorem atomic_2812_agreement : RowAgreement 2811 2812 := RowAgreement.single (first := 2811) modelrow_agrees_2812
private def atomic_2813 : CountSegment 1 10 13120 2812 2813 1385 1385 := CountSegment.single (first := 2812) (initial := 1385) row_count_2813 (by decide)
private theorem atomic_2813_agreement : RowAgreement 2812 2813 := RowAgreement.single (first := 2812) modelrow_agrees_2813
private def atomic_2814 : CountSegment 1 10 13120 2813 2814 1385 1386 := CountSegment.single (first := 2813) (initial := 1385) row_count_2814 (by decide)
private theorem atomic_2814_agreement : RowAgreement 2813 2814 := RowAgreement.single (first := 2813) modelrow_agrees_2814
private def atomic_2815 : CountSegment 1 10 13120 2814 2815 1386 1386 := CountSegment.single (first := 2814) (initial := 1386) row_count_2815 (by decide)
private theorem atomic_2815_agreement : RowAgreement 2814 2815 := RowAgreement.single (first := 2814) modelrow_agrees_2815
private def atomic_2816 : CountSegment 1 10 13120 2815 2816 1386 1386 := CountSegment.single (first := 2815) (initial := 1386) row_count_2816 (by decide)
private theorem atomic_2816_agreement : RowAgreement 2815 2816 := RowAgreement.single (first := 2815) modelrow_agrees_2816
private def within_021_000 : CountSegment 1 10 13120 2688 2690 1332 1332 := CountSegment.join atomic_2689 atomic_2690
private theorem within_021_000_agreement : RowAgreement 2688 2690 := RowAgreement.join atomic_2689_agreement atomic_2690_agreement
private def within_021_001 : CountSegment 1 10 13120 2690 2692 1332 1334 := CountSegment.join atomic_2691 atomic_2692
private theorem within_021_001_agreement : RowAgreement 2690 2692 := RowAgreement.join atomic_2691_agreement atomic_2692_agreement
private def within_021_002 : CountSegment 1 10 13120 2688 2692 1332 1334 := CountSegment.join within_021_000 within_021_001
private theorem within_021_002_agreement : RowAgreement 2688 2692 := RowAgreement.join within_021_000_agreement within_021_001_agreement
private def within_021_003 : CountSegment 1 10 13120 2692 2694 1334 1334 := CountSegment.join atomic_2693 atomic_2694
private theorem within_021_003_agreement : RowAgreement 2692 2694 := RowAgreement.join atomic_2693_agreement atomic_2694_agreement
private def within_021_004 : CountSegment 1 10 13120 2694 2696 1334 1335 := CountSegment.join atomic_2695 atomic_2696
private theorem within_021_004_agreement : RowAgreement 2694 2696 := RowAgreement.join atomic_2695_agreement atomic_2696_agreement
private def within_021_005 : CountSegment 1 10 13120 2692 2696 1334 1335 := CountSegment.join within_021_003 within_021_004
private theorem within_021_005_agreement : RowAgreement 2692 2696 := RowAgreement.join within_021_003_agreement within_021_004_agreement
private def within_021_006 : CountSegment 1 10 13120 2688 2696 1332 1335 := CountSegment.join within_021_002 within_021_005
private theorem within_021_006_agreement : RowAgreement 2688 2696 := RowAgreement.join within_021_002_agreement within_021_005_agreement
private def within_021_007 : CountSegment 1 10 13120 2696 2698 1335 1335 := CountSegment.join atomic_2697 atomic_2698
private theorem within_021_007_agreement : RowAgreement 2696 2698 := RowAgreement.join atomic_2697_agreement atomic_2698_agreement
private def within_021_008 : CountSegment 1 10 13120 2698 2700 1335 1336 := CountSegment.join atomic_2699 atomic_2700
private theorem within_021_008_agreement : RowAgreement 2698 2700 := RowAgreement.join atomic_2699_agreement atomic_2700_agreement
private def within_021_009 : CountSegment 1 10 13120 2696 2700 1335 1336 := CountSegment.join within_021_007 within_021_008
private theorem within_021_009_agreement : RowAgreement 2696 2700 := RowAgreement.join within_021_007_agreement within_021_008_agreement
private def within_021_010 : CountSegment 1 10 13120 2700 2702 1336 1337 := CountSegment.join atomic_2701 atomic_2702
private theorem within_021_010_agreement : RowAgreement 2700 2702 := RowAgreement.join atomic_2701_agreement atomic_2702_agreement
private def within_021_011 : CountSegment 1 10 13120 2702 2704 1337 1338 := CountSegment.join atomic_2703 atomic_2704
private theorem within_021_011_agreement : RowAgreement 2702 2704 := RowAgreement.join atomic_2703_agreement atomic_2704_agreement
private def within_021_012 : CountSegment 1 10 13120 2700 2704 1336 1338 := CountSegment.join within_021_010 within_021_011
private theorem within_021_012_agreement : RowAgreement 2700 2704 := RowAgreement.join within_021_010_agreement within_021_011_agreement
private def within_021_013 : CountSegment 1 10 13120 2696 2704 1335 1338 := CountSegment.join within_021_009 within_021_012
private theorem within_021_013_agreement : RowAgreement 2696 2704 := RowAgreement.join within_021_009_agreement within_021_012_agreement
private def within_021_014 : CountSegment 1 10 13120 2688 2704 1332 1338 := CountSegment.join within_021_006 within_021_013
private theorem within_021_014_agreement : RowAgreement 2688 2704 := RowAgreement.join within_021_006_agreement within_021_013_agreement
private def within_021_015 : CountSegment 1 10 13120 2704 2706 1338 1339 := CountSegment.join atomic_2705 atomic_2706
private theorem within_021_015_agreement : RowAgreement 2704 2706 := RowAgreement.join atomic_2705_agreement atomic_2706_agreement
private def within_021_016 : CountSegment 1 10 13120 2706 2708 1339 1339 := CountSegment.join atomic_2707 atomic_2708
private theorem within_021_016_agreement : RowAgreement 2706 2708 := RowAgreement.join atomic_2707_agreement atomic_2708_agreement
private def within_021_017 : CountSegment 1 10 13120 2704 2708 1338 1339 := CountSegment.join within_021_015 within_021_016
private theorem within_021_017_agreement : RowAgreement 2704 2708 := RowAgreement.join within_021_015_agreement within_021_016_agreement
private def within_021_018 : CountSegment 1 10 13120 2708 2710 1339 1340 := CountSegment.join atomic_2709 atomic_2710
private theorem within_021_018_agreement : RowAgreement 2708 2710 := RowAgreement.join atomic_2709_agreement atomic_2710_agreement
private def within_021_019 : CountSegment 1 10 13120 2710 2712 1340 1341 := CountSegment.join atomic_2711 atomic_2712
private theorem within_021_019_agreement : RowAgreement 2710 2712 := RowAgreement.join atomic_2711_agreement atomic_2712_agreement
private def within_021_020 : CountSegment 1 10 13120 2708 2712 1339 1341 := CountSegment.join within_021_018 within_021_019
private theorem within_021_020_agreement : RowAgreement 2708 2712 := RowAgreement.join within_021_018_agreement within_021_019_agreement
private def within_021_021 : CountSegment 1 10 13120 2704 2712 1338 1341 := CountSegment.join within_021_017 within_021_020
private theorem within_021_021_agreement : RowAgreement 2704 2712 := RowAgreement.join within_021_017_agreement within_021_020_agreement
private def within_021_022 : CountSegment 1 10 13120 2712 2714 1341 1341 := CountSegment.join atomic_2713 atomic_2714
private theorem within_021_022_agreement : RowAgreement 2712 2714 := RowAgreement.join atomic_2713_agreement atomic_2714_agreement
private def within_021_023 : CountSegment 1 10 13120 2714 2716 1341 1343 := CountSegment.join atomic_2715 atomic_2716
private theorem within_021_023_agreement : RowAgreement 2714 2716 := RowAgreement.join atomic_2715_agreement atomic_2716_agreement
private def within_021_024 : CountSegment 1 10 13120 2712 2716 1341 1343 := CountSegment.join within_021_022 within_021_023
private theorem within_021_024_agreement : RowAgreement 2712 2716 := RowAgreement.join within_021_022_agreement within_021_023_agreement
private def within_021_025 : CountSegment 1 10 13120 2716 2718 1343 1343 := CountSegment.join atomic_2717 atomic_2718
private theorem within_021_025_agreement : RowAgreement 2716 2718 := RowAgreement.join atomic_2717_agreement atomic_2718_agreement
private def within_021_026 : CountSegment 1 10 13120 2718 2720 1343 1343 := CountSegment.join atomic_2719 atomic_2720
private theorem within_021_026_agreement : RowAgreement 2718 2720 := RowAgreement.join atomic_2719_agreement atomic_2720_agreement
private def within_021_027 : CountSegment 1 10 13120 2716 2720 1343 1343 := CountSegment.join within_021_025 within_021_026
private theorem within_021_027_agreement : RowAgreement 2716 2720 := RowAgreement.join within_021_025_agreement within_021_026_agreement
private def within_021_028 : CountSegment 1 10 13120 2712 2720 1341 1343 := CountSegment.join within_021_024 within_021_027
private theorem within_021_028_agreement : RowAgreement 2712 2720 := RowAgreement.join within_021_024_agreement within_021_027_agreement
private def within_021_029 : CountSegment 1 10 13120 2704 2720 1338 1343 := CountSegment.join within_021_021 within_021_028
private theorem within_021_029_agreement : RowAgreement 2704 2720 := RowAgreement.join within_021_021_agreement within_021_028_agreement
private def within_021_030 : CountSegment 1 10 13120 2688 2720 1332 1343 := CountSegment.join within_021_014 within_021_029
private theorem within_021_030_agreement : RowAgreement 2688 2720 := RowAgreement.join within_021_014_agreement within_021_029_agreement
private def within_021_031 : CountSegment 1 10 13120 2720 2722 1343 1343 := CountSegment.join atomic_2721 atomic_2722
private theorem within_021_031_agreement : RowAgreement 2720 2722 := RowAgreement.join atomic_2721_agreement atomic_2722_agreement
private def within_021_032 : CountSegment 1 10 13120 2722 2724 1343 1345 := CountSegment.join atomic_2723 atomic_2724
private theorem within_021_032_agreement : RowAgreement 2722 2724 := RowAgreement.join atomic_2723_agreement atomic_2724_agreement
private def within_021_033 : CountSegment 1 10 13120 2720 2724 1343 1345 := CountSegment.join within_021_031 within_021_032
private theorem within_021_033_agreement : RowAgreement 2720 2724 := RowAgreement.join within_021_031_agreement within_021_032_agreement
private def within_021_034 : CountSegment 1 10 13120 2724 2726 1345 1345 := CountSegment.join atomic_2725 atomic_2726
private theorem within_021_034_agreement : RowAgreement 2724 2726 := RowAgreement.join atomic_2725_agreement atomic_2726_agreement
private def within_021_035 : CountSegment 1 10 13120 2726 2728 1345 1346 := CountSegment.join atomic_2727 atomic_2728
private theorem within_021_035_agreement : RowAgreement 2726 2728 := RowAgreement.join atomic_2727_agreement atomic_2728_agreement
private def within_021_036 : CountSegment 1 10 13120 2724 2728 1345 1346 := CountSegment.join within_021_034 within_021_035
private theorem within_021_036_agreement : RowAgreement 2724 2728 := RowAgreement.join within_021_034_agreement within_021_035_agreement
private def within_021_037 : CountSegment 1 10 13120 2720 2728 1343 1346 := CountSegment.join within_021_033 within_021_036
private theorem within_021_037_agreement : RowAgreement 2720 2728 := RowAgreement.join within_021_033_agreement within_021_036_agreement
private def within_021_038 : CountSegment 1 10 13120 2728 2730 1346 1347 := CountSegment.join atomic_2729 atomic_2730
private theorem within_021_038_agreement : RowAgreement 2728 2730 := RowAgreement.join atomic_2729_agreement atomic_2730_agreement
private def within_021_039 : CountSegment 1 10 13120 2730 2732 1347 1347 := CountSegment.join atomic_2731 atomic_2732
private theorem within_021_039_agreement : RowAgreement 2730 2732 := RowAgreement.join atomic_2731_agreement atomic_2732_agreement
private def within_021_040 : CountSegment 1 10 13120 2728 2732 1346 1347 := CountSegment.join within_021_038 within_021_039
private theorem within_021_040_agreement : RowAgreement 2728 2732 := RowAgreement.join within_021_038_agreement within_021_039_agreement
private def within_021_041 : CountSegment 1 10 13120 2732 2734 1347 1348 := CountSegment.join atomic_2733 atomic_2734
private theorem within_021_041_agreement : RowAgreement 2732 2734 := RowAgreement.join atomic_2733_agreement atomic_2734_agreement
private def within_021_042 : CountSegment 1 10 13120 2734 2736 1348 1348 := CountSegment.join atomic_2735 atomic_2736
private theorem within_021_042_agreement : RowAgreement 2734 2736 := RowAgreement.join atomic_2735_agreement atomic_2736_agreement
private def within_021_043 : CountSegment 1 10 13120 2732 2736 1347 1348 := CountSegment.join within_021_041 within_021_042
private theorem within_021_043_agreement : RowAgreement 2732 2736 := RowAgreement.join within_021_041_agreement within_021_042_agreement
private def within_021_044 : CountSegment 1 10 13120 2728 2736 1346 1348 := CountSegment.join within_021_040 within_021_043
private theorem within_021_044_agreement : RowAgreement 2728 2736 := RowAgreement.join within_021_040_agreement within_021_043_agreement
private def within_021_045 : CountSegment 1 10 13120 2720 2736 1343 1348 := CountSegment.join within_021_037 within_021_044
private theorem within_021_045_agreement : RowAgreement 2720 2736 := RowAgreement.join within_021_037_agreement within_021_044_agreement
private def within_021_046 : CountSegment 1 10 13120 2736 2738 1348 1350 := CountSegment.join atomic_2737 atomic_2738
private theorem within_021_046_agreement : RowAgreement 2736 2738 := RowAgreement.join atomic_2737_agreement atomic_2738_agreement
private def within_021_047 : CountSegment 1 10 13120 2738 2740 1350 1350 := CountSegment.join atomic_2739 atomic_2740
private theorem within_021_047_agreement : RowAgreement 2738 2740 := RowAgreement.join atomic_2739_agreement atomic_2740_agreement
private def within_021_048 : CountSegment 1 10 13120 2736 2740 1348 1350 := CountSegment.join within_021_046 within_021_047
private theorem within_021_048_agreement : RowAgreement 2736 2740 := RowAgreement.join within_021_046_agreement within_021_047_agreement
private def within_021_049 : CountSegment 1 10 13120 2740 2742 1350 1351 := CountSegment.join atomic_2741 atomic_2742
private theorem within_021_049_agreement : RowAgreement 2740 2742 := RowAgreement.join atomic_2741_agreement atomic_2742_agreement
private def within_021_050 : CountSegment 1 10 13120 2742 2744 1351 1353 := CountSegment.join atomic_2743 atomic_2744
private theorem within_021_050_agreement : RowAgreement 2742 2744 := RowAgreement.join atomic_2743_agreement atomic_2744_agreement
private def within_021_051 : CountSegment 1 10 13120 2740 2744 1350 1353 := CountSegment.join within_021_049 within_021_050
private theorem within_021_051_agreement : RowAgreement 2740 2744 := RowAgreement.join within_021_049_agreement within_021_050_agreement
private def within_021_052 : CountSegment 1 10 13120 2736 2744 1348 1353 := CountSegment.join within_021_048 within_021_051
private theorem within_021_052_agreement : RowAgreement 2736 2744 := RowAgreement.join within_021_048_agreement within_021_051_agreement
private def within_021_053 : CountSegment 1 10 13120 2744 2746 1353 1354 := CountSegment.join atomic_2745 atomic_2746
private theorem within_021_053_agreement : RowAgreement 2744 2746 := RowAgreement.join atomic_2745_agreement atomic_2746_agreement
private def within_021_054 : CountSegment 1 10 13120 2746 2748 1354 1354 := CountSegment.join atomic_2747 atomic_2748
private theorem within_021_054_agreement : RowAgreement 2746 2748 := RowAgreement.join atomic_2747_agreement atomic_2748_agreement
private def within_021_055 : CountSegment 1 10 13120 2744 2748 1353 1354 := CountSegment.join within_021_053 within_021_054
private theorem within_021_055_agreement : RowAgreement 2744 2748 := RowAgreement.join within_021_053_agreement within_021_054_agreement
private def within_021_056 : CountSegment 1 10 13120 2748 2750 1354 1355 := CountSegment.join atomic_2749 atomic_2750
private theorem within_021_056_agreement : RowAgreement 2748 2750 := RowAgreement.join atomic_2749_agreement atomic_2750_agreement
private def within_021_057 : CountSegment 1 10 13120 2750 2752 1355 1356 := CountSegment.join atomic_2751 atomic_2752
private theorem within_021_057_agreement : RowAgreement 2750 2752 := RowAgreement.join atomic_2751_agreement atomic_2752_agreement
private def within_021_058 : CountSegment 1 10 13120 2748 2752 1354 1356 := CountSegment.join within_021_056 within_021_057
private theorem within_021_058_agreement : RowAgreement 2748 2752 := RowAgreement.join within_021_056_agreement within_021_057_agreement
private def within_021_059 : CountSegment 1 10 13120 2744 2752 1353 1356 := CountSegment.join within_021_055 within_021_058
private theorem within_021_059_agreement : RowAgreement 2744 2752 := RowAgreement.join within_021_055_agreement within_021_058_agreement
private def within_021_060 : CountSegment 1 10 13120 2736 2752 1348 1356 := CountSegment.join within_021_052 within_021_059
private theorem within_021_060_agreement : RowAgreement 2736 2752 := RowAgreement.join within_021_052_agreement within_021_059_agreement
private def within_021_061 : CountSegment 1 10 13120 2720 2752 1343 1356 := CountSegment.join within_021_045 within_021_060
private theorem within_021_061_agreement : RowAgreement 2720 2752 := RowAgreement.join within_021_045_agreement within_021_060_agreement
private def within_021_062 : CountSegment 1 10 13120 2688 2752 1332 1356 := CountSegment.join within_021_030 within_021_061
private theorem within_021_062_agreement : RowAgreement 2688 2752 := RowAgreement.join within_021_030_agreement within_021_061_agreement
private def within_021_063 : CountSegment 1 10 13120 2752 2754 1356 1357 := CountSegment.join atomic_2753 atomic_2754
private theorem within_021_063_agreement : RowAgreement 2752 2754 := RowAgreement.join atomic_2753_agreement atomic_2754_agreement
private def within_021_064 : CountSegment 1 10 13120 2754 2756 1357 1358 := CountSegment.join atomic_2755 atomic_2756
private theorem within_021_064_agreement : RowAgreement 2754 2756 := RowAgreement.join atomic_2755_agreement atomic_2756_agreement
private def within_021_065 : CountSegment 1 10 13120 2752 2756 1356 1358 := CountSegment.join within_021_063 within_021_064
private theorem within_021_065_agreement : RowAgreement 2752 2756 := RowAgreement.join within_021_063_agreement within_021_064_agreement
private def within_021_066 : CountSegment 1 10 13120 2756 2758 1358 1360 := CountSegment.join atomic_2757 atomic_2758
private theorem within_021_066_agreement : RowAgreement 2756 2758 := RowAgreement.join atomic_2757_agreement atomic_2758_agreement
private def within_021_067 : CountSegment 1 10 13120 2758 2760 1360 1360 := CountSegment.join atomic_2759 atomic_2760
private theorem within_021_067_agreement : RowAgreement 2758 2760 := RowAgreement.join atomic_2759_agreement atomic_2760_agreement
private def within_021_068 : CountSegment 1 10 13120 2756 2760 1358 1360 := CountSegment.join within_021_066 within_021_067
private theorem within_021_068_agreement : RowAgreement 2756 2760 := RowAgreement.join within_021_066_agreement within_021_067_agreement
private def within_021_069 : CountSegment 1 10 13120 2752 2760 1356 1360 := CountSegment.join within_021_065 within_021_068
private theorem within_021_069_agreement : RowAgreement 2752 2760 := RowAgreement.join within_021_065_agreement within_021_068_agreement
private def within_021_070 : CountSegment 1 10 13120 2760 2762 1360 1362 := CountSegment.join atomic_2761 atomic_2762
private theorem within_021_070_agreement : RowAgreement 2760 2762 := RowAgreement.join atomic_2761_agreement atomic_2762_agreement
private def within_021_071 : CountSegment 1 10 13120 2762 2764 1362 1363 := CountSegment.join atomic_2763 atomic_2764
private theorem within_021_071_agreement : RowAgreement 2762 2764 := RowAgreement.join atomic_2763_agreement atomic_2764_agreement
private def within_021_072 : CountSegment 1 10 13120 2760 2764 1360 1363 := CountSegment.join within_021_070 within_021_071
private theorem within_021_072_agreement : RowAgreement 2760 2764 := RowAgreement.join within_021_070_agreement within_021_071_agreement
private def within_021_073 : CountSegment 1 10 13120 2764 2766 1363 1364 := CountSegment.join atomic_2765 atomic_2766
private theorem within_021_073_agreement : RowAgreement 2764 2766 := RowAgreement.join atomic_2765_agreement atomic_2766_agreement
private def within_021_074 : CountSegment 1 10 13120 2766 2768 1364 1365 := CountSegment.join atomic_2767 atomic_2768
private theorem within_021_074_agreement : RowAgreement 2766 2768 := RowAgreement.join atomic_2767_agreement atomic_2768_agreement
private def within_021_075 : CountSegment 1 10 13120 2764 2768 1363 1365 := CountSegment.join within_021_073 within_021_074
private theorem within_021_075_agreement : RowAgreement 2764 2768 := RowAgreement.join within_021_073_agreement within_021_074_agreement
private def within_021_076 : CountSegment 1 10 13120 2760 2768 1360 1365 := CountSegment.join within_021_072 within_021_075
private theorem within_021_076_agreement : RowAgreement 2760 2768 := RowAgreement.join within_021_072_agreement within_021_075_agreement
private def within_021_077 : CountSegment 1 10 13120 2752 2768 1356 1365 := CountSegment.join within_021_069 within_021_076
private theorem within_021_077_agreement : RowAgreement 2752 2768 := RowAgreement.join within_021_069_agreement within_021_076_agreement
private def within_021_078 : CountSegment 1 10 13120 2768 2770 1365 1366 := CountSegment.join atomic_2769 atomic_2770
private theorem within_021_078_agreement : RowAgreement 2768 2770 := RowAgreement.join atomic_2769_agreement atomic_2770_agreement
private def within_021_079 : CountSegment 1 10 13120 2770 2772 1366 1366 := CountSegment.join atomic_2771 atomic_2772
private theorem within_021_079_agreement : RowAgreement 2770 2772 := RowAgreement.join atomic_2771_agreement atomic_2772_agreement
private def within_021_080 : CountSegment 1 10 13120 2768 2772 1365 1366 := CountSegment.join within_021_078 within_021_079
private theorem within_021_080_agreement : RowAgreement 2768 2772 := RowAgreement.join within_021_078_agreement within_021_079_agreement
private def within_021_081 : CountSegment 1 10 13120 2772 2774 1366 1366 := CountSegment.join atomic_2773 atomic_2774
private theorem within_021_081_agreement : RowAgreement 2772 2774 := RowAgreement.join atomic_2773_agreement atomic_2774_agreement
private def within_021_082 : CountSegment 1 10 13120 2774 2776 1366 1369 := CountSegment.join atomic_2775 atomic_2776
private theorem within_021_082_agreement : RowAgreement 2774 2776 := RowAgreement.join atomic_2775_agreement atomic_2776_agreement
private def within_021_083 : CountSegment 1 10 13120 2772 2776 1366 1369 := CountSegment.join within_021_081 within_021_082
private theorem within_021_083_agreement : RowAgreement 2772 2776 := RowAgreement.join within_021_081_agreement within_021_082_agreement
private def within_021_084 : CountSegment 1 10 13120 2768 2776 1365 1369 := CountSegment.join within_021_080 within_021_083
private theorem within_021_084_agreement : RowAgreement 2768 2776 := RowAgreement.join within_021_080_agreement within_021_083_agreement
private def within_021_085 : CountSegment 1 10 13120 2776 2778 1369 1369 := CountSegment.join atomic_2777 atomic_2778
private theorem within_021_085_agreement : RowAgreement 2776 2778 := RowAgreement.join atomic_2777_agreement atomic_2778_agreement
private def within_021_086 : CountSegment 1 10 13120 2778 2780 1369 1370 := CountSegment.join atomic_2779 atomic_2780
private theorem within_021_086_agreement : RowAgreement 2778 2780 := RowAgreement.join atomic_2779_agreement atomic_2780_agreement
private def within_021_087 : CountSegment 1 10 13120 2776 2780 1369 1370 := CountSegment.join within_021_085 within_021_086
private theorem within_021_087_agreement : RowAgreement 2776 2780 := RowAgreement.join within_021_085_agreement within_021_086_agreement
private def within_021_088 : CountSegment 1 10 13120 2780 2782 1370 1372 := CountSegment.join atomic_2781 atomic_2782
private theorem within_021_088_agreement : RowAgreement 2780 2782 := RowAgreement.join atomic_2781_agreement atomic_2782_agreement
private def within_021_089 : CountSegment 1 10 13120 2782 2784 1372 1373 := CountSegment.join atomic_2783 atomic_2784
private theorem within_021_089_agreement : RowAgreement 2782 2784 := RowAgreement.join atomic_2783_agreement atomic_2784_agreement
private def within_021_090 : CountSegment 1 10 13120 2780 2784 1370 1373 := CountSegment.join within_021_088 within_021_089
private theorem within_021_090_agreement : RowAgreement 2780 2784 := RowAgreement.join within_021_088_agreement within_021_089_agreement
private def within_021_091 : CountSegment 1 10 13120 2776 2784 1369 1373 := CountSegment.join within_021_087 within_021_090
private theorem within_021_091_agreement : RowAgreement 2776 2784 := RowAgreement.join within_021_087_agreement within_021_090_agreement
private def within_021_092 : CountSegment 1 10 13120 2768 2784 1365 1373 := CountSegment.join within_021_084 within_021_091
private theorem within_021_092_agreement : RowAgreement 2768 2784 := RowAgreement.join within_021_084_agreement within_021_091_agreement
private def within_021_093 : CountSegment 1 10 13120 2752 2784 1356 1373 := CountSegment.join within_021_077 within_021_092
private theorem within_021_093_agreement : RowAgreement 2752 2784 := RowAgreement.join within_021_077_agreement within_021_092_agreement
private def within_021_094 : CountSegment 1 10 13120 2784 2786 1373 1375 := CountSegment.join atomic_2785 atomic_2786
private theorem within_021_094_agreement : RowAgreement 2784 2786 := RowAgreement.join atomic_2785_agreement atomic_2786_agreement
private def within_021_095 : CountSegment 1 10 13120 2786 2788 1375 1376 := CountSegment.join atomic_2787 atomic_2788
private theorem within_021_095_agreement : RowAgreement 2786 2788 := RowAgreement.join atomic_2787_agreement atomic_2788_agreement
private def within_021_096 : CountSegment 1 10 13120 2784 2788 1373 1376 := CountSegment.join within_021_094 within_021_095
private theorem within_021_096_agreement : RowAgreement 2784 2788 := RowAgreement.join within_021_094_agreement within_021_095_agreement
private def within_021_097 : CountSegment 1 10 13120 2788 2790 1376 1376 := CountSegment.join atomic_2789 atomic_2790
private theorem within_021_097_agreement : RowAgreement 2788 2790 := RowAgreement.join atomic_2789_agreement atomic_2790_agreement
private def within_021_098 : CountSegment 1 10 13120 2790 2792 1376 1377 := CountSegment.join atomic_2791 atomic_2792
private theorem within_021_098_agreement : RowAgreement 2790 2792 := RowAgreement.join atomic_2791_agreement atomic_2792_agreement
private def within_021_099 : CountSegment 1 10 13120 2788 2792 1376 1377 := CountSegment.join within_021_097 within_021_098
private theorem within_021_099_agreement : RowAgreement 2788 2792 := RowAgreement.join within_021_097_agreement within_021_098_agreement
private def within_021_100 : CountSegment 1 10 13120 2784 2792 1373 1377 := CountSegment.join within_021_096 within_021_099
private theorem within_021_100_agreement : RowAgreement 2784 2792 := RowAgreement.join within_021_096_agreement within_021_099_agreement
private def within_021_101 : CountSegment 1 10 13120 2792 2794 1377 1378 := CountSegment.join atomic_2793 atomic_2794
private theorem within_021_101_agreement : RowAgreement 2792 2794 := RowAgreement.join atomic_2793_agreement atomic_2794_agreement
private def within_021_102 : CountSegment 1 10 13120 2794 2796 1378 1379 := CountSegment.join atomic_2795 atomic_2796
private theorem within_021_102_agreement : RowAgreement 2794 2796 := RowAgreement.join atomic_2795_agreement atomic_2796_agreement
private def within_021_103 : CountSegment 1 10 13120 2792 2796 1377 1379 := CountSegment.join within_021_101 within_021_102
private theorem within_021_103_agreement : RowAgreement 2792 2796 := RowAgreement.join within_021_101_agreement within_021_102_agreement
private def within_021_104 : CountSegment 1 10 13120 2796 2798 1379 1380 := CountSegment.join atomic_2797 atomic_2798
private theorem within_021_104_agreement : RowAgreement 2796 2798 := RowAgreement.join atomic_2797_agreement atomic_2798_agreement
private def within_021_105 : CountSegment 1 10 13120 2798 2800 1380 1380 := CountSegment.join atomic_2799 atomic_2800
private theorem within_021_105_agreement : RowAgreement 2798 2800 := RowAgreement.join atomic_2799_agreement atomic_2800_agreement
private def within_021_106 : CountSegment 1 10 13120 2796 2800 1379 1380 := CountSegment.join within_021_104 within_021_105
private theorem within_021_106_agreement : RowAgreement 2796 2800 := RowAgreement.join within_021_104_agreement within_021_105_agreement
private def within_021_107 : CountSegment 1 10 13120 2792 2800 1377 1380 := CountSegment.join within_021_103 within_021_106
private theorem within_021_107_agreement : RowAgreement 2792 2800 := RowAgreement.join within_021_103_agreement within_021_106_agreement
private def within_021_108 : CountSegment 1 10 13120 2784 2800 1373 1380 := CountSegment.join within_021_100 within_021_107
private theorem within_021_108_agreement : RowAgreement 2784 2800 := RowAgreement.join within_021_100_agreement within_021_107_agreement
private def within_021_109 : CountSegment 1 10 13120 2800 2802 1380 1382 := CountSegment.join atomic_2801 atomic_2802
private theorem within_021_109_agreement : RowAgreement 2800 2802 := RowAgreement.join atomic_2801_agreement atomic_2802_agreement
private def within_021_110 : CountSegment 1 10 13120 2802 2804 1382 1382 := CountSegment.join atomic_2803 atomic_2804
private theorem within_021_110_agreement : RowAgreement 2802 2804 := RowAgreement.join atomic_2803_agreement atomic_2804_agreement
private def within_021_111 : CountSegment 1 10 13120 2800 2804 1380 1382 := CountSegment.join within_021_109 within_021_110
private theorem within_021_111_agreement : RowAgreement 2800 2804 := RowAgreement.join within_021_109_agreement within_021_110_agreement
private def within_021_112 : CountSegment 1 10 13120 2804 2806 1382 1383 := CountSegment.join atomic_2805 atomic_2806
private theorem within_021_112_agreement : RowAgreement 2804 2806 := RowAgreement.join atomic_2805_agreement atomic_2806_agreement
private def within_021_113 : CountSegment 1 10 13120 2806 2808 1383 1383 := CountSegment.join atomic_2807 atomic_2808
private theorem within_021_113_agreement : RowAgreement 2806 2808 := RowAgreement.join atomic_2807_agreement atomic_2808_agreement
private def within_021_114 : CountSegment 1 10 13120 2804 2808 1382 1383 := CountSegment.join within_021_112 within_021_113
private theorem within_021_114_agreement : RowAgreement 2804 2808 := RowAgreement.join within_021_112_agreement within_021_113_agreement
private def within_021_115 : CountSegment 1 10 13120 2800 2808 1380 1383 := CountSegment.join within_021_111 within_021_114
private theorem within_021_115_agreement : RowAgreement 2800 2808 := RowAgreement.join within_021_111_agreement within_021_114_agreement
private def within_021_116 : CountSegment 1 10 13120 2808 2810 1383 1384 := CountSegment.join atomic_2809 atomic_2810
private theorem within_021_116_agreement : RowAgreement 2808 2810 := RowAgreement.join atomic_2809_agreement atomic_2810_agreement
private def within_021_117 : CountSegment 1 10 13120 2810 2812 1384 1385 := CountSegment.join atomic_2811 atomic_2812
private theorem within_021_117_agreement : RowAgreement 2810 2812 := RowAgreement.join atomic_2811_agreement atomic_2812_agreement
private def within_021_118 : CountSegment 1 10 13120 2808 2812 1383 1385 := CountSegment.join within_021_116 within_021_117
private theorem within_021_118_agreement : RowAgreement 2808 2812 := RowAgreement.join within_021_116_agreement within_021_117_agreement
private def within_021_119 : CountSegment 1 10 13120 2812 2814 1385 1386 := CountSegment.join atomic_2813 atomic_2814
private theorem within_021_119_agreement : RowAgreement 2812 2814 := RowAgreement.join atomic_2813_agreement atomic_2814_agreement
private def within_021_120 : CountSegment 1 10 13120 2814 2816 1386 1386 := CountSegment.join atomic_2815 atomic_2816
private theorem within_021_120_agreement : RowAgreement 2814 2816 := RowAgreement.join atomic_2815_agreement atomic_2816_agreement
private def within_021_121 : CountSegment 1 10 13120 2812 2816 1385 1386 := CountSegment.join within_021_119 within_021_120
private theorem within_021_121_agreement : RowAgreement 2812 2816 := RowAgreement.join within_021_119_agreement within_021_120_agreement
private def within_021_122 : CountSegment 1 10 13120 2808 2816 1383 1386 := CountSegment.join within_021_118 within_021_121
private theorem within_021_122_agreement : RowAgreement 2808 2816 := RowAgreement.join within_021_118_agreement within_021_121_agreement
private def within_021_123 : CountSegment 1 10 13120 2800 2816 1380 1386 := CountSegment.join within_021_115 within_021_122
private theorem within_021_123_agreement : RowAgreement 2800 2816 := RowAgreement.join within_021_115_agreement within_021_122_agreement
private def within_021_124 : CountSegment 1 10 13120 2784 2816 1373 1386 := CountSegment.join within_021_108 within_021_123
private theorem within_021_124_agreement : RowAgreement 2784 2816 := RowAgreement.join within_021_108_agreement within_021_123_agreement
private def within_021_125 : CountSegment 1 10 13120 2752 2816 1356 1386 := CountSegment.join within_021_093 within_021_124
private theorem within_021_125_agreement : RowAgreement 2752 2816 := RowAgreement.join within_021_093_agreement within_021_124_agreement
private def within_021_126 : CountSegment 1 10 13120 2688 2816 1332 1386 := CountSegment.join within_021_062 within_021_125
private theorem within_021_126_agreement : RowAgreement 2688 2816 := RowAgreement.join within_021_062_agreement within_021_125_agreement
def segment_021 : CountSegment 1 10 13120 2688 2816 1332 1386 := within_021_126
theorem agreement_021 : RowAgreement 2688 2816 := within_021_126_agreement
end ProjectEulerSolutions.P58Numerical
