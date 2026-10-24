import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9857 : RowCertificate 9857 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9857 1) := by change ¬ Nat.Prime 388661511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129553837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9857 2) := by change ¬ Nat.Prime 388641797; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6371177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9857 3) := by change Nat.Prime 388622083; exact ProjectEulerSolutions.P58PrimeNodes.p_388622083.prime
    exact decide_eq_true prime
theorem row_count_9857 : row 9857 = 1 := (row_9857.checked).trans rfl
theorem model_count_9857 : modelRow 9857 = 1 := by decide
theorem modelrow_agrees_9857 : modelRow 9857 = row 9857 := model_count_9857.trans row_count_9857.symm
def row_9858 : RowCertificate 9858 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9858 1) := by change ¬ Nat.Prime 388740373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55534339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9858 2) := by change ¬ Nat.Prime 388720657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29901589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9858 3) := by change ¬ Nat.Prime 388700941; exact PrattCertificates.composite_of_product (factor := 10729) (cofactor := 36229) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9858 : row 9858 = 0 := (row_9858.checked).trans rfl
theorem model_count_9858 : modelRow 9858 = 0 := by decide
theorem modelrow_agrees_9858 : modelRow 9858 = row 9858 := model_count_9858.trans row_count_9858.symm
def row_9859 : RowCertificate 9859 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9859 1) := by change ¬ Nat.Prime 388819243; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5326291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9859 2) := by change ¬ Nat.Prime 388799525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77759905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9859 3) := by change ¬ Nat.Prime 388779807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129593269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9859 : row 9859 = 0 := (row_9859.checked).trans rfl
theorem model_count_9859 : modelRow 9859 = 0 := by decide
theorem modelrow_agrees_9859 : modelRow 9859 = row 9859 := model_count_9859.trans row_count_9859.symm
def row_9860 : RowCertificate 9860 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9860 1) := by change ¬ Nat.Prime 388898121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129632707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9860 2) := by change ¬ Nat.Prime 388878401; exact PrattCertificates.composite_of_product (factor := 12637) (cofactor := 30773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9860 3) := by change ¬ Nat.Prime 388858681; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3775327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9860 : row 9860 = 0 := (row_9860.checked).trans rfl
theorem model_count_9860 : modelRow 9860 = 0 := by decide
theorem modelrow_agrees_9860 : modelRow 9860 = row 9860 := model_count_9860.trans row_count_9860.symm
def row_9861 : RowCertificate 9861 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9861 1) := by change Nat.Prime 388977007; exact ProjectEulerSolutions.P58PrimeNodes.p_388977007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9861 2) := by change ¬ Nat.Prime 388957285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77791457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9861 3) := by change ¬ Nat.Prime 388937563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55562509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9861 : row 9861 = 1 := (row_9861.checked).trans rfl
theorem model_count_9861 : modelRow 9861 = 1 := by decide
theorem modelrow_agrees_9861 : modelRow 9861 = row 9861 := model_count_9861.trans row_count_9861.symm
def row_9862 : RowCertificate 9862 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9862 1) := by change ¬ Nat.Prime 389055901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29927377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9862 2) := by change ¬ Nat.Prime 389036177; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22884481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9862 3) := by change ¬ Nat.Prime 389016453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129672151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9862 : row 9862 = 0 := (row_9862.checked).trans rfl
theorem model_count_9862 : modelRow 9862 = 0 := by decide
theorem modelrow_agrees_9862 : modelRow 9862 = row 9862 := model_count_9862.trans row_count_9862.symm
def row_9863 : RowCertificate 9863 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9863 1) := by change ¬ Nat.Prime 389134803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129711601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9863 2) := by change ¬ Nat.Prime 389115077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29931929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9863 3) := by change Nat.Prime 389095351; exact ProjectEulerSolutions.P58PrimeNodes.p_389095351.prime
    exact decide_eq_true prime
theorem row_count_9863 : row 9863 = 1 := (row_9863.checked).trans rfl
theorem model_count_9863 : modelRow 9863 = 1 := by decide
theorem modelrow_agrees_9863 : modelRow 9863 = row 9863 := model_count_9863.trans row_count_9863.symm
def row_9864 : RowCertificate 9864 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9864 1) := by change ¬ Nat.Prime 389213713; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55601959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9864 2) := by change ¬ Nat.Prime 389193985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77838797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9864 3) := by change ¬ Nat.Prime 389174257; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5808571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9864 : row 9864 = 0 := (row_9864.checked).trans rfl
theorem model_count_9864 : modelRow 9864 = 0 := by decide
theorem modelrow_agrees_9864 : modelRow 9864 = row 9864 := model_count_9864.trans row_count_9864.symm
def row_9865 : RowCertificate 9865 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9865 1) := by change ¬ Nat.Prime 389292631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55613233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9865 2) := by change ¬ Nat.Prime 389272901; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9494461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9865 3) := by change ¬ Nat.Prime 389253171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129751057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9865 : row 9865 = 0 := (row_9865.checked).trans rfl
theorem model_count_9865 : modelRow 9865 = 0 := by decide
theorem modelrow_agrees_9865 : modelRow 9865 = row 9865 := model_count_9865.trans row_count_9865.symm
def row_9866 : RowCertificate 9866 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9866 1) := by change ¬ Nat.Prime 389371557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129790519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9866 2) := by change ¬ Nat.Prime 389351825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77870365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9866 3) := by change ¬ Nat.Prime 389332093; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10522489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9866 : row 9866 = 0 := (row_9866.checked).trans rfl
theorem model_count_9866 : modelRow 9866 = 0 := by decide
theorem modelrow_agrees_9866 : modelRow 9866 = row 9866 := model_count_9866.trans row_count_9866.symm
def row_9867 : RowCertificate 9867 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9867 1) := by change ¬ Nat.Prime 389450491; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 1061173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9867 2) := by change ¬ Nat.Prime 389430757; exact PrattCertificates.composite_of_product (factor := 809) (cofactor := 481373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9867 3) := by change ¬ Nat.Prime 389411023; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20495317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9867 : row 9867 = 0 := (row_9867.checked).trans rfl
theorem model_count_9867 : modelRow 9867 = 0 := by decide
theorem modelrow_agrees_9867 : modelRow 9867 = row 9867 := model_count_9867.trans row_count_9867.symm
def row_9868 : RowCertificate 9868 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9868 1) := by change Nat.Prime 389529433; exact ProjectEulerSolutions.P58PrimeNodes.p_389529433.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9868 2) := by change Nat.Prime 389509697; exact ProjectEulerSolutions.P58PrimeNodes.p_389509697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9868 3) := by change ¬ Nat.Prime 389489961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129829987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9868 : row 9868 = 2 := (row_9868.checked).trans rfl
theorem model_count_9868 : modelRow 9868 = 2 := by decide
theorem modelrow_agrees_9868 : modelRow 9868 = row 9868 := model_count_9868.trans row_count_9868.symm
def row_9869 : RowCertificate 9869 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9869 1) := by change ¬ Nat.Prime 389608383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129869461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9869 2) := by change ¬ Nat.Prime 389588645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77917729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9869 3) := by change ¬ Nat.Prime 389568907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55652701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9869 : row 9869 = 0 := (row_9869.checked).trans rfl
theorem model_count_9869 : modelRow 9869 = 0 := by decide
theorem modelrow_agrees_9869 : modelRow 9869 = row 9869 := model_count_9869.trans row_count_9869.symm
def row_9870 : RowCertificate 9870 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9870 1) := by change ¬ Nat.Prime 389687341; exact PrattCertificates.composite_of_product (factor := 3613) (cofactor := 107857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9870 2) := by change Nat.Prime 389667601; exact ProjectEulerSolutions.P58PrimeNodes.p_389667601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9870 3) := by change ¬ Nat.Prime 389647861; exact PrattCertificates.composite_of_product (factor := 3109) (cofactor := 125329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9870 : row 9870 = 1 := (row_9870.checked).trans rfl
theorem model_count_9870 : modelRow 9870 = 1 := by decide
theorem modelrow_agrees_9870 : modelRow 9870 = row 9870 := model_count_9870.trans row_count_9870.symm
def row_9871 : RowCertificate 9871 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9871 1) := by change ¬ Nat.Prime 389766307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55680901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9871 2) := by change ¬ Nat.Prime 389746565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77949313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9871 3) := by change ¬ Nat.Prime 389726823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129908941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9871 : row 9871 = 0 := (row_9871.checked).trans rfl
theorem model_count_9871 : modelRow 9871 = 0 := by decide
theorem modelrow_agrees_9871 : modelRow 9871 = row 9871 := model_count_9871.trans row_count_9871.symm
def row_9872 : RowCertificate 9872 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9872 1) := by change ¬ Nat.Prime 389845281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129948427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9872 2) := by change ¬ Nat.Prime 389825537; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1245449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9872 3) := by change ¬ Nat.Prime 389805793; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29985061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9872 : row 9872 = 0 := (row_9872.checked).trans rfl
theorem model_count_9872 : modelRow 9872 = 0 := by decide
theorem modelrow_agrees_9872 : modelRow 9872 = row 9872 := model_count_9872.trans row_count_9872.symm
def row_9873 : RowCertificate 9873 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9873 1) := by change ¬ Nat.Prime 389924263; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1617943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9873 2) := by change ¬ Nat.Prime 389904517; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7356689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9873 3) := by change Nat.Prime 389884771; exact ProjectEulerSolutions.P58PrimeNodes.p_389884771.prime
    exact decide_eq_true prime
theorem row_count_9873 : row 9873 = 1 := (row_9873.checked).trans rfl
theorem model_count_9873 : modelRow 9873 = 1 := by decide
theorem modelrow_agrees_9873 : modelRow 9873 = row 9873 := model_count_9873.trans row_count_9873.symm
def row_9874 : RowCertificate 9874 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9874 1) := by change ¬ Nat.Prime 390003253; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20526487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9874 2) := by change ¬ Nat.Prime 389983505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77996701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9874 3) := by change ¬ Nat.Prime 389963757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129987919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9874 : row 9874 = 0 := (row_9874.checked).trans rfl
theorem model_count_9874 : modelRow 9874 = 0 := by decide
theorem modelrow_agrees_9874 : modelRow 9874 = row 9874 := model_count_9874.trans row_count_9874.symm
def row_9875 : RowCertificate 9875 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9875 1) := by change ¬ Nat.Prime 390082251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130027417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9875 2) := by change ¬ Nat.Prime 390062501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22944853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9875 3) := by change ¬ Nat.Prime 390042751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55720393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9875 : row 9875 = 0 := (row_9875.checked).trans rfl
theorem model_count_9875 : modelRow 9875 = 0 := by decide
theorem modelrow_agrees_9875 : modelRow 9875 = row 9875 := model_count_9875.trans row_count_9875.symm
def row_9876 : RowCertificate 9876 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9876 1) := by change ¬ Nat.Prime 390161257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20534803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9876 2) := by change ¬ Nat.Prime 390141505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78028301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9876 3) := by change ¬ Nat.Prime 390121753; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55731679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9876 : row 9876 = 0 := (row_9876.checked).trans rfl
theorem model_count_9876 : modelRow 9876 = 0 := by decide
theorem modelrow_agrees_9876 : modelRow 9876 = row 9876 := model_count_9876.trans row_count_9876.symm
def row_9877 : RowCertificate 9877 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9877 1) := by change ¬ Nat.Prime 390240271; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 1440001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9877 2) := by change ¬ Nat.Prime 390220517; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2485481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9877 3) := by change ¬ Nat.Prime 390200763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130066921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9877 : row 9877 = 0 := (row_9877.checked).trans rfl
theorem model_count_9877 : modelRow 9877 = 0 := by decide
theorem modelrow_agrees_9877 : modelRow 9877 = row 9877 := model_count_9877.trans row_count_9877.symm
def row_9878 : RowCertificate 9878 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9878 1) := by change ¬ Nat.Prime 390319293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130106431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9878 2) := by change ¬ Nat.Prime 390299537; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5346569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9878 3) := by change ¬ Nat.Prime 390279781; exact PrattCertificates.composite_of_product (factor := 643) (cofactor := 606967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9878 : row 9878 = 0 := (row_9878.checked).trans rfl
theorem model_count_9878 : modelRow 9878 = 0 := by decide
theorem modelrow_agrees_9878 : modelRow 9878 = row 9878 := model_count_9878.trans row_count_9878.symm
def row_9879 : RowCertificate 9879 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9879 1) := by change ¬ Nat.Prime 390398323; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55771189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9879 2) := by change ¬ Nat.Prime 390378565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78075713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9879 3) := by change Nat.Prime 390358807; exact ProjectEulerSolutions.P58PrimeNodes.p_390358807.prime
    exact decide_eq_true prime
theorem row_count_9879 : row 9879 = 1 := (row_9879.checked).trans rfl
theorem model_count_9879 : modelRow 9879 = 1 := by decide
theorem modelrow_agrees_9879 : modelRow 9879 = row 9879 := model_count_9879.trans row_count_9879.symm
def row_9880 : RowCertificate 9880 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9880 1) := by change Nat.Prime 390477361; exact ProjectEulerSolutions.P58PrimeNodes.p_390477361.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9880 2) := by change ¬ Nat.Prime 390457601; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3455377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9880 3) := by change ¬ Nat.Prime 390437841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130145947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9880 : row 9880 = 1 := (row_9880.checked).trans rfl
theorem model_count_9880 : modelRow 9880 = 1 := by decide
theorem modelrow_agrees_9880 : modelRow 9880 = row 9880 := model_count_9880.trans row_count_9880.symm
def row_9881 : RowCertificate 9881 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9881 1) := by change ¬ Nat.Prime 390556407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130185469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9881 2) := by change ¬ Nat.Prime 390536645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78107329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9881 3) := by change Nat.Prime 390516883; exact ProjectEulerSolutions.P58PrimeNodes.p_390516883.prime
    exact decide_eq_true prime
theorem row_count_9881 : row 9881 = 1 := (row_9881.checked).trans rfl
theorem model_count_9881 : modelRow 9881 = 1 := by decide
theorem modelrow_agrees_9881 : modelRow 9881 = row 9881 := model_count_9881.trans row_count_9881.symm
def row_9882 : RowCertificate 9882 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9882 1) := by change Nat.Prime 390635461; exact ProjectEulerSolutions.P58PrimeNodes.p_390635461.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9882 2) := by change ¬ Nat.Prime 390615697; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10557181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9882 3) := by change ¬ Nat.Prime 390595933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55799419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9882 : row 9882 = 1 := (row_9882.checked).trans rfl
theorem model_count_9882 : modelRow 9882 = 1 := by decide
theorem modelrow_agrees_9882 : modelRow 9882 = row 9882 := model_count_9882.trans row_count_9882.symm
def row_9883 : RowCertificate 9883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9883 1) := by change ¬ Nat.Prime 390714523; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 928063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9883 2) := by change ¬ Nat.Prime 390694757; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13472233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9883 3) := by change ¬ Nat.Prime 390674991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130224997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9883 : row 9883 = 0 := (row_9883.checked).trans rfl
theorem model_count_9883 : modelRow 9883 = 0 := by decide
theorem modelrow_agrees_9883 : modelRow 9883 = row 9883 := model_count_9883.trans row_count_9883.symm
def row_9884 : RowCertificate 9884 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9884 1) := by change ¬ Nat.Prime 390793593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130264531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9884 2) := by change ¬ Nat.Prime 390773825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78154765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9884 3) := by change ¬ Nat.Prime 390754057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20566003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9884 : row 9884 = 0 := (row_9884.checked).trans rfl
theorem model_count_9884 : modelRow 9884 = 0 := by decide
theorem modelrow_agrees_9884 : modelRow 9884 = row 9884 := model_count_9884.trans row_count_9884.symm
def row_9885 : RowCertificate 9885 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9885 1) := by change ¬ Nat.Prime 390872671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55838953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9885 2) := by change ¬ Nat.Prime 390852901; exact PrattCertificates.composite_of_product (factor := 14437) (cofactor := 27073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9885 3) := by change ¬ Nat.Prime 390833131; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30064087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9885 : row 9885 = 0 := (row_9885.checked).trans rfl
theorem model_count_9885 : modelRow 9885 = 0 := by decide
theorem modelrow_agrees_9885 : modelRow 9885 = row 9885 := model_count_9885.trans row_count_9885.symm
def row_9886 : RowCertificate 9886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9886 1) := by change ¬ Nat.Prime 390951757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55850251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9886 2) := by change ¬ Nat.Prime 390931985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78186397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9886 3) := by change ¬ Nat.Prime 390912213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130304071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9886 : row 9886 = 0 := (row_9886.checked).trans rfl
theorem model_count_9886 : modelRow 9886 = 0 := by decide
theorem modelrow_agrees_9886 : modelRow 9886 = row 9886 := model_count_9886.trans row_count_9886.symm
def row_9887 : RowCertificate 9887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9887 1) := by change ¬ Nat.Prime 391030851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130343617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9887 2) := by change ¬ Nat.Prime 391011077; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 244841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9887 3) := by change ¬ Nat.Prime 390991303; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9092821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9887 : row 9887 = 0 := (row_9887.checked).trans rfl
theorem model_count_9887 : modelRow 9887 = 0 := by decide
theorem modelrow_agrees_9887 : modelRow 9887 = row 9887 := model_count_9887.trans row_count_9887.symm
def row_9888 : RowCertificate 9888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9888 1) := by change ¬ Nat.Prime 391109953; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30085381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9888 2) := by change ¬ Nat.Prime 391090177; exact PrattCertificates.composite_of_product (factor := 13997) (cofactor := 27941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9888 3) := by change Nat.Prime 391070401; exact ProjectEulerSolutions.P58PrimeNodes.p_391070401.prime
    exact decide_eq_true prime
theorem row_count_9888 : row 9888 = 1 := (row_9888.checked).trans rfl
theorem model_count_9888 : modelRow 9888 = 1 := by decide
theorem modelrow_agrees_9888 : modelRow 9888 = row 9888 := model_count_9888.trans row_count_9888.symm
def row_9889 : RowCertificate 9889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9889 1) := by change ¬ Nat.Prime 391189063; exact PrattCertificates.composite_of_product (factor := 2287) (cofactor := 171049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9889 2) := by change ¬ Nat.Prime 391169285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78233857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9889 3) := by change ¬ Nat.Prime 391149507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130383169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9889 : row 9889 = 0 := (row_9889.checked).trans rfl
theorem model_count_9889 : modelRow 9889 = 0 := by decide
theorem modelrow_agrees_9889 : modelRow 9889 = row 9889 := model_count_9889.trans row_count_9889.symm
def row_9890 : RowCertificate 9890 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9890 1) := by change ¬ Nat.Prime 391268181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130422727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9890 2) := by change Nat.Prime 391248401; exact ProjectEulerSolutions.P58PrimeNodes.p_391248401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9890 3) := by change ¬ Nat.Prime 391228621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55889803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9890 : row 9890 = 1 := (row_9890.checked).trans rfl
theorem model_count_9890 : modelRow 9890 = 1 := by decide
theorem modelrow_agrees_9890 : modelRow 9890 = row 9890 := model_count_9890.trans row_count_9890.symm
def row_9891 : RowCertificate 9891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9891 1) := by change ¬ Nat.Prime 391347307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30103639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9891 2) := by change ¬ Nat.Prime 391327525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78265505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9891 3) := by change ¬ Nat.Prime 391307743; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2400661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9891 : row 9891 = 0 := (row_9891.checked).trans rfl
theorem model_count_9891 : modelRow 9891 = 0 := by decide
theorem modelrow_agrees_9891 : modelRow 9891 = row 9891 := model_count_9891.trans row_count_9891.symm
def row_9892 : RowCertificate 9892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9892 1) := by change ¬ Nat.Prime 391426441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55918063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9892 2) := by change ¬ Nat.Prime 391406657; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23023921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9892 3) := by change ¬ Nat.Prime 391386873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130462291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9892 : row 9892 = 0 := (row_9892.checked).trans rfl
theorem model_count_9892 : modelRow 9892 = 0 := by decide
theorem modelrow_agrees_9892 : modelRow 9892 = row 9892 := model_count_9892.trans row_count_9892.symm
def row_9893 : RowCertificate 9893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9893 1) := by change ¬ Nat.Prime 391505583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130501861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9893 2) := by change ¬ Nat.Prime 391485797; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3876097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9893 3) := by change ¬ Nat.Prime 391466011; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 566521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9893 : row 9893 = 0 := (row_9893.checked).trans rfl
theorem model_count_9893 : modelRow 9893 = 0 := by decide
theorem modelrow_agrees_9893 : modelRow 9893 = row 9893 := model_count_9893.trans row_count_9893.symm
def row_9894 : RowCertificate 9894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9894 1) := by change ¬ Nat.Prime 391584733; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1709977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9894 2) := by change ¬ Nat.Prime 391564945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78312989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9894 3) := by change Nat.Prime 391545157; exact ProjectEulerSolutions.P58PrimeNodes.p_391545157.prime
    exact decide_eq_true prime
theorem row_count_9894 : row 9894 = 1 := (row_9894.checked).trans rfl
theorem model_count_9894 : modelRow 9894 = 1 := by decide
theorem modelrow_agrees_9894 : modelRow 9894 = row 9894 := model_count_9894.trans row_count_9894.symm
def row_9895 : RowCertificate 9895 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9895 1) := by change ¬ Nat.Prime 391663891; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20613889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9895 2) := by change ¬ Nat.Prime 391644101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13504969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9895 3) := by change ¬ Nat.Prime 391624311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130541437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9895 : row 9895 = 0 := (row_9895.checked).trans rfl
theorem model_count_9895 : modelRow 9895 = 0 := by decide
theorem modelrow_agrees_9895 : modelRow 9895 = row 9895 := model_count_9895.trans row_count_9895.symm
def row_9896 : RowCertificate 9896 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9896 1) := by change ¬ Nat.Prime 391743057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130581019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9896 2) := by change ¬ Nat.Prime 391723265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78344653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9896 3) := by change ¬ Nat.Prime 391703473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55957639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9896 : row 9896 = 0 := (row_9896.checked).trans rfl
theorem model_count_9896 : modelRow 9896 = 0 := by decide
theorem modelrow_agrees_9896 : modelRow 9896 = row 9896 := model_count_9896.trans row_count_9896.symm
def row_9897 : RowCertificate 9897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9897 1) := by change ¬ Nat.Prime 391822231; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5848093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9897 2) := by change ¬ Nat.Prime 391802437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30138649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9897 3) := by change ¬ Nat.Prime 391782643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55968949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9897 : row 9897 = 0 := (row_9897.checked).trans rfl
theorem model_count_9897 : modelRow 9897 = 0 := by decide
theorem modelrow_agrees_9897 : modelRow 9897 = row 9897 := model_count_9897.trans row_count_9897.symm
def row_9898 : RowCertificate 9898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9898 1) := by change ¬ Nat.Prime 391901413; exact PrattCertificates.composite_of_product (factor := 14767) (cofactor := 26539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9898 2) := by change Nat.Prime 391881617; exact ProjectEulerSolutions.P58PrimeNodes.p_391881617.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9898 3) := by change ¬ Nat.Prime 391861821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130620607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9898 : row 9898 = 1 := (row_9898.checked).trans rfl
theorem model_count_9898 : modelRow 9898 = 1 := by decide
theorem modelrow_agrees_9898 : modelRow 9898 = row 9898 := model_count_9898.trans row_count_9898.symm
def row_9899 : RowCertificate 9899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9899 1) := by change ¬ Nat.Prime 391980603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130660201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9899 2) := by change ¬ Nat.Prime 391960805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78392161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9899 3) := by change Nat.Prime 391941007; exact ProjectEulerSolutions.P58PrimeNodes.p_391941007.prime
    exact decide_eq_true prime
theorem row_count_9899 : row 9899 = 1 := (row_9899.checked).trans rfl
theorem model_count_9899 : modelRow 9899 = 1 := by decide
theorem modelrow_agrees_9899 : modelRow 9899 = row 9899 := model_count_9899.trans row_count_9899.symm
def row_9900 : RowCertificate 9900 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9900 1) := by change ¬ Nat.Prime 392059801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56008543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9900 2) := by change Nat.Prime 392040001; exact ProjectEulerSolutions.P58PrimeNodes.p_392040001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9900 3) := by change ¬ Nat.Prime 392020201; exact PrattCertificates.composite_of_product (factor := 9769) (cofactor := 40129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9900 : row 9900 = 1 := (row_9900.checked).trans rfl
theorem model_count_9900 : modelRow 9900 = 1 := by decide
theorem modelrow_agrees_9900 : modelRow 9900 = row 9900 := model_count_9900.trans row_count_9900.symm
def row_9901 : RowCertificate 9901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9901 1) := by change ¬ Nat.Prime 392139007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30164539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9901 2) := by change ¬ Nat.Prime 392119205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78423841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9901 3) := by change ¬ Nat.Prime 392099403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130699801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9901 : row 9901 = 0 := (row_9901.checked).trans rfl
theorem model_count_9901 : modelRow 9901 = 0 := by decide
theorem modelrow_agrees_9901 : modelRow 9901 = row 9901 := model_count_9901.trans row_count_9901.symm
def row_9902 : RowCertificate 9902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9902 1) := by change ¬ Nat.Prime 392218221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130739407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9902 2) := by change ¬ Nat.Prime 392198417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30169109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9902 3) := by change ¬ Nat.Prime 392178613; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12650923) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9902 : row 9902 = 0 := (row_9902.checked).trans rfl
theorem model_count_9902 : modelRow 9902 = 0 := by decide
theorem modelrow_agrees_9902 : modelRow 9902 = row 9902 := model_count_9902.trans row_count_9902.symm
def row_9903 : RowCertificate 9903 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9903 1) := by change ¬ Nat.Prime 392297443; exact PrattCertificates.composite_of_product (factor := 1783) (cofactor := 220021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9903 2) := by change ¬ Nat.Prime 392277637; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3883937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9903 3) := by change ¬ Nat.Prime 392257831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56036833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9903 : row 9903 = 0 := (row_9903.checked).trans rfl
theorem model_count_9903 : modelRow 9903 = 0 := by decide
theorem modelrow_agrees_9903 : modelRow 9903 = row 9903 := model_count_9903.trans row_count_9903.symm
def row_9904 : RowCertificate 9904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9904 1) := by change ¬ Nat.Prime 392376673; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30182821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9904 2) := by change ¬ Nat.Prime 392356865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78471373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9904 3) := by change ¬ Nat.Prime 392337057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130779019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9904 : row 9904 = 0 := (row_9904.checked).trans rfl
theorem model_count_9904 : modelRow 9904 = 0 := by decide
theorem modelrow_agrees_9904 : modelRow 9904 = row 9904 := model_count_9904.trans row_count_9904.symm
def row_9905 : RowCertificate 9905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9905 1) := by change ¬ Nat.Prime 392455911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130818637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9905 2) := by change ¬ Nat.Prime 392436101; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5375837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9905 3) := by change ¬ Nat.Prime 392416291; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20653489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9905 : row 9905 = 0 := (row_9905.checked).trans rfl
theorem model_count_9905 : modelRow 9905 = 0 := by decide
theorem modelrow_agrees_9905 : modelRow 9905 = row 9905 := model_count_9905.trans row_count_9905.symm
def row_9906 : RowCertificate 9906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9906 1) := by change ¬ Nat.Prime 392535157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56076451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9906 2) := by change ¬ Nat.Prime 392515345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78503069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9906 3) := by change ¬ Nat.Prime 392495533; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6434353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9906 : row 9906 = 0 := (row_9906.checked).trans rfl
theorem model_count_9906 : modelRow 9906 = 0 := by decide
theorem modelrow_agrees_9906 : modelRow 9906 = row 9906 := model_count_9906.trans row_count_9906.symm
def row_9907 : RowCertificate 9907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9907 1) := by change ¬ Nat.Prime 392614411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56087773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9907 2) := by change ¬ Nat.Prime 392594597; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6435977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9907 3) := by change ¬ Nat.Prime 392574783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130858261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9907 : row 9907 = 0 := (row_9907.checked).trans rfl
theorem model_count_9907 : modelRow 9907 = 0 := by decide
theorem modelrow_agrees_9907 : modelRow 9907 = row 9907 := model_count_9907.trans row_count_9907.symm
def row_9908 : RowCertificate 9908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9908 1) := by change ¬ Nat.Prime 392693673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130897891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9908 2) := by change ¬ Nat.Prime 392673857; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1714733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9908 3) := by change ¬ Nat.Prime 392654041; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30204157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9908 : row 9908 = 0 := (row_9908.checked).trans rfl
theorem model_count_9908 : modelRow 9908 = 0 := by decide
theorem modelrow_agrees_9908 : modelRow 9908 = row 9908 := model_count_9908.trans row_count_9908.symm
def row_9909 : RowCertificate 9909 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9909 1) := by change Nat.Prime 392772943; exact ProjectEulerSolutions.P58PrimeNodes.p_392772943.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9909 2) := by change ¬ Nat.Prime 392753125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78550625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9909 3) := by change ¬ Nat.Prime 392733307; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1254739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9909 : row 9909 = 1 := (row_9909.checked).trans rfl
theorem model_count_9909 : modelRow 9909 = 1 := by decide
theorem modelrow_agrees_9909 : modelRow 9909 = row 9909 := model_count_9909.trans row_count_9909.symm
def row_9910 : RowCertificate 9910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9910 1) := by change ¬ Nat.Prime 392852221; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 787279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9910 2) := by change ¬ Nat.Prime 392832401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30217877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9910 3) := by change ¬ Nat.Prime 392812581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130937527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9910 : row 9910 = 0 := (row_9910.checked).trans rfl
theorem model_count_9910 : modelRow 9910 = 0 := by decide
theorem modelrow_agrees_9910 : modelRow 9910 = row 9910 := model_count_9910.trans row_count_9910.symm
def row_9911 : RowCertificate 9911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9911 1) := by change ¬ Nat.Prime 392931507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 130977169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9911 2) := by change ¬ Nat.Prime 392911685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78582337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9911 3) := by change ¬ Nat.Prime 392891863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56127409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9911 : row 9911 = 0 := (row_9911.checked).trans rfl
theorem model_count_9911 : modelRow 9911 = 0 := by decide
theorem modelrow_agrees_9911 : modelRow 9911 = row 9911 := model_count_9911.trans row_count_9911.symm
def row_9912 : RowCertificate 9912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9912 1) := by change ¬ Nat.Prime 393010801; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20684779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9912 2) := by change ¬ Nat.Prime 392990977; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13551413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9912 3) := by change ¬ Nat.Prime 392971153; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4051249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9912 : row 9912 = 0 := (row_9912.checked).trans rfl
theorem model_count_9912 : modelRow 9912 = 0 := by decide
theorem modelrow_agrees_9912 : modelRow 9912 = row 9912 := model_count_9912.trans row_count_9912.symm
def row_9913 : RowCertificate 9913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9913 1) := by change ¬ Nat.Prime 393090103; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56155729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9913 2) := by change ¬ Nat.Prime 393070277; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23121781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9913 3) := by change ¬ Nat.Prime 393050451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131016817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9913 : row 9913 = 0 := (row_9913.checked).trans rfl
theorem model_count_9913 : modelRow 9913 = 0 := by decide
theorem modelrow_agrees_9913 : modelRow 9913 = row 9913 := model_count_9913.trans row_count_9913.symm
def row_9914 : RowCertificate 9914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9914 1) := by change ¬ Nat.Prime 393169413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131056471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9914 2) := by change ¬ Nat.Prime 393149585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78629917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9914 3) := by change ¬ Nat.Prime 393129757; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2411839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9914 : row 9914 = 0 := (row_9914.checked).trans rfl
theorem model_count_9914 : modelRow 9914 = 0 := by decide
theorem modelrow_agrees_9914 : modelRow 9914 = row 9914 := model_count_9914.trans row_count_9914.symm
def row_9915 : RowCertificate 9915 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9915 1) := by change Nat.Prime 393248731; exact ProjectEulerSolutions.P58PrimeNodes.p_393248731.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9915 2) := by change ¬ Nat.Prime 393228901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30248377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9915 3) := by change ¬ Nat.Prime 393209071; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9144397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9915 : row 9915 = 1 := (row_9915.checked).trans rfl
theorem model_count_9915 : modelRow 9915 = 1 := by decide
theorem modelrow_agrees_9915 : modelRow 9915 = row 9915 := model_count_9915.trans row_count_9915.symm
def row_9916 : RowCertificate 9916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9916 1) := by change ¬ Nat.Prime 393328057; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3818719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9916 2) := by change ¬ Nat.Prime 393308225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78661645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9916 3) := by change ¬ Nat.Prime 393288393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131096131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9916 : row 9916 = 0 := (row_9916.checked).trans rfl
theorem model_count_9916 : modelRow 9916 = 0 := by decide
theorem modelrow_agrees_9916 : modelRow 9916 = row 9916 := model_count_9916.trans row_count_9916.symm
def row_9917 : RowCertificate 9917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9917 1) := by change ¬ Nat.Prime 393407391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131135797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9917 2) := by change ¬ Nat.Prime 393387557; exact PrattCertificates.composite_of_product (factor := 1181) (cofactor := 333097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9917 3) := by change ¬ Nat.Prime 393367723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56195389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9917 : row 9917 = 0 := (row_9917.checked).trans rfl
theorem model_count_9917 : modelRow 9917 = 0 := by decide
theorem modelrow_agrees_9917 : modelRow 9917 = row 9917 := model_count_9917.trans row_count_9917.symm
def row_9918 : RowCertificate 9918 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9918 1) := by change Nat.Prime 393486733; exact ProjectEulerSolutions.P58PrimeNodes.p_393486733.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9918 2) := by change ¬ Nat.Prime 393466897; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6450277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9918 3) := by change ¬ Nat.Prime 393447061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56206723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9918 : row 9918 = 1 := (row_9918.checked).trans rfl
theorem model_count_9918 : modelRow 9918 = 1 := by decide
theorem modelrow_agrees_9918 : modelRow 9918 = row 9918 := model_count_9918.trans row_count_9918.symm
def row_9919 : RowCertificate 9919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9919 1) := by change ¬ Nat.Prime 393566083; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6451903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9919 2) := by change ¬ Nat.Prime 393546245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78709249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9919 3) := by change ¬ Nat.Prime 393526407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131175469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9919 : row 9919 = 0 := (row_9919.checked).trans rfl
theorem model_count_9919 : modelRow 9919 = 0 := by decide
theorem modelrow_agrees_9919 : modelRow 9919 = row 9919 := model_count_9919.trans row_count_9919.symm
def row_9920 : RowCertificate 9920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9920 1) := by change ¬ Nat.Prime 393645441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131215147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9920 2) := by change ¬ Nat.Prime 393625601; exact PrattCertificates.composite_of_product (factor := 2633) (cofactor := 149497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9920 3) := by change ¬ Nat.Prime 393605761; exact PrattCertificates.composite_of_product (factor := 853) (cofactor := 461437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9920 : row 9920 = 0 := (row_9920.checked).trans rfl
theorem model_count_9920 : modelRow 9920 = 0 := by decide
theorem modelrow_agrees_9920 : modelRow 9920 = row 9920 := model_count_9920.trans row_count_9920.symm
def row_9921 : RowCertificate 9921 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9921 1) := by change ¬ Nat.Prime 393724807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56246401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9921 2) := by change ¬ Nat.Prime 393704965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78740993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9921 3) := by change ¬ Nat.Prime 393685123; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30283471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9921 : row 9921 = 0 := (row_9921.checked).trans rfl
theorem model_count_9921 : modelRow 9921 = 0 := by decide
theorem modelrow_agrees_9921 : modelRow 9921 = row 9921 := model_count_9921.trans row_count_9921.symm
def row_9922 : RowCertificate 9922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9922 1) := by change ¬ Nat.Prime 393804181; exact PrattCertificates.composite_of_product (factor := 6043) (cofactor := 65167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9922 2) := by change Nat.Prime 393784337; exact ProjectEulerSolutions.P58PrimeNodes.p_393784337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9922 3) := by change ¬ Nat.Prime 393764493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131254831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9922 : row 9922 = 1 := (row_9922.checked).trans rfl
theorem model_count_9922 : modelRow 9922 = 1 := by decide
theorem modelrow_agrees_9922 : modelRow 9922 = row 9922 := model_count_9922.trans row_count_9922.symm
def row_9923 : RowCertificate 9923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9923 1) := by change ¬ Nat.Prime 393883563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131294521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9923 2) := by change ¬ Nat.Prime 393863717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30297209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9923 3) := by change ¬ Nat.Prime 393843871; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12704641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9923 : row 9923 = 0 := (row_9923.checked).trans rfl
theorem model_count_9923 : modelRow 9923 = 0 := by decide
theorem modelrow_agrees_9923 : modelRow 9923 = row 9923 := model_count_9923.trans row_count_9923.symm
def row_9924 : RowCertificate 9924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9924 1) := by change ¬ Nat.Prime 393962953; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1867123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9924 2) := by change ¬ Nat.Prime 393943105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78788621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9924 3) := by change ¬ Nat.Prime 393923257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56274751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9924 : row 9924 = 0 := (row_9924.checked).trans rfl
theorem model_count_9924 : modelRow 9924 = 0 := by decide
theorem modelrow_agrees_9924 : modelRow 9924 = row 9924 := model_count_9924.trans row_count_9924.symm
def row_9925 : RowCertificate 9925 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9925 1) := by change Nat.Prime 394042351; exact ProjectEulerSolutions.P58PrimeNodes.p_394042351.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9925 2) := by change ¬ Nat.Prime 394022501; exact PrattCertificates.composite_of_product (factor := 1553) (cofactor := 253717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9925 3) := by change ¬ Nat.Prime 394002651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131334217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9925 : row 9925 = 1 := (row_9925.checked).trans rfl
theorem model_count_9925 : modelRow 9925 = 1 := by decide
theorem modelrow_agrees_9925 : modelRow 9925 = row 9925 := model_count_9925.trans row_count_9925.symm
def row_9926 : RowCertificate 9926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9926 1) := by change ¬ Nat.Prime 394121757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131373919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9926 2) := by change ¬ Nat.Prime 394101905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78820381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9926 3) := by change Nat.Prime 394082053; exact ProjectEulerSolutions.P58PrimeNodes.p_394082053.prime
    exact decide_eq_true prime
theorem row_count_9926 : row 9926 = 1 := (row_9926.checked).trans rfl
theorem model_count_9926 : modelRow 9926 = 1 := by decide
theorem modelrow_agrees_9926 : modelRow 9926 = row 9926 := model_count_9926.trans row_count_9926.symm
def row_9927 : RowCertificate 9927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9927 1) := by change ¬ Nat.Prime 394201171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56314453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9927 2) := by change ¬ Nat.Prime 394181317; exact PrattCertificates.composite_of_product (factor := 9749) (cofactor := 40433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9927 3) := by change ¬ Nat.Prime 394161463; exact PrattCertificates.composite_of_product (factor := 1063) (cofactor := 370801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9927 : row 9927 = 0 := (row_9927.checked).trans rfl
theorem model_count_9927 : modelRow 9927 = 0 := by decide
theorem modelrow_agrees_9927 : modelRow 9927 = row 9927 := model_count_9927.trans row_count_9927.symm
def row_9928 : RowCertificate 9928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9928 1) := by change ¬ Nat.Prime 394280593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56325799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9928 2) := by change ¬ Nat.Prime 394260737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30327749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9928 3) := by change ¬ Nat.Prime 394240881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131413627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9928 : row 9928 = 0 := (row_9928.checked).trans rfl
theorem model_count_9928 : modelRow 9928 = 0 := by decide
theorem modelrow_agrees_9928 : modelRow 9928 = row 9928 := model_count_9928.trans row_count_9928.symm
def row_9929 : RowCertificate 9929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9929 1) := by change ¬ Nat.Prime 394360023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131453341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9929 2) := by change ¬ Nat.Prime 394340165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78868033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9929 3) := by change ¬ Nat.Prime 394320307; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 1057159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9929 : row 9929 = 0 := (row_9929.checked).trans rfl
theorem model_count_9929 : modelRow 9929 = 0 := by decide
theorem modelrow_agrees_9929 : modelRow 9929 = row 9929 := model_count_9929.trans row_count_9929.symm
def row_9930 : RowCertificate 9930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9930 1) := by change ¬ Nat.Prime 394439461; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30341497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9930 2) := by change ¬ Nat.Prime 394419601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23201153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9930 3) := by change ¬ Nat.Prime 394399741; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9172087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9930 : row 9930 = 0 := (row_9930.checked).trans rfl
theorem model_count_9930 : modelRow 9930 = 0 := by decide
theorem modelrow_agrees_9930 : modelRow 9930 = row 9930 := model_count_9930.trans row_count_9930.symm
def row_9931 : RowCertificate 9931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9931 1) := by change ¬ Nat.Prime 394518907; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20764153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9931 2) := by change ¬ Nat.Prime 394499045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78899809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9931 3) := by change ¬ Nat.Prime 394479183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131493061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9931 : row 9931 = 0 := (row_9931.checked).trans rfl
theorem model_count_9931 : modelRow 9931 = 0 := by decide
theorem modelrow_agrees_9931 : modelRow 9931 = row 9931 := model_count_9931.trans row_count_9931.symm
def row_9932 : RowCertificate 9932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9932 1) := by change ¬ Nat.Prime 394598361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131532787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9932 2) := by change ¬ Nat.Prime 394578497; exact PrattCertificates.composite_of_product (factor := 6277) (cofactor := 62861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9932 3) := by change ¬ Nat.Prime 394558633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56365519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9932 : row 9932 = 0 := (row_9932.checked).trans rfl
theorem model_count_9932 : modelRow 9932 = 0 := by decide
theorem modelrow_agrees_9932 : modelRow 9932 = row 9932 := model_count_9932.trans row_count_9932.symm
def row_9933 : RowCertificate 9933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9933 1) := by change ¬ Nat.Prime 394677823; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20772517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9933 2) := by change ¬ Nat.Prime 394657957; exact PrattCertificates.composite_of_product (factor := 1873) (cofactor := 210709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9933 3) := by change ¬ Nat.Prime 394638091; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12730261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9933 : row 9933 = 0 := (row_9933.checked).trans rfl
theorem model_count_9933 : modelRow 9933 = 0 := by decide
theorem modelrow_agrees_9933 : modelRow 9933 = row 9933 := model_count_9933.trans row_count_9933.symm
def row_9934 : RowCertificate 9934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9934 1) := by change ¬ Nat.Prime 394757293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56393899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9934 2) := by change ¬ Nat.Prime 394737425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78947485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9934 3) := by change ¬ Nat.Prime 394717557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131572519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9934 : row 9934 = 0 := (row_9934.checked).trans rfl
theorem model_count_9934 : modelRow 9934 = 0 := by decide
theorem modelrow_agrees_9934 : modelRow 9934 = row 9934 := model_count_9934.trans row_count_9934.symm
def row_9935 : RowCertificate 9935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9935 1) := by change ¬ Nat.Prime 394836771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131612257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9935 2) := by change Nat.Prime 394816901; exact ProjectEulerSolutions.P58PrimeNodes.p_394816901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9935 3) := by change ¬ Nat.Prime 394797031; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5892493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9935 : row 9935 = 1 := (row_9935.checked).trans rfl
theorem model_count_9935 : modelRow 9935 = 1 := by decide
theorem modelrow_agrees_9935 : modelRow 9935 = row 9935 := model_count_9935.trans row_count_9935.symm
def row_9936 : RowCertificate 9936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9936 1) := by change ¬ Nat.Prime 394916257; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9184099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9936 2) := by change ¬ Nat.Prime 394896385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 78979277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9936 3) := by change ¬ Nat.Prime 394876513; exact PrattCertificates.composite_of_product (factor := 2617) (cofactor := 150889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9936 : row 9936 = 0 := (row_9936.checked).trans rfl
theorem model_count_9936 : modelRow 9936 = 0 := by decide
theorem modelrow_agrees_9936 : modelRow 9936 = row 9936 := model_count_9936.trans row_count_9936.symm
def row_9937 : RowCertificate 9937 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9937 1) := by change Nat.Prime 394995751; exact ProjectEulerSolutions.P58PrimeNodes.p_394995751.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9937 2) := by change Nat.Prime 394975877; exact ProjectEulerSolutions.P58PrimeNodes.p_394975877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9937 3) := by change ¬ Nat.Prime 394956003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131652001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9937 : row 9937 = 2 := (row_9937.checked).trans rfl
theorem model_count_9937 : modelRow 9937 = 2 := by decide
theorem modelrow_agrees_9937 : modelRow 9937 = row 9937 := model_count_9937.trans row_count_9937.symm
def row_9938 : RowCertificate 9938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9938 1) := by change ¬ Nat.Prime 395075253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131691751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9938 2) := by change ¬ Nat.Prime 395055377; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9635497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9938 3) := by change ¬ Nat.Prime 395035501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56433643) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9938 : row 9938 = 0 := (row_9938.checked).trans rfl
theorem model_count_9938 : modelRow 9938 = 0 := by decide
theorem modelrow_agrees_9938 : modelRow 9938 = row 9938 := model_count_9938.trans row_count_9938.symm
def row_9939 : RowCertificate 9939 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9939 1) := by change Nat.Prime 395154763; exact ProjectEulerSolutions.P58PrimeNodes.p_395154763.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9939 2) := by change ¬ Nat.Prime 395134885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79026977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9939 3) := by change ¬ Nat.Prime 395115007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56445001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9939 : row 9939 = 1 := (row_9939.checked).trans rfl
theorem model_count_9939 : modelRow 9939 = 1 := by decide
theorem modelrow_agrees_9939 : modelRow 9939 = row 9939 := model_count_9939.trans row_count_9939.symm
def row_9940 : RowCertificate 9940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9940 1) := by change ¬ Nat.Prime 395234281; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30402637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9940 2) := by change ¬ Nat.Prime 395214401; exact PrattCertificates.composite_of_product (factor := 9433) (cofactor := 41897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9940 3) := by change ¬ Nat.Prime 395194521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131731507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9940 : row 9940 = 0 := (row_9940.checked).trans rfl
theorem model_count_9940 : modelRow 9940 = 0 := by decide
theorem modelrow_agrees_9940 : modelRow 9940 = row 9940 := model_count_9940.trans row_count_9940.symm
def row_9941 : RowCertificate 9941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9941 1) := by change ¬ Nat.Prime 395313807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131771269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9941 2) := by change ¬ Nat.Prime 395293925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79058785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9941 3) := by change ¬ Nat.Prime 395274043; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20803897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9941 : row 9941 = 0 := (row_9941.checked).trans rfl
theorem model_count_9941 : modelRow 9941 = 0 := by decide
theorem modelrow_agrees_9941 : modelRow 9941 = row 9941 := model_count_9941.trans row_count_9941.symm
def row_9942 : RowCertificate 9942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9942 1) := by change ¬ Nat.Prime 395393341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56484763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9942 2) := by change ¬ Nat.Prime 395373457; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1427341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9942 3) := by change ¬ Nat.Prime 395353573; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3627097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9942 : row 9942 = 0 := (row_9942.checked).trans rfl
theorem model_count_9942 : modelRow 9942 = 0 := by decide
theorem modelrow_agrees_9942 : modelRow 9942 = row 9942 := model_count_9942.trans row_count_9942.symm
def row_9943 : RowCertificate 9943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9943 1) := by change ¬ Nat.Prime 395472883; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30420991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9943 2) := by change ¬ Nat.Prime 395452997; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23261941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9943 3) := by change ¬ Nat.Prime 395433111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131811037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9943 : row 9943 = 0 := (row_9943.checked).trans rfl
theorem model_count_9943 : modelRow 9943 = 0 := by decide
theorem modelrow_agrees_9943 : modelRow 9943 = row 9943 := model_count_9943.trans row_count_9943.symm
def row_9944 : RowCertificate 9944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9944 1) := by change ¬ Nat.Prime 395552433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131850811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9944 2) := by change ¬ Nat.Prime 395532545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79106509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9944 3) := by change ¬ Nat.Prime 395512657; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 854239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9944 : row 9944 = 0 := (row_9944.checked).trans rfl
theorem model_count_9944 : modelRow 9944 = 0 := by decide
theorem modelrow_agrees_9944 : modelRow 9944 = row 9944 := model_count_9944.trans row_count_9944.symm
def row_9945 : RowCertificate 9945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9945 1) := by change ¬ Nat.Prime 395631991; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 316759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9945 2) := by change ¬ Nat.Prime 395612101; exact PrattCertificates.composite_of_product (factor := 1373) (cofactor := 288137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9945 3) := by change ¬ Nat.Prime 395592211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56513173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9945 : row 9945 = 0 := (row_9945.checked).trans rfl
theorem model_count_9945 : modelRow 9945 = 0 := by decide
theorem modelrow_agrees_9945 : modelRow 9945 = row 9945 := model_count_9945.trans row_count_9945.symm
def row_9946 : RowCertificate 9946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9946 1) := by change ¬ Nat.Prime 395711557; exact PrattCertificates.composite_of_product (factor := 12739) (cofactor := 31063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9946 2) := by change ¬ Nat.Prime 395691665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79138333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9946 3) := by change ¬ Nat.Prime 395671773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131890591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9946 : row 9946 = 0 := (row_9946.checked).trans rfl
theorem model_count_9946 : modelRow 9946 = 0 := by decide
theorem modelrow_agrees_9946 : modelRow 9946 = row 9946 := model_count_9946.trans row_count_9946.symm
def row_9947 : RowCertificate 9947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9947 1) := by change ¬ Nat.Prime 395791131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131930377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9947 2) := by change ¬ Nat.Prime 395771237; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23280661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9947 3) := by change ¬ Nat.Prime 395751343; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30442411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9947 : row 9947 = 0 := (row_9947.checked).trans rfl
theorem model_count_9947 : modelRow 9947 = 0 := by decide
theorem modelrow_agrees_9947 : modelRow 9947 = row 9947 := model_count_9947.trans row_count_9947.symm
def row_9948 : RowCertificate 9948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9948 1) := by change ¬ Nat.Prime 395870713; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56552959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9948 2) := by change Nat.Prime 395850817; exact ProjectEulerSolutions.P58PrimeNodes.p_395850817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9948 3) := by change ¬ Nat.Prime 395830921; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10698133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9948 : row 9948 = 1 := (row_9948.checked).trans rfl
theorem model_count_9948 : modelRow 9948 = 1 := by decide
theorem modelrow_agrees_9948 : modelRow 9948 = row 9948 := model_count_9948.trans row_count_9948.symm
def row_9949 : RowCertificate 9949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9949 1) := by change ¬ Nat.Prime 395950303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56564329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9949 2) := by change ¬ Nat.Prime 395930405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79186081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9949 3) := by change ¬ Nat.Prime 395910507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 131970169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9949 : row 9949 = 0 := (row_9949.checked).trans rfl
theorem model_count_9949 : modelRow 9949 = 0 := by decide
theorem modelrow_agrees_9949 : modelRow 9949 = row 9949 := model_count_9949.trans row_count_9949.symm
def row_9950 : RowCertificate 9950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9950 1) := by change ¬ Nat.Prime 396029901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132009967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9950 2) := by change ¬ Nat.Prime 396010001; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10702973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9950 3) := by change ¬ Nat.Prime 395990101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30460777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9950 : row 9950 = 0 := (row_9950.checked).trans rfl
theorem model_count_9950 : modelRow 9950 = 0 := by decide
theorem modelrow_agrees_9950 : modelRow 9950 = row 9950 := model_count_9950.trans row_count_9950.symm
def row_9951 : RowCertificate 9951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9951 1) := by change ¬ Nat.Prime 396109507; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9211849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9951 2) := by change ¬ Nat.Prime 396089605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79217921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9951 3) := by change ¬ Nat.Prime 396069703; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3633667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9951 : row 9951 = 0 := (row_9951.checked).trans rfl
theorem model_count_9951 : modelRow 9951 = 0 := by decide
theorem modelrow_agrees_9951 : modelRow 9951 = row 9951 := model_count_9951.trans row_count_9951.symm
def row_9952 : RowCertificate 9952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9952 1) := by change ¬ Nat.Prime 396189121; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20852059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9952 2) := by change Nat.Prime 396169217; exact ProjectEulerSolutions.P58PrimeNodes.p_396169217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9952 3) := by change ¬ Nat.Prime 396149313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132049771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9952 : row 9952 = 1 := (row_9952.checked).trans rfl
theorem model_count_9952 : modelRow 9952 = 1 := by decide
theorem modelrow_agrees_9952 : modelRow 9952 = row 9952 := model_count_9952.trans row_count_9952.symm
def row_9953 : RowCertificate 9953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9953 1) := by change ¬ Nat.Prime 396268743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132089581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9953 2) := by change ¬ Nat.Prime 396248837; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13663753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9953 3) := by change ¬ Nat.Prime 396228931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56604133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9953 : row 9953 = 0 := (row_9953.checked).trans rfl
theorem model_count_9953 : modelRow 9953 = 0 := by decide
theorem modelrow_agrees_9953 : modelRow 9953 = row 9953 := model_count_9953.trans row_count_9953.symm
def row_9954 : RowCertificate 9954 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9954 1) := by change Nat.Prime 396348373; exact ProjectEulerSolutions.P58PrimeNodes.p_396348373.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9954 2) := by change ¬ Nat.Prime 396328465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79265693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9954 3) := by change ¬ Nat.Prime 396308557; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12784147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9954 : row 9954 = 1 := (row_9954.checked).trans rfl
theorem model_count_9954 : modelRow 9954 = 1 := by decide
theorem modelrow_agrees_9954 : modelRow 9954 = row 9954 := model_count_9954.trans row_count_9954.symm
def row_9955 : RowCertificate 9955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9955 1) := by change ¬ Nat.Prime 396428011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56632573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9955 2) := by change ¬ Nat.Prime 396408101; exact PrattCertificates.composite_of_product (factor := 4441) (cofactor := 89261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9955 3) := by change ¬ Nat.Prime 396388191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132129397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9955 : row 9955 = 0 := (row_9955.checked).trans rfl
theorem model_count_9955 : modelRow 9955 = 0 := by decide
theorem modelrow_agrees_9955 : modelRow 9955 = row 9955 := model_count_9955.trans row_count_9955.symm
def row_9956 : RowCertificate 9956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9956 1) := by change ¬ Nat.Prime 396507657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132169219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9956 2) := by change ¬ Nat.Prime 396487745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79297549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9956 3) := by change Nat.Prime 396467833; exact ProjectEulerSolutions.P58PrimeNodes.p_396467833.prime
    exact decide_eq_true prime
theorem row_count_9956 : row 9956 = 1 := (row_9956.checked).trans rfl
theorem model_count_9956 : modelRow 9956 = 1 := by decide
theorem modelrow_agrees_9956 : modelRow 9956 = row 9956 := model_count_9956.trans row_count_9956.symm
def row_9957 : RowCertificate 9957 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9957 1) := by change Nat.Prime 396587311; exact ProjectEulerSolutions.P58PrimeNodes.p_396587311.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9957 2) := by change ¬ Nat.Prime 396567397; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3638233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9957 3) := by change ¬ Nat.Prime 396547483; exact PrattCertificates.composite_of_product (factor := 11827) (cofactor := 33529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9957 : row 9957 = 1 := (row_9957.checked).trans rfl
theorem model_count_9957 : modelRow 9957 = 1 := by decide
theorem modelrow_agrees_9957 : modelRow 9957 = row 9957 := model_count_9957.trans row_count_9957.symm
def row_9958 : RowCertificate 9958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9958 1) := by change ¬ Nat.Prime 396666973; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10720729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9958 2) := by change Nat.Prime 396647057; exact ProjectEulerSolutions.P58PrimeNodes.p_396647057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9958 3) := by change ¬ Nat.Prime 396627141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132209047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9958 : row 9958 = 1 := (row_9958.checked).trans rfl
theorem model_count_9958 : modelRow 9958 = 1 := by decide
theorem modelrow_agrees_9958 : modelRow 9958 = row 9958 := model_count_9958.trans row_count_9958.symm
def row_9959 : RowCertificate 9959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9959 1) := by change ¬ Nat.Prime 396746643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132248881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9959 2) := by change ¬ Nat.Prime 396726725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79345345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9959 3) := by change ¬ Nat.Prime 396706807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56672401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9959 : row 9959 = 0 := (row_9959.checked).trans rfl
theorem model_count_9959 : modelRow 9959 = 0 := by decide
theorem modelrow_agrees_9959 : modelRow 9959 = row 9959 := model_count_9959.trans row_count_9959.symm
def row_9960 : RowCertificate 9960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9960 1) := by change ¬ Nat.Prime 396826321; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5435977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9960 2) := by change ¬ Nat.Prime 396806401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23341553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9960 3) := by change ¬ Nat.Prime 396786481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56683783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9960 : row 9960 = 0 := (row_9960.checked).trans rfl
theorem model_count_9960 : modelRow 9960 = 0 := by decide
theorem modelrow_agrees_9960 : modelRow 9960 = row 9960 := model_count_9960.trans row_count_9960.symm
def row_9961 : RowCertificate 9961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9961 1) := by change ¬ Nat.Prime 396906007; exact PrattCertificates.composite_of_product (factor := 10243) (cofactor := 38749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9961 2) := by change ¬ Nat.Prime 396886085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79377217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9961 3) := by change ¬ Nat.Prime 396866163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132288721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9961 : row 9961 = 0 := (row_9961.checked).trans rfl
theorem model_count_9961 : modelRow 9961 = 0 := by decide
theorem modelrow_agrees_9961 : modelRow 9961 = row 9961 := model_count_9961.trans row_count_9961.symm
def row_9962 : RowCertificate 9962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9962 1) := by change ¬ Nat.Prime 396985701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132328567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9962 2) := by change ¬ Nat.Prime 396965777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30535829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9962 3) := by change ¬ Nat.Prime 396945853; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20891887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9962 : row 9962 = 0 := (row_9962.checked).trans rfl
theorem model_count_9962 : modelRow 9962 = 0 := by decide
theorem modelrow_agrees_9962 : modelRow 9962 = row 9962 := model_count_9962.trans row_count_9962.symm
def row_9963 : RowCertificate 9963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9963 1) := by change ¬ Nat.Prime 397065403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56723629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9963 2) := by change Nat.Prime 397045477; exact ProjectEulerSolutions.P58PrimeNodes.p_397045477.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9963 3) := by change ¬ Nat.Prime 397025551; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30540427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9963 : row 9963 = 1 := (row_9963.checked).trans rfl
theorem model_count_9963 : modelRow 9963 = 1 := by decide
theorem modelrow_agrees_9963 : modelRow 9963 = row 9963 := model_count_9963.trans row_count_9963.symm
def row_9964 : RowCertificate 9964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9964 1) := by change ¬ Nat.Prime 397145113; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5927539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9964 2) := by change ¬ Nat.Prime 397125185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79425037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9964 3) := by change ¬ Nat.Prime 397105257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132368419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9964 : row 9964 = 0 := (row_9964.checked).trans rfl
theorem model_count_9964 : modelRow 9964 = 0 := by decide
theorem modelrow_agrees_9964 : modelRow 9964 = row 9964 := model_count_9964.trans row_count_9964.symm
def row_9965 : RowCertificate 9965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9965 1) := by change ¬ Nat.Prime 397224831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132408277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9965 2) := by change Nat.Prime 397204901; exact ProjectEulerSolutions.P58PrimeNodes.p_397204901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9965 3) := by change ¬ Nat.Prime 397184971; exact PrattCertificates.composite_of_product (factor := 12841) (cofactor := 30931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9965 : row 9965 = 1 := (row_9965.checked).trans rfl
theorem model_count_9965 : modelRow 9965 = 1 := by decide
theorem modelrow_agrees_9965 : modelRow 9965 = row 9965 := model_count_9965.trans row_count_9965.symm
def row_9966 : RowCertificate 9966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9966 1) := by change ¬ Nat.Prime 397304557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30561889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9966 2) := by change ¬ Nat.Prime 397284625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79456925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9966 3) := by change ¬ Nat.Prime 397264693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56752099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9966 : row 9966 = 0 := (row_9966.checked).trans rfl
theorem model_count_9966 : modelRow 9966 = 0 := by decide
theorem modelrow_agrees_9966 : modelRow 9966 = row 9966 := model_count_9966.trans row_count_9966.symm
def row_9967 : RowCertificate 9967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9967 1) := by change ¬ Nat.Prime 397384291; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 759817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9967 2) := by change ¬ Nat.Prime 397364357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30566489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9967 3) := by change ¬ Nat.Prime 397344423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132448141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9967 : row 9967 = 0 := (row_9967.checked).trans rfl
theorem model_count_9967 : modelRow 9967 = 0 := by decide
theorem modelrow_agrees_9967 : modelRow 9967 = row 9967 := model_count_9967.trans row_count_9967.symm
def row_9968 : RowCertificate 9968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9968 1) := by change ¬ Nat.Prime 397464033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132488011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9968 2) := by change ¬ Nat.Prime 397444097; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6515477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9968 3) := by change ¬ Nat.Prime 397424161; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3858487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9968 : row 9968 = 0 := (row_9968.checked).trans rfl
theorem model_count_9968 : modelRow 9968 = 0 := by decide
theorem modelrow_agrees_9968 : modelRow 9968 = row 9968 := model_count_9968.trans row_count_9968.symm
def row_9969 : RowCertificate 9969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9969 1) := by change ¬ Nat.Prime 397543783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56791969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9969 2) := by change ¬ Nat.Prime 397523845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79504769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9969 3) := by change ¬ Nat.Prime 397503907; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5445259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9969 : row 9969 = 0 := (row_9969.checked).trans rfl
theorem model_count_9969 : modelRow 9969 = 0 := by decide
theorem modelrow_agrees_9969 : modelRow 9969 = row 9969 := model_count_9969.trans row_count_9969.symm
def row_9970 : RowCertificate 9970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9970 1) := by change ¬ Nat.Prime 397623541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56803363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9970 2) := by change ¬ Nat.Prime 397603601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13710469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9970 3) := by change ¬ Nat.Prime 397583661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132527887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9970 : row 9970 = 0 := (row_9970.checked).trans rfl
theorem model_count_9970 : modelRow 9970 = 0 := by decide
theorem modelrow_agrees_9970 : modelRow 9970 = row 9970 := model_count_9970.trans row_count_9970.symm
def row_9971 : RowCertificate 9971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9971 1) := by change ¬ Nat.Prime 397703307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132567769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9971 2) := by change ¬ Nat.Prime 397683365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79536673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9971 3) := by change ¬ Nat.Prime 397663423; exact PrattCertificates.composite_of_product (factor := 7639) (cofactor := 52057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9971 : row 9971 = 0 := (row_9971.checked).trans rfl
theorem model_count_9971 : modelRow 9971 = 0 := by decide
theorem modelrow_agrees_9971 : modelRow 9971 = row 9971 := model_count_9971.trans row_count_9971.symm
def row_9972 : RowCertificate 9972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9972 1) := by change ¬ Nat.Prime 397783081; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2440387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9972 2) := by change Nat.Prime 397763137; exact ProjectEulerSolutions.P58PrimeNodes.p_397763137.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9972 3) := by change Nat.Prime 397743193; exact ProjectEulerSolutions.P58PrimeNodes.p_397743193.prime
    exact decide_eq_true prime
theorem row_count_9972 : row 9972 = 2 := (row_9972.checked).trans rfl
theorem model_count_9972 : modelRow 9972 = 2 := by decide
theorem modelrow_agrees_9972 : modelRow 9972 = row 9972 := model_count_9972.trans row_count_9972.symm
def row_9973 : RowCertificate 9973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9973 1) := by change ¬ Nat.Prime 397862863; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4101679) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9973 2) := by change Nat.Prime 397842917; exact ProjectEulerSolutions.P58PrimeNodes.p_397842917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9973 3) := by change ¬ Nat.Prime 397822971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132607657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9973 : row 9973 = 1 := (row_9973.checked).trans rfl
theorem model_count_9973 : modelRow 9973 = 1 := by decide
theorem modelrow_agrees_9973 : modelRow 9973 = row 9973 := model_count_9973.trans row_count_9973.symm
def row_9974 : RowCertificate 9974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9974 1) := by change ¬ Nat.Prime 397942653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132647551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9974 2) := by change ¬ Nat.Prime 397922705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79584541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9974 3) := by change ¬ Nat.Prime 397902757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56843251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9974 : row 9974 = 0 := (row_9974.checked).trans rfl
theorem model_count_9974 : modelRow 9974 = 0 := by decide
theorem modelrow_agrees_9974 : modelRow 9974 = row 9974 := model_count_9974.trans row_count_9974.symm
def row_9975 : RowCertificate 9975 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9975 1) := by change Nat.Prime 398022451; exact ProjectEulerSolutions.P58PrimeNodes.p_398022451.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9975 2) := by change ¬ Nat.Prime 398002501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30615577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9975 3) := by change ¬ Nat.Prime 397982551; exact PrattCertificates.composite_of_product (factor := 8563) (cofactor := 46477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9975 : row 9975 = 1 := (row_9975.checked).trans rfl
theorem model_count_9975 : modelRow 9975 = 1 := by decide
theorem modelrow_agrees_9975 : modelRow 9975 = row 9975 := model_count_9975.trans row_count_9975.symm
def row_9976 : RowCertificate 9976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9976 1) := by change ¬ Nat.Prime 398102257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56871751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9976 2) := by change ¬ Nat.Prime 398082305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79616461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9976 3) := by change ¬ Nat.Prime 398062353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132687451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9976 : row 9976 = 0 := (row_9976.checked).trans rfl
theorem model_count_9976 : modelRow 9976 = 0 := by decide
theorem modelrow_agrees_9976 : modelRow 9976 = row 9976 := model_count_9976.trans row_count_9976.symm
def row_9977 : RowCertificate 9977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9977 1) := by change ¬ Nat.Prime 398182071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132727357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9977 2) := by change ¬ Nat.Prime 398162117; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23421301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9977 3) := by change ¬ Nat.Prime 398142163; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10760599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9977 : row 9977 = 0 := (row_9977.checked).trans rfl
theorem model_count_9977 : modelRow 9977 = 0 := by decide
theorem modelrow_agrees_9977 : modelRow 9977 = row 9977 := model_count_9977.trans row_count_9977.symm
def row_9978 : RowCertificate 9978 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9978 1) := by change Nat.Prime 398261893; exact ProjectEulerSolutions.P58PrimeNodes.p_398261893.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9978 2) := by change ¬ Nat.Prime 398241937; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5455369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9978 3) := by change Nat.Prime 398221981; exact ProjectEulerSolutions.P58PrimeNodes.p_398221981.prime
    exact decide_eq_true prime
theorem row_count_9978 : row 9978 = 2 := (row_9978.checked).trans rfl
theorem model_count_9978 : modelRow 9978 = 2 := by decide
theorem modelrow_agrees_9978 : modelRow 9978 = row 9978 := model_count_9978.trans row_count_9978.symm
def row_9979 : RowCertificate 9979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9979 1) := by change ¬ Nat.Prime 398341723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30641671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9979 2) := by change ¬ Nat.Prime 398321765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79664353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9979 3) := by change ¬ Nat.Prime 398301807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132767269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9979 : row 9979 = 0 := (row_9979.checked).trans rfl
theorem model_count_9979 : modelRow 9979 = 0 := by decide
theorem modelrow_agrees_9979 : modelRow 9979 = row 9979 := model_count_9979.trans row_count_9979.symm
def row_9980 : RowCertificate 9980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9980 1) := by change ¬ Nat.Prime 398421561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132807187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9980 2) := by change ¬ Nat.Prime 398401601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30646277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9980 3) := by change ¬ Nat.Prime 398381641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56911663) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9980 : row 9980 = 0 := (row_9980.checked).trans rfl
theorem model_count_9980 : modelRow 9980 = 0 := by decide
theorem modelrow_agrees_9980 : modelRow 9980 = row 9980 := model_count_9980.trans row_count_9980.symm
def row_9981 : RowCertificate 9981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9981 1) := by change ¬ Nat.Prime 398501407; exact PrattCertificates.composite_of_product (factor := 2341) (cofactor := 170227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9981 2) := by change ¬ Nat.Prime 398481445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79696289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9981 3) := by change ¬ Nat.Prime 398461483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56923069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9981 : row 9981 = 0 := (row_9981.checked).trans rfl
theorem model_count_9981 : modelRow 9981 = 0 := by decide
theorem modelrow_agrees_9981 : modelRow 9981 = row 9981 := model_count_9981.trans row_count_9981.symm
def row_9982 : RowCertificate 9982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9982 1) := by change ¬ Nat.Prime 398581261; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30660097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9982 2) := by change ¬ Nat.Prime 398561297; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13743493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9982 3) := by change ¬ Nat.Prime 398541333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132847111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9982 : row 9982 = 0 := (row_9982.checked).trans rfl
theorem model_count_9982 : modelRow 9982 = 0 := by decide
theorem modelrow_agrees_9982 : modelRow 9982 = row 9982 := model_count_9982.trans row_count_9982.symm
def row_9983 : RowCertificate 9983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9983 1) := by change ¬ Nat.Prime 398661123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132887041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9983 2) := by change Nat.Prime 398641157; exact ProjectEulerSolutions.P58PrimeNodes.p_398641157.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9983 3) := by change Nat.Prime 398621191; exact ProjectEulerSolutions.P58PrimeNodes.p_398621191.prime
    exact decide_eq_true prime
theorem row_count_9983 : row 9983 = 2 := (row_9983.checked).trans rfl
theorem model_count_9983 : modelRow 9983 = 2 := by decide
theorem modelrow_agrees_9983 : modelRow 9983 = row 9983 := model_count_9983.trans row_count_9983.symm
def row_9984 : RowCertificate 9984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9984 1) := by change ¬ Nat.Prime 398740993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56962999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9984 2) := by change ¬ Nat.Prime 398721025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79744205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9984 3) := by change Nat.Prime 398701057; exact ProjectEulerSolutions.P58PrimeNodes.p_398701057.prime
    exact decide_eq_true prime
theorem row_count_9984 : row 9984 = 1 := (row_9984.checked).trans rfl
theorem model_count_9984 : modelRow 9984 = 1 := by decide
theorem modelrow_agrees_9984 : modelRow 9984 = row 9984 := model_count_9984.trans row_count_9984.symm
private def atomic_9857 : CountSegment 1 10 13120 9856 9857 4085 4086 := CountSegment.single (first := 9856) (initial := 4085) row_count_9857 (by decide)
private theorem atomic_9857_agreement : RowAgreement 9856 9857 := RowAgreement.single (first := 9856) modelrow_agrees_9857
private def atomic_9858 : CountSegment 1 10 13120 9857 9858 4086 4086 := CountSegment.single (first := 9857) (initial := 4086) row_count_9858 (by decide)
private theorem atomic_9858_agreement : RowAgreement 9857 9858 := RowAgreement.single (first := 9857) modelrow_agrees_9858
private def atomic_9859 : CountSegment 1 10 13120 9858 9859 4086 4086 := CountSegment.single (first := 9858) (initial := 4086) row_count_9859 (by decide)
private theorem atomic_9859_agreement : RowAgreement 9858 9859 := RowAgreement.single (first := 9858) modelrow_agrees_9859
private def atomic_9860 : CountSegment 1 10 13120 9859 9860 4086 4086 := CountSegment.single (first := 9859) (initial := 4086) row_count_9860 (by decide)
private theorem atomic_9860_agreement : RowAgreement 9859 9860 := RowAgreement.single (first := 9859) modelrow_agrees_9860
private def atomic_9861 : CountSegment 1 10 13120 9860 9861 4086 4087 := CountSegment.single (first := 9860) (initial := 4086) row_count_9861 (by decide)
private theorem atomic_9861_agreement : RowAgreement 9860 9861 := RowAgreement.single (first := 9860) modelrow_agrees_9861
private def atomic_9862 : CountSegment 1 10 13120 9861 9862 4087 4087 := CountSegment.single (first := 9861) (initial := 4087) row_count_9862 (by decide)
private theorem atomic_9862_agreement : RowAgreement 9861 9862 := RowAgreement.single (first := 9861) modelrow_agrees_9862
private def atomic_9863 : CountSegment 1 10 13120 9862 9863 4087 4088 := CountSegment.single (first := 9862) (initial := 4087) row_count_9863 (by decide)
private theorem atomic_9863_agreement : RowAgreement 9862 9863 := RowAgreement.single (first := 9862) modelrow_agrees_9863
private def atomic_9864 : CountSegment 1 10 13120 9863 9864 4088 4088 := CountSegment.single (first := 9863) (initial := 4088) row_count_9864 (by decide)
private theorem atomic_9864_agreement : RowAgreement 9863 9864 := RowAgreement.single (first := 9863) modelrow_agrees_9864
private def atomic_9865 : CountSegment 1 10 13120 9864 9865 4088 4088 := CountSegment.single (first := 9864) (initial := 4088) row_count_9865 (by decide)
private theorem atomic_9865_agreement : RowAgreement 9864 9865 := RowAgreement.single (first := 9864) modelrow_agrees_9865
private def atomic_9866 : CountSegment 1 10 13120 9865 9866 4088 4088 := CountSegment.single (first := 9865) (initial := 4088) row_count_9866 (by decide)
private theorem atomic_9866_agreement : RowAgreement 9865 9866 := RowAgreement.single (first := 9865) modelrow_agrees_9866
private def atomic_9867 : CountSegment 1 10 13120 9866 9867 4088 4088 := CountSegment.single (first := 9866) (initial := 4088) row_count_9867 (by decide)
private theorem atomic_9867_agreement : RowAgreement 9866 9867 := RowAgreement.single (first := 9866) modelrow_agrees_9867
private def atomic_9868 : CountSegment 1 10 13120 9867 9868 4088 4090 := CountSegment.single (first := 9867) (initial := 4088) row_count_9868 (by decide)
private theorem atomic_9868_agreement : RowAgreement 9867 9868 := RowAgreement.single (first := 9867) modelrow_agrees_9868
private def atomic_9869 : CountSegment 1 10 13120 9868 9869 4090 4090 := CountSegment.single (first := 9868) (initial := 4090) row_count_9869 (by decide)
private theorem atomic_9869_agreement : RowAgreement 9868 9869 := RowAgreement.single (first := 9868) modelrow_agrees_9869
private def atomic_9870 : CountSegment 1 10 13120 9869 9870 4090 4091 := CountSegment.single (first := 9869) (initial := 4090) row_count_9870 (by decide)
private theorem atomic_9870_agreement : RowAgreement 9869 9870 := RowAgreement.single (first := 9869) modelrow_agrees_9870
private def atomic_9871 : CountSegment 1 10 13120 9870 9871 4091 4091 := CountSegment.single (first := 9870) (initial := 4091) row_count_9871 (by decide)
private theorem atomic_9871_agreement : RowAgreement 9870 9871 := RowAgreement.single (first := 9870) modelrow_agrees_9871
private def atomic_9872 : CountSegment 1 10 13120 9871 9872 4091 4091 := CountSegment.single (first := 9871) (initial := 4091) row_count_9872 (by decide)
private theorem atomic_9872_agreement : RowAgreement 9871 9872 := RowAgreement.single (first := 9871) modelrow_agrees_9872
private def atomic_9873 : CountSegment 1 10 13120 9872 9873 4091 4092 := CountSegment.single (first := 9872) (initial := 4091) row_count_9873 (by decide)
private theorem atomic_9873_agreement : RowAgreement 9872 9873 := RowAgreement.single (first := 9872) modelrow_agrees_9873
private def atomic_9874 : CountSegment 1 10 13120 9873 9874 4092 4092 := CountSegment.single (first := 9873) (initial := 4092) row_count_9874 (by decide)
private theorem atomic_9874_agreement : RowAgreement 9873 9874 := RowAgreement.single (first := 9873) modelrow_agrees_9874
private def atomic_9875 : CountSegment 1 10 13120 9874 9875 4092 4092 := CountSegment.single (first := 9874) (initial := 4092) row_count_9875 (by decide)
private theorem atomic_9875_agreement : RowAgreement 9874 9875 := RowAgreement.single (first := 9874) modelrow_agrees_9875
private def atomic_9876 : CountSegment 1 10 13120 9875 9876 4092 4092 := CountSegment.single (first := 9875) (initial := 4092) row_count_9876 (by decide)
private theorem atomic_9876_agreement : RowAgreement 9875 9876 := RowAgreement.single (first := 9875) modelrow_agrees_9876
private def atomic_9877 : CountSegment 1 10 13120 9876 9877 4092 4092 := CountSegment.single (first := 9876) (initial := 4092) row_count_9877 (by decide)
private theorem atomic_9877_agreement : RowAgreement 9876 9877 := RowAgreement.single (first := 9876) modelrow_agrees_9877
private def atomic_9878 : CountSegment 1 10 13120 9877 9878 4092 4092 := CountSegment.single (first := 9877) (initial := 4092) row_count_9878 (by decide)
private theorem atomic_9878_agreement : RowAgreement 9877 9878 := RowAgreement.single (first := 9877) modelrow_agrees_9878
private def atomic_9879 : CountSegment 1 10 13120 9878 9879 4092 4093 := CountSegment.single (first := 9878) (initial := 4092) row_count_9879 (by decide)
private theorem atomic_9879_agreement : RowAgreement 9878 9879 := RowAgreement.single (first := 9878) modelrow_agrees_9879
private def atomic_9880 : CountSegment 1 10 13120 9879 9880 4093 4094 := CountSegment.single (first := 9879) (initial := 4093) row_count_9880 (by decide)
private theorem atomic_9880_agreement : RowAgreement 9879 9880 := RowAgreement.single (first := 9879) modelrow_agrees_9880
private def atomic_9881 : CountSegment 1 10 13120 9880 9881 4094 4095 := CountSegment.single (first := 9880) (initial := 4094) row_count_9881 (by decide)
private theorem atomic_9881_agreement : RowAgreement 9880 9881 := RowAgreement.single (first := 9880) modelrow_agrees_9881
private def atomic_9882 : CountSegment 1 10 13120 9881 9882 4095 4096 := CountSegment.single (first := 9881) (initial := 4095) row_count_9882 (by decide)
private theorem atomic_9882_agreement : RowAgreement 9881 9882 := RowAgreement.single (first := 9881) modelrow_agrees_9882
private def atomic_9883 : CountSegment 1 10 13120 9882 9883 4096 4096 := CountSegment.single (first := 9882) (initial := 4096) row_count_9883 (by decide)
private theorem atomic_9883_agreement : RowAgreement 9882 9883 := RowAgreement.single (first := 9882) modelrow_agrees_9883
private def atomic_9884 : CountSegment 1 10 13120 9883 9884 4096 4096 := CountSegment.single (first := 9883) (initial := 4096) row_count_9884 (by decide)
private theorem atomic_9884_agreement : RowAgreement 9883 9884 := RowAgreement.single (first := 9883) modelrow_agrees_9884
private def atomic_9885 : CountSegment 1 10 13120 9884 9885 4096 4096 := CountSegment.single (first := 9884) (initial := 4096) row_count_9885 (by decide)
private theorem atomic_9885_agreement : RowAgreement 9884 9885 := RowAgreement.single (first := 9884) modelrow_agrees_9885
private def atomic_9886 : CountSegment 1 10 13120 9885 9886 4096 4096 := CountSegment.single (first := 9885) (initial := 4096) row_count_9886 (by decide)
private theorem atomic_9886_agreement : RowAgreement 9885 9886 := RowAgreement.single (first := 9885) modelrow_agrees_9886
private def atomic_9887 : CountSegment 1 10 13120 9886 9887 4096 4096 := CountSegment.single (first := 9886) (initial := 4096) row_count_9887 (by decide)
private theorem atomic_9887_agreement : RowAgreement 9886 9887 := RowAgreement.single (first := 9886) modelrow_agrees_9887
private def atomic_9888 : CountSegment 1 10 13120 9887 9888 4096 4097 := CountSegment.single (first := 9887) (initial := 4096) row_count_9888 (by decide)
private theorem atomic_9888_agreement : RowAgreement 9887 9888 := RowAgreement.single (first := 9887) modelrow_agrees_9888
private def atomic_9889 : CountSegment 1 10 13120 9888 9889 4097 4097 := CountSegment.single (first := 9888) (initial := 4097) row_count_9889 (by decide)
private theorem atomic_9889_agreement : RowAgreement 9888 9889 := RowAgreement.single (first := 9888) modelrow_agrees_9889
private def atomic_9890 : CountSegment 1 10 13120 9889 9890 4097 4098 := CountSegment.single (first := 9889) (initial := 4097) row_count_9890 (by decide)
private theorem atomic_9890_agreement : RowAgreement 9889 9890 := RowAgreement.single (first := 9889) modelrow_agrees_9890
private def atomic_9891 : CountSegment 1 10 13120 9890 9891 4098 4098 := CountSegment.single (first := 9890) (initial := 4098) row_count_9891 (by decide)
private theorem atomic_9891_agreement : RowAgreement 9890 9891 := RowAgreement.single (first := 9890) modelrow_agrees_9891
private def atomic_9892 : CountSegment 1 10 13120 9891 9892 4098 4098 := CountSegment.single (first := 9891) (initial := 4098) row_count_9892 (by decide)
private theorem atomic_9892_agreement : RowAgreement 9891 9892 := RowAgreement.single (first := 9891) modelrow_agrees_9892
private def atomic_9893 : CountSegment 1 10 13120 9892 9893 4098 4098 := CountSegment.single (first := 9892) (initial := 4098) row_count_9893 (by decide)
private theorem atomic_9893_agreement : RowAgreement 9892 9893 := RowAgreement.single (first := 9892) modelrow_agrees_9893
private def atomic_9894 : CountSegment 1 10 13120 9893 9894 4098 4099 := CountSegment.single (first := 9893) (initial := 4098) row_count_9894 (by decide)
private theorem atomic_9894_agreement : RowAgreement 9893 9894 := RowAgreement.single (first := 9893) modelrow_agrees_9894
private def atomic_9895 : CountSegment 1 10 13120 9894 9895 4099 4099 := CountSegment.single (first := 9894) (initial := 4099) row_count_9895 (by decide)
private theorem atomic_9895_agreement : RowAgreement 9894 9895 := RowAgreement.single (first := 9894) modelrow_agrees_9895
private def atomic_9896 : CountSegment 1 10 13120 9895 9896 4099 4099 := CountSegment.single (first := 9895) (initial := 4099) row_count_9896 (by decide)
private theorem atomic_9896_agreement : RowAgreement 9895 9896 := RowAgreement.single (first := 9895) modelrow_agrees_9896
private def atomic_9897 : CountSegment 1 10 13120 9896 9897 4099 4099 := CountSegment.single (first := 9896) (initial := 4099) row_count_9897 (by decide)
private theorem atomic_9897_agreement : RowAgreement 9896 9897 := RowAgreement.single (first := 9896) modelrow_agrees_9897
private def atomic_9898 : CountSegment 1 10 13120 9897 9898 4099 4100 := CountSegment.single (first := 9897) (initial := 4099) row_count_9898 (by decide)
private theorem atomic_9898_agreement : RowAgreement 9897 9898 := RowAgreement.single (first := 9897) modelrow_agrees_9898
private def atomic_9899 : CountSegment 1 10 13120 9898 9899 4100 4101 := CountSegment.single (first := 9898) (initial := 4100) row_count_9899 (by decide)
private theorem atomic_9899_agreement : RowAgreement 9898 9899 := RowAgreement.single (first := 9898) modelrow_agrees_9899
private def atomic_9900 : CountSegment 1 10 13120 9899 9900 4101 4102 := CountSegment.single (first := 9899) (initial := 4101) row_count_9900 (by decide)
private theorem atomic_9900_agreement : RowAgreement 9899 9900 := RowAgreement.single (first := 9899) modelrow_agrees_9900
private def atomic_9901 : CountSegment 1 10 13120 9900 9901 4102 4102 := CountSegment.single (first := 9900) (initial := 4102) row_count_9901 (by decide)
private theorem atomic_9901_agreement : RowAgreement 9900 9901 := RowAgreement.single (first := 9900) modelrow_agrees_9901
private def atomic_9902 : CountSegment 1 10 13120 9901 9902 4102 4102 := CountSegment.single (first := 9901) (initial := 4102) row_count_9902 (by decide)
private theorem atomic_9902_agreement : RowAgreement 9901 9902 := RowAgreement.single (first := 9901) modelrow_agrees_9902
private def atomic_9903 : CountSegment 1 10 13120 9902 9903 4102 4102 := CountSegment.single (first := 9902) (initial := 4102) row_count_9903 (by decide)
private theorem atomic_9903_agreement : RowAgreement 9902 9903 := RowAgreement.single (first := 9902) modelrow_agrees_9903
private def atomic_9904 : CountSegment 1 10 13120 9903 9904 4102 4102 := CountSegment.single (first := 9903) (initial := 4102) row_count_9904 (by decide)
private theorem atomic_9904_agreement : RowAgreement 9903 9904 := RowAgreement.single (first := 9903) modelrow_agrees_9904
private def atomic_9905 : CountSegment 1 10 13120 9904 9905 4102 4102 := CountSegment.single (first := 9904) (initial := 4102) row_count_9905 (by decide)
private theorem atomic_9905_agreement : RowAgreement 9904 9905 := RowAgreement.single (first := 9904) modelrow_agrees_9905
private def atomic_9906 : CountSegment 1 10 13120 9905 9906 4102 4102 := CountSegment.single (first := 9905) (initial := 4102) row_count_9906 (by decide)
private theorem atomic_9906_agreement : RowAgreement 9905 9906 := RowAgreement.single (first := 9905) modelrow_agrees_9906
private def atomic_9907 : CountSegment 1 10 13120 9906 9907 4102 4102 := CountSegment.single (first := 9906) (initial := 4102) row_count_9907 (by decide)
private theorem atomic_9907_agreement : RowAgreement 9906 9907 := RowAgreement.single (first := 9906) modelrow_agrees_9907
private def atomic_9908 : CountSegment 1 10 13120 9907 9908 4102 4102 := CountSegment.single (first := 9907) (initial := 4102) row_count_9908 (by decide)
private theorem atomic_9908_agreement : RowAgreement 9907 9908 := RowAgreement.single (first := 9907) modelrow_agrees_9908
private def atomic_9909 : CountSegment 1 10 13120 9908 9909 4102 4103 := CountSegment.single (first := 9908) (initial := 4102) row_count_9909 (by decide)
private theorem atomic_9909_agreement : RowAgreement 9908 9909 := RowAgreement.single (first := 9908) modelrow_agrees_9909
private def atomic_9910 : CountSegment 1 10 13120 9909 9910 4103 4103 := CountSegment.single (first := 9909) (initial := 4103) row_count_9910 (by decide)
private theorem atomic_9910_agreement : RowAgreement 9909 9910 := RowAgreement.single (first := 9909) modelrow_agrees_9910
private def atomic_9911 : CountSegment 1 10 13120 9910 9911 4103 4103 := CountSegment.single (first := 9910) (initial := 4103) row_count_9911 (by decide)
private theorem atomic_9911_agreement : RowAgreement 9910 9911 := RowAgreement.single (first := 9910) modelrow_agrees_9911
private def atomic_9912 : CountSegment 1 10 13120 9911 9912 4103 4103 := CountSegment.single (first := 9911) (initial := 4103) row_count_9912 (by decide)
private theorem atomic_9912_agreement : RowAgreement 9911 9912 := RowAgreement.single (first := 9911) modelrow_agrees_9912
private def atomic_9913 : CountSegment 1 10 13120 9912 9913 4103 4103 := CountSegment.single (first := 9912) (initial := 4103) row_count_9913 (by decide)
private theorem atomic_9913_agreement : RowAgreement 9912 9913 := RowAgreement.single (first := 9912) modelrow_agrees_9913
private def atomic_9914 : CountSegment 1 10 13120 9913 9914 4103 4103 := CountSegment.single (first := 9913) (initial := 4103) row_count_9914 (by decide)
private theorem atomic_9914_agreement : RowAgreement 9913 9914 := RowAgreement.single (first := 9913) modelrow_agrees_9914
private def atomic_9915 : CountSegment 1 10 13120 9914 9915 4103 4104 := CountSegment.single (first := 9914) (initial := 4103) row_count_9915 (by decide)
private theorem atomic_9915_agreement : RowAgreement 9914 9915 := RowAgreement.single (first := 9914) modelrow_agrees_9915
private def atomic_9916 : CountSegment 1 10 13120 9915 9916 4104 4104 := CountSegment.single (first := 9915) (initial := 4104) row_count_9916 (by decide)
private theorem atomic_9916_agreement : RowAgreement 9915 9916 := RowAgreement.single (first := 9915) modelrow_agrees_9916
private def atomic_9917 : CountSegment 1 10 13120 9916 9917 4104 4104 := CountSegment.single (first := 9916) (initial := 4104) row_count_9917 (by decide)
private theorem atomic_9917_agreement : RowAgreement 9916 9917 := RowAgreement.single (first := 9916) modelrow_agrees_9917
private def atomic_9918 : CountSegment 1 10 13120 9917 9918 4104 4105 := CountSegment.single (first := 9917) (initial := 4104) row_count_9918 (by decide)
private theorem atomic_9918_agreement : RowAgreement 9917 9918 := RowAgreement.single (first := 9917) modelrow_agrees_9918
private def atomic_9919 : CountSegment 1 10 13120 9918 9919 4105 4105 := CountSegment.single (first := 9918) (initial := 4105) row_count_9919 (by decide)
private theorem atomic_9919_agreement : RowAgreement 9918 9919 := RowAgreement.single (first := 9918) modelrow_agrees_9919
private def atomic_9920 : CountSegment 1 10 13120 9919 9920 4105 4105 := CountSegment.single (first := 9919) (initial := 4105) row_count_9920 (by decide)
private theorem atomic_9920_agreement : RowAgreement 9919 9920 := RowAgreement.single (first := 9919) modelrow_agrees_9920
private def atomic_9921 : CountSegment 1 10 13120 9920 9921 4105 4105 := CountSegment.single (first := 9920) (initial := 4105) row_count_9921 (by decide)
private theorem atomic_9921_agreement : RowAgreement 9920 9921 := RowAgreement.single (first := 9920) modelrow_agrees_9921
private def atomic_9922 : CountSegment 1 10 13120 9921 9922 4105 4106 := CountSegment.single (first := 9921) (initial := 4105) row_count_9922 (by decide)
private theorem atomic_9922_agreement : RowAgreement 9921 9922 := RowAgreement.single (first := 9921) modelrow_agrees_9922
private def atomic_9923 : CountSegment 1 10 13120 9922 9923 4106 4106 := CountSegment.single (first := 9922) (initial := 4106) row_count_9923 (by decide)
private theorem atomic_9923_agreement : RowAgreement 9922 9923 := RowAgreement.single (first := 9922) modelrow_agrees_9923
private def atomic_9924 : CountSegment 1 10 13120 9923 9924 4106 4106 := CountSegment.single (first := 9923) (initial := 4106) row_count_9924 (by decide)
private theorem atomic_9924_agreement : RowAgreement 9923 9924 := RowAgreement.single (first := 9923) modelrow_agrees_9924
private def atomic_9925 : CountSegment 1 10 13120 9924 9925 4106 4107 := CountSegment.single (first := 9924) (initial := 4106) row_count_9925 (by decide)
private theorem atomic_9925_agreement : RowAgreement 9924 9925 := RowAgreement.single (first := 9924) modelrow_agrees_9925
private def atomic_9926 : CountSegment 1 10 13120 9925 9926 4107 4108 := CountSegment.single (first := 9925) (initial := 4107) row_count_9926 (by decide)
private theorem atomic_9926_agreement : RowAgreement 9925 9926 := RowAgreement.single (first := 9925) modelrow_agrees_9926
private def atomic_9927 : CountSegment 1 10 13120 9926 9927 4108 4108 := CountSegment.single (first := 9926) (initial := 4108) row_count_9927 (by decide)
private theorem atomic_9927_agreement : RowAgreement 9926 9927 := RowAgreement.single (first := 9926) modelrow_agrees_9927
private def atomic_9928 : CountSegment 1 10 13120 9927 9928 4108 4108 := CountSegment.single (first := 9927) (initial := 4108) row_count_9928 (by decide)
private theorem atomic_9928_agreement : RowAgreement 9927 9928 := RowAgreement.single (first := 9927) modelrow_agrees_9928
private def atomic_9929 : CountSegment 1 10 13120 9928 9929 4108 4108 := CountSegment.single (first := 9928) (initial := 4108) row_count_9929 (by decide)
private theorem atomic_9929_agreement : RowAgreement 9928 9929 := RowAgreement.single (first := 9928) modelrow_agrees_9929
private def atomic_9930 : CountSegment 1 10 13120 9929 9930 4108 4108 := CountSegment.single (first := 9929) (initial := 4108) row_count_9930 (by decide)
private theorem atomic_9930_agreement : RowAgreement 9929 9930 := RowAgreement.single (first := 9929) modelrow_agrees_9930
private def atomic_9931 : CountSegment 1 10 13120 9930 9931 4108 4108 := CountSegment.single (first := 9930) (initial := 4108) row_count_9931 (by decide)
private theorem atomic_9931_agreement : RowAgreement 9930 9931 := RowAgreement.single (first := 9930) modelrow_agrees_9931
private def atomic_9932 : CountSegment 1 10 13120 9931 9932 4108 4108 := CountSegment.single (first := 9931) (initial := 4108) row_count_9932 (by decide)
private theorem atomic_9932_agreement : RowAgreement 9931 9932 := RowAgreement.single (first := 9931) modelrow_agrees_9932
private def atomic_9933 : CountSegment 1 10 13120 9932 9933 4108 4108 := CountSegment.single (first := 9932) (initial := 4108) row_count_9933 (by decide)
private theorem atomic_9933_agreement : RowAgreement 9932 9933 := RowAgreement.single (first := 9932) modelrow_agrees_9933
private def atomic_9934 : CountSegment 1 10 13120 9933 9934 4108 4108 := CountSegment.single (first := 9933) (initial := 4108) row_count_9934 (by decide)
private theorem atomic_9934_agreement : RowAgreement 9933 9934 := RowAgreement.single (first := 9933) modelrow_agrees_9934
private def atomic_9935 : CountSegment 1 10 13120 9934 9935 4108 4109 := CountSegment.single (first := 9934) (initial := 4108) row_count_9935 (by decide)
private theorem atomic_9935_agreement : RowAgreement 9934 9935 := RowAgreement.single (first := 9934) modelrow_agrees_9935
private def atomic_9936 : CountSegment 1 10 13120 9935 9936 4109 4109 := CountSegment.single (first := 9935) (initial := 4109) row_count_9936 (by decide)
private theorem atomic_9936_agreement : RowAgreement 9935 9936 := RowAgreement.single (first := 9935) modelrow_agrees_9936
private def atomic_9937 : CountSegment 1 10 13120 9936 9937 4109 4111 := CountSegment.single (first := 9936) (initial := 4109) row_count_9937 (by decide)
private theorem atomic_9937_agreement : RowAgreement 9936 9937 := RowAgreement.single (first := 9936) modelrow_agrees_9937
private def atomic_9938 : CountSegment 1 10 13120 9937 9938 4111 4111 := CountSegment.single (first := 9937) (initial := 4111) row_count_9938 (by decide)
private theorem atomic_9938_agreement : RowAgreement 9937 9938 := RowAgreement.single (first := 9937) modelrow_agrees_9938
private def atomic_9939 : CountSegment 1 10 13120 9938 9939 4111 4112 := CountSegment.single (first := 9938) (initial := 4111) row_count_9939 (by decide)
private theorem atomic_9939_agreement : RowAgreement 9938 9939 := RowAgreement.single (first := 9938) modelrow_agrees_9939
private def atomic_9940 : CountSegment 1 10 13120 9939 9940 4112 4112 := CountSegment.single (first := 9939) (initial := 4112) row_count_9940 (by decide)
private theorem atomic_9940_agreement : RowAgreement 9939 9940 := RowAgreement.single (first := 9939) modelrow_agrees_9940
private def atomic_9941 : CountSegment 1 10 13120 9940 9941 4112 4112 := CountSegment.single (first := 9940) (initial := 4112) row_count_9941 (by decide)
private theorem atomic_9941_agreement : RowAgreement 9940 9941 := RowAgreement.single (first := 9940) modelrow_agrees_9941
private def atomic_9942 : CountSegment 1 10 13120 9941 9942 4112 4112 := CountSegment.single (first := 9941) (initial := 4112) row_count_9942 (by decide)
private theorem atomic_9942_agreement : RowAgreement 9941 9942 := RowAgreement.single (first := 9941) modelrow_agrees_9942
private def atomic_9943 : CountSegment 1 10 13120 9942 9943 4112 4112 := CountSegment.single (first := 9942) (initial := 4112) row_count_9943 (by decide)
private theorem atomic_9943_agreement : RowAgreement 9942 9943 := RowAgreement.single (first := 9942) modelrow_agrees_9943
private def atomic_9944 : CountSegment 1 10 13120 9943 9944 4112 4112 := CountSegment.single (first := 9943) (initial := 4112) row_count_9944 (by decide)
private theorem atomic_9944_agreement : RowAgreement 9943 9944 := RowAgreement.single (first := 9943) modelrow_agrees_9944
private def atomic_9945 : CountSegment 1 10 13120 9944 9945 4112 4112 := CountSegment.single (first := 9944) (initial := 4112) row_count_9945 (by decide)
private theorem atomic_9945_agreement : RowAgreement 9944 9945 := RowAgreement.single (first := 9944) modelrow_agrees_9945
private def atomic_9946 : CountSegment 1 10 13120 9945 9946 4112 4112 := CountSegment.single (first := 9945) (initial := 4112) row_count_9946 (by decide)
private theorem atomic_9946_agreement : RowAgreement 9945 9946 := RowAgreement.single (first := 9945) modelrow_agrees_9946
private def atomic_9947 : CountSegment 1 10 13120 9946 9947 4112 4112 := CountSegment.single (first := 9946) (initial := 4112) row_count_9947 (by decide)
private theorem atomic_9947_agreement : RowAgreement 9946 9947 := RowAgreement.single (first := 9946) modelrow_agrees_9947
private def atomic_9948 : CountSegment 1 10 13120 9947 9948 4112 4113 := CountSegment.single (first := 9947) (initial := 4112) row_count_9948 (by decide)
private theorem atomic_9948_agreement : RowAgreement 9947 9948 := RowAgreement.single (first := 9947) modelrow_agrees_9948
private def atomic_9949 : CountSegment 1 10 13120 9948 9949 4113 4113 := CountSegment.single (first := 9948) (initial := 4113) row_count_9949 (by decide)
private theorem atomic_9949_agreement : RowAgreement 9948 9949 := RowAgreement.single (first := 9948) modelrow_agrees_9949
private def atomic_9950 : CountSegment 1 10 13120 9949 9950 4113 4113 := CountSegment.single (first := 9949) (initial := 4113) row_count_9950 (by decide)
private theorem atomic_9950_agreement : RowAgreement 9949 9950 := RowAgreement.single (first := 9949) modelrow_agrees_9950
private def atomic_9951 : CountSegment 1 10 13120 9950 9951 4113 4113 := CountSegment.single (first := 9950) (initial := 4113) row_count_9951 (by decide)
private theorem atomic_9951_agreement : RowAgreement 9950 9951 := RowAgreement.single (first := 9950) modelrow_agrees_9951
private def atomic_9952 : CountSegment 1 10 13120 9951 9952 4113 4114 := CountSegment.single (first := 9951) (initial := 4113) row_count_9952 (by decide)
private theorem atomic_9952_agreement : RowAgreement 9951 9952 := RowAgreement.single (first := 9951) modelrow_agrees_9952
private def atomic_9953 : CountSegment 1 10 13120 9952 9953 4114 4114 := CountSegment.single (first := 9952) (initial := 4114) row_count_9953 (by decide)
private theorem atomic_9953_agreement : RowAgreement 9952 9953 := RowAgreement.single (first := 9952) modelrow_agrees_9953
private def atomic_9954 : CountSegment 1 10 13120 9953 9954 4114 4115 := CountSegment.single (first := 9953) (initial := 4114) row_count_9954 (by decide)
private theorem atomic_9954_agreement : RowAgreement 9953 9954 := RowAgreement.single (first := 9953) modelrow_agrees_9954
private def atomic_9955 : CountSegment 1 10 13120 9954 9955 4115 4115 := CountSegment.single (first := 9954) (initial := 4115) row_count_9955 (by decide)
private theorem atomic_9955_agreement : RowAgreement 9954 9955 := RowAgreement.single (first := 9954) modelrow_agrees_9955
private def atomic_9956 : CountSegment 1 10 13120 9955 9956 4115 4116 := CountSegment.single (first := 9955) (initial := 4115) row_count_9956 (by decide)
private theorem atomic_9956_agreement : RowAgreement 9955 9956 := RowAgreement.single (first := 9955) modelrow_agrees_9956
private def atomic_9957 : CountSegment 1 10 13120 9956 9957 4116 4117 := CountSegment.single (first := 9956) (initial := 4116) row_count_9957 (by decide)
private theorem atomic_9957_agreement : RowAgreement 9956 9957 := RowAgreement.single (first := 9956) modelrow_agrees_9957
private def atomic_9958 : CountSegment 1 10 13120 9957 9958 4117 4118 := CountSegment.single (first := 9957) (initial := 4117) row_count_9958 (by decide)
private theorem atomic_9958_agreement : RowAgreement 9957 9958 := RowAgreement.single (first := 9957) modelrow_agrees_9958
private def atomic_9959 : CountSegment 1 10 13120 9958 9959 4118 4118 := CountSegment.single (first := 9958) (initial := 4118) row_count_9959 (by decide)
private theorem atomic_9959_agreement : RowAgreement 9958 9959 := RowAgreement.single (first := 9958) modelrow_agrees_9959
private def atomic_9960 : CountSegment 1 10 13120 9959 9960 4118 4118 := CountSegment.single (first := 9959) (initial := 4118) row_count_9960 (by decide)
private theorem atomic_9960_agreement : RowAgreement 9959 9960 := RowAgreement.single (first := 9959) modelrow_agrees_9960
private def atomic_9961 : CountSegment 1 10 13120 9960 9961 4118 4118 := CountSegment.single (first := 9960) (initial := 4118) row_count_9961 (by decide)
private theorem atomic_9961_agreement : RowAgreement 9960 9961 := RowAgreement.single (first := 9960) modelrow_agrees_9961
private def atomic_9962 : CountSegment 1 10 13120 9961 9962 4118 4118 := CountSegment.single (first := 9961) (initial := 4118) row_count_9962 (by decide)
private theorem atomic_9962_agreement : RowAgreement 9961 9962 := RowAgreement.single (first := 9961) modelrow_agrees_9962
private def atomic_9963 : CountSegment 1 10 13120 9962 9963 4118 4119 := CountSegment.single (first := 9962) (initial := 4118) row_count_9963 (by decide)
private theorem atomic_9963_agreement : RowAgreement 9962 9963 := RowAgreement.single (first := 9962) modelrow_agrees_9963
private def atomic_9964 : CountSegment 1 10 13120 9963 9964 4119 4119 := CountSegment.single (first := 9963) (initial := 4119) row_count_9964 (by decide)
private theorem atomic_9964_agreement : RowAgreement 9963 9964 := RowAgreement.single (first := 9963) modelrow_agrees_9964
private def atomic_9965 : CountSegment 1 10 13120 9964 9965 4119 4120 := CountSegment.single (first := 9964) (initial := 4119) row_count_9965 (by decide)
private theorem atomic_9965_agreement : RowAgreement 9964 9965 := RowAgreement.single (first := 9964) modelrow_agrees_9965
private def atomic_9966 : CountSegment 1 10 13120 9965 9966 4120 4120 := CountSegment.single (first := 9965) (initial := 4120) row_count_9966 (by decide)
private theorem atomic_9966_agreement : RowAgreement 9965 9966 := RowAgreement.single (first := 9965) modelrow_agrees_9966
private def atomic_9967 : CountSegment 1 10 13120 9966 9967 4120 4120 := CountSegment.single (first := 9966) (initial := 4120) row_count_9967 (by decide)
private theorem atomic_9967_agreement : RowAgreement 9966 9967 := RowAgreement.single (first := 9966) modelrow_agrees_9967
private def atomic_9968 : CountSegment 1 10 13120 9967 9968 4120 4120 := CountSegment.single (first := 9967) (initial := 4120) row_count_9968 (by decide)
private theorem atomic_9968_agreement : RowAgreement 9967 9968 := RowAgreement.single (first := 9967) modelrow_agrees_9968
private def atomic_9969 : CountSegment 1 10 13120 9968 9969 4120 4120 := CountSegment.single (first := 9968) (initial := 4120) row_count_9969 (by decide)
private theorem atomic_9969_agreement : RowAgreement 9968 9969 := RowAgreement.single (first := 9968) modelrow_agrees_9969
private def atomic_9970 : CountSegment 1 10 13120 9969 9970 4120 4120 := CountSegment.single (first := 9969) (initial := 4120) row_count_9970 (by decide)
private theorem atomic_9970_agreement : RowAgreement 9969 9970 := RowAgreement.single (first := 9969) modelrow_agrees_9970
private def atomic_9971 : CountSegment 1 10 13120 9970 9971 4120 4120 := CountSegment.single (first := 9970) (initial := 4120) row_count_9971 (by decide)
private theorem atomic_9971_agreement : RowAgreement 9970 9971 := RowAgreement.single (first := 9970) modelrow_agrees_9971
private def atomic_9972 : CountSegment 1 10 13120 9971 9972 4120 4122 := CountSegment.single (first := 9971) (initial := 4120) row_count_9972 (by decide)
private theorem atomic_9972_agreement : RowAgreement 9971 9972 := RowAgreement.single (first := 9971) modelrow_agrees_9972
private def atomic_9973 : CountSegment 1 10 13120 9972 9973 4122 4123 := CountSegment.single (first := 9972) (initial := 4122) row_count_9973 (by decide)
private theorem atomic_9973_agreement : RowAgreement 9972 9973 := RowAgreement.single (first := 9972) modelrow_agrees_9973
private def atomic_9974 : CountSegment 1 10 13120 9973 9974 4123 4123 := CountSegment.single (first := 9973) (initial := 4123) row_count_9974 (by decide)
private theorem atomic_9974_agreement : RowAgreement 9973 9974 := RowAgreement.single (first := 9973) modelrow_agrees_9974
private def atomic_9975 : CountSegment 1 10 13120 9974 9975 4123 4124 := CountSegment.single (first := 9974) (initial := 4123) row_count_9975 (by decide)
private theorem atomic_9975_agreement : RowAgreement 9974 9975 := RowAgreement.single (first := 9974) modelrow_agrees_9975
private def atomic_9976 : CountSegment 1 10 13120 9975 9976 4124 4124 := CountSegment.single (first := 9975) (initial := 4124) row_count_9976 (by decide)
private theorem atomic_9976_agreement : RowAgreement 9975 9976 := RowAgreement.single (first := 9975) modelrow_agrees_9976
private def atomic_9977 : CountSegment 1 10 13120 9976 9977 4124 4124 := CountSegment.single (first := 9976) (initial := 4124) row_count_9977 (by decide)
private theorem atomic_9977_agreement : RowAgreement 9976 9977 := RowAgreement.single (first := 9976) modelrow_agrees_9977
private def atomic_9978 : CountSegment 1 10 13120 9977 9978 4124 4126 := CountSegment.single (first := 9977) (initial := 4124) row_count_9978 (by decide)
private theorem atomic_9978_agreement : RowAgreement 9977 9978 := RowAgreement.single (first := 9977) modelrow_agrees_9978
private def atomic_9979 : CountSegment 1 10 13120 9978 9979 4126 4126 := CountSegment.single (first := 9978) (initial := 4126) row_count_9979 (by decide)
private theorem atomic_9979_agreement : RowAgreement 9978 9979 := RowAgreement.single (first := 9978) modelrow_agrees_9979
private def atomic_9980 : CountSegment 1 10 13120 9979 9980 4126 4126 := CountSegment.single (first := 9979) (initial := 4126) row_count_9980 (by decide)
private theorem atomic_9980_agreement : RowAgreement 9979 9980 := RowAgreement.single (first := 9979) modelrow_agrees_9980
private def atomic_9981 : CountSegment 1 10 13120 9980 9981 4126 4126 := CountSegment.single (first := 9980) (initial := 4126) row_count_9981 (by decide)
private theorem atomic_9981_agreement : RowAgreement 9980 9981 := RowAgreement.single (first := 9980) modelrow_agrees_9981
private def atomic_9982 : CountSegment 1 10 13120 9981 9982 4126 4126 := CountSegment.single (first := 9981) (initial := 4126) row_count_9982 (by decide)
private theorem atomic_9982_agreement : RowAgreement 9981 9982 := RowAgreement.single (first := 9981) modelrow_agrees_9982
private def atomic_9983 : CountSegment 1 10 13120 9982 9983 4126 4128 := CountSegment.single (first := 9982) (initial := 4126) row_count_9983 (by decide)
private theorem atomic_9983_agreement : RowAgreement 9982 9983 := RowAgreement.single (first := 9982) modelrow_agrees_9983
private def atomic_9984 : CountSegment 1 10 13120 9983 9984 4128 4129 := CountSegment.single (first := 9983) (initial := 4128) row_count_9984 (by decide)
private theorem atomic_9984_agreement : RowAgreement 9983 9984 := RowAgreement.single (first := 9983) modelrow_agrees_9984
private def within_077_000 : CountSegment 1 10 13120 9856 9858 4085 4086 := CountSegment.join atomic_9857 atomic_9858
private theorem within_077_000_agreement : RowAgreement 9856 9858 := RowAgreement.join atomic_9857_agreement atomic_9858_agreement
private def within_077_001 : CountSegment 1 10 13120 9858 9860 4086 4086 := CountSegment.join atomic_9859 atomic_9860
private theorem within_077_001_agreement : RowAgreement 9858 9860 := RowAgreement.join atomic_9859_agreement atomic_9860_agreement
private def within_077_002 : CountSegment 1 10 13120 9856 9860 4085 4086 := CountSegment.join within_077_000 within_077_001
private theorem within_077_002_agreement : RowAgreement 9856 9860 := RowAgreement.join within_077_000_agreement within_077_001_agreement
private def within_077_003 : CountSegment 1 10 13120 9860 9862 4086 4087 := CountSegment.join atomic_9861 atomic_9862
private theorem within_077_003_agreement : RowAgreement 9860 9862 := RowAgreement.join atomic_9861_agreement atomic_9862_agreement
private def within_077_004 : CountSegment 1 10 13120 9862 9864 4087 4088 := CountSegment.join atomic_9863 atomic_9864
private theorem within_077_004_agreement : RowAgreement 9862 9864 := RowAgreement.join atomic_9863_agreement atomic_9864_agreement
private def within_077_005 : CountSegment 1 10 13120 9860 9864 4086 4088 := CountSegment.join within_077_003 within_077_004
private theorem within_077_005_agreement : RowAgreement 9860 9864 := RowAgreement.join within_077_003_agreement within_077_004_agreement
private def within_077_006 : CountSegment 1 10 13120 9856 9864 4085 4088 := CountSegment.join within_077_002 within_077_005
private theorem within_077_006_agreement : RowAgreement 9856 9864 := RowAgreement.join within_077_002_agreement within_077_005_agreement
private def within_077_007 : CountSegment 1 10 13120 9864 9866 4088 4088 := CountSegment.join atomic_9865 atomic_9866
private theorem within_077_007_agreement : RowAgreement 9864 9866 := RowAgreement.join atomic_9865_agreement atomic_9866_agreement
private def within_077_008 : CountSegment 1 10 13120 9866 9868 4088 4090 := CountSegment.join atomic_9867 atomic_9868
private theorem within_077_008_agreement : RowAgreement 9866 9868 := RowAgreement.join atomic_9867_agreement atomic_9868_agreement
private def within_077_009 : CountSegment 1 10 13120 9864 9868 4088 4090 := CountSegment.join within_077_007 within_077_008
private theorem within_077_009_agreement : RowAgreement 9864 9868 := RowAgreement.join within_077_007_agreement within_077_008_agreement
private def within_077_010 : CountSegment 1 10 13120 9868 9870 4090 4091 := CountSegment.join atomic_9869 atomic_9870
private theorem within_077_010_agreement : RowAgreement 9868 9870 := RowAgreement.join atomic_9869_agreement atomic_9870_agreement
private def within_077_011 : CountSegment 1 10 13120 9870 9872 4091 4091 := CountSegment.join atomic_9871 atomic_9872
private theorem within_077_011_agreement : RowAgreement 9870 9872 := RowAgreement.join atomic_9871_agreement atomic_9872_agreement
private def within_077_012 : CountSegment 1 10 13120 9868 9872 4090 4091 := CountSegment.join within_077_010 within_077_011
private theorem within_077_012_agreement : RowAgreement 9868 9872 := RowAgreement.join within_077_010_agreement within_077_011_agreement
private def within_077_013 : CountSegment 1 10 13120 9864 9872 4088 4091 := CountSegment.join within_077_009 within_077_012
private theorem within_077_013_agreement : RowAgreement 9864 9872 := RowAgreement.join within_077_009_agreement within_077_012_agreement
private def within_077_014 : CountSegment 1 10 13120 9856 9872 4085 4091 := CountSegment.join within_077_006 within_077_013
private theorem within_077_014_agreement : RowAgreement 9856 9872 := RowAgreement.join within_077_006_agreement within_077_013_agreement
private def within_077_015 : CountSegment 1 10 13120 9872 9874 4091 4092 := CountSegment.join atomic_9873 atomic_9874
private theorem within_077_015_agreement : RowAgreement 9872 9874 := RowAgreement.join atomic_9873_agreement atomic_9874_agreement
private def within_077_016 : CountSegment 1 10 13120 9874 9876 4092 4092 := CountSegment.join atomic_9875 atomic_9876
private theorem within_077_016_agreement : RowAgreement 9874 9876 := RowAgreement.join atomic_9875_agreement atomic_9876_agreement
private def within_077_017 : CountSegment 1 10 13120 9872 9876 4091 4092 := CountSegment.join within_077_015 within_077_016
private theorem within_077_017_agreement : RowAgreement 9872 9876 := RowAgreement.join within_077_015_agreement within_077_016_agreement
private def within_077_018 : CountSegment 1 10 13120 9876 9878 4092 4092 := CountSegment.join atomic_9877 atomic_9878
private theorem within_077_018_agreement : RowAgreement 9876 9878 := RowAgreement.join atomic_9877_agreement atomic_9878_agreement
private def within_077_019 : CountSegment 1 10 13120 9878 9880 4092 4094 := CountSegment.join atomic_9879 atomic_9880
private theorem within_077_019_agreement : RowAgreement 9878 9880 := RowAgreement.join atomic_9879_agreement atomic_9880_agreement
private def within_077_020 : CountSegment 1 10 13120 9876 9880 4092 4094 := CountSegment.join within_077_018 within_077_019
private theorem within_077_020_agreement : RowAgreement 9876 9880 := RowAgreement.join within_077_018_agreement within_077_019_agreement
private def within_077_021 : CountSegment 1 10 13120 9872 9880 4091 4094 := CountSegment.join within_077_017 within_077_020
private theorem within_077_021_agreement : RowAgreement 9872 9880 := RowAgreement.join within_077_017_agreement within_077_020_agreement
private def within_077_022 : CountSegment 1 10 13120 9880 9882 4094 4096 := CountSegment.join atomic_9881 atomic_9882
private theorem within_077_022_agreement : RowAgreement 9880 9882 := RowAgreement.join atomic_9881_agreement atomic_9882_agreement
private def within_077_023 : CountSegment 1 10 13120 9882 9884 4096 4096 := CountSegment.join atomic_9883 atomic_9884
private theorem within_077_023_agreement : RowAgreement 9882 9884 := RowAgreement.join atomic_9883_agreement atomic_9884_agreement
private def within_077_024 : CountSegment 1 10 13120 9880 9884 4094 4096 := CountSegment.join within_077_022 within_077_023
private theorem within_077_024_agreement : RowAgreement 9880 9884 := RowAgreement.join within_077_022_agreement within_077_023_agreement
private def within_077_025 : CountSegment 1 10 13120 9884 9886 4096 4096 := CountSegment.join atomic_9885 atomic_9886
private theorem within_077_025_agreement : RowAgreement 9884 9886 := RowAgreement.join atomic_9885_agreement atomic_9886_agreement
private def within_077_026 : CountSegment 1 10 13120 9886 9888 4096 4097 := CountSegment.join atomic_9887 atomic_9888
private theorem within_077_026_agreement : RowAgreement 9886 9888 := RowAgreement.join atomic_9887_agreement atomic_9888_agreement
private def within_077_027 : CountSegment 1 10 13120 9884 9888 4096 4097 := CountSegment.join within_077_025 within_077_026
private theorem within_077_027_agreement : RowAgreement 9884 9888 := RowAgreement.join within_077_025_agreement within_077_026_agreement
private def within_077_028 : CountSegment 1 10 13120 9880 9888 4094 4097 := CountSegment.join within_077_024 within_077_027
private theorem within_077_028_agreement : RowAgreement 9880 9888 := RowAgreement.join within_077_024_agreement within_077_027_agreement
private def within_077_029 : CountSegment 1 10 13120 9872 9888 4091 4097 := CountSegment.join within_077_021 within_077_028
private theorem within_077_029_agreement : RowAgreement 9872 9888 := RowAgreement.join within_077_021_agreement within_077_028_agreement
private def within_077_030 : CountSegment 1 10 13120 9856 9888 4085 4097 := CountSegment.join within_077_014 within_077_029
private theorem within_077_030_agreement : RowAgreement 9856 9888 := RowAgreement.join within_077_014_agreement within_077_029_agreement
private def within_077_031 : CountSegment 1 10 13120 9888 9890 4097 4098 := CountSegment.join atomic_9889 atomic_9890
private theorem within_077_031_agreement : RowAgreement 9888 9890 := RowAgreement.join atomic_9889_agreement atomic_9890_agreement
private def within_077_032 : CountSegment 1 10 13120 9890 9892 4098 4098 := CountSegment.join atomic_9891 atomic_9892
private theorem within_077_032_agreement : RowAgreement 9890 9892 := RowAgreement.join atomic_9891_agreement atomic_9892_agreement
private def within_077_033 : CountSegment 1 10 13120 9888 9892 4097 4098 := CountSegment.join within_077_031 within_077_032
private theorem within_077_033_agreement : RowAgreement 9888 9892 := RowAgreement.join within_077_031_agreement within_077_032_agreement
private def within_077_034 : CountSegment 1 10 13120 9892 9894 4098 4099 := CountSegment.join atomic_9893 atomic_9894
private theorem within_077_034_agreement : RowAgreement 9892 9894 := RowAgreement.join atomic_9893_agreement atomic_9894_agreement
private def within_077_035 : CountSegment 1 10 13120 9894 9896 4099 4099 := CountSegment.join atomic_9895 atomic_9896
private theorem within_077_035_agreement : RowAgreement 9894 9896 := RowAgreement.join atomic_9895_agreement atomic_9896_agreement
private def within_077_036 : CountSegment 1 10 13120 9892 9896 4098 4099 := CountSegment.join within_077_034 within_077_035
private theorem within_077_036_agreement : RowAgreement 9892 9896 := RowAgreement.join within_077_034_agreement within_077_035_agreement
private def within_077_037 : CountSegment 1 10 13120 9888 9896 4097 4099 := CountSegment.join within_077_033 within_077_036
private theorem within_077_037_agreement : RowAgreement 9888 9896 := RowAgreement.join within_077_033_agreement within_077_036_agreement
private def within_077_038 : CountSegment 1 10 13120 9896 9898 4099 4100 := CountSegment.join atomic_9897 atomic_9898
private theorem within_077_038_agreement : RowAgreement 9896 9898 := RowAgreement.join atomic_9897_agreement atomic_9898_agreement
private def within_077_039 : CountSegment 1 10 13120 9898 9900 4100 4102 := CountSegment.join atomic_9899 atomic_9900
private theorem within_077_039_agreement : RowAgreement 9898 9900 := RowAgreement.join atomic_9899_agreement atomic_9900_agreement
private def within_077_040 : CountSegment 1 10 13120 9896 9900 4099 4102 := CountSegment.join within_077_038 within_077_039
private theorem within_077_040_agreement : RowAgreement 9896 9900 := RowAgreement.join within_077_038_agreement within_077_039_agreement
private def within_077_041 : CountSegment 1 10 13120 9900 9902 4102 4102 := CountSegment.join atomic_9901 atomic_9902
private theorem within_077_041_agreement : RowAgreement 9900 9902 := RowAgreement.join atomic_9901_agreement atomic_9902_agreement
private def within_077_042 : CountSegment 1 10 13120 9902 9904 4102 4102 := CountSegment.join atomic_9903 atomic_9904
private theorem within_077_042_agreement : RowAgreement 9902 9904 := RowAgreement.join atomic_9903_agreement atomic_9904_agreement
private def within_077_043 : CountSegment 1 10 13120 9900 9904 4102 4102 := CountSegment.join within_077_041 within_077_042
private theorem within_077_043_agreement : RowAgreement 9900 9904 := RowAgreement.join within_077_041_agreement within_077_042_agreement
private def within_077_044 : CountSegment 1 10 13120 9896 9904 4099 4102 := CountSegment.join within_077_040 within_077_043
private theorem within_077_044_agreement : RowAgreement 9896 9904 := RowAgreement.join within_077_040_agreement within_077_043_agreement
private def within_077_045 : CountSegment 1 10 13120 9888 9904 4097 4102 := CountSegment.join within_077_037 within_077_044
private theorem within_077_045_agreement : RowAgreement 9888 9904 := RowAgreement.join within_077_037_agreement within_077_044_agreement
private def within_077_046 : CountSegment 1 10 13120 9904 9906 4102 4102 := CountSegment.join atomic_9905 atomic_9906
private theorem within_077_046_agreement : RowAgreement 9904 9906 := RowAgreement.join atomic_9905_agreement atomic_9906_agreement
private def within_077_047 : CountSegment 1 10 13120 9906 9908 4102 4102 := CountSegment.join atomic_9907 atomic_9908
private theorem within_077_047_agreement : RowAgreement 9906 9908 := RowAgreement.join atomic_9907_agreement atomic_9908_agreement
private def within_077_048 : CountSegment 1 10 13120 9904 9908 4102 4102 := CountSegment.join within_077_046 within_077_047
private theorem within_077_048_agreement : RowAgreement 9904 9908 := RowAgreement.join within_077_046_agreement within_077_047_agreement
private def within_077_049 : CountSegment 1 10 13120 9908 9910 4102 4103 := CountSegment.join atomic_9909 atomic_9910
private theorem within_077_049_agreement : RowAgreement 9908 9910 := RowAgreement.join atomic_9909_agreement atomic_9910_agreement
private def within_077_050 : CountSegment 1 10 13120 9910 9912 4103 4103 := CountSegment.join atomic_9911 atomic_9912
private theorem within_077_050_agreement : RowAgreement 9910 9912 := RowAgreement.join atomic_9911_agreement atomic_9912_agreement
private def within_077_051 : CountSegment 1 10 13120 9908 9912 4102 4103 := CountSegment.join within_077_049 within_077_050
private theorem within_077_051_agreement : RowAgreement 9908 9912 := RowAgreement.join within_077_049_agreement within_077_050_agreement
private def within_077_052 : CountSegment 1 10 13120 9904 9912 4102 4103 := CountSegment.join within_077_048 within_077_051
private theorem within_077_052_agreement : RowAgreement 9904 9912 := RowAgreement.join within_077_048_agreement within_077_051_agreement
private def within_077_053 : CountSegment 1 10 13120 9912 9914 4103 4103 := CountSegment.join atomic_9913 atomic_9914
private theorem within_077_053_agreement : RowAgreement 9912 9914 := RowAgreement.join atomic_9913_agreement atomic_9914_agreement
private def within_077_054 : CountSegment 1 10 13120 9914 9916 4103 4104 := CountSegment.join atomic_9915 atomic_9916
private theorem within_077_054_agreement : RowAgreement 9914 9916 := RowAgreement.join atomic_9915_agreement atomic_9916_agreement
private def within_077_055 : CountSegment 1 10 13120 9912 9916 4103 4104 := CountSegment.join within_077_053 within_077_054
private theorem within_077_055_agreement : RowAgreement 9912 9916 := RowAgreement.join within_077_053_agreement within_077_054_agreement
private def within_077_056 : CountSegment 1 10 13120 9916 9918 4104 4105 := CountSegment.join atomic_9917 atomic_9918
private theorem within_077_056_agreement : RowAgreement 9916 9918 := RowAgreement.join atomic_9917_agreement atomic_9918_agreement
private def within_077_057 : CountSegment 1 10 13120 9918 9920 4105 4105 := CountSegment.join atomic_9919 atomic_9920
private theorem within_077_057_agreement : RowAgreement 9918 9920 := RowAgreement.join atomic_9919_agreement atomic_9920_agreement
private def within_077_058 : CountSegment 1 10 13120 9916 9920 4104 4105 := CountSegment.join within_077_056 within_077_057
private theorem within_077_058_agreement : RowAgreement 9916 9920 := RowAgreement.join within_077_056_agreement within_077_057_agreement
private def within_077_059 : CountSegment 1 10 13120 9912 9920 4103 4105 := CountSegment.join within_077_055 within_077_058
private theorem within_077_059_agreement : RowAgreement 9912 9920 := RowAgreement.join within_077_055_agreement within_077_058_agreement
private def within_077_060 : CountSegment 1 10 13120 9904 9920 4102 4105 := CountSegment.join within_077_052 within_077_059
private theorem within_077_060_agreement : RowAgreement 9904 9920 := RowAgreement.join within_077_052_agreement within_077_059_agreement
private def within_077_061 : CountSegment 1 10 13120 9888 9920 4097 4105 := CountSegment.join within_077_045 within_077_060
private theorem within_077_061_agreement : RowAgreement 9888 9920 := RowAgreement.join within_077_045_agreement within_077_060_agreement
private def within_077_062 : CountSegment 1 10 13120 9856 9920 4085 4105 := CountSegment.join within_077_030 within_077_061
private theorem within_077_062_agreement : RowAgreement 9856 9920 := RowAgreement.join within_077_030_agreement within_077_061_agreement
private def within_077_063 : CountSegment 1 10 13120 9920 9922 4105 4106 := CountSegment.join atomic_9921 atomic_9922
private theorem within_077_063_agreement : RowAgreement 9920 9922 := RowAgreement.join atomic_9921_agreement atomic_9922_agreement
private def within_077_064 : CountSegment 1 10 13120 9922 9924 4106 4106 := CountSegment.join atomic_9923 atomic_9924
private theorem within_077_064_agreement : RowAgreement 9922 9924 := RowAgreement.join atomic_9923_agreement atomic_9924_agreement
private def within_077_065 : CountSegment 1 10 13120 9920 9924 4105 4106 := CountSegment.join within_077_063 within_077_064
private theorem within_077_065_agreement : RowAgreement 9920 9924 := RowAgreement.join within_077_063_agreement within_077_064_agreement
private def within_077_066 : CountSegment 1 10 13120 9924 9926 4106 4108 := CountSegment.join atomic_9925 atomic_9926
private theorem within_077_066_agreement : RowAgreement 9924 9926 := RowAgreement.join atomic_9925_agreement atomic_9926_agreement
private def within_077_067 : CountSegment 1 10 13120 9926 9928 4108 4108 := CountSegment.join atomic_9927 atomic_9928
private theorem within_077_067_agreement : RowAgreement 9926 9928 := RowAgreement.join atomic_9927_agreement atomic_9928_agreement
private def within_077_068 : CountSegment 1 10 13120 9924 9928 4106 4108 := CountSegment.join within_077_066 within_077_067
private theorem within_077_068_agreement : RowAgreement 9924 9928 := RowAgreement.join within_077_066_agreement within_077_067_agreement
private def within_077_069 : CountSegment 1 10 13120 9920 9928 4105 4108 := CountSegment.join within_077_065 within_077_068
private theorem within_077_069_agreement : RowAgreement 9920 9928 := RowAgreement.join within_077_065_agreement within_077_068_agreement
private def within_077_070 : CountSegment 1 10 13120 9928 9930 4108 4108 := CountSegment.join atomic_9929 atomic_9930
private theorem within_077_070_agreement : RowAgreement 9928 9930 := RowAgreement.join atomic_9929_agreement atomic_9930_agreement
private def within_077_071 : CountSegment 1 10 13120 9930 9932 4108 4108 := CountSegment.join atomic_9931 atomic_9932
private theorem within_077_071_agreement : RowAgreement 9930 9932 := RowAgreement.join atomic_9931_agreement atomic_9932_agreement
private def within_077_072 : CountSegment 1 10 13120 9928 9932 4108 4108 := CountSegment.join within_077_070 within_077_071
private theorem within_077_072_agreement : RowAgreement 9928 9932 := RowAgreement.join within_077_070_agreement within_077_071_agreement
private def within_077_073 : CountSegment 1 10 13120 9932 9934 4108 4108 := CountSegment.join atomic_9933 atomic_9934
private theorem within_077_073_agreement : RowAgreement 9932 9934 := RowAgreement.join atomic_9933_agreement atomic_9934_agreement
private def within_077_074 : CountSegment 1 10 13120 9934 9936 4108 4109 := CountSegment.join atomic_9935 atomic_9936
private theorem within_077_074_agreement : RowAgreement 9934 9936 := RowAgreement.join atomic_9935_agreement atomic_9936_agreement
private def within_077_075 : CountSegment 1 10 13120 9932 9936 4108 4109 := CountSegment.join within_077_073 within_077_074
private theorem within_077_075_agreement : RowAgreement 9932 9936 := RowAgreement.join within_077_073_agreement within_077_074_agreement
private def within_077_076 : CountSegment 1 10 13120 9928 9936 4108 4109 := CountSegment.join within_077_072 within_077_075
private theorem within_077_076_agreement : RowAgreement 9928 9936 := RowAgreement.join within_077_072_agreement within_077_075_agreement
private def within_077_077 : CountSegment 1 10 13120 9920 9936 4105 4109 := CountSegment.join within_077_069 within_077_076
private theorem within_077_077_agreement : RowAgreement 9920 9936 := RowAgreement.join within_077_069_agreement within_077_076_agreement
private def within_077_078 : CountSegment 1 10 13120 9936 9938 4109 4111 := CountSegment.join atomic_9937 atomic_9938
private theorem within_077_078_agreement : RowAgreement 9936 9938 := RowAgreement.join atomic_9937_agreement atomic_9938_agreement
private def within_077_079 : CountSegment 1 10 13120 9938 9940 4111 4112 := CountSegment.join atomic_9939 atomic_9940
private theorem within_077_079_agreement : RowAgreement 9938 9940 := RowAgreement.join atomic_9939_agreement atomic_9940_agreement
private def within_077_080 : CountSegment 1 10 13120 9936 9940 4109 4112 := CountSegment.join within_077_078 within_077_079
private theorem within_077_080_agreement : RowAgreement 9936 9940 := RowAgreement.join within_077_078_agreement within_077_079_agreement
private def within_077_081 : CountSegment 1 10 13120 9940 9942 4112 4112 := CountSegment.join atomic_9941 atomic_9942
private theorem within_077_081_agreement : RowAgreement 9940 9942 := RowAgreement.join atomic_9941_agreement atomic_9942_agreement
private def within_077_082 : CountSegment 1 10 13120 9942 9944 4112 4112 := CountSegment.join atomic_9943 atomic_9944
private theorem within_077_082_agreement : RowAgreement 9942 9944 := RowAgreement.join atomic_9943_agreement atomic_9944_agreement
private def within_077_083 : CountSegment 1 10 13120 9940 9944 4112 4112 := CountSegment.join within_077_081 within_077_082
private theorem within_077_083_agreement : RowAgreement 9940 9944 := RowAgreement.join within_077_081_agreement within_077_082_agreement
private def within_077_084 : CountSegment 1 10 13120 9936 9944 4109 4112 := CountSegment.join within_077_080 within_077_083
private theorem within_077_084_agreement : RowAgreement 9936 9944 := RowAgreement.join within_077_080_agreement within_077_083_agreement
private def within_077_085 : CountSegment 1 10 13120 9944 9946 4112 4112 := CountSegment.join atomic_9945 atomic_9946
private theorem within_077_085_agreement : RowAgreement 9944 9946 := RowAgreement.join atomic_9945_agreement atomic_9946_agreement
private def within_077_086 : CountSegment 1 10 13120 9946 9948 4112 4113 := CountSegment.join atomic_9947 atomic_9948
private theorem within_077_086_agreement : RowAgreement 9946 9948 := RowAgreement.join atomic_9947_agreement atomic_9948_agreement
private def within_077_087 : CountSegment 1 10 13120 9944 9948 4112 4113 := CountSegment.join within_077_085 within_077_086
private theorem within_077_087_agreement : RowAgreement 9944 9948 := RowAgreement.join within_077_085_agreement within_077_086_agreement
private def within_077_088 : CountSegment 1 10 13120 9948 9950 4113 4113 := CountSegment.join atomic_9949 atomic_9950
private theorem within_077_088_agreement : RowAgreement 9948 9950 := RowAgreement.join atomic_9949_agreement atomic_9950_agreement
private def within_077_089 : CountSegment 1 10 13120 9950 9952 4113 4114 := CountSegment.join atomic_9951 atomic_9952
private theorem within_077_089_agreement : RowAgreement 9950 9952 := RowAgreement.join atomic_9951_agreement atomic_9952_agreement
private def within_077_090 : CountSegment 1 10 13120 9948 9952 4113 4114 := CountSegment.join within_077_088 within_077_089
private theorem within_077_090_agreement : RowAgreement 9948 9952 := RowAgreement.join within_077_088_agreement within_077_089_agreement
private def within_077_091 : CountSegment 1 10 13120 9944 9952 4112 4114 := CountSegment.join within_077_087 within_077_090
private theorem within_077_091_agreement : RowAgreement 9944 9952 := RowAgreement.join within_077_087_agreement within_077_090_agreement
private def within_077_092 : CountSegment 1 10 13120 9936 9952 4109 4114 := CountSegment.join within_077_084 within_077_091
private theorem within_077_092_agreement : RowAgreement 9936 9952 := RowAgreement.join within_077_084_agreement within_077_091_agreement
private def within_077_093 : CountSegment 1 10 13120 9920 9952 4105 4114 := CountSegment.join within_077_077 within_077_092
private theorem within_077_093_agreement : RowAgreement 9920 9952 := RowAgreement.join within_077_077_agreement within_077_092_agreement
private def within_077_094 : CountSegment 1 10 13120 9952 9954 4114 4115 := CountSegment.join atomic_9953 atomic_9954
private theorem within_077_094_agreement : RowAgreement 9952 9954 := RowAgreement.join atomic_9953_agreement atomic_9954_agreement
private def within_077_095 : CountSegment 1 10 13120 9954 9956 4115 4116 := CountSegment.join atomic_9955 atomic_9956
private theorem within_077_095_agreement : RowAgreement 9954 9956 := RowAgreement.join atomic_9955_agreement atomic_9956_agreement
private def within_077_096 : CountSegment 1 10 13120 9952 9956 4114 4116 := CountSegment.join within_077_094 within_077_095
private theorem within_077_096_agreement : RowAgreement 9952 9956 := RowAgreement.join within_077_094_agreement within_077_095_agreement
private def within_077_097 : CountSegment 1 10 13120 9956 9958 4116 4118 := CountSegment.join atomic_9957 atomic_9958
private theorem within_077_097_agreement : RowAgreement 9956 9958 := RowAgreement.join atomic_9957_agreement atomic_9958_agreement
private def within_077_098 : CountSegment 1 10 13120 9958 9960 4118 4118 := CountSegment.join atomic_9959 atomic_9960
private theorem within_077_098_agreement : RowAgreement 9958 9960 := RowAgreement.join atomic_9959_agreement atomic_9960_agreement
private def within_077_099 : CountSegment 1 10 13120 9956 9960 4116 4118 := CountSegment.join within_077_097 within_077_098
private theorem within_077_099_agreement : RowAgreement 9956 9960 := RowAgreement.join within_077_097_agreement within_077_098_agreement
private def within_077_100 : CountSegment 1 10 13120 9952 9960 4114 4118 := CountSegment.join within_077_096 within_077_099
private theorem within_077_100_agreement : RowAgreement 9952 9960 := RowAgreement.join within_077_096_agreement within_077_099_agreement
private def within_077_101 : CountSegment 1 10 13120 9960 9962 4118 4118 := CountSegment.join atomic_9961 atomic_9962
private theorem within_077_101_agreement : RowAgreement 9960 9962 := RowAgreement.join atomic_9961_agreement atomic_9962_agreement
private def within_077_102 : CountSegment 1 10 13120 9962 9964 4118 4119 := CountSegment.join atomic_9963 atomic_9964
private theorem within_077_102_agreement : RowAgreement 9962 9964 := RowAgreement.join atomic_9963_agreement atomic_9964_agreement
private def within_077_103 : CountSegment 1 10 13120 9960 9964 4118 4119 := CountSegment.join within_077_101 within_077_102
private theorem within_077_103_agreement : RowAgreement 9960 9964 := RowAgreement.join within_077_101_agreement within_077_102_agreement
private def within_077_104 : CountSegment 1 10 13120 9964 9966 4119 4120 := CountSegment.join atomic_9965 atomic_9966
private theorem within_077_104_agreement : RowAgreement 9964 9966 := RowAgreement.join atomic_9965_agreement atomic_9966_agreement
private def within_077_105 : CountSegment 1 10 13120 9966 9968 4120 4120 := CountSegment.join atomic_9967 atomic_9968
private theorem within_077_105_agreement : RowAgreement 9966 9968 := RowAgreement.join atomic_9967_agreement atomic_9968_agreement
private def within_077_106 : CountSegment 1 10 13120 9964 9968 4119 4120 := CountSegment.join within_077_104 within_077_105
private theorem within_077_106_agreement : RowAgreement 9964 9968 := RowAgreement.join within_077_104_agreement within_077_105_agreement
private def within_077_107 : CountSegment 1 10 13120 9960 9968 4118 4120 := CountSegment.join within_077_103 within_077_106
private theorem within_077_107_agreement : RowAgreement 9960 9968 := RowAgreement.join within_077_103_agreement within_077_106_agreement
private def within_077_108 : CountSegment 1 10 13120 9952 9968 4114 4120 := CountSegment.join within_077_100 within_077_107
private theorem within_077_108_agreement : RowAgreement 9952 9968 := RowAgreement.join within_077_100_agreement within_077_107_agreement
private def within_077_109 : CountSegment 1 10 13120 9968 9970 4120 4120 := CountSegment.join atomic_9969 atomic_9970
private theorem within_077_109_agreement : RowAgreement 9968 9970 := RowAgreement.join atomic_9969_agreement atomic_9970_agreement
private def within_077_110 : CountSegment 1 10 13120 9970 9972 4120 4122 := CountSegment.join atomic_9971 atomic_9972
private theorem within_077_110_agreement : RowAgreement 9970 9972 := RowAgreement.join atomic_9971_agreement atomic_9972_agreement
private def within_077_111 : CountSegment 1 10 13120 9968 9972 4120 4122 := CountSegment.join within_077_109 within_077_110
private theorem within_077_111_agreement : RowAgreement 9968 9972 := RowAgreement.join within_077_109_agreement within_077_110_agreement
private def within_077_112 : CountSegment 1 10 13120 9972 9974 4122 4123 := CountSegment.join atomic_9973 atomic_9974
private theorem within_077_112_agreement : RowAgreement 9972 9974 := RowAgreement.join atomic_9973_agreement atomic_9974_agreement
private def within_077_113 : CountSegment 1 10 13120 9974 9976 4123 4124 := CountSegment.join atomic_9975 atomic_9976
private theorem within_077_113_agreement : RowAgreement 9974 9976 := RowAgreement.join atomic_9975_agreement atomic_9976_agreement
private def within_077_114 : CountSegment 1 10 13120 9972 9976 4122 4124 := CountSegment.join within_077_112 within_077_113
private theorem within_077_114_agreement : RowAgreement 9972 9976 := RowAgreement.join within_077_112_agreement within_077_113_agreement
private def within_077_115 : CountSegment 1 10 13120 9968 9976 4120 4124 := CountSegment.join within_077_111 within_077_114
private theorem within_077_115_agreement : RowAgreement 9968 9976 := RowAgreement.join within_077_111_agreement within_077_114_agreement
private def within_077_116 : CountSegment 1 10 13120 9976 9978 4124 4126 := CountSegment.join atomic_9977 atomic_9978
private theorem within_077_116_agreement : RowAgreement 9976 9978 := RowAgreement.join atomic_9977_agreement atomic_9978_agreement
private def within_077_117 : CountSegment 1 10 13120 9978 9980 4126 4126 := CountSegment.join atomic_9979 atomic_9980
private theorem within_077_117_agreement : RowAgreement 9978 9980 := RowAgreement.join atomic_9979_agreement atomic_9980_agreement
private def within_077_118 : CountSegment 1 10 13120 9976 9980 4124 4126 := CountSegment.join within_077_116 within_077_117
private theorem within_077_118_agreement : RowAgreement 9976 9980 := RowAgreement.join within_077_116_agreement within_077_117_agreement
private def within_077_119 : CountSegment 1 10 13120 9980 9982 4126 4126 := CountSegment.join atomic_9981 atomic_9982
private theorem within_077_119_agreement : RowAgreement 9980 9982 := RowAgreement.join atomic_9981_agreement atomic_9982_agreement
private def within_077_120 : CountSegment 1 10 13120 9982 9984 4126 4129 := CountSegment.join atomic_9983 atomic_9984
private theorem within_077_120_agreement : RowAgreement 9982 9984 := RowAgreement.join atomic_9983_agreement atomic_9984_agreement
private def within_077_121 : CountSegment 1 10 13120 9980 9984 4126 4129 := CountSegment.join within_077_119 within_077_120
private theorem within_077_121_agreement : RowAgreement 9980 9984 := RowAgreement.join within_077_119_agreement within_077_120_agreement
private def within_077_122 : CountSegment 1 10 13120 9976 9984 4124 4129 := CountSegment.join within_077_118 within_077_121
private theorem within_077_122_agreement : RowAgreement 9976 9984 := RowAgreement.join within_077_118_agreement within_077_121_agreement
private def within_077_123 : CountSegment 1 10 13120 9968 9984 4120 4129 := CountSegment.join within_077_115 within_077_122
private theorem within_077_123_agreement : RowAgreement 9968 9984 := RowAgreement.join within_077_115_agreement within_077_122_agreement
private def within_077_124 : CountSegment 1 10 13120 9952 9984 4114 4129 := CountSegment.join within_077_108 within_077_123
private theorem within_077_124_agreement : RowAgreement 9952 9984 := RowAgreement.join within_077_108_agreement within_077_123_agreement
private def within_077_125 : CountSegment 1 10 13120 9920 9984 4105 4129 := CountSegment.join within_077_093 within_077_124
private theorem within_077_125_agreement : RowAgreement 9920 9984 := RowAgreement.join within_077_093_agreement within_077_124_agreement
private def within_077_126 : CountSegment 1 10 13120 9856 9984 4085 4129 := CountSegment.join within_077_062 within_077_125
private theorem within_077_126_agreement : RowAgreement 9856 9984 := RowAgreement.join within_077_062_agreement within_077_125_agreement
def segment_077 : CountSegment 1 10 13120 9856 9984 4085 4129 := within_077_126
theorem agreement_077 : RowAgreement 9856 9984 := within_077_126_agreement
end ProjectEulerSolutions.P58Numerical
