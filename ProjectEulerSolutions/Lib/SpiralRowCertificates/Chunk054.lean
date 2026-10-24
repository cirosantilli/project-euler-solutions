import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_6913 : RowCertificate 6913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6913 1) := by change ¬ Nat.Prime 191172103; exact PrattCertificates.composite_of_product (factor := 9103) (cofactor := 21001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6913 2) := by change ¬ Nat.Prime 191158277; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4662397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6913 3) := by change ¬ Nat.Prime 191144451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63714817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6913 : row 6913 = 0 := (row_6913.checked).trans rfl
theorem model_count_6913 : modelRow 6913 = 0 := by decide
theorem modelrow_agrees_6913 : modelRow 6913 = row 6913 := model_count_6913.trans row_count_6913.symm
def row_6914 : RowCertificate 6914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6914 1) := by change ¬ Nat.Prime 191227413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63742471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6914 2) := by change ¬ Nat.Prime 191213585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38242717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6914 3) := by change ¬ Nat.Prime 191199757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27314251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6914 : row 6914 = 0 := (row_6914.checked).trans rfl
theorem model_count_6914 : modelRow 6914 = 0 := by decide
theorem modelrow_agrees_6914 : modelRow 6914 = row 6914 := model_count_6914.trans row_count_6914.symm
def row_6915 : RowCertificate 6915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6915 1) := by change ¬ Nat.Prime 191282731; exact PrattCertificates.composite_of_product (factor := 3229) (cofactor := 59239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6915 2) := by change Nat.Prime 191268901; exact ProjectEulerSolutions.P58PrimeNodes.p_191268901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6915 3) := by change ¬ Nat.Prime 191255071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27322153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6915 : row 6915 = 1 := (row_6915.checked).trans rfl
theorem model_count_6915 : modelRow 6915 = 1 := by decide
theorem modelrow_agrees_6915 : modelRow 6915 = row 6915 := model_count_6915.trans row_count_6915.symm
def row_6916 : RowCertificate 6916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6916 1) := by change ¬ Nat.Prime 191338057; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 623251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6916 2) := by change ¬ Nat.Prime 191324225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38264845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6916 3) := by change ¬ Nat.Prime 191310393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63770131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6916 : row 6916 = 0 := (row_6916.checked).trans rfl
theorem model_count_6916 : modelRow 6916 = 0 := by decide
theorem modelrow_agrees_6916 : modelRow 6916 = row 6916 := model_count_6916.trans row_count_6916.symm
def row_6917 : RowCertificate 6917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6917 1) := by change ¬ Nat.Prime 191393391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63797797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6917 2) := by change ¬ Nat.Prime 191379557; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11257621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6917 3) := by change ¬ Nat.Prime 191365723; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3137143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6917 : row 6917 = 0 := (row_6917.checked).trans rfl
theorem model_count_6917 : modelRow 6917 = 0 := by decide
theorem modelrow_agrees_6917 : modelRow 6917 = row 6917 := model_count_6917.trans row_count_6917.symm
def row_6918 : RowCertificate 6918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6918 1) := by change ¬ Nat.Prime 191448733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27349819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6918 2) := by change ¬ Nat.Prime 191434897; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3138277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6918 3) := by change ¬ Nat.Prime 191421061; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14724697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6918 : row 6918 = 0 := (row_6918.checked).trans rfl
theorem model_count_6918 : modelRow 6918 = 0 := by decide
theorem modelrow_agrees_6918 : modelRow 6918 = row 6918 := model_count_6918.trans row_count_6918.symm
def row_6919 : RowCertificate 6919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6919 1) := by change ¬ Nat.Prime 191504083; exact PrattCertificates.composite_of_product (factor := 631) (cofactor := 303493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6919 2) := by change ¬ Nat.Prime 191490245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38298049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6919 3) := by change ¬ Nat.Prime 191476407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63825469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6919 : row 6919 = 0 := (row_6919.checked).trans rfl
theorem model_count_6919 : modelRow 6919 = 0 := by decide
theorem modelrow_agrees_6919 : modelRow 6919 = row 6919 := model_count_6919.trans row_count_6919.symm
def row_6920 : RowCertificate 6920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6920 1) := by change ¬ Nat.Prime 191559441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63853147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6920 2) := by change ¬ Nat.Prime 191545601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14734277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6920 3) := by change ¬ Nat.Prime 191531761; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10080619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6920 : row 6920 = 0 := (row_6920.checked).trans rfl
theorem model_count_6920 : modelRow 6920 = 0 := by decide
theorem modelrow_agrees_6920 : modelRow 6920 = row 6920 := model_count_6920.trans row_count_6920.symm
def row_6921 : RowCertificate 6921 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6921 1) := by change Nat.Prime 191614807; exact ProjectEulerSolutions.P58PrimeNodes.p_191614807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6921 2) := by change ¬ Nat.Prime 191600965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38320193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6921 3) := by change ¬ Nat.Prime 191587123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27369589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6921 : row 6921 = 1 := (row_6921.checked).trans rfl
theorem model_count_6921 : modelRow 6921 = 1 := by decide
theorem modelrow_agrees_6921 : modelRow 6921 = row 6921 := model_count_6921.trans row_count_6921.symm
def row_6922 : RowCertificate 6922 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6922 1) := by change Nat.Prime 191670181; exact ProjectEulerSolutions.P58PrimeNodes.p_191670181.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6922 2) := by change ¬ Nat.Prime 191656337; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5179901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6922 3) := by change ¬ Nat.Prime 191642493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63880831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6922 : row 6922 = 1 := (row_6922.checked).trans rfl
theorem model_count_6922 : modelRow 6922 = 1 := by decide
theorem modelrow_agrees_6922 : modelRow 6922 = row 6922 := model_count_6922.trans row_count_6922.symm
def row_6923 : RowCertificate 6923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6923 1) := by change ¬ Nat.Prime 191725563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63908521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6923 2) := by change Nat.Prime 191711717; exact ProjectEulerSolutions.P58PrimeNodes.p_191711717.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6923 3) := by change Nat.Prime 191697871; exact ProjectEulerSolutions.P58PrimeNodes.p_191697871.prime
    exact decide_eq_true prime
theorem row_count_6923 : row 6923 = 2 := (row_6923.checked).trans rfl
theorem model_count_6923 : modelRow 6923 = 2 := by decide
theorem modelrow_agrees_6923 : modelRow 6923 = row 6923 := model_count_6923.trans row_count_6923.symm
def row_6924 : RowCertificate 6924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6924 1) := by change ¬ Nat.Prime 191780953; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27397279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6924 2) := by change ¬ Nat.Prime 191767105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38353421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6924 3) := by change ¬ Nat.Prime 191753257; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 319057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6924 : row 6924 = 0 := (row_6924.checked).trans rfl
theorem model_count_6924 : modelRow 6924 = 0 := by decide
theorem modelrow_agrees_6924 : modelRow 6924 = row 6924 := model_count_6924.trans row_count_6924.symm
def row_6925 : RowCertificate 6925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6925 1) := by change ¬ Nat.Prime 191836351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27405193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6925 2) := by change ¬ Nat.Prime 191822501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14755577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6925 3) := by change ¬ Nat.Prime 191808651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63936217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6925 : row 6925 = 0 := (row_6925.checked).trans rfl
theorem model_count_6925 : modelRow 6925 = 0 := by decide
theorem modelrow_agrees_6925 : modelRow 6925 = row 6925 := model_count_6925.trans row_count_6925.symm
def row_6926 : RowCertificate 6926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6926 1) := by change ¬ Nat.Prime 191891757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63963919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6926 2) := by change ¬ Nat.Prime 191877905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38375581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6926 3) := by change ¬ Nat.Prime 191864053; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6189163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6926 : row 6926 = 0 := (row_6926.checked).trans rfl
theorem model_count_6926 : modelRow 6926 = 0 := by decide
theorem modelrow_agrees_6926 : modelRow 6926 = row 6926 := model_count_6926.trans row_count_6926.symm
def row_6927 : RowCertificate 6927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6927 1) := by change ¬ Nat.Prime 191947171; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14765167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6927 2) := by change Nat.Prime 191933317; exact ProjectEulerSolutions.P58PrimeNodes.p_191933317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6927 3) := by change ¬ Nat.Prime 191919463; exact PrattCertificates.composite_of_product (factor := 8233) (cofactor := 23311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6927 : row 6927 = 1 := (row_6927.checked).trans rfl
theorem model_count_6927 : modelRow 6927 = 1 := by decide
theorem modelrow_agrees_6927 : modelRow 6927 = row 6927 := model_count_6927.trans row_count_6927.symm
def row_6928 : RowCertificate 6928 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6928 1) := by change Nat.Prime 192002593; exact ProjectEulerSolutions.P58PrimeNodes.p_192002593.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6928 2) := by change ¬ Nat.Prime 191988737; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 3622429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6928 3) := by change ¬ Nat.Prime 191974881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 63991627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6928 : row 6928 = 1 := (row_6928.checked).trans rfl
theorem model_count_6928 : modelRow 6928 = 1 := by decide
theorem modelrow_agrees_6928 : modelRow 6928 = row 6928 := model_count_6928.trans row_count_6928.symm
def row_6929 : RowCertificate 6929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6929 1) := by change ¬ Nat.Prime 192058023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64019341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6929 2) := by change ¬ Nat.Prime 192044165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38408833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6929 3) := by change ¬ Nat.Prime 192030307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27432901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6929 : row 6929 = 0 := (row_6929.checked).trans rfl
theorem model_count_6929 : modelRow 6929 = 0 := by decide
theorem modelrow_agrees_6929 : modelRow 6929 = row 6929 := model_count_6929.trans row_count_6929.symm
def row_6930 : RowCertificate 6930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6930 1) := by change ¬ Nat.Prime 192113461; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3149401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6930 2) := by change ¬ Nat.Prime 192099601; exact PrattCertificates.composite_of_product (factor := 5741) (cofactor := 33461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6930 3) := by change ¬ Nat.Prime 192085741; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1272091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6930 : row 6930 = 0 := (row_6930.checked).trans rfl
theorem model_count_6930 : modelRow 6930 = 0 := by decide
theorem modelrow_agrees_6930 : modelRow 6930 = row 6930 := model_count_6930.trans row_count_6930.symm
def row_6931 : RowCertificate 6931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6931 1) := by change ¬ Nat.Prime 192168907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27452701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6931 2) := by change ¬ Nat.Prime 192155045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38431009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6931 3) := by change ¬ Nat.Prime 192141183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64047061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6931 : row 6931 = 0 := (row_6931.checked).trans rfl
theorem model_count_6931 : modelRow 6931 = 0 := by decide
theorem modelrow_agrees_6931 : modelRow 6931 = row 6931 := model_count_6931.trans row_count_6931.symm
def row_6932 : RowCertificate 6932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6932 1) := by change ¬ Nat.Prime 192224361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64074787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6932 2) := by change ¬ Nat.Prime 192210497; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 456557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6932 3) := by change ¬ Nat.Prime 192196633; exact PrattCertificates.composite_of_product (factor := 2791) (cofactor := 68863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6932 : row 6932 = 0 := (row_6932.checked).trans rfl
theorem model_count_6932 : modelRow 6932 = 0 := by decide
theorem modelrow_agrees_6932 : modelRow 6932 = row 6932 := model_count_6932.trans row_count_6932.symm
def row_6933 : RowCertificate 6933 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6933 1) := by change Nat.Prime 192279823; exact ProjectEulerSolutions.P58PrimeNodes.p_192279823.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6933 2) := by change ¬ Nat.Prime 192265957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14789689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6933 3) := by change ¬ Nat.Prime 192252091; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 862117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6933 : row 6933 = 1 := (row_6933.checked).trans rfl
theorem model_count_6933 : modelRow 6933 = 1 := by decide
theorem modelrow_agrees_6933 : modelRow 6933 = row 6933 := model_count_6933.trans row_count_6933.symm
def row_6934 : RowCertificate 6934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6934 1) := by change ¬ Nat.Prime 192335293; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 415411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6934 2) := by change ¬ Nat.Prime 192321425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38464285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6934 3) := by change ¬ Nat.Prime 192307557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64102519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6934 : row 6934 = 0 := (row_6934.checked).trans rfl
theorem model_count_6934 : modelRow 6934 = 0 := by decide
theorem modelrow_agrees_6934 : modelRow 6934 = row 6934 := model_count_6934.trans row_count_6934.symm
def row_6935 : RowCertificate 6935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6935 1) := by change ¬ Nat.Prime 192390771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64130257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6935 2) := by change Nat.Prime 192376901; exact ProjectEulerSolutions.P58PrimeNodes.p_192376901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6935 3) := by change ¬ Nat.Prime 192363031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27480433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6935 : row 6935 = 1 := (row_6935.checked).trans rfl
theorem model_count_6935 : modelRow 6935 = 1 := by decide
theorem modelrow_agrees_6935 : modelRow 6935 = row 6935 := model_count_6935.trans row_count_6935.symm
def row_6936 : RowCertificate 6936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6936 1) := by change ¬ Nat.Prime 192446257; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 457117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6936 2) := by change ¬ Nat.Prime 192432385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38486477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6936 3) := by change ¬ Nat.Prime 192418513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27488359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6936 : row 6936 = 0 := (row_6936.checked).trans rfl
theorem model_count_6936 : modelRow 6936 = 0 := by decide
theorem modelrow_agrees_6936 : modelRow 6936 = row 6936 := model_count_6936.trans row_count_6936.symm
def row_6937 : RowCertificate 6937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6937 1) := by change ¬ Nat.Prime 192501751; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14807827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6937 2) := by change ¬ Nat.Prime 192487877; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6637513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6937 3) := by change ¬ Nat.Prime 192474003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64158001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6937 : row 6937 = 0 := (row_6937.checked).trans rfl
theorem model_count_6937 : modelRow 6937 = 0 := by decide
theorem modelrow_agrees_6937 : modelRow 6937 = row 6937 := model_count_6937.trans row_count_6937.symm
def row_6938 : RowCertificate 6938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6938 1) := by change ¬ Nat.Prime 192557253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64185751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6938 2) := by change ¬ Nat.Prime 192543377; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14811029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6938 3) := by change ¬ Nat.Prime 192529501; exact PrattCertificates.composite_of_product (factor := 4021) (cofactor := 47881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6938 : row 6938 = 0 := (row_6938.checked).trans rfl
theorem model_count_6938 : modelRow 6938 = 0 := by decide
theorem modelrow_agrees_6938 : modelRow 6938 = row 6938 := model_count_6938.trans row_count_6938.symm
def row_6939 : RowCertificate 6939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6939 1) := by change ¬ Nat.Prime 192612763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27516109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6939 2) := by change ¬ Nat.Prime 192598885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38519777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6939 3) := by change ¬ Nat.Prime 192585007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10136053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6939 : row 6939 = 0 := (row_6939.checked).trans rfl
theorem model_count_6939 : modelRow 6939 = 0 := by decide
theorem modelrow_agrees_6939 : modelRow 6939 = row 6939 := model_count_6939.trans row_count_6939.symm
def row_6940 : RowCertificate 6940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6940 1) := by change ¬ Nat.Prime 192668281; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14820637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6940 2) := by change Nat.Prime 192654401; exact ProjectEulerSolutions.P58PrimeNodes.p_192654401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6940 3) := by change ¬ Nat.Prime 192640521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64213507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6940 : row 6940 = 1 := (row_6940.checked).trans rfl
theorem model_count_6940 : modelRow 6940 = 1 := by decide
theorem modelrow_agrees_6940 : modelRow 6940 = row 6940 := model_count_6940.trans row_count_6940.symm
def row_6941 : RowCertificate 6941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6941 1) := by change ¬ Nat.Prime 192723807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64241269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6941 2) := by change ¬ Nat.Prime 192709925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38541985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6941 3) := by change ¬ Nat.Prime 192696043; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10141897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6941 : row 6941 = 0 := (row_6941.checked).trans rfl
theorem model_count_6941 : modelRow 6941 = 0 := by decide
theorem modelrow_agrees_6941 : modelRow 6941 = row 6941 := model_count_6941.trans row_count_6941.symm
def row_6942 : RowCertificate 6942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6942 1) := by change ¬ Nat.Prime 192779341; exact PrattCertificates.composite_of_product (factor := 1489) (cofactor := 129469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6942 2) := by change ¬ Nat.Prime 192765457; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 1705889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6942 3) := by change ¬ Nat.Prime 192751573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27535939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6942 : row 6942 = 0 := (row_6942.checked).trans rfl
theorem model_count_6942 : modelRow 6942 = 0 := by decide
theorem modelrow_agrees_6942 : modelRow 6942 = row 6942 := model_count_6942.trans row_count_6942.symm
def row_6943 : RowCertificate 6943 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6943 1) := by change Nat.Prime 192834883; exact ProjectEulerSolutions.P58PrimeNodes.p_192834883.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6943 2) := by change Nat.Prime 192820997; exact ProjectEulerSolutions.P58PrimeNodes.p_192820997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6943 3) := by change ¬ Nat.Prime 192807111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64269037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6943 : row 6943 = 2 := (row_6943.checked).trans rfl
theorem model_count_6943 : modelRow 6943 = 2 := by decide
theorem modelrow_agrees_6943 : modelRow 6943 = row 6943 := model_count_6943.trans row_count_6943.symm
def row_6944 : RowCertificate 6944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6944 1) := by change ¬ Nat.Prime 192890433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64296811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6944 2) := by change ¬ Nat.Prime 192876545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38575309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6944 3) := by change ¬ Nat.Prime 192862657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14835589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6944 : row 6944 = 0 := (row_6944.checked).trans rfl
theorem model_count_6944 : modelRow 6944 = 0 := by decide
theorem modelrow_agrees_6944 : modelRow 6944 = row 6944 := model_count_6944.trans row_count_6944.symm
def row_6945 : RowCertificate 6945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6945 1) := by change ¬ Nat.Prime 192945991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27563713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6945 2) := by change ¬ Nat.Prime 192932101; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4705661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6945 3) := by change ¬ Nat.Prime 192918211; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 517207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6945 : row 6945 = 0 := (row_6945.checked).trans rfl
theorem model_count_6945 : modelRow 6945 = 0 := by decide
theorem modelrow_agrees_6945 : modelRow 6945 = row 6945 := model_count_6945.trans row_count_6945.symm
def row_6946 : RowCertificate 6946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6946 1) := by change ¬ Nat.Prime 193001557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27571651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6946 2) := by change ¬ Nat.Prime 192987665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38597533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6946 3) := by change ¬ Nat.Prime 192973773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64324591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6946 : row 6946 = 0 := (row_6946.checked).trans rfl
theorem model_count_6946 : modelRow 6946 = 0 := by decide
theorem modelrow_agrees_6946 : modelRow 6946 = row 6946 := model_count_6946.trans row_count_6946.symm
def row_6947 : RowCertificate 6947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6947 1) := by change ¬ Nat.Prime 193057131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64352377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6947 2) := by change ¬ Nat.Prime 193043237; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 751141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6947 3) := by change ¬ Nat.Prime 193029343; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14848411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6947 : row 6947 = 0 := (row_6947.checked).trans rfl
theorem model_count_6947 : modelRow 6947 = 0 := by decide
theorem modelrow_agrees_6947 : modelRow 6947 = row 6947 := model_count_6947.trans row_count_6947.symm
def row_6948 : RowCertificate 6948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6948 1) := by change ¬ Nat.Prime 193112713; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10163827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6948 2) := by change ¬ Nat.Prime 193098817; exact PrattCertificates.composite_of_product (factor := 1061) (cofactor := 181997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6948 3) := by change ¬ Nat.Prime 193084921; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4490347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6948 : row 6948 = 0 := (row_6948.checked).trans rfl
theorem model_count_6948 : modelRow 6948 = 0 := by decide
theorem modelrow_agrees_6948 : modelRow 6948 = row 6948 := model_count_6948.trans row_count_6948.symm
def row_6949 : RowCertificate 6949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6949 1) := by change ¬ Nat.Prime 193168303; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2883109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6949 2) := by change ¬ Nat.Prime 193154405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38630881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6949 3) := by change ¬ Nat.Prime 193140507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64380169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6949 : row 6949 = 0 := (row_6949.checked).trans rfl
theorem model_count_6949 : modelRow 6949 = 0 := by decide
theorem modelrow_agrees_6949 : modelRow 6949 = row 6949 := model_count_6949.trans row_count_6949.symm
def row_6950 : RowCertificate 6950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6950 1) := by change ¬ Nat.Prime 193223901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64407967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6950 2) := by change Nat.Prime 193210001; exact ProjectEulerSolutions.P58PrimeNodes.p_193210001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6950 3) := by change ¬ Nat.Prime 193196101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27599443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6950 : row 6950 = 1 := (row_6950.checked).trans rfl
theorem model_count_6950 : modelRow 6950 = 1 := by decide
theorem modelrow_agrees_6950 : modelRow 6950 = row 6950 := model_count_6950.trans row_count_6950.symm
def row_6951 : RowCertificate 6951 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6951 1) := by change Nat.Prime 193279507; exact ProjectEulerSolutions.P58PrimeNodes.p_193279507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6951 2) := by change ¬ Nat.Prime 193265605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38653121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6951 3) := by change ¬ Nat.Prime 193251703; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5223019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6951 : row 6951 = 1 := (row_6951.checked).trans rfl
theorem model_count_6951 : modelRow 6951 = 1 := by decide
theorem modelrow_agrees_6951 : modelRow 6951 = row 6951 := model_count_6951.trans row_count_6951.symm
def row_6952 : RowCertificate 6952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6952 1) := by change ¬ Nat.Prime 193335121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27619303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6952 2) := by change ¬ Nat.Prime 193321217; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 251393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6952 3) := by change ¬ Nat.Prime 193307313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64435771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6952 : row 6952 = 0 := (row_6952.checked).trans rfl
theorem model_count_6952 : modelRow 6952 = 0 := by decide
theorem modelrow_agrees_6952 : modelRow 6952 = row 6952 := model_count_6952.trans row_count_6952.symm
def row_6953 : RowCertificate 6953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6953 1) := by change ¬ Nat.Prime 193390743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64463581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6953 2) := by change ¬ Nat.Prime 193376837; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5226401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6953 3) := by change Nat.Prime 193362931; exact ProjectEulerSolutions.P58PrimeNodes.p_193362931.prime
    exact decide_eq_true prime
theorem row_count_6953 : row 6953 = 1 := (row_6953.checked).trans rfl
theorem model_count_6953 : modelRow 6953 = 1 := by decide
theorem modelrow_agrees_6953 : modelRow 6953 = row 6953 := model_count_6953.trans row_count_6953.symm
def row_6954 : RowCertificate 6954 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6954 1) := by change Nat.Prime 193446373; exact ProjectEulerSolutions.P58PrimeNodes.p_193446373.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6954 2) := by change ¬ Nat.Prime 193432465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38686493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6954 3) := by change ¬ Nat.Prime 193418557; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 156361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6954 : row 6954 = 1 := (row_6954.checked).trans rfl
theorem model_count_6954 : modelRow 6954 = 1 := by decide
theorem modelrow_agrees_6954 : modelRow 6954 = row 6954 := model_count_6954.trans row_count_6954.symm
def row_6955 : RowCertificate 6955 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6955 1) := by change Nat.Prime 193502011; exact ProjectEulerSolutions.P58PrimeNodes.p_193502011.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6955 2) := by change ¬ Nat.Prime 193488101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11381653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6955 3) := by change ¬ Nat.Prime 193474191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64491397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6955 : row 6955 = 1 := (row_6955.checked).trans rfl
theorem model_count_6955 : modelRow 6955 = 1 := by decide
theorem modelrow_agrees_6955 : modelRow 6955 = row 6955 := model_count_6955.trans row_count_6955.symm
def row_6956 : RowCertificate 6956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6956 1) := by change ¬ Nat.Prime 193557657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64519219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6956 2) := by change ¬ Nat.Prime 193543745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38708749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6956 3) := by change ¬ Nat.Prime 193529833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27647119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6956 : row 6956 = 0 := (row_6956.checked).trans rfl
theorem model_count_6956 : modelRow 6956 = 0 := by decide
theorem modelrow_agrees_6956 : modelRow 6956 = row 6956 := model_count_6956.trans row_count_6956.symm
def row_6957 : RowCertificate 6957 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6957 1) := by change Nat.Prime 193613311; exact ProjectEulerSolutions.P58PrimeNodes.p_193613311.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6957 2) := by change ¬ Nat.Prime 193599397; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 1713269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6957 3) := by change ¬ Nat.Prime 193585483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27655069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6957 : row 6957 = 1 := (row_6957.checked).trans rfl
theorem model_count_6957 : modelRow 6957 = 1 := by decide
theorem modelrow_agrees_6957 : modelRow 6957 = row 6957 := model_count_6957.trans row_count_6957.symm
def row_6958 : RowCertificate 6958 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6958 1) := by change Nat.Prime 193668973; exact ProjectEulerSolutions.P58PrimeNodes.p_193668973.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6958 2) := by change ¬ Nat.Prime 193655057; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 3653869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6958 3) := by change ¬ Nat.Prime 193641141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64547047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6958 : row 6958 = 1 := (row_6958.checked).trans rfl
theorem model_count_6958 : modelRow 6958 = 1 := by decide
theorem modelrow_agrees_6958 : modelRow 6958 = row 6958 := model_count_6958.trans row_count_6958.symm
def row_6959 : RowCertificate 6959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6959 1) := by change ¬ Nat.Prime 193724643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64574881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6959 2) := by change ¬ Nat.Prime 193710725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38742145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6959 3) := by change ¬ Nat.Prime 193696807; exact PrattCertificates.composite_of_product (factor := 811) (cofactor := 238837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6959 : row 6959 = 0 := (row_6959.checked).trans rfl
theorem model_count_6959 : modelRow 6959 = 0 := by decide
theorem modelrow_agrees_6959 : modelRow 6959 = row 6959 := model_count_6959.trans row_count_6959.symm
def row_6960 : RowCertificate 6960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6960 1) := by change ¬ Nat.Prime 193780321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27682903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6960 2) := by change ¬ Nat.Prime 193766401; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 1120037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6960 3) := by change ¬ Nat.Prime 193752481; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14904037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6960 : row 6960 = 0 := (row_6960.checked).trans rfl
theorem model_count_6960 : modelRow 6960 = 0 := by decide
theorem modelrow_agrees_6960 : modelRow 6960 = row 6960 := model_count_6960.trans row_count_6960.symm
def row_6961 : RowCertificate 6961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6961 1) := by change ¬ Nat.Prime 193836007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5238811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6961 2) := by change ¬ Nat.Prime 193822085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38764417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6961 3) := by change ¬ Nat.Prime 193808163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64602721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6961 : row 6961 = 0 := (row_6961.checked).trans rfl
theorem model_count_6961 : modelRow 6961 = 0 := by decide
theorem modelrow_agrees_6961 : modelRow 6961 = row 6961 := model_count_6961.trans row_count_6961.symm
def row_6962 : RowCertificate 6962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6962 1) := by change ¬ Nat.Prime 193891701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64630567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6962 2) := by change Nat.Prime 193877777; exact ProjectEulerSolutions.P58PrimeNodes.p_193877777.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6962 3) := by change Nat.Prime 193863853; exact ProjectEulerSolutions.P58PrimeNodes.p_193863853.prime
    exact decide_eq_true prime
theorem row_count_6962 : row 6962 = 2 := (row_6962.checked).trans rfl
theorem model_count_6962 : modelRow 6962 = 2 := by decide
theorem modelrow_agrees_6962 : modelRow 6962 = row 6962 := model_count_6962.trans row_count_6962.symm
def row_6963 : RowCertificate 6963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6963 1) := by change ¬ Nat.Prime 193947403; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14919031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6963 2) := by change ¬ Nat.Prime 193933477; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 661889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6963 3) := by change ¬ Nat.Prime 193919551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27702793) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6963 : row 6963 = 0 := (row_6963.checked).trans rfl
theorem model_count_6963 : modelRow 6963 = 0 := by decide
theorem modelrow_agrees_6963 : modelRow 6963 = row 6963 := model_count_6963.trans row_count_6963.symm
def row_6964 : RowCertificate 6964 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6964 1) := by change Nat.Prime 194003113; exact ProjectEulerSolutions.P58PrimeNodes.p_194003113.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6964 2) := by change ¬ Nat.Prime 193989185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38797837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6964 3) := by change ¬ Nat.Prime 193975257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64658419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6964 : row 6964 = 1 := (row_6964.checked).trans rfl
theorem model_count_6964 : modelRow 6964 = 1 := by decide
theorem modelrow_agrees_6964 : modelRow 6964 = row 6964 := model_count_6964.trans row_count_6964.symm
def row_6965 : RowCertificate 6965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6965 1) := by change ¬ Nat.Prime 194058831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64686277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6965 2) := by change ¬ Nat.Prime 194044901; exact PrattCertificates.composite_of_product (factor := 977) (cofactor := 198613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6965 3) := by change ¬ Nat.Prime 194030971; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 488743) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6965 : row 6965 = 0 := (row_6965.checked).trans rfl
theorem model_count_6965 : modelRow 6965 = 0 := by decide
theorem modelrow_agrees_6965 : modelRow 6965 = row 6965 := model_count_6965.trans row_count_6965.symm
def row_6966 : RowCertificate 6966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6966 1) := by change ¬ Nat.Prime 194114557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27730651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6966 2) := by change ¬ Nat.Prime 194100625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38820125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6966 3) := by change Nat.Prime 194086693; exact ProjectEulerSolutions.P58PrimeNodes.p_194086693.prime
    exact decide_eq_true prime
theorem row_count_6966 : row 6966 = 1 := (row_6966.checked).trans rfl
theorem model_count_6966 : modelRow 6966 = 1 := by decide
theorem modelrow_agrees_6966 : modelRow 6966 = row 6966 := model_count_6966.trans row_count_6966.symm
def row_6967 : RowCertificate 6967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6967 1) := by change ¬ Nat.Prime 194170291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27738613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6967 2) := by change ¬ Nat.Prime 194156357; exact PrattCertificates.composite_of_product (factor := 1277) (cofactor := 152041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6967 3) := by change ¬ Nat.Prime 194142423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64714141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6967 : row 6967 = 0 := (row_6967.checked).trans rfl
theorem model_count_6967 : modelRow 6967 = 0 := by decide
theorem modelrow_agrees_6967 : modelRow 6967 = row 6967 := model_count_6967.trans row_count_6967.symm
def row_6968 : RowCertificate 6968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6968 1) := by change ¬ Nat.Prime 194226033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64742011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6968 2) := by change ¬ Nat.Prime 194212097; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11424241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6968 3) := by change ¬ Nat.Prime 194198161; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 172009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6968 : row 6968 = 0 := (row_6968.checked).trans rfl
theorem model_count_6968 : modelRow 6968 = 0 := by decide
theorem modelrow_agrees_6968 : modelRow 6968 = row 6968 := model_count_6968.trans row_count_6968.symm
def row_6969 : RowCertificate 6969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6969 1) := by change ¬ Nat.Prime 194281783; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10225357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6969 2) := by change ¬ Nat.Prime 194267845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38853569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6969 3) := by change Nat.Prime 194253907; exact ProjectEulerSolutions.P58PrimeNodes.p_194253907.prime
    exact decide_eq_true prime
theorem row_count_6969 : row 6969 = 1 := (row_6969.checked).trans rfl
theorem model_count_6969 : modelRow 6969 = 1 := by decide
theorem modelrow_agrees_6969 : modelRow 6969 = row 6969 := model_count_6969.trans row_count_6969.symm
def row_6970 : RowCertificate 6970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6970 1) := by change ¬ Nat.Prime 194337541; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 921031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6970 2) := by change Nat.Prime 194323601; exact ProjectEulerSolutions.P58PrimeNodes.p_194323601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6970 3) := by change ¬ Nat.Prime 194309661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64769887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6970 : row 6970 = 1 := (row_6970.checked).trans rfl
theorem model_count_6970 : modelRow 6970 = 1 := by decide
theorem modelrow_agrees_6970 : modelRow 6970 = row 6970 := model_count_6970.trans row_count_6970.symm
def row_6971 : RowCertificate 6971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6971 1) := by change ¬ Nat.Prime 194393307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64797769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6971 2) := by change ¬ Nat.Prime 194379365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38875873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6971 3) := by change ¬ Nat.Prime 194365423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27766489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6971 : row 6971 = 0 := (row_6971.checked).trans rfl
theorem model_count_6971 : modelRow 6971 = 0 := by decide
theorem modelrow_agrees_6971 : modelRow 6971 = row 6971 := model_count_6971.trans row_count_6971.symm
def row_6972 : RowCertificate 6972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6972 1) := by change ¬ Nat.Prime 194449081; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6272551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6972 2) := by change ¬ Nat.Prime 194435137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14956549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6972 3) := by change ¬ Nat.Prime 194421193; exact PrattCertificates.composite_of_product (factor := 5737) (cofactor := 33889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6972 : row 6972 = 0 := (row_6972.checked).trans rfl
theorem model_count_6972 : modelRow 6972 = 0 := by decide
theorem modelrow_agrees_6972 : modelRow 6972 = row 6972 := model_count_6972.trans row_count_6972.symm
def row_6973 : RowCertificate 6973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6973 1) := by change ¬ Nat.Prime 194504863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27786409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6973 2) := by change ¬ Nat.Prime 194490917; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2005061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6973 3) := by change ¬ Nat.Prime 194476971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64825657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6973 : row 6973 = 0 := (row_6973.checked).trans rfl
theorem model_count_6973 : modelRow 6973 = 0 := by decide
theorem modelrow_agrees_6973 : modelRow 6973 = row 6973 := model_count_6973.trans row_count_6973.symm
def row_6974 : RowCertificate 6974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6974 1) := by change ¬ Nat.Prime 194560653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64853551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6974 2) := by change ¬ Nat.Prime 194546705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38909341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6974 3) := by change Nat.Prime 194532757; exact ProjectEulerSolutions.P58PrimeNodes.p_194532757.prime
    exact decide_eq_true prime
theorem row_count_6974 : row 6974 = 1 := (row_6974.checked).trans rfl
theorem model_count_6974 : modelRow 6974 = 1 := by decide
theorem modelrow_agrees_6974 : modelRow 6974 = row 6974 := model_count_6974.trans row_count_6974.symm
def row_6975 : RowCertificate 6975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6975 1) := by change ¬ Nat.Prime 194616451; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1532413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6975 2) := by change ¬ Nat.Prime 194602501; exact PrattCertificates.composite_of_product (factor := 7069) (cofactor := 27529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6975 3) := by change Nat.Prime 194588551; exact ProjectEulerSolutions.P58PrimeNodes.p_194588551.prime
    exact decide_eq_true prime
theorem row_count_6975 : row 6975 = 1 := (row_6975.checked).trans rfl
theorem model_count_6975 : modelRow 6975 = 1 := by decide
theorem modelrow_agrees_6975 : modelRow 6975 = row 6975 := model_count_6975.trans row_count_6975.symm
def row_6976 : RowCertificate 6976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6976 1) := by change ¬ Nat.Prime 194672257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14974789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6976 2) := by change ¬ Nat.Prime 194658305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38931661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6976 3) := by change ¬ Nat.Prime 194644353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64881451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6976 : row 6976 = 0 := (row_6976.checked).trans rfl
theorem model_count_6976 : modelRow 6976 = 0 := by decide
theorem modelrow_agrees_6976 : modelRow 6976 = row 6976 := model_count_6976.trans row_count_6976.symm
def row_6977 : RowCertificate 6977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6977 1) := by change ¬ Nat.Prime 194728071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64909357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6977 2) := by change ¬ Nat.Prime 194714117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14978009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6977 3) := by change ¬ Nat.Prime 194700163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27814309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6977 : row 6977 = 0 := (row_6977.checked).trans rfl
theorem model_count_6977 : modelRow 6977 = 0 := by decide
theorem modelrow_agrees_6977 : modelRow 6977 = row 6977 := model_count_6977.trans row_count_6977.symm
def row_6978 : RowCertificate 6978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6978 1) := by change ¬ Nat.Prime 194783893; exact PrattCertificates.composite_of_product (factor := 13537) (cofactor := 14389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6978 2) := by change ¬ Nat.Prime 194769937; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1076077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6978 3) := by change ¬ Nat.Prime 194755981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27822283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6978 : row 6978 = 0 := (row_6978.checked).trans rfl
theorem model_count_6978 : modelRow 6978 = 0 := by decide
theorem modelrow_agrees_6978 : modelRow 6978 = row 6978 := model_count_6978.trans row_count_6978.symm
def row_6979 : RowCertificate 6979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6979 1) := by change ¬ Nat.Prime 194839723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14987671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6979 2) := by change ¬ Nat.Prime 194825765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38965153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6979 3) := by change ¬ Nat.Prime 194811807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64937269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6979 : row 6979 = 0 := (row_6979.checked).trans rfl
theorem model_count_6979 : modelRow 6979 = 0 := by decide
theorem modelrow_agrees_6979 : modelRow 6979 = row 6979 := model_count_6979.trans row_count_6979.symm
def row_6980 : RowCertificate 6980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6980 1) := by change ¬ Nat.Prime 194895561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64965187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6980 2) := by change ¬ Nat.Prime 194881601; exact PrattCertificates.composite_of_product (factor := 3581) (cofactor := 54421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6980 3) := by change ¬ Nat.Prime 194867641; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5266693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6980 : row 6980 = 0 := (row_6980.checked).trans rfl
theorem model_count_6980 : modelRow 6980 = 0 := by decide
theorem modelrow_agrees_6980 : modelRow 6980 = row 6980 := model_count_6980.trans row_count_6980.symm
def row_6981 : RowCertificate 6981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6981 1) := by change ¬ Nat.Prime 194951407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27850201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6981 2) := by change ¬ Nat.Prime 194937445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 38987489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6981 3) := by change ¬ Nat.Prime 194923483; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1892461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6981 : row 6981 = 0 := (row_6981.checked).trans rfl
theorem model_count_6981 : modelRow 6981 = 0 := by decide
theorem modelrow_agrees_6981 : modelRow 6981 = row 6981 := model_count_6981.trans row_count_6981.symm
def row_6982 : RowCertificate 6982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6982 1) := by change ¬ Nat.Prime 195007261; exact PrattCertificates.composite_of_product (factor := 2953) (cofactor := 66037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6982 2) := by change Nat.Prime 194993297; exact ProjectEulerSolutions.P58PrimeNodes.p_194993297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6982 3) := by change ¬ Nat.Prime 194979333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 64993111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6982 : row 6982 = 1 := (row_6982.checked).trans rfl
theorem model_count_6982 : modelRow 6982 = 1 := by decide
theorem modelrow_agrees_6982 : modelRow 6982 = row 6982 := model_count_6982.trans row_count_6982.symm
def row_6983 : RowCertificate 6983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6983 1) := by change ¬ Nat.Prime 195063123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65021041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6983 2) := by change ¬ Nat.Prime 195049157; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6725833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6983 3) := by change ¬ Nat.Prime 195035191; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15002707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6983 : row 6983 = 0 := (row_6983.checked).trans rfl
theorem model_count_6983 : modelRow 6983 = 0 := by decide
theorem modelrow_agrees_6983 : modelRow 6983 = row 6983 := model_count_6983.trans row_count_6983.symm
def row_6984 : RowCertificate 6984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6984 1) := by change ¬ Nat.Prime 195118993; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4537651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6984 2) := by change ¬ Nat.Prime 195105025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39021005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6984 3) := by change ¬ Nat.Prime 195091057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27870151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6984 : row 6984 = 0 := (row_6984.checked).trans rfl
theorem model_count_6984 : modelRow 6984 = 0 := by decide
theorem modelrow_agrees_6984 : modelRow 6984 = row 6984 := model_count_6984.trans row_count_6984.symm
def row_6985 : RowCertificate 6985 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6985 1) := by change Nat.Prime 195174871; exact ProjectEulerSolutions.P58PrimeNodes.p_195174871.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6985 2) := by change ¬ Nat.Prime 195160901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15012377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6985 3) := by change ¬ Nat.Prime 195146931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65048977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6985 : row 6985 = 1 := (row_6985.checked).trans rfl
theorem model_count_6985 : modelRow 6985 = 1 := by decide
theorem modelrow_agrees_6985 : modelRow 6985 = row 6985 := model_count_6985.trans row_count_6985.symm
def row_6986 : RowCertificate 6986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6986 1) := by change ¬ Nat.Prime 195230757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65076919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6986 2) := by change ¬ Nat.Prime 195216785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39043357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6986 3) := by change ¬ Nat.Prime 195202813; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15015601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6986 : row 6986 = 0 := (row_6986.checked).trans rfl
theorem model_count_6986 : modelRow 6986 = 0 := by decide
theorem modelrow_agrees_6986 : modelRow 6986 = row 6986 := model_count_6986.trans row_count_6986.symm
def row_6987 : RowCertificate 6987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6987 1) := by change ¬ Nat.Prime 195286651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27898093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6987 2) := by change ¬ Nat.Prime 195272677; exact PrattCertificates.composite_of_product (factor := 2441) (cofactor := 79997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6987 3) := by change ¬ Nat.Prime 195258703; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2914309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6987 : row 6987 = 0 := (row_6987.checked).trans rfl
theorem model_count_6987 : modelRow 6987 = 0 := by decide
theorem modelrow_agrees_6987 : modelRow 6987 = row 6987 := model_count_6987.trans row_count_6987.symm
def row_6988 : RowCertificate 6988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6988 1) := by change ¬ Nat.Prime 195342553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27906079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6988 2) := by change ¬ Nat.Prime 195328577; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 169409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6988 3) := by change ¬ Nat.Prime 195314601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65104867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6988 : row 6988 = 0 := (row_6988.checked).trans rfl
theorem model_count_6988 : modelRow 6988 = 0 := by decide
theorem modelrow_agrees_6988 : modelRow 6988 = row 6988 := model_count_6988.trans row_count_6988.symm
def row_6989 : RowCertificate 6989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6989 1) := by change ¬ Nat.Prime 195398463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65132821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6989 2) := by change ¬ Nat.Prime 195384485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39076897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6989 3) := by change ¬ Nat.Prime 195370507; exact PrattCertificates.composite_of_product (factor := 1693) (cofactor := 115399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6989 : row 6989 = 0 := (row_6989.checked).trans rfl
theorem model_count_6989 : modelRow 6989 = 0 := by decide
theorem modelrow_agrees_6989 : modelRow 6989 = row 6989 := model_count_6989.trans row_count_6989.symm
def row_6990 : RowCertificate 6990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6990 1) := by change ¬ Nat.Prime 195454381; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1012717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6990 2) := by change ¬ Nat.Prime 195440401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15033877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6990 3) := by change Nat.Prime 195426421; exact ProjectEulerSolutions.P58PrimeNodes.p_195426421.prime
    exact decide_eq_true prime
theorem row_count_6990 : row 6990 = 1 := (row_6990.checked).trans rfl
theorem model_count_6990 : modelRow 6990 = 1 := by decide
theorem modelrow_agrees_6990 : modelRow 6990 = row 6990 := model_count_6990.trans row_count_6990.symm
def row_6991 : RowCertificate 6991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6991 1) := by change ¬ Nat.Prime 195510307; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 3205087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6991 2) := by change ¬ Nat.Prime 195496325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39099265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6991 3) := by change ¬ Nat.Prime 195482343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65160781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6991 : row 6991 = 0 := (row_6991.checked).trans rfl
theorem model_count_6991 : modelRow 6991 = 0 := by decide
theorem modelrow_agrees_6991 : modelRow 6991 = row 6991 := model_count_6991.trans row_count_6991.symm
def row_6992 : RowCertificate 6992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6992 1) := by change ¬ Nat.Prime 195566241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65188747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6992 2) := by change Nat.Prime 195552257; exact ProjectEulerSolutions.P58PrimeNodes.p_195552257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6992 3) := by change ¬ Nat.Prime 195538273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27934039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6992 : row 6992 = 1 := (row_6992.checked).trans rfl
theorem model_count_6992 : modelRow 6992 = 1 := by decide
theorem modelrow_agrees_6992 : modelRow 6992 = row 6992 := model_count_6992.trans row_count_6992.symm
def row_6993 : RowCertificate 6993 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6993 1) := by change ¬ Nat.Prime 195622183; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6310393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6993 2) := by change ¬ Nat.Prime 195608197; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 435653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6993 3) := by change ¬ Nat.Prime 195594211; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1245823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6993 : row 6993 = 0 := (row_6993.checked).trans rfl
theorem model_count_6993 : modelRow 6993 = 0 := by decide
theorem modelrow_agrees_6993 : modelRow 6993 = row 6993 := model_count_6993.trans row_count_6993.symm
def row_6994 : RowCertificate 6994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6994 1) := by change ¬ Nat.Prime 195678133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27954019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6994 2) := by change ¬ Nat.Prime 195664145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39132829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6994 3) := by change ¬ Nat.Prime 195650157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65216719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6994 : row 6994 = 0 := (row_6994.checked).trans rfl
theorem model_count_6994 : modelRow 6994 = 0 := by decide
theorem modelrow_agrees_6994 : modelRow 6994 = row 6994 := model_count_6994.trans row_count_6994.symm
def row_6995 : RowCertificate 6995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6995 1) := by change ¬ Nat.Prime 195734091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65244697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6995 2) := by change ¬ Nat.Prime 195720101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6748969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6995 3) := by change ¬ Nat.Prime 195706111; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1540993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6995 : row 6995 = 0 := (row_6995.checked).trans rfl
theorem model_count_6995 : modelRow 6995 = 0 := by decide
theorem modelrow_agrees_6995 : modelRow 6995 = row 6995 := model_count_6995.trans row_count_6995.symm
def row_6996 : RowCertificate 6996 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6996 1) := by change Nat.Prime 195790057; exact ProjectEulerSolutions.P58PrimeNodes.p_195790057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6996 2) := by change ¬ Nat.Prime 195776065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39155213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6996 3) := by change ¬ Nat.Prime 195762073; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15058621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6996 : row 6996 = 1 := (row_6996.checked).trans rfl
theorem model_count_6996 : modelRow 6996 = 1 := by decide
theorem modelrow_agrees_6996 : modelRow 6996 = row 6996 := model_count_6996.trans row_count_6996.symm
def row_6997 : RowCertificate 6997 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6997 1) := by change Nat.Prime 195846031; exact ProjectEulerSolutions.P58PrimeNodes.p_195846031.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6997 2) := by change Nat.Prime 195832037; exact ProjectEulerSolutions.P58PrimeNodes.p_195832037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6997 3) := by change ¬ Nat.Prime 195818043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65272681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6997 : row 6997 = 2 := (row_6997.checked).trans rfl
theorem model_count_6997 : modelRow 6997 = 2 := by decide
theorem modelrow_agrees_6997 : modelRow 6997 = row 6997 := model_count_6997.trans row_count_6997.symm
def row_6998 : RowCertificate 6998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6998 1) := by change ¬ Nat.Prime 195902013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65300671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6998 2) := by change ¬ Nat.Prime 195888017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15068309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6998 3) := by change ¬ Nat.Prime 195874021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27982003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6998 : row 6998 = 0 := (row_6998.checked).trans rfl
theorem model_count_6998 : modelRow 6998 = 0 := by decide
theorem modelrow_agrees_6998 : modelRow 6998 = row 6998 := model_count_6998.trans row_count_6998.symm
def row_6999 : RowCertificate 6999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6999 1) := by change ¬ Nat.Prime 195958003; exact PrattCertificates.composite_of_product (factor := 8923) (cofactor := 21961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6999 2) := by change ¬ Nat.Prime 195944005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39188801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6999 3) := by change ¬ Nat.Prime 195930007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 27990001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6999 : row 6999 = 0 := (row_6999.checked).trans rfl
theorem model_count_6999 : modelRow 6999 = 0 := by decide
theorem modelrow_agrees_6999 : modelRow 6999 = row 6999 := model_count_6999.trans row_count_6999.symm
def row_7000 : RowCertificate 7000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7000 1) := by change ¬ Nat.Prime 196014001; exact PrattCertificates.composite_of_product (factor := 3433) (cofactor := 57097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7000 2) := by change Nat.Prime 196000001; exact ProjectEulerSolutions.P58PrimeNodes.p_196000001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7000 3) := by change ¬ Nat.Prime 195986001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65328667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7000 : row 7000 = 1 := (row_7000.checked).trans rfl
theorem model_count_7000 : modelRow 7000 = 1 := by decide
theorem modelrow_agrees_7000 : modelRow 7000 = row 7000 := model_count_7000.trans row_count_7000.symm
def row_7001 : RowCertificate 7001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7001 1) := by change ¬ Nat.Prime 196070007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65356669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7001 2) := by change ¬ Nat.Prime 196056005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39211201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7001 3) := by change Nat.Prime 196042003; exact ProjectEulerSolutions.P58PrimeNodes.p_196042003.prime
    exact decide_eq_true prime
theorem row_count_7001 : row 7001 = 1 := (row_7001.checked).trans rfl
theorem model_count_7001 : modelRow 7001 = 1 := by decide
theorem modelrow_agrees_7001 : modelRow 7001 = row 7001 := model_count_7001.trans row_count_7001.symm
def row_7002 : RowCertificate 7002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7002 1) := by change ¬ Nat.Prime 196126021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28018003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7002 2) := by change ¬ Nat.Prime 196112017; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11536001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7002 3) := by change ¬ Nat.Prime 196098013; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 2021629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7002 : row 7002 = 0 := (row_7002.checked).trans rfl
theorem model_count_7002 : modelRow 7002 = 0 := by decide
theorem modelrow_agrees_7002 : modelRow 7002 = row 7002 := model_count_7002.trans row_count_7002.symm
def row_7003 : RowCertificate 7003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7003 1) := by change ¬ Nat.Prime 196182043; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 6328453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7003 2) := by change ¬ Nat.Prime 196168037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15089849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7003 3) := by change ¬ Nat.Prime 196154031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65384677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7003 : row 7003 = 0 := (row_7003.checked).trans rfl
theorem model_count_7003 : modelRow 7003 = 0 := by decide
theorem modelrow_agrees_7003 : modelRow 7003 = row 7003 := model_count_7003.trans row_count_7003.symm
def row_7004 : RowCertificate 7004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7004 1) := by change ¬ Nat.Prime 196238073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65412691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7004 2) := by change ¬ Nat.Prime 196224065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39244813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7004 3) := by change ¬ Nat.Prime 196210057; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 2687809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7004 : row 7004 = 0 := (row_7004.checked).trans rfl
theorem model_count_7004 : modelRow 7004 = 0 := by decide
theorem modelrow_agrees_7004 : modelRow 7004 = row 7004 := model_count_7004.trans row_count_7004.symm
def row_7005 : RowCertificate 7005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7005 1) := by change ¬ Nat.Prime 196294111; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15099547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7005 2) := by change Nat.Prime 196280101; exact ProjectEulerSolutions.P58PrimeNodes.p_196280101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7005 3) := by change ¬ Nat.Prime 196266091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28038013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7005 : row 7005 = 1 := (row_7005.checked).trans rfl
theorem model_count_7005 : modelRow 7005 = 1 := by decide
theorem modelrow_agrees_7005 : modelRow 7005 = row 7005 := model_count_7005.trans row_count_7005.symm
def row_7006 : RowCertificate 7006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7006 1) := by change ¬ Nat.Prime 196350157; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5306761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7006 2) := by change ¬ Nat.Prime 196336145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39267229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7006 3) := by change ¬ Nat.Prime 196322133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65440711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7006 : row 7006 = 0 := (row_7006.checked).trans rfl
theorem model_count_7006 : modelRow 7006 = 0 := by decide
theorem modelrow_agrees_7006 : modelRow 7006 = row 7006 := model_count_7006.trans row_count_7006.symm
def row_7007 : RowCertificate 7007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7007 1) := by change ¬ Nat.Prime 196406211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65468737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7007 2) := by change ¬ Nat.Prime 196392197; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 363017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7007 3) := by change Nat.Prime 196378183; exact ProjectEulerSolutions.P58PrimeNodes.p_196378183.prime
    exact decide_eq_true prime
theorem row_count_7007 : row 7007 = 1 := (row_7007.checked).trans rfl
theorem model_count_7007 : modelRow 7007 = 1 := by decide
theorem modelrow_agrees_7007 : modelRow 7007 = row 7007 := model_count_7007.trans row_count_7007.symm
def row_7008 : RowCertificate 7008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7008 1) := by change ¬ Nat.Prime 196462273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28066039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7008 2) := by change Nat.Prime 196448257; exact ProjectEulerSolutions.P58PrimeNodes.p_196448257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7008 3) := by change ¬ Nat.Prime 196434241; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1802149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7008 : row 7008 = 1 := (row_7008.checked).trans rfl
theorem model_count_7008 : modelRow 7008 = 1 := by decide
theorem modelrow_agrees_7008 : modelRow 7008 = row 7008 := model_count_7008.trans row_count_7008.symm
def row_7009 : RowCertificate 7009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7009 1) := by change ¬ Nat.Prime 196518343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28074049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7009 2) := by change ¬ Nat.Prime 196504325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39300865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7009 3) := by change ¬ Nat.Prime 196490307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65496769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7009 : row 7009 = 0 := (row_7009.checked).trans rfl
theorem model_count_7009 : modelRow 7009 = 0 := by decide
theorem modelrow_agrees_7009 : modelRow 7009 = row 7009 := model_count_7009.trans row_count_7009.symm
def row_7010 : RowCertificate 7010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7010 1) := by change ¬ Nat.Prime 196574421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65524807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7010 2) := by change Nat.Prime 196560401; exact ProjectEulerSolutions.P58PrimeNodes.p_196560401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7010 3) := by change Nat.Prime 196546381; exact ProjectEulerSolutions.P58PrimeNodes.p_196546381.prime
    exact decide_eq_true prime
theorem row_count_7010 : row 7010 = 2 := (row_7010.checked).trans rfl
theorem model_count_7010 : modelRow 7010 = 2 := by decide
theorem modelrow_agrees_7010 : modelRow 7010 = row 7010 := model_count_7010.trans row_count_7010.symm
def row_7011 : RowCertificate 7011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7011 1) := by change ¬ Nat.Prime 196630507; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 988093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7011 2) := by change ¬ Nat.Prime 196616485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39323297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7011 3) := by change Nat.Prime 196602463; exact ProjectEulerSolutions.P58PrimeNodes.p_196602463.prime
    exact decide_eq_true prime
theorem row_count_7011 : row 7011 = 1 := (row_7011.checked).trans rfl
theorem model_count_7011 : modelRow 7011 = 1 := by decide
theorem modelrow_agrees_7011 : modelRow 7011 = row 7011 := model_count_7011.trans row_count_7011.symm
def row_7012 : RowCertificate 7012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7012 1) := by change ¬ Nat.Prime 196686601; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4574107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7012 2) := by change ¬ Nat.Prime 196672577; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6781813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7012 3) := by change ¬ Nat.Prime 196658553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65552851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7012 : row 7012 = 0 := (row_7012.checked).trans rfl
theorem model_count_7012 : modelRow 7012 = 0 := by decide
theorem modelrow_agrees_7012 : modelRow 7012 = row 7012 := model_count_7012.trans row_count_7012.symm
def row_7013 : RowCertificate 7013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7013 1) := by change ¬ Nat.Prime 196742703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65580901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7013 2) := by change Nat.Prime 196728677; exact ProjectEulerSolutions.P58PrimeNodes.p_196728677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7013 3) := by change ¬ Nat.Prime 196714651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28102093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7013 : row 7013 = 1 := (row_7013.checked).trans rfl
theorem model_count_7013 : modelRow 7013 = 1 := by decide
theorem modelrow_agrees_7013 : modelRow 7013 = row 7013 := model_count_7013.trans row_count_7013.symm
def row_7014 : RowCertificate 7014 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7014 1) := by change Nat.Prime 196798813; exact ProjectEulerSolutions.P58PrimeNodes.p_196798813.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7014 2) := by change ¬ Nat.Prime 196784785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39356957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7014 3) := by change Nat.Prime 196770757; exact ProjectEulerSolutions.P58PrimeNodes.p_196770757.prime
    exact decide_eq_true prime
theorem row_count_7014 : row 7014 = 2 := (row_7014.checked).trans rfl
theorem model_count_7014 : modelRow 7014 = 2 := by decide
theorem modelrow_agrees_7014 : modelRow 7014 = row 7014 := model_count_7014.trans row_count_7014.symm
def row_7015 : RowCertificate 7015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7015 1) := by change ¬ Nat.Prime 196854931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28122133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7015 2) := by change ¬ Nat.Prime 196840901; exact PrattCertificates.composite_of_product (factor := 7057) (cofactor := 27893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7015 3) := by change ¬ Nat.Prime 196826871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65608957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7015 : row 7015 = 0 := (row_7015.checked).trans rfl
theorem model_count_7015 : modelRow 7015 = 0 := by decide
theorem modelrow_agrees_7015 : modelRow 7015 = row 7015 := model_count_7015.trans row_count_7015.symm
def row_7016 : RowCertificate 7016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7016 1) := by change ¬ Nat.Prime 196911057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65637019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7016 2) := by change ¬ Nat.Prime 196897025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39379405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7016 3) := by change Nat.Prime 196882993; exact ProjectEulerSolutions.P58PrimeNodes.p_196882993.prime
    exact decide_eq_true prime
theorem row_count_7016 : row 7016 = 1 := (row_7016.checked).trans rfl
theorem model_count_7016 : modelRow 7016 = 1 := by decide
theorem modelrow_agrees_7016 : modelRow 7016 = row 7016 := model_count_7016.trans row_count_7016.symm
def row_7017 : RowCertificate 7017 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7017 1) := by change Nat.Prime 196967191; exact ProjectEulerSolutions.P58PrimeNodes.p_196967191.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7017 2) := by change Nat.Prime 196953157; exact ProjectEulerSolutions.P58PrimeNodes.p_196953157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7017 3) := by change ¬ Nat.Prime 196939123; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10365217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7017 : row 7017 = 2 := (row_7017.checked).trans rfl
theorem model_count_7017 : modelRow 7017 = 2 := by decide
theorem modelrow_agrees_7017 : modelRow 7017 = row 7017 := model_count_7017.trans row_count_7017.symm
def row_7018 : RowCertificate 7018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7018 1) := by change ¬ Nat.Prime 197023333; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15155641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7018 2) := by change Nat.Prime 197009297; exact ProjectEulerSolutions.P58PrimeNodes.p_197009297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7018 3) := by change ¬ Nat.Prime 196995261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65665087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7018 : row 7018 = 1 := (row_7018.checked).trans rfl
theorem model_count_7018 : modelRow 7018 = 1 := by decide
theorem modelrow_agrees_7018 : modelRow 7018 = row 7018 := model_count_7018.trans row_count_7018.symm
def row_7019 : RowCertificate 7019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7019 1) := by change ¬ Nat.Prime 197079483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65693161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7019 2) := by change ¬ Nat.Prime 197065445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39413089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7019 3) := by change ¬ Nat.Prime 197051407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28150201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7019 : row 7019 = 0 := (row_7019.checked).trans rfl
theorem model_count_7019 : modelRow 7019 = 0 := by decide
theorem modelrow_agrees_7019 : modelRow 7019 = row 7019 := model_count_7019.trans row_count_7019.symm
def row_7020 : RowCertificate 7020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7020 1) := by change ¬ Nat.Prime 197135641; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2942323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 7020 2) := by change Nat.Prime 197121601; exact ProjectEulerSolutions.P58PrimeNodes.p_197121601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7020 3) := by change ¬ Nat.Prime 197107561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28158223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7020 : row 7020 = 1 := (row_7020.checked).trans rfl
theorem model_count_7020 : modelRow 7020 = 1 := by decide
theorem modelrow_agrees_7020 : modelRow 7020 = row 7020 := model_count_7020.trans row_count_7020.symm
def row_7021 : RowCertificate 7021 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7021 1) := by change Nat.Prime 197191807; exact ProjectEulerSolutions.P58PrimeNodes.p_197191807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7021 2) := by change ¬ Nat.Prime 197177765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39435553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7021 3) := by change ¬ Nat.Prime 197163723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65721241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7021 : row 7021 = 1 := (row_7021.checked).trans rfl
theorem model_count_7021 : modelRow 7021 = 1 := by decide
theorem modelrow_agrees_7021 : modelRow 7021 = row 7021 := model_count_7021.trans row_count_7021.symm
def row_7022 : RowCertificate 7022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7022 1) := by change ¬ Nat.Prime 197247981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65749327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7022 2) := by change ¬ Nat.Prime 197233937; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 387493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7022 3) := by change ¬ Nat.Prime 197219893; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15170761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7022 : row 7022 = 0 := (row_7022.checked).trans rfl
theorem model_count_7022 : modelRow 7022 = 0 := by decide
theorem modelrow_agrees_7022 : modelRow 7022 = row 7022 := model_count_7022.trans row_count_7022.symm
def row_7023 : RowCertificate 7023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7023 1) := by change ¬ Nat.Prime 197304163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28186309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7023 2) := by change ¬ Nat.Prime 197290117; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11605301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7023 3) := by change ¬ Nat.Prime 197276071; exact PrattCertificates.composite_of_product (factor := 9619) (cofactor := 20509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7023 : row 7023 = 0 := (row_7023.checked).trans rfl
theorem model_count_7023 : modelRow 7023 = 0 := by decide
theorem modelrow_agrees_7023 : modelRow 7023 = row 7023 := model_count_7023.trans row_count_7023.symm
def row_7024 : RowCertificate 7024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7024 1) := by change ¬ Nat.Prime 197360353; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10387387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7024 2) := by change ¬ Nat.Prime 197346305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39469261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7024 3) := by change ¬ Nat.Prime 197332257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65777419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7024 : row 7024 = 0 := (row_7024.checked).trans rfl
theorem model_count_7024 : modelRow 7024 = 0 := by decide
theorem modelrow_agrees_7024 : modelRow 7024 = row 7024 := model_count_7024.trans row_count_7024.symm
def row_7025 : RowCertificate 7025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7025 1) := by change ¬ Nat.Prime 197416551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65805517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7025 2) := by change ¬ Nat.Prime 197402501; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 1324849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7025 3) := by change ¬ Nat.Prime 197388451; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15183727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7025 : row 7025 = 0 := (row_7025.checked).trans rfl
theorem model_count_7025 : modelRow 7025 = 0 := by decide
theorem modelrow_agrees_7025 : modelRow 7025 = row 7025 := model_count_7025.trans row_count_7025.symm
def row_7026 : RowCertificate 7026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7026 1) := by change ¬ Nat.Prime 197472757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 10393303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7026 2) := by change ¬ Nat.Prime 197458705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39491741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7026 3) := by change ¬ Nat.Prime 197444653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28206379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7026 : row 7026 = 0 := (row_7026.checked).trans rfl
theorem model_count_7026 : modelRow 7026 = 0 := by decide
theorem modelrow_agrees_7026 : modelRow 7026 = row 7026 := model_count_7026.trans row_count_7026.symm
def row_7027 : RowCertificate 7027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7027 1) := by change ¬ Nat.Prime 197528971; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4593697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7027 2) := by change ¬ Nat.Prime 197514917; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5338241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7027 3) := by change ¬ Nat.Prime 197500863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65833621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7027 : row 7027 = 0 := (row_7027.checked).trans rfl
theorem model_count_7027 : modelRow 7027 = 0 := by decide
theorem modelrow_agrees_7027 : modelRow 7027 = row 7027 := model_count_7027.trans row_count_7027.symm
def row_7028 : RowCertificate 7028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7028 1) := by change ¬ Nat.Prime 197585193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65861731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7028 2) := by change ¬ Nat.Prime 197571137; exact PrattCertificates.composite_of_product (factor := 11953) (cofactor := 16529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 7028 3) := by change Nat.Prime 197557081; exact ProjectEulerSolutions.P58PrimeNodes.p_197557081.prime
    exact decide_eq_true prime
theorem row_count_7028 : row 7028 = 1 := (row_7028.checked).trans rfl
theorem model_count_7028 : modelRow 7028 = 1 := by decide
theorem modelrow_agrees_7028 : modelRow 7028 = row 7028 := model_count_7028.trans row_count_7028.symm
def row_7029 : RowCertificate 7029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7029 1) := by change ¬ Nat.Prime 197641423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28234489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7029 2) := by change ¬ Nat.Prime 197627365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39525473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7029 3) := by change ¬ Nat.Prime 197613307; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 180799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7029 : row 7029 = 0 := (row_7029.checked).trans rfl
theorem model_count_7029 : modelRow 7029 = 0 := by decide
theorem modelrow_agrees_7029 : modelRow 7029 = row 7029 := model_count_7029.trans row_count_7029.symm
def row_7030 : RowCertificate 7030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7030 1) := by change ¬ Nat.Prime 197697661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28242523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7030 2) := by change ¬ Nat.Prime 197683601; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 631577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7030 3) := by change ¬ Nat.Prime 197669541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65889847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7030 : row 7030 = 0 := (row_7030.checked).trans rfl
theorem model_count_7030 : modelRow 7030 = 0 := by decide
theorem modelrow_agrees_7030 : modelRow 7030 = row 7030 := model_count_7030.trans row_count_7030.symm
def row_7031 : RowCertificate 7031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7031 1) := by change ¬ Nat.Prime 197753907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65917969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7031 2) := by change ¬ Nat.Prime 197739845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39547969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7031 3) := by change ¬ Nat.Prime 197725783; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 342679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7031 : row 7031 = 0 := (row_7031.checked).trans rfl
theorem model_count_7031 : modelRow 7031 = 0 := by decide
theorem modelrow_agrees_7031 : modelRow 7031 = row 7031 := model_count_7031.trans row_count_7031.symm
def row_7032 : RowCertificate 7032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7032 1) := by change ¬ Nat.Prime 197810161; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1920487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7032 2) := by change ¬ Nat.Prime 197796097; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 322669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7032 3) := by change ¬ Nat.Prime 197782033; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 863677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7032 : row 7032 = 0 := (row_7032.checked).trans rfl
theorem model_count_7032 : modelRow 7032 = 0 := by decide
theorem modelrow_agrees_7032 : modelRow 7032 = row 7032 := model_count_7032.trans row_count_7032.symm
def row_7033 : RowCertificate 7033 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7033 1) := by change Nat.Prime 197866423; exact ProjectEulerSolutions.P58PrimeNodes.p_197866423.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7033 2) := by change ¬ Nat.Prime 197852357; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5347361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7033 3) := by change ¬ Nat.Prime 197838291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65946097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7033 : row 7033 = 1 := (row_7033.checked).trans rfl
theorem model_count_7033 : modelRow 7033 = 1 := by decide
theorem modelrow_agrees_7033 : modelRow 7033 = row 7033 := model_count_7033.trans row_count_7033.symm
def row_7034 : RowCertificate 7034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7034 1) := by change ¬ Nat.Prime 197922693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 65974231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7034 2) := by change ¬ Nat.Prime 197908625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39581725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7034 3) := by change ¬ Nat.Prime 197894557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28270651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7034 : row 7034 = 0 := (row_7034.checked).trans rfl
theorem model_count_7034 : modelRow 7034 = 0 := by decide
theorem modelrow_agrees_7034 : modelRow 7034 = row 7034 := model_count_7034.trans row_count_7034.symm
def row_7035 : RowCertificate 7035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7035 1) := by change ¬ Nat.Prime 197978971; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 5350783) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7035 2) := by change ¬ Nat.Prime 197964901; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 735929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7035 3) := by change ¬ Nat.Prime 197950831; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15226987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7035 : row 7035 = 0 := (row_7035.checked).trans rfl
theorem model_count_7035 : modelRow 7035 = 0 := by decide
theorem modelrow_agrees_7035 : modelRow 7035 = row 7035 := model_count_7035.trans row_count_7035.symm
def row_7036 : RowCertificate 7036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7036 1) := by change ¬ Nat.Prime 198035257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28290751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7036 2) := by change ¬ Nat.Prime 198021185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39604237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7036 3) := by change ¬ Nat.Prime 198007113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66002371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7036 : row 7036 = 0 := (row_7036.checked).trans rfl
theorem model_count_7036 : modelRow 7036 = 0 := by decide
theorem modelrow_agrees_7036 : modelRow 7036 = row 7036 := model_count_7036.trans row_count_7036.symm
def row_7037 : RowCertificate 7037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7037 1) := by change ¬ Nat.Prime 198091551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66030517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7037 2) := by change ¬ Nat.Prime 198077477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15236729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7037 3) := by change ¬ Nat.Prime 198063403; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 286633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7037 : row 7037 = 0 := (row_7037.checked).trans rfl
theorem model_count_7037 : modelRow 7037 = 0 := by decide
theorem modelrow_agrees_7037 : modelRow 7037 = row 7037 := model_count_7037.trans row_count_7037.symm
def row_7038 : RowCertificate 7038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7038 1) := by change ¬ Nat.Prime 198147853; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1215631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7038 2) := by change ¬ Nat.Prime 198133777; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 865213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7038 3) := by change ¬ Nat.Prime 198119701; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 15239977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7038 : row 7038 = 0 := (row_7038.checked).trans rfl
theorem model_count_7038 : modelRow 7038 = 0 := by decide
theorem modelrow_agrees_7038 : modelRow 7038 = row 7038 := model_count_7038.trans row_count_7038.symm
def row_7039 : RowCertificate 7039 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 7039 1) := by change Nat.Prime 198204163; exact ProjectEulerSolutions.P58PrimeNodes.p_198204163.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7039 2) := by change ¬ Nat.Prime 198190085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 39638017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7039 3) := by change ¬ Nat.Prime 198176007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66058669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7039 : row 7039 = 1 := (row_7039.checked).trans rfl
theorem model_count_7039 : modelRow 7039 = 1 := by decide
theorem modelrow_agrees_7039 : modelRow 7039 = row 7039 := model_count_7039.trans row_count_7039.symm
def row_7040 : RowCertificate 7040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 7040 1) := by change ¬ Nat.Prime 198260481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 66086827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 7040 2) := by change ¬ Nat.Prime 198246401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 11661553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 7040 3) := by change ¬ Nat.Prime 198232321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 28318903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_7040 : row 7040 = 0 := (row_7040.checked).trans rfl
theorem model_count_7040 : modelRow 7040 = 0 := by decide
theorem modelrow_agrees_7040 : modelRow 7040 = row 7040 := model_count_7040.trans row_count_7040.symm
private def atomic_6913 : CountSegment 1 10 13120 6912 6913 2961 2961 := CountSegment.single (first := 6912) (initial := 2961) row_count_6913 (by decide)
private theorem atomic_6913_agreement : RowAgreement 6912 6913 := RowAgreement.single (first := 6912) modelrow_agrees_6913
private def atomic_6914 : CountSegment 1 10 13120 6913 6914 2961 2961 := CountSegment.single (first := 6913) (initial := 2961) row_count_6914 (by decide)
private theorem atomic_6914_agreement : RowAgreement 6913 6914 := RowAgreement.single (first := 6913) modelrow_agrees_6914
private def atomic_6915 : CountSegment 1 10 13120 6914 6915 2961 2962 := CountSegment.single (first := 6914) (initial := 2961) row_count_6915 (by decide)
private theorem atomic_6915_agreement : RowAgreement 6914 6915 := RowAgreement.single (first := 6914) modelrow_agrees_6915
private def atomic_6916 : CountSegment 1 10 13120 6915 6916 2962 2962 := CountSegment.single (first := 6915) (initial := 2962) row_count_6916 (by decide)
private theorem atomic_6916_agreement : RowAgreement 6915 6916 := RowAgreement.single (first := 6915) modelrow_agrees_6916
private def atomic_6917 : CountSegment 1 10 13120 6916 6917 2962 2962 := CountSegment.single (first := 6916) (initial := 2962) row_count_6917 (by decide)
private theorem atomic_6917_agreement : RowAgreement 6916 6917 := RowAgreement.single (first := 6916) modelrow_agrees_6917
private def atomic_6918 : CountSegment 1 10 13120 6917 6918 2962 2962 := CountSegment.single (first := 6917) (initial := 2962) row_count_6918 (by decide)
private theorem atomic_6918_agreement : RowAgreement 6917 6918 := RowAgreement.single (first := 6917) modelrow_agrees_6918
private def atomic_6919 : CountSegment 1 10 13120 6918 6919 2962 2962 := CountSegment.single (first := 6918) (initial := 2962) row_count_6919 (by decide)
private theorem atomic_6919_agreement : RowAgreement 6918 6919 := RowAgreement.single (first := 6918) modelrow_agrees_6919
private def atomic_6920 : CountSegment 1 10 13120 6919 6920 2962 2962 := CountSegment.single (first := 6919) (initial := 2962) row_count_6920 (by decide)
private theorem atomic_6920_agreement : RowAgreement 6919 6920 := RowAgreement.single (first := 6919) modelrow_agrees_6920
private def atomic_6921 : CountSegment 1 10 13120 6920 6921 2962 2963 := CountSegment.single (first := 6920) (initial := 2962) row_count_6921 (by decide)
private theorem atomic_6921_agreement : RowAgreement 6920 6921 := RowAgreement.single (first := 6920) modelrow_agrees_6921
private def atomic_6922 : CountSegment 1 10 13120 6921 6922 2963 2964 := CountSegment.single (first := 6921) (initial := 2963) row_count_6922 (by decide)
private theorem atomic_6922_agreement : RowAgreement 6921 6922 := RowAgreement.single (first := 6921) modelrow_agrees_6922
private def atomic_6923 : CountSegment 1 10 13120 6922 6923 2964 2966 := CountSegment.single (first := 6922) (initial := 2964) row_count_6923 (by decide)
private theorem atomic_6923_agreement : RowAgreement 6922 6923 := RowAgreement.single (first := 6922) modelrow_agrees_6923
private def atomic_6924 : CountSegment 1 10 13120 6923 6924 2966 2966 := CountSegment.single (first := 6923) (initial := 2966) row_count_6924 (by decide)
private theorem atomic_6924_agreement : RowAgreement 6923 6924 := RowAgreement.single (first := 6923) modelrow_agrees_6924
private def atomic_6925 : CountSegment 1 10 13120 6924 6925 2966 2966 := CountSegment.single (first := 6924) (initial := 2966) row_count_6925 (by decide)
private theorem atomic_6925_agreement : RowAgreement 6924 6925 := RowAgreement.single (first := 6924) modelrow_agrees_6925
private def atomic_6926 : CountSegment 1 10 13120 6925 6926 2966 2966 := CountSegment.single (first := 6925) (initial := 2966) row_count_6926 (by decide)
private theorem atomic_6926_agreement : RowAgreement 6925 6926 := RowAgreement.single (first := 6925) modelrow_agrees_6926
private def atomic_6927 : CountSegment 1 10 13120 6926 6927 2966 2967 := CountSegment.single (first := 6926) (initial := 2966) row_count_6927 (by decide)
private theorem atomic_6927_agreement : RowAgreement 6926 6927 := RowAgreement.single (first := 6926) modelrow_agrees_6927
private def atomic_6928 : CountSegment 1 10 13120 6927 6928 2967 2968 := CountSegment.single (first := 6927) (initial := 2967) row_count_6928 (by decide)
private theorem atomic_6928_agreement : RowAgreement 6927 6928 := RowAgreement.single (first := 6927) modelrow_agrees_6928
private def atomic_6929 : CountSegment 1 10 13120 6928 6929 2968 2968 := CountSegment.single (first := 6928) (initial := 2968) row_count_6929 (by decide)
private theorem atomic_6929_agreement : RowAgreement 6928 6929 := RowAgreement.single (first := 6928) modelrow_agrees_6929
private def atomic_6930 : CountSegment 1 10 13120 6929 6930 2968 2968 := CountSegment.single (first := 6929) (initial := 2968) row_count_6930 (by decide)
private theorem atomic_6930_agreement : RowAgreement 6929 6930 := RowAgreement.single (first := 6929) modelrow_agrees_6930
private def atomic_6931 : CountSegment 1 10 13120 6930 6931 2968 2968 := CountSegment.single (first := 6930) (initial := 2968) row_count_6931 (by decide)
private theorem atomic_6931_agreement : RowAgreement 6930 6931 := RowAgreement.single (first := 6930) modelrow_agrees_6931
private def atomic_6932 : CountSegment 1 10 13120 6931 6932 2968 2968 := CountSegment.single (first := 6931) (initial := 2968) row_count_6932 (by decide)
private theorem atomic_6932_agreement : RowAgreement 6931 6932 := RowAgreement.single (first := 6931) modelrow_agrees_6932
private def atomic_6933 : CountSegment 1 10 13120 6932 6933 2968 2969 := CountSegment.single (first := 6932) (initial := 2968) row_count_6933 (by decide)
private theorem atomic_6933_agreement : RowAgreement 6932 6933 := RowAgreement.single (first := 6932) modelrow_agrees_6933
private def atomic_6934 : CountSegment 1 10 13120 6933 6934 2969 2969 := CountSegment.single (first := 6933) (initial := 2969) row_count_6934 (by decide)
private theorem atomic_6934_agreement : RowAgreement 6933 6934 := RowAgreement.single (first := 6933) modelrow_agrees_6934
private def atomic_6935 : CountSegment 1 10 13120 6934 6935 2969 2970 := CountSegment.single (first := 6934) (initial := 2969) row_count_6935 (by decide)
private theorem atomic_6935_agreement : RowAgreement 6934 6935 := RowAgreement.single (first := 6934) modelrow_agrees_6935
private def atomic_6936 : CountSegment 1 10 13120 6935 6936 2970 2970 := CountSegment.single (first := 6935) (initial := 2970) row_count_6936 (by decide)
private theorem atomic_6936_agreement : RowAgreement 6935 6936 := RowAgreement.single (first := 6935) modelrow_agrees_6936
private def atomic_6937 : CountSegment 1 10 13120 6936 6937 2970 2970 := CountSegment.single (first := 6936) (initial := 2970) row_count_6937 (by decide)
private theorem atomic_6937_agreement : RowAgreement 6936 6937 := RowAgreement.single (first := 6936) modelrow_agrees_6937
private def atomic_6938 : CountSegment 1 10 13120 6937 6938 2970 2970 := CountSegment.single (first := 6937) (initial := 2970) row_count_6938 (by decide)
private theorem atomic_6938_agreement : RowAgreement 6937 6938 := RowAgreement.single (first := 6937) modelrow_agrees_6938
private def atomic_6939 : CountSegment 1 10 13120 6938 6939 2970 2970 := CountSegment.single (first := 6938) (initial := 2970) row_count_6939 (by decide)
private theorem atomic_6939_agreement : RowAgreement 6938 6939 := RowAgreement.single (first := 6938) modelrow_agrees_6939
private def atomic_6940 : CountSegment 1 10 13120 6939 6940 2970 2971 := CountSegment.single (first := 6939) (initial := 2970) row_count_6940 (by decide)
private theorem atomic_6940_agreement : RowAgreement 6939 6940 := RowAgreement.single (first := 6939) modelrow_agrees_6940
private def atomic_6941 : CountSegment 1 10 13120 6940 6941 2971 2971 := CountSegment.single (first := 6940) (initial := 2971) row_count_6941 (by decide)
private theorem atomic_6941_agreement : RowAgreement 6940 6941 := RowAgreement.single (first := 6940) modelrow_agrees_6941
private def atomic_6942 : CountSegment 1 10 13120 6941 6942 2971 2971 := CountSegment.single (first := 6941) (initial := 2971) row_count_6942 (by decide)
private theorem atomic_6942_agreement : RowAgreement 6941 6942 := RowAgreement.single (first := 6941) modelrow_agrees_6942
private def atomic_6943 : CountSegment 1 10 13120 6942 6943 2971 2973 := CountSegment.single (first := 6942) (initial := 2971) row_count_6943 (by decide)
private theorem atomic_6943_agreement : RowAgreement 6942 6943 := RowAgreement.single (first := 6942) modelrow_agrees_6943
private def atomic_6944 : CountSegment 1 10 13120 6943 6944 2973 2973 := CountSegment.single (first := 6943) (initial := 2973) row_count_6944 (by decide)
private theorem atomic_6944_agreement : RowAgreement 6943 6944 := RowAgreement.single (first := 6943) modelrow_agrees_6944
private def atomic_6945 : CountSegment 1 10 13120 6944 6945 2973 2973 := CountSegment.single (first := 6944) (initial := 2973) row_count_6945 (by decide)
private theorem atomic_6945_agreement : RowAgreement 6944 6945 := RowAgreement.single (first := 6944) modelrow_agrees_6945
private def atomic_6946 : CountSegment 1 10 13120 6945 6946 2973 2973 := CountSegment.single (first := 6945) (initial := 2973) row_count_6946 (by decide)
private theorem atomic_6946_agreement : RowAgreement 6945 6946 := RowAgreement.single (first := 6945) modelrow_agrees_6946
private def atomic_6947 : CountSegment 1 10 13120 6946 6947 2973 2973 := CountSegment.single (first := 6946) (initial := 2973) row_count_6947 (by decide)
private theorem atomic_6947_agreement : RowAgreement 6946 6947 := RowAgreement.single (first := 6946) modelrow_agrees_6947
private def atomic_6948 : CountSegment 1 10 13120 6947 6948 2973 2973 := CountSegment.single (first := 6947) (initial := 2973) row_count_6948 (by decide)
private theorem atomic_6948_agreement : RowAgreement 6947 6948 := RowAgreement.single (first := 6947) modelrow_agrees_6948
private def atomic_6949 : CountSegment 1 10 13120 6948 6949 2973 2973 := CountSegment.single (first := 6948) (initial := 2973) row_count_6949 (by decide)
private theorem atomic_6949_agreement : RowAgreement 6948 6949 := RowAgreement.single (first := 6948) modelrow_agrees_6949
private def atomic_6950 : CountSegment 1 10 13120 6949 6950 2973 2974 := CountSegment.single (first := 6949) (initial := 2973) row_count_6950 (by decide)
private theorem atomic_6950_agreement : RowAgreement 6949 6950 := RowAgreement.single (first := 6949) modelrow_agrees_6950
private def atomic_6951 : CountSegment 1 10 13120 6950 6951 2974 2975 := CountSegment.single (first := 6950) (initial := 2974) row_count_6951 (by decide)
private theorem atomic_6951_agreement : RowAgreement 6950 6951 := RowAgreement.single (first := 6950) modelrow_agrees_6951
private def atomic_6952 : CountSegment 1 10 13120 6951 6952 2975 2975 := CountSegment.single (first := 6951) (initial := 2975) row_count_6952 (by decide)
private theorem atomic_6952_agreement : RowAgreement 6951 6952 := RowAgreement.single (first := 6951) modelrow_agrees_6952
private def atomic_6953 : CountSegment 1 10 13120 6952 6953 2975 2976 := CountSegment.single (first := 6952) (initial := 2975) row_count_6953 (by decide)
private theorem atomic_6953_agreement : RowAgreement 6952 6953 := RowAgreement.single (first := 6952) modelrow_agrees_6953
private def atomic_6954 : CountSegment 1 10 13120 6953 6954 2976 2977 := CountSegment.single (first := 6953) (initial := 2976) row_count_6954 (by decide)
private theorem atomic_6954_agreement : RowAgreement 6953 6954 := RowAgreement.single (first := 6953) modelrow_agrees_6954
private def atomic_6955 : CountSegment 1 10 13120 6954 6955 2977 2978 := CountSegment.single (first := 6954) (initial := 2977) row_count_6955 (by decide)
private theorem atomic_6955_agreement : RowAgreement 6954 6955 := RowAgreement.single (first := 6954) modelrow_agrees_6955
private def atomic_6956 : CountSegment 1 10 13120 6955 6956 2978 2978 := CountSegment.single (first := 6955) (initial := 2978) row_count_6956 (by decide)
private theorem atomic_6956_agreement : RowAgreement 6955 6956 := RowAgreement.single (first := 6955) modelrow_agrees_6956
private def atomic_6957 : CountSegment 1 10 13120 6956 6957 2978 2979 := CountSegment.single (first := 6956) (initial := 2978) row_count_6957 (by decide)
private theorem atomic_6957_agreement : RowAgreement 6956 6957 := RowAgreement.single (first := 6956) modelrow_agrees_6957
private def atomic_6958 : CountSegment 1 10 13120 6957 6958 2979 2980 := CountSegment.single (first := 6957) (initial := 2979) row_count_6958 (by decide)
private theorem atomic_6958_agreement : RowAgreement 6957 6958 := RowAgreement.single (first := 6957) modelrow_agrees_6958
private def atomic_6959 : CountSegment 1 10 13120 6958 6959 2980 2980 := CountSegment.single (first := 6958) (initial := 2980) row_count_6959 (by decide)
private theorem atomic_6959_agreement : RowAgreement 6958 6959 := RowAgreement.single (first := 6958) modelrow_agrees_6959
private def atomic_6960 : CountSegment 1 10 13120 6959 6960 2980 2980 := CountSegment.single (first := 6959) (initial := 2980) row_count_6960 (by decide)
private theorem atomic_6960_agreement : RowAgreement 6959 6960 := RowAgreement.single (first := 6959) modelrow_agrees_6960
private def atomic_6961 : CountSegment 1 10 13120 6960 6961 2980 2980 := CountSegment.single (first := 6960) (initial := 2980) row_count_6961 (by decide)
private theorem atomic_6961_agreement : RowAgreement 6960 6961 := RowAgreement.single (first := 6960) modelrow_agrees_6961
private def atomic_6962 : CountSegment 1 10 13120 6961 6962 2980 2982 := CountSegment.single (first := 6961) (initial := 2980) row_count_6962 (by decide)
private theorem atomic_6962_agreement : RowAgreement 6961 6962 := RowAgreement.single (first := 6961) modelrow_agrees_6962
private def atomic_6963 : CountSegment 1 10 13120 6962 6963 2982 2982 := CountSegment.single (first := 6962) (initial := 2982) row_count_6963 (by decide)
private theorem atomic_6963_agreement : RowAgreement 6962 6963 := RowAgreement.single (first := 6962) modelrow_agrees_6963
private def atomic_6964 : CountSegment 1 10 13120 6963 6964 2982 2983 := CountSegment.single (first := 6963) (initial := 2982) row_count_6964 (by decide)
private theorem atomic_6964_agreement : RowAgreement 6963 6964 := RowAgreement.single (first := 6963) modelrow_agrees_6964
private def atomic_6965 : CountSegment 1 10 13120 6964 6965 2983 2983 := CountSegment.single (first := 6964) (initial := 2983) row_count_6965 (by decide)
private theorem atomic_6965_agreement : RowAgreement 6964 6965 := RowAgreement.single (first := 6964) modelrow_agrees_6965
private def atomic_6966 : CountSegment 1 10 13120 6965 6966 2983 2984 := CountSegment.single (first := 6965) (initial := 2983) row_count_6966 (by decide)
private theorem atomic_6966_agreement : RowAgreement 6965 6966 := RowAgreement.single (first := 6965) modelrow_agrees_6966
private def atomic_6967 : CountSegment 1 10 13120 6966 6967 2984 2984 := CountSegment.single (first := 6966) (initial := 2984) row_count_6967 (by decide)
private theorem atomic_6967_agreement : RowAgreement 6966 6967 := RowAgreement.single (first := 6966) modelrow_agrees_6967
private def atomic_6968 : CountSegment 1 10 13120 6967 6968 2984 2984 := CountSegment.single (first := 6967) (initial := 2984) row_count_6968 (by decide)
private theorem atomic_6968_agreement : RowAgreement 6967 6968 := RowAgreement.single (first := 6967) modelrow_agrees_6968
private def atomic_6969 : CountSegment 1 10 13120 6968 6969 2984 2985 := CountSegment.single (first := 6968) (initial := 2984) row_count_6969 (by decide)
private theorem atomic_6969_agreement : RowAgreement 6968 6969 := RowAgreement.single (first := 6968) modelrow_agrees_6969
private def atomic_6970 : CountSegment 1 10 13120 6969 6970 2985 2986 := CountSegment.single (first := 6969) (initial := 2985) row_count_6970 (by decide)
private theorem atomic_6970_agreement : RowAgreement 6969 6970 := RowAgreement.single (first := 6969) modelrow_agrees_6970
private def atomic_6971 : CountSegment 1 10 13120 6970 6971 2986 2986 := CountSegment.single (first := 6970) (initial := 2986) row_count_6971 (by decide)
private theorem atomic_6971_agreement : RowAgreement 6970 6971 := RowAgreement.single (first := 6970) modelrow_agrees_6971
private def atomic_6972 : CountSegment 1 10 13120 6971 6972 2986 2986 := CountSegment.single (first := 6971) (initial := 2986) row_count_6972 (by decide)
private theorem atomic_6972_agreement : RowAgreement 6971 6972 := RowAgreement.single (first := 6971) modelrow_agrees_6972
private def atomic_6973 : CountSegment 1 10 13120 6972 6973 2986 2986 := CountSegment.single (first := 6972) (initial := 2986) row_count_6973 (by decide)
private theorem atomic_6973_agreement : RowAgreement 6972 6973 := RowAgreement.single (first := 6972) modelrow_agrees_6973
private def atomic_6974 : CountSegment 1 10 13120 6973 6974 2986 2987 := CountSegment.single (first := 6973) (initial := 2986) row_count_6974 (by decide)
private theorem atomic_6974_agreement : RowAgreement 6973 6974 := RowAgreement.single (first := 6973) modelrow_agrees_6974
private def atomic_6975 : CountSegment 1 10 13120 6974 6975 2987 2988 := CountSegment.single (first := 6974) (initial := 2987) row_count_6975 (by decide)
private theorem atomic_6975_agreement : RowAgreement 6974 6975 := RowAgreement.single (first := 6974) modelrow_agrees_6975
private def atomic_6976 : CountSegment 1 10 13120 6975 6976 2988 2988 := CountSegment.single (first := 6975) (initial := 2988) row_count_6976 (by decide)
private theorem atomic_6976_agreement : RowAgreement 6975 6976 := RowAgreement.single (first := 6975) modelrow_agrees_6976
private def atomic_6977 : CountSegment 1 10 13120 6976 6977 2988 2988 := CountSegment.single (first := 6976) (initial := 2988) row_count_6977 (by decide)
private theorem atomic_6977_agreement : RowAgreement 6976 6977 := RowAgreement.single (first := 6976) modelrow_agrees_6977
private def atomic_6978 : CountSegment 1 10 13120 6977 6978 2988 2988 := CountSegment.single (first := 6977) (initial := 2988) row_count_6978 (by decide)
private theorem atomic_6978_agreement : RowAgreement 6977 6978 := RowAgreement.single (first := 6977) modelrow_agrees_6978
private def atomic_6979 : CountSegment 1 10 13120 6978 6979 2988 2988 := CountSegment.single (first := 6978) (initial := 2988) row_count_6979 (by decide)
private theorem atomic_6979_agreement : RowAgreement 6978 6979 := RowAgreement.single (first := 6978) modelrow_agrees_6979
private def atomic_6980 : CountSegment 1 10 13120 6979 6980 2988 2988 := CountSegment.single (first := 6979) (initial := 2988) row_count_6980 (by decide)
private theorem atomic_6980_agreement : RowAgreement 6979 6980 := RowAgreement.single (first := 6979) modelrow_agrees_6980
private def atomic_6981 : CountSegment 1 10 13120 6980 6981 2988 2988 := CountSegment.single (first := 6980) (initial := 2988) row_count_6981 (by decide)
private theorem atomic_6981_agreement : RowAgreement 6980 6981 := RowAgreement.single (first := 6980) modelrow_agrees_6981
private def atomic_6982 : CountSegment 1 10 13120 6981 6982 2988 2989 := CountSegment.single (first := 6981) (initial := 2988) row_count_6982 (by decide)
private theorem atomic_6982_agreement : RowAgreement 6981 6982 := RowAgreement.single (first := 6981) modelrow_agrees_6982
private def atomic_6983 : CountSegment 1 10 13120 6982 6983 2989 2989 := CountSegment.single (first := 6982) (initial := 2989) row_count_6983 (by decide)
private theorem atomic_6983_agreement : RowAgreement 6982 6983 := RowAgreement.single (first := 6982) modelrow_agrees_6983
private def atomic_6984 : CountSegment 1 10 13120 6983 6984 2989 2989 := CountSegment.single (first := 6983) (initial := 2989) row_count_6984 (by decide)
private theorem atomic_6984_agreement : RowAgreement 6983 6984 := RowAgreement.single (first := 6983) modelrow_agrees_6984
private def atomic_6985 : CountSegment 1 10 13120 6984 6985 2989 2990 := CountSegment.single (first := 6984) (initial := 2989) row_count_6985 (by decide)
private theorem atomic_6985_agreement : RowAgreement 6984 6985 := RowAgreement.single (first := 6984) modelrow_agrees_6985
private def atomic_6986 : CountSegment 1 10 13120 6985 6986 2990 2990 := CountSegment.single (first := 6985) (initial := 2990) row_count_6986 (by decide)
private theorem atomic_6986_agreement : RowAgreement 6985 6986 := RowAgreement.single (first := 6985) modelrow_agrees_6986
private def atomic_6987 : CountSegment 1 10 13120 6986 6987 2990 2990 := CountSegment.single (first := 6986) (initial := 2990) row_count_6987 (by decide)
private theorem atomic_6987_agreement : RowAgreement 6986 6987 := RowAgreement.single (first := 6986) modelrow_agrees_6987
private def atomic_6988 : CountSegment 1 10 13120 6987 6988 2990 2990 := CountSegment.single (first := 6987) (initial := 2990) row_count_6988 (by decide)
private theorem atomic_6988_agreement : RowAgreement 6987 6988 := RowAgreement.single (first := 6987) modelrow_agrees_6988
private def atomic_6989 : CountSegment 1 10 13120 6988 6989 2990 2990 := CountSegment.single (first := 6988) (initial := 2990) row_count_6989 (by decide)
private theorem atomic_6989_agreement : RowAgreement 6988 6989 := RowAgreement.single (first := 6988) modelrow_agrees_6989
private def atomic_6990 : CountSegment 1 10 13120 6989 6990 2990 2991 := CountSegment.single (first := 6989) (initial := 2990) row_count_6990 (by decide)
private theorem atomic_6990_agreement : RowAgreement 6989 6990 := RowAgreement.single (first := 6989) modelrow_agrees_6990
private def atomic_6991 : CountSegment 1 10 13120 6990 6991 2991 2991 := CountSegment.single (first := 6990) (initial := 2991) row_count_6991 (by decide)
private theorem atomic_6991_agreement : RowAgreement 6990 6991 := RowAgreement.single (first := 6990) modelrow_agrees_6991
private def atomic_6992 : CountSegment 1 10 13120 6991 6992 2991 2992 := CountSegment.single (first := 6991) (initial := 2991) row_count_6992 (by decide)
private theorem atomic_6992_agreement : RowAgreement 6991 6992 := RowAgreement.single (first := 6991) modelrow_agrees_6992
private def atomic_6993 : CountSegment 1 10 13120 6992 6993 2992 2992 := CountSegment.single (first := 6992) (initial := 2992) row_count_6993 (by decide)
private theorem atomic_6993_agreement : RowAgreement 6992 6993 := RowAgreement.single (first := 6992) modelrow_agrees_6993
private def atomic_6994 : CountSegment 1 10 13120 6993 6994 2992 2992 := CountSegment.single (first := 6993) (initial := 2992) row_count_6994 (by decide)
private theorem atomic_6994_agreement : RowAgreement 6993 6994 := RowAgreement.single (first := 6993) modelrow_agrees_6994
private def atomic_6995 : CountSegment 1 10 13120 6994 6995 2992 2992 := CountSegment.single (first := 6994) (initial := 2992) row_count_6995 (by decide)
private theorem atomic_6995_agreement : RowAgreement 6994 6995 := RowAgreement.single (first := 6994) modelrow_agrees_6995
private def atomic_6996 : CountSegment 1 10 13120 6995 6996 2992 2993 := CountSegment.single (first := 6995) (initial := 2992) row_count_6996 (by decide)
private theorem atomic_6996_agreement : RowAgreement 6995 6996 := RowAgreement.single (first := 6995) modelrow_agrees_6996
private def atomic_6997 : CountSegment 1 10 13120 6996 6997 2993 2995 := CountSegment.single (first := 6996) (initial := 2993) row_count_6997 (by decide)
private theorem atomic_6997_agreement : RowAgreement 6996 6997 := RowAgreement.single (first := 6996) modelrow_agrees_6997
private def atomic_6998 : CountSegment 1 10 13120 6997 6998 2995 2995 := CountSegment.single (first := 6997) (initial := 2995) row_count_6998 (by decide)
private theorem atomic_6998_agreement : RowAgreement 6997 6998 := RowAgreement.single (first := 6997) modelrow_agrees_6998
private def atomic_6999 : CountSegment 1 10 13120 6998 6999 2995 2995 := CountSegment.single (first := 6998) (initial := 2995) row_count_6999 (by decide)
private theorem atomic_6999_agreement : RowAgreement 6998 6999 := RowAgreement.single (first := 6998) modelrow_agrees_6999
private def atomic_7000 : CountSegment 1 10 13120 6999 7000 2995 2996 := CountSegment.single (first := 6999) (initial := 2995) row_count_7000 (by decide)
private theorem atomic_7000_agreement : RowAgreement 6999 7000 := RowAgreement.single (first := 6999) modelrow_agrees_7000
private def atomic_7001 : CountSegment 1 10 13120 7000 7001 2996 2997 := CountSegment.single (first := 7000) (initial := 2996) row_count_7001 (by decide)
private theorem atomic_7001_agreement : RowAgreement 7000 7001 := RowAgreement.single (first := 7000) modelrow_agrees_7001
private def atomic_7002 : CountSegment 1 10 13120 7001 7002 2997 2997 := CountSegment.single (first := 7001) (initial := 2997) row_count_7002 (by decide)
private theorem atomic_7002_agreement : RowAgreement 7001 7002 := RowAgreement.single (first := 7001) modelrow_agrees_7002
private def atomic_7003 : CountSegment 1 10 13120 7002 7003 2997 2997 := CountSegment.single (first := 7002) (initial := 2997) row_count_7003 (by decide)
private theorem atomic_7003_agreement : RowAgreement 7002 7003 := RowAgreement.single (first := 7002) modelrow_agrees_7003
private def atomic_7004 : CountSegment 1 10 13120 7003 7004 2997 2997 := CountSegment.single (first := 7003) (initial := 2997) row_count_7004 (by decide)
private theorem atomic_7004_agreement : RowAgreement 7003 7004 := RowAgreement.single (first := 7003) modelrow_agrees_7004
private def atomic_7005 : CountSegment 1 10 13120 7004 7005 2997 2998 := CountSegment.single (first := 7004) (initial := 2997) row_count_7005 (by decide)
private theorem atomic_7005_agreement : RowAgreement 7004 7005 := RowAgreement.single (first := 7004) modelrow_agrees_7005
private def atomic_7006 : CountSegment 1 10 13120 7005 7006 2998 2998 := CountSegment.single (first := 7005) (initial := 2998) row_count_7006 (by decide)
private theorem atomic_7006_agreement : RowAgreement 7005 7006 := RowAgreement.single (first := 7005) modelrow_agrees_7006
private def atomic_7007 : CountSegment 1 10 13120 7006 7007 2998 2999 := CountSegment.single (first := 7006) (initial := 2998) row_count_7007 (by decide)
private theorem atomic_7007_agreement : RowAgreement 7006 7007 := RowAgreement.single (first := 7006) modelrow_agrees_7007
private def atomic_7008 : CountSegment 1 10 13120 7007 7008 2999 3000 := CountSegment.single (first := 7007) (initial := 2999) row_count_7008 (by decide)
private theorem atomic_7008_agreement : RowAgreement 7007 7008 := RowAgreement.single (first := 7007) modelrow_agrees_7008
private def atomic_7009 : CountSegment 1 10 13120 7008 7009 3000 3000 := CountSegment.single (first := 7008) (initial := 3000) row_count_7009 (by decide)
private theorem atomic_7009_agreement : RowAgreement 7008 7009 := RowAgreement.single (first := 7008) modelrow_agrees_7009
private def atomic_7010 : CountSegment 1 10 13120 7009 7010 3000 3002 := CountSegment.single (first := 7009) (initial := 3000) row_count_7010 (by decide)
private theorem atomic_7010_agreement : RowAgreement 7009 7010 := RowAgreement.single (first := 7009) modelrow_agrees_7010
private def atomic_7011 : CountSegment 1 10 13120 7010 7011 3002 3003 := CountSegment.single (first := 7010) (initial := 3002) row_count_7011 (by decide)
private theorem atomic_7011_agreement : RowAgreement 7010 7011 := RowAgreement.single (first := 7010) modelrow_agrees_7011
private def atomic_7012 : CountSegment 1 10 13120 7011 7012 3003 3003 := CountSegment.single (first := 7011) (initial := 3003) row_count_7012 (by decide)
private theorem atomic_7012_agreement : RowAgreement 7011 7012 := RowAgreement.single (first := 7011) modelrow_agrees_7012
private def atomic_7013 : CountSegment 1 10 13120 7012 7013 3003 3004 := CountSegment.single (first := 7012) (initial := 3003) row_count_7013 (by decide)
private theorem atomic_7013_agreement : RowAgreement 7012 7013 := RowAgreement.single (first := 7012) modelrow_agrees_7013
private def atomic_7014 : CountSegment 1 10 13120 7013 7014 3004 3006 := CountSegment.single (first := 7013) (initial := 3004) row_count_7014 (by decide)
private theorem atomic_7014_agreement : RowAgreement 7013 7014 := RowAgreement.single (first := 7013) modelrow_agrees_7014
private def atomic_7015 : CountSegment 1 10 13120 7014 7015 3006 3006 := CountSegment.single (first := 7014) (initial := 3006) row_count_7015 (by decide)
private theorem atomic_7015_agreement : RowAgreement 7014 7015 := RowAgreement.single (first := 7014) modelrow_agrees_7015
private def atomic_7016 : CountSegment 1 10 13120 7015 7016 3006 3007 := CountSegment.single (first := 7015) (initial := 3006) row_count_7016 (by decide)
private theorem atomic_7016_agreement : RowAgreement 7015 7016 := RowAgreement.single (first := 7015) modelrow_agrees_7016
private def atomic_7017 : CountSegment 1 10 13120 7016 7017 3007 3009 := CountSegment.single (first := 7016) (initial := 3007) row_count_7017 (by decide)
private theorem atomic_7017_agreement : RowAgreement 7016 7017 := RowAgreement.single (first := 7016) modelrow_agrees_7017
private def atomic_7018 : CountSegment 1 10 13120 7017 7018 3009 3010 := CountSegment.single (first := 7017) (initial := 3009) row_count_7018 (by decide)
private theorem atomic_7018_agreement : RowAgreement 7017 7018 := RowAgreement.single (first := 7017) modelrow_agrees_7018
private def atomic_7019 : CountSegment 1 10 13120 7018 7019 3010 3010 := CountSegment.single (first := 7018) (initial := 3010) row_count_7019 (by decide)
private theorem atomic_7019_agreement : RowAgreement 7018 7019 := RowAgreement.single (first := 7018) modelrow_agrees_7019
private def atomic_7020 : CountSegment 1 10 13120 7019 7020 3010 3011 := CountSegment.single (first := 7019) (initial := 3010) row_count_7020 (by decide)
private theorem atomic_7020_agreement : RowAgreement 7019 7020 := RowAgreement.single (first := 7019) modelrow_agrees_7020
private def atomic_7021 : CountSegment 1 10 13120 7020 7021 3011 3012 := CountSegment.single (first := 7020) (initial := 3011) row_count_7021 (by decide)
private theorem atomic_7021_agreement : RowAgreement 7020 7021 := RowAgreement.single (first := 7020) modelrow_agrees_7021
private def atomic_7022 : CountSegment 1 10 13120 7021 7022 3012 3012 := CountSegment.single (first := 7021) (initial := 3012) row_count_7022 (by decide)
private theorem atomic_7022_agreement : RowAgreement 7021 7022 := RowAgreement.single (first := 7021) modelrow_agrees_7022
private def atomic_7023 : CountSegment 1 10 13120 7022 7023 3012 3012 := CountSegment.single (first := 7022) (initial := 3012) row_count_7023 (by decide)
private theorem atomic_7023_agreement : RowAgreement 7022 7023 := RowAgreement.single (first := 7022) modelrow_agrees_7023
private def atomic_7024 : CountSegment 1 10 13120 7023 7024 3012 3012 := CountSegment.single (first := 7023) (initial := 3012) row_count_7024 (by decide)
private theorem atomic_7024_agreement : RowAgreement 7023 7024 := RowAgreement.single (first := 7023) modelrow_agrees_7024
private def atomic_7025 : CountSegment 1 10 13120 7024 7025 3012 3012 := CountSegment.single (first := 7024) (initial := 3012) row_count_7025 (by decide)
private theorem atomic_7025_agreement : RowAgreement 7024 7025 := RowAgreement.single (first := 7024) modelrow_agrees_7025
private def atomic_7026 : CountSegment 1 10 13120 7025 7026 3012 3012 := CountSegment.single (first := 7025) (initial := 3012) row_count_7026 (by decide)
private theorem atomic_7026_agreement : RowAgreement 7025 7026 := RowAgreement.single (first := 7025) modelrow_agrees_7026
private def atomic_7027 : CountSegment 1 10 13120 7026 7027 3012 3012 := CountSegment.single (first := 7026) (initial := 3012) row_count_7027 (by decide)
private theorem atomic_7027_agreement : RowAgreement 7026 7027 := RowAgreement.single (first := 7026) modelrow_agrees_7027
private def atomic_7028 : CountSegment 1 10 13120 7027 7028 3012 3013 := CountSegment.single (first := 7027) (initial := 3012) row_count_7028 (by decide)
private theorem atomic_7028_agreement : RowAgreement 7027 7028 := RowAgreement.single (first := 7027) modelrow_agrees_7028
private def atomic_7029 : CountSegment 1 10 13120 7028 7029 3013 3013 := CountSegment.single (first := 7028) (initial := 3013) row_count_7029 (by decide)
private theorem atomic_7029_agreement : RowAgreement 7028 7029 := RowAgreement.single (first := 7028) modelrow_agrees_7029
private def atomic_7030 : CountSegment 1 10 13120 7029 7030 3013 3013 := CountSegment.single (first := 7029) (initial := 3013) row_count_7030 (by decide)
private theorem atomic_7030_agreement : RowAgreement 7029 7030 := RowAgreement.single (first := 7029) modelrow_agrees_7030
private def atomic_7031 : CountSegment 1 10 13120 7030 7031 3013 3013 := CountSegment.single (first := 7030) (initial := 3013) row_count_7031 (by decide)
private theorem atomic_7031_agreement : RowAgreement 7030 7031 := RowAgreement.single (first := 7030) modelrow_agrees_7031
private def atomic_7032 : CountSegment 1 10 13120 7031 7032 3013 3013 := CountSegment.single (first := 7031) (initial := 3013) row_count_7032 (by decide)
private theorem atomic_7032_agreement : RowAgreement 7031 7032 := RowAgreement.single (first := 7031) modelrow_agrees_7032
private def atomic_7033 : CountSegment 1 10 13120 7032 7033 3013 3014 := CountSegment.single (first := 7032) (initial := 3013) row_count_7033 (by decide)
private theorem atomic_7033_agreement : RowAgreement 7032 7033 := RowAgreement.single (first := 7032) modelrow_agrees_7033
private def atomic_7034 : CountSegment 1 10 13120 7033 7034 3014 3014 := CountSegment.single (first := 7033) (initial := 3014) row_count_7034 (by decide)
private theorem atomic_7034_agreement : RowAgreement 7033 7034 := RowAgreement.single (first := 7033) modelrow_agrees_7034
private def atomic_7035 : CountSegment 1 10 13120 7034 7035 3014 3014 := CountSegment.single (first := 7034) (initial := 3014) row_count_7035 (by decide)
private theorem atomic_7035_agreement : RowAgreement 7034 7035 := RowAgreement.single (first := 7034) modelrow_agrees_7035
private def atomic_7036 : CountSegment 1 10 13120 7035 7036 3014 3014 := CountSegment.single (first := 7035) (initial := 3014) row_count_7036 (by decide)
private theorem atomic_7036_agreement : RowAgreement 7035 7036 := RowAgreement.single (first := 7035) modelrow_agrees_7036
private def atomic_7037 : CountSegment 1 10 13120 7036 7037 3014 3014 := CountSegment.single (first := 7036) (initial := 3014) row_count_7037 (by decide)
private theorem atomic_7037_agreement : RowAgreement 7036 7037 := RowAgreement.single (first := 7036) modelrow_agrees_7037
private def atomic_7038 : CountSegment 1 10 13120 7037 7038 3014 3014 := CountSegment.single (first := 7037) (initial := 3014) row_count_7038 (by decide)
private theorem atomic_7038_agreement : RowAgreement 7037 7038 := RowAgreement.single (first := 7037) modelrow_agrees_7038
private def atomic_7039 : CountSegment 1 10 13120 7038 7039 3014 3015 := CountSegment.single (first := 7038) (initial := 3014) row_count_7039 (by decide)
private theorem atomic_7039_agreement : RowAgreement 7038 7039 := RowAgreement.single (first := 7038) modelrow_agrees_7039
private def atomic_7040 : CountSegment 1 10 13120 7039 7040 3015 3015 := CountSegment.single (first := 7039) (initial := 3015) row_count_7040 (by decide)
private theorem atomic_7040_agreement : RowAgreement 7039 7040 := RowAgreement.single (first := 7039) modelrow_agrees_7040
private def within_054_000 : CountSegment 1 10 13120 6912 6914 2961 2961 := CountSegment.join atomic_6913 atomic_6914
private theorem within_054_000_agreement : RowAgreement 6912 6914 := RowAgreement.join atomic_6913_agreement atomic_6914_agreement
private def within_054_001 : CountSegment 1 10 13120 6914 6916 2961 2962 := CountSegment.join atomic_6915 atomic_6916
private theorem within_054_001_agreement : RowAgreement 6914 6916 := RowAgreement.join atomic_6915_agreement atomic_6916_agreement
private def within_054_002 : CountSegment 1 10 13120 6912 6916 2961 2962 := CountSegment.join within_054_000 within_054_001
private theorem within_054_002_agreement : RowAgreement 6912 6916 := RowAgreement.join within_054_000_agreement within_054_001_agreement
private def within_054_003 : CountSegment 1 10 13120 6916 6918 2962 2962 := CountSegment.join atomic_6917 atomic_6918
private theorem within_054_003_agreement : RowAgreement 6916 6918 := RowAgreement.join atomic_6917_agreement atomic_6918_agreement
private def within_054_004 : CountSegment 1 10 13120 6918 6920 2962 2962 := CountSegment.join atomic_6919 atomic_6920
private theorem within_054_004_agreement : RowAgreement 6918 6920 := RowAgreement.join atomic_6919_agreement atomic_6920_agreement
private def within_054_005 : CountSegment 1 10 13120 6916 6920 2962 2962 := CountSegment.join within_054_003 within_054_004
private theorem within_054_005_agreement : RowAgreement 6916 6920 := RowAgreement.join within_054_003_agreement within_054_004_agreement
private def within_054_006 : CountSegment 1 10 13120 6912 6920 2961 2962 := CountSegment.join within_054_002 within_054_005
private theorem within_054_006_agreement : RowAgreement 6912 6920 := RowAgreement.join within_054_002_agreement within_054_005_agreement
private def within_054_007 : CountSegment 1 10 13120 6920 6922 2962 2964 := CountSegment.join atomic_6921 atomic_6922
private theorem within_054_007_agreement : RowAgreement 6920 6922 := RowAgreement.join atomic_6921_agreement atomic_6922_agreement
private def within_054_008 : CountSegment 1 10 13120 6922 6924 2964 2966 := CountSegment.join atomic_6923 atomic_6924
private theorem within_054_008_agreement : RowAgreement 6922 6924 := RowAgreement.join atomic_6923_agreement atomic_6924_agreement
private def within_054_009 : CountSegment 1 10 13120 6920 6924 2962 2966 := CountSegment.join within_054_007 within_054_008
private theorem within_054_009_agreement : RowAgreement 6920 6924 := RowAgreement.join within_054_007_agreement within_054_008_agreement
private def within_054_010 : CountSegment 1 10 13120 6924 6926 2966 2966 := CountSegment.join atomic_6925 atomic_6926
private theorem within_054_010_agreement : RowAgreement 6924 6926 := RowAgreement.join atomic_6925_agreement atomic_6926_agreement
private def within_054_011 : CountSegment 1 10 13120 6926 6928 2966 2968 := CountSegment.join atomic_6927 atomic_6928
private theorem within_054_011_agreement : RowAgreement 6926 6928 := RowAgreement.join atomic_6927_agreement atomic_6928_agreement
private def within_054_012 : CountSegment 1 10 13120 6924 6928 2966 2968 := CountSegment.join within_054_010 within_054_011
private theorem within_054_012_agreement : RowAgreement 6924 6928 := RowAgreement.join within_054_010_agreement within_054_011_agreement
private def within_054_013 : CountSegment 1 10 13120 6920 6928 2962 2968 := CountSegment.join within_054_009 within_054_012
private theorem within_054_013_agreement : RowAgreement 6920 6928 := RowAgreement.join within_054_009_agreement within_054_012_agreement
private def within_054_014 : CountSegment 1 10 13120 6912 6928 2961 2968 := CountSegment.join within_054_006 within_054_013
private theorem within_054_014_agreement : RowAgreement 6912 6928 := RowAgreement.join within_054_006_agreement within_054_013_agreement
private def within_054_015 : CountSegment 1 10 13120 6928 6930 2968 2968 := CountSegment.join atomic_6929 atomic_6930
private theorem within_054_015_agreement : RowAgreement 6928 6930 := RowAgreement.join atomic_6929_agreement atomic_6930_agreement
private def within_054_016 : CountSegment 1 10 13120 6930 6932 2968 2968 := CountSegment.join atomic_6931 atomic_6932
private theorem within_054_016_agreement : RowAgreement 6930 6932 := RowAgreement.join atomic_6931_agreement atomic_6932_agreement
private def within_054_017 : CountSegment 1 10 13120 6928 6932 2968 2968 := CountSegment.join within_054_015 within_054_016
private theorem within_054_017_agreement : RowAgreement 6928 6932 := RowAgreement.join within_054_015_agreement within_054_016_agreement
private def within_054_018 : CountSegment 1 10 13120 6932 6934 2968 2969 := CountSegment.join atomic_6933 atomic_6934
private theorem within_054_018_agreement : RowAgreement 6932 6934 := RowAgreement.join atomic_6933_agreement atomic_6934_agreement
private def within_054_019 : CountSegment 1 10 13120 6934 6936 2969 2970 := CountSegment.join atomic_6935 atomic_6936
private theorem within_054_019_agreement : RowAgreement 6934 6936 := RowAgreement.join atomic_6935_agreement atomic_6936_agreement
private def within_054_020 : CountSegment 1 10 13120 6932 6936 2968 2970 := CountSegment.join within_054_018 within_054_019
private theorem within_054_020_agreement : RowAgreement 6932 6936 := RowAgreement.join within_054_018_agreement within_054_019_agreement
private def within_054_021 : CountSegment 1 10 13120 6928 6936 2968 2970 := CountSegment.join within_054_017 within_054_020
private theorem within_054_021_agreement : RowAgreement 6928 6936 := RowAgreement.join within_054_017_agreement within_054_020_agreement
private def within_054_022 : CountSegment 1 10 13120 6936 6938 2970 2970 := CountSegment.join atomic_6937 atomic_6938
private theorem within_054_022_agreement : RowAgreement 6936 6938 := RowAgreement.join atomic_6937_agreement atomic_6938_agreement
private def within_054_023 : CountSegment 1 10 13120 6938 6940 2970 2971 := CountSegment.join atomic_6939 atomic_6940
private theorem within_054_023_agreement : RowAgreement 6938 6940 := RowAgreement.join atomic_6939_agreement atomic_6940_agreement
private def within_054_024 : CountSegment 1 10 13120 6936 6940 2970 2971 := CountSegment.join within_054_022 within_054_023
private theorem within_054_024_agreement : RowAgreement 6936 6940 := RowAgreement.join within_054_022_agreement within_054_023_agreement
private def within_054_025 : CountSegment 1 10 13120 6940 6942 2971 2971 := CountSegment.join atomic_6941 atomic_6942
private theorem within_054_025_agreement : RowAgreement 6940 6942 := RowAgreement.join atomic_6941_agreement atomic_6942_agreement
private def within_054_026 : CountSegment 1 10 13120 6942 6944 2971 2973 := CountSegment.join atomic_6943 atomic_6944
private theorem within_054_026_agreement : RowAgreement 6942 6944 := RowAgreement.join atomic_6943_agreement atomic_6944_agreement
private def within_054_027 : CountSegment 1 10 13120 6940 6944 2971 2973 := CountSegment.join within_054_025 within_054_026
private theorem within_054_027_agreement : RowAgreement 6940 6944 := RowAgreement.join within_054_025_agreement within_054_026_agreement
private def within_054_028 : CountSegment 1 10 13120 6936 6944 2970 2973 := CountSegment.join within_054_024 within_054_027
private theorem within_054_028_agreement : RowAgreement 6936 6944 := RowAgreement.join within_054_024_agreement within_054_027_agreement
private def within_054_029 : CountSegment 1 10 13120 6928 6944 2968 2973 := CountSegment.join within_054_021 within_054_028
private theorem within_054_029_agreement : RowAgreement 6928 6944 := RowAgreement.join within_054_021_agreement within_054_028_agreement
private def within_054_030 : CountSegment 1 10 13120 6912 6944 2961 2973 := CountSegment.join within_054_014 within_054_029
private theorem within_054_030_agreement : RowAgreement 6912 6944 := RowAgreement.join within_054_014_agreement within_054_029_agreement
private def within_054_031 : CountSegment 1 10 13120 6944 6946 2973 2973 := CountSegment.join atomic_6945 atomic_6946
private theorem within_054_031_agreement : RowAgreement 6944 6946 := RowAgreement.join atomic_6945_agreement atomic_6946_agreement
private def within_054_032 : CountSegment 1 10 13120 6946 6948 2973 2973 := CountSegment.join atomic_6947 atomic_6948
private theorem within_054_032_agreement : RowAgreement 6946 6948 := RowAgreement.join atomic_6947_agreement atomic_6948_agreement
private def within_054_033 : CountSegment 1 10 13120 6944 6948 2973 2973 := CountSegment.join within_054_031 within_054_032
private theorem within_054_033_agreement : RowAgreement 6944 6948 := RowAgreement.join within_054_031_agreement within_054_032_agreement
private def within_054_034 : CountSegment 1 10 13120 6948 6950 2973 2974 := CountSegment.join atomic_6949 atomic_6950
private theorem within_054_034_agreement : RowAgreement 6948 6950 := RowAgreement.join atomic_6949_agreement atomic_6950_agreement
private def within_054_035 : CountSegment 1 10 13120 6950 6952 2974 2975 := CountSegment.join atomic_6951 atomic_6952
private theorem within_054_035_agreement : RowAgreement 6950 6952 := RowAgreement.join atomic_6951_agreement atomic_6952_agreement
private def within_054_036 : CountSegment 1 10 13120 6948 6952 2973 2975 := CountSegment.join within_054_034 within_054_035
private theorem within_054_036_agreement : RowAgreement 6948 6952 := RowAgreement.join within_054_034_agreement within_054_035_agreement
private def within_054_037 : CountSegment 1 10 13120 6944 6952 2973 2975 := CountSegment.join within_054_033 within_054_036
private theorem within_054_037_agreement : RowAgreement 6944 6952 := RowAgreement.join within_054_033_agreement within_054_036_agreement
private def within_054_038 : CountSegment 1 10 13120 6952 6954 2975 2977 := CountSegment.join atomic_6953 atomic_6954
private theorem within_054_038_agreement : RowAgreement 6952 6954 := RowAgreement.join atomic_6953_agreement atomic_6954_agreement
private def within_054_039 : CountSegment 1 10 13120 6954 6956 2977 2978 := CountSegment.join atomic_6955 atomic_6956
private theorem within_054_039_agreement : RowAgreement 6954 6956 := RowAgreement.join atomic_6955_agreement atomic_6956_agreement
private def within_054_040 : CountSegment 1 10 13120 6952 6956 2975 2978 := CountSegment.join within_054_038 within_054_039
private theorem within_054_040_agreement : RowAgreement 6952 6956 := RowAgreement.join within_054_038_agreement within_054_039_agreement
private def within_054_041 : CountSegment 1 10 13120 6956 6958 2978 2980 := CountSegment.join atomic_6957 atomic_6958
private theorem within_054_041_agreement : RowAgreement 6956 6958 := RowAgreement.join atomic_6957_agreement atomic_6958_agreement
private def within_054_042 : CountSegment 1 10 13120 6958 6960 2980 2980 := CountSegment.join atomic_6959 atomic_6960
private theorem within_054_042_agreement : RowAgreement 6958 6960 := RowAgreement.join atomic_6959_agreement atomic_6960_agreement
private def within_054_043 : CountSegment 1 10 13120 6956 6960 2978 2980 := CountSegment.join within_054_041 within_054_042
private theorem within_054_043_agreement : RowAgreement 6956 6960 := RowAgreement.join within_054_041_agreement within_054_042_agreement
private def within_054_044 : CountSegment 1 10 13120 6952 6960 2975 2980 := CountSegment.join within_054_040 within_054_043
private theorem within_054_044_agreement : RowAgreement 6952 6960 := RowAgreement.join within_054_040_agreement within_054_043_agreement
private def within_054_045 : CountSegment 1 10 13120 6944 6960 2973 2980 := CountSegment.join within_054_037 within_054_044
private theorem within_054_045_agreement : RowAgreement 6944 6960 := RowAgreement.join within_054_037_agreement within_054_044_agreement
private def within_054_046 : CountSegment 1 10 13120 6960 6962 2980 2982 := CountSegment.join atomic_6961 atomic_6962
private theorem within_054_046_agreement : RowAgreement 6960 6962 := RowAgreement.join atomic_6961_agreement atomic_6962_agreement
private def within_054_047 : CountSegment 1 10 13120 6962 6964 2982 2983 := CountSegment.join atomic_6963 atomic_6964
private theorem within_054_047_agreement : RowAgreement 6962 6964 := RowAgreement.join atomic_6963_agreement atomic_6964_agreement
private def within_054_048 : CountSegment 1 10 13120 6960 6964 2980 2983 := CountSegment.join within_054_046 within_054_047
private theorem within_054_048_agreement : RowAgreement 6960 6964 := RowAgreement.join within_054_046_agreement within_054_047_agreement
private def within_054_049 : CountSegment 1 10 13120 6964 6966 2983 2984 := CountSegment.join atomic_6965 atomic_6966
private theorem within_054_049_agreement : RowAgreement 6964 6966 := RowAgreement.join atomic_6965_agreement atomic_6966_agreement
private def within_054_050 : CountSegment 1 10 13120 6966 6968 2984 2984 := CountSegment.join atomic_6967 atomic_6968
private theorem within_054_050_agreement : RowAgreement 6966 6968 := RowAgreement.join atomic_6967_agreement atomic_6968_agreement
private def within_054_051 : CountSegment 1 10 13120 6964 6968 2983 2984 := CountSegment.join within_054_049 within_054_050
private theorem within_054_051_agreement : RowAgreement 6964 6968 := RowAgreement.join within_054_049_agreement within_054_050_agreement
private def within_054_052 : CountSegment 1 10 13120 6960 6968 2980 2984 := CountSegment.join within_054_048 within_054_051
private theorem within_054_052_agreement : RowAgreement 6960 6968 := RowAgreement.join within_054_048_agreement within_054_051_agreement
private def within_054_053 : CountSegment 1 10 13120 6968 6970 2984 2986 := CountSegment.join atomic_6969 atomic_6970
private theorem within_054_053_agreement : RowAgreement 6968 6970 := RowAgreement.join atomic_6969_agreement atomic_6970_agreement
private def within_054_054 : CountSegment 1 10 13120 6970 6972 2986 2986 := CountSegment.join atomic_6971 atomic_6972
private theorem within_054_054_agreement : RowAgreement 6970 6972 := RowAgreement.join atomic_6971_agreement atomic_6972_agreement
private def within_054_055 : CountSegment 1 10 13120 6968 6972 2984 2986 := CountSegment.join within_054_053 within_054_054
private theorem within_054_055_agreement : RowAgreement 6968 6972 := RowAgreement.join within_054_053_agreement within_054_054_agreement
private def within_054_056 : CountSegment 1 10 13120 6972 6974 2986 2987 := CountSegment.join atomic_6973 atomic_6974
private theorem within_054_056_agreement : RowAgreement 6972 6974 := RowAgreement.join atomic_6973_agreement atomic_6974_agreement
private def within_054_057 : CountSegment 1 10 13120 6974 6976 2987 2988 := CountSegment.join atomic_6975 atomic_6976
private theorem within_054_057_agreement : RowAgreement 6974 6976 := RowAgreement.join atomic_6975_agreement atomic_6976_agreement
private def within_054_058 : CountSegment 1 10 13120 6972 6976 2986 2988 := CountSegment.join within_054_056 within_054_057
private theorem within_054_058_agreement : RowAgreement 6972 6976 := RowAgreement.join within_054_056_agreement within_054_057_agreement
private def within_054_059 : CountSegment 1 10 13120 6968 6976 2984 2988 := CountSegment.join within_054_055 within_054_058
private theorem within_054_059_agreement : RowAgreement 6968 6976 := RowAgreement.join within_054_055_agreement within_054_058_agreement
private def within_054_060 : CountSegment 1 10 13120 6960 6976 2980 2988 := CountSegment.join within_054_052 within_054_059
private theorem within_054_060_agreement : RowAgreement 6960 6976 := RowAgreement.join within_054_052_agreement within_054_059_agreement
private def within_054_061 : CountSegment 1 10 13120 6944 6976 2973 2988 := CountSegment.join within_054_045 within_054_060
private theorem within_054_061_agreement : RowAgreement 6944 6976 := RowAgreement.join within_054_045_agreement within_054_060_agreement
private def within_054_062 : CountSegment 1 10 13120 6912 6976 2961 2988 := CountSegment.join within_054_030 within_054_061
private theorem within_054_062_agreement : RowAgreement 6912 6976 := RowAgreement.join within_054_030_agreement within_054_061_agreement
private def within_054_063 : CountSegment 1 10 13120 6976 6978 2988 2988 := CountSegment.join atomic_6977 atomic_6978
private theorem within_054_063_agreement : RowAgreement 6976 6978 := RowAgreement.join atomic_6977_agreement atomic_6978_agreement
private def within_054_064 : CountSegment 1 10 13120 6978 6980 2988 2988 := CountSegment.join atomic_6979 atomic_6980
private theorem within_054_064_agreement : RowAgreement 6978 6980 := RowAgreement.join atomic_6979_agreement atomic_6980_agreement
private def within_054_065 : CountSegment 1 10 13120 6976 6980 2988 2988 := CountSegment.join within_054_063 within_054_064
private theorem within_054_065_agreement : RowAgreement 6976 6980 := RowAgreement.join within_054_063_agreement within_054_064_agreement
private def within_054_066 : CountSegment 1 10 13120 6980 6982 2988 2989 := CountSegment.join atomic_6981 atomic_6982
private theorem within_054_066_agreement : RowAgreement 6980 6982 := RowAgreement.join atomic_6981_agreement atomic_6982_agreement
private def within_054_067 : CountSegment 1 10 13120 6982 6984 2989 2989 := CountSegment.join atomic_6983 atomic_6984
private theorem within_054_067_agreement : RowAgreement 6982 6984 := RowAgreement.join atomic_6983_agreement atomic_6984_agreement
private def within_054_068 : CountSegment 1 10 13120 6980 6984 2988 2989 := CountSegment.join within_054_066 within_054_067
private theorem within_054_068_agreement : RowAgreement 6980 6984 := RowAgreement.join within_054_066_agreement within_054_067_agreement
private def within_054_069 : CountSegment 1 10 13120 6976 6984 2988 2989 := CountSegment.join within_054_065 within_054_068
private theorem within_054_069_agreement : RowAgreement 6976 6984 := RowAgreement.join within_054_065_agreement within_054_068_agreement
private def within_054_070 : CountSegment 1 10 13120 6984 6986 2989 2990 := CountSegment.join atomic_6985 atomic_6986
private theorem within_054_070_agreement : RowAgreement 6984 6986 := RowAgreement.join atomic_6985_agreement atomic_6986_agreement
private def within_054_071 : CountSegment 1 10 13120 6986 6988 2990 2990 := CountSegment.join atomic_6987 atomic_6988
private theorem within_054_071_agreement : RowAgreement 6986 6988 := RowAgreement.join atomic_6987_agreement atomic_6988_agreement
private def within_054_072 : CountSegment 1 10 13120 6984 6988 2989 2990 := CountSegment.join within_054_070 within_054_071
private theorem within_054_072_agreement : RowAgreement 6984 6988 := RowAgreement.join within_054_070_agreement within_054_071_agreement
private def within_054_073 : CountSegment 1 10 13120 6988 6990 2990 2991 := CountSegment.join atomic_6989 atomic_6990
private theorem within_054_073_agreement : RowAgreement 6988 6990 := RowAgreement.join atomic_6989_agreement atomic_6990_agreement
private def within_054_074 : CountSegment 1 10 13120 6990 6992 2991 2992 := CountSegment.join atomic_6991 atomic_6992
private theorem within_054_074_agreement : RowAgreement 6990 6992 := RowAgreement.join atomic_6991_agreement atomic_6992_agreement
private def within_054_075 : CountSegment 1 10 13120 6988 6992 2990 2992 := CountSegment.join within_054_073 within_054_074
private theorem within_054_075_agreement : RowAgreement 6988 6992 := RowAgreement.join within_054_073_agreement within_054_074_agreement
private def within_054_076 : CountSegment 1 10 13120 6984 6992 2989 2992 := CountSegment.join within_054_072 within_054_075
private theorem within_054_076_agreement : RowAgreement 6984 6992 := RowAgreement.join within_054_072_agreement within_054_075_agreement
private def within_054_077 : CountSegment 1 10 13120 6976 6992 2988 2992 := CountSegment.join within_054_069 within_054_076
private theorem within_054_077_agreement : RowAgreement 6976 6992 := RowAgreement.join within_054_069_agreement within_054_076_agreement
private def within_054_078 : CountSegment 1 10 13120 6992 6994 2992 2992 := CountSegment.join atomic_6993 atomic_6994
private theorem within_054_078_agreement : RowAgreement 6992 6994 := RowAgreement.join atomic_6993_agreement atomic_6994_agreement
private def within_054_079 : CountSegment 1 10 13120 6994 6996 2992 2993 := CountSegment.join atomic_6995 atomic_6996
private theorem within_054_079_agreement : RowAgreement 6994 6996 := RowAgreement.join atomic_6995_agreement atomic_6996_agreement
private def within_054_080 : CountSegment 1 10 13120 6992 6996 2992 2993 := CountSegment.join within_054_078 within_054_079
private theorem within_054_080_agreement : RowAgreement 6992 6996 := RowAgreement.join within_054_078_agreement within_054_079_agreement
private def within_054_081 : CountSegment 1 10 13120 6996 6998 2993 2995 := CountSegment.join atomic_6997 atomic_6998
private theorem within_054_081_agreement : RowAgreement 6996 6998 := RowAgreement.join atomic_6997_agreement atomic_6998_agreement
private def within_054_082 : CountSegment 1 10 13120 6998 7000 2995 2996 := CountSegment.join atomic_6999 atomic_7000
private theorem within_054_082_agreement : RowAgreement 6998 7000 := RowAgreement.join atomic_6999_agreement atomic_7000_agreement
private def within_054_083 : CountSegment 1 10 13120 6996 7000 2993 2996 := CountSegment.join within_054_081 within_054_082
private theorem within_054_083_agreement : RowAgreement 6996 7000 := RowAgreement.join within_054_081_agreement within_054_082_agreement
private def within_054_084 : CountSegment 1 10 13120 6992 7000 2992 2996 := CountSegment.join within_054_080 within_054_083
private theorem within_054_084_agreement : RowAgreement 6992 7000 := RowAgreement.join within_054_080_agreement within_054_083_agreement
private def within_054_085 : CountSegment 1 10 13120 7000 7002 2996 2997 := CountSegment.join atomic_7001 atomic_7002
private theorem within_054_085_agreement : RowAgreement 7000 7002 := RowAgreement.join atomic_7001_agreement atomic_7002_agreement
private def within_054_086 : CountSegment 1 10 13120 7002 7004 2997 2997 := CountSegment.join atomic_7003 atomic_7004
private theorem within_054_086_agreement : RowAgreement 7002 7004 := RowAgreement.join atomic_7003_agreement atomic_7004_agreement
private def within_054_087 : CountSegment 1 10 13120 7000 7004 2996 2997 := CountSegment.join within_054_085 within_054_086
private theorem within_054_087_agreement : RowAgreement 7000 7004 := RowAgreement.join within_054_085_agreement within_054_086_agreement
private def within_054_088 : CountSegment 1 10 13120 7004 7006 2997 2998 := CountSegment.join atomic_7005 atomic_7006
private theorem within_054_088_agreement : RowAgreement 7004 7006 := RowAgreement.join atomic_7005_agreement atomic_7006_agreement
private def within_054_089 : CountSegment 1 10 13120 7006 7008 2998 3000 := CountSegment.join atomic_7007 atomic_7008
private theorem within_054_089_agreement : RowAgreement 7006 7008 := RowAgreement.join atomic_7007_agreement atomic_7008_agreement
private def within_054_090 : CountSegment 1 10 13120 7004 7008 2997 3000 := CountSegment.join within_054_088 within_054_089
private theorem within_054_090_agreement : RowAgreement 7004 7008 := RowAgreement.join within_054_088_agreement within_054_089_agreement
private def within_054_091 : CountSegment 1 10 13120 7000 7008 2996 3000 := CountSegment.join within_054_087 within_054_090
private theorem within_054_091_agreement : RowAgreement 7000 7008 := RowAgreement.join within_054_087_agreement within_054_090_agreement
private def within_054_092 : CountSegment 1 10 13120 6992 7008 2992 3000 := CountSegment.join within_054_084 within_054_091
private theorem within_054_092_agreement : RowAgreement 6992 7008 := RowAgreement.join within_054_084_agreement within_054_091_agreement
private def within_054_093 : CountSegment 1 10 13120 6976 7008 2988 3000 := CountSegment.join within_054_077 within_054_092
private theorem within_054_093_agreement : RowAgreement 6976 7008 := RowAgreement.join within_054_077_agreement within_054_092_agreement
private def within_054_094 : CountSegment 1 10 13120 7008 7010 3000 3002 := CountSegment.join atomic_7009 atomic_7010
private theorem within_054_094_agreement : RowAgreement 7008 7010 := RowAgreement.join atomic_7009_agreement atomic_7010_agreement
private def within_054_095 : CountSegment 1 10 13120 7010 7012 3002 3003 := CountSegment.join atomic_7011 atomic_7012
private theorem within_054_095_agreement : RowAgreement 7010 7012 := RowAgreement.join atomic_7011_agreement atomic_7012_agreement
private def within_054_096 : CountSegment 1 10 13120 7008 7012 3000 3003 := CountSegment.join within_054_094 within_054_095
private theorem within_054_096_agreement : RowAgreement 7008 7012 := RowAgreement.join within_054_094_agreement within_054_095_agreement
private def within_054_097 : CountSegment 1 10 13120 7012 7014 3003 3006 := CountSegment.join atomic_7013 atomic_7014
private theorem within_054_097_agreement : RowAgreement 7012 7014 := RowAgreement.join atomic_7013_agreement atomic_7014_agreement
private def within_054_098 : CountSegment 1 10 13120 7014 7016 3006 3007 := CountSegment.join atomic_7015 atomic_7016
private theorem within_054_098_agreement : RowAgreement 7014 7016 := RowAgreement.join atomic_7015_agreement atomic_7016_agreement
private def within_054_099 : CountSegment 1 10 13120 7012 7016 3003 3007 := CountSegment.join within_054_097 within_054_098
private theorem within_054_099_agreement : RowAgreement 7012 7016 := RowAgreement.join within_054_097_agreement within_054_098_agreement
private def within_054_100 : CountSegment 1 10 13120 7008 7016 3000 3007 := CountSegment.join within_054_096 within_054_099
private theorem within_054_100_agreement : RowAgreement 7008 7016 := RowAgreement.join within_054_096_agreement within_054_099_agreement
private def within_054_101 : CountSegment 1 10 13120 7016 7018 3007 3010 := CountSegment.join atomic_7017 atomic_7018
private theorem within_054_101_agreement : RowAgreement 7016 7018 := RowAgreement.join atomic_7017_agreement atomic_7018_agreement
private def within_054_102 : CountSegment 1 10 13120 7018 7020 3010 3011 := CountSegment.join atomic_7019 atomic_7020
private theorem within_054_102_agreement : RowAgreement 7018 7020 := RowAgreement.join atomic_7019_agreement atomic_7020_agreement
private def within_054_103 : CountSegment 1 10 13120 7016 7020 3007 3011 := CountSegment.join within_054_101 within_054_102
private theorem within_054_103_agreement : RowAgreement 7016 7020 := RowAgreement.join within_054_101_agreement within_054_102_agreement
private def within_054_104 : CountSegment 1 10 13120 7020 7022 3011 3012 := CountSegment.join atomic_7021 atomic_7022
private theorem within_054_104_agreement : RowAgreement 7020 7022 := RowAgreement.join atomic_7021_agreement atomic_7022_agreement
private def within_054_105 : CountSegment 1 10 13120 7022 7024 3012 3012 := CountSegment.join atomic_7023 atomic_7024
private theorem within_054_105_agreement : RowAgreement 7022 7024 := RowAgreement.join atomic_7023_agreement atomic_7024_agreement
private def within_054_106 : CountSegment 1 10 13120 7020 7024 3011 3012 := CountSegment.join within_054_104 within_054_105
private theorem within_054_106_agreement : RowAgreement 7020 7024 := RowAgreement.join within_054_104_agreement within_054_105_agreement
private def within_054_107 : CountSegment 1 10 13120 7016 7024 3007 3012 := CountSegment.join within_054_103 within_054_106
private theorem within_054_107_agreement : RowAgreement 7016 7024 := RowAgreement.join within_054_103_agreement within_054_106_agreement
private def within_054_108 : CountSegment 1 10 13120 7008 7024 3000 3012 := CountSegment.join within_054_100 within_054_107
private theorem within_054_108_agreement : RowAgreement 7008 7024 := RowAgreement.join within_054_100_agreement within_054_107_agreement
private def within_054_109 : CountSegment 1 10 13120 7024 7026 3012 3012 := CountSegment.join atomic_7025 atomic_7026
private theorem within_054_109_agreement : RowAgreement 7024 7026 := RowAgreement.join atomic_7025_agreement atomic_7026_agreement
private def within_054_110 : CountSegment 1 10 13120 7026 7028 3012 3013 := CountSegment.join atomic_7027 atomic_7028
private theorem within_054_110_agreement : RowAgreement 7026 7028 := RowAgreement.join atomic_7027_agreement atomic_7028_agreement
private def within_054_111 : CountSegment 1 10 13120 7024 7028 3012 3013 := CountSegment.join within_054_109 within_054_110
private theorem within_054_111_agreement : RowAgreement 7024 7028 := RowAgreement.join within_054_109_agreement within_054_110_agreement
private def within_054_112 : CountSegment 1 10 13120 7028 7030 3013 3013 := CountSegment.join atomic_7029 atomic_7030
private theorem within_054_112_agreement : RowAgreement 7028 7030 := RowAgreement.join atomic_7029_agreement atomic_7030_agreement
private def within_054_113 : CountSegment 1 10 13120 7030 7032 3013 3013 := CountSegment.join atomic_7031 atomic_7032
private theorem within_054_113_agreement : RowAgreement 7030 7032 := RowAgreement.join atomic_7031_agreement atomic_7032_agreement
private def within_054_114 : CountSegment 1 10 13120 7028 7032 3013 3013 := CountSegment.join within_054_112 within_054_113
private theorem within_054_114_agreement : RowAgreement 7028 7032 := RowAgreement.join within_054_112_agreement within_054_113_agreement
private def within_054_115 : CountSegment 1 10 13120 7024 7032 3012 3013 := CountSegment.join within_054_111 within_054_114
private theorem within_054_115_agreement : RowAgreement 7024 7032 := RowAgreement.join within_054_111_agreement within_054_114_agreement
private def within_054_116 : CountSegment 1 10 13120 7032 7034 3013 3014 := CountSegment.join atomic_7033 atomic_7034
private theorem within_054_116_agreement : RowAgreement 7032 7034 := RowAgreement.join atomic_7033_agreement atomic_7034_agreement
private def within_054_117 : CountSegment 1 10 13120 7034 7036 3014 3014 := CountSegment.join atomic_7035 atomic_7036
private theorem within_054_117_agreement : RowAgreement 7034 7036 := RowAgreement.join atomic_7035_agreement atomic_7036_agreement
private def within_054_118 : CountSegment 1 10 13120 7032 7036 3013 3014 := CountSegment.join within_054_116 within_054_117
private theorem within_054_118_agreement : RowAgreement 7032 7036 := RowAgreement.join within_054_116_agreement within_054_117_agreement
private def within_054_119 : CountSegment 1 10 13120 7036 7038 3014 3014 := CountSegment.join atomic_7037 atomic_7038
private theorem within_054_119_agreement : RowAgreement 7036 7038 := RowAgreement.join atomic_7037_agreement atomic_7038_agreement
private def within_054_120 : CountSegment 1 10 13120 7038 7040 3014 3015 := CountSegment.join atomic_7039 atomic_7040
private theorem within_054_120_agreement : RowAgreement 7038 7040 := RowAgreement.join atomic_7039_agreement atomic_7040_agreement
private def within_054_121 : CountSegment 1 10 13120 7036 7040 3014 3015 := CountSegment.join within_054_119 within_054_120
private theorem within_054_121_agreement : RowAgreement 7036 7040 := RowAgreement.join within_054_119_agreement within_054_120_agreement
private def within_054_122 : CountSegment 1 10 13120 7032 7040 3013 3015 := CountSegment.join within_054_118 within_054_121
private theorem within_054_122_agreement : RowAgreement 7032 7040 := RowAgreement.join within_054_118_agreement within_054_121_agreement
private def within_054_123 : CountSegment 1 10 13120 7024 7040 3012 3015 := CountSegment.join within_054_115 within_054_122
private theorem within_054_123_agreement : RowAgreement 7024 7040 := RowAgreement.join within_054_115_agreement within_054_122_agreement
private def within_054_124 : CountSegment 1 10 13120 7008 7040 3000 3015 := CountSegment.join within_054_108 within_054_123
private theorem within_054_124_agreement : RowAgreement 7008 7040 := RowAgreement.join within_054_108_agreement within_054_123_agreement
private def within_054_125 : CountSegment 1 10 13120 6976 7040 2988 3015 := CountSegment.join within_054_093 within_054_124
private theorem within_054_125_agreement : RowAgreement 6976 7040 := RowAgreement.join within_054_093_agreement within_054_124_agreement
private def within_054_126 : CountSegment 1 10 13120 6912 7040 2961 3015 := CountSegment.join within_054_062 within_054_125
private theorem within_054_126_agreement : RowAgreement 6912 7040 := RowAgreement.join within_054_062_agreement within_054_125_agreement
def segment_054 : CountSegment 1 10 13120 6912 7040 2961 3015 := within_054_126
theorem agreement_054 : RowAgreement 6912 7040 := within_054_126_agreement
end ProjectEulerSolutions.P58Numerical
