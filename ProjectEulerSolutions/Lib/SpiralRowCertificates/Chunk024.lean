import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3073 : RowCertificate 3073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3073 1) := by change ¬ Nat.Prime 37779463; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 389479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3073 2) := by change ¬ Nat.Prime 37773317; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 108233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3073 3) := by change ¬ Nat.Prime 37767171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12589057) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3073 : row 3073 = 0 := (row_3073.checked).trans rfl
theorem model_count_3073 : modelRow 3073 = 0 := by decide
theorem modelrow_agrees_3073 : modelRow 3073 = row 3073 := model_count_3073.trans row_count_3073.symm
def row_3074 : RowCertificate 3074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3074 1) := by change ¬ Nat.Prime 37804053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12601351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3074 2) := by change ¬ Nat.Prime 37797905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7559581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3074 3) := by change ¬ Nat.Prime 37791757; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 619537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3074 : row 3074 = 0 := (row_3074.checked).trans rfl
theorem model_count_3074 : modelRow 3074 = 0 := by decide
theorem modelrow_agrees_3074 : modelRow 3074 = row 3074 := model_count_3074.trans row_count_3074.symm
def row_3075 : RowCertificate 3075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3075 1) := by change ¬ Nat.Prime 37828651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5404093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3075 2) := by change ¬ Nat.Prime 37822501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2224853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3075 3) := by change ¬ Nat.Prime 37816351; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 346939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3075 : row 3075 = 0 := (row_3075.checked).trans rfl
theorem model_count_3075 : modelRow 3075 = 0 := by decide
theorem modelrow_agrees_3075 : modelRow 3075 = row 3075 := model_count_3075.trans row_count_3075.symm
def row_3076 : RowCertificate 3076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3076 1) := by change ¬ Nat.Prime 37853257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2911789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3076 2) := by change ¬ Nat.Prime 37847105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7569421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3076 3) := by change ¬ Nat.Prime 37840953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12613651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3076 : row 3076 = 0 := (row_3076.checked).trans rfl
theorem model_count_3076 : modelRow 3076 = 0 := by decide
theorem modelrow_agrees_3076 : modelRow 3076 = row 3076 := model_count_3076.trans row_count_3076.symm
def row_3077 : RowCertificate 3077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3077 1) := by change ¬ Nat.Prime 37877871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12625957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3077 2) := by change ¬ Nat.Prime 37871717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2913209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3077 3) := by change ¬ Nat.Prime 37865563; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 53407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3077 : row 3077 = 0 := (row_3077.checked).trans rfl
theorem model_count_3077 : modelRow 3077 = 0 := by decide
theorem modelrow_agrees_3077 : modelRow 3077 = row 3077 := model_count_3077.trans row_count_3077.symm
def row_3078 : RowCertificate 3078 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3078 1) := by change Nat.Prime 37902493; exact ProjectEulerSolutions.P58PrimeNodes.p_37902493.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3078 2) := by change Nat.Prime 37896337; exact ProjectEulerSolutions.P58PrimeNodes.p_37896337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3078 3) := by change ¬ Nat.Prime 37890181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5412883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3078 : row 3078 = 2 := (row_3078.checked).trans rfl
theorem model_count_3078 : modelRow 3078 = 2 := by decide
theorem modelrow_agrees_3078 : modelRow 3078 = row 3078 := model_count_3078.trans row_count_3078.symm
def row_3079 : RowCertificate 3079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3079 1) := by change ¬ Nat.Prime 37927123; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2917471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3079 2) := by change ¬ Nat.Prime 37920965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7584193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3079 3) := by change ¬ Nat.Prime 37914807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12638269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3079 : row 3079 = 0 := (row_3079.checked).trans rfl
theorem model_count_3079 : modelRow 3079 = 0 := by decide
theorem modelrow_agrees_3079 : modelRow 3079 = row 3079 := model_count_3079.trans row_count_3079.symm
def row_3080 : RowCertificate 3080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3080 1) := by change ¬ Nat.Prime 37951761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12650587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3080 2) := by change ¬ Nat.Prime 37945601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1308469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3080 3) := by change ¬ Nat.Prime 37939441; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 108709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3080 : row 3080 = 0 := (row_3080.checked).trans rfl
theorem model_count_3080 : modelRow 3080 = 0 := by decide
theorem modelrow_agrees_3080 : modelRow 3080 = row 3080 := model_count_3080.trans row_count_3080.symm
def row_3081 : RowCertificate 3081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3081 1) := by change ¬ Nat.Prime 37976407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5425201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3081 2) := by change ¬ Nat.Prime 37970245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7594049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3081 3) := by change ¬ Nat.Prime 37964083; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 121291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3081 : row 3081 = 0 := (row_3081.checked).trans rfl
theorem model_count_3081 : modelRow 3081 = 0 := by decide
theorem modelrow_agrees_3081 : modelRow 3081 = row 3081 := model_count_3081.trans row_count_3081.symm
def row_3082 : RowCertificate 3082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3082 1) := by change ¬ Nat.Prime 38001061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5428723) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3082 2) := by change Nat.Prime 37994897; exact ProjectEulerSolutions.P58PrimeNodes.p_37994897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3082 3) := by change ¬ Nat.Prime 37988733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12662911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3082 : row 3082 = 1 := (row_3082.checked).trans rfl
theorem model_count_3082 : modelRow 3082 = 1 := by decide
theorem modelrow_agrees_3082 : modelRow 3082 = row 3082 := model_count_3082.trans row_count_3082.symm
def row_3083 : RowCertificate 3083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3083 1) := by change ¬ Nat.Prime 38025723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12675241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3083 2) := by change Nat.Prime 38019557; exact ProjectEulerSolutions.P58PrimeNodes.p_38019557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3083 3) := by change ¬ Nat.Prime 38013391; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2924107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3083 : row 3083 = 1 := (row_3083.checked).trans rfl
theorem model_count_3083 : modelRow 3083 = 1 := by decide
theorem modelrow_agrees_3083 : modelRow 3083 = row 3083 := model_count_3083.trans row_count_3083.symm
def row_3084 : RowCertificate 3084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3084 1) := by change ¬ Nat.Prime 38050393; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1028389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3084 2) := by change ¬ Nat.Prime 38044225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7608845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3084 3) := by change ¬ Nat.Prime 38038057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2002003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3084 : row 3084 = 0 := (row_3084.checked).trans rfl
theorem model_count_3084 : modelRow 3084 = 0 := by decide
theorem modelrow_agrees_3084 : modelRow 3084 = row 3084 := model_count_3084.trans row_count_3084.symm
def row_3085 : RowCertificate 3085 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3085 1) := by change Nat.Prime 38075071; exact ProjectEulerSolutions.P58PrimeNodes.p_38075071.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3085 2) := by change ¬ Nat.Prime 38068901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2928377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3085 3) := by change ¬ Nat.Prime 38062731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12687577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3085 : row 3085 = 1 := (row_3085.checked).trans rfl
theorem model_count_3085 : modelRow 3085 = 1 := by decide
theorem modelrow_agrees_3085 : modelRow 3085 = row 3085 := model_count_3085.trans row_count_3085.symm
def row_3086 : RowCertificate 3086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3086 1) := by change ¬ Nat.Prime 38099757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12699919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3086 2) := by change ¬ Nat.Prime 38093585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7618717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3086 3) := by change ¬ Nat.Prime 38087413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5441059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3086 : row 3086 = 0 := (row_3086.checked).trans rfl
theorem model_count_3086 : modelRow 3086 = 0 := by decide
theorem modelrow_agrees_3086 : modelRow 3086 = row 3086 := model_count_3086.trans row_count_3086.symm
def row_3087 : RowCertificate 3087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3087 1) := by change ¬ Nat.Prime 38124451; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1229821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3087 2) := by change ¬ Nat.Prime 38118277; exact PrattCertificates.composite_of_product (factor := 701) (cofactor := 54377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3087 3) := by change Nat.Prime 38112103; exact ProjectEulerSolutions.P58PrimeNodes.p_38112103.prime
    exact decide_eq_true prime
theorem row_count_3087 : row 3087 = 1 := (row_3087.checked).trans rfl
theorem model_count_3087 : modelRow 3087 = 1 := by decide
theorem modelrow_agrees_3087 : modelRow 3087 = row 3087 := model_count_3087.trans row_count_3087.symm
def row_3088 : RowCertificate 3088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3088 1) := by change ¬ Nat.Prime 38149153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5449879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3088 2) := by change Nat.Prime 38142977; exact ProjectEulerSolutions.P58PrimeNodes.p_38142977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3088 3) := by change ¬ Nat.Prime 38136801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12712267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3088 : row 3088 = 1 := (row_3088.checked).trans rfl
theorem model_count_3088 : modelRow 3088 = 1 := by decide
theorem modelrow_agrees_3088 : modelRow 3088 = row 3088 := model_count_3088.trans row_count_3088.symm
def row_3089 : RowCertificate 3089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3089 1) := by change ¬ Nat.Prime 38173863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12724621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3089 2) := by change ¬ Nat.Prime 38167685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7633537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3089 3) := by change Nat.Prime 38161507; exact ProjectEulerSolutions.P58PrimeNodes.p_38161507.prime
    exact decide_eq_true prime
theorem row_count_3089 : row 3089 = 1 := (row_3089.checked).trans rfl
theorem model_count_3089 : modelRow 3089 = 1 := by decide
theorem modelrow_agrees_3089 : modelRow 3089 = row 3089 := model_count_3089.trans row_count_3089.symm
def row_3090 : RowCertificate 3090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3090 1) := by change ¬ Nat.Prime 38198581; exact PrattCertificates.composite_of_product (factor := 4099) (cofactor := 9319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3090 2) := by change ¬ Nat.Prime 38192401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2937877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3090 3) := by change ¬ Nat.Prime 38186221; exact PrattCertificates.composite_of_product (factor := 2281) (cofactor := 16741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3090 : row 3090 = 0 := (row_3090.checked).trans rfl
theorem model_count_3090 : modelRow 3090 = 0 := by decide
theorem modelrow_agrees_3090 : modelRow 3090 = row 3090 := model_count_3090.trans row_count_3090.symm
def row_3091 : RowCertificate 3091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3091 1) := by change ¬ Nat.Prime 38223307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2011753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3091 2) := by change ¬ Nat.Prime 38217125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7643425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3091 3) := by change ¬ Nat.Prime 38210943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12736981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3091 : row 3091 = 0 := (row_3091.checked).trans rfl
theorem model_count_3091 : modelRow 3091 = 0 := by decide
theorem modelrow_agrees_3091 : modelRow 3091 = row 3091 := model_count_3091.trans row_count_3091.symm
def row_3092 : RowCertificate 3092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3092 1) := by change ¬ Nat.Prime 38248041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12749347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3092 2) := by change ¬ Nat.Prime 38241857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2249521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3092 3) := by change ¬ Nat.Prime 38235673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5462239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3092 : row 3092 = 0 := (row_3092.checked).trans rfl
theorem model_count_3092 : modelRow 3092 = 0 := by decide
theorem modelrow_agrees_3092 : modelRow 3092 = row 3092 := model_count_3092.trans row_count_3092.symm
def row_3093 : RowCertificate 3093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3093 1) := by change ¬ Nat.Prime 38272783; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2014357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3093 2) := by change ¬ Nat.Prime 38266597; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 394501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3093 3) := by change ¬ Nat.Prime 38260411; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5465773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3093 : row 3093 = 0 := (row_3093.checked).trans rfl
theorem model_count_3093 : modelRow 3093 = 0 := by decide
theorem modelrow_agrees_3093 : modelRow 3093 = row 3093 := model_count_3093.trans row_count_3093.symm
def row_3094 : RowCertificate 3094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3094 1) := by change ¬ Nat.Prime 38297533; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 93637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3094 2) := by change ¬ Nat.Prime 38291345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7658269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3094 3) := by change ¬ Nat.Prime 38285157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12761719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3094 : row 3094 = 0 := (row_3094.checked).trans rfl
theorem model_count_3094 : modelRow 3094 = 0 := by decide
theorem modelrow_agrees_3094 : modelRow 3094 = row 3094 := model_count_3094.trans row_count_3094.symm
def row_3095 : RowCertificate 3095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3095 1) := by change ¬ Nat.Prime 38322291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12774097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3095 2) := by change Nat.Prime 38316101; exact ProjectEulerSolutions.P58PrimeNodes.p_38316101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3095 3) := by change ¬ Nat.Prime 38309911; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1035403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3095 : row 3095 = 1 := (row_3095.checked).trans rfl
theorem model_count_3095 : modelRow 3095 = 1 := by decide
theorem modelrow_agrees_3095 : modelRow 3095 = row 3095 := model_count_3095.trans row_count_3095.symm
def row_3096 : RowCertificate 3096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3096 1) := by change ¬ Nat.Prime 38347057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5478151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3096 2) := by change ¬ Nat.Prime 38340865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7668173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3096 3) := by change ¬ Nat.Prime 38334673; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2948821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3096 : row 3096 = 0 := (row_3096.checked).trans rfl
theorem model_count_3096 : modelRow 3096 = 0 := by decide
theorem modelrow_agrees_3096 : modelRow 3096 = row 3096 := model_count_3096.trans row_count_3096.symm
def row_3097 : RowCertificate 3097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3097 1) := by change ¬ Nat.Prime 38371831; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1237801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3097 2) := by change ¬ Nat.Prime 38365637; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1322953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3097 3) := by change ¬ Nat.Prime 38359443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12786481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3097 : row 3097 = 0 := (row_3097.checked).trans rfl
theorem model_count_3097 : modelRow 3097 = 0 := by decide
theorem modelrow_agrees_3097 : modelRow 3097 = row 3097 := model_count_3097.trans row_count_3097.symm
def row_3098 : RowCertificate 3098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3098 1) := by change ¬ Nat.Prime 38396613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12798871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3098 2) := by change ¬ Nat.Prime 38390417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2953109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3098 3) := by change ¬ Nat.Prime 38384221; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 62617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3098 : row 3098 = 0 := (row_3098.checked).trans rfl
theorem model_count_3098 : modelRow 3098 = 0 := by decide
theorem modelrow_agrees_3098 : modelRow 3098 = row 3098 := model_count_3098.trans row_count_3098.symm
def row_3099 : RowCertificate 3099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3099 1) := by change ¬ Nat.Prime 38421403; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 893521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3099 2) := by change ¬ Nat.Prime 38415205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7683041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3099 3) := by change ¬ Nat.Prime 38409007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5487001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3099 : row 3099 = 0 := (row_3099.checked).trans rfl
theorem model_count_3099 : modelRow 3099 = 0 := by decide
theorem modelrow_agrees_3099 : modelRow 3099 = row 3099 := model_count_3099.trans row_count_3099.symm
def row_3100 : RowCertificate 3100 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3100 1) := by change Nat.Prime 38446201; exact ProjectEulerSolutions.P58PrimeNodes.p_38446201.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3100 2) := by change ¬ Nat.Prime 38440001; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 937561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3100 3) := by change ¬ Nat.Prime 38433801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12811267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3100 : row 3100 = 1 := (row_3100.checked).trans rfl
theorem model_count_3100 : modelRow 3100 = 1 := by decide
theorem modelrow_agrees_3100 : modelRow 3100 = row 3100 := model_count_3100.trans row_count_3100.symm
def row_3101 : RowCertificate 3101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3101 1) := by change ¬ Nat.Prime 38471007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12823669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3101 2) := by change ¬ Nat.Prime 38464805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7692961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3101 3) := by change ¬ Nat.Prime 38458603; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2024137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3101 : row 3101 = 0 := (row_3101.checked).trans rfl
theorem model_count_3101 : modelRow 3101 = 0 := by decide
theorem modelrow_agrees_3101 : modelRow 3101 = row 3101 := model_count_3101.trans row_count_3101.symm
def row_3102 : RowCertificate 3102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3102 1) := by change ¬ Nat.Prime 38495821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5499403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3102 2) := by change ¬ Nat.Prime 38489617; exact PrattCertificates.composite_of_product (factor := 4801) (cofactor := 8017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3102 3) := by change ¬ Nat.Prime 38483413; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 303019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3102 : row 3102 = 0 := (row_3102.checked).trans rfl
theorem model_count_3102 : modelRow 3102 = 0 := by decide
theorem modelrow_agrees_3102 : modelRow 3102 = row 3102 := model_count_3102.trans row_count_3102.symm
def row_3103 : RowCertificate 3103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3103 1) := by change ¬ Nat.Prime 38520643; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5502949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3103 2) := by change ¬ Nat.Prime 38514437; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2962649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3103 3) := by change ¬ Nat.Prime 38508231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12836077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3103 : row 3103 = 0 := (row_3103.checked).trans rfl
theorem model_count_3103 : modelRow 3103 = 0 := by decide
theorem modelrow_agrees_3103 : modelRow 3103 = row 3103 := model_count_3103.trans row_count_3103.symm
def row_3104 : RowCertificate 3104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3104 1) := by change ¬ Nat.Prime 38545473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12848491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3104 2) := by change ¬ Nat.Prime 38539265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7707853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3104 3) := by change Nat.Prime 38533057; exact ProjectEulerSolutions.P58PrimeNodes.p_38533057.prime
    exact decide_eq_true prime
theorem row_count_3104 : row 3104 = 1 := (row_3104.checked).trans rfl
theorem model_count_3104 : modelRow 3104 = 1 := by decide
theorem modelrow_agrees_3104 : modelRow 3104 = row 3104 := model_count_3104.trans row_count_3104.symm
def row_3105 : RowCertificate 3105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3105 1) := by change ¬ Nat.Prime 38570311; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2966947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3105 2) := by change ¬ Nat.Prime 38564101; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1042273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3105 3) := by change ¬ Nat.Prime 38557891; exact PrattCertificates.composite_of_product (factor := 2137) (cofactor := 18043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3105 : row 3105 = 0 := (row_3105.checked).trans rfl
theorem model_count_3105 : modelRow 3105 = 0 := by decide
theorem modelrow_agrees_3105 : modelRow 3105 = row 3105 := model_count_3105.trans row_count_3105.symm
def row_3106 : RowCertificate 3106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3106 1) := by change ¬ Nat.Prime 38595157; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 277663) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3106 2) := by change ¬ Nat.Prime 38588945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7717789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3106 3) := by change ¬ Nat.Prime 38582733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12860911) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3106 : row 3106 = 0 := (row_3106.checked).trans rfl
theorem model_count_3106 : modelRow 3106 = 0 := by decide
theorem modelrow_agrees_3106 : modelRow 3106 = row 3106 := model_count_3106.trans row_count_3106.symm
def row_3107 : RowCertificate 3107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3107 1) := by change ¬ Nat.Prime 38620011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12873337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3107 2) := by change ¬ Nat.Prime 38613797; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 259153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3107 3) := by change ¬ Nat.Prime 38607583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5515369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3107 : row 3107 = 0 := (row_3107.checked).trans rfl
theorem model_count_3107 : modelRow 3107 = 0 := by decide
theorem modelrow_agrees_3107 : modelRow 3107 = row 3107 := model_count_3107.trans row_count_3107.symm
def row_3108 : RowCertificate 3108 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3108 1) := by change Nat.Prime 38644873; exact ProjectEulerSolutions.P58PrimeNodes.p_38644873.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3108 2) := by change Nat.Prime 38638657; exact ProjectEulerSolutions.P58PrimeNodes.p_38638657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3108 3) := by change ¬ Nat.Prime 38632441; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 73867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3108 : row 3108 = 2 := (row_3108.checked).trans rfl
theorem model_count_3108 : modelRow 3108 = 2 := by decide
theorem modelrow_agrees_3108 : modelRow 3108 = row 3108 := model_count_3108.trans row_count_3108.symm
def row_3109 : RowCertificate 3109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3109 1) := by change ¬ Nat.Prime 38669743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5524249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3109 2) := by change ¬ Nat.Prime 38663525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7732705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3109 3) := by change ¬ Nat.Prime 38657307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12885769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3109 : row 3109 = 0 := (row_3109.checked).trans rfl
theorem model_count_3109 : modelRow 3109 = 0 := by decide
theorem modelrow_agrees_3109 : modelRow 3109 = row 3109 := model_count_3109.trans row_count_3109.symm
def row_3110 : RowCertificate 3110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3110 1) := by change ¬ Nat.Prime 38694621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12898207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3110 2) := by change Nat.Prime 38688401; exact ProjectEulerSolutions.P58PrimeNodes.p_38688401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3110 3) := by change Nat.Prime 38682181; exact ProjectEulerSolutions.P58PrimeNodes.p_38682181.prime
    exact decide_eq_true prime
theorem row_count_3110 : row 3110 = 2 := (row_3110.checked).trans rfl
theorem model_count_3110 : modelRow 3110 = 2 := by decide
theorem modelrow_agrees_3110 : modelRow 3110 = row 3110 := model_count_3110.trans row_count_3110.symm
def row_3111 : RowCertificate 3111 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3111 1) := by change Nat.Prime 38719507; exact ProjectEulerSolutions.P58PrimeNodes.p_38719507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3111 2) := by change ¬ Nat.Prime 38713285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7742657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3111 3) := by change ¬ Nat.Prime 38707063; exact PrattCertificates.composite_of_product (factor := 2659) (cofactor := 14557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3111 : row 3111 = 1 := (row_3111.checked).trans rfl
theorem model_count_3111 : modelRow 3111 = 1 := by decide
theorem modelrow_agrees_3111 : modelRow 3111 = row 3111 := model_count_3111.trans row_count_3111.symm
def row_3112 : RowCertificate 3112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3112 1) := by change ¬ Nat.Prime 38744401; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2039179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3112 2) := by change ¬ Nat.Prime 38738177; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 730909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3112 3) := by change ¬ Nat.Prime 38731953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12910651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3112 : row 3112 = 0 := (row_3112.checked).trans rfl
theorem model_count_3112 : modelRow 3112 = 0 := by decide
theorem modelrow_agrees_3112 : modelRow 3112 = row 3112 := model_count_3112.trans row_count_3112.symm
def row_3113 : RowCertificate 3113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3113 1) := by change ¬ Nat.Prime 38769303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12923101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3113 2) := by change ¬ Nat.Prime 38763077; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2280181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3113 3) := by change ¬ Nat.Prime 38756851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5536693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3113 : row 3113 = 0 := (row_3113.checked).trans rfl
theorem model_count_3113 : modelRow 3113 = 0 := by decide
theorem modelrow_agrees_3113 : modelRow 3113 = row 3113 := model_count_3113.trans row_count_3113.symm
def row_3114 : RowCertificate 3114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3114 1) := by change ¬ Nat.Prime 38794213; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 902191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3114 2) := by change ¬ Nat.Prime 38787985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7757597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3114 3) := by change ¬ Nat.Prime 38781757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5540251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3114 : row 3114 = 0 := (row_3114.checked).trans rfl
theorem model_count_3114 : modelRow 3114 = 0 := by decide
theorem modelrow_agrees_3114 : modelRow 3114 = row 3114 := model_count_3114.trans row_count_3114.symm
def row_3115 : RowCertificate 3115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3115 1) := by change ¬ Nat.Prime 38819131; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2986087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3115 2) := by change ¬ Nat.Prime 38812901; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 400133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3115 3) := by change ¬ Nat.Prime 38806671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12935557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3115 : row 3115 = 0 := (row_3115.checked).trans rfl
theorem model_count_3115 : modelRow 3115 = 0 := by decide
theorem modelrow_agrees_3115 : modelRow 3115 = row 3115 := model_count_3115.trans row_count_3115.symm
def row_3116 : RowCertificate 3116 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3116 1) := by change ¬ Nat.Prime 38844057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12948019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3116 2) := by change ¬ Nat.Prime 38837825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7767565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3116 3) := by change Nat.Prime 38831593; exact ProjectEulerSolutions.P58PrimeNodes.p_38831593.prime
    exact decide_eq_true prime
theorem row_count_3116 : row 3116 = 1 := (row_3116.checked).trans rfl
theorem model_count_3116 : modelRow 3116 = 1 := by decide
theorem modelrow_agrees_3116 : modelRow 3116 = row 3116 := model_count_3116.trans row_count_3116.symm
def row_3117 : RowCertificate 3117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3117 1) := by change ¬ Nat.Prime 38868991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5552713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3117 2) := by change Nat.Prime 38862757; exact ProjectEulerSolutions.P58PrimeNodes.p_38862757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3117 3) := by change ¬ Nat.Prime 38856523; exact PrattCertificates.composite_of_product (factor := 4027) (cofactor := 9649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3117 : row 3117 = 1 := (row_3117.checked).trans rfl
theorem model_count_3117 : modelRow 3117 = 1 := by decide
theorem modelrow_agrees_3117 : modelRow 3117 = row 3117 := model_count_3117.trans row_count_3117.symm
def row_3118 : RowCertificate 3118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3118 1) := by change ¬ Nat.Prime 38893933; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2991841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3118 2) := by change Nat.Prime 38887697; exact ProjectEulerSolutions.P58PrimeNodes.p_38887697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3118 3) := by change ¬ Nat.Prime 38881461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12960487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3118 : row 3118 = 1 := (row_3118.checked).trans rfl
theorem model_count_3118 : modelRow 3118 = 1 := by decide
theorem modelrow_agrees_3118 : modelRow 3118 = row 3118 := model_count_3118.trans row_count_3118.symm
def row_3119 : RowCertificate 3119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3119 1) := by change ¬ Nat.Prime 38918883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12972961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3119 2) := by change ¬ Nat.Prime 38912645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7782529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3119 3) := by change Nat.Prime 38906407; exact ProjectEulerSolutions.P58PrimeNodes.p_38906407.prime
    exact decide_eq_true prime
theorem row_count_3119 : row 3119 = 1 := (row_3119.checked).trans rfl
theorem model_count_3119 : modelRow 3119 = 1 := by decide
theorem modelrow_agrees_3119 : modelRow 3119 = row 3119 := model_count_3119.trans row_count_3119.symm
def row_3120 : RowCertificate 3120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3120 1) := by change ¬ Nat.Prime 38943841; exact PrattCertificates.composite_of_product (factor := 3169) (cofactor := 12289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3120 2) := by change Nat.Prime 38937601; exact ProjectEulerSolutions.P58PrimeNodes.p_38937601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3120 3) := by change ¬ Nat.Prime 38931361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5561623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3120 : row 3120 = 1 := (row_3120.checked).trans rfl
theorem model_count_3120 : modelRow 3120 = 1 := by decide
theorem modelrow_agrees_3120 : modelRow 3120 = row 3120 := model_count_3120.trans row_count_3120.symm
def row_3121 : RowCertificate 3121 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3121 1) := by change ¬ Nat.Prime 38968807; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1053211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3121 2) := by change ¬ Nat.Prime 38962565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7792513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3121 3) := by change ¬ Nat.Prime 38956323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12985441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3121 : row 3121 = 0 := (row_3121.checked).trans rfl
theorem model_count_3121 : modelRow 3121 = 0 := by decide
theorem modelrow_agrees_3121 : modelRow 3121 = row 3121 := model_count_3121.trans row_count_3121.symm
def row_3122 : RowCertificate 3122 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3122 1) := by change ¬ Nat.Prime 38993781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12997927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3122 2) := by change ¬ Nat.Prime 38987537; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 53189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3122 3) := by change ¬ Nat.Prime 38981293; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2998561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3122 : row 3122 = 0 := (row_3122.checked).trans rfl
theorem model_count_3122 : modelRow 3122 = 0 := by decide
theorem modelrow_agrees_3122 : modelRow 3122 = row 3122 := model_count_3122.trans row_count_3122.symm
def row_3123 : RowCertificate 3123 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3123 1) := by change ¬ Nat.Prime 39018763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5574109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3123 2) := by change ¬ Nat.Prime 39012517; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 357913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3123 3) := by change ¬ Nat.Prime 39006271; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 59011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3123 : row 3123 = 0 := (row_3123.checked).trans rfl
theorem model_count_3123 : modelRow 3123 = 0 := by decide
theorem modelrow_agrees_3123 : modelRow 3123 = row 3123 := model_count_3123.trans row_count_3123.symm
def row_3124 : RowCertificate 3124 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3124 1) := by change ¬ Nat.Prime 39043753; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5577679) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3124 2) := by change ¬ Nat.Prime 39037505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7807501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3124 3) := by change ¬ Nat.Prime 39031257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13010419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3124 : row 3124 = 0 := (row_3124.checked).trans rfl
theorem model_count_3124 : modelRow 3124 = 0 := by decide
theorem modelrow_agrees_3124 : modelRow 3124 = row 3124 := model_count_3124.trans row_count_3124.symm
def row_3125 : RowCertificate 3125 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3125 1) := by change ¬ Nat.Prime 39068751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13022917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3125 2) := by change ¬ Nat.Prime 39062501; exact PrattCertificates.composite_of_product (factor := 3529) (cofactor := 11069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3125 3) := by change ¬ Nat.Prime 39056251; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3004327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3125 : row 3125 = 0 := (row_3125.checked).trans rfl
theorem model_count_3125 : modelRow 3125 = 0 := by decide
theorem modelrow_agrees_3125 : modelRow 3125 = row 3125 := model_count_3125.trans row_count_3125.symm
def row_3126 : RowCertificate 3126 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3126 1) := by change ¬ Nat.Prime 39093757; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 239839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3126 2) := by change ¬ Nat.Prime 39087505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7817501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3126 3) := by change Nat.Prime 39081253; exact ProjectEulerSolutions.P58PrimeNodes.p_39081253.prime
    exact decide_eq_true prime
theorem row_count_3126 : row 3126 = 1 := (row_3126.checked).trans rfl
theorem model_count_3126 : modelRow 3126 = 1 := by decide
theorem modelrow_agrees_3126 : modelRow 3126 = row 3126 := model_count_3126.trans row_count_3126.symm
def row_3127 : RowCertificate 3127 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3127 1) := by change ¬ Nat.Prime 39118771; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 141223) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3127 2) := by change Nat.Prime 39112517; exact ProjectEulerSolutions.P58PrimeNodes.p_39112517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3127 3) := by change ¬ Nat.Prime 39106263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13035421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3127 : row 3127 = 1 := (row_3127.checked).trans rfl
theorem model_count_3127 : modelRow 3127 = 1 := by decide
theorem modelrow_agrees_3127 : modelRow 3127 = row 3127 := model_count_3127.trans row_count_3127.symm
def row_3128 : RowCertificate 3128 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3128 1) := by change ¬ Nat.Prime 39143793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13047931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3128 2) := by change ¬ Nat.Prime 39137537; exact PrattCertificates.composite_of_product (factor := 641) (cofactor := 61057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3128 3) := by change ¬ Nat.Prime 39131281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5590183) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3128 : row 3128 = 0 := (row_3128.checked).trans rfl
theorem model_count_3128 : modelRow 3128 = 0 := by decide
theorem modelrow_agrees_3128 : modelRow 3128 = row 3128 := model_count_3128.trans row_count_3128.symm
def row_3129 : RowCertificate 3129 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3129 1) := by change ¬ Nat.Prime 39168823; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2061517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3129 2) := by change ¬ Nat.Prime 39162565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7832513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3129 3) := by change ¬ Nat.Prime 39156307; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 118297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3129 : row 3129 = 0 := (row_3129.checked).trans rfl
theorem model_count_3129 : modelRow 3129 = 0 := by decide
theorem modelrow_agrees_3129 : modelRow 3129 = row 3129 := model_count_3129.trans row_count_3129.symm
def row_3130 : RowCertificate 3130 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3130 1) := by change ¬ Nat.Prime 39193861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5599123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3130 2) := by change ¬ Nat.Prime 39187601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2305153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3130 3) := by change ¬ Nat.Prime 39181341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13060447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3130 : row 3130 = 0 := (row_3130.checked).trans rfl
theorem model_count_3130 : modelRow 3130 = 0 := by decide
theorem modelrow_agrees_3130 : modelRow 3130 = row 3130 := model_count_3130.trans row_count_3130.symm
def row_3131 : RowCertificate 3131 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3131 1) := by change ¬ Nat.Prime 39218907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13072969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3131 2) := by change ¬ Nat.Prime 39212645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7842529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3131 3) := by change ¬ Nat.Prime 39206383; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 197017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3131 : row 3131 = 0 := (row_3131.checked).trans rfl
theorem model_count_3131 : modelRow 3131 = 0 := by decide
theorem modelrow_agrees_3131 : modelRow 3131 = row 3131 := model_count_3131.trans row_count_3131.symm
def row_3132 : RowCertificate 3132 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3132 1) := by change ¬ Nat.Prime 39243961; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 496759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3132 2) := by change ¬ Nat.Prime 39237697; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 957017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3132 3) := by change ¬ Nat.Prime 39231433; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1060309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3132 : row 3132 = 0 := (row_3132.checked).trans rfl
theorem model_count_3132 : modelRow 3132 = 0 := by decide
theorem modelrow_agrees_3132 : modelRow 3132 = row 3132 := model_count_3132.trans row_count_3132.symm
def row_3133 : RowCertificate 3133 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3133 1) := by change ¬ Nat.Prime 39269023; exact PrattCertificates.composite_of_product (factor := 631) (cofactor := 62233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3133 2) := by change Nat.Prime 39262757; exact ProjectEulerSolutions.P58PrimeNodes.p_39262757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3133 3) := by change ¬ Nat.Prime 39256491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13085497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3133 : row 3133 = 1 := (row_3133.checked).trans rfl
theorem model_count_3133 : modelRow 3133 = 1 := by decide
theorem modelrow_agrees_3133 : modelRow 3133 = row 3133 := model_count_3133.trans row_count_3133.symm
def row_3134 : RowCertificate 3134 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3134 1) := by change ¬ Nat.Prime 39294093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13098031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3134 2) := by change ¬ Nat.Prime 39287825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7857565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3134 3) := by change ¬ Nat.Prime 39281557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5611651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3134 : row 3134 = 0 := (row_3134.checked).trans rfl
theorem model_count_3134 : modelRow 3134 = 0 := by decide
theorem modelrow_agrees_3134 : modelRow 3134 = row 3134 := model_count_3134.trans row_count_3134.symm
def row_3135 : RowCertificate 3135 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3135 1) := by change ¬ Nat.Prime 39319171; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 138937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3135 2) := by change ¬ Nat.Prime 39312901; exact PrattCertificates.composite_of_product (factor := 4789) (cofactor := 8209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3135 3) := by change ¬ Nat.Prime 39306631; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5615233) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3135 : row 3135 = 0 := (row_3135.checked).trans rfl
theorem model_count_3135 : modelRow 3135 = 0 := by decide
theorem modelrow_agrees_3135 : modelRow 3135 = row 3135 := model_count_3135.trans row_count_3135.symm
def row_3136 : RowCertificate 3136 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3136 1) := by change Nat.Prime 39344257; exact ProjectEulerSolutions.P58PrimeNodes.p_39344257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3136 2) := by change ¬ Nat.Prime 39337985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7867597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3136 3) := by change ¬ Nat.Prime 39331713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13110571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3136 : row 3136 = 1 := (row_3136.checked).trans rfl
theorem model_count_3136 : modelRow 3136 = 1 := by decide
theorem modelrow_agrees_3136 : modelRow 3136 = row 3136 := model_count_3136.trans row_count_3136.symm
def row_3137 : RowCertificate 3137 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3137 1) := by change ¬ Nat.Prime 39369351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13123117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3137 2) := by change ¬ Nat.Prime 39363077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3027929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3137 3) := by change ¬ Nat.Prime 39356803; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 45817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3137 : row 3137 = 0 := (row_3137.checked).trans rfl
theorem model_count_3137 : modelRow 3137 = 0 := by decide
theorem modelrow_agrees_3137 : modelRow 3137 = row 3137 := model_count_3137.trans row_count_3137.symm
def row_3138 : RowCertificate 3138 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3138 1) := by change ¬ Nat.Prime 39394453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5627779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3138 2) := by change ¬ Nat.Prime 39388177; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1358213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3138 3) := by change ¬ Nat.Prime 39381901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3029377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3138 : row 3138 = 0 := (row_3138.checked).trans rfl
theorem model_count_3138 : modelRow 3138 = 0 := by decide
theorem modelrow_agrees_3138 : modelRow 3138 = row 3138 := model_count_3138.trans row_count_3138.symm
def row_3139 : RowCertificate 3139 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3139 1) := by change ¬ Nat.Prime 39419563; exact PrattCertificates.composite_of_product (factor := 4129) (cofactor := 9547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3139 2) := by change ¬ Nat.Prime 39413285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7882657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3139 3) := by change ¬ Nat.Prime 39407007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13135669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3139 : row 3139 = 0 := (row_3139.checked).trans rfl
theorem model_count_3139 : modelRow 3139 = 0 := by decide
theorem modelrow_agrees_3139 : modelRow 3139 = row 3139 := model_count_3139.trans row_count_3139.symm
def row_3140 : RowCertificate 3140 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3140 1) := by change ¬ Nat.Prime 39444681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13148227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3140 2) := by change ¬ Nat.Prime 39438401; exact PrattCertificates.composite_of_product (factor := 2017) (cofactor := 19553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3140 3) := by change ¬ Nat.Prime 39432121; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1065733) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3140 : row 3140 = 0 := (row_3140.checked).trans rfl
theorem model_count_3140 : modelRow 3140 = 0 := by decide
theorem modelrow_agrees_3140 : modelRow 3140 = row 3140 := model_count_3140.trans row_count_3140.symm
def row_3141 : RowCertificate 3141 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3141 1) := by change ¬ Nat.Prime 39469807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3036139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3141 2) := by change ¬ Nat.Prime 39463525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7892705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3141 3) := by change ¬ Nat.Prime 39457243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5636749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3141 : row 3141 = 0 := (row_3141.checked).trans rfl
theorem model_count_3141 : modelRow 3141 = 0 := by decide
theorem modelrow_agrees_3141 : modelRow 3141 = row 3141 := model_count_3141.trans row_count_3141.symm
def row_3142 : RowCertificate 3142 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3142 1) := by change ¬ Nat.Prime 39494941; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 918487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3142 2) := by change ¬ Nat.Prime 39488657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3037589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3142 3) := by change ¬ Nat.Prime 39482373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13160791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3142 : row 3142 = 0 := (row_3142.checked).trans rfl
theorem model_count_3142 : modelRow 3142 = 0 := by decide
theorem modelrow_agrees_3142 : modelRow 3142 = row 3142 := model_count_3142.trans row_count_3142.symm
def row_3143 : RowCertificate 3143 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3143 1) := by change ¬ Nat.Prime 39520083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13173361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3143 2) := by change ¬ Nat.Prime 39513797; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2324341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3143 3) := by change Nat.Prime 39507511; exact ProjectEulerSolutions.P58PrimeNodes.p_39507511.prime
    exact decide_eq_true prime
theorem row_count_3143 : row 3143 = 1 := (row_3143.checked).trans rfl
theorem model_count_3143 : modelRow 3143 = 1 := by decide
theorem modelrow_agrees_3143 : modelRow 3143 = row 3143 := model_count_3143.trans row_count_3143.symm
def row_3144 : RowCertificate 3144 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3144 1) := by change ¬ Nat.Prime 39545233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5649319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3144 2) := by change ¬ Nat.Prime 39538945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7907789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3144 3) := by change ¬ Nat.Prime 39532657; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1275247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3144 : row 3144 = 0 := (row_3144.checked).trans rfl
theorem model_count_3144 : modelRow 3144 = 0 := by decide
theorem modelrow_agrees_3144 : modelRow 3144 = row 3144 := model_count_3144.trans row_count_3144.symm
def row_3145 : RowCertificate 3145 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3145 1) := by change ¬ Nat.Prime 39570391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5652913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3145 2) := by change ¬ Nat.Prime 39564101; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 200833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3145 3) := by change ¬ Nat.Prime 39557811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13185937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3145 : row 3145 = 0 := (row_3145.checked).trans rfl
theorem model_count_3145 : modelRow 3145 = 0 := by decide
theorem modelrow_agrees_3145 : modelRow 3145 = row 3145 := model_count_3145.trans row_count_3145.symm
def row_3146 : RowCertificate 3146 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3146 1) := by change ¬ Nat.Prime 39595557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13198519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3146 2) := by change ¬ Nat.Prime 39589265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7917853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3146 3) := by change ¬ Nat.Prime 39582973; exact PrattCertificates.composite_of_product (factor := 1993) (cofactor := 19861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3146 : row 3146 = 0 := (row_3146.checked).trans rfl
theorem model_count_3146 : modelRow 3146 = 0 := by decide
theorem modelrow_agrees_3146 : modelRow 3146 = row 3146 := model_count_3146.trans row_count_3146.symm
def row_3147 : RowCertificate 3147 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3147 1) := by change ¬ Nat.Prime 39620731; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 94111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3147 2) := by change ¬ Nat.Prime 39614437; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2330261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3147 3) := by change ¬ Nat.Prime 39608143; exact PrattCertificates.composite_of_product (factor := 4549) (cofactor := 8707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3147 : row 3147 = 0 := (row_3147.checked).trans rfl
theorem model_count_3147 : modelRow 3147 = 0 := by decide
theorem modelrow_agrees_3147 : modelRow 3147 = row 3147 := model_count_3147.trans row_count_3147.symm
def row_3148 : RowCertificate 3148 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3148 1) := by change ¬ Nat.Prime 39645913; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2086627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3148 2) := by change ¬ Nat.Prime 39639617; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1071341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3148 3) := by change ¬ Nat.Prime 39633321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13211107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3148 : row 3148 = 0 := (row_3148.checked).trans rfl
theorem model_count_3148 : modelRow 3148 = 0 := by decide
theorem modelrow_agrees_3148 : modelRow 3148 = row 3148 := model_count_3148.trans row_count_3148.symm
def row_3149 : RowCertificate 3149 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3149 1) := by change ¬ Nat.Prime 39671103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13223701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3149 2) := by change ¬ Nat.Prime 39664805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7932961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3149 3) := by change ¬ Nat.Prime 39658507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5665501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3149 : row 3149 = 0 := (row_3149.checked).trans rfl
theorem model_count_3149 : modelRow 3149 = 0 := by decide
theorem modelrow_agrees_3149 : modelRow 3149 = row 3149 := model_count_3149.trans row_count_3149.symm
def row_3150 : RowCertificate 3150 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3150 1) := by change ¬ Nat.Prime 39696301; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2089279) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3150 2) := by change ¬ Nat.Prime 39690001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3053077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3150 3) := by change ¬ Nat.Prime 39683701; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 24481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3150 : row 3150 = 0 := (row_3150.checked).trans rfl
theorem model_count_3150 : modelRow 3150 = 0 := by decide
theorem modelrow_agrees_3150 : modelRow 3150 = row 3150 := model_count_3150.trans row_count_3150.symm
def row_3151 : RowCertificate 3151 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3151 1) := by change ¬ Nat.Prime 39721507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5674501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3151 2) := by change ¬ Nat.Prime 39715205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7943041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3151 3) := by change ¬ Nat.Prime 39708903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13236301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3151 : row 3151 = 0 := (row_3151.checked).trans rfl
theorem model_count_3151 : modelRow 3151 = 0 := by decide
theorem modelrow_agrees_3151 : modelRow 3151 = row 3151 := model_count_3151.trans row_count_3151.symm
def row_3152 : RowCertificate 3152 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3152 1) := by change ¬ Nat.Prime 39746721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13248907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3152 2) := by change ¬ Nat.Prime 39740417; exact PrattCertificates.composite_of_product (factor := 853) (cofactor := 46589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3152 3) := by change Nat.Prime 39734113; exact ProjectEulerSolutions.P58PrimeNodes.p_39734113.prime
    exact decide_eq_true prime
theorem row_count_3152 : row 3152 = 1 := (row_3152.checked).trans rfl
theorem model_count_3152 : modelRow 3152 = 1 := by decide
theorem modelrow_agrees_3152 : modelRow 3152 = row 3152 := model_count_3152.trans row_count_3152.symm
def row_3153 : RowCertificate 3153 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3153 1) := by change Nat.Prime 39771943; exact ProjectEulerSolutions.P58PrimeNodes.p_39771943.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3153 2) := by change Nat.Prime 39765637; exact ProjectEulerSolutions.P58PrimeNodes.p_39765637.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3153 3) := by change Nat.Prime 39759331; exact ProjectEulerSolutions.P58PrimeNodes.p_39759331.prime
    exact decide_eq_true prime
theorem row_count_3153 : row 3153 = 3 := (row_3153.checked).trans rfl
theorem model_count_3153 : modelRow 3153 = 3 := by decide
theorem modelrow_agrees_3153 : modelRow 3153 = row 3153 := model_count_3153.trans row_count_3153.symm
def row_3154 : RowCertificate 3154 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3154 1) := by change ¬ Nat.Prime 39797173; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3061321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3154 2) := by change ¬ Nat.Prime 39790865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7958173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3154 3) := by change ¬ Nat.Prime 39784557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13261519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3154 : row 3154 = 0 := (row_3154.checked).trans rfl
theorem model_count_3154 : modelRow 3154 = 0 := by decide
theorem modelrow_agrees_3154 : modelRow 3154 = row 3154 := model_count_3154.trans row_count_3154.symm
def row_3155 : RowCertificate 3155 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3155 1) := by change ¬ Nat.Prime 39822411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13274137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3155 2) := by change ¬ Nat.Prime 39816101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3062777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3155 3) := by change ¬ Nat.Prime 39809791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5687113) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3155 : row 3155 = 0 := (row_3155.checked).trans rfl
theorem model_count_3155 : modelRow 3155 = 0 := by decide
theorem modelrow_agrees_3155 : modelRow 3155 = row 3155 := model_count_3155.trans row_count_3155.symm
def row_3156 : RowCertificate 3156 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3156 1) := by change ¬ Nat.Prime 39847657; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 59209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3156 2) := by change ¬ Nat.Prime 39841345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7968269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3156 3) := by change ¬ Nat.Prime 39835033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5690719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3156 : row 3156 = 0 := (row_3156.checked).trans rfl
theorem model_count_3156 : modelRow 3156 = 0 := by decide
theorem modelrow_agrees_3156 : modelRow 3156 = row 3156 := model_count_3156.trans row_count_3156.symm
def row_3157 : RowCertificate 3157 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3157 1) := by change ¬ Nat.Prime 39872911; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3067147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3157 2) := by change Nat.Prime 39866597; exact ProjectEulerSolutions.P58PrimeNodes.p_39866597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3157 3) := by change ¬ Nat.Prime 39860283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13286761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3157 : row 3157 = 1 := (row_3157.checked).trans rfl
theorem model_count_3157 : modelRow 3157 = 1 := by decide
theorem modelrow_agrees_3157 : modelRow 3157 = row 3157 := model_count_3157.trans row_count_3157.symm
def row_3158 : RowCertificate 3158 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3158 1) := by change ¬ Nat.Prime 39898173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13299391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3158 2) := by change ¬ Nat.Prime 39891857; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 220397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3158 3) := by change ¬ Nat.Prime 39885541; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2099239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3158 : row 3158 = 0 := (row_3158.checked).trans rfl
theorem model_count_3158 : modelRow 3158 = 0 := by decide
theorem modelrow_agrees_3158 : modelRow 3158 = row 3158 := model_count_3158.trans row_count_3158.symm
def row_3159 : RowCertificate 3159 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3159 1) := by change ¬ Nat.Prime 39923443; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5703349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3159 2) := by change ¬ Nat.Prime 39917125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7983425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3159 3) := by change ¬ Nat.Prime 39910807; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 174283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3159 : row 3159 = 0 := (row_3159.checked).trans rfl
theorem model_count_3159 : modelRow 3159 = 0 := by decide
theorem modelrow_agrees_3159 : modelRow 3159 = row 3159 := model_count_3159.trans row_count_3159.symm
def row_3160 : RowCertificate 3160 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3160 1) := by change ¬ Nat.Prime 39948721; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 120691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3160 2) := by change ¬ Nat.Prime 39942401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2349553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3160 3) := by change ¬ Nat.Prime 39936081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13312027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3160 : row 3160 = 0 := (row_3160.checked).trans rfl
theorem model_count_3160 : modelRow 3160 = 0 := by decide
theorem modelrow_agrees_3160 : modelRow 3160 = row 3160 := model_count_3160.trans row_count_3160.symm
def row_3161 : RowCertificate 3161 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3161 1) := by change ¬ Nat.Prime 39974007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13324669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3161 2) := by change ¬ Nat.Prime 39967685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7993537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3161 3) := by change ¬ Nat.Prime 39961363; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3073951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3161 : row 3161 = 0 := (row_3161.checked).trans rfl
theorem model_count_3161 : modelRow 3161 = 0 := by decide
theorem modelrow_agrees_3161 : modelRow 3161 = row 3161 := model_count_3161.trans row_count_3161.symm
def row_3162 : RowCertificate 3162 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3162 1) := by change Nat.Prime 39999301; exact ProjectEulerSolutions.P58PrimeNodes.p_39999301.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3162 2) := by change ¬ Nat.Prime 39992977; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 547849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3162 3) := by change ¬ Nat.Prime 39986653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5712379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3162 : row 3162 = 1 := (row_3162.checked).trans rfl
theorem model_count_3162 : modelRow 3162 = 1 := by decide
theorem modelrow_agrees_3162 : modelRow 3162 = row 3162 := model_count_3162.trans row_count_3162.symm
def row_3163 : RowCertificate 3163 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3163 1) := by change Nat.Prime 40024603; exact ProjectEulerSolutions.P58PrimeNodes.p_40024603.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3163 2) := by change ¬ Nat.Prime 40018277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3078329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3163 3) := by change ¬ Nat.Prime 40011951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13337317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3163 : row 3163 = 1 := (row_3163.checked).trans rfl
theorem model_count_3163 : modelRow 3163 = 1 := by decide
theorem modelrow_agrees_3163 : modelRow 3163 = row 3163 := model_count_3163.trans row_count_3163.symm
def row_3164 : RowCertificate 3164 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3164 1) := by change ¬ Nat.Prime 40049913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13349971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3164 2) := by change ¬ Nat.Prime 40043585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8008717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3164 3) := by change ¬ Nat.Prime 40037257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3079789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3164 : row 3164 = 0 := (row_3164.checked).trans rfl
theorem model_count_3164 : modelRow 3164 = 0 := by decide
theorem modelrow_agrees_3164 : modelRow 3164 = row 3164 := model_count_3164.trans row_count_3164.symm
def row_3165 : RowCertificate 3165 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3165 1) := by change ¬ Nat.Prime 40075231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5725033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3165 2) := by change ¬ Nat.Prime 40068901; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 756017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3165 3) := by change ¬ Nat.Prime 40062571; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1292341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3165 : row 3165 = 0 := (row_3165.checked).trans rfl
theorem model_count_3165 : modelRow 3165 = 0 := by decide
theorem modelrow_agrees_3165 : modelRow 3165 = row 3165 := model_count_3165.trans row_count_3165.symm
def row_3166 : RowCertificate 3166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3166 1) := by change ¬ Nat.Prime 40100557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5728651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3166 2) := by change ¬ Nat.Prime 40094225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8018845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3166 3) := by change ¬ Nat.Prime 40087893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13362631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3166 : row 3166 = 0 := (row_3166.checked).trans rfl
theorem model_count_3166 : modelRow 3166 = 0 := by decide
theorem modelrow_agrees_3166 : modelRow 3166 = row 3166 := model_count_3166.trans row_count_3166.symm
def row_3167 : RowCertificate 3167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3167 1) := by change ¬ Nat.Prime 40125891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13375297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3167 2) := by change ¬ Nat.Prime 40119557; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1383433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3167 3) := by change ¬ Nat.Prime 40113223; exact PrattCertificates.composite_of_product (factor := 1663) (cofactor := 24121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3167 : row 3167 = 0 := (row_3167.checked).trans rfl
theorem model_count_3167 : modelRow 3167 = 0 := by decide
theorem modelrow_agrees_3167 : modelRow 3167 = row 3167 := model_count_3167.trans row_count_3167.symm
def row_3168 : RowCertificate 3168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3168 1) := by change ¬ Nat.Prime 40151233; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 76771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3168 2) := by change ¬ Nat.Prime 40144897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3088069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3168 3) := by change ¬ Nat.Prime 40138561; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 599083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3168 : row 3168 = 0 := (row_3168.checked).trans rfl
theorem model_count_3168 : modelRow 3168 = 0 := by decide
theorem modelrow_agrees_3168 : modelRow 3168 = row 3168 := model_count_3168.trans row_count_3168.symm
def row_3169 : RowCertificate 3169 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3169 1) := by change ¬ Nat.Prime 40176583; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2114557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3169 2) := by change ¬ Nat.Prime 40170245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8034049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3169 3) := by change ¬ Nat.Prime 40163907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13387969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3169 : row 3169 = 0 := (row_3169.checked).trans rfl
theorem model_count_3169 : modelRow 3169 = 0 := by decide
theorem modelrow_agrees_3169 : modelRow 3169 = row 3169 := model_count_3169.trans row_count_3169.symm
def row_3170 : RowCertificate 3170 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3170 1) := by change ¬ Nat.Prime 40201941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13400647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3170 2) := by change Nat.Prime 40195601; exact ProjectEulerSolutions.P58PrimeNodes.p_40195601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3170 3) := by change ¬ Nat.Prime 40189261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5741323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3170 : row 3170 = 1 := (row_3170.checked).trans rfl
theorem model_count_3170 : modelRow 3170 = 1 := by decide
theorem modelrow_agrees_3170 : modelRow 3170 = row 3170 := model_count_3170.trans row_count_3170.symm
def row_3171 : RowCertificate 3171 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3171 1) := by change ¬ Nat.Prime 40227307; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 551059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3171 2) := by change ¬ Nat.Prime 40220965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8044193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3171 3) := by change Nat.Prime 40214623; exact ProjectEulerSolutions.P58PrimeNodes.p_40214623.prime
    exact decide_eq_true prime
theorem row_count_3171 : row 3171 = 1 := (row_3171.checked).trans rfl
theorem model_count_3171 : modelRow 3171 = 1 := by decide
theorem modelrow_agrees_3171 : modelRow 3171 = row 3171 := model_count_3171.trans row_count_3171.symm
def row_3172 : RowCertificate 3172 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3172 1) := by change ¬ Nat.Prime 40252681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5750383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3172 2) := by change ¬ Nat.Prime 40246337; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 95597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3172 3) := by change ¬ Nat.Prime 40239993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13413331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3172 : row 3172 = 0 := (row_3172.checked).trans rfl
theorem model_count_3172 : modelRow 3172 = 0 := by decide
theorem modelrow_agrees_3172 : modelRow 3172 = row 3172 := model_count_3172.trans row_count_3172.symm
def row_3173 : RowCertificate 3173 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3173 1) := by change ¬ Nat.Prime 40278063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13426021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3173 2) := by change ¬ Nat.Prime 40271717; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 982237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3173 3) := by change ¬ Nat.Prime 40265371; exact PrattCertificates.composite_of_product (factor := 2017) (cofactor := 19963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3173 : row 3173 = 0 := (row_3173.checked).trans rfl
theorem model_count_3173 : modelRow 3173 = 0 := by decide
theorem modelrow_agrees_3173 : modelRow 3173 = row 3173 := model_count_3173.trans row_count_3173.symm
def row_3174 : RowCertificate 3174 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3174 1) := by change Nat.Prime 40303453; exact ProjectEulerSolutions.P58PrimeNodes.p_40303453.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3174 2) := by change ¬ Nat.Prime 40297105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8059421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3174 3) := by change ¬ Nat.Prime 40290757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3099289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3174 : row 3174 = 1 := (row_3174.checked).trans rfl
theorem model_count_3174 : modelRow 3174 = 1 := by decide
theorem modelrow_agrees_3174 : modelRow 3174 = row 3174 := model_count_3174.trans row_count_3174.symm
def row_3175 : RowCertificate 3175 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3175 1) := by change Nat.Prime 40328851; exact ProjectEulerSolutions.P58PrimeNodes.p_40328851.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3175 2) := by change Nat.Prime 40322501; exact ProjectEulerSolutions.P58PrimeNodes.p_40322501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3175 3) := by change ¬ Nat.Prime 40316151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13438717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3175 : row 3175 = 2 := (row_3175.checked).trans rfl
theorem model_count_3175 : modelRow 3175 = 2 := by decide
theorem modelrow_agrees_3175 : modelRow 3175 = row 3175 := model_count_3175.trans row_count_3175.symm
def row_3176 : RowCertificate 3176 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3176 1) := by change ¬ Nat.Prime 40354257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13451419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3176 2) := by change ¬ Nat.Prime 40347905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8069581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3176 3) := by change ¬ Nat.Prime 40341553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5763079) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3176 : row 3176 = 0 := (row_3176.checked).trans rfl
theorem model_count_3176 : modelRow 3176 = 0 := by decide
theorem modelrow_agrees_3176 : modelRow 3176 = row 3176 := model_count_3176.trans row_count_3176.symm
def row_3177 : RowCertificate 3177 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3177 1) := by change Nat.Prime 40379671; exact ProjectEulerSolutions.P58PrimeNodes.p_40379671.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3177 2) := by change ¬ Nat.Prime 40373317; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2374901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3177 3) := by change ¬ Nat.Prime 40366963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5766709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3177 : row 3177 = 1 := (row_3177.checked).trans rfl
theorem model_count_3177 : modelRow 3177 = 1 := by decide
theorem modelrow_agrees_3177 : modelRow 3177 = row 3177 := model_count_3177.trans row_count_3177.symm
def row_3178 : RowCertificate 3178 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3178 1) := by change ¬ Nat.Prime 40405093; exact PrattCertificates.composite_of_product (factor := 2053) (cofactor := 19681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3178 2) := by change ¬ Nat.Prime 40398737; exact PrattCertificates.composite_of_product (factor := 1873) (cofactor := 21569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3178 3) := by change ¬ Nat.Prime 40392381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13464127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3178 : row 3178 = 0 := (row_3178.checked).trans rfl
theorem model_count_3178 : modelRow 3178 = 0 := by decide
theorem modelrow_agrees_3178 : modelRow 3178 = row 3178 := model_count_3178.trans row_count_3178.symm
def row_3179 : RowCertificate 3179 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3179 1) := by change ¬ Nat.Prime 40430523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13476841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3179 2) := by change ¬ Nat.Prime 40424165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8084833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3179 3) := by change ¬ Nat.Prime 40417807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2127253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3179 : row 3179 = 0 := (row_3179.checked).trans rfl
theorem model_count_3179 : modelRow 3179 = 0 := by decide
theorem modelrow_agrees_3179 : modelRow 3179 = row 3179 := model_count_3179.trans row_count_3179.symm
def row_3180 : RowCertificate 3180 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3180 1) := by change ¬ Nat.Prime 40455961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5779423) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3180 2) := by change Nat.Prime 40449601; exact ProjectEulerSolutions.P58PrimeNodes.p_40449601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3180 3) := by change ¬ Nat.Prime 40443241; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 554017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3180 : row 3180 = 1 := (row_3180.checked).trans rfl
theorem model_count_3180 : modelRow 3180 = 1 := by decide
theorem modelrow_agrees_3180 : modelRow 3180 = row 3180 := model_count_3180.trans row_count_3180.symm
def row_3181 : RowCertificate 3181 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3181 1) := by change ¬ Nat.Prime 40481407; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 74827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3181 2) := by change ¬ Nat.Prime 40475045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8095009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3181 3) := by change ¬ Nat.Prime 40468683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13489561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3181 : row 3181 = 0 := (row_3181.checked).trans rfl
theorem model_count_3181 : modelRow 3181 = 0 := by decide
theorem modelrow_agrees_3181 : modelRow 3181 = row 3181 := model_count_3181.trans row_count_3181.symm
def row_3182 : RowCertificate 3182 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3182 1) := by change ¬ Nat.Prime 40506861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13502287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3182 2) := by change ¬ Nat.Prime 40500497; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 987817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3182 3) := by change Nat.Prime 40494133; exact ProjectEulerSolutions.P58PrimeNodes.p_40494133.prime
    exact decide_eq_true prime
theorem row_count_3182 : row 3182 = 1 := (row_3182.checked).trans rfl
theorem model_count_3182 : modelRow 3182 = 1 := by decide
theorem modelrow_agrees_3182 : modelRow 3182 = row 3182 := model_count_3182.trans row_count_3182.symm
def row_3183 : RowCertificate 3183 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3183 1) := by change ¬ Nat.Prime 40532323; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3117871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3183 2) := by change Nat.Prime 40525957; exact ProjectEulerSolutions.P58PrimeNodes.p_40525957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3183 3) := by change ¬ Nat.Prime 40519591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5788513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3183 : row 3183 = 1 := (row_3183.checked).trans rfl
theorem model_count_3183 : modelRow 3183 = 1 := by decide
theorem modelrow_agrees_3183 : modelRow 3183 = row 3183 := model_count_3183.trans row_count_3183.symm
def row_3184 : RowCertificate 3184 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3184 1) := by change Nat.Prime 40557793; exact ProjectEulerSolutions.P58PrimeNodes.p_40557793.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3184 2) := by change ¬ Nat.Prime 40551425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8110285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3184 3) := by change ¬ Nat.Prime 40545057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13515019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3184 : row 3184 = 1 := (row_3184.checked).trans rfl
theorem model_count_3184 : modelRow 3184 = 1 := by decide
theorem modelrow_agrees_3184 : modelRow 3184 = row 3184 := model_count_3184.trans row_count_3184.symm
def row_3185 : RowCertificate 3185 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3185 1) := by change ¬ Nat.Prime 40583271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13527757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3185 2) := by change ¬ Nat.Prime 40576901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1096673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3185 3) := by change ¬ Nat.Prime 40570531; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 319453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3185 : row 3185 = 0 := (row_3185.checked).trans rfl
theorem model_count_3185 : modelRow 3185 = 0 := by decide
theorem modelrow_agrees_3185 : modelRow 3185 = row 3185 := model_count_3185.trans row_count_3185.symm
def row_3186 : RowCertificate 3186 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3186 1) := by change ¬ Nat.Prime 40608757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5801251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3186 2) := by change ¬ Nat.Prime 40602385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8120477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3186 3) := by change Nat.Prime 40596013; exact ProjectEulerSolutions.P58PrimeNodes.p_40596013.prime
    exact decide_eq_true prime
theorem row_count_3186 : row 3186 = 1 := (row_3186.checked).trans rfl
theorem model_count_3186 : modelRow 3186 = 1 := by decide
theorem modelrow_agrees_3186 : modelRow 3186 = row 3186 := model_count_3186.trans row_count_3186.symm
def row_3187 : RowCertificate 3187 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3187 1) := by change ¬ Nat.Prime 40634251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5804893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3187 2) := by change ¬ Nat.Prime 40627877; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 456493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3187 3) := by change ¬ Nat.Prime 40621503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13540501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3187 : row 3187 = 0 := (row_3187.checked).trans rfl
theorem model_count_3187 : modelRow 3187 = 0 := by decide
theorem modelrow_agrees_3187 : modelRow 3187 = row 3187 := model_count_3187.trans row_count_3187.symm
def row_3188 : RowCertificate 3188 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3188 1) := by change ¬ Nat.Prime 40659753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13553251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3188 2) := by change ¬ Nat.Prime 40653377; exact PrattCertificates.composite_of_product (factor := 4229) (cofactor := 9613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3188 3) := by change ¬ Nat.Prime 40647001; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 514519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3188 : row 3188 = 0 := (row_3188.checked).trans rfl
theorem model_count_3188 : modelRow 3188 = 0 := by decide
theorem modelrow_agrees_3188 : modelRow 3188 = row 3188 := model_count_3188.trans row_count_3188.symm
def row_3189 : RowCertificate 3189 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3189 1) := by change Nat.Prime 40685263; exact ProjectEulerSolutions.P58PrimeNodes.p_40685263.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3189 2) := by change ¬ Nat.Prime 40678885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8135777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3189 3) := by change Nat.Prime 40672507; exact ProjectEulerSolutions.P58PrimeNodes.p_40672507.prime
    exact decide_eq_true prime
theorem row_count_3189 : row 3189 = 2 := (row_3189.checked).trans rfl
theorem model_count_3189 : modelRow 3189 = 2 := by decide
theorem modelrow_agrees_3189 : modelRow 3189 = row 3189 := model_count_3189.trans row_count_3189.symm
def row_3190 : RowCertificate 3190 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3190 1) := by change ¬ Nat.Prime 40710781; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1313251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3190 2) := by change ¬ Nat.Prime 40704401; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 419633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3190 3) := by change ¬ Nat.Prime 40698021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13566007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3190 : row 3190 = 0 := (row_3190.checked).trans rfl
theorem model_count_3190 : modelRow 3190 = 0 := by decide
theorem modelrow_agrees_3190 : modelRow 3190 = row 3190 := model_count_3190.trans row_count_3190.symm
def row_3191 : RowCertificate 3191 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3191 1) := by change ¬ Nat.Prime 40736307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13578769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3191 2) := by change ¬ Nat.Prime 40729925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8145985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3191 3) := by change ¬ Nat.Prime 40723543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5817649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3191 : row 3191 = 0 := (row_3191.checked).trans rfl
theorem model_count_3191 : modelRow 3191 = 0 := by decide
theorem modelrow_agrees_3191 : modelRow 3191 = row 3191 := model_count_3191.trans row_count_3191.symm
def row_3192 : RowCertificate 3192 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3192 1) := by change ¬ Nat.Prime 40761841; exact PrattCertificates.composite_of_product (factor := 4621) (cofactor := 8821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3192 2) := by change ¬ Nat.Prime 40755457; exact PrattCertificates.composite_of_product (factor := 5021) (cofactor := 8117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3192 3) := by change ¬ Nat.Prime 40749073; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 225133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3192 : row 3192 = 0 := (row_3192.checked).trans rfl
theorem model_count_3192 : modelRow 3192 = 0 := by decide
theorem modelrow_agrees_3192 : modelRow 3192 = row 3192 := model_count_3192.trans row_count_3192.symm
def row_3193 : RowCertificate 3193 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3193 1) := by change ¬ Nat.Prime 40787383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5826769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3193 2) := by change ¬ Nat.Prime 40780997; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 49193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3193 3) := by change ¬ Nat.Prime 40774611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13591537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3193 : row 3193 = 0 := (row_3193.checked).trans rfl
theorem model_count_3193 : modelRow 3193 = 0 := by decide
theorem modelrow_agrees_3193 : modelRow 3193 = row 3193 := model_count_3193.trans row_count_3193.symm
def row_3194 : RowCertificate 3194 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3194 1) := by change ¬ Nat.Prime 40812933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13604311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3194 2) := by change ¬ Nat.Prime 40806545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8161309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3194 3) := by change ¬ Nat.Prime 40800157; exact PrattCertificates.composite_of_product (factor := 2677) (cofactor := 15241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3194 : row 3194 = 0 := (row_3194.checked).trans rfl
theorem model_count_3194 : modelRow 3194 = 0 := by decide
theorem modelrow_agrees_3194 : modelRow 3194 = row 3194 := model_count_3194.trans row_count_3194.symm
def row_3195 : RowCertificate 3195 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3195 1) := by change ¬ Nat.Prime 40838491; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1103743) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3195 2) := by change ¬ Nat.Prime 40832101; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 770417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3195 3) := by change Nat.Prime 40825711; exact ProjectEulerSolutions.P58PrimeNodes.p_40825711.prime
    exact decide_eq_true prime
theorem row_count_3195 : row 3195 = 1 := (row_3195.checked).trans rfl
theorem model_count_3195 : modelRow 3195 = 1 := by decide
theorem modelrow_agrees_3195 : modelRow 3195 = row 3195 := model_count_3195.trans row_count_3195.symm
def row_3196 : RowCertificate 3196 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3196 1) := by change ¬ Nat.Prime 40864057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3143389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3196 2) := by change ¬ Nat.Prime 40857665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8171533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3196 3) := by change ¬ Nat.Prime 40851273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13617091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3196 : row 3196 = 0 := (row_3196.checked).trans rfl
theorem model_count_3196 : modelRow 3196 = 0 := by decide
theorem modelrow_agrees_3196 : modelRow 3196 = row 3196 := model_count_3196.trans row_count_3196.symm
def row_3197 : RowCertificate 3197 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3197 1) := by change ¬ Nat.Prime 40889631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13629877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3197 2) := by change ¬ Nat.Prime 40883237; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 670217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3197 3) := by change ¬ Nat.Prime 40876843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5839549) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3197 : row 3197 = 0 := (row_3197.checked).trans rfl
theorem model_count_3197 : modelRow 3197 = 0 := by decide
theorem modelrow_agrees_3197 : modelRow 3197 = row 3197 := model_count_3197.trans row_count_3197.symm
def row_3198 : RowCertificate 3198 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3198 1) := by change Nat.Prime 40915213; exact ProjectEulerSolutions.P58PrimeNodes.p_40915213.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3198 2) := by change ¬ Nat.Prime 40908817; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2406401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3198 3) := by change ¬ Nat.Prime 40902421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5843203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3198 : row 3198 = 1 := (row_3198.checked).trans rfl
theorem model_count_3198 : modelRow 3198 = 1 := by decide
theorem modelrow_agrees_3198 : modelRow 3198 = row 3198 := model_count_3198.trans row_count_3198.symm
def row_3199 : RowCertificate 3199 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3199 1) := by change ¬ Nat.Prime 40940803; exact PrattCertificates.composite_of_product (factor := 1699) (cofactor := 24097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3199 2) := by change ¬ Nat.Prime 40934405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 8186881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3199 3) := by change ¬ Nat.Prime 40928007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13642669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3199 : row 3199 = 0 := (row_3199.checked).trans rfl
theorem model_count_3199 : modelRow 3199 = 0 := by decide
theorem modelrow_agrees_3199 : modelRow 3199 = row 3199 := model_count_3199.trans row_count_3199.symm
def row_3200 : RowCertificate 3200 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3200 1) := by change ¬ Nat.Prime 40966401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 13655467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3200 2) := by change Nat.Prime 40960001; exact ProjectEulerSolutions.P58PrimeNodes.p_40960001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3200 3) := by change ¬ Nat.Prime 40953601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 3150277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3200 : row 3200 = 1 := (row_3200.checked).trans rfl
theorem model_count_3200 : modelRow 3200 = 1 := by decide
theorem modelrow_agrees_3200 : modelRow 3200 = row 3200 := model_count_3200.trans row_count_3200.symm
private def atomic_3073 : CountSegment 1 10 13120 3072 3073 1493 1493 := CountSegment.single (first := 3072) (initial := 1493) row_count_3073 (by decide)
private theorem atomic_3073_agreement : RowAgreement 3072 3073 := RowAgreement.single (first := 3072) modelrow_agrees_3073
private def atomic_3074 : CountSegment 1 10 13120 3073 3074 1493 1493 := CountSegment.single (first := 3073) (initial := 1493) row_count_3074 (by decide)
private theorem atomic_3074_agreement : RowAgreement 3073 3074 := RowAgreement.single (first := 3073) modelrow_agrees_3074
private def atomic_3075 : CountSegment 1 10 13120 3074 3075 1493 1493 := CountSegment.single (first := 3074) (initial := 1493) row_count_3075 (by decide)
private theorem atomic_3075_agreement : RowAgreement 3074 3075 := RowAgreement.single (first := 3074) modelrow_agrees_3075
private def atomic_3076 : CountSegment 1 10 13120 3075 3076 1493 1493 := CountSegment.single (first := 3075) (initial := 1493) row_count_3076 (by decide)
private theorem atomic_3076_agreement : RowAgreement 3075 3076 := RowAgreement.single (first := 3075) modelrow_agrees_3076
private def atomic_3077 : CountSegment 1 10 13120 3076 3077 1493 1493 := CountSegment.single (first := 3076) (initial := 1493) row_count_3077 (by decide)
private theorem atomic_3077_agreement : RowAgreement 3076 3077 := RowAgreement.single (first := 3076) modelrow_agrees_3077
private def atomic_3078 : CountSegment 1 10 13120 3077 3078 1493 1495 := CountSegment.single (first := 3077) (initial := 1493) row_count_3078 (by decide)
private theorem atomic_3078_agreement : RowAgreement 3077 3078 := RowAgreement.single (first := 3077) modelrow_agrees_3078
private def atomic_3079 : CountSegment 1 10 13120 3078 3079 1495 1495 := CountSegment.single (first := 3078) (initial := 1495) row_count_3079 (by decide)
private theorem atomic_3079_agreement : RowAgreement 3078 3079 := RowAgreement.single (first := 3078) modelrow_agrees_3079
private def atomic_3080 : CountSegment 1 10 13120 3079 3080 1495 1495 := CountSegment.single (first := 3079) (initial := 1495) row_count_3080 (by decide)
private theorem atomic_3080_agreement : RowAgreement 3079 3080 := RowAgreement.single (first := 3079) modelrow_agrees_3080
private def atomic_3081 : CountSegment 1 10 13120 3080 3081 1495 1495 := CountSegment.single (first := 3080) (initial := 1495) row_count_3081 (by decide)
private theorem atomic_3081_agreement : RowAgreement 3080 3081 := RowAgreement.single (first := 3080) modelrow_agrees_3081
private def atomic_3082 : CountSegment 1 10 13120 3081 3082 1495 1496 := CountSegment.single (first := 3081) (initial := 1495) row_count_3082 (by decide)
private theorem atomic_3082_agreement : RowAgreement 3081 3082 := RowAgreement.single (first := 3081) modelrow_agrees_3082
private def atomic_3083 : CountSegment 1 10 13120 3082 3083 1496 1497 := CountSegment.single (first := 3082) (initial := 1496) row_count_3083 (by decide)
private theorem atomic_3083_agreement : RowAgreement 3082 3083 := RowAgreement.single (first := 3082) modelrow_agrees_3083
private def atomic_3084 : CountSegment 1 10 13120 3083 3084 1497 1497 := CountSegment.single (first := 3083) (initial := 1497) row_count_3084 (by decide)
private theorem atomic_3084_agreement : RowAgreement 3083 3084 := RowAgreement.single (first := 3083) modelrow_agrees_3084
private def atomic_3085 : CountSegment 1 10 13120 3084 3085 1497 1498 := CountSegment.single (first := 3084) (initial := 1497) row_count_3085 (by decide)
private theorem atomic_3085_agreement : RowAgreement 3084 3085 := RowAgreement.single (first := 3084) modelrow_agrees_3085
private def atomic_3086 : CountSegment 1 10 13120 3085 3086 1498 1498 := CountSegment.single (first := 3085) (initial := 1498) row_count_3086 (by decide)
private theorem atomic_3086_agreement : RowAgreement 3085 3086 := RowAgreement.single (first := 3085) modelrow_agrees_3086
private def atomic_3087 : CountSegment 1 10 13120 3086 3087 1498 1499 := CountSegment.single (first := 3086) (initial := 1498) row_count_3087 (by decide)
private theorem atomic_3087_agreement : RowAgreement 3086 3087 := RowAgreement.single (first := 3086) modelrow_agrees_3087
private def atomic_3088 : CountSegment 1 10 13120 3087 3088 1499 1500 := CountSegment.single (first := 3087) (initial := 1499) row_count_3088 (by decide)
private theorem atomic_3088_agreement : RowAgreement 3087 3088 := RowAgreement.single (first := 3087) modelrow_agrees_3088
private def atomic_3089 : CountSegment 1 10 13120 3088 3089 1500 1501 := CountSegment.single (first := 3088) (initial := 1500) row_count_3089 (by decide)
private theorem atomic_3089_agreement : RowAgreement 3088 3089 := RowAgreement.single (first := 3088) modelrow_agrees_3089
private def atomic_3090 : CountSegment 1 10 13120 3089 3090 1501 1501 := CountSegment.single (first := 3089) (initial := 1501) row_count_3090 (by decide)
private theorem atomic_3090_agreement : RowAgreement 3089 3090 := RowAgreement.single (first := 3089) modelrow_agrees_3090
private def atomic_3091 : CountSegment 1 10 13120 3090 3091 1501 1501 := CountSegment.single (first := 3090) (initial := 1501) row_count_3091 (by decide)
private theorem atomic_3091_agreement : RowAgreement 3090 3091 := RowAgreement.single (first := 3090) modelrow_agrees_3091
private def atomic_3092 : CountSegment 1 10 13120 3091 3092 1501 1501 := CountSegment.single (first := 3091) (initial := 1501) row_count_3092 (by decide)
private theorem atomic_3092_agreement : RowAgreement 3091 3092 := RowAgreement.single (first := 3091) modelrow_agrees_3092
private def atomic_3093 : CountSegment 1 10 13120 3092 3093 1501 1501 := CountSegment.single (first := 3092) (initial := 1501) row_count_3093 (by decide)
private theorem atomic_3093_agreement : RowAgreement 3092 3093 := RowAgreement.single (first := 3092) modelrow_agrees_3093
private def atomic_3094 : CountSegment 1 10 13120 3093 3094 1501 1501 := CountSegment.single (first := 3093) (initial := 1501) row_count_3094 (by decide)
private theorem atomic_3094_agreement : RowAgreement 3093 3094 := RowAgreement.single (first := 3093) modelrow_agrees_3094
private def atomic_3095 : CountSegment 1 10 13120 3094 3095 1501 1502 := CountSegment.single (first := 3094) (initial := 1501) row_count_3095 (by decide)
private theorem atomic_3095_agreement : RowAgreement 3094 3095 := RowAgreement.single (first := 3094) modelrow_agrees_3095
private def atomic_3096 : CountSegment 1 10 13120 3095 3096 1502 1502 := CountSegment.single (first := 3095) (initial := 1502) row_count_3096 (by decide)
private theorem atomic_3096_agreement : RowAgreement 3095 3096 := RowAgreement.single (first := 3095) modelrow_agrees_3096
private def atomic_3097 : CountSegment 1 10 13120 3096 3097 1502 1502 := CountSegment.single (first := 3096) (initial := 1502) row_count_3097 (by decide)
private theorem atomic_3097_agreement : RowAgreement 3096 3097 := RowAgreement.single (first := 3096) modelrow_agrees_3097
private def atomic_3098 : CountSegment 1 10 13120 3097 3098 1502 1502 := CountSegment.single (first := 3097) (initial := 1502) row_count_3098 (by decide)
private theorem atomic_3098_agreement : RowAgreement 3097 3098 := RowAgreement.single (first := 3097) modelrow_agrees_3098
private def atomic_3099 : CountSegment 1 10 13120 3098 3099 1502 1502 := CountSegment.single (first := 3098) (initial := 1502) row_count_3099 (by decide)
private theorem atomic_3099_agreement : RowAgreement 3098 3099 := RowAgreement.single (first := 3098) modelrow_agrees_3099
private def atomic_3100 : CountSegment 1 10 13120 3099 3100 1502 1503 := CountSegment.single (first := 3099) (initial := 1502) row_count_3100 (by decide)
private theorem atomic_3100_agreement : RowAgreement 3099 3100 := RowAgreement.single (first := 3099) modelrow_agrees_3100
private def atomic_3101 : CountSegment 1 10 13120 3100 3101 1503 1503 := CountSegment.single (first := 3100) (initial := 1503) row_count_3101 (by decide)
private theorem atomic_3101_agreement : RowAgreement 3100 3101 := RowAgreement.single (first := 3100) modelrow_agrees_3101
private def atomic_3102 : CountSegment 1 10 13120 3101 3102 1503 1503 := CountSegment.single (first := 3101) (initial := 1503) row_count_3102 (by decide)
private theorem atomic_3102_agreement : RowAgreement 3101 3102 := RowAgreement.single (first := 3101) modelrow_agrees_3102
private def atomic_3103 : CountSegment 1 10 13120 3102 3103 1503 1503 := CountSegment.single (first := 3102) (initial := 1503) row_count_3103 (by decide)
private theorem atomic_3103_agreement : RowAgreement 3102 3103 := RowAgreement.single (first := 3102) modelrow_agrees_3103
private def atomic_3104 : CountSegment 1 10 13120 3103 3104 1503 1504 := CountSegment.single (first := 3103) (initial := 1503) row_count_3104 (by decide)
private theorem atomic_3104_agreement : RowAgreement 3103 3104 := RowAgreement.single (first := 3103) modelrow_agrees_3104
private def atomic_3105 : CountSegment 1 10 13120 3104 3105 1504 1504 := CountSegment.single (first := 3104) (initial := 1504) row_count_3105 (by decide)
private theorem atomic_3105_agreement : RowAgreement 3104 3105 := RowAgreement.single (first := 3104) modelrow_agrees_3105
private def atomic_3106 : CountSegment 1 10 13120 3105 3106 1504 1504 := CountSegment.single (first := 3105) (initial := 1504) row_count_3106 (by decide)
private theorem atomic_3106_agreement : RowAgreement 3105 3106 := RowAgreement.single (first := 3105) modelrow_agrees_3106
private def atomic_3107 : CountSegment 1 10 13120 3106 3107 1504 1504 := CountSegment.single (first := 3106) (initial := 1504) row_count_3107 (by decide)
private theorem atomic_3107_agreement : RowAgreement 3106 3107 := RowAgreement.single (first := 3106) modelrow_agrees_3107
private def atomic_3108 : CountSegment 1 10 13120 3107 3108 1504 1506 := CountSegment.single (first := 3107) (initial := 1504) row_count_3108 (by decide)
private theorem atomic_3108_agreement : RowAgreement 3107 3108 := RowAgreement.single (first := 3107) modelrow_agrees_3108
private def atomic_3109 : CountSegment 1 10 13120 3108 3109 1506 1506 := CountSegment.single (first := 3108) (initial := 1506) row_count_3109 (by decide)
private theorem atomic_3109_agreement : RowAgreement 3108 3109 := RowAgreement.single (first := 3108) modelrow_agrees_3109
private def atomic_3110 : CountSegment 1 10 13120 3109 3110 1506 1508 := CountSegment.single (first := 3109) (initial := 1506) row_count_3110 (by decide)
private theorem atomic_3110_agreement : RowAgreement 3109 3110 := RowAgreement.single (first := 3109) modelrow_agrees_3110
private def atomic_3111 : CountSegment 1 10 13120 3110 3111 1508 1509 := CountSegment.single (first := 3110) (initial := 1508) row_count_3111 (by decide)
private theorem atomic_3111_agreement : RowAgreement 3110 3111 := RowAgreement.single (first := 3110) modelrow_agrees_3111
private def atomic_3112 : CountSegment 1 10 13120 3111 3112 1509 1509 := CountSegment.single (first := 3111) (initial := 1509) row_count_3112 (by decide)
private theorem atomic_3112_agreement : RowAgreement 3111 3112 := RowAgreement.single (first := 3111) modelrow_agrees_3112
private def atomic_3113 : CountSegment 1 10 13120 3112 3113 1509 1509 := CountSegment.single (first := 3112) (initial := 1509) row_count_3113 (by decide)
private theorem atomic_3113_agreement : RowAgreement 3112 3113 := RowAgreement.single (first := 3112) modelrow_agrees_3113
private def atomic_3114 : CountSegment 1 10 13120 3113 3114 1509 1509 := CountSegment.single (first := 3113) (initial := 1509) row_count_3114 (by decide)
private theorem atomic_3114_agreement : RowAgreement 3113 3114 := RowAgreement.single (first := 3113) modelrow_agrees_3114
private def atomic_3115 : CountSegment 1 10 13120 3114 3115 1509 1509 := CountSegment.single (first := 3114) (initial := 1509) row_count_3115 (by decide)
private theorem atomic_3115_agreement : RowAgreement 3114 3115 := RowAgreement.single (first := 3114) modelrow_agrees_3115
private def atomic_3116 : CountSegment 1 10 13120 3115 3116 1509 1510 := CountSegment.single (first := 3115) (initial := 1509) row_count_3116 (by decide)
private theorem atomic_3116_agreement : RowAgreement 3115 3116 := RowAgreement.single (first := 3115) modelrow_agrees_3116
private def atomic_3117 : CountSegment 1 10 13120 3116 3117 1510 1511 := CountSegment.single (first := 3116) (initial := 1510) row_count_3117 (by decide)
private theorem atomic_3117_agreement : RowAgreement 3116 3117 := RowAgreement.single (first := 3116) modelrow_agrees_3117
private def atomic_3118 : CountSegment 1 10 13120 3117 3118 1511 1512 := CountSegment.single (first := 3117) (initial := 1511) row_count_3118 (by decide)
private theorem atomic_3118_agreement : RowAgreement 3117 3118 := RowAgreement.single (first := 3117) modelrow_agrees_3118
private def atomic_3119 : CountSegment 1 10 13120 3118 3119 1512 1513 := CountSegment.single (first := 3118) (initial := 1512) row_count_3119 (by decide)
private theorem atomic_3119_agreement : RowAgreement 3118 3119 := RowAgreement.single (first := 3118) modelrow_agrees_3119
private def atomic_3120 : CountSegment 1 10 13120 3119 3120 1513 1514 := CountSegment.single (first := 3119) (initial := 1513) row_count_3120 (by decide)
private theorem atomic_3120_agreement : RowAgreement 3119 3120 := RowAgreement.single (first := 3119) modelrow_agrees_3120
private def atomic_3121 : CountSegment 1 10 13120 3120 3121 1514 1514 := CountSegment.single (first := 3120) (initial := 1514) row_count_3121 (by decide)
private theorem atomic_3121_agreement : RowAgreement 3120 3121 := RowAgreement.single (first := 3120) modelrow_agrees_3121
private def atomic_3122 : CountSegment 1 10 13120 3121 3122 1514 1514 := CountSegment.single (first := 3121) (initial := 1514) row_count_3122 (by decide)
private theorem atomic_3122_agreement : RowAgreement 3121 3122 := RowAgreement.single (first := 3121) modelrow_agrees_3122
private def atomic_3123 : CountSegment 1 10 13120 3122 3123 1514 1514 := CountSegment.single (first := 3122) (initial := 1514) row_count_3123 (by decide)
private theorem atomic_3123_agreement : RowAgreement 3122 3123 := RowAgreement.single (first := 3122) modelrow_agrees_3123
private def atomic_3124 : CountSegment 1 10 13120 3123 3124 1514 1514 := CountSegment.single (first := 3123) (initial := 1514) row_count_3124 (by decide)
private theorem atomic_3124_agreement : RowAgreement 3123 3124 := RowAgreement.single (first := 3123) modelrow_agrees_3124
private def atomic_3125 : CountSegment 1 10 13120 3124 3125 1514 1514 := CountSegment.single (first := 3124) (initial := 1514) row_count_3125 (by decide)
private theorem atomic_3125_agreement : RowAgreement 3124 3125 := RowAgreement.single (first := 3124) modelrow_agrees_3125
private def atomic_3126 : CountSegment 1 10 13120 3125 3126 1514 1515 := CountSegment.single (first := 3125) (initial := 1514) row_count_3126 (by decide)
private theorem atomic_3126_agreement : RowAgreement 3125 3126 := RowAgreement.single (first := 3125) modelrow_agrees_3126
private def atomic_3127 : CountSegment 1 10 13120 3126 3127 1515 1516 := CountSegment.single (first := 3126) (initial := 1515) row_count_3127 (by decide)
private theorem atomic_3127_agreement : RowAgreement 3126 3127 := RowAgreement.single (first := 3126) modelrow_agrees_3127
private def atomic_3128 : CountSegment 1 10 13120 3127 3128 1516 1516 := CountSegment.single (first := 3127) (initial := 1516) row_count_3128 (by decide)
private theorem atomic_3128_agreement : RowAgreement 3127 3128 := RowAgreement.single (first := 3127) modelrow_agrees_3128
private def atomic_3129 : CountSegment 1 10 13120 3128 3129 1516 1516 := CountSegment.single (first := 3128) (initial := 1516) row_count_3129 (by decide)
private theorem atomic_3129_agreement : RowAgreement 3128 3129 := RowAgreement.single (first := 3128) modelrow_agrees_3129
private def atomic_3130 : CountSegment 1 10 13120 3129 3130 1516 1516 := CountSegment.single (first := 3129) (initial := 1516) row_count_3130 (by decide)
private theorem atomic_3130_agreement : RowAgreement 3129 3130 := RowAgreement.single (first := 3129) modelrow_agrees_3130
private def atomic_3131 : CountSegment 1 10 13120 3130 3131 1516 1516 := CountSegment.single (first := 3130) (initial := 1516) row_count_3131 (by decide)
private theorem atomic_3131_agreement : RowAgreement 3130 3131 := RowAgreement.single (first := 3130) modelrow_agrees_3131
private def atomic_3132 : CountSegment 1 10 13120 3131 3132 1516 1516 := CountSegment.single (first := 3131) (initial := 1516) row_count_3132 (by decide)
private theorem atomic_3132_agreement : RowAgreement 3131 3132 := RowAgreement.single (first := 3131) modelrow_agrees_3132
private def atomic_3133 : CountSegment 1 10 13120 3132 3133 1516 1517 := CountSegment.single (first := 3132) (initial := 1516) row_count_3133 (by decide)
private theorem atomic_3133_agreement : RowAgreement 3132 3133 := RowAgreement.single (first := 3132) modelrow_agrees_3133
private def atomic_3134 : CountSegment 1 10 13120 3133 3134 1517 1517 := CountSegment.single (first := 3133) (initial := 1517) row_count_3134 (by decide)
private theorem atomic_3134_agreement : RowAgreement 3133 3134 := RowAgreement.single (first := 3133) modelrow_agrees_3134
private def atomic_3135 : CountSegment 1 10 13120 3134 3135 1517 1517 := CountSegment.single (first := 3134) (initial := 1517) row_count_3135 (by decide)
private theorem atomic_3135_agreement : RowAgreement 3134 3135 := RowAgreement.single (first := 3134) modelrow_agrees_3135
private def atomic_3136 : CountSegment 1 10 13120 3135 3136 1517 1518 := CountSegment.single (first := 3135) (initial := 1517) row_count_3136 (by decide)
private theorem atomic_3136_agreement : RowAgreement 3135 3136 := RowAgreement.single (first := 3135) modelrow_agrees_3136
private def atomic_3137 : CountSegment 1 10 13120 3136 3137 1518 1518 := CountSegment.single (first := 3136) (initial := 1518) row_count_3137 (by decide)
private theorem atomic_3137_agreement : RowAgreement 3136 3137 := RowAgreement.single (first := 3136) modelrow_agrees_3137
private def atomic_3138 : CountSegment 1 10 13120 3137 3138 1518 1518 := CountSegment.single (first := 3137) (initial := 1518) row_count_3138 (by decide)
private theorem atomic_3138_agreement : RowAgreement 3137 3138 := RowAgreement.single (first := 3137) modelrow_agrees_3138
private def atomic_3139 : CountSegment 1 10 13120 3138 3139 1518 1518 := CountSegment.single (first := 3138) (initial := 1518) row_count_3139 (by decide)
private theorem atomic_3139_agreement : RowAgreement 3138 3139 := RowAgreement.single (first := 3138) modelrow_agrees_3139
private def atomic_3140 : CountSegment 1 10 13120 3139 3140 1518 1518 := CountSegment.single (first := 3139) (initial := 1518) row_count_3140 (by decide)
private theorem atomic_3140_agreement : RowAgreement 3139 3140 := RowAgreement.single (first := 3139) modelrow_agrees_3140
private def atomic_3141 : CountSegment 1 10 13120 3140 3141 1518 1518 := CountSegment.single (first := 3140) (initial := 1518) row_count_3141 (by decide)
private theorem atomic_3141_agreement : RowAgreement 3140 3141 := RowAgreement.single (first := 3140) modelrow_agrees_3141
private def atomic_3142 : CountSegment 1 10 13120 3141 3142 1518 1518 := CountSegment.single (first := 3141) (initial := 1518) row_count_3142 (by decide)
private theorem atomic_3142_agreement : RowAgreement 3141 3142 := RowAgreement.single (first := 3141) modelrow_agrees_3142
private def atomic_3143 : CountSegment 1 10 13120 3142 3143 1518 1519 := CountSegment.single (first := 3142) (initial := 1518) row_count_3143 (by decide)
private theorem atomic_3143_agreement : RowAgreement 3142 3143 := RowAgreement.single (first := 3142) modelrow_agrees_3143
private def atomic_3144 : CountSegment 1 10 13120 3143 3144 1519 1519 := CountSegment.single (first := 3143) (initial := 1519) row_count_3144 (by decide)
private theorem atomic_3144_agreement : RowAgreement 3143 3144 := RowAgreement.single (first := 3143) modelrow_agrees_3144
private def atomic_3145 : CountSegment 1 10 13120 3144 3145 1519 1519 := CountSegment.single (first := 3144) (initial := 1519) row_count_3145 (by decide)
private theorem atomic_3145_agreement : RowAgreement 3144 3145 := RowAgreement.single (first := 3144) modelrow_agrees_3145
private def atomic_3146 : CountSegment 1 10 13120 3145 3146 1519 1519 := CountSegment.single (first := 3145) (initial := 1519) row_count_3146 (by decide)
private theorem atomic_3146_agreement : RowAgreement 3145 3146 := RowAgreement.single (first := 3145) modelrow_agrees_3146
private def atomic_3147 : CountSegment 1 10 13120 3146 3147 1519 1519 := CountSegment.single (first := 3146) (initial := 1519) row_count_3147 (by decide)
private theorem atomic_3147_agreement : RowAgreement 3146 3147 := RowAgreement.single (first := 3146) modelrow_agrees_3147
private def atomic_3148 : CountSegment 1 10 13120 3147 3148 1519 1519 := CountSegment.single (first := 3147) (initial := 1519) row_count_3148 (by decide)
private theorem atomic_3148_agreement : RowAgreement 3147 3148 := RowAgreement.single (first := 3147) modelrow_agrees_3148
private def atomic_3149 : CountSegment 1 10 13120 3148 3149 1519 1519 := CountSegment.single (first := 3148) (initial := 1519) row_count_3149 (by decide)
private theorem atomic_3149_agreement : RowAgreement 3148 3149 := RowAgreement.single (first := 3148) modelrow_agrees_3149
private def atomic_3150 : CountSegment 1 10 13120 3149 3150 1519 1519 := CountSegment.single (first := 3149) (initial := 1519) row_count_3150 (by decide)
private theorem atomic_3150_agreement : RowAgreement 3149 3150 := RowAgreement.single (first := 3149) modelrow_agrees_3150
private def atomic_3151 : CountSegment 1 10 13120 3150 3151 1519 1519 := CountSegment.single (first := 3150) (initial := 1519) row_count_3151 (by decide)
private theorem atomic_3151_agreement : RowAgreement 3150 3151 := RowAgreement.single (first := 3150) modelrow_agrees_3151
private def atomic_3152 : CountSegment 1 10 13120 3151 3152 1519 1520 := CountSegment.single (first := 3151) (initial := 1519) row_count_3152 (by decide)
private theorem atomic_3152_agreement : RowAgreement 3151 3152 := RowAgreement.single (first := 3151) modelrow_agrees_3152
private def atomic_3153 : CountSegment 1 10 13120 3152 3153 1520 1523 := CountSegment.single (first := 3152) (initial := 1520) row_count_3153 (by decide)
private theorem atomic_3153_agreement : RowAgreement 3152 3153 := RowAgreement.single (first := 3152) modelrow_agrees_3153
private def atomic_3154 : CountSegment 1 10 13120 3153 3154 1523 1523 := CountSegment.single (first := 3153) (initial := 1523) row_count_3154 (by decide)
private theorem atomic_3154_agreement : RowAgreement 3153 3154 := RowAgreement.single (first := 3153) modelrow_agrees_3154
private def atomic_3155 : CountSegment 1 10 13120 3154 3155 1523 1523 := CountSegment.single (first := 3154) (initial := 1523) row_count_3155 (by decide)
private theorem atomic_3155_agreement : RowAgreement 3154 3155 := RowAgreement.single (first := 3154) modelrow_agrees_3155
private def atomic_3156 : CountSegment 1 10 13120 3155 3156 1523 1523 := CountSegment.single (first := 3155) (initial := 1523) row_count_3156 (by decide)
private theorem atomic_3156_agreement : RowAgreement 3155 3156 := RowAgreement.single (first := 3155) modelrow_agrees_3156
private def atomic_3157 : CountSegment 1 10 13120 3156 3157 1523 1524 := CountSegment.single (first := 3156) (initial := 1523) row_count_3157 (by decide)
private theorem atomic_3157_agreement : RowAgreement 3156 3157 := RowAgreement.single (first := 3156) modelrow_agrees_3157
private def atomic_3158 : CountSegment 1 10 13120 3157 3158 1524 1524 := CountSegment.single (first := 3157) (initial := 1524) row_count_3158 (by decide)
private theorem atomic_3158_agreement : RowAgreement 3157 3158 := RowAgreement.single (first := 3157) modelrow_agrees_3158
private def atomic_3159 : CountSegment 1 10 13120 3158 3159 1524 1524 := CountSegment.single (first := 3158) (initial := 1524) row_count_3159 (by decide)
private theorem atomic_3159_agreement : RowAgreement 3158 3159 := RowAgreement.single (first := 3158) modelrow_agrees_3159
private def atomic_3160 : CountSegment 1 10 13120 3159 3160 1524 1524 := CountSegment.single (first := 3159) (initial := 1524) row_count_3160 (by decide)
private theorem atomic_3160_agreement : RowAgreement 3159 3160 := RowAgreement.single (first := 3159) modelrow_agrees_3160
private def atomic_3161 : CountSegment 1 10 13120 3160 3161 1524 1524 := CountSegment.single (first := 3160) (initial := 1524) row_count_3161 (by decide)
private theorem atomic_3161_agreement : RowAgreement 3160 3161 := RowAgreement.single (first := 3160) modelrow_agrees_3161
private def atomic_3162 : CountSegment 1 10 13120 3161 3162 1524 1525 := CountSegment.single (first := 3161) (initial := 1524) row_count_3162 (by decide)
private theorem atomic_3162_agreement : RowAgreement 3161 3162 := RowAgreement.single (first := 3161) modelrow_agrees_3162
private def atomic_3163 : CountSegment 1 10 13120 3162 3163 1525 1526 := CountSegment.single (first := 3162) (initial := 1525) row_count_3163 (by decide)
private theorem atomic_3163_agreement : RowAgreement 3162 3163 := RowAgreement.single (first := 3162) modelrow_agrees_3163
private def atomic_3164 : CountSegment 1 10 13120 3163 3164 1526 1526 := CountSegment.single (first := 3163) (initial := 1526) row_count_3164 (by decide)
private theorem atomic_3164_agreement : RowAgreement 3163 3164 := RowAgreement.single (first := 3163) modelrow_agrees_3164
private def atomic_3165 : CountSegment 1 10 13120 3164 3165 1526 1526 := CountSegment.single (first := 3164) (initial := 1526) row_count_3165 (by decide)
private theorem atomic_3165_agreement : RowAgreement 3164 3165 := RowAgreement.single (first := 3164) modelrow_agrees_3165
private def atomic_3166 : CountSegment 1 10 13120 3165 3166 1526 1526 := CountSegment.single (first := 3165) (initial := 1526) row_count_3166 (by decide)
private theorem atomic_3166_agreement : RowAgreement 3165 3166 := RowAgreement.single (first := 3165) modelrow_agrees_3166
private def atomic_3167 : CountSegment 1 10 13120 3166 3167 1526 1526 := CountSegment.single (first := 3166) (initial := 1526) row_count_3167 (by decide)
private theorem atomic_3167_agreement : RowAgreement 3166 3167 := RowAgreement.single (first := 3166) modelrow_agrees_3167
private def atomic_3168 : CountSegment 1 10 13120 3167 3168 1526 1526 := CountSegment.single (first := 3167) (initial := 1526) row_count_3168 (by decide)
private theorem atomic_3168_agreement : RowAgreement 3167 3168 := RowAgreement.single (first := 3167) modelrow_agrees_3168
private def atomic_3169 : CountSegment 1 10 13120 3168 3169 1526 1526 := CountSegment.single (first := 3168) (initial := 1526) row_count_3169 (by decide)
private theorem atomic_3169_agreement : RowAgreement 3168 3169 := RowAgreement.single (first := 3168) modelrow_agrees_3169
private def atomic_3170 : CountSegment 1 10 13120 3169 3170 1526 1527 := CountSegment.single (first := 3169) (initial := 1526) row_count_3170 (by decide)
private theorem atomic_3170_agreement : RowAgreement 3169 3170 := RowAgreement.single (first := 3169) modelrow_agrees_3170
private def atomic_3171 : CountSegment 1 10 13120 3170 3171 1527 1528 := CountSegment.single (first := 3170) (initial := 1527) row_count_3171 (by decide)
private theorem atomic_3171_agreement : RowAgreement 3170 3171 := RowAgreement.single (first := 3170) modelrow_agrees_3171
private def atomic_3172 : CountSegment 1 10 13120 3171 3172 1528 1528 := CountSegment.single (first := 3171) (initial := 1528) row_count_3172 (by decide)
private theorem atomic_3172_agreement : RowAgreement 3171 3172 := RowAgreement.single (first := 3171) modelrow_agrees_3172
private def atomic_3173 : CountSegment 1 10 13120 3172 3173 1528 1528 := CountSegment.single (first := 3172) (initial := 1528) row_count_3173 (by decide)
private theorem atomic_3173_agreement : RowAgreement 3172 3173 := RowAgreement.single (first := 3172) modelrow_agrees_3173
private def atomic_3174 : CountSegment 1 10 13120 3173 3174 1528 1529 := CountSegment.single (first := 3173) (initial := 1528) row_count_3174 (by decide)
private theorem atomic_3174_agreement : RowAgreement 3173 3174 := RowAgreement.single (first := 3173) modelrow_agrees_3174
private def atomic_3175 : CountSegment 1 10 13120 3174 3175 1529 1531 := CountSegment.single (first := 3174) (initial := 1529) row_count_3175 (by decide)
private theorem atomic_3175_agreement : RowAgreement 3174 3175 := RowAgreement.single (first := 3174) modelrow_agrees_3175
private def atomic_3176 : CountSegment 1 10 13120 3175 3176 1531 1531 := CountSegment.single (first := 3175) (initial := 1531) row_count_3176 (by decide)
private theorem atomic_3176_agreement : RowAgreement 3175 3176 := RowAgreement.single (first := 3175) modelrow_agrees_3176
private def atomic_3177 : CountSegment 1 10 13120 3176 3177 1531 1532 := CountSegment.single (first := 3176) (initial := 1531) row_count_3177 (by decide)
private theorem atomic_3177_agreement : RowAgreement 3176 3177 := RowAgreement.single (first := 3176) modelrow_agrees_3177
private def atomic_3178 : CountSegment 1 10 13120 3177 3178 1532 1532 := CountSegment.single (first := 3177) (initial := 1532) row_count_3178 (by decide)
private theorem atomic_3178_agreement : RowAgreement 3177 3178 := RowAgreement.single (first := 3177) modelrow_agrees_3178
private def atomic_3179 : CountSegment 1 10 13120 3178 3179 1532 1532 := CountSegment.single (first := 3178) (initial := 1532) row_count_3179 (by decide)
private theorem atomic_3179_agreement : RowAgreement 3178 3179 := RowAgreement.single (first := 3178) modelrow_agrees_3179
private def atomic_3180 : CountSegment 1 10 13120 3179 3180 1532 1533 := CountSegment.single (first := 3179) (initial := 1532) row_count_3180 (by decide)
private theorem atomic_3180_agreement : RowAgreement 3179 3180 := RowAgreement.single (first := 3179) modelrow_agrees_3180
private def atomic_3181 : CountSegment 1 10 13120 3180 3181 1533 1533 := CountSegment.single (first := 3180) (initial := 1533) row_count_3181 (by decide)
private theorem atomic_3181_agreement : RowAgreement 3180 3181 := RowAgreement.single (first := 3180) modelrow_agrees_3181
private def atomic_3182 : CountSegment 1 10 13120 3181 3182 1533 1534 := CountSegment.single (first := 3181) (initial := 1533) row_count_3182 (by decide)
private theorem atomic_3182_agreement : RowAgreement 3181 3182 := RowAgreement.single (first := 3181) modelrow_agrees_3182
private def atomic_3183 : CountSegment 1 10 13120 3182 3183 1534 1535 := CountSegment.single (first := 3182) (initial := 1534) row_count_3183 (by decide)
private theorem atomic_3183_agreement : RowAgreement 3182 3183 := RowAgreement.single (first := 3182) modelrow_agrees_3183
private def atomic_3184 : CountSegment 1 10 13120 3183 3184 1535 1536 := CountSegment.single (first := 3183) (initial := 1535) row_count_3184 (by decide)
private theorem atomic_3184_agreement : RowAgreement 3183 3184 := RowAgreement.single (first := 3183) modelrow_agrees_3184
private def atomic_3185 : CountSegment 1 10 13120 3184 3185 1536 1536 := CountSegment.single (first := 3184) (initial := 1536) row_count_3185 (by decide)
private theorem atomic_3185_agreement : RowAgreement 3184 3185 := RowAgreement.single (first := 3184) modelrow_agrees_3185
private def atomic_3186 : CountSegment 1 10 13120 3185 3186 1536 1537 := CountSegment.single (first := 3185) (initial := 1536) row_count_3186 (by decide)
private theorem atomic_3186_agreement : RowAgreement 3185 3186 := RowAgreement.single (first := 3185) modelrow_agrees_3186
private def atomic_3187 : CountSegment 1 10 13120 3186 3187 1537 1537 := CountSegment.single (first := 3186) (initial := 1537) row_count_3187 (by decide)
private theorem atomic_3187_agreement : RowAgreement 3186 3187 := RowAgreement.single (first := 3186) modelrow_agrees_3187
private def atomic_3188 : CountSegment 1 10 13120 3187 3188 1537 1537 := CountSegment.single (first := 3187) (initial := 1537) row_count_3188 (by decide)
private theorem atomic_3188_agreement : RowAgreement 3187 3188 := RowAgreement.single (first := 3187) modelrow_agrees_3188
private def atomic_3189 : CountSegment 1 10 13120 3188 3189 1537 1539 := CountSegment.single (first := 3188) (initial := 1537) row_count_3189 (by decide)
private theorem atomic_3189_agreement : RowAgreement 3188 3189 := RowAgreement.single (first := 3188) modelrow_agrees_3189
private def atomic_3190 : CountSegment 1 10 13120 3189 3190 1539 1539 := CountSegment.single (first := 3189) (initial := 1539) row_count_3190 (by decide)
private theorem atomic_3190_agreement : RowAgreement 3189 3190 := RowAgreement.single (first := 3189) modelrow_agrees_3190
private def atomic_3191 : CountSegment 1 10 13120 3190 3191 1539 1539 := CountSegment.single (first := 3190) (initial := 1539) row_count_3191 (by decide)
private theorem atomic_3191_agreement : RowAgreement 3190 3191 := RowAgreement.single (first := 3190) modelrow_agrees_3191
private def atomic_3192 : CountSegment 1 10 13120 3191 3192 1539 1539 := CountSegment.single (first := 3191) (initial := 1539) row_count_3192 (by decide)
private theorem atomic_3192_agreement : RowAgreement 3191 3192 := RowAgreement.single (first := 3191) modelrow_agrees_3192
private def atomic_3193 : CountSegment 1 10 13120 3192 3193 1539 1539 := CountSegment.single (first := 3192) (initial := 1539) row_count_3193 (by decide)
private theorem atomic_3193_agreement : RowAgreement 3192 3193 := RowAgreement.single (first := 3192) modelrow_agrees_3193
private def atomic_3194 : CountSegment 1 10 13120 3193 3194 1539 1539 := CountSegment.single (first := 3193) (initial := 1539) row_count_3194 (by decide)
private theorem atomic_3194_agreement : RowAgreement 3193 3194 := RowAgreement.single (first := 3193) modelrow_agrees_3194
private def atomic_3195 : CountSegment 1 10 13120 3194 3195 1539 1540 := CountSegment.single (first := 3194) (initial := 1539) row_count_3195 (by decide)
private theorem atomic_3195_agreement : RowAgreement 3194 3195 := RowAgreement.single (first := 3194) modelrow_agrees_3195
private def atomic_3196 : CountSegment 1 10 13120 3195 3196 1540 1540 := CountSegment.single (first := 3195) (initial := 1540) row_count_3196 (by decide)
private theorem atomic_3196_agreement : RowAgreement 3195 3196 := RowAgreement.single (first := 3195) modelrow_agrees_3196
private def atomic_3197 : CountSegment 1 10 13120 3196 3197 1540 1540 := CountSegment.single (first := 3196) (initial := 1540) row_count_3197 (by decide)
private theorem atomic_3197_agreement : RowAgreement 3196 3197 := RowAgreement.single (first := 3196) modelrow_agrees_3197
private def atomic_3198 : CountSegment 1 10 13120 3197 3198 1540 1541 := CountSegment.single (first := 3197) (initial := 1540) row_count_3198 (by decide)
private theorem atomic_3198_agreement : RowAgreement 3197 3198 := RowAgreement.single (first := 3197) modelrow_agrees_3198
private def atomic_3199 : CountSegment 1 10 13120 3198 3199 1541 1541 := CountSegment.single (first := 3198) (initial := 1541) row_count_3199 (by decide)
private theorem atomic_3199_agreement : RowAgreement 3198 3199 := RowAgreement.single (first := 3198) modelrow_agrees_3199
private def atomic_3200 : CountSegment 1 10 13120 3199 3200 1541 1542 := CountSegment.single (first := 3199) (initial := 1541) row_count_3200 (by decide)
private theorem atomic_3200_agreement : RowAgreement 3199 3200 := RowAgreement.single (first := 3199) modelrow_agrees_3200
private def within_024_000 : CountSegment 1 10 13120 3072 3074 1493 1493 := CountSegment.join atomic_3073 atomic_3074
private theorem within_024_000_agreement : RowAgreement 3072 3074 := RowAgreement.join atomic_3073_agreement atomic_3074_agreement
private def within_024_001 : CountSegment 1 10 13120 3074 3076 1493 1493 := CountSegment.join atomic_3075 atomic_3076
private theorem within_024_001_agreement : RowAgreement 3074 3076 := RowAgreement.join atomic_3075_agreement atomic_3076_agreement
private def within_024_002 : CountSegment 1 10 13120 3072 3076 1493 1493 := CountSegment.join within_024_000 within_024_001
private theorem within_024_002_agreement : RowAgreement 3072 3076 := RowAgreement.join within_024_000_agreement within_024_001_agreement
private def within_024_003 : CountSegment 1 10 13120 3076 3078 1493 1495 := CountSegment.join atomic_3077 atomic_3078
private theorem within_024_003_agreement : RowAgreement 3076 3078 := RowAgreement.join atomic_3077_agreement atomic_3078_agreement
private def within_024_004 : CountSegment 1 10 13120 3078 3080 1495 1495 := CountSegment.join atomic_3079 atomic_3080
private theorem within_024_004_agreement : RowAgreement 3078 3080 := RowAgreement.join atomic_3079_agreement atomic_3080_agreement
private def within_024_005 : CountSegment 1 10 13120 3076 3080 1493 1495 := CountSegment.join within_024_003 within_024_004
private theorem within_024_005_agreement : RowAgreement 3076 3080 := RowAgreement.join within_024_003_agreement within_024_004_agreement
private def within_024_006 : CountSegment 1 10 13120 3072 3080 1493 1495 := CountSegment.join within_024_002 within_024_005
private theorem within_024_006_agreement : RowAgreement 3072 3080 := RowAgreement.join within_024_002_agreement within_024_005_agreement
private def within_024_007 : CountSegment 1 10 13120 3080 3082 1495 1496 := CountSegment.join atomic_3081 atomic_3082
private theorem within_024_007_agreement : RowAgreement 3080 3082 := RowAgreement.join atomic_3081_agreement atomic_3082_agreement
private def within_024_008 : CountSegment 1 10 13120 3082 3084 1496 1497 := CountSegment.join atomic_3083 atomic_3084
private theorem within_024_008_agreement : RowAgreement 3082 3084 := RowAgreement.join atomic_3083_agreement atomic_3084_agreement
private def within_024_009 : CountSegment 1 10 13120 3080 3084 1495 1497 := CountSegment.join within_024_007 within_024_008
private theorem within_024_009_agreement : RowAgreement 3080 3084 := RowAgreement.join within_024_007_agreement within_024_008_agreement
private def within_024_010 : CountSegment 1 10 13120 3084 3086 1497 1498 := CountSegment.join atomic_3085 atomic_3086
private theorem within_024_010_agreement : RowAgreement 3084 3086 := RowAgreement.join atomic_3085_agreement atomic_3086_agreement
private def within_024_011 : CountSegment 1 10 13120 3086 3088 1498 1500 := CountSegment.join atomic_3087 atomic_3088
private theorem within_024_011_agreement : RowAgreement 3086 3088 := RowAgreement.join atomic_3087_agreement atomic_3088_agreement
private def within_024_012 : CountSegment 1 10 13120 3084 3088 1497 1500 := CountSegment.join within_024_010 within_024_011
private theorem within_024_012_agreement : RowAgreement 3084 3088 := RowAgreement.join within_024_010_agreement within_024_011_agreement
private def within_024_013 : CountSegment 1 10 13120 3080 3088 1495 1500 := CountSegment.join within_024_009 within_024_012
private theorem within_024_013_agreement : RowAgreement 3080 3088 := RowAgreement.join within_024_009_agreement within_024_012_agreement
private def within_024_014 : CountSegment 1 10 13120 3072 3088 1493 1500 := CountSegment.join within_024_006 within_024_013
private theorem within_024_014_agreement : RowAgreement 3072 3088 := RowAgreement.join within_024_006_agreement within_024_013_agreement
private def within_024_015 : CountSegment 1 10 13120 3088 3090 1500 1501 := CountSegment.join atomic_3089 atomic_3090
private theorem within_024_015_agreement : RowAgreement 3088 3090 := RowAgreement.join atomic_3089_agreement atomic_3090_agreement
private def within_024_016 : CountSegment 1 10 13120 3090 3092 1501 1501 := CountSegment.join atomic_3091 atomic_3092
private theorem within_024_016_agreement : RowAgreement 3090 3092 := RowAgreement.join atomic_3091_agreement atomic_3092_agreement
private def within_024_017 : CountSegment 1 10 13120 3088 3092 1500 1501 := CountSegment.join within_024_015 within_024_016
private theorem within_024_017_agreement : RowAgreement 3088 3092 := RowAgreement.join within_024_015_agreement within_024_016_agreement
private def within_024_018 : CountSegment 1 10 13120 3092 3094 1501 1501 := CountSegment.join atomic_3093 atomic_3094
private theorem within_024_018_agreement : RowAgreement 3092 3094 := RowAgreement.join atomic_3093_agreement atomic_3094_agreement
private def within_024_019 : CountSegment 1 10 13120 3094 3096 1501 1502 := CountSegment.join atomic_3095 atomic_3096
private theorem within_024_019_agreement : RowAgreement 3094 3096 := RowAgreement.join atomic_3095_agreement atomic_3096_agreement
private def within_024_020 : CountSegment 1 10 13120 3092 3096 1501 1502 := CountSegment.join within_024_018 within_024_019
private theorem within_024_020_agreement : RowAgreement 3092 3096 := RowAgreement.join within_024_018_agreement within_024_019_agreement
private def within_024_021 : CountSegment 1 10 13120 3088 3096 1500 1502 := CountSegment.join within_024_017 within_024_020
private theorem within_024_021_agreement : RowAgreement 3088 3096 := RowAgreement.join within_024_017_agreement within_024_020_agreement
private def within_024_022 : CountSegment 1 10 13120 3096 3098 1502 1502 := CountSegment.join atomic_3097 atomic_3098
private theorem within_024_022_agreement : RowAgreement 3096 3098 := RowAgreement.join atomic_3097_agreement atomic_3098_agreement
private def within_024_023 : CountSegment 1 10 13120 3098 3100 1502 1503 := CountSegment.join atomic_3099 atomic_3100
private theorem within_024_023_agreement : RowAgreement 3098 3100 := RowAgreement.join atomic_3099_agreement atomic_3100_agreement
private def within_024_024 : CountSegment 1 10 13120 3096 3100 1502 1503 := CountSegment.join within_024_022 within_024_023
private theorem within_024_024_agreement : RowAgreement 3096 3100 := RowAgreement.join within_024_022_agreement within_024_023_agreement
private def within_024_025 : CountSegment 1 10 13120 3100 3102 1503 1503 := CountSegment.join atomic_3101 atomic_3102
private theorem within_024_025_agreement : RowAgreement 3100 3102 := RowAgreement.join atomic_3101_agreement atomic_3102_agreement
private def within_024_026 : CountSegment 1 10 13120 3102 3104 1503 1504 := CountSegment.join atomic_3103 atomic_3104
private theorem within_024_026_agreement : RowAgreement 3102 3104 := RowAgreement.join atomic_3103_agreement atomic_3104_agreement
private def within_024_027 : CountSegment 1 10 13120 3100 3104 1503 1504 := CountSegment.join within_024_025 within_024_026
private theorem within_024_027_agreement : RowAgreement 3100 3104 := RowAgreement.join within_024_025_agreement within_024_026_agreement
private def within_024_028 : CountSegment 1 10 13120 3096 3104 1502 1504 := CountSegment.join within_024_024 within_024_027
private theorem within_024_028_agreement : RowAgreement 3096 3104 := RowAgreement.join within_024_024_agreement within_024_027_agreement
private def within_024_029 : CountSegment 1 10 13120 3088 3104 1500 1504 := CountSegment.join within_024_021 within_024_028
private theorem within_024_029_agreement : RowAgreement 3088 3104 := RowAgreement.join within_024_021_agreement within_024_028_agreement
private def within_024_030 : CountSegment 1 10 13120 3072 3104 1493 1504 := CountSegment.join within_024_014 within_024_029
private theorem within_024_030_agreement : RowAgreement 3072 3104 := RowAgreement.join within_024_014_agreement within_024_029_agreement
private def within_024_031 : CountSegment 1 10 13120 3104 3106 1504 1504 := CountSegment.join atomic_3105 atomic_3106
private theorem within_024_031_agreement : RowAgreement 3104 3106 := RowAgreement.join atomic_3105_agreement atomic_3106_agreement
private def within_024_032 : CountSegment 1 10 13120 3106 3108 1504 1506 := CountSegment.join atomic_3107 atomic_3108
private theorem within_024_032_agreement : RowAgreement 3106 3108 := RowAgreement.join atomic_3107_agreement atomic_3108_agreement
private def within_024_033 : CountSegment 1 10 13120 3104 3108 1504 1506 := CountSegment.join within_024_031 within_024_032
private theorem within_024_033_agreement : RowAgreement 3104 3108 := RowAgreement.join within_024_031_agreement within_024_032_agreement
private def within_024_034 : CountSegment 1 10 13120 3108 3110 1506 1508 := CountSegment.join atomic_3109 atomic_3110
private theorem within_024_034_agreement : RowAgreement 3108 3110 := RowAgreement.join atomic_3109_agreement atomic_3110_agreement
private def within_024_035 : CountSegment 1 10 13120 3110 3112 1508 1509 := CountSegment.join atomic_3111 atomic_3112
private theorem within_024_035_agreement : RowAgreement 3110 3112 := RowAgreement.join atomic_3111_agreement atomic_3112_agreement
private def within_024_036 : CountSegment 1 10 13120 3108 3112 1506 1509 := CountSegment.join within_024_034 within_024_035
private theorem within_024_036_agreement : RowAgreement 3108 3112 := RowAgreement.join within_024_034_agreement within_024_035_agreement
private def within_024_037 : CountSegment 1 10 13120 3104 3112 1504 1509 := CountSegment.join within_024_033 within_024_036
private theorem within_024_037_agreement : RowAgreement 3104 3112 := RowAgreement.join within_024_033_agreement within_024_036_agreement
private def within_024_038 : CountSegment 1 10 13120 3112 3114 1509 1509 := CountSegment.join atomic_3113 atomic_3114
private theorem within_024_038_agreement : RowAgreement 3112 3114 := RowAgreement.join atomic_3113_agreement atomic_3114_agreement
private def within_024_039 : CountSegment 1 10 13120 3114 3116 1509 1510 := CountSegment.join atomic_3115 atomic_3116
private theorem within_024_039_agreement : RowAgreement 3114 3116 := RowAgreement.join atomic_3115_agreement atomic_3116_agreement
private def within_024_040 : CountSegment 1 10 13120 3112 3116 1509 1510 := CountSegment.join within_024_038 within_024_039
private theorem within_024_040_agreement : RowAgreement 3112 3116 := RowAgreement.join within_024_038_agreement within_024_039_agreement
private def within_024_041 : CountSegment 1 10 13120 3116 3118 1510 1512 := CountSegment.join atomic_3117 atomic_3118
private theorem within_024_041_agreement : RowAgreement 3116 3118 := RowAgreement.join atomic_3117_agreement atomic_3118_agreement
private def within_024_042 : CountSegment 1 10 13120 3118 3120 1512 1514 := CountSegment.join atomic_3119 atomic_3120
private theorem within_024_042_agreement : RowAgreement 3118 3120 := RowAgreement.join atomic_3119_agreement atomic_3120_agreement
private def within_024_043 : CountSegment 1 10 13120 3116 3120 1510 1514 := CountSegment.join within_024_041 within_024_042
private theorem within_024_043_agreement : RowAgreement 3116 3120 := RowAgreement.join within_024_041_agreement within_024_042_agreement
private def within_024_044 : CountSegment 1 10 13120 3112 3120 1509 1514 := CountSegment.join within_024_040 within_024_043
private theorem within_024_044_agreement : RowAgreement 3112 3120 := RowAgreement.join within_024_040_agreement within_024_043_agreement
private def within_024_045 : CountSegment 1 10 13120 3104 3120 1504 1514 := CountSegment.join within_024_037 within_024_044
private theorem within_024_045_agreement : RowAgreement 3104 3120 := RowAgreement.join within_024_037_agreement within_024_044_agreement
private def within_024_046 : CountSegment 1 10 13120 3120 3122 1514 1514 := CountSegment.join atomic_3121 atomic_3122
private theorem within_024_046_agreement : RowAgreement 3120 3122 := RowAgreement.join atomic_3121_agreement atomic_3122_agreement
private def within_024_047 : CountSegment 1 10 13120 3122 3124 1514 1514 := CountSegment.join atomic_3123 atomic_3124
private theorem within_024_047_agreement : RowAgreement 3122 3124 := RowAgreement.join atomic_3123_agreement atomic_3124_agreement
private def within_024_048 : CountSegment 1 10 13120 3120 3124 1514 1514 := CountSegment.join within_024_046 within_024_047
private theorem within_024_048_agreement : RowAgreement 3120 3124 := RowAgreement.join within_024_046_agreement within_024_047_agreement
private def within_024_049 : CountSegment 1 10 13120 3124 3126 1514 1515 := CountSegment.join atomic_3125 atomic_3126
private theorem within_024_049_agreement : RowAgreement 3124 3126 := RowAgreement.join atomic_3125_agreement atomic_3126_agreement
private def within_024_050 : CountSegment 1 10 13120 3126 3128 1515 1516 := CountSegment.join atomic_3127 atomic_3128
private theorem within_024_050_agreement : RowAgreement 3126 3128 := RowAgreement.join atomic_3127_agreement atomic_3128_agreement
private def within_024_051 : CountSegment 1 10 13120 3124 3128 1514 1516 := CountSegment.join within_024_049 within_024_050
private theorem within_024_051_agreement : RowAgreement 3124 3128 := RowAgreement.join within_024_049_agreement within_024_050_agreement
private def within_024_052 : CountSegment 1 10 13120 3120 3128 1514 1516 := CountSegment.join within_024_048 within_024_051
private theorem within_024_052_agreement : RowAgreement 3120 3128 := RowAgreement.join within_024_048_agreement within_024_051_agreement
private def within_024_053 : CountSegment 1 10 13120 3128 3130 1516 1516 := CountSegment.join atomic_3129 atomic_3130
private theorem within_024_053_agreement : RowAgreement 3128 3130 := RowAgreement.join atomic_3129_agreement atomic_3130_agreement
private def within_024_054 : CountSegment 1 10 13120 3130 3132 1516 1516 := CountSegment.join atomic_3131 atomic_3132
private theorem within_024_054_agreement : RowAgreement 3130 3132 := RowAgreement.join atomic_3131_agreement atomic_3132_agreement
private def within_024_055 : CountSegment 1 10 13120 3128 3132 1516 1516 := CountSegment.join within_024_053 within_024_054
private theorem within_024_055_agreement : RowAgreement 3128 3132 := RowAgreement.join within_024_053_agreement within_024_054_agreement
private def within_024_056 : CountSegment 1 10 13120 3132 3134 1516 1517 := CountSegment.join atomic_3133 atomic_3134
private theorem within_024_056_agreement : RowAgreement 3132 3134 := RowAgreement.join atomic_3133_agreement atomic_3134_agreement
private def within_024_057 : CountSegment 1 10 13120 3134 3136 1517 1518 := CountSegment.join atomic_3135 atomic_3136
private theorem within_024_057_agreement : RowAgreement 3134 3136 := RowAgreement.join atomic_3135_agreement atomic_3136_agreement
private def within_024_058 : CountSegment 1 10 13120 3132 3136 1516 1518 := CountSegment.join within_024_056 within_024_057
private theorem within_024_058_agreement : RowAgreement 3132 3136 := RowAgreement.join within_024_056_agreement within_024_057_agreement
private def within_024_059 : CountSegment 1 10 13120 3128 3136 1516 1518 := CountSegment.join within_024_055 within_024_058
private theorem within_024_059_agreement : RowAgreement 3128 3136 := RowAgreement.join within_024_055_agreement within_024_058_agreement
private def within_024_060 : CountSegment 1 10 13120 3120 3136 1514 1518 := CountSegment.join within_024_052 within_024_059
private theorem within_024_060_agreement : RowAgreement 3120 3136 := RowAgreement.join within_024_052_agreement within_024_059_agreement
private def within_024_061 : CountSegment 1 10 13120 3104 3136 1504 1518 := CountSegment.join within_024_045 within_024_060
private theorem within_024_061_agreement : RowAgreement 3104 3136 := RowAgreement.join within_024_045_agreement within_024_060_agreement
private def within_024_062 : CountSegment 1 10 13120 3072 3136 1493 1518 := CountSegment.join within_024_030 within_024_061
private theorem within_024_062_agreement : RowAgreement 3072 3136 := RowAgreement.join within_024_030_agreement within_024_061_agreement
private def within_024_063 : CountSegment 1 10 13120 3136 3138 1518 1518 := CountSegment.join atomic_3137 atomic_3138
private theorem within_024_063_agreement : RowAgreement 3136 3138 := RowAgreement.join atomic_3137_agreement atomic_3138_agreement
private def within_024_064 : CountSegment 1 10 13120 3138 3140 1518 1518 := CountSegment.join atomic_3139 atomic_3140
private theorem within_024_064_agreement : RowAgreement 3138 3140 := RowAgreement.join atomic_3139_agreement atomic_3140_agreement
private def within_024_065 : CountSegment 1 10 13120 3136 3140 1518 1518 := CountSegment.join within_024_063 within_024_064
private theorem within_024_065_agreement : RowAgreement 3136 3140 := RowAgreement.join within_024_063_agreement within_024_064_agreement
private def within_024_066 : CountSegment 1 10 13120 3140 3142 1518 1518 := CountSegment.join atomic_3141 atomic_3142
private theorem within_024_066_agreement : RowAgreement 3140 3142 := RowAgreement.join atomic_3141_agreement atomic_3142_agreement
private def within_024_067 : CountSegment 1 10 13120 3142 3144 1518 1519 := CountSegment.join atomic_3143 atomic_3144
private theorem within_024_067_agreement : RowAgreement 3142 3144 := RowAgreement.join atomic_3143_agreement atomic_3144_agreement
private def within_024_068 : CountSegment 1 10 13120 3140 3144 1518 1519 := CountSegment.join within_024_066 within_024_067
private theorem within_024_068_agreement : RowAgreement 3140 3144 := RowAgreement.join within_024_066_agreement within_024_067_agreement
private def within_024_069 : CountSegment 1 10 13120 3136 3144 1518 1519 := CountSegment.join within_024_065 within_024_068
private theorem within_024_069_agreement : RowAgreement 3136 3144 := RowAgreement.join within_024_065_agreement within_024_068_agreement
private def within_024_070 : CountSegment 1 10 13120 3144 3146 1519 1519 := CountSegment.join atomic_3145 atomic_3146
private theorem within_024_070_agreement : RowAgreement 3144 3146 := RowAgreement.join atomic_3145_agreement atomic_3146_agreement
private def within_024_071 : CountSegment 1 10 13120 3146 3148 1519 1519 := CountSegment.join atomic_3147 atomic_3148
private theorem within_024_071_agreement : RowAgreement 3146 3148 := RowAgreement.join atomic_3147_agreement atomic_3148_agreement
private def within_024_072 : CountSegment 1 10 13120 3144 3148 1519 1519 := CountSegment.join within_024_070 within_024_071
private theorem within_024_072_agreement : RowAgreement 3144 3148 := RowAgreement.join within_024_070_agreement within_024_071_agreement
private def within_024_073 : CountSegment 1 10 13120 3148 3150 1519 1519 := CountSegment.join atomic_3149 atomic_3150
private theorem within_024_073_agreement : RowAgreement 3148 3150 := RowAgreement.join atomic_3149_agreement atomic_3150_agreement
private def within_024_074 : CountSegment 1 10 13120 3150 3152 1519 1520 := CountSegment.join atomic_3151 atomic_3152
private theorem within_024_074_agreement : RowAgreement 3150 3152 := RowAgreement.join atomic_3151_agreement atomic_3152_agreement
private def within_024_075 : CountSegment 1 10 13120 3148 3152 1519 1520 := CountSegment.join within_024_073 within_024_074
private theorem within_024_075_agreement : RowAgreement 3148 3152 := RowAgreement.join within_024_073_agreement within_024_074_agreement
private def within_024_076 : CountSegment 1 10 13120 3144 3152 1519 1520 := CountSegment.join within_024_072 within_024_075
private theorem within_024_076_agreement : RowAgreement 3144 3152 := RowAgreement.join within_024_072_agreement within_024_075_agreement
private def within_024_077 : CountSegment 1 10 13120 3136 3152 1518 1520 := CountSegment.join within_024_069 within_024_076
private theorem within_024_077_agreement : RowAgreement 3136 3152 := RowAgreement.join within_024_069_agreement within_024_076_agreement
private def within_024_078 : CountSegment 1 10 13120 3152 3154 1520 1523 := CountSegment.join atomic_3153 atomic_3154
private theorem within_024_078_agreement : RowAgreement 3152 3154 := RowAgreement.join atomic_3153_agreement atomic_3154_agreement
private def within_024_079 : CountSegment 1 10 13120 3154 3156 1523 1523 := CountSegment.join atomic_3155 atomic_3156
private theorem within_024_079_agreement : RowAgreement 3154 3156 := RowAgreement.join atomic_3155_agreement atomic_3156_agreement
private def within_024_080 : CountSegment 1 10 13120 3152 3156 1520 1523 := CountSegment.join within_024_078 within_024_079
private theorem within_024_080_agreement : RowAgreement 3152 3156 := RowAgreement.join within_024_078_agreement within_024_079_agreement
private def within_024_081 : CountSegment 1 10 13120 3156 3158 1523 1524 := CountSegment.join atomic_3157 atomic_3158
private theorem within_024_081_agreement : RowAgreement 3156 3158 := RowAgreement.join atomic_3157_agreement atomic_3158_agreement
private def within_024_082 : CountSegment 1 10 13120 3158 3160 1524 1524 := CountSegment.join atomic_3159 atomic_3160
private theorem within_024_082_agreement : RowAgreement 3158 3160 := RowAgreement.join atomic_3159_agreement atomic_3160_agreement
private def within_024_083 : CountSegment 1 10 13120 3156 3160 1523 1524 := CountSegment.join within_024_081 within_024_082
private theorem within_024_083_agreement : RowAgreement 3156 3160 := RowAgreement.join within_024_081_agreement within_024_082_agreement
private def within_024_084 : CountSegment 1 10 13120 3152 3160 1520 1524 := CountSegment.join within_024_080 within_024_083
private theorem within_024_084_agreement : RowAgreement 3152 3160 := RowAgreement.join within_024_080_agreement within_024_083_agreement
private def within_024_085 : CountSegment 1 10 13120 3160 3162 1524 1525 := CountSegment.join atomic_3161 atomic_3162
private theorem within_024_085_agreement : RowAgreement 3160 3162 := RowAgreement.join atomic_3161_agreement atomic_3162_agreement
private def within_024_086 : CountSegment 1 10 13120 3162 3164 1525 1526 := CountSegment.join atomic_3163 atomic_3164
private theorem within_024_086_agreement : RowAgreement 3162 3164 := RowAgreement.join atomic_3163_agreement atomic_3164_agreement
private def within_024_087 : CountSegment 1 10 13120 3160 3164 1524 1526 := CountSegment.join within_024_085 within_024_086
private theorem within_024_087_agreement : RowAgreement 3160 3164 := RowAgreement.join within_024_085_agreement within_024_086_agreement
private def within_024_088 : CountSegment 1 10 13120 3164 3166 1526 1526 := CountSegment.join atomic_3165 atomic_3166
private theorem within_024_088_agreement : RowAgreement 3164 3166 := RowAgreement.join atomic_3165_agreement atomic_3166_agreement
private def within_024_089 : CountSegment 1 10 13120 3166 3168 1526 1526 := CountSegment.join atomic_3167 atomic_3168
private theorem within_024_089_agreement : RowAgreement 3166 3168 := RowAgreement.join atomic_3167_agreement atomic_3168_agreement
private def within_024_090 : CountSegment 1 10 13120 3164 3168 1526 1526 := CountSegment.join within_024_088 within_024_089
private theorem within_024_090_agreement : RowAgreement 3164 3168 := RowAgreement.join within_024_088_agreement within_024_089_agreement
private def within_024_091 : CountSegment 1 10 13120 3160 3168 1524 1526 := CountSegment.join within_024_087 within_024_090
private theorem within_024_091_agreement : RowAgreement 3160 3168 := RowAgreement.join within_024_087_agreement within_024_090_agreement
private def within_024_092 : CountSegment 1 10 13120 3152 3168 1520 1526 := CountSegment.join within_024_084 within_024_091
private theorem within_024_092_agreement : RowAgreement 3152 3168 := RowAgreement.join within_024_084_agreement within_024_091_agreement
private def within_024_093 : CountSegment 1 10 13120 3136 3168 1518 1526 := CountSegment.join within_024_077 within_024_092
private theorem within_024_093_agreement : RowAgreement 3136 3168 := RowAgreement.join within_024_077_agreement within_024_092_agreement
private def within_024_094 : CountSegment 1 10 13120 3168 3170 1526 1527 := CountSegment.join atomic_3169 atomic_3170
private theorem within_024_094_agreement : RowAgreement 3168 3170 := RowAgreement.join atomic_3169_agreement atomic_3170_agreement
private def within_024_095 : CountSegment 1 10 13120 3170 3172 1527 1528 := CountSegment.join atomic_3171 atomic_3172
private theorem within_024_095_agreement : RowAgreement 3170 3172 := RowAgreement.join atomic_3171_agreement atomic_3172_agreement
private def within_024_096 : CountSegment 1 10 13120 3168 3172 1526 1528 := CountSegment.join within_024_094 within_024_095
private theorem within_024_096_agreement : RowAgreement 3168 3172 := RowAgreement.join within_024_094_agreement within_024_095_agreement
private def within_024_097 : CountSegment 1 10 13120 3172 3174 1528 1529 := CountSegment.join atomic_3173 atomic_3174
private theorem within_024_097_agreement : RowAgreement 3172 3174 := RowAgreement.join atomic_3173_agreement atomic_3174_agreement
private def within_024_098 : CountSegment 1 10 13120 3174 3176 1529 1531 := CountSegment.join atomic_3175 atomic_3176
private theorem within_024_098_agreement : RowAgreement 3174 3176 := RowAgreement.join atomic_3175_agreement atomic_3176_agreement
private def within_024_099 : CountSegment 1 10 13120 3172 3176 1528 1531 := CountSegment.join within_024_097 within_024_098
private theorem within_024_099_agreement : RowAgreement 3172 3176 := RowAgreement.join within_024_097_agreement within_024_098_agreement
private def within_024_100 : CountSegment 1 10 13120 3168 3176 1526 1531 := CountSegment.join within_024_096 within_024_099
private theorem within_024_100_agreement : RowAgreement 3168 3176 := RowAgreement.join within_024_096_agreement within_024_099_agreement
private def within_024_101 : CountSegment 1 10 13120 3176 3178 1531 1532 := CountSegment.join atomic_3177 atomic_3178
private theorem within_024_101_agreement : RowAgreement 3176 3178 := RowAgreement.join atomic_3177_agreement atomic_3178_agreement
private def within_024_102 : CountSegment 1 10 13120 3178 3180 1532 1533 := CountSegment.join atomic_3179 atomic_3180
private theorem within_024_102_agreement : RowAgreement 3178 3180 := RowAgreement.join atomic_3179_agreement atomic_3180_agreement
private def within_024_103 : CountSegment 1 10 13120 3176 3180 1531 1533 := CountSegment.join within_024_101 within_024_102
private theorem within_024_103_agreement : RowAgreement 3176 3180 := RowAgreement.join within_024_101_agreement within_024_102_agreement
private def within_024_104 : CountSegment 1 10 13120 3180 3182 1533 1534 := CountSegment.join atomic_3181 atomic_3182
private theorem within_024_104_agreement : RowAgreement 3180 3182 := RowAgreement.join atomic_3181_agreement atomic_3182_agreement
private def within_024_105 : CountSegment 1 10 13120 3182 3184 1534 1536 := CountSegment.join atomic_3183 atomic_3184
private theorem within_024_105_agreement : RowAgreement 3182 3184 := RowAgreement.join atomic_3183_agreement atomic_3184_agreement
private def within_024_106 : CountSegment 1 10 13120 3180 3184 1533 1536 := CountSegment.join within_024_104 within_024_105
private theorem within_024_106_agreement : RowAgreement 3180 3184 := RowAgreement.join within_024_104_agreement within_024_105_agreement
private def within_024_107 : CountSegment 1 10 13120 3176 3184 1531 1536 := CountSegment.join within_024_103 within_024_106
private theorem within_024_107_agreement : RowAgreement 3176 3184 := RowAgreement.join within_024_103_agreement within_024_106_agreement
private def within_024_108 : CountSegment 1 10 13120 3168 3184 1526 1536 := CountSegment.join within_024_100 within_024_107
private theorem within_024_108_agreement : RowAgreement 3168 3184 := RowAgreement.join within_024_100_agreement within_024_107_agreement
private def within_024_109 : CountSegment 1 10 13120 3184 3186 1536 1537 := CountSegment.join atomic_3185 atomic_3186
private theorem within_024_109_agreement : RowAgreement 3184 3186 := RowAgreement.join atomic_3185_agreement atomic_3186_agreement
private def within_024_110 : CountSegment 1 10 13120 3186 3188 1537 1537 := CountSegment.join atomic_3187 atomic_3188
private theorem within_024_110_agreement : RowAgreement 3186 3188 := RowAgreement.join atomic_3187_agreement atomic_3188_agreement
private def within_024_111 : CountSegment 1 10 13120 3184 3188 1536 1537 := CountSegment.join within_024_109 within_024_110
private theorem within_024_111_agreement : RowAgreement 3184 3188 := RowAgreement.join within_024_109_agreement within_024_110_agreement
private def within_024_112 : CountSegment 1 10 13120 3188 3190 1537 1539 := CountSegment.join atomic_3189 atomic_3190
private theorem within_024_112_agreement : RowAgreement 3188 3190 := RowAgreement.join atomic_3189_agreement atomic_3190_agreement
private def within_024_113 : CountSegment 1 10 13120 3190 3192 1539 1539 := CountSegment.join atomic_3191 atomic_3192
private theorem within_024_113_agreement : RowAgreement 3190 3192 := RowAgreement.join atomic_3191_agreement atomic_3192_agreement
private def within_024_114 : CountSegment 1 10 13120 3188 3192 1537 1539 := CountSegment.join within_024_112 within_024_113
private theorem within_024_114_agreement : RowAgreement 3188 3192 := RowAgreement.join within_024_112_agreement within_024_113_agreement
private def within_024_115 : CountSegment 1 10 13120 3184 3192 1536 1539 := CountSegment.join within_024_111 within_024_114
private theorem within_024_115_agreement : RowAgreement 3184 3192 := RowAgreement.join within_024_111_agreement within_024_114_agreement
private def within_024_116 : CountSegment 1 10 13120 3192 3194 1539 1539 := CountSegment.join atomic_3193 atomic_3194
private theorem within_024_116_agreement : RowAgreement 3192 3194 := RowAgreement.join atomic_3193_agreement atomic_3194_agreement
private def within_024_117 : CountSegment 1 10 13120 3194 3196 1539 1540 := CountSegment.join atomic_3195 atomic_3196
private theorem within_024_117_agreement : RowAgreement 3194 3196 := RowAgreement.join atomic_3195_agreement atomic_3196_agreement
private def within_024_118 : CountSegment 1 10 13120 3192 3196 1539 1540 := CountSegment.join within_024_116 within_024_117
private theorem within_024_118_agreement : RowAgreement 3192 3196 := RowAgreement.join within_024_116_agreement within_024_117_agreement
private def within_024_119 : CountSegment 1 10 13120 3196 3198 1540 1541 := CountSegment.join atomic_3197 atomic_3198
private theorem within_024_119_agreement : RowAgreement 3196 3198 := RowAgreement.join atomic_3197_agreement atomic_3198_agreement
private def within_024_120 : CountSegment 1 10 13120 3198 3200 1541 1542 := CountSegment.join atomic_3199 atomic_3200
private theorem within_024_120_agreement : RowAgreement 3198 3200 := RowAgreement.join atomic_3199_agreement atomic_3200_agreement
private def within_024_121 : CountSegment 1 10 13120 3196 3200 1540 1542 := CountSegment.join within_024_119 within_024_120
private theorem within_024_121_agreement : RowAgreement 3196 3200 := RowAgreement.join within_024_119_agreement within_024_120_agreement
private def within_024_122 : CountSegment 1 10 13120 3192 3200 1539 1542 := CountSegment.join within_024_118 within_024_121
private theorem within_024_122_agreement : RowAgreement 3192 3200 := RowAgreement.join within_024_118_agreement within_024_121_agreement
private def within_024_123 : CountSegment 1 10 13120 3184 3200 1536 1542 := CountSegment.join within_024_115 within_024_122
private theorem within_024_123_agreement : RowAgreement 3184 3200 := RowAgreement.join within_024_115_agreement within_024_122_agreement
private def within_024_124 : CountSegment 1 10 13120 3168 3200 1526 1542 := CountSegment.join within_024_108 within_024_123
private theorem within_024_124_agreement : RowAgreement 3168 3200 := RowAgreement.join within_024_108_agreement within_024_123_agreement
private def within_024_125 : CountSegment 1 10 13120 3136 3200 1518 1542 := CountSegment.join within_024_093 within_024_124
private theorem within_024_125_agreement : RowAgreement 3136 3200 := RowAgreement.join within_024_093_agreement within_024_124_agreement
private def within_024_126 : CountSegment 1 10 13120 3072 3200 1493 1542 := CountSegment.join within_024_062 within_024_125
private theorem within_024_126_agreement : RowAgreement 3072 3200 := RowAgreement.join within_024_062_agreement within_024_125_agreement
def segment_024 : CountSegment 1 10 13120 3072 3200 1493 1542 := within_024_126
theorem agreement_024 : RowAgreement 3072 3200 := within_024_126_agreement
end ProjectEulerSolutions.P58Numerical
