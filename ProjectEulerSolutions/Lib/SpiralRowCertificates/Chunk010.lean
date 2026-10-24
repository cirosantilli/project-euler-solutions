import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1281 : RowCertificate 1281 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1281 1) := by change Nat.Prime 6566407; exact ProjectEulerSolutions.P58PrimeNodes.p_6566407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1281 2) := by change ¬ Nat.Prime 6563845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1312769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1281 3) := by change Nat.Prime 6561283; exact ProjectEulerSolutions.P58PrimeNodes.p_6561283.prime
    exact decide_eq_true prime
theorem row_count_1281 : row 1281 = 2 := (row_1281.checked).trans rfl
theorem model_count_1281 : modelRow 1281 = 2 := by decide
theorem modelrow_agrees_1281 : modelRow 1281 = row 1281 := model_count_1281.trans row_count_1281.symm
def row_1282 : RowCertificate 1282 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1282 1) := by change ¬ Nat.Prime 6576661; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 939523) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1282 2) := by change ¬ Nat.Prime 6574097; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 226693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1282 3) := by change ¬ Nat.Prime 6571533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2190511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1282 : row 1282 = 0 := (row_1282.checked).trans rfl
theorem model_count_1282 : modelRow 1282 = 0 := by decide
theorem modelrow_agrees_1282 : modelRow 1282 = row 1282 := model_count_1282.trans row_count_1282.symm
def row_1283 : RowCertificate 1283 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1283 1) := by change ¬ Nat.Prime 6586923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2195641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1283 2) := by change ¬ Nat.Prime 6584357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 506489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1283 3) := by change ¬ Nat.Prime 6581791; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 18859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1283 : row 1283 = 0 := (row_1283.checked).trans rfl
theorem model_count_1283 : modelRow 1283 = 0 := by decide
theorem modelrow_agrees_1283 : modelRow 1283 = row 1283 := model_count_1283.trans row_count_1283.symm
def row_1284 : RowCertificate 1284 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1284 1) := by change Nat.Prime 6597193; exact ProjectEulerSolutions.P58PrimeNodes.p_6597193.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1284 2) := by change ¬ Nat.Prime 6594625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1318925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1284 3) := by change ¬ Nat.Prime 6592057; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 212647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1284 : row 1284 = 1 := (row_1284.checked).trans rfl
theorem model_count_1284 : modelRow 1284 = 1 := by decide
theorem modelrow_agrees_1284 : modelRow 1284 = row 1284 := model_count_1284.trans row_count_1284.symm
def row_1285 : RowCertificate 1285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1285 1) := by change ¬ Nat.Prime 6607471; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 508267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1285 2) := by change Nat.Prime 6604901; exact ProjectEulerSolutions.P58PrimeNodes.p_6604901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1285 3) := by change ¬ Nat.Prime 6602331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2200777) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1285 : row 1285 = 1 := (row_1285.checked).trans rfl
theorem model_count_1285 : modelRow 1285 = 1 := by decide
theorem modelrow_agrees_1285 : modelRow 1285 = row 1285 := model_count_1285.trans row_count_1285.symm
def row_1286 : RowCertificate 1286 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1286 1) := by change ¬ Nat.Prime 6617757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2205919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1286 2) := by change ¬ Nat.Prime 6615185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1323037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1286 3) := by change ¬ Nat.Prime 6612613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 944659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1286 : row 1286 = 0 := (row_1286.checked).trans rfl
theorem model_count_1286 : modelRow 1286 = 0 := by decide
theorem modelrow_agrees_1286 : modelRow 1286 = row 1286 := model_count_1286.trans row_count_1286.symm
def row_1287 : RowCertificate 1287 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1287 1) := by change Nat.Prime 6628051; exact ProjectEulerSolutions.P58PrimeNodes.p_6628051.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1287 2) := by change ¬ Nat.Prime 6625477; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 161597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1287 3) := by change ¬ Nat.Prime 6622903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 946129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1287 : row 1287 = 1 := (row_1287.checked).trans rfl
theorem model_count_1287 : modelRow 1287 = 1 := by decide
theorem modelrow_agrees_1287 : modelRow 1287 = row 1287 := model_count_1287.trans row_count_1287.symm
def row_1288 : RowCertificate 1288 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1288 1) := by change ¬ Nat.Prime 6638353; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 349387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1288 2) := by change Nat.Prime 6635777; exact ProjectEulerSolutions.P58PrimeNodes.p_6635777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1288 3) := by change ¬ Nat.Prime 6633201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2211067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1288 : row 1288 = 1 := (row_1288.checked).trans rfl
theorem model_count_1288 : modelRow 1288 = 1 := by decide
theorem modelrow_agrees_1288 : modelRow 1288 = row 1288 := model_count_1288.trans row_count_1288.symm
def row_1289 : RowCertificate 1289 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1289 1) := by change ¬ Nat.Prime 6648663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2216221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1289 2) := by change ¬ Nat.Prime 6646085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1329217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1289 3) := by change ¬ Nat.Prime 6643507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 511039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1289 : row 1289 = 0 := (row_1289.checked).trans rfl
theorem model_count_1289 : modelRow 1289 = 0 := by decide
theorem modelrow_agrees_1289 : modelRow 1289 = row 1289 := model_count_1289.trans row_count_1289.symm
def row_1290 : RowCertificate 1290 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1290 1) := by change ¬ Nat.Prime 6658981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 951283) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1290 2) := by change ¬ Nat.Prime 6656401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 391553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1290 3) := by change ¬ Nat.Prime 6653821; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 179833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1290 : row 1290 = 0 := (row_1290.checked).trans rfl
theorem model_count_1290 : modelRow 1290 = 0 := by decide
theorem modelrow_agrees_1290 : modelRow 1290 = row 1290 := model_count_1290.trans row_count_1290.symm
def row_1291 : RowCertificate 1291 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1291 1) := by change ¬ Nat.Prime 6669307; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 36847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1291 2) := by change ¬ Nat.Prime 6666725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1333345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1291 3) := by change ¬ Nat.Prime 6664143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2221381) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1291 : row 1291 = 0 := (row_1291.checked).trans rfl
theorem model_count_1291 : modelRow 1291 = 0 := by decide
theorem modelrow_agrees_1291 : modelRow 1291 = row 1291 := model_count_1291.trans row_count_1291.symm
def row_1292 : RowCertificate 1292 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1292 1) := by change ¬ Nat.Prime 6679641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2226547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1292 2) := by change ¬ Nat.Prime 6677057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 180461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1292 3) := by change ¬ Nat.Prime 6674473; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 513421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1292 : row 1292 = 0 := (row_1292.checked).trans rfl
theorem model_count_1292 : modelRow 1292 = 0 := by decide
theorem modelrow_agrees_1292 : modelRow 1292 = row 1292 := model_count_1292.trans row_count_1292.symm
def row_1293 : RowCertificate 1293 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1293 1) := by change ¬ Nat.Prime 6689983; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 155581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1293 2) := by change ¬ Nat.Prime 6687397; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 26021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1293 3) := by change ¬ Nat.Prime 6684811; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 954973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1293 : row 1293 = 0 := (row_1293.checked).trans rfl
theorem model_count_1293 : modelRow 1293 = 0 := by decide
theorem modelrow_agrees_1293 : modelRow 1293 = row 1293 := model_count_1293.trans row_count_1293.symm
def row_1294 : RowCertificate 1294 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1294 1) := by change ¬ Nat.Prime 6700333; exact PrattCertificates.composite_of_product (factor := 1231) (cofactor := 5443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1294 2) := by change ¬ Nat.Prime 6697745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1339549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1294 3) := by change ¬ Nat.Prime 6695157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2231719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1294 : row 1294 = 0 := (row_1294.checked).trans rfl
theorem model_count_1294 : modelRow 1294 = 0 := by decide
theorem modelrow_agrees_1294 : modelRow 1294 = row 1294 := model_count_1294.trans row_count_1294.symm
def row_1295 : RowCertificate 1295 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1295 1) := by change ¬ Nat.Prime 6710691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2236897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1295 2) := by change ¬ Nat.Prime 6708101; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 34757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1295 3) := by change Nat.Prime 6705511; exact ProjectEulerSolutions.P58PrimeNodes.p_6705511.prime
    exact decide_eq_true prime
theorem row_count_1295 : row 1295 = 1 := (row_1295.checked).trans rfl
theorem model_count_1295 : modelRow 1295 = 1 := by decide
theorem modelrow_agrees_1295 : modelRow 1295 = row 1295 := model_count_1295.trans row_count_1295.symm
def row_1296 : RowCertificate 1296 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1296 1) := by change ¬ Nat.Prime 6721057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 960151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1296 2) := by change ¬ Nat.Prime 6718465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1343693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1296 3) := by change ¬ Nat.Prime 6715873; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 353467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1296 : row 1296 = 0 := (row_1296.checked).trans rfl
theorem model_count_1296 : modelRow 1296 = 0 := by decide
theorem modelrow_agrees_1296 : modelRow 1296 = row 1296 := model_count_1296.trans row_count_1296.symm
def row_1297 : RowCertificate 1297 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1297 1) := by change ¬ Nat.Prime 6731431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 961633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1297 2) := by change Nat.Prime 6728837; exact ProjectEulerSolutions.P58PrimeNodes.p_6728837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1297 3) := by change ¬ Nat.Prime 6726243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2242081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1297 : row 1297 = 1 := (row_1297.checked).trans rfl
theorem model_count_1297 : modelRow 1297 = 1 := by decide
theorem modelrow_agrees_1297 : modelRow 1297 = row 1297 := model_count_1297.trans row_count_1297.symm
def row_1298 : RowCertificate 1298 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1298 1) := by change ¬ Nat.Prime 6741813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2247271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1298 2) := by change ¬ Nat.Prime 6739217; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 182141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1298 3) := by change ¬ Nat.Prime 6736621; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 354559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1298 : row 1298 = 0 := (row_1298.checked).trans rfl
theorem model_count_1298 : modelRow 1298 = 0 := by decide
theorem modelrow_agrees_1298 : modelRow 1298 = row 1298 := model_count_1298.trans row_count_1298.symm
def row_1299 : RowCertificate 1299 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1299 1) := by change ¬ Nat.Prime 6752203; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 217813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1299 2) := by change ¬ Nat.Prime 6749605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1349921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1299 3) := by change Nat.Prime 6747007; exact ProjectEulerSolutions.P58PrimeNodes.p_6747007.prime
    exact decide_eq_true prime
theorem row_count_1299 : row 1299 = 1 := (row_1299.checked).trans rfl
theorem model_count_1299 : modelRow 1299 = 1 := by decide
theorem modelrow_agrees_1299 : modelRow 1299 = row 1299 := model_count_1299.trans row_count_1299.symm
def row_1300 : RowCertificate 1300 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1300 1) := by change ¬ Nat.Prime 6762601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 182773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1300 2) := by change Nat.Prime 6760001; exact ProjectEulerSolutions.P58PrimeNodes.p_6760001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1300 3) := by change ¬ Nat.Prime 6757401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2252467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1300 : row 1300 = 1 := (row_1300.checked).trans rfl
theorem model_count_1300 : modelRow 1300 = 1 := by decide
theorem modelrow_agrees_1300 : modelRow 1300 = row 1300 := model_count_1300.trans row_count_1300.symm
def row_1301 : RowCertificate 1301 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1301 1) := by change ¬ Nat.Prime 6773007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2257669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1301 2) := by change ¬ Nat.Prime 6770405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1354081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1301 3) := by change ¬ Nat.Prime 6767803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 966829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1301 : row 1301 = 0 := (row_1301.checked).trans rfl
theorem model_count_1301 : modelRow 1301 = 0 := by decide
theorem modelrow_agrees_1301 : modelRow 1301 = row 1301 := model_count_1301.trans row_count_1301.symm
def row_1302 : RowCertificate 1302 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1302 1) := by change Nat.Prime 6783421; exact ProjectEulerSolutions.P58PrimeNodes.p_6783421.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1302 2) := by change Nat.Prime 6780817; exact ProjectEulerSolutions.P58PrimeNodes.p_6780817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1302 3) := by change ¬ Nat.Prime 6778213; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 521401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1302 : row 1302 = 2 := (row_1302.checked).trans rfl
theorem model_count_1302 : modelRow 1302 = 2 := by decide
theorem modelrow_agrees_1302 : modelRow 1302 = row 1302 := model_count_1302.trans row_count_1302.symm
def row_1303 : RowCertificate 1303 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1303 1) := by change ¬ Nat.Prime 6793843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 970549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1303 2) := by change ¬ Nat.Prime 6791237; exact PrattCertificates.composite_of_product (factor := 797) (cofactor := 8521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1303 3) := by change ¬ Nat.Prime 6788631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2262877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1303 : row 1303 = 0 := (row_1303.checked).trans rfl
theorem model_count_1303 : modelRow 1303 = 0 := by decide
theorem modelrow_agrees_1303 : modelRow 1303 = row 1303 := model_count_1303.trans row_count_1303.symm
def row_1304 : RowCertificate 1304 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1304 1) := by change ¬ Nat.Prime 6804273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2268091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1304 2) := by change ¬ Nat.Prime 6801665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1360333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1304 3) := by change Nat.Prime 6799057; exact ProjectEulerSolutions.P58PrimeNodes.p_6799057.prime
    exact decide_eq_true prime
theorem row_count_1304 : row 1304 = 1 := (row_1304.checked).trans rfl
theorem model_count_1304 : modelRow 1304 = 1 := by decide
theorem modelrow_agrees_1304 : modelRow 1304 = row 1304 := model_count_1304.trans row_count_1304.symm
def row_1305 : RowCertificate 1305 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1305 1) := by change ¬ Nat.Prime 6814711; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 358669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1305 2) := by change ¬ Nat.Prime 6812101; exact PrattCertificates.composite_of_product (factor := 1873) (cofactor := 3637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1305 3) := by change ¬ Nat.Prime 6809491; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 523807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1305 : row 1305 = 0 := (row_1305.checked).trans rfl
theorem model_count_1305 : modelRow 1305 = 0 := by decide
theorem modelrow_agrees_1305 : modelRow 1305 = row 1305 := model_count_1305.trans row_count_1305.symm
def row_1306 : RowCertificate 1306 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1306 1) := by change ¬ Nat.Prime 6825157; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 8233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1306 2) := by change ¬ Nat.Prime 6822545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1364509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1306 3) := by change ¬ Nat.Prime 6819933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2273311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1306 : row 1306 = 0 := (row_1306.checked).trans rfl
theorem model_count_1306 : modelRow 1306 = 0 := by decide
theorem modelrow_agrees_1306 : modelRow 1306 = row 1306 := model_count_1306.trans row_count_1306.symm
def row_1307 : RowCertificate 1307 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1307 1) := by change ¬ Nat.Prime 6835611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2278537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1307 2) := by change ¬ Nat.Prime 6832997; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 401941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1307 3) := by change ¬ Nat.Prime 6830383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 975769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1307 : row 1307 = 0 := (row_1307.checked).trans rfl
theorem model_count_1307 : modelRow 1307 = 0 := by decide
theorem modelrow_agrees_1307 : modelRow 1307 = row 1307 := model_count_1307.trans row_count_1307.symm
def row_1308 : RowCertificate 1308 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1308 1) := by change ¬ Nat.Prime 6846073; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 526621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1308 2) := by change ¬ Nat.Prime 6843457; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 67757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1308 3) := by change ¬ Nat.Prime 6840841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 977263) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1308 : row 1308 = 0 := (row_1308.checked).trans rfl
theorem model_count_1308 : modelRow 1308 = 0 := by decide
theorem modelrow_agrees_1308 : modelRow 1308 = row 1308 := model_count_1308.trans row_count_1308.symm
def row_1309 : RowCertificate 1309 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1309 1) := by change Nat.Prime 6856543; exact ProjectEulerSolutions.P58PrimeNodes.p_6856543.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1309 2) := by change ¬ Nat.Prime 6853925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1370785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1309 3) := by change ¬ Nat.Prime 6851307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2283769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1309 : row 1309 = 1 := (row_1309.checked).trans rfl
theorem model_count_1309 : modelRow 1309 = 1 := by decide
theorem modelrow_agrees_1309 : modelRow 1309 = row 1309 := model_count_1309.trans row_count_1309.symm
def row_1310 : RowCertificate 1310 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1310 1) := by change ¬ Nat.Prime 6867021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2289007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1310 2) := by change ¬ Nat.Prime 6864401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 129517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1310 3) := by change ¬ Nat.Prime 6861781; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 93997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1310 : row 1310 = 0 := (row_1310.checked).trans rfl
theorem model_count_1310 : modelRow 1310 = 0 := by decide
theorem modelrow_agrees_1310 : modelRow 1310 = row 1310 := model_count_1310.trans row_count_1310.symm
def row_1311 : RowCertificate 1311 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1311 1) := by change ¬ Nat.Prime 6877507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 982501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1311 2) := by change ¬ Nat.Prime 6874885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1374977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1311 3) := by change ¬ Nat.Prime 6872263; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 66721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1311 : row 1311 = 0 := (row_1311.checked).trans rfl
theorem model_count_1311 : modelRow 1311 = 0 := by decide
theorem modelrow_agrees_1311 : modelRow 1311 = row 1311 := model_count_1311.trans row_count_1311.symm
def row_1312 : RowCertificate 1312 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1312 1) := by change ¬ Nat.Prime 6888001; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 9397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1312 2) := by change Nat.Prime 6885377; exact ProjectEulerSolutions.P58PrimeNodes.p_6885377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1312 3) := by change ¬ Nat.Prime 6882753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2294251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1312 : row 1312 = 1 := (row_1312.checked).trans rfl
theorem model_count_1312 : modelRow 1312 = 1 := by decide
theorem modelrow_agrees_1312 : modelRow 1312 = row 1312 := model_count_1312.trans row_count_1312.symm
def row_1313 : RowCertificate 1313 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1313 1) := by change ¬ Nat.Prime 6898503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2299501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1313 2) := by change ¬ Nat.Prime 6895877; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 30113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1313 3) := by change Nat.Prime 6893251; exact ProjectEulerSolutions.P58PrimeNodes.p_6893251.prime
    exact decide_eq_true prime
theorem row_count_1313 : row 1313 = 1 := (row_1313.checked).trans rfl
theorem model_count_1313 : modelRow 1313 = 1 := by decide
theorem modelrow_agrees_1313 : modelRow 1313 = row 1313 := model_count_1313.trans row_count_1313.symm
def row_1314 : RowCertificate 1314 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1314 1) := by change Nat.Prime 6909013; exact ProjectEulerSolutions.P58PrimeNodes.p_6909013.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1314 2) := by change ¬ Nat.Prime 6906385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1381277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1314 3) := by change ¬ Nat.Prime 6903757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 986251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1314 : row 1314 = 1 := (row_1314.checked).trans rfl
theorem model_count_1314 : modelRow 1314 = 1 := by decide
theorem modelrow_agrees_1314 : modelRow 1314 = row 1314 := model_count_1314.trans row_count_1314.symm
def row_1315 : RowCertificate 1315 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1315 1) := by change ¬ Nat.Prime 6919531; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 87589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1315 2) := by change ¬ Nat.Prime 6916901; exact PrattCertificates.composite_of_product (factor := 1297) (cofactor := 5333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1315 3) := by change ¬ Nat.Prime 6914271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2304757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1315 : row 1315 = 0 := (row_1315.checked).trans rfl
theorem model_count_1315 : modelRow 1315 = 0 := by decide
theorem modelrow_agrees_1315 : modelRow 1315 = row 1315 := model_count_1315.trans row_count_1315.symm
def row_1316 : RowCertificate 1316 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1316 1) := by change ¬ Nat.Prime 6930057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2310019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1316 2) := by change ¬ Nat.Prime 6927425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1385485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1316 3) := by change ¬ Nat.Prime 6924793; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 67231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1316 : row 1316 = 0 := (row_1316.checked).trans rfl
theorem model_count_1316 : modelRow 1316 = 0 := by decide
theorem modelrow_agrees_1316 : modelRow 1316 = row 1316 := model_count_1316.trans row_count_1316.symm
def row_1317 : RowCertificate 1317 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1317 1) := by change ¬ Nat.Prime 6940591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 991513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1317 2) := by change ¬ Nat.Prime 6937957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 533689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1317 3) := by change ¬ Nat.Prime 6935323; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 365017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1317 : row 1317 = 0 := (row_1317.checked).trans rfl
theorem model_count_1317 : modelRow 1317 = 0 := by decide
theorem modelrow_agrees_1317 : modelRow 1317 = row 1317 := model_count_1317.trans row_count_1317.symm
def row_1318 : RowCertificate 1318 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1318 1) := by change ¬ Nat.Prime 6951133; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 993019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1318 2) := by change ¬ Nat.Prime 6948497; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 78073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1318 3) := by change ¬ Nat.Prime 6945861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2315287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1318 : row 1318 = 0 := (row_1318.checked).trans rfl
theorem model_count_1318 : modelRow 1318 = 0 := by decide
theorem modelrow_agrees_1318 : modelRow 1318 = row 1318 := model_count_1318.trans row_count_1318.symm
def row_1319 : RowCertificate 1319 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1319 1) := by change ¬ Nat.Prime 6961683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2320561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1319 2) := by change ¬ Nat.Prime 6959045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1391809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1319 3) := by change ¬ Nat.Prime 6956407; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 188011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1319 : row 1319 = 0 := (row_1319.checked).trans rfl
theorem model_count_1319 : modelRow 1319 = 0 := by decide
theorem modelrow_agrees_1319 : modelRow 1319 = row 1319 := model_count_1319.trans row_count_1319.symm
def row_1320 : RowCertificate 1320 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1320 1) := by change ¬ Nat.Prime 6972241; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 224911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1320 2) := by change ¬ Nat.Prime 6969601; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 50873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1320 3) := by change Nat.Prime 6966961; exact ProjectEulerSolutions.P58PrimeNodes.p_6966961.prime
    exact decide_eq_true prime
theorem row_count_1320 : row 1320 = 1 := (row_1320.checked).trans rfl
theorem model_count_1320 : modelRow 1320 = 1 := by decide
theorem modelrow_agrees_1320 : modelRow 1320 = row 1320 := model_count_1320.trans row_count_1320.symm
def row_1321 : RowCertificate 1321 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1321 1) := by change ¬ Nat.Prime 6982807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 537139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1321 2) := by change ¬ Nat.Prime 6980165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1396033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1321 3) := by change ¬ Nat.Prime 6977523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2325841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1321 : row 1321 = 0 := (row_1321.checked).trans rfl
theorem model_count_1321 : modelRow 1321 = 0 := by decide
theorem modelrow_agrees_1321 : modelRow 1321 = row 1321 := model_count_1321.trans row_count_1321.symm
def row_1322 : RowCertificate 1322 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1322 1) := by change ¬ Nat.Prime 6993381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2331127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1322 2) := by change ¬ Nat.Prime 6990737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 537749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1322 3) := by change ¬ Nat.Prime 6988093; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 998299) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1322 : row 1322 = 0 := (row_1322.checked).trans rfl
theorem model_count_1322 : modelRow 1322 = 0 := by decide
theorem modelrow_agrees_1322 : modelRow 1322 = row 1322 := model_count_1322.trans row_count_1322.symm
def row_1323 : RowCertificate 1323 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1323 1) := by change Nat.Prime 7003963; exact ProjectEulerSolutions.P58PrimeNodes.p_7003963.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1323 2) := by change Nat.Prime 7001317; exact ProjectEulerSolutions.P58PrimeNodes.p_7001317.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1323 3) := by change ¬ Nat.Prime 6998671; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 6199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1323 : row 1323 = 2 := (row_1323.checked).trans rfl
theorem model_count_1323 : modelRow 1323 = 2 := by decide
theorem modelrow_agrees_1323 : modelRow 1323 = row 1323 := model_count_1323.trans row_count_1323.symm
def row_1324 : RowCertificate 1324 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1324 1) := by change ¬ Nat.Prime 7014553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1002079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1324 2) := by change ¬ Nat.Prime 7011905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1402381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1324 3) := by change ¬ Nat.Prime 7009257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2336419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1324 : row 1324 = 0 := (row_1324.checked).trans rfl
theorem model_count_1324 : modelRow 1324 = 0 := by decide
theorem modelrow_agrees_1324 : modelRow 1324 = row 1324 := model_count_1324.trans row_count_1324.symm
def row_1325 : RowCertificate 1325 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1325 1) := by change ¬ Nat.Prime 7025151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2341717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1325 2) := by change ¬ Nat.Prime 7022501; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 22153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1325 3) := by change Nat.Prime 7019851; exact ProjectEulerSolutions.P58PrimeNodes.p_7019851.prime
    exact decide_eq_true prime
theorem row_count_1325 : row 1325 = 1 := (row_1325.checked).trans rfl
theorem model_count_1325 : modelRow 1325 = 1 := by decide
theorem modelrow_agrees_1325 : modelRow 1325 = row 1325 := model_count_1325.trans row_count_1325.symm
def row_1326 : RowCertificate 1326 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1326 1) := by change ¬ Nat.Prime 7035757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 370303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1326 2) := by change ¬ Nat.Prime 7033105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1406621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1326 3) := by change Nat.Prime 7030453; exact ProjectEulerSolutions.P58PrimeNodes.p_7030453.prime
    exact decide_eq_true prime
theorem row_count_1326 : row 1326 = 1 := (row_1326.checked).trans rfl
theorem model_count_1326 : modelRow 1326 = 1 := by decide
theorem modelrow_agrees_1326 : modelRow 1326 = row 1326 := model_count_1326.trans row_count_1326.symm
def row_1327 : RowCertificate 1327 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1327 1) := by change ¬ Nat.Prime 7046371; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 72643) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1327 2) := by change Nat.Prime 7043717; exact ProjectEulerSolutions.P58PrimeNodes.p_7043717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1327 3) := by change ¬ Nat.Prime 7041063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2347021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1327 : row 1327 = 1 := (row_1327.checked).trans rfl
theorem model_count_1327 : modelRow 1327 = 1 := by decide
theorem modelrow_agrees_1327 : modelRow 1327 = row 1327 := model_count_1327.trans row_count_1327.symm
def row_1328 : RowCertificate 1328 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1328 1) := by change ¬ Nat.Prime 7056993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2352331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1328 2) := by change ¬ Nat.Prime 7054337; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 414961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1328 3) := by change ¬ Nat.Prime 7051681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1007383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1328 : row 1328 = 0 := (row_1328.checked).trans rfl
theorem model_count_1328 : modelRow 1328 = 0 := by decide
theorem modelrow_agrees_1328 : modelRow 1328 = row 1328 := model_count_1328.trans row_count_1328.symm
def row_1329 : RowCertificate 1329 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1329 1) := by change Nat.Prime 7067623; exact ProjectEulerSolutions.P58PrimeNodes.p_7067623.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1329 2) := by change ¬ Nat.Prime 7064965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1412993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1329 3) := by change ¬ Nat.Prime 7062307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1008901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1329 : row 1329 = 1 := (row_1329.checked).trans rfl
theorem model_count_1329 : modelRow 1329 = 1 := by decide
theorem modelrow_agrees_1329 : modelRow 1329 = row 1329 := model_count_1329.trans row_count_1329.symm
def row_1330 : RowCertificate 1330 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1330 1) := by change ¬ Nat.Prime 7078261; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 228331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1330 2) := by change ¬ Nat.Prime 7075601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 544277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1330 3) := by change ¬ Nat.Prime 7072941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2357647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1330 : row 1330 = 0 := (row_1330.checked).trans rfl
theorem model_count_1330 : modelRow 1330 = 0 := by decide
theorem modelrow_agrees_1330 : modelRow 1330 = row 1330 := model_count_1330.trans row_count_1330.symm
def row_1331 : RowCertificate 1331 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1331 1) := by change ¬ Nat.Prime 7088907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2362969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1331 2) := by change ¬ Nat.Prime 7086245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1417249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1331 3) := by change ¬ Nat.Prime 7083583; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 544891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1331 : row 1331 = 0 := (row_1331.checked).trans rfl
theorem model_count_1331 : modelRow 1331 = 0 := by decide
theorem modelrow_agrees_1331 : modelRow 1331 = row 1331 := model_count_1331.trans row_count_1331.symm
def row_1332 : RowCertificate 1332 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1332 1) := by change ¬ Nat.Prime 7099561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1014223) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1332 2) := by change Nat.Prime 7096897; exact ProjectEulerSolutions.P58PrimeNodes.p_7096897.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1332 3) := by change Nat.Prime 7094233; exact ProjectEulerSolutions.P58PrimeNodes.p_7094233.prime
    exact decide_eq_true prime
theorem row_count_1332 : row 1332 = 2 := (row_1332.checked).trans rfl
theorem model_count_1332 : modelRow 1332 = 2 := by decide
theorem modelrow_agrees_1332 : modelRow 1332 = row 1332 := model_count_1332.trans row_count_1332.symm
def row_1333 : RowCertificate 1333 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1333 1) := by change ¬ Nat.Prime 7110223; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 43621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1333 2) := by change Nat.Prime 7107557; exact ProjectEulerSolutions.P58PrimeNodes.p_7107557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1333 3) := by change ¬ Nat.Prime 7104891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2368297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1333 : row 1333 = 1 := (row_1333.checked).trans rfl
theorem model_count_1333 : modelRow 1333 = 1 := by decide
theorem modelrow_agrees_1333 : modelRow 1333 = row 1333 := model_count_1333.trans row_count_1333.symm
def row_1334 : RowCertificate 1334 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1334 1) := by change ¬ Nat.Prime 7120893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2373631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1334 2) := by change ¬ Nat.Prime 7118225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1423645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1334 3) := by change ¬ Nat.Prime 7115557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 374503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1334 : row 1334 = 0 := (row_1334.checked).trans rfl
theorem model_count_1334 : modelRow 1334 = 0 := by decide
theorem modelrow_agrees_1334 : modelRow 1334 = row 1334 := model_count_1334.trans row_count_1334.symm
def row_1335 : RowCertificate 1335 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1335 1) := by change ¬ Nat.Prime 7131571; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 116911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1335 2) := by change ¬ Nat.Prime 7128901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 548377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1335 3) := by change ¬ Nat.Prime 7126231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1018033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1335 : row 1335 = 0 := (row_1335.checked).trans rfl
theorem model_count_1335 : modelRow 1335 = 0 := by decide
theorem modelrow_agrees_1335 : modelRow 1335 = row 1335 := model_count_1335.trans row_count_1335.symm
def row_1336 : RowCertificate 1336 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1336 1) := by change ¬ Nat.Prime 7142257; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 166099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1336 2) := by change ¬ Nat.Prime 7139585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1427917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1336 3) := by change ¬ Nat.Prime 7136913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2378971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1336 : row 1336 = 0 := (row_1336.checked).trans rfl
theorem model_count_1336 : modelRow 1336 = 0 := by decide
theorem modelrow_agrees_1336 : modelRow 1336 = row 1336 := model_count_1336.trans row_count_1336.symm
def row_1337 : RowCertificate 1337 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1337 1) := by change ¬ Nat.Prime 7152951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2384317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1337 2) := by change ¬ Nat.Prime 7150277; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 174397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1337 3) := by change Nat.Prime 7147603; exact ProjectEulerSolutions.P58PrimeNodes.p_7147603.prime
    exact decide_eq_true prime
theorem row_count_1337 : row 1337 = 1 := (row_1337.checked).trans rfl
theorem model_count_1337 : modelRow 1337 = 1 := by decide
theorem modelrow_agrees_1337 : modelRow 1337 = row 1337 := model_count_1337.trans row_count_1337.symm
def row_1338 : RowCertificate 1338 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1338 1) := by change ¬ Nat.Prime 7163653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1023379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1338 2) := by change Nat.Prime 7160977; exact ProjectEulerSolutions.P58PrimeNodes.p_7160977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1338 3) := by change ¬ Nat.Prime 7158301; exact PrattCertificates.composite_of_product (factor := 523) (cofactor := 13687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1338 : row 1338 = 1 := (row_1338.checked).trans rfl
theorem model_count_1338 : modelRow 1338 = 1 := by decide
theorem modelrow_agrees_1338 : modelRow 1338 = row 1338 := model_count_1338.trans row_count_1338.symm
def row_1339 : RowCertificate 1339 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1339 1) := by change ¬ Nat.Prime 7174363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1024909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1339 2) := by change ¬ Nat.Prime 7171685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1434337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1339 3) := by change ¬ Nat.Prime 7169007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2389669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1339 : row 1339 = 0 := (row_1339.checked).trans rfl
theorem model_count_1339 : modelRow 1339 = 0 := by decide
theorem modelrow_agrees_1339 : modelRow 1339 = row 1339 := model_count_1339.trans row_count_1339.symm
def row_1340 : RowCertificate 1340 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1340 1) := by change ¬ Nat.Prime 7185081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2395027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1340 2) := by change ¬ Nat.Prime 7182401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 247669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1340 3) := by change ¬ Nat.Prime 7179721; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 65869) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1340 : row 1340 = 0 := (row_1340.checked).trans rfl
theorem model_count_1340 : modelRow 1340 = 0 := by decide
theorem modelrow_agrees_1340 : modelRow 1340 = row 1340 := model_count_1340.trans row_count_1340.symm
def row_1341 : RowCertificate 1341 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1341 1) := by change Nat.Prime 7195807; exact ProjectEulerSolutions.P58PrimeNodes.p_7195807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1341 2) := by change ¬ Nat.Prime 7193125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1438625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1341 3) := by change ¬ Nat.Prime 7190443; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 553111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1341 : row 1341 = 1 := (row_1341.checked).trans rfl
theorem model_count_1341 : modelRow 1341 = 1 := by decide
theorem modelrow_agrees_1341 : modelRow 1341 = row 1341 := model_count_1341.trans row_count_1341.symm
def row_1342 : RowCertificate 1342 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1342 1) := by change Nat.Prime 7206541; exact ProjectEulerSolutions.P58PrimeNodes.p_7206541.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1342 2) := by change Nat.Prime 7203857; exact ProjectEulerSolutions.P58PrimeNodes.p_7203857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1342 3) := by change ¬ Nat.Prime 7201173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2400391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1342 : row 1342 = 2 := (row_1342.checked).trans rfl
theorem model_count_1342 : modelRow 1342 = 2 := by decide
theorem modelrow_agrees_1342 : modelRow 1342 = row 1342 := model_count_1342.trans row_count_1342.symm
def row_1343 : RowCertificate 1343 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1343 1) := by change ¬ Nat.Prime 7217283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2405761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1343 2) := by change ¬ Nat.Prime 7214597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 554969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1343 3) := by change ¬ Nat.Prime 7211911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1030273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1343 : row 1343 = 0 := (row_1343.checked).trans rfl
theorem model_count_1343 : modelRow 1343 = 0 := by decide
theorem modelrow_agrees_1343 : modelRow 1343 = row 1343 := model_count_1343.trans row_count_1343.symm
def row_1344 : RowCertificate 1344 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1344 1) := by change Nat.Prime 7228033; exact ProjectEulerSolutions.P58PrimeNodes.p_7228033.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1344 2) := by change ¬ Nat.Prime 7225345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1445069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1344 3) := by change ¬ Nat.Prime 7222657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 555589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1344 : row 1344 = 1 := (row_1344.checked).trans rfl
theorem model_count_1344 : modelRow 1344 = 1 := by decide
theorem modelrow_agrees_1344 : modelRow 1344 = row 1344 := model_count_1344.trans row_count_1344.symm
def row_1345 : RowCertificate 1345 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1345 1) := by change ¬ Nat.Prime 7238791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1034113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1345 2) := by change ¬ Nat.Prime 7236101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 425653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1345 3) := by change ¬ Nat.Prime 7233411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2411137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1345 : row 1345 = 0 := (row_1345.checked).trans rfl
theorem model_count_1345 : modelRow 1345 = 0 := by decide
theorem modelrow_agrees_1345 : modelRow 1345 = row 1345 := model_count_1345.trans row_count_1345.symm
def row_1346 : RowCertificate 1346 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1346 1) := by change ¬ Nat.Prime 7249557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2416519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1346 2) := by change ¬ Nat.Prime 7246865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1449373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1346 3) := by change ¬ Nat.Prime 7244173; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 233683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1346 : row 1346 = 0 := (row_1346.checked).trans rfl
theorem model_count_1346 : modelRow 1346 = 0 := by decide
theorem modelrow_agrees_1346 : modelRow 1346 = row 1346 := model_count_1346.trans row_count_1346.symm
def row_1347 : RowCertificate 1347 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1347 1) := by change ¬ Nat.Prime 7260331; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 558487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1347 2) := by change ¬ Nat.Prime 7257637; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 74821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1347 3) := by change ¬ Nat.Prime 7254943; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 36457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1347 : row 1347 = 0 := (row_1347.checked).trans rfl
theorem model_count_1347 : modelRow 1347 = 0 := by decide
theorem modelrow_agrees_1347 : modelRow 1347 = row 1347 := model_count_1347.trans row_count_1347.symm
def row_1348 : RowCertificate 1348 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1348 1) := by change Nat.Prime 7271113; exact ProjectEulerSolutions.P58PrimeNodes.p_7271113.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1348 2) := by change ¬ Nat.Prime 7268417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 559109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1348 3) := by change ¬ Nat.Prime 7265721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2421907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1348 : row 1348 = 1 := (row_1348.checked).trans rfl
theorem model_count_1348 : modelRow 1348 = 1 := by decide
theorem modelrow_agrees_1348 : modelRow 1348 = row 1348 := model_count_1348.trans row_count_1348.symm
def row_1349 : RowCertificate 1349 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1349 1) := by change ¬ Nat.Prime 7281903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2427301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1349 2) := by change ¬ Nat.Prime 7279205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1455841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1349 3) := by change ¬ Nat.Prime 7276507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1039501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1349 : row 1349 = 0 := (row_1349.checked).trans rfl
theorem model_count_1349 : modelRow 1349 = 0 := by decide
theorem modelrow_agrees_1349 : modelRow 1349 = row 1349 := model_count_1349.trans row_count_1349.symm
def row_1350 : RowCertificate 1350 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1350 1) := by change ¬ Nat.Prime 7292701; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 560977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1350 2) := by change Nat.Prime 7290001; exact ProjectEulerSolutions.P58PrimeNodes.p_7290001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1350 3) := by change ¬ Nat.Prime 7287301; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1041043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1350 : row 1350 = 1 := (row_1350.checked).trans rfl
theorem model_count_1350 : modelRow 1350 = 1 := by decide
theorem modelrow_agrees_1350 : modelRow 1350 = row 1350 := model_count_1350.trans row_count_1350.symm
def row_1351 : RowCertificate 1351 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1351 1) := by change ¬ Nat.Prime 7303507; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 235597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1351 2) := by change ¬ Nat.Prime 7300805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1460161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1351 3) := by change ¬ Nat.Prime 7298103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2432701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1351 : row 1351 = 0 := (row_1351.checked).trans rfl
theorem model_count_1351 : modelRow 1351 = 0 := by decide
theorem modelrow_agrees_1351 : modelRow 1351 = row 1351 := model_count_1351.trans row_count_1351.symm
def row_1352 : RowCertificate 1352 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1352 1) := by change ¬ Nat.Prime 7314321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2438107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1352 2) := by change ¬ Nat.Prime 7311617; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 82153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1352 3) := by change Nat.Prime 7308913; exact ProjectEulerSolutions.P58PrimeNodes.p_7308913.prime
    exact decide_eq_true prime
theorem row_count_1352 : row 1352 = 1 := (row_1352.checked).trans rfl
theorem model_count_1352 : modelRow 1352 = 1 := by decide
theorem modelrow_agrees_1352 : modelRow 1352 = row 1352 := model_count_1352.trans row_count_1352.symm
def row_1353 : RowCertificate 1353 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1353 1) := by change ¬ Nat.Prime 7325143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1046449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1353 2) := by change Nat.Prime 7322437; exact ProjectEulerSolutions.P58PrimeNodes.p_7322437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1353 3) := by change ¬ Nat.Prime 7319731; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 385249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1353 : row 1353 = 1 := (row_1353.checked).trans rfl
theorem model_count_1353 : modelRow 1353 = 1 := by decide
theorem modelrow_agrees_1353 : modelRow 1353 = row 1353 := model_count_1353.trans row_count_1353.symm
def row_1354 : RowCertificate 1354 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1354 1) := by change Nat.Prime 7335973; exact ProjectEulerSolutions.P58PrimeNodes.p_7335973.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1354 2) := by change ¬ Nat.Prime 7333265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1466653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1354 3) := by change ¬ Nat.Prime 7330557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2443519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1354 : row 1354 = 1 := (row_1354.checked).trans rfl
theorem model_count_1354 : modelRow 1354 = 1 := by decide
theorem modelrow_agrees_1354 : modelRow 1354 = row 1354 := model_count_1354.trans row_count_1354.symm
def row_1355 : RowCertificate 1355 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1355 1) := by change ¬ Nat.Prime 7346811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2448937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1355 2) := by change ¬ Nat.Prime 7344101; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 26513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1355 3) := by change ¬ Nat.Prime 7341391; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 386389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1355 : row 1355 = 0 := (row_1355.checked).trans rfl
theorem model_count_1355 : modelRow 1355 = 0 := by decide
theorem modelrow_agrees_1355 : modelRow 1355 = row 1355 := model_count_1355.trans row_count_1355.symm
def row_1356 : RowCertificate 1356 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1356 1) := by change ¬ Nat.Prime 7357657; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 45139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1356 2) := by change ¬ Nat.Prime 7354945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1470989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1356 3) := by change ¬ Nat.Prime 7352233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1050319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1356 : row 1356 = 0 := (row_1356.checked).trans rfl
theorem model_count_1356 : modelRow 1356 = 0 := by decide
theorem modelrow_agrees_1356 : modelRow 1356 = row 1356 := model_count_1356.trans row_count_1356.symm
def row_1357 : RowCertificate 1357 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1357 1) := by change Nat.Prime 7368511; exact ProjectEulerSolutions.P58PrimeNodes.p_7368511.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1357 2) := by change ¬ Nat.Prime 7365797; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 253993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1357 3) := by change ¬ Nat.Prime 7363083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2454361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1357 : row 1357 = 1 := (row_1357.checked).trans rfl
theorem model_count_1357 : modelRow 1357 = 1 := by decide
theorem modelrow_agrees_1357 : modelRow 1357 = row 1357 := model_count_1357.trans row_count_1357.symm
def row_1358 : RowCertificate 1358 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1358 1) := by change ¬ Nat.Prime 7379373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2459791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1358 2) := by change ¬ Nat.Prime 7376657; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 433921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1358 3) := by change ¬ Nat.Prime 7373941; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 171487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1358 : row 1358 = 0 := (row_1358.checked).trans rfl
theorem model_count_1358 : modelRow 1358 = 0 := by decide
theorem modelrow_agrees_1358 : modelRow 1358 = row 1358 := model_count_1358.trans row_count_1358.symm
def row_1359 : RowCertificate 1359 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1359 1) := by change ¬ Nat.Prime 7390243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1055749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1359 2) := by change ¬ Nat.Prime 7387525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1477505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1359 3) := by change ¬ Nat.Prime 7384807; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 110221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1359 : row 1359 = 0 := (row_1359.checked).trans rfl
theorem model_count_1359 : modelRow 1359 = 0 := by decide
theorem modelrow_agrees_1359 : modelRow 1359 = row 1359 := model_count_1359.trans row_count_1359.symm
def row_1360 : RowCertificate 1360 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1360 1) := by change ¬ Nat.Prime 7401121; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1057303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1360 2) := by change ¬ Nat.Prime 7398401; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 18089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1360 3) := by change ¬ Nat.Prime 7395681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2465227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1360 : row 1360 = 0 := (row_1360.checked).trans rfl
theorem model_count_1360 : modelRow 1360 = 0 := by decide
theorem modelrow_agrees_1360 : modelRow 1360 = row 1360 := model_count_1360.trans row_count_1360.symm
def row_1361 : RowCertificate 1361 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1361 1) := by change ¬ Nat.Prime 7412007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2470669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1361 2) := by change ¬ Nat.Prime 7409285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1481857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1361 3) := by change Nat.Prime 7406563; exact ProjectEulerSolutions.P58PrimeNodes.p_7406563.prime
    exact decide_eq_true prime
theorem row_count_1361 : row 1361 = 1 := (row_1361.checked).trans rfl
theorem model_count_1361 : modelRow 1361 = 1 := by decide
theorem modelrow_agrees_1361 : modelRow 1361 = row 1361 := model_count_1361.trans row_count_1361.symm
def row_1362 : RowCertificate 1362 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1362 1) := by change ¬ Nat.Prime 7422901; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 390679) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1362 2) := by change ¬ Nat.Prime 7420177; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 436481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1362 3) := by change Nat.Prime 7417453; exact ProjectEulerSolutions.P58PrimeNodes.p_7417453.prime
    exact decide_eq_true prime
theorem row_count_1362 : row 1362 = 1 := (row_1362.checked).trans rfl
theorem model_count_1362 : modelRow 1362 = 1 := by decide
theorem modelrow_agrees_1362 : modelRow 1362 = row 1362 := model_count_1362.trans row_count_1362.symm
def row_1363 : RowCertificate 1363 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1363 1) := by change ¬ Nat.Prime 7433803; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 571831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1363 2) := by change ¬ Nat.Prime 7431077; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 140209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1363 3) := by change ¬ Nat.Prime 7428351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2476117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1363 : row 1363 = 0 := (row_1363.checked).trans rfl
theorem model_count_1363 : modelRow 1363 = 0 := by decide
theorem modelrow_agrees_1363 : modelRow 1363 = row 1363 := model_count_1363.trans row_count_1363.symm
def row_1364 : RowCertificate 1364 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1364 1) := by change ¬ Nat.Prime 7444713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2481571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1364 2) := by change ¬ Nat.Prime 7441985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1488397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1364 3) := by change ¬ Nat.Prime 7439257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1062751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1364 : row 1364 = 0 := (row_1364.checked).trans rfl
theorem model_count_1364 : modelRow 1364 = 0 := by decide
theorem modelrow_agrees_1364 : modelRow 1364 = row 1364 := model_count_1364.trans row_count_1364.symm
def row_1365 : RowCertificate 1365 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1365 1) := by change Nat.Prime 7455631; exact ProjectEulerSolutions.P58PrimeNodes.p_7455631.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1365 2) := by change Nat.Prime 7452901; exact ProjectEulerSolutions.P58PrimeNodes.p_7452901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1365 3) := by change Nat.Prime 7450171; exact ProjectEulerSolutions.P58PrimeNodes.p_7450171.prime
    exact decide_eq_true prime
theorem row_count_1365 : row 1365 = 3 := (row_1365.checked).trans rfl
theorem model_count_1365 : modelRow 1365 = 3 := by decide
theorem modelrow_agrees_1365 : modelRow 1365 = row 1365 := model_count_1365.trans row_count_1365.symm
def row_1366 : RowCertificate 1366 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1366 1) := by change ¬ Nat.Prime 7466557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1066651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1366 2) := by change ¬ Nat.Prime 7463825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1492765) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1366 3) := by change ¬ Nat.Prime 7461093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2487031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1366 : row 1366 = 0 := (row_1366.checked).trans rfl
theorem model_count_1366 : modelRow 1366 = 0 := by decide
theorem modelrow_agrees_1366 : modelRow 1366 = row 1366 := model_count_1366.trans row_count_1366.symm
def row_1367 : RowCertificate 1367 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1367 1) := by change ¬ Nat.Prime 7477491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2492497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1367 2) := by change ¬ Nat.Prime 7474757; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 122537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1367 3) := by change ¬ Nat.Prime 7472023; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 574771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1367 : row 1367 = 0 := (row_1367.checked).trans rfl
theorem model_count_1367 : modelRow 1367 = 0 := by decide
theorem modelrow_agrees_1367 : modelRow 1367 = row 1367 := model_count_1367.trans row_count_1367.symm
def row_1368 : RowCertificate 1368 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1368 1) := by change Nat.Prime 7488433; exact ProjectEulerSolutions.P58PrimeNodes.p_7488433.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1368 2) := by change Nat.Prime 7485697; exact ProjectEulerSolutions.P58PrimeNodes.p_7485697.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1368 3) := by change Nat.Prime 7482961; exact ProjectEulerSolutions.P58PrimeNodes.p_7482961.prime
    exact decide_eq_true prime
theorem row_count_1368 : row 1368 = 3 := (row_1368.checked).trans rfl
theorem model_count_1368 : modelRow 1368 = 3 := by decide
theorem modelrow_agrees_1368 : modelRow 1368 = row 1368 := model_count_1368.trans row_count_1368.symm
def row_1369 : RowCertificate 1369 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1369 1) := by change ¬ Nat.Prime 7499383; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 27673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1369 2) := by change ¬ Nat.Prime 7496645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1499329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1369 3) := by change ¬ Nat.Prime 7493907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2497969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1369 : row 1369 = 0 := (row_1369.checked).trans rfl
theorem model_count_1369 : modelRow 1369 = 0 := by decide
theorem modelrow_agrees_1369 : modelRow 1369 = row 1369 := model_count_1369.trans row_count_1369.symm
def row_1370 : RowCertificate 1370 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1370 1) := by change ¬ Nat.Prime 7510341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2503447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1370 2) := by change ¬ Nat.Prime 7507601; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 10589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1370 3) := by change ¬ Nat.Prime 7504861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1072123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1370 : row 1370 = 0 := (row_1370.checked).trans rfl
theorem model_count_1370 : modelRow 1370 = 0 := by decide
theorem modelrow_agrees_1370 : modelRow 1370 = row 1370 := model_count_1370.trans row_count_1370.symm
def row_1371 : RowCertificate 1371 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1371 1) := by change Nat.Prime 7521307; exact ProjectEulerSolutions.P58PrimeNodes.p_7521307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1371 2) := by change ¬ Nat.Prime 7518565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1503713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1371 3) := by change ¬ Nat.Prime 7515823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1073689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1371 : row 1371 = 1 := (row_1371.checked).trans rfl
theorem model_count_1371 : modelRow 1371 = 1 := by decide
theorem modelrow_agrees_1371 : modelRow 1371 = row 1371 := model_count_1371.trans row_count_1371.symm
def row_1372 : RowCertificate 1372 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1372 1) := by change ¬ Nat.Prime 7532281; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 18973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1372 2) := by change ¬ Nat.Prime 7529537; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 203501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1372 3) := by change ¬ Nat.Prime 7526793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2508931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1372 : row 1372 = 0 := (row_1372.checked).trans rfl
theorem model_count_1372 : modelRow 1372 = 0 := by decide
theorem modelrow_agrees_1372 : modelRow 1372 = row 1372 := model_count_1372.trans row_count_1372.symm
def row_1373 : RowCertificate 1373 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1373 1) := by change ¬ Nat.Prime 7543263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2514421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1373 2) := by change Nat.Prime 7540517; exact ProjectEulerSolutions.P58PrimeNodes.p_7540517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1373 3) := by change ¬ Nat.Prime 7537771; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 175297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1373 : row 1373 = 1 := (row_1373.checked).trans rfl
theorem model_count_1373 : modelRow 1373 = 1 := by decide
theorem modelrow_agrees_1373 : modelRow 1373 = row 1373 := model_count_1373.trans row_count_1373.symm
def row_1374 : RowCertificate 1374 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1374 1) := by change ¬ Nat.Prime 7554253; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1079179) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1374 2) := by change ¬ Nat.Prime 7551505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1510301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1374 3) := by change ¬ Nat.Prime 7548757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 397303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1374 : row 1374 = 0 := (row_1374.checked).trans rfl
theorem model_count_1374 : modelRow 1374 = 0 := by decide
theorem modelrow_agrees_1374 : modelRow 1374 = row 1374 := model_count_1374.trans row_count_1374.symm
def row_1375 : RowCertificate 1375 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1375 1) := by change ¬ Nat.Prime 7565251; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 50101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1375 2) := by change ¬ Nat.Prime 7562501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 444853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1375 3) := by change ¬ Nat.Prime 7559751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2519917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1375 : row 1375 = 0 := (row_1375.checked).trans rfl
theorem model_count_1375 : modelRow 1375 = 0 := by decide
theorem modelrow_agrees_1375 : modelRow 1375 = row 1375 := model_count_1375.trans row_count_1375.symm
def row_1376 : RowCertificate 1376 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1376 1) := by change ¬ Nat.Prime 7576257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2525419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1376 2) := by change ¬ Nat.Prime 7573505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1514701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1376 3) := by change ¬ Nat.Prime 7570753; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 78049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1376 : row 1376 = 0 := (row_1376.checked).trans rfl
theorem model_count_1376 : modelRow 1376 = 0 := by decide
theorem modelrow_agrees_1376 : modelRow 1376 = row 1376 := model_count_1376.trans row_count_1376.symm
def row_1377 : RowCertificate 1377 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1377 1) := by change ¬ Nat.Prime 7587271; exact PrattCertificates.composite_of_product (factor := 2161) (cofactor := 3511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1377 2) := by change Nat.Prime 7584517; exact ProjectEulerSolutions.P58PrimeNodes.p_7584517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1377 3) := by change ¬ Nat.Prime 7581763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1083109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1377 : row 1377 = 1 := (row_1377.checked).trans rfl
theorem model_count_1377 : modelRow 1377 = 1 := by decide
theorem modelrow_agrees_1377 : modelRow 1377 = row 1377 := model_count_1377.trans row_count_1377.symm
def row_1378 : RowCertificate 1378 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1378 1) := by change ¬ Nat.Prime 7598293; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 16411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1378 2) := by change ¬ Nat.Prime 7595537; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 185257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1378 3) := by change ¬ Nat.Prime 7592781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2530927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1378 : row 1378 = 0 := (row_1378.checked).trans rfl
theorem model_count_1378 : modelRow 1378 = 0 := by decide
theorem modelrow_agrees_1378 : modelRow 1378 = row 1378 := model_count_1378.trans row_count_1378.symm
def row_1379 : RowCertificate 1379 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1379 1) := by change ¬ Nat.Prime 7609323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2536441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1379 2) := by change ¬ Nat.Prime 7606565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1521313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1379 3) := by change ¬ Nat.Prime 7603807; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 36037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1379 : row 1379 = 0 := (row_1379.checked).trans rfl
theorem model_count_1379 : modelRow 1379 = 0 := by decide
theorem modelrow_agrees_1379 : modelRow 1379 = row 1379 := model_count_1379.trans row_count_1379.symm
def row_1380 : RowCertificate 1380 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1380 1) := by change ¬ Nat.Prime 7620361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1088623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1380 2) := by change Nat.Prime 7617601; exact ProjectEulerSolutions.P58PrimeNodes.p_7617601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1380 3) := by change ¬ Nat.Prime 7614841; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 585757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1380 : row 1380 = 1 := (row_1380.checked).trans rfl
theorem model_count_1380 : modelRow 1380 = 1 := by decide
theorem modelrow_agrees_1380 : modelRow 1380 = row 1380 := model_count_1380.trans row_count_1380.symm
def row_1381 : RowCertificate 1381 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1381 1) := by change ¬ Nat.Prime 7631407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1090201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1381 2) := by change ¬ Nat.Prime 7628645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1525729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1381 3) := by change ¬ Nat.Prime 7625883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2541961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1381 : row 1381 = 0 := (row_1381.checked).trans rfl
theorem model_count_1381 : modelRow 1381 = 0 := by decide
theorem modelrow_agrees_1381 : modelRow 1381 = row 1381 := model_count_1381.trans row_count_1381.symm
def row_1382 : RowCertificate 1382 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1382 1) := by change ¬ Nat.Prime 7642461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2547487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1382 2) := by change ¬ Nat.Prime 7639697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 587669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1382 3) := by change ¬ Nat.Prime 7636933; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 42193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1382 : row 1382 = 0 := (row_1382.checked).trans rfl
theorem model_count_1382 : modelRow 1382 = 0 := by decide
theorem modelrow_agrees_1382 : modelRow 1382 = row 1382 := model_count_1382.trans row_count_1382.symm
def row_1383 : RowCertificate 1383 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1383 1) := by change ¬ Nat.Prime 7653523; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 402817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1383 2) := by change Nat.Prime 7650757; exact ProjectEulerSolutions.P58PrimeNodes.p_7650757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1383 3) := by change ¬ Nat.Prime 7647991; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 588307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1383 : row 1383 = 1 := (row_1383.checked).trans rfl
theorem model_count_1383 : modelRow 1383 = 1 := by decide
theorem modelrow_agrees_1383 : modelRow 1383 = row 1383 := model_count_1383.trans row_count_1383.symm
def row_1384 : RowCertificate 1384 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1384 1) := by change ¬ Nat.Prime 7664593; exact PrattCertificates.composite_of_product (factor := 787) (cofactor := 9739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1384 2) := by change ¬ Nat.Prime 7661825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1532365) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1384 3) := by change ¬ Nat.Prime 7659057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2553019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1384 : row 1384 = 0 := (row_1384.checked).trans rfl
theorem model_count_1384 : modelRow 1384 = 0 := by decide
theorem modelrow_agrees_1384 : modelRow 1384 = row 1384 := model_count_1384.trans row_count_1384.symm
def row_1385 : RowCertificate 1385 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1385 1) := by change ¬ Nat.Prime 7675671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2558557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1385 2) := by change Nat.Prime 7672901; exact ProjectEulerSolutions.P58PrimeNodes.p_7672901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1385 3) := by change ¬ Nat.Prime 7670131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1095733) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1385 : row 1385 = 1 := (row_1385.checked).trans rfl
theorem model_count_1385 : modelRow 1385 = 1 := by decide
theorem modelrow_agrees_1385 : modelRow 1385 = row 1385 := model_count_1385.trans row_count_1385.symm
def row_1386 : RowCertificate 1386 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1386 1) := by change ¬ Nat.Prime 7686757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 591289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1386 2) := by change ¬ Nat.Prime 7683985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1536797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1386 3) := by change Nat.Prime 7681213; exact ProjectEulerSolutions.P58PrimeNodes.p_7681213.prime
    exact decide_eq_true prime
theorem row_count_1386 : row 1386 = 1 := (row_1386.checked).trans rfl
theorem model_count_1386 : modelRow 1386 = 1 := by decide
theorem modelrow_agrees_1386 : modelRow 1386 = row 1386 := model_count_1386.trans row_count_1386.symm
def row_1387 : RowCertificate 1387 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1387 1) := by change ¬ Nat.Prime 7697851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1099693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1387 2) := by change ¬ Nat.Prime 7695077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 591929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1387 3) := by change ¬ Nat.Prime 7692303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2564101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1387 : row 1387 = 0 := (row_1387.checked).trans rfl
theorem model_count_1387 : modelRow 1387 = 0 := by decide
theorem modelrow_agrees_1387 : modelRow 1387 = row 1387 := model_count_1387.trans row_count_1387.symm
def row_1388 : RowCertificate 1388 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1388 1) := by change ¬ Nat.Prime 7708953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2569651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1388 2) := by change Nat.Prime 7706177; exact ProjectEulerSolutions.P58PrimeNodes.p_7706177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1388 3) := by change ¬ Nat.Prime 7703401; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 14083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1388 : row 1388 = 1 := (row_1388.checked).trans rfl
theorem model_count_1388 : modelRow 1388 = 1 := by decide
theorem modelrow_agrees_1388 : modelRow 1388 = row 1388 := model_count_1388.trans row_count_1388.symm
def row_1389 : RowCertificate 1389 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1389 1) := by change ¬ Nat.Prime 7720063; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 593851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1389 2) := by change ¬ Nat.Prime 7717285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1543457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1389 3) := by change ¬ Nat.Prime 7714507; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 79531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1389 : row 1389 = 0 := (row_1389.checked).trans rfl
theorem model_count_1389 : modelRow 1389 = 0 := by decide
theorem modelrow_agrees_1389 : modelRow 1389 = row 1389 := model_count_1389.trans row_count_1389.symm
def row_1390 : RowCertificate 1390 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1390 1) := by change ¬ Nat.Prime 7731181; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 25183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1390 2) := by change Nat.Prime 7728401; exact ProjectEulerSolutions.P58PrimeNodes.p_7728401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1390 3) := by change ¬ Nat.Prime 7725621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2575207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1390 : row 1390 = 1 := (row_1390.checked).trans rfl
theorem model_count_1390 : modelRow 1390 = 1 := by decide
theorem modelrow_agrees_1390 : modelRow 1390 = row 1390 := model_count_1390.trans row_count_1390.symm
def row_1391 : RowCertificate 1391 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1391 1) := by change ¬ Nat.Prime 7742307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2580769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1391 2) := by change ¬ Nat.Prime 7739525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1547905) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1391 3) := by change ¬ Nat.Prime 7736743; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1105249) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1391 : row 1391 = 0 := (row_1391.checked).trans rfl
theorem model_count_1391 : modelRow 1391 = 0 := by decide
theorem modelrow_agrees_1391 : modelRow 1391 = row 1391 := model_count_1391.trans row_count_1391.symm
def row_1392 : RowCertificate 1392 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1392 1) := by change ¬ Nat.Prime 7753441; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 250111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1392 2) := by change ¬ Nat.Prime 7750657; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 455921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1392 3) := by change ¬ Nat.Prime 7747873; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1106839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1392 : row 1392 = 0 := (row_1392.checked).trans rfl
theorem model_count_1392 : modelRow 1392 = 0 := by decide
theorem modelrow_agrees_1392 : modelRow 1392 = row 1392 := model_count_1392.trans row_count_1392.symm
def row_1393 : RowCertificate 1393 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1393 1) := by change ¬ Nat.Prime 7764583; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 40231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1393 2) := by change ¬ Nat.Prime 7761797; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 146449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1393 3) := by change ¬ Nat.Prime 7759011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2586337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1393 : row 1393 = 0 := (row_1393.checked).trans rfl
theorem model_count_1393 : modelRow 1393 = 0 := by decide
theorem modelrow_agrees_1393 : modelRow 1393 = row 1393 := model_count_1393.trans row_count_1393.symm
def row_1394 : RowCertificate 1394 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1394 1) := by change ¬ Nat.Prime 7775733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2591911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1394 2) := by change ¬ Nat.Prime 7772945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1554589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1394 3) := by change Nat.Prime 7770157; exact ProjectEulerSolutions.P58PrimeNodes.p_7770157.prime
    exact decide_eq_true prime
theorem row_count_1394 : row 1394 = 1 := (row_1394.checked).trans rfl
theorem model_count_1394 : modelRow 1394 = 1 := by decide
theorem modelrow_agrees_1394 : modelRow 1394 = row 1394 := model_count_1394.trans row_count_1394.symm
def row_1395 : RowCertificate 1395 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1395 1) := by change ¬ Nat.Prime 7786891; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1112413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1395 2) := by change ¬ Nat.Prime 7784101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 598777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1395 3) := by change Nat.Prime 7781311; exact ProjectEulerSolutions.P58PrimeNodes.p_7781311.prime
    exact decide_eq_true prime
theorem row_count_1395 : row 1395 = 1 := (row_1395.checked).trans rfl
theorem model_count_1395 : modelRow 1395 = 1 := by decide
theorem modelrow_agrees_1395 : modelRow 1395 = row 1395 := model_count_1395.trans row_count_1395.symm
def row_1396 : RowCertificate 1396 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1396 1) := by change ¬ Nat.Prime 7798057; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 127837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1396 2) := by change ¬ Nat.Prime 7795265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1559053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1396 3) := by change ¬ Nat.Prime 7792473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2597491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1396 : row 1396 = 0 := (row_1396.checked).trans rfl
theorem model_count_1396 : modelRow 1396 = 0 := by decide
theorem modelrow_agrees_1396 : modelRow 1396 = row 1396 := model_count_1396.trans row_count_1396.symm
def row_1397 : RowCertificate 1397 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1397 1) := by change ¬ Nat.Prime 7809231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2603077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1397 2) := by change Nat.Prime 7806437; exact ProjectEulerSolutions.P58PrimeNodes.p_7806437.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1397 3) := by change Nat.Prime 7803643; exact ProjectEulerSolutions.P58PrimeNodes.p_7803643.prime
    exact decide_eq_true prime
theorem row_count_1397 : row 1397 = 2 := (row_1397.checked).trans rfl
theorem model_count_1397 : modelRow 1397 = 2 := by decide
theorem modelrow_agrees_1397 : modelRow 1397 = row 1397 := model_count_1397.trans row_count_1397.symm
def row_1398 : RowCertificate 1398 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1398 1) := by change ¬ Nat.Prime 7820413; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 18061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1398 2) := by change ¬ Nat.Prime 7817617; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 269573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1398 3) := by change ¬ Nat.Prime 7814821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1116403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1398 : row 1398 = 0 := (row_1398.checked).trans rfl
theorem model_count_1398 : modelRow 1398 = 0 := by decide
theorem modelrow_agrees_1398 : modelRow 1398 = row 1398 := model_count_1398.trans row_count_1398.symm
def row_1399 : RowCertificate 1399 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1399 1) := by change ¬ Nat.Prime 7831603; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 602431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1399 2) := by change ¬ Nat.Prime 7828805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1565761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1399 3) := by change ¬ Nat.Prime 7826007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2608669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1399 : row 1399 = 0 := (row_1399.checked).trans rfl
theorem model_count_1399 : modelRow 1399 = 0 := by decide
theorem modelrow_agrees_1399 : modelRow 1399 = row 1399 := model_count_1399.trans row_count_1399.symm
def row_1400 : RowCertificate 1400 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1400 1) := by change ¬ Nat.Prime 7842801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2614267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1400 2) := by change ¬ Nat.Prime 7840001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 603077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1400 3) := by change ¬ Nat.Prime 7837201; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 16927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1400 : row 1400 = 0 := (row_1400.checked).trans rfl
theorem model_count_1400 : modelRow 1400 = 0 := by decide
theorem modelrow_agrees_1400 : modelRow 1400 = row 1400 := model_count_1400.trans row_count_1400.symm
def row_1401 : RowCertificate 1401 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1401 1) := by change ¬ Nat.Prime 7854007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1122001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1401 2) := by change ¬ Nat.Prime 7851205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1570241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1401 3) := by change ¬ Nat.Prime 7848403; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 212119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1401 : row 1401 = 0 := (row_1401.checked).trans rfl
theorem model_count_1401 : modelRow 1401 = 0 := by decide
theorem modelrow_agrees_1401 : modelRow 1401 = row 1401 := model_count_1401.trans row_count_1401.symm
def row_1402 : RowCertificate 1402 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1402 1) := by change ¬ Nat.Prime 7865221; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1123603) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1402 2) := by change Nat.Prime 7862417; exact ProjectEulerSolutions.P58PrimeNodes.p_7862417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1402 3) := by change ¬ Nat.Prime 7859613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2619871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1402 : row 1402 = 1 := (row_1402.checked).trans rfl
theorem model_count_1402 : modelRow 1402 = 1 := by decide
theorem modelrow_agrees_1402 : modelRow 1402 = row 1402 := model_count_1402.trans row_count_1402.symm
def row_1403 : RowCertificate 1403 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1403 1) := by change ¬ Nat.Prime 7876443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2625481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1403 2) := by change ¬ Nat.Prime 7873637; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 212801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1403 3) := by change ¬ Nat.Prime 7870831; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 17929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1403 : row 1403 = 0 := (row_1403.checked).trans rfl
theorem model_count_1403 : modelRow 1403 = 0 := by decide
theorem modelrow_agrees_1403 : modelRow 1403 = row 1403 := model_count_1403.trans row_count_1403.symm
def row_1404 : RowCertificate 1404 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1404 1) := by change ¬ Nat.Prime 7887673; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 6841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1404 2) := by change ¬ Nat.Prime 7884865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1576973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1404 3) := by change ¬ Nat.Prime 7882057; exact PrattCertificates.composite_of_product (factor := 1999) (cofactor := 3943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1404 : row 1404 = 0 := (row_1404.checked).trans rfl
theorem model_count_1404 : modelRow 1404 = 0 := by decide
theorem modelrow_agrees_1404 : modelRow 1404 = row 1404 := model_count_1404.trans row_count_1404.symm
def row_1405 : RowCertificate 1405 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1405 1) := by change ¬ Nat.Prime 7898911; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 40927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1405 2) := by change ¬ Nat.Prime 7896101; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 69877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1405 3) := by change ¬ Nat.Prime 7893291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2631097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1405 : row 1405 = 0 := (row_1405.checked).trans rfl
theorem model_count_1405 : modelRow 1405 = 0 := by decide
theorem modelrow_agrees_1405 : modelRow 1405 = row 1405 := model_count_1405.trans row_count_1405.symm
def row_1406 : RowCertificate 1406 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1406 1) := by change ¬ Nat.Prime 7910157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2636719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1406 2) := by change ¬ Nat.Prime 7907345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1581469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1406 3) := by change ¬ Nat.Prime 7904533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1129219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1406 : row 1406 = 0 := (row_1406.checked).trans rfl
theorem model_count_1406 : modelRow 1406 = 0 := by decide
theorem modelrow_agrees_1406 : modelRow 1406 = row 1406 := model_count_1406.trans row_count_1406.symm
def row_1407 : RowCertificate 1407 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1407 1) := by change Nat.Prime 7921411; exact ProjectEulerSolutions.P58PrimeNodes.p_7921411.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1407 2) := by change ¬ Nat.Prime 7918597; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 88973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1407 3) := by change ¬ Nat.Prime 7915783; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 62329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1407 : row 1407 = 1 := (row_1407.checked).trans rfl
theorem model_count_1407 : modelRow 1407 = 1 := by decide
theorem modelrow_agrees_1407 : modelRow 1407 = row 1407 := model_count_1407.trans row_count_1407.symm
def row_1408 : RowCertificate 1408 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1408 1) := by change ¬ Nat.Prime 7932673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1133239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1408 2) := by change ¬ Nat.Prime 7929857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 609989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1408 3) := by change ¬ Nat.Prime 7927041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2642347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1408 : row 1408 = 0 := (row_1408.checked).trans rfl
theorem model_count_1408 : modelRow 1408 = 0 := by decide
theorem modelrow_agrees_1408 : modelRow 1408 = row 1408 := model_count_1408.trans row_count_1408.symm
private def atomic_1281 : CountSegment 1 10 13120 1280 1281 691 693 := CountSegment.single (first := 1280) (initial := 691) row_count_1281 (by decide)
private theorem atomic_1281_agreement : RowAgreement 1280 1281 := RowAgreement.single (first := 1280) modelrow_agrees_1281
private def atomic_1282 : CountSegment 1 10 13120 1281 1282 693 693 := CountSegment.single (first := 1281) (initial := 693) row_count_1282 (by decide)
private theorem atomic_1282_agreement : RowAgreement 1281 1282 := RowAgreement.single (first := 1281) modelrow_agrees_1282
private def atomic_1283 : CountSegment 1 10 13120 1282 1283 693 693 := CountSegment.single (first := 1282) (initial := 693) row_count_1283 (by decide)
private theorem atomic_1283_agreement : RowAgreement 1282 1283 := RowAgreement.single (first := 1282) modelrow_agrees_1283
private def atomic_1284 : CountSegment 1 10 13120 1283 1284 693 694 := CountSegment.single (first := 1283) (initial := 693) row_count_1284 (by decide)
private theorem atomic_1284_agreement : RowAgreement 1283 1284 := RowAgreement.single (first := 1283) modelrow_agrees_1284
private def atomic_1285 : CountSegment 1 10 13120 1284 1285 694 695 := CountSegment.single (first := 1284) (initial := 694) row_count_1285 (by decide)
private theorem atomic_1285_agreement : RowAgreement 1284 1285 := RowAgreement.single (first := 1284) modelrow_agrees_1285
private def atomic_1286 : CountSegment 1 10 13120 1285 1286 695 695 := CountSegment.single (first := 1285) (initial := 695) row_count_1286 (by decide)
private theorem atomic_1286_agreement : RowAgreement 1285 1286 := RowAgreement.single (first := 1285) modelrow_agrees_1286
private def atomic_1287 : CountSegment 1 10 13120 1286 1287 695 696 := CountSegment.single (first := 1286) (initial := 695) row_count_1287 (by decide)
private theorem atomic_1287_agreement : RowAgreement 1286 1287 := RowAgreement.single (first := 1286) modelrow_agrees_1287
private def atomic_1288 : CountSegment 1 10 13120 1287 1288 696 697 := CountSegment.single (first := 1287) (initial := 696) row_count_1288 (by decide)
private theorem atomic_1288_agreement : RowAgreement 1287 1288 := RowAgreement.single (first := 1287) modelrow_agrees_1288
private def atomic_1289 : CountSegment 1 10 13120 1288 1289 697 697 := CountSegment.single (first := 1288) (initial := 697) row_count_1289 (by decide)
private theorem atomic_1289_agreement : RowAgreement 1288 1289 := RowAgreement.single (first := 1288) modelrow_agrees_1289
private def atomic_1290 : CountSegment 1 10 13120 1289 1290 697 697 := CountSegment.single (first := 1289) (initial := 697) row_count_1290 (by decide)
private theorem atomic_1290_agreement : RowAgreement 1289 1290 := RowAgreement.single (first := 1289) modelrow_agrees_1290
private def atomic_1291 : CountSegment 1 10 13120 1290 1291 697 697 := CountSegment.single (first := 1290) (initial := 697) row_count_1291 (by decide)
private theorem atomic_1291_agreement : RowAgreement 1290 1291 := RowAgreement.single (first := 1290) modelrow_agrees_1291
private def atomic_1292 : CountSegment 1 10 13120 1291 1292 697 697 := CountSegment.single (first := 1291) (initial := 697) row_count_1292 (by decide)
private theorem atomic_1292_agreement : RowAgreement 1291 1292 := RowAgreement.single (first := 1291) modelrow_agrees_1292
private def atomic_1293 : CountSegment 1 10 13120 1292 1293 697 697 := CountSegment.single (first := 1292) (initial := 697) row_count_1293 (by decide)
private theorem atomic_1293_agreement : RowAgreement 1292 1293 := RowAgreement.single (first := 1292) modelrow_agrees_1293
private def atomic_1294 : CountSegment 1 10 13120 1293 1294 697 697 := CountSegment.single (first := 1293) (initial := 697) row_count_1294 (by decide)
private theorem atomic_1294_agreement : RowAgreement 1293 1294 := RowAgreement.single (first := 1293) modelrow_agrees_1294
private def atomic_1295 : CountSegment 1 10 13120 1294 1295 697 698 := CountSegment.single (first := 1294) (initial := 697) row_count_1295 (by decide)
private theorem atomic_1295_agreement : RowAgreement 1294 1295 := RowAgreement.single (first := 1294) modelrow_agrees_1295
private def atomic_1296 : CountSegment 1 10 13120 1295 1296 698 698 := CountSegment.single (first := 1295) (initial := 698) row_count_1296 (by decide)
private theorem atomic_1296_agreement : RowAgreement 1295 1296 := RowAgreement.single (first := 1295) modelrow_agrees_1296
private def atomic_1297 : CountSegment 1 10 13120 1296 1297 698 699 := CountSegment.single (first := 1296) (initial := 698) row_count_1297 (by decide)
private theorem atomic_1297_agreement : RowAgreement 1296 1297 := RowAgreement.single (first := 1296) modelrow_agrees_1297
private def atomic_1298 : CountSegment 1 10 13120 1297 1298 699 699 := CountSegment.single (first := 1297) (initial := 699) row_count_1298 (by decide)
private theorem atomic_1298_agreement : RowAgreement 1297 1298 := RowAgreement.single (first := 1297) modelrow_agrees_1298
private def atomic_1299 : CountSegment 1 10 13120 1298 1299 699 700 := CountSegment.single (first := 1298) (initial := 699) row_count_1299 (by decide)
private theorem atomic_1299_agreement : RowAgreement 1298 1299 := RowAgreement.single (first := 1298) modelrow_agrees_1299
private def atomic_1300 : CountSegment 1 10 13120 1299 1300 700 701 := CountSegment.single (first := 1299) (initial := 700) row_count_1300 (by decide)
private theorem atomic_1300_agreement : RowAgreement 1299 1300 := RowAgreement.single (first := 1299) modelrow_agrees_1300
private def atomic_1301 : CountSegment 1 10 13120 1300 1301 701 701 := CountSegment.single (first := 1300) (initial := 701) row_count_1301 (by decide)
private theorem atomic_1301_agreement : RowAgreement 1300 1301 := RowAgreement.single (first := 1300) modelrow_agrees_1301
private def atomic_1302 : CountSegment 1 10 13120 1301 1302 701 703 := CountSegment.single (first := 1301) (initial := 701) row_count_1302 (by decide)
private theorem atomic_1302_agreement : RowAgreement 1301 1302 := RowAgreement.single (first := 1301) modelrow_agrees_1302
private def atomic_1303 : CountSegment 1 10 13120 1302 1303 703 703 := CountSegment.single (first := 1302) (initial := 703) row_count_1303 (by decide)
private theorem atomic_1303_agreement : RowAgreement 1302 1303 := RowAgreement.single (first := 1302) modelrow_agrees_1303
private def atomic_1304 : CountSegment 1 10 13120 1303 1304 703 704 := CountSegment.single (first := 1303) (initial := 703) row_count_1304 (by decide)
private theorem atomic_1304_agreement : RowAgreement 1303 1304 := RowAgreement.single (first := 1303) modelrow_agrees_1304
private def atomic_1305 : CountSegment 1 10 13120 1304 1305 704 704 := CountSegment.single (first := 1304) (initial := 704) row_count_1305 (by decide)
private theorem atomic_1305_agreement : RowAgreement 1304 1305 := RowAgreement.single (first := 1304) modelrow_agrees_1305
private def atomic_1306 : CountSegment 1 10 13120 1305 1306 704 704 := CountSegment.single (first := 1305) (initial := 704) row_count_1306 (by decide)
private theorem atomic_1306_agreement : RowAgreement 1305 1306 := RowAgreement.single (first := 1305) modelrow_agrees_1306
private def atomic_1307 : CountSegment 1 10 13120 1306 1307 704 704 := CountSegment.single (first := 1306) (initial := 704) row_count_1307 (by decide)
private theorem atomic_1307_agreement : RowAgreement 1306 1307 := RowAgreement.single (first := 1306) modelrow_agrees_1307
private def atomic_1308 : CountSegment 1 10 13120 1307 1308 704 704 := CountSegment.single (first := 1307) (initial := 704) row_count_1308 (by decide)
private theorem atomic_1308_agreement : RowAgreement 1307 1308 := RowAgreement.single (first := 1307) modelrow_agrees_1308
private def atomic_1309 : CountSegment 1 10 13120 1308 1309 704 705 := CountSegment.single (first := 1308) (initial := 704) row_count_1309 (by decide)
private theorem atomic_1309_agreement : RowAgreement 1308 1309 := RowAgreement.single (first := 1308) modelrow_agrees_1309
private def atomic_1310 : CountSegment 1 10 13120 1309 1310 705 705 := CountSegment.single (first := 1309) (initial := 705) row_count_1310 (by decide)
private theorem atomic_1310_agreement : RowAgreement 1309 1310 := RowAgreement.single (first := 1309) modelrow_agrees_1310
private def atomic_1311 : CountSegment 1 10 13120 1310 1311 705 705 := CountSegment.single (first := 1310) (initial := 705) row_count_1311 (by decide)
private theorem atomic_1311_agreement : RowAgreement 1310 1311 := RowAgreement.single (first := 1310) modelrow_agrees_1311
private def atomic_1312 : CountSegment 1 10 13120 1311 1312 705 706 := CountSegment.single (first := 1311) (initial := 705) row_count_1312 (by decide)
private theorem atomic_1312_agreement : RowAgreement 1311 1312 := RowAgreement.single (first := 1311) modelrow_agrees_1312
private def atomic_1313 : CountSegment 1 10 13120 1312 1313 706 707 := CountSegment.single (first := 1312) (initial := 706) row_count_1313 (by decide)
private theorem atomic_1313_agreement : RowAgreement 1312 1313 := RowAgreement.single (first := 1312) modelrow_agrees_1313
private def atomic_1314 : CountSegment 1 10 13120 1313 1314 707 708 := CountSegment.single (first := 1313) (initial := 707) row_count_1314 (by decide)
private theorem atomic_1314_agreement : RowAgreement 1313 1314 := RowAgreement.single (first := 1313) modelrow_agrees_1314
private def atomic_1315 : CountSegment 1 10 13120 1314 1315 708 708 := CountSegment.single (first := 1314) (initial := 708) row_count_1315 (by decide)
private theorem atomic_1315_agreement : RowAgreement 1314 1315 := RowAgreement.single (first := 1314) modelrow_agrees_1315
private def atomic_1316 : CountSegment 1 10 13120 1315 1316 708 708 := CountSegment.single (first := 1315) (initial := 708) row_count_1316 (by decide)
private theorem atomic_1316_agreement : RowAgreement 1315 1316 := RowAgreement.single (first := 1315) modelrow_agrees_1316
private def atomic_1317 : CountSegment 1 10 13120 1316 1317 708 708 := CountSegment.single (first := 1316) (initial := 708) row_count_1317 (by decide)
private theorem atomic_1317_agreement : RowAgreement 1316 1317 := RowAgreement.single (first := 1316) modelrow_agrees_1317
private def atomic_1318 : CountSegment 1 10 13120 1317 1318 708 708 := CountSegment.single (first := 1317) (initial := 708) row_count_1318 (by decide)
private theorem atomic_1318_agreement : RowAgreement 1317 1318 := RowAgreement.single (first := 1317) modelrow_agrees_1318
private def atomic_1319 : CountSegment 1 10 13120 1318 1319 708 708 := CountSegment.single (first := 1318) (initial := 708) row_count_1319 (by decide)
private theorem atomic_1319_agreement : RowAgreement 1318 1319 := RowAgreement.single (first := 1318) modelrow_agrees_1319
private def atomic_1320 : CountSegment 1 10 13120 1319 1320 708 709 := CountSegment.single (first := 1319) (initial := 708) row_count_1320 (by decide)
private theorem atomic_1320_agreement : RowAgreement 1319 1320 := RowAgreement.single (first := 1319) modelrow_agrees_1320
private def atomic_1321 : CountSegment 1 10 13120 1320 1321 709 709 := CountSegment.single (first := 1320) (initial := 709) row_count_1321 (by decide)
private theorem atomic_1321_agreement : RowAgreement 1320 1321 := RowAgreement.single (first := 1320) modelrow_agrees_1321
private def atomic_1322 : CountSegment 1 10 13120 1321 1322 709 709 := CountSegment.single (first := 1321) (initial := 709) row_count_1322 (by decide)
private theorem atomic_1322_agreement : RowAgreement 1321 1322 := RowAgreement.single (first := 1321) modelrow_agrees_1322
private def atomic_1323 : CountSegment 1 10 13120 1322 1323 709 711 := CountSegment.single (first := 1322) (initial := 709) row_count_1323 (by decide)
private theorem atomic_1323_agreement : RowAgreement 1322 1323 := RowAgreement.single (first := 1322) modelrow_agrees_1323
private def atomic_1324 : CountSegment 1 10 13120 1323 1324 711 711 := CountSegment.single (first := 1323) (initial := 711) row_count_1324 (by decide)
private theorem atomic_1324_agreement : RowAgreement 1323 1324 := RowAgreement.single (first := 1323) modelrow_agrees_1324
private def atomic_1325 : CountSegment 1 10 13120 1324 1325 711 712 := CountSegment.single (first := 1324) (initial := 711) row_count_1325 (by decide)
private theorem atomic_1325_agreement : RowAgreement 1324 1325 := RowAgreement.single (first := 1324) modelrow_agrees_1325
private def atomic_1326 : CountSegment 1 10 13120 1325 1326 712 713 := CountSegment.single (first := 1325) (initial := 712) row_count_1326 (by decide)
private theorem atomic_1326_agreement : RowAgreement 1325 1326 := RowAgreement.single (first := 1325) modelrow_agrees_1326
private def atomic_1327 : CountSegment 1 10 13120 1326 1327 713 714 := CountSegment.single (first := 1326) (initial := 713) row_count_1327 (by decide)
private theorem atomic_1327_agreement : RowAgreement 1326 1327 := RowAgreement.single (first := 1326) modelrow_agrees_1327
private def atomic_1328 : CountSegment 1 10 13120 1327 1328 714 714 := CountSegment.single (first := 1327) (initial := 714) row_count_1328 (by decide)
private theorem atomic_1328_agreement : RowAgreement 1327 1328 := RowAgreement.single (first := 1327) modelrow_agrees_1328
private def atomic_1329 : CountSegment 1 10 13120 1328 1329 714 715 := CountSegment.single (first := 1328) (initial := 714) row_count_1329 (by decide)
private theorem atomic_1329_agreement : RowAgreement 1328 1329 := RowAgreement.single (first := 1328) modelrow_agrees_1329
private def atomic_1330 : CountSegment 1 10 13120 1329 1330 715 715 := CountSegment.single (first := 1329) (initial := 715) row_count_1330 (by decide)
private theorem atomic_1330_agreement : RowAgreement 1329 1330 := RowAgreement.single (first := 1329) modelrow_agrees_1330
private def atomic_1331 : CountSegment 1 10 13120 1330 1331 715 715 := CountSegment.single (first := 1330) (initial := 715) row_count_1331 (by decide)
private theorem atomic_1331_agreement : RowAgreement 1330 1331 := RowAgreement.single (first := 1330) modelrow_agrees_1331
private def atomic_1332 : CountSegment 1 10 13120 1331 1332 715 717 := CountSegment.single (first := 1331) (initial := 715) row_count_1332 (by decide)
private theorem atomic_1332_agreement : RowAgreement 1331 1332 := RowAgreement.single (first := 1331) modelrow_agrees_1332
private def atomic_1333 : CountSegment 1 10 13120 1332 1333 717 718 := CountSegment.single (first := 1332) (initial := 717) row_count_1333 (by decide)
private theorem atomic_1333_agreement : RowAgreement 1332 1333 := RowAgreement.single (first := 1332) modelrow_agrees_1333
private def atomic_1334 : CountSegment 1 10 13120 1333 1334 718 718 := CountSegment.single (first := 1333) (initial := 718) row_count_1334 (by decide)
private theorem atomic_1334_agreement : RowAgreement 1333 1334 := RowAgreement.single (first := 1333) modelrow_agrees_1334
private def atomic_1335 : CountSegment 1 10 13120 1334 1335 718 718 := CountSegment.single (first := 1334) (initial := 718) row_count_1335 (by decide)
private theorem atomic_1335_agreement : RowAgreement 1334 1335 := RowAgreement.single (first := 1334) modelrow_agrees_1335
private def atomic_1336 : CountSegment 1 10 13120 1335 1336 718 718 := CountSegment.single (first := 1335) (initial := 718) row_count_1336 (by decide)
private theorem atomic_1336_agreement : RowAgreement 1335 1336 := RowAgreement.single (first := 1335) modelrow_agrees_1336
private def atomic_1337 : CountSegment 1 10 13120 1336 1337 718 719 := CountSegment.single (first := 1336) (initial := 718) row_count_1337 (by decide)
private theorem atomic_1337_agreement : RowAgreement 1336 1337 := RowAgreement.single (first := 1336) modelrow_agrees_1337
private def atomic_1338 : CountSegment 1 10 13120 1337 1338 719 720 := CountSegment.single (first := 1337) (initial := 719) row_count_1338 (by decide)
private theorem atomic_1338_agreement : RowAgreement 1337 1338 := RowAgreement.single (first := 1337) modelrow_agrees_1338
private def atomic_1339 : CountSegment 1 10 13120 1338 1339 720 720 := CountSegment.single (first := 1338) (initial := 720) row_count_1339 (by decide)
private theorem atomic_1339_agreement : RowAgreement 1338 1339 := RowAgreement.single (first := 1338) modelrow_agrees_1339
private def atomic_1340 : CountSegment 1 10 13120 1339 1340 720 720 := CountSegment.single (first := 1339) (initial := 720) row_count_1340 (by decide)
private theorem atomic_1340_agreement : RowAgreement 1339 1340 := RowAgreement.single (first := 1339) modelrow_agrees_1340
private def atomic_1341 : CountSegment 1 10 13120 1340 1341 720 721 := CountSegment.single (first := 1340) (initial := 720) row_count_1341 (by decide)
private theorem atomic_1341_agreement : RowAgreement 1340 1341 := RowAgreement.single (first := 1340) modelrow_agrees_1341
private def atomic_1342 : CountSegment 1 10 13120 1341 1342 721 723 := CountSegment.single (first := 1341) (initial := 721) row_count_1342 (by decide)
private theorem atomic_1342_agreement : RowAgreement 1341 1342 := RowAgreement.single (first := 1341) modelrow_agrees_1342
private def atomic_1343 : CountSegment 1 10 13120 1342 1343 723 723 := CountSegment.single (first := 1342) (initial := 723) row_count_1343 (by decide)
private theorem atomic_1343_agreement : RowAgreement 1342 1343 := RowAgreement.single (first := 1342) modelrow_agrees_1343
private def atomic_1344 : CountSegment 1 10 13120 1343 1344 723 724 := CountSegment.single (first := 1343) (initial := 723) row_count_1344 (by decide)
private theorem atomic_1344_agreement : RowAgreement 1343 1344 := RowAgreement.single (first := 1343) modelrow_agrees_1344
private def atomic_1345 : CountSegment 1 10 13120 1344 1345 724 724 := CountSegment.single (first := 1344) (initial := 724) row_count_1345 (by decide)
private theorem atomic_1345_agreement : RowAgreement 1344 1345 := RowAgreement.single (first := 1344) modelrow_agrees_1345
private def atomic_1346 : CountSegment 1 10 13120 1345 1346 724 724 := CountSegment.single (first := 1345) (initial := 724) row_count_1346 (by decide)
private theorem atomic_1346_agreement : RowAgreement 1345 1346 := RowAgreement.single (first := 1345) modelrow_agrees_1346
private def atomic_1347 : CountSegment 1 10 13120 1346 1347 724 724 := CountSegment.single (first := 1346) (initial := 724) row_count_1347 (by decide)
private theorem atomic_1347_agreement : RowAgreement 1346 1347 := RowAgreement.single (first := 1346) modelrow_agrees_1347
private def atomic_1348 : CountSegment 1 10 13120 1347 1348 724 725 := CountSegment.single (first := 1347) (initial := 724) row_count_1348 (by decide)
private theorem atomic_1348_agreement : RowAgreement 1347 1348 := RowAgreement.single (first := 1347) modelrow_agrees_1348
private def atomic_1349 : CountSegment 1 10 13120 1348 1349 725 725 := CountSegment.single (first := 1348) (initial := 725) row_count_1349 (by decide)
private theorem atomic_1349_agreement : RowAgreement 1348 1349 := RowAgreement.single (first := 1348) modelrow_agrees_1349
private def atomic_1350 : CountSegment 1 10 13120 1349 1350 725 726 := CountSegment.single (first := 1349) (initial := 725) row_count_1350 (by decide)
private theorem atomic_1350_agreement : RowAgreement 1349 1350 := RowAgreement.single (first := 1349) modelrow_agrees_1350
private def atomic_1351 : CountSegment 1 10 13120 1350 1351 726 726 := CountSegment.single (first := 1350) (initial := 726) row_count_1351 (by decide)
private theorem atomic_1351_agreement : RowAgreement 1350 1351 := RowAgreement.single (first := 1350) modelrow_agrees_1351
private def atomic_1352 : CountSegment 1 10 13120 1351 1352 726 727 := CountSegment.single (first := 1351) (initial := 726) row_count_1352 (by decide)
private theorem atomic_1352_agreement : RowAgreement 1351 1352 := RowAgreement.single (first := 1351) modelrow_agrees_1352
private def atomic_1353 : CountSegment 1 10 13120 1352 1353 727 728 := CountSegment.single (first := 1352) (initial := 727) row_count_1353 (by decide)
private theorem atomic_1353_agreement : RowAgreement 1352 1353 := RowAgreement.single (first := 1352) modelrow_agrees_1353
private def atomic_1354 : CountSegment 1 10 13120 1353 1354 728 729 := CountSegment.single (first := 1353) (initial := 728) row_count_1354 (by decide)
private theorem atomic_1354_agreement : RowAgreement 1353 1354 := RowAgreement.single (first := 1353) modelrow_agrees_1354
private def atomic_1355 : CountSegment 1 10 13120 1354 1355 729 729 := CountSegment.single (first := 1354) (initial := 729) row_count_1355 (by decide)
private theorem atomic_1355_agreement : RowAgreement 1354 1355 := RowAgreement.single (first := 1354) modelrow_agrees_1355
private def atomic_1356 : CountSegment 1 10 13120 1355 1356 729 729 := CountSegment.single (first := 1355) (initial := 729) row_count_1356 (by decide)
private theorem atomic_1356_agreement : RowAgreement 1355 1356 := RowAgreement.single (first := 1355) modelrow_agrees_1356
private def atomic_1357 : CountSegment 1 10 13120 1356 1357 729 730 := CountSegment.single (first := 1356) (initial := 729) row_count_1357 (by decide)
private theorem atomic_1357_agreement : RowAgreement 1356 1357 := RowAgreement.single (first := 1356) modelrow_agrees_1357
private def atomic_1358 : CountSegment 1 10 13120 1357 1358 730 730 := CountSegment.single (first := 1357) (initial := 730) row_count_1358 (by decide)
private theorem atomic_1358_agreement : RowAgreement 1357 1358 := RowAgreement.single (first := 1357) modelrow_agrees_1358
private def atomic_1359 : CountSegment 1 10 13120 1358 1359 730 730 := CountSegment.single (first := 1358) (initial := 730) row_count_1359 (by decide)
private theorem atomic_1359_agreement : RowAgreement 1358 1359 := RowAgreement.single (first := 1358) modelrow_agrees_1359
private def atomic_1360 : CountSegment 1 10 13120 1359 1360 730 730 := CountSegment.single (first := 1359) (initial := 730) row_count_1360 (by decide)
private theorem atomic_1360_agreement : RowAgreement 1359 1360 := RowAgreement.single (first := 1359) modelrow_agrees_1360
private def atomic_1361 : CountSegment 1 10 13120 1360 1361 730 731 := CountSegment.single (first := 1360) (initial := 730) row_count_1361 (by decide)
private theorem atomic_1361_agreement : RowAgreement 1360 1361 := RowAgreement.single (first := 1360) modelrow_agrees_1361
private def atomic_1362 : CountSegment 1 10 13120 1361 1362 731 732 := CountSegment.single (first := 1361) (initial := 731) row_count_1362 (by decide)
private theorem atomic_1362_agreement : RowAgreement 1361 1362 := RowAgreement.single (first := 1361) modelrow_agrees_1362
private def atomic_1363 : CountSegment 1 10 13120 1362 1363 732 732 := CountSegment.single (first := 1362) (initial := 732) row_count_1363 (by decide)
private theorem atomic_1363_agreement : RowAgreement 1362 1363 := RowAgreement.single (first := 1362) modelrow_agrees_1363
private def atomic_1364 : CountSegment 1 10 13120 1363 1364 732 732 := CountSegment.single (first := 1363) (initial := 732) row_count_1364 (by decide)
private theorem atomic_1364_agreement : RowAgreement 1363 1364 := RowAgreement.single (first := 1363) modelrow_agrees_1364
private def atomic_1365 : CountSegment 1 10 13120 1364 1365 732 735 := CountSegment.single (first := 1364) (initial := 732) row_count_1365 (by decide)
private theorem atomic_1365_agreement : RowAgreement 1364 1365 := RowAgreement.single (first := 1364) modelrow_agrees_1365
private def atomic_1366 : CountSegment 1 10 13120 1365 1366 735 735 := CountSegment.single (first := 1365) (initial := 735) row_count_1366 (by decide)
private theorem atomic_1366_agreement : RowAgreement 1365 1366 := RowAgreement.single (first := 1365) modelrow_agrees_1366
private def atomic_1367 : CountSegment 1 10 13120 1366 1367 735 735 := CountSegment.single (first := 1366) (initial := 735) row_count_1367 (by decide)
private theorem atomic_1367_agreement : RowAgreement 1366 1367 := RowAgreement.single (first := 1366) modelrow_agrees_1367
private def atomic_1368 : CountSegment 1 10 13120 1367 1368 735 738 := CountSegment.single (first := 1367) (initial := 735) row_count_1368 (by decide)
private theorem atomic_1368_agreement : RowAgreement 1367 1368 := RowAgreement.single (first := 1367) modelrow_agrees_1368
private def atomic_1369 : CountSegment 1 10 13120 1368 1369 738 738 := CountSegment.single (first := 1368) (initial := 738) row_count_1369 (by decide)
private theorem atomic_1369_agreement : RowAgreement 1368 1369 := RowAgreement.single (first := 1368) modelrow_agrees_1369
private def atomic_1370 : CountSegment 1 10 13120 1369 1370 738 738 := CountSegment.single (first := 1369) (initial := 738) row_count_1370 (by decide)
private theorem atomic_1370_agreement : RowAgreement 1369 1370 := RowAgreement.single (first := 1369) modelrow_agrees_1370
private def atomic_1371 : CountSegment 1 10 13120 1370 1371 738 739 := CountSegment.single (first := 1370) (initial := 738) row_count_1371 (by decide)
private theorem atomic_1371_agreement : RowAgreement 1370 1371 := RowAgreement.single (first := 1370) modelrow_agrees_1371
private def atomic_1372 : CountSegment 1 10 13120 1371 1372 739 739 := CountSegment.single (first := 1371) (initial := 739) row_count_1372 (by decide)
private theorem atomic_1372_agreement : RowAgreement 1371 1372 := RowAgreement.single (first := 1371) modelrow_agrees_1372
private def atomic_1373 : CountSegment 1 10 13120 1372 1373 739 740 := CountSegment.single (first := 1372) (initial := 739) row_count_1373 (by decide)
private theorem atomic_1373_agreement : RowAgreement 1372 1373 := RowAgreement.single (first := 1372) modelrow_agrees_1373
private def atomic_1374 : CountSegment 1 10 13120 1373 1374 740 740 := CountSegment.single (first := 1373) (initial := 740) row_count_1374 (by decide)
private theorem atomic_1374_agreement : RowAgreement 1373 1374 := RowAgreement.single (first := 1373) modelrow_agrees_1374
private def atomic_1375 : CountSegment 1 10 13120 1374 1375 740 740 := CountSegment.single (first := 1374) (initial := 740) row_count_1375 (by decide)
private theorem atomic_1375_agreement : RowAgreement 1374 1375 := RowAgreement.single (first := 1374) modelrow_agrees_1375
private def atomic_1376 : CountSegment 1 10 13120 1375 1376 740 740 := CountSegment.single (first := 1375) (initial := 740) row_count_1376 (by decide)
private theorem atomic_1376_agreement : RowAgreement 1375 1376 := RowAgreement.single (first := 1375) modelrow_agrees_1376
private def atomic_1377 : CountSegment 1 10 13120 1376 1377 740 741 := CountSegment.single (first := 1376) (initial := 740) row_count_1377 (by decide)
private theorem atomic_1377_agreement : RowAgreement 1376 1377 := RowAgreement.single (first := 1376) modelrow_agrees_1377
private def atomic_1378 : CountSegment 1 10 13120 1377 1378 741 741 := CountSegment.single (first := 1377) (initial := 741) row_count_1378 (by decide)
private theorem atomic_1378_agreement : RowAgreement 1377 1378 := RowAgreement.single (first := 1377) modelrow_agrees_1378
private def atomic_1379 : CountSegment 1 10 13120 1378 1379 741 741 := CountSegment.single (first := 1378) (initial := 741) row_count_1379 (by decide)
private theorem atomic_1379_agreement : RowAgreement 1378 1379 := RowAgreement.single (first := 1378) modelrow_agrees_1379
private def atomic_1380 : CountSegment 1 10 13120 1379 1380 741 742 := CountSegment.single (first := 1379) (initial := 741) row_count_1380 (by decide)
private theorem atomic_1380_agreement : RowAgreement 1379 1380 := RowAgreement.single (first := 1379) modelrow_agrees_1380
private def atomic_1381 : CountSegment 1 10 13120 1380 1381 742 742 := CountSegment.single (first := 1380) (initial := 742) row_count_1381 (by decide)
private theorem atomic_1381_agreement : RowAgreement 1380 1381 := RowAgreement.single (first := 1380) modelrow_agrees_1381
private def atomic_1382 : CountSegment 1 10 13120 1381 1382 742 742 := CountSegment.single (first := 1381) (initial := 742) row_count_1382 (by decide)
private theorem atomic_1382_agreement : RowAgreement 1381 1382 := RowAgreement.single (first := 1381) modelrow_agrees_1382
private def atomic_1383 : CountSegment 1 10 13120 1382 1383 742 743 := CountSegment.single (first := 1382) (initial := 742) row_count_1383 (by decide)
private theorem atomic_1383_agreement : RowAgreement 1382 1383 := RowAgreement.single (first := 1382) modelrow_agrees_1383
private def atomic_1384 : CountSegment 1 10 13120 1383 1384 743 743 := CountSegment.single (first := 1383) (initial := 743) row_count_1384 (by decide)
private theorem atomic_1384_agreement : RowAgreement 1383 1384 := RowAgreement.single (first := 1383) modelrow_agrees_1384
private def atomic_1385 : CountSegment 1 10 13120 1384 1385 743 744 := CountSegment.single (first := 1384) (initial := 743) row_count_1385 (by decide)
private theorem atomic_1385_agreement : RowAgreement 1384 1385 := RowAgreement.single (first := 1384) modelrow_agrees_1385
private def atomic_1386 : CountSegment 1 10 13120 1385 1386 744 745 := CountSegment.single (first := 1385) (initial := 744) row_count_1386 (by decide)
private theorem atomic_1386_agreement : RowAgreement 1385 1386 := RowAgreement.single (first := 1385) modelrow_agrees_1386
private def atomic_1387 : CountSegment 1 10 13120 1386 1387 745 745 := CountSegment.single (first := 1386) (initial := 745) row_count_1387 (by decide)
private theorem atomic_1387_agreement : RowAgreement 1386 1387 := RowAgreement.single (first := 1386) modelrow_agrees_1387
private def atomic_1388 : CountSegment 1 10 13120 1387 1388 745 746 := CountSegment.single (first := 1387) (initial := 745) row_count_1388 (by decide)
private theorem atomic_1388_agreement : RowAgreement 1387 1388 := RowAgreement.single (first := 1387) modelrow_agrees_1388
private def atomic_1389 : CountSegment 1 10 13120 1388 1389 746 746 := CountSegment.single (first := 1388) (initial := 746) row_count_1389 (by decide)
private theorem atomic_1389_agreement : RowAgreement 1388 1389 := RowAgreement.single (first := 1388) modelrow_agrees_1389
private def atomic_1390 : CountSegment 1 10 13120 1389 1390 746 747 := CountSegment.single (first := 1389) (initial := 746) row_count_1390 (by decide)
private theorem atomic_1390_agreement : RowAgreement 1389 1390 := RowAgreement.single (first := 1389) modelrow_agrees_1390
private def atomic_1391 : CountSegment 1 10 13120 1390 1391 747 747 := CountSegment.single (first := 1390) (initial := 747) row_count_1391 (by decide)
private theorem atomic_1391_agreement : RowAgreement 1390 1391 := RowAgreement.single (first := 1390) modelrow_agrees_1391
private def atomic_1392 : CountSegment 1 10 13120 1391 1392 747 747 := CountSegment.single (first := 1391) (initial := 747) row_count_1392 (by decide)
private theorem atomic_1392_agreement : RowAgreement 1391 1392 := RowAgreement.single (first := 1391) modelrow_agrees_1392
private def atomic_1393 : CountSegment 1 10 13120 1392 1393 747 747 := CountSegment.single (first := 1392) (initial := 747) row_count_1393 (by decide)
private theorem atomic_1393_agreement : RowAgreement 1392 1393 := RowAgreement.single (first := 1392) modelrow_agrees_1393
private def atomic_1394 : CountSegment 1 10 13120 1393 1394 747 748 := CountSegment.single (first := 1393) (initial := 747) row_count_1394 (by decide)
private theorem atomic_1394_agreement : RowAgreement 1393 1394 := RowAgreement.single (first := 1393) modelrow_agrees_1394
private def atomic_1395 : CountSegment 1 10 13120 1394 1395 748 749 := CountSegment.single (first := 1394) (initial := 748) row_count_1395 (by decide)
private theorem atomic_1395_agreement : RowAgreement 1394 1395 := RowAgreement.single (first := 1394) modelrow_agrees_1395
private def atomic_1396 : CountSegment 1 10 13120 1395 1396 749 749 := CountSegment.single (first := 1395) (initial := 749) row_count_1396 (by decide)
private theorem atomic_1396_agreement : RowAgreement 1395 1396 := RowAgreement.single (first := 1395) modelrow_agrees_1396
private def atomic_1397 : CountSegment 1 10 13120 1396 1397 749 751 := CountSegment.single (first := 1396) (initial := 749) row_count_1397 (by decide)
private theorem atomic_1397_agreement : RowAgreement 1396 1397 := RowAgreement.single (first := 1396) modelrow_agrees_1397
private def atomic_1398 : CountSegment 1 10 13120 1397 1398 751 751 := CountSegment.single (first := 1397) (initial := 751) row_count_1398 (by decide)
private theorem atomic_1398_agreement : RowAgreement 1397 1398 := RowAgreement.single (first := 1397) modelrow_agrees_1398
private def atomic_1399 : CountSegment 1 10 13120 1398 1399 751 751 := CountSegment.single (first := 1398) (initial := 751) row_count_1399 (by decide)
private theorem atomic_1399_agreement : RowAgreement 1398 1399 := RowAgreement.single (first := 1398) modelrow_agrees_1399
private def atomic_1400 : CountSegment 1 10 13120 1399 1400 751 751 := CountSegment.single (first := 1399) (initial := 751) row_count_1400 (by decide)
private theorem atomic_1400_agreement : RowAgreement 1399 1400 := RowAgreement.single (first := 1399) modelrow_agrees_1400
private def atomic_1401 : CountSegment 1 10 13120 1400 1401 751 751 := CountSegment.single (first := 1400) (initial := 751) row_count_1401 (by decide)
private theorem atomic_1401_agreement : RowAgreement 1400 1401 := RowAgreement.single (first := 1400) modelrow_agrees_1401
private def atomic_1402 : CountSegment 1 10 13120 1401 1402 751 752 := CountSegment.single (first := 1401) (initial := 751) row_count_1402 (by decide)
private theorem atomic_1402_agreement : RowAgreement 1401 1402 := RowAgreement.single (first := 1401) modelrow_agrees_1402
private def atomic_1403 : CountSegment 1 10 13120 1402 1403 752 752 := CountSegment.single (first := 1402) (initial := 752) row_count_1403 (by decide)
private theorem atomic_1403_agreement : RowAgreement 1402 1403 := RowAgreement.single (first := 1402) modelrow_agrees_1403
private def atomic_1404 : CountSegment 1 10 13120 1403 1404 752 752 := CountSegment.single (first := 1403) (initial := 752) row_count_1404 (by decide)
private theorem atomic_1404_agreement : RowAgreement 1403 1404 := RowAgreement.single (first := 1403) modelrow_agrees_1404
private def atomic_1405 : CountSegment 1 10 13120 1404 1405 752 752 := CountSegment.single (first := 1404) (initial := 752) row_count_1405 (by decide)
private theorem atomic_1405_agreement : RowAgreement 1404 1405 := RowAgreement.single (first := 1404) modelrow_agrees_1405
private def atomic_1406 : CountSegment 1 10 13120 1405 1406 752 752 := CountSegment.single (first := 1405) (initial := 752) row_count_1406 (by decide)
private theorem atomic_1406_agreement : RowAgreement 1405 1406 := RowAgreement.single (first := 1405) modelrow_agrees_1406
private def atomic_1407 : CountSegment 1 10 13120 1406 1407 752 753 := CountSegment.single (first := 1406) (initial := 752) row_count_1407 (by decide)
private theorem atomic_1407_agreement : RowAgreement 1406 1407 := RowAgreement.single (first := 1406) modelrow_agrees_1407
private def atomic_1408 : CountSegment 1 10 13120 1407 1408 753 753 := CountSegment.single (first := 1407) (initial := 753) row_count_1408 (by decide)
private theorem atomic_1408_agreement : RowAgreement 1407 1408 := RowAgreement.single (first := 1407) modelrow_agrees_1408
private def within_010_000 : CountSegment 1 10 13120 1280 1282 691 693 := CountSegment.join atomic_1281 atomic_1282
private theorem within_010_000_agreement : RowAgreement 1280 1282 := RowAgreement.join atomic_1281_agreement atomic_1282_agreement
private def within_010_001 : CountSegment 1 10 13120 1282 1284 693 694 := CountSegment.join atomic_1283 atomic_1284
private theorem within_010_001_agreement : RowAgreement 1282 1284 := RowAgreement.join atomic_1283_agreement atomic_1284_agreement
private def within_010_002 : CountSegment 1 10 13120 1280 1284 691 694 := CountSegment.join within_010_000 within_010_001
private theorem within_010_002_agreement : RowAgreement 1280 1284 := RowAgreement.join within_010_000_agreement within_010_001_agreement
private def within_010_003 : CountSegment 1 10 13120 1284 1286 694 695 := CountSegment.join atomic_1285 atomic_1286
private theorem within_010_003_agreement : RowAgreement 1284 1286 := RowAgreement.join atomic_1285_agreement atomic_1286_agreement
private def within_010_004 : CountSegment 1 10 13120 1286 1288 695 697 := CountSegment.join atomic_1287 atomic_1288
private theorem within_010_004_agreement : RowAgreement 1286 1288 := RowAgreement.join atomic_1287_agreement atomic_1288_agreement
private def within_010_005 : CountSegment 1 10 13120 1284 1288 694 697 := CountSegment.join within_010_003 within_010_004
private theorem within_010_005_agreement : RowAgreement 1284 1288 := RowAgreement.join within_010_003_agreement within_010_004_agreement
private def within_010_006 : CountSegment 1 10 13120 1280 1288 691 697 := CountSegment.join within_010_002 within_010_005
private theorem within_010_006_agreement : RowAgreement 1280 1288 := RowAgreement.join within_010_002_agreement within_010_005_agreement
private def within_010_007 : CountSegment 1 10 13120 1288 1290 697 697 := CountSegment.join atomic_1289 atomic_1290
private theorem within_010_007_agreement : RowAgreement 1288 1290 := RowAgreement.join atomic_1289_agreement atomic_1290_agreement
private def within_010_008 : CountSegment 1 10 13120 1290 1292 697 697 := CountSegment.join atomic_1291 atomic_1292
private theorem within_010_008_agreement : RowAgreement 1290 1292 := RowAgreement.join atomic_1291_agreement atomic_1292_agreement
private def within_010_009 : CountSegment 1 10 13120 1288 1292 697 697 := CountSegment.join within_010_007 within_010_008
private theorem within_010_009_agreement : RowAgreement 1288 1292 := RowAgreement.join within_010_007_agreement within_010_008_agreement
private def within_010_010 : CountSegment 1 10 13120 1292 1294 697 697 := CountSegment.join atomic_1293 atomic_1294
private theorem within_010_010_agreement : RowAgreement 1292 1294 := RowAgreement.join atomic_1293_agreement atomic_1294_agreement
private def within_010_011 : CountSegment 1 10 13120 1294 1296 697 698 := CountSegment.join atomic_1295 atomic_1296
private theorem within_010_011_agreement : RowAgreement 1294 1296 := RowAgreement.join atomic_1295_agreement atomic_1296_agreement
private def within_010_012 : CountSegment 1 10 13120 1292 1296 697 698 := CountSegment.join within_010_010 within_010_011
private theorem within_010_012_agreement : RowAgreement 1292 1296 := RowAgreement.join within_010_010_agreement within_010_011_agreement
private def within_010_013 : CountSegment 1 10 13120 1288 1296 697 698 := CountSegment.join within_010_009 within_010_012
private theorem within_010_013_agreement : RowAgreement 1288 1296 := RowAgreement.join within_010_009_agreement within_010_012_agreement
private def within_010_014 : CountSegment 1 10 13120 1280 1296 691 698 := CountSegment.join within_010_006 within_010_013
private theorem within_010_014_agreement : RowAgreement 1280 1296 := RowAgreement.join within_010_006_agreement within_010_013_agreement
private def within_010_015 : CountSegment 1 10 13120 1296 1298 698 699 := CountSegment.join atomic_1297 atomic_1298
private theorem within_010_015_agreement : RowAgreement 1296 1298 := RowAgreement.join atomic_1297_agreement atomic_1298_agreement
private def within_010_016 : CountSegment 1 10 13120 1298 1300 699 701 := CountSegment.join atomic_1299 atomic_1300
private theorem within_010_016_agreement : RowAgreement 1298 1300 := RowAgreement.join atomic_1299_agreement atomic_1300_agreement
private def within_010_017 : CountSegment 1 10 13120 1296 1300 698 701 := CountSegment.join within_010_015 within_010_016
private theorem within_010_017_agreement : RowAgreement 1296 1300 := RowAgreement.join within_010_015_agreement within_010_016_agreement
private def within_010_018 : CountSegment 1 10 13120 1300 1302 701 703 := CountSegment.join atomic_1301 atomic_1302
private theorem within_010_018_agreement : RowAgreement 1300 1302 := RowAgreement.join atomic_1301_agreement atomic_1302_agreement
private def within_010_019 : CountSegment 1 10 13120 1302 1304 703 704 := CountSegment.join atomic_1303 atomic_1304
private theorem within_010_019_agreement : RowAgreement 1302 1304 := RowAgreement.join atomic_1303_agreement atomic_1304_agreement
private def within_010_020 : CountSegment 1 10 13120 1300 1304 701 704 := CountSegment.join within_010_018 within_010_019
private theorem within_010_020_agreement : RowAgreement 1300 1304 := RowAgreement.join within_010_018_agreement within_010_019_agreement
private def within_010_021 : CountSegment 1 10 13120 1296 1304 698 704 := CountSegment.join within_010_017 within_010_020
private theorem within_010_021_agreement : RowAgreement 1296 1304 := RowAgreement.join within_010_017_agreement within_010_020_agreement
private def within_010_022 : CountSegment 1 10 13120 1304 1306 704 704 := CountSegment.join atomic_1305 atomic_1306
private theorem within_010_022_agreement : RowAgreement 1304 1306 := RowAgreement.join atomic_1305_agreement atomic_1306_agreement
private def within_010_023 : CountSegment 1 10 13120 1306 1308 704 704 := CountSegment.join atomic_1307 atomic_1308
private theorem within_010_023_agreement : RowAgreement 1306 1308 := RowAgreement.join atomic_1307_agreement atomic_1308_agreement
private def within_010_024 : CountSegment 1 10 13120 1304 1308 704 704 := CountSegment.join within_010_022 within_010_023
private theorem within_010_024_agreement : RowAgreement 1304 1308 := RowAgreement.join within_010_022_agreement within_010_023_agreement
private def within_010_025 : CountSegment 1 10 13120 1308 1310 704 705 := CountSegment.join atomic_1309 atomic_1310
private theorem within_010_025_agreement : RowAgreement 1308 1310 := RowAgreement.join atomic_1309_agreement atomic_1310_agreement
private def within_010_026 : CountSegment 1 10 13120 1310 1312 705 706 := CountSegment.join atomic_1311 atomic_1312
private theorem within_010_026_agreement : RowAgreement 1310 1312 := RowAgreement.join atomic_1311_agreement atomic_1312_agreement
private def within_010_027 : CountSegment 1 10 13120 1308 1312 704 706 := CountSegment.join within_010_025 within_010_026
private theorem within_010_027_agreement : RowAgreement 1308 1312 := RowAgreement.join within_010_025_agreement within_010_026_agreement
private def within_010_028 : CountSegment 1 10 13120 1304 1312 704 706 := CountSegment.join within_010_024 within_010_027
private theorem within_010_028_agreement : RowAgreement 1304 1312 := RowAgreement.join within_010_024_agreement within_010_027_agreement
private def within_010_029 : CountSegment 1 10 13120 1296 1312 698 706 := CountSegment.join within_010_021 within_010_028
private theorem within_010_029_agreement : RowAgreement 1296 1312 := RowAgreement.join within_010_021_agreement within_010_028_agreement
private def within_010_030 : CountSegment 1 10 13120 1280 1312 691 706 := CountSegment.join within_010_014 within_010_029
private theorem within_010_030_agreement : RowAgreement 1280 1312 := RowAgreement.join within_010_014_agreement within_010_029_agreement
private def within_010_031 : CountSegment 1 10 13120 1312 1314 706 708 := CountSegment.join atomic_1313 atomic_1314
private theorem within_010_031_agreement : RowAgreement 1312 1314 := RowAgreement.join atomic_1313_agreement atomic_1314_agreement
private def within_010_032 : CountSegment 1 10 13120 1314 1316 708 708 := CountSegment.join atomic_1315 atomic_1316
private theorem within_010_032_agreement : RowAgreement 1314 1316 := RowAgreement.join atomic_1315_agreement atomic_1316_agreement
private def within_010_033 : CountSegment 1 10 13120 1312 1316 706 708 := CountSegment.join within_010_031 within_010_032
private theorem within_010_033_agreement : RowAgreement 1312 1316 := RowAgreement.join within_010_031_agreement within_010_032_agreement
private def within_010_034 : CountSegment 1 10 13120 1316 1318 708 708 := CountSegment.join atomic_1317 atomic_1318
private theorem within_010_034_agreement : RowAgreement 1316 1318 := RowAgreement.join atomic_1317_agreement atomic_1318_agreement
private def within_010_035 : CountSegment 1 10 13120 1318 1320 708 709 := CountSegment.join atomic_1319 atomic_1320
private theorem within_010_035_agreement : RowAgreement 1318 1320 := RowAgreement.join atomic_1319_agreement atomic_1320_agreement
private def within_010_036 : CountSegment 1 10 13120 1316 1320 708 709 := CountSegment.join within_010_034 within_010_035
private theorem within_010_036_agreement : RowAgreement 1316 1320 := RowAgreement.join within_010_034_agreement within_010_035_agreement
private def within_010_037 : CountSegment 1 10 13120 1312 1320 706 709 := CountSegment.join within_010_033 within_010_036
private theorem within_010_037_agreement : RowAgreement 1312 1320 := RowAgreement.join within_010_033_agreement within_010_036_agreement
private def within_010_038 : CountSegment 1 10 13120 1320 1322 709 709 := CountSegment.join atomic_1321 atomic_1322
private theorem within_010_038_agreement : RowAgreement 1320 1322 := RowAgreement.join atomic_1321_agreement atomic_1322_agreement
private def within_010_039 : CountSegment 1 10 13120 1322 1324 709 711 := CountSegment.join atomic_1323 atomic_1324
private theorem within_010_039_agreement : RowAgreement 1322 1324 := RowAgreement.join atomic_1323_agreement atomic_1324_agreement
private def within_010_040 : CountSegment 1 10 13120 1320 1324 709 711 := CountSegment.join within_010_038 within_010_039
private theorem within_010_040_agreement : RowAgreement 1320 1324 := RowAgreement.join within_010_038_agreement within_010_039_agreement
private def within_010_041 : CountSegment 1 10 13120 1324 1326 711 713 := CountSegment.join atomic_1325 atomic_1326
private theorem within_010_041_agreement : RowAgreement 1324 1326 := RowAgreement.join atomic_1325_agreement atomic_1326_agreement
private def within_010_042 : CountSegment 1 10 13120 1326 1328 713 714 := CountSegment.join atomic_1327 atomic_1328
private theorem within_010_042_agreement : RowAgreement 1326 1328 := RowAgreement.join atomic_1327_agreement atomic_1328_agreement
private def within_010_043 : CountSegment 1 10 13120 1324 1328 711 714 := CountSegment.join within_010_041 within_010_042
private theorem within_010_043_agreement : RowAgreement 1324 1328 := RowAgreement.join within_010_041_agreement within_010_042_agreement
private def within_010_044 : CountSegment 1 10 13120 1320 1328 709 714 := CountSegment.join within_010_040 within_010_043
private theorem within_010_044_agreement : RowAgreement 1320 1328 := RowAgreement.join within_010_040_agreement within_010_043_agreement
private def within_010_045 : CountSegment 1 10 13120 1312 1328 706 714 := CountSegment.join within_010_037 within_010_044
private theorem within_010_045_agreement : RowAgreement 1312 1328 := RowAgreement.join within_010_037_agreement within_010_044_agreement
private def within_010_046 : CountSegment 1 10 13120 1328 1330 714 715 := CountSegment.join atomic_1329 atomic_1330
private theorem within_010_046_agreement : RowAgreement 1328 1330 := RowAgreement.join atomic_1329_agreement atomic_1330_agreement
private def within_010_047 : CountSegment 1 10 13120 1330 1332 715 717 := CountSegment.join atomic_1331 atomic_1332
private theorem within_010_047_agreement : RowAgreement 1330 1332 := RowAgreement.join atomic_1331_agreement atomic_1332_agreement
private def within_010_048 : CountSegment 1 10 13120 1328 1332 714 717 := CountSegment.join within_010_046 within_010_047
private theorem within_010_048_agreement : RowAgreement 1328 1332 := RowAgreement.join within_010_046_agreement within_010_047_agreement
private def within_010_049 : CountSegment 1 10 13120 1332 1334 717 718 := CountSegment.join atomic_1333 atomic_1334
private theorem within_010_049_agreement : RowAgreement 1332 1334 := RowAgreement.join atomic_1333_agreement atomic_1334_agreement
private def within_010_050 : CountSegment 1 10 13120 1334 1336 718 718 := CountSegment.join atomic_1335 atomic_1336
private theorem within_010_050_agreement : RowAgreement 1334 1336 := RowAgreement.join atomic_1335_agreement atomic_1336_agreement
private def within_010_051 : CountSegment 1 10 13120 1332 1336 717 718 := CountSegment.join within_010_049 within_010_050
private theorem within_010_051_agreement : RowAgreement 1332 1336 := RowAgreement.join within_010_049_agreement within_010_050_agreement
private def within_010_052 : CountSegment 1 10 13120 1328 1336 714 718 := CountSegment.join within_010_048 within_010_051
private theorem within_010_052_agreement : RowAgreement 1328 1336 := RowAgreement.join within_010_048_agreement within_010_051_agreement
private def within_010_053 : CountSegment 1 10 13120 1336 1338 718 720 := CountSegment.join atomic_1337 atomic_1338
private theorem within_010_053_agreement : RowAgreement 1336 1338 := RowAgreement.join atomic_1337_agreement atomic_1338_agreement
private def within_010_054 : CountSegment 1 10 13120 1338 1340 720 720 := CountSegment.join atomic_1339 atomic_1340
private theorem within_010_054_agreement : RowAgreement 1338 1340 := RowAgreement.join atomic_1339_agreement atomic_1340_agreement
private def within_010_055 : CountSegment 1 10 13120 1336 1340 718 720 := CountSegment.join within_010_053 within_010_054
private theorem within_010_055_agreement : RowAgreement 1336 1340 := RowAgreement.join within_010_053_agreement within_010_054_agreement
private def within_010_056 : CountSegment 1 10 13120 1340 1342 720 723 := CountSegment.join atomic_1341 atomic_1342
private theorem within_010_056_agreement : RowAgreement 1340 1342 := RowAgreement.join atomic_1341_agreement atomic_1342_agreement
private def within_010_057 : CountSegment 1 10 13120 1342 1344 723 724 := CountSegment.join atomic_1343 atomic_1344
private theorem within_010_057_agreement : RowAgreement 1342 1344 := RowAgreement.join atomic_1343_agreement atomic_1344_agreement
private def within_010_058 : CountSegment 1 10 13120 1340 1344 720 724 := CountSegment.join within_010_056 within_010_057
private theorem within_010_058_agreement : RowAgreement 1340 1344 := RowAgreement.join within_010_056_agreement within_010_057_agreement
private def within_010_059 : CountSegment 1 10 13120 1336 1344 718 724 := CountSegment.join within_010_055 within_010_058
private theorem within_010_059_agreement : RowAgreement 1336 1344 := RowAgreement.join within_010_055_agreement within_010_058_agreement
private def within_010_060 : CountSegment 1 10 13120 1328 1344 714 724 := CountSegment.join within_010_052 within_010_059
private theorem within_010_060_agreement : RowAgreement 1328 1344 := RowAgreement.join within_010_052_agreement within_010_059_agreement
private def within_010_061 : CountSegment 1 10 13120 1312 1344 706 724 := CountSegment.join within_010_045 within_010_060
private theorem within_010_061_agreement : RowAgreement 1312 1344 := RowAgreement.join within_010_045_agreement within_010_060_agreement
private def within_010_062 : CountSegment 1 10 13120 1280 1344 691 724 := CountSegment.join within_010_030 within_010_061
private theorem within_010_062_agreement : RowAgreement 1280 1344 := RowAgreement.join within_010_030_agreement within_010_061_agreement
private def within_010_063 : CountSegment 1 10 13120 1344 1346 724 724 := CountSegment.join atomic_1345 atomic_1346
private theorem within_010_063_agreement : RowAgreement 1344 1346 := RowAgreement.join atomic_1345_agreement atomic_1346_agreement
private def within_010_064 : CountSegment 1 10 13120 1346 1348 724 725 := CountSegment.join atomic_1347 atomic_1348
private theorem within_010_064_agreement : RowAgreement 1346 1348 := RowAgreement.join atomic_1347_agreement atomic_1348_agreement
private def within_010_065 : CountSegment 1 10 13120 1344 1348 724 725 := CountSegment.join within_010_063 within_010_064
private theorem within_010_065_agreement : RowAgreement 1344 1348 := RowAgreement.join within_010_063_agreement within_010_064_agreement
private def within_010_066 : CountSegment 1 10 13120 1348 1350 725 726 := CountSegment.join atomic_1349 atomic_1350
private theorem within_010_066_agreement : RowAgreement 1348 1350 := RowAgreement.join atomic_1349_agreement atomic_1350_agreement
private def within_010_067 : CountSegment 1 10 13120 1350 1352 726 727 := CountSegment.join atomic_1351 atomic_1352
private theorem within_010_067_agreement : RowAgreement 1350 1352 := RowAgreement.join atomic_1351_agreement atomic_1352_agreement
private def within_010_068 : CountSegment 1 10 13120 1348 1352 725 727 := CountSegment.join within_010_066 within_010_067
private theorem within_010_068_agreement : RowAgreement 1348 1352 := RowAgreement.join within_010_066_agreement within_010_067_agreement
private def within_010_069 : CountSegment 1 10 13120 1344 1352 724 727 := CountSegment.join within_010_065 within_010_068
private theorem within_010_069_agreement : RowAgreement 1344 1352 := RowAgreement.join within_010_065_agreement within_010_068_agreement
private def within_010_070 : CountSegment 1 10 13120 1352 1354 727 729 := CountSegment.join atomic_1353 atomic_1354
private theorem within_010_070_agreement : RowAgreement 1352 1354 := RowAgreement.join atomic_1353_agreement atomic_1354_agreement
private def within_010_071 : CountSegment 1 10 13120 1354 1356 729 729 := CountSegment.join atomic_1355 atomic_1356
private theorem within_010_071_agreement : RowAgreement 1354 1356 := RowAgreement.join atomic_1355_agreement atomic_1356_agreement
private def within_010_072 : CountSegment 1 10 13120 1352 1356 727 729 := CountSegment.join within_010_070 within_010_071
private theorem within_010_072_agreement : RowAgreement 1352 1356 := RowAgreement.join within_010_070_agreement within_010_071_agreement
private def within_010_073 : CountSegment 1 10 13120 1356 1358 729 730 := CountSegment.join atomic_1357 atomic_1358
private theorem within_010_073_agreement : RowAgreement 1356 1358 := RowAgreement.join atomic_1357_agreement atomic_1358_agreement
private def within_010_074 : CountSegment 1 10 13120 1358 1360 730 730 := CountSegment.join atomic_1359 atomic_1360
private theorem within_010_074_agreement : RowAgreement 1358 1360 := RowAgreement.join atomic_1359_agreement atomic_1360_agreement
private def within_010_075 : CountSegment 1 10 13120 1356 1360 729 730 := CountSegment.join within_010_073 within_010_074
private theorem within_010_075_agreement : RowAgreement 1356 1360 := RowAgreement.join within_010_073_agreement within_010_074_agreement
private def within_010_076 : CountSegment 1 10 13120 1352 1360 727 730 := CountSegment.join within_010_072 within_010_075
private theorem within_010_076_agreement : RowAgreement 1352 1360 := RowAgreement.join within_010_072_agreement within_010_075_agreement
private def within_010_077 : CountSegment 1 10 13120 1344 1360 724 730 := CountSegment.join within_010_069 within_010_076
private theorem within_010_077_agreement : RowAgreement 1344 1360 := RowAgreement.join within_010_069_agreement within_010_076_agreement
private def within_010_078 : CountSegment 1 10 13120 1360 1362 730 732 := CountSegment.join atomic_1361 atomic_1362
private theorem within_010_078_agreement : RowAgreement 1360 1362 := RowAgreement.join atomic_1361_agreement atomic_1362_agreement
private def within_010_079 : CountSegment 1 10 13120 1362 1364 732 732 := CountSegment.join atomic_1363 atomic_1364
private theorem within_010_079_agreement : RowAgreement 1362 1364 := RowAgreement.join atomic_1363_agreement atomic_1364_agreement
private def within_010_080 : CountSegment 1 10 13120 1360 1364 730 732 := CountSegment.join within_010_078 within_010_079
private theorem within_010_080_agreement : RowAgreement 1360 1364 := RowAgreement.join within_010_078_agreement within_010_079_agreement
private def within_010_081 : CountSegment 1 10 13120 1364 1366 732 735 := CountSegment.join atomic_1365 atomic_1366
private theorem within_010_081_agreement : RowAgreement 1364 1366 := RowAgreement.join atomic_1365_agreement atomic_1366_agreement
private def within_010_082 : CountSegment 1 10 13120 1366 1368 735 738 := CountSegment.join atomic_1367 atomic_1368
private theorem within_010_082_agreement : RowAgreement 1366 1368 := RowAgreement.join atomic_1367_agreement atomic_1368_agreement
private def within_010_083 : CountSegment 1 10 13120 1364 1368 732 738 := CountSegment.join within_010_081 within_010_082
private theorem within_010_083_agreement : RowAgreement 1364 1368 := RowAgreement.join within_010_081_agreement within_010_082_agreement
private def within_010_084 : CountSegment 1 10 13120 1360 1368 730 738 := CountSegment.join within_010_080 within_010_083
private theorem within_010_084_agreement : RowAgreement 1360 1368 := RowAgreement.join within_010_080_agreement within_010_083_agreement
private def within_010_085 : CountSegment 1 10 13120 1368 1370 738 738 := CountSegment.join atomic_1369 atomic_1370
private theorem within_010_085_agreement : RowAgreement 1368 1370 := RowAgreement.join atomic_1369_agreement atomic_1370_agreement
private def within_010_086 : CountSegment 1 10 13120 1370 1372 738 739 := CountSegment.join atomic_1371 atomic_1372
private theorem within_010_086_agreement : RowAgreement 1370 1372 := RowAgreement.join atomic_1371_agreement atomic_1372_agreement
private def within_010_087 : CountSegment 1 10 13120 1368 1372 738 739 := CountSegment.join within_010_085 within_010_086
private theorem within_010_087_agreement : RowAgreement 1368 1372 := RowAgreement.join within_010_085_agreement within_010_086_agreement
private def within_010_088 : CountSegment 1 10 13120 1372 1374 739 740 := CountSegment.join atomic_1373 atomic_1374
private theorem within_010_088_agreement : RowAgreement 1372 1374 := RowAgreement.join atomic_1373_agreement atomic_1374_agreement
private def within_010_089 : CountSegment 1 10 13120 1374 1376 740 740 := CountSegment.join atomic_1375 atomic_1376
private theorem within_010_089_agreement : RowAgreement 1374 1376 := RowAgreement.join atomic_1375_agreement atomic_1376_agreement
private def within_010_090 : CountSegment 1 10 13120 1372 1376 739 740 := CountSegment.join within_010_088 within_010_089
private theorem within_010_090_agreement : RowAgreement 1372 1376 := RowAgreement.join within_010_088_agreement within_010_089_agreement
private def within_010_091 : CountSegment 1 10 13120 1368 1376 738 740 := CountSegment.join within_010_087 within_010_090
private theorem within_010_091_agreement : RowAgreement 1368 1376 := RowAgreement.join within_010_087_agreement within_010_090_agreement
private def within_010_092 : CountSegment 1 10 13120 1360 1376 730 740 := CountSegment.join within_010_084 within_010_091
private theorem within_010_092_agreement : RowAgreement 1360 1376 := RowAgreement.join within_010_084_agreement within_010_091_agreement
private def within_010_093 : CountSegment 1 10 13120 1344 1376 724 740 := CountSegment.join within_010_077 within_010_092
private theorem within_010_093_agreement : RowAgreement 1344 1376 := RowAgreement.join within_010_077_agreement within_010_092_agreement
private def within_010_094 : CountSegment 1 10 13120 1376 1378 740 741 := CountSegment.join atomic_1377 atomic_1378
private theorem within_010_094_agreement : RowAgreement 1376 1378 := RowAgreement.join atomic_1377_agreement atomic_1378_agreement
private def within_010_095 : CountSegment 1 10 13120 1378 1380 741 742 := CountSegment.join atomic_1379 atomic_1380
private theorem within_010_095_agreement : RowAgreement 1378 1380 := RowAgreement.join atomic_1379_agreement atomic_1380_agreement
private def within_010_096 : CountSegment 1 10 13120 1376 1380 740 742 := CountSegment.join within_010_094 within_010_095
private theorem within_010_096_agreement : RowAgreement 1376 1380 := RowAgreement.join within_010_094_agreement within_010_095_agreement
private def within_010_097 : CountSegment 1 10 13120 1380 1382 742 742 := CountSegment.join atomic_1381 atomic_1382
private theorem within_010_097_agreement : RowAgreement 1380 1382 := RowAgreement.join atomic_1381_agreement atomic_1382_agreement
private def within_010_098 : CountSegment 1 10 13120 1382 1384 742 743 := CountSegment.join atomic_1383 atomic_1384
private theorem within_010_098_agreement : RowAgreement 1382 1384 := RowAgreement.join atomic_1383_agreement atomic_1384_agreement
private def within_010_099 : CountSegment 1 10 13120 1380 1384 742 743 := CountSegment.join within_010_097 within_010_098
private theorem within_010_099_agreement : RowAgreement 1380 1384 := RowAgreement.join within_010_097_agreement within_010_098_agreement
private def within_010_100 : CountSegment 1 10 13120 1376 1384 740 743 := CountSegment.join within_010_096 within_010_099
private theorem within_010_100_agreement : RowAgreement 1376 1384 := RowAgreement.join within_010_096_agreement within_010_099_agreement
private def within_010_101 : CountSegment 1 10 13120 1384 1386 743 745 := CountSegment.join atomic_1385 atomic_1386
private theorem within_010_101_agreement : RowAgreement 1384 1386 := RowAgreement.join atomic_1385_agreement atomic_1386_agreement
private def within_010_102 : CountSegment 1 10 13120 1386 1388 745 746 := CountSegment.join atomic_1387 atomic_1388
private theorem within_010_102_agreement : RowAgreement 1386 1388 := RowAgreement.join atomic_1387_agreement atomic_1388_agreement
private def within_010_103 : CountSegment 1 10 13120 1384 1388 743 746 := CountSegment.join within_010_101 within_010_102
private theorem within_010_103_agreement : RowAgreement 1384 1388 := RowAgreement.join within_010_101_agreement within_010_102_agreement
private def within_010_104 : CountSegment 1 10 13120 1388 1390 746 747 := CountSegment.join atomic_1389 atomic_1390
private theorem within_010_104_agreement : RowAgreement 1388 1390 := RowAgreement.join atomic_1389_agreement atomic_1390_agreement
private def within_010_105 : CountSegment 1 10 13120 1390 1392 747 747 := CountSegment.join atomic_1391 atomic_1392
private theorem within_010_105_agreement : RowAgreement 1390 1392 := RowAgreement.join atomic_1391_agreement atomic_1392_agreement
private def within_010_106 : CountSegment 1 10 13120 1388 1392 746 747 := CountSegment.join within_010_104 within_010_105
private theorem within_010_106_agreement : RowAgreement 1388 1392 := RowAgreement.join within_010_104_agreement within_010_105_agreement
private def within_010_107 : CountSegment 1 10 13120 1384 1392 743 747 := CountSegment.join within_010_103 within_010_106
private theorem within_010_107_agreement : RowAgreement 1384 1392 := RowAgreement.join within_010_103_agreement within_010_106_agreement
private def within_010_108 : CountSegment 1 10 13120 1376 1392 740 747 := CountSegment.join within_010_100 within_010_107
private theorem within_010_108_agreement : RowAgreement 1376 1392 := RowAgreement.join within_010_100_agreement within_010_107_agreement
private def within_010_109 : CountSegment 1 10 13120 1392 1394 747 748 := CountSegment.join atomic_1393 atomic_1394
private theorem within_010_109_agreement : RowAgreement 1392 1394 := RowAgreement.join atomic_1393_agreement atomic_1394_agreement
private def within_010_110 : CountSegment 1 10 13120 1394 1396 748 749 := CountSegment.join atomic_1395 atomic_1396
private theorem within_010_110_agreement : RowAgreement 1394 1396 := RowAgreement.join atomic_1395_agreement atomic_1396_agreement
private def within_010_111 : CountSegment 1 10 13120 1392 1396 747 749 := CountSegment.join within_010_109 within_010_110
private theorem within_010_111_agreement : RowAgreement 1392 1396 := RowAgreement.join within_010_109_agreement within_010_110_agreement
private def within_010_112 : CountSegment 1 10 13120 1396 1398 749 751 := CountSegment.join atomic_1397 atomic_1398
private theorem within_010_112_agreement : RowAgreement 1396 1398 := RowAgreement.join atomic_1397_agreement atomic_1398_agreement
private def within_010_113 : CountSegment 1 10 13120 1398 1400 751 751 := CountSegment.join atomic_1399 atomic_1400
private theorem within_010_113_agreement : RowAgreement 1398 1400 := RowAgreement.join atomic_1399_agreement atomic_1400_agreement
private def within_010_114 : CountSegment 1 10 13120 1396 1400 749 751 := CountSegment.join within_010_112 within_010_113
private theorem within_010_114_agreement : RowAgreement 1396 1400 := RowAgreement.join within_010_112_agreement within_010_113_agreement
private def within_010_115 : CountSegment 1 10 13120 1392 1400 747 751 := CountSegment.join within_010_111 within_010_114
private theorem within_010_115_agreement : RowAgreement 1392 1400 := RowAgreement.join within_010_111_agreement within_010_114_agreement
private def within_010_116 : CountSegment 1 10 13120 1400 1402 751 752 := CountSegment.join atomic_1401 atomic_1402
private theorem within_010_116_agreement : RowAgreement 1400 1402 := RowAgreement.join atomic_1401_agreement atomic_1402_agreement
private def within_010_117 : CountSegment 1 10 13120 1402 1404 752 752 := CountSegment.join atomic_1403 atomic_1404
private theorem within_010_117_agreement : RowAgreement 1402 1404 := RowAgreement.join atomic_1403_agreement atomic_1404_agreement
private def within_010_118 : CountSegment 1 10 13120 1400 1404 751 752 := CountSegment.join within_010_116 within_010_117
private theorem within_010_118_agreement : RowAgreement 1400 1404 := RowAgreement.join within_010_116_agreement within_010_117_agreement
private def within_010_119 : CountSegment 1 10 13120 1404 1406 752 752 := CountSegment.join atomic_1405 atomic_1406
private theorem within_010_119_agreement : RowAgreement 1404 1406 := RowAgreement.join atomic_1405_agreement atomic_1406_agreement
private def within_010_120 : CountSegment 1 10 13120 1406 1408 752 753 := CountSegment.join atomic_1407 atomic_1408
private theorem within_010_120_agreement : RowAgreement 1406 1408 := RowAgreement.join atomic_1407_agreement atomic_1408_agreement
private def within_010_121 : CountSegment 1 10 13120 1404 1408 752 753 := CountSegment.join within_010_119 within_010_120
private theorem within_010_121_agreement : RowAgreement 1404 1408 := RowAgreement.join within_010_119_agreement within_010_120_agreement
private def within_010_122 : CountSegment 1 10 13120 1400 1408 751 753 := CountSegment.join within_010_118 within_010_121
private theorem within_010_122_agreement : RowAgreement 1400 1408 := RowAgreement.join within_010_118_agreement within_010_121_agreement
private def within_010_123 : CountSegment 1 10 13120 1392 1408 747 753 := CountSegment.join within_010_115 within_010_122
private theorem within_010_123_agreement : RowAgreement 1392 1408 := RowAgreement.join within_010_115_agreement within_010_122_agreement
private def within_010_124 : CountSegment 1 10 13120 1376 1408 740 753 := CountSegment.join within_010_108 within_010_123
private theorem within_010_124_agreement : RowAgreement 1376 1408 := RowAgreement.join within_010_108_agreement within_010_123_agreement
private def within_010_125 : CountSegment 1 10 13120 1344 1408 724 753 := CountSegment.join within_010_093 within_010_124
private theorem within_010_125_agreement : RowAgreement 1344 1408 := RowAgreement.join within_010_093_agreement within_010_124_agreement
private def within_010_126 : CountSegment 1 10 13120 1280 1408 691 753 := CountSegment.join within_010_062 within_010_125
private theorem within_010_126_agreement : RowAgreement 1280 1408 := RowAgreement.join within_010_062_agreement within_010_125_agreement
def segment_010 : CountSegment 1 10 13120 1280 1408 691 753 := within_010_126
theorem agreement_010 : RowAgreement 1280 1408 := within_010_126_agreement
end ProjectEulerSolutions.P58Numerical
