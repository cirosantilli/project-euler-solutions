import ProjectEulerSolutions.Lib.PrimePairSearch.SmallP3Chunk002
import ProjectEulerSolutions.Lib.CliqueSearch

/-! Generated scan steps for the first small-domain branch. -/
namespace ProjectEulerSolutions.PrimePairSearch
open PrimePairCertificate CliqueSearch

def smallTail28 : List Nat := 739 :: smallTail29
def smallCandidates28 : List Nat := [6793]
set_option maxRecDepth 2048 in
theorem smallCandidates28_checked : candidates relation 4 3 739 smallTail29 smallBest.bound = smallCandidates28 := by decide

theorem smallScan28 : scan relation (search relation 3) 4 [3] 3 smallTail28 smallBest = smallBest := by
  rw [smallTail28, scan, if_neg (by decide), smallCandidates28_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail29
    (if 3 ≤ smallCandidates28.length then search relation 3 [3, 739] smallCandidates28 742 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan29

def smallTail27 : List Nat := 733 :: smallTail28
def smallCandidates27 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates27_checked : candidates relation 4 3 733 smallTail28 smallBest.bound = smallCandidates27 := by decide

theorem smallScan27 : scan relation (search relation 3) 4 [3] 3 smallTail27 smallBest = smallBest := by
  rw [smallTail27, scan, if_neg (by decide), smallCandidates27_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail28
    (if 3 ≤ smallCandidates27.length then search relation 3 [3, 733] smallCandidates27 736 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan28

def smallTail26 : List Nat := 719 :: smallTail27
def smallCandidates26 : List Nat := [947, 2579]
set_option maxRecDepth 2048 in
theorem smallCandidates26_checked : candidates relation 4 3 719 smallTail27 smallBest.bound = smallCandidates26 := by decide

theorem smallScan26 : scan relation (search relation 3) 4 [3] 3 smallTail26 smallBest = smallBest := by
  rw [smallTail26, scan, if_neg (by decide), smallCandidates26_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail27
    (if 3 ≤ smallCandidates26.length then search relation 3 [3, 719] smallCandidates26 722 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan27

def smallTail25 : List Nat := 701 :: smallTail26
def smallCandidates25 : List Nat := [1181, 1607, 4583]
set_option maxRecDepth 2048 in
theorem smallCandidates25_checked : candidates relation 4 3 701 smallTail26 smallBest.bound = smallCandidates25 := by decide

set_option maxRecDepth 2048 in
theorem smallChild25 : search relation 3 [3, 701] smallCandidates25 704 smallBest = smallBest := by decide

theorem smallScan25 : scan relation (search relation 3) 4 [3] 3 smallTail25 smallBest = smallBest := by
  rw [smallTail25, scan, if_neg (by decide), smallCandidates25_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail26
    (if 3 ≤ smallCandidates25.length then search relation 3 [3, 701] smallCandidates25 704 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild25]
  exact smallScan26

def smallTail24 : List Nat := 673 :: smallTail25
def smallCandidates24 : List Nat := [1657]
set_option maxRecDepth 2048 in
theorem smallCandidates24_checked : candidates relation 4 3 673 smallTail25 smallBest.bound = smallCandidates24 := by decide

theorem smallScan24 : scan relation (search relation 3) 4 [3] 3 smallTail24 smallBest = smallBest := by
  rw [smallTail24, scan, if_neg (by decide), smallCandidates24_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail25
    (if 3 ≤ smallCandidates24.length then search relation 3 [3, 673] smallCandidates24 676 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan25

def smallTail23 : List Nat := 617 :: smallTail24
def smallCandidates23 : List Nat := [4253]
set_option maxRecDepth 2048 in
theorem smallCandidates23_checked : candidates relation 4 3 617 smallTail24 smallBest.bound = smallCandidates23 := by decide

theorem smallScan23 : scan relation (search relation 3) 4 [3] 3 smallTail23 smallBest = smallBest := by
  rw [smallTail23, scan, if_neg (by decide), smallCandidates23_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail24
    (if 3 ≤ smallCandidates23.length then search relation 3 [3, 617] smallCandidates23 620 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan24

def smallTail22 : List Nat := 613 :: smallTail23
def smallCandidates22 : List Nat := [673, 3769]
set_option maxRecDepth 2048 in
theorem smallCandidates22_checked : candidates relation 4 3 613 smallTail23 smallBest.bound = smallCandidates22 := by decide

theorem smallScan22 : scan relation (search relation 3) 4 [3] 3 smallTail22 smallBest = smallBest := by
  rw [smallTail22, scan, if_neg (by decide), smallCandidates22_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail23
    (if 3 ≤ smallCandidates22.length then search relation 3 [3, 613] smallCandidates22 616 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan23

def smallTail21 : List Nat := 607 :: smallTail22
def smallCandidates21 : List Nat := [7039]
set_option maxRecDepth 2048 in
theorem smallCandidates21_checked : candidates relation 4 3 607 smallTail22 smallBest.bound = smallCandidates21 := by decide

theorem smallScan21 : scan relation (search relation 3) 4 [3] 3 smallTail21 smallBest = smallBest := by
  rw [smallTail21, scan, if_neg (by decide), smallCandidates21_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail22
    (if 3 ≤ smallCandidates21.length then search relation 3 [3, 607] smallCandidates21 610 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan22

def smallTail20 : List Nat := 557 :: smallTail21
def smallCandidates20 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates20_checked : candidates relation 4 3 557 smallTail21 smallBest.bound = smallCandidates20 := by decide

theorem smallScan20 : scan relation (search relation 3) 4 [3] 3 smallTail20 smallBest = smallBest := by
  rw [smallTail20, scan, if_neg (by decide), smallCandidates20_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail21
    (if 3 ≤ smallCandidates20.length then search relation 3 [3, 557] smallCandidates20 560 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan21

def smallTail19 : List Nat := 541 :: smallTail20
def smallCandidates19 : List Nat := [1033, 3301, 4159, 7369]
set_option maxRecDepth 2048 in
theorem smallCandidates19_checked : candidates relation 4 3 541 smallTail20 smallBest.bound = smallCandidates19 := by decide

set_option maxRecDepth 2048 in
theorem smallChild19 : search relation 3 [3, 541] smallCandidates19 544 smallBest = smallBest := by decide

theorem smallScan19 : scan relation (search relation 3) 4 [3] 3 smallTail19 smallBest = smallBest := by
  rw [smallTail19, scan, if_neg (by decide), smallCandidates19_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail20
    (if 3 ≤ smallCandidates19.length then search relation 3 [3, 541] smallCandidates19 544 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild19]
  exact smallScan20

def smallTail18 : List Nat := 499 :: smallTail19
def smallCandidates18 : List Nat := [673, 1033, 1237, 2707, 5281]
set_option maxRecDepth 2048 in
theorem smallCandidates18_checked : candidates relation 4 3 499 smallTail19 smallBest.bound = smallCandidates18 := by decide

set_option maxRecDepth 2048 in
theorem smallChild18 : search relation 3 [3, 499] smallCandidates18 502 smallBest = smallBest := by decide

theorem smallScan18 : scan relation (search relation 3) 4 [3] 3 smallTail18 smallBest = smallBest := by
  rw [smallTail18, scan, if_neg (by decide), smallCandidates18_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail19
    (if 3 ≤ smallCandidates18.length then search relation 3 [3, 499] smallCandidates18 502 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild18]
  exact smallScan19

def smallTail17 : List Nat := 467 :: smallTail18
def smallCandidates17 : List Nat := [617, 4253]
set_option maxRecDepth 2048 in
theorem smallCandidates17_checked : candidates relation 4 3 467 smallTail18 smallBest.bound = smallCandidates17 := by decide

theorem smallScan17 : scan relation (search relation 3) 4 [3] 3 smallTail17 smallBest = smallBest := by
  rw [smallTail17, scan, if_neg (by decide), smallCandidates17_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail18
    (if 3 ≤ smallCandidates17.length then search relation 3 [3, 467] smallCandidates17 470 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan18

def smallTail16 : List Nat := 449 :: smallTail17
def smallCandidates16 : List Nat := [557, 2069, 2441, 5171, 5801, 6353, 6599]
set_option maxRecDepth 2048 in
theorem smallCandidates16_checked : candidates relation 4 3 449 smallTail17 smallBest.bound = smallCandidates16 := by decide

set_option maxRecDepth 2048 in
theorem smallChild16 : search relation 3 [3, 449] smallCandidates16 452 smallBest = smallBest := by decide

theorem smallScan16 : scan relation (search relation 3) 4 [3] 3 smallTail16 smallBest = smallBest := by
  rw [smallTail16, scan, if_neg (by decide), smallCandidates16_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail17
    (if 3 ≤ smallCandidates16.length then search relation 3 [3, 449] smallCandidates16 452 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild16]
  exact smallScan17

def smallTail15 : List Nat := 373 :: smallTail16
def smallCandidates15 : List Nat := [823, 1237, 1531, 5449, 6793, 7057]
set_option maxRecDepth 2048 in
theorem smallCandidates15_checked : candidates relation 4 3 373 smallTail16 smallBest.bound = smallCandidates15 := by decide

set_option maxRecDepth 2048 in
theorem smallChild15 : search relation 3 [3, 373] smallCandidates15 376 smallBest = smallBest := by decide

theorem smallScan15 : scan relation (search relation 3) 4 [3] 3 smallTail15 smallBest = smallBest := by
  rw [smallTail15, scan, if_neg (by decide), smallCandidates15_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail16
    (if 3 ≤ smallCandidates15.length then search relation 3 [3, 373] smallCandidates15 376 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild15]
  exact smallScan16

def smallTail14 : List Nat := 359 :: smallTail15
def smallCandidates14 : List Nat := [701, 1667]
set_option maxRecDepth 2048 in
theorem smallCandidates14_checked : candidates relation 4 3 359 smallTail15 smallBest.bound = smallCandidates14 := by decide

theorem smallScan14 : scan relation (search relation 3) 4 [3] 3 smallTail14 smallBest = smallBest := by
  rw [smallTail14, scan, if_neg (by decide), smallCandidates14_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail15
    (if 3 ≤ smallCandidates14.length then search relation 3 [3, 359] smallCandidates14 362 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan15

def smallTail13 : List Nat := 331 :: smallTail14
def smallCandidates13 : List Nat := [739, 1531, 3637, 3769, 6451, 8377]
set_option maxRecDepth 2048 in
theorem smallCandidates13_checked : candidates relation 4 3 331 smallTail14 smallBest.bound = smallCandidates13 := by decide

set_option maxRecDepth 2048 in
theorem smallChild13 : search relation 3 [3, 331] smallCandidates13 334 smallBest = smallBest := by decide

theorem smallScan13 : scan relation (search relation 3) 4 [3] 3 smallTail13 smallBest = smallBest := by
  rw [smallTail13, scan, if_neg (by decide), smallCandidates13_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail14
    (if 3 ≤ smallCandidates13.length then search relation 3 [3, 331] smallCandidates13 334 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild13]
  exact smallScan14

def smallTail12 : List Nat := 271 :: smallTail13
def smallCandidates12 : List Nat := [6451]
set_option maxRecDepth 2048 in
theorem smallCandidates12_checked : candidates relation 4 3 271 smallTail13 smallBest.bound = smallCandidates12 := by decide

theorem smallScan12 : scan relation (search relation 3) 4 [3] 3 smallTail12 smallBest = smallBest := by
  rw [smallTail12, scan, if_neg (by decide), smallCandidates12_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail13
    (if 3 ≤ smallCandidates12.length then search relation 3 [3, 271] smallCandidates12 274 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan13

def smallTail11 : List Nat := 229 :: smallTail12
def smallCandidates11 : List Nat := [373, 499, 613, 3301, 5869, 7591]
set_option maxRecDepth 2048 in
theorem smallCandidates11_checked : candidates relation 4 3 229 smallTail12 smallBest.bound = smallCandidates11 := by decide

set_option maxRecDepth 2048 in
theorem smallChild11 : search relation 3 [3, 229] smallCandidates11 232 smallBest = smallBest := by decide

theorem smallScan11 : scan relation (search relation 3) 4 [3] 3 smallTail11 smallBest = smallBest := by
  rw [smallTail11, scan, if_neg (by decide), smallCandidates11_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail12
    (if 3 ≤ smallCandidates11.length then search relation 3 [3, 229] smallCandidates11 232 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild11]
  exact smallScan12

def smallTail10 : List Nat := 191 :: smallTail11
def smallCandidates10 : List Nat := [2069, 2213, 5099, 6599, 8237]
set_option maxRecDepth 2048 in
theorem smallCandidates10_checked : candidates relation 4 3 191 smallTail11 smallBest.bound = smallCandidates10 := by decide

set_option maxRecDepth 2048 in
theorem smallChild10 : search relation 3 [3, 191] smallCandidates10 194 smallBest = smallBest := by decide

theorem smallScan10 : scan relation (search relation 3) 4 [3] 3 smallTail10 smallBest = smallBest := by
  rw [smallTail10, scan, if_neg (by decide), smallCandidates10_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail11
    (if 3 ≤ smallCandidates10.length then search relation 3 [3, 191] smallCandidates10 194 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild10]
  exact smallScan11

def smallTail9 : List Nat := 137 :: smallTail10
def smallCandidates9 : List Nat := [191, 359, 947, 8219]
set_option maxRecDepth 2048 in
theorem smallCandidates9_checked : candidates relation 4 3 137 smallTail10 smallBest.bound = smallCandidates9 := by decide

set_option maxRecDepth 2048 in
theorem smallChild9 : search relation 3 [3, 137] smallCandidates9 140 smallBest = smallBest := by decide

theorem smallScan9 : scan relation (search relation 3) 4 [3] 3 smallTail9 smallBest = smallBest := by
  rw [smallTail9, scan, if_neg (by decide), smallCandidates9_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail10
    (if 3 ≤ smallCandidates9.length then search relation 3 [3, 137] smallCandidates9 140 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild9]
  exact smallScan10

def smallTail8 : List Nat := 109 :: smallTail9
def smallCandidates8 : List Nat := [673]
set_option maxRecDepth 2048 in
theorem smallCandidates8_checked : candidates relation 4 3 109 smallTail9 smallBest.bound = smallCandidates8 := by decide

theorem smallScan8 : scan relation (search relation 3) 4 [3] 3 smallTail8 smallBest = smallBest := by
  rw [smallTail8, scan, if_neg (by decide), smallCandidates8_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail9
    (if 3 ≤ smallCandidates8.length then search relation 3 [3, 109] smallCandidates8 112 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan9

def smallTail7 : List Nat := 73 :: smallTail8
def smallCandidates7 : List Nat := [607, 823, 6277, 6793, 7159, 7591]
set_option maxRecDepth 2048 in
theorem smallCandidates7_checked : candidates relation 4 3 73 smallTail8 smallBest.bound = smallCandidates7 := by decide

set_option maxRecDepth 2048 in
theorem smallChild7 : search relation 3 [3, 73] smallCandidates7 76 smallBest = smallBest := by decide

theorem smallScan7 : scan relation (search relation 3) 4 [3] 3 smallTail7 smallBest = smallBest := by
  rw [smallTail7, scan, if_neg (by decide), smallCandidates7_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail8
    (if 3 ≤ smallCandidates7.length then search relation 3 [3, 73] smallCandidates7 76 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild7]
  exact smallScan8

def smallTail6 : List Nat := 67 :: smallTail7
def smallCandidates6 : List Nat := [2377, 3637, 5419, 5923]
set_option maxRecDepth 2048 in
theorem smallCandidates6_checked : candidates relation 4 3 67 smallTail7 smallBest.bound = smallCandidates6 := by decide

set_option maxRecDepth 2048 in
theorem smallChild6 : search relation 3 [3, 67] smallCandidates6 70 smallBest = smallBest := by decide

theorem smallScan6 : scan relation (search relation 3) 4 [3] 3 smallTail6 smallBest = smallBest := by
  rw [smallTail6, scan, if_neg (by decide), smallCandidates6_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail7
    (if 3 ≤ smallCandidates6.length then search relation 3 [3, 67] smallCandidates6 70 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild6]
  exact smallScan7

def smallTail5 : List Nat := 59 :: smallTail6
def smallCandidates5 : List Nat := [929, 1193, 5381, 6653, 6899, 7253]
set_option maxRecDepth 2048 in
theorem smallCandidates5_checked : candidates relation 4 3 59 smallTail6 smallBest.bound = smallCandidates5 := by decide

set_option maxRecDepth 2048 in
theorem smallChild5 : search relation 3 [3, 59] smallCandidates5 62 smallBest = smallBest := by decide

theorem smallScan5 : scan relation (search relation 3) 4 [3] 3 smallTail5 smallBest = smallBest := by
  rw [smallTail5, scan, if_neg (by decide), smallCandidates5_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail6
    (if 3 ≤ smallCandidates5.length then search relation 3 [3, 59] smallCandidates5 62 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild5]
  exact smallScan6

def smallTail4 : List Nat := 37 :: smallTail5
def smallCandidates4 : List Nat := [67, 607, 1237, 1699, 2377, 2707, 3049, 4159, 4729, 5281, 5449, 5923, 6469, 6529, 7369, 7963]
set_option maxRecDepth 2048 in
theorem smallCandidates4_checked : candidates relation 4 3 37 smallTail5 smallBest.bound = smallCandidates4 := by decide

set_option maxRecDepth 2048 in
theorem smallChild4 : search relation 3 [3, 37] smallCandidates4 40 smallBest = smallBest := by decide

theorem smallScan4 : scan relation (search relation 3) 4 [3] 3 smallTail4 smallBest = smallBest := by
  rw [smallTail4, scan, if_neg (by decide), smallCandidates4_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail5
    (if 3 ≤ smallCandidates4.length then search relation 3 [3, 37] smallCandidates4 40 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild4]
  exact smallScan5

def smallTail3 : List Nat := 31 :: smallTail4
def smallCandidates3 : List Nat := [1237, 3301, 3637, 4159, 5281, 6037, 6073, 6571, 7591, 8377]
set_option maxRecDepth 2048 in
theorem smallCandidates3_checked : candidates relation 4 3 31 smallTail4 smallBest.bound = smallCandidates3 := by decide

set_option maxRecDepth 2048 in
theorem smallChild3 : search relation 3 [3, 31] smallCandidates3 34 smallBest = smallBest := by decide

theorem smallScan3 : scan relation (search relation 3) 4 [3] 3 smallTail3 smallBest = smallBest := by
  rw [smallTail3, scan, if_neg (by decide), smallCandidates3_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail4
    (if 3 ≤ smallCandidates3.length then search relation 3 [3, 31] smallCandidates3 34 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild3]
  exact smallScan4

def smallTail2 : List Nat := 17 :: smallTail3
def smallCandidates2 : List Nat := [449, 2069, 2213, 2297, 2411, 4019, 4583, 6353, 6599, 6899, 7691]
set_option maxRecDepth 2048 in
theorem smallCandidates2_checked : candidates relation 4 3 17 smallTail3 smallBest.bound = smallCandidates2 := by decide

set_option maxRecDepth 2048 in
theorem smallChild2 : search relation 3 [3, 17] smallCandidates2 20 smallBest = smallBest := by decide

theorem smallScan2 : scan relation (search relation 3) 4 [3] 3 smallTail2 smallBest = smallBest := by
  rw [smallTail2, scan, if_neg (by decide), smallCandidates2_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail3
    (if 3 ≤ smallCandidates2.length then search relation 3 [3, 17] smallCandidates2 20 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild2]
  exact smallScan3

def smallTail1 : List Nat := 11 :: smallTail2
def smallCandidates1 : List Nat := [701, 2069, 2213, 2297, 2843, 5099, 8219]
set_option maxRecDepth 2048 in
theorem smallCandidates1_checked : candidates relation 4 3 11 smallTail2 smallBest.bound = smallCandidates1 := by decide

set_option maxRecDepth 2048 in
theorem smallChild1 : search relation 3 [3, 11] smallCandidates1 14 smallBest = smallBest := by decide

theorem smallScan1 : scan relation (search relation 3) 4 [3] 3 smallTail1 smallBest = smallBest := by
  rw [smallTail1, scan, if_neg (by decide), smallCandidates1_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail2
    (if 3 ≤ smallCandidates1.length then search relation 3 [3, 11] smallCandidates1 14 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild1]
  exact smallScan2

def smallTail0 : List Nat := 7 :: smallTail1
def smallCandidates0 : List Nat := [109, 229, 541, 673, 823, 1237, 2503, 2707, 4159, 4729, 5521]
set_option maxRecDepth 2048 in
theorem smallCandidates0_checked : candidates relation 4 3 7 smallTail1 smallBest.bound = smallCandidates0 := by decide

set_option maxRecDepth 2048 in
theorem smallChild0 : search relation 3 [3, 7] smallCandidates0 10 smallBest = smallBest := by decide

theorem smallScan0 : scan relation (search relation 3) 4 [3] 3 smallTail0 smallBest = smallBest := by
  rw [smallTail0, scan, if_neg (by decide), smallCandidates0_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail1
    (if 3 ≤ smallCandidates0.length then search relation 3 [3, 7] smallCandidates0 10 smallBest else smallBest) = smallBest
  rw [if_pos (by decide), smallChild0]
  exact smallScan1

theorem small_p3_branch : search relation 4 [3] PrimePairRows.p3_small 3 smallBest = smallBest := by
  rw [search_succ_eq_scan]
  change scan relation (search relation 3) 4 [3] 3 smallTail0 smallBest = smallBest
  exact smallScan0

end ProjectEulerSolutions.PrimePairSearch
