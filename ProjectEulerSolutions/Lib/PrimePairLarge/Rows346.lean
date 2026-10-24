import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11497_small : List Nat := []
theorem row11497_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11497 row11497_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11497_block_005 : List Nat := []
theorem row11497_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11497 row11497_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11497_block_006 : List Nat := [11677]
def plan11497_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .second, .first, (.secondFactor 29), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11497_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11497 row11497_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11497_block_006
  · decide
  · rfl

def row11497_block_007 : List Nat := [12541, 12583]
def plan11497_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 23), .first, .first, .possible, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 13), .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .second, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 29), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11497_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11497 row11497_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11497_block_007
  · decide
  · rfl

def row11497_block_008 : List Nat := []
def plan11497_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11497_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11497 row11497_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11497_block_008
  · decide
  · rfl

def row11497_block_009 : List Nat := []
theorem row11497_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11497 row11497_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11497_block_010 : List Nat := []
theorem row11497_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11497 row11497_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11497_block_011 : List Nat := []
theorem row11497_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11497 row11497_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11497_block_012 : List Nat := []
theorem row11497_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11497 row11497_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11497_block_013 : List Nat := []
theorem row11497_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11497 row11497_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11497_large_left_left : List Nat := row11497_small ++ row11497_block_005
theorem row11497_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11497 row11497_large_left_left :=
  Budget.RowUpperBound.branch row11497_small_covers row11497_block_005_covers

def row11497_large_left_right_right : List Nat := row11497_block_007 ++ row11497_block_008
theorem row11497_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11497 row11497_large_left_right_right :=
  Budget.RowUpperBound.branch row11497_block_007_covers row11497_block_008_covers

def row11497_large_left_right : List Nat := row11497_block_006 ++ row11497_large_left_right_right
theorem row11497_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11497 row11497_large_left_right :=
  Budget.RowUpperBound.branch row11497_block_006_covers row11497_large_left_right_right_covers

def row11497_large_left : List Nat := row11497_large_left_left ++ row11497_large_left_right
theorem row11497_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11497 row11497_large_left :=
  Budget.RowUpperBound.branch row11497_large_left_left_covers row11497_large_left_right_covers

def row11497_large_right_left : List Nat := row11497_block_009 ++ row11497_block_010
theorem row11497_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11497 row11497_large_right_left :=
  Budget.RowUpperBound.branch row11497_block_009_covers row11497_block_010_covers

def row11497_large_right_right_right : List Nat := row11497_block_012 ++ row11497_block_013
theorem row11497_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11497 row11497_large_right_right_right :=
  Budget.RowUpperBound.branch row11497_block_012_covers row11497_block_013_covers

def row11497_large_right_right : List Nat := row11497_block_011 ++ row11497_large_right_right_right
theorem row11497_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11497 row11497_large_right_right :=
  Budget.RowUpperBound.branch row11497_block_011_covers row11497_large_right_right_right_covers

def row11497_large_right : List Nat := row11497_large_right_left ++ row11497_large_right_right
theorem row11497_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11497 row11497_large_right :=
  Budget.RowUpperBound.branch row11497_large_right_left_covers row11497_large_right_right_covers

def row11497_large : List Nat := row11497_large_left ++ row11497_large_right
theorem row11497_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11497 row11497_large :=
  Budget.RowUpperBound.branch row11497_large_left_covers row11497_large_right_covers

def row11497 : List Nat := row11497_large
theorem row11497_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11497 row11497 := row11497_large_covers

def row11503_small : List Nat := []
theorem row11503_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11503 row11503_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11503_block_005 : List Nat := []
theorem row11503_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11503 row11503_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11503_block_006 : List Nat := []
def plan11503_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, (.secondFactor 31), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row11503_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11503 row11503_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11503_block_006
  · decide
  · rfl

def row11503_block_007 : List Nat := [12577, 12613, 12697, 13627, 13921]
def plan11503_block_007 : List Budget.Choice := [(.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .second, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 29), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.secondFactor 19), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 37), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row11503_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11503 row11503_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11503_block_007
  · decide
  · rfl

def row11503_block_008 : List Nat := []
def plan11503_block_008 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11503_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11503 row11503_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11503_block_008
  · decide
  · rfl

def row11503_block_009 : List Nat := []
theorem row11503_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11503 row11503_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11503_block_010 : List Nat := []
theorem row11503_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11503 row11503_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11503_block_011 : List Nat := []
theorem row11503_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11503 row11503_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11503_block_012 : List Nat := []
theorem row11503_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11503 row11503_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11503_block_013 : List Nat := []
theorem row11503_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11503 row11503_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11503_large_left_left : List Nat := row11503_small ++ row11503_block_005
theorem row11503_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11503 row11503_large_left_left :=
  Budget.RowUpperBound.branch row11503_small_covers row11503_block_005_covers

def row11503_large_left_right_right : List Nat := row11503_block_007 ++ row11503_block_008
theorem row11503_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11503 row11503_large_left_right_right :=
  Budget.RowUpperBound.branch row11503_block_007_covers row11503_block_008_covers

def row11503_large_left_right : List Nat := row11503_block_006 ++ row11503_large_left_right_right
theorem row11503_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11503 row11503_large_left_right :=
  Budget.RowUpperBound.branch row11503_block_006_covers row11503_large_left_right_right_covers

def row11503_large_left : List Nat := row11503_large_left_left ++ row11503_large_left_right
theorem row11503_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11503 row11503_large_left :=
  Budget.RowUpperBound.branch row11503_large_left_left_covers row11503_large_left_right_covers

def row11503_large_right_left : List Nat := row11503_block_009 ++ row11503_block_010
theorem row11503_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11503 row11503_large_right_left :=
  Budget.RowUpperBound.branch row11503_block_009_covers row11503_block_010_covers

def row11503_large_right_right_right : List Nat := row11503_block_012 ++ row11503_block_013
theorem row11503_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11503 row11503_large_right_right_right :=
  Budget.RowUpperBound.branch row11503_block_012_covers row11503_block_013_covers

def row11503_large_right_right : List Nat := row11503_block_011 ++ row11503_large_right_right_right
theorem row11503_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11503 row11503_large_right_right :=
  Budget.RowUpperBound.branch row11503_block_011_covers row11503_large_right_right_right_covers

def row11503_large_right : List Nat := row11503_large_right_left ++ row11503_large_right_right
theorem row11503_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11503 row11503_large_right :=
  Budget.RowUpperBound.branch row11503_large_right_left_covers row11503_large_right_right_covers

def row11503_large : List Nat := row11503_large_left ++ row11503_large_right
theorem row11503_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11503 row11503_large :=
  Budget.RowUpperBound.branch row11503_large_left_covers row11503_large_right_covers

def row11503 : List Nat := row11503_large
theorem row11503_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11503 row11503 := row11503_large_covers

def row11519_small : List Nat := []
theorem row11519_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11519 row11519_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11519_block_005 : List Nat := []
theorem row11519_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11519 row11519_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11519_block_006 : List Nat := [11633, 11681, 11927, 12149, 12239]
def plan11519_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 23), .possible, .first, (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11519_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11519 row11519_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11519_block_006
  · decide
  · rfl

def row11519_block_007 : List Nat := [12569, 12689, 13331, 14447]
def plan11519_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 31), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .second, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 37), .second, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row11519_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11519 row11519_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11519_block_007
  · decide
  · rfl

def row11519_block_008 : List Nat := []
def plan11519_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11519_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11519 row11519_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11519_block_008
  · decide
  · rfl

def row11519_block_009 : List Nat := []
theorem row11519_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11519 row11519_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11519_block_010 : List Nat := []
theorem row11519_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11519 row11519_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11519_block_011 : List Nat := []
theorem row11519_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11519 row11519_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11519_block_012 : List Nat := []
theorem row11519_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11519 row11519_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11519_block_013 : List Nat := []
theorem row11519_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11519 row11519_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11519_large_left_left : List Nat := row11519_small ++ row11519_block_005
theorem row11519_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11519 row11519_large_left_left :=
  Budget.RowUpperBound.branch row11519_small_covers row11519_block_005_covers

def row11519_large_left_right_right : List Nat := row11519_block_007 ++ row11519_block_008
theorem row11519_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11519 row11519_large_left_right_right :=
  Budget.RowUpperBound.branch row11519_block_007_covers row11519_block_008_covers

def row11519_large_left_right : List Nat := row11519_block_006 ++ row11519_large_left_right_right
theorem row11519_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11519 row11519_large_left_right :=
  Budget.RowUpperBound.branch row11519_block_006_covers row11519_large_left_right_right_covers

def row11519_large_left : List Nat := row11519_large_left_left ++ row11519_large_left_right
theorem row11519_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11519 row11519_large_left :=
  Budget.RowUpperBound.branch row11519_large_left_left_covers row11519_large_left_right_covers

def row11519_large_right_left : List Nat := row11519_block_009 ++ row11519_block_010
theorem row11519_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11519 row11519_large_right_left :=
  Budget.RowUpperBound.branch row11519_block_009_covers row11519_block_010_covers

def row11519_large_right_right_right : List Nat := row11519_block_012 ++ row11519_block_013
theorem row11519_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11519 row11519_large_right_right_right :=
  Budget.RowUpperBound.branch row11519_block_012_covers row11519_block_013_covers

def row11519_large_right_right : List Nat := row11519_block_011 ++ row11519_large_right_right_right
theorem row11519_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11519 row11519_large_right_right :=
  Budget.RowUpperBound.branch row11519_block_011_covers row11519_large_right_right_right_covers

def row11519_large_right : List Nat := row11519_large_right_left ++ row11519_large_right_right
theorem row11519_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11519 row11519_large_right :=
  Budget.RowUpperBound.branch row11519_large_right_left_covers row11519_large_right_right_covers

def row11519_large : List Nat := row11519_large_left ++ row11519_large_right
theorem row11519_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11519 row11519_large :=
  Budget.RowUpperBound.branch row11519_large_left_covers row11519_large_right_covers

def row11519 : List Nat := row11519_large
theorem row11519_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11519 row11519 := row11519_large_covers

def row11527_small : List Nat := []
theorem row11527_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11527 row11527_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11527_block_005 : List Nat := []
theorem row11527_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11527 row11527_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11527_block_006 : List Nat := [12073]
def plan11527_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 19), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .second, .second, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 37), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row11527_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11527 row11527_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11527_block_006
  · decide
  · rfl

def row11527_block_007 : List Nat := [12703, 13567, 14449]
def plan11527_block_007 : List Budget.Choice := [(.firstFactor 7), .second, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 37), (.secondFactor 19), .first, .first, .second, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .possible, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row11527_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11527 row11527_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11527_block_007
  · decide
  · rfl

def row11527_block_008 : List Nat := []
def plan11527_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11527_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11527 row11527_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11527_block_008
  · decide
  · rfl

def row11527_block_009 : List Nat := []
theorem row11527_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11527 row11527_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11527_block_010 : List Nat := []
theorem row11527_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11527 row11527_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11527_block_011 : List Nat := []
theorem row11527_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11527 row11527_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11527_block_012 : List Nat := []
theorem row11527_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11527 row11527_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11527_block_013 : List Nat := []
theorem row11527_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11527 row11527_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11527_large_left_left : List Nat := row11527_small ++ row11527_block_005
theorem row11527_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11527 row11527_large_left_left :=
  Budget.RowUpperBound.branch row11527_small_covers row11527_block_005_covers

def row11527_large_left_right_right : List Nat := row11527_block_007 ++ row11527_block_008
theorem row11527_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11527 row11527_large_left_right_right :=
  Budget.RowUpperBound.branch row11527_block_007_covers row11527_block_008_covers

def row11527_large_left_right : List Nat := row11527_block_006 ++ row11527_large_left_right_right
theorem row11527_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11527 row11527_large_left_right :=
  Budget.RowUpperBound.branch row11527_block_006_covers row11527_large_left_right_right_covers

def row11527_large_left : List Nat := row11527_large_left_left ++ row11527_large_left_right
theorem row11527_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11527 row11527_large_left :=
  Budget.RowUpperBound.branch row11527_large_left_left_covers row11527_large_left_right_covers

def row11527_large_right_left : List Nat := row11527_block_009 ++ row11527_block_010
theorem row11527_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11527 row11527_large_right_left :=
  Budget.RowUpperBound.branch row11527_block_009_covers row11527_block_010_covers

def row11527_large_right_right_right : List Nat := row11527_block_012 ++ row11527_block_013
theorem row11527_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11527 row11527_large_right_right_right :=
  Budget.RowUpperBound.branch row11527_block_012_covers row11527_block_013_covers

def row11527_large_right_right : List Nat := row11527_block_011 ++ row11527_large_right_right_right
theorem row11527_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11527 row11527_large_right_right :=
  Budget.RowUpperBound.branch row11527_block_011_covers row11527_large_right_right_right_covers

def row11527_large_right : List Nat := row11527_large_right_left ++ row11527_large_right_right
theorem row11527_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11527 row11527_large_right :=
  Budget.RowUpperBound.branch row11527_large_right_left_covers row11527_large_right_right_covers

def row11527_large : List Nat := row11527_large_left ++ row11527_large_right
theorem row11527_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11527 row11527_large :=
  Budget.RowUpperBound.branch row11527_large_left_covers row11527_large_right_covers

def row11527 : List Nat := row11527_large
theorem row11527_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11527 row11527 := row11527_large_covers

end ProjectEulerSolutions.PrimePairLarge
