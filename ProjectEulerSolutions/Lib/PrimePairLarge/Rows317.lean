import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10357_small : List Nat := []
theorem row10357_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10357 row10357_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10357_block_005 : List Nat := []
def plan10357_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10357_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10357 row10357_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10357_block_005
  · decide
  · rfl

def row10357_block_006 : List Nat := [11239, 11731, 11959, 12157, 12289]
def plan10357_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, .second, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .possible, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 37), .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row10357_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10357 row10357_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10357_block_006
  · decide
  · rfl

def row10357_block_007 : List Nat := [12967, 13033]
def plan10357_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .second, (.firstFactor 37), (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10357_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10357 row10357_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10357_block_007
  · decide
  · rfl

def row10357_block_008 : List Nat := [14731, 15349, 15427, 15511]
def plan10357_block_008 : List Budget.Choice := [.second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .second, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 23), (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), (.secondFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 37), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10357_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10357 row10357_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10357_block_008
  · decide
  · rfl

def row10357_block_009 : List Nat := []
theorem row10357_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10357 row10357_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10357_block_010 : List Nat := []
theorem row10357_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10357 row10357_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10357_block_011 : List Nat := []
theorem row10357_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10357 row10357_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10357_block_012 : List Nat := []
theorem row10357_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10357 row10357_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10357_block_013 : List Nat := []
theorem row10357_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10357 row10357_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10357_large_left_left : List Nat := row10357_small ++ row10357_block_005
theorem row10357_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10357 row10357_large_left_left :=
  Budget.RowUpperBound.branch row10357_small_covers row10357_block_005_covers

def row10357_large_left_right_right : List Nat := row10357_block_007 ++ row10357_block_008
theorem row10357_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10357 row10357_large_left_right_right :=
  Budget.RowUpperBound.branch row10357_block_007_covers row10357_block_008_covers

def row10357_large_left_right : List Nat := row10357_block_006 ++ row10357_large_left_right_right
theorem row10357_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10357 row10357_large_left_right :=
  Budget.RowUpperBound.branch row10357_block_006_covers row10357_large_left_right_right_covers

def row10357_large_left : List Nat := row10357_large_left_left ++ row10357_large_left_right
theorem row10357_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10357 row10357_large_left :=
  Budget.RowUpperBound.branch row10357_large_left_left_covers row10357_large_left_right_covers

def row10357_large_right_left : List Nat := row10357_block_009 ++ row10357_block_010
theorem row10357_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10357 row10357_large_right_left :=
  Budget.RowUpperBound.branch row10357_block_009_covers row10357_block_010_covers

def row10357_large_right_right_right : List Nat := row10357_block_012 ++ row10357_block_013
theorem row10357_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10357 row10357_large_right_right_right :=
  Budget.RowUpperBound.branch row10357_block_012_covers row10357_block_013_covers

def row10357_large_right_right : List Nat := row10357_block_011 ++ row10357_large_right_right_right
theorem row10357_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10357 row10357_large_right_right :=
  Budget.RowUpperBound.branch row10357_block_011_covers row10357_large_right_right_right_covers

def row10357_large_right : List Nat := row10357_large_right_left ++ row10357_large_right_right
theorem row10357_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10357 row10357_large_right :=
  Budget.RowUpperBound.branch row10357_large_right_left_covers row10357_large_right_right_covers

def row10357_large : List Nat := row10357_large_left ++ row10357_large_right
theorem row10357_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10357 row10357_large :=
  Budget.RowUpperBound.branch row10357_large_left_covers row10357_large_right_covers

def row10357 : List Nat := row10357_large
theorem row10357_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10357 row10357 := row10357_large_covers

def row10369_small : List Nat := []
theorem row10369_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10369 row10369_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10369_block_005 : List Nat := []
def plan10369_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10369_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10369 row10369_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10369_block_005
  · decide
  · rfl

def row10369_block_006 : List Nat := [11119]
def plan10369_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row10369_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10369 row10369_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10369_block_006
  · decide
  · rfl

def row10369_block_007 : List Nat := [12577, 13177, 13723, 13873]
def plan10369_block_007 : List Budget.Choice := [.second, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 19), .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 37), (.secondFactor 13), (.firstFactor 17), (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 37), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 23), .first, (.secondFactor 29), .first, .first, .possible, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .second, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10369_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10369 row10369_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10369_block_007
  · decide
  · rfl

def row10369_block_008 : List Nat := [14479, 14653, 14731]
def plan10369_block_008 : List Budget.Choice := [.possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 37), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .second, .second, (.firstFactor 13), .first, .first, .second, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .second, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10369_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10369 row10369_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10369_block_008
  · decide
  · rfl

def row10369_block_009 : List Nat := []
theorem row10369_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10369 row10369_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10369_block_010 : List Nat := []
theorem row10369_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10369 row10369_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10369_block_011 : List Nat := []
theorem row10369_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10369 row10369_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10369_block_012 : List Nat := []
theorem row10369_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10369 row10369_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10369_block_013 : List Nat := []
theorem row10369_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10369 row10369_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10369_large_left_left : List Nat := row10369_small ++ row10369_block_005
theorem row10369_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10369 row10369_large_left_left :=
  Budget.RowUpperBound.branch row10369_small_covers row10369_block_005_covers

def row10369_large_left_right_right : List Nat := row10369_block_007 ++ row10369_block_008
theorem row10369_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10369 row10369_large_left_right_right :=
  Budget.RowUpperBound.branch row10369_block_007_covers row10369_block_008_covers

def row10369_large_left_right : List Nat := row10369_block_006 ++ row10369_large_left_right_right
theorem row10369_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10369 row10369_large_left_right :=
  Budget.RowUpperBound.branch row10369_block_006_covers row10369_large_left_right_right_covers

def row10369_large_left : List Nat := row10369_large_left_left ++ row10369_large_left_right
theorem row10369_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10369 row10369_large_left :=
  Budget.RowUpperBound.branch row10369_large_left_left_covers row10369_large_left_right_covers

def row10369_large_right_left : List Nat := row10369_block_009 ++ row10369_block_010
theorem row10369_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10369 row10369_large_right_left :=
  Budget.RowUpperBound.branch row10369_block_009_covers row10369_block_010_covers

def row10369_large_right_right_right : List Nat := row10369_block_012 ++ row10369_block_013
theorem row10369_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10369 row10369_large_right_right_right :=
  Budget.RowUpperBound.branch row10369_block_012_covers row10369_block_013_covers

def row10369_large_right_right : List Nat := row10369_block_011 ++ row10369_large_right_right_right
theorem row10369_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10369 row10369_large_right_right :=
  Budget.RowUpperBound.branch row10369_block_011_covers row10369_large_right_right_right_covers

def row10369_large_right : List Nat := row10369_large_right_left ++ row10369_large_right_right
theorem row10369_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10369 row10369_large_right :=
  Budget.RowUpperBound.branch row10369_large_right_left_covers row10369_large_right_right_covers

def row10369_large : List Nat := row10369_large_left ++ row10369_large_right
theorem row10369_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10369 row10369_large :=
  Budget.RowUpperBound.branch row10369_large_left_covers row10369_large_right_covers

def row10369 : List Nat := row10369_large
theorem row10369_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10369 row10369 := row10369_large_covers

def row10391_small : List Nat := []
theorem row10391_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10391 row10391_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10391_block_005 : List Nat := []
theorem row10391_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10391 row10391_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10391_block_006 : List Nat := [11399, 11897]
def plan10391_block_006 : List Budget.Choice := [.outside, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row10391_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10391 row10391_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10391_block_006
  · decide
  · rfl

def row10391_block_007 : List Nat := []
def plan10391_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .second, .first, (.firstFactor 29), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10391_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10391 row10391_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10391_block_007
  · decide
  · rfl

def row10391_block_008 : List Nat := [14969]
def plan10391_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, (.secondFactor 7), .second, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10391_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10391 row10391_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10391_block_008
  · decide
  · rfl

def row10391_block_009 : List Nat := []
theorem row10391_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10391 row10391_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10391_block_010 : List Nat := []
theorem row10391_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10391 row10391_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10391_block_011 : List Nat := []
theorem row10391_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10391 row10391_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10391_block_012 : List Nat := []
theorem row10391_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10391 row10391_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10391_block_013 : List Nat := []
theorem row10391_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10391 row10391_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10391_large_left_left : List Nat := row10391_small ++ row10391_block_005
theorem row10391_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10391 row10391_large_left_left :=
  Budget.RowUpperBound.branch row10391_small_covers row10391_block_005_covers

def row10391_large_left_right_right : List Nat := row10391_block_007 ++ row10391_block_008
theorem row10391_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10391 row10391_large_left_right_right :=
  Budget.RowUpperBound.branch row10391_block_007_covers row10391_block_008_covers

def row10391_large_left_right : List Nat := row10391_block_006 ++ row10391_large_left_right_right
theorem row10391_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10391 row10391_large_left_right :=
  Budget.RowUpperBound.branch row10391_block_006_covers row10391_large_left_right_right_covers

def row10391_large_left : List Nat := row10391_large_left_left ++ row10391_large_left_right
theorem row10391_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10391 row10391_large_left :=
  Budget.RowUpperBound.branch row10391_large_left_left_covers row10391_large_left_right_covers

def row10391_large_right_left : List Nat := row10391_block_009 ++ row10391_block_010
theorem row10391_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10391 row10391_large_right_left :=
  Budget.RowUpperBound.branch row10391_block_009_covers row10391_block_010_covers

def row10391_large_right_right_right : List Nat := row10391_block_012 ++ row10391_block_013
theorem row10391_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10391 row10391_large_right_right_right :=
  Budget.RowUpperBound.branch row10391_block_012_covers row10391_block_013_covers

def row10391_large_right_right : List Nat := row10391_block_011 ++ row10391_large_right_right_right
theorem row10391_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10391 row10391_large_right_right :=
  Budget.RowUpperBound.branch row10391_block_011_covers row10391_large_right_right_right_covers

def row10391_large_right : List Nat := row10391_large_right_left ++ row10391_large_right_right
theorem row10391_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10391 row10391_large_right :=
  Budget.RowUpperBound.branch row10391_large_right_left_covers row10391_large_right_right_covers

def row10391_large : List Nat := row10391_large_left ++ row10391_large_right
theorem row10391_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10391 row10391_large :=
  Budget.RowUpperBound.branch row10391_large_left_covers row10391_large_right_covers

def row10391 : List Nat := row10391_large
theorem row10391_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10391 row10391 := row10391_large_covers

def row10399_small : List Nat := []
theorem row10399_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10399 row10399_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10399_block_005 : List Nat := []
theorem row10399_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10399 row10399_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10399_block_006 : List Nat := [10909, 11329]
def plan10399_block_006 : List Budget.Choice := [.outside, .outside, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 23), (.firstFactor 31), .possible, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 23), .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row10399_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10399 row10399_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10399_block_006
  · decide
  · rfl

def row10399_block_007 : List Nat := [12619, 13249, 13669]
def plan10399_block_007 : List Budget.Choice := [(.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 23), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10399_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10399 row10399_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10399_block_007
  · decide
  · rfl

def row10399_block_008 : List Nat := [14851]
def plan10399_block_008 : List Budget.Choice := [(.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .second, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10399_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10399 row10399_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10399_block_008
  · decide
  · rfl

def row10399_block_009 : List Nat := []
theorem row10399_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10399 row10399_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10399_block_010 : List Nat := []
theorem row10399_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10399 row10399_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10399_block_011 : List Nat := []
theorem row10399_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10399 row10399_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10399_block_012 : List Nat := []
theorem row10399_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10399 row10399_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10399_block_013 : List Nat := []
theorem row10399_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10399 row10399_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10399_large_left_left : List Nat := row10399_small ++ row10399_block_005
theorem row10399_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10399 row10399_large_left_left :=
  Budget.RowUpperBound.branch row10399_small_covers row10399_block_005_covers

def row10399_large_left_right_right : List Nat := row10399_block_007 ++ row10399_block_008
theorem row10399_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10399 row10399_large_left_right_right :=
  Budget.RowUpperBound.branch row10399_block_007_covers row10399_block_008_covers

def row10399_large_left_right : List Nat := row10399_block_006 ++ row10399_large_left_right_right
theorem row10399_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10399 row10399_large_left_right :=
  Budget.RowUpperBound.branch row10399_block_006_covers row10399_large_left_right_right_covers

def row10399_large_left : List Nat := row10399_large_left_left ++ row10399_large_left_right
theorem row10399_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10399 row10399_large_left :=
  Budget.RowUpperBound.branch row10399_large_left_left_covers row10399_large_left_right_covers

def row10399_large_right_left : List Nat := row10399_block_009 ++ row10399_block_010
theorem row10399_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10399 row10399_large_right_left :=
  Budget.RowUpperBound.branch row10399_block_009_covers row10399_block_010_covers

def row10399_large_right_right_right : List Nat := row10399_block_012 ++ row10399_block_013
theorem row10399_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10399 row10399_large_right_right_right :=
  Budget.RowUpperBound.branch row10399_block_012_covers row10399_block_013_covers

def row10399_large_right_right : List Nat := row10399_block_011 ++ row10399_large_right_right_right
theorem row10399_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10399 row10399_large_right_right :=
  Budget.RowUpperBound.branch row10399_block_011_covers row10399_large_right_right_right_covers

def row10399_large_right : List Nat := row10399_large_right_left ++ row10399_large_right_right
theorem row10399_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10399 row10399_large_right :=
  Budget.RowUpperBound.branch row10399_large_right_left_covers row10399_large_right_right_covers

def row10399_large : List Nat := row10399_large_left ++ row10399_large_right
theorem row10399_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10399 row10399_large :=
  Budget.RowUpperBound.branch row10399_large_left_covers row10399_large_right_covers

def row10399 : List Nat := row10399_large
theorem row10399_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10399 row10399 := row10399_large_covers

end ProjectEulerSolutions.PrimePairLarge
