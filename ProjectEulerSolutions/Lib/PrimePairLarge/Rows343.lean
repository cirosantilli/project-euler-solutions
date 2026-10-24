import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11393_small : List Nat := []
theorem row11393_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11393 row11393_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11393_block_005 : List Nat := []
theorem row11393_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11393 row11393_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11393_block_006 : List Nat := [11471, 11633]
def plan11393_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 19), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 37), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11393_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11393 row11393_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11393_block_006
  · decide
  · rfl

def row11393_block_007 : List Nat := [12893, 13697, 13709]
def plan11393_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 29), .first, .first, .first, .first, .first, .possible, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 13), (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .possible, .possible, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11393_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11393 row11393_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11393_block_007
  · decide
  · rfl

def row11393_block_008 : List Nat := [14633]
def plan11393_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11393_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11393 row11393_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11393_block_008
  · decide
  · rfl

def row11393_block_009 : List Nat := []
theorem row11393_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11393 row11393_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11393_block_010 : List Nat := []
theorem row11393_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11393 row11393_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11393_block_011 : List Nat := []
theorem row11393_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11393 row11393_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11393_block_012 : List Nat := []
theorem row11393_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11393 row11393_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11393_block_013 : List Nat := []
theorem row11393_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11393 row11393_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11393_large_left_left : List Nat := row11393_small ++ row11393_block_005
theorem row11393_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11393 row11393_large_left_left :=
  Budget.RowUpperBound.branch row11393_small_covers row11393_block_005_covers

def row11393_large_left_right_right : List Nat := row11393_block_007 ++ row11393_block_008
theorem row11393_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11393 row11393_large_left_right_right :=
  Budget.RowUpperBound.branch row11393_block_007_covers row11393_block_008_covers

def row11393_large_left_right : List Nat := row11393_block_006 ++ row11393_large_left_right_right
theorem row11393_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11393 row11393_large_left_right :=
  Budget.RowUpperBound.branch row11393_block_006_covers row11393_large_left_right_right_covers

def row11393_large_left : List Nat := row11393_large_left_left ++ row11393_large_left_right
theorem row11393_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11393 row11393_large_left :=
  Budget.RowUpperBound.branch row11393_large_left_left_covers row11393_large_left_right_covers

def row11393_large_right_left : List Nat := row11393_block_009 ++ row11393_block_010
theorem row11393_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11393 row11393_large_right_left :=
  Budget.RowUpperBound.branch row11393_block_009_covers row11393_block_010_covers

def row11393_large_right_right_right : List Nat := row11393_block_012 ++ row11393_block_013
theorem row11393_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11393 row11393_large_right_right_right :=
  Budget.RowUpperBound.branch row11393_block_012_covers row11393_block_013_covers

def row11393_large_right_right : List Nat := row11393_block_011 ++ row11393_large_right_right_right
theorem row11393_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11393 row11393_large_right_right :=
  Budget.RowUpperBound.branch row11393_block_011_covers row11393_large_right_right_right_covers

def row11393_large_right : List Nat := row11393_large_right_left ++ row11393_large_right_right
theorem row11393_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11393 row11393_large_right :=
  Budget.RowUpperBound.branch row11393_large_right_left_covers row11393_large_right_right_covers

def row11393_large : List Nat := row11393_large_left ++ row11393_large_right
theorem row11393_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11393 row11393_large :=
  Budget.RowUpperBound.branch row11393_large_left_covers row11393_large_right_covers

def row11393 : List Nat := row11393_large
theorem row11393_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11393 row11393 := row11393_large_covers

def row11399_small : List Nat := []
theorem row11399_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11399 row11399_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11399_block_005 : List Nat := []
theorem row11399_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11399 row11399_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11399_block_006 : List Nat := [12161]
def plan11399_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.secondFactor 13), .first, .possible, .first, (.secondFactor 31), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row11399_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11399 row11399_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11399_block_006
  · decide
  · rfl

def row11399_block_007 : List Nat := [12911, 14249]
def plan11399_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.secondFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 31), (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11399_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11399 row11399_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11399_block_007
  · decide
  · rfl

def row11399_block_008 : List Nat := []
def plan11399_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11399_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11399 row11399_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11399_block_008
  · decide
  · rfl

def row11399_block_009 : List Nat := []
theorem row11399_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11399 row11399_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11399_block_010 : List Nat := []
theorem row11399_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11399 row11399_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11399_block_011 : List Nat := []
theorem row11399_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11399 row11399_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11399_block_012 : List Nat := []
theorem row11399_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11399 row11399_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11399_block_013 : List Nat := []
theorem row11399_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11399 row11399_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11399_large_left_left : List Nat := row11399_small ++ row11399_block_005
theorem row11399_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11399 row11399_large_left_left :=
  Budget.RowUpperBound.branch row11399_small_covers row11399_block_005_covers

def row11399_large_left_right_right : List Nat := row11399_block_007 ++ row11399_block_008
theorem row11399_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11399 row11399_large_left_right_right :=
  Budget.RowUpperBound.branch row11399_block_007_covers row11399_block_008_covers

def row11399_large_left_right : List Nat := row11399_block_006 ++ row11399_large_left_right_right
theorem row11399_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11399 row11399_large_left_right :=
  Budget.RowUpperBound.branch row11399_block_006_covers row11399_large_left_right_right_covers

def row11399_large_left : List Nat := row11399_large_left_left ++ row11399_large_left_right
theorem row11399_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11399 row11399_large_left :=
  Budget.RowUpperBound.branch row11399_large_left_left_covers row11399_large_left_right_covers

def row11399_large_right_left : List Nat := row11399_block_009 ++ row11399_block_010
theorem row11399_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11399 row11399_large_right_left :=
  Budget.RowUpperBound.branch row11399_block_009_covers row11399_block_010_covers

def row11399_large_right_right_right : List Nat := row11399_block_012 ++ row11399_block_013
theorem row11399_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11399 row11399_large_right_right_right :=
  Budget.RowUpperBound.branch row11399_block_012_covers row11399_block_013_covers

def row11399_large_right_right : List Nat := row11399_block_011 ++ row11399_large_right_right_right
theorem row11399_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11399 row11399_large_right_right :=
  Budget.RowUpperBound.branch row11399_block_011_covers row11399_large_right_right_right_covers

def row11399_large_right : List Nat := row11399_large_right_left ++ row11399_large_right_right
theorem row11399_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11399 row11399_large_right :=
  Budget.RowUpperBound.branch row11399_large_right_left_covers row11399_large_right_right_covers

def row11399_large : List Nat := row11399_large_left ++ row11399_large_right
theorem row11399_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11399 row11399_large :=
  Budget.RowUpperBound.branch row11399_large_left_covers row11399_large_right_covers

def row11399 : List Nat := row11399_large
theorem row11399_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11399 row11399 := row11399_large_covers

def row11411_small : List Nat := []
theorem row11411_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11411 row11411_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11411_block_005 : List Nat := []
theorem row11411_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11411 row11411_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11411_block_006 : List Nat := [12281]
def plan11411_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 37), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11411_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11411 row11411_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11411_block_006
  · decide
  · rfl

def row11411_block_007 : List Nat := [12941, 13907]
def plan11411_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 37), (.secondFactor 37), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 37), .second, .first, (.secondFactor 37), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .second, (.secondFactor 19), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .possible, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row11411_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11411 row11411_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11411_block_007
  · decide
  · rfl

def row11411_block_008 : List Nat := []
def plan11411_block_008 : List Budget.Choice := [.first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 29), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11411_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11411 row11411_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11411_block_008
  · decide
  · rfl

def row11411_block_009 : List Nat := []
theorem row11411_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11411 row11411_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11411_block_010 : List Nat := []
theorem row11411_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11411 row11411_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11411_block_011 : List Nat := []
theorem row11411_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11411 row11411_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11411_block_012 : List Nat := []
theorem row11411_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11411 row11411_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11411_block_013 : List Nat := []
theorem row11411_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11411 row11411_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11411_large_left_left : List Nat := row11411_small ++ row11411_block_005
theorem row11411_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11411 row11411_large_left_left :=
  Budget.RowUpperBound.branch row11411_small_covers row11411_block_005_covers

def row11411_large_left_right_right : List Nat := row11411_block_007 ++ row11411_block_008
theorem row11411_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11411 row11411_large_left_right_right :=
  Budget.RowUpperBound.branch row11411_block_007_covers row11411_block_008_covers

def row11411_large_left_right : List Nat := row11411_block_006 ++ row11411_large_left_right_right
theorem row11411_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11411 row11411_large_left_right :=
  Budget.RowUpperBound.branch row11411_block_006_covers row11411_large_left_right_right_covers

def row11411_large_left : List Nat := row11411_large_left_left ++ row11411_large_left_right
theorem row11411_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11411 row11411_large_left :=
  Budget.RowUpperBound.branch row11411_large_left_left_covers row11411_large_left_right_covers

def row11411_large_right_left : List Nat := row11411_block_009 ++ row11411_block_010
theorem row11411_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11411 row11411_large_right_left :=
  Budget.RowUpperBound.branch row11411_block_009_covers row11411_block_010_covers

def row11411_large_right_right_right : List Nat := row11411_block_012 ++ row11411_block_013
theorem row11411_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11411 row11411_large_right_right_right :=
  Budget.RowUpperBound.branch row11411_block_012_covers row11411_block_013_covers

def row11411_large_right_right : List Nat := row11411_block_011 ++ row11411_large_right_right_right
theorem row11411_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11411 row11411_large_right_right :=
  Budget.RowUpperBound.branch row11411_block_011_covers row11411_large_right_right_right_covers

def row11411_large_right : List Nat := row11411_large_right_left ++ row11411_large_right_right
theorem row11411_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11411 row11411_large_right :=
  Budget.RowUpperBound.branch row11411_large_right_left_covers row11411_large_right_right_covers

def row11411_large : List Nat := row11411_large_left ++ row11411_large_right
theorem row11411_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11411 row11411_large :=
  Budget.RowUpperBound.branch row11411_large_left_covers row11411_large_right_covers

def row11411 : List Nat := row11411_large
theorem row11411_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11411 row11411 := row11411_large_covers

def row11423_small : List Nat := []
theorem row11423_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11423 row11423_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11423_block_005 : List Nat := []
theorem row11423_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11423 row11423_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11423_block_006 : List Nat := [11579, 12401]
def plan11423_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row11423_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11423 row11423_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11423_block_006
  · decide
  · rfl

def row11423_block_007 : List Nat := [12473, 12653, 12959, 13553, 13577, 14057]
def plan11423_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .possible, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.secondFactor 7), .second, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .possible, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 23), .possible, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row11423_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11423 row11423_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11423_block_007
  · decide
  · rfl

def row11423_block_008 : List Nat := []
def plan11423_block_008 : List Budget.Choice := [.first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11423_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11423 row11423_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11423_block_008
  · decide
  · rfl

def row11423_block_009 : List Nat := []
theorem row11423_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11423 row11423_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11423_block_010 : List Nat := []
theorem row11423_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11423 row11423_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11423_block_011 : List Nat := []
theorem row11423_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11423 row11423_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11423_block_012 : List Nat := []
theorem row11423_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11423 row11423_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11423_block_013 : List Nat := []
theorem row11423_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11423 row11423_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11423_large_left_left : List Nat := row11423_small ++ row11423_block_005
theorem row11423_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11423 row11423_large_left_left :=
  Budget.RowUpperBound.branch row11423_small_covers row11423_block_005_covers

def row11423_large_left_right_right : List Nat := row11423_block_007 ++ row11423_block_008
theorem row11423_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11423 row11423_large_left_right_right :=
  Budget.RowUpperBound.branch row11423_block_007_covers row11423_block_008_covers

def row11423_large_left_right : List Nat := row11423_block_006 ++ row11423_large_left_right_right
theorem row11423_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11423 row11423_large_left_right :=
  Budget.RowUpperBound.branch row11423_block_006_covers row11423_large_left_right_right_covers

def row11423_large_left : List Nat := row11423_large_left_left ++ row11423_large_left_right
theorem row11423_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11423 row11423_large_left :=
  Budget.RowUpperBound.branch row11423_large_left_left_covers row11423_large_left_right_covers

def row11423_large_right_left : List Nat := row11423_block_009 ++ row11423_block_010
theorem row11423_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11423 row11423_large_right_left :=
  Budget.RowUpperBound.branch row11423_block_009_covers row11423_block_010_covers

def row11423_large_right_right_right : List Nat := row11423_block_012 ++ row11423_block_013
theorem row11423_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11423 row11423_large_right_right_right :=
  Budget.RowUpperBound.branch row11423_block_012_covers row11423_block_013_covers

def row11423_large_right_right : List Nat := row11423_block_011 ++ row11423_large_right_right_right
theorem row11423_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11423 row11423_large_right_right :=
  Budget.RowUpperBound.branch row11423_block_011_covers row11423_large_right_right_right_covers

def row11423_large_right : List Nat := row11423_large_right_left ++ row11423_large_right_right
theorem row11423_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11423 row11423_large_right :=
  Budget.RowUpperBound.branch row11423_large_right_left_covers row11423_large_right_right_covers

def row11423_large : List Nat := row11423_large_left ++ row11423_large_right
theorem row11423_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11423 row11423_large :=
  Budget.RowUpperBound.branch row11423_large_left_covers row11423_large_right_covers

def row11423 : List Nat := row11423_large
theorem row11423_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11423 row11423 := row11423_large_covers

end ProjectEulerSolutions.PrimePairLarge
