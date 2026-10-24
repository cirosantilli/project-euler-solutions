import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11689_small : List Nat := []
theorem row11689_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11689 row11689_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11689_block_005 : List Nat := []
theorem row11689_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11689 row11689_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11689_block_006 : List Nat := [11701]
def plan11689_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .possible, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11689_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11689 row11689_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11689_block_006
  · decide
  · rfl

def row11689_block_007 : List Nat := [12517, 12763, 13033]
def plan11689_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 37), (.firstFactor 23), .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 31), .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 29), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11689_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11689 row11689_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11689_block_007
  · decide
  · rfl

def row11689_block_008 : List Nat := []
theorem row11689_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11689 row11689_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11689_block_009 : List Nat := []
theorem row11689_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11689 row11689_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11689_block_010 : List Nat := []
theorem row11689_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11689 row11689_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11689_block_011 : List Nat := []
theorem row11689_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11689 row11689_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11689_block_012 : List Nat := []
theorem row11689_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11689 row11689_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11689_block_013 : List Nat := []
theorem row11689_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11689 row11689_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11689_large_left_left : List Nat := row11689_small ++ row11689_block_005
theorem row11689_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11689 row11689_large_left_left :=
  Budget.RowUpperBound.branch row11689_small_covers row11689_block_005_covers

def row11689_large_left_right_right : List Nat := row11689_block_007 ++ row11689_block_008
theorem row11689_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11689 row11689_large_left_right_right :=
  Budget.RowUpperBound.branch row11689_block_007_covers row11689_block_008_covers

def row11689_large_left_right : List Nat := row11689_block_006 ++ row11689_large_left_right_right
theorem row11689_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11689 row11689_large_left_right :=
  Budget.RowUpperBound.branch row11689_block_006_covers row11689_large_left_right_right_covers

def row11689_large_left : List Nat := row11689_large_left_left ++ row11689_large_left_right
theorem row11689_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11689 row11689_large_left :=
  Budget.RowUpperBound.branch row11689_large_left_left_covers row11689_large_left_right_covers

def row11689_large_right_left : List Nat := row11689_block_009 ++ row11689_block_010
theorem row11689_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11689 row11689_large_right_left :=
  Budget.RowUpperBound.branch row11689_block_009_covers row11689_block_010_covers

def row11689_large_right_right_right : List Nat := row11689_block_012 ++ row11689_block_013
theorem row11689_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11689 row11689_large_right_right_right :=
  Budget.RowUpperBound.branch row11689_block_012_covers row11689_block_013_covers

def row11689_large_right_right : List Nat := row11689_block_011 ++ row11689_large_right_right_right
theorem row11689_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11689 row11689_large_right_right :=
  Budget.RowUpperBound.branch row11689_block_011_covers row11689_large_right_right_right_covers

def row11689_large_right : List Nat := row11689_large_right_left ++ row11689_large_right_right
theorem row11689_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11689 row11689_large_right :=
  Budget.RowUpperBound.branch row11689_large_right_left_covers row11689_large_right_right_covers

def row11689_large : List Nat := row11689_large_left ++ row11689_large_right
theorem row11689_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11689 row11689_large :=
  Budget.RowUpperBound.branch row11689_large_left_covers row11689_large_right_covers

def row11689 : List Nat := row11689_large
theorem row11689_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11689 row11689 := row11689_large_covers

def row11699_small : List Nat := []
theorem row11699_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11699 row11699_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11699_block_005 : List Nat := []
theorem row11699_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11699 row11699_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11699_block_006 : List Nat := []
def plan11699_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 19), .first, .second, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second]
set_option maxRecDepth 2048 in
theorem row11699_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11699 row11699_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11699_block_006
  · decide
  · rfl

def row11699_block_007 : List Nat := [12959]
def plan11699_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 37), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, .second, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 31), (.secondFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .second, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11699_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11699 row11699_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11699_block_007
  · decide
  · rfl

def row11699_block_008 : List Nat := []
theorem row11699_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11699 row11699_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11699_block_009 : List Nat := []
theorem row11699_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11699 row11699_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11699_block_010 : List Nat := []
theorem row11699_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11699 row11699_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11699_block_011 : List Nat := []
theorem row11699_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11699 row11699_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11699_block_012 : List Nat := []
theorem row11699_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11699 row11699_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11699_block_013 : List Nat := []
theorem row11699_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11699 row11699_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11699_large_left_left : List Nat := row11699_small ++ row11699_block_005
theorem row11699_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11699 row11699_large_left_left :=
  Budget.RowUpperBound.branch row11699_small_covers row11699_block_005_covers

def row11699_large_left_right_right : List Nat := row11699_block_007 ++ row11699_block_008
theorem row11699_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11699 row11699_large_left_right_right :=
  Budget.RowUpperBound.branch row11699_block_007_covers row11699_block_008_covers

def row11699_large_left_right : List Nat := row11699_block_006 ++ row11699_large_left_right_right
theorem row11699_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11699 row11699_large_left_right :=
  Budget.RowUpperBound.branch row11699_block_006_covers row11699_large_left_right_right_covers

def row11699_large_left : List Nat := row11699_large_left_left ++ row11699_large_left_right
theorem row11699_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11699 row11699_large_left :=
  Budget.RowUpperBound.branch row11699_large_left_left_covers row11699_large_left_right_covers

def row11699_large_right_left : List Nat := row11699_block_009 ++ row11699_block_010
theorem row11699_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11699 row11699_large_right_left :=
  Budget.RowUpperBound.branch row11699_block_009_covers row11699_block_010_covers

def row11699_large_right_right_right : List Nat := row11699_block_012 ++ row11699_block_013
theorem row11699_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11699 row11699_large_right_right_right :=
  Budget.RowUpperBound.branch row11699_block_012_covers row11699_block_013_covers

def row11699_large_right_right : List Nat := row11699_block_011 ++ row11699_large_right_right_right
theorem row11699_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11699 row11699_large_right_right :=
  Budget.RowUpperBound.branch row11699_block_011_covers row11699_large_right_right_right_covers

def row11699_large_right : List Nat := row11699_large_right_left ++ row11699_large_right_right
theorem row11699_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11699 row11699_large_right :=
  Budget.RowUpperBound.branch row11699_large_right_left_covers row11699_large_right_right_covers

def row11699_large : List Nat := row11699_large_left ++ row11699_large_right
theorem row11699_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11699 row11699_large :=
  Budget.RowUpperBound.branch row11699_large_left_covers row11699_large_right_covers

def row11699 : List Nat := row11699_large
theorem row11699_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11699 row11699 := row11699_large_covers

def row11701_small : List Nat := []
theorem row11701_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11701 row11701_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11701_block_005 : List Nat := []
theorem row11701_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11701 row11701_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11701_block_006 : List Nat := [11779]
def plan11701_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11701_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11701 row11701_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11701_block_006
  · decide
  · rfl

def row11701_block_007 : List Nat := [12781, 13597, 13759]
def plan11701_block_007 : List Budget.Choice := [.second, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 31), .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 37), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11701_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11701 row11701_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11701_block_007
  · decide
  · rfl

def row11701_block_008 : List Nat := []
theorem row11701_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11701 row11701_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11701_block_009 : List Nat := []
theorem row11701_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11701 row11701_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11701_block_010 : List Nat := []
theorem row11701_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11701 row11701_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11701_block_011 : List Nat := []
theorem row11701_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11701 row11701_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11701_block_012 : List Nat := []
theorem row11701_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11701 row11701_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11701_block_013 : List Nat := []
theorem row11701_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11701 row11701_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11701_large_left_left : List Nat := row11701_small ++ row11701_block_005
theorem row11701_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11701 row11701_large_left_left :=
  Budget.RowUpperBound.branch row11701_small_covers row11701_block_005_covers

def row11701_large_left_right_right : List Nat := row11701_block_007 ++ row11701_block_008
theorem row11701_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11701 row11701_large_left_right_right :=
  Budget.RowUpperBound.branch row11701_block_007_covers row11701_block_008_covers

def row11701_large_left_right : List Nat := row11701_block_006 ++ row11701_large_left_right_right
theorem row11701_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11701 row11701_large_left_right :=
  Budget.RowUpperBound.branch row11701_block_006_covers row11701_large_left_right_right_covers

def row11701_large_left : List Nat := row11701_large_left_left ++ row11701_large_left_right
theorem row11701_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11701 row11701_large_left :=
  Budget.RowUpperBound.branch row11701_large_left_left_covers row11701_large_left_right_covers

def row11701_large_right_left : List Nat := row11701_block_009 ++ row11701_block_010
theorem row11701_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11701 row11701_large_right_left :=
  Budget.RowUpperBound.branch row11701_block_009_covers row11701_block_010_covers

def row11701_large_right_right_right : List Nat := row11701_block_012 ++ row11701_block_013
theorem row11701_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11701 row11701_large_right_right_right :=
  Budget.RowUpperBound.branch row11701_block_012_covers row11701_block_013_covers

def row11701_large_right_right : List Nat := row11701_block_011 ++ row11701_large_right_right_right
theorem row11701_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11701 row11701_large_right_right :=
  Budget.RowUpperBound.branch row11701_block_011_covers row11701_large_right_right_right_covers

def row11701_large_right : List Nat := row11701_large_right_left ++ row11701_large_right_right
theorem row11701_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11701 row11701_large_right :=
  Budget.RowUpperBound.branch row11701_large_right_left_covers row11701_large_right_right_covers

def row11701_large : List Nat := row11701_large_left ++ row11701_large_right
theorem row11701_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11701 row11701_large :=
  Budget.RowUpperBound.branch row11701_large_left_covers row11701_large_right_covers

def row11701 : List Nat := row11701_large
theorem row11701_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11701 row11701 := row11701_large_covers

def row11717_small : List Nat := []
theorem row11717_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11717 row11717_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11717_block_005 : List Nat := []
theorem row11717_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11717 row11717_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11717_block_006 : List Nat := [12251]
def plan11717_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row11717_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11717 row11717_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11717_block_006
  · decide
  · rfl

def row11717_block_007 : List Nat := [12821]
def plan11717_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .second, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .second, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11717_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11717 row11717_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11717_block_007
  · decide
  · rfl

def row11717_block_008 : List Nat := []
theorem row11717_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11717 row11717_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11717_block_009 : List Nat := []
theorem row11717_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11717 row11717_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11717_block_010 : List Nat := []
theorem row11717_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11717 row11717_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11717_block_011 : List Nat := []
theorem row11717_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11717 row11717_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11717_block_012 : List Nat := []
theorem row11717_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11717 row11717_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11717_block_013 : List Nat := []
theorem row11717_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11717 row11717_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11717_large_left_left : List Nat := row11717_small ++ row11717_block_005
theorem row11717_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11717 row11717_large_left_left :=
  Budget.RowUpperBound.branch row11717_small_covers row11717_block_005_covers

def row11717_large_left_right_right : List Nat := row11717_block_007 ++ row11717_block_008
theorem row11717_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11717 row11717_large_left_right_right :=
  Budget.RowUpperBound.branch row11717_block_007_covers row11717_block_008_covers

def row11717_large_left_right : List Nat := row11717_block_006 ++ row11717_large_left_right_right
theorem row11717_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11717 row11717_large_left_right :=
  Budget.RowUpperBound.branch row11717_block_006_covers row11717_large_left_right_right_covers

def row11717_large_left : List Nat := row11717_large_left_left ++ row11717_large_left_right
theorem row11717_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11717 row11717_large_left :=
  Budget.RowUpperBound.branch row11717_large_left_left_covers row11717_large_left_right_covers

def row11717_large_right_left : List Nat := row11717_block_009 ++ row11717_block_010
theorem row11717_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11717 row11717_large_right_left :=
  Budget.RowUpperBound.branch row11717_block_009_covers row11717_block_010_covers

def row11717_large_right_right_right : List Nat := row11717_block_012 ++ row11717_block_013
theorem row11717_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11717 row11717_large_right_right_right :=
  Budget.RowUpperBound.branch row11717_block_012_covers row11717_block_013_covers

def row11717_large_right_right : List Nat := row11717_block_011 ++ row11717_large_right_right_right
theorem row11717_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11717 row11717_large_right_right :=
  Budget.RowUpperBound.branch row11717_block_011_covers row11717_large_right_right_right_covers

def row11717_large_right : List Nat := row11717_large_right_left ++ row11717_large_right_right
theorem row11717_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11717 row11717_large_right :=
  Budget.RowUpperBound.branch row11717_large_right_left_covers row11717_large_right_right_covers

def row11717_large : List Nat := row11717_large_left ++ row11717_large_right
theorem row11717_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11717 row11717_large :=
  Budget.RowUpperBound.branch row11717_large_left_covers row11717_large_right_covers

def row11717 : List Nat := row11717_large
theorem row11717_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11717 row11717 := row11717_large_covers

end ProjectEulerSolutions.PrimePairLarge
