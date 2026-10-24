import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11177_small : List Nat := []
theorem row11177_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11177 row11177_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11177_block_005 : List Nat := []
theorem row11177_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11177 row11177_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11177_block_006 : List Nat := [11351, 11369, 11807]
def plan11177_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .possible, .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 19), (.firstFactor 19), .first, .first, (.secondFactor 31), .first, (.secondFactor 23), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 23), .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .second, .first, .second, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row11177_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11177 row11177_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11177_block_006
  · decide
  · rfl

def row11177_block_007 : List Nat := [13757]
def plan11177_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, .second, (.firstFactor 37), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 31), (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 31), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11177_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11177 row11177_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11177_block_007
  · decide
  · rfl

def row11177_block_008 : List Nat := [14519]
def plan11177_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11177_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11177 row11177_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11177_block_008
  · decide
  · rfl

def row11177_block_009 : List Nat := []
theorem row11177_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11177 row11177_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11177_block_010 : List Nat := []
theorem row11177_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11177 row11177_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11177_block_011 : List Nat := []
theorem row11177_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11177 row11177_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11177_block_012 : List Nat := []
theorem row11177_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11177 row11177_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11177_block_013 : List Nat := []
theorem row11177_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11177 row11177_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11177_large_left_left : List Nat := row11177_small ++ row11177_block_005
theorem row11177_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11177 row11177_large_left_left :=
  Budget.RowUpperBound.branch row11177_small_covers row11177_block_005_covers

def row11177_large_left_right_right : List Nat := row11177_block_007 ++ row11177_block_008
theorem row11177_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11177 row11177_large_left_right_right :=
  Budget.RowUpperBound.branch row11177_block_007_covers row11177_block_008_covers

def row11177_large_left_right : List Nat := row11177_block_006 ++ row11177_large_left_right_right
theorem row11177_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11177 row11177_large_left_right :=
  Budget.RowUpperBound.branch row11177_block_006_covers row11177_large_left_right_right_covers

def row11177_large_left : List Nat := row11177_large_left_left ++ row11177_large_left_right
theorem row11177_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11177 row11177_large_left :=
  Budget.RowUpperBound.branch row11177_large_left_left_covers row11177_large_left_right_covers

def row11177_large_right_left : List Nat := row11177_block_009 ++ row11177_block_010
theorem row11177_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11177 row11177_large_right_left :=
  Budget.RowUpperBound.branch row11177_block_009_covers row11177_block_010_covers

def row11177_large_right_right_right : List Nat := row11177_block_012 ++ row11177_block_013
theorem row11177_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11177 row11177_large_right_right_right :=
  Budget.RowUpperBound.branch row11177_block_012_covers row11177_block_013_covers

def row11177_large_right_right : List Nat := row11177_block_011 ++ row11177_large_right_right_right
theorem row11177_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11177 row11177_large_right_right :=
  Budget.RowUpperBound.branch row11177_block_011_covers row11177_large_right_right_right_covers

def row11177_large_right : List Nat := row11177_large_right_left ++ row11177_large_right_right
theorem row11177_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11177 row11177_large_right :=
  Budget.RowUpperBound.branch row11177_large_right_left_covers row11177_large_right_right_covers

def row11177_large : List Nat := row11177_large_left ++ row11177_large_right
theorem row11177_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11177 row11177_large :=
  Budget.RowUpperBound.branch row11177_large_left_covers row11177_large_right_covers

def row11177 : List Nat := row11177_large
theorem row11177_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11177 row11177 := row11177_large_covers

def row11197_small : List Nat := []
theorem row11197_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11197 row11197_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11197_block_005 : List Nat := []
theorem row11197_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11197 row11197_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11197_block_006 : List Nat := [11941, 12043, 12163, 12409]
def plan11197_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .second, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 29), (.firstFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row11197_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11197 row11197_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11197_block_006
  · decide
  · rfl

def row11197_block_007 : List Nat := [12457, 13093, 13297, 13711, 13879]
def plan11197_block_007 : List Budget.Choice := [.second, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 17), (.secondFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 37), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .possible, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row11197_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11197 row11197_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11197_block_007
  · decide
  · rfl

def row11197_block_008 : List Nat := []
def plan11197_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11197_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11197 row11197_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11197_block_008
  · decide
  · rfl

def row11197_block_009 : List Nat := []
theorem row11197_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11197 row11197_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11197_block_010 : List Nat := []
theorem row11197_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11197 row11197_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11197_block_011 : List Nat := []
theorem row11197_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11197 row11197_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11197_block_012 : List Nat := []
theorem row11197_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11197 row11197_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11197_block_013 : List Nat := []
theorem row11197_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11197 row11197_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11197_large_left_left : List Nat := row11197_small ++ row11197_block_005
theorem row11197_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11197 row11197_large_left_left :=
  Budget.RowUpperBound.branch row11197_small_covers row11197_block_005_covers

def row11197_large_left_right_right : List Nat := row11197_block_007 ++ row11197_block_008
theorem row11197_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11197 row11197_large_left_right_right :=
  Budget.RowUpperBound.branch row11197_block_007_covers row11197_block_008_covers

def row11197_large_left_right : List Nat := row11197_block_006 ++ row11197_large_left_right_right
theorem row11197_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11197 row11197_large_left_right :=
  Budget.RowUpperBound.branch row11197_block_006_covers row11197_large_left_right_right_covers

def row11197_large_left : List Nat := row11197_large_left_left ++ row11197_large_left_right
theorem row11197_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11197 row11197_large_left :=
  Budget.RowUpperBound.branch row11197_large_left_left_covers row11197_large_left_right_covers

def row11197_large_right_left : List Nat := row11197_block_009 ++ row11197_block_010
theorem row11197_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11197 row11197_large_right_left :=
  Budget.RowUpperBound.branch row11197_block_009_covers row11197_block_010_covers

def row11197_large_right_right_right : List Nat := row11197_block_012 ++ row11197_block_013
theorem row11197_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11197 row11197_large_right_right_right :=
  Budget.RowUpperBound.branch row11197_block_012_covers row11197_block_013_covers

def row11197_large_right_right : List Nat := row11197_block_011 ++ row11197_large_right_right_right
theorem row11197_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11197 row11197_large_right_right :=
  Budget.RowUpperBound.branch row11197_block_011_covers row11197_large_right_right_right_covers

def row11197_large_right : List Nat := row11197_large_right_left ++ row11197_large_right_right
theorem row11197_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11197 row11197_large_right :=
  Budget.RowUpperBound.branch row11197_large_right_left_covers row11197_large_right_right_covers

def row11197_large : List Nat := row11197_large_left ++ row11197_large_right
theorem row11197_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11197 row11197_large :=
  Budget.RowUpperBound.branch row11197_large_left_covers row11197_large_right_covers

def row11197 : List Nat := row11197_large
theorem row11197_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11197 row11197 := row11197_large_covers

def row11213_small : List Nat := []
theorem row11213_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11213 row11213_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11213_block_005 : List Nat := []
theorem row11213_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11213 row11213_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11213_block_006 : List Nat := [11813]
def plan11213_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .second, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row11213_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11213 row11213_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11213_block_006
  · decide
  · rfl

def row11213_block_007 : List Nat := [12497, 13001, 13763]
def plan11213_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, (.firstFactor 17), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row11213_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11213 row11213_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11213_block_007
  · decide
  · rfl

def row11213_block_008 : List Nat := [14771]
def plan11213_block_008 : List Budget.Choice := [.first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 37), .first, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .possible, .first, (.firstFactor 29), .first, (.secondFactor 29), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11213_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11213 row11213_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11213_block_008
  · decide
  · rfl

def row11213_block_009 : List Nat := []
theorem row11213_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11213 row11213_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11213_block_010 : List Nat := []
theorem row11213_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11213 row11213_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11213_block_011 : List Nat := []
theorem row11213_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11213 row11213_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11213_block_012 : List Nat := []
theorem row11213_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11213 row11213_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11213_block_013 : List Nat := []
theorem row11213_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11213 row11213_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11213_large_left_left : List Nat := row11213_small ++ row11213_block_005
theorem row11213_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11213 row11213_large_left_left :=
  Budget.RowUpperBound.branch row11213_small_covers row11213_block_005_covers

def row11213_large_left_right_right : List Nat := row11213_block_007 ++ row11213_block_008
theorem row11213_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11213 row11213_large_left_right_right :=
  Budget.RowUpperBound.branch row11213_block_007_covers row11213_block_008_covers

def row11213_large_left_right : List Nat := row11213_block_006 ++ row11213_large_left_right_right
theorem row11213_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11213 row11213_large_left_right :=
  Budget.RowUpperBound.branch row11213_block_006_covers row11213_large_left_right_right_covers

def row11213_large_left : List Nat := row11213_large_left_left ++ row11213_large_left_right
theorem row11213_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11213 row11213_large_left :=
  Budget.RowUpperBound.branch row11213_large_left_left_covers row11213_large_left_right_covers

def row11213_large_right_left : List Nat := row11213_block_009 ++ row11213_block_010
theorem row11213_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11213 row11213_large_right_left :=
  Budget.RowUpperBound.branch row11213_block_009_covers row11213_block_010_covers

def row11213_large_right_right_right : List Nat := row11213_block_012 ++ row11213_block_013
theorem row11213_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11213 row11213_large_right_right_right :=
  Budget.RowUpperBound.branch row11213_block_012_covers row11213_block_013_covers

def row11213_large_right_right : List Nat := row11213_block_011 ++ row11213_large_right_right_right
theorem row11213_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11213 row11213_large_right_right :=
  Budget.RowUpperBound.branch row11213_block_011_covers row11213_large_right_right_right_covers

def row11213_large_right : List Nat := row11213_large_right_left ++ row11213_large_right_right
theorem row11213_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11213 row11213_large_right :=
  Budget.RowUpperBound.branch row11213_large_right_left_covers row11213_large_right_right_covers

def row11213_large : List Nat := row11213_large_left ++ row11213_large_right
theorem row11213_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11213 row11213_large :=
  Budget.RowUpperBound.branch row11213_large_left_covers row11213_large_right_covers

def row11213 : List Nat := row11213_large
theorem row11213_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11213 row11213 := row11213_large_covers

def row11239_small : List Nat := []
theorem row11239_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11239 row11239_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11239_block_005 : List Nat := []
theorem row11239_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11239 row11239_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11239_block_006 : List Nat := [11779, 12157]
def plan11239_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 29), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 31), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row11239_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11239 row11239_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11239_block_006
  · decide
  · rfl

def row11239_block_007 : List Nat := [14437]
def plan11239_block_007 : List Budget.Choice := [(.secondFactor 17), (.firstFactor 7), .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 31), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 23), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row11239_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11239 row11239_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11239_block_007
  · decide
  · rfl

def row11239_block_008 : List Nat := [14629]
def plan11239_block_008 : List Budget.Choice := [(.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11239_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11239 row11239_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11239_block_008
  · decide
  · rfl

def row11239_block_009 : List Nat := []
theorem row11239_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11239 row11239_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11239_block_010 : List Nat := []
theorem row11239_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11239 row11239_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11239_block_011 : List Nat := []
theorem row11239_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11239 row11239_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11239_block_012 : List Nat := []
theorem row11239_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11239 row11239_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11239_block_013 : List Nat := []
theorem row11239_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11239 row11239_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11239_large_left_left : List Nat := row11239_small ++ row11239_block_005
theorem row11239_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11239 row11239_large_left_left :=
  Budget.RowUpperBound.branch row11239_small_covers row11239_block_005_covers

def row11239_large_left_right_right : List Nat := row11239_block_007 ++ row11239_block_008
theorem row11239_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11239 row11239_large_left_right_right :=
  Budget.RowUpperBound.branch row11239_block_007_covers row11239_block_008_covers

def row11239_large_left_right : List Nat := row11239_block_006 ++ row11239_large_left_right_right
theorem row11239_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11239 row11239_large_left_right :=
  Budget.RowUpperBound.branch row11239_block_006_covers row11239_large_left_right_right_covers

def row11239_large_left : List Nat := row11239_large_left_left ++ row11239_large_left_right
theorem row11239_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11239 row11239_large_left :=
  Budget.RowUpperBound.branch row11239_large_left_left_covers row11239_large_left_right_covers

def row11239_large_right_left : List Nat := row11239_block_009 ++ row11239_block_010
theorem row11239_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11239 row11239_large_right_left :=
  Budget.RowUpperBound.branch row11239_block_009_covers row11239_block_010_covers

def row11239_large_right_right_right : List Nat := row11239_block_012 ++ row11239_block_013
theorem row11239_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11239 row11239_large_right_right_right :=
  Budget.RowUpperBound.branch row11239_block_012_covers row11239_block_013_covers

def row11239_large_right_right : List Nat := row11239_block_011 ++ row11239_large_right_right_right
theorem row11239_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11239 row11239_large_right_right :=
  Budget.RowUpperBound.branch row11239_block_011_covers row11239_large_right_right_right_covers

def row11239_large_right : List Nat := row11239_large_right_left ++ row11239_large_right_right
theorem row11239_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11239 row11239_large_right :=
  Budget.RowUpperBound.branch row11239_large_right_left_covers row11239_large_right_right_covers

def row11239_large : List Nat := row11239_large_left ++ row11239_large_right
theorem row11239_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11239 row11239_large :=
  Budget.RowUpperBound.branch row11239_large_left_covers row11239_large_right_covers

def row11239 : List Nat := row11239_large
theorem row11239_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11239 row11239 := row11239_large_covers

end ProjectEulerSolutions.PrimePairLarge
