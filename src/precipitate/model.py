from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Model(Protocol[T]):
    def fit(self, inputs: list[str], outputs: list[T]) -> None: ...

    def predict(self, inputs: list[str]) -> list[T]: ...

    def distill(
        self, student: "Model[T]", inputs: list[str], outputs: list[T]
    ) -> None: ...
