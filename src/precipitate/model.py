from typing import BinaryIO, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Model(Protocol[T]):
    def fit(self, inputs: list[str], outputs: list[T], *args, **kwargs) -> None: ...

    def predict(self, inputs: list[str]) -> list[T]: ...

    def distill(
        self, student: "Model[T]", inputs: list[str], outputs: list[T]
    ) -> None: ...

    def save(self, stream: BinaryIO) -> None: ...

    @classmethod
    def load(cls, stream: BinaryIO) -> "Model": ...
